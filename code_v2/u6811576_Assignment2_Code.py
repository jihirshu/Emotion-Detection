from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from torch.utils import data
import os
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import cv2
import random
from mtcnn import MTCNN
from copy import deepcopy
import matplotlib.pyplot as plt
seed=0
pre_processdata = False
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def split_data():
    train_images = []
    test_images = []
    train_labels = []
    test_labels = []
    valid_labels = []
    train_data = []
    test_data = []
    valida_data = []
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    detector = MTCNN()
    haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    seg_values = np.random.rand(675)

    dirs = ['Train', 'Test', 'Valid', 'Train/Happy', 'Train/Angry', 'Train/Sad', 'Train/Surprise', 'Train/Neutral',
            'Train/Fear', 'Train/Disgust',
            'Test/Happy', 'Test/Angry', 'Test/Sad', 'Test/Surprise', 'Test/Neutral', 'Test/Fear', 'Test/Disgust',
            'Valid/Happy', 'Valid/Angry', 'Valid/Sad', 'Valid/Surprise', 'Valid/Neutral', 'Valid/Fear',
            'Valid/Disgust', ]
    for dir in dirs:
        try:
            os.mkdir(dir)
        except FileExistsError:
            print("Please delete the existing Train, Test and Valid Folders")
            exit()
    i=0
    for subdir, dir, files in os.walk('./Subset For Assignment SFEW/'):
        for file in files:
            path = os.path.join(subdir, file)
            if ("cropped" in path):
                continue
            if (path[-3:] == "png"):
                image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                print(path)
                result = detector.detect_faces(image)
                if not result:
                    faces_rects = haar_cascade_face.detectMultiScale(image, scaleFactor=1.2, minNeighbors=10)
                    for tuple in faces_rects:
                        bounding_box = tuple
                else:
                    bounding_box = result[0]['box']
                # keypoints = result[0]['keypoints']
                bounding_box = [0 if i < 0 else i for i in bounding_box]
                cropped_image = image[bounding_box[1]:bounding_box[1] + bounding_box[3],
                                bounding_box[0]:bounding_box[0] + bounding_box[2]]
                cropped_image = cv2.resize(cropped_image, (48, 48), interpolation=cv2.INTER_LINEAR)
                label = 0
                newimg = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)

                if ("Angry" in path):
                    file = "Angry/" + file
                    label = 0
                elif ("Disgust" in path):
                    file = "Disgust/" + file
                    label = 1
                elif ("Fear" in path):
                    file = "Fear/" + file
                    label = 2
                elif ("Happy" in path):
                    file = "Happy/" + file
                    label = 3
                elif ("Neutral" in path):
                    file = "Neutral/" + file
                    label = 4
                elif ("Sad" in path):
                    file = "Sad/" + file
                    label = 5
                elif ("Surprise" in path):
                    file = "Surprise/" + file
                    label = 6
                r = seg_values[i]
                i=i+1
                if (r < 0.1):
                    file = "Test/" + file
                    test_data.append(newimg)
                    test_labels.append(label)
                elif (r < 0.2):
                    file = "Valid/" + file
                    valida_data.append(newimg)
                    valid_labels.append(label)
                else:
                    file = "Train/" + file
                    train_data.append(newimg)
                    train_labels.append(label)
                print(file)
                cv2.imwrite(file, newimg)

    train = np.asarray(train_data)
    test = np.asarray(test_data)
    valid = np.asarray(valida_data)
    train_y = np.asarray(train_labels)
    test_y = np.asarray(test_labels)
    valid_y = np.asarray(valid_labels)

    np.save("Train_Data.npy", train)
    np.save("Train_Labels.npy", train_y)
    np.save("Test_Data.npy", test)
    np.save("Test_Labels.npy", test_y)
    np.save("Valid_Data.npy", valid)
    np.save("Valid_Labels.npy", valid_y)

class CustomConvNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomConvNet, self).__init__()

        self.layer1 = self.conv_module(3, 16)
        self.layer2 = self.conv_module(16, 32)
        self.layer3 = self.conv_module(32, 64)
        self.layer4 = self.conv_module(64, 128)
        self.layer5 = self.conv_module(128, 256)
        self.gap = self.global_avg_pool(256, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        pattern_space = self.layer5(out)
        out = self.gap(pattern_space)
        out = out.view(-1, 7)

        return out, pattern_space

    def conv_module(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def global_avg_pool(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))

learning_rate = 0.0001
num_classes = 7
if pre_processdata:
    split_data()

model = CustomConvNet(num_classes=num_classes)

def train_model(epoch, train_loader, optimizer):
    model.train()
    correct = 0
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output, p = model(data)
        loss = F.nll_loss(output, target)
        train_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        loss.backward()
        optimizer.step()
        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader)
    acc = 100. * int(correct) / (len(train_loader.dataset))
    print("Train Set : ")
    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))

    return acc

def test_model(test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        output, p = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
    acc = 100. * int(correct) / (len(test_loader.dataset))
    test_loss /= len(test_loader.dataset)
    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        acc))

    return acc



traindir = './Train/'
validdir = './Valid/'
testdir = './Test/'

batch_size = 10

categories = []
img_categories = []
n_train = []
n_valid = []
n_test = []
hs = []
ws = []
folders = ['Happy', 'Angry','Disgust','Fear','Surprise','Neutral','Sad']
# Iterate through each category
for d in os.listdir(traindir):
    if d not in folders:
        continue
    categories.append(d)
    # Number of each image
    train_imgs = os.listdir(traindir + d)
    valid_imgs = os.listdir(validdir + d)
    test_imgs = os.listdir(testdir + d)
    n_train.append(len(train_imgs))
    n_test.append(len(test_imgs))
    n_valid.append(len(valid_imgs))
    # Find stats for train images
    for i in train_imgs:
        if (i[-3:] != "png"):
            continue
        img_categories.append(d)
        img = Image.open(traindir + d + '/' + i)
        img_array = np.array(img)
        # Shape
        hs.append(img_array.shape[0])
        ws.append(img_array.shape[1])


cat_df = pd.DataFrame({'category': categories,
                       'n_train': n_train,
					   'n_valid': n_valid,
                        'n_test': n_test}).\
    sort_values('category')

# Dataframe of training images
image_df = pd.DataFrame({
    'category': img_categories,
    'height': hs,
    'width': ws
})
cat_df.sort_values('n_train', ascending=False, inplace=True)

image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        # transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        # transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    # Test does not use augmentation
    'test':
    transforms.Compose([
        # transforms.Resize(size=256),
        # transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


data = {
    'train':
    datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
	'val' :
	datasets.ImageFolder(root=validdir, transform=image_transforms['test']),
    'test':
    datasets.ImageFolder(root=testdir, transform=image_transforms['test'])
}

# Dataloader iterators
dataloaders = {
    'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
	'val': DataLoader(data['val'], batch_size=batch_size, shuffle=True),
    'test': DataLoader(data['test'], batch_size=batch_size, shuffle=True)
}

trainiter = iter(dataloaders['train'])
features, labels = next(trainiter)
print(features.shape, labels.shape)

n_classes = len(cat_df)

avg_test_acc = 0


optimizer = optim.Adam(model.parameters(), lr=learning_rate)
epochs = 40
train_acc = []
test_acc = []
for epoch in range(1, epochs+1):
    trainingAcc = train_model(epoch, dataloaders['train'], optimizer)
    train_acc.append(trainingAcc)
    print("Test Set : ")
    testAcc = test_model(dataloaders['test'])
    test_acc.append(testAcc)
    avg_test_acc = avg_test_acc + testAcc

plt.plot(train_acc)
plt.plot(test_acc)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Training Accuracy", "Testing Accuracy"], loc = "upper left")
plt.show()

print("Average test accuracy : ", avg_test_acc / epochs)
print("Validation Set : ")
acc_before_pruning = test_model(dataloaders['val'])

dataloader = {
    'train': DataLoader(data['train'], batch_size=549, shuffle=True),
	'val': DataLoader(data['val'], batch_size=batch_size, shuffle=True),
    'test': DataLoader(data['test'], batch_size=batch_size, shuffle=True)
}

p = None
for batch_idx, (data, target) in enumerate(dataloader['train']):
    out, p = model(data)

p = p.detach().numpy()
p = p / (np.max(p) - np.min(p))
p = p - 0.5

activation_values = p[:,:,0,0]
weights = model.state_dict()['gap.0.weight']

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

# Function which returns the angle between two vectors in radians
def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

initial_weights = model.state_dict()['gap.0.weight']
neurons_pruned = []
acc_after_pruning = []
for i in range(15):
    model.state_dict()['gap.0.weight'] = initial_weights
    threshold_angle = i
    hidden_size=256
    neurons_to_be_removed = []
    neurons_to_be_retained = []
    original_weights = deepcopy(weights)
    neurons_dict = {x : 0 for x in range(hidden_size)}
    number_of_neurons_pruned = 0
    for i in range(hidden_size):
        v1 = activation_values[:,i]
        # v1 = v1 / (2*torch.max(v1))
        for j in range(i+1,hidden_size):
            if (i!=j):
                v2 = activation_values[:,j]
                # v2 = v2 / (2 * torch.max(v2))
                angle = np.degrees(angle_between(v1, v2))
                if (angle < threshold_angle):# or (angle > 170):
                    if (i in neurons_to_be_removed):
                        weights[:,j,:,:] = weights[:,j, :, :] + original_weights[:,i, :, :]
                    elif (j in neurons_to_be_removed):
                        weights[:,i,:,:] = weights[:,i, :,:] + original_weights[:,j, :,:]
                    else:
                        neurons_to_be_removed.append(i)
                        weights[:,j, :,:] = weights[:,j, :,:] + original_weights[:,i, :,:]
                        weights[:,i, :,:] = 0
                if (angle > (180 - threshold_angle)):
                    neurons_to_be_removed.append(i)
                    neurons_to_be_removed.append(j)
                    weights[:,j,:,:] = 0
                    weights[:,i,:,:] = 0


    print("Neurons to be Removed : ", neurons_to_be_removed)
    print("Number of Neurons Removed : ", len(neurons_to_be_removed))

    model.state_dict()['gap.0.weight'] = weights

    print("Validation Accuracy after Pruning : ")
    acc = test_model(dataloaders['val'])
    neurons_pruned.append(len(neurons_to_be_removed))
    acc_after_pruning.append(acc)

print(neurons_pruned)
print(acc_after_pruning)
plt.plot([acc_before_pruning]*16)
plt.plot(acc_after_pruning)
plt.plot(neurons_pruned)
plt.xlabel('Threshold Angle for Pruning')
plt.legend(['Accuracy before pruning','Accuracy after pruning','Number of feature maps pruned'], loc='upper left')
plt.show()