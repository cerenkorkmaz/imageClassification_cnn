import torchvision
import torchvision.datasets as datasets
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, Sequential, Flatten, Dropout, Linear, ReLU, BatchNorm2d, LogSoftmax, NLLLoss
from collections import OrderedDict
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns

folder_path = r"D:\2022-BAHAR\416\ass3\dataset"
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

transform = transforms.Compose(
    [transforms.Resize((64, 64)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = datasets.ImageFolder(root=folder_path, transform=transform)
dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)  # %80 for train
size = int(dataset_size * 0.2)  # % 10 for validation and test

trainset, set = torch.utils.data.random_split(dataset, [train_size, size])
new_test_size = int(size * 0.5)  # %10 for test
valid_size = int(size * 0.5)  # %10 for validation
testset, validset = torch.utils.data.random_split(set, [new_test_size, valid_size])


def CNN(batch_size, learning_rate, epoch_value):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5, 1, 2)  # input size=3 output size=6 kernel=5x5 stride=1 padding=2
            self.pool = nn.MaxPool2d(2, 2)  # pool to downsize image
            self.conv2 = nn.Conv2d(6, 16, 5, 1, 2)  # input size=6 output size=16 kernel=5x5 stride=1 padding=2
            self.conv3 = nn.Conv2d(16, 10, 5, 1, 2)  # input size=16 output size=10 kernel=5x5 stride=1 padding=2
            self.conv4 = nn.Conv2d(10, 10, 5, 1, 2)  # input size=10 output size=10 kernel=5x5 stride=1 padding=2
            self.conv5 = nn.Conv2d(10, 5, 5, 1, 2)  # input size=10 output size=5 kernel=5x5 stride=1 padding=2
            self.fc1 = nn.Linear(20, 15)  # features=20 output size(classes)=15
            # self.dropout = nn.Dropout(0.10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))  # pool to downsize image
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = self.pool(F.relu(self.conv4(x)))
            x = self.pool(F.relu(self.conv5(x)))
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            # x = self.dropout(x)
            x = F.relu(self.fc1(x))  # activation function RELU

            return x

    net = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), learning_rate, momentum=0.9)

    train_loss, valid_loss = [], []
    path = r"D:\2022-BAHAR\416\ass3\dataset\model.pth"
    min_valid_loss = np.inf
    for epoch in range(epoch_value):  # loop over the dataset multiple times

        running_loss = 0.0
        t_loss = 0.0
        for i, (data, labels) in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            t_loss += loss.item()
            if i % 250 == 249:  # print every 250 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 250 :.3f}')
                running_loss = 0.0

        v_loss = 0.0
        for j, (data, labels) in enumerate(validloader):
            inputs, labels = data.to(device), labels.to(device)
            outputs = net(inputs)
            loss_v = criterion(outputs, labels)
            v_loss += loss_v.item()

        v_loss = v_loss / len(validloader)
        t_loss = t_loss / len(trainloader)
        train_loss.append(t_loss)
        valid_loss.append(v_loss)
        print(f'Epoch {epoch + 1} Validation Loss: {v_loss}')
        if min_valid_loss > v_loss:
            torch.save(net.state_dict(), path)  # save if valid loss decreased
            print("Validation loss has decreased. Saving the model...")
            min_valid_loss = v_loss

    print('Finished Training')
    list1 = list(range(0, len(train_loss)))
    plt.plot(train_loss, list1)
    plt.title('Train Loss Change')
    plt.show()
    list2 = list(range(0, len(valid_loss)))
    plt.plot(valid_loss, list2)
    plt.title('Validation Loss Change')
    plt.show()

    net.eval()
    predictions = []
    labels = []
    for data, target in testloader:
        for label in target.data.numpy():
            labels.append(label)
        for prediction in net(data).data.numpy().argmax(1):
            predictions.append(prediction)

    cm = confusion_matrix(labels, predictions)
    df_cm = pd.DataFrame(cm, index=CATEGORIES, columns=CATEGORIES)
    plt.figure(figsize=(16, 16))
    sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
    plt.xlabel('Prediction', fontsize=15)
    plt.ylabel('True', fontsize=15)
    plt.show()

    net = Net().to(device)
    net.load_state_dict(torch.load(path))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data, labels in validloader:
            images, labels = data.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct // total
    print(f'Accuracy of the network on the validation images: {100 * correct // total} %')

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data, labels in testloader:
            images, labels = data.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct // total
    print(f'Accuracy of the network on the test images: {100 * correct // total} %')


def Resnet(batch_size, learning_rate, epoch_value):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)

    resnet18 = torchvision.models.resnet18(pretrained=True).to(device)
    for param in resnet18.parameters():
        param.requires_grad = False

    # last_layer = Sequential(OrderedDict([
    #    ('conv1', Conv2d(256, 25, kernel_size=3, stride=1, padding=2)),
    #    ('conv2', Conv2d(25, 25, kernel_size=3, stride=1, padding=2)),
    #    ('relu', ReLU()),
    #    ('pool1', MaxPool2d(2, 2)),
    #    ('conv3', Conv2d(25, 10, kernel_size=3, stride=1, padding=2)),
    #    ('conv4', Conv2d(10, 256, kernel_size=3, stride=1, padding=2)),
    #    ('conv5', Conv2d(256, 512, kernel_size=3, stride=1, padding=2)),
    #    ('relu', ReLU())
    # ]))
    fc = Sequential(OrderedDict([
        ('fc1', Linear(512, 256)),
        ('relu', ReLU()),
        ('fc2', Linear(256, 15)),
        ('output', LogSoftmax(dim=1))
    ]))

    # resnet18.layer4 = last_layer
    resnet18.fc = fc

    net = resnet18.to(device)

    optimizer = optim.SGD(resnet18.fc.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_loss, valid_loss = [], []
    path = r"D:\2022-BAHAR\416\ass3\dataset\resnet_model.pth"
    min_valid_loss = np.inf
    for epoch in range(epoch_value):  # loop over the dataset multiple times

        running_loss = 0.0
        t_loss = 0.0
        for i, (data, labels) in enumerate(trainloader):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            t_loss += loss.item()
            if i % 250 == 249:  # print every 250 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 250 :.3f}')
                running_loss = 0.0

        v_loss = 0.0
        for j, (data, labels) in enumerate(validloader):
            inputs, labels = data.to(device), labels.to(device)
            outputs = net(inputs)
            loss_v = criterion(outputs, labels)
            v_loss += loss_v.item()

        v_loss = v_loss / len(validloader)
        t_loss = t_loss / len(trainloader)
        train_loss.append(t_loss)
        valid_loss.append(v_loss)
        print(f'Epoch {epoch + 1} Validation Loss: {v_loss}')
        if min_valid_loss > v_loss:
            torch.save(net.state_dict(), path)  # save if valid loss decreased
            print("Validation loss has decreased. Saving the model...")
            min_valid_loss = v_loss

    print('Finished Training')
    list1 = list(range(0, len(train_loss)))
    plt.plot(train_loss, list1)
    plt.title('Train Loss Change')
    plt.show()
    list2 = list(range(0, len(valid_loss)))
    plt.plot(valid_loss, list2)
    plt.title('Validation Loss Change')
    plt.show()

    net.eval()
    predictions = []
    labels = []
    for data, target in testloader:
        for label in target.data.numpy():
            labels.append(label)
        for prediction in net(data).data.numpy().argmax(1):
            predictions.append(prediction)

    cm = confusion_matrix(labels, predictions)
    df_cm = pd.DataFrame(cm, index=CATEGORIES, columns=CATEGORIES)
    plt.figure(figsize=(16, 16))
    sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
    plt.xlabel('Prediction', fontsize=15)
    plt.ylabel('True', fontsize=15)
    plt.show()

    net = resnet18.to(device)
    net.load_state_dict(torch.load(path))
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data, labels in testloader:
            images, labels = data.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct // total
    print(f'Accuracy of the network on the test images: {100 * correct // total} %')


BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCH_COUNT = 20
CATEGORIES = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot',
              'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

# cnn accuracy
CNN(BATCH_SIZE, LEARNING_RATE, EPOCH_COUNT)

# resnet accuracy
Resnet(BATCH_SIZE, LEARNING_RATE, EPOCH_COUNT)
