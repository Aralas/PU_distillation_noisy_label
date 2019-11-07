#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2019/10/12 14:28

@author: Jingyi
"""

import time
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from copy import deepcopy
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

batch_size = 128
lr = 0.003
epoch_nums = 100
lr_decay = 0.5
noise = 0.5
seed = 99
np.random.seed(seed)

# gpu or cpu
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda")


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, **kwargs):
        super().__init__(**kwargs)
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.identity = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.identity = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
            )

    def forward(self, x):
        out = self.residual(x)
        out += self.identity(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_class=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, 2)
        self.layer3 = self._make_layer(128, 256, 2, 2)
        self.layer4 = self._make_layer(256, 512, 2, 2)
        self.avg_pool = nn.AvgPool2d(4)
        self.linear = nn.Linear(512, num_class)

    def _make_layer(self, in_channel, out_channel, bloch_num, stride=1):
        blocks = []
        blocks.append(ResidualBlock(in_channel, out_channel, stride))
        for i in range(1, bloch_num):
            blocks.append(ResidualBlock(out_channel, out_channel))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        return x


train_transform = transforms.Compose([
    transforms.Resize(40),
    transforms.RandomResizedCrop(32, scale=(0.64, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

valid_transform = transforms.Compose([
    transforms.Resize(40),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.Resize(40),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

train_set = torchvision.datasets.CIFAR10(
    root='./data/',
    train=True,
    transform=train_transform,
    download=True,
)


valid_set = torchvision.datasets.CIFAR10(
    root='./data/',
    train=True,
    transform=valid_transform,
    download=True,
)

num_train = len(train_set)
indices = list(range(num_train))
split = int(np.floor(0.2 * num_train))

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

y_train = np.array(train_set.targets[split:])
y_train_org = deepcopy(y_train)
# Generate clean dataset
clean_index = []
for i in range(10):
    positive_index = list(np.where(y_train == i)[0])
    clean_index = np.append(clean_index, np.random.choice(positive_index, 200, replace=False)).astype(int)

noisy_index = list(set(range(len(train_idx)))-set(clean_index))

# Add noise
num_noise = int(noise * len(noisy_index))
incorrect_index = np.random.choice(noisy_index, num_noise, replace=False)
label_slice = y_train[incorrect_index]
new_label = np.random.randint(low=0, high=10, size=num_noise)
while sum(label_slice == new_label) > 0:
    n = sum(label_slice == new_label)
    new_label[label_slice == new_label] = np.random.randint(low=0, high=10, size=n)
y_train[incorrect_index] = new_label
train_set.targets[split:] = y_train
print(np.mean(y_train == y_train_org))

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    sampler=train_sampler,
    batch_size=batch_size,
)

valid_loader = torch.utils.data.DataLoader(
    dataset=valid_set,
    sampler=valid_sampler,
    batch_size=batch_size,
)

test_set = torchvision.datasets.CIFAR10(
    root='./data/',
    train=False,
    transform=test_transform,
    download=True,
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False,
)

# net = torchvision.models.resnet18().to(device)
net = ResNet18(num_class=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

f = open("record/acc_lr_%.3f_noise_%.1f.txt" % (lr, noise), "a+")
for epoch in range(epoch_nums):
    # 每10个epoch就decay一下学习率
    if epoch > 0 and epoch % 10 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_decay

    loss_sum = 0
    start_time = time.time()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.long()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

    with torch.no_grad():
        test_acc = 0.0
        valid_acc = 0.0
        total = 0

        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.shape[0]
            test_acc += (predicted == y).sum()

        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            _, predicted = torch.max(outputs.data, 1)
            valid_acc += (predicted == y).sum()
    print('epoch: %d, train loss: %.03f, valid acc: %.03f, test acc: %.03f, time cost: %.1f sec' % (
        epoch + 1, loss_sum / len(train_loader), valid_acc.item() / total, test_acc.item() / total,
        time.time() - start_time))
    f.write('epoch: %d, train loss: %.03f, valid acc: %.03f, test acc: %.03f, time cost: %.1f sec \n' % (
        epoch + 1, loss_sum / len(train_loader), valid_acc.item() / total, test_acc.item() / total,
        time.time() - start_time))
    f.flush()
f.close()