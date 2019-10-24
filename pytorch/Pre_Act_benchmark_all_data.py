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
epoch_nums = 100
lr_decay = 0.1
seed = 99
np.random.seed(seed)

# gpu or cpu
device = torch.device("cuda")


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, lin=0, lout=5):
        out = x
        if lin < 1 and lout > -1:
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
        if lin < 2 and lout > 0:
            out = self.layer1(out)
        if lin < 3 and lout > 1:
            out = self.layer2(out)
        if lin < 4 and lout > 2:
            out = self.layer3(out)
        if lin < 5 and lout > 3:
            out = self.layer4(out)
        if lout > 4:
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(PreActBlock, [2,2,2,2], num_classes=num_classes)


def main(lr, noise, index):
    train_transform = transforms.Compose([
        transforms.Resize(40),
        transforms.RandomResizedCrop(32, scale=(0.64, 1.0)),
        transforms.RandomHorizontalFlip(),
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
    
    test_set = torchvision.datasets.CIFAR10(
        root='./data/',
        train=False,
        transform=test_transform,
        download=True,
    )

    y_train = np.array(train_set.targets)
    y_train_orig = deepcopy(y_train)
    # Generate clean dataset
    clean_index = []
    for i in range(10):
        positive_index = list(np.where(y_train == i)[0])
        clean_index = np.append(clean_index, np.random.choice(positive_index, 200, replace=False)).astype(int)

    noisy_index = list(set(range(len(y_train)))-set(clean_index))

    # Add noise
    num_noise = int(noise * len(noisy_index))
    incorrect_index = np.random.choice(noisy_index, num_noise, replace=False)
    label_slice = y_train[incorrect_index]
    new_label = np.random.randint(low=0, high=10, size=num_noise)
    while sum(label_slice == new_label) > 0:
        n = sum(label_slice == new_label)
        new_label[label_slice == new_label] = np.random.randint(low=0, high=10, size=n)
    y_train[incorrect_index] = new_label
    train_set.targets = y_train
    print(np.mean(y_train == y_train_orig))

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        shuffle=True,
        batch_size=batch_size,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
    )

    # net = torchvision.models.resnet18().to(device)
    net = ResNet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    path = "record/PreAct_Benchmark_all_data/"
    if not os.path.exists(path):
        os.makedirs(path)
    f = open(path + "acc_lr_%.3f_noise_%.1f_%d.txt" % (lr, noise, index), "a+")
    for epoch in range(epoch_nums):
        # 每10个epoch就decay一下学习率
        if epoch > 0 and epoch % 30 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] * lr_decay, 1e-5)

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

        print('epoch: %d, train loss: %.03f, test acc: %.03f, time cost: %.1f sec' % (
            epoch + 1, loss_sum / len(train_loader), test_acc.item() / total,
            time.time() - start_time))
        f.write('epoch: %d, train loss: %.03f, test acc: %.03f, time cost: %.1f sec \n' % (
            epoch + 1, loss_sum / len(train_loader), test_acc.item() / total,
            time.time() - start_time))
        f.flush()
    f.close()

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
for lr in [0.2]:
    for noise in [0.3, 0.5, 0.7]:
        for index in range(1):
            main(lr, noise, index)