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

np.set_printoptions(threshold=np.inf)


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


def main(positive_threshold, add_criterion, label, lr):  
    seed = 99
    np.random.seed(seed)
    
    path_name = '/positive_threshold_%.2f_add_criterion_%d_learning_rate_%.3f_without_validation'%(positive_threshold, add_criterion, lr)
    model_dir = 'saved_model/' + path_name + '_label' + str(label)
    precision_dir = os.path.join(os.getcwd(), 'saved_precision')
    precision_dir += path_name

    print(model_dir)
    print(precision_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(precision_dir):
        os.makedirs(precision_dir)
    
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
#     valid_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std),
#     ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='./data/',
        train=True,
        transform=train_transform,
        download=True,
    )

#     valid_set = torchvision.datasets.CIFAR10(
#         root='./data/',
#         train=True,
#         transform=valid_transform,
#         download=True,
#     )
    
    test_set = torchvision.datasets.CIFAR10(
        root='./data/',
        train=False,
        transform=test_transform,
        download=True,
    )

#     num_train = len(train_set)
#     indices = list(range(num_train))
#     split = int(np.floor(0.2 * num_train))

#     train_idx, valid_idx = indices[split:], indices[:split]
#     train_sampler = SubsetRandomSampler(train_idx)
#     valid_sampler = SubsetRandomSampler(valid_idx)

#     y_train = np.array(train_set.targets[split:])
#     y_train_orig = deepcopy(y_train)
    # Generate clean dataset
    clean_index = []
    y_train = np.array(train_set.targets)
    for i in range(10):
        positive_index = list(np.where(y_train == i)[0])
        clean_index = np.append(clean_index, np.random.choice(positive_index, 200, replace=False)).astype(int)

    noisy_index = list(set(range(len(y_train)))-set(clean_index))
    
    train_all_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        # sampler=train_sampler,
        batch_size=batch_size,
        shuffle=False,
    )
    
#     valid_loader = torch.utils.data.DataLoader(
#         dataset=valid_set,
#         sampler=valid_sampler,
#         batch_size=batch_size,
#         shuffle=False,
#     )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
    )
    
    additional_data_index = []
    binary_classifier_list = []
    
    # Initialize n models
    for n in range(N_bagging):
        if n % 10 == 9:
            torch.cuda.empty_cache()
        net = ResNet18(num_classes=1).to(device)
        filepath = os.path.join(model_dir, 'model%d.pkl' % n)
        torch.save(net.state_dict(), filepath)
    
    for k in range(K_iteration):
        y_pred_train = np.zeros((len(y_train), N_bagging))
#         y_pred_validation = np.zeros((int(0.2 * num_train), N_bagging))
    
        # train N binary classifiers for one label
        for n in range(N_bagging):
            if n % 10 == 9:
                torch.cuda.empty_cache()
            print('iteration: %d, label: %d, model: %d' % (k, label, n))
            
            milestones = [15]
            net = ResNet18(num_classes=1).to(device)
            criterion = nn.BCELoss()
            # criterion = nn.MSELoss()
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
            filepath = os.path.join(model_dir, 'model%d.pkl' % n)
            net.load_state_dict(torch.load(filepath))
            train_pos_index_clean = clean_index[clean_data_size*label:clean_data_size*(label+1)]

            if len(additional_data_index) < clean_data_size:
                add_positive_index = additional_data_index
            else:
                add_positive_index = np.random.choice(additional_data_index, clean_data_size, replace=False)

            n_pos = len(train_pos_index_clean) + len(add_positive_index)
            n_neg_clean = n_pos // 2
            n_neg_noisy = n_pos - n_neg_clean
            clean_index_neg = list(set(clean_index) - set(train_pos_index_clean))
            train_neg_index_clean = np.random.choice(clean_index_neg, n_neg_clean, replace=False)
            neg_index_noisy = list(set(np.arange(len(y_train))) - set(additional_data_index))
            train_neg_index_noisy = np.random.choice(neg_index_noisy, n_neg_noisy, replace=False)
            train_index = np.concatenate([train_pos_index_clean, add_positive_index, train_neg_index_clean, train_neg_index_noisy]).astype(int)
#             train_index += int(0.2 * num_train) # the first 10000 samples are use as validation set
            
#             y_train = np.array(train_set.targets)

#             y_train[train_index] = [1] * n_pos + [0] * n_pos
#             train_set.targets = y_train
            bagging_train_set = torch.utils.data.Subset(train_set, train_index)
            train_loader = torch.utils.data.DataLoader(
                dataset=bagging_train_set,
                batch_size=batch_size,
                shuffle=True,
            )
        
            y_train_new = np.array(train_loader.dataset.dataset.targets)
            y_train_new[train_index] = [1] * n_pos + [0] * n_pos
            
            train_loader.dataset.dataset.targets = y_train_new
            
            for epoch in range(epoch_nums):
                scheduler.step()
                net.train()
                loss_sum = 0
                start_time = time.time()

                for inputs, labels in train_loader:             
                    labels = labels.float()
                    labels = labels.reshape((-1, 1))
                    inputs, labels = inputs.to(device), labels.to(device)                
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    outputs = torch.sigmoid(outputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.item()
            
                net.eval()
                with torch.no_grad():
                    test_acc = 0.0
                    total = 0
                    test_acc1 = 0.0
                    test_acc2 = 0.0
                    total1 = 0
                    total2 = 0

                    for x, y in test_loader:
                        x, y = x.to(device), y.to(device)
                        outputs = net(x)
                        outputs = torch.sigmoid(outputs)
                        total += y.shape[0]
                        test_acc += ((outputs>=0.5).reshape((1,-1)) == (y==label)).sum()
                        total1 += y[y==label].shape[0]
                        total2 += y[y!=label].shape[0]
                        test_acc1 += (outputs[y==label]>=0.5).sum()
                        test_acc2 += (outputs[y!=label]<0.5).sum()
                print('epoch: %d, train loss: %.03f, test acc: %.03f, positive %.03f, negative %.03f, time cost: %.1f sec' % (epoch + 1, loss_sum / len(train_loader), test_acc.item() / total, test_acc1.item() / total1, test_acc2.item() / total2, time.time() - start_time))
            
            torch.save(net.state_dict(), filepath)
            
            net.eval()
            pred_train = []
            pred_val = []
            with torch.no_grad():
                for x, y in train_all_loader:
                    x, y = x.to(device), y.to(device)
                    outputs = net(x)
                    outputs = torch.sigmoid(outputs)
                    pred_train += outputs.tolist()
                    
#                 for x, y in valid_loader:
#                     x, y = x.to(device), y.to(device)
#                     outputs = net(x)
#                     outputs = torch.sigmoid(outputs)
#                     pred_val += outputs.tolist()  
            y_pred_train[:, n] = np.array(pred_train).flatten()
#             y_pred_validation[:, n] = np.array(pred_val).flatten()

        # Generate additional data from training data set
        y_pred1 = np.sum(y_pred_train > positive_threshold, axis=1)
        add_index = np.where(y_pred1 > add_criterion)[0]
        if len(add_index) > additional_data_limitation[1]:
            add_index = np.argsort(-y_pred1, axis=0)[0:additional_data_limitation[1]].reshape(-1)
        elif len(add_index) < additional_data_limitation[0]:
            y_pred2 = np.sum(y_pred_train, axis=1)
            add_index = np.argsort(-y_pred2, axis=0)[0:additional_data_limitation[0]].reshape(-1)
        additional_data_index = add_index

        record_file = open(os.path.join(precision_dir, 'training_label%d.txt' % label), 'a+')
        record_file.write(str(additional_data_index) + '\n')
        record_file.close()

        # Test the precision on validation set
#         y_pred1 = np.sum(y_pred_validation > positive_threshold, axis=1)
#         add_index = np.where(y_pred1 > add_criterion)[0]

#         record_file = open(os.path.join(precision_dir, 'validation_label%d.txt' % label), 'a+')
#         record_file.write(str(add_index) + '\n')
#         record_file.close()

# gpu or cpu
device = torch.device("cuda")

batch_size = 64
epoch_nums = 20
lr_decay = 0.1

clean_data_size = 250
additional_data_limitation = [100, 4000]
K_iteration = 10
N_bagging = 100    
    
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

positive_threshold = 0.9
add_criterion = 90

label = 0
lr = 0.01
main(positive_threshold, add_criterion, label, lr)
