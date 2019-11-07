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
from torch.utils.data.dataset import Subset
from PreResNet import ResNet18



def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def main(lr, index, mixup=False):
    seed = 99
    np.random.seed(seed)
    torch.manual_seed(seed)

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
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

    valid_set = torchvision.datasets.CIFAR10(
        root='./data/',
        train=True,
        transform=valid_transform,
        download=True,
    )
    
    test_set = torchvision.datasets.CIFAR10(
        root='./data/',
        train=False,
        transform=test_transform,
        download=True,
    )

    # Split validation set (the first 20% samples) and training set
    num_train = len(train_set)
    indices = list(range(num_train))
    split = int(np.floor(0.2 * num_train))

    train_idx, valid_idx = indices[split:], indices[:split]
    train_set_new = Subset(train_set, train_idx)
    valid_set_new = Subset(valid_set, valid_idx)
    
    y_train = np.array(train_set.targets)[train_idx]
    y_train_orig = np.array(train_set.targets)[train_idx]
    y_valid = np.array(valid_set.targets)[valid_idx]
            
    # Generate clean dataset
    clean_index = []
    for i in range(10):
        positive_index = list(np.where(y_train == i)[0])
        clean_index = np.append(clean_index, np.random.choice(positive_index, clean_data_size, replace=False)).astype(int)

    # Get Additional Data Index
    file = open('additional_data_index.txt')
    lines = file.readlines()
    add_index = '['
    for line in lines:
        add_index += line.replace('\n', '').replace(' ', ',')
    add_index += ']'
    add_index = eval(add_index.replace('][', '], [').replace(',,,,', ',').replace(',,,', ',').replace(',,', ',').replace('[,', '['))
    
    add_number = []
    for label in range(10):
        add_number.append(len(add_index[label]))
    bootstrap_size = min(add_number)
    

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set_new,
        batch_size=batch_size,
        shuffle=True,
    )
    
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_set_new,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
    )    

    # net = torchvision.models.resnet18().to(device)
    milestones = [30, 50, 80]
    net = ResNet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    if mixup:
        path = "record/teacher_mixup_new/"
        model_path = "saved_model/teacher_mixup_new/"
    else:
        path = "record/teacher_new/"
        model_path = "saved_model/teacher_new/"
    
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    f = open(path + "acc_lr_%.3f_%d.txt" % (lr, index), "a+")
    model_name = model_path + 'lr_%.3f_%d.pkl'%(lr, index)
    
    for epoch in range(epoch_nums):        
        # Generate indices of bootstrap subset
        train_index = clean_index
        y_slice = sum([[i]*clean_data_size for i in range(10)], [])
        for label in range(10):
            index = add_index[label]
            index_bootstrap = np.random.choice(index, bootstrap_size, replace=False) 
            train_index = np.concatenate([train_index, index_bootstrap])
            y_slice += [label]*bootstrap_size
        
        bootstrap_train_set = Subset(train_set, train_index+len(y_valid))
        train_loader_bootstrap = torch.utils.data.DataLoader(
            dataset=bootstrap_train_set,
            batch_size=batch_size,
            shuffle=True,
        )

        y_train_new = np.array(train_set.targets)
        y_train_new[train_index+len(y_valid)] = y_slice
        train_loader_bootstrap.dataset.dataset.targets = y_train_new
        print('percentage of correct labels:', np.mean(y_train_new[train_index+len(y_valid)] == y_train_orig[train_index]))
        
        scheduler.step()
        net.train()
        loss_sum = 0
        start_time = time.time()
        for inputs, labels in train_loader_bootstrap:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.long()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()            
            loss_sum += loss.item()
            if mixup:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, 0.5)
                inputs, targets_a, targets_b = map(Variable, (inputs, labels_a, labels_b))
                outputs = net(inputs)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                loss_sum += loss.item()
                optimizer.zero_grad()
                loss.backward()
            optimizer.step()
                         
        net.eval()
        with torch.no_grad():
            best_val_acc = 0.0
            test_acc = 0.0
            valid_acc = 0.0
            total = 0

            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = net(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.shape[0]
                acc = (predicted == y).sum()
                test_acc += acc

            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                outputs = net(x)
                _, predicted = torch.max(outputs.data, 1)
                acc = (predicted == y).sum()
                valid_acc += acc
                
            if (valid_acc.item() / total) > best_val_acc:
                best_val_acc = valid_acc.item() / total
                torch.save(net.state_dict(), model_name)
        print('epoch: %d, train loss: %.03f, valid acc: %.03f, test acc: %.03f, time cost: %.1f sec' % (
            epoch + 1, loss_sum / len(train_loader), valid_acc.item() / total, test_acc.item() / total,
            time.time() - start_time))
        f.write('epoch: %d, train loss: %.03f, valid acc: %.03f, test acc: %.03f, time cost: %.1f sec \n' % (
            epoch + 1, loss_sum / len(train_loader), valid_acc.item() / total, test_acc.item() / total,
            time.time() - start_time))
        f.flush()
    f.close()


# gpu or cpu
device = torch.device("cuda")
    
batch_size = 128
epoch_nums = 100
lr_decay = 0.1
clean_data_size = 200

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
for lr in [0.1]:
    for index in range(4, 10):
        for mixup in [True]:
            main(lr, index, mixup)