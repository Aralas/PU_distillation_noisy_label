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

def main(threshold, criterion, lr, label):
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
    noisy_index = list(set(range(len(y_train)))-set(clean_index))
    
    # train loader cannot shuffle because we need the fixed indices
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set_new,
        batch_size=batch_size,
        shuffle=False,
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
    
    path = "saved_precision/threshold_%.2f_criterion_%d_lr_%.3f/"%(threshold, criterion, lr)
    model_path = "saved_model/threshold_%.2f_criterion_%d_lr_%.3f/"%(threshold, criterion, lr)
    
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # Initialize N binary classifiers
    for n in range(N_bagging):
        if n % 10 == 9:
            torch.cuda.empty_cache()
        net = ResNet18(num_classes=1).to(device)
        model_name = os.path.join(model_path, 'model%d.pkl' % n)
        torch.save(net.state_dict(), model_name)
    
    additional_data_index = []
    
    for k in range(K_iteration):
        y_pred_train = np.zeros((len(y_train), N_bagging))
        y_pred_valid = np.zeros((len(y_valid), N_bagging))

        # train N binary classifiers for one label
        for n in range(N_bagging):
            
            if n % 10 == 9:
                torch.cuda.empty_cache()
                
            print('iteration: %d, label: %d, model: %d' % (k, label, n))
            net = ResNet18(num_classes=1).to(device)
            model_name = os.path.join(model_path, 'model%d.pkl' % n)
            net.load_state_dict(torch.load(model_name))
            milestones = [20]
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
            
            # generate indices of bootstrap training set
            clean_pos_index = clean_index[label*clean_data_size:(label+1)*clean_data_size]
            if len(additional_data_index) < clean_data_size:
                noisy_pos_index = additional_data_index
            else:
                noisy_pos_index = np.random.choice(additional_data_index, clean_data_size, replace=False)
            
            n_pos = len(clean_pos_index) + len(noisy_pos_index)
            n_neg_clean = n_pos // 2
            n_neg_noisy = n_pos - n_neg_clean
            
            clean_neg_index = np.random.choice(list(set(clean_index) - set(clean_pos_index)), n_neg_clean, replace=False)
            noisy_neg_index = np.random.choice(list(set(noisy_index) - set(additional_data_index)), n_neg_noisy, replace=False)
            
            train_index = np.concatenate((clean_pos_index, noisy_pos_index, clean_neg_index, noisy_neg_index)).astype(int)
            y_slice = [1] * n_pos + [0] * n_pos
            
            bootstrap_train_set = Subset(train_set, train_index+len(y_valid))
            train_loader_bootstrap = torch.utils.data.DataLoader(
                dataset=bootstrap_train_set,
                batch_size=batch_size,
                shuffle=True,
            )

            y_train_new = np.array(train_set.targets)
            y_train_new[train_index+len(y_valid)] = y_slice
            train_loader_bootstrap.dataset.dataset.targets = y_train_new           
     
            for epoch in range(epoch_nums):        
                scheduler.step()
                net.train()
                loss_sum = 0
                start_time = time.time()
                for inputs, labels in train_loader_bootstrap:
                    
                    inputs, labels = inputs.to(device), labels.to(device)
                    labels = labels.float()
                    # labels = torch.reshape(labels, (-1, 1))
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    outputs = torch.sigmoid(outputs).reshape(-1)
                    loss = criterion(outputs, labels)
                    loss_sum += loss.item()
                    # mixup augmentation
                    inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, 0.5)
                    inputs, targets_a, targets_b = map(Variable, (inputs, labels_a, labels_b))
                    outputs = net(inputs)
                    outputs = torch.sigmoid(outputs).reshape(-1)
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                    loss_sum += loss.item()
                    loss.backward()
                    optimizer.step()

                net.eval()
                with torch.no_grad():
                    valid_acc = 0.0
                    valid_acc_pos = 0.0
                    valid_acc_neg = 0.0
                    total = 0
                    total_pos = 0.0
                    total_neg = 0.0

                    for x, y in valid_loader:
                        x, y = x.to(device), y.to(device)
                        outputs = net(x)
                        outputs = torch.sigmoid(outputs).reshape(-1)
                        valid_acc += ((outputs>0.5) == (y==label)).sum()
                        valid_acc_pos += ((outputs[y==label]>0.5) == (y[y==label]==label)).sum()
                        valid_acc_neg += ((outputs[y!=label]>0.5) == (y[y!=label]==label)).sum()
                        total += y.shape[0]
                        total_pos += y[y==label].shape[0]
                        total_neg += y[y!=label].shape[0]

                print('epoch: %d, train loss: %.03f, valid acc: %.03f, (positive: %.3f, negative: %.3f), time cost: %.1f sec' % (
                    epoch + 1, loss_sum / len(train_loader), valid_acc.item() / total, valid_acc_pos.item() / total_pos, 
                    valid_acc_neg.item() / total_neg, time.time() - start_time))        
                        
            torch.save(net.state_dict(), model_name)
            
            # Record the prediction
            net.eval()
            pred_train = []
            pred_valid = []
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                outputs = net(x)
                outputs = torch.sigmoid(outputs)
                pred_train += output.tolist()
            y_pred_train[:, n] = pre_train
            
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                outputs = net(x)
                outputs = torch.sigmoid(outputs)
                pred_valid += output.tolist()
            y_pred_valid[:, n] = pre_valid        

            # Generate additional data from training data set
            pos_pred_train = np.sum(y_pred_train > threshold, axis=1)
            add_index = np.where(pos_pred_train > criterion)[0]
            if len(add_index) > additional_data_limitation[1]:
                add_index = np.argsort(-pos_pred_train, axis=0)[0:additional_data_limitation[1]].reshape(-1)
            elif len(add_index) < additional_data_limitation[0]:
                add_index = np.argsort(-np.sum(y_pred_train, axis=1), axis=0)[0:additional_data_limitation[0]].reshape(-1)
            additional_data_index = add_index

            record_file = open(os.path.join(precision_dir, 'training_label%d.txt' % i), 'a+')
            record_file.write(str(additional_data_index) + '\n')
            record_file.close()

            # Test the precision on validation set
            pos_pred_valid = np.sum(y_pred_valid > threshold, axis=1)
            add_index_valid = np.where(pos_pred_valid > criterion)[0]           

            record_file = open(os.path.join(precision_dir, 'validation_label%d.txt' % i), 'a+')
            record_file.write(str(add_index_valid) + '\n')
            record_file.close()
            
            evaluate_file = open(os.path.join(precision_dir, 'evaluation_label%d.txt' % i), 'a+')
            train_precision = len(list(set(np.where(y_train == label)[0]) & set(additional_data_index)))/len(additional_data_index)
            evaluate_file.write('iteration: %d, training set precision: %.3f, number: %d'%(train_precision, len(additional_data_index)) + '\n')
            valid_precision = len(list(set(np.where(y_valid == label)[0]) & set(add_index_valid)))/(max(0.1, len(add_index_valid)))
            evaluate_file.write('iteration: %d, validation set precision: %.3f, number: %d'%(valid_precision, len(add_index_valid)) + '\n')
            evaluate_file.close()           

# gpu or cpu
device = torch.device("cuda")
    
batch_size = 64
epoch_nums = 30
lr_decay = 0.1
N_bagging = 20
K_iteration = 10
clean_data_size = 200
label = 2

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
for lr in [0.01]:
    for threshold in [0.8]:
        for criterion in [16]:
            main(threshold, criterion, lr, label)