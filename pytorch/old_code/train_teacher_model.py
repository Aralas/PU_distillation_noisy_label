#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
lr_decay = 0.5
seed = 99
np.random.seed(seed)

# gpu or cpu
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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


def main(lr, noise):
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
    
for lr in [0.003]:
    for noise in [0.7, 0.9]:
        main(lr, noise)

















'''
***** Set parameters *****
'''
seed = 99
noise_level = 0
clean_data_size = 200

batch_size = 64
epochs = 50
learning_rate = 0.001
data_augmentation = True
bagging = True

n = 2
depth = n * 9 + 2
file_index = 0

if not bagging:
    path_name = '/teacher_model'
else:
    path_name = '/teacher_model_bagging'
model_dir = os.path.join(os.getcwd(), 'saved_models')
model_dir += path_name
precision_dir = os.path.join(os.getcwd(), 'saved_precision')
precision_dir += path_name

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

np.set_printoptions(threshold=np.inf)

np.random.seed(seed)
tf.set_random_seed(seed)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(precision_dir):
    os.makedirs(precision_dir)

'''
***** Load data *****
'''
# Load the CIFAR10 data.
(x_train_, y_train_), (x_test, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices.
num_classes = 10
y_train_ = keras.utils.to_categorical(y_train_, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Input image dimensions.
input_shape = x_train_.shape[1:]

# Normalize data.
x_train_ = x_train_.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Subtract pixel mean is enabled
x_train_mean = np.mean(x_train_, axis=0)
x_train_ -= x_train_mean
x_test -= x_train_mean

# Seperate validation set
x_validation = x_train_[0:10000, ]
y_validation = y_train_[0:10000, ]
x_train = x_train_[10000:50000, ]
y_train = y_train_[10000:50000, ]

# Generate clean dataset
clean_index = []
for i in range(10):
    positive_index = list(np.where(y_train[:, i] == 1)[0])
    clean_index = np.append(clean_index, np.random.choice(positive_index, clean_data_size, replace=False)).astype(
        int)

x_clean = x_train[clean_index]
y_clean = y_train[clean_index]
x_train = np.delete(x_train, clean_index, axis=0)
y_train = np.delete(y_train, clean_index, axis=0)
y_train_orig = deepcopy(y_train)

# Add additional data
add_number = []
file = open('additional_data_index.txt')
lines = file.readlines()
precision = '['
for line in lines:
    precision += line.replace('\n', '').replace(' ', ',')
precision += ']'
precision = eval(
    precision.replace('][', '], [').replace(',,,,', ',').replace(',,,', ',').replace(',,', ',').replace('[,', '['))

for label in range(10):
    add_number.append(len(precision[label]))
bootstrap_size = min(add_number)


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = learning_rate
    if epoch > 40:
        lr *= 0.5e-3
    elif epoch > 30:
        lr *= 1e-3
    elif epoch > 20:
        lr *= 1e-2
    elif epoch > 10:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v2(input_shape, depth, lr=lr_schedule(0), num_classes=10):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2  # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr),
                  metrics=['accuracy'])
    return model


"""
# Generate model.
"""
model = resnet_v2(input_shape, depth)
model_name = 'cifar10_file%d.h5' % file_index
filepath = os.path.join(model_dir, model_name)

# Run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    # History = model.fit(x_clean, y_clean,
    #                   batch_size=batch_size,
    #                   epochs=epochs,
    #                   validation_data=(x_validation, y_validation),
    #                   shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=15,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.1,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    train_acc = []
    val_acc = []
    test_acc = []
    np.random.seed(seed + file_index)

    for epoch in range(epochs):
        x = deepcopy(x_clean)
        y = deepcopy(y_clean)
        for label in range(10):
            index = precision[label]
            if bagging:
                index = np.random.choice(index, bootstrap_size, replace=False)           
            x = np.concatenate((x, x_train[index]), axis=0)
            y = np.concatenate((y, tf.contrib.keras.utils.to_categorical([label] * len(index), 10)), axis=0)

        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=lr_schedule(epoch)),
                      metrics=['accuracy'])
        History = model.fit_generator(datagen.flow(x, y, batch_size=batch_size),
                                      steps_per_epoch=x.shape[0],
                                      validation_data=(x_validation, y_validation),
                                      epochs=1, verbose=1, workers=4)
        val_acc.append(History.history['val_acc'][0])
        train_acc.append(History.history['acc'][0])
        test_acc.append(model.evaluate(x_test, y_test)[1])
        print('epoch:%d, training accuracy:%.3f, validation accuracy:%.3f, test accuracy:%.3f'
              % (epoch, train_acc[-1], val_acc[-1], test_acc[-1]))
        if val_acc[-1] >= np.max(val_acc):
            model.save_weights(filepath)

# Record accuracy.
record_file = open(os.path.join(precision_dir, 'accuracy_file%d.txt' % file_index), 'a+')
record_file.write('training accuracy\n' + str(train_acc))
record_file.write('\nvalidation accuracy\n' + str(val_acc))
record_file.write('\ntest accuracy\n' + str(test_acc))
record_file.close()
