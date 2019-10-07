#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 7/28/2019 10:49 PM

@author: Jingyi
"""

from __future__ import print_function

import os
from copy import deepcopy

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.datasets import cifar10
from keras.layers import AveragePooling2D, Input, Flatten
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

'''
***** Set parameters *****
'''
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)
np.set_printoptions(threshold=np.inf)

seed = 99
np.random.seed(seed)
tf.set_random_seed(seed)

noise_level = 0.5
student_lambda = 0.2
student_threshold = 0.8
clean_data_size = 200

batch_size = 64
epochs = 500
learning_rate = 0.001
data_augmentation = True
additional_index = 1

n = 2
depth = n * 9 + 2
file_index = 2

path_name = '/student_model_dynamic' + str(additional_index)
model_dir = os.path.join(os.getcwd(), 'saved_models')
model_dir += path_name
precision_dir = os.path.join(os.getcwd(), 'saved_precision')
precision_dir += path_name

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

# Add noise
num_noise = int(noise_level * y_train.shape[0])
noise_index = np.random.choice(y_train.shape[0], num_noise, replace=False)
label_slice = np.argmax(y_train[noise_index], axis=1)
new_label = np.random.randint(low=0, high=10, size=num_noise)
while sum(label_slice == new_label) > 0:
    n = sum(label_slice == new_label)
    new_label[label_slice == new_label] = np.random.randint(low=0, high=10, size=n)
y_train[noise_index] = tf.contrib.keras.utils.to_categorical(new_label, 10)

# Correct labels of additional data
file = open('additional_data_index' + str(additional_index) + '.txt')
lines = file.readlines()
precision = '['
for line in lines:
    precision += line.replace('\n', '').replace(' ', ',')
precision += ']'
precision = eval(
    precision.replace('][', '], [').replace(',,,,', ',').replace(',,,', ',').replace(',,', ',').replace('[,', '['))

for label in range(10):
    if label != 3:
        index = precision[label]
        y_train[index] = tf.contrib.keras.utils.to_categorical([label] * len(index), 10)

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
    if epoch > 450:
        lr *= 0.5e-3
    elif epoch > 400:
        lr *= 1e-3
    elif epoch > 300:
        lr *= 1e-2
    elif epoch > 200:
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


def resnet_v2(input_shape, depth, num_classes=10):
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
                  optimizer=Adam(lr=lr_schedule(0)),
                  metrics=['accuracy'])
    return model


"""
# Get pseudo label.
"""
teacher_path_name = '/teacher_model_bagging1'
teacher_model_dir = os.path.join(os.getcwd(), 'saved_models')
teacher_model_dir += teacher_path_name

y_pred = np.zeros((20, len(x_train), 10))
teacher_model = resnet_v2(input_shape, depth, num_classes)
for teacher_index in range(20):
    model_name = 'cifar10_file%d.h5' % teacher_index
    filepath = os.path.join(teacher_model_dir, model_name)    
    teacher_model.load_weights(filepath)
    y_pred[teacher_index] = teacher_model.predict(x_train)
y_pred = np.mean(y_pred, axis=0)
y_max = np.max(y_pred, axis=1)
index_high = np.where(y_max >= student_threshold)[0]
index_low = np.where(student_threshold > y_max)[0]
index_high_acc = np.mean(np.argmax(y_pred[index_high], axis=1)==np.argmax(y_train_orig[index_high], axis=1))
print("accuracy of high confident samples: %.3f"%index_high_acc)

y_pseudo = deepcopy(y_train)
y_pseudo[index_high] = student_lambda * y_train[index_high] + (1 - student_lambda) * y_pred[index_high]
y_pseudo[index_low] = student_lambda * y_pred[index_low] + (1 - student_lambda) * y_train[index_low]

x_train = np.concatenate((x_train, x_clean), axis=0)
y_pseudo = np.concatenate((y_pseudo, y_clean), axis=0)

K.clear_session()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

"""
# Generate model.
"""
model = resnet_v2(input_shape, depth, num_classes)
model_name = 'cifar10_noise_%.1f_lambda_%.2f_threshold_%.2f_file%d.h5' % (noise_level, student_lambda, student_threshold, file_index)
filepath = os.path.join(model_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

# Run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    History = model.fit(x_train, y_pseudo,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_validation, y_validation),
                        shuffle=True,
                        callbacks=callbacks)
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
    History = model.fit_generator(datagen.flow(x_train, y_pseudo, batch_size=batch_size),
                                  steps_per_epoch=int(x_train.shape[0]/50),
                                  validation_data=(x_validation, y_validation),
                                  epochs=epochs, verbose=1, workers=4,
                                  callbacks=callbacks)

# Score trained model.
model.load_weights(filepath)
scores = model.evaluate(x_test, y_test, verbose=1)

# Record accuracy.
record_file = open(os.path.join(precision_dir, 'noise_%.1f_lambda_%.2f_accuracy_file%d.txt' % (noise_level, student_lambda, file_index)), 'a+')
record_file.write('training accuracy\n' + str(History.history['acc']))
record_file.write('\nvalidation accuracy\n' + str(History.history['val_acc']))
record_file.write('\ntest accuracy\n' + str(scores[1]))
record_file.close()
