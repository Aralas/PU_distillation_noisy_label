#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 7/13/2019 10:52 AM

@author: Jingyi

parameters:

clean_data_size: number of clean data for each class
noise_level: fraction of noisy labels for the rest data
data_augmentation: whether to use data augmentation in training
n: number of blocks in ResNet

"""

import os
from copy import deepcopy

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

from generate_CNN import CNN

'''
***** Set parameters *****
'''
seed = 99
noise_level = 0.8
clean_data_size = 200

batch_size = 32
epochs = 100
learning_rate = 0.003
data_augmentation = True

n = 3
depth = n * 9 + 2
file_index = 0

precision_dir = os.path.join(os.getcwd(), 'saved_precision')
precision_dir += '/benchmark_ResNet' + str(depth) + '_pre-activation/'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

# Add noise
num_noise = int(noise_level * y_train.shape[0])
noise_index = np.random.choice(y_train.shape[0], num_noise, replace=False)
label_slice = np.argmax(y_train[noise_index], axis=1)
new_label = np.random.randint(low=0, high=10, size=num_noise)
while sum(label_slice == new_label) > 0:
    n = sum(label_slice == new_label)
    new_label[label_slice == new_label] = np.random.randint(low=0, high=10, size=n)
y_train[noise_index] = tf.contrib.keras.utils.to_categorical(new_label, 10)


def clear_session():
    K.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

















if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

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
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
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
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
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
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        validation_data=(x_test, y_test),
                        epochs=epochs, verbose=1, workers=4,
                        callbacks=callbacks)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])





'''
## Train teacher model on additional clean data
'''

'''
## Train student model
'''

