# coding: utf-8


import tensorflow as tf
import os
import numpy as np
from copy import deepcopy
import sys

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

clean_data_size = 250

# load data from CIFAR10
def load_data(clean_data_size):
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()    
    x_train = preprocess_input(x_train)
    x_test = preprocess_input(x_test)

    # transform labels to one-hot vectors
    y_train = tf.contrib.keras.utils.to_categorical(y_train, 10)
    y_test = tf.contrib.keras.utils.to_categorical(y_test, 10)

    clean_index = []
    for label in range(10):
        positive_index = list(np.where(y_train[:, label] == 1)[0])
        clean_index = np.append(clean_index, np.random.choice(positive_index, clean_data_size, replace=False)).astype(
            int)

    x_clean = x_train[clean_index]
    y_clean = y_train[clean_index]
    x_train = np.delete(x_train, clean_index, axis=0)
    y_train = np.delete(y_train, clean_index, axis=0)
    return x_train, y_train, x_test, y_test, x_clean, y_clean


def generate_noise_labels(y_train, noise_level):
    num_noise = int(noise_level * y_train.shape[0])
    noise_index = np.random.choice(y_train.shape[0], num_noise, replace=False)
    label_slice = np.argmax(y_train[noise_index], axis=1)
    new_label = np.random.randint(low=0, high=10, size=num_noise)
    while sum(label_slice == new_label) > 0:
        n = sum(label_slice == new_label)
        new_label[label_slice == new_label] = np.random.randint(low=0, high=10, size=n)
    y_train[noise_index] = tf.contrib.keras.utils.to_categorical(new_label, 10)
    return y_train





def run_benchmark(file_index, noise_level, learning_rate, seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)

    x_train, y_train, x_test, y_test, x_clean, y_clean = load_data(clean_data_size)
    y_train_orig = deepcopy(y_train)
    y_train = generate_noise_labels(y_train, noise_level)

    # create record files
    dirs = 'record_new_preprocessing/benchmark_VGG16/noise_' + str(noise_level) + '_lr_' + str(learning_rate) + '_clean_data_size_' + str(clean_data_size) + '/'
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    # train benchmark
    model = VGG16(weights=None, include_top=True, classes=10, input_shape=(32,32,3))
    model.summary()
    adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)    
    
    x = np.concatenate((x_train, x_clean), axis=0)
    y = np.concatenate((y_train, y_clean), axis=0)
    file_benchmark = open(dirs + 'benchmark' + str(file_index) + '.txt', 'a+')
    History_benchmark = model.fit(x, y, validation_data=(x_test, y_test), batch_size=64, epochs=100, shuffle=True)
    
    
    file_benchmark.write('training accuracy' + '\n')
    file_benchmark.write(str(History_benchmark.history['acc']) + '\n')
    file_benchmark.write('test accuracy' + '\n')
    file_benchmark.write(str(History_benchmark.history['val_acc']) + '\n')
    file_benchmark.close()


for file_index in range(2, 5):
    for noise_level in [0.8]:
        for learning_rate in [0.0003, 0.0001, 0.00003]:
            seed = 10 * file_index
            run_benchmark(file_index, noise_level, learning_rate, seed)


