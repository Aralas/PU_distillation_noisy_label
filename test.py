# coding: utf-8
import tensorflow as tf
import os
import numpy as np
from copy import deepcopy
import sys
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import EarlyStopping

np.set_printoptions(threshold=np.inf)

# load data from CIFAR10
def load_data():
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 127.5 - 1, x_test / 127.5 - 1
    x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)

    # transform labels to one-hot vectors
    y_train = tf.contrib.keras.utils.to_categorical(y_train, 10)
    y_test = tf.contrib.keras.utils.to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test


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


def create_model(architecture, num_classes, learning_rate, dropout=0.5):
    model = Sequential()
    for layer_index in range(len(architecture)):
        layer = architecture[layer_index]
        if len(layer) == 3:
            if layer_index == 0:
                model.add(Conv2D(layer[0], kernel_size=(layer[1], layer[2]), input_shape=(32, 32, 3),
                                 kernel_initializer='glorot_normal', activation='relu', padding='same'))
            else:
                model.add(Conv2D(layer[0], kernel_size=(layer[1], layer[2]), kernel_initializer='glorot_normal',
                                 activation='relu', padding='same'))
            if layer_index < 3:
                model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        elif len(layer) == 1:
            if len(architecture[layer_index - 1]) == 3:
                model.add(Flatten())
            model.add(Dense(layer[0], activation='relu', kernel_initializer='glorot_normal'))
        else:
            print('Invalid architecture /(ㄒoㄒ)/~~')
    model.add(Dropout(dropout))
    if num_classes > 2:
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
        adam = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
    elif num_classes == 2:
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        adam = Adam(lr=learning_rate)
        model.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer=adam)
    return model

def train_teacher_model(x, y):
    accuracy = np.zeros((5, 30))
    architecture = [[32, 5, 5], [32, 5, 5], [32, 5, 5], [1000]]
    for i in range(5):
        teacher_model = create_model(architecture, num_classes=10, learning_rate=learning_rate)
        History_teacher = teacher_model.fit(x, y, validation_data=(x_test, y_test), batch_size=64, epochs=30, shuffle=True)
        accuracy[i] = History_teacher.history['val_acc']
    return np.mean(accuracy, axis=0)


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

seed = 20
np.random.seed(seed)
tf.set_random_seed(seed)

learning_rate = 0.0003

x_train, y_train, x_test, y_test = load_data()


x1 = x_train[0:10000]
y1 = y_train[0:10000]

x_train, y_train, x_test, y_test = load_data()
x2 = x_train[0:10000]
y2 = y_train[0:10000]
y2 = generate_noise_labels(y2, 0.5)

x_train, y_train, x_test, y_test = load_data()
x3 = x_train[0:10000]
y3 = y_train[0:10000]
y3 = generate_noise_labels(y3, 0.6)

x_train, y_train, x_test, y_test = load_data()
x4 = x_train[0:20000]
y4 = y_train[0:20000]

x_train, y_train, x_test, y_test = load_data()
x5 = x_train[0:20000]
y5 = y_train[0:20000]
y5 = generate_noise_labels(y5, 0.5)

x_train, y_train, x_test, y_test = load_data()
x6 = x_train[0:20000]
y6 = y_train[0:20000]
y6 = generate_noise_labels(y6, 0.6)




acc1 = train_teacher_model(x1, y1)
acc2 = train_teacher_model(x2, y2)
acc3 = train_teacher_model(x3, y3)
acc4 = train_teacher_model(x4, y4)
acc5 = train_teacher_model(x5, y5)
acc6 = train_teacher_model(x6, y6)

plt.figure(figsize=(10,6)) # 创建图表1

plt.plot(np.arange(30)+1, acc1, '--', label='10000 data, clean')
plt.plot(np.arange(30)+1, acc2, label='10000 data, noise=0.5')
plt.plot(np.arange(30)+1, acc3, label='10000 data, noise=0.6')
plt.plot(np.arange(30)+1, acc4, '--', label='20000 data, clean')
plt.plot(np.arange(30)+1, acc5, label='20000 data, noise=0.5')
plt.plot(np.arange(30)+1, acc6, label='20000 data, noise=0.6')

plt.legend()
plt.grid()
plt.ylim(0,1)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('teacher model accuracy', fontsize=20)

plt.savefig('teacher_accuracy.png')








