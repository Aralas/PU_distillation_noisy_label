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
from keras.callbacks import EarlyStopping

np.set_printoptions(threshold=np.inf)

# load data from CIFAR10
def load_data(clean_data_size):
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 127.5 - 1, x_test / 127.5 - 1
    x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)

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


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

file_index = 1
noise_level = 0.8
clean_data_size = 50
seed = 10 * file_index
additional_data_size = 2000
learning_rate = 0.0003
iteration_num = 10
bagging_threshold = 0.9
add_criterion = 75
minimum_addtional_size = 50

np.random.seed(seed)
tf.set_random_seed(seed)

x_train, y_train, x_test, y_test, x_clean, y_clean = load_data(clean_data_size)
y_train_orig = deepcopy(y_train)
y_train = generate_noise_labels(y_train, noise_level)

dirs = 'record_new_preprocessing/bagging_without_renew_models_PU/noise_' + str(noise_level) + '_clean_data_size_' + str(clean_data_size) + '_additional_data_size_' + str(additional_data_size)  + '/bagging_threshold_' + str(bagging_threshold) + '_add_criterion_' + str(add_criterion) + '_minimum_additional_size_' + str(minimum_addtional_size) + '/seed_' + str(seed) + '/'
if not os.path.exists(dirs):
    os.makedirs(dirs)

final_additional_data = [[] for i in range(10)]
add_number = []

# read index of additional data
for label in range(10):
    file = open(dirs + 'label_' + str(label) + '.txt')
    lines = file.readlines()
    new_line = ''
    for line in lines:
        new_line += line.strip('\n').replace('array','')
    final_additional_data[label] = eval(new_line)[-1]
    add_number.append(len(eval(new_line)[-1]))

    
# get additional data and train teacher model
bootstrap_size = min(add_number)
for k in range(5): 
    y_pred = np.zeros((20, len(x_train), 10))
    for teacher_bagging_i in range(20):
        x_add = deepcopy(x_clean)
        y_add = deepcopy(y_clean)
        for label in range(10):
            index = final_additional_data[label]
            index = np.random.choice(index, bootstrap_size, replace=False)
            x_add = np.concatenate((x_add, x_train[index]), axis=0)
            y_add = np.concatenate((y_add, tf.contrib.keras.utils.to_categorical([label] * len(index), 10)))

        architecture = [[64, 5, 5], [64, 5, 5], [32, 5, 5], [32, 5, 5], [16, 5, 5], [1000]]
        teacher_model = create_model(architecture, num_classes=10, learning_rate=learning_rate)
        early_stopping = EarlyStopping(monitor='loss', patience=3)
        History_teacher = teacher_model.fit(x_add, y_add, validation_data=(x_test, y_test), batch_size=64, epochs=50,
                                            shuffle=True, callbacks=[early_stopping])
        
        y_pred[teacher_bagging_i] = teacher_model.predict(x_train)
    y_pred = np.mean(y_pred, axis=0)
    # generate a multi-classifier
    for lambda_teacher in [0.8, 0.85, 0.9, 0.95]:
        architecture = [[64, 5, 5], [64, 5, 5], [32, 5, 5], [32, 5, 5], [16, 5, 5], [1000]]
        student_model = create_model(architecture, num_classes=10, learning_rate=learning_rate)
        
        y_pseudo = lambda_teacher * y_train + (1 - lambda_teacher) * y_pred
        early_stopping = EarlyStopping(monitor='loss', patience=3)
        History_student = student_model.fit(x_train, y_pseudo, validation_data=(x_test, y_test), batch_size=64, epochs=50,
                                            shuffle=True, callbacks=[early_stopping])

        file_student  = open(dirs + 'bagging_teacher_file_student_' + str(k) + '.txt', 'a+')
        file_student.write('training accuracy when lambda=' + str(lambda_teacher) + '\n')
        file_student.write(str(History_student.history['acc']) + '\n')
        file_student.write('test accuracy when lambda=' + str(lambda_teacher) + '\n')
        file_student.write(str(History_student.history['val_acc']) + '\n')
        file_student.close()

    
       