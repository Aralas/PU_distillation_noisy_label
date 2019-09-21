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
from keras.datasets import cifar10

np.set_printoptions(threshold=np.inf)

# load data from CIFAR10
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

def run_test(noise_level, learning_rate):
    file_index = 1    
    clean_data_size = 250
    seed = 10 * file_index    
    learning_rate = 0.0003
    
    minimum_addtional_size = 50

    np.random.seed(seed)
    tf.set_random_seed(seed)

    x_train, y_train, x_test, y_test, x_clean, y_clean = load_data(clean_data_size)
    y_train_orig = deepcopy(y_train)
    y_train = generate_noise_labels(y_train, noise_level)
    architecture = [[32, 5, 5], [32, 5, 5], [32, 5, 5], [1000]]

    dirs = 'record_new_preprocessing/original_distillation/noise_level_' + str(noise_level) + '_learning_rate_' + str(learning_rate) + '/seed_' + str(seed) + '/architecture_' + str(architecture)

    if not os.path.exists(dirs):
        os.makedirs(dirs)

    # train teacher model
    for k in range(5): 
        teacher_model = create_model(architecture, num_classes=10, learning_rate=learning_rate)        
        History_teacher = teacher_model.fit(x_clean, y_clean, validation_data=(x_test, y_test), batch_size=64, epochs=100, shuffle=True)
        
        file_teacher  = open(dirs + '/teacher_' + str(k) + '.txt', 'a+')
        file_teacher.write('training accuracy \n')
        file_teacher.write(str(History_teacher.history['acc']) + '\n')
        file_teacher.write('test accuracy \n')
        file_teacher.write(str(History_teacher.history['val_acc']) + '\n')
        file_teacher.close()        
        
        # generate a multi-classifier
        y_pred = teacher_model.predict(x_train)
        for lambda_teacher in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
            student_model = create_model(architecture, num_classes=10, learning_rate=learning_rate)

            y_pseudo = lambda_teacher * y_train + (1 - lambda_teacher) * y_pred
            x = np.concatenate((x_train, x_clean), axis=0)
            y = np.concatenate((y_pseudo, y_clean), axis=0)
            History_student = student_model.fit(x, y, validation_data=(x_test, y_test), batch_size=64, epochs=100, shuffle=True)

            file_student  = open(dirs + '/student_' + str(k) + '.txt', 'a+')
            file_student.write('training accuracy when lambda=' + str(lambda_teacher) + '\n')
            file_student.write(str(History_student.history['acc']) + '\n')
            file_student.write('test accuracy when lambda=' + str(lambda_teacher) + '\n')
            file_student.write(str(History_student.history['val_acc']) + '\n')
            file_student.close()

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)
            
for noise_level in [0.5, 0.8, 0.9, 0.95]: 
    for learning_rate in [0.0003, 0.0001]:
        run_test(noise_level, learning_rate)       
        K.clear_session()   
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)



