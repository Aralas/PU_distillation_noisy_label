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


os.environ["CUDA_VISIBLE_DEVICES"] = "5"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

file_index = 1
noise_level = 0.5
clean_data_size = 50
seed = 10 * file_index
additional_data_size = 1000
learning_rate = 0.0003
iteration = 20

# create record files
dirs = 'record/noise_' + str(noise_level) + '_iteration_' + str(iteraion) + '_cleansize_' + str(clean_data_size) + '_add_' + str(additional_data_size) + '_lr_' + str(learning_rate) + '/test' + str(file_index) + '/'
if not os.path.exists(dirs):
    os.makedirs(dirs)

file_setting = open(dirs + 'file_setting.txt', 'a+')
file_additional_data = open(dirs + 'file_additional_data.txt', 'a+')
file_teacher = open(dirs + 'file_teacher.txt', 'a+')
file_student = open(dirs + 'file_student.txt', 'a+')
file_benchmark = open(dirs + 'file_benchmark.txt', 'a+')


# load data from CIFAR10
def load_data(clean_data_size):
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
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




np.random.seed(seed)
tf.set_random_seed(seed)

x_train, y_train, x_test, y_test, x_clean, y_clean = load_data(clean_data_size)
y_train_orig = deepcopy(y_train)
y_train = generate_noise_labels(y_train, noise_level)

file_setting.write('noise level: ' + str(noise_level) + '\n')
file_setting.write('clean data size for each class: ' + str(clean_data_size) + '\n')
file_setting.close()


def create_model(architecture, num_classes, learning_rate=learning_rate, dropout=0.5):
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


# generate 10 binary classifier
binary_classifier_list = []
architecture = [[32, 5, 5], [32, 5, 5], [32, 5, 5], [256]]
for label in range(10):
    model = create_model(architecture, num_classes=2)
    binary_classifier_list.append(model)

file_setting = open(dirs + 'file_setting.txt', 'a+')
file_setting.write('*' * 10 + 'architecture of binary classifier' + '*' * 10 + '\n')
orig_stdout = sys.stdout
sys.stdout = file_setting
print(model.summary())
sys.stdout = orig_stdout
file_setting.close()

# use the idea of PU learning to augment positive data
epochs = 10
additional_data_index = [[] for i in range(10)]
for epoch in range(epochs):
    for label in range(10):
        print('epoch', epoch, 'label', label)
        positive_index = list(np.where(y_clean[:, label] == 1)[0])
        x = x_clean[positive_index]
        x = np.concatenate((x, x_train[additional_data_index[label]]), axis=0)
        n_p = len(x)
        n_n = min(400, n_p)
        negative_index = list(np.where(y_clean[:, label] != 1)[0])
        negative_index = np.random.choice(negative_index, n_n, replace=False)
        x = np.concatenate((x, x_clean[negative_index]), axis=0)
        y = [1] * n_p + [0] * n_n
        classifier = binary_classifier_list[label]
        classifier.fit(x, y, batch_size=32, epochs=20, shuffle=True)
        pred_train = classifier.predict(x_train)
        candidate_index = np.where(pred_train > 0.98)[0]
        if len(candidate_index) < additional_data_size:
            additional_data_index[label] = list(candidate_index)
        else:
            additional_data_index[label] = np.argsort(-pred_train, axis=0)[0:additional_data_size].reshape(-1)

    # estimate additional clean data
    precision_additional_data = []
    number_additional_data = []
    for label in range(10):
        index = additional_data_index[label]
        true_positive_index = list(np.where(y_train_orig[:, label] == 1)[0])
        TP = len(list(set(index) & set(true_positive_index)))
        if len(index) == 0:
            precision_additional_data.append(0)
        else:
            precision_additional_data.append(TP / len(index))
        number_additional_data.append(len(index))
    print(precision_additional_data)
    print(number_additional_data)

    file_additional_data.write('iteraion ' + str(epoch) + ', precision of additional data for each class' + '\n')
    file_additional_data.write(str(precision_additional_data) + '\n')
    file_additional_data.write('iteraion ' + str(epoch) + ', number of additional data for each class' + '\n')
    file_additional_data.write(str(number_additional_data) + '\n')
    file_additional_data.flush()
file_additional_data.close()

# get additional data and train teacher model
x_add = deepcopy(x_clean)
y_add = deepcopy(y_clean)
for label in range(10):
    index = additional_data_index[label]
    x_add = np.concatenate((x_add, x_train[index]), axis=0)
    y_add = np.concatenate((y_add, tf.contrib.keras.utils.to_categorical([label] * len(index), 10)))

architecture = [[32, 5, 5], [32, 5, 5], [32, 5, 5], [500]]
teacher_model = create_model(architecture, num_classes=10)

file_setting = open(dirs + 'file_setting.txt', 'a+')
file_setting.write('*' * 10 + 'architecture of teacher classifier' + '*' * 10 + '\n')
orig_stdout = sys.stdout
sys.stdout = file_setting
print(teacher_model.summary())
sys.stdout = orig_stdout
file_setting.close()

History_teacher = teacher_model.fit(x_add, y_add, validation_data=(x_test, y_test), batch_size=64, epochs=100,
                                    shuffle=True)

file_teacher.write('training accuracy' + '\n')
file_teacher.write(str(History_teacher.history['acc']) + '\n')
file_teacher.write('test accuracy' + '\n')
file_teacher.write(str(History_teacher.history['val_acc']) + '\n')
file_teacher.close()


# calculate lambda
def get_precision(x, y, model):
    mean_precision = []
    prediction = model.predict(x)
    y_pred = prediction.argmax(axis=-1)
    for label in range(10):
        pred_positive_index = list(np.where(y_pred == label)[0])
        true_positive_index = list(np.where(y[:, label] == 1)[0])
        TP = len(list(set(pred_positive_index) & set(true_positive_index)))
        print(label, len(pred_positive_index))
        mean_precision.append(TP / len(pred_positive_index))
    return mean_precision


precision_whole = get_precision(x_train, y_train, teacher_model)
precision_clean = get_precision(x_clean, y_clean, teacher_model)
lambda_teacher = sum(precision_clean) / (sum(precision_clean) + sum(precision_whole))
print(precision_whole, precision_clean, lambda_teacher)

file_additional_data = open(dirs + 'file_additional_data.txt', 'a+')
file_additional_data.write('precision of teacher model on whole training dataset' + '\n')
file_additional_data.write(str(precision_whole) + '\n')
file_additional_data.write('precision of teacher model on clean dataset' + '\n')
file_additional_data.write(str(precision_clean) + '\n')
file_additional_data.write('lambda: ' + str(lambda_teacher) + '\n')
file_additional_data.close()

# generate a multi-classifier
architecture = [[32, 5, 5], [32, 5, 5], [32, 5, 5], [500]]
student_model = create_model(architecture, num_classes=10)

file_setting = open(dirs + 'file_setting.txt', 'a+')
file_setting.write('*' * 10 + 'architecture of student classifier' + '*' * 10 + '\n')
orig_stdout = sys.stdout
sys.stdout = file_setting
print(student_model.summary())
sys.stdout = orig_stdout
file_setting.close()

y_pred = teacher_model.predict(x_train)
y_pseudo = lambda_teacher * y_train + (1 - lambda_teacher) * y_pred
History_student = student_model.fit(x_train, y_pseudo, validation_data=(x_test, y_test), batch_size=64, epochs=100,
                                    shuffle=True)

file_student.write('training accuracy' + '\n')
file_student.write(str(History_student.history['acc']) + '\n')
file_student.write('test accuracy' + '\n')
file_student.write(str(History_student.history['val_acc']) + '\n')
file_student.close()

# train benchmark
architecture = [[32, 5, 5], [32, 5, 5], [32, 5, 5], [500]]
benchmark_model = create_model(architecture, num_classes=10)

file_setting = open(dirs + 'file_setting.txt', 'a+')
file_setting.write('*' * 10 + 'architecture of benchmark classifier' + '*' * 10 + '\n')
orig_stdout = sys.stdout
sys.stdout = file_setting
print(benchmark_model.summary())
sys.stdout = orig_stdout
file_setting.close()

History_benchmark = student_model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=100,
                                      shuffle=True)

file_benchmark.write('training accuracy' + '\n')
file_benchmark.write(str(History_benchmark.history['acc']) + '\n')
file_benchmark.write('test accuracy' + '\n')
file_benchmark.write(str(History_benchmark.history['val_acc']) + '\n')
file_benchmark.close()

for lambda_teacher in [0.2, 0.4, 0.6]:
    # generate a multi-classifier
    architecture = [[32, 5, 5], [32, 5, 5], [32, 5, 5], [500]]
    student_model = create_model(architecture, num_classes=10)

    y_pred = teacher_model.predict(x_train)
    y_pseudo = lambda_teacher * y_train + (1 - lambda_teacher) * y_pred
    History_student = student_model.fit(x_train, y_pseudo, validation_data=(x_test, y_test), batch_size=64, epochs=100,
                                        shuffle=True)

    file_student = open(dirs + 'file_student_lambda_' + str(lambda_teacher) + '.txt', 'a+')
    file_student.write('training accuracy' + '\n')
    file_student.write(str(History_student.history['acc']) + '\n')
    file_student.write('test accuracy' + '\n')
    file_student.write(str(History_student.history['val_acc']) + '\n')
    file_student.close()