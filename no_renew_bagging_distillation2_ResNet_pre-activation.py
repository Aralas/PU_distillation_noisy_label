# coding: utf-8
from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K
from keras.datasets import cifar10

import tensorflow as tf
import os
import numpy as np
from copy import deepcopy
import sys



np.set_printoptions(threshold=np.inf)

# load data from CIFAR10
def load_data(clean_data_size):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Input image dimensions.
    input_shape = x_train.shape[1:]
    
    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Subtract pixel mean    
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

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
    return x_train, y_train, x_test, y_test, x_clean, y_clean, input_shape


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


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 3e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
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
                    strides = 2    # downsample

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
                    activation='sigmoid',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def run_experiment(additional_data_size, learning_rate, bagging_threshold, add_criterion, n):
    # Training parameters
    batch_size = 32  # orig paper trained all networks with batch_size=128
    epochs = 50
    num_classes = 1

    # Subtracting pixel mean improves accuracy
    subtract_pixel_mean = True
        
    depth = n * 9 + 2   
   
    file_index = 1
    clean_data_size = 250
    seed = 10 * file_index
    iteration_num = 5
    minimum_addtional_size = 50
    np.random.seed(seed)
    tf.set_random_seed(seed)

    x_train, y_train, x_test, y_test, x_clean, y_clean, input_shape = load_data(clean_data_size)
    y_train_orig = deepcopy(y_train)

    dirs = 'record_new_preprocessing/bagging_cold_start_positives_' + str(2*clean_data_size) + '_clean_data_size_' + str(clean_data_size) + '_additional_data_size_' + str(additional_data_size) + '/ResNet' + str(depth) + ' _pre-activation' + '/bagging_threshold_' + str(bagging_threshold) + '_add_criterion_' + str(add_criterion) + '_minimum_additional_size_' + str(minimum_addtional_size) + '_lr_' + str(learning_rate) + '/seed_' + str(seed) + '/'
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    
    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    callbacks = [lr_reducer, lr_scheduler]
    
    for label in range(10):
        additional_data_index = [[] for i in range(iteration_num)]    
        binary_classifier_list = []
        for index in range(100):
            model = resnet_v2(input_shape=input_shape, depth=depth, num_classes=num_classes)
            model.compile(loss='mean_squared_error',
                  optimizer=Adam(lr=lr_schedule(0)),
                  metrics=['accuracy'])
            binary_classifier_list.append(model)
        for iteration in range(iteration_num):           
            y_pred = np.zeros((len(x_train), 100))
            # train 100 binary classifiers for one label
            for index in range(100):
                print('iteration', iteration, 'label', label, 'index', index)
                model = binary_classifier_list[index]

                positive_index = list(np.where(y_clean[:, label] == 1)[0])
                x = x_clean[positive_index]

                if len(additional_data_index[label]) < clean_data_size:                
                    add_positive_index = additional_data_index[label]
                else:
                    add_positive_index = np.random.choice(additional_data_index[label], clean_data_size, replace=False)
                x = np.concatenate((x, x_train[add_positive_index]), axis=0)

                n_p = len(x)
                n_n_clean = n_p//2
                n_n_noisy = n_p - n_n_clean
                negative_index_clean = list(np.where(y_clean[:, label] != 1)[0])
                negative_index_clean = np.random.choice(negative_index_clean, n_n_clean, replace=False)
                x = np.concatenate((x, x_clean[negative_index_clean]), axis=0)
                negative_index_noisy = set(np.arange(len(x_train))) - set(additional_data_index[label])
                negative_index_noisy = np.random.choice(list(negative_index_noisy), n_n_noisy, replace=False)
                x = np.concatenate((x, x_train[negative_index_noisy]), axis=0)

                y = [1] * n_p + [0] * n_p
                
               

                # Fit the model on the batches generated by datagen.flow().
                model.fit(x, y,
                  batch_size=batch_size,
                  epochs=epochs,
                  shuffle=True,
                  callbacks=callbacks)
                y_pred[:, index] = model.predict(x_train).reshape(-1,)                

            y_pred1 = np.sum(y_pred>bagging_threshold, axis=1)   
            add_index = np.where(y_pred1 > add_criterion)[0]
            if len(add_index) > additional_data_size:
                add_index = np.argsort(-y_pred1, axis=0)[0:additional_data_size].reshape(-1)
            elif len(add_index) < minimum_addtional_size:
                y_pred2 = np.sum(y_pred, axis=1) 
                add_index = np.argsort(-y_pred2, axis=0)[0:minimum_addtional_size].reshape(-1)
            additional_data_index[iteration] = add_index

        K.clear_session()   
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)

        record_file = open(dirs + 'label_' + str(label) + '.txt', 'a+')      
        record_file.write(str(additional_data_index) + '\n')
        record_file.close()

        # estimate additional clean data  
        precision_additional_data = []
        number_additional_data = []              
        true_positive_index = list(np.where(y_train_orig[:, label] == 1)[0]) 
        for iteration in range(iteration_num):
            index = additional_data_index[iteration]        
            TP = len(list(set(index) & set(true_positive_index)))
            if len(index) == 0:
                precision_additional_data.append(0)
            else:
                precision_additional_data.append(TP / len(index))
                number_additional_data.append(len(index))

        precision_file = open(dirs + 'precision_file.txt', 'a+')    
        precision_file.write(str(precision_additional_data) + '\n')
        precision_file.write(str(number_additional_data) + '\n') 
        precision_file.close()
  

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

additional_data_size = 2000
bagging_threshold = 0.95
add_criterion = 90

for learning_rate in [0.003]:
    run_experiment(additional_data_size, learning_rate, bagging_threshold, add_criterion, 2)