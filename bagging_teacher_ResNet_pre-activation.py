# coding: utf-8
from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import tensorflow as tf
import os
import numpy as np
import sys
import tensorflow as tf
import os
import numpy as np
from copy import deepcopy
import sys


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
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def generate_model(n, input_shape):
    
   
    depth = n * 9 + 2    
    model = resnet_v2(input_shape=input_shape, depth=depth)
  
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr_schedule(0)),
                  metrics=['accuracy'])
    model.summary()
    return(model)


    


def run_test(noise_level, additional_data_size, bagging_threshold, add_criterion, n):
    file_index = 1    
    clean_data_size = 250
    seed = 10 * file_index  
    learning_rate = 0.0003
    minimum_addtional_size = 50
    depth = n * 9 + 2
    
    # Training parameters
    batch_size = 32  # orig paper trained all networks with batch_size=128
    epochs = 100
    data_augmentation = False
    num_classes = 10

    # Subtracting pixel mean improves accuracy
    subtract_pixel_mean = True

    np.random.seed(seed)
    tf.set_random_seed(seed)

    x_train, y_train, x_test, y_test, x_clean, y_clean = load_data(clean_data_size)
    y_train_orig = deepcopy(y_train)
    y_train = generate_noise_labels(y_train, noise_level)
    architecture = [[32, 5, 5], [32, 5, 5], [32, 5, 5], [256]]

    dirs = 'record_new_preprocessing/bagging_cold_start_positives_' + str(2*clean_data_size) + '_clean_data_size_' + str(clean_data_size) + '_additional_data_size_' + str(additional_data_size) + '/' + str(architecture) + '/bagging_threshold_' + str(bagging_threshold) + '_add_criterion_' + str(add_criterion) + '_minimum_additional_size_' + str(minimum_addtional_size) + '_lr_' + str(learning_rate) + '/seed_' + str(seed) + '/'
    
    dirs1 = dirs + 'noise_' + str(noise_level) + '_ResNet' + str(depth) + '_pre-activation_lr_0.003'
    if not os.path.exists(dirs1):
        os.makedirs(dirs1)

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

    # Prepare callbacks for model saving and for learning rate adjustment.
    

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    callbacks = [lr_reducer, lr_scheduler]

    input_shape = x_train.shape[1:]
    # get additional data and train teacher model
    bootstrap_size = min(add_number)
    for k in range(1): 
        y_pred = np.zeros((20, len(x_train), 10))
        for teacher_bagging_i in range(20):
            x_add = deepcopy(x_clean)
            y_add = deepcopy(y_clean)
            for label in range(10):
                index = final_additional_data[label]
                index = np.random.choice(index, bootstrap_size, replace=False)
                x_add = np.concatenate((x_add, x_train[index]), axis=0)
                y_add = np.concatenate((y_add, tf.contrib.keras.utils.to_categorical([label] * len(index), 10)))

            teacher_model = generate_model(n, input_shape)
            History_teacher = teacher_model.fit(x_add, y_add,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=callbacks)

            y_pred[teacher_bagging_i] = teacher_model.predict(x_train)
        y_pred = np.mean(y_pred, axis=0)
        
        # generate a multi-classifier
        for lambda_teacher in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            student_model = generate_model(n, input_shape)

            y_pseudo = lambda_teacher * y_train + (1 - lambda_teacher) * y_pred
            x = np.concatenate((x_train, x_clean), axis=0)
            y = np.concatenate((y_pseudo, y_clean), axis=0)
            History_student = student_model.fit(x, y,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=callbacks)

            file_student  = open(dirs1 + '/bagging_teacher_file_student_' + str(k) + '.txt', 'a+')
            file_student.write('training accuracy when lambda=' + str(lambda_teacher) + '\n')
            file_student.write(str(History_student.history['acc']) + '\n')
            file_student.write('test accuracy when lambda=' + str(lambda_teacher) + '\n')
            file_student.write(str(History_student.history['val_acc']) + '\n')
            file_student.close()
            
            K.clear_session()   
            sess = tf.Session(config=config)
            K.set_session(sess)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)            
            
for noise_level in [0.8]:
    for additional_data_size in [2000]:
        for bagging_threshold in [0.9]:
            for add_criterion in [95]:
                n = 3
                run_test(noise_level, additional_data_size, bagging_threshold, add_criterion, n)       