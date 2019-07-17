from __future__ import print_function

import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential


class CNN(object):
    def __init__(self, num_classes, input_shape, learning_rate):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.model = self.generate_model()

    def generate_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='glorot_normal',
                         input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3), kernel_initializer='glorot_normal'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='glorot_normal'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), kernel_initializer='glorot_normal'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        if self.num_classes > 1:
            model.add(Activation('softmax'))
            loss = 'categorical_crossentropy'
        else:
            model.add(Activation('sigmoid'))
            loss = 'mean_squared_error'

        # initiate RMSprop optimizer
        opt = keras.optimizers.Adam(lr=self.learning_rate)

        # Let's train the model using RMSprop
        model.compile(loss=loss,
                      optimizer=opt,
                      metrics=['accuracy'])
        return model
