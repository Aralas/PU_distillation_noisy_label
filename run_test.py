'''

parameters:

clean_data_size: number of clean data for each class
noise_level: fraction of noisy labels for the rest data
additional_data_limitation: minimum and maximum limitation of additional data for each class
positive_threshold: threshold of positives and negatives
add_criterion: criterion to generate additional data
K_iteration: number of iterations to generate additional data
N_bagging: number of binary classifiers for each class
data_augmentation: whether to use data augmentation in training

'''

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
additional_data_limitation = [100, 2000]
positive_threshold = 0.95
add_criterion = 90
K_iteration = 10
N_bagging = 100
label = 1

batch_size = 32
epochs = 20
learning_rate = 0.0001
data_augmentation = True

path_name = '/positive_threshold_%.2f_add_criterion_%d_learning_rate_%.4f'%(positive_threshold, add_criterion, learning_rate)
model_dir = os.path.join(os.getcwd(), 'saved_models')
if label == None:
    model_dir += path_name
else:
    model_dir += path_name + '_label' + str(label)
precision_dir = os.path.join(os.getcwd(), 'saved_precision')
precision_dir += path_name

print(model_dir)
print(precision_dir)

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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

# record parameters    
record_file = open(os.path.join(precision_dir, 'parameter.txt'), 'a+')
record_file.write('seed:%d\n noise_level:%.1f\n clean_data_size:%d\n additional_data_limitation:%d,%d\n positive_threshold:%.2f\n add_criterion:%d\n K_iteration:%d\n N_bagging:%d\n batch_size:%d\n epochs:%d\n learning_rate:%f\n data_augmentation:%s\n' % (seed, noise_level, clean_data_size, additional_data_limitation[0], additional_data_limitation[1], positive_threshold, add_criterion, K_iteration, N_bagging, batch_size, epochs, learning_rate, str(data_augmentation)))
record_file.close()    
    
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


'''
***** clean data augmentation *****
'''
if label == None:
    label_list = np.arange(num_classes)
else:
    label_list = [label]

for i in label_list:
    additional_data_index = []
    binary_classifier_list = []

    # Initialize N binary classifiers
    for n in range(N_bagging):
        if n % 10 == 9:
            clear_session()
        model = CNN(1, input_shape, learning_rate).model
        filepath = os.path.join(model_dir, 'model%d.h5' % n)
        model.save_weights(filepath)
        clear_session()

    for k in range(K_iteration):
        y_pred_train = np.zeros((len(x_train), N_bagging))
        y_pred_validation = np.zeros((len(x_validation), N_bagging))

        # train N binary classifiers for one label
        for n in range(N_bagging):
            if n % 10 == 9:
                clear_session()
            print('iteration: %d, label: %d, model: %d' % (k, i, n))
            model = CNN(1, input_shape, learning_rate).model
            filepath = os.path.join(model_dir, 'model%d.h5' % n)
            model.load_weights(filepath)

            positive_index = list(np.where(y_clean[:, i] == 1)[0])
            x = x_clean[positive_index]

            if len(additional_data_index) < clean_data_size:
                add_positive_index = additional_data_index
            else:
                add_positive_index = np.random.choice(additional_data_index, clean_data_size, replace=False)
            x = np.concatenate((x, x_train[add_positive_index]), axis=0)

            n_p = len(x)
            n_n_clean = n_p // 2
            n_n_noisy = n_p - n_n_clean
            negative_index_clean = list(np.where(y_clean[:, i] != 1)[0])
            negative_index_clean = np.random.choice(negative_index_clean, n_n_clean, replace=False)
            x = np.concatenate((x, x_clean[negative_index_clean]), axis=0)
            negative_index_noisy = set(np.arange(len(x_train))) - set(additional_data_index)
            negative_index_noisy = np.random.choice(list(negative_index_noisy), n_n_noisy, replace=False)
            x = np.concatenate((x, x_train[negative_index_noisy]), axis=0)

            y = [1] * n_p + [0] * n_p

            # Data augmentation
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

            # Compute quantities required for feature-wise normalization
            # (std, mean, and principal components if ZCA whitening is applied).
            datagen.fit(x)

            if data_augmentation:
                # Fit the model on the batches generated by datagen.flow().
                model.fit_generator(datagen.flow(x, y, batch_size=batch_size, ),
                                    steps_per_epoch=x.shape[0],
                                    epochs=epochs, verbose=1, workers=4)
            else:
                model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=1)
                
            model.save_weights(filepath)
            y_pred_train[:, n] = model.predict(x_train).reshape(-1, )
            y_pred_validation[:, n] = model.predict(x_validation).reshape(-1, )

        # Generate additional data from training data set
        y_pred1 = np.sum(y_pred_train > positive_threshold, axis=1)
        add_index = np.where(y_pred1 > add_criterion)[0]
        if len(add_index) > additional_data_limitation[1]:
            add_index = np.argsort(-y_pred1, axis=0)[0:additional_data_limitation[1]].reshape(-1)
        elif len(add_index) < additional_data_limitation[0]:
            y_pred2 = np.sum(y_pred_train, axis=1)
            add_index = np.argsort(-y_pred2, axis=0)[0:additional_data_limitation[0]].reshape(-1)
        additional_data_index = add_index

        record_file = open(os.path.join(precision_dir, 'training_label%d.txt' % i), 'a+')
        record_file.write(str(additional_data_index) + '\n')
        record_file.close()

        # Test the precision on validation set
        y_pred1 = np.sum(y_pred_validation > positive_threshold, axis=1)
        add_index = np.where(y_pred1 > add_criterion)[0]

        record_file = open(os.path.join(precision_dir, 'validation_label%d.txt' % i), 'a+')
        record_file.write(str(add_index) + '\n')
        record_file.close()

'''
## Train teacher model on additional clean data
'''

'''
## Train student model
'''
