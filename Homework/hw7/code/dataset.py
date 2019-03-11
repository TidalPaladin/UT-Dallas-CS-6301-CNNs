#!python3

import tensorflow as tf
from   tensorflow         import keras
from   tensorflow         import contrib
from   tensorflow.contrib import autograph
from tensorflow.keras import layers

# additional libraries
import numpy             as np
import matplotlib.pyplot as plt


# training
TRAINING_IMAGE_SIZE        = 32
TRAINING_ZERO_PADDED_SIZE  = 40
TRAINING_SHUFFLE_BUFFER    = 5000
TRAINING_BATCH_SIZE        = 128


################################################################################
#
# PRE PROCESSING
#
################################################################################

def pre_processing_train(image, label):

    # note: this function operates on 8 bit data then normalizes to a float

    # add a boarder of 0s
    image = tf.image.resize_image_with_crop_or_pad(image, TRAINING_ZERO_PADDED_SIZE, TRAINING_ZERO_PADDED_SIZE)

    # random horizontal flip
    image = tf.image.random_flip_left_right(image)

    # random hue, contrast, brightness and saturation
    # image = tf.image.random_hue(image, max_delta=0.05)
    # image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
    # image = tf.image.random_brightness(image, max_delta=0.2)
    # image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

    # random crop back to original size
    image = tf.random_crop(image, size=[TRAINING_IMAGE_SIZE, TRAINING_IMAGE_SIZE, 3])

    # normalization
    image = (tf.cast(image, tf.float32) - data_mean)/data_std

    return image, label

def pre_processing_test(image, label):

    # note: this function operates on 8 bit data then normalizes to a float

    # normalization
    image = (tf.cast(image, tf.float32) - data_mean)/data_std

    return image, label


################################################################################
#
# DATASET
#
################################################################################

# download
cifar10 = keras.datasets.cifar10

# training and testing split
(data_train, labels_train), (data_test, labels_test) = cifar10.load_data()

# normalization values
data_mean = np.mean(data_train, dtype=np.float32)
data_std  = np.std(data_train, dtype=np.float32)

# label typecast
labels_train = labels_train.astype(np.int32)
labels_test  = labels_test.astype(np.int32)
labels_train = np.squeeze(labels_train)
labels_test  = np.squeeze(labels_test)

# dataset
dataset_train = tf.data.Dataset.from_tensor_slices((data_train, labels_train))
dataset_test  = tf.data.Dataset.from_tensor_slices((data_test,  labels_test))

# transformation
dataset_train = dataset_train.shuffle(TRAINING_SHUFFLE_BUFFER).repeat().map(pre_processing_train).batch(TRAINING_BATCH_SIZE)
dataset_test  = dataset_test.repeat().map(pre_processing_test).batch(TRAINING_BATCH_SIZE)


################################################################################
#
# ITERATOR
#
################################################################################

# iterator
iterator            = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
iterator_init_train = iterator.make_initializer(dataset_train)
iterator_init_test  = iterator.make_initializer(dataset_test)

# example
# data.shape   = TRAINING_BATCH_SIZE x rows x cols
# labels.shape = TRAINING_BATCH_SIZE x 1
data, labels = iterator.get_next()

