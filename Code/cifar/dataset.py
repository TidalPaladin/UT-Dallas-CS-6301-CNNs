#!python3
################################################################################
#
# xNNs_Code_02_Vision_Class_CIFAR.py
#
# DESCRIPTION
#
#    TensorFlow CIFAR example
#
# INSTRUCTIONS
#
#    1. Go to Google Colaboratory: https://colab.research.google.com/notebooks/welcome.ipynb
#    2. File - New Python 3 notebook
#    3. Cut and paste this file into the cell (feel free to divide into multiple cells)
#    4. Runtime - Change runtime type - Hardware accelerator - GPU
#    5. Runtime - Run all
#
################################################################################


################################################################################
#
# IMPORT
#
################################################################################

# tenorflow
import tensorflow         as     tf
from   tensorflow         import keras
from   tensorflow         import contrib
from   tensorflow.contrib import autograph

# additional libraries
import numpy             as np
import matplotlib.pyplot as plt
import datetime
#%matplotlib inline


################################################################################
#
# PARAMETERS
#
################################################################################

# data
DATA_NUM_CLASSES = 10

# model
MODEL_LEVEL_0_BLOCKS = 4
MODEL_LEVEL_1_BLOCKS = 6
MODEL_LEVEL_2_BLOCKS = 3

# training
TRAINING_IMAGE_SIZE        = 32
TRAINING_ZERO_PADDED_SIZE  = 40
TRAINING_SHUFFLE_BUFFER    = 5000
TRAINING_BATCH_SIZE        = 128
TRAINING_NUM_EPOCHS        = 72
TRAINING_MOMENTUM          = 0.9                    # currently not used
TRAINING_REGULARIZER_SCALE = 0.1                    # currently not used
TRAINING_LR_INITIAL        = 0.01
TRAINING_LR_SCALE          = 0.1
TRAINING_LR_EPOCHS         = 64
TRAINING_LR_STAIRCASE      = True
TRAINING_MAX_CHECKPOINTS   = 5
TRAINING_CHECKPOINT_FILE   = './logs/model_{}.ckpt' # currently not used



def pre_processing_train(image, label):

    # note: this function operates on 8 bit data then normalizes to a float

    # add a boarder of 0s
    image = tf.image.resize_image_with_crop_or_pad(
            image,
            target_height=TRAINING_ZERO_PADDED_SIZE,
            target_width=TRAINING_ZERO_PADDED_SIZE)

    # random horizontal flip
    image = tf.image.random_flip_left_right(image)

    # random crop back to original size
    crop_size = [TRAINING_IMAGE_SIZE, TRAINING_IMAGE_SIZE, 3]
    image = tf.random_crop(
            image,
            size=crop_size)

    # normalization
    image = (tf.cast(image, tf.float32) - data_mean) / data_std

    return image, label

def pre_processing_test(image, label):

    # note: this function operates on 8 bit data then normalizes to a float

    # normalization
    image = (tf.cast(image, tf.float32) - data_mean)/data_std

    return image, label

def pre_processing_test(image, label):

################################################################################
#
# DATASET
#
################################################################################

class Dataset(tf.data.Dataset):

    def __init__():

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
        dataset_train = tf.data.Dataset.from_tensor_slices(
                (data_train, labels_train))
        dataset_test  = tf.data.Dataset.from_tensor_slices(
                (data_test,  labels_test))

        # transformation
        dataset_train = dataset_train.shuffle(TRAINING_SHUFFLE_BUFFER)
            .repeat()
            .map(pre_processing_train)
            .batch(TRAINING_BATCH_SIZE)
        dataset_test  = dataset_test
            .repeat()
            .map(pre_processing_test)
            .batch(TRAINING_BATCH_SIZE)


    def __iter__():
        iterator = tf.data.Iterator.from_structure(
                dataset_train.output_types,
                dataset_train.output_shapes)
        iterator_init_train = iterator.make_initializer(dataset_train)
        iterator_init_test  = iterator.make_initializer(dataset_test)

