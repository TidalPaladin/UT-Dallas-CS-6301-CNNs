#!python3

# tenorflow
import tensorflow         as     tf
from   tensorflow         import keras
from   tensorflow         import contrib
from   tensorflow.contrib import autograph

from tensorflow.keras import layers as layers

# additional libraries
import numpy             as np
import matplotlib.pyplot as plt
import datetime


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



################################################################################
#
# MODEL - RESNET V2
#
################################################################################
class ResnetV2(tf.keras.Model):

    def __init__(self):
        """
        Here we define the types of layers to be used. These constructs
        say nothing about the sequence of operations. We are only defining
        the types of layers that will be used, the ordering will be specified
        in call()
        """

        # Specify default convolution properties
        DEFAULT_CONV = {
                'padding' : 'same',
                'data_format' : 'channels_last',
                'training' : train_state,
                'activation' : None,
                'use_bias' : False,
                'dilation_rate' : (1, 1),
                'strides' : (1, 1)
        }

        # Simple batch norm and ReLU nonlinearity
        self.batch_norm = layers.BatchNormalization()
        self.nonlin = layers.ReLU()

        # Tail

    def tail(inc
        self.tail = layers.Conv2D(
                filters=16,
                kernel_size=(3, 3),
                padding='same',
                data_format='channels_last',
                dilation_rate=(1, 1),
                use_bias=False)

    def level_0_bottleneck():


        # encoder - level 0 special bottleneck x1
        # input:  32 x 32 x 32
        # filter: 16 x 32 x 1 x 1 / 1
        # filter: 16 x 16 x 3 x 3
        # filter: 64 x 16 x 1 x 1
        # main:   64 x 32 x 1 x 1 / 1
        # output: 64 x 32 x 32
        self.l0_f1 = layers.Conv2D(
                filters=8,
                kernel_size=(1, 1), **DEFAULT_CONV)

        fm_residual = tf.layers.batch_normalization(fm_residual, training=train_state)
        fm_residual = tf.nn.relu(fm_residual)
        fm_residual = tf.layers.conv2d(fm_residual, 8, (3, 3), **DEFAULT_CONV)

        fm_residual = tf.layers.batch_normalization(fm_residual, training=train_state)
        fm_residual = tf.nn.relu(fm_residual)
        fm_residual = tf.layers.conv2d(fm_residual, 32, (1, 1), **DEFAULT_CONV)

        fm_id = tf.layers.conv2d(fm_id, 32, (1, 1), DEFAULT_CONV)
        fm_id = tf.add(fm_id, fm_residual)


    # encoder - level 0 standard bottleneck x(level_0_blocks - 1)
    # input:  64 x 32 x 32
    # filter: 16 x 64 x 1 x 1
    # filter: 16 x 16 x 3 x 3
    # filter: 64 x 16 x 1 x 1
    # main:   identity
    # output: 64 x 32 x 32
    for block_repeat_0 in range(level_0_blocks - 1):
        fm_residual = tf.layers.batch_normalization(fm_id, training=train_state)
        fm_residual = tf.nn.relu(fm_residual)
        fm_residual = tf.layers.conv2d(fm_residual, 8, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
        fm_residual = tf.layers.batch_normalization(fm_residual, training=train_state)
        fm_residual = tf.nn.relu(fm_residual)
        fm_residual = tf.layers.conv2d(fm_residual, 8, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
        fm_residual = tf.layers.batch_normalization(fm_residual, training=train_state)
        fm_residual = tf.nn.relu(fm_residual)
        fm_residual = tf.layers.conv2d(fm_residual, 32, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
        fm_id       = tf.add(fm_id, fm_residual)

@autograph.convert()
def model_resnet(data, train_state, level_0_blocks, level_1_blocks, level_2_blocks, num_classes):


    # encoder - level 0 standard bottleneck x(level_0_blocks - 1)
    # input:  64 x 32 x 32
    # filter: 16 x 64 x 1 x 1
    # filter: 16 x 16 x 3 x 3
    # filter: 64 x 16 x 1 x 1
    # main:   identity
    # output: 64 x 32 x 32
    for block_repeat_0 in range(level_0_blocks - 1):
        fm_residual = tf.layers.batch_normalization(fm_id, training=train_state)
        fm_residual = tf.nn.relu(fm_residual)
        fm_residual = tf.layers.conv2d(fm_residual, 8, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
        fm_residual = tf.layers.batch_normalization(fm_residual, training=train_state)
        fm_residual = tf.nn.relu(fm_residual)
        fm_residual = tf.layers.conv2d(fm_residual, 8, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
        fm_residual = tf.layers.batch_normalization(fm_residual, training=train_state)
        fm_residual = tf.nn.relu(fm_residual)
        fm_residual = tf.layers.conv2d(fm_residual, 32, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
        fm_id       = tf.add(fm_id, fm_residual)

    # encoder - level 1 down sampling bottleneck x1
    # input:   64 x 32 x 32
    # filter:  32 x 64 x 1 x 1 / 2
    # filter:  32 x 32 x 3 x 3
    # filter: 128 x 32 x 1 x 1
    # main:   128 x 64 x 1 x 1 / 2
    # output: 128 x 16 x 16
    fm_residual = tf.layers.batch_normalization(fm_id, training=train_state)
    fm_residual = tf.nn.relu(fm_residual)
    fm_residual = tf.layers.conv2d(fm_residual,  16, (1, 1), strides=(2, 2), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm_residual = tf.layers.batch_normalization(fm_residual, training=train_state)
    fm_residual = tf.nn.relu(fm_residual)
    fm_residual = tf.layers.conv2d(fm_residual,  16, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm_residual = tf.layers.batch_normalization(fm_residual, training=train_state)
    fm_residual = tf.nn.relu(fm_residual)
    fm_residual = tf.layers.conv2d(fm_residual, 64, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm_id       = tf.layers.conv2d(fm_id,       64, (1, 1), strides=(2, 2), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm_id       = tf.add(fm_id, fm_residual)

    # encoder - level 1 standard bottleneck x(level_1_blocks - 1)
    # input:  128 x  16 x 16
    # filter:  32 x 128 x 1 x 1
    # filter:  32 x  32 x 3 x 3
    # filter: 128 x  32 x 1 x 1
    # main:   identity
    # output: 128 x  16 x 16
    for block_repeat_1 in range(level_1_blocks - 1):
        fm_residual = tf.layers.batch_normalization(fm_id, training=train_state)
        fm_residual = tf.nn.relu(fm_residual)
        fm_residual = tf.layers.conv2d(fm_residual,  16, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
        fm_residual = tf.layers.batch_normalization(fm_residual, training=train_state)
        fm_residual = tf.nn.relu(fm_residual)
        fm_residual = tf.layers.conv2d(fm_residual,  16, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
        fm_residual = tf.layers.batch_normalization(fm_residual, training=train_state)
        fm_residual = tf.nn.relu(fm_residual)
        fm_residual = tf.layers.conv2d(fm_residual, 64, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
        fm_id       = tf.add(fm_id, fm_residual)

    # encoder - level 2 down sampling bottleneck x1
    # input:  128 x  16 x 16
    # filter:  64 x 128 x 1 x 1 / 2
    # filter:  64 x  64 x 3 x 3
    # filter: 256 x  64 x 1 x 1
    # main:   256 x 128 x 1 x 1 / 2
    # output: 256 x   8 x 8
    fm_residual = tf.layers.batch_normalization(fm_id, training=train_state)
    fm_residual = tf.nn.relu(fm_residual)
    fm_residual = tf.layers.conv2d(fm_residual,  32, (1, 1), strides=(2, 2), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm_residual = tf.layers.batch_normalization(fm_residual, training=train_state)
    fm_residual = tf.nn.relu(fm_residual)
    fm_residual = tf.layers.conv2d(fm_residual,  32, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm_residual = tf.layers.batch_normalization(fm_residual, training=train_state)
    fm_residual = tf.nn.relu(fm_residual)
    fm_residual = tf.layers.conv2d(fm_residual, 128, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm_id       = tf.layers.conv2d(fm_id,       128, (1, 1), strides=(2, 2), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
    fm_id       = tf.add(fm_id, fm_residual)

    # encoder - level 2 standard bottleneck x(level_2_blocks - 1)
    # input:  256 x   8 x 8
    # filter:  64 x 256 x 1 x 1
    # filter:  64 x  64 x 3 x 3
    # filter: 256 x  64 x 1 x 1
    # main:   identity
    # output: 256 x   8 x 8
    for block_repeat_2 in range(level_2_blocks - 1):
        fm_residual = tf.layers.batch_normalization(fm_id, training=train_state)
        fm_residual = tf.nn.relu(fm_residual)
        fm_residual = tf.layers.conv2d(fm_residual,  32, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
        fm_residual = tf.layers.batch_normalization(fm_residual, training=train_state)
        fm_residual = tf.nn.relu(fm_residual)
        fm_residual = tf.layers.conv2d(fm_residual,  32, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
        fm_residual = tf.layers.batch_normalization(fm_residual, training=train_state)
        fm_residual = tf.nn.relu(fm_residual)
        fm_residual = tf.layers.conv2d(fm_residual, 128, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=False)
        fm_id       = tf.add(fm_id, fm_residual)

    # encoder - level 2 special block x1
    # input:  256 x   8 x 8
    # output: 256 x   8 x 8
    fm_id       = tf.layers.batch_normalization(fm_id, training=train_state)
    fm_id       = tf.nn.relu(fm_id)

    # decoder
    # predictions.shape = TRAINING_BATCH_SIZE x num_classes
    fm_id       = tf.reduce_mean(fm_id, axis=[1, 2])
    predictions = tf.layers.dense(fm_id, num_classes, activation=None, use_bias=True)

    # return
    return predictions

    def call(self, inputs, training=False):
