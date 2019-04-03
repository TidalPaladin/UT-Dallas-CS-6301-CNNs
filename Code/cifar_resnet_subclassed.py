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
from tensorflow.keras import layers

# additional libraries
import numpy             as np
import matplotlib.pyplot as plt
import datetime
#%matplotlib inline

import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

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

WIDTH=16

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

# display
# print(data_train.shape)
# print(data_test.shape)
# print(labels_train.shape)
# print(labels_test.shape)


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


################################################################################
#
# MODEL - RESNET V2
#
################################################################################

class Resnet(tf.keras.Model):

    def __init__(self, classes, filters, levels, *args, **kwargs):
        super(Resnet, self).__init__()

        # Lists to hold various layers
        self.blocks = list()

        # Tail
        self.tail = layers.Conv2D(
                filters,
                (3, 3),
                padding='same',
                data_format='channels_last',
                use_bias=False)

        # Special bottleneck layer with convolution on main path
        self.level_0_special = SpecialBottleneck(filters)

        # Loop through levels and their parameterized repeat counts
        for level, repeats in enumerate(levels):
            for block in range(repeats):
                # Append a bottleneck block for each repeat
                self.blocks.append(Bottleneck(filters))

            # Downsample and double feature maps at end of level
            self.blocks.append( Downsample(filters))
            filters *= 2

        # encoder - level 2 special block x1
        # input:  256 x   8 x 8
        # output: 256 x   8 x 8
        self.level2_batch_norm = layers.BatchNormalization()
        self.level2_relu = layers.ReLU()

        # decoder
        self.global_avg = layers.GlobalAveragePooling2D(
                data_format='channels_last')
        self.dense = layers.Dense(classes, use_bias=True)


    def call(self, inputs):
        x = self.tail(inputs)
        x = self.level_0_special(x)

        # Loop over layers by level
        for layer in self.blocks:
            x = layer(x)

        # Finish up specials in level 2
        x = self.level2_batch_norm(x)
        x = self.level2_relu(x)

        # Decoder
        x = self.global_avg(x)
        return self.dense(x)

class ResnetBasic(tf.keras.Model):
    """
    Respresents the batch norm, relu, conv2d resnet fundamental
    """

    def __init__(self, filters, kernel_size, *args, **kwargs):
        super(ResnetBasic, self).__init__()
        self.batch_norm = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv2d = layers.Conv2D(
                filters,
                kernel_size,
                padding='same',
                data_format='channels_last',
                activation=None,
                use_bias=False,
                *args,**kwargs)

    def call(self, inputs):
        x = self.batch_norm(inputs)
        x = self.relu(x)
        return self.conv2d(x)

class Bottleneck(tf.keras.Model):
    """
    encoder - standard bottleneck
        input:  32 x 32 x 32
        filter: 16 x 32 x 1 x 1 / 1
        filter: 16 x 16 x 3 x 3
        filter: 64 x 16 x 1 x 1
        main:   64 x 32 x 1 x 1 / 1
        output: 64 x 32 x 32
    """

    def __init__(self, Ni, *args, **kwargs):
        """
        Pass Ni, the input feature map size
        """
        # Parent constructor call
        super(Bottleneck, self).__init__(*args, **kwargs)

        # Three residual convolution blocks
        self.residual_filter_1 = ResnetBasic(int(Ni/4), (1,1))
        self.residual_filter_2 = ResnetBasic(int(Ni/4), (3,3))
        self.residual_filter_3 = ResnetBasic(Ni, (1,1))

        # Join residual and main path
        self.merge = layers.Add()

    def call(self, inputs):
        res = self.residual_filter_1(inputs)
        res = self.residual_filter_2(res)
        res = self.residual_filter_3(res)

        return self.merge([inputs, res])

class SpecialBottleneck(Bottleneck):
    """
    This bottleneck is special because the main path
    receives a 2d convolution

        encoder - speical bottleneck
        input:  N x 32 x 32
        filter: N/4 x N x 1 x 1
        filter: N/4 x N/4 x 3 x 3
        filter: N x N/4 x 1 x 1
        main:   identity
        output: N x 32 x 32
    """
    def __init__(self, Ni, *args, **kwargs):
        """
        Override of standard bottleneck to add main path convolution
        """

        # Residual layers as with normal bottleneck
        super(SpecialBottleneck, self).__init__(Ni, *args, **kwargs)

        # Add convolution for main path
        self.main = layers.Conv2D(
                Ni,
                (1, 1),
                padding='same',
                data_format='channels_last',
                activation=None,
                use_bias=False)

    def call(self, inputs):

        res = self.residual_filter_1(inputs)
        res = self.residual_filter_2(res)
        res = self.residual_filter_3(res)
        main = self.main(inputs)
        return self.merge([main, res])


class Downsample(tf.keras.Model):
    """
    encoder - down sampling bottleneck
    input:   N x Lr x Lc
    filter:  N/2 x N x 1 x 1 / 2
    filter:  N/2 x N/2 x 3 x 3
    filter:  2N x N/2 x 1 x 1
    main:    2N x N x 1 x 1 / 2
    output:  2N x Lr/2 x Lc/2
    """

    def __init__(self, Ni, *args, **kwargs):
        """
        Pass Ni, the input feature map size
        """
        # Parent constructor call
        super(Downsample, self).__init__(*args, **kwargs)

        # Three residual convolution blocks
        self.residual_filter_1 = ResnetBasic(int(Ni/2), (1,1), strides=(2,2))
        self.residual_filter_2 = ResnetBasic(int(Ni/2), (3,3))
        self.residual_filter_3 = ResnetBasic(2*Ni, (1,1))

        self.main = ResnetBasic(2*Ni, (1,1), strides=(2,2))

        # Join residual and main path
        self.merge = layers.Add()

    def call(self, inputs):
        res = self.residual_filter_1(inputs)
        res = self.residual_filter_2(res)
        res = self.residual_filter_3(res)

        main = self.main(inputs)
        return self.merge([main, res])


################################################################################
#
# TRAINING
#
################################################################################

# state
train_state = tf.placeholder(tf.bool, name='train_state')

# data
num_train         = len(data_train)
num_test          = len(data_test)
num_batches_train = int(num_train/TRAINING_BATCH_SIZE)
num_batches_test  = int(num_test/TRAINING_BATCH_SIZE)

levels = (
        MODEL_LEVEL_0_BLOCKS,
        MODEL_LEVEL_1_BLOCKS,
        MODEL_LEVEL_2_BLOCKS)

model = Resnet(DATA_NUM_CLASSES, WIDTH, levels)
predictions = model(data)
predictions_test = np.zeros((num_test, DATA_NUM_CLASSES), dtype=np.float32)

# accuracy
accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(predictions, 1), tf.cast(labels, tf.int64)), tf.float32))

# loss
loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=predictions)

# optimizer
global_step   = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(TRAINING_LR_INITIAL, global_step, TRAINING_LR_EPOCHS*num_batches_train, TRAINING_LR_SCALE, staircase=TRAINING_LR_STAIRCASE)
update_ops    = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # optimizer = tf.train.MomentumOptimizer(learning_rate, TRAINING_MOMENTUM, use_nesterov=True).minimize(loss, global_step=global_step)

# saver
# saver = tf.train.Saver(max_to_keep=TRAINING_MAX_CHECKPOINTS)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.75

# create a session
session = tf.Session(config=config)

# initialize global variables
session.run(tf.global_variables_initializer())

# cycle through the epochs
for epoch_index in range(TRAINING_NUM_EPOCHS):

    # train
    # initialize the iterator to the training dataset
    # cycle through the training batches
    # example, encoder, decoder, error, gradient computation and update
    session.run(iterator_init_train)
    for batch_index in range(num_batches_train):
        session.run(optimizer, feed_dict={train_state: True})

    # validate
    # initialize the iterator to the testing dataset
    # reset the accuracy statistics
    # cycle through the testing batches
    # example, encoder, decoder, accuracy
    session.run(iterator_init_test)
    num_correct = 0
    for batch_index in range(num_batches_test):
        num_correct_batch, predictions_batch    = session.run([accuracy, predictions], feed_dict={train_state: False})
        num_correct                            += num_correct_batch
        row_start                               = batch_index*TRAINING_BATCH_SIZE
        row_end                                 = (batch_index + 1)*TRAINING_BATCH_SIZE
        predictions_test[row_start:row_end, :]  = predictions_batch

    # display
    print('Epoch {0:3d}: top 1 accuracy on the test set is {1:5.2f} %'.format(epoch_index, (100.0*num_correct)/(TRAINING_BATCH_SIZE*num_batches_test)))
    print('Timestamp: %s' % datetime.datetime.now())

    # save
    # saver.save(session, TRAINING_CHECKPOINT_FILE.format(epoch_index))

# close the session
session.close()


################################################################################
#
# DISPLAY
#
################################################################################

# create a session
session = tf.Session()

# initialize global variables
session.run(tf.global_variables_initializer())

# initialize the test iterator
session.run(iterator_init_test)

# cycle through a few batches
for batch_index in range(1):

    # generate data and labels
    data_batch, labels_batch = session.run([data, labels])

    # normalize to [0, 1]
    data_batch = ((data_batch*data_std) + data_mean)/255.0;

    # convert the final saved predictions to labels
    row_start          = batch_index*TRAINING_BATCH_SIZE
    row_end            = (batch_index + 1)*TRAINING_BATCH_SIZE
    predictions_labels = np.argmax(predictions_test[row_start:row_end, :], axis=1)

    # cycle through the images in the batch
    for image_index in range(TRAINING_BATCH_SIZE):

        # display the predicted label, actual label and image
        print('Predicted label: {0:1d} and actual label: {1:1d}'.format(predictions_labels[image_index], labels_batch[image_index]))
        plt.imshow(data_batch[image_index, :, :, :])
        plt.show()

# close the session
session.close()
