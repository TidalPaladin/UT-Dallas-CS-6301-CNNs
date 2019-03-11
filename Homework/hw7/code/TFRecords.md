---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.0'
      jupytext_version: 1.0.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Homework 07


# Problem 2 <a name="introduction"></a>

## 26 <a name="26"></a>

We have a residual layer with the following reverse graph

```python
%load_ext tikzmagic
```

To calculate $\frac{\partial e}{\partial x}$ we will need
to apply the chain rule through $f(x)$ and sum on the
residual merge. This gives

$$
\frac{\partial e}{\partial x} = 
    \frac{\partial f}{\partial x}
    \cdot
    \frac{\partial e}{\partial y} 
$$
## 27 <a name="27"></a>

## 28 <a name="28"></a>

# Problem 3 <a name="p3"></a>
The Tiny ImageNet dataset can be downloaded 
[here](http://cs231n.stanford.edu/tiny-imagenet-200.zip)
Extract the zip onto a fast disk drive. First we will set up the
python environment and imports

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import re
from IPython.display import clear_output

from tensorflow.keras.callbacks import ModelCheckpoint, ProgbarLogger
from tensorflow.losses import sparse_softmax_cross_entropy as softmax_xent
from tensorflow.data import TFRecordDataset
from tensorflow.data.experimental import TFRecordWriter
#from tensorflow.data.experimental import naive_shard

# For TFRecord demo
tf.logging.set_verbosity(tf.logging.INFO)
#tf.enable_eager_execution()

# Train on secondary GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

```

Looking at the directory
structure of the dataset, we see that there are subdirectory for
training, validating, and testing. The training set contains
200 classes where images of a class are grouped by directory.




## Preprocessing <a name="introduction"></a>

We can import the training set with preprocessing as follows
(documentation available 
[here](https://keras.io/preprocessing/image/).)

We will use the `ImageDataGenerator`. This works will on the
training set by default

```python

class Constants(object):
    def __init__(self, d):
        self.__dict__ = d

# Constants for important filepaths
DATASET_ROOT = '/home/tidal/tiny-imagenet-200/tiny-imagenet-200'
DIRS = {
    'root' : '/',
    'tfrecord' :  'tfrecords',
    'train' : 'train',
    'test' :  'test',
    'val' : 'val',
    'checkpoint' : 'checkpoints'
}
DIRS = { k : os.path.join(DATASET_ROOT, v) for k, v in DIRS.items() }
DIRS = Constants(DIRS)

# Input shape, batching, and data type
inputs = tf.keras.layers.Input(
    shape=[3, 64, 64],
    name='input',
    dtype=tf.float32
)

# Ground truth sparse labels
labels = tf.placeholder(
    dtype=tf.int32,
    shape=[None, 1],
    name='label'
)

# Training parameters
TRAIN = {
    'num_classes' : 200,
    'batch_size' : 128,
    'shuffle' : 5000,
    'num_epochs' : 72,
    'momentum' : 0.9,
    'regularizer_scale' : 0.1,
    'lr_initial' : 0.01,
    'lr_scale' : 0.1,
    'lr_epoch' : 64,
    'val_split' : 0.2,
    'lr_staircase' : True,
    'max_checkpoint' : 5,
    'checkpoint_fmt' : 'resnet_{epoch:02d}.hdf5'
}
TRAIN = Constants(TRAIN)

# Shard generation parameters
TFRECORD = {
    'file_format' : os.path.join(DIRS.tfrecord, 'tin_train_%i.tfrecord'),
    'train_glob' : os.path.join(DIRS.tfrecord, '.*_train_.*tfrecord'),
    'val_glob' : os.path.join(DIRS.tfrecord, '.*_val_.*tfrecord'),
    'num_shards' : 30
}
TFRECORD = Constants(TFRECORD)

CONST = Constants({'train' : TRAIN, 'tfrecord' : TFRECORD, 'dirs' : DIRS})
```

### Importing to a `Dataset`

Now that we have defined constants, we can begin reading training data in
preparation for writing sharded `TFRecord` files. Keras provides the
`ImageDataGenerator` which accepts formatting and preprocessing information
as arguments and returns an `ImageDataGenerator` object. One method of this
returned object is the `flow_from_directory` method which automatically
interprets the file structure of the training set and returns an iterator
over training files.

First we create the `ImageDataGenerator`. We will use the following args
 * `samplewise_center=True` - Normalize to zero mean
 * `samplewise_std_normalization=True` - Normalize to unit variance
 * `horizontal_flip=True` - Flip images
 * `data_format=True` - Generator should yield images with channels on axis 0
 * `rescale=1./255` - Rescale 8 bit images to a float on [0, 1]

**Note** that no shuffling was used - according to the documentation,
shuffling should be done after any sharding operations.

```python
# Define a seprate graph to handle TFRecord writing
shard_graph = tf.Graph()

# Define a data generator with preprocessing 
# Includes a ratio to reserve for validation
with shard_graph.as_default():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization=True,
            horizontal_flip=True,
            data_format='channels_first',
            rescale=1./255)
```

We next create a wrapper function for `flow_from_directory`. As stated
earlier, this method will return an iterator over the dataset. 
The purpose of using a wrapper function is to maintain code cleanliness
when the callable `flow_from_directory` will need to be passed to other
functions (will be more clear below). The arguments are as follows

 * `DIRS.train` - The directory to flow from
 * `target_size` - Shape of outputs from iterator
 * `batch_size` - Batching
 * `class_mode='sparse'` - Give an integer for label class, rather than a one
 hot vector

 **Note:** When using `class_mode='sparse'`, loss functions should be sparse
 as well, ie `sparse_softmax_cross_entropy`. By default `categorical` will be
 used, which yields the full one-hot label vector. In such cases, do **not**
 use a sparse loss function. Ambiguous errors will be produced regarding the
 dimentionality of inputs to the loss function when training begins.

We can also call the generator method to see a message on the number of
examples and classes found.

```python
# Wrap flow_from_directory in simple callable
def generator():

    return train_datagen.flow_from_directory(
            DIRS.train,
            target_size=inputs.shape[2:4].as_list(),
            batch_size=1,
            class_mode='sparse')
```

Now we have a function `generator()` that returns an iterator over the
training dataset. The iterator yields correctly batched groups of training
images / labels, with preprocessing already applied. Finally, we will
use this generator to produce a `Dataset` object. Note that there are 
other approaches where the generator can be used without this wrapping.

After wrapping we can verify the shapes and types of the `Dataset` are
as expected (with batching on the first dimension).

```python
with shard_graph.as_default():
    output_types = inputs.dtype, labels.dtype
    output_shapes = inputs.shape, labels.shape
    ds_train = tf.data.Dataset.from_generator(
            generator,
            output_types=output_types,
            output_shapes=output_shapes)
    ds_train
```

### Writing `.tfrecord` Files

Finally, we can produce `.tfrecord` files for our `Dataset`.

THIS JUST BUILDS THE GRAPH, NOTHING RUNS YET

```python

with shard_graph.as_default():
    for current_shard in range(0,TFRECORD.num_shards):
        filepath = TFRECORD.file_format % current_shard
        clear_output(wait=True)
        #print('Processing shard %i / %i' % (current_shard+1, TFRECORD.num_shards))
        #print('Path: %s' % filepath)

        writer = TFRecordWriter(filepath)

        # Create a Dataset with 1/num_shards elements
        shard = ds_train.shard(TFRECORD.num_shards, current_shard)

        # New way, wouldn't work with my tensorflow (no filter_for_shard)
        # shard_func = filter_for_shard(current_shard, TFRECORD.num_shards)
        # shard = ds_train.apply(shard_func)

        def serialize_tensor_tuple(img, label):
            # Serialize two separate tensors
            img_s = tf.serialize_tensor(img)
            label_s = tf.serialize_tensor(label)
            return tf.string_join([img_s, label_s])

        shard = shard.map(serialize_tensor_tuple)
        writer.write(shard)
```

```python

```

```python
# A method to extract an example from a record file
def parse_record(example_proto, clip=False):

    # The features contained in the written TFRecord
    tfrecord_read_features = {
           'image': tf.FixedLenFeature(shape=[], dtype=tf.string),
           'label': tf.FixedLenFeature(shape=[], dtype=tf.string)
    }

    example = tf.parse_single_example(example_proto, tfrecord_read_features)
    img = tf.decode_raw(example['image'], tf.float32)
    label = tf.decode_raw(example['label'], tf.float32)

    img = tf.reshape(img, inputs.shape[1:4])
    label = tf.reshape(label, (1,))
    label = tf.squeeze(label)
    label = tf.cast(label, int32)
    return img, label
```

```python
```
```python

# Construct a TFRecordDataset mapped over all shards
filenames = tf.data.Dataset.list_files(TFRECORD.train_glob)
ds_train = tf.data.TFRecordDataset(filenames).map(parse_record)
ds_train = ds_train.shuffle(TRAIN.shuffle)
ds_train = ds_train.batch(TRAIN.batch_size)
ds_train
```




```python
def make_shards():

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    def _int64_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    dataset_iterator = generator()
    print('Writing %i shards...' % TFRECORD.num_shards)

    for shard, batch in enumerate(dataset_iterator):
        if shard > TFRECORD.num_shards: break
        filepath = TFRECORD.file_format % shard
        print('  |-- %s' % os.path.basename(filepath))

        with tf.python_io.TFRecordWriter(filepath) as writer:

            for img, label in zip(batch[0], batch[1]):

                feature = {
                    'image': _bytes_feature(img.tostring()),
                    'label': _bytes_feature(label.tostring()),
                }
                
                features=tf.train.Features(feature=feature)
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())

make_shards()
```

But for the validation and testing sets the `ImageDataGenerator`
expects directory for each label. We must manually fix labels

# Problem 4 <a name="p4"></a>
Understood

# Problem 5 - Modifying Resnet <a name="p5"></a>

We can construct Resnet using a subclassed approach. This involves
creating modular blocks of layers that can be reused as needed, thus
increasing code reuseability and ease of maintainance. 

Specifically, we subclass `tf.keras.Model` and implement the methods
`__init__()` and `call()`. Our choice of `__init__()` method will define
the the types of layers in this block, but says nothing about how they
are connected. In the `call()` method we will define the connections
between layers. This method takes an input as a parameter and returns
an ouput that represents the feature maps after a forward pass through
all layers in the block.

## Basic Block <a name="basic"></a>

First we will define the fundamental CNN style 2D convolution block
of Resnet, ie

Note that the number of filters and the kernel size are 
parameterized, and that parameter packs `*args, **kwargs`
are forwarded to the convolution layer. This is important
as it enables the reuse of this model for the various
types of convolutions that we will need.

```python
class ResnetBasic(tf.keras.Model):

    def __init__(self, filters, kernel_size, strides=(1,1), *args, **kwargs):
        super(ResnetBasic, self).__init__(*args, **kwargs)
        self.batch_norm = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv2d = layers.Conv2D(
                filters,
                kernel_size,
                padding='same',
                data_format='channels_last',
                activation=None,
                use_bias=False,
                strides=strides)

    def call(self, inputs, **kwargs):
        x = self.batch_norm(inputs, **kwargs)
        x = self.relu(x, **kwargs)
        return self.conv2d(x, **kwargs)
```


## Standard Bottleneck

Recognizing this, we can define a bottleneck layer. Again,
the number of input feature maps is parameterized. 
We no longer parameterize the kernel dimensions, as these
are intrinsic to this type of block.


```python
class Bottleneck(tf.keras.Model):

    def __init__(self, Ni, *args, **kwargs):
        super(Bottleneck, self).__init__(*args, **kwargs)

        # Three residual convolution blocks
        kernels = [(1, 1), (3, 3), (1, 1)]
        feature_maps = [Ni // 4, Ni // 4, Ni]
        self.residual_filters = [
            ResnetBasic(N, K) 
            for N, K in zip(feature_maps, kernels) 
        ] 

        # Merge operation
        self.merge = layers.Add()

    def call(self, inputs, **kwargs):

        # Residual forward pass
        res = inputs
        for res_layer in self.residual_filters:
            res = res_layer(res, **kwargs)

        # Combine residual pass with identity
        return self.merge([inputs, res], **kwargs)
```

## Special Bottleneck

We can define the special bottleneck layer by subclassing
the `Bottleneck` class as follows.

```python
class SpecialBottleneck(Bottleneck):

    def __init__(self, Ni, *args, **kwargs):

        # Layers that also appear in standard bottleneck
        super(SpecialBottleneck, self).__init__(Ni, *args, **kwargs)

        # Add convolution layer along main path
        self.main = layers.Conv2D(
                Ni,
                (1, 1),
                padding='same',
                data_format='channels_last',
                activation=None,
                use_bias=False)

    def call(self, inputs, **kwargs):

        # Residual forward pass
        res = inputs
        for res_layer in self.residual_filters:
            res = res_layer(res, **kwargs)

        # Convolution on main forward pass
        main = self.main(inputs, **kwargs)

        # Merge residual and main
        return self.merge([main, res])
```

## Downsampling

Next we need to define the downsampling layer.

```python
class Downsample(tf.keras.Model):

    def __init__(self, Ni, *args, **kwargs):
        super(Downsample, self).__init__(*args, **kwargs)

        # Three residual convolution blocks
        kernels = [(1, 1), (3, 3), (1, 1)]
        strides = [(2, 2), (1, 1), (1, 1)]
        feature_maps = [Ni // 2, Ni // 2, 2*Ni]

        self.residual_filters = [
            ResnetBasic(N, K, strides=S) 
            for N, K, S in zip(feature_maps, kernels, strides) 
        ] 

        # Convolution on main path
        self.main = ResnetBasic(2*Ni, (1,1), strides=(2,2))

        # Merge operation for residual and main
        self.merge = layers.Add()

    def call(self, inputs, **kwargs):

        # Residual forward pass
        res = inputs
        for res_layer in self.residual_filters:
            res = res_layer(res,**kwargs)

        # Main forward pass
        main = self.main(inputs, **kwargs)

        # Merge residual and main
        return self.merge([main, res])

```

## Final Model

Finally, we can assemble these blocks into the final model. Note
that the tail and other simple layers are defined within the 
Resnet model class, rather than being subclassed as we did
with the other building blocks. This choice came down to the
simplicity of tail and other non-subclassed layers.

Also worth noting is the use of `layers.GlobalAveragePooling2D`. There
is no `keras.layers.reduce_mean()` layer, but this operation represents
global average pooling so we simply choose the correct layer class.

```python
class Resnet(tf.keras.Model):

    def __init__(self, classes, filters, levels, *args, **kwargs):
        super(Resnet, self).__init__(*args, **kwargs)


        # Lists to hold various layers
        self.blocks = list()

        # Tail
        self.tail = layers.Conv2D(
                filters,
                (3, 3),
                padding='same',
                data_format='channels_last',
                use_bias=False,
                name='tail')

        # Special bottleneck layer with convolution on main path
        self.level_0_special = SpecialBottleneck(filters)

        # Loop through levels and their parameterized repeat counts
        for level, repeats in enumerate(levels):
            for block in range(repeats):
                # Append a bottleneck block for each repeat
                name = 'bottleneck_%i_%i' % (level, block)
                layer = Bottleneck(filters, name=name)
                self.blocks.append(layer)

            # Downsample and double feature maps at end of level
            name = 'downsample_%i' % (level)
            layer = Downsample(filters, name=name)
            self.blocks.append(layer) 
            filters *= 2

        # encoder - level 2 special block x1
        # input:  256 x   8 x 8
        # output: 256 x   8 x 8
        self.level2_batch_norm = layers.BatchNormalization()
        self.level2_relu = layers.ReLU()

        # Decoder - global average pool and fully connected
        self.global_avg = layers.GlobalAveragePooling2D(
                data_format='channels_last')
        self.dense = layers.Dense(classes, 
                use_bias=True)


    def call(self, inputs, **kwargs):
        x = self.tail(inputs, **kwargs)
        x = self.level_0_special(x)

        # Loop over layers by level
        for layer in self.blocks:
            x = layer(x, **kwargs)

        # Finish up specials in level 2
        x = self.level2_batch_norm(x, **kwargs)
        x = self.level2_relu(x)

        # Decoder
        x = self.global_avg(x)
        return self.dense(x, **kwargs)
```

## Using the Model

Now that we have defined a subclassed model, we need to
incorproate it into a training / testing environment. This is
where the beauty of the subclassed approach comes in. 
In our case
we want construct Resnet modified for Tiny Imagenet, where the
modifications are as follows:

 * Third level of residual blocks + downsampling
 * Full and half width versions

Our Resnet class accepts an interable of integers to define the
number of repeats at each level. As such, we need only add an
integer for the number of repeats at level 3 to our constructor call.
Similarly, we can scale the number of feature maps as needed to adjust
width.

Lastly we will provide the number of classes in Tiny Imagenet, ie $200$.

```python
# Add another level
standard_levels = [4, 6, 3]
new_level_count = 2
modified_levels = standard_levels + [new_level_count]

# Define full and half width feature map count
full_width = 64
half_width = full_width / 2

# Tiny Imagenet properties


model = Resnet(TRAIN.num_classes, full_width, modified_levels)
outputs = model(inputs)
```

Note that `model` returned by our class constructor is callable.
Thus our forward pass mapping inputs to outputs is invoked by
"calling" `model` on the inputs and storing the returned outputs.
Note that a call on model is simply calling the 
`Resnet.call()` method we wrote earlier. More on this when we get
to training.

Finally, we can build the model for the appropriate input shape
and get a summary of the included layers

```python
model.summary()
```
# Problem 6 - Saving Validation

Now we can construct a training loop with the following additional
features

 * Saving validation loss/accuracy on a per epoch basis
 * Checkpointing after each epoch with ability to restore from 
   checkpoint

First we will define training hyperparameters to be used later on

```python

```

## Defining Metrics

Next we define an accuracy and loss metric, as well as an optimizer

```python
# accuracy
metrics = ['accuracy']

# loss
loss = softmax_xent
#loss = 'categorical_crossentropy'

# optimizer
optimizer = tf.train.AdamOptimizer()

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
```

Checkpointing and whatnot

```python
checkpoint_path = os.path.join(DIRS.checkpoint, TRAIN.checkpoint_fmt)
callbacks = [ 
        ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True),
        #ProgbarLogger(),
]
```

```python
step_size = 100000 * TFRECORD.num_shards // TRAIN.batch_size
model.fit(
        ds_train,
        #callbacks=callbacks,
        steps_per_epoch=step_size,
        epochs=TRAIN.num_epochs)
```

```python
#for x, y in ds_train.take(1):
    #print(y.shape)
    #y.shape
steps_per_epoch
128*72
```

```python
flops = tf.profiler.profile(options = tf.profiler.ProfileOptionBuilder.float_operation())
mem = tf.profiler.profile(options = tf.profiler.ProfileOptionBuilder.time_and_memory())
if flops is not None:
    print('Calculated FLOP', flops.total_float_ops)
if mem is not None:
    print(mem)
```
