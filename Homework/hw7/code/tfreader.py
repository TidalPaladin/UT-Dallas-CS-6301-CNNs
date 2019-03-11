from PIL import Image
import numpy as np
import tensorflow as tf
import IPython.display as display
from IPython.display import Image

import os
tf.enable_eager_execution()
from matplotlib import pyplot as plt



def show():
    def _parse_image_function(example_proto):
        # Create a dictionary describing the features.
        image_feature_description = {
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
        # Parse the input tf.Example proto using the dictionary above.
        return tf.parse_single_example(example_proto, image_feature_description)

    cwd = '/home/tidal/tiny-imagenet-200/tiny-imagenet-200'
    shards = '/home/tidal/shards.tfrecord'
    raw_image_dataset = tf.data.TFRecordDataset(shards)

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    for image_features in parsed_image_dataset:
        image_raw = image_features['label']
        print(image_raw)

if __name__ == '__main__':
    show()
