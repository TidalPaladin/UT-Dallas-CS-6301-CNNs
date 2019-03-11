from PIL import Image
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import os

class TFRecordFactory():

    def __init__(self, filename, source):
        """
        Construct a TFRecord factory producing files with a given basename

        Params
        ===
        filename : str
            The basename for the TFRecord files

        source : str
            Source directory for Tiny - Imagenet
        """
        self.source = source
        self.filename = filename
        self.classes = dict()

        # Read the words.txt file, build list of names and captions
        training_dir = os.path.join(source, 'train')
        self.training_labels = os.listdir(training_dir)

        with open(os.path.join(source, 'words.txt'), 'r') as f:
            for line in f:
                name, caption = line.split('\t')
                self.classes[name] = caption



    def run(self, shard_size):
        """
        Run the record generator, including a given number of examples
        in a single shard

        Params
        ===
        shard_size : int > 0
            Number of examples per shard

        Post
        ===
        TFRecord files written
        """

        with tf.python_io.TFRecordWriter(self.filename) as writer:

            for label, name in enumerate(self.training_labels):

                # Look up directory with training images
                class_dir = os.path.join(
                        self.source,
                        'train',
                        name)

                print(label)
                img_dir = os.path.join(class_dir, 'images')
                for img_file in os.listdir(img_dir):
                    img_file = os.path.join(img_dir, img_file)
                    img_raw = open(img_file, 'rb').read()

                    feature = {
                      'label': self._int64_feature(label),
                      'image_raw': self._bytes_feature(img_raw)
                    }

                    example = tf.train.Example(features=tf.train.Features(feature=feature))

                    writer.write(example.SerializeToString())


    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


if __name__ == '__main__':
    fac = TFRecordFactory(
            '/home/tidal/shards.tfrecord',
            '/home/tidal/tiny-imagenet-200/tiny-imagenet-200')
    fac.run(4)

