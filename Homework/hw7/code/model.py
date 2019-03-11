#!python3

import tensorflow as tf
from tensorflow.keras import layers

class Resnet463(Resnet):

    def __init__(self, classes, filters, *args, **kwargs):
        super(Resnet463, self).__init__(
                classes,
                filters,
                *args,
                **kwargs)

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

