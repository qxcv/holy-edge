# Adapted from : VGG 16 model : https://github.com/machrisaa/tensorflow-vgg
import time
import os

import numpy as np
import tensorflow as tf

from hed.losses import sigmoid_cross_entropy_balanced
from hed.utils.io import IO


class Vgg16():
    def __init__(self, cfgs, run='training'):
        self.cfgs = cfgs
        self.io = IO()

        self.training = tf.placeholder(shape=(), dtype=tf.bool)

        base_path = os.path.abspath(os.path.dirname(__file__))
        weights_file = os.path.join(base_path, self.cfgs['model_weights_path'])

        self.data_dict = np.load(weights_file, encoding='latin1').item()
        self.io.print_info("Model weights loaded from {}".format(
            self.cfgs['model_weights_path']))

        self.images = tf.placeholder(tf.float32, [
            None, self.cfgs[run]['image_height'],
            self.cfgs[run]['image_width'], self.cfgs[run]['n_channels']
        ])
        self.edgemaps = tf.placeholder(tf.float32, [
            None, self.cfgs[run]['image_height'],
            self.cfgs[run]['image_width'], 1
        ])

        self.define_model()

    def define_model(self):
        """
        Load VGG params from disk without FC layers A
        Add branch layers (with deconv) after each CONV block
        """
        start_time = time.time()

        # try using static shapes
        in_hw = self.get_hw(self.images)

        self.conv1_1 = self.conv_layer(self.images, 64, (3, 3), name="conv1_1")
        self.conv1_2 = self.conv_layer(
            self.conv1_1, 64, (3, 3), name="conv1_2")
        self.side_1 = self.side_layer(self.conv1_2, "side_1", in_hw)
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.io.print_info('Added CONV-BLOCK-1+SIDE-1')

        self.conv2_1 = self.conv_layer(self.pool1, 128, (3, 3), name="conv2_1")
        self.conv2_2 = self.conv_layer(
            self.conv2_1, 128, (3, 3), name="conv2_2")
        self.side_2 = self.side_layer(self.conv2_2, "side_2", in_hw)
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.io.print_info('Added CONV-BLOCK-2+SIDE-2')

        self.conv3_1 = self.conv_layer(self.pool2, 256, (3, 3), name="conv3_1")
        self.conv3_2 = self.conv_layer(
            self.conv3_1, 256, (3, 3), name="conv3_2")
        self.conv3_3 = self.conv_layer(
            self.conv3_2, 256, (3, 3), name="conv3_3")
        self.side_3 = self.side_layer(self.conv3_3, "side_3", in_hw)
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.io.print_info('Added CONV-BLOCK-3+SIDE-3')

        self.conv4_1 = self.conv_layer(self.pool3, 512, (3, 3), name="conv4_1")
        self.conv4_2 = self.conv_layer(
            self.conv4_1, 512, (3, 3), name="conv4_2")
        self.conv4_3 = self.conv_layer(
            self.conv4_2, 512, (3, 3), name="conv4_3")
        self.side_4 = self.side_layer(self.conv4_3, "side_4", in_hw)
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.io.print_info('Added CONV-BLOCK-4+SIDE-4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, (3, 3), name="conv5_1")
        self.conv5_2 = self.conv_layer(
            self.conv5_1, 512, (3, 3), name="conv5_2")
        self.conv5_3 = self.conv_layer(
            self.conv5_2, 512, (3, 3), name="conv5_3")
        self.side_5 = self.side_layer(self.conv5_3, "side_5", in_hw)

        self.io.print_info('Added CONV-BLOCK-5+SIDE-5')

        self.side_outputs = [
            self.side_1, self.side_2, self.side_3, self.side_4, self.side_5
        ]

        def upscale_cat_2x(small_x, same_x, *, name='upscale', out_chans=32):
            # first reduce dimensionality
            if small_x.shape.as_list()[3] > out_chans:
                # 1x1 downsample
                small_x = self.conv_layer(
                    small_x,
                    out_chans, (1, 1),
                    name=name + '/small_conv1',
                    padding='same')
            if same_x.shape.as_list()[3] > out_chans:
                # do same
                same_x = self.conv_layer(
                    same_x,
                    out_chans, (1, 1),
                    name=name + '/same_conv1',
                    padding='same')

            # now resize smaller one and join together two layers
            small_out_shape = self.get_hw(same_x)
            scaled_small_x = tf.image.resize_bilinear(
                small_x, size=small_out_shape, align_corners=True)

            def double_conv(first_map, second_map, kernel_size=(3, 3),
                            name=''):
                out_filters = first_map.shape.as_list()[-1]
                # only first_conv has bias
                first_conv = tf.layers.conv2d(
                    first_map,
                    out_filters,
                    kernel_size,
                    padding='same',
                    name=name + '_conv1')
                second_conv = tf.layers.conv2d(
                    second_map,
                    out_filters,
                    kernel_size,
                    use_bias=False,
                    padding='same',
                    name=name + '_conv2')
                sums = first_conv + second_conv
                return tf.nn.sigmoid(sums)

            # GRU-ish update, where we treat scaled_small_x as hidden value and
            # small_x as new input
            reset = double_conv(
                scaled_small_x, same_x, name=name + '/gate/reset')
            update = double_conv(
                scaled_small_x, same_x, name=name + '/gate/update')
            attenuated_in = scaled_small_x * reset
            new_state = double_conv(
                attenuated_in, same_x, name=name + '/gate/new_state')
            out = update * scaled_small_x + (1 - update) * new_state

            return out

        # new fuse strategy: go from lowest to highest, upscaling +
        # concatenating
        side_convs = [
            self.conv1_2, self.conv2_2, self.conv3_3, self.conv4_3,
            self.conv5_3
        ]
        small_side = side_convs[-1]
        for idx, next_side in enumerate(side_convs[:-1][::-1]):
            small_side = upscale_cat_2x(
                small_x=small_side, same_x=next_side, name='fuse_up_%d' % idx)
        # big conv for smoothing
        small_side = self.conv_layer(small_side, 32, (5, 5), name='fuse_conv')

        self.fuse = tf.layers.conv2d(
            small_side, 1, (1, 1), name='fuse_1', use_bias=False)

        self.io.print_info('Added FUSE layer')

        # complete output maps from side layer and fuse layers
        self.outputs = self.side_outputs + [self.fuse]

        self.data_dict = None
        self.io.print_info(
            "Build model finished: {:.4f}s".format(time.time() - start_time))

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(
            bottom,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name=name)

    def get_hw(self, batch):
        """Get height & width of images in NHWC batch. Tries to get static
        height/width, but returns Tensor if that fails."""
        hw = batch.shape.as_list()[1:3]
        if any(d is None for d in hw):
            # or use dynamic shape instead
            hw = tf.shape(batch)[1:3]
        return hw

    def conv_layer(self,
                   x,
                   filters,
                   shape,
                   *,
                   name=None,
                   padding='same',
                   use_bn=True,
                   activation=tf.nn.relu):
        assert name is not None
        # TODO: add summary histogram for conv2d and batch norm weights
        if name in self.data_dict:
            W, b = self.data_dict[name]
            kernel_initializer = tf.constant_initializer(W)
            bias_initializer = tf.constant_initializer(b)
        else:
            self.io.print_info('Using new weights for layer "%s"' % name)
            kernel_initializer = None
            bias_initializer = tf.zeros_initializer()
        x = tf.layers.conv2d(
            x,
            filters,
            shape,
            padding=padding,
            name=name + '/conv',
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer)
        x = activation(x, name=name + '/nonlin')
        if use_bn:
            x = tf.layers.batch_normalization(
                x, training=self.training, name=name + '/bn')
        return x

    def side_layer(self, inputs, name, out_shape):
        """
            https://github.com/s9xie/hed/blob/9e74dd710773d8d8a469ad905c76f4a7fa08f945/examples/hed/train_val.prototxt#L122
            1x1 conv followed with Deconvoltion layer to upscale the size of
            input image sans color
        """
        with tf.variable_scope(name):
            # classify + upscale (saves memory late in the net)
            classifier = tf.layers.conv2d(
                inputs, 1, (1, 1), name=name + '_reduction')
            classifier = tf.image.resize_bilinear(
                classifier,
                size=out_shape,
                align_corners=True,
                name=name + '_upscale')

            return classifier

    def setup_testing(self, session):
        """
            Apply sigmoid non-linearity to side layer ouputs + fuse layer
            outputs for predictions
        """
        self.predictions = []

        for idx, b in enumerate(self.outputs):
            output = tf.nn.sigmoid(b, name='output_{}'.format(idx))
            self.predictions.append(output)

    def setup_training(self, session):
        """
            Apply sigmoid non-linearity to side layer ouputs + fuse layer
            outputs

            Compute total loss := side_layer_loss + fuse_layer_loss

            Compute predicted edge maps from fuse layer as pseudo performance
            metric to track
        """
        self.predictions = []
        self.loss = 0

        self.io.print_warning('Deep supervision application set to {}'.format(
            self.cfgs['deep_supervision']))

        for idx, b in enumerate(self.side_outputs):
            output = tf.nn.sigmoid(b, name='output_{}'.format(idx))
            cost = sigmoid_cross_entropy_balanced(
                b, self.edgemaps, name='cross_entropy{}'.format(idx))

            self.predictions.append(output)
            if self.cfgs['deep_supervision']:
                self.loss += (self.cfgs['loss_weights'] * cost)

        fuse_output = tf.nn.sigmoid(self.fuse, name='fuse')
        fuse_cost = sigmoid_cross_entropy_balanced(
            self.fuse, self.edgemaps, name='cross_entropy_fuse')

        self.predictions.append(fuse_output)
        self.loss += (self.cfgs['loss_weights'] * fuse_cost)

        pred = tf.cast(
            tf.greater(fuse_output, 0.5), tf.int32, name='predictions')
        error = tf.cast(
            tf.not_equal(pred, tf.cast(self.edgemaps, tf.int32)), tf.float32)
        self.error = tf.reduce_mean(error, name='pixel_error')

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('error', self.error)

        self.merged_summary = tf.summary.merge_all()

        with tf.name_scope(name='val_images'):
            chan_swap = self.cfgs['channel_swap']
            inv_chan_swap = tf.invert_permutation(chan_swap)
            mean_px = np.asarray(
                self.cfgs['mean_pixel_value'], dtype='float32') \
                .reshape(1, 1, 1, 3)
            in_images_normed = tf.gather(
                self.images + mean_px, inv_chan_swap, axis=3) / 255

            def tf_grey2rgb(grey):
                return tf.tile(grey, (1, 1, 1, 3))

            # we need to convert edge maps to RGB and undo the channel swap on
            # self.images, then make a 3x3 grid of outputs
            cells = [tf_grey2rgb(self.edgemaps), in_images_normed] \
                + [tf_grey2rgb(out) for out in self.predictions] \
                + [tf_grey2rgb(tf.cast(pred, tf.float32))]
            assert len(cells) == 9
            grid = [cells[:3], cells[3:6], cells[6:]]
            im_summary_tensor = tf.concat(
                [tf.concat(r, axis=2) for r in grid],
                axis=1,
                name='im_summary_tensor')
            self.val_im_summary = tf.summary.image(
                'summary_op', im_summary_tensor, max_outputs=16)

        self.train_writer = tf.summary.FileWriter(
            self.cfgs['save_dir'] + '/train', session.graph)
        self.val_writer = tf.summary.FileWriter(self.cfgs['save_dir'] + '/val')
