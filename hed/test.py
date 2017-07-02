import os
import sys
import argparse
import yaml
import urlparse
import urllib
import StringIO
import cStringIO
import numpy as np
from PIL import Image
import tensorflow as tf

from hed.models.vgg16 import Vgg16
from hed.utils.io import IO


class HEDTester():

    def __init__(self, config_file):

        self.io = IO()

        try:
            pfile = open(config_file)
            self.cfgs = yaml.load(pfile)
            pfile.close()

        except Exception as err:

            self.io.print_error('Error reading config file {}, {}'.format(config_file), err)

    def setup(self, session):

        try:

            self.model = Vgg16(self.cfgs, run='testing')

            meta_model_file = os.path.join(self.cfgs['save_dir'], 'models/hed-model-{}'.format(self.cfgs['test_snapshot']))
            saver = tf.train.Saver()
            saver.restore(session, meta_model_file)

            self.io.print_info('Done restoring VGG-16 model from {}'.format(meta_model_file))

        except Exception as err:

            self.io.print_error('Error setting up VGG-16 model, {}'.format(err))

    def run(self, session):

        self.model.setup_testing(session)

        train_list = self.io.read_file_list(self.cfgs['testing']['list'])
        # np.random.shuffle(train_list)

        for idx, img in enumerate(train_list[:self.cfgs['test_samples']]):

            test_filename = os.path.join(self.cfgs['testing']['dir'], img)

            im = self.fetch_image(test_filename)
            self.io.print_info('Testing {}, {}'.format(test_filename, im.shape))

            edgemap = session.run(self.model.predictions, feed_dict={self.model.images: [im]})

            # Take the edge map from the network from side layers and fuse layer
            fused_edge_map = 255.0 * np.mean(np.array(edgemap), axis=4, keepdims=True)[0][0]
            em = np.tile(fused_edge_map, [1, 1, 3])

            em = Image.fromarray(np.uint8(em))
            em.save(os.path.join(self.cfgs['test_output'], 'testing-{}.png'.format(idx)))

    def fetch_image(self, test_image):

        # is url
        image = None

        if not urlparse.urlparse(test_image).scheme == "":

            url_response = urllib.urlopen(test_image)

            if url_response.code == 404:
                print self.io.print_error('[Testing] URL error code : {1} for {0}'.format(test_image, url_response.code))
                return None

            try:

                image_buffer = cStringIO.StringIO(url_response.read())
                image = self.capture_pixels(image_buffer)

            except Exception as err:

                print self.io.print_error('[Testing] Error with URL {0} {1}'.format(test_image, err))
                return None

        # read from disk
        elif os.path.exists(test_image):

            try:

                fid = open(test_image, 'r')
                stream = fid.read()
                fid.close()

                image_buffer = cStringIO.StringIO(stream)
                image = self.capture_pixels(image_buffer)

            except Exception as err:

                print self.io.print_error('[Testing] Error with image file {0} {1}'.format(test_image, err))
                return None

        return image

    def capture_pixels(self, image_buffer):

        image = Image.open(image_buffer)
        image = image.resize((self.cfgs['testing']['image_width'], self.cfgs['testing']['image_height']))
        image = np.array(image, np.float32)
        image = self.colorize(image)

        image = image[:, :, self.cfgs['channel_swap']]
        image -= self.cfgs['mean_pixel_value']

        return image

    def colorize(self, image):

        # BW to 3 channel RGB image
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
            image = np.tile(image, (1, 1, 3))
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        return image