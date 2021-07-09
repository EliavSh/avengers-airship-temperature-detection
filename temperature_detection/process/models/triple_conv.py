import tensorflow as tf

from temperature_detection.pre_process.pre_process_conf import PreProcessConf
from temperature_detection.utils import Utils

IMG_HEIGHT = PreProcessConf.IMG_HEIGHT
IMG_WIDTH = PreProcessConf.IMG_WIDTH
IMAGES_TO_SUMMARY = PreProcessConf.IMAGES_TO_SUMMARY
DISPLAY_IMAGES = PreProcessConf.DISPLAY_IMAGES
NUM_LABELS = PreProcessConf.NUM_LABELS


class TripleConv(tf.keras.Model):
    """
    The class is responsible for building a 3-conv layered model with a final dense layer for specific output
    """

    def get_config(self):
        return {'num_filters': 64}

    def __init__(self):
        super(TripleConv, self).__init__()
        self._name = "TripleConv"

        # build model
        self.conv1 = tf.keras.layers.Conv2D(filters=self.get_config()['num_filters'], kernel_size=3, activation='relu')
        self.max_pool1 = tf.keras.layers.MaxPooling2D()
        self.conv2 = tf.keras.layers.Conv2D(filters=self.get_config()['num_filters'], kernel_size=3, activation='relu')
        self.max_pool2 = tf.keras.layers.MaxPooling2D()
        self.conv3 = tf.keras.layers.Conv2D(filters=self.get_config()['num_filters'], kernel_size=3, activation='relu')
        self.max_pool3 = tf.keras.layers.MaxPooling2D()

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(NUM_LABELS, activation='softmax')

        # build and summary
        self.build(input_shape=(None, IMG_WIDTH, IMG_HEIGHT, 3))
        self.summary()

    def call(self, inputs, **kwargs):
        # forward pass of the model

        conv1_out = self.conv1(inputs)  # out: (None, 194, 194, 128)
        max_pool1_out = self.max_pool1(conv1_out)

        conv2_out = self.conv2(max_pool1_out)
        max_pool2_out = self.max_pool2(conv2_out)

        conv3_out = self.conv3(max_pool2_out)
        max_pool3_out = self.max_pool3(conv3_out)

        flatten_out = self.flatten(max_pool3_out)
        outputs = self.dense(flatten_out)

        # displaying input image and convolutional filters images
        if DISPLAY_IMAGES:
            Utils.image_summary(inputs, 'input image', 1, self._train_counter)
            Utils.image_summary(conv1_out, 'after_conv_1', IMAGES_TO_SUMMARY, self._train_counter)
            Utils.image_summary(conv2_out, 'after_conv_2', IMAGES_TO_SUMMARY, self._train_counter)
            Utils.image_summary(conv3_out, 'after_conv_3', IMAGES_TO_SUMMARY, self._train_counter)

        # summary the chosen class for tensorboard visualization
        tf.summary.histogram('outputs', tf.argmax(outputs, axis=1), self._train_counter)

        return outputs
