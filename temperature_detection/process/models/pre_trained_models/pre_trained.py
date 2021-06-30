import tensorflow as tf

from temperature_detection.pre_process.pre_process_conf import PreProcessConf
from temperature_detection.process.models.pre_trained_models.pre_trained_enum import PreTrainedEnum

IMG_HEIGHT = PreProcessConf.IMG_HEIGHT
IMG_WIDTH = PreProcessConf.IMG_WIDTH
IMAGES_TO_SUMMARY = PreProcessConf.IMAGES_TO_SUMMARY
DISPLAY_IMAGES = PreProcessConf.DISPLAY_IMAGES
NUM_LABELS = PreProcessConf.NUM_LABELS


class PreTrained(tf.keras.Model):
    """
    The class is responsible for building a 2-conv layered model with a final dense layer for specific output
    """

    def get_config(self):
        return {}

    def __init__(self, pre_trained_model: PreTrainedEnum):
        super(PreTrained, self).__init__()

        # build model
        self.base_model = pre_trained_model.get(include_top=False)  # load pre-trained model without last dense layer/s

        self.global_average_pooling = tf.keras.layers.GlobalAveragePooling2D()

        # TODO - should we use single or double dense here? search the articles of 'PreTrainedEnum' for last layers architecture
        self.dense_1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(PreProcessConf.NUM_LABELS, activation='softmax')

        # build and summary
        self.build(input_shape=(None, IMG_WIDTH, IMG_HEIGHT, 3))
        self.summary()

    def call(self, inputs, **kwargs):
        # forward pass of the model
        base_model_out = self.base_model(inputs)  # out: (None, 194, 194, 128)
        global_average_pooling_out = self.global_average_pooling(base_model_out)

        dense_1_out = self.dense_1(global_average_pooling_out)
        dense_2_out = self.dense_2(dense_1_out)

        # displaying input image and convolutional filters images
        if DISPLAY_IMAGES:
            self.image_summary(inputs, 'input image', 1)
            self.image_summary(base_model_out, 'after_inception_v3', IMAGES_TO_SUMMARY)

        # summary the chosen class for tensorboard visualization
        tf.summary.histogram('outputs', tf.argmax(dense_2_out, axis=1))
        return dense_2_out

    @staticmethod
    def image_summary(x, name, num_images):
        # change dim of image and summary
        image_format = tf.transpose(x, [3, 1, 2, 0])
        image_format = image_format[0:num_images, :, :, 0]
        image_format = tf.expand_dims(image_format, -1)
        tf.summary.image(name=name, data=image_format, max_outputs=num_images)
