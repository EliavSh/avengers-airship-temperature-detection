import cv2
import numpy as np

from temperature_detection.pre_process.load_data.abstract_loader import AbstractLoader
from temperature_detection.pre_process.pre_process_conf import PreProcessConf

IMG_HEIGHT = PreProcessConf.IMG_HEIGHT
IMG_WIDTH = PreProcessConf.IMG_WIDTH


class ImageLoader(AbstractLoader):
    """
    Loading a single image file, used for testing.
    """

    def load(self, img_path: str):
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        image = np.true_divide(np.array(image).astype('float32'), 255.)

        return image
