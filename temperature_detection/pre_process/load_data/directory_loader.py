import os
from typing import List

import cv2
import numpy as np
from sklearn.utils import shuffle

from temperature_detection.pre_process.load_data.abstract_loader import AbstractLoader
from temperature_detection.pre_process.pre_process_conf import PreProcessConf

IMG_HEIGHT = PreProcessConf.IMG_HEIGHT
IMG_WIDTH = PreProcessConf.IMG_WIDTH


class DirectoryLoader(AbstractLoader):
    """
    Loading .png files from a directory that consists of directories corresponding to each class
    """

    def load(self, img_folder: str, is_shuffle=False):
        img_data_array = []
        class_name = []

        for dir1 in os.listdir(img_folder):
            for file in os.listdir(os.path.join(img_folder, dir1)):
                image_path = os.path.join(img_folder, dir1, file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
                image = np.array(image).astype('float32') / 255.0
                img_data_array.append(image)

                class_name.append(dir1)

        target_value = self.to_target_value(class_name)

        if is_shuffle:
            img_data_array, target_value, class_name = shuffle(img_data_array, target_value, class_name)

        # finally, expand dims of image to contain 3-channels - as all our models expect
        x = np.repeat(np.array(img_data_array)[..., np.newaxis], 3, -1)

        return x, np.array(target_value), np.array(class_name)

    @staticmethod
    def to_target_value(class_arr: List[str]):
        target_dict = {k: v for v, k in enumerate(np.unique(class_arr))}
        return [target_dict[class_arr[i]] for i in range(len(class_arr))]
