from typing import List

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from temperature_detection.pre_process.augment_data.abstarct_augment import AbstractAugment


class RotationAugment(AbstractAugment):

    def __init__(self, image_folder, rotation_range):
        super().__init__(image_folder)
        self.image_data_generator = ImageDataGenerator(rotation_range=rotation_range)
        print('Preparing to augment images by random rotation within range of: [', '-' + str(abs(rotation_range)), ',', str(abs(rotation_range)), '] ...')

    def internal_augment(self, img) -> List[np.array]:
        it = self.image_data_generator.flow(img, batch_size=1)
        return [it.next()[0].astype('uint8')]
