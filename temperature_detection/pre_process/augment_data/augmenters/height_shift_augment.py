from typing import List

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from temperature_detection.pre_process.augment_data.abstarct_augment import AbstractAugment


class HeightShiftAugment(AbstractAugment):

    def __init__(self, image_folder, height_shift_range):
        super().__init__(image_folder)
        self.image_data_generator = ImageDataGenerator(height_shift_range=height_shift_range, fill_mode='constant', cval=255)
        print('Preparing to augment images by height shift within range of: [', '-' + str(abs(height_shift_range)), ',', str(abs(height_shift_range)), '] ...')

    def internal_augment(self, img) -> List[np.array]:
        it = self.image_data_generator.flow(img, batch_size=1)
        return [it.next()[0].astype('uint8')]
