import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import numpy as np
from matplotlib.image import imsave
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

from temperature_detection.pre_process.pre_process_conf import PreProcessConf


class AbstractAugment(ABC):
    def __init__(self, image_folder):
        self.image_folder = image_folder

    def augment(self, num_of_samples):
        for dir1 in os.listdir(self.image_folder):
            all_image_files = os.listdir(os.path.join(self.image_folder, dir1))

            for file in all_image_files:
                # load original image
                image = img_to_array(load_img(os.path.join(self.image_folder, dir1, file), color_mode="grayscale"))

                for i in range(num_of_samples):
                    augmented_images = self.internal_augment(np.expand_dims(image, 0))

                    # make sure all folders exist
                    Path(os.path.join(PreProcessConf.AUGMENT_DIR, dir1)).mkdir(parents=True, exist_ok=True)

                    for j, augmented_image in enumerate(augmented_images):
                        imsave(os.path.join(PreProcessConf.AUGMENT_DIR, dir1, file.split('.')[0] + '_' + str(i) + '_' + str(j) + '.png'), np.squeeze(augmented_image, -1))

                # finally, save the original image too
                imsave(os.path.join(PreProcessConf.AUGMENT_DIR, dir1, file), np.squeeze(image, -1))
            print('Finished augmenting images for class:', dir1 + ',', 'Total of', str(len(all_image_files) * (num_of_samples + len(augmented_images))) + '.')

    @abstractmethod
    def internal_augment(self, img) -> List[np.array]:
        raise NotImplementedError


"""
We need:
1. augment and save to folder
2. augment and save to some another folder

3. composite augmenter

4. do we want different type of augmentations for each class ?
"""
