import numpy as np

from temperature_detection.pre_process.augment_data.abstarct_augment import AbstractAugment


class CompositeAugment(AbstractAugment):

    def __init__(self, image_folder, *args, **kwargs):
        super().__init__(image_folder)
        self.augmenters = [augmenter.value(self.image_folder, param[1]) for augmenter, param in zip([*args], kwargs.items())]

    def internal_augment(self, img) -> np.array:
        images = []
        for augmenter in self.augmenters:
            images.append(augmenter.internal_augment(img)[0])
        return images
