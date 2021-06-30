from enum import Enum

from .augmenters import *


class AugmentEnum(Enum):
    RotationAugment = RotationAugment
    WidthShiftAugment = WidthShiftAugment
    HeightShiftAugment = HeightShiftAugment
    CompositeAugment = CompositeAugment

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)
