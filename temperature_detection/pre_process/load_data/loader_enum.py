from enum import Enum
from temperature_detection.pre_process.load_data.image_loader import ImageLoader


class LoaderEnum(Enum):
    ImageLoader = ImageLoader

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)
