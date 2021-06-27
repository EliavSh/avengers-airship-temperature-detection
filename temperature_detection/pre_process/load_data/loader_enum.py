from enum import Enum
from temperature_detection.pre_process.load_data.directory_loader import DirectoryLoader
from temperature_detection.pre_process.load_data.image_loader import ImageLoader


class LoaderEnum(Enum):
    """
    An Enum class which is responsible for calling a specific loader object.
    """
    DirectoryLoader = DirectoryLoader
    ImageLoader = ImageLoader

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)
