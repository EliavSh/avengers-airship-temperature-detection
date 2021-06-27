from abc import ABC, abstractmethod


class AbstractLoader(ABC):
    """
    An abstract loader object. any loader should have the 'load' method for a generic use.
    """
    @abstractmethod
    def load(self, img_folder: str):
        raise NotImplementedError
