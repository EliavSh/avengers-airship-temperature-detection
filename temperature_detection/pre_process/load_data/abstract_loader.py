from abc import ABC, abstractmethod


class AbstractLoader(ABC):
    @abstractmethod
    def load(self, img_folder: str):
        raise NotImplementedError
