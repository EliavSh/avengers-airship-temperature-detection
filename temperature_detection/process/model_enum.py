from enum import Enum

from .models import *


class ModelEnum(Enum):
    SimpleModel = SimpleModel

    def get(self, *args):
        return self.value(*args)
