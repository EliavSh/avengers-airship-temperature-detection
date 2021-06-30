from enum import Enum

from .models import *


class ModelEnum(Enum):
    """
    An Enum that is responsible of model selection.
    """
    DoubleConv = DoubleConv
    TripleConv = TripleConv
    PreTrained = PreTrained

    def get(self, *args):
        return self.value(*args)
