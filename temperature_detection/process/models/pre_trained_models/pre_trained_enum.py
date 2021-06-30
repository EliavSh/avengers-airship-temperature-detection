from enum import Enum

from tensorflow.keras.applications import *


class PreTrainedEnum(Enum):
    DenseNet121 = DenseNet121,
    DenseNet169 = DenseNet169,
    DenseNet201 = DenseNet201,
    EfficientNetB0 = EfficientNetB0,
    EfficientNetB1 = EfficientNetB1,
    EfficientNetB2 = EfficientNetB2,
    EfficientNetB3 = EfficientNetB3,
    EfficientNetB4 = EfficientNetB4,
    EfficientNetB5 = EfficientNetB5,
    EfficientNetB6 = EfficientNetB6,
    EfficientNetB7 = EfficientNetB7,
    InceptionResNetV2 = InceptionResNetV2,
    InceptionV3 = InceptionV3,
    MobileNet = MobileNet,
    MobileNetV2 = MobileNetV2,
    MobileNetV3Large = MobileNetV3Large,
    MobileNetV3Small = MobileNetV3Small,
    NASNetLarge = NASNetLarge,
    NASNetMobile = NASNetMobile,
    ResNet101 = ResNet101,
    ResNet152 = ResNet152,
    ResNet50 = ResNet50,
    ResNet101V2 = ResNet101V2,
    ResNet152V2 = ResNet152V2,
    ResNet50V2 = ResNet50V2,
    VGG16 = VGG16,
    VGG19 = VGG19,
    Xception = Xception

    def get(self, *args, **kwargs):
        return self.value[0](*args, **kwargs)
