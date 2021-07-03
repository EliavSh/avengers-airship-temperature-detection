from sklearn.model_selection import train_test_split

from temperature_detection import *

# Welcome to the main script! Here we are going to train, evaluate and save the model in six simple steps.

# AugmentEnum.CompositeAugment(PreProcessConf.SOURCE_IMAGE_DIR, AugmentEnum.HeightShiftAugment, AugmentEnum.WidthShiftAugment,
#                              AugmentEnum.RotationAugment, height_shift_range=0.4, width_shift_range=0.4, rotation_range=180,
#                              ).augment(4)

# load images
# TODO - decide whether to load from 'augmented' or 'outputs' by the existence of 'augmented'
x, y, text_labels = LoaderEnum.DirectoryLoader().load(PreProcessConf.AUGMENT_DIR, is_shuffle=True)

# split to train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

coach = CrossValidationCoach(x_train, y_train, x_test, y_test, text_labels)

coach.train(ModelEnum.DoubleConv, batch_size=20, epochs=15)

print('king')
