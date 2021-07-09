from datetime import datetime

from sklearn.model_selection import train_test_split

from temperature_detection import *

model_enums = [ModelEnum.PreTrained.get(PreTrainedEnum.EfficientNetB0), ModelEnum.PreTrained.get(PreTrainedEnum.EfficientNetB1),
               ModelEnum.PreTrained.get(PreTrainedEnum.EfficientNetB2), ModelEnum.PreTrained.get(PreTrainedEnum.EfficientNetB3)]
# model = ModelEnum.PreTrained.get(PreTrainedEnum.EfficientNetB0)  # example for pretrained model

batch_size = 30

# AugmentEnum.CompositeAugment(PreProcessConf.SOURCE_IMAGE_DIR, AugmentEnum.HeightShiftAugment, AugmentEnum.WidthShiftAugment,
#                              AugmentEnum.RotationAugment, height_shift_range=0.3, width_shift_range=0.3, rotation_range=180,
#                              ).augment(1)

main_writer_dir = './temp/' + datetime.now().strftime("%Y%m%d-%H%M%S")

for model_enum in model_enums:
    # tab_prefix = './logs/' + model_enum.name + '_batch_' + str(batch_size)
    log_dir = main_writer_dir + '/' + model_enum.name + '_batch_' + str(batch_size)

    # load images
    # TODO - decide whether to load from 'augmented' or 'outputs' by the existence of 'augmented'
    x, y, text_labels = LoaderEnum.DirectoryLoader().load(PreProcessConf.AUGMENT_DIR, is_shuffle=True)

    # split to train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    coach = CrossValidationCoach(x_train, y_train, x_test, y_test, text_labels, log_dir)

    coach.train(model_enum, batch_size=batch_size, epochs=40)

print('king')
