from sklearn.model_selection import train_test_split

from temperature_detection import *

# TODO - fixed init files in 'models' - their imports logic should follow enum instead of specific classes
# Welcome to the main script! Here we are going to train, evaluate and save the model in six simple steps.

# AugmentEnum.CompositeAugment(PreProcessConf.SOURCE_IMAGE_DIR, AugmentEnum.HeightShiftAugment, AugmentEnum.WidthShiftAugment,
#                              AugmentEnum.RotationAugment, height_shift_range=0.4, width_shift_range=0.4, rotation_range=180,
#                              ).augment(4)

# load images ======================================================================================================================= 1
# TODO - decide whether to load from 'augmented' or 'outputs' by the existence of 'augmented'
x, y, text_labels = LoaderEnum.DirectoryLoader().load(PreProcessConf.SOURCE_IMAGE_DIR, is_shuffle=True)

# split to train and test =========================================================================================================== 2
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

coach = CrossValidationCoach(x_train, y_train, x_test, y_test, text_labels)

coach.train(ModelEnum.DoubleConv, batch_size=20, epochs=20)

# save training graph and model ===================================================================================================== 6
# Utils.plot_training_graph(histories, 0).savefig(PreProcessConf.log_dir + '/training_graph.png')

# model.save(PreProcessConf.log_dir + '/model/')

print('king')
