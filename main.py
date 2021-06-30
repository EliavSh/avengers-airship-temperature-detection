from sklearn.model_selection import train_test_split

from temperature_detection import *

# Welcome to the main script! Here we are going to train, evaluate and save the model in six simple steps.

# load images ======================================================================================================================= 1
# TODO - decide whether to load from 'augmented' or 'outputs' by the existence of 'augmented'
x, y, text_labels = LoaderEnum.DirectoryLoader().load('./augmented', is_shuffle=True)

# split to train and test =========================================================================================================== 2
x = np.repeat(x[..., np.newaxis], 3, -1)  # expand dims before entering the model - expects 3 channels.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# build model ======================================================================================================================= 3
model = ModelEnum.DoubleConv.get()
# model = ModelEnum.PreTrained.get(PreTrainedEnum.EfficientNetB0)  # example for pretrained model

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=["accuracy"])

# create callbacks ================================================================================================================== 4
tb_callback, cm_callback = Utils.create_callbacks(model, x_test, y_test, text_labels, PreProcessConf.log_dir)

# fit the model ===================================================================================================================== 5
history = model.fit(x_train, y_train, batch_size=15, epochs=40, validation_split=0.2, callbacks=[tb_callback, cm_callback])

# save training graph and model ===================================================================================================== 6
Utils.plot_training_graph(history, 0).savefig(PreProcessConf.log_dir + '/training_graph.png')

model.save(PreProcessConf.log_dir + '/model/')

print('king')
