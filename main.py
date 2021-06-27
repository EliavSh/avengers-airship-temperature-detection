from sklearn.model_selection import train_test_split

from temperature_detection import *

# load images ======================================================================================================================= 1
x, y, text_labels = LoaderEnum.ImageLoader().load('./outputs', is_shuffle=True)

# split to train and test =========================================================================================================== 2
# expand dim for our models (we are going to deal with conv layers that expects 4-dim input)
x = np.expand_dims(x, -1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# build model ======================================================================================================================= 3
model = ModelEnum.SimpleModel.get()

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=["accuracy"])

# create callbacks ================================================================================================================== 4
tb_callback, cm_callback = Utils.create_callbacks(model, x_test, y_test, text_labels, PreProcessConf.log_dir)

# fit the model ====================================================================================
history = model.fit(x_train, y_train, batch_size=10, epochs=15, validation_split=0.2, callbacks=[tb_callback, cm_callback])

# save training graph and model ===================================================================================================== 5
Utils.plot_training_graph(history, 0).savefig(PreProcessConf.log_dir + '/training_graph.png')

model.save(PreProcessConf.log_dir + '/model/')

print('king')
