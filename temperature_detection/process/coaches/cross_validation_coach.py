from sklearn.model_selection import StratifiedKFold
from tensorflow import keras

from temperature_detection.pre_process.pre_process_conf import PreProcessConf
from temperature_detection.utils import Utils

"""
This class should run over:
1. different train-validation sets
2. different batch sizes

"""


class CrossValidationCoach:
    def __init__(self, x_train, y_train, x_test, y_test, text_labels):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.text_labels = text_labels

        self.k_fold = StratifiedKFold(n_splits=PreProcessConf.N_SPLITS, shuffle=True)

        self.model = None

        self.train_histories = {"train_loss": [], "train_accuracy": []}

    def train(self, model_enum, *model_args, batch_size, epochs):
        for train, validation in self.k_fold.split(self.x_train, self.y_train):
            # build model
            self.model = model_enum.get(*model_args)
            # model = ModelEnum.PreTrained.get(PreTrainedEnum.EfficientNetB0)  # example for pretrained model

            self.model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=["accuracy"])

            # create callbacks ================================================================================================================== 4
            tb_callback, cm_callback, acc_callback = Utils.create_callbacks(self.model, self.x_train[validation], self.y_train[validation], self.x_test,
                                                                            self.y_test, self.text_labels, PreProcessConf.log_dir)

            # fit the model ===================================================================================================================== 5
            history = self.model.fit(self.x_train[train], self.y_train[train], batch_size=batch_size, epochs=epochs, callbacks=[tb_callback, cm_callback, acc_callback])

            self.train_histories['train_loss'].append(history.history['loss'])
            self.train_histories['train_accuracy'].append(history.history['accuracy'])

        # finally, summary the mean values as image to tensorboard + save the figure locally
        Utils.plot_training_graph(self.train_histories['train_loss'], self.train_histories['train_accuracy'], 0).savefig(PreProcessConf.log_dir + '/training_graph.png')
