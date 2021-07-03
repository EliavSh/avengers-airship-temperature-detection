import os

import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

from temperature_detection.pre_process.pre_process_conf import PreProcessConf
from temperature_detection.process.callbacks import *
from temperature_detection.utils import Utils

"""
This class should run over:
1. different train-validation sets
2. different batch sizes

"""


class CrossValidationCoach:
    # set different validation metrics per entire training by coach
    validation_histories = {"validation_loss": [], "validation_accuracy": []}

    def __init__(self, x_train, y_train, x_test, y_test, text_labels):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.text_labels = text_labels

        self.k_fold = StratifiedKFold(n_splits=PreProcessConf.N_SPLITS, shuffle=True)

        self.train_histories = {"train_loss": [], "train_accuracy": []}
        self.validation_histories = {"validation_loss": [], "validation_accuracy": []}

        self.writer = tf.summary.create_file_writer(os.path.join(PreProcessConf.log_dir, 'train', 'Cross Validation Summary'))

    def train(self, model_enum, *model_args, batch_size, epochs):
        # create cm callback outside loop for storing all confusion matrices
        cm_callback = ConfusionMatrixCallback(x_test=self.x_test, y_test=self.y_test, text_labels=self.text_labels)

        # loop over each fold and aggregate scores
        for i, (train, validation) in enumerate(self.k_fold.split(self.x_train, self.y_train)):
            current_writer_path = os.path.join(PreProcessConf.log_dir, 'fold ' + str(i))
            # set writer
            writer = tf.summary.create_file_writer(current_writer_path)

            # build model
            model = model_enum.get(*model_args)
            # model = ModelEnum.PreTrained.get(PreTrainedEnum.EfficientNetB0)  # example for pretrained model

            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=["accuracy"])

            # create metrics-callback and update cm-callback with current model, data, etc.
            # tb_callback = tf.keras.callbacks.TensorBoard(PreProcessConf.log_dir, profile_batch=0, update_freq=1, write_images=True, write_graph=False)
            m_callback = MetricsCallback(model=model, x_validation=self.x_train[validation], y_validation=self.y_train[validation], writer=writer)
            cm_callback.build(model=model, writer=writer, plot_results=(i == PreProcessConf.N_SPLITS - 1))

            # fit the model
            history = model.fit(self.x_train[train], self.y_train[train], batch_size=batch_size, epochs=epochs, callbacks=[cm_callback, m_callback])

            # aggregate data for later
            self.train_histories['train_loss'].append(history.history['loss'])
            self.train_histories['train_accuracy'].append(history.history['accuracy'])
            self.validation_histories['validation_loss'].append(m_callback.get_validation_loss())
            self.validation_histories['validation_accuracy'].append(m_callback.get_validation_accuracy())

            # save the model
            model.save(os.path.join(current_writer_path, 'model'))

        # finally, summary the mean values as image to tensorboard + save the figure locally
        Utils.plot_training_graph(self.train_histories['train_loss'], self.train_histories['train_accuracy'],
                                  self.validation_histories['validation_loss'], self.validation_histories['validation_accuracy'], 0,
                                  self.writer).savefig(PreProcessConf.log_dir + '/training_graph.png')
