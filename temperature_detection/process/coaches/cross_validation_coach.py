import os

import numpy as np
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
    def __init__(self, x_train, y_train, x_test, y_test, text_labels, log_dir):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.text_labels = text_labels

        self.log_dir = log_dir
        self.tab_prefix = log_dir.split('/')[-1]

        self.k_fold = StratifiedKFold(n_splits=PreProcessConf.N_SPLITS, shuffle=True)
        self.num_folds = PreProcessConf.N_SPLITS

    def train(self, model, batch_size, epochs):
        # create_writer
        general_summary_writer = tf.summary.create_file_writer(os.path.join('/'.join(self.log_dir.split('/')[:-1]), 'Cross Validation Summary', self.tab_prefix))

        # create cm callback outside loop for storing all confusion matrices
        cm_callback = ConfusionMatrixCallback(x_test=self.x_test, y_test=self.y_test, text_labels=self.text_labels, tab_prefix=self.tab_prefix, writer=general_summary_writer)
        m_callback = MetricsCallback(cross_validation_writer=general_summary_writer, tab_prefix=self.tab_prefix, k_fold=self.num_folds)

        # loop over each fold and aggregate scores
        for i, (train, validation) in enumerate(self.k_fold.split(self.x_train, self.y_train)):
            current_writer_path = os.path.join(self.log_dir, 'fold ' + str(i))
            # set writer
            writer = tf.summary.create_file_writer(current_writer_path)

            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=["accuracy"])

            # create metrics-callback and update cm-callback with current model, data, etc.
            # tb_callback = tf.keras.callbacks.TensorBoard(PreProcessConf.log_dir, profile_batch=0, update_freq=1, write_images=True, write_graph=False)
            m_callback.build(model=model, x_validation=self.x_train[validation], y_validation=self.y_train[validation], writer=writer,
                             plot_results=(i == PreProcessConf.N_SPLITS - 1))
            cm_callback.build(model=model, plot_results=(i == PreProcessConf.N_SPLITS - 1))

            # fit the model
            model.fit(self.x_train[train], self.y_train[train], batch_size=batch_size, epochs=epochs, callbacks=[cm_callback, m_callback])

            # save the model
            model.save(os.path.join(current_writer_path, 'model'))

        # print summary of all scores as the mean of all folds
        self.final_summary(m_callback.scores, general_summary_writer)

    def final_summary(self, scores, general_summary_writer):
        mean_train_loss = np.divide(list(scores['train_loss'].values()), self.num_folds)
        mean_train_accuracy = np.divide(list(scores['train_accuracy'].values()), self.num_folds)
        mean_validation_loss = np.divide(list(scores['validation_loss'].values()), self.num_folds)
        mean_validation_accuracy = np.divide(list(scores['validation_accuracy'].values()), self.num_folds)

        # finally, summary the mean values as image to tensorboard + save the figure locally
        Utils.plot_training_graph(mean_train_loss, mean_train_accuracy, mean_validation_loss, mean_validation_accuracy, 0, general_summary_writer, self.tab_prefix).savefig(
            self.log_dir + '/training_graph.png')
