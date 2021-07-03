import itertools

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import tensorflow as tf

from temperature_detection.utils import Utils


class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_test, y_test, text_labels):
        super().__init__()

        self.x_test = x_test
        self.y_test = y_test
        self.class_names = np.unique(text_labels)

        self.model = None
        self.writer = None
        self.plot_results = None

        self.confusion_matrices = {}

    def build(self, model, writer, plot_results):
        self.model = model
        self.writer = writer
        # plot only in the last run - confusion matrix should be the average of all models
        self.plot_results = plot_results

    def on_epoch_end(self, epoch, logs=None):
        # Use the model to predict the values from the validation dataset.
        test_pred_raw = self.model.predict(self.x_test)
        test_pred = np.argmax(test_pred_raw, axis=1)

        # Calculate the confusion matrix.
        cm = sklearn.metrics.confusion_matrix(self.y_test, test_pred)

        # aggregate confusion matrix by epoch
        if epoch in list(self.confusion_matrices.keys()):
            self.confusion_matrices[epoch] += cm
        else:
            self.confusion_matrices[epoch] = cm

        if self.plot_results:
            # Log the confusion matrix as an image summary.
            figure = self.plot_confusion_matrix(self.confusion_matrices[epoch], class_names=np.unique(self.class_names))
            cm_image = Utils.plot_to_image(figure)

            # Log the confusion matrix as an image summary.
            with self.writer.as_default():
                tf.summary.image("1. Final Results/Confusion Matrix", cm_image, step=epoch)

    def plot_confusion_matrix(self, cm, class_names):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.

        Args:
          cm (array, shape = [n, n]): a confusion matrix of integer classes
          class_names (array, shape = [n]): String names of the integer classes
        """
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Compute the labels from the normalized confusion matrix.
        labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

        plt.tight_layout(rect=(0.03, 0.03, 0.95, 0.95))
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return figure
