import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sklearn.metrics
import io
import itertools
from tensorflow import keras
from datetime import datetime


class Utils:
    test_images = None
    test_labels = None
    class_names = None
    model = None

    @staticmethod
    def plot_training_graph(train_data, start_epoch):
        # Plot the training and validation data
        tacc = train_data.history['accuracy']
        tloss = train_data.history['loss']
        vacc = train_data.history['val_accuracy']
        vloss = train_data.history['val_loss']
        Epoch_count = len(tacc) + start_epoch
        Epochs = []
        for i in range(start_epoch, Epoch_count):
            Epochs.append(i + 1)
        index_loss = np.argmin(vloss)  # this is the epoch with the lowest validation loss
        val_lowest = vloss[index_loss]
        index_acc = np.argmax(vacc)
        acc_highest = vacc[index_acc]
        plt.style.use('fivethirtyeight')
        sc_label = 'best epoch= ' + str(index_loss + 1 + start_epoch)
        vc_label = 'best epoch= ' + str(index_acc + 1 + start_epoch)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))
        axes[0].plot(Epochs, tloss, 'r', label='Training loss')
        axes[0].plot(Epochs, vloss, 'g', label='Validation loss')
        axes[0].scatter(index_loss + 1 + start_epoch, val_lowest, s=150, c='blue', label=sc_label)
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[1].plot(Epochs, tacc, 'r', label='Training Accuracy')
        axes[1].plot(Epochs, vacc, 'g', label='Validation Accuracy')
        axes[1].scatter(index_acc + 1 + start_epoch, acc_highest, s=150, c='blue', label=vc_label)
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        plt.tight_layout()
        # plt.style.use('fivethirtyeight')
        plt.show()
        return fig

    @staticmethod
    def log_confusion_matrix(epoch, logs):
        # Use the model to predict the values from the validation dataset.
        test_pred_raw = Utils.model.predict(Utils.test_images)
        test_pred = np.argmax(test_pred_raw, axis=1)

        # Calculate the confusion matrix.
        cm = sklearn.metrics.confusion_matrix(Utils.test_labels, test_pred)
        # Log the confusion matrix as an image summary.
        figure = Utils.plot_confusion_matrix(cm, class_names=np.unique(Utils.class_names))
        cm_image = Utils.plot_to_image(figure)

        # Log the confusion matrix as an image summary.
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)

    @staticmethod
    def plot_confusion_matrix(cm, class_names):
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

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return figure

    @staticmethod
    def plot_to_image(figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    @staticmethod
    def create_callbacks(model, x_test, y_test, text_labels, log_dir):
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir, update_freq=1)

        Utils.test_images = x_test
        Utils.test_labels = y_test
        Utils.class_names = np.unique(text_labels)
        Utils.model = model

        # Define the per-epoch callback.
        cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=Utils.log_confusion_matrix)

        return tb_callback, cm_callback
