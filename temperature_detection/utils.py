import io

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class Utils:
    @staticmethod
    def image_summary(image, name, num_images, step):
        # change the dimension of an image and summary
        image_format = tf.transpose(image, [3, 1, 2, 0])  # (batch, 200-k*x, 200-k*x, None)
        image_format = image_format[0:num_images, :, :, 0]  # (1, 200-k*x, 200-k*x)
        image_format = tf.expand_dims(image_format, -1)  # (1, 200-k*x, 200-k*x, 1)
        tf.summary.image(name=name, data=image_format, max_outputs=num_images, step=step)

    @staticmethod
    def plot_training_graph(train_loss, train_accuracy, validation_loss, validation_accuracy, start_epoch, writer):
        # Plot the training and validation data
        tloss = np.array(train_loss).mean(axis=0)
        tacc = np.array(train_accuracy).mean(axis=0)
        vloss = np.array(validation_loss).mean(axis=0)
        vacc = np.array(validation_accuracy).mean(axis=0)
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
        with writer.as_default():
            tf.summary.image("1. Final Results/Training progress", Utils.plot_to_image(fig), step=0)
        return fig

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
