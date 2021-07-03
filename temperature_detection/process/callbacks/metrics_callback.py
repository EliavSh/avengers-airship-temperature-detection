import tensorflow as tf


class MetricsCallback(tf.keras.callbacks.Callback):

    def __init__(self, model, x_validation, y_validation, writer):
        super().__init__()

        self.model = model
        self.x_validation = x_validation
        self.y_validation = y_validation

        self.writer = writer

        self.validation_loss = []
        self.validation_accuracy = []

    def on_epoch_end(self, epoch, logs=None):
        validation_loss, validation_accuracy = self.model.evaluate(self.x_validation, self.y_validation, verbose=1)

        self.validation_loss.append(validation_loss)
        self.validation_accuracy.append(validation_accuracy)

        # summary scores for tensorboard
        with self.writer.as_default():
            tf.summary.scalar('train/loss', logs['loss'], step=epoch)
            tf.summary.scalar('train/accuracy', logs['accuracy'], step=epoch)
            tf.summary.scalar('validation/loss', validation_loss, step=epoch)
            tf.summary.scalar('validation/accuracy', validation_accuracy, step=epoch)

    def get_validation_loss(self):
        return self.validation_loss

    def get_validation_accuracy(self):
        return self.validation_accuracy
