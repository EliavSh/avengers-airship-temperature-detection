import tensorflow as tf


class MetricsCallback(tf.keras.callbacks.Callback):

    def __init__(self, cross_validation_writer, tab_prefix, k_fold):
        super().__init__()

        self.model = None
        self.x_validation = None
        self.y_validation = None
        self.writer = None

        self.cross_validation_writer = cross_validation_writer
        self.k_fold = k_fold
        self.tab_prefix = tab_prefix

        self.scores = {'train_loss': {}, 'train_accuracy': {}, 'validation_loss': {}, 'validation_accuracy': {}}

        self.plot_results = False

    def build(self, model, x_validation, y_validation, writer, plot_results):
        self.model = model
        self.x_validation = x_validation
        self.y_validation = y_validation
        self.writer = writer
        self.plot_results = plot_results

    def on_epoch_end(self, epoch, logs=None):
        validation_loss, validation_accuracy = self.model.evaluate(self.x_validation, self.y_validation, verbose=1)

        train_loss = logs['loss']
        train_accuracy = logs['accuracy']

        # summary scores for tensorboard
        with self.writer.as_default():
            tf.summary.scalar(self.tab_prefix + '/train_loss', train_loss, step=epoch)
            tf.summary.scalar(self.tab_prefix + '/train_accuracy', train_accuracy, step=epoch)
            tf.summary.scalar(self.tab_prefix + '/validation_loss', validation_loss, step=epoch)
            tf.summary.scalar(self.tab_prefix + '/validation_accuracy', validation_accuracy, step=epoch)

        # aggregate confusion matrix by epoch
        for metric_name, value in zip(list(self.scores.keys()), [train_loss, train_accuracy, validation_loss, validation_accuracy]):
            if epoch in list(self.scores[metric_name].keys()):
                self.scores[metric_name][epoch] += value
            else:
                self.scores[metric_name][epoch] = value

        if self.plot_results:
            with self.cross_validation_writer.as_default():
                for metric_name in list(self.scores.keys()):
                    tf.summary.scalar('mean/' + metric_name, self.scores[metric_name][epoch] / self.k_fold, step=epoch)
