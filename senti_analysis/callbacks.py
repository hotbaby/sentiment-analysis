# encoding: utf8

import logging
import tensorflow as tf

_logger = logging.getLogger()


class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
    def on_train_batch_end(self, batch, logs=None):
        # _logger.info('train batch {} end'.format(batch))

        # self._log_metrics(logs, prefix='', step=batch)
        super(CustomTensorBoard, self).on_train_batch_end(batch, logs)
