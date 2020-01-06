# encoding: utf8

import os
import uuid
import logging
import datetime
import numpy as np
import tensorflow as tf

from senti_analysis import config
from senti_analysis.data import x_data, y_data
from senti_analysis.callbacks import CustomTensorBoard

_logger = logging.getLogger()


class StoppingCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        _logger.info('batch {} end'.format(batch))
        if batch >= 1:
            self.model.stop_training = True


def train(model, epochs=config.EPOCHS, learning_rate=config.LEARNING_RATE):
    _logger.info('load data')
    # service waiters attitude classification.
    x_train, x_val = x_data()
    y_train, y_val = y_data()

    _logger.info('compile model')

    log_dir = os.path.join(config.LOG_DIR, 'fit/{}/{}'.format(model.name,
                                                              datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    tensorboard_callback = CustomTensorBoard(log_dir=log_dir, update_freq='batch')
    checkpoint_path = os.path.join(config.MODEL_CHECKPOINT_PATH, '{}'.format(model.name))
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    _logger.info('fit model')
    history = model.fit(x_train, y_train,
                        batch_size=config.BATCH_SIZE,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_val, y_val),
                        callbacks=[tensorboard_callback, cp_callback],
                        steps_per_epoch=len(x_train) // config.BATCH_SIZE,
                        validation_steps=len(x_val) // config.BATCH_SIZE,
                        workers=config.WORKER_NUM)

    _logger.info('save model')
    model_name = model.name or str(uuid.uuid1())
    model_path = os.path.join(config.MODEL_PATH, '{}.h5'.format(model_name))
    # model.save(model_path)
    model.save_weights(model_path)

    _logger.info('done')
    return history
