# encoding: utf8

import os
import uuid
import logging
import datetime
import tensorflow as tf

from senti_analysis import config
from senti_analysis.data import x_data, y_data
from senti_analysis.preprocess import load_embedding_matrix

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
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    log_dir = os.path.join(config.LOG_DIR, 'fit/{}/{}'.format(model.name,
                                                              datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='batch')
    checkpoint_path = os.path.join(config.MODEL_CHECKPOINT_PATH, '{}'.format(model.name))
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    _logger.info('fit model')
    history = model.fit({'input': x_train}, y_train,
                        batch_size=config.BATCH_SIZE,
                        epochs=epochs,
                        verbose=1,
                        validation_data=({'input': x_val}, y_val),
                        callbacks=[tensorboard_callback, cp_callback],
                        workers=config.WORKER_NUM)

    _logger.info('save model')
    model_name = model.name or str(uuid.uuid1())
    model_path = os.path.join(config.MODEL_PATH, model_name)
    # model.save(model_path)
    model.save_weights(model_path)

    _logger.info('done')
    return history
