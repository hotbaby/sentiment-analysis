# encoding: utf8

import os
import uuid
import datetime
import tensorflow as tf

from senti_analysis import config
from senti_analysis.data import x_data, y_data
from senti_analysis.preprocess import load_embedding_matrix


def train(model, epochs=config.EPOCHS, learning_rate=config.LEARNING_RATE):
    # service waiters attitude classification.
    x_train, x_val = x_data()
    y_train, y_val = y_data()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    log_dir = os.path.join(config.LOG_DIR, 'fit/{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(x_train, y_train, batch_size=config.BATCH_SIZE, epochs=epochs, verbose=1,
                        validation_data=(x_val, y_val),
                        callbacks=[tensorboard_callback],
                        workers=config.WORKER_NUM)

    model_name = model.name or str(uuid.uuid1())
    model_path = os.path.join(config.MODEL_PATH, model_name)
    model.save(model_path)

    return history
