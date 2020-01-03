# encoding: utf8

import os
import logging
import datetime
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.initializers import Constant

from senti_analysis import config
from senti_analysis.preprocess import (load_sentences, load_tokenizer,
                                       encode_sentence, label_transform,
                                       load_embedding_matrix)

_logger = logging.getLogger()


def get_model(embedding_matrix, name='baseline_model'):
    """
    create model.
    :return: model
    """
    num_class = 4

    inputs = tf.keras.layers.Input(shape=(config.MAX_SEQUENCE_LENGTH,))
    embedding = tf.keras.layers.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                          embeddings_initializer=Constant(embedding_matrix),
                                          input_length=config.MAX_SEQUENCE_LENGTH,
                                          trainable=False)(inputs)
    hidden = tf.keras.layers.GRU(64, activation='relu', return_sequences=True)(embedding)
    hidden = tf.keras.layers.GRU(32, activation='relu')(hidden)
    outputs = tf.keras.layers.Dense(num_class, activation='softmax')(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    return model


def train_data():
    train_set = pd.read_csv(config.TRAIN_SET_PATH,)
    val_set = pd.read_csv(config.VALIDATION_SET_PATH)

    tokenizer = load_tokenizer()
    train_sentences, val_sentences, test_sentences = load_sentences()
    x_train = encode_sentence(train_sentences, padding=True, max_length=config.MAX_SEQUENCE_LENGTH,
                              tokenizer=tokenizer)
    x_val = encode_sentence(val_sentences, padding=True, max_length=config.MAX_SEQUENCE_LENGTH,
                            tokenizer=tokenizer)

    y_train = train_set['service_waiters_attitude']
    y_val = val_set['service_waiters_attitude']

    y_train, y_val = np.array(label_transform(y_train)), np.array(label_transform(y_val))

    return x_train, y_train, x_val, y_val


def train(epochs=config.EPOCHS, learning_rate=config.LEARNING_RATE):
    # service waiters attitude classification.
    _logger.info('load data.')
    x_train, y_train, x_val, y_val = train_data()

    train_data_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data_set = train_data_set.batch(config.BATCH_SIZE)

    val_data_set = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_data_set = val_data_set.batch(config.BATCH_SIZE)

    _logger.info('load embedding matrix')
    embedding_matrix = load_embedding_matrix()

    _logger.info('get and compile model')
    model = get_model(embedding_matrix)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    log_dir = os.path.join(config.LOG_DIR, 'fit/{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=2, baseline=0.9)

    _logger.info('training model')

    # history = model.fit(x_train, y_train, batch_size=config.BATCH_SIZE, epochs=epochs, verbose=1,
    #                     validation_data=(x_val, y_val),
    #                     callbacks=[tensorboard_callback, early_stopping_callback],
    #                     workers=config.WORKER_NUM)

    history = model.fit(train_data_set, epochs=epochs, verbose=1,
                        validation_data=val_data_set,
                        callbacks=[tensorboard_callback, early_stopping_callback],
                        workers=config.WORKER_NUM)

    model_path = os.path.join(config.MODEL_PATH, 'baseline.model')
    model.save(model_path)

    return history
