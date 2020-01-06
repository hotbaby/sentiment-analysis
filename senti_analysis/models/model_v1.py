# encoding: utf8

import os
import datetime
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.initializers import Constant

from senti_analysis import config
from senti_analysis import constants
from senti_analysis.preprocess import load_embedding_matrix


def fc_nn(share_hidden, name=None):
    """
    Fully connected layer.
    :param share_hidden: share hidden layer
    :param name: output layer name
    :return: outputs
    """
    hidden = tf.keras.layers.Dense(32, activation='relu')(share_hidden)
    hidden = tf.keras.layers.Dense(16, activation='relu')(hidden)
    outputs = tf.keras.layers.Dense(4, activation='softmax', name=name)(hidden)

    return outputs


def get_model(learning_rate=config.LEARNING_RATE, name='model_v1'):
    """
    create model.
    :return: model
    """
    num_class = 4
    embedding_matrix = load_embedding_matrix()

    inputs = tf.keras.layers.Input(shape=(config.MAX_SEQUENCE_LENGTH,), name='input')
    embedding = tf.keras.layers.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                          embeddings_initializer=Constant(embedding_matrix),
                                          input_length=config.MAX_SEQUENCE_LENGTH,
                                          trainable=False)(inputs)
    share_hidden = tf.keras.layers.GRU(64, activation='relu', return_sequences=True, reset_after=True)(embedding)
    share_hidden = tf.keras.layers.GRU(32, activation='relu', reset_after=True)(share_hidden)

    outputs = []
    for col in constants.COLS:
        # outputs.append(fc_nn(share_hidden, name=col))
        outputs.append(tf.keras.layers.Dense(num_class, activation='softmax', name=col)(share_hidden))

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
                           tfa.metrics.F1Score(num_class, average='micro')],)

    return model
