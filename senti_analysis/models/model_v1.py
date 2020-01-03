# encoding: utf8

import os
import datetime
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.initializers import Constant

from senti_analysis import config
from senti_analysis import constants


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


def get_model(embedding_matrix, name='model_v1'):
    """
    create model.
    :return: model
    """
    num_class = 4

    inputs = tf.keras.layers.Input(shape=(config.MAX_SEQUENCE_LENGTH,), name='input')
    embedding = tf.keras.layers.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                          embeddings_initializer=Constant(embedding_matrix),
                                          input_length=config.MAX_SEQUENCE_LENGTH,
                                          trainable=False)(inputs)
    share_hidden = tf.keras.layers.GRU(64, activation='relu', return_sequences=True)(embedding)
    # share_hidden = tf.keras.layers.GRU(32, activation='relu')(share_hidden)

    outputs = []
    for col in constants.COLS:
        outputs.append(fc_nn(share_hidden, name=col))

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    return model
