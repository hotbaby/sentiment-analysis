# encoding: utf8

# encoding: utf8

import os
import uuid
import logging
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

from senti_analysis import config
from senti_analysis import constants
from senti_analysis.data import x_data, y_data
from senti_analysis.preprocess import load_sentences, load_tokenizer, encode_sentence, label_transform

_logger = logging.getLogger()


def validate_model(model, epochs=1, samples=200):
    _logger.info('load data')
    train_set = pd.read_csv(config.TRAIN_SET_PATH)

    tokenizer = load_tokenizer()
    train_sentences, val_sentences, test_sentences = load_sentences()
    x_train = encode_sentence(train_sentences[:samples], padding=True, max_length=config.MAX_SEQUENCE_LENGTH,
                              tokenizer=tokenizer)

    y_train = {}

    for col in constants.COLS:
        y_train[col] = np.array(label_transform(train_set[col]))[:samples]

    _logger.info('test fit model')
    history = model.fit(x_train, y_train,
                        batch_size=config.BATCH_SIZE,
                        epochs=epochs,
                        verbose=1,
                        workers=config.WORKER_NUM)

    _logger.info('test successfully')
    return history
