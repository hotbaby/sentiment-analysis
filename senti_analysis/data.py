# encoding: utf8

import numpy as np
import pandas as pd
from collections import OrderedDict

from senti_analysis import config
from senti_analysis import constants
from senti_analysis.preprocess import (load_tokenizer, load_sentences,
                                       encode_sentence, label_transform)


def load_data_set():
    """
    Load data set.
    :return: train_data_set, validation_data_set, test_data_set
    """
    train_data_set = pd.read_csv(config.TRAIN_SET_PATH)
    validation_data_set = pd.read_csv(config.VALIDATION_SET_PATH)
    test_data_set = pd.read_csv(config.TEST_SET_PATH)

    return train_data_set, validation_data_set, test_data_set


def x_data():
    train_set = pd.read_csv(config.TRAIN_SET_PATH)
    val_set = pd.read_csv(config.VALIDATION_SET_PATH)

    tokenizer = load_tokenizer()
    train_sentences, val_sentences, test_sentences = load_sentences()
    x_train = encode_sentence(train_sentences, padding=True, max_length=config.MAX_SEQUENCE_LENGTH,
                              tokenizer=tokenizer)
    x_val = encode_sentence(val_sentences, padding=True, max_length=config.MAX_SEQUENCE_LENGTH,
                            tokenizer=tokenizer)
    return x_train, x_val


def transform_y_data(train_set, val_set, cols):
    y_train = OrderedDict()
    y_val = OrderedDict()

    for col in cols:
        y_train[col] = np.array(label_transform(train_set[col]))
        y_val[col] = np.array(label_transform(val_set[col]))

    return y_train, y_val


def y_data():
    """
    generate y label data.
    :return: train_label_data dict, validation_label_data dict
    """
    train_set = pd.read_csv(config.TRAIN_SET_PATH)
    val_set = pd.read_csv(config.VALIDATION_SET_PATH)

    y_train, y_val = transform_y_data(train_set, val_set, constants.COLS)

    return y_train, y_val
