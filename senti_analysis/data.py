# encoding: utf8

import pandas as pd
from senti_analysis import config


def load_data_set():
    """
    Load data set.
    :return: train_data_set, validation_data_set, test_data_set
    """
    train_data_set = pd.read_csv(config.TRAIN_SET_PATH)
    validation_data_set = pd.read_csv(config.VALIDATION_SET_PATH)
    test_data_set = pd.read_csv(config.TEST_SET_PATH)

    return train_data_set, validation_data_set, test_data_set
