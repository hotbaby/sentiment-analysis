# encoding: utf8

import logging

from senti_analysis import config
from senti_analysis.models.model_v1 import get_model
from senti_analysis.data import load_val_data_set

_logger = logging.getLogger()


if __name__ == '__main__':
    model = get_model()

    x_val, y_val = load_val_data_set()
    loss, acc = model.evaluate(x_val, y_val, batch_size=config.BATCH_SIZE)
    _logger.info('loss: {}'.format(loss))
    _logger.info('acc: {}'.format(acc))
