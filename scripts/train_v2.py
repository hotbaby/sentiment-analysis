# encoding: utf8

import env

from senti_analysis.train import train
from senti_analysis.models.model_v2 import get_model


if __name__ == '__main__':
    model = get_model()
    train(model)
