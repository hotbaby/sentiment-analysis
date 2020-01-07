# encoding: utf8

import env

import os
import sys
import importlib
from senti_analysis.train import train


if __name__ == '__main__':
    assert len(sys.argv) == 2, 'please input model name!'
    model_path = '.'.join(['senti_analysis', 'models', sys.argv[1]])
    module = importlib.import_module(model_path)

    model = module.get_model()
    train(model)
