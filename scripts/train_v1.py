# encoding: utf8

import env

from senti_analysis.train import train
from senti_analysis.preprocess import load_embedding_matrix
from senti_analysis.models.model_v1 import get_model


if __name__ == '__main__':
    embedding_matrix = load_embedding_matrix()
    model = get_model(embedding_matrix)
    train(model)
