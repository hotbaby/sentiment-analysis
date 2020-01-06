# encoding: utf8

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from senti_analysis import config
from senti_analysis.models.model_v1 import get_model
from senti_analysis.preprocess import encode_sentence, text2sentence


def predict(model, text_list:list):
    inverse_label_map = {
        0: -2,
        1: -1,
        2: 0,
        3: 1
    }

    # model = get_model()
    # model.load_weights(os.path.join(config.MODEL_PATH, 'model_v1.h5'))

    sentences = text2sentence(text_list)
    encoded_sentences = encode_sentence(sentences, padding=True, max_length=config.MAX_SEQUENCE_LENGTH)

    predictions = model.predict(encoded_sentences)

    pred_results = []
    for pred in predictions:
        pred_result = []
        for label in pred:
            transform_label = inverse_label_map[np.argmax(label)]
            pred_result.append(transform_label)
        pred_results.append(pred_result)

    return pd.DataFrame(np.array(pred_results).T, columns=model.output_names)
