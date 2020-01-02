# encoding: utf8

import os
import json
import gensim
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

from senti_analysis import config
from senti_analysis.utils import dump, load, cut


def text2sentence(text_list):
    """
    text to sentence, that can be use by gensim.
    :param text_list:
    :return: sentence list
    """
    sentences = []
    for text in text_list:
        sentences.append(' '.join(cut(text)))

    return sentences


def load_sentences():
    """
    Load sentences.
    :return: train_sentences, val_sentences, test_sentences.
    """

    def _load(corpus_file_path, dump_file_path):
        if os.path.exists(dump_file_path):
            # already generated train sentences.
            with open(dump_file_path) as f:
                return json.load(f)

        data_set = pd.read_csv(corpus_file_path)
        sentences = text2sentence(data_set['content'])

        # serialize sentences.
        with open(dump_file_path, 'w+') as f:
            f.write(json.dumps(sentences, ensure_ascii=False))

        return sentences

    train_sentences = _load(config.TRAIN_SET_PATH, config.TRAIN_SENTENCE_PATH)
    val_sentences = _load(config.VALIDATION_SET_PATH, config.VALIDATION_SENTENCE_PATH)
    test_sentences = _load(config.TEST_SET_PATH, config.TEST_SENTENCE_PATH)

    return train_sentences, val_sentences, test_sentences


def load_tokenizer():
    if os.path.exists(config.TOKENIZER_PATH):
        with open(config.TOKENIZER_PATH) as f:
            return tokenizer_from_json(f.read())

    train_sentences, val_sentences, test_sentences = load_sentences()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_sentences)
    tokenizer.fit_on_texts(val_sentences)
    tokenizer.fit_on_texts(test_sentences)

    # persistent to file.
    with open(config.TOKENIZER_PATH, 'w') as f:
        f.write(tokenizer.to_json(ensure_ascii=False))

    return tokenizer


def load_vocab(tokenizer=None):
    """
    load vocabulary.
    :return: vocabulary.
    """
    if tokenizer is None:
        tokenizer = load_tokenizer()

    return set(tokenizer.word_index.keys())


def encode_sentence(sentences, padding=False, max_length=None, tokenizer=None):
    """
    Encode sentences.
    :param sentences: sentence list.
    :param padding: whether padding sentences or not.
    :param max_length: padding sentence max length.
    :param tokenizer: tokenizer
    :return: encoded sentences.
    """
    assert isinstance(sentences, list)

    if tokenizer is None:
        tokenizer = load_tokenizer()

    encoded_sentences = tokenizer.texts_to_sequences(sentences)
    if padding:
        assert max_length is not None, 'if padding is True, must provide max_length param.'
        encoded_sentences = pad_sequences(encoded_sentences, maxlen=max_length)

    return encoded_sentences


def load_embedding_matrix(word_index=None, wv=None):
    """
    Initialize embedding matrix.
    :param word_index: word index dict.
    :param wv: word2vec model.
    :return: embedding matrix.
    """
    if os.path.exists(config.EMBEDDING_MATRIX_PATH):
        return np.load(config.EMBEDDING_MATRIX_PATH)

    vocab_size = len(word_index)
    embedding_dim = wv.vector_size

    embedding_matrix = np.zeros((vocab_size+1, embedding_dim))

    for word, index in word_index.items():
        if word not in wv:
            continue
        embedding_matrix[index] = wv[word]

    np.save(config.EMBEDDING_MATRIX_PATH, embedding_matrix)

    return embedding_matrix


def label_transform(y_data):
    """
    :Label transform.
    :param y_data: y_data
    :return: transformed y_data
    """
    y_data[y_data[y_data == 1].index] = 3
    y_data[y_data[y_data == 0].index] = 2
    y_data[y_data[y_data == -1].index] = 1
    y_data[y_data[y_data == -2].index] = 0

    return y_data


def load_w2v_model():
    return gensim.models.Word2Vec.load(config.W2V_MODEL_PATH)
