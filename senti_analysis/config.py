# encoding: utf8

import os


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'senti_analysis/datasets/')

# data set paths
TRAIN_SET_PATH = os.path.join(DATASET_DIR, 'train_set.csv')
VALIDATION_SET_PATH = os.path.join(DATASET_DIR, 'validation_set.csv')
TEST_SET_PATH = os.path.join(DATASET_DIR, 'test_set.csv')
STOPWORDS_PATH = os.path.join(DATASET_DIR, 'stopwords.txt')

# corpus sentences path
TRAIN_SENTENCE_PATH = os.path.join(DATASET_DIR, 'train_sentence.json')
VALIDATION_SENTENCE_PATH = os.path.join(DATASET_DIR, 'validation_sentence.json')
TEST_SENTENCE_PATH = os.path.join(DATASET_DIR, 'test_sentence.json')

# tokenizer
TOKENIZER_PATH = os.path.join(DATASET_DIR, 'tokenizer.json')

# word embedding
EMBEDDING_MATRIX_PATH = os.path.join(DATASET_DIR, 'embedding_matrix.npy')
W2V_MODEL_PATH = os.path.join(DATASET_DIR, 'w2v.model')
VOCAB_SIZE = 253154
EMBEDDING_DIM = 100

MAX_SEQUENCE_LENGTH = 1500

# model
MODEL_PATH = os.path.join(BASE_DIR, 'senti_analysis/models')


# log
LOG_DIR = os.path.join(BASE_DIR, 'logs')
