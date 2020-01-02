# encoding: utf8

import env
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

from senti_analysis.preprocess import load_embedding_matrix, load_tokenizer


tencent_pretrained_word_embedding = '/Users/hotbaby/Datasets/Tencent_AILab_ChineseEmbedding.txt'
corpus_word_embdding = '/Users/hotbaby/code/github/sentiment-analysis/notebooks/w2v.model'


if __name__ == '__main__':
    wv_model = KeyedVectors.load_word2vec_format(tencent_pretrained_word_embedding, binary=False)
    # wv_model = Word2Vec.load(corpus_word_embdding)
    tokenizer = load_tokenizer()
    load_embedding_matrix(tokenizer.word_index, wv_model)
