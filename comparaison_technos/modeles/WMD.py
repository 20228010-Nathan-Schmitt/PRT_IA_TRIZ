from word_mover_distance import model
from nltk.corpus import stopwords
from nltk import download
from compare_tools import transposeList
import gensim.downloader as api

download('stopwords')
stop_words = stopwords('english')


def preprocess(sentence):
    return [w for w in sentence.lower().split() if w not in stop_words]


def wmd_dist(sentence_emb):
    sentence1_emb, sentence2_emb = transposeList(sentences_emb)
    sentence1_emb = preprocess(sentence1_emb)
    sentence2_emb = preprocess(sentence2_emb)

    model = api.load('word2vec-google-news-300')
    return model.wmdistance(sentence1_emb, sentence2_emb)