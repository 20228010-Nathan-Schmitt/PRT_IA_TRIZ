#from word_mover_distance import model
from nltk.corpus import stopwords
from nltk import download
from compare_tools import transposeList
import gensim.downloader as api
import numpy as np

download('stopwords')
stop_words = stopwords.words('english')


def preprocess(sentences):
    return [[w for w in sentence.lower().split() if w not in stop_words] for sentence in sentences]
def preprocess1(sentence):
    return [w for w in sentence.lower().split() if w not in stop_words]

def wmd_dist(sentences):
    sentences1, sentences2 = transposeList(sentences)
        
    #sentences1 = preprocess(sentences1)
    #sentences2 = preprocess(sentences2)

    model = api.load('word2vec-google-news-300')
    return np.fromiter((model.wmdistance(preprocess1(sentences1[i]), preprocess1(sentences2[i])) for i in range(len(sentences1))),float)