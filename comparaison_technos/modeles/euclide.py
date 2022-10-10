import numpy as np

def euc_dist(sentences_emb):
    """ Euclidian distance between two columns of sentence embedding"""

    sentence1_emb, sentence2_emb = np.array(sentences_emb).T
    difference = sentence1_emb - sentence2_emb
    return np.linalg.norm(difference)

