import numpy as np

def euc_dist(sentences_emb):
    """ Euclidian distance between two columns of sentence embedding"""

    sentence1_emb, sentence2_emb = np.swapaxes(sentences_emb,0,1)
    difference = sentence1_emb - sentence2_emb
    return np.linalg.norm(difference, axis=1)

