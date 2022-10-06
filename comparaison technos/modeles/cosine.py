from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from compare_tools import transposeList

def cos_sim(sentences_emb):
    """
    Cosine similarity between two columns of sentence embeddings

    Args:
      sentence1_emb: sentence1 embedding column
      sentence2_emb: sentence2 embedding column

    Returns:
      The row-wise cosine similarity between the two columns.
      For instance is sentence1_emb=[a,b,c] and sentence2_emb=[x,y,z]
      Then the result is [cosine_similarity(a,x), cosine_similarity(b,y), cosine_similarity(c,z)]
    """

    sentence1_emb, sentence2_emb = transposeList(sentences_emb)
    cos_sim = cosine_similarity(sentence1_emb, sentence2_emb)
    return np.diag(cos_sim)
