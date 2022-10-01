from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# https://towardsdatascience.com/semantic-textual-similarity-83b3ca4a840e

def cos_sim(sentence1_emb, sentence2_emb):
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
    cos_sim = cosine_similarity(sentence1_emb, sentence2_emb)
    return np.diag(cos_sim)


model = SentenceTransformer('princeton-nlp/unsup-simcse-roberta-large')

sentence1 = [
    'this is such an amazing movie!',
    'this is such an amazing movie!',
    'this is such an amazing movie!',
    "When the stroller 1 moves over a lawn or uneven road surfaces, it is necessary for the stroller wheels to have a large diameter so as to ensure the comfort of the baby. However, if each of the front wheel assemblies 11 has two large-diameter front wheels 13, the total volume and weight of the stroller 1 will increase significantly so that it is difficult to push the stroller 1.",
    "When the stroller 1 moves over a lawn or uneven road surfaces, it is necessary for the stroller wheels to have a large diameter so as to ensure the comfort of the baby. However, if each of the front wheel assemblies 11 has two large-diameter front wheels 13, the total volume and weight of the stroller 1 will increase significantly so that it is difficult to push the stroller 1.",
    "When the stroller 1 moves over a lawn or uneven road surfaces, it is necessary for the stroller wheels to have a large diameter so as to ensure the comfort of the baby. However, if each of the front wheel assemblies 11 has two large-diameter front wheels 13, the total volume and weight of the stroller 1 will increase significantly so that it is difficult to push the stroller 1."
]
sentence2 = [
    'this is such an amazing movie!',
    'The film was nice to see.',
    'I don\'t like my tea!',
    "Big wheels are required but it increase mass and space.",
    "Increasing dimension also increase mass and volume",
    "Small wheels are required but it increase complexity "
]
# Generate Embeddings
sentence1_emb = model.encode(sentence1, show_progress_bar=True)
sentence2_emb = model.encode(sentence2, show_progress_bar=True)

# Cosine Similarity
result = cos_sim(sentence1_emb, sentence2_emb)

print(result)
