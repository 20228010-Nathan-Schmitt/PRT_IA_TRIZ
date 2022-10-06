from sentence_transformers import SentenceTransformer
from compare_tools import transposeList

model_simcse = None

def embeddings_simcse(sentences):
    global model_simcse
    if model_simcse is None : model_simcse = SentenceTransformer('princeton-nlp/unsup-simcse-roberta-large')

    sentence1, sentence2 = transposeList(sentences)
    sentence1_emb_simcse = model_simcse.encode(sentence1, show_progress_bar=True)
    sentence2_emb_simcse = model_simcse.encode(sentence2, show_progress_bar=True)
    return transposeList([sentence1_emb_simcse, sentence2_emb_simcse])
