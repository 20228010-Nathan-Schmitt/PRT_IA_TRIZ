from compare_tools import transposeList

model_simcse = None

def embeddings_simcse(sentences):
    global model_simcse
    if model_simcse is None : 
        from sentence_transformers import SentenceTransformer
        model_simcse = SentenceTransformer('princeton-nlp/unsup-simcse-roberta-large')

    sentences_emb_simcse = model_simcse.encode(sentences)
    return sentences_emb_simcse
