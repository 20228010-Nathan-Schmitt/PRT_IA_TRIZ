model_distilroberta = None

def embeddings_distilroberta(sentences, batch_size=32, once=False):
    global model_distilroberta
    if model_distilroberta is None : 
        from sentence_transformers import SentenceTransformer
        model_distilroberta = SentenceTransformer('sentence-transformers/all-distilroberta-v1')

    sentences_emb_distilroberta = model_distilroberta.encode(sentences, show_progress_bar=True, batch_size=batch_size)
    if once: model_distilroberta = None
    return sentences_emb_distilroberta
