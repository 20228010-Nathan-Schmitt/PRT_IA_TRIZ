model_deberta = None

def embeddings_deberta(sentences, batch_size=32, once=False):
    global model_deberta
    if model_deberta is None : 
        from sentence_transformers import SentenceTransformer
        model_deberta = SentenceTransformer('microsoft/deberta-v3-base')

    sentences_emb_simcse = model_deberta.encode(sentences, show_progress_bar=True, batch_size=batch_size)
    if once: model_deberta = None
    return sentences_emb_simcse

