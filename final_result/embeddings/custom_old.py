model_custom = None

def embeddings_custom(sentences, batch_size=32, once=False):
    global model_custom
    if model_custom is None : 
        from sentence_transformers import SentenceTransformer
        model_custom = SentenceTransformer('./my_model_sbert')

    sentences_emb_custom = model_custom.encode(sentences, show_progress_bar=True, batch_size=batch_size)
    if once: model_custom = None
    return sentences_emb_custom
