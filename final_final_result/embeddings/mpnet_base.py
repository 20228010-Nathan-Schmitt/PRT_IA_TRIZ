model_mpnet_base = None

def embeddings_mpnet_base(sentences, batch_size=32, once=False):
    global model_mpnet_base
    if model_mpnet_base is None : 
        from sentence_transformers import SentenceTransformer
        model_mpnet_base = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    sentences_emb_simcse = model_mpnet_base.encode(sentences, show_progress_bar=True, batch_size=batch_size)
    if once: model_mpnet_base = None
    return sentences_emb_simcse

