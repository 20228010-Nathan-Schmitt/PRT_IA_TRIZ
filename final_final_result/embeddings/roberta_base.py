model_roberta_base = None

def embeddings_roberta_base(sentences, batch_size=32, once=False):
    global model_roberta_base
    if model_roberta_base is None : 
        from sentence_transformers import SentenceTransformer
        model_roberta_base = SentenceTransformer('xlm-roberta-base')

    sentences_emb_roberta_base = model_roberta_base.encode(sentences, show_progress_bar=True, batch_size=batch_size)
    if once: model_roberta_base = None
    return sentences_emb_roberta_base
