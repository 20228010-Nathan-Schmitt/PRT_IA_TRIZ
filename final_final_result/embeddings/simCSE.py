model_simcse = None

def embeddings_simcse(sentences, batch_size=32, once=False):
    global model_simcse
    if model_simcse is None : 
        from sentence_transformers import SentenceTransformer
        model_simcse = SentenceTransformer('princeton-nlp/unsup-simcse-roberta-large')

    sentences_emb_simcse = model_simcse.encode(sentences, show_progress_bar=True, batch_size=batch_size)
    if once: model_simcse = None
    return sentences_emb_simcse
