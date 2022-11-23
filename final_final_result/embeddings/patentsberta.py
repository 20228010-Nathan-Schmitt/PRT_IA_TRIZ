model_patentsberta = None

def embeddings_patentsberta(sentences, batch_size=32, once=False):
    global model_patentsberta
    if model_patentsberta is None : 
        from sentence_transformers import SentenceTransformer
        model_patentsberta = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')

    sentences_emb_patentsberta = model_patentsberta.encode(sentences, show_progress_bar=True, batch_size=batch_size)
    if once: model_patentsberta = None
    return sentences_emb_patentsberta
