from compare_tools import transposeList

model_mpnet_base = None

def embeddings_mpnet_base(sentences):
    global model_mpnet_base
    if model_mpnet_base is None : 
        from sentence_transformers import SentenceTransformer
        model_mpnet_base = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    sentences_emb_simcse = model_mpnet_base.encode(sentences)
    return sentences_emb_simcse
