from compare_tools import transposeList

model_mpnet_base = None


def embeddings_mpnet_base(sentences):
    global model_mpnet_base
    if model_mpnet_base is None:
        from sentence_transformers import SentenceTransformer
        model_mpnet_base = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    sentence1, sentence2 = transposeList(sentences)
    sentence1_emb_simcse = model_mpnet_base.encode(sentence1)
    sentence2_emb_simcse = model_mpnet_base.encode(sentence2)
    return transposeList([sentence1_emb_simcse, sentence2_emb_simcse])
