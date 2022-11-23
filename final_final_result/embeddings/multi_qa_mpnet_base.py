model_multi_qa_mpnet_base = None

def embeddings_multi_qa_mpnet_base(sentences, batch_size=32, once=False):
    global model_multi_qa_mpnet_base
    if model_multi_qa_mpnet_base is None : 
        from sentence_transformers import SentenceTransformer
        model_multi_qa_mpnet_base = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')

    sentences_emb_multi_qa_mpnet_base = model_multi_qa_mpnet_base.encode(sentences, show_progress_bar=True, batch_size=batch_size)
    if once: model_multi_qa_mpnet_base = None
    return sentences_emb_multi_qa_mpnet_base
