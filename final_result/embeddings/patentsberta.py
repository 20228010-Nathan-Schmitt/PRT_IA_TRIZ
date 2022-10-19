from compare_tools import transposeList

model_patentsberta = None

def embeddings_patentsberta(sentences):
    global model_patentsberta
    if model_patentsberta is None : 
        from sentence_transformers import SentenceTransformer
        model_patentsberta = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')

    sentences_emb_patentsberta = model_patentsberta.encode(sentences)
    return sentences_emb_patentsberta
