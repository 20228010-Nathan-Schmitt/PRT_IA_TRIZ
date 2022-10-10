from compare_tools import transposeList

model_simcse = None

def embeddings_simcse(sentences):
    global model_simcse
    if model_simcse is None : 
        from sentence_transformers import SentenceTransformer
        model_simcse = SentenceTransformer('princeton-nlp/unsup-simcse-roberta-large')

    sentence1, sentence2 = transposeList(sentences)
    sentence1_emb_simcse = model_simcse.encode(sentence1)
    sentence2_emb_simcse = model_simcse.encode(sentence2)
    return transposeList([sentence1_emb_simcse, sentence2_emb_simcse])
