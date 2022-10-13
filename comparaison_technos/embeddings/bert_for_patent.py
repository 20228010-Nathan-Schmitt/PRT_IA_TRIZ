from compare_tools import transposeList
import numpy as np

model_bert_patent = None

def embeddings_bert_patent(sentences):
    global model_bert_patent
    if model_bert_patent is None : 
        from .bertForPatent.bertForPatent import bert_predictor
        model_bert_patent=bert_predictor

    sentence1, sentence2 = transposeList(sentences)
    response1,_,_ = model_bert_patent.predict(sentence1)
    response2,_,_ = model_bert_patent.predict(sentence2)

    sentence1_emb = response1["cls_token"]
    sentence2_emb = response2["cls_token"]
    return np.swapaxes([sentence1_emb, sentence2_emb], 0,1)
