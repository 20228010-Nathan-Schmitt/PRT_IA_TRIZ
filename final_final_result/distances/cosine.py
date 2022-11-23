import numpy as np

def cos_sim(sentences_emb):
    cos = np.einsum("ij,ij->i",sentences_emb[:,0,:],sentences_emb[:,1,:])
    cos_sim2= cos/(np.linalg.norm(sentences_emb[:,0,:], axis=1)*np.linalg.norm(sentences_emb[:,1,:], axis=1))
    return cos_sim2