from embeddings.embeddings import embeddings
from compare_tools import makePairsToCompare, load_database, merge_array
import tensorflow as tf
import numpy as np
from numpy.lib import recfunctions as rfn
import os.path

def compute_embedding(i):
    database = load_database(i)
    ids_save = database["id"]
    sentences_save = database["sentence"]
    f_triz_save = database["F_TRIZ_PARAMS"]
    s_triz_save = database["S_TRIZ_PARAMS"]


    for embedding in embeddings:
        ids = ids_save.copy()
        sentences = sentences_save.copy()
        f_triz = f_triz_save.copy()
        s_triz = s_triz_save.copy()
        
        print(embedding)
        
        #on esaye d'ouvrir les embeddings deja calculés
        filename = "save/embedding_"+embedding+".npy"
        if os.path.isfile(filename):
            #si on les trouve, on les charge
            prev_emb_with_id = np.load(filename)
            
            #on enlève ceux qu'on a deja calculés
            prev_ids = set(prev_emb_with_id[:,0])
            for i in range(len(sentences)-1, -1, -1):
                if ids[i] in prev_ids: 
                    ids = np.delete(ids, i, 0)
                    sentences = np.delete(sentences, i, 0)
                    f_triz = np.delete(f_triz, i, 0)
                    s_triz = np.delete(s_triz, i, 0)
                    
            #si il en reste, on calcule les embeddings
            if ids.shape[0]:
                sentences_emb = embeddings[embedding](sentences)
                emb_with_id = merge_array(np.expand_dims(ids,axis=1), sentences_emb, "U32", "f4",sentences_emb.shape[1])
                emb_with_id = np.concatenate((prev_emb_with_id, emb_with_id),axis=0)
                emb_with_id = np.unique(emb_with_id, axis=0)
            else:
                emb_with_id = prev_emb_with_id
        else:
            #si aucune sauvegarde n'existe, on calcule tout
            sentences_emb = embeddings[embedding](sentences)
            emb_with_id = merge_array(np.expand_dims(ids,axis=1), sentences_emb, "U32", "f4",sentences_emb.shape[1])
        
        #on enregistre les embeddings avec les nouveaux calculés
        print(emb_with_id.shape)
        np.save(filename, emb_with_id)
        print("="*50, "\n")

for i in range(10):
    compute_embedding(i)