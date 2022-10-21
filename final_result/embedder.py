from embeddings.embeddings import embeddings
from compare_tools import makePairsToCompare, load_database, merge_array
import tensorflow as tf
import numpy as np
from numpy.lib import recfunctions as rfn
import os.path

def compute_embedding(force_start=None):
    batch_size = 200
    filename_computed = "save/computed.txt"

    if force_start is not None:
        start = force_start
    else:
        if os.path.isfile(filename_computed):
            f = open(filename_computed, "r")
            start = int(f.read())
            f.close()
        else:
            start=0

    print("\n===== STEP {} started =====".format(start))

    database = load_database(start, batch_size)
    ids = database["id"]
    sentences = database["sentence"]
    f_triz = database["F_TRIZ_PARAMS"]
    s_triz = database["S_TRIZ_PARAMS"]

    filename_ids = "save/ids.npy"
    filename_ids_compr = "save/ids_compr.npz"
    if os.path.isfile(filename_ids):
        prev_ids = np.load(filename_ids)
        for i in range(len(sentences)-1, -1, -1):
            if ids[i] in prev_ids: 
                ids = np.delete(ids, i, 0)
                sentences = np.delete(sentences, i, 0)
                f_triz = np.delete(f_triz, i, 0)
                s_triz = np.delete(s_triz, i, 0)
        ids = np.concatenate((prev_ids, ids),axis=0)
    else:
        prev_ids = np.empty((0,))
    

    sentences_emb_dict={}
    for embedding in embeddings:
        print(embedding)
        
        #on esaye d'ouvrir les embeddings deja calculés
        filename = "save/embedding_"+embedding+".npy"
        if os.path.isfile(filename):
            #si on les trouve, on les charge
            prev_sentences_emb = np.load(filename)
                    
            #si il en reste, on calcule les embeddings
            if sentences.shape[0]:
                sentences_emb = embeddings[embedding](sentences)
                sentences_emb = np.concatenate((prev_sentences_emb, sentences_emb),axis=0)
            else:
                sentences_emb = prev_sentences_emb
        else:
            #si aucune sauvegarde n'existe, on calcule tout
            sentences_emb = embeddings[embedding](sentences)
        
        print("\nNumber of embeddings computed for {} : {}".format(embedding,sentences_emb.shape[0]))
        sentences_emb_dict[embedding] = sentences_emb

    print("===== DON'T STOP until save =====")
    #on enregistre les embeddings avec les nouveaux calculés
    for embedding in embeddings:
        filename = "save/embedding_"+embedding+".npy"
        np.save(filename, sentences_emb_dict[embedding])
    np.save(filename_ids, ids)
    np.savez_compressed(filename_ids_compr, ids=ids)

    f = open(filename_computed, "w")
    f.write(str(prev_ids.shape[0]+batch_size))
    f.close()

    print("===== STEP {} saved =====".format(start))


for i in range(1000):
    compute_embedding()