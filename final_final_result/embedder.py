import sys

from embeddings.embeddings import embeddings
from compare_tools import load_local_database

import numpy as np


#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #Work on CPU

def compute_embedding_queue(embedding_to_compute):
    print("===== START =====")

    database = load_local_database("databases/response_10000.json")
    ids = database["id"]
    sentences = database["sentence"]

    for embedding in embedding_to_compute:
        print(embedding)
        
        #on essaye d'ouvrir les embeddings deja calculés
        filename = "save/embedding_"+embedding+".npy"
        sentences_emb = embeddings[embedding](sentences, 32)
        
        print("Number of embeddings computed for {} : {}\n".format(embedding,sentences_emb.shape[0]))

        #on enregistre les embeddings avec les nouveaux calculés
        filename = "save/embedding_"+embedding+".npy"
        np.save(filename, sentences_emb)

    filename_ids = "save/ids.npy"
    np.save(filename_ids, ids)

    print("===== DONE =====")

if __name__ == "__main__":
    if len(sys.argv)>1:
        #si des embeddings sont précisé dans les arguments, on ne fait que ceux ci
        embedding_to_compute = sys.argv[1:]
    else:
        #sinon on les fait tous
        embedding_to_compute=list(embeddings.keys())
    compute_embedding_queue(embedding_to_compute)
