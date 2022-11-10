from torch import embedding
from compare_tools import load_database
import numpy as np
from sentence_transformers import SentenceTransformer
from numpy.lib import recfunctions as rfn
import os.path

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #Work on CPU

def compute_embedding_queue():
    print("===== START =====")

    filename_computed = "save/computed.txt"

    database = load_database(0, None)
    ids = database["id"]
    sentences = database["sentence"]

    embedding = "my_model_sbert"
    model = SentenceTransformer("./"+embedding)
    print(embedding)
    
    #on esaye d'ouvrir les embeddings deja calculés
    filename = "save/embedding_"+embedding+".npy"
    sentences_emb = model.encode(sentences, batch_size=32, show_progress_bar=True)
    
    print("\nNumber of embeddings computed for {} : {}".format(embedding,sentences_emb.shape[0]))

    #on enregistre les embeddings avec les nouveaux calculés
    filename = "save/embedding_"+embedding+".npy"
    np.save(filename, sentences_emb)

    filename_ids = "save/ids.npy"
    np.save(filename_ids, ids)

    f = open(filename_computed, "w")
    f.write(str(ids.shape[0]))
    f.close()

    print("===== DONE =====")

"""for i in range(1000):
    compute_embedding_batch()"""
compute_embedding_queue()