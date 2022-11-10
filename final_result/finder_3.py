import numpy as np
import os.path
import os
from sentence_transformers import SentenceTransformer
from modeles.cosine import cos_sim
import torch
from embeddings.embeddings import embeddings


def embed(sentence,embedding):
    sentences_emb = embeddings[embedding](sentence, once=True)
    return sentences_emb
    
def load_database_embed(embedding):
    ids = []

    filename_ids = "save/ids.npy"
    if os.path.isfile(filename_ids):
        ids  = np.load(filename_ids)
    else:
        print("Id file not found")
        0/0

    filename = "save/embedding_"+embedding+".npy"
    if os.path.isfile(filename):
        current_embedding  = np.load(filename)
        database_sentences_emb = current_embedding.astype(float)
        sentences_emb = database_sentences_emb
    else:
        print("Embedding not found : ", embedding)
        0/0
    return sentences_emb,ids


sentence_to_compare = "Batteries need to be bigger but it will be heavier"
#sentence_to_compare = "Big wheels are better for comfort but it will be harder to push."
#sentence_to_compare = "A high temperature is needed for the chemical reaction but it can damage the environnement"
#sentence_to_compare = "The dimensions of trench power MOSFETs metal-oxide-semiconductor field-effect transistor may be reduced for improving the electrical performance and decreasing the costs from generation to generation, which may be enabled both through better lithography systems and more powerful tools with an improved process control. While the field plate resistance may be rather uncritical due to its direct connection to the source metal, the gate resistance may provide difficulties as the gate trench is arranged between the columns of the field plate electrode."
#sentence_to_compare = "Sharing server improve server usage and reduce cost but do not allow to share common data. "




embedding_to_test=["custom_mpnet", "mpnet_base","deberta"]

for embedding in embedding_to_test:
    print(embedding)
    database_emb,id_ = load_database_embed(embedding)
    sentence_emb = embed(sentence_to_compare, embedding)
    sentence_emb/=np.linalg.norm(sentence_emb)
    results=[]
    pairs_emb = []
    for database_sentence in database_emb:
        pairs_emb.append([sentence_emb, database_sentence])
    pairs_emb = np.array(pairs_emb)
    results = cos_sim(pairs_emb)
    results = np.array(results).T
    print(results)
    
    print(np.average(results))
    print(np.min(results))
    print(np.max(results))
    
    print(np.linalg.norm(sentence_emb))
    print(np.min(sentence_emb))
    print(np.max(sentence_emb))

    number_to_keep=10
    ind = np.argpartition(results, -number_to_keep)[-number_to_keep:]
    ind = ind[np.argsort(results[ind])]
    for index in ind:
        print(index, results[index], id_[index])
    
    print("\n\n")