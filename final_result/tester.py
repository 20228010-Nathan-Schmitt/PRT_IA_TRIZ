import os
import numpy as np
import os.path
import torch
from modeles.cosine import cos_sim
from compare_tools import load_local_database
import matplotlib.pyplot as plt



def makePairsToCompare3(dataset):
    ids = dataset["id"]

    pairs = []
    similarities_int = []
    similarities_float = []

    for i in range(len(ids)):
        for j in range(len(ids)):
            if j > i:
                break
            # if len(pairs)>=5000:break
            pairs.append((i, j))

            patent1_triz_f = set(dataset["F_TRIZ_PARAMS"][i])
            patent1_triz_s = set(dataset["S_TRIZ_PARAMS"][i])
            patent2_triz_f = set(dataset["F_TRIZ_PARAMS"][j])
            patent2_triz_s = set(dataset["S_TRIZ_PARAMS"][j])
            size_intersection_f = len(
                list(patent1_triz_f.intersection(patent2_triz_f)))
            size_intersection_s = len(
                list(patent1_triz_s.intersection(patent2_triz_s)))

            similarities_int.append(int(not (not size_intersection_f or not size_intersection_s)))
            similarities_float.append((size_intersection_f+ size_intersection_s) / (max(len(patent1_triz_f), len(patent2_triz_f))+ max(len(patent1_triz_s), len(patent2_triz_s))))

    return np.array(pairs), np.array(similarities_int),np.array(similarities_float).astype("f4"), ids

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

def test(sentences_emb, pairs, similarities_int,similarities_float, model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(pairs.shape)
    f = lambda pair: [sentences_emb[pair[0]], sentences_emb[pair[1]]]
    pairs_emb = np.array(list(map(f, pairs)))
    results = cos_sim(pairs_emb)

    print("Moyenne similaritées",np.average(similarities_float))
    print("Moyenne resultat",np.average(results))

    diff = np.abs(results-similarities_float)
    mean_squared_error = np.sqrt((diff**2).mean())
    print("MSE",mean_squared_error)
    print("Ecart max",np.max(diff))
    print("Ecart min",np.min(diff))
    print("Ecart moy",np.average(diff))
    print("Ecart med",np.median(diff))
    
    plt.hist(diff,bins=100)
    plt.show()

   
embedding_to_test=["custom_mpnet_ultime"]

dataset = load_local_database("response_1000.json")
ids = dataset["id"]
pairs, similarities_int, similarities_float, ids = makePairsToCompare3(dataset)

number_to_keep=100000
pairs = pairs[:number_to_keep]
similarities_int = similarities_int[:number_to_keep]
similarities_float = similarities_float[:number_to_keep]
ids = ids[:number_to_keep]
for embedding in embedding_to_test:
    print("starting ", embedding)
    database_emb,id_ = load_database_embed(embedding)
    test(database_emb, pairs, similarities_int, similarities_float, embedding)

print("end", __name__)
