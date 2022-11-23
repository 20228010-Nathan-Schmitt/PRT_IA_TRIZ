import os
import numpy as np
import os.path
import torch
from modeles.modeles import models
from embeddings.embeddings import embeddings
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

def embed(embeddings_to_test, sentences):
    sentences_emb=[]
    ids = []

    for embedding in embeddings_to_test:
        filename = "training/test_embedding_"+embedding+".npy"
        if os.path.isfile(filename):
            current_embedding  = np.load(filename)
            database_sentences_emb = current_embedding.astype(float)
        else:
            current_embedding = embeddings[embedding](sentences, once=True)
            database_sentences_emb = current_embedding.astype(float)
            np.save(filename, current_embedding)
        sentences_emb.append(database_sentences_emb)
    return sentences_emb,ids

def test(sentences_emb, pairs, similarities_int,similarities_float, model_name):    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(pairs.shape)
    distances=[]
    for i in range(len(sentences_emb)):
        print(sentences_emb[i][0][:6])
        print(sentences_emb[i][1][:6])
        f = lambda pair: [sentences_emb[i][pair[0]], sentences_emb[i][pair[1]]]
        pairs_emb = np.array(list(map(f, pairs)))
        for model in models:
            distances.append(models[model](pairs_emb))
    distances = np.array(distances).T
    print(distances)
    distances = torch.from_numpy(distances).float().to(device)

    model = torch.load(model_name).to(device)
    print(model)
    results  = model(distances).detach().cpu().numpy().T
    print(results[:,:20])
    print(similarities_float[:20])


    print("Moyenne similaritées",np.average(similarities_float))
    print("Moyenne resultat",np.average(results))

    diff = np.abs(results-similarities_float)
    mean_squared_error = np.sqrt((diff**2).mean())
    print("MSE",mean_squared_error)
    print("Ecart max",np.max(diff))
    print("Ecart min",np.min(diff))
    print("Ecart moy",np.average(diff))
    print("Ecart med",np.median(diff))

   
embedding_to_test = ["custom_mpnet_ultime", "mpnet_base"]

dataset = load_local_database()
pairs, similarities_int, similarities_float, ids = makePairsToCompare3(dataset)

number_to_keep=100000
pairs = pairs[:number_to_keep]
similarities_int = similarities_int[:number_to_keep]
similarities_float = similarities_float[:number_to_keep]
database_emb,id_ = embed(embedding_to_test,dataset["sentence"])
test(database_emb, pairs, similarities_int, similarities_float, "my_models/my_model_mpnet_and_mpnet_custom")

print("end", __name__)
