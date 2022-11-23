import requests
import json
import numpy as np

def load_local_database(filename="response_1000.json"):
    #import json file as dictionary
    f=open(filename, "r", encoding="utf8")
    response = json.load(f)
    f.close()    
    
    #extract only id, sentences and triz parameters fromeach patent
    patents = []
    for patent in response["hits"]["hits"]:
        patents.append((
            patent["_id"],
            patent["fields"]["F_SENTS"][0] + " " +patent["fields"]["S_SENTS"][0], #contradiction
            patent["fields"]["F_TRIZ_PARAMS"],
            patent["fields"]["S_TRIZ_PARAMS"],
        ))
    return np.array(patents,dtype=[("id", "U32"),("sentence",np.unicode,1024),("F_TRIZ_PARAMS",object), ("S_TRIZ_PARAMS",object)])

def make_pairs_to_compare(dataset):
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

