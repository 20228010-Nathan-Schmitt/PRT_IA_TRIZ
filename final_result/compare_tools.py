import requests
import json
import numpy as np

def load_training_dataset():
    f=open("training_dataset.txt", "r", encoding="utf8")
    lines = [line.rstrip('\n') for line in f]
    f.close()

    sentences = {}    
    for line in lines:
        sentences_raw = line.split(" ")

        if len(sentences_raw)<3:continue
        patentNumber = sentences_raw[0]

        if not patentNumber in sentences: sentences[patentNumber]={"patent":"", "short":[]}
        if sentences_raw[1]=="F" or sentences_raw[1]=="S":
            sentences[patentNumber]["patent"] += " ".join(sentences_raw[2:])+" "
        if sentences_raw[1]=="R":
            sentences[patentNumber]["short"].append(" ".join(sentences_raw[2:]))
    return sentences

def makePairsToCompare(dataset, keys):
    pairs = []
    n=len(keys)
    import numpy as np
    similarities=np.array([])
    for i in range(len(keys)):
        patentNumber = keys[i]
        for j in range(len(keys)):
            if j>i:break
            patentNumber2 = keys[j]
            
            #between patents
            pairs.append((i, j))
            similarities = np.insert(similarities, similarities.size, patentNumber==patentNumber2)
            
            #between patent1 and short2
            for short in range(len(dataset[patentNumber2]["short"])):
                pairs.append((i, n+j+short))
                similarities = np.insert(similarities, similarities.size, patentNumber==patentNumber2)
            if j!=i:    
                #between patent2 and short1
                for short in range(len(dataset[patentNumber]["short"])):
                    pairs.append((j, n+i+short))
                    similarities = np.insert(similarities, similarities.size, patentNumber==patentNumber2)
                
    return pairs, similarities

def makePairsToCompare2(dataset):
    ids = dataset["id"]
    
    pairs = []
    similarities=[]
    
    for i in range(len(ids)):
        for j in range(len(ids)):
            if j>i:break
            pairs.append((i, j))

            patent1_triz_f = set(dataset["F_TRIZ_PARAMS"][i])
            patent1_triz_s = set(dataset["S_TRIZ_PARAMS"][i])
            patent2_triz_f = set(dataset["F_TRIZ_PARAMS"][j])
            patent2_triz_s = set(dataset["S_TRIZ_PARAMS"][j])
            size_intersection_f = len(list(patent1_triz_f.intersection(patent2_triz_f)))
            size_intersection_s = len(list(patent1_triz_s.intersection(patent2_triz_s)))
            """size_union_f = len(list(patent1_triz_f.union(patent2_triz_f)))
            size_union_s = len(list(patent1_triz_s.union(patent2_triz_s)))

            similarities.append((size_intersection_f+ size_intersection_s) / (size_union_f+size_union_s))"""
            similarities.append((size_intersection_f+ size_intersection_s) / (max(len(patent1_triz_f), len(patent2_triz_f))+ max(len(patent1_triz_s), len(patent2_triz_s))))
    return np.array(pairs), np.array(similarities), ids

def transposeList(pairs):
    import numpy as np
    return np.swapaxes(pairs, 0,1)
    


database = []
def load_database(from_, size):
    global database, sorted_keys

    
    request_body = {
      "from": from_,
      "size": size,
      "query": {
        "bool": {
          "must": []
        }
      },
      "sort": [
        {
          "_id": {
            "order": "asc"
          }
        }
      ],
        "fields":[
            "CONTRADICTION_SCORE",
            "F_SENTS",
            "S_SENTS",
            "F_TRIZ_PARAMS",
            "S_TRIZ_PARAMS"
        ],
        "_source": False
    }
    
    """r = requests.get("https://vm-csip-es.icube.unistra.fr/db/db_solve/patents/_search", headers = {"Authorization":"ApiKey ekRWY3ZINEI1b1ktTzQzX3ZhRGM6aTRlbVJjQXZUdzY2a3hDTmFmMVhoZw=="}, json = request_body, verify=False)
    response  =r.json()

    """
    if not len(database):
        f=open("response_10000.json", "r", encoding="utf8")
        response = json.load(f)
        f.close()
        database = response["hits"]["hits"]

    database_extract = database[from_:from_+size] if size is not None else database[from_:]
        
    patents = []
    for patent in database_extract:
        patents.append((
            patent["_id"],
            patent["fields"]["F_SENTS"][0] + " " +patent["fields"]["S_SENTS"][0], #contradiction
            patent["fields"]["F_TRIZ_PARAMS"],
            patent["fields"]["S_TRIZ_PARAMS"],
        ))
    return np.array(patents,dtype=[("id", "U32"),("sentence",np.compat.unicode,1024),("F_TRIZ_PARAMS",object), ("S_TRIZ_PARAMS",object)])


def load_local_database(filename="response_1000.json"):
    f=open(filename, "r", encoding="utf8")
    response = json.load(f)
    f.close()
        
    
    patents = []
    for patent in response["hits"]["hits"]:
        patents.append((
            patent["_id"],
            patent["fields"]["F_SENTS"][0] + " " +patent["fields"]["S_SENTS"][0], #contradiction
            patent["fields"]["F_TRIZ_PARAMS"],
            patent["fields"]["S_TRIZ_PARAMS"],
        ))
    return np.array(patents,dtype=[("id", "U32"),("sentence",np.unicode,1024),("F_TRIZ_PARAMS",object), ("S_TRIZ_PARAMS",object)])


def merge_array(arr1, arr2,type1, type2, number2):
    a = np.concatenate((arr1, arr2), axis=1)
    return a
    return np.core.records.fromarrays(a.T, names='id, '+", ".join(["c"+str(u) for u in range(number2)]), formats = type1+(", "+type2)*number2)


def remove_field_num(a, i):
    names = list(a.dtype.names)
    new_names = names[:i] + names[i+1:]
    b = a[new_names]
    return b