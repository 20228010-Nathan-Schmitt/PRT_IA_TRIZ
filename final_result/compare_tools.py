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
def makePairsToCompare2():
    f=open("response_1000.json", "r", encoding="utf8")
    response = json.load(f)
    f.close()

    pairs = []
    import numpy as np
    similarities=np.array([])
    
    keys = response["hits"]["hits"][:100]
    
    for i in range(len(keys)):
        patent1 = keys[i]
        for j in range(len(keys)):
            if j>i:break
            patent2 = keys[j]
            pairs.append((patent1["_id"], patent2["_id"]))
            similarities = np.insert(
                similarities, 
                similarities.size,
                len(list(set(patent1["fields"]["F_TRIZ_PARAMS"]).intersection(patent2["fields"]["F_TRIZ_PARAMS"]))) + len(list(set(patent1["fields"]["S_TRIZ_PARAMS"]).intersection(patent2["fields"]["S_TRIZ_PARAMS"])))
            )
    return pairs, similarities

def transposeList(pairs):
    import numpy as np
    return np.swapaxes(pairs, 0,1)
    
def load_database(start):
    size = 50
    
    request_body = {
      "from": size*start,
      "size": size,
      "query": {
        "bool": {
          "must": [
            {
              "match": {
                "INVENTION_TITLE": "dev"
              }
            },
            {
              "match": {
                "REF_PATENT": "US1"
              }
            },
            {
              "range": {
                "PUBLICATION_DATE": {
                  "gte": "01/01/2021",
                  "format": "dd/MM/yyyy||yyyy"
                }
              }
            }
          ]
        }
      }
    }
    
    header = {
        "key": "Authorization",
        "value": "ApiKey ekRWY3ZINEI1b1ktTzQzX3ZhRGM6aTRlbVJjQXZUdzY2a3hDTmFmMVhoZw==",
        "type": "text"
    }
    r = requests.get("https://vm-csip-es.icube.unistra.fr/db/db_solve/patents/_search", headers = {"Authorization":"ApiKey ekRWY3ZINEI1b1ktTzQzX3ZhRGM6aTRlbVJjQXZUdzY2a3hDTmFmMVhoZw=="}, json = request_body, verify=False)
    response  =r.json()
    """
    f=open("response.json", "r", encoding="utf8")
    response = json.load(f)
    f.close()"""
        
    
    patents = []
    for patent in response["hits"]["hits"]:
        patents.append((
            patent["_id"],
            patent["_source"]["F_SENTS"][0] + " " +patent["_source"]["S_SENTS"][0], #contradiction
            patent["_source"]["F_TRIZ_PARAMS"],
            patent["_source"]["S_TRIZ_PARAMS"],
        ))
    return np.array(patents,dtype=[("id", "U32"),("sentence",np.unicode,1024),("F_TRIZ_PARAMS",object), ("S_TRIZ_PARAMS",object)])
    
def load_local_database():
    f=open("response_1000.json", "r", encoding="utf8")
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