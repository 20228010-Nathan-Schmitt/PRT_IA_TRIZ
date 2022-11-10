import numpy as np
import os.path
import os
from modeles.cosine import cos_sim
from embeddings.embeddings import embeddings
import json


def embed(sentence, embedding):
    sentences_emb = embeddings[embedding](sentence, once=True)
    return sentences_emb


def load_database_embed(embedding):
    ids = []

    filename_ids = "save/ids.npy"
    if os.path.isfile(filename_ids):
        ids = np.load(filename_ids)
    else:
        print("Id file not found")
        0 / 0

    filename = "save/embedding_" + embedding + ".npy"
    if os.path.isfile(filename):
        current_embedding = np.load(filename)
        database_sentences_emb = current_embedding.astype(float)
        sentences_emb = database_sentences_emb
    else:
        print("Embedding not found : ", embedding)
        0 / 0
    return sentences_emb, ids


f = open("test_sentences.json", "r", encoding="utf8")
response = json.load(f)
f.close()

sentence_list = []
for sentence in response["hits"]:
    #print(sentence["_id"])
    sentence_list.append((
        sentence["_id"],
        sentence["contradiction"]
    ))


embedding_to_test = ["mpnet_base"]

for embedding in embedding_to_test:
    print(embedding)
    database_emb, id_ = load_database_embed(embedding)
    for sentence in sentence_list:
        sentence_to_compare = sentence[1]
        sentence_id = sentence[0]
        print(sentence_to_compare, '\n', sentence_id, '\n')
        sentence_emb = embed(sentence_to_compare, embedding)
        results = []
        pairs_emb = []
        for database_sentence in database_emb:
            pairs_emb.append([sentence_emb, database_sentence])
        pairs_emb = np.array(pairs_emb)
        results = cos_sim(pairs_emb)
        results = np.array(results).T
        print(results)

        number_to_keep = 10
        ind = np.argpartition(results, -number_to_keep)[-number_to_keep:]
        ind = ind[np.argsort(results[ind])]
        for index in ind:
            print(index, results[index], id_[index])

        print("\n\n")
