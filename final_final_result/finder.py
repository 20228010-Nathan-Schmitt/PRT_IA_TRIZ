import numpy as np
import os.path
import os
import sys
import getopt

from distances.cosine import cos_sim
import torch
from embeddings.embeddings import embeddings

def parse_args_finder(argv):
    arg_sentence=0
    arg_embedding_names=[]
    arg_help = "{0} --sentence 1 [<embedding_name1> <embedding_name2> ...]".format(argv[0])

    try:
        opts, args = getopt.getopt(argv[1:], "hs:", ["help", "sentence="])
        if len(args):
            arg_embedding_names = args
        for opt, arg in opts:
            if opt in ("-s", "--sentence"):
                arg_sentence=int(arg)
            if opt in ("-h", "--help"):
                print(arg_help)
                sys.exit(2)
    except:
        print(arg_help)
        sys.exit(2)

    return arg_embedding_names, arg_sentence


def embed(sentence, embedding):
    sentences_emb = embeddings[embedding](sentence, once=True)
    return sentences_emb


def load_database_embed(embedding, triz_params):
    ids = []

    #load triz parameter filters
    f_filters = np.load("databases/f_parameters_list.npy", allow_pickle=True).item()
    s_filters = np.load("databases/s_parameters_list.npy", allow_pickle=True).item()

    triz_filter = np.ones_like(f_filters["Speed"])
    for param in triz_params:
        triz_filter *= np.logical_or(f_filters[param],s_filters[param])
    triz_filter = triz_filter>0.5
    
    filename_ids = "save/ids.npy"
    if os.path.isfile(filename_ids):
        ids = np.load(filename_ids)[triz_filter]
    else:
        print("Id file not found")
        0/0

    filename = "save/embedding_"+embedding+".npy"
    if os.path.isfile(filename):
        current_embedding = np.load(filename, mmap_mode="r")[triz_filter]
        database_sentences_emb = current_embedding.astype(float)
        sentences_emb = database_sentences_emb
    else:
        print("Embedding not found : ", embedding)
        0/0
    return sentences_emb, ids

sentences = {
    1:"Batteries need to be bigger but it will be heavier",
    2:"Big wheels are better for comfort but it will be harder to push.",
    3:"A high temperature is needed for the chemical reaction but it can damage the machine",
    4:"The dimensions of trench power MOSFETs metal-oxide-semiconductor field-effect transistor may be reduced for improving the electrical performance and decreasing the costs from generation to generation, which may be enabled both through better lithography systems and more powerful tools with an improved process control. While the field plate resistance may be rather uncritical due to its direct connection to the source metal, the gate resistance may provide difficulties as the gate trench is arranged between the columns of the field plate electrode.",
    5:"Sharing server improve server usage and reduce cost but do not allow to share common data. ",
    6:"For CAD applications, it can be long and tedious to manually display the two or three dimensions of each component in a new design. It can also be difficult for designers who choose the right combination of components for the first time.",
    7:"L'intrusion d'animaux et de plantes dans le système photovoltaïque est l'une des causes de la dégradation des performances et des temps d'arrêt du système. Certaines techniques existantes de pliage du métal impliquent un contact de surface important et une friction de glissement entre la machine et le matériau, ce qui les rend inadaptées à une utilisation manuelle.",
    8:"A higher voltage is needed to power the circuit. However it increase electromagnetic emmission and causes interference. ",
    9:"Having batteries that charge faster reduces the number of replacement batteries needed. But a faster charge increases the temperature and therefore the risk of fire"
}

def find(embedding_to_test, sentence_to_compare, triz_params):
    for embedding in embedding_to_test:
        print(embedding)

        print("Start embeddings loading")
        database_emb, id_ = load_database_embed(embedding, triz_params)
        print("Embeddings loaded")

        sentence_emb = embed(sentence_to_compare, embedding)
        sentence_emb /= np.linalg.norm(sentence_emb)

        batch_size=55_000
        results = np.array([])
        for i in range(database_emb.shape[0]//batch_size +1):
            print("Start batch",i)
            start = i* batch_size
            end = min(start+batch_size, database_emb.shape[0])
            pairs_emb = np.empty((end-start, 2, database_emb.shape[1]))
            pairs_emb[:,0] = sentence_emb
            pairs_emb[:,1] = database_emb[start:end]
            results = np.concatenate((results,cos_sim(pairs_emb)))
            print("Batch",i,"done")
        """pairs_emb = []
        for database_sentence in database_emb:
            pairs_emb.append([sentence_emb, database_sentence])
        database_emb=None
        sentence_emb=None
        pairs_emb = np.array(pairs_emb)"""


        results = np.array(results).T
        print(results)

        print(np.average(results))
        print(np.min(results))
        print(np.max(results))


        number_to_keep = 10
        ind = np.argpartition(results, -number_to_keep)[-number_to_keep:]
        ind = ind[np.argsort(results[ind])]
        for index in ind:
            print(index, results[index], id_[index])

        print("\n\n")

if __name__ == "__main__":
    embedding_to_test, sentence = parse_args_finder(sys.argv)
    
    print("Embeddings qui vont être testés : ", "  ".join(embedding_to_test))
    find(embedding_to_test,sentences[sentence], ["Speed", "Temperature"])
