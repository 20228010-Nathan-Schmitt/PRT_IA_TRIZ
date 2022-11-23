import sys
import getopt
import json

from embeddings.embeddings import embeddings
from distances.cosine import cos_sim
from compare_tools import load_embed

import numpy as np
import matplotlib.pyplot as plt


def parse_args_embedder(argv):
    arg_type = ""
    arg_embedding_names = []
    arg_help = "{0} --type [<embedding_name1> <embedding_name2> ...]".format(argv[0])

    try:
        opts, args = getopt.getopt(argv[1:], "ht:", ["help", "type="])
        if len(args):
            arg_embedding_names = args
        for opt, arg in opts:
            if opt in ("-h", "--help"):
                print(arg_help)
                sys.exit(2)
            elif opt in ("-t", "--type"):
                arg_type = int(arg)
    except:
        print(arg_help)
        sys.exit(2)

    missing_param = False
    if arg_type == "":
        print("type parameter is missing")
        missing_param = True
    if missing_param:
        sys.exit(2)

    return arg_embedding_names, arg_type


def load_test_sentence():
    f = open("databases/test_sentences.json", "r", encoding="utf8")
    response = json.load(f)
    f.close()

    return [(sentence["_id"], sentence["contradiction"]) for sentence in response["hits"]]


def test(embedding_to_test, model_type):

    # check for incorrect values
    if model_type != 1 and model_type != 2 and model_type != 3:
        print("Incorrect type")
        sys.exit(2)

    test_sentences = load_test_sentence()

    for embedding in embedding_to_test:
        print(embedding)
        result_list = []

        # chargement des embeddings
        database_emb, id_ = load_embed("save/embedding_mesure_", embedding, "save/ids_mesure.npy")

        for sentence in test_sentences:
            sentence_id = sentence[0]
            sentence_to_compare = sentence[1]
            sentence_emb = embeddings[embedding](sentence_to_compare, once=False)

            # construct all the pairs to compare : current reformulation vs whole database
            pairs_emb = []
            for database_sentence in database_emb:
                pairs_emb.append([sentence_emb, database_sentence])
            pairs_emb = np.array(pairs_emb)

            if model_type==1 or model_type==2:
                # compute cosine distance for each pair
                results = cos_sim(pairs_emb)
                results = np.array(results).T
            elif model_type == 3:
                print("Not implemented yet")

            # find the rank of the original sentence
            sorted_results = np.flip(np.argsort(results))
            id_rank = np.where(id_ == sentence_id)[0][0]
            rank_orignal_sentence = np.where(sorted_results == id_rank)[0][0]
            result_list.append(rank_orignal_sentence)
        embeddings[embedding]("remove", once=True)  # once=True remove the model from the GPU
        plt.hist(result_list, bins=range(0, 1+np.amax(result_list)))
        plt.xlabel('Rank')
        plt.ylabel('Occurences')
        plt.title(embedding + "   avg="+str(np.average(result_list)) + "   med="+str(np.median(result_list)))
        plt.savefig("test_results/"+embedding+".png", dpi=300)
        plt.close('all')


if __name__ == "__main__":
    embedding_to_test, model_type = parse_args_embedder(sys.argv)
    # si aucun embeddings n'est précisé dans les arguments, on les fait tous
    if not len(embedding_to_test):
        embedding_to_test = list(embeddings.keys())
    test(embedding_to_test, model_type)
