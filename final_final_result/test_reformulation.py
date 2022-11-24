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

        # afin de calculer le F1 score, on va enregistrer pour chaque embedding les vais positifs etc.
        true_positive = 0
        false_positive = 0
        false_negative = 0
        #on souhaite aussi calculer le score d'exact match en considérant successivement qu'exact signifie retrouver le brevet original dans les 5 premiers et en premier
        EM_1 = 0
        EM_5 = 0

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
            if rank_orignal_sentence <= 4:
                EM_5 += EM_5
                if rank_orignal_sentence == 0:
                    EM_1 += EM_1
            result_list.append(rank_orignal_sentence)

            # pour le F1score, on choisit de considérer un résultat comme positif s'il a une similarité de plus de 70%
            #pour chaque phrase on incrément le nombre de vrais/faux positifs et negatifs
            p_or_n = sorted_results > 0.7
            for i in range(p_or_n):
                if i == id_rank:
                    if p_or_n[i]:
                        true_positive += true_positive
                    else:
                        false_negative += false_negative
                elif p_or_n[i]:
                    false_positive += false_positive

        embeddings[embedding]("remove", once=True)  # once=True remove the model from the GPU

        #compute metrics
        MRR = np.mean(np.reciprocal(result_list+1))

        precision = true_positive/(true_positive+false_positive)
        recall = true_positive/(true_positive+false_negative)
        F1score = 2 * precision * recall /(precision + recall)

        print(embedding + ":\n MRR = " + MRR + ":\n F1 score = " + F1score + ":\n exact match in 5 = " + EM_5 + ":\n exact match in 1 = " + EM_1)

        #plot histogram
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
