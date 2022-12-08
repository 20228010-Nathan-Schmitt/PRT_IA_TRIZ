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


def find_best_f1_and_threshold(scores, labels, high_score_more_similar: bool):
    assert len(scores) == len(labels)

    scores = np.asarray(scores)
    labels = np.asarray(labels)

    rows = list(zip(scores, labels))

    rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

    best_f1 = best_precision = best_recall = 0
    threshold = 0
    nextract = 0
    ncorrect = 0
    total_num_duplicates = sum(labels)

    for i in range(len(rows) - 1):
        score, label = rows[i]
        nextract += 1

        if label == 1:
            ncorrect += 1

        if ncorrect > 0:
            precision = ncorrect / nextract
            recall = ncorrect / total_num_duplicates
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best_f1 = f1
                best_precision = precision
                best_recall = recall
                threshold = (rows[i][0] + rows[i + 1][0]) / 2

    return best_f1, best_precision, best_recall, threshold


def test():
    '''
    # check for incorrect values
    if model_type != 1 and model_type != 2 and model_type != 3:
        print("Incorrect type")
        sys.exit(2)
    '''

    test_sentences = load_test_sentence()
    sentence_ids, sentences = zip(*test_sentences)

    for embedding in "simCSE":
        print(embedding)
        result_list = []
        all_results = []
        all_labels = []

        # chargement des embeddings
        database_emb, id_ = load_embed("save/embedding_mesure_", embedding, "save/ids_mesure.npy")

        # on souhaite calculer le score d'exact match en considérant successivement qu'exact signifie retrouver le
        # brevet original dans les 5 premiers et en premier
        EM_1 = 0
        EM_5 = 0

        sentences_emb = embeddings[embedding](sentences, once=True)

        for i in range(len(sentence_ids)):
            sentence_id = sentence_ids[i]
            sentence_emb = sentences_emb[i]

            # construct all the pairs to compare : current reformulation vs whole database
            pairs_emb = []
            for database_sentence in database_emb:
                pairs_emb.append([sentence_emb, database_sentence])
            pairs_emb = np.array(pairs_emb)

            if model_type == 1 or model_type == 2:
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
                EM_5 += 1
                if rank_orignal_sentence == 0:
                    EM_1 += 1
            result_list.append(rank_orignal_sentence)

            # on sauvegarde les similaritées calculées et theorique pour le calcul de f1
            all_results += list(results)
            all_labels += [0 if j != id_rank else 1 for j in range(results.shape[0])]
        F1score, _, _, F1_threshold = find_best_f1_and_threshold(all_results, all_labels, True)

        # compute metrics
        MRR = np.mean(np.reciprocal(np.array(result_list) + 1))

        print(embedding + ":")
        print("MRR =", MRR)
        print("F1 score =", F1score, "(", F1_threshold, ")")
        print("Exact match in 5 =", EM_5)
        print("Exact match in 1 =", EM_1)

        f = open("test_results/metrics_" + embedding + ".txt", "w")
        f.write("MRR = " + str(MRR) + "\n")
        f.write("F1 score = " + str(F1score) + " (" + str(F1_threshold) + ")" + "\n")
        f.write("Exact match in 5 = " + str(EM_5) + "\n")
        f.write("Exact match in 1 = " + str(EM_1) + "\n")
        f.close()

        # plot histogram
        plt.hist(result_list, bins=range(0, 1 + np.amax(result_list)))
        plt.xlabel('Rank')
        plt.ylabel('Occurences')
        plt.title(embedding + "   avg=" + str(np.average(result_list)) + "   med=" + str(np.median(result_list)))
        plt.savefig("test_results/" + embedding + ".png", dpi=300)
        plt.close('all')


if __name__ == "__main__":
    embedding_to_test, model_type = parse_args_embedder(sys.argv)
    # si aucun embeddings n'est précisé dans les arguments, on les fait tous
    if not len(embedding_to_test):
        embedding_to_test = list(embeddings.keys())
    test(embedding_to_test, model_type)
