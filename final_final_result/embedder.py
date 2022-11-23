import sys
import getopt

from embeddings.embeddings import embeddings
from compare_tools import load_local_database

import numpy as np


def parse_args_embedder(argv):
    arg_test = False
    arg_embedding_names=[]
    arg_help = "{0} --test [<embedding_name1> <embedding_name2> ...]".format(argv[0])

    try:
        opts, args = getopt.getopt(argv[1:], "ht", ["help", "test"])
        if len(args):
            arg_embedding_names = args
        for opt, arg in opts:
            if opt in ("-h", "--help"):
                print(arg_help)
                sys.exit(2)
            elif opt in ("-t", "--test"):
                arg_test = True
    except:
        print(arg_help)
        sys.exit(2)

    return arg_embedding_names, arg_test


def compute_embedding_queue(embedding_to_compute, sentences_file, basename, id_basename):
    print("===== START =====")

    database = load_local_database(sentences_file)
    ids = database["id"]
    sentences = database["sentence"]

    for embedding in embedding_to_compute:
        print(embedding)

        # on essaye d'ouvrir les embeddings deja calculés
        filename = "save/embedding_"+embedding+".npy"
        sentences_emb = embeddings[embedding](sentences, 32, once=True)

        print("Number of embeddings computed for {} : {}\n".format(embedding, sentences_emb.shape[0]))

        # on enregistre les embeddings avec les nouveaux calculés
        filename = basename+embedding+".npy"
        np.save(filename, sentences_emb)

    filename_ids = id_basename+".npy"
    np.save(filename_ids, ids)
    print("===== DONE =====")


if __name__ == "__main__":
    embedding_to_compute, is_test = parse_args_embedder(sys.argv)
    # si aucun embeddings n'est précisé dans les arguments, on les fait tous
    if not len(embedding_to_compute):
        embedding_to_compute = list(embeddings.keys())
    
    if is_test:
        sentences_file = "databases/response_1000_mesure.json"
        basename = "save/embedding_mesure_"
        id_basename = "save/ids_mesure"
    else:
        sentences_file = "databases/response_10000.json"
        basename = "save/embedding_"
        id_basename = "save/ids"

    print("Embeddings qui vont être calculés : ", "  ".join(embedding_to_compute))
    compute_embedding_queue(embedding_to_compute, sentences_file, basename, id_basename)
