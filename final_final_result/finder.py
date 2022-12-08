import numpy as np
import os.path
import os
import sys
import getopt

from distances.cosine import cos_sim
from embeddings.embeddings import embeddings

import torch


def parse_args_finder(argv):  # récupération des arguments dans l'appel de la fonction
    arg_sentence = 0
    arg_embedding_names = []
    arg_type = []
    arg_help = "{0} --type 1 --sentence 1 [<embedding_name1> <embedding_name2> ...]".format(argv[0])

    try:
        opts, args = getopt.getopt(argv[1:], "ht:s:", ["help", "sentence="])
        if len(args):
            arg_embedding_names = args
        for opt, arg in opts:
            if opt in ("-s", "--sentence"):
                arg_sentence = int(arg)
            if opt in ("-t", "--type"):
                arg_type = int(arg)
            if opt in ("-h", "--help"):
                print(arg_help)
                sys.exit(2)
    except:
        print(arg_help)
        sys.exit(2)

    return arg_embedding_names, arg_sentence, arg_type

def interface():
    print("Bonjour.")
    validation = "False"
    while validation != "":
        sentence = input("Enoncez votre problème (contradiction) :")

        param = "1"
        param_list = []
        while param != "":
            print()
            param = input("Ajoutez un parametre TRIZ à votre contradiction (peu importe qu'il soit amélioré ou déterioré)"
                          "\nIl doit y avoir au moins un parametre \n"
                          "appuyez sur entrée pour cesser d'ajouter des parametres : ")
            if param != "": param_list.append(param)

        print()
        model_type = int(input("quel type de modèle souhaitez vous utiliser ? (1 ou 2)"))
        while model_type>2 or model_type<1:
            model_type = int(input("entrez un chiffre valide : 1 ou 2"))

        print()
        if model_type == 1:
            testModel = input('quel embedding souhaitez vous utiliser ? \n (mpnet_base, patentsberta)')
            while testModel != "mpnet_base" and testModel != "patents berta":
                testModel = input('donnez un nom valide (mpnet_base ou patentsberta)')
        elif model_type == 2:
            testModel = input("quel modèle souhaitez vous utiliser ?")

        print("\nLa recherche va etre effectuée avec ces parametres : \n contradiction : ", sentence, "parametres : ",
              param_list, "\n type de modèle : ", model_type, "\n modèle/embedding : ", testModel)
        validation = input("\nAppuyez sur entrée pour valider ou d'autres caractères pour recommencer")
        print()
    return [testModel], sentence, model_type, param_list

def embed(sentence, embedding):  # calcul de l'embedding de la phrase de l'utilisateur
    sentences_emb = embeddings[embedding](sentence, once=True)
    return sentences_emb


def load_database_embed(embedding, triz_params):  # chargement des embeddings pré-enregistrés des brevets
    ids = []

    # load triz parameter filters
    f_filters = np.load("databases/f_parameters_list.npy", allow_pickle=True).item()
    s_filters = np.load("databases/s_parameters_list.npy", allow_pickle=True).item()

    triz_filter = np.ones_like(f_filters["Speed"])  # création d'un tableau récapitulant les parametres TRIZ
    for param in triz_params:
        triz_filter *= np.logical_or(f_filters[param], s_filters[param])
    triz_filter = triz_filter > 0.5

    filename_ids = "save/ids.npy"  # chargement fichier id des brevets
    if os.path.isfile(filename_ids):
        ids = np.load(filename_ids)[triz_filter]
    else:
        print("Id file not found")
        0 / 0

    filename = "save/embedding_" + embedding + ".npy"  # chargement fichier embedding des brevets
    if os.path.isfile(filename):
        current_embedding = np.load(filename, mmap_mode="r")[triz_filter]
        database_sentences_emb = current_embedding.astype(float)
        sentences_emb = database_sentences_emb
    else:
        print("Embedding not found : ", embedding)
        0 / 0
    return sentences_emb, ids

    # définition de phrases d'exemples




sentences = {
    1: "Batteries need to be bigger but it will be heavier",
    2: "Big wheels are better for comfort but it will be harder to push.",
    3: "A high temperature is needed for the chemical reaction but it can damage the machine",
    4: "The dimensions of trench power MOSFETs metal-oxide-semiconductor field-effect transistor may be reduced for improving the electrical performance and decreasing the costs from generation to generation, which may be enabled both through better lithography systems and more powerful tools with an improved process control. While the field plate resistance may be rather uncritical due to its direct connection to the source metal, the gate resistance may provide difficulties as the gate trench is arranged between the columns of the field plate electrode.",
    5: "Sharing server improve server usage and reduce cost but do not allow to share common data. ",
    6: "For CAD applications, it can be long and tedious to manually display the two or three dimensions of each component in a new design. It can also be difficult for designers who choose the right combination of components for the first time.",
    7: "L'intrusion d'animaux et de plantes dans le système photovoltaïque est l'une des causes de la dégradation des performances et des temps d'arrêt du système. Certaines techniques existantes de pliage du métal impliquent un contact de surface important et une friction de glissement entre la machine et le matériau, ce qui les rend inadaptées à une utilisation manuelle.",
    8: "A higher voltage is needed to power the circuit. However it increase electromagnetic emmission and causes interference. ",
    9: "Having batteries that charge faster reduces the number of replacement batteries needed. But a faster charge increases the temperature and therefore the risk of fire",
    10: "1"
}

def get_best_n(results, number_to_show):
    ind = np.argpartition(results, -number_to_show)[-number_to_show:]   # classement des résultats conservés par performance
    return ind[np.argsort(results[ind])][::-1]


def show_results(results, id):
    results = np.array(results).T           # mise en forme des résultats

    print("======DEBUG======")
    print("score moyen", np.average(results))
    print("score min", np.min(results))
    print("score max", np.max(results))
    print("=================")
    print()

    number_to_show = 10
    print("======RESULTATS======")
    ind = get_best_n(results, number_to_show)
    for i,index in enumerate(ind): 
        print("{}\tscore:{:.8f} - brevet : {}".format(i+1, results[index], id[index]))
    print("yo")
    input()
    print("ya")
    while input("Entre quelquechose pour afficher la suite : ")!="":
        number_to_show+=10
        ind = get_best_n(results, number_to_show)
        for i,index in enumerate(ind):
            if i>= number_to_show-10:
                print("{}\tscore:{:.8f} - brevet : {}".format(i+1, results[index], id[index]))
    print("Au revoir ❤️")


def find_type1(embedding_to_test, sentence_to_compare, triz_params):# fonction finder proprement dite
    for embedding in embedding_to_test:                             # on cherche en utilisant une liste d'embeddings
        print(embedding)

        print("Start embeddings loading")
        database_emb, id_ = load_database_embed(embedding, triz_params)
        print("Embeddings loaded")

        sentence_emb = embed(sentence_to_compare, embedding)  # calcul de l'embedding de la phrase utilisateur
        sentence_emb /= np.linalg.norm(sentence_emb)

        batch_size = 50_000
        results = np.array([])
        for i in range(database_emb.shape[0] // batch_size + 1):  # calcul des résultats par groupe de 55000 brevets
            print("Start batch", i)
            start = i * batch_size
            end = min(start + batch_size, database_emb.shape[0])
            pairs_emb = np.empty((end - start, 2, database_emb.shape[1]))
            pairs_emb[:, 0] = sentence_emb
            pairs_emb[:, 1] = database_emb[start:end]
            results = np.concatenate((results, cos_sim(pairs_emb)))
            print("Batch", i, "done")
        """pairs_emb = []
        for database_sentence in database_emb:
            pairs_emb.append([sentence_emb, database_sentence])
        database_emb=None
        sentence_emb=None
        pairs_emb = np.array(pairs_emb)"""
        show_results(results, id_)

        
def find_type2(model_to_test, sentence_to_compare, triz_params):
    MODEL_FOLDER = "my_models"
    model_to_test = model_to_test[0]
    print(model_to_test)
    f = open(MODEL_FOLDER + "/" + model_to_test + "/" + model_to_test + "_embedding_names.txt", "r")
    embedding = f.read()
    print(embedding)

    print("Start embeddings loading")
    database_emb, id_ = load_database_embed(embedding, triz_params)
    print("Embeddings loaded")

    model = torch.load(MODEL_FOLDER + "/" + model_to_test + "/" + model_to_test)

    sentence_emb = embed(sentence_to_compare, embedding)
    sentence_emb /= np.linalg.norm(sentence_emb)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 50_000
    results = np.array([])
    for i in range(database_emb.shape[0] // batch_size + 1):
        print("Start batch", i)
        start = i * batch_size
        end = min(start + batch_size, database_emb.shape[0])
        pairs_emb = np.empty((end - start, 2, database_emb.shape[1]))
        pairs_emb[:, 0] = sentence_emb
        pairs_emb[:, 1] = database_emb[start:end]
        pairs_emb = torch.from_numpy(pairs_emb).float().to(device)

        pairs_output = model(pairs_emb).detach().cpu().numpy()
        results = np.concatenate((results, cos_sim(pairs_output)))
        print("Batch", i, "done")
    """pairs_emb = []
    for database_sentence in database_emb:
        pairs_emb.append([sentence_emb, database_sentence])
    database_emb=None
    sentence_emb=None
    pairs_emb = np.array(pairs_emb)"""

    results = np.array(results).T

    print("Min in array", np.min(results))
    print("Avg of array", np.average(results))
    print("Max in array", np.max(results))

    number_to_keep = 10
    ind = np.argpartition(results, -number_to_keep)[-number_to_keep:]
    ind = ind[np.argsort(results[ind])]
    for i, index in enumerate(ind):
        print(number_to_keep - i, index, results[index], id_[index], sep=" \t")

    print("\n\n")




if __name__ == "__main__":
    stop=False
    while not stop:

        triz_parameters = {
            1:"Weight of Moving Object", 
            2:"Weight of Stationary Object",
            3:"Length of Moving Object",
            4:"Length of Stationary Object",
            5:"Area of Moving Object",
            6:"Area of Stationary Object",
            7:"Volume of Moving Object",
            8:"Volume of Stationary Object",
            9:"Speed",
            10:"Force Torque",
            11:"Tension Pressure",
            12:"Shape",
            13:"Stability of Object",
            14:"Strength",
            15:"Durability of Moving Object",
            16:"Durability of Stationary Object",
            17:"Temperature",
            18:"Brightness",
            19:"Energy Spent by Moving Object",
            20:"Energy Spent by Stationary Object",
            21:"Power",
            22:"Waste of Energy",
            23:"Waste of Substance",
            24:"Loss of Information",
            25:"Waste of Time",
            26:"Amount of Substance",
            27:"Reliability",
            28:"Accuracy of Measurement",
            29:"Accuracy of Manufacturing",
            30:"Harmful Factors Acting on Object",
            31:"Harmful Side Effects",
            32:"Manufacturability",
            33:"Convenience of Use",
            34:"Reparability",
            35:"Adaptability",
            36:"Complexity of Device",
            37:"Complexity of Control",
            38:"Level of Automation",
            39:"Productivity",
        }
        model_to_test, sentence, model_type, param_list = interface()

        print("Embeddings qui vont être testés : ", "  ".join(model_to_test))
        if model_type == 1:
            find_type1(model_to_test, sentence, param_list)
        elif model_type == 2:
            find_type2(model_to_test, sentence, param_list)

        stop = (input("Entre quelquechose pour refaire une recherche") == "" )
