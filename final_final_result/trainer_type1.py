import sys
import getopt
import random

from compare_tools import load_local_database, make_pairs_to_compare

import torch
from torch import nn
from torch.utils.data import DataLoader

from sentence_transformers import InputExample, losses, SentenceTransformer, models
from sentence_transformers.evaluation import SimilarityFunction, EmbeddingSimilarityEvaluator


def parse_args_trainer(argv):
    """récupère les arguments envoyé en ligne de commande"""

    #tous les arguments à renvoyer en sortie
    arg_loss = ""
    arg_epochs = ""
    arg_output = ""
    arg_embedding_name = ""

    #aide des commandes à afficher à l'utilisateur
    arg_help = "{0} [--loss loss_name] [--epoch 10] -o <output_name> <embedding_name>".format(argv[0])

    try:
        #récupération des paramètres envoyés 
        opts, args = getopt.getopt(argv[1:], "hl:e:o:", ["help", "loss=", "epochs=", "output="])
        if len(args):
            arg_embedding_name = args[0]
        for opt, arg in opts:
            if opt in ("-h", "--help"):
                print(arg_help)
                sys.exit(2)
            elif opt in ("-l", "--loss"):
                arg_loss = arg
            elif opt in ("-e", "--epochs"):
                arg_epochs = int(arg)
            elif opt in ("-o", "--output"):
                arg_output = arg
    except:
        #si l'utilisateur a fait une erreur, on lui montre ce qu'il doit faire
        print(arg_help)
        sys.exit(2)

    #on regarde quel paramètre(s) manque(nt)
    missing_param = False
    if arg_output == "":
        print("output parameter is missing")
        missing_param = True
    if arg_embedding_name == "":
        print("embedding_name parameter is missing")
        missing_param = True
    if missing_param: #si un paramètre manque, on arrete tout
        sys.exit(2)
    return arg_loss, arg_epochs, arg_output, arg_embedding_name


def print_eval(score, epoch, step):
    """Affichage du résultat des evaluation"""
    print("Epoch {} - Step {} : {}".format(epoch, step, score))


def train(sentence_file, model_name,  output_model, loss_name, epochs):
    """fonction d'entrainement
    
    Arguments:
    sentence_file -- fichier des phrases d'entrainement
    model_name -- model sentece_transformers à affiner
    output_model -- nom du model entrainé
    loss_name -- nom de la fonction loss à utiliser : cosine, contrastive ou MNR 
    epochs -- nombre d'epochs 
    """
    BATCH_SIZE = 8
    TRAINING_DATA_SIZE = 8192
    TEST_DATA_SIZE = 8192
    OUTPUT_FOLDER = "my_models/"

    # check for incorrect values
    if loss_name != "" and not (loss_name == "cosine" or loss_name == "contrastive" or loss_name == "MNR"):
        print("unknown loss_name (either cosine, contrastive or none)")
        sys.exit(2)
    if type(epochs) is not int or epochs <= 0:
        print("epochs must be a non zero integer")
        sys.exit(2)
    if output_model == "":
        print("output_model can't be empty")
        sys.exit(2)
    if model_name == "":
        print("model_name can't be empty")
        sys.exit(2)
    if sentence_file == "":
        print("sentence_file can't be empty")
        sys.exit(2)

    #chargement du model dans le GPU (ou CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)

    # convertit loss_name en loss
    if loss_name == "" or loss_name == "cosine":
        train_loss = losses.CosineSimilarityLoss(model)
    elif loss_name == "contrastive":
        train_loss = losses.ContrastiveLoss(model)
    elif loss_name == "MNR":
        train_loss = losses.MultipleNegativesRankingLoss(model)

    # load sentences
    dataset = load_local_database(sentence_file)
    pairs, similarities_int, similarities_float, ids = make_pairs_to_compare(dataset)
    sentences = dataset["sentence"]

    # prepare training and test sentences
    train_data = []
    sentences1 = []
    sentences2 = []
    test_data = []
    for i in range(len(pairs)):
        test_data.append((sentences[pairs[i][0]], sentences[pairs[i][1]], similarities_int[i]))
        if loss_name == "MNR":
            if similarities_int[i]:
                train_data.append(InputExample(texts=[sentences[pairs[i][0]], sentences[pairs[i][1]]]))
        else:
            train_data.append(InputExample(texts=[sentences[pairs[i][0]], sentences[pairs[i][1]]], label=similarities_int[i]))


    random.shuffle(train_data)
    random.shuffle(test_data)
    dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    sentences1, sentences2, similarities_trunc = zip(*test_data[:TEST_DATA_SIZE])

    evaluator = EmbeddingSimilarityEvaluator(sentences1=sentences1, sentences2=sentences2, scores=similarities_trunc,  name="eval.csv", show_progress_bar=True, main_similarity=SimilarityFunction.COSINE, batch_size=128)

    # training
    print("init score : ", model.evaluate(evaluator))
    print("Start training")
    warmup_steps = int(len(dataloader) * epochs * 0.1)
    model.fit(train_objectives=[(dataloader, train_loss)], evaluator=evaluator, epochs=epochs, warmup_steps=warmup_steps, evaluation_steps=10000, output_path=OUTPUT_FOLDER+output_model, use_amp=True, callback=print_eval, steps_per_epoch=TRAINING_DATA_SIZE//BATCH_SIZE)
    print("Training Done")


if __name__ == "__main__":
    # parse input
    loss, epochs, output, embedding_name = parse_args_trainer(sys.argv)

    train("databases/response_1000.json", embedding_name, output, loss, epochs)
