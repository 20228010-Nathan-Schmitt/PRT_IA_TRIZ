import sys
import getopt
import random
import os
import os.path

from compare_tools import load_local_database, make_pairs_to_compare
from distances.distances import distances
from embeddings.embeddings import embeddings

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


def parse_args_trainer(argv):
    arg_layers = []
    arg_epochs = ""
    arg_output = ""
    arg_embedding_name = []
    arg_help = "{0} -e 10 -l <layer1_layer2> -o <output_name> <embedding_name1> <embedding_name2> ...".format(argv[0])

    try:
        opts, args = getopt.getopt(argv[1:], "hl:e:o:", ["help", "layers=", "epochs=", "output="])
        if len(args):
            arg_embedding_name = args
        for opt, arg in opts:
            if opt in ("-h", "--help"):
                print(arg_help)
                sys.exit(2)
            elif opt in ("-l", "--layers"):
                arg_layers = [int(i) for i in arg.split("_")]
            elif opt in ("-e", "--epochs"):
                arg_epochs = int(arg)
            elif opt in ("-o", "--output"):
                arg_output = arg
    except:
        print(arg_help)
        sys.exit(2)

    missing_param = False
    if arg_output == "":
        print("output parameter is missing")
        missing_param = True
    if not len(arg_embedding_name):
        print("embedding_name parameter is missing")
        missing_param = True
    if missing_param:
        sys.exit(2)
    return arg_layers, arg_epochs, arg_output, arg_embedding_name

# compute and return the distance between pairs
# only compute once and save the result for future usage


def distance_between_pairs(embedding, distance, sentences, pairs):
    # compute and save sentence embeddings
    print("Getting embeddings for ", embedding)
    filename = "training/embedding_type3_"+embedding+".npy"
    if os.path.isfile(filename):
        sentences_emb = np.load(filename)
    else:
        sentences_emb = embeddings[embedding](sentences, once=True)
        np.save(filename, sentences_emb)

    print("Getting distances for ", distance)
    filename_result = "training/result2_"+embedding+"_"+distance+".npy"
    if os.path.isfile(filename_result):
        distance_between_embeddings = np.load(filename_result)
    else:
        def f(pair): return [sentences_emb[pair[0]], sentences_emb[pair[1]]]
        pairs_emb = np.array(list(map(f, pairs)))
        distance_between_embeddings = distances[distance](pairs_emb)
        np.save(filename_result, distance_between_embeddings)
    return distance_between_embeddings


def train(sentence_file, model_names,  output_model, layers_size, epochs):     # definition de la routine d'entrainement
    BATCH_SIZE = 128
    OUTPUT_FOLDER = "my_models/"
    LEARNING_RATE = 0.01

    # check for incorrect values
    if type(epochs) is not int or epochs <= 0:
        print("epochs must be a non zero integer")
        sys.exit(2)
    if output_model == "":
        print("output_model can't be empty")
        sys.exit(2)
    if not len(model_names):
        print("model_names can't be empty")
        sys.exit(2)
    if sentence_file == "":
        print("sentence_file can't be empty")
        sys.exit(2)

    device = "cuda" if torch.cuda.is_available() else "cpu"     # utilisation du GPU si disponible

    # construct distances between pairs and their similarities
    dataset = load_local_database(sentence_file)
    pairs, similarities_int, similarities_float, ids = make_pairs_to_compare(dataset)
    distance_between_embeddings = np.empty((0, len(pairs)))
    for embedding in model_names:
        print(embedding)
        for distance in distances:
            print(" └", distance)
            distance_between_embeddings = np.vstack((distance_between_embeddings, distance_between_pairs(embedding, distance, dataset["sentence"], pairs)))

    distance_between_embeddings = torch.from_numpy(distance_between_embeddings.T).float().to(device)
    similarities = torch.from_numpy(np.expand_dims(similarities_float, axis=1)).float().to(device)

    dataset = TensorDataset(distance_between_embeddings, similarities)
    kwargs = {'num_workers': 0, 'pin_memory': False} if device == 'cuda' else {}
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    layers = []         # definition du MLP et des fonctions d'activation pour chaque layer
    last_layer_size = distance_between_embeddings.shape[1]
    for layer in layers_size:
        layers.append(nn.Linear(last_layer_size, layer))
        layers.append(nn.Sigmoid())
        last_layer_size = layer
    model = nn.Sequential(*layers)
    model.to(device)

    loss_function = nn.MSELoss()        # definition de la fonction de perte
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    losses = []
    pred_y = model(distance_between_embeddings)
    loss = loss_function(pred_y, similarities)
    print("Loss : ", loss.item())
    losses.append(loss.item())

    for epoch in range(epochs):                         # entrainement pendant x epochs
        print("Epoch ", epoch)
        for inputs, outputs in dataloader:
            pred_y = model(inputs)                      # prediction
            loss = loss_function(pred_y, outputs)       # calcul de pertes
            model.zero_grad()                           # descente de gradient
            loss.backward()

            optimizer.step()
        pred_y = model(distance_between_embeddings)
        loss = loss_function(pred_y, similarities)
        print("Loss : ", loss.item())
        losses.append(loss.item())

    os.mkdir("./"+OUTPUT_FOLDER+"/"+output_model)       # définition d'un dossier et sauvegarde de modèle
    torch.save(model, OUTPUT_FOLDER + "/"+output_model+"/"+output_model)

    f = open(OUTPUT_FOLDER + "/"+output_model+"/" + output_model + "_embedding_names.txt", "w")
    for embedding in model_names:
        f.write(embedding+"\n")
    f.close()


if __name__ == "__main__":
    # parse input
    layers, epochs, output, embedding_name = parse_args_trainer(sys.argv)

    train("databases/response_1000.json", embedding_name, output, layers, epochs)
