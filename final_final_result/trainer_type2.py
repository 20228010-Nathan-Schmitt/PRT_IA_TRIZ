import os
import sys
import getopt
import random
import numpy as np

from compare_tools import load_local_database, make_pairs_to_compare
from embeddings.embeddings import embeddings

import torch
from torch import nn
from torch.utils.data import TensorDataset, IterableDataset, DataLoader

from sentence_transformers import InputExample, losses, SentenceTransformer, models
from sentence_transformers.evaluation import SimilarityFunction, EmbeddingSimilarityEvaluator


def parse_args_trainer(argv):
    arg_loss = ""
    arg_layers = []
    arg_epochs = ""
    arg_output = ""
    arg_embedding_name = ""
    arg_help = "{0} -t <type (1/2)> [-L <layer1_layer2>] [-l loss_name] [-e 10] -o <output_name> <embedding_name>".format(argv[0])

    try:
        opts, args = getopt.getopt(argv[1:], "hL:l:e:o:", ["help", "loss=", "layers=", "epochs=", "output="])
        if len(args):
            arg_embedding_name = args[0]
        for opt, arg in opts:
            if opt in ("-h", "--help"):
                print(arg_help)
                sys.exit(2)
            elif opt in ("-l", "--loss"):
                arg_loss = arg
            elif opt in ("-L", "--layers"):
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
    if arg_embedding_name == "":
        print("embedding_name parameter is missing")
        missing_param = True
    if missing_param:
        sys.exit(2)
    return arg_loss, arg_layers, arg_epochs, arg_output, arg_embedding_name


def print_eval(score, epoch, step):
    print("Epoch {} - Step {} : {}".format(epoch, step, score))


def train(sentence_file, embedding,  output_model, loss_name, layers, epochs):
    BATCH_SIZE = 1024
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
    if embedding == "":
        print("model_name can't be empty")
        sys.exit(2)
    embedding_filename = "training/embedding_"+embedding+".npy"
    dataset = load_local_database(sentence_file)
    if os.path.isfile(embedding_filename):
        sentences_emb = np.load(embedding_filename)
    else:        
        sentences_emb = embeddings[embedding](dataset["sentence"], once=True)
        np.save(embedding_filename, sentences_emb)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # prepare training and test sentences
    pairs, similarities_int, similarities_float, ids = make_pairs_to_compare(dataset[:1000])
    similarities_int = similarities_int*2-1

    print(pairs.shape[0])


    #sentences1 = torch.from_numpy(pairs[:,0]).float().to(device)
    #sentences2 = torch.from_numpy(pairs[:,1]).float().to(device)
    sentences1_emb = np.reshape(sentences_emb[pairs[:,0]], (-1, sentences_emb.shape[1]))
    sentences2_emb = np.reshape(sentences_emb[pairs[:,1]], (-1, sentences_emb.shape[1]))
    sentences1_emb = torch.from_numpy(sentences1_emb).float().to(device)
    sentences2_emb = torch.from_numpy(sentences2_emb).float().to(device)
    similarities_int = torch.from_numpy(similarities_int).to(device)
    dataset = TensorDataset(sentences1_emb, sentences2_emb, similarities_int)
    print(similarities_int)

    kwargs = {'num_workers': 0, 'pin_memory': False} if device=='cuda' else {}
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    denses_layers = [
        nn.Linear(sentences_emb.shape[1], layers[0]),
    ]
    for i in range(1,len(layers)):
        denses_layers.append(nn.ReLU())
        #denses_layers.append(nn.Sigmoid())
        #denses_layers.append(nn.GELU())
        denses_layers.append(nn.Linear(layers[i-1], layers[i]))
    denses_layers.append(nn.Tanh())
    model = nn.Sequential(*denses_layers)
    print(model)
    model.to(device)

    train_loss = torch.nn.CosineEmbeddingLoss(0.5)
    optimizer = torch.optim.Adam(model.parameters())


    # training
    losses = []
    """pred_y_s1 = model(sentences1_emb)
    pred_y_s2 = model(sentences2_emb)
    loss = train_loss(pred_y_s1, pred_y_s2, similarities_int)
    print("Loss : ",loss.item())
    losses.append(loss.item())"""

    print("Start training")
    for epoch in range(epochs):
        print("Epoch ",epoch)
        for i, data in enumerate(dataloader):
            if not i%1000: print(i)
            s1, s2, simi = data
            model.zero_grad()
            #s1 = np.reshape(s1, (8, -1))
            #s2 = np.reshape(s2, (8, -1))
            pred_y_s1 = model(s1)
            pred_y_s2 = model(s2)
            loss = train_loss(pred_y_s1, pred_y_s2, simi)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
       
        print(i, losses[-1])

        """pred_y_s1 = model(sentences1_emb)
        pred_y_s2 = model(sentences2_emb)
        loss = train_loss(pred_y_s1, pred_y_s2, similarities_int)
        print("Loss : ",loss.item())
        losses.append(loss.item())"""
    os.makedirs("./"+OUTPUT_FOLDER+"/"+output_model, exist_ok=True)
    torch.save(model, OUTPUT_FOLDER + "/"+output_model+"/"+output_model)
    
    f = open(OUTPUT_FOLDER + "/"+output_model+"/" + output_model + "_embedding_names.txt", "w")
    f.write(embedding)
    f.close()

    print("Training Done")
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.yscale("log")
    plt.gca().set_ylim(bottom=0)
    plt.title("Loss")
    plt.show()


if __name__ == "__main__":
    # parse input
    loss, layers, epochs, output, embedding_name = parse_args_trainer(sys.argv)

    train("databases/response_1000.json", embedding_name, output, loss, layers, epochs)
