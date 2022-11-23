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
    arg_type = ""
    arg_loss = ""
    arg_layers = []
    arg_epochs = ""
    arg_output = ""
    arg_embedding_name = ""
    arg_help = "{0} -t <type (1/2)> [-l <layer1_layer2>] -o <output_name> <embedding_name>".format(argv[0])

    try:
        opts, args = getopt.getopt(argv[1:], "ht:L:l:e:o:", ["help", "type=", "loss=", "layers=", "epochs=", "output="])
        if len(args):
            arg_embedding_name = args[0]
        for opt, arg in opts:
            if opt in ("-h", "--help"):
                print(arg_help)
                sys.exit(2)
            elif opt in ("-t", "--type"):
                arg_type = int(arg)
            elif opt in ("-L", "--loss"):
                arg_loss = arg
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
    if arg_type == "":
        print("type parameter is missing")
        missing_param = True
    if arg_output == "":
        print("output parameter is missing")
        missing_param = True
    if arg_embedding_name == "":
        print("embedding_name parameter is missing")
        missing_param = True
    if missing_param:
        sys.exit(2)
    return arg_type, arg_loss, arg_layers, arg_epochs, arg_output, arg_embedding_name


def print_eval(score, epoch, step):
    print("Epoch {} - Step {} : {}".format(epoch, step, score))


def train(sentence_file, model_name,  output_model, model_type, loss_name, layers, epochs):
    BATCH_SIZE = 8
    TRAINING_DATA_SIZE = 8192
    TEST_DATA_SIZE = 8192
    OUTPUT_FOLDER = "my_models/"

    # check for incorrect values
    if model_type != 1 and model_type != 2:
        print("Incorrect type")
        sys.exit(2)
    if model_type == 2 and not len(layers):
        print("Type 2 must have at least one layer")
        sys.exit(2)
    if loss_name != "" and not (loss_name == "cosine" or loss_name == "contrastive"):
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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # convert type to model
    if model_type == 1:
        model = SentenceTransformer(model_name, device=device)
    elif model_type == 2:
        word_embedding_model = models.Transformer(model_name, max_seq_length=256)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

        denses_layers = [
            models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=layers[0], activation_function=nn.Tanh())
        ]
        for layer_output in layers[1:]:
            denses_layers.append(models.Dense(in_features=denses_layers[-1].get_sentence_embedding_dimension(), out_features=layer_output, activation_function=nn.Tanh()))

        model = SentenceTransformer(modules=[word_embedding_model, pooling_model]+denses_layers, device=device)

    # convert loss_name to loss
    if loss_name == "" or loss_name == "cosine":
        train_loss = losses.CosineSimilarityLoss(model)
    elif loss_name == "contrastive":
        train_loss = losses.ContrastiveLoss(model)

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
        train_data.append(InputExample(texts=[sentences[pairs[i][0]], sentences[pairs[i][1]]], label=similarities_int[i]))
        test_data.append((sentences[pairs[i][0]], sentences[pairs[i][1]], similarities_int[i]))
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
    model_type, loss, layers, epochs, output, embedding_name = parse_args_trainer(sys.argv)

    train("databases/response_1000.json", embedding_name, output, model_type, loss, layers, epochs)
