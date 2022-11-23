import os
import numpy as np
import os.path
import torch
from torch.utils.data import DataLoader
from torch import nn
import random

from compare_tools import load_local_database
from sentence_transformers import InputExample, losses, SentenceTransformer, models
from sentence_transformers.evaluation import SimilarityFunction, EmbeddingSimilarityEvaluator,BinaryClassificationEvaluator


def printEval(score, epoch, step):
    print("Epoch {} - Step {} : {}".format(epoch, step, score))


def makePairsToCompare3(dataset):
    ids = dataset["id"]

    pairs = []
    similarities_int = []
    similarities_float = []

    for i in range(len(ids)):
        for j in range(len(ids)):
            if j > i:
                break
            # if len(pairs)>=5000:break
            pairs.append((i, j))

            patent1_triz_f = set(dataset["F_TRIZ_PARAMS"][i])
            patent1_triz_s = set(dataset["S_TRIZ_PARAMS"][i])
            patent2_triz_f = set(dataset["F_TRIZ_PARAMS"][j])
            patent2_triz_s = set(dataset["S_TRIZ_PARAMS"][j])
            size_intersection_f = len(
                list(patent1_triz_f.intersection(patent2_triz_f)))
            size_intersection_s = len(
                list(patent1_triz_s.intersection(patent2_triz_s)))

            similarities_int.append(int(not (not size_intersection_f or not size_intersection_s)))
            similarities_float.append((size_intersection_f+ size_intersection_s) / (max(len(patent1_triz_f), len(patent2_triz_f))+ max(len(patent1_triz_s), len(patent2_triz_s))))

    return np.array(pairs), np.array(similarities_int),np.array(similarities_float).astype("f4"), ids


def train(sentences, pairs, similarities_int,similarities_float, model_name,  output_model, combinaison = 0, continue_training=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device="cpu"
    combinaison_model = combinaison%10
    if continue_training:
        model = SentenceTransformer(model_name+"_"+str(combinaison), device=device)
    elif combinaison_model==0:
        model = SentenceTransformer(model_name, device=device)
    elif combinaison_model==1:
        word_embedding_model = models.Transformer(model_name, max_seq_length=256)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(
        ), out_features=1, activation_function=nn.Tanh())
        model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model, dense_model], device=device)
    elif combinaison_model==2:
        word_embedding_model = models.Transformer(model_name, max_seq_length=256)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(
        ), out_features=40, activation_function=nn.Tanh())
        model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model, dense_model], device=device)
    elif combinaison_model==3:
        word_embedding_model = models.Transformer(model_name, max_seq_length=256)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(
        ), out_features=256, activation_function=nn.Tanh())
        dense_model_2 = models.Dense(in_features=dense_model.get_sentence_embedding_dimension(
        ), out_features=40, activation_function=nn.Tanh())
        model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model, dense_model,dense_model_2], device=device)
   
    combinaison_loss = combinaison//100 %10
    if combinaison_loss==0:
        train_loss = losses.ContrastiveLoss(model)
    elif combinaison_loss==1:
        train_loss = losses.OnlineContrastiveLoss(model)
    elif combinaison_loss==2:
        train_loss = losses.CosineSimilarityLoss(model)
    elif combinaison_loss==3:
        train_loss = losses.SoftmaxLoss(model,sentence_embedding_dimension=model.get_sentence_embedding_dimension(),num_labels=2)

    train_data = []
    sentences1 = []
    sentences2 = []
    test_data = []
    for i in range(len(pairs)):
        if combinaison_loss==2:
            train_data.append(InputExample(texts=[sentences[pairs[i][0]], sentences[pairs[i][1]]], label=similarities_float[i]))
            test_data.append((sentences[pairs[i][0]],sentences[pairs[i][1]], similarities_float[i]))
        else:
            train_data.append(InputExample(texts=[sentences[pairs[i][0]], sentences[pairs[i][1]]], label=similarities_int[i]))
            test_data.append((sentences[pairs[i][0]],sentences[pairs[i][1]], similarities_int[i]))
    random.shuffle(train_data)
    random.shuffle(test_data)

    training_data_size = 8192
    test_data_size = 8#192

    sentences1,sentences2,similarities_trunc = zip(*test_data[:test_data_size])


    batch_size = 4
    epochs = 5

    kwargs = {'num_workers': 0, 'pin_memory': False} if device == 'cuda' else {}
    dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, **kwargs)

    print("start model", model_name, " - ", combinaison)

    combinaison_evaluator = combinaison//10 %10
    if combinaison_evaluator==0:
        evaluator = EmbeddingSimilarityEvaluator(sentences1=sentences1, sentences2=sentences2, scores=similarities_trunc,  name="eval.csv", show_progress_bar=True, main_similarity=SimilarityFunction.COSINE, batch_size=128)
    elif combinaison_evaluator==1:
        evaluator = BinaryClassificationEvaluator(sentences1, sentences2, similarities_trunc, "eval2.csv",batch_size=128,show_progress_bar=True)



    print("start tune")

    # Tune the model
    warmup_steps = 0  # int(len(dataloader) * epochs * 0.1)
    print("init score : ",model.evaluate(evaluator))
    model.fit(train_objectives=[(dataloader, train_loss)], evaluator=evaluator, epochs=epochs,
              warmup_steps=warmup_steps, evaluation_steps=10000, output_path=output_model+"_"+str(combinaison), use_amp=True, callback=printEval, steps_per_epoch=training_data_size//batch_size, scheduler="constantlr")
    print("\n\n")


dataset = load_local_database()
ids = dataset["id"]
pairs, similarities_int, similarities_float, ids = makePairsToCompare3(dataset)


#train(dataset["sentence"], pairs, similarities_int,similarities_float,'sentence-transformers/all-mpnet-base-v2', "./my_models/my_model_mpnet",200 )
#train(dataset["sentence"], pairs, similarities_int,similarities_float,'sentence-transformers/all-mpnet-base-v2', "./my_models/my_model_mpnet",201 )
####train(dataset["sentence"], pairs, similarities_int,similarities_float, 'sentence-transformers/all-mpnet-base-v2', "./my_models/my_model_mpnet",202 )
#train(dataset["sentence"], pairs, similarities_int,similarities_float,'sentence-transformers/all-mpnet-base-v2', "./my_models/my_model_mpnet",203 )
#train(dataset["sentence"], pairs, similarities_int,similarities_float,'sentence-transformers/all-mpnet-base-v2', "./my_models/my_model_mpnet",300 )
train(dataset["sentence"], pairs, similarities_int,similarities_float,'sentence-transformers/all-mpnet-base-v2', "./my_models/my_model_mpnet",301 )
train(dataset["sentence"], pairs, similarities_int,similarities_float,'sentence-transformers/all-mpnet-base-v2', "./my_models/my_model_mpnet",302 )
train(dataset["sentence"], pairs, similarities_int,similarities_float,'sentence-transformers/all-mpnet-base-v2', "./my_models/my_model_mpnet",303 )
train(dataset["sentence"], pairs, similarities_int,similarities_float,'sentence-transformers/all-mpnet-base-v2', "./my_models/my_model_mpnet",000)
train(dataset["sentence"], pairs, similarities_int,similarities_float,'sentence-transformers/all-mpnet-base-v2', "./my_models/my_model_mpnet",1 )
train(dataset["sentence"], pairs, similarities_int,similarities_float,'sentence-transformers/all-mpnet-base-v2', "./my_models/my_model_mpnet",2 )
train(dataset["sentence"], pairs, similarities_int,similarities_float,'sentence-transformers/all-mpnet-base-v2', "./my_models/my_model_mpnet",3 )

print("end", __name__)
