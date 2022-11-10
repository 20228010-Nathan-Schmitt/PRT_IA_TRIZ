import os
import numpy as np
import os.path
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from torch import nn
import random

from compare_tools import load_local_database
from sentence_transformers import InputExample, losses, SentenceTransformer, models
from sentence_transformers.evaluation import SimilarityFunction,EmbeddingSimilarityEvaluator

def printEval(score, epoch, step):
    print("Epoch {} - Step {} : {}".format(epoch, step,score))

def makePairsToCompare3(dataset):
    ids = dataset["id"]
    
    pairs = []
    similarities=[]
    
    for i in range(len(ids)):
        for j in range(len(ids)):
            if j>i:break
            #if len(pairs)>=5000:break
            pairs.append((i, j))

            patent1_triz_f = set(dataset["F_TRIZ_PARAMS"][i])
            patent1_triz_s = set(dataset["S_TRIZ_PARAMS"][i])
            patent2_triz_f = set(dataset["F_TRIZ_PARAMS"][j])
            patent2_triz_s = set(dataset["S_TRIZ_PARAMS"][j])
            size_intersection_f = len(list(patent1_triz_f.intersection(patent2_triz_f)))
            size_intersection_s = len(list(patent1_triz_s.intersection(patent2_triz_s)))

            similarities.append(int(size_intersection_f and size_intersection_s))
            #similarities.append((size_intersection_f+ size_intersection_s) / (max(len(patent1_triz_f), len(patent2_triz_f))+ max(len(patent1_triz_s), len(patent2_triz_s))))

    return np.array(pairs), np.array(similarities).astype("f4"), ids


def train(sentences, pairs, similarities,model_name,  output_model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device="cpu"

    train_data = []
    sentences1 = []
    sentences2 = []
    for i in range(len(pairs)):
        train_data.append(InputExample(texts=[sentences[pairs[i][0]],sentences[pairs[i][1]]], label=similarities[i]))
        sentences1.append(sentences[pairs[i][0]])
        sentences2.append(sentences[pairs[i][1]])
    random.shuffle(train_data)
    train_data = train_data[:128]
    sentences1 = sentences1[:8192]
    sentences2 = sentences2[:8192]
    similarities_trunc = similarities[:8192]
    
    print(np.average(similarities_trunc))


    batch_size=32
    epochs = 10

    kwargs = {'num_workers': 0, 'pin_memory': False} if device=='cuda' else {}
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)

    print("start model")
    model = SentenceTransformer(model_name, device=device)
    
    """word_embedding_model = models.Transformer(model_name, max_seq_length=512)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=40, activation_function=nn.Sigmoid())

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])"""

    #Define your train dataset, the dataloader and the train loss
    train_loss = losses.ContrastiveLoss(model)
    #train_loss = losses.CosineSimilarityLoss(model)
    evaluator = EmbeddingSimilarityEvaluator(sentences1=sentences1, sentences2=sentences2, scores=similarities_trunc, name="eval.csv", show_progress_bar=True, main_similarity=SimilarityFunction.COSINE, batch_size=128)


    print("start tune")
    #Tune the model
    warmup_steps = int(len(dataloader) * epochs * 0.1)
    model.fit(train_objectives=[(dataloader, train_loss)],evaluator=evaluator, epochs=epochs, warmup_steps=warmup_steps, evaluation_steps=10000,output_path=output_model, use_amp=True, callback=printEval)


print(os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
dataset = load_local_database()
ids = dataset["id"]
pairs, similarities, ids = makePairsToCompare3(dataset)


print(len(pairs))
print(similarities.size)


train(dataset["sentence"], pairs, similarities,'sentence-transformers/all-mpnet-base-v2',"./my_model_mpnet_2")

print("end", __name__)