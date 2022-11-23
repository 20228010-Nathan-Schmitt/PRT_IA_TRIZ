import os
import numpy as np
import os.path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from embeddings.embeddings import embeddings
from modeles.modeles import models
from compare_tools import load_local_database, makePairsToCompare2


def get_result(embedding, model, sentences, pairs,ids):
    filename = "training/embedding2_"+embedding+".npy"
    if os.path.isfile(filename):
        sentences_emb = np.load(filename)
    else:
        sentences_emb = embeddings[embedding](sentences, once=True)
        np.save(filename, sentences_emb)
        
    print(sentences_emb[0][:6])
    print(sentences_emb[1][:6])
    print("start compute")
    
    filename_result = "training/result2_"+embedding+"_"+model+".npy"
    if os.path.isfile(filename_result):
        results = np.load(filename_result)
    else:
        f = lambda pair: [sentences_emb[pair[0]], sentences_emb[pair[1]]]
        pairs_emb = np.array(list(map(f, pairs)))
        print("starting ", model)
        results = models[model](pairs_emb)
        np.save(filename_result, results)
    print(results[1])
    return results


def classify(similarities,results):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = torch.from_numpy(results.T).float().to(device)
    similarities = torch.from_numpy(np.expand_dims(similarities, axis=1)).float().to(device)

    n_input, n_hidden, n_hidden2, n_out, batch_size, learning_rate, epochs = results.shape[1], 4,4, 1, 128, 0.01,4

    dataset = TensorDataset(results, similarities)
    kwargs = {'num_workers': 0, 'pin_memory': False} if device=='cuda' else {}
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)

    model = nn.Sequential(
        nn.Linear(n_input, n_hidden),
        nn.Sigmoid(),
        nn.Linear(n_hidden, n_hidden2),
        nn.Sigmoid(),
        nn.Linear(n_hidden2, n_out),
        nn.Sigmoid())
    model.to(device)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    pred_y = model(results)
    loss = loss_function(pred_y, similarities)
    print("Loss : ",loss.item())
    losses.append(loss.item())

    for epoch in range(epochs):
        print("Epoch ",epoch)
        for inputs, outputs in dataloader:
            pred_y = model(inputs)
            loss = loss_function(pred_y, outputs)
            model.zero_grad()
            loss.backward()

            optimizer.step()
        pred_y = model(results)
        loss = loss_function(pred_y, similarities)
        print("Loss : ",loss.item())
        losses.append(loss.item())
    torch.save(model, "my_models/my_model_mpnet_and_mpnet_custom")
    print(results)
    print(results.shape)
    print(model(results))
    print(similarities)
    print(similarities[:10])
        
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.gca().set_ylim(bottom=0)
    plt.title("Learning rate %f"%(learning_rate))
    plt.show()

print("yo")
print(__name__ )
dataset = load_local_database()
ids = dataset["id"]
pairs, similarities, ids = makePairsToCompare2(dataset)


print(len(pairs))
print(similarities.size)
sentences = []
for i in range(len(ids)):
    sentences.append(dataset["sentence"][i])

embedding_to_test = ["custom_mpnet_ultime", "mpnet_base"]

results = np.empty((0,len(pairs)))
for embedding in embedding_to_test:
    print(embedding)    
    for model in models:
        print(" └",model)
        results= np.vstack((results, get_result(embedding, model, sentences, pairs,ids)))
        
print("new ", __name__, results.shape)


if __name__=="__main__":
    classify(similarities, results)

print("end", __name__)