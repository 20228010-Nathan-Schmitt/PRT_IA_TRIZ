import os
import numpy as np
import os.path
import torch
import torch.nn as nn

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
    return results


def classify(similarities,results):
    results = torch.from_numpy(results.T).float()
    similarities = torch.from_numpy(np.expand_dims(similarities, axis=1)).float()

    n_input, n_hidden, n_out, batch_size, learning_rate = results.shape[1], 4, 1, 128, 0.01
    epochs = 10

    model = nn.Sequential(
        nn.Linear(n_input, n_hidden),
        nn.Sigmoid(),
        nn.Linear(n_hidden, n_out),
        nn.Sigmoid())

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    pred_y = model(results)
    loss = loss_function(pred_y, similarities)
    print("Loss : ",loss.item())
    losses.append(loss.item())
    for epoch in range(epochs):
        print("Epoch ",epoch)
        for i in range(results.shape[0]//batch_size):
            pred_y = model(results[i*batch_size: (i+1)*batch_size])
            loss = loss_function(pred_y, similarities[i*batch_size: (i+1)*batch_size])
            model.zero_grad()
            loss.backward()

            optimizer.step()
        pred_y = model(results)
        loss = loss_function(pred_y, similarities)
        print("Loss : ",loss.item())
        losses.append(loss.item())
    torch.save(model, "my_model")

    """for i in range(len(model.layers)):
        print(model.layers[i].get_weights())
        
    loss, accuracy = model.evaluate(
        x= results.T,
        y= similarities,
        batch_size=batch_size
    )
        
    model.save("my_model")
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)"""
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.gca().set_ylim(bottom=0)
    plt.title("Learning rate %f"%(learning_rate))
    plt.show()


#dataset = load_training_dataset()
dataset = load_local_database()
ids = dataset["id"]
pairs, similarities, ids = makePairsToCompare2(dataset)


print(len(pairs))
print(similarities.size)
sentences = []
for i in range(len(ids)):
    sentences.append(dataset["sentence"][i])

results = np.empty((0,len(pairs)))
for embedding in embeddings:
    print(embedding)    
    for model in models:
        print(" └",model)
        results= np.vstack((results, get_result(embedding, model, sentences, pairs,ids)))
        

classify(similarities, results)