import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
import tensorflow as tf
import numpy as np
import os.path

from embeddings.embeddings import embeddings
from modeles.modeles import models
from compare_tools import load_local_database, makePairsToCompare2


def get_result(embedding, model, sentences, dico_des_ids, pairs):

    

    filename = "training/embedding2_"+embedding+".npy"
    if os.path.isfile(filename):
        sentences_emb = np.load(filename)
    else:
        sentences_emb = embeddings[embedding](sentences)
        np.save(filename, sentences_emb)
        
    
    
    print("start compute")
    
    filename_result = "training/result2_"+embedding+"_"+model+".npy"
    if os.path.isfile(filename_result):
        results = np.load(filename_result)
    else:
        pairs_emb = np.empty((0,2,sentences_emb.shape[1],))
    
        for a,b in pairs:
            pairs_emb = np.append(pairs_emb, [[sentences_emb[dico_des_ids[a]], sentences_emb[dico_des_ids[b]]]], axis=0)
        results = models[model](pairs_emb)
        np.save(filename_result, results)

    return results


def classify(sentences,similarities,results):

    print(results.shape)
    print(similarities.shape)
    dataset_np = np.vstack([results, similarities]).T

    training_split = 0.8
    training_size = int(training_split * len(sentences))
    validation_size = int((1-training_split) * len(sentences))

    dataset_full = tf.data.Dataset.from_tensor_slices(dataset_np).shuffle(10)
    dataset_train = dataset_full.take(training_size)
    dataset_val = dataset_full.skip(training_size).take(validation_size)

    model = tf.keras.Sequential([
      layers.InputLayer(input_shape=results.shape[0]),
      layers.Dense(3),
      layers.Dense(1)])
      
    model.compile(loss='mean_squared_error',
                  metrics=tf.keras.metrics.MeanSquaredError())

    model.summary()
    
    epochs = 100
    history = model.fit(
        x= results.T,
        y= similarities,
        validation_split = 0.3,
        epochs=epochs,
        shuffle = True,
        verbose=2)

    for i in range(len(model.layers)):
        print(model.layers[i].get_weights())
        
        
    loss, accuracy = model.evaluate(
        x= results.T,
        y= similarities
    )
    
    print(model.predict(results.T))
    
    model.save("my_model")
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)


#dataset = load_training_dataset()
dataset = load_local_database()
ids = dataset["id"]
pairs, similarities = makePairsToCompare2()

print(len(pairs))
print(similarities.size)
sentences = []
dico_des_ids={}
for i in range(len(ids)):
    sentences.append(dataset["sentence"][i])
    dico_des_ids[ids[i]]=i

print(sentences)

results = np.empty((0,len(pairs)))
for embedding in embeddings:
    print(embedding)    
    for model in models:
        print(" └",model)
        results= np.vstack((results, get_result(embedding, model, sentences, dico_des_ids, pairs)))
        

classify(sentences,similarities,results)