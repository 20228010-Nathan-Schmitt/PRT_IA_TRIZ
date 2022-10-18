import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses

from compare_tools import load_sentences, makePairsToCompare


import embeddings.simCSE as simCSE
import embeddings.bert_for_patent as bert_for_patent
import embeddings.mpnet_base as mpnet_base
import modeles.cosine as cosine
import modeles.euclide as euclide
import cross_encoders.WMD as WMD
import cross_encoders.nli_deberta as nli_deberta
from compare_tools import load_sentences, makePairsToCompare
import tensorflow as tf
import numpy as np
import os.path


embeddings = ["simCSE", "mpnet_base"]
models = {"cosine": cosine.cos_sim, "euclide": euclide.euc_dist}
def get_result(embedding, model):
    print(embedding)
    print(model)

    filename = "save/embedding_"+embedding+".npy"
    if os.path.isfile(filename):
        sentences_emb = np.load(filename)
    else:
        print("Embedding not found : ", embedding)
        a=0/0
    results = models[model](sentences_emb)
    return results


def classify(sentences,similarities,results):
    dataset_np = np.vstack([results, similarities]).T
    print(dataset_np)

    training_split = 0.8
    training_size = int(training_split * len(sentences))
    validation_size = int((1-training_split) * len(sentences))

    dataset_full = tf.data.Dataset.from_tensor_slices(dataset_np).shuffle(10)
    dataset_train = dataset_full.take(training_size)
    dataset_val = dataset_full.skip(training_size).take(validation_size)

        
    model = tf.keras.Sequential([
      layers.InputLayer(input_shape=4),
      layers.Dense(10),
      layers.Dense(1)])
      
    model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=tf.metrics.BinaryAccuracy(threshold=0))

    model.summary()
    
    epochs = 1000
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
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)


sentences,similarities = makePairsToCompare(load_sentences())
results = get_result(embeddings[0], "cosine")
results= np.vstack((results, get_result(embeddings[0], "euclide")))
results= np.vstack((results, get_result(embeddings[1], "cosine")))
results= np.vstack((results, get_result(embeddings[1], "euclide")))
classify(sentences,similarities,results)