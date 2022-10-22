import os
import tensorflow as tf

from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import os.path

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

    print(results.shape)
    print(similarities.shape)

    validation_split = 0.3


    model = tf.keras.Sequential([
      layers.InputLayer(input_shape=results.shape[0]),
      layers.Dense(4),
      layers.Dense(1)])
      
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                  metrics=tf.keras.metrics.MeanSquaredError())

    model.summary()
    
    epochs = 10
    batch_size = 128
    history = model.fit(
        x= results.T,
        y= similarities,
        validation_split = validation_split,
        epochs=epochs,
        batch_size=batch_size,
        shuffle = True)

    for i in range(len(model.layers)):
        print(model.layers[i].get_weights())
        
    loss, accuracy = model.evaluate(
        x= results.T,
        y= similarities,
        batch_size=batch_size
    )
        
    model.save("my_model")
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)


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