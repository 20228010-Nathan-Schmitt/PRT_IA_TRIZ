import tensorflow as tf
import numpy as np
import os.path
import os

from embeddings.embeddings import embeddings, embed_size
from modeles.modeles import models
from compare_tools import load_database,remove_field_num
from tensorflow import keras

def embed(sentence):
    sentence_emb = []
    for embedding in embeddings:
        print(embedding)
        sentence_emb.append(embeddings[embedding](sentence, once=True))
    return sentence_emb
    
def load_database_embed():
    sentences_emb=[]
    ids = []

    filename_ids = "save/ids.npy"
    if os.path.isfile(filename_ids):
        ids  = np.load(filename_ids)
    else:
        print("Id file not found")
        0/0

    for embedding in embeddings:
        filename = "save/embedding_"+embedding+".npy"
        if os.path.isfile(filename):
            current_embedding  = np.load(filename)
            database_sentences_emb = current_embedding.astype(float)
            sentences_emb.append(database_sentences_emb)
        else:
            print("Embedding not found : ", embedding)
            0/0
    return sentences_emb,ids


sentence_to_compare = "Batteries need to be bigger but it will be heavier"
sentence_to_compare = "Big wheels are better for comfort but it will be harder to push."
#sentence_to_compare = "A high temperature is needed for the chemical reaction but it can damage the environnement"
#sentence_to_compare = "The dimensions of trench power MOSFETs metal-oxide-semiconductor field-effect transistor may be reduced for improving the electrical performance and decreasing the costs from generation to generation, which may be enabled both through better lithography systems and more powerful tools with an improved process control. While the field plate resistance may be rather uncritical due to its direct connection to the source metal, the gate resistance may provide difficulties as the gate trench is arranged between the columns of the field plate electrode."



database_emb,id_ = load_database_embed()
sentence_emb = embed(sentence_to_compare)


results=[]
for i in range(len(embeddings)):  
    pairs_emb = []
    for database_sentence in database_emb[i]:
        pairs_emb.append([sentence_emb[i], database_sentence])
    pairs_emb = np.array(pairs_emb)
    for model in models:
        results.append(models[model](pairs_emb))
results = np.array(results).T


model = keras.models.load_model('my_model')
model.summary()
prediction  = model.predict(results)

number_to_keep=10
prediction = prediction[:,0]
ind = np.argpartition(prediction, -number_to_keep)[-number_to_keep:]
ind = ind[np.argsort(prediction[ind])]
for index in ind:
    print(index, prediction[index], id_[index])


"""print(prediction)

imax = np.argmax(prediction)
print(imax)
print(prediction[imax])
print(id_[imax])"""