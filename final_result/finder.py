from tensorflow.keras import layers
from tensorflow.keras import losses
import tensorflow as tf
import numpy as np
import os.path

from embeddings.embeddings import embeddings, embed_size
from modeles.modeles import models
from compare_tools import load_database,remove_field_num


def embed(sentence):
    sentence_emb = []
    for embedding in embeddings:
        print(embedding)
        sentence_emb.append(embeddings[embedding](sentence))
    return sentence_emb
    
def load_database_embed():
    sentences_emb=[]
    ids = []
    for embedding in embeddings:
        filename = "save/embedding_"+embedding+".npy"
        if os.path.isfile(filename):
            current_embedding  = np.load(filename)
            
            ids = current_embedding[:,0]
            database_sentences_emb = np.delete(current_embedding, 0, axis=1)
            database_sentences_emb = database_sentences_emb.astype(float)
            sentences_emb.append(database_sentences_emb)
        else:
            print("Embedding not found : ", embedding)
            0/0
    return sentences_emb,ids


sentence_to_compare = "Batteries need to be bigger but it will be heavier"
sentence_to_compare = "The dimensions of trench power MOSFETs metal-oxide-semiconductor field-effect transistor may be reduced for improving the electrical performance and decreasing the costs from generation to generation, which may be enabled both through better lithography systems and more powerful tools with an improved process control."

database_emb,id = load_database_embed()
sentence_emb = embed(sentence_to_compare)


results=[]
for i in range(len(embeddings)):  
    pairs_emb = []
    for database_sentence in database_emb[i]:
        pairs_emb.append((sentence_emb[i], database_sentence))
    for model in models:
        results.append(models[model](pairs_emb))
        print(results)
results = np.array(results).T
   
from tensorflow import keras
model = keras.models.load_model('my_model')
model.summary()

prediction  = model.predict(results)

print(prediction)

imax = np.argmax(prediction)
print(imax)
print(prediction[imax])
print(id[imax])