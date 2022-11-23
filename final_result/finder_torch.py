import numpy as np
import os.path
import os

from embeddings.embeddings import embeddings
from modeles.modeles import models
import torch

def embed(sentence, embedding_to_test):
    sentence_emb = []
    for embedding in embedding_to_test:
        print(embedding)
        sentence_emb.append(embeddings[embedding](sentence, once=True))
    return sentence_emb
    
def load_database_embed(embedding_to_test):
    sentences_emb=[]
    ids = []

    filename_ids = "save/ids.npy"
    if os.path.isfile(filename_ids):
        ids  = np.load(filename_ids)
    else:
        print("Id file not found")
        0/0

    for embedding in embedding_to_test:
        filename = "save/embedding_"+embedding+".npy"
        if os.path.isfile(filename):
            current_embedding  = np.load(filename)
            database_sentences_emb = current_embedding.astype(float)
            sentences_emb.append(database_sentences_emb)
        else:
            print("Embedding not found : ", embedding)
            0/0
    return sentences_emb,ids


#sentence_to_compare = "Batteries need to be bigger but it will be heavier"
#sentence_to_compare = "Big wheels are better for comfort but it will be harder to push."
sentence_to_compare = "In addition, the remote control is a high-performance device because it takes time to perform operations such as image processing. Therefore, when using multiple portable information terminals, the number of console units increases with the number of portable information terminals, which reduces portability."
#sentence_to_compare = "A high temperature is needed for the chemical reaction but it can damage the environnement"
#sentence_to_compare = "The dimensions of trench power MOSFETs metal-oxide-semiconductor field-effect transistor may be reduced for improving the electrical performance and decreasing the costs from generation to generation, which may be enabled both through better lithography systems and more powerful tools with an improved process control. While the field plate resistance may be rather uncritical due to its direct connection to the source metal, the gate resistance may provide difficulties as the gate trench is arranged between the columns of the field plate electrode."



embedding_to_test = ["custom_mpnet_ultime", "mpnet_base"]

database_emb,id_ = load_database_embed(embedding_to_test)
sentence_emb = embed(sentence_to_compare, embedding_to_test)


results=[]
for i in range(len(embedding_to_test)):  
    pairs_emb = []
    for database_sentence in database_emb[i]:
        pairs_emb.append([sentence_emb[i], database_sentence])
    pairs_emb = np.array(pairs_emb)
    for model in models:
        results.append(models[model](pairs_emb))
results = np.array(results).T
results_tensor = torch.from_numpy(results).float().to("cuda")

model = torch.load('my_models/my_model_mpnet_and_mpnet_custom').to("cuda")
print(model)
prediction  = model(results_tensor).detach().cpu().numpy()

print(prediction)

number_to_keep=10
prediction = prediction[:,0]
ind = np.argpartition(prediction, -number_to_keep)[-number_to_keep:]
ind = ind[np.argsort(prediction[ind])]
for index in ind:
    print(index, prediction[index], id_[index], results[index,0], results[index,1], results[index,2], results[index,3])


"""print(prediction)

imax = np.argmax(prediction)
print(imax)
print(prediction[imax])
print(id_[imax])"""