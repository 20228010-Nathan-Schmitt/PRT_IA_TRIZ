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


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    except RuntimeError as e:
        print(e)

#embeddings = {"simCSE": simCSE.embeddings_simcse, "Bert For Patent": bert_for_patent.embeddings_bert_patent, "mpnet_base":mpnet_base.embeddings_mpnet_base}
embeddings = {"simCSE": simCSE.embeddings_simcse, "mpnet_base":mpnet_base.embeddings_mpnet_base}
models = {"cosine": cosine.cos_sim, "euclide": euclide.euc_dist}
cross_encoders={"WMD":WMD.wmd_dist, "nli_deberta":nli_deberta.dist_nli_deberta}


sentences,similarities = makePairsToCompare(load_sentences())
print("\n","="*50,"\n", "="*50, "\n", "="*50,"\n", sep="")

for cross_encoder in cross_encoders:
    filename = "save/cross_encoder_"+cross_encoder+".npy"
    if os.path.isfile(filename):
        results = np.load(filename)
    else:
        results = cross_encoders[cross_encoder](sentences)
        np.save(filename, results)
    print(cross_encoder)
    error = results-similarities
    error_positive = 0
    error_negative = 0
    for i in range(len(sentences)):
        if similarities[i]: 
            error_positive += error[i]**2
        else: 
            error_negative += error[i]**2
    print("Errors : ", error_positive, "\t", error_negative)
    """for i in range(len(sentences)):
        print("{:.40}  &  {:.40} =>  {}".format(sentences[i][0], sentences[i][1], results[i]))"""

print("\n","="*50,"\n", "="*50, "\n", "="*50,"\n", sep="")

for embedding in embeddings:
    filename = "save/embedding_"+embedding+".npy"
    if os.path.isfile(filename):
        sentences_emb = np.load(filename)
    else:
        sentences_emb = embeddings[embedding](sentences)
        np.save(filename, sentences_emb)
    print(embedding)
    for model in models:
        print(model)
        results = models[model](sentences_emb)
        
        error = results-similarities
        error_positive = 0
        error_negative = 0
        for i in range(len(sentences)):
            if similarities[i]: 
                error_positive += error[i]**2
            else: 
                error_negative += error[i]**2
        print("Errors : ", error_positive, "\t", error_negative)

        """for i in range(len(sentences)):
            print("{:.40}  &  {:.40} =>  {}".format(sentences[i][0], sentences[i][1], results[i]))"""
        print()
    print("="*50, "\n")