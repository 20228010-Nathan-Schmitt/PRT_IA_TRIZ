import embeddings.simCSE as simCSE
import embeddings.bert_for_patent as bert_for_patent
import modeles.cosine as cosine
import modeles.euclide as euclide
from compare_tools import load_sentences, makePairsToCompare
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    except RuntimeError as e:
        print(e)

embeddings = {"simCSE": simCSE.embeddings_simcse, "Bert For Patent": bert_for_patent.embeddings_bert_patent}
models = {"cosine": cosine.cos_sim, "euclide": euclide.euc_dist}

sentences = makePairsToCompare(load_sentences())
for embedding in embeddings:
    sentences_emb = embeddings[embedding](sentences)
    for model in models:
        print(model)
        results = models[model](sentences_emb)
        for i in range(len(sentences)):
            print("{:.40}  &  {:.40} =>  {}".format(sentences[i][0], sentences[i][1], results[i]))
"""

sentences_emb = bert_for_patent.embeddings_bert_patent(sentences[:3])
print(sentences_emb)
for model in models:
    results = models[model](sentences_emb)

    print(model)
    print(results)
    for i in range(len(sentences[:3])):
        print(results[i])
        print("{:.40}  &  {:.40} =>  {}".format(sentences[i][0], sentences[i][1], results[i]))"""
