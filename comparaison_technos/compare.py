import embeddings.simCSE as simCSE
import embeddings.bert_for_patent as bert_for_patent
import modeles.cosine as cosine
import modeles.euclide as euclide
import cross_encoders.WMD as WMD
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
cross_encoders={"WMD":WMD.wmd_dist}


sentences = makePairsToCompare(load_sentences())

for cross_encoder in cross_encoders:
    print(cross_encoder)
    results = cross_encoders[cross_encoder](sentences)
    for i in range(len(sentences)):
        print("{:.40}  &  {:.40} =>  {}".format(sentences[i][0], sentences[i][1], results[i]))

for embedding in embeddings:
    sentences_emb = embeddings[embedding](sentences)
    for model in models:
        print(model)
        results = models[model](sentences_emb)
        for i in range(len(sentences)):
            print("{:.40}  &  {:.40} =>  {}".format(sentences[i][0], sentences[i][1], results[i]))