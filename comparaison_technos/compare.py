import embeddings.simCSE as simCSE
import embeddings.bert_for_patent as bert_for_patent
import modeles.cosine as cosine
import modeles.euclide as euclide
from compare_tools import load_sentences, makePairsToCompare

embeddings = {"simCSE":simCSE.embeddings_simcse, "Bert For Patent": bert_for_patent.embeddings_bert_patent}
models = {"cosine":cosine.cos_sim, "euclide":euclide.euc_dist}

sentences =makePairsToCompare(load_sentences())
"""for embedding in embeddings:
    sentences_emb = embeddings[embedding](sentences)
    for model in models:
        results = models[model](sentences_emb)
        for i in range(len(sentences)):
            print("{:.40}  &  {:.40} =>  {}".format(sentences[i][0], sentences[i][1], results[i]))
"""

sentences_emb = bert_for_patent.embeddings_bert_patent(sentences[:2])
for model in models:
    results = models[model](sentences_emb)
    for i in range(len(sentences[:2])):
        print("{:.40}  &  {:.40} =>  {}".format(sentences[i][0], sentences[i][1], results[i]))
