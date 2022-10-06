import embeddings.simCSE as simCSE
import modeles.cosine as cosine
from compare_tools import load_sentences, makePairsToCompare

embeddings = {"simCSE":simCSE.embeddings_simcse}
models = {"cosine":cosine.cos_sim}

sentences =makePairsToCompare(load_sentences())
for embedding in embeddings:
    sentences_emb = embeddings[embedding](sentences)
    for model in models:
        results = models[model](sentences_emb)
        for i in range(len(sentences)):
            print("{:.40}  &  {:.40} =>  {}".format(sentences[i][0], sentences[i][1], results[i]))
