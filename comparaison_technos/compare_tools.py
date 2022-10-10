def load_sentences():
    f=open("tools/dataset.txt", "r", encoding="utf8")
    lines = [line.rstrip('\n') for line in f]
    f.close()

    sentences = {}    
    for line in lines:
        sentences_raw = line.split(" ")

        if len(sentences_raw)<3:continue
        patentNumber = sentences_raw[0]

        if not patentNumber in sentences: sentences[patentNumber]={"patent":"", "short":[]}
        if sentences_raw[1]=="F" or sentences_raw[1]=="S":
            sentences[patentNumber]["patent"] += " ".join(sentences_raw[2:])+" "
        if sentences_raw[1]=="R":
            sentences[patentNumber]["short"].append(" ".join(sentences_raw[2:]))
    return sentences

def makePairsToCompare(sentences):
    pairs = []

    for patentNumber in sentences:
        patent = sentences[patentNumber]["patent"]
        for short in sentences[patentNumber]["short"]:
            pairs.append((patent, short))
    return pairs

def transposeList(pairs):
    import numpy as np
    return np.array(pairs).T.tolist()