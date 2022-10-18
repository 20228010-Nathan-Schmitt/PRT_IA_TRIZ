def load_sentences():
    f=open("dataset.txt", "r", encoding="utf8")
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
    import numpy as np
    similarities=np.array([])
    for patentNumber in sentences:
        patent = sentences[patentNumber]["patent"]
        for patentNumber2 in sentences:
            for short in sentences[patentNumber2]["short"]:
                pairs.append((patent, short))
                similarities = np.insert(similarities, similarities.size, patentNumber==patentNumber2)
    return pairs, similarities

def transposeList(pairs):
    import numpy as np
    return np.swapaxes(pairs, 0,1)