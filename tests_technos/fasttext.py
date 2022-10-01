import io
import scipy as sp

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

fasttext = load_vectors("wiki-news-300d-1M.vec")

seat = list(fasttext["Uranus"])
chair = list(fasttext["Neptune"])
print(seat)
print(chair)
print()
print(sp.spatial.distance.cosine(seat, chair))
