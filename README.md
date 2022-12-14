# PRT_IA_TRIZ

## Mise en place et requirements
  
  Le projet est prévu pour etre utilisé via pycharm (certains IDE utilisent des chemins différents pour les fichiers référencés).
  
  Il est nécessaire d'installer les package du fichier requirements.txt : une fois dans pycharm, il suffit de faire click droit -> installer tous les packages.
  
  Packages à installer : numpy, scipy, scikit-learn, tensorflow, sentence_transformers, gensin, nltk, pyemd (https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyemd)


## Installation en L1.12

Dans le cmd

cd C:\Users\nschmitt01\Downloads\PortableGit\PRT_IA_TRIZ\final_result
python.exe -m venv C:\Users\nschmitt01\Downloads\PortableGit\PRT_IA_TRIZ\final_result\myvenv
"myvenv\Scripts\activate.bat"
pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu117
pip3 install numpy sentence_transformers

## Utilisation de `final_final_result`
Tout fonctionne avec `pytorch` et `sentence_transformers`

3 types de réseaux
 1. Embedding + pooling + distance
 2. Embedding + pooling + dense Layer + distance
 3. Embeddings + pooling + distances + classifier (plusieurs embeddings en parallèle)

### Entrainement
 1. trainer.py --type 1 --loss loss_name --epochs 10 --output output_name embedding_name
 2. trainer.py --type 2 --loss loss_name --layers 256_40 --epochs 10 --output output_name embedding_name
 3. 3 étapes
    1. Pour les embeddings : trainer.py --type 1 --loss loss_name --epochs 10 --output output_name embedding_name
    2. Calcule de chaque [embedding](#Embedding)
    3. Pour le classifier : trainer_type3.py --epochs 10 --layers 256_40_3 --output output_name embedding1_name [embedding2_name] [embedding1_name] ... 

### Embedding
embedder.py embedding1_name [embedding2_name] [embedding1_name]

### Test de performance
 1. test_reformulation.py --type 1  embedding_name
 2. test_reformulation.py --type 2 embedding_name
 3. test_reformulation.py --type 3 model_name

### Recherche dans la bdd
 1. finder.py --type 1 embedding_name
 2. finder.py --type 2 embedding_name
 3. finder.py --type 3 model_name

## Useful tutorials
- [x] Text Similarities (Medium) : https://medium.com/@adriensieg/text-similarities-da019229c894
- [x] Semantic Textual Similarity (Towards Data Science) : https://towardsdatascience.com/semantic-textual-similarity-83b3ca4a840e
- [x] Difference between BERT and following : https://zilliz.com/learn/7-nlp-models

## Stemming/lemmatisation

Le stemming permet de convertir un mot en sa version la plus simple, sa racine : better -> good

over-stemming : regrouper sur la meme racien des mots qui ont des sens différents.

under-stemming : ne pas regrouper sur la meme racine des mots qui ont le meme sens.

La racine peut ne pas etre un mot valide. Si on se limite à des mots valides, on parle de lemmisation.


On peut retenir quelques algorithmes :
- krovetz stemmer : convertit les noms vers leur forme au singulier, les verbes vers leur présent simple. il est plutot considéré comme un pré-stemmer
- porter stemmer : convertit les mots vers leur racine, parfois la racine atteinte n'est pas un mot réel. est tout de meme considéré comme un des meilleurs algorithmes
- snowball stemming : parfois appelé porter2 stemmer, il est juste mieux et multilingue. importatble par le package nltk.

tutos youtube:
- https://www.youtube.com/watch?v=HHAilAC3cXw
- https://www.youtube.com/watch?v=tmY-G6sngk8
- https://www.youtube.com/watch?v=p1ccbR2P_xA

Articles :
- https://www.geeksforgeeks.org/introduction-to-stemming/
- https://en.wikipedia.org/wiki/Lemmatisation
- https://en.wikipedia.org/wiki/Stemming
  
## Useful techno
### Gensim
- [x] https://pypi.org/project/gensim/
- [x] https://radimrehurek.com/gensim/auto_examples/

### BERT
- [x] https://github.com/google-research/bert
- [x] https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt

### word2vec
- [x] See [Gensim](#gensim)

### fastText
- [x] https://fasttext.cc/docs/en/english-vectors.html

### Sentence Tranformers
- [x] https://www.sbert.net/

### word Mover Distance
- [x] https://radimrehurek.com/gensim/auto_examples/tutorials/run_wmd.html
- [x] https://vene.ro/blog/word-movers-distance-in-python.html
- [x] https://pypi.org/project/word-mover-distance/
