import scipy as sp
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

tfhub_handle_preprocess  ="https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
tfhub_handle_encoder    = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"

f=open("vocab.txt", "r", encoding="utf8")
vocab = [line.rstrip('\n') for line in f]
f.close()
def get_vocab_by_id(id):
    return vocab[id]


def get_embedding(sentences:list[str]):
    bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
    bert_model = hub.KerasLayer(tfhub_handle_encoder)


    text_preprocessed = bert_preprocess_model(sentences)
    bert_results = bert_model(text_preprocessed)

    print(f'Type Ids   : {text_preprocessed["input_word_ids"][:5, :12]}')
    print(f'Type Ids   : {text_preprocessed["input_mask"][:5, :12]}')


    embeddings = (bert_results["encoder_outputs"][11]+bert_results["encoder_outputs"][10]+bert_results["encoder_outputs"][9]+bert_results["encoder_outputs"][8])/4

    wordEmbedMatch=[]
    for j in range(len(sentences)):
        wordEmbedMatch.append([])
        for i in range(text_preprocessed["input_word_ids"].shape[1]):
            if not text_preprocessed["input_mask"][j,i]: break
            id = int(text_preprocessed["input_word_ids"][j,i])

            wordEmbedMatch[j].append([get_vocab_by_id(id), embeddings[j,i].numpy()])
    return wordEmbedMatch

    #print(bert_results["encoder_outputs"][11][0,:12,:10])

text_test = ['this is such an amazing movie!', 'The film was nice to see.', 'I don\'t like my tea!', 'this is such an amazing movie!']
text_test = ['germany sells arms to saudi arabia', 'arms bend at the elbow', 'wave your arms around']
#text_test = ["nice movie", "good movie", "bad movie"]
embeddings = get_embedding(text_test)
"""for sentence in embeddings:
    for word, embedding in sentence:
        print(word)
    print()
"""

def cosineDistance(embeddings, sentence1, index1, sentence2, index2):
    print(f"Distance between {embeddings[sentence1][index1][0]} and {embeddings[sentence2][index2][0]} is {sp.spatial.distance.cosine(embeddings[sentence1][index1][1], embeddings[sentence2][index2][1])}")

#cosineDistance(embeddings, 0,6,1,2)
cosineDistance(embeddings, 0,3,1,1)
cosineDistance(embeddings, 0,3,2,3)
cosineDistance(embeddings, 1,1,2,3)
"""cosineDistance(embeddings, 0,1,1,2)
cosineDistance(embeddings, 0,1,2,2)
cosineDistance(embeddings, 1,2,2,2)"""