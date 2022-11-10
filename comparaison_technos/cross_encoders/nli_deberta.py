from compare_tools import transposeList
import numpy as np

model_nli_deberta = None
# label_mapping = ['contradiction', 'entailment', 'neutral']
label_mapping = [0, 1, 0.3]


def dist_nli_deberta(sentences):
    global model_nli_deberta
    if model_nli_deberta is None:
        from sentence_transformers import CrossEncoder
        model_nli_deberta = CrossEncoder('cross-encoder/nli-deberta-v3-base')

    scores = model_nli_deberta.predict(sentences)
    return np.fromiter((label_mapping[score_max] for score_max in scores.argmax(axis=1)), float)
