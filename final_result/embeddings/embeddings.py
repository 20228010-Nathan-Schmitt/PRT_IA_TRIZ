import embeddings.simCSE as simCSE
import embeddings.mpnet_base as mpnet_base
import embeddings.patentsberta as patentsberta
import embeddings.roberta_base as roberta_base
import embeddings.multi_qa_mpnet_base as multi_qa_mpnet_base
import embeddings.distilroberta as distilroberta
import embeddings.custom as custom
import embeddings.custom_old as custom_old
import embeddings.deberta as deberta


embeddings = {
    #"patentsberta": patentsberta.embeddings_patentsberta,
    "mpnet_base":mpnet_base.embeddings_mpnet_base,
    "deberta":deberta.embeddings_deberta,
    #"multi_qa_mpnet_base":multi_qa_mpnet_base.embeddings_multi_qa_mpnet_base,
    #"distilroberta":distilroberta.embeddings_distilroberta,
    #"simCSE": simCSE.embeddings_simcse,
    #"roberta_base":roberta_base.embeddings_roberta_base,
    "custom32k_1":custom.CustomSentenceTransformer("my_model_sbert_trained_32k"),
    "custom32k_2":custom.CustomSentenceTransformer("my_model_sbert_trained_32k_2"),
    "custom":custom.CustomSentenceTransformer("my_model_sbert"),
    "custom_old":custom.CustomSentenceTransformer("my_model_sbert_old")
}



embed_size = {
    "patentsberta": 768,
    "mpnet_base":768,
    "multi_qa_mpnet_base":768,
    "distilroberta":768,
    "simCSE": 1024,
    "roberta_base":768
}

