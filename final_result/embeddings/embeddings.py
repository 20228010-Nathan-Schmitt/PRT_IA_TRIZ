import embeddings.simCSE as simCSE
import embeddings.mpnet_base as mpnet_base
import embeddings.patentsberta as patentsberta
import embeddings.roberta_base as roberta_base
import embeddings.multi_qa_mpnet_base as multi_qa_mpnet_base
import embeddings.distilroberta as distilroberta


embeddings = {
    "patentsberta": patentsberta.embeddings_patentsberta,
    "mpnet_base":mpnet_base.embeddings_mpnet_base,
    "multi_qa_mpnet_base":multi_qa_mpnet_base.embeddings_multi_qa_mpnet_base,
    "distilroberta":distilroberta.embeddings_distilroberta,
    "simCSE": simCSE.embeddings_simcse,
    "roberta_base":roberta_base.embeddings_roberta_base
}


embed_size = {
    "patentsberta": 768,
    "mpnet_base":768,
    "multi_qa_mpnet_base":768,
    "distilroberta":768,
    "simCSE": 1024,
    "roberta_base":768
}
