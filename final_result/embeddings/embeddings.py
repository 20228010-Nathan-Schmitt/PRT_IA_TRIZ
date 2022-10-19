import embeddings.simCSE as simCSE
import embeddings.mpnet_base as mpnet_base
import embeddings.patentsberta as patentsberta


embeddings = {"simCSE": simCSE.embeddings_simcse,"patentsberta": patentsberta.embeddings_patentsberta, "mpnet_base":mpnet_base.embeddings_mpnet_base}


embed_size = {"simCSE": 1024,"patentsberta": 768, "mpnet_base":768}
