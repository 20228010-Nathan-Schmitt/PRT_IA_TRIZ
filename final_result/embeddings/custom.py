from sentence_transformers import SentenceTransformer
model_custom = None

class CustomSentenceTransformer:
    def __init__(self, folder_name):
        self.folder_name = folder_name
        self.model = None

    def __call__(self, sentences, batch_size=32, once=False):
        if self.model is None : 
            self.model = SentenceTransformer('./'+self.folder_name)
        return self.model.encode(sentences, show_progress_bar=True, batch_size=batch_size)
