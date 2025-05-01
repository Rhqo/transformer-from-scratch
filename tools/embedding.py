import numpy as np

class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Xavier init
        self.weight = np.random.randn(num_embeddings, embedding_dim) * np.sqrt(2.0/(num_embeddings + embedding_dim))

    def __call__(self, indices):
        return self.weight[indices]
