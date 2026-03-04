import faiss
import numpy as np
import torch

class SessionVectorStore:
    """
    Local session memory using FAISS.
    Useful for storing and retrieving past interactions in a session.
    """
    def __init__(self, dim=768):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    def add(self, vector, text_metadata):
        # vector: [D] or [1, D]
        if isinstance(vector, torch.Tensor):
            vector = vector.detach().cpu().numpy()
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        self.index.add(vector.astype('float32'))
        self.metadata.append(text_metadata)

    def search(self, query_vector, k=5):
        if isinstance(query_vector, torch.Tensor):
            query_vector = query_vector.detach().cpu().numpy()
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        distances, indices = self.index.search(query_vector.astype('float32'), k)
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1 and idx < len(self.metadata):
                results.append({
                    "metadata": self.metadata[idx],
                    "distance": float(distances[0][i])
                })
        return results

    def clear(self):
        self.index.reset()
        self.metadata = []
