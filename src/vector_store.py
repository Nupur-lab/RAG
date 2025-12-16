import faiss
import numpy as np
import os
import pickle

class VectorStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.text_chunks = []

    def add(self, embeddings, chunks):
        self.index.add(np.array(embeddings))
        self.text_chunks.extend(chunks)

    def search(self, query_embedding, top_k=3):
        distances, indices = self.index.search(
            np.array(query_embedding), top_k
        )
        return [self.text_chunks[i] for i in indices[0]]

    def save(self, path):
        faiss.write_index(self.index, f"{path}/faiss.index")
        with open(f"{path}/chunks.pkl", "wb") as f:
            pickle.dump(self.text_chunks, f)

    def load(self, path):
        self.index = faiss.read_index(f"{path}/faiss.index")
        with open(f"{path}/chunks.pkl", "rb") as f:
            self.text_chunks = pickle.load(f)
