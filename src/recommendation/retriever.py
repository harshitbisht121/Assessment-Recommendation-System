"""
Retriever Module for SHL Assessment Recommendation System
Performs fast semantic search over SHL catalog using FAISS
"""

import numpy as np
import pandas as pd
import faiss
from pathlib import Path

class SHLRetriever:
    def __init__(self, data_dir="data/processed"):
        """
        Loads embeddings + metadata and builds FAISS index
        """
        self.data_dir = Path(data_dir)

        embeddings_path = self.data_dir / "embeddings.npy"
        metadata_path = self.data_dir / "metadata.csv"

        # Load embeddings
        self.embeddings = np.load(embeddings_path)
        self.metadata = pd.read_csv(metadata_path)

        if len(self.embeddings) != len(self.metadata):
            raise ValueError(
                f"Embeddings ({len(self.embeddings)}) and metadata ({len(self.metadata)}) size mismatch!"
            )

        self.index = self._build_faiss_index()

        print(f"✓ Retriever initialized with {len(self.metadata)} assessments")

    def _build_faiss_index(self):
        """
        Build cosine similarity FAISS index
        """
        dimension = self.embeddings.shape[1]

        # Normalize for cosine similarity
        normalized_embeddings = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )

        index = faiss.IndexFlatIP(dimension)   # inner product = cosine similarity
        index.add(normalized_embeddings.astype("float32"))

        return index

    def search(self, query_embedding, top_k=20):
        """
        Perform similarity search
        Args:
            query_embedding -> numpy vector
            top_k -> number of results
        Returns:
            indices list, scores list
        """

        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        scores, indices = self.index.search(
            query_embedding.reshape(1, -1).astype("float32"),
            top_k
        )

        return indices[0].tolist(), scores[0].tolist()

    def get_results(self, indices):
        """
        Return metadata rows for given indices
        """
        return self.metadata.iloc[indices].copy()

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    
    print("\nInitializing Retriever…")
    retriever = SHLRetriever()

    model = SentenceTransformer("all-MiniLM-L6-v2")

    query = "Hiring Java developers who can collaborate with business teams"
    print(f"\nQuery: {query}")

    emb = model.encode(query)
    indices, scores = retriever.search(emb, top_k=5)

    print("\nTop 5 Results:")
    results = retriever.get_results(indices)

    for i, (idx, score) in enumerate(zip(indices, scores), 1):
        row = results.iloc[i-1]
        print(f"\n{i}. {row['name']}")
        print(f"   URL: {row['url']}")
        print(f"   Score: {score}")
