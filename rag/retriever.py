import os
import pickle
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

EMBEDDING_MODEL = "openai/text-embedding-3-small"  # must match embed_and_index.py exactly


class Retriever:
    def __init__(self, vectorstore_dir):
        """
        Loads the persisted FAISS index and chunk metadata once,
        at startup — not on every query. Re-loading from disk per
        query would be slow and pointless since the index doesn't
        change between queries.
        """
        index_path = os.path.join(vectorstore_dir, "index.faiss")
        chunks_path = os.path.join(vectorstore_dir, "chunks.pkl")

        self.index = faiss.read_index(index_path)

        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        print(f"Retriever loaded: {len(self.chunks)} chunks in index")

    def embed_query(self, query):
        """Embed the user's question the same way chunks were embedded."""
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query]
        )
        embedding = response.data[0].embedding
        return np.array([embedding], dtype="float32")

    def retrieve(self, query, top_k=5):
        """
        Returns the top_k most relevant chunks for a query, along with
        similarity scores, ranked best-first.

        FAISS's .search() returns two arrays:
        - distances: similarity scores (higher = more similar, since
          we're using inner product on normalized vectors = cosine sim)
        - indices: positions in the index -> map back to self.chunks
        """
        query_vector = self.embed_query(query)
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 if fewer than top_k results exist
                continue
            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "source_file": chunk["source_file"],
                "page_number": chunk["page_number"],
                "score": float(score)
            })
        return results


if __name__ == "__main__":
    import pathlib

    project_root = pathlib.Path(__file__).resolve().parent.parent
    vectorstore_dir = project_root / "vectorstore"

    retriever = Retriever(str(vectorstore_dir))

    # Quick manual test — try a question you know the answer exists for
    test_query = "What are Apple's main risk factors?"
    results = retriever.retrieve(test_query, top_k=3)

    print(f"\nQuery: {test_query}\n")
    for i, r in enumerate(results):
        print(f"--- Result {i+1} (score: {r['score']:.4f}, page {r['page_number']}) ---")
        print(r["text"][:300])
        print()