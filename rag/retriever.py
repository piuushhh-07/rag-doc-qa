import os
import re
import pickle
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

EMBEDDING_MODEL = "openai/text-embedding-3-small"


def simple_tokenize(text):
    """
    Lowercase, strip punctuation, split on whitespace.
    Stripping punctuation matters -- without it, "CEO?" and "CEO,"
    never match "CEO" as separate tokens under exact BM25 matching.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


class Retriever:
    def __init__(self, vectorstore_dir):
        index_path = os.path.join(vectorstore_dir, "index.faiss")
        chunks_path = os.path.join(vectorstore_dir, "chunks.pkl")

        self.index = faiss.read_index(index_path)

        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        tokenized_corpus = [simple_tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print(f"Retriever loaded: {len(self.chunks)} chunks in index (dense + BM25)")

    def embed_query(self, query):
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
        embedding = response.data[0].embedding
        return np.array([embedding], dtype="float32")

    def dense_retrieve(self, query, top_k=5):
        """Original embedding-only retrieval, kept for comparison/debugging."""
        query_vector = self.embed_query(query)
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({**chunk, "score": float(score)})
        return results

    def hybrid_retrieve(self, query, top_k=5, k=60):
        """
        Combines dense and BM25 retrieval using Reciprocal Rank Fusion (RRF),
        the standard approach for hybrid search in production systems.

        Why RRF instead of a weighted linear blend of raw scores: raw BM25
        and dense scores live on different, incompatible scales, and
        min-max normalization can let a single outlier score dominate the
        combination regardless of true relevance -- we hit exactly this
        failure case with a false-positive "CEO" keyword match on an
        unrelated exhibit-list page.

        RRF instead uses only RANK POSITION from each method:
        score = sum over each method of 1 / (k + rank)
        This makes fusion robust to score-scale differences and to one-off
        high scores, since only relative ordering matters. k=60 is the
        standard constant from the original RRF paper.
        """
        query_vector = self.embed_query(query)
        _, dense_indices = self.index.search(query_vector, len(self.chunks))
        dense_ranking = [int(i) for i in dense_indices[0]]

        tokenized_query = simple_tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranking = [int(i) for i in np.argsort(bm25_scores)[::-1]]

        rrf_scores = {}
        for rank, idx in enumerate(dense_ranking):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)
        for rank, idx in enumerate(bm25_ranking):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)

        top_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "source_file": chunk["source_file"],
                "page_number": chunk["page_number"],
                "score": rrf_scores[idx],
                "dense_rank": dense_ranking.index(idx) + 1 if idx in dense_ranking else None,
                "bm25_rank": bm25_ranking.index(idx) + 1 if idx in bm25_ranking else None
            })
        return results

    def retrieve(self, query, top_k=5, mode="hybrid"):
        """Unified entry point -- mode='hybrid' (RRF, default) or 'dense'."""
        if mode == "dense":
            return self.dense_retrieve(query, top_k)
        return self.hybrid_retrieve(query, top_k)


if __name__ == "__main__":
    import pathlib
    project_root = pathlib.Path(__file__).resolve().parent.parent
    vectorstore_dir = project_root / "vectorstore"

    retriever = Retriever(str(vectorstore_dir))

    test_query = "Who is Apple's CEO?"
    print(f"\n=== HYBRID (RRF): '{test_query}' ===")
    for r in retriever.retrieve(test_query, top_k=5, mode="hybrid"):
        print(f"  page {r['page_number']}: rrf={r['score']:.5f}, "
              f"dense_rank={r['dense_rank']}, bm25_rank={r['bm25_rank']}")