import os
import pickle
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# OpenRouter exposes an OpenAI-compatible API — same SDK, different base_url + key
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

EMBEDDING_MODEL = "openai/text-embedding-3-small"  # OpenRouter prefixes model names with provider
EMBEDDING_DIM = 1536


def embed_texts(texts, batch_size=100):
    """
    Sends chunk texts to the embeddings API in batches.
    Batching matters: one API call per chunk would be slow and
    wasteful — the API accepts many texts per request.
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        print(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks")

    return np.array(all_embeddings, dtype="float32")


def build_faiss_index(embeddings):
    """
    IndexFlatIP = exact search using inner product (cosine similarity,
    since these embeddings are normalized to unit length).
    'Flat' = no approximation — exact but slower at huge scale;
    fine for our few-hundred-chunk corpus.
    """
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    return index


def save_index(index, chunks, output_dir):
    """
    FAISS only stores vectors, not text/metadata — so we persist
    chunk metadata separately and rely on position-matching:
    index position N <-> chunks[N].
    """
    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, "index.faiss"))
    with open(os.path.join(output_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved index and {len(chunks)} chunk records to {output_dir}")


if __name__ == "__main__":
    import pathlib
    import sys

    project_root = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    from ingestion.pdf_loader import load_all_pdfs
    from ingestion.chunker import chunk_pages

    pdf_dir = project_root / "data" / "raw_pdfs"
    vectorstore_dir = project_root / "vectorstore"

    print("Loading PDFs...")
    pages = load_all_pdfs(str(pdf_dir))

    print("Chunking...")
    chunks = chunk_pages(pages)
    print(f"Produced {len(chunks)} chunks")

    print("Embedding chunks (calling OpenRouter API)...")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    save_index(index, chunks, str(vectorstore_dir))