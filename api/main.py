import pathlib
import sys

project_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag.pipeline import RAGPipeline

app = FastAPI(
    title="RAG Document Q&A Assistant",
    description="Ask questions over a document corpus and get grounded, cited answers.",
    version="1.0.0"
)

# Loaded once at startup -- not per request. This is the same reasoning
# as the Retriever class: loading a FAISS index from disk is expensive,
# so we do it exactly once when the server boots.
vectorstore_dir = str(project_root / "vectorstore")
pipeline = RAGPipeline(vectorstore_dir, top_k=5)


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


class Citation(BaseModel):
    number: int
    source_file: str
    page_number: int
    score: float


class AskResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]


@app.get("/health")
def health_check():
    """
    Simple liveness check -- lets you (or a deployment platform,
    or Docker healthcheck later) confirm the server is up and the
    index loaded correctly, without doing a full expensive query.
    """
    return {"status": "ok", "chunks_indexed": len(pipeline.retriever.chunks)}


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    Main endpoint: takes a question, runs it through the full RAG
    pipeline, returns a grounded answer with citations.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = pipeline.ask(request.question, top_k=request.top_k)
    return result