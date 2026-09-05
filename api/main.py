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

from fastapi.responses import StreamingResponse
from rag.generator import generate_answer_stream
import json


@app.post("/ask/stream")
def ask_question_stream(request: AskRequest):
    """
    Streaming version of /ask. Retrieval happens synchronously first
    (fast, and we need citations before we can tell the client anything
    useful), then the answer text is streamed token-by-token.

    Protocol: plain text chunks for the answer, followed by a special
    delimiter line, followed by a JSON blob containing citations.
    This is a simplified custom protocol (not full Server-Sent Events)
    -- good enough for our own Streamlit client, though a "real" public
    API would typically use proper SSE with `data: ` prefixes.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    top_k = request.top_k or pipeline.top_k
    retrieved_chunks = pipeline.retriever.retrieve(request.question, top_k=top_k)

    if not retrieved_chunks:
        def empty_gen():
            yield "I don't have enough information in the provided documents to answer this."
        return StreamingResponse(empty_gen(), media_type="text/plain")

    def event_generator():
        # Stream the answer text first
        for token in generate_answer_stream(request.question, retrieved_chunks):
            yield token

        # Then send citations as a delimited final block
        citations = [
            {
                "number": i + 1,
                "source_file": c["source_file"],
                "page_number": c["page_number"],
                "score": round(c["score"], 4)
            }
            for i, c in enumerate(retrieved_chunks)
        ]
        yield "\n[[CITATIONS]]\n"
        yield json.dumps(citations)

    return StreamingResponse(event_generator(), media_type="text/plain")