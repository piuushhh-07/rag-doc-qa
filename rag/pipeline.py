import pathlib
import sys

project_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from rag.retriever import Retriever
from rag.generator import generate_answer


class RAGPipeline:
    def __init__(self, vectorstore_dir, top_k=5):
        """
        Loads the retriever once at startup (expensive: reads FAISS
        index + chunk metadata from disk). top_k is stored as a default
        so callers don't have to specify it every time, but can override
        per query if needed.
        """
        self.retriever = Retriever(vectorstore_dir)
        self.top_k = top_k

    def ask(self, query, top_k=None):
        """
        Full RAG flow: retrieve relevant chunks -> generate an answer
        grounded in them -> return answer + explicitly numbered sources.

        Returns:
        {
            "query": str,
            "answer": str,
            "citations": [
                {"number": 1, "source_file": ..., "page_number": ..., "score": ...},
                ...
            ]
        }
        """
        k = top_k or self.top_k
        retrieved_chunks = self.retriever.retrieve(query, top_k=k)

        if not retrieved_chunks:
            return {
                "query": query,
                "answer": "I don't have enough information in the provided documents to answer this.",
                "citations": []
            }

        result = generate_answer(query, retrieved_chunks)

        # Explicitly number citations to match [1], [2], etc. used in the
        # prompt/answer -- this is the part that was implicit before.
        # generator.py builds the prompt with chunks in this exact order,
        # so position in this list == the bracket number in the answer.
        citations = [
            {
                "number": i + 1,
                "source_file": s["source_file"],
                "page_number": s["page_number"],
                "score": round(s["score"], 4)
            }
            for i, s in enumerate(result["sources"])
        ]

        return {
            "query": query,
            "answer": result["answer"],
            "citations": citations
        }


def format_answer(result):
    """Pretty-print a pipeline result for terminal/debug use."""
    lines = [f"Q: {result['query']}\n", f"A: {result['answer']}\n", "Citations:"]
    for c in result["citations"]:
        lines.append(f"  [{c['number']}] {c['source_file']}, page {c['page_number']} (score: {c['score']})")
    return "\n".join(lines)


if __name__ == "__main__":
    vectorstore_dir = project_root / "vectorstore"
    pipeline = RAGPipeline(str(vectorstore_dir), top_k=5)

    test_queries = [
        "What are Apple's main risk factors?",
        "What was Apple's revenue?",
         "What are Apple's main risk factors?",
        "Who is the CEO of Apple?"  # deliberately testing a question that might NOT be answerable from context alone
    ]

    for q in test_queries:
        result = pipeline.ask(q)
        print(format_answer(result))
        print("\n" + "="*80 + "\n")