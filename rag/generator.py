import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

GENERATION_MODEL = "openai/gpt-4o-mini"


def build_prompt(query, retrieved_chunks):
    """
    Builds a prompt that:
    1. Gives the model only the retrieved context (not open knowledge)
    2. Numbers each chunk so the model can cite [1], [2], etc.
    3. Explicitly instructs it to say "I don't know" rather than guess

    This last point is the single biggest lever against hallucination —
    an LLM asked "answer using ONLY this context" behaves very
    differently from one asked an open question.
    """
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks):
        context_blocks.append(
            f"[{i+1}] (Source: {chunk['source_file']}, Page {chunk['page_number']})\n{chunk['text']}"
        )
    context_text = "\n\n".join(context_blocks)

    prompt = f"""You are a document Q&A assistant. Answer the question using ONLY the context provided below.

Rules:
- If the answer is not contained in the context, say "I don't have enough information in the provided documents to answer this."
- Do NOT use outside knowledge, even if you know the answer.
- Cite your sources using the bracket numbers, e.g. [1], [2], matching the context blocks below.
- Be concise and direct.

Context:
{context_text}

Question: {query}

Answer:"""
    return prompt


def generate_answer(query, retrieved_chunks):
    """
    Sends the prompt to the LLM and returns both the answer text
    and the source chunks used, so the caller (pipeline.py, then
    the UI) can display citations alongside the answer.
    """
    prompt = build_prompt(query, retrieved_chunks)

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,  # low temperature: we want faithful, consistent
                           # answers grounded in context, not creative variation
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": [
            {
                "source_file": c["source_file"],
                "page_number": c["page_number"],
                "score": c["score"]
            }
            for c in retrieved_chunks
        ]
    }


if __name__ == "__main__":
    import pathlib
    import sys

    project_root = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    from rag.retriever import Retriever

    vectorstore_dir = project_root / "vectorstore"
    retriever = Retriever(str(vectorstore_dir))

    test_query = "What are Apple's main risk factors?"
    retrieved = retriever.retrieve(test_query, top_k=5)
    result = generate_answer(test_query, retrieved)

    print(f"Query: {test_query}\n")
    print(f"Answer:\n{result['answer']}\n")
    print("Sources used:")
    for s in result["sources"]:
        print(f"  - {s['source_file']}, page {s['page_number']} (score: {s['score']:.4f})")