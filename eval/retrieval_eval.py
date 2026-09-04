import json
import pathlib
import sys

project_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from rag.pipeline import RAGPipeline
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

JUDGE_MODEL = "openai/gpt-4o-mini"


def load_test_set(path):
    with open(path, "r") as f:
        return json.load(f)


def check_retrieval_precision(citations, expected_page, tolerance=1):
    """
    Precision@K check: was the expected source page actually retrieved
    in the top-k results?

    tolerance=1 allows off-by-one page matches, since content near a
    page boundary can legitimately land on the adjacent page depending
    on how PDF extraction/chunking split it -- this isn't cheating,
    it's accounting for a real, known imprecision in page-level chunking.
    """
    retrieved_pages = [c["page_number"] for c in citations]
    for page in retrieved_pages:
        if abs(page - expected_page) <= tolerance:
            return True
    return False


def check_answer_contains_keywords(answer, expected_keywords):
    """
    Simple lexical check: does the answer contain at least one of the
    expected keywords/phrases? Case-insensitive.

    This is a cheap, fast first-pass check -- not a substitute for the
    LLM-judge faithfulness score below, but useful because it's free
    and catches obvious total misses instantly.
    """
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in expected_keywords)


def llm_judge_faithfulness(question, answer, expected_keywords):
    """
    Uses an LLM to judge whether the generated answer is faithful --
    i.e. factually consistent with what we expect, not hallucinating,
    and actually addressing the question.

    This is the standard "LLM-as-judge" pattern used when you don't
    have humans available to grade every answer by hand. It's not
    perfect (the judge can be wrong too), but it's a defensible,
    scalable proxy -- and a real technique used in production eval
    pipelines, worth naming by name in an interview.

    Returns a dict: {"score": 1-5, "reasoning": "..."}
    """
    prompt = f"""You are evaluating the faithfulness of an AI-generated answer.

Question: {question}
Expected key facts/keywords the answer should reflect: {expected_keywords}
Generated answer: {answer}

Rate the answer's faithfulness on a scale of 1-5:
5 = Fully correct, matches expected facts, no hallucination
3 = Partially correct or vague
1 = Wrong, contradicts expected facts, or hallucinated

Respond ONLY in this exact JSON format, nothing else:
{{"score": <1-5>, "reasoning": "<one sentence>"}}"""

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()
    # Models sometimes wrap JSON in markdown fences despite instructions --
    # strip those defensively rather than letting json.loads crash.
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"score": None, "reasoning": f"Could not parse judge output: {raw}"}


def run_evaluation(test_set_path, vectorstore_dir, top_k=5):
    test_set = load_test_set(test_set_path)
    pipeline = RAGPipeline(vectorstore_dir, top_k=top_k)

    results = []
    retrieval_hits = 0
    keyword_hits = 0
    faithfulness_scores = []

    for item in test_set:
        result = pipeline.ask(item["question"])

        retrieval_ok = check_retrieval_precision(result["citations"], item["expected_source_page"])
        keyword_ok = check_answer_contains_keywords(result["answer"], item["expected_answer_contains"])
        judge = llm_judge_faithfulness(item["question"], result["answer"], item["expected_answer_contains"])

        if retrieval_ok:
            retrieval_hits += 1
        if keyword_ok:
            keyword_hits += 1
        if judge["score"] is not None:
            faithfulness_scores.append(judge["score"])

        results.append({
            "id": item["id"],
            "question": item["question"],
            "answer": result["answer"],
            "citations": result["citations"],  #it save actual retrieved pages for debugging
            "retrieval_hit": retrieval_ok,
            "keyword_hit": keyword_ok,
            "faithfulness_score": judge["score"],
            "faithfulness_reasoning": judge["reasoning"]
        })

        print(f"[{item['id']}] retrieval={'✓' if retrieval_ok else '✗'} "
              f"keyword={'✓' if keyword_ok else '✗'} "
              f"faithfulness={judge['score']}/5")

    n = len(test_set)
    summary = {
        "total_questions": n,
        "retrieval_precision_at_k": round(retrieval_hits / n, 3),
        "keyword_match_rate": round(keyword_hits / n, 3),
        "avg_faithfulness_score": round(sum(faithfulness_scores) / len(faithfulness_scores), 2) if faithfulness_scores else None
    }

    return summary, results


if __name__ == "__main__":
    vectorstore_dir = str(project_root / "vectorstore")
    test_set_path = str(project_root / "eval" / "qa_test_set.json")

    summary, results = run_evaluation(test_set_path, vectorstore_dir, top_k=5)

    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for k, v in summary.items():
        print(f"{k}: {v}")

    # Save detailed results for later inspection / README screenshots
    output_path = project_root / "eval" / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")