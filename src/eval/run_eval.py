"""
Evaluation harness.

This is the piece that turns "I built a RAG demo" into "I built a RAG
system I can prove works" -- the difference that matters most in an
interview.

We measure two SEPARATE things, because they fail independently and for
different reasons:

  1. RETRIEVAL RECALL: for a question with a known correct source chunk,
     did that chunk actually appear in the retrieved set? If not, no LLM
     in the world can answer correctly -- the failure is in chunking or
     search, not generation.

  2. ANSWER FAITHFULNESS: given the correct chunk was retrieved, did the
     LLM's answer actually reflect it accurately (vs. hallucinating extra
     detail, or contradicting it)? We use an LLM-as-judge approach: ask
     the LLM itself (or a stronger model) to score the answer against the
     ground-truth chunk on a 1-5 scale with justification.

The eval set (eval_set.json) is a curated list of questions with the
EXPECTED file_path + symbol_name that should be retrieved -- these need to
be written by hand, by someone who actually knows the codebase (that's the
point: your judgment as the project author is the ground truth here).

Run this after every re-index (see scripts/watch_and_reindex.py) so a
retrieval regression is caught immediately, not discovered by a confused
user later -- this is "CI/CD for a RAG index."
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_eval_set(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())


def evaluate_retrieval(eval_set: list[dict], retrieve_fn) -> dict:
    """
    retrieve_fn(question: str) -> list of chunk metadata dicts (must
    include 'file_path' and 'symbol_name').

    Each eval item may specify EITHER a single expected answer:
        {"expected_file_path": ..., "expected_symbol_name": ...}
    OR a list of acceptable alternatives (use this when more than one
    chunk would legitimately answer the question -- e.g. a method that
    delegates to a related standalone function is often a fair answer
    too, not just the one "canonical" symbol):
        {"acceptable_answers": [{"file_path": ..., "symbol_name": ...}, ...]}

    Returns per-question hit/miss plus an aggregate recall@k score.
    """
    results = []
    hits = 0
    for item in eval_set:
        retrieved = retrieve_fn(item["question"])
        retrieved_keys = {(r.get("file_path"), r.get("symbol_name")) for r in retrieved}

        if "acceptable_answers" in item:
            expected_keys = {(a["file_path"], a["symbol_name"]) for a in item["acceptable_answers"]}
        else:
            expected_keys = {(item["expected_file_path"], item["expected_symbol_name"])}

        hit = bool(retrieved_keys & expected_keys)
        hits += int(hit)
        results.append({
            "question": item["question"],
            "expected": sorted(expected_keys),
            "hit": hit,
            "retrieved_top_k": [f"{r.get('file_path')}::{r.get('symbol_name')}" for r in retrieved],
        })

    recall_at_k = hits / len(eval_set) if eval_set else 0.0
    return {
        "recall_at_k": round(recall_at_k, 3),
        "hits": hits,
        "total": len(eval_set),
        "per_question": results,
    }


JUDGE_PROMPT_TEMPLATE = """You are grading an AI code assistant's answer for faithfulness.

Question: {question}

Ground truth source code/docs the answer should be based on:
{ground_truth_chunk}

The assistant's answer:
{answer}

Score the answer from 1-5 on FAITHFULNESS to the ground truth:
5 = fully accurate, no unsupported claims
3 = mostly accurate but includes minor unsupported details
1 = contradicts the ground truth or is mostly fabricated

Respond in this exact JSON format, nothing else:
{{"score": <int 1-5>, "justification": "<one sentence>"}}
"""


def evaluate_faithfulness(eval_set: list[dict], answer_fn, judge_fn) -> dict:
    """
    answer_fn(question: str) -> (answer_text: str, ground_truth_chunk: str)
    judge_fn(prompt: str) -> str (raw LLM response, expected to be JSON)
    """
    results = []
    scores = []
    for item in eval_set:
        answer, ground_truth = answer_fn(item["question"])
        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=item["question"], ground_truth_chunk=ground_truth, answer=answer,
        )
        raw_judgment = judge_fn(judge_prompt)
        try:
            judgment = json.loads(raw_judgment)
            score = judgment["score"]
        except (json.JSONDecodeError, KeyError):
            score = None
            judgment = {"score": None, "justification": f"unparseable judge response: {raw_judgment[:100]}"}

        if score is not None:
            scores.append(score)
        results.append({"question": item["question"], "answer": answer, **judgment})

    avg_score = sum(scores) / len(scores) if scores else 0.0
    return {
        "avg_faithfulness_score": round(avg_score, 2),
        "scored_count": len(scores),
        "total": len(eval_set),
        "per_question": results,
    }


def print_report(retrieval_results: dict, faithfulness_results: dict | None = None) -> None:
    print("=" * 60)
    print("EVAL REPORT")
    print("=" * 60)
    print(f"Retrieval recall@k: {retrieval_results['recall_at_k']:.1%} "
          f"({retrieval_results['hits']}/{retrieval_results['total']})")
    for r in retrieval_results["per_question"]:
        status = "PASS" if r["hit"] else "MISS"
        print(f"  [{status}] {r['question'][:60]}")

    if faithfulness_results:
        print(f"\nAvg faithfulness score: {faithfulness_results['avg_faithfulness_score']}/5 "
              f"({faithfulness_results['scored_count']}/{faithfulness_results['total']} scored)")


if __name__ == "__main__":
    # Self-test with a fake retrieve_fn -- verifies scoring logic without
    # needing the real embedding/LLM pipeline running.
    fake_eval_set = [
        {"question": "How does dependency resolution work?",
         "expected_file_path": "dependencies/utils.py", "expected_symbol_name": "solve_dependencies"},
        {"question": "How is a JWT decoded?",
         "expected_file_path": "auth.py", "expected_symbol_name": "decode_token"},
    ]

    def fake_retrieve(question: str) -> list[dict]:
        if "dependency" in question.lower():
            return [{"file_path": "dependencies/utils.py", "symbol_name": "solve_dependencies"}]
        return [{"file_path": "wrong_file.py", "symbol_name": "wrong_symbol"}]  # deliberate miss

    retrieval_report = evaluate_retrieval(fake_eval_set, fake_retrieve)
    print_report(retrieval_report)
    assert retrieval_report["recall_at_k"] == 0.5, "expected exactly 1/2 hit rate in this fake scenario"
    print("\nPASS: eval harness correctly distinguishes hit vs miss")
