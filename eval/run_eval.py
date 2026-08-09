"""Offline accuracy eval harness for F1rstAid.

Runs a fixed set of F-1 visa questions with known-correct facts against the
live QA pipeline and scores the responses. Costs real OpenAI API calls --
this is a manual/periodic tool, not part of the pytest unit test suite.

Usage:
    export OPENAI_API_KEY=sk-...
    python eval/run_eval.py --label baseline
    python eval/run_eval.py --label after-reranker --model gpt-4o-mini --k 5
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1rstaid import AppConfig, F1rstAidApp  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
DECLINE_MARKER = "Relevance Check"

logging.getLogger().setLevel(logging.WARNING)


def load_dataset(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def score_question(question: dict, answer: dict) -> dict:
    """Score a single question/answer pair. Returns a result dict."""
    result_text = (answer or {}).get("result", "") or ""
    lower_text = result_text.lower()
    declined = DECLINE_MARKER.lower() in lower_text

    if question.get("must_decline"):
        correct = declined
        detail = "declined as expected" if correct else "did NOT decline an out-of-scope question"
        return {
            "id": question["id"],
            "category": question["category"],
            "correct": correct,
            "declined": declined,
            "detail": detail,
            "matched_keypoints": [],
            "missing_keypoints": [],
            "answer": result_text,
        }

    if declined:
        return {
            "id": question["id"],
            "category": question["category"],
            "correct": False,
            "declined": True,
            "detail": "incorrectly declined an in-scope question",
            "matched_keypoints": [],
            "missing_keypoints": question.get("expected_keypoints", []),
            "answer": result_text,
        }

    all_kw = question.get("expected_keypoints", [])
    any_kw = question.get("expected_keypoints_any", [])

    matched = [kw for kw in all_kw if kw.lower() in lower_text]
    missing = [kw for kw in all_kw if kw.lower() not in lower_text]
    any_matched = [kw for kw in any_kw if kw.lower() in lower_text]

    all_ok = len(missing) == 0
    any_ok = (len(any_kw) == 0) or (len(any_matched) > 0)
    correct = all_ok and any_ok

    detail_parts = []
    if missing:
        detail_parts.append(f"missing required: {missing}")
    if any_kw and not any_matched:
        detail_parts.append(f"missing any-of: {any_kw}")
    detail = "; ".join(detail_parts) if detail_parts else "all keypoints present"

    return {
        "id": question["id"],
        "category": question["category"],
        "correct": correct,
        "declined": False,
        "detail": detail,
        "matched_keypoints": matched + any_matched,
        "missing_keypoints": missing,
        "answer": result_text,
    }


def run(dataset: dict, config: AppConfig) -> list:
    app = F1rstAidApp(config)
    if not app.initialize():
        print("ERROR: app.initialize() failed -- check OPENAI_API_KEY and faiss_index/", file=sys.stderr)
        sys.exit(1)

    results = []
    questions = dataset["questions"]
    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {question['id']}: {question['question'][:70]}", file=sys.stderr)
        answer = app.get_answer(question["question"])
        results.append(score_question(question, answer))
    return results


def summarize(results: list) -> dict:
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    by_category = {}
    for r in results:
        cat = r["category"]
        bucket = by_category.setdefault(cat, {"total": 0, "correct": 0})
        bucket["total"] += 1
        bucket["correct"] += 1 if r["correct"] else 0

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "by_category": {
            cat: {
                "total": b["total"],
                "correct": b["correct"],
                "accuracy": round(b["correct"] / b["total"], 4) if b["total"] else 0.0,
            }
            for cat, b in sorted(by_category.items())
        },
    }


def print_report(summary: dict, results: list, label: str):
    print(f"\n=== F1rstAid Eval Report: {label} ===")
    print(f"Overall: {summary['correct']}/{summary['total']} ({summary['accuracy']:.1%})\n")
    print(f"{'Category':<28}{'Correct':>10}{'Total':>8}{'Accuracy':>12}")
    for cat, b in summary["by_category"].items():
        print(f"{cat:<28}{b['correct']:>10}{b['total']:>8}{b['accuracy']:>11.1%}")

    failures = [r for r in results if not r["correct"]]
    if failures:
        print(f"\n--- {len(failures)} failing question(s) ---")
        for r in failures:
            print(f"  [{r['id']}] {r['detail']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(EVAL_DIR / "dataset.json"))
    parser.add_argument("--output-dir", default=str(EVAL_DIR / "results"))
    parser.add_argument("--label", default="run", help="Short name for this run, used in the output filename")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="Chat model name")
    parser.add_argument("--k", type=int, default=3, help="Retriever search_k")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--vector-store-path", default="faiss_index")
    args = parser.parse_args()

    dataset = load_dataset(Path(args.dataset))
    config = AppConfig(
        model_name=args.model,
        vector_store_path=args.vector_store_path,
        search_k=args.k,
        temperature=args.temperature,
    )

    results = run(dataset, config)
    summary = summarize(results)
    print_report(summary, results, args.label)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"{timestamp}_{args.label}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label": args.label,
                "timestamp": timestamp,
                "config": asdict(config) if hasattr(config, "__dataclass_fields__") else vars(config),
                "dataset_version": dataset.get("version"),
                "summary": summary,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
