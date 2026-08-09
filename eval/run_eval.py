"""Offline accuracy eval harness for F1rstAid.

Runs a fixed set of F-1 visa questions with known-correct facts against the
live QA pipeline, scores the responses, and appends a summary to
eval/results/history.jsonl so accuracy can be tracked run-over-run against
the change that produced it. Costs real OpenAI API calls -- this is a
manual/periodic tool, not part of the pytest unit test suite.

Usage:
    export OPENAI_API_KEY=sk-...
    python eval/run_eval.py --label baseline
    python eval/run_eval.py --label after-reranker --model gpt-4o-mini --k 5 \
        --note "switched reranker to per-query cross-encoder"
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1rstaid import AppConfig, F1rstAidApp  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
DECLINE_MARKER = "Relevance Check"
ABSTAIN_MARKER = "don't have enough information"
NON_SUBSTANTIVE_CATEGORIES = {"out_of_scope", "help"}

logging.getLogger().setLevel(logging.WARNING)


def load_dataset(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def git_provenance() -> dict:
    """Best-effort git commit/branch/dirty-tree info for this run. Never raises."""

    def _run(args):
        try:
            return subprocess.run(
                args, cwd=REPO_ROOT, capture_output=True, text=True, timeout=5, check=True
            ).stdout.strip()
        except Exception:
            return None

    commit = _run(["git", "rev-parse", "--short", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
    }


def _normalize(text: str) -> str:
    """Lowercase and fold hyphens to spaces so '90-day' and '90 day' compare equal --
    the model phrases the same fact either way depending on retrieved chunk wording."""
    return text.lower().replace("-", " ")


def score_question(question: dict, answer: dict, latency_ms: float = None) -> dict:
    """Score a single question/answer pair. Returns a result dict."""
    result_text = (answer or {}).get("result", "") or ""
    lower_text = _normalize(result_text)
    declined = DECLINE_MARKER.lower() in lower_text
    abstained = ABSTAIN_MARKER in lower_text

    base = {
        "id": question["id"],
        "category": question["category"],
        "declined": declined,
        "abstained": abstained,
        "latency_ms": latency_ms,
        "answer": result_text,
    }

    if question.get("must_decline"):
        # Either refusal path counts: the explicit relevance-gate decline, or
        # a RAG-layer abstain (e.g. a question that shares an in-vocabulary
        # keyword with an F-1 topic -- like "grace period" -- but is
        # actually off-topic, so it passes the keyword gate and correctly
        # finds no supporting context instead). Both are safe non-answers;
        # only a confident wrong answer is the failure mode this guards.
        correct = declined or abstained
        detail = "declined/abstained as expected" if correct else "did NOT decline an out-of-scope question"
        return {**base, "correct": correct, "detail": detail, "matched_keypoints": [], "missing_keypoints": []}

    if declined:
        return {
            **base,
            "correct": False,
            "detail": "incorrectly declined an in-scope question",
            "matched_keypoints": [],
            "missing_keypoints": question.get("expected_keypoints", []),
        }

    all_kw = question.get("expected_keypoints", [])
    any_kw = question.get("expected_keypoints_any", [])

    matched = [kw for kw in all_kw if _normalize(kw) in lower_text]
    missing = [kw for kw in all_kw if _normalize(kw) not in lower_text]
    any_matched = [kw for kw in any_kw if _normalize(kw) in lower_text]

    all_ok = len(missing) == 0
    any_ok = (len(any_kw) == 0) or (len(any_matched) > 0)
    correct = all_ok and any_ok

    detail_parts = []
    if missing:
        detail_parts.append(f"missing required: {missing}")
    if any_kw and not any_matched:
        detail_parts.append(f"missing any-of: {any_kw}")
    if abstained:
        detail_parts.append("model abstained")
    detail = "; ".join(detail_parts) if detail_parts else "all keypoints present"

    return {
        **base,
        "correct": correct,
        "detail": detail,
        "matched_keypoints": matched + any_matched,
        "missing_keypoints": missing,
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
        started = time.perf_counter()
        answer = app.get_answer(question["question"])
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        results.append(score_question(question, answer, latency_ms))
    return results


def _percentile(values: list, pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(len(ordered) * pct), len(ordered) - 1)
    return ordered[idx]


def summarize(results: list) -> dict:
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    by_category = {}
    for r in results:
        cat = r["category"]
        bucket = by_category.setdefault(cat, {"total": 0, "correct": 0})
        bucket["total"] += 1
        bucket["correct"] += 1 if r["correct"] else 0

    substantive = [r for r in results if r["category"] not in NON_SUBSTANTIVE_CATEGORIES]
    abstentions = sum(1 for r in substantive if r.get("abstained"))

    false_declines = sum(
        1 for r in results if r.get("declined") and r["category"] not in NON_SUBSTANTIVE_CATEGORIES
    )
    false_accepts = sum(1 for r in results if r["category"] == "out_of_scope" and not r.get("declined"))

    latencies = [r["latency_ms"] for r in results if r.get("latency_ms") is not None]

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
        "abstention_rate": round(abstentions / len(substantive), 4) if substantive else 0.0,
        "false_decline_count": false_declines,
        "false_accept_count": false_accepts,
        "latency_ms": {
            "avg": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            "p50": round(_percentile(latencies, 0.5), 1),
            "p95": round(_percentile(latencies, 0.95), 1),
            "max": round(max(latencies), 1) if latencies else 0.0,
        },
    }


def print_report(summary: dict, results: list, label: str):
    print(f"\n=== F1rstAid Eval Report: {label} ===")
    print(f"Overall: {summary['correct']}/{summary['total']} ({summary['accuracy']:.1%})")
    print(
        f"Abstention rate (in-scope questions): {summary['abstention_rate']:.1%}  |  "
        f"False declines: {summary['false_decline_count']}  |  "
        f"False accepts: {summary['false_accept_count']}"
    )
    lat = summary["latency_ms"]
    print(f"Latency ms: avg={lat['avg']} p50={lat['p50']} p95={lat['p95']} max={lat['max']}\n")

    print(f"{'Category':<28}{'Correct':>10}{'Total':>8}{'Accuracy':>12}")
    for cat, b in summary["by_category"].items():
        print(f"{cat:<28}{b['correct']:>10}{b['total']:>8}{b['accuracy']:>11.1%}")

    failures = [r for r in results if not r["correct"]]
    if failures:
        print(f"\n--- {len(failures)} failing question(s) ---")
        for r in failures:
            print(f"  [{r['id']}] {r['detail']}")


def append_history(history_path: Path, entry: dict) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(EVAL_DIR / "dataset.json"))
    parser.add_argument("--output-dir", default=str(EVAL_DIR / "results"))
    parser.add_argument("--history-path", default=str(EVAL_DIR / "results" / "history.jsonl"))
    parser.add_argument("--label", default="run", help="Short name for this run, used in the output filename")
    parser.add_argument("--note", default="", help="What changed for this run, e.g. 'switched to gpt-4o-mini'")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="Chat model name")
    parser.add_argument("--k", type=int, default=3, help="Retriever search_k")
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Defaults to 0 (not the app's production 0.2) for reproducible, comparable eval runs. "
             "Sampling noise at 0.2 was observed to flip 2-4 questions between identical-config runs.",
    )
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

    provenance = git_provenance()
    if provenance.get("dirty"):
        print(
            "\nWARNING: working tree has uncommitted changes -- this run's accuracy "
            "can't be reliably attributed to a specific commit.",
            file=sys.stderr,
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config_dict = asdict(config) if hasattr(config, "__dataclass_fields__") else vars(config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{timestamp}_{args.label}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label": args.label,
                "note": args.note,
                "timestamp": timestamp,
                "git": provenance,
                "config": config_dict,
                "dataset_version": dataset.get("version"),
                "summary": summary,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved: {out_path}")

    append_history(
        Path(args.history_path),
        {
            "timestamp": timestamp,
            "label": args.label,
            "note": args.note,
            "git": provenance,
            "config": config_dict,
            "summary": summary,
            "result_file": str(out_path.relative_to(REPO_ROOT)),
        },
    )
    print(f"Appended to: {args.history_path}")


if __name__ == "__main__":
    main()
