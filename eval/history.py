"""Print the accuracy trend across all recorded eval runs.

Reads eval/results/history.jsonl (one line per run, appended by run_eval.py)
and prints a chronological table with the delta from the previous run, so
you can see whether a given change (model swap, k, prompt edit, etc.) moved
accuracy up or down -- and by how much.

Usage:
    python eval/history.py
    python eval/history.py --last 5
    python eval/history.py --category eligibility_multi_condition
"""

import argparse
import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent


def load_history(path: Path) -> list:
    if not path.exists():
        return []
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def fmt_delta(delta: float) -> str:
    if delta > 0:
        return f"+{delta:.1%}"
    if delta < 0:
        return f"{delta:.1%}"
    return "±0.0%"


def print_trend(entries: list, category: str = None):
    if not entries:
        print("No runs recorded yet. Run eval/run_eval.py first.")
        return

    header_metric = f"acc[{category}]" if category else "accuracy"
    print(f"{'Timestamp':<18}{'Label':<20}{'Commit':<10}{header_metric:>12}{'Delta':>10}{'Abstain':>10}{'p50 ms':>9}")

    prev_value = None
    for e in entries:
        summary = e["summary"]
        if category:
            cat_bucket = summary.get("by_category", {}).get(category)
            value = cat_bucket["accuracy"] if cat_bucket else None
        else:
            value = summary["accuracy"]

        commit = (e.get("git") or {}).get("commit") or "?"
        dirty = " *" if (e.get("git") or {}).get("dirty") else ""
        abstain = summary.get("abstention_rate")
        p50 = (summary.get("latency_ms") or {}).get("p50")

        value_str = f"{value:.1%}" if value is not None else "n/a"
        delta_str = fmt_delta(value - prev_value) if (value is not None and prev_value is not None) else ""
        abstain_str = f"{abstain:.1%}" if abstain is not None else "n/a"
        p50_str = f"{p50:.0f}" if p50 is not None else "n/a"

        print(
            f"{e['timestamp']:<18}{e['label']:<20}{commit + dirty:<10}"
            f"{value_str:>12}{delta_str:>10}{abstain_str:>10}{p50_str:>9}"
        )
        if e.get("note"):
            print(f"{'':<18}  note: {e['note']}")

        if value is not None:
            prev_value = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-path", default=str(EVAL_DIR / "results" / "history.jsonl"))
    parser.add_argument("--last", type=int, default=None, help="Only show the last N runs")
    parser.add_argument("--category", default=None, help="Track a single category's accuracy instead of overall")
    args = parser.parse_args()

    entries = load_history(Path(args.history_path))
    if args.last:
        entries = entries[-args.last:]
    print_trend(entries, category=args.category)


if __name__ == "__main__":
    main()
