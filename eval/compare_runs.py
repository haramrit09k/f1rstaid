"""Diff two eval result JSON files (e.g. baseline vs. after a change).

Usage:
    python eval/compare_runs.py eval/results/<baseline>.json eval/results/<after>.json
"""

import argparse
import json


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()

    before = load(args.before)
    after = load(args.after)

    before_by_id = {r["id"]: r for r in before["results"]}
    after_by_id = {r["id"]: r for r in after["results"]}

    print(f"Before ({before['label']}): {before['summary']['correct']}/{before['summary']['total']} "
          f"({before['summary']['accuracy']:.1%})")
    print(f"After  ({after['label']}): {after['summary']['correct']}/{after['summary']['total']} "
          f"({after['summary']['accuracy']:.1%})")

    delta = after["summary"]["accuracy"] - before["summary"]["accuracy"]
    print(f"Delta: {delta:+.1%}\n")

    regressions = []
    improvements = []
    for qid, after_r in after_by_id.items():
        before_r = before_by_id.get(qid)
        if before_r is None:
            continue
        if before_r["correct"] and not after_r["correct"]:
            regressions.append((qid, before_r, after_r))
        elif not before_r["correct"] and after_r["correct"]:
            improvements.append((qid, before_r, after_r))

    if improvements:
        print(f"--- {len(improvements)} question(s) newly correct ---")
        for qid, _, after_r in improvements:
            print(f"  [{qid}] {after_r['detail']}")

    if regressions:
        print(f"\n--- {len(regressions)} REGRESSION(s): was correct, now failing ---")
        for qid, _, after_r in regressions:
            print(f"  [{qid}] {after_r['detail']}")

    if not improvements and not regressions:
        print("No per-question changes between runs.")


if __name__ == "__main__":
    main()
