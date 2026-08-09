"""Offline sanity checks for the eval harness itself (no API calls).

These validate dataset.json is well-formed and that score_question() scores
correctly -- they do NOT run the eval against a live model. See eval/README.md
for that.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.history import load_history, print_trend  # noqa: E402
from eval.run_eval import append_history, score_question, summarize  # noqa: E402

DATASET_PATH = Path(__file__).resolve().parent.parent / "eval" / "dataset.json"

REQUIRED_FIELDS = {"id", "category", "difficulty", "question", "must_decline"}
VALID_CATEGORIES = {
    "factual_lookup",
    "eligibility_multi_condition",
    "timeline_math",
    "edge_case",
    "out_of_scope",
    "help",
}


def load_dataset():
    with open(DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_dataset_is_well_formed():
    dataset = load_dataset()
    questions = dataset["questions"]
    assert len(questions) >= 20

    seen_ids = set()
    for q in questions:
        assert REQUIRED_FIELDS.issubset(q.keys()), f"{q.get('id')} missing required fields"
        assert q["id"] not in seen_ids, f"duplicate id {q['id']}"
        seen_ids.add(q["id"])
        assert q["category"] in VALID_CATEGORIES
        assert q["difficulty"] in {"easy", "medium", "hard"}
        if not q["must_decline"]:
            has_all = bool(q.get("expected_keypoints"))
            has_any = bool(q.get("expected_keypoints_any"))
            assert has_all or has_any, f"{q['id']} has no scorable expectation"


def test_dataset_has_all_categories_represented():
    dataset = load_dataset()
    categories = {q["category"] for q in dataset["questions"]}
    assert categories == VALID_CATEGORIES


def test_score_question_all_keypoints_present():
    question = {"id": "x", "category": "factual_lookup", "must_decline": False,
                "expected_keypoints": ["I-765"]}
    answer = {"result": "You need Form I-765 to apply."}
    result = score_question(question, answer)
    assert result["correct"] is True
    assert "I-765" in result["matched_keypoints"]


def test_score_question_missing_keypoint():
    question = {"id": "x", "category": "factual_lookup", "must_decline": False,
                "expected_keypoints": ["I-765"]}
    answer = {"result": "You need to fill out some paperwork."}
    result = score_question(question, answer)
    assert result["correct"] is False
    assert "I-765" in result["missing_keypoints"]


def test_score_question_any_of_semantics():
    question = {"id": "x", "category": "timeline_math", "must_decline": False,
                "expected_keypoints": [], "expected_keypoints_any": ["September 30", "October 1"]}
    assert score_question(question, {"result": "work is authorized until October 1."})["correct"] is True
    assert score_question(question, {"result": "no relevant date mentioned"})["correct"] is False


def test_score_question_out_of_scope_correctly_declined():
    question = {"id": "x", "category": "out_of_scope", "must_decline": True}
    answer = {"result": "🚦 **Relevance Check**\nThis doesn't appear to be an F-1 visa question."}
    result = score_question(question, answer)
    assert result["correct"] is True
    assert result["declined"] is True


def test_score_question_out_of_scope_not_declined_is_a_failure():
    question = {"id": "x", "category": "out_of_scope", "must_decline": True}
    answer = {"result": "Here is the weather forecast..."}
    result = score_question(question, answer)
    assert result["correct"] is False


def test_score_question_in_scope_incorrectly_declined_is_a_failure():
    question = {"id": "x", "category": "factual_lookup", "must_decline": False,
                "expected_keypoints": ["I-765"]}
    answer = {"result": "🚦 **Relevance Check**\nThis doesn't appear to be an F-1 visa question."}
    result = score_question(question, answer)
    assert result["correct"] is False
    assert result["declined"] is True


def test_score_question_detects_abstention():
    question = {"id": "x", "category": "factual_lookup", "must_decline": False,
                "expected_keypoints": ["I-765"]}
    answer = {"result": "I don't have enough information to answer this accurately -- "
                         "please consult your DSO."}
    result = score_question(question, answer)
    assert result["abstained"] is True
    assert result["correct"] is False


def test_score_question_records_latency():
    question = {"id": "x", "category": "factual_lookup", "must_decline": False,
                "expected_keypoints": ["I-765"]}
    result = score_question(question, {"result": "Form I-765."}, latency_ms=123.4)
    assert result["latency_ms"] == 123.4


def _result(category, correct, declined=False, abstained=False, latency_ms=100.0):
    return {
        "id": f"{category}-x",
        "category": category,
        "correct": correct,
        "declined": declined,
        "abstained": abstained,
        "latency_ms": latency_ms,
    }


def test_summarize_computes_per_category_accuracy():
    results = [
        _result("factual_lookup", True),
        _result("factual_lookup", False),
        _result("help", True),
    ]
    summary = summarize(results)
    assert summary["total"] == 3
    assert summary["correct"] == 2
    assert summary["by_category"]["factual_lookup"]["accuracy"] == 0.5
    assert summary["by_category"]["help"]["accuracy"] == 1.0


def test_summarize_computes_abstention_rate_excluding_out_of_scope_and_help():
    results = [
        _result("factual_lookup", True, abstained=False),
        _result("eligibility_multi_condition", False, abstained=True),
        _result("out_of_scope", True, declined=True, abstained=False),
        _result("help", True, abstained=False),
    ]
    summary = summarize(results)
    # Only 2 substantive (non out_of_scope/help) questions, 1 abstained.
    assert summary["abstention_rate"] == 0.5


def test_summarize_computes_false_decline_and_false_accept_counts():
    results = [
        _result("factual_lookup", False, declined=True),  # false decline
        _result("out_of_scope", False, declined=False),  # false accept
        _result("out_of_scope", True, declined=True),  # correctly declined
    ]
    summary = summarize(results)
    assert summary["false_decline_count"] == 1
    assert summary["false_accept_count"] == 1


def test_summarize_computes_latency_percentiles():
    results = [_result("factual_lookup", True, latency_ms=v) for v in [100, 200, 300, 400, 500]]
    summary = summarize(results)
    assert summary["latency_ms"]["avg"] == 300.0
    assert summary["latency_ms"]["max"] == 500.0
    assert summary["latency_ms"]["p50"] > 0


def test_append_history_writes_jsonl_line(tmp_path):
    history_path = tmp_path / "history.jsonl"
    append_history(history_path, {"timestamp": "t1", "label": "baseline", "summary": {"accuracy": 0.5}})
    append_history(history_path, {"timestamp": "t2", "label": "after-fix", "summary": {"accuracy": 0.7}})

    entries = load_history(history_path)
    assert len(entries) == 2
    assert entries[0]["label"] == "baseline"
    assert entries[1]["summary"]["accuracy"] == 0.7


def test_load_history_missing_file_returns_empty_list(tmp_path):
    assert load_history(tmp_path / "does_not_exist.jsonl") == []


def test_print_trend_does_not_raise_on_empty_or_populated_history(capsys):
    print_trend([])
    captured = capsys.readouterr()
    assert "No runs recorded" in captured.out

    entries = [
        {"timestamp": "t1", "label": "baseline", "git": {"commit": "abc123"},
         "summary": {"accuracy": 0.5, "abstention_rate": 0.1, "latency_ms": {"p50": 900}}},
        {"timestamp": "t2", "label": "gpt4o", "git": {"commit": "def456"}, "note": "swapped model",
         "summary": {"accuracy": 0.75, "abstention_rate": 0.05, "latency_ms": {"p50": 700}}},
    ]
    print_trend(entries)
    captured = capsys.readouterr()
    assert "50.0%" in captured.out
    assert "+25.0%" in captured.out
    assert "swapped model" in captured.out


def test_score_question_hyphen_and_space_variants_are_equivalent():
    question = {"id": "x", "category": "factual_lookup", "must_decline": False,
                "expected_keypoints": ["60 day"]}
    assert score_question(question, {"result": "a standard 60-day grace period applies."})["correct"] is True

    question_hyphenated = {"id": "y", "category": "factual_lookup", "must_decline": False,
                            "expected_keypoints": ["60-day"]}
    assert score_question(question_hyphenated, {"result": "you get 60 days total."})["correct"] is True
