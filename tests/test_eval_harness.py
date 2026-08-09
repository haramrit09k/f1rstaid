"""Offline sanity checks for the eval harness itself (no API calls).

These validate dataset.json is well-formed and that score_question() scores
correctly -- they do NOT run the eval against a live model. See eval/README.md
for that.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.run_eval import score_question, summarize  # noqa: E402

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


def test_summarize_computes_per_category_accuracy():
    results = [
        {"id": "a", "category": "factual_lookup", "correct": True},
        {"id": "b", "category": "factual_lookup", "correct": False},
        {"id": "c", "category": "help", "correct": True},
    ]
    summary = summarize(results)
    assert summary["total"] == 3
    assert summary["correct"] == 2
    assert summary["by_category"]["factual_lookup"]["accuracy"] == 0.5
    assert summary["by_category"]["help"]["accuracy"] == 1.0
