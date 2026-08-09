"""Tests for rules_engine.py. compute_answer/_general_rule_answer are pure
logic, tested directly with no mocking. match_and_answer needs an LLM for
extraction, mocked here so these run with no API calls/cost -- live behavior
against the real LLM was separately smoke-tested by hand during development
(see the project plan / commit history for elig-001/elig-002/time-003/
time-005 transcripts)."""

from rules_engine import (
    UnemploymentFields,
    _general_rule_answer,
    _is_trigger_match,
    compute_answer,
    match_and_answer,
)


class FakeExtractor:
    def __init__(self, result):
        self._result = result

    def invoke(self, prompt):
        return self._result


class FakeLLM:
    """Stands in for a ChatOpenAI instance -- only needs to support
    .with_structured_output(...).invoke(...), which is all rules_engine
    calls on the real llm."""

    def __init__(self, result):
        self._result = result

    def with_structured_output(self, model):
        return FakeExtractor(self._result)


# --- compute_answer: pure logic, exhaustive ---

def test_compute_answer_initial_opt_under_cap():
    fields = UnemploymentFields(applies=True, opt_phase="initial_opt", unemployment_days_used=60)
    result = compute_answer(fields)
    assert "90" in result
    assert "30" in result  # 90 - 60 remaining
    assert "exceeds" not in result.lower()


def test_compute_answer_initial_opt_at_cap_exactly():
    fields = UnemploymentFields(applies=True, opt_phase="initial_opt", unemployment_days_used=90)
    result = compute_answer(fields)
    # Exactly at the cap is not yet "exceeds" (used > cap is strictly greater)
    assert "0 days remaining" in result
    assert "exceeds" not in result.lower()


def test_compute_answer_initial_opt_over_cap():
    fields = UnemploymentFields(applies=True, opt_phase="initial_opt", unemployment_days_used=95)
    result = compute_answer(fields)
    assert "exceeds" in result.lower()
    assert "90" in result
    assert "DSO" in result


def test_compute_answer_stem_extension_under_cap():
    fields = UnemploymentFields(applies=True, opt_phase="stem_opt_extension", unemployment_days_used=60)
    result = compute_answer(fields)
    assert "150" in result
    assert "90 days remaining" in result  # 150 - 60


def test_compute_answer_stem_extension_over_cap():
    fields = UnemploymentFields(applies=True, opt_phase="stem_opt_extension", unemployment_days_used=160)
    result = compute_answer(fields)
    assert "exceeds" in result.lower()
    assert "150" in result
    assert "DSO" in result


# --- _general_rule_answer: no personal data, always safely answerable ---

def test_general_rule_answer_no_phase_mentions_both_caps():
    result = _general_rule_answer(None)
    assert "90" in result
    assert "150" in result


def test_general_rule_answer_initial_opt_phase():
    result = _general_rule_answer("initial_opt")
    assert "90" in result
    assert "150" not in result


def test_general_rule_answer_stem_phase():
    result = _general_rule_answer("stem_opt_extension")
    assert "150" in result


# --- trigger matching ---

def test_trigger_matches_unemployment_phrasing():
    assert _is_trigger_match("I've been unemployed for 95 days") is True


def test_trigger_does_not_match_unrelated_question():
    assert _is_trigger_match("What form do I use to apply for OPT?") is False


# --- match_and_answer: the three branches, mocked LLM ---

def test_match_and_answer_no_trigger_falls_through_to_rag():
    result = match_and_answer("What form do I use to apply for OPT?", llm=FakeLLM(None))
    assert result is None


def test_match_and_answer_no_days_stated_gives_general_rule():
    fake_fields = UnemploymentFields(applies=True, opt_phase="initial_opt", unemployment_days_used=None)
    result = match_and_answer(
        "How many unemployment days am I allowed during initial OPT?",
        llm=FakeLLM(fake_fields),
    )
    assert result is not None
    assert "90" in result["result"]
    assert result["source_documents"] == []


def test_match_and_answer_days_stated_but_phase_missing_asks_clarifying_question():
    fake_fields = UnemploymentFields(applies=True, opt_phase=None, unemployment_days_used=95)
    result = match_and_answer(
        "I've been unemployed for 95 days, am I okay?",
        llm=FakeLLM(fake_fields),
    )
    assert result is not None
    assert "initial" in result["result"].lower() or "stem" in result["result"].lower()
    assert result["source_documents"] == []


def test_match_and_answer_complete_fields_computes_answer():
    fake_fields = UnemploymentFields(applies=True, opt_phase="initial_opt", unemployment_days_used=95)
    result = match_and_answer(
        "I'm on initial OPT and have been unemployed for 95 days, am I okay?",
        llm=FakeLLM(fake_fields),
    )
    assert result is not None
    assert "exceeds" in result["result"].lower()
    assert result["source_documents"] == []


def test_match_and_answer_trigger_hit_but_extraction_says_not_applicable_falls_through():
    """Trigger word present ('unemployment') but extraction correctly judges
    this is a different topic (e.g. insurance benefits, not the day-count
    status rule) -- must fall through to RAG, not answer with the
    irrelevant general-rule text. This is exactly the false-positive
    pattern the edge-005/edge-006 eval questions caught."""
    fake_fields = UnemploymentFields(applies=False, opt_phase=None, unemployment_days_used=None)
    result = match_and_answer(
        "Can I collect unemployment insurance benefits as an F-1 student on OPT?",
        llm=FakeLLM(fake_fields),
    )
    assert result is None


def test_match_and_answer_extraction_failure_falls_through_to_rag():
    class BrokenExtractor:
        def invoke(self, prompt):
            raise RuntimeError("simulated extraction failure")

    class BrokenLLM:
        def with_structured_output(self, model):
            return BrokenExtractor()

    result = match_and_answer("I've been unemployed for 95 days", llm=BrokenLLM())
    assert result is None
