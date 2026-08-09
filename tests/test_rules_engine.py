"""Tests for rules_engine.py. compute_answer/_general_rule_answer are pure
logic, tested directly with no mocking. match_and_answer needs an LLM for
extraction, mocked here so these run with no API calls/cost -- live behavior
against the real LLM was separately smoke-tested by hand during development
(see the project plan / commit history for elig-001/elig-002/time-003/
time-005 transcripts)."""

from rules_engine import (
    CapGapFields,
    DegreeListFields,
    GracePeriodFields,
    UnemploymentFields,
    _general_cap_gap_answer,
    _general_degree_list_answer,
    _general_grace_period_answer,
    _general_rule_answer,
    _is_cap_gap_trigger_match,
    _is_degree_list_trigger_match,
    _is_grace_period_trigger_match,
    _is_trigger_match,
    compute_answer,
    compute_cap_gap_answer,
    compute_degree_list_answer,
    compute_grace_period_answer,
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


# --- compute_grace_period_answer: pure date arithmetic ---

def test_compute_grace_period_no_year_stays_in_same_year():
    fields = GracePeriodFields(applies=True, end_month=5, end_day=15)
    result = compute_grace_period_answer(fields)
    assert "May 15" in result
    assert "July 14" in result  # May 15 + 60 days


def test_compute_grace_period_wraps_into_next_year():
    fields = GracePeriodFields(applies=True, end_month=11, end_day=15, end_year=2026)
    result = compute_grace_period_answer(fields)
    assert "November 15, 2026" in result
    assert "January 14, 2027" in result


def test_compute_grace_period_no_wrap_keeps_same_stated_year():
    fields = GracePeriodFields(applies=True, end_month=3, end_day=1, end_year=2026)
    result = compute_grace_period_answer(fields)
    assert "March 1, 2026" in result
    assert "April 30, 2026" in result


# --- grace period trigger matching ---

def test_grace_period_trigger_matches():
    assert _is_grace_period_trigger_match("What's my grace period after I graduate?") is True


def test_grace_period_trigger_does_not_match_unrelated_question():
    assert _is_grace_period_trigger_match("What form do I use to apply for OPT?") is False


# --- match_and_answer: grace period branches ---

def test_match_and_answer_grace_period_no_date_gives_general_rule():
    fake_fields = GracePeriodFields(applies=True, end_month=None, end_day=None)
    result = match_and_answer(
        "How long is the grace period after my program ends?",
        llm=FakeLLM(fake_fields),
    )
    assert result is not None
    assert "60-day" in result["result"]
    assert result["source_documents"] == []


def test_match_and_answer_grace_period_with_date_computes_answer():
    fake_fields = GracePeriodFields(applies=True, end_month=5, end_day=15)
    result = match_and_answer(
        "My F-1 program end date is May 15. By what date must I leave the US "
        "under the standard grace period?",
        llm=FakeLLM(fake_fields),
    )
    assert result is not None
    assert "July 14" in result["result"]
    assert result["source_documents"] == []


def test_match_and_answer_grace_period_false_positive_falls_through():
    """Trigger word present ('grace period') but extraction correctly judges
    this isn't about the F-1 grace period at all (e.g. a billing grace
    period) -- must fall through to RAG."""
    fake_fields = GracePeriodFields(applies=False)
    result = match_and_answer(
        "Is there a grace period for paying my credit card bill late?",
        llm=FakeLLM(fake_fields),
    )
    assert result is None


# --- compute_cap_gap_answer: exhaustive branches ---

def test_compute_cap_gap_filed_on_time_not_denied_gives_general_answer():
    fields = CapGapFields(applies=True, filed_on_time=True, petition_denied=False)
    result = compute_cap_gap_answer(fields)
    assert "September 30" in result
    assert "denied" in result.lower()  # caveat mentioned


def test_compute_cap_gap_denied_overrides_everything():
    fields = CapGapFields(applies=True, filed_on_time=True, petition_denied=True)
    result = compute_cap_gap_answer(fields)
    assert "denied" in result.lower()
    assert "grace period" in result.lower()
    assert "September 30" not in result


def test_compute_cap_gap_not_filed_on_time():
    fields = CapGapFields(applies=True, filed_on_time=False, petition_denied=None)
    result = compute_cap_gap_answer(fields)
    assert "doesn't sound like it applies" in result


# --- cap-gap trigger matching ---

def test_cap_gap_trigger_matches():
    assert _is_cap_gap_trigger_match("What happens to my status under cap-gap?") is True


def test_cap_gap_trigger_does_not_match_unrelated_question():
    assert _is_cap_gap_trigger_match("What form do I use to apply for OPT?") is False


# --- match_and_answer: cap-gap branches ---

def test_match_and_answer_cap_gap_no_facts_gives_general_rule():
    fake_fields = CapGapFields(applies=True, filed_on_time=None, petition_denied=None)
    result = match_and_answer(
        "How does cap-gap work?",
        llm=FakeLLM(fake_fields),
    )
    assert result is not None
    assert "September 30" in result["result"]
    assert result["source_documents"] == []


def test_match_and_answer_cap_gap_false_positive_falls_through():
    """Trigger word present ('cap-gap') but extraction correctly judges this
    doesn't actually describe the cap-gap scenario -- must fall through to
    RAG rather than answering with the general cap-gap text."""
    fake_fields = CapGapFields(applies=False, filed_on_time=None, petition_denied=None)
    result = match_and_answer(
        "I got an H-1B job offer but haven't filed a petition yet -- is "
        "there a cap-gap for that?",
        llm=FakeLLM(fake_fields),
    )
    assert result is None


# --- rule-family ordering: only the matching family's extractor fires ---

def test_match_and_answer_tries_families_in_order_and_stops_at_first_match():
    """Unemployment trigger doesn't match, grace-period trigger doesn't
    match, cap-gap trigger does -- confirms the dispatcher actually walks
    all three families rather than only ever trying the first."""
    fake_fields = CapGapFields(applies=True, filed_on_time=True, petition_denied=False)
    result = match_and_answer(
        "I'm under cap-gap -- how long am I authorized to work?",
        llm=FakeLLM(fake_fields),
    )
    assert result is not None
    assert "September 30" in result["result"]


# --- compute_degree_list_answer: exhaustive branches ---

def test_compute_degree_list_not_on_list():
    fields = DegreeListFields(applies=True, degree_on_list=False)
    result = compute_degree_list_answer(fields)
    assert result.startswith("No")
    assert "10 years" in result  # prior-qualifying-degree caveat mentioned


def test_compute_degree_list_on_list():
    fields = DegreeListFields(applies=True, degree_on_list=True)
    result = compute_degree_list_answer(fields)
    assert "E-Verify" in result
    assert "I-983" in result


def test_general_degree_list_answer_mentions_the_list():
    result = _general_degree_list_answer()
    assert "STEM Designated Degree Program List" in result
    assert "CIP code" in result


# --- degree-list trigger matching ---

def test_degree_list_trigger_matches():
    assert _is_degree_list_trigger_match("Is my major on the STEM designated degree list?") is True


def test_degree_list_trigger_does_not_match_unrelated_question():
    assert _is_degree_list_trigger_match("What form do I use to apply for OPT?") is False


# --- match_and_answer: degree-list branches ---

def test_match_and_answer_degree_list_not_on_list():
    fake_fields = DegreeListFields(applies=True, degree_on_list=False)
    result = match_and_answer(
        "Can I apply for a STEM OPT extension if my degree is not on the "
        "DHS STEM Designated Degree Program list?",
        llm=FakeLLM(fake_fields),
    )
    assert result is not None
    assert "STEM Designated Degree Program List" in result["result"]
    assert result["source_documents"] == []


def test_match_and_answer_degree_list_no_facts_gives_general_rule():
    fake_fields = DegreeListFields(applies=True, degree_on_list=None)
    result = match_and_answer(
        "How does the STEM designated degree program list work?",
        llm=FakeLLM(fake_fields),
    )
    assert result is not None
    assert "STEM Designated Degree Program List" in result["result"]


def test_match_and_answer_degree_list_false_positive_falls_through():
    fake_fields = DegreeListFields(applies=False, degree_on_list=None)
    result = match_and_answer(
        "What's the CIP code format supposed to look like on any I-20?",
        llm=FakeLLM(fake_fields),
    )
    assert result is None
