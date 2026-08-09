import json
from datetime import date

import httpx
import pytest
from openai import RateLimitError

import f1rstaid
from f1rstaid import (
    DAILY_SHARED_KEY_LIMIT,
    FREE_TRIAL_QUERY_LIMIT,
    AppConfig,
    CONDENSE_HISTORY_TURNS,
    F1rstAidApp,
    RelevanceFields,
    _increment_daily_shared_key_usage,
    _read_daily_shared_key_usage,
    classify_answer,
    shared_key_limit_reached,
)
from rules_engine import UnemploymentFields


def _fake_rate_limit_error() -> RateLimitError:
    response = httpx.Response(
        status_code=429,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )
    return RateLimitError("insufficient_quota", response=response, body=None)


class _FakeLLMResponse:
    def __init__(self, content):
        self.content = content


class _FakeChatLLM:
    """Mimics ChatOpenAI's .invoke(prompt) -> response.content shape used
    by condense_question(), with no real API call. Records the last prompt
    sent so tests can inspect what history actually made it in."""

    def __init__(self, response_text=None, raise_error=False):
        self._response_text = response_text
        self._raise_error = raise_error
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        if self._raise_error:
            raise RuntimeError("simulated LLM failure")
        return _FakeLLMResponse(self._response_text)


class _FakeExtractor:
    def __init__(self, result):
        self._result = result

    def invoke(self, prompt):
        return self._result


class _FakeStructuredLLM:
    """Mimics ChatOpenAI's .with_structured_output(model).invoke(...) shape
    used by rules_engine's extraction calls -- no real API call."""

    def __init__(self, result):
        self._result = result

    def with_structured_output(self, model):
        return _FakeExtractor(self._result)


class _BrokenExtractor:
    def invoke(self, prompt):
        raise RuntimeError("simulated relevance-check LLM failure")


class _BrokenStructuredLLM:
    def with_structured_output(self, model):
        return _BrokenExtractor()


# --- _is_relevant_question: two-stage keyword + LLM-judged check ---

def test_relevance_check_no_keyword_match_declines_without_calling_llm():
    """A question with zero F1 keywords must decline at the free,
    keyword-only stage -- the LLM should never even be consulted. Proven
    by giving it an LLM that would raise if called at all."""
    app = F1rstAidApp(AppConfig())
    app.llm = _BrokenStructuredLLM()

    relevant, explanation = app._is_relevant_question("What's the weather like today?")

    assert relevant is False
    assert "doesn't appear to be an F-1 visa question" in explanation


def test_relevance_check_keyword_match_llm_confirms_relevant():
    app = F1rstAidApp(AppConfig())
    app.llm = _FakeStructuredLLM(RelevanceFields(applies=True))

    relevant, explanation = app._is_relevant_question("How long does OPT processing take?")

    assert relevant is True


def test_relevance_check_keyword_match_but_llm_says_not_applicable_declines():
    """The real bug this was built for: 'on F-1 OPT, can we invest in Roth
    IRA?' contains 'F-1' and 'OPT' (passes the keyword stage) but is really
    a tax/retirement-account question with no supporting content in this
    app's knowledge base at all."""
    app = F1rstAidApp(AppConfig())
    app.llm = _FakeStructuredLLM(RelevanceFields(applies=False))

    relevant, explanation = app._is_relevant_question(
        "on F-1 OPT, can we invest in Roth IRA?"
    )

    assert relevant is False
    assert "doesn't appear to be an F-1 visa question" in explanation


def test_relevance_check_llm_failure_falls_back_to_keyword_match():
    """The LLM stage is an enhancement, not a requirement -- a failure here
    must never be the reason a genuinely relevant question goes unanswered."""
    app = F1rstAidApp(AppConfig())
    app.llm = _BrokenStructuredLLM()

    relevant, explanation = app._is_relevant_question("How long does OPT processing take?")

    assert relevant is True


def test_relevance_check_no_llm_set_falls_back_to_keyword_match():
    """Mirrors how tests construct F1rstAidApp without calling initialize()
    -- self.llm stays None, and the check must still work via stage 1 alone."""
    app = F1rstAidApp(AppConfig())
    assert app.llm is None

    relevant, _ = app._is_relevant_question("How long does OPT processing take?")

    assert relevant is True


# --- condense_question: pure logic against a fake LLM, no API calls ---

def test_condense_question_no_history_returns_original_without_calling_llm():
    app = F1rstAidApp(AppConfig())
    fake_llm = _FakeChatLLM("should never be used")
    app.llm = fake_llm

    result = app.condense_question("What about STEM OPT?", [])

    assert result == "What about STEM OPT?"
    assert fake_llm.last_prompt is None  # never called -- no history to condense from


def test_condense_question_rewrites_using_history():
    app = F1rstAidApp(AppConfig())
    fake_llm = _FakeChatLLM(
        "I used 60 days of unemployment on initial OPT -- how many days do I "
        "have left if I move to the STEM OPT extension?"
    )
    app.llm = fake_llm
    history = [
        {"role": "user", "content": "I used 60 days of unemployment on initial OPT"},
        {"role": "assistant", "content": "You have 30 days remaining..."},
    ]

    result = app.condense_question("What about STEM OPT?", history)

    assert "STEM OPT" in result
    assert fake_llm.last_prompt is not None
    assert "60 days" in fake_llm.last_prompt  # history text reached the prompt
    assert "What about STEM OPT?" in fake_llm.last_prompt  # so did the follow-up


def test_condense_question_bounds_history_to_recent_turns():
    app = F1rstAidApp(AppConfig())
    fake_llm = _FakeChatLLM("rewritten")
    app.llm = fake_llm

    total_turns = CONDENSE_HISTORY_TURNS + 4
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn-{i}"}
        for i in range(total_turns)
    ]

    app.condense_question("follow-up", history)

    prompt = fake_llm.last_prompt
    assert "turn-0" not in prompt  # earliest turns excluded
    assert f"turn-{total_turns - 1}" in prompt  # most recent turn included


def test_condense_question_falls_back_to_original_on_llm_failure():
    app = F1rstAidApp(AppConfig())
    app.llm = _FakeChatLLM(raise_error=True)
    history = [{"role": "user", "content": "..."}]

    result = app.condense_question("What about STEM OPT?", history)

    assert result == "What about STEM OPT?"


# --- get_answer(chat_history=...): condensation wiring, no API calls ---

def test_get_answer_no_chat_history_skips_condensing():
    """Default/no chat_history must behave identically to before this
    feature existed -- the eval harness and every existing single-turn
    caller depend on that."""
    app = F1rstAidApp(AppConfig())
    calls = []
    app.condense_question = lambda q, h: calls.append((q, h)) or "SHOULD NOT BE USED"

    result = app.get_answer("What's the weather today?")

    assert calls == []
    assert "doesn't appear to be an F-1 visa question" in result["result"]


def test_get_answer_empty_chat_history_list_skips_condensing():
    app = F1rstAidApp(AppConfig())
    calls = []
    app.condense_question = lambda q, h: calls.append((q, h)) or "SHOULD NOT BE USED"

    app.get_answer("What's the weather today?", chat_history=[])

    assert calls == []


def test_get_answer_help_question_checked_before_condensing():
    """A mid-conversation 'help' should still hit its canned trigger
    directly, not risk being paraphrased by condensation first."""
    app = F1rstAidApp(AppConfig())
    calls = []
    app.condense_question = lambda q, h: calls.append((q, h)) or "SHOULD NOT BE USED"

    result = app.get_answer("help", chat_history=[{"role": "user", "content": "hi"}])

    assert calls == []
    assert "My Expertise" in result["result"]


def test_get_answer_uses_condensed_question_for_downstream_routing():
    """The condensed question -- not the original -- must be what actually
    reaches rules_engine/relevance checking. Proven here by having the fake
    condenser return text with a real rules_engine trigger word that the
    original question doesn't contain, then checking the result reflects
    the rule firing rather than a decline."""
    app = F1rstAidApp(AppConfig())
    app.llm = _FakeStructuredLLM(
        UnemploymentFields(applies=True, opt_phase="initial_opt", unemployment_days_used=None)
    )
    calls = []

    def fake_condense(question, chat_history):
        calls.append((question, chat_history))
        return "How many unemployment days am I allowed during initial OPT?"

    app.condense_question = fake_condense
    history = [{"role": "user", "content": "I'm on initial OPT"}]

    result = app.get_answer("what about that?", chat_history=history)

    assert calls == [("what about that?", history)]
    assert result is not None
    assert "90" in result["result"]


@pytest.mark.live
def test_initialize(app_config):
    """Test application initialization."""
    app = F1rstAidApp(app_config)
    assert app.initialize() is True
    assert app.embeddings is not None
    assert app.db is not None
    assert app.qa_chain is not None

def test_relevance_check(app_config):
    """Test question relevance checking."""
    app = F1rstAidApp(app_config)
    
    # Test relevant question
    relevant, _ = app._is_relevant_question("How long does OPT processing take?")
    assert relevant is True
    
    # Test irrelevant question
    relevant, _ = app._is_relevant_question("What's the weather like today?")
    assert relevant is False

@pytest.mark.live
def test_get_answer(app_config):
    """Test answer generation."""
    app = F1rstAidApp(app_config)
    app.initialize()
    
    # Test help question
    answer = app.get_answer("help")
    assert "My Expertise" in answer["result"]
    assert len(answer["source_documents"]) == 0
    
    # Test relevant question
    answer = app.get_answer("What is OPT?")
    assert answer is not None
    assert "result" in answer
    assert "source_documents" in answer


# --- get_answer: RateLimitError gets a distinct message, not the generic one ---

def test_get_answer_rate_limit_error_gets_distinct_message():
    """A genuine OpenAI quota/rate-limit error (HTTP 429) must be
    distinguishable from a generic processing error -- the shared-key usage
    caps are a soft, best-effort limit (see shared_key_limit_reached()'s
    docstring), so a real quota exhaustion getting through them is exactly
    the failure mode that needs an honest, actionable message."""

    class _RateLimitedQAChain:
        def invoke(self, *args, **kwargs):
            raise _fake_rate_limit_error()

    app = F1rstAidApp(AppConfig())
    app.llm = _FakeStructuredLLM(RelevanceFields(applies=True))
    app.qa_chain = _RateLimitedQAChain()

    result = app.get_answer("What is OPT?")

    assert "hit its OpenAI usage limit" in result["result"]
    assert classify_answer(result) == "rate_limited"


# --- daily shared-key usage counter: pure file I/O, isolated per test ---

def test_daily_usage_counter_persists_and_increments(tmp_path, monkeypatch):
    tracker_path = tmp_path / "usage_tracker.json"
    monkeypatch.setattr(f1rstaid, "USAGE_TRACKER_PATH", tracker_path)

    assert _read_daily_shared_key_usage() == 0
    assert _increment_daily_shared_key_usage() == 1
    assert _increment_daily_shared_key_usage() == 2
    assert _read_daily_shared_key_usage() == 2

    saved = json.loads(tracker_path.read_text())
    assert saved["count"] == 2


def test_daily_usage_counter_resets_on_a_new_date(tmp_path, monkeypatch):
    tracker_path = tmp_path / "usage_tracker.json"
    tracker_path.write_text(json.dumps({"date": "2000-01-01", "count": 999}))
    monkeypatch.setattr(f1rstaid, "USAGE_TRACKER_PATH", tracker_path)

    assert _read_daily_shared_key_usage() == 0  # stale date -- doesn't count


def test_daily_usage_counter_handles_missing_or_corrupt_file(tmp_path, monkeypatch):
    monkeypatch.setattr(f1rstaid, "USAGE_TRACKER_PATH", tmp_path / "does_not_exist.json")
    assert _read_daily_shared_key_usage() == 0

    corrupt_path = tmp_path / "corrupt.json"
    corrupt_path.write_text("not valid json{{{")
    monkeypatch.setattr(f1rstaid, "USAGE_TRACKER_PATH", corrupt_path)
    assert _read_daily_shared_key_usage() == 0


# --- shared_key_limit_reached: session + daily checks together ---

def test_shared_key_limit_not_reached_when_under_both_limits(monkeypatch, tmp_path):
    import streamlit as st

    monkeypatch.setattr(f1rstaid, "USAGE_TRACKER_PATH", tmp_path / "usage.json")
    st.session_state["shared_key_query_count"] = 0

    limited, message = shared_key_limit_reached()

    assert limited is False
    assert message == ""


def test_shared_key_limit_reached_by_session_count(monkeypatch, tmp_path):
    import streamlit as st

    monkeypatch.setattr(f1rstaid, "USAGE_TRACKER_PATH", tmp_path / "usage.json")
    st.session_state["shared_key_query_count"] = FREE_TRIAL_QUERY_LIMIT

    limited, message = shared_key_limit_reached()

    assert limited is True
    assert "free trial questions" in message


def test_shared_key_limit_reached_by_daily_cap(monkeypatch, tmp_path):
    import streamlit as st

    tracker_path = tmp_path / "usage.json"
    tracker_path.write_text(
        json.dumps({"date": str(date.today()), "count": DAILY_SHARED_KEY_LIMIT})
    )
    monkeypatch.setattr(f1rstaid, "USAGE_TRACKER_PATH", tracker_path)
    st.session_state["shared_key_query_count"] = 0  # under the per-session limit

    limited, message = shared_key_limit_reached()

    assert limited is True
    assert "shared free-trial usage limit for today" in message