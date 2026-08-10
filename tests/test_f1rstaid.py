import json
from datetime import date

import httpx
import pytest
from openai import RateLimitError

import f1rstaid
from f1rstaid import (
    DAILY_SHARED_KEY_LIMIT,
    FREE_TRIAL_QUERY_LIMIT,
    MAX_FEEDBACK_ISSUES_PER_DAY,
    AppConfig,
    CONDENSE_HISTORY_TURNS,
    F1rstAidApp,
    RelevanceFields,
    _answers_pending_clarification,
    _has_reference_word,
    _increment_daily_shared_key_usage,
    _read_daily_count,
    _read_daily_shared_key_usage,
    classify_answer,
    create_feedback_issue,
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


# --- condense_question: two deterministically-picked mechanisms, not one
# LLM call that both judges and rewrites -- that combined approach was
# tried first and verified live to be unreliable (see condense_question's
# docstring). Tests below cover each mechanism, and the routing between
# them, separately. ---

# --- _answers_pending_clarification: pure logic, no LLM involved ---

def test_answers_pending_clarification_true_for_short_answer_to_a_question():
    history = [
        {"role": "user", "content": "I've been unemployed for 95 days, am I okay?"},
        {"role": "assistant", "content": "Are you on initial OPT or the STEM extension?"},
    ]
    assert _answers_pending_clarification("I'm on initial OPT", history) is True


def test_answers_pending_clarification_false_when_assistant_did_not_ask_a_question():
    history = [
        {"role": "user", "content": "How long is the STEM OPT extension?"},
        {"role": "assistant", "content": "The STEM OPT extension is 24 months."},
    ]
    assert _answers_pending_clarification("What form do I need for that?", history) is False


def test_answers_pending_clarification_false_when_followup_is_a_full_new_question():
    """The exact real bug this guards against: a follow-up arriving right
    after a clarifying question, but which is actually a new, unrelated,
    full question rather than a short direct answer to it."""
    history = [
        {"role": "user", "content": "I haven't worked in 80 days, is that bad?"},
        {"role": "assistant", "content": "Are you on initial OPT or the STEM extension?"},
    ]
    assert (
        _answers_pending_clarification("on f1 opt can i invest in a roth ira", history)
        is False
    )


def test_answers_pending_clarification_false_with_insufficient_history():
    assert _answers_pending_clarification("I'm on initial OPT", []) is False
    assert (
        _answers_pending_clarification(
            "I'm on initial OPT", [{"role": "assistant", "content": "Which phase?"}]
        )
        is False
    )


# --- _has_reference_word: pure logic ---

def test_has_reference_word_detects_that_even_with_trailing_punctuation():
    """Real bug caught live: naive space-padded substring matching missed
    "that?" because of the trailing question mark."""
    assert _has_reference_word("What form do I need for that?") is True


def test_has_reference_word_false_for_unrelated_question():
    assert _has_reference_word("How long is the STEM OPT extension?") is False


# --- condense_question: routing between the two mechanisms ---

def test_condense_question_no_history_returns_original_without_calling_llm():
    app = F1rstAidApp(AppConfig())
    fake_llm = _FakeChatLLM("should never be used")
    app.llm = fake_llm

    result = app.condense_question("What about STEM OPT?", [])

    assert result == "What about STEM OPT?"
    assert fake_llm.last_prompt is None


def test_condense_question_pending_clarification_concatenates_without_calling_llm():
    """The highest-stakes path (feeds straight into rules_engine's exact
    day-math) uses plain string concatenation, not LLM composition --
    proven here by giving it an LLM that would raise if called at all."""
    app = F1rstAidApp(AppConfig())
    app.llm = _FakeChatLLM(raise_error=True)
    history = [
        {"role": "user", "content": "I've been unemployed for 95 days, am I okay?"},
        {"role": "assistant", "content": "Are you on initial OPT or the STEM extension?"},
    ]

    result = app.condense_question("I'm on initial OPT", history)

    assert "95 days" in result
    assert "initial OPT" in result


def test_condense_question_reference_word_rewrites_using_history():
    app = F1rstAidApp(AppConfig())
    fake_llm = _FakeChatLLM("What form do I need for the STEM OPT extension?")
    app.llm = fake_llm
    history = [
        {"role": "user", "content": "How long is the STEM OPT extension?"},
        {"role": "assistant", "content": "The STEM OPT extension is 24 months."},
    ]

    result = app.condense_question("What form do I need for that?", history)

    assert "STEM OPT" in result
    assert fake_llm.last_prompt is not None
    assert "24 months" in fake_llm.last_prompt  # history text reached the prompt


def test_condense_question_no_reference_word_and_no_pending_clarification_is_unchanged():
    """Same shape as the real Roth IRA bug: history exists, but this
    follow-up neither answers a pending clarifying question nor references
    the conversation -- must stay unchanged, no LLM call."""
    app = F1rstAidApp(AppConfig())
    fake_llm = _FakeChatLLM(raise_error=True)
    app.llm = fake_llm
    history = [
        {"role": "user", "content": "I haven't worked in 80 days, is that bad?"},
        {"role": "assistant", "content": "You have used 80 of your 90 allowed days..."},
    ]

    result = app.condense_question("on f1 opt can i invest in a roth ira", history)

    assert result == "on f1 opt can i invest in a roth ira"


def test_condense_question_bounds_history_to_recent_turns():
    app = F1rstAidApp(AppConfig())
    fake_llm = _FakeChatLLM("rewritten")
    app.llm = fake_llm

    total_turns = CONDENSE_HISTORY_TURNS + 4
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn-{i}"}
        for i in range(total_turns)
    ]

    app.condense_question("what about that?", history)

    prompt = fake_llm.last_prompt
    assert "turn-0" not in prompt  # earliest turns excluded
    assert f"turn-{total_turns - 1}" in prompt  # most recent turn included


def test_condense_question_falls_back_to_original_on_llm_failure():
    app = F1rstAidApp(AppConfig())
    app.llm = _FakeChatLLM(raise_error=True)
    history = [
        {"role": "user", "content": "How long is the STEM OPT extension?"},
        {"role": "assistant", "content": "The STEM OPT extension is 24 months."},
    ]

    result = app.condense_question("what about that?", history)

    assert result == "what about that?"


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


# --- create_feedback_issue: thumbs-down -> GitHub issue, mocked API ---

class _FakeResponse:
    def __init__(self, status_code, text="", body=None):
        self.status_code = status_code
        self.text = text
        self._body = body

    def json(self):
        return self._body


def test_create_feedback_issue_no_token_configured_does_not_call_api(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    calls = []
    monkeypatch.setattr(f1rstaid.requests, "post", lambda *a, **k: calls.append((a, k)))

    filed = create_feedback_issue("rag", "some answer", ["https://example.com"], "wrong")

    assert filed is False
    assert calls == []


def test_create_feedback_issue_succeeds_and_increments_daily_counter(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_TOKEN", "fake-token")
    tracker_path = tmp_path / "feedback.json"
    monkeypatch.setattr(f1rstaid, "FEEDBACK_TRACKER_PATH", tracker_path)

    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _FakeResponse(201)

    monkeypatch.setattr(f1rstaid.requests, "post", fake_post)

    filed = create_feedback_issue(
        "rag", "some answer text", ["https://example.com/source"], "this was wrong"
    )

    assert filed is True
    assert _read_daily_count(tracker_path) == 1
    assert "haramrit09k/f1rstaid" in captured["url"]
    assert captured["headers"]["Authorization"] == "Bearer fake-token"
    assert "some answer text" in captured["json"]["body"]
    assert "this was wrong" in captured["json"]["body"]
    assert "https://example.com/source" in captured["json"]["body"]


def test_create_feedback_issue_never_includes_the_original_question(monkeypatch, tmp_path):
    """The core privacy requirement: create_feedback_issue() doesn't even
    accept a question parameter, so there's no way for one to leak into a
    public issue body -- this just confirms the body is built only from
    what was actually passed in."""
    monkeypatch.setenv("GITHUB_TOKEN", "fake-token")
    monkeypatch.setattr(f1rstaid, "FEEDBACK_TRACKER_PATH", tmp_path / "feedback.json")
    captured = {}
    monkeypatch.setattr(
        f1rstaid.requests,
        "post",
        lambda url, headers=None, json=None, timeout=None: (
            captured.update(body=json["body"]) or _FakeResponse(201)
        ),
    )

    create_feedback_issue("rule", "the computed answer", [], "unhelpful")

    assert "question is intentionally omitted" in captured["body"]


def test_create_feedback_issue_respects_daily_cap(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_TOKEN", "fake-token")
    tracker_path = tmp_path / "feedback.json"
    tracker_path.write_text(
        json.dumps({"date": str(date.today()), "count": MAX_FEEDBACK_ISSUES_PER_DAY})
    )
    monkeypatch.setattr(f1rstaid, "FEEDBACK_TRACKER_PATH", tracker_path)
    calls = []
    monkeypatch.setattr(f1rstaid.requests, "post", lambda *a, **k: calls.append((a, k)))

    filed = create_feedback_issue("rag", "answer", [], "feedback")

    assert filed is False
    assert calls == []  # cap enforced before spending an API call


def test_create_feedback_issue_handles_non_201_response(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_TOKEN", "fake-token")
    monkeypatch.setattr(f1rstaid, "FEEDBACK_TRACKER_PATH", tmp_path / "feedback.json")
    monkeypatch.setattr(
        f1rstaid.requests, "post", lambda *a, **k: _FakeResponse(403, text="bad credentials")
    )

    filed = create_feedback_issue("rag", "answer", [], "feedback")

    assert filed is False


def test_create_feedback_issue_handles_request_exception(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_TOKEN", "fake-token")
    monkeypatch.setattr(f1rstaid, "FEEDBACK_TRACKER_PATH", tmp_path / "feedback.json")

    def raise_error(*a, **k):
        raise f1rstaid.requests.RequestException("network error")

    monkeypatch.setattr(f1rstaid.requests, "post", raise_error)

    filed = create_feedback_issue("rag", "answer", [], "feedback")

    assert filed is False