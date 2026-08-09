import pytest
from f1rstaid import AppConfig, CONDENSE_HISTORY_TURNS, F1rstAidApp
from rules_engine import UnemploymentFields


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