import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

import f1rstaid
import rules_engine
from chat_store import (
    CHAT_TTL_SECONDS,
    MAX_MESSAGES,
    ChatStore,
    is_valid_chat_id,
    new_chat_id,
)
from f1rstaid import AppConfig, F1rstAidApp
from follow_ups import MAX_SUGGESTIONS, all_suggested_questions, suggest_follow_ups

DATASET = json.loads((Path(__file__).parent.parent / "eval" / "dataset.json").read_text())


# --- follow-up suggestions -----------------------------------------------------

def test_every_suggestion_is_a_question_the_eval_harness_already_checks():
    eval_questions = {q["question"] for q in DATASET["questions"]}
    for label, question in all_suggested_questions().items():
        assert question in eval_questions, f"{label!r} is not an eval-set question"


def test_suggestions_match_the_topic_skip_asked_questions_and_cap_at_three():
    picks = suggest_follow_ups("How many days of unemployment am I allowed on OPT?")
    assert 1 <= len(picks) <= MAX_SUGGESTIONS
    first_question = picks[0][1]

    again = suggest_follow_ups("unemployment days?", already_asked=[first_question])
    assert first_question not in [q for _, q in again]


def test_unknown_topics_get_no_suggestions():
    assert suggest_follow_ups("What's the weather like?") == []
    assert suggest_follow_ups("") == []


@pytest.mark.parametrize(
    "question,expected_label",
    [
        ("Can I do CPT before one year?", "CPT before one academic year?"),
        ("When do I file for a STEM extension?", "When to file for STEM OPT?"),
        ("Does cap-gap extend my work authorization?", "Cap-gap with an H-1B offer?"),
    ],
)
def test_topic_routing(question, expected_label):
    assert expected_label in [label for label, _ in suggest_follow_ups(question)]


# --- tappable clarifying-question options ------------------------------------------

def test_clarifying_options_fit_the_direct_answer_word_limit():
    """A tapped option must be merged with the original question by
    condense_question(), which only treats replies up to this length as a
    direct answer to the clarifying question."""
    for label, reply in rules_engine._UNEMPLOYMENT_PHASE_OPTIONS:
        assert len(reply.split()) <= f1rstaid._DIRECT_ANSWER_WORD_LIMIT, reply


def test_a_tapped_option_is_recognised_as_answering_the_clarifying_question():
    history = [
        {"role": "user", "content": "I've been unemployed for 40 days. Am I okay?"},
        {"role": "assistant", "content": rules_engine._UNEMPLOYMENT_PHASE_CLARIFYING_QUESTION},
    ]
    for _, reply in rules_engine._UNEMPLOYMENT_PHASE_OPTIONS:
        assert f1rstaid._answers_pending_clarification(reply, history) is True


# --- answer first ---------------------------------------------------------------------

def test_source_caveats_come_after_the_answer_not_before():
    app = F1rstAidApp(AppConfig())
    docs = [
        Document(page_content="n", metadata={"type": "news", "source": "https://news.example/x"}),
        Document(page_content="r", metadata={"type": "reddit", "source": "https://reddit.com/x"}),
    ]
    text = app.format_answer("You can apply 90 days before.", docs)

    assert text.startswith("You can apply 90 days before.")
    assert text.index("news reports") > text.index("90 days")
    assert text.index("Reddit") > text.index("90 days")


def test_answers_without_caveats_are_returned_unchanged():
    app = F1rstAidApp(AppConfig())
    web = [Document(page_content="w", metadata={"type": "web", "source": "https://www.uscis.gov/x"})]
    assert app.format_answer("Plain answer.", web) == "Plain answer."


# --- chat store -----------------------------------------------------------------------

class _Redis:
    def __init__(self):
        self.data, self.ttl = {}, {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None):
        self.data[key], self.ttl[key] = value, ex

    def delete(self, key):
        self.data.pop(key, None)


def _store():
    redis = _Redis()
    return ChatStore(lambda: redis), redis


def _messages():
    doc = Document(page_content="OPT text " * 400, metadata={"source": "https://www.uscis.gov/x", "type": "web", "created_utc": 1.5})
    return [
        {"role": "user", "content": "How long is OPT?"},
        {"role": "assistant", "content": "12 months.", "answer": {"result": "12 months.", "source_documents": [doc], "clarify_options": [["a", "b"]]}},
    ]


def test_chat_round_trips_documents_options_and_counter():
    store, redis = _store()
    chat_id = new_chat_id()
    assert store.save(chat_id, _messages(), shared_key_count=3) is True
    assert redis.ttl[f"f1rstaid:chat:{chat_id}"] == CHAT_TTL_SECONDS

    loaded = store.load(chat_id)
    assert loaded["shared_key_count"] == 3
    answer = loaded["messages"][1]["answer"]
    assert answer["result"] == "12 months."
    assert isinstance(answer["source_documents"][0], Document)
    assert answer["source_documents"][0].metadata["created_utc"] == 1.5
    assert len(answer["source_documents"][0].page_content) <= 1500  # size-capped
    assert answer["clarify_options"] == [["a", "b"]]


def test_chat_ids_are_unguessable_and_validated():
    assert len({new_chat_id() for _ in range(50)}) == 50
    assert is_valid_chat_id(new_chat_id())
    for bad in [None, "", "short", "has spaces in it!!", "../../etc/passwd", "x" * 100]:
        assert is_valid_chat_id(bad) is False


def test_invalid_ids_never_touch_the_store():
    store, redis = _store()
    assert store.save("../evil", _messages()) is False
    assert store.load("../evil") is None
    assert redis.data == {}


def test_only_the_most_recent_messages_are_kept():
    store, _ = _store()
    chat_id = new_chat_id()
    many = [{"role": "user", "content": f"q{i}"} for i in range(MAX_MESSAGES + 10)]
    store.save(chat_id, many)
    loaded = store.load(chat_id)["messages"]
    assert len(loaded) == MAX_MESSAGES and loaded[-1]["content"] == f"q{MAX_MESSAGES + 9}"


def test_delete_removes_the_chat():
    store, redis = _store()
    chat_id = new_chat_id()
    store.save(chat_id, _messages())
    store.delete(chat_id)
    assert store.load(chat_id) is None and redis.data == {}


def test_store_fails_soft_without_redis_or_on_errors():
    none_store = ChatStore(lambda: None)
    assert none_store.available() is False
    assert none_store.save(new_chat_id(), _messages()) is False
    assert none_store.load(new_chat_id()) is None
    none_store.delete(new_chat_id())  # no error

    class _Broken:
        def get(self, *a, **k):
            raise ConnectionError("down")

        set = delete = get

    broken = ChatStore(lambda: _Broken())
    assert broken.save(new_chat_id(), _messages()) is False
    assert broken.load(new_chat_id()) is None
    broken.delete(new_chat_id())


def test_a_corrupt_saved_chat_loads_as_nothing():
    store, redis = _store()
    chat_id = new_chat_id()
    redis.data[f"f1rstaid:chat:{chat_id}"] = "{not json"
    assert store.load(chat_id) is None
