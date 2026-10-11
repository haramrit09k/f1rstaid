"""UI tests: run the real f1rstaid.main() through Streamlit's AppTest with a
stubbed model (no OpenAI key, network or credits)."""
from pathlib import Path

import json

import pytest
from streamlit.testing.v1 import AppTest

REPO = str(Path(__file__).resolve().parent.parent)


def _script():
    import os
    import sys

    sys.path.insert(0, os.environ["F1_REPO"])
    import f1rstaid
    from langchain_core.documents import Document

    # keep the usage counters out of the repo directory
    f1rstaid.USAGE_TRACKER_PATH = f1rstaid.Path(os.environ["F1_TMP"]) / "usage.json"
    f1rstaid.IP_TRACKER_PATH = f1rstaid.Path(os.environ["F1_TMP"]) / "ip.json"

    class _Stub(f1rstaid.F1rstAidApp):
        def __init__(self):
            super().__init__(f1rstaid.AppConfig())

        def get_knowledge_base_last_updated(self):
            return os.environ["F1_KB_DATE"]

        def get_answer(self, question, chat_history=None):
            with open(os.environ["F1_TMP"] + "/prompts.txt", "a") as f:
                f.write(question + "\n")
            if os.environ.get("F1_MODE") == "clarify" and "unemployment" in question.lower():
                return {
                    "result": "Are you on initial OPT or the STEM OPT extension?",
                    "source_documents": [],
                    "clarify_options": [
                        ("Initial OPT", "I'm on initial OPT"),
                        ("STEM OPT extension", "I'm on the STEM OPT extension"),
                    ],
                }
            return {
                "result": "You can apply up to 90 days before your program end date.",
                "source_documents": [
                    Document(
                        page_content="F-1 students may apply for OPT 90 days before program end.",
                        metadata={"source": "https://www.uscis.gov/opt", "type": "web"},
                    )
                ],
            }

    stub = _Stub()
    f1rstaid.get_cached_app = lambda key: stub
    f1rstaid.main()


@pytest.fixture
def run_app(monkeypatch, tmp_path):
    monkeypatch.setenv("F1_REPO", REPO)
    monkeypatch.setenv("F1_TMP", str(tmp_path))
    monkeypatch.setenv("F1_KB_DATE", "January 1, 2020")  # always stale
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def _run():
        return AppTest.from_function(_script, default_timeout=30).run()

    return _run


def _popover_labels(at):
    return [block.proto.popover.label for block in at.get("popover")]


def test_home_renders_with_everything_essential_outside_the_sidebar(run_app):
    at = run_app()

    assert not at.exception
    assert [t.value for t in at.title] == ["🎓 F1rstAid"]
    assert len(at.sidebar.children) == 0  # hidden behind a tiny arrow on phones
    assert _popover_labels(at) == ["ℹ️ About", "🎟️ 5 free left"]
    assert len(at.button) == 4  # example-question chips
    assert [s.value for s in at.subheader] == ["What do you want to know?"]


def test_stale_knowledge_base_shows_a_visible_warning(run_app):
    at = run_app()

    assert any("days ago" in w.value for w in at.warning)


def test_clicking_a_chip_answers_and_updates_the_free_question_counter(run_app):
    at = run_app()
    at.button[0].click().run()

    assert not at.exception
    assert any(
        "90 days before" in md.value for m in at.chat_message for md in m.markdown
    )
    assert _popover_labels(at)[1] == "🎟️ 4 free left"
    labels = [b.label for b in at.button]
    assert "🔄 New chat" in labels
    # the starting chips are gone (related-question chips take their place)
    assert not any((b.key or "").startswith("example_") for b in at.button)


def test_new_chat_returns_to_the_start_screen(run_app):
    at = run_app()
    at.button[0].click().run()
    new_chat = next(b for b in at.button if b.label == "🔄 New chat")
    new_chat.click().run()

    assert [s.value for s in at.subheader] == ["What do you want to know?"]
    assert at.session_state.messages == []
    # (AppTest keeps the pre-rerun "New chat" button in its tree, a real
    # browser does not -- so check the chips, not the total button count.)
    chips = [b for b in at.button if (b.key or "").startswith("example_")]
    assert len(chips) == 4


def test_typing_your_own_key_switches_the_top_bar_label(run_app):
    at = run_app()
    at.text_input[0].set_value("sk-my-own-key").run()

    assert not at.exception
    assert _popover_labels(at)[1] == "🔑 Your key"


def test_without_any_key_the_page_tells_you_where_to_add_one(run_app, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    at = run_app()

    assert not at.exception
    assert _popover_labels(at)[1] == "🔑 Add key"
    assert any("Add key" in e.value for e in at.error)


def _keys(at, prefix):
    return [b for b in at.button if (b.key or "").startswith(prefix)]


def _prompts(tmp_path):
    path = tmp_path / "prompts.txt"
    return path.read_text().splitlines() if path.exists() else []


# --- related-question chips ---------------------------------------------------

def test_an_answer_offers_related_question_chips_and_tapping_one_asks_it(run_app, tmp_path):
    at = run_app()
    at.button[0].click().run()  # "Unemployment days on OPT?" -> unemployment topic

    chips = _keys(at, "followup_")
    assert 1 <= len(chips) <= 3
    label, tapped = chips[0].label, chips[0]
    tapped.click().run()

    asked = _prompts(tmp_path)
    assert len(asked) == 2 and asked[1] != asked[0]
    assert label  # the chip had a readable label


# --- tappable clarifying question -----------------------------------------------

def test_clarifying_question_shows_tap_options_instead_of_follow_ups(run_app, monkeypatch, tmp_path):
    monkeypatch.setenv("F1_MODE", "clarify")
    at = run_app()
    at.button[0].click().run()

    options = _keys(at, "clarify_")
    assert [b.label for b in options] == ["Initial OPT", "STEM OPT extension"]
    assert not _keys(at, "followup_")
    # a question is not an answer: nothing to rate, no "answer" badge/disclaimer
    assert not [b for b in at.button if b.label in ("👍", "👎")]
    shown = " ".join(md.value for m in at.chat_message for md in m.markdown)
    assert "not legal advice" not in shown and "Instantly calculated" not in shown

    options[0].click().run()
    assert _prompts(tmp_path)[-1] == "I'm on initial OPT"


# --- keeping the chat across a refresh ---------------------------------------------

class _FakeRedis:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None):
        self.data[key] = value

    def delete(self, key):
        self.data.pop(key, None)

    def incr(self, key):
        self.data[key] = int(self.data.get(key, 0)) + 1
        return self.data[key]

    def expire(self, key, seconds):
        pass


@pytest.fixture
def fake_redis(monkeypatch):
    import f1rstaid

    fake = _FakeRedis()
    monkeypatch.setattr(f1rstaid, "_get_redis", lambda: fake)
    return fake


def test_chat_is_saved_and_restored_after_a_refresh(run_app, fake_redis):
    at = run_app()
    at.button[0].click().run()

    chat_id = at.query_params["c"][0] if isinstance(at.query_params["c"], list) else at.query_params["c"]
    saved = [k for k in fake_redis.data if k.startswith("f1rstaid:chat:")]
    assert saved == [f"f1rstaid:chat:{chat_id}"]
    payload = json.loads(fake_redis.data[saved[0]])
    assert payload["shared_key_count"] == 1
    assert payload["messages"][0]["role"] == "user"

    # a "refresh": brand-new session, same URL
    fresh = AppTest.from_function(_script, default_timeout=30)
    fresh.query_params["c"] = chat_id
    fresh.run()

    assert not fresh.exception
    assert any("90 days before" in md.value for m in fresh.chat_message for md in m.markdown)
    assert [b.label for b in fresh.button][0] == "🔄 New chat"
    assert _popover_labels(fresh)[1] == "🎟️ 4 free left"  # the counter survived too


def test_new_chat_deletes_the_saved_chat(run_app, fake_redis):
    at = run_app()
    at.button[0].click().run()
    assert any(k.startswith("f1rstaid:chat:") for k in fake_redis.data)

    next(b for b in at.button if b.label == "🔄 New chat").click().run()

    assert not any(k.startswith("f1rstaid:chat:") for k in fake_redis.data)


def test_without_redis_nothing_is_saved_and_the_url_is_left_alone(run_app):
    at = run_app()
    at.button[0].click().run()

    assert not at.exception
    assert "c" not in at.query_params


def test_a_made_up_or_expired_chat_id_just_starts_fresh(run_app, fake_redis):
    at = AppTest.from_function(_script, default_timeout=30)
    at.query_params["c"] = "does-not-exist-1234"
    at.run()

    assert not at.exception
    assert [s.value for s in at.subheader] == ["What do you want to know?"]


def test_related_chips_after_a_one_tap_reply_use_the_original_question_as_topic(run_app, monkeypatch, tmp_path):
    monkeypatch.setenv("F1_MODE", "clarify")
    at = run_app()
    at.button[0].click().run()  # unemployment question -> clarifying options
    _keys(at, "clarify_")[0].click().run()  # "I'm on initial OPT"

    labels = [b.label for b in _keys(at, "followup_")]
    # topic = unemployment (from the question it answered), not just "opt"
    assert "Unemployment benefits on OPT?" in labels
