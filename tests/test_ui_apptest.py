"""UI tests: run the real f1rstaid.main() through Streamlit's AppTest with a
stubbed model (no OpenAI key, network or credits)."""
from pathlib import Path

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
    assert not any("Unemployment" in label for label in labels)  # chips go away


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
