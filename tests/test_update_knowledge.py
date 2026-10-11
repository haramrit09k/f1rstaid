import json

import pytest
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from ingest import ContentProcessor
import update_knowledge
from update_knowledge import _write_last_updated, append_to_vector_store

@pytest.mark.live
def test_vector_store_update(tmp_path, mock_documents):
    """Test vector store updating in an isolated tmp index -- must never touch
    the real faiss_index/ used by the running app."""
    index_path = str(tmp_path / "test_faiss_index")
    embeddings = OpenAIEmbeddings()
    FAISS.from_documents(mock_documents[:1], embeddings).save_local(index_path)

    success = append_to_vector_store(mock_documents[1:], vector_store_path=index_path)
    assert success is True

    # append_to_vector_store() writes freshness metadata on every successful
    # update -- this is what f1rstaid.py surfaces to students so they know
    # how current the answers actually are.
    metadata = json.loads((tmp_path / "test_faiss_index" / "last_updated.json").read_text())
    assert "last_updated" in metadata


def test_write_last_updated_writes_a_valid_timestamp(tmp_path):
    _write_last_updated(str(tmp_path))

    data = json.loads((tmp_path / "last_updated.json").read_text())
    # Must be parseable by f1rstaid.py's reader (datetime.fromisoformat) --
    # the whole point of this file existing.
    from datetime import datetime
    datetime.fromisoformat(data["last_updated"])


def test_write_last_updated_handles_unwritable_path_without_raising(tmp_path):
    unwritable = tmp_path / "does" / "not" / "exist"
    _write_last_updated(str(unwritable))  # must not raise

def test_document_preprocessing(mock_documents):
    """Test document preprocessing."""
    processor = ContentProcessor()
    
    for doc in mock_documents:
        processed_text = processor.preprocess_text(doc.page_content)
        assert processed_text != ""
        if "F student" in doc.page_content:
            assert "F-1 student" in processed_text

def test_append_failure_is_recorded_as_a_real_error(tmp_path, mock_documents, monkeypatch):
    """A failed append (e.g. OpenAI 429 insufficient_quota) must be recorded
    so main() exits non-zero -- it used to be log-only, leaving the weekly
    run green while nothing was committed."""
    monkeypatch.setattr(update_knowledge, "_errors", [])

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("insufficient_quota")

    monkeypatch.setattr(update_knowledge, "OpenAIEmbeddings", _Boom)

    assert append_to_vector_store(mock_documents, vector_store_path=str(tmp_path)) is False
    assert any("insufficient_quota" in e for e in update_knowledge._errors)


def test_main_returns_nonzero_when_errors_were_recorded(monkeypatch):
    async def _fake_updates():
        update_knowledge._errors.append("simulated failure")
        return [False, False, False]

    monkeypatch.setattr(update_knowledge, "_errors", [])
    monkeypatch.setattr(update_knowledge, "_run_all_updates", _fake_updates)

    assert update_knowledge.main() == 1


def test_main_returns_zero_when_nothing_failed(monkeypatch):
    async def _fake_updates():
        return [True, False, False]

    monkeypatch.setattr(update_knowledge, "_errors", [])
    monkeypatch.setattr(update_knowledge, "_run_all_updates", _fake_updates)

    assert update_knowledge.main() == 0


def test_main_returns_zero_when_there_was_simply_nothing_new(monkeypatch):
    """No new content is not an error -- only recorded failures are."""
    async def _fake_updates():
        return [False, False, False]

    monkeypatch.setattr(update_knowledge, "_errors", [])
    monkeypatch.setattr(update_knowledge, "_run_all_updates", _fake_updates)

    assert update_knowledge.main() == 0
