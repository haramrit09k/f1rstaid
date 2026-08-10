import json

import pytest
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from ingest import ContentProcessor
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