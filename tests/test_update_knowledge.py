import pytest
from update_knowledge import append_to_vector_store

def test_vector_store_update(mock_documents):
    """Test vector store updating."""
    success = append_to_vector_store(mock_documents)
    assert success is True

def test_document_preprocessing(mock_documents):
    """Test document preprocessing."""
    processor = ContentProcessor()
    
    for doc in mock_documents:
        processed_text = processor.preprocess_text(doc.page_content)
        assert processed_text != ""
        if "F student" in doc.page_content:
            assert "F-1 student" in processed_text