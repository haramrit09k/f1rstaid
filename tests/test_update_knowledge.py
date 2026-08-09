from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from ingest import ContentProcessor
from update_knowledge import append_to_vector_store

def test_vector_store_update(tmp_path, mock_documents):
    """Test vector store updating in an isolated tmp index -- must never touch
    the real faiss_index/ used by the running app."""
    index_path = str(tmp_path / "test_faiss_index")
    embeddings = OpenAIEmbeddings()
    FAISS.from_documents(mock_documents[:1], embeddings).save_local(index_path)

    success = append_to_vector_store(mock_documents[1:], vector_store_path=index_path)
    assert success is True

def test_document_preprocessing(mock_documents):
    """Test document preprocessing."""
    processor = ContentProcessor()
    
    for doc in mock_documents:
        processed_text = processor.preprocess_text(doc.page_content)
        assert processed_text != ""
        if "F student" in doc.page_content:
            assert "F-1 student" in processed_text