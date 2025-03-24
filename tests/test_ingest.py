import pytest
from ingest import ContentProcessor

def test_content_validation(mock_documents):
    """Test document content validation."""
    processor = ContentProcessor()
    
    # Test valid documents
    for doc in mock_documents:
        assert processor.validate_content(doc) is True
    
    # Test invalid document
    invalid_doc = Document(
        page_content="",
        metadata={"type": "web"}
    )
    assert processor.validate_content(invalid_doc) is False

@pytest.mark.asyncio
async def test_reddit_scraping():
    """Test Reddit content scraping."""
    processor = ContentProcessor()
    docs = await processor.scrape_reddit()
    
    assert len(docs) > 0
    for doc in docs:
        assert doc.metadata["type"] == "reddit"
        assert "score" in doc.metadata
        assert len(doc.page_content) > 100