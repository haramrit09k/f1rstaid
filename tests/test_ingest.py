import asyncio
import time
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
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

@pytest.mark.live
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


@pytest.mark.asyncio
async def test_scrape_reddit_does_not_block_event_loop():
    """scrape_reddit() must offload PRAW's blocking calls to a worker thread
    so coroutines gathered alongside it can still make progress. This was
    the root cause of the weekly knowledge-base update workflow hanging for
    6h/run for months: a synchronous PRAW scrape inside an async function
    fully monopolized the event loop, so update_web_sources()/
    update_from_rss() couldn't run concurrently despite asyncio.gather.
    """
    processor = ContentProcessor()
    events = []

    def blocking_scrape():
        time.sleep(0.3)
        events.append("scrape_done")
        return []

    async def dummy_coroutine():
        await asyncio.sleep(0.05)
        events.append("dummy_done")

    with patch.object(processor, "_scrape_reddit_sync", side_effect=blocking_scrape):
        await asyncio.gather(processor.scrape_reddit(), dummy_coroutine())

    # If scrape_reddit blocked the event loop, dummy_coroutine (0.05s)
    # couldn't run until the blocking call (0.3s) returned, so "scrape_done"
    # would come first. With the asyncio.to_thread fix, the scrape runs in a
    # worker thread and the dummy finishes first while it's still running.
    assert events == ["dummy_done", "scrape_done"]