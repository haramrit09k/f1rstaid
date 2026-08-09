import pytest
from langchain_core.documents import Document
from f1rstaid import F1rstAidApp, AppConfig

@pytest.fixture
def app_config():
    """Provide test configuration."""
    return AppConfig(
        model_name="gpt-3.5-turbo",
        vector_store_path="tests/test_faiss_index",
        search_k=2,
        temperature=0
    )

@pytest.fixture
def mock_documents():
    """Provide test documents. Content is deliberately >50 chars -- that's
    ContentProcessor.validate_content()'s minimum length, and real scraped/
    chunked content is always far longer than a short label, so these need
    to be realistic-length to actually exercise validation rather than fail
    it by construction."""
    return [
        Document(
            page_content=(
                "F-1 students must maintain full-time enrollment each "
                "academic term to remain in valid status."
            ),
            metadata={"source": "test.pdf", "type": "pdf"}
        ),
        Document(
            page_content=(
                "Optional Practical Training (OPT) allows F-1 students to "
                "work in their field of study for up to 12 months."
            ),
            metadata={"source": "https://test.com", "type": "web"}
        ),
        Document(
            page_content=(
                "Reddit: My OPT application (Form I-765) was approved in "
                "just under 2 months this year, faster than I expected."
            ),
            metadata={
                "source": "https://reddit.com/r/f1visa",
                "type": "reddit",
                "score": 10
            }
        )
    ]