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
    """Provide test documents."""
    return [
        Document(
            page_content="F-1 students must maintain full-time enrollment.",
            metadata={"source": "test.pdf", "type": "pdf"}
        ),
        Document(
            page_content="OPT allows students to work for 12 months.",
            metadata={"source": "https://test.com", "type": "web"}
        ),
        Document(
            page_content="Reddit: My OPT was approved in 2 months.",
            metadata={
                "source": "https://reddit.com/r/f1visa",
                "type": "reddit",
                "score": 10
            }
        )
    ]