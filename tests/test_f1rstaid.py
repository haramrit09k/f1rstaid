import pytest
from f1rstaid import F1rstAidApp

def test_initialize(app_config):
    """Test application initialization."""
    app = F1rstAidApp(app_config)
    assert app.initialize() is True
    assert app.embeddings is not None
    assert app.db is not None
    assert app.qa_chain is not None

def test_relevance_check(app_config):
    """Test question relevance checking."""
    app = F1rstAidApp(app_config)
    
    # Test relevant question
    relevant, _ = app._is_relevant_question("How long does OPT processing take?")
    assert relevant is True
    
    # Test irrelevant question
    relevant, _ = app._is_relevant_question("What's the weather like today?")
    assert relevant is False

def test_get_answer(app_config):
    """Test answer generation."""
    app = F1rstAidApp(app_config)
    app.initialize()
    
    # Test help question
    answer = app.get_answer("help")
    assert "My Expertise" in answer["result"]
    assert len(answer["source_documents"]) == 0
    
    # Test relevant question
    answer = app.get_answer("What is OPT?")
    assert answer is not None
    assert "result" in answer
    assert "source_documents" in answer