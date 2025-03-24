import pytest
from crawler.crawler import WebCrawler

def test_url_validation():
    """Test URL validation."""
    crawler = WebCrawler()
    
    # Test valid URLs
    assert crawler.is_valid_url("https://www.uscis.gov/opt") is True
    assert crawler.is_valid_url("not_a_url") is False

@pytest.mark.asyncio
async def test_page_scraping():
    """Test web page scraping."""
    crawler = WebCrawler()
    content = await crawler.scrape_page("https://www.uscis.gov/working-in-the-united-states/students-and-exchange-visitors")
    
    assert content is not None
    assert "F-1" in content