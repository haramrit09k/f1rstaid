import pytest
from crawler.crawler import is_allowed, is_relevant, parse_robots_txt


def test_robots_txt_parsing():
    """Test robots.txt parsing extracts disallowed and allowed paths."""
    robots_txt = "User-agent: * Disallow: /private/ Allow: /public/"
    disallowed, allowed = parse_robots_txt(robots_txt)

    assert "/private/" in disallowed
    assert "/public/" in allowed


def test_robots_txt_empty():
    """Test robots.txt parsing handles empty input."""
    disallowed, allowed = parse_robots_txt("")
    assert disallowed == []
    assert allowed == []


def test_is_allowed_disallowed_path():
    """Test that disallowed paths are blocked."""
    disallowed = ["/admin/"]
    allowed = []
    assert is_allowed("https://example.com/admin/page", disallowed, allowed) is False


def test_is_allowed_permitted_path():
    """Test that paths not in disallow list are allowed."""
    disallowed = ["/admin/"]
    allowed = []
    assert is_allowed("https://example.com/students/opt", disallowed, allowed) is True


def test_is_allowed_override():
    """Test that allowed paths override disallowed paths."""
    disallowed = ["/sevis/"]
    allowed = ["/sevis/overview"]
    assert is_allowed("https://www.ice.gov/sevis/overview", disallowed, allowed) is True


def test_is_relevant_f1_content():
    """Test that F-1 relevant content scores above the threshold."""
    content = " ".join(["OPT Optional Practical Training F-1 visa SEVIS SEVP"] * 20)
    assert is_relevant(content) is True


def test_is_relevant_unrelated_content():
    """Test that unrelated content scores below the threshold."""
    content = "This website is about gardening tips and plant care."
    assert is_relevant(content) is False
