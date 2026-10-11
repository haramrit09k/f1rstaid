from datetime import datetime, timedelta, timezone

import feedparser
import pytest
from langchain_community.embeddings.fake import DeterministicFakeEmbedding
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import update_knowledge
from kb_freshness import (
    date_caption,
    delete_is_safe,
    doc_label,
    freshness_multiplier,
    parse_date,
    plan_index_changes,
    published_date,
    stamp_documents,
)

NOW = datetime(2026, 10, 11, tzinfo=timezone.utc)


def _d(text, source="https://a.example/p", **meta):
    return Document(page_content=text, metadata={"source": source, "type": "web", **meta})


# --- planner -----------------------------------------------------------------

def test_unchanged_chunks_are_skipped_not_re_added():
    existing = {"1": _d("alpha text"), "2": _d("beta text", "https://b.example")}
    to_delete, to_add, stats = plan_index_changes(existing, [_d("alpha text")])
    assert (to_delete, to_add) == ([], [])
    assert stats["skipped_unchanged"] == 1


def test_changed_page_replaces_its_old_chunks():
    existing = {"1": _d("old policy text"), "2": _d("other page", "https://b.example")}
    to_delete, to_add, stats = plan_index_changes(existing, [_d("new policy text")])
    assert to_delete == ["1"]  # only that source's stale chunk
    assert [d.page_content for d in to_add] == ["new policy text"]
    assert stats["removed_stale"] == 1


def test_existing_exact_duplicates_are_collapsed_to_one():
    existing = {"1": _d("same"), "2": _d("same"), "3": _d("same")}
    to_delete, to_add, stats = plan_index_changes(existing, [])
    assert sorted(to_delete) == ["2", "3"]
    assert stats["removed_duplicates"] == 2


def test_sources_not_in_this_batch_are_left_alone():
    existing = {"1": _d("keep me", "https://untouched.example")}
    to_delete, to_add, _ = plan_index_changes(existing, [_d("something", "https://other.example")])
    assert to_delete == []
    assert len(to_add) == 1


def test_duplicate_chunks_within_the_new_batch_are_added_once():
    _, to_add, _ = plan_index_changes({}, [_d("x"), _d("x")])
    assert len(to_add) == 1


def test_delete_cap():
    assert delete_is_safe(100, 50) is True
    assert delete_is_safe(100, 51) is False
    assert delete_is_safe(0, 0) is True


# --- dates -------------------------------------------------------------------

def test_parse_date_handles_iso_epoch_and_garbage():
    assert parse_date("2026-09-30T00:00:00Z").year == 2026
    assert parse_date(1_780_000_000).tzinfo is not None
    assert parse_date("not a date") is None
    assert parse_date(None) is None


def test_fetched_at_is_not_treated_as_publication_date():
    assert published_date({"fetched_at": "2026-10-01T00:00:00+00:00"}) is None
    assert published_date({"created_utc": 1_780_000_000}) is not None


def test_freshness_multiplier_favours_recent_and_is_neutral_when_undated():
    recent = {"published": (NOW - timedelta(days=10)).isoformat()}
    old = {"published": (NOW - timedelta(days=900)).isoformat()}
    assert freshness_multiplier(recent, NOW) > 1.0 > freshness_multiplier(old, NOW)
    assert freshness_multiplier({}, NOW) == 1.0


def test_labels_and_captions():
    news = {"type": "news", "published": "2026-09-30T00:00:00+00:00"}
    assert doc_label(news) == "news report, published 2026-09-30"
    assert date_caption(news) == "Published Sep 30, 2026"
    page = {"type": "web", "fetched_at": "2026-10-05T00:00:00+00:00"}
    assert doc_label(page) == "official/web page, retrieved 2026-10-05"
    assert date_caption(page) == "Retrieved Oct 05, 2026"
    assert doc_label({"type": "pdf"}) == "official document, date unknown"
    assert date_caption({}) == ""


def test_stamp_documents_sets_fetch_time_and_hash():
    (doc,) = stamp_documents([_d("hello")], NOW)
    assert doc.metadata["fetched_at"] == NOW.isoformat()
    assert doc.metadata["content_hash"]


# --- real FAISS merge (deterministic fake embeddings, no API) ------------------

@pytest.fixture
def index_path(tmp_path, monkeypatch):
    embeddings = DeterministicFakeEmbedding(size=16)
    store = FAISS.from_documents(
        [_d("OPT lasts 12 months."), _d("OPT lasts 12 months."), _d("Cap gap info.", "https://b.example")],
        embeddings,
    )
    store.save_local(str(tmp_path))
    monkeypatch.setattr(update_knowledge, "OpenAIEmbeddings", lambda **k: embeddings)
    monkeypatch.setattr(update_knowledge, "_errors", [])
    return str(tmp_path)


def _contents(path):
    store = FAISS.load_local(path, DeterministicFakeEmbedding(size=16), allow_dangerous_deserialization=True)
    return sorted(store.docstore.search(i).page_content for i in store.index_to_docstore_id.values())


def test_append_cleans_existing_duplicates_and_adds_only_new_content(index_path):
    ok = update_knowledge.append_to_vector_store(
        [_d("OPT lasts 12 months."), _d("Brand new alert.", "https://c.example")],
        vector_store_path=index_path,
    )
    assert ok is True
    assert _contents(index_path) == ["Brand new alert.", "Cap gap info.", "OPT lasts 12 months."]


def test_append_replaces_a_changed_page_in_the_real_index(index_path):
    update_knowledge.append_to_vector_store(
        [_d("OPT lasts 12 months, updated 2026.", "https://a.example/p")], vector_store_path=index_path
    )
    contents = _contents(index_path)
    assert "OPT lasts 12 months." not in contents
    assert "OPT lasts 12 months, updated 2026." in contents


def test_rerunning_the_same_update_changes_nothing(index_path):
    batch = lambda: [_d("Brand new alert.", "https://c.example")]  # noqa: E731
    update_knowledge.append_to_vector_store(batch(), vector_store_path=index_path)
    before = _contents(index_path)
    update_knowledge.append_to_vector_store(batch(), vector_store_path=index_path)
    assert _contents(index_path) == before


def test_append_refuses_a_mass_replacement(index_path):
    # 3 indexed chunks -> 2 unique; replacing both uniques (100%) must abort.
    ok = update_knowledge.append_to_vector_store(
        [_d("totally different A", "https://a.example/p"), _d("totally different B", "https://b.example")],
        vector_store_path=index_path,
    )
    assert ok is False
    assert any("Refusing to replace" in e for e in update_knowledge._errors)
    assert "Cap gap info." in _contents(index_path)  # untouched


# --- feeds -------------------------------------------------------------------

NEWS_RSS = """<?xml version="1.0"?><rss version="2.0"><channel><title>Google News</title>
<item><title>Court blocks new OPT rule for F-1 students - Inside Higher Ed</title>
<link>https://news.example/opt-rule</link><pubDate>Tue, 30 Sep 2026 12:00:00 GMT</pubDate>
<description>&lt;a href="https://news.example/opt-rule"&gt;Court blocks new OPT rule&lt;/a&gt; &lt;font&gt;Inside Higher Ed&lt;/font&gt;</description>
<source url="https://insidehighered.com">Inside Higher Ed</source></item>
<item><title>Local bakery wins award</title><link>https://news.example/bakery</link>
<pubDate>Tue, 30 Sep 2026 12:00:00 GMT</pubDate></item>
<item><title>Old F-1 visa story</title><link>https://news.example/old</link>
<pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate></item></channel></rss>"""


def test_news_feed_keeps_recent_on_topic_entries_only():
    feed = feedparser.parse(NEWS_RSS)
    docs = update_knowledge.news_documents_from_feed(feed, now=NOW)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.metadata["type"] == "news"
    assert doc.metadata["source"] == "https://news.example/opt-rule"
    assert doc.metadata["published"].startswith("2026-09-30")
    assert "Outlet: Inside Higher Ed" in doc.page_content
    assert "<" not in doc.page_content


def test_federal_register_payload_becomes_dated_official_documents():
    payload = {
        "results": [
            {
                "title": "Practical Training Reform",
                "type": "Proposed Rule",
                "abstract": "DHS proposes changes to OPT.",
                "html_url": "https://www.federalregister.gov/d/2026-12345",
                "publication_date": "2026-09-15",
                "agencies": [{"name": "Homeland Security Department"}],
            },
            {"title": "", "html_url": "https://x"},
        ]
    }
    (doc,) = update_knowledge.federal_register_documents(payload)
    assert doc.metadata["type"] == "web"
    assert doc.metadata["published"].startswith("2026-09-15")
    assert "Proposed Rule" in doc.page_content and "Homeland Security" in doc.page_content


# --- trust tiers, H-1B sources and Federal Register queries -----------------------

from urllib.parse import parse_qs, urlparse  # noqa: E402

from config import reddit_config, sources  # noqa: E402
from kb_freshness import source_tier  # noqa: E402


@pytest.mark.parametrize(
    "source,doc_type,expected",
    [
        ("https://www.uscis.gov/i-765", "web", "web"),
        ("https://studyinthestates.dhs.gov/x", "web", "web"),
        ("https://www.fragomen.com/insights/index.html", "web", "commentary"),
        ("https://blog.sprintax.com/f1-tax", "web", "commentary"),
        ("https://www.nafsa.org/regulatory-information", "web", "commentary"),
        ("https://notfragomen.com/x", "web", "web"),  # not a suffix match
        ("https://reddit.com/r/h1b/x", "reddit", "reddit"),
        ("https://news.example/x", "news", "news"),
        ("docs/guide.pdf", "pdf", "pdf"),
    ],
)
def test_source_tier_derived_from_domain(source, doc_type, expected):
    assert source_tier({"source": source, "type": doc_type}) == expected


def test_commentary_ranks_below_official_and_above_news_and_is_labelled():
    from f1rstaid import SOURCE_TYPE_WEIGHTS

    assert SOURCE_TYPE_WEIGHTS["web"] > SOURCE_TYPE_WEIGHTS["commentary"] > SOURCE_TYPE_WEIGHTS["news"]
    label = doc_label({"source": "https://www.murthy.com/x", "type": "web", "published": "2026-09-30T00:00:00+00:00"})
    assert label == "law-firm/expert commentary, published 2026-09-30"


def test_h1b_sources_are_wired_into_both_source_lists_without_duplicates():
    assert sources.H1B_TRANSITION_SOURCES
    for url in sources.H1B_TRANSITION_SOURCES:
        assert url.startswith("https://")
        assert url in sources.WEBSITE_SOURCES  # full rebuild
        assert url in update_knowledge.WEB_UPDATE_SOURCES  # weekly job
    assert len(sources.WEBSITE_SOURCES) == len(set(sources.WEBSITE_SOURCES))
    assert any("uscis.gov" in u and "h-1b" in u.lower() for u in sources.H1B_TRANSITION_SOURCES)


def test_reddit_h1b_coverage_added():
    assert "h1b" in reddit_config.SUBREDDITS
    assert "H-1B lottery" in reddit_config.SEARCH_TERMS


def test_federal_register_url_includes_term_and_every_agency():
    url = update_knowledge.federal_register_url('"H-1B"', ["homeland-security-department", "labor-department"])
    query = parse_qs(urlparse(url).query)
    assert query["conditions[term]"] == ['"H-1B"']
    assert query["conditions[agencies][]"] == ["homeland-security-department", "labor-department"]
    assert query["order"] == ["newest"]


def test_federal_register_queries_cover_h1b_and_other_agencies():
    terms = [t for t, _ in update_knowledge.FEDERAL_REGISTER_QUERIES]
    agencies = {a for _, ags in update_knowledge.FEDERAL_REGISTER_QUERIES for a in ags}
    assert '"H-1B"' in terms and '"specialty occupation"' in terms
    assert {"homeland-security-department", "labor-department", "state-department"} <= agencies
