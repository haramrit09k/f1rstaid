import asyncio
import html
import json
import logging
import re
import sys
from calendar import timegm
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote_plus

import aiohttp
import feedparser
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from ingest import ContentProcessor
from kb_freshness import delete_is_safe, plan_index_changes, stamp_documents

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("update_knowledge.log"), logging.StreamHandler()],
)

# High-churn pages most likely to have weekly updates
WEB_UPDATE_SOURCES = [
    "https://www.uscis.gov/newsroom",
    "https://www.uscis.gov/newsroom/alerts",
    "https://www.uscis.gov/policy-manual/updates",
    "https://studyinthestates.dhs.gov/whats-new",
    "https://www.uscis.gov/working-in-the-united-states/students-and-exchange-visitors/optional-practical-training-opt-for-f-1-students",
    "https://www.uscis.gov/working-in-the-united-states/students-and-exchange-visitors/optional-practical-training-extension-for-stem-students-stem-opt",
    "https://www.murthy.com/view-all-news/",
    "https://www.fragomen.com/insights/index.html",
    "https://www.aila.org/blog/",
]

# Machine-readable RSS feeds for alerts and news
RSS_FEEDS = [
    "https://www.uscis.gov/rss/alerts",
    "https://www.nafsa.org/rss-feed.xml?feed=am_news",
]

# News coverage (type "news"): headline + outlet + date + snippet from Google
# News search RSS -- we index the headline/snippet and link back, not the
# article body (paywalls, copyright). Ranked below official sources and
# labelled as news in the answer prompt and citations.
NEWS_QUERIES = [
    '"F-1" OPT students',
    '"STEM OPT" OR "optional practical training" rule',
    'international students visa policy F-1',
]
NEWS_FEED_URL = "https://news.google.com/rss/search?q={query}+when:60d&hl=en-US&gl=US&ceid=US:en"
NEWS_MAX_AGE_DAYS = 120
NEWS_MAX_PER_QUERY = 20
_NEWS_RELEVANCE = re.compile(
    r"\b(f-?1|opt|cpt|stem|sevis|sevp|i-20|international students?|student visas?|visas?)\b",
    re.IGNORECASE,
)

# Federal Register (official): rules and notices from DHS/ICE/USCIS that
# mention practical training, F-1 or SEVIS -- where OPT rule changes show up
# first, long before any explainer page is updated.
FEDERAL_REGISTER_URL = (
    "https://www.federalregister.gov/api/v1/documents.json"
    "?conditions[term]={term}&conditions[agencies][]=homeland-security-department"
    "&order=newest&per_page=15"
)
FEDERAL_REGISTER_TERMS = ['"practical training"', '"F-1" students', "SEVIS"]


# Real failures (an OpenAI quota/auth error while embedding, a source blowing
# up) as opposed to "nothing new this week". Each updater returns False for
# both, and the job used to exit 0 either way -- so a run that failed every
# week for ~7 weeks (OpenAI credits exhausted) still showed a green check.
# main() exits non-zero when anything is recorded here.
_errors: List[str] = []


def _record_error(message: str) -> None:
    logging.error(message)
    _errors.append(message)


def _write_last_updated(vector_store_path: str) -> None:
    """Records when the index was actually last refreshed, inside
    faiss_index/ itself so weekly-update.yml's existing `git add
    faiss_index/` picks it up automatically -- no separate commit step
    needed. f1rstaid.py reads this to show students how fresh the answers
    actually are, instead of leaving that invisible (a real gap: the
    freshness-pipeline fix earlier this project never had anywhere to
    actually show up to the person using the app).

    Called multiple times per pipeline run (once per source type) -- each
    call just overwrites with the current time, so the file always reflects
    whenever this run's last successful update happened, which is exactly
    what "last updated" should mean.
    """
    try:
        path = Path(vector_store_path) / "last_updated.json"
        path.write_text(json.dumps({"last_updated": datetime.now(timezone.utc).isoformat()}))
    except OSError as e:
        logging.error(f"Failed to write last_updated.json: {e}")


def append_to_vector_store(docs: List[Document], vector_store_path: str = "faiss_index") -> bool:
    """Merge new documents into the existing vector store.

    Not a blind append: unchanged chunks are skipped (no embedding call, no
    duplicate), chunks of a re-fetched source whose content changed replace
    the old version, and any pre-existing exact duplicates are cleaned up
    (see kb_freshness.plan_index_changes). Every new chunk is stamped with
    fetched_at and a content_hash. Returns True when the store was updated
    successfully (even if nothing needed to change), False on an error.
    """
    try:
        # request_timeout: a stalled embeddings call must raise, not hang --
        # the weekly update workflow silently hung for 6h/run for months
        # before this was added.
        embeddings = OpenAIEmbeddings(request_timeout=60)
        vector_store = FAISS.load_local(
            vector_store_path, embeddings, allow_dangerous_deserialization=True
        )
        stamp_documents(docs)
        existing = {
            doc_id: vector_store.docstore.search(doc_id)
            for doc_id in vector_store.index_to_docstore_id.values()
        }
        to_delete, to_add, stats = plan_index_changes(existing, docs)

        # Exact duplicates are always safe to drop (one copy stays, and
        # there may be many after months of blind appends); the cap guards
        # the other kind of deletion -- replacing stale versions -- against
        # a bug or a bad fetch wiping out the index.
        unique_count = len(existing) - stats["removed_duplicates"]
        if not delete_is_safe(unique_count, stats["removed_stale"]):
            raise RuntimeError(
                f"Refusing to replace {stats['removed_stale']} of {unique_count} indexed chunks in one run"
            )
        if to_delete:
            vector_store.delete(to_delete)
        if to_add:
            vector_store.add_documents(to_add)
        if to_delete or to_add:
            vector_store.save_local(vector_store_path)
        logging.info(f"Vector store update: {stats}")
        _write_last_updated(vector_store_path)
        return True
    except Exception as e:
        _record_error(f"Error appending to vector store: {e}")
        return False


async def update_knowledge_base() -> bool:
    """Update the knowledge base with new Reddit content."""
    try:
        processor = ContentProcessor()

        reddit_docs = await processor.scrape_reddit()
        if not reddit_docs:
            logging.warning("No new Reddit content found")
            return False

        valid_docs = []
        for doc in reddit_docs:
            if processor.validate_content(doc):
                doc.page_content = processor.preprocess_text(doc.page_content)
                valid_docs.append(doc)

        if not valid_docs:
            logging.warning("No valid Reddit documents found")
            return False

        chunks = []
        for doc in valid_docs:
            chunks.extend(processor.text_splitter.split_documents([doc]))

        if append_to_vector_store(chunks):
            logging.info(f"Added {len(chunks)} Reddit chunks to knowledge base")
            return True
        return False

    except Exception as e:
        _record_error(f"Failed to update Reddit content: {e}")
        return False


async def update_web_sources() -> bool:
    """Fetch and append updates from high-churn USCIS/DHS and law firm pages."""
    try:
        processor = ContentProcessor()
        docs = []

        async with aiohttp.ClientSession() as session:
            for url in WEB_UPDATE_SOURCES:
                doc = await processor.process_website(url)
                if doc and processor.validate_content(doc):
                    doc.page_content = processor.preprocess_text(doc.page_content)
                    docs.append(doc)

        if not docs:
            logging.warning("No web content retrieved")
            return False

        chunks = []
        for doc in docs:
            chunks.extend(processor.text_splitter.split_documents([doc]))

        if append_to_vector_store(chunks):
            logging.info(f"Appended {len(chunks)} web chunks to knowledge base")
            return True
        return False

    except Exception as e:
        _record_error(f"Web source update failed: {e}")
        return False


async def update_from_rss() -> bool:
    """Fetch and append updates from RSS feeds (USCIS alerts, NAFSA)."""
    try:
        processor = ContentProcessor()
        docs = []

        for feed_url in RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    content = f"{entry.get('title', '')}\n\n{entry.get('summary', '')}"
                    if len(content) < 50:
                        continue
                    metadata = {
                        "source": entry.get("link", feed_url),
                        "type": "web",
                        "title": entry.get("title", ""),
                    }
                    published = _entry_published(entry)
                    if published:
                        metadata["published"] = published
                    doc = Document(page_content=content, metadata=metadata)
                    if processor.validate_content(doc):
                        doc.page_content = processor.preprocess_text(doc.page_content)
                        docs.append(doc)
            except Exception as e:
                logging.error(f"Error parsing RSS feed {feed_url}: {e}")
                continue

        if not docs:
            logging.warning("No RSS content retrieved")
            return False

        chunks = []
        for doc in docs:
            chunks.extend(processor.text_splitter.split_documents([doc]))

        if append_to_vector_store(chunks):
            logging.info(f"Appended {len(chunks)} RSS chunks to knowledge base")
            return True
        return False

    except Exception as e:
        _record_error(f"RSS update failed: {e}")
        return False


def _entry_published(entry) -> Optional[str]:
    """ISO timestamp from a feedparser entry's parsed date, if present."""
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    try:
        return datetime.fromtimestamp(timegm(parsed), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OverflowError):
        return None


def _strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text or ""))).strip()


def news_documents_from_feed(feed, now: Optional[datetime] = None) -> List[Document]:
    """Turn a parsed news feed into type='news' Documents: recent, on-topic
    entries only, headline + outlet + date (+ snippet when it adds
    something beyond the headline)."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=NEWS_MAX_AGE_DAYS)
    docs: List[Document] = []
    for entry in feed.entries[:NEWS_MAX_PER_QUERY]:
        title = _strip_html(entry.get("title", ""))
        published = _entry_published(entry)
        if not title or not published or datetime.fromisoformat(published) < cutoff:
            continue
        if not _NEWS_RELEVANCE.search(title):
            continue
        outlet = _strip_html(
            (entry.get("source") or {}).get("title", "") if isinstance(entry.get("source"), dict) else ""
        )
        snippet = _strip_html(entry.get("summary", ""))
        parts = [title]
        if outlet:
            parts.append(f"Outlet: {outlet}")
        parts.append(f"Published: {published[:10]}")
        if snippet and snippet.lower() not in title.lower() and title.lower() not in snippet.lower():
            parts.append(snippet)
        docs.append(
            Document(
                page_content="\n\n".join(parts),
                metadata={
                    "source": entry.get("link", ""),
                    "type": "news",
                    "title": title,
                    "published": published,
                },
            )
        )
    return docs


def federal_register_documents(payload: dict) -> List[Document]:
    """Documents from a Federal Register API response (type='web', i.e.
    official), dated with the document's publication date."""
    docs: List[Document] = []
    for item in payload.get("results", []):
        title = (item.get("title") or "").strip()
        abstract = (item.get("abstract") or "").strip()
        url = item.get("html_url") or ""
        if not title or not url:
            continue
        agencies = ", ".join(a.get("name", "") for a in item.get("agencies", []) if a.get("name"))
        parts = [f"Federal Register ({item.get('type', 'document')}): {title}"]
        if agencies:
            parts.append(f"Agency: {agencies}")
        if item.get("publication_date"):
            parts.append(f"Published: {item['publication_date']}")
        if abstract:
            parts.append(abstract)
        metadata = {"source": url, "type": "web", "title": title}
        if item.get("publication_date"):
            metadata["published"] = f"{item['publication_date']}T00:00:00+00:00"
        docs.append(Document(page_content="\n\n".join(parts), metadata=metadata))
    return docs


async def update_from_federal_register() -> bool:
    """Official rules/notices from the Federal Register."""
    try:
        processor = ContentProcessor()
        docs: List[Document] = []
        failures = 0
        async with aiohttp.ClientSession() as session:
            for term in FEDERAL_REGISTER_TERMS:
                try:
                    url = FEDERAL_REGISTER_URL.format(term=term.replace(" ", "+"))
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        if resp.status != 200:
                            raise RuntimeError(f"HTTP {resp.status}")
                        docs.extend(federal_register_documents(await resp.json()))
                except Exception as e:
                    failures += 1
                    logging.warning(f"Federal Register query {term!r} failed: {e}")
        if failures == len(FEDERAL_REGISTER_TERMS):
            _record_error("Every Federal Register query failed (check FEDERAL_REGISTER_URL)")
            return False
        docs = [d for d in docs if processor.validate_content(d)]
        if not docs:
            logging.warning("No Federal Register content retrieved")
            return False
        return append_to_vector_store(processor.text_splitter.split_documents(docs))
    except Exception as e:
        _record_error(f"Federal Register update failed: {e}")
        return False


async def update_from_news() -> bool:
    """Recent news coverage (type='news') via search-RSS queries."""
    try:
        processor = ContentProcessor()
        docs: List[Document] = []
        failures = 0
        for query in NEWS_QUERIES:
            try:
                feed = feedparser.parse(NEWS_FEED_URL.format(query=quote_plus(query)))
                if not feed.entries and getattr(feed, "bozo", False):
                    raise RuntimeError(str(getattr(feed, "bozo_exception", "unparseable feed")))
                docs.extend(news_documents_from_feed(feed))
            except Exception as e:
                failures += 1
                logging.warning(f"News query {query!r} failed: {e}")
        if failures == len(NEWS_QUERIES):
            _record_error("Every news query failed (check NEWS_FEED_URL)")
            return False
        docs = [d for d in docs if processor.validate_content(d)]
        if not docs:
            logging.warning("No news content retrieved")
            return False
        return append_to_vector_store(processor.text_splitter.split_documents(docs))
    except Exception as e:
        _record_error(f"News update failed: {e}")
        return False


async def _run_all_updates() -> list:
    """Run sequentially, not via asyncio.gather.

    Each of these ends by loading, modifying, and saving the *same*
    faiss_index/ path (append_to_vector_store). Running them concurrently
    races three independent load->add->save cycles against the same files --
    each function already handles its own errors and returns a bool rather
    than raising, so sequencing them costs some wall-clock time but has no
    other downside, and removes the race entirely.
    """
    reddit_result = await update_knowledge_base()  # Reddit
    web_result = await update_web_sources()         # USCIS/DHS/law firm pages
    rss_result = await update_from_rss()             # RSS feeds (USCIS alerts, NAFSA)
    fr_result = await update_from_federal_register()  # official rules/notices
    news_result = await update_from_news()           # news coverage
    return [reddit_result, web_result, rss_result, fr_result, news_result]


def main() -> int:
    """Runs all updaters; returns the process exit code (non-zero if any
    real error was recorded -- see _errors)."""
    print(
        f"\n🔄 Starting knowledge base update at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    results = asyncio.run(_run_all_updates())

    if any(results):
        print("\n✅ Knowledge base update complete!")
        print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("\n❌ Knowledge base update failed.")
        print("Check the update_knowledge.log file for details.")

    if _errors:
        print(f"\n❌ {len(_errors)} error(s) during the update:")
        for message in _errors:
            print(f"  - {message}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
