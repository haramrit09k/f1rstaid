import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import aiohttp
import feedparser
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from ingest import ContentProcessor

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
    """Append new documents to existing vector store."""
    try:
        # request_timeout: a stalled embeddings call must raise, not hang --
        # the weekly update workflow silently hung for 6h/run for months
        # before this was added.
        embeddings = OpenAIEmbeddings(request_timeout=60)
        vector_store = FAISS.load_local(
            vector_store_path, embeddings, allow_dangerous_deserialization=True
        )
        vector_store.add_documents(docs)
        vector_store.save_local(vector_store_path)
        _write_last_updated(vector_store_path)
        return True
    except Exception as e:
        logging.error(f"Error appending to vector store: {e}")
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
        logging.error(f"Failed to update Reddit content: {e}")
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
        logging.error(f"Web source update failed: {e}")
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
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": entry.get("link", feed_url),
                            "type": "web",
                            "title": entry.get("title", ""),
                        },
                    )
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
        logging.error(f"RSS update failed: {e}")
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
    return [reddit_result, web_result, rss_result]


if __name__ == "__main__":
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
