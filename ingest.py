import os
import logging
import sys
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import requests
from bs4 import BeautifulSoup
import html2text
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import praw

from config.reddit_config import REDDIT_CONFIG, SUBREDDITS, SEARCH_TERMS
from config.sources import WEBSITE_SOURCES

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("ingest.log"), logging.StreamHandler()],
)


@dataclass
class ProcessingMetrics:
    """Metrics for document processing."""

    total_pdfs: int = 0
    valid_pdfs: int = 0
    invalid_pdfs: int = 0
    total_chunks: int = 0
    valid_chunks: int = 0
    invalid_chunks: int = 0
    total_websites: int = 0
    valid_websites: int = 0
    total_reddit: int = 0


class ContentProcessor:
    """Handles document processing and validation."""

    def __init__(self):
        self.metrics = ProcessingMetrics()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", ". ", "\n", " ", ""],
        )

    @staticmethod
    @lru_cache(maxsize=100)
    def preprocess_text(text: str) -> str:
        """Preprocess text with caching for efficiency."""
        replacements = {
            "F student": "F-1 student",
            "F Students": "F-1 Students",
            "F visa": "F-1 visa",
            "OPT ": "Optional Practical Training (OPT) ",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def validate_content(self, doc: Document) -> bool:
        """Validate document content."""
        content = doc.page_content.strip()
        source_type = doc.metadata.get("type", "unknown")

        if not content or len(content) < 50 or not any(c.isalpha() for c in content):
            return False

        if source_type == "web" and any(
            term in content.lower()
            for term in ["[advertisement]", "cookie", "privacy policy"]
        ):
            return False

        return True

    async def process_pdf(self, file_path: str) -> List[Document]:
        """Process individual PDF file."""
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            filename = os.path.basename(file_path)
            for doc in docs:
                doc.metadata.update(
                    {"type": "pdf", "source": file_path, "filename": filename}
                )
            return docs
        except Exception as e:
            logging.error(f"Error processing PDF {file_path}: {e}")
            return []

    async def process_website(self, url: str) -> Optional[Document]:
        """Process individual website."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, "html.parser")
                        for element in soup.find_all(
                            ["nav", "footer", "script", "style"]
                        ):
                            element.decompose()

                        h = html2text.HTML2Text()
                        h.ignore_links = False
                        content = h.handle(str(soup))

                        return Document(
                            page_content=content,
                            metadata={"source": url, "type": "web"},
                        )
            return None
        except Exception as e:
            logging.error(f"Error processing website {url}: {e}")
            return None

    async def scrape_reddit(self) -> List[Document]:
        """Scrape relevant Reddit posts and comments."""
        try:
            logging.info("Starting Reddit content scraping...")
            reddit = praw.Reddit(**REDDIT_CONFIG)
            documents = []

            for subreddit_name in SUBREDDITS:
                try:
                    subreddit = reddit.subreddit(subreddit_name)
                    logging.info(f"Scraping r/{subreddit_name}")

                    for search_term in SEARCH_TERMS:
                        for submission in subreddit.search(
                            search_term, limit=10, sort="relevance"
                        ):
                            # Process post content
                            post_content = (
                                f"Title: {submission.title}\n\n"
                                f"Content: {submission.selftext}\n\n"
                                f"Score: {submission.score}"
                            )

                            if len(submission.selftext) > 100:  # Filter short posts
                                doc = Document(
                                    page_content=post_content,
                                    metadata={
                                        "source": f"https://reddit.com{submission.permalink}",
                                        "type": "reddit",
                                        "score": submission.score,
                                        "created_utc": submission.created_utc,
                                        "subreddit": subreddit_name,
                                        "title": submission.title,
                                    },
                                )
                                documents.append(doc)
                                self.metrics.total_reddit += 1

                            # Process top comments
                            submission.comments.replace_more(limit=0)
                            for comment in submission.comments[:5]:  # Top 5 comments
                                if len(comment.body) > 100:  # Filter short comments
                                    comment_doc = Document(
                                        page_content=(
                                            f"Comment on: {submission.title}\n\n"
                                            f"Content: {comment.body}\n\n"
                                            f"Score: {comment.score}"
                                        ),
                                        metadata={
                                            "source": f"https://reddit.com{comment.permalink}",
                                            "type": "reddit",
                                            "score": comment.score,
                                            "created_utc": comment.created_utc,
                                            "subreddit": subreddit_name,
                                            "parent_title": submission.title,
                                        },
                                    )
                                    documents.append(comment_doc)
                                    self.metrics.total_reddit += 1

                except Exception as e:
                    logging.error(f"Error scraping subreddit {subreddit_name}: {e}")
                    continue

            logging.info(f"Scraped {len(documents)} Reddit documents")
            return documents

        except Exception as e:
            logging.error(f"Reddit scraping failed: {e}")
            return []

    async def load_sources(self) -> List[Document]:
        """Load documents from all sources concurrently."""
        try:
            documents = []

            # Process PDFs
            pdf_files = [f for f in os.listdir("docs") if f.endswith(".pdf")]
            pdf_tasks = []
            for file in pdf_files:
                file_path = os.path.join("docs", file)
                pdf_tasks.append(self.process_pdf(file_path))

            # Process PDFs concurrently
            pdf_results = await asyncio.gather(*pdf_tasks)
            for docs in pdf_results:
                documents.extend(docs)
                self.metrics.total_pdfs += len(docs)

            # Process websites concurrently
            async with aiohttp.ClientSession() as session:
                website_tasks = []
                for url in WEBSITE_SOURCES:
                    website_tasks.append(self.process_website(url))
                website_results = await asyncio.gather(*website_tasks)

                for doc in website_results:
                    if doc:
                        documents.append(doc)
                        self.metrics.total_websites += 1

            # Process Reddit content
            reddit_docs = await self.scrape_reddit()
            documents.extend(reddit_docs)

            logging.info(f"Loaded {len(documents)} documents in total")
            logging.info(
                f"Metrics: PDFs={self.metrics.total_pdfs}, "
                f"Websites={self.metrics.total_websites}, "
                f"Reddit={self.metrics.total_reddit}"
            )

            return documents

        except Exception as e:
            logging.error(f"Error loading sources: {e}")
            raise

    def create_vector_store(self, chunks: List[Document]) -> FAISS:
        """Create and validate vector store."""
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(chunks, embeddings)

        # Validate vector store
        test_queries = ["What is OPT?", "How to apply for OPT?"]
        for query in test_queries:
            results = db.similarity_search(query, k=2)
            if not results:
                raise ValueError(f"Vector store validation failed for: {query}")

        db.save_local("faiss_index")
        return db


async def main():
    """Main execution function."""
    try:
        processor = ContentProcessor()

        async with aiohttp.ClientSession() as session:
            documents = await processor.load_sources()

            # Process and validate documents
            valid_docs = [doc for doc in documents if processor.validate_content(doc)]
            if not valid_docs:
                raise ValueError("No valid documents found")

            # Create chunks
            chunks = []
            for doc in valid_docs:
                doc.page_content = processor.preprocess_text(doc.page_content)
                chunks.extend(processor.text_splitter.split_documents([doc]))

            # Create vector store
            db = processor.create_vector_store(chunks)
            logging.info("Processing completed successfully")
            return True

    except Exception as e:
        logging.error(f"Processing failed: {e}")
        return False


if __name__ == "__main__":
    import asyncio
    import aiohttp

    # On MacOS, use a different event loop policy
    if sys.platform == "darwin":
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

    success = asyncio.run(main())
    if not success:
        exit(1)
