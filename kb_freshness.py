"""Keeping the knowledge base fresh: dating, de-duplicating and replacing.

Pure functions (no OpenAI / network) so they're cheap to test. Used by
update_knowledge.py when writing the index and by f1rstaid.py when ranking
and labelling retrieved chunks.

Why this exists: the weekly job used to only ever *append*. Re-fetching an
unchanged page re-embedded and re-added the same chunks every week (the
index grew ~6 MB/week), an updated page left its old version in the index
next to the new one, and no chunk carried a date, so neither retrieval nor
the model could tell last week's alert from a 2022 FAQ.
"""

import hashlib
import re
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

from langchain_core.documents import Document

# Refuse to apply a change plan that would delete more than this share of
# the existing index in one run -- a bug or a bad fetch must not be able to
# silently gut the knowledge base.
MAX_DELETE_FRACTION = 0.5


def content_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip().lower()
    return hashlib.sha256(normalized.encode()).hexdigest()[:20]


def _hash_of(doc: Document) -> str:
    return doc.metadata.get("content_hash") or content_hash(doc.page_content)


def parse_date(value) -> Optional[datetime]:
    """ISO string, epoch seconds (Reddit's created_utc) or datetime -> aware
    UTC datetime; None if it can't be understood."""
    if value in (None, ""):
        return None
    try:
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, (int, float)):
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
        else:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, OSError, OverflowError):
        return None


def published_date(metadata: Dict) -> Optional[datetime]:
    """When the content itself was written -- `published` (news, Federal
    Register, RSS) or `created_utc` (Reddit). Deliberately NOT fetched_at:
    that is only when we last downloaded a page, not how current it is."""
    return parse_date(metadata.get("published")) or parse_date(metadata.get("created_utc"))


def stamp_documents(docs: Iterable[Document], now: Optional[datetime] = None) -> List[Document]:
    now = now or datetime.now(timezone.utc)
    docs = list(docs)
    for doc in docs:
        doc.metadata["fetched_at"] = now.isoformat()
        doc.metadata["content_hash"] = content_hash(doc.page_content)
    return docs


def plan_index_changes(
    existing: Dict[str, Document], new_docs: List[Document]
) -> Tuple[List[str], List[Document], Dict[str, int]]:
    """Decide what to remove from / add to the index.

    existing: {docstore_id: Document} for everything currently indexed.
    Returns (ids_to_delete, docs_to_add, stats).

    1. Exact duplicates already in the index are removed (keep one).
    2. A new chunk whose content is already indexed is skipped -- it costs
       no embedding call.
    3. For each source URL present in this batch, previously indexed chunks
       of that source whose content is no longer present are removed: the
       page changed, so its old version is stale.
    """
    to_delete: List[str] = []
    kept_hash_to_id: Dict[str, str] = {}
    duplicates = 0
    for doc_id, doc in existing.items():
        h = _hash_of(doc)
        if h in kept_hash_to_id:
            to_delete.append(doc_id)
            duplicates += 1
        else:
            kept_hash_to_id[h] = doc_id

    new_hashes_by_source: Dict[str, set] = {}
    for doc in new_docs:
        new_hashes_by_source.setdefault(doc.metadata.get("source", ""), set()).add(_hash_of(doc))

    replaced = 0
    deleted = set(to_delete)
    for doc_id, doc in existing.items():
        if doc_id in deleted:
            continue
        source = doc.metadata.get("source", "")
        fresh = new_hashes_by_source.get(source)
        if fresh is not None and _hash_of(doc) not in fresh:
            to_delete.append(doc_id)
            deleted.add(doc_id)
            replaced += 1

    still_indexed = {
        _hash_of(doc) for doc_id, doc in existing.items() if doc_id not in deleted
    }
    to_add: List[Document] = []
    skipped = 0
    seen_new = set()
    for doc in new_docs:
        h = _hash_of(doc)
        if h in still_indexed or h in seen_new:
            skipped += 1
            continue
        seen_new.add(h)
        to_add.append(doc)

    stats = {
        "added": len(to_add),
        "skipped_unchanged": skipped,
        "removed_duplicates": duplicates,
        "removed_stale": replaced,
    }
    return to_delete, to_add, stats


def delete_is_safe(existing_count: int, delete_count: int) -> bool:
    return existing_count == 0 or delete_count / existing_count <= MAX_DELETE_FRACTION


# --- retrieval: freshness weighting and labels -------------------------------

def freshness_multiplier(metadata: Dict, now: Optional[datetime] = None) -> float:
    """Small nudge toward recently-published content (applied to the
    similarity weight, so >1 helps). Undated content is neutral -- most
    official pages have no publication date and must not be penalised."""
    published = published_date(metadata)
    if published is None:
        return 1.0
    age_days = ((now or datetime.now(timezone.utc)) - published).days
    if age_days <= 60:
        return 1.2
    if age_days <= 180:
        return 1.1
    if age_days <= 540:
        return 1.0
    return 0.9


# Law firms, associations and tax-help blogs: useful and fast, but secondary
# (and some sell services). They are indexed with type "web" like the
# government pages, so the tier is derived from the source URL at retrieval
# time -- that also re-tiers everything already in the index, with no
# re-embedding or migration.
COMMENTARY_DOMAINS = (
    "murthy.com",
    "fragomen.com",
    "aila.org",
    "nafsa.org",
    "bal.com",
    "ogletree.com",
    "mcenteelaw.com",
    "alcorn.law",
    "immigrationlawgroup.net",
    "sprintax.com",
    "cptdog.com",
    "stilt.com",
)


# University international-office sites -> school name. Their pages describe
# that school's own procedures, so they get their own tier and a label naming
# the school (see doc_label); keep in sync with
# config.sources.SCHOOL_GUIDANCE_SOURCES (a test checks every URL there maps).
SCHOOL_DOMAINS = {
    "nyu.edu": "NYU",
    "northeastern.edu": "Northeastern University",
    "columbia.edu": "Columbia University",
    "asu.edu": "Arizona State University",
    "usc.edu": "USC",
    "illinois.edu": "University of Illinois Urbana-Champaign",
    "cmu.edu": "Carnegie Mellon University",
    "purdue.edu": "Purdue University",
    "ucla.edu": "UCLA",
    "bu.edu": "Boston University",
    "gmu.edu": "George Mason University",
    "nec.edu": "New England College",
    "utdallas.edu": "UT Dallas",
    "uta.edu": "UT Arlington",
}


def _host(metadata: Dict) -> str:
    return (urlparse(metadata.get("source", "")).hostname or "").lower()


def school_name(metadata: Dict) -> Optional[str]:
    host = _host(metadata)
    for domain, name in SCHOOL_DOMAINS.items():
        if host == domain or host.endswith("." + domain):
            return name
    return None


def source_tier(metadata: Dict) -> str:
    """Trust tier used for ranking and labelling: pdf, web (official),
    commentary, news or reddit. Anything unrecognised falls back to its
    stored type."""
    doc_type = metadata.get("type")
    if doc_type == "web":
        if school_name(metadata):
            return "school"
        host = _host(metadata)
        if any(host == d or host.endswith("." + d) for d in COMMENTARY_DOMAINS):
            return "commentary"
    return doc_type or "unknown"


_TYPE_LABELS = {
    "pdf": "official document",
    "web": "official/web page",
    "commentary": "law-firm/expert commentary",
    "school": "university international-office guidance",
    "news": "news report",
    "reddit": "community post (Reddit)",
}


def doc_label(metadata: Dict) -> str:
    """Short label shown to the model with each chunk, e.g.
    'news report, published 2026-09-30'."""
    tier = source_tier(metadata)
    label = _TYPE_LABELS.get(tier, "document")
    if tier == "school":
        label = f"{label} ({school_name(metadata)} only)"
    published = published_date(metadata)
    if published:
        return f"{label}, published {published.date().isoformat()}"
    fetched = parse_date(metadata.get("fetched_at"))
    if fetched:
        return f"{label}, retrieved {fetched.date().isoformat()}"
    return f"{label}, date unknown"


def date_caption(metadata: Dict) -> str:
    """Human caption for a citation card ('Published Mar 03, 2026')."""
    published = published_date(metadata)
    if published:
        return f"Published {published.strftime('%b %d, %Y')}"
    fetched = parse_date(metadata.get("fetched_at"))
    if fetched:
        return f"Retrieved {fetched.strftime('%b %d, %Y')}"
    return ""
