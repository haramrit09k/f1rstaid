from html import escape
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlparse
import os.path

import requests
import streamlit as st
from dotenv import load_dotenv
from openai import RateLimitError
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Optional

import rules_engine

# Picks up OPENAI_API_KEY (and anything else) from a local .env if present,
# so the app works without the user having to export it manually into the
# shell before running `streamlit run` -- the sidebar text_input is still
# the primary path and always wins if a key is entered there.
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("f1rstaid.log"), logging.StreamHandler()],
)

_F1_KEYWORDS = {
    # Visa & status
    "f-1", "f1", "visa", "sevis", "sevp", "i-20", "i20", "status", "grace period",
    "cap gap", "out of status", "full-time", "enrollment", "dso",
    "designated school official",
    # Work authorization
    "opt", "cpt", "ead", "employment authorization", "work authorization",
    "optional practical training", "curricular practical training",
    "stem opt", "stem extension", "i-765", "i-983",
    # Agencies & forms
    "uscis", "dhs", "ice", "cbp", "i-94", "i-539", "sevis fee", "i-901",
    # Processes
    "transfer", "school transfer", "change of status", "reinstatement",
    "travel", "reentry", "re-entry", "port of entry", "travel signature",
    # Tax & finance
    "tax", "itin", "w-2", "1042-s", "social security", "ssn", "fica",
    # Employment
    "internship", "employer", "h-1b", "h1b", "sponsor", "lottery",
    "off-campus", "on-campus",
}


class RelevanceFields(BaseModel):
    """Second-stage relevance judgment, used only when a question already
    matched an _F1_KEYWORDS keyword. That match is necessary but not
    sufficient -- e.g. "on F-1 OPT, can we invest in Roth IRA?" contains
    'F-1' and 'OPT' but is really a tax/retirement-account question with no
    supporting content in this app's knowledge base at all. Mirrors the
    `applies` field pattern already used throughout rules_engine.py.

    Deliberately a lower, more inclusive bar than rules_engine's `applies`
    fields: those gate a *canned, deterministic* answer, so precision
    matters (a wrong rule match is a wrong answer). This just gates whether
    to *attempt* a RAG answer at all -- and RAG already has its own safety
    net (it abstains when the retrieved context doesn't support an answer).
    So this should only filter out what's genuinely unrelated to F-1
    status, not everything RAG might not have a clean answer for. Tuned
    live against a real false-decline: "can I collect unemployment
    insurance benefits on OPT?" is a legitimate status-adjacent question
    (does this affect maintaining F-1 status?) even though the honest
    answer is "ask your DSO" -- very different from the Roth IRA case,
    where the question has nothing to do with visa status at all.
    """

    applies: bool = Field(
        description=(
            "True if this question is about F-1 student visa status, "
            "maintaining status, OPT/CPT, SEVIS, employment authorization, "
            "or something DHS/SEVIS regulations for F-1 students actually "
            "govern -- including secondary effects on status, like whether "
            "receiving a benefit affects the unemployment-day count SEVIS "
            "tracks during OPT. The test: would a DSO (who only handles "
            "immigration status, not personal finance/tax/general life "
            "advice) actually be the right person to ask, because the "
            "answer depends on immigration regulations -- or would this "
            "really be a question for an accountant, bank, or financial "
            "advisor regardless of the person's visa status, with F-1/OPT "
            "just being incidental context about who's asking? Set this "
            "False for the latter (e.g. general investment, retirement "
            "account, or banking questions -- opening a Roth IRA has "
            "nothing to do with F-1/SEVIS regulations even if the person "
            "asking happens to be on OPT). When genuinely uncertain "
            "whether DHS/SEVIS regulations are actually implicated, set "
            "this True."
        )
    )

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are F1rstAid, an expert assistant specializing in F-1 student visa "
        "regulations in the United States. You help international students understand "
        "USCIS, DHS, and related immigration requirements.\n\n"
        "Use ONLY the following context to answer the question. If the context does not "
        "contain enough information to answer confidently, say exactly: \"I don't have "
        "enough information to answer this accurately — please consult your DSO or visit "
        "uscis.gov directly.\"\n\n"
        "Guidelines:\n"
        "- Reference the source when possible (e.g., \"Per USCIS...\", \"According to "
        "DHS Study in the States...\")\n"
        "- Be specific about form numbers, timelines, and eligibility requirements when "
        "the context supports it\n"
        "- Never invent facts, numbers, or rules that are not present in the context. "
        "If the context explicitly states a requirement or condition, you may state the "
        "direct consequence of it being met or not met -- that is applying the stated "
        "rule, not guessing. Still abstain if the context is silent or ambiguous about "
        "the student's specific situation.\n"
        "- Immigration regulations change frequently — always recommend verifying with a "
        "DSO or USCIS\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
)

# Bounds the condense_question() LLM call to the last few exchanges rather
# than the whole conversation -- keeps that extra call's cost/latency flat
# regardless of how long a session gets, and a follow-up almost never
# depends on anything more than a couple of turns back.
CONDENSE_HISTORY_TURNS = 6  # 3 user/assistant exchanges

# Reference words that mean a follow-up is pointing back at the
# conversation ("what form do I need for THAT?"). Checked with word
# boundaries, not bare substring, so e.g. "this" doesn't false-positive on
# an unrelated word that happens to contain it.
_CONDENSE_REFERENCE_WORDS = ["that", "it", "this", "those", "these", "the same", "same as"]

# A follow-up this short, arriving right after the assistant asked a
# clarifying question, is almost certainly a direct answer to it ("I'm on
# initial OPT", "stem opt") rather than a new, independent question -- a
# genuinely new question is essentially never this short. Verified live
# this distinguishes the two cases that matter: "I'm on initial OPT" (4
# words) and "stem opt" (2 words) are direct answers; "on f1 opt can i
# invest in a roth ira" (9 words) is not, even though it also arrived right
# after a clarifying question about OPT phase.
_DIRECT_ANSWER_WORD_LIMIT = 6


def _answers_pending_clarification(question: str, chat_history: List[Dict]) -> bool:
    """True if the assistant's last message was a clarifying question and
    this follow-up looks like a short, direct answer to it -- the
    highest-confidence, highest-stakes case (it feeds straight into
    rules_engine's exact day-math), so it's detected deterministically
    rather than by asking an LLM to judge it. See condense_question()'s
    docstring for why LLM composition isn't used for this case either."""
    if len(chat_history) < 2:
        return False
    last, prior = chat_history[-1], chat_history[-2]
    return (
        last.get("role") == "assistant"
        and "?" in (last.get("content") or "")
        and prior.get("role") == "user"
        and len(question.split()) <= _DIRECT_ANSWER_WORD_LIMIT
    )


def _has_reference_word(question: str) -> bool:
    """True if the follow-up uses a word like 'that'/'it'/'this' that could
    be pointing back at the conversation (e.g. 'what form do I need for
    THAT?'). A much lower-stakes signal than
    _answers_pending_clarification -- feeds only into an LLM composition
    call for the RAG path, not a hard-coded arithmetic rule, so occasional
    imperfection here is tolerable in a way it isn't for the day-math case.

    Uses a real word-boundary regex, not space-padded substring matching --
    verified live that the naive version missed "what form do I need for
    THAT?" because the trailing "?" meant "that?" never matched " that "
    padded with plain spaces on both sides.
    """
    clean_q = question.strip().lower()
    return any(re.search(rf"\b{re.escape(w)}\b", clean_q) for w in _CONDENSE_REFERENCE_WORDS)


# Only ever invoked after _has_reference_word() has already decided (in
# code, not by asking the LLM) that this follow-up points back at the
# conversation -- so this prompt's only job is composing the merged
# question, not judging whether merging is warranted. Splitting judgment
# from composition like this, rather than asking one call to do both, is
# what actually fixed the unreliability -- see condense_question()'s
# docstring for what was tried first.
CONDENSE_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=(
        "This follow-up question depends on the conversation below to be "
        "understood on its own -- either it's a direct answer to a "
        "clarifying question the assistant just asked, or it uses a word "
        "like 'that'/'it'/'this' referring back to something already "
        "discussed. Rewrite it as a standalone question that includes "
        "every fact from the conversation needed to answer it -- dates, "
        "day counts, which OPT phase, or whatever else was already "
        "stated. Do not answer the question yourself, only rewrite it.\n\n"
        "Example: assistant asked 'are you on initial OPT or the STEM "
        "extension?' after the student said they'd used 80 of their "
        "unemployment days; follow-up is 'I'm on initial OPT.' -> 'I've "
        "used 80 of my unemployment days and I'm on initial OPT -- how "
        "many days do I have left?'\n\n"
        "Conversation so far:\n{chat_history}\n\n"
        "Follow-up question: {question}\n\n"
        "Standalone question:"
    ),
)

# Applied at query time (see SourceWeightedRetriever), not ingest time.
# ingest.py's rerank_documents() applies these same weights once when
# building the index, which has no effect on FAISS's similarity ranking at
# query time -- plain cosine/L2 similarity search ranks purely by embedding
# distance, and casually-phrased Reddit posts often embed *closer* to a
# casually-phrased user question than formally-worded official policy text
# does, even though the official source is more authoritative. This
# resurfaces official sources by discounting Reddit's effective distance.
SOURCE_TYPE_WEIGHTS = {"pdf": 2.0, "web": 2.2, "reddit": 0.5}
DEFAULT_SOURCE_WEIGHT = 1.0

# A few starter questions so a first-time visitor isn't staring at a blank
# input box -- there's no other onboarding in this app.
# (short chip label, full question actually submitted) -- kept separate so
# the chips can read at a glance without changing what rules_engine/RAG
# actually sees, which is the exact wording already verified this session.
EXAMPLE_QUESTIONS = [
    ("Unemployment days on OPT?", "How many days of unemployment am I allowed on OPT?"),
    ("60-day grace period?", "What is the 60-day grace period after I graduate?"),
    ("Does cap-gap extend work auth?", "Does cap-gap extend my work authorization?"),
    ("STEM OPT extension docs?", "What documents do I need for a STEM OPT extension?"),
]

# --- Shared-key usage limiting ----------------------------------------------
#
# Without this, anyone visiting the deployed app uses the owner's own
# OPENAI_API_KEY automatically (the sidebar falls back to it whenever no one
# types their own) -- with zero limit on how many real API calls a random
# visitor could rack up. Two layers, checked together:
#
#   1. Per-session limit: lets someone try the app for real, then asks them
#      to bring their own key to keep going. This alone is NOT a hard
#      guarantee -- a new private/incognito window is a fresh session, so
#      this is a soft, honest "try it out" gate, not abuse-proofing.
#   2. A global daily counter, persisted to a small JSON file on disk, so
#      the limit isn't purely per-session and can't be trivially reset by
#      opening new sessions. Heroku's filesystem is ephemeral -- this
#      resets on every dyno restart/redeploy, and under concurrent
#      requests the read-modify-write below isn't atomic, so this is a
#      best-effort soft cap, not a precise one. The one real, hard backstop
#      against actually running up an unexpected bill is a spending limit
#      configured directly in the OpenAI account dashboard -- that's a
#      platform.openai.com setting, not something this app can set on its
#      own behalf.
FREE_TRIAL_QUERY_LIMIT = 5  # per browser session, using the shared/owner key
DAILY_SHARED_KEY_LIMIT = 50  # total shared-key queries per day, all sessions
USAGE_TRACKER_PATH = Path("usage_tracker.json")


def _read_daily_count(path: Path) -> int:
    """Generic same-day counter read, shared by the shared-key usage cap
    and the feedback-issue cap below -- both need the identical
    read-today's-count-or-zero logic, just against a different file."""
    today = str(date.today())
    try:
        data = json.loads(path.read_text())
        if data.get("date") == today:
            return int(data.get("count", 0))
    except (FileNotFoundError, json.JSONDecodeError, ValueError, OSError):
        pass
    return 0


def _increment_daily_count(path: Path) -> int:
    today = str(date.today())
    count = _read_daily_count(path) + 1
    try:
        path.write_text(json.dumps({"date": today, "count": count}))
    except OSError as e:
        logging.error(f"Failed to persist daily counter at {path}: {e}")
    return count


def _read_daily_shared_key_usage() -> int:
    return _read_daily_count(USAGE_TRACKER_PATH)


def _increment_daily_shared_key_usage() -> int:
    return _increment_daily_count(USAGE_TRACKER_PATH)


def shared_key_limit_reached() -> Tuple[bool, str]:
    """Checked before spending a shared-key API call on a new question --
    never after. Only meaningful when the visitor is using the owner's
    fallback key; a visitor who entered their own key is spending their own
    credits and isn't subject to either limit (see main()'s call site)."""
    session_count = st.session_state.get("shared_key_query_count", 0)
    if session_count >= FREE_TRIAL_QUERY_LIMIT:
        return True, (
            f"🚦 You've used all {FREE_TRIAL_QUERY_LIMIT} free trial questions "
            "for this session. To keep chatting, enter your own OpenAI API key "
            "in the sidebar -- get one at "
            "[platform.openai.com/api-keys](https://platform.openai.com/api-keys)."
        )

    if _read_daily_shared_key_usage() >= DAILY_SHARED_KEY_LIMIT:
        return True, (
            "🚦 This app has reached its shared free-trial usage limit for "
            "today. Please try again tomorrow, or enter your own OpenAI API "
            "key in the sidebar to continue right now -- get one at "
            "[platform.openai.com/api-keys](https://platform.openai.com/api-keys)."
        )

    return False, ""


# --- Thumbs-down feedback -> GitHub issue -----------------------------------
#
# Same daily-cap shape as the shared-key usage limit above, for the same
# reason: without a cap, a burst of downvotes on one broken answer (e.g.
# many visitors hitting the same router false-positive) would file a wall
# of near-duplicate issues instead of one signal worth acting on.
GITHUB_FEEDBACK_REPO = "haramrit09k/f1rstaid"
MAX_FEEDBACK_ISSUES_PER_DAY = 10
FEEDBACK_TRACKER_PATH = Path("feedback_issue_tracker.json")


def create_feedback_issue(
    answer_type: str, answer_text: str, sources: List[str], feedback_text: str
) -> bool:
    """Files a GitHub issue from a thumbs-down. Returns True only if an
    issue was actually created -- callers use this to show an honest
    status ("thanks for the feedback" vs. "flagged for review") rather than
    claiming an issue was filed when the token's missing, the daily cap is
    hit, or the API call itself failed.

    The original question is deliberately NOT included in the issue body.
    This repo is public, and a downvoter's exact question could contain
    personal specifics (employer name, exact dates, individual
    circumstances) that shouldn't become permanent and public just because
    they hit an unhelpful answer. The answer given, its type, the sources
    it cited, and the student's own feedback text are enough to actually
    debug from without that risk.
    """
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        logging.info("GITHUB_TOKEN not configured -- feedback not filed as an issue")
        return False

    if _read_daily_count(FEEDBACK_TRACKER_PATH) >= MAX_FEEDBACK_ISSUES_PER_DAY:
        logging.info("Daily feedback-issue cap reached -- not filing another issue")
        return False

    body = (
        f"**Answer type:** {answer_type}\n\n"
        f"**Answer given:**\n> {answer_text}\n\n"
        f"**Sources cited:** {', '.join(sources) if sources else 'none'}\n\n"
        f"**Student feedback:** {feedback_text.strip() if feedback_text and feedback_text.strip() else '(no comment given)'}\n\n"
        "_Filed automatically from a \U0001f44e in the app. The original "
        "question is intentionally omitted -- this repo is public._"
    )

    try:
        response = requests.post(
            f"https://api.github.com/repos/{GITHUB_FEEDBACK_REPO}/issues",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
            json={
                "title": f"User feedback: {answer_type} answer flagged as unhelpful",
                "body": body,
                "labels": ["user-feedback"],
            },
            timeout=10,
        )
        if response.status_code == 201:
            _increment_daily_count(FEEDBACK_TRACKER_PATH)
            return True
        logging.error(
            f"GitHub issue creation failed: {response.status_code} {response.text[:200]}"
        )
        return False
    except requests.RequestException as e:
        logging.error(f"GitHub issue creation request failed: {e}")
        return False


class SourceWeightedRetriever(BaseRetriever):
    """Retrieves a wider candidate pool via similarity search, then re-ranks
    by source-type weight before truncating to k."""

    db: FAISS
    k: int = 3
    candidate_pool_size: int = 30

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        candidates = self.db.similarity_search_with_score(query, k=self.candidate_pool_size)

        def adjusted_distance(doc_and_score):
            doc, distance = doc_and_score
            weight = SOURCE_TYPE_WEIGHTS.get(doc.metadata.get("type"), DEFAULT_SOURCE_WEIGHT)
            return distance / weight

        ranked = sorted(candidates, key=adjusted_distance)
        return [doc for doc, _ in ranked[: self.k]]


@dataclass
class AppConfig:
    """Configuration for F1rstAid application."""

    model_name: str = "gpt-3.5-turbo"
    vector_store_path: str = "faiss_index"
    search_k: int = 3
    temperature: float = 0.2
    GENERIC_HELP_QUESTIONS = {
        "help": {
            "response": """
Hello! I'm F1rstAid, your virtual assistant for F-1 visa questions.
📚 **My Expertise**:\n
I specialize in F-1 visa regulations including:
- OPT/CPT requirements and applications
- STEM OPT extensions (Form I-983)
- Employment authorization documents (Form I-765)
- Maintaining visa status
- Travel restrictions and re-entry requirements

Ask me specific questions like:
- 'How long does OPT processing take after submitting Form I-765?'
- 'What are the CPT requirements for summer internships?'
                """,
            "triggers": [
                "what can you do",
                "how to use",
                "help",
                "expertise",
                "what do i ask you",
                "what's your name"
            ],
        },
        "question_guidance": {
            "response": """
🔍 **How to Ask Effective Questions**:\n\n
1. Include specific terms: 'OPT', 'CPT', 'I-765', 'I-983'\n
2. Mention your situation: 'After H1B denial...', 'As a STEM student...'\n
3. Ask about timelines: 'How long...', 'Processing time for...'\n
4. Request form guidance: 'Section 5 of I-983...'\n

Example: 'What documents do I need for STEM OPT extension?'\n
                """,
            "triggers": [
                "ask a question",
                "ask you a question",
                "how do i ask",
                "formulate",
                "effective questions",
                "how to ask you",
            ],
        },
    }


# --- Answer classification for UI presentation -----------------------------
# get_answer() doesn't tag which code path produced a given answer, and we
# can't change its return shape (that's backend/rules_engine territory). So
# this infers the type purely from what's already observable: an empty
# source_documents list plus which (if any) of the app's own known
# boilerplate strings the result matches. This is UI-only best-effort
# classification, used solely to pick a badge/disclaimer -- never to change
# behavior.
_EMPTY_QUESTION_RESULT = "Please enter a question about F-1 visas, OPT, or CPT."
_ERROR_RESULT = "Error processing request. Please try again."
_RATE_LIMIT_MARKER = "hit its OpenAI usage limit"
_DECLINE_MARKER = "🚦 **Relevance Check**"
_ABSTAIN_MARKER = "don't have enough information"
_HELP_MARKERS = ("Hello! I'm F1rstAid", "🔍 **How to Ask Effective Questions**")

_ANSWER_BADGES = {
    # (label, tooltip, css class)
    "rule": (
        "⚡ Instantly calculated",
        "Computed directly from stated F-1 rules -- no AI involved in this answer.",
        "badge-rule",
    ),
    "rag": (
        "🤖 AI-generated from sources",
        "Synthesized by an LLM from the retrieved documents below -- verify specifics.",
        "badge-rag",
    ),
    "abstained": (
        "🤷 Not enough info",
        "The retrieved sources didn't contain a confident answer -- nothing below was actually used.",
        "badge-abstained",
    ),
    "declined": ("🚦 Off-topic", None, "badge-declined"),
    "help": ("ℹ️ Guidance", None, "badge-help"),
    "rate_limited": ("🔌 Usage limit", None, "badge-declined"),
}


def classify_answer(answer: Dict) -> str:
    """Best-effort classification of which code path produced this answer,
    for UI presentation only (badge + disclaimer + whether to show sources).
    See module note above.

    rules_engine answers now carry a fixed citation Document too (not just
    RAG answers), so "has source_documents" alone no longer distinguishes
    them -- each rules_engine citation is tagged metadata["rule_based"]=True
    specifically so this can still tell them apart correctly.

    Checked before the source-based rag/rule branches: the retriever always
    returns its top-k docs regardless of whether the LLM found them
    sufficient, so a RAG call that abstains still has non-empty
    source_documents. Without this check first, an abstention ("I don't
    have enough information...") got classified as "rag" and rendered with
    a "🤖 AI-generated from sources" badge, a citation row, and the actual
    (unused) retrieved docs below it -- implying those sources backed an
    answer that was never actually given.
    """
    result = (answer.get("result") or "").strip()
    docs = answer.get("source_documents") or []
    if _ABSTAIN_MARKER in result.lower():
        return "abstained"
    if docs and all(d.metadata.get("rule_based") for d in docs):
        return "rule"
    if docs:
        return "rag"
    if result == _EMPTY_QUESTION_RESULT:
        return "empty"
    if result == _ERROR_RESULT:
        return "error"
    if _RATE_LIMIT_MARKER in result:
        return "rate_limited"
    if _DECLINE_MARKER in result:
        return "declined"
    if any(marker in result for marker in _HELP_MARKERS):
        return "help"
    # No sources and none of the known boilerplate templates match -- treat
    # as a rule answer defensively (matches the old fallback behavior).
    return "rule"


# Injected once per script run (see main()) rather than once per source
# block, which is what the old code did -- format_sources() used to embed
# a full <style> tag in its return value and got called twice per answer
# (once for official sources, once for Reddit), duplicating the CSS.
APP_CSS = """
<style>
.answer-badge {
    display: inline-block;
    font-size: 0.8em;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 12px;
    margin: 4px 0 12px 0;
}
.badge-rule { background-color: #e6f4ea; color: #1e7e34; }
.badge-rag { background-color: #e8f0fe; color: #1a56c4; }
.badge-declined { background-color: #fdecea; color: #b3261e; }
.badge-help { background-color: #fff4e5; color: #a05a00; }
.badge-abstained { background-color: #eaeef2; color: #57606a; }

/* Compact by design -- these render inside a collapsed st.expander (see
   display_answer), so they no longer need to visually announce themselves
   as their own section the way the old always-open cards did. A one-line
   heading row (index + type + link) plus a short preview, not a bordered
   card with its own sub-headings for "Type:"/"Source:"/"Preview:". */
.source-block {
    /* Background here is a fixed light color regardless of Streamlit's
       theme -- so text color inside has to be fixed too, not inherited.
       Left as inherited before, it picked up Streamlit's dark-theme body
       color (near-white), which is invisible on this always-light card. */
    background-color: #ffffff;
    color: #24292f;
    border-left: 3px solid #0366d6;
    margin: 6px 0;
    padding: 8px 12px;
    border-radius: 4px;
}
.source-block.source-reddit {
    border-left-color: #ff4500;
}
.source-heading {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
    font-size: 0.85em;
}
.source-index {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 1.3em;
    height: 1.3em;
    border-radius: 50%;
    background-color: #f1f8ff;
    color: #0366d6;
    font-weight: 600;
    font-size: 0.85em;
}
.source-type {
    color: #6a737d;
    font-size: 0.8em;
    letter-spacing: 0.03em;
}
.source-block a {
    color: #0366d6;
    text-decoration: none;
    word-break: break-all;
}
.source-block a:hover {
    text-decoration: underline;
}
.preview-text {
    margin: 4px 0 0 0;
    font-size: 0.82em;
    line-height: 1.4;
    color: #57606a;
}

.disclaimer-box {
    font-size: 0.85em;
    color: #6a737d;
    background-color: #f6f8fa;
    border-left: 3px solid #d0d7de;
    padding: 8px 12px;
    border-radius: 4px;
    margin: 8px 0 20px 0;
}

/* Keep long/unbroken source URLs and code from forcing horizontal scroll
   on narrow (mobile) viewports. */
.source-block, .preview-text {
    overflow-wrap: anywhere;
}

@media (max-width: 640px) {
    .source-block { padding: 6px 10px; }
}
</style>
"""


class F1rstAidApp:
    """Main application class for F1rstAid."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.qa_chain = None
        self.embeddings = None
        self.db = None
        self.llm = None

    def initialize(self) -> bool:
        """Initialize the application components."""
        try:
            if not self._check_environment():
                return False

            logging.info("Initializing embeddings...")
            self.embeddings = OpenAIEmbeddings()

            logging.info("Loading vector store...")
            self.db = FAISS.load_local(
                self.config.vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

            logging.info("Setting up retriever and QA chain...")
            retriever = SourceWeightedRetriever(
                db=self.db,
                k=self.config.search_k,
                # 10 wasn't enough -- verified live that Reddit dominates
                # the raw candidate pool deeply enough that the first
                # official (pdf) result didn't appear until rank 20 for a
                # typical eligibility question. 30 reliably captures at
                # least one official source to weight up.
                candidate_pool_size=max(self.config.search_k * 10, 30),
            )

            self.llm = ChatOpenAI(
                model_name=self.config.model_name,
                temperature=self.config.temperature,
            )
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=retriever,
                chain_type="stuff",
                return_source_documents=True,
                chain_type_kwargs={"prompt": QA_PROMPT},
            )

            logging.info("Application initialized successfully")
            return True

        except Exception as e:
            logging.error(f"Initialization failed: {str(e)}")
            return False

    def get_knowledge_base_last_updated(self) -> Optional[str]:
        """Human-readable freshness date for the currently-loaded index,
        read from faiss_index/last_updated.json -- written by
        update_knowledge.py's pipeline on every successful refresh (see
        _write_last_updated() there). Without this surfaced somewhere,
        there's no way for a student to know whether yesterday's policy
        change made it into an answer, or won't until next week's
        scheduled run -- exactly the freshness-pipeline work earlier this
        project never had anywhere to actually show up.

        Returns None (never raises) if the file doesn't exist or can't be
        parsed -- e.g. an isolated test index that was never run through
        the real pipeline -- callers should treat this as optional context,
        not something the app depends on.
        """
        try:
            path = Path(self.config.vector_store_path) / "last_updated.json"
            data = json.loads(path.read_text())
            dt = datetime.fromisoformat(data["last_updated"])
            return dt.strftime("%B %d, %Y")
        except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError, OSError):
            return None

    def get_secret(self, group, key, env_var=None):
        """
        Retrieve a secret from st.secrets first, then fall back to environment variables.

        Parameters:
        group (str): The group name in the TOML configuration (e.g., "openai").
        key (str): The key within that group (e.g., "api_key").
        env_var (str): Optional environment variable name. If not provided,
                        defaults to GROUP_KEY in uppercase (e.g., "OPENAI_API_KEY").

        Returns:
        The secret value or None if not found.
        """
        # Determine the environment variable name if not provided.
        if env_var is None:
            env_var = f"{group.upper()}_{key.upper()}"

        logging.info(f"Fetching secret: {group}/{key}")
        logging.info(f"Environment variable: {env_var}")

        # Try to fetch from st.secrets.
        try:
            # Check if st.secrets exists and has the requested group/key.
            if hasattr(st, "secrets") and st.secrets:
                if group in st.secrets and key in st.secrets[group]:
                    return st.secrets[group][key]
        except Exception as e:
            logging.info(f"st.secrets not available: {e}")

        # Fallback to using os.getenv.
        return os.getenv(env_var)

    def _check_environment(self) -> bool:
        """Verify environment setup."""
        api_key = get_api_key()
        if not api_key:
            logging.error("OPENAI_API_KEY not found")
            return False
        os.environ["OPENAI_API_KEY"] = api_key
        return True

    _OFF_TOPIC_EXPLANATION = (
        "This doesn't appear to be an F-1 visa question. "
        "I can help with OPT/CPT eligibility, SEVIS, employment authorization, "
        "maintaining F-1 status, travel requirements, and related topics."
    )

    def _is_relevant_question(self, question: str) -> Tuple[bool, str]:
        """Two-stage relevance check. Stage 1 (free, keyword-only): if no
        _F1_KEYWORDS term appears at all, decline immediately -- this alone
        correctly catches the obvious case ("what's the weather today?").
        Stage 2 (LLM-judged, only reached if stage 1 matched): a keyword
        match is necessary but not sufficient -- a tax/financial/general
        question that merely mentions "F-1" or "OPT" as context (e.g. "on
        F-1 OPT, can we invest in Roth IRA?") would pass stage 1 and then
        just waste a RAG retrieval+generation call that can only abstain,
        since nothing in the knowledge base actually covers it. Stage 2
        catches that case with one cheap classification call instead --
        net cheaper than today for false positives, and gives an honest
        "off-topic" decline instead of a vague "not enough information."

        Gracefully degrades to stage-1-only if self.llm isn't set (e.g. a
        test constructing F1rstAidApp without calling initialize()) or if
        the LLM call itself fails -- this check enhances relevance
        detection, it should never be why a genuinely relevant question
        can't be answered.
        """
        clean_q = question.strip().lower()

        # First check for predefined help questions
        for entry in self.config.GENERIC_HELP_QUESTIONS.values():
            if any(trigger in clean_q for trigger in entry["triggers"]):
                return True, "Help question detected"

        if not any(kw in clean_q for kw in _F1_KEYWORDS):
            return False, self._OFF_TOPIC_EXPLANATION

        if self.llm is not None:
            try:
                # Concrete contrastive examples, not just the abstract rule
                # in RelevanceFields.applies's description -- verified live
                # that description text alone wasn't enough for gpt-3.5-turbo
                # to reliably draw this particular distinction (same class
                # of instruction-following limit as the earlier elig-008
                # finding). A directly analogous pair (unemployment
                # insurance vs. Roth IRA -- both financial topics, only one
                # of which SEVIS regulations actually track) anchors the
                # judgment far better than the rule stated abstractly.
                fields = self.llm.with_structured_output(RelevanceFields).invoke(
                    "Is this question genuinely about F-1 student visa status, "
                    "OPT/CPT, SEVIS, employment authorization, or something "
                    "DHS/SEVIS regulations actually govern -- or does it just "
                    "mention F-1/OPT as incidental context for a topic (tax, "
                    "investing, banking, general life advice) that has "
                    "nothing to do with visa status?\n\n"
                    "Examples:\n"
                    "- 'Can I collect unemployment insurance while on OPT?' "
                    "-> True (SEVIS tracks unemployment days during OPT, so "
                    "this touches status maintenance directly)\n"
                    "- 'On F-1 OPT, can I invest in a Roth IRA?' -> False "
                    "(retirement account eligibility is an IRS/tax matter "
                    "with no connection to F-1/SEVIS status)\n"
                    "- 'Do I need to file taxes as an F-1 student?' -> True "
                    "(nonresident-alien tax filing obligations are a common, "
                    "genuinely F-1-specific question)\n"
                    "- 'What's a good budgeting app?' -> False (general "
                    "personal finance, unrelated to visa status)\n\n"
                    f"Question: {question}"
                )
                if not fields.applies:
                    return False, self._OFF_TOPIC_EXPLANATION
            except Exception as e:
                logging.error(f"Relevance LLM check failed, falling back to keyword match: {e}")

        return True, "F-1 topic detected"

    @staticmethod
    def _parse_response_section(response: str, header: str) -> str:
        """Extract specific section from formatted response."""
        try:
            return response.split(header)[1].split("\n")[0].strip()
        except (IndexError, AttributeError):
            return "Unable to parse response."

    def condense_question(self, question: str, chat_history: List[Dict]) -> str:
        """Rewrites a follow-up question into a standalone one using recent
        chat history, so get_answer()'s existing single-turn pipeline
        (help matching, rules_engine dispatch, relevance check, RAG
        retrieval) stays completely unchanged downstream -- it only ever
        sees a self-contained question and never has to reason about
        history itself. This is what turns "what about STEM OPT?" into
        something rules_engine or the retriever can actually act on.

        Two different mechanisms, picked deterministically rather than by
        asking one LLM call to both judge and rewrite in one shot -- that
        combined approach was tried first and verified unreliable across
        several prompt iterations: it either injected an earlier turn's
        facts into a genuinely new, unrelated question (an F-1 unemployment
        question got merged into an unrelated Roth IRA question, which then
        got answered as if it were about unemployment days), or failed to
        inject facts into a real follow-up that needed them (a direct
        answer to the assistant's own clarifying question stopped
        producing a computed result), or even echoed the assistant's
        question back instead of merging in the student's answer.

        1. Answering a pending clarifying question (_answers_pending_
           clarification): plain string concatenation of the question that
           prompted it plus this answer -- no LLM composition at all. This
           is the highest-stakes case (feeds straight into rules_engine's
           exact day-math), so it gets the most reliable possible
           mechanism rather than trusting a paraphrase.
        2. A reference word like "that"/"it" pointing back at the
           conversation (_has_reference_word): LLM composition, since this
           only feeds the lower-stakes RAG path. Falls back to the original
           question on any failure -- condensation is an enhancement,
           never something that should be the reason a question goes
           unanswered.
        3. Neither applies: return the question unchanged, no LLM call.
        """
        if _answers_pending_clarification(question, chat_history):
            prior_question = chat_history[-2].get("content", "")
            merged = f"{prior_question.rstrip('?.! ')}. {question.rstrip('.! ')}."
            logging.info(f"Concatenated clarifying-question answer: {question!r} -> {merged!r}")
            return merged

        if not chat_history or not _has_reference_word(question):
            return question

        recent = chat_history[-CONDENSE_HISTORY_TURNS:]
        history_text = "\n".join(
            f"{'Student' if m.get('role') == 'user' else 'F1rstAid'}: {m.get('content', '')}"
            for m in recent
        )

        try:
            rewritten = self.llm.invoke(
                CONDENSE_PROMPT.format(chat_history=history_text, question=question)
            ).content.strip()
            logging.info(f"Condensed follow-up: {question!r} -> {rewritten!r}")
            return rewritten or question
        except Exception as e:
            logging.error(f"Question condensation failed, using original question: {e}")
            return question

    def get_answer(self, question: str, chat_history: Optional[List[Dict]] = None) -> Optional[Dict]:
        """Process question with layered relevance handling.

        chat_history, if given, is a list of prior {"role", "content"}
        turns -- used only to rewrite `question` into a standalone version
        via condense_question() before any of the real routing happens.
        Optional and defaults to None so existing single-turn callers (the
        eval harness, tests) see no change in behavior at all.
        """
        try:
            # Handle empty questions
            if not question.strip():
                return {
                    "result": "Please enter a question about F-1 visas, OPT, or CPT.",
                    "source_documents": [],
                }

            # Check for predefined help questions -- checked against the
            # raw question, before condensing, so a mid-conversation "help"
            # still matches its trigger phrase directly rather than risking
            # the condense step paraphrasing it into something that doesn't.
            clean_q = question.strip().lower()
            for key, entry in self.config.GENERIC_HELP_QUESTIONS.items():
                if any(trigger in clean_q for trigger in entry["triggers"]):
                    logging.info(f"Help question detected: {key}")
                    return {"result": entry["response"], "source_documents": []}

            if chat_history:
                question = self.condense_question(question, chat_history)

            # Deterministic rules (day-count/logic questions with exact
            # answers) -- bypasses RAG entirely for questions they match.
            # Returns None for anything that isn't a match, falling through
            # to the normal RAG path unchanged. Runs on the (possibly
            # condensed) question, so a clarifying question rules_engine
            # asked last turn -- e.g. "are you on initial OPT or STEM
            # extension?" -- can actually be answered in the next turn
            # instead of being a dead end.
            rule_answer = rules_engine.match_and_answer(question, self.llm)
            if rule_answer is not None:
                return rule_answer

            # LLM relevance analysis
            relevant, explanation = self._is_relevant_question(question)

            if not relevant:
                return {
                    "result": f"""
🚦 **Relevance Check**\n
{explanation}\n
💡 **Ask About**:
- OPT/CPT eligibility
- Form I-765 processing
- Maintaining F-1 status
- STEM OPT requirements
- Travel signatures""",
                    "source_documents": [],
                }

            # Process relevant questions
            answer = self.qa_chain.invoke(
                {"query": question, "return_only_outputs": True}
            )
            logging.info(f"Answer generated: {answer}")
            if answer and "source_documents" in answer:
                answer["source_documents"] = self._sort_by_source_priority(
                    answer["source_documents"]
                )
            return answer

        except RateLimitError as e:
            # OpenAI returns HTTP 429 for both true rate limiting and an
            # exhausted billing quota -- either way, this is genuinely
            # different from "something in this app broke" and deserves an
            # honest, distinct message rather than the generic catch-all
            # below, especially since the shared-key usage caps above are a
            # soft, best-effort limit (see their module docstring) and a
            # real quota exhaustion is exactly the failure mode they can't
            # fully prevent.
            logging.error(f"OpenAI rate limit / quota error: {str(e)}")
            return {
                "result": (
                    "⚠️ This app has hit its OpenAI usage limit right now. "
                    "Please try again later, or enter your own API key in "
                    "the sidebar to continue immediately."
                ),
                "source_documents": [],
            }

        except Exception as e:
            logging.error(f"Processing error: {str(e)}")
            return {
                "result": "Error processing request. Please try again.",
                "source_documents": [],
            }

    @staticmethod
    def _get_source_link(source: str, doc_type: str) -> str:
        """Generate appropriate hyperlink based on source type."""
        try:
            if doc_type == "web" or doc_type == "reddit":
                # Check if URL is valid
                parsed = urlparse(source)
                if parsed.scheme and parsed.netloc:
                    return f"<a href='{source}' target='_blank'>{source} 🔗</a>"
                else:
                    return "Invalid URL ❌"
            elif doc_type == "pdf":
                # Using st.markdown's native PDF handling
                filename = os.path.basename(source)
                full_path = os.path.abspath(os.path.join("docs", filename))
                if os.path.exists(full_path):
                    return (
                        f"<a href='data:application/pdf;base64,{F1rstAidApp._encode_pdf(full_path)}' "
                        f"download='{filename}'>Download {filename} 📄</a>"
                    )
            return "Source unavailable ❌"
        except Exception as e:
            logging.error(f"Error creating source link: {e}")
            return "Source link error ⚠️"

    @staticmethod
    def _encode_pdf(file_path: str) -> str:
        """Encode PDF file to base64 for browser download."""
        import base64

        try:
            with open(file_path, "rb") as file:
                return base64.b64encode(file.read()).decode()
        except Exception as e:
            logging.error(f"Error encoding PDF {file_path}: {e}")
            return ""

    @staticmethod
    def clean_markdown(text: str) -> str:
        """Remove common markdown link and emphasis syntax from text."""
        # Remove code blocks
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        text = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)

        # Remove inline code markers (`)
        text = re.sub(r"`([^`]+)`", r"\1", text)

        # Remove markdown links (updated regex to handle truncated URLs)
        text = re.sub(
            r"\[([^\]]+)\]\([^)]*\)?", r"\1", text
        )  # Added optional closing )

        # Remove emphasis markers: *, _, ~
        text = re.sub(r"[*_~]", "", text)

        # Remove any leading heading markers
        text = re.sub(r"^\s*#+\s*", "", text, flags=re.MULTILINE)

        return text.strip()

    @staticmethod
    def format_sources(docs: List[Document], start_index: int = 1) -> str:
        """Format source documents for display with enhanced metadata and
        links. Styling lives in the module-level APP_CSS, injected once per
        script run in main() -- this only builds the markup.

        start_index lets numbering stay continuous when official and
        community sources are rendered as two separate calls (see
        display_answer) -- otherwise both groups would restart at "Source 1".
        No id='source-N' anchors anymore -- this whole block now renders
        inside a collapsed st.expander, and an anchor link can't auto-open
        a collapsed expander, so a jump-to-source link would just be dead.
        """
        sources = []
        for offset, doc in enumerate(docs):
            i = start_index + offset
            source = doc.metadata.get("source", "Unknown")
            doc_type = doc.metadata.get("type", "unknown")
            block_class = "source-reddit" if doc_type == "reddit" else "source-official"

            # 1) Grab raw content snippet
            raw_preview = doc.page_content[:200].replace("\n", " ").strip()

            # 2) Clean markdown
            preview = F1rstAidApp.clean_markdown(raw_preview)
            preview = escape(preview)

            source_block = [
                f"<div class='source-block {block_class}'>",
                f"<div class='source-heading'>",
                f"<span class='source-index'>{i}</span>",
                f"<span class='source-type'>{doc_type.upper()}</span>",
                f"{F1rstAidApp._get_source_link(source, doc_type)}",
                "</div>",
                f"<p class='preview-text'>{preview}...</p>",
                "</div>",
            ]
            sources.append("\n".join(source_block))

        return "\n".join(sources)

    @staticmethod
    def _sort_by_source_priority(docs: List[Document]) -> List[Document]:
        """Re-sort retrieved docs so official sources appear before Reddit."""
        priority = {"pdf": 0, "web": 1, "reddit": 2}
        return sorted(docs, key=lambda d: priority.get(d.metadata.get("type", "unknown"), 3))

    def format_answer(self, result: str, sources: List[Document]) -> str:
        """Format answer with source context."""
        has_reddit_sources = any(
            doc.metadata.get("type") == "reddit" for doc in sources
        )

        formatted_answer = result
        if has_reddit_sources:
            formatted_answer = (
                "⚠️ Note: Some of this information comes from Reddit community experiences "
                "and should be verified with official sources.\n\n" + formatted_answer
            )

        return formatted_answer

    def display_answer(self, answer: Dict):
        """Display formatted answer and sources."""
        answer_type = classify_answer(answer)
        logging.info(f"Answer: {answer['result']} (classified as: {answer_type})")

        # The retriever always returns its top-k docs regardless of whether
        # the LLM found them sufficient -- an abstained RAG answer still has
        # non-empty source_documents. Gating here, once, so neither the
        # Reddit-sourcing note below nor the source cards further down can
        # imply that unused retrieved docs backed an answer that was never
        # actually given.
        displayable_sources = (
            answer.get("source_documents", []) if answer_type in ("rule", "rag") else []
        )

        st.markdown("### 📝 Answer")

        badge = _ANSWER_BADGES.get(answer_type)
        if badge:
            label, tooltip, css_class = badge
            title_attr = f" title='{escape(tooltip)}'" if tooltip else ""
            st.markdown(
                f"<span class='answer-badge {css_class}'{title_attr}>{label}</span>",
                unsafe_allow_html=True,
            )

        formatted_answer = self.format_answer(
            F1rstAidApp.clean_markdown(answer["result"]).strip(),
            displayable_sources,
        )
        st.markdown(formatted_answer)

        # A substantive answer (computed or AI-generated) is where a
        # not-legal-advice reminder actually matters -- skip it for
        # off-topic declines, help text, etc. where it'd just be noise.
        if answer_type in ("rule", "rag"):
            st.markdown(
                "<div class='disclaimer-box'>⚖️ This is not legal advice. "
                "Immigration rules and individual circumstances vary -- "
                "please confirm your specific situation with your DSO or "
                "an immigration attorney.</div>",
                unsafe_allow_html=True,
            )

        if displayable_sources:
            official_sources = []
            community_sources = []

            for doc in displayable_sources:
                if doc.metadata.get("type") == "reddit":
                    community_sources.append(doc)
                else:
                    official_sources.append(doc)

            # Collapsed by default -- reported live: with sources always
            # expanded, the source cards routinely took more vertical space
            # than the answer itself, which read as broken/overwhelming
            # rather than like supporting detail. An expander keeps the
            # count visible (so "sources exist, click to check them" is
            # still obvious) without forcing every answer to be that long.
            total = len(official_sources) + len(community_sources)
            label = f"📚 {total} source{'s' if total != 1 else ''}"
            with st.expander(label, expanded=False):
                if official_sources:
                    st.markdown("**Official Sources**")
                    st.markdown(
                        self.format_sources(official_sources, start_index=1),
                        unsafe_allow_html=True,
                    )

                if community_sources:
                    st.markdown("**Community Experiences (Reddit)**")
                    st.markdown(
                        self.format_sources(
                            community_sources, start_index=len(official_sources) + 1
                        ),
                        unsafe_allow_html=True,
                    )

                st.caption("PDF links open in your default PDF viewer.")


def get_api_key() -> Optional[str]:
    """Get API key from session state or environment."""
    if "OPENAI_API_KEY" in st.session_state:
        return st.session_state.OPENAI_API_KEY
    return os.getenv("OPENAI_API_KEY")

def set_api_key(api_key: str) -> None:
    """Set API key in session state and environment."""
    st.session_state.OPENAI_API_KEY = api_key
    os.environ["OPENAI_API_KEY"] = api_key


@st.cache_resource
def get_cached_app(api_key: str) -> Optional[F1rstAidApp]:
    """Initialize and cache the app — runs once per unique API key per session."""
    config = AppConfig()
    app = F1rstAidApp(config)
    if not app.initialize():
        return None
    return app


def render_feedback_widget(msg_index: int, answer: Dict):
    """Thumbs up/down under an assistant message. Thumbs-down reveals an
    optional comment box; submitting it calls create_feedback_issue().
    State is keyed by msg_index so re-rendering the same message on a
    later rerun (the whole history replays every time -- see main()) shows
    "already recorded" instead of the buttons again, and so widget keys
    stay unique across the whole conversation.
    """
    feedback_given = st.session_state.setdefault("feedback_given", {})

    if msg_index in feedback_given:
        if feedback_given[msg_index] == "up":
            st.caption("👍 Thanks for the feedback!")
        elif st.session_state.get(f"feedback_filed_{msg_index}"):
            st.caption("👎 Thanks -- this has been flagged for review.")
        else:
            st.caption("👎 Thanks for the feedback!")
        return

    col1, col2, _ = st.columns([1, 1, 10])
    with col1:
        if st.button("👍", key=f"thumbs_up_{msg_index}"):
            feedback_given[msg_index] = "up"
            st.rerun()
    with col2:
        if st.button("👎", key=f"thumbs_down_{msg_index}"):
            st.session_state[f"show_feedback_form_{msg_index}"] = True
            st.rerun()

    if st.session_state.get(f"show_feedback_form_{msg_index}"):
        comment = st.text_area(
            "What went wrong? (optional)", key=f"feedback_text_{msg_index}"
        )
        if st.button("Submit feedback", key=f"submit_feedback_{msg_index}"):
            sources = [
                d.metadata.get("source", "")
                for d in answer.get("source_documents", [])
            ]
            filed = create_feedback_issue(
                classify_answer(answer), answer.get("result", ""), sources, comment
            )
            feedback_given[msg_index] = "down"
            st.session_state[f"feedback_filed_{msg_index}"] = filed
            st.session_state.pop(f"show_feedback_form_{msg_index}", None)
            st.rerun()


def main():
    """Main application entry point."""
    try:
        # Setup Streamlit UI
        st.markdown(APP_CSS, unsafe_allow_html=True)
        st.title("🎓 F1rstAid: Your F-1 Visa Helper")
        st.caption(
            "An unofficial assistant for F-1 student visa questions -- "
            "not a substitute for advice from your DSO or an immigration attorney."
        )

        # API Key Input Section
        with st.sidebar:
            with st.expander("ℹ️ About F1rstAid", expanded=True):
                st.markdown(
                    """
**What this is:** A RAG-based assistant over official F-1 guidance
(USCIS/DHS/Study in the States) plus Reddit community experiences, with a
few exact, rule-based answers (like OPT unemployment-day limits and the
60-day grace period) computed directly instead of guessed by an LLM.

**What this isn't:** Legal advice. Immigration rules change and individual
cases vary -- always confirm anything important with your DSO or a
qualified immigration attorney before acting on it.

**Tips:**
- Mention specifics (dates, form numbers, your OPT phase) for a sharper answer.
- Official sources are weighted above Reddit posts, but Reddit answers are
  still shown separately when relevant -- read them as anecdotes, not policy.
                    """
                )

            st.markdown("### 🔑 API Access")
            st.caption(
                f"Try it free -- {FREE_TRIAL_QUERY_LIMIT} questions per "
                "session on the shared key, no setup needed. Have your own "
                "OpenAI key? Add it below for unlimited questions."
            )
            api_key = st.text_input(
                "Your own OpenAI API key (optional):",
                type="password",
                help=(
                    "Get one at https://platform.openai.com/api-keys -- only "
                    "needed if you want unlimited questions instead of the "
                    "free trial."
                ),
                key="api_key_input"
            )

            if api_key:
                set_api_key(api_key)
                st.session_state["using_own_key"] = True
                st.success("✅ Using your own key -- no question limit.")
            elif get_api_key():
                # A key is already available from a local .env or a prior
                # session-state set (e.g. loaded via load_dotenv() at import
                # time) -- the manual field being empty just means the user
                # hasn't (re-)typed one, not that no key exists. This is the
                # shared/owner key, so the free-trial limits apply -- see
                # shared_key_limit_reached().
                st.session_state["using_own_key"] = False
                remaining = max(
                    0,
                    FREE_TRIAL_QUERY_LIMIT - st.session_state.get("shared_key_query_count", 0),
                )
                st.info(
                    f"🎟️ Using the shared trial key -- {remaining} of "
                    f"{FREE_TRIAL_QUERY_LIMIT} free questions left this session."
                )
            else:
                # Only reachable when no shared key is configured for this
                # deployment at all (e.g. running locally with no .env) --
                # on the live deployment, get_api_key() above always finds
                # the shared key, so a visitor never actually hits this.
                st.warning(
                    "⚠️ No shared trial key is configured for this "
                    "deployment. Please enter your own OpenAI API key above "
                    "to continue."
                )
                return

            st.markdown("""
            ### ℹ️ About API Keys
            - **Shared trial key**: a few free questions per session, plus a
              daily cap shared across everyone, so it doesn't run dry for
              other visitors.
            - **Your own key**: unlimited questions, billed to your own
              OpenAI account. Stored only in this browser session -- never
              saved or logged.
            - Get a key at [OpenAI Platform](https://platform.openai.com/api-keys),
              check [pricing](https://openai.com/pricing).
            """)

            if st.session_state.get("messages"):
                st.markdown("---")
                if st.button("🔄 New conversation", use_container_width=True):
                    st.session_state.messages = []
                    st.rerun()

            # Footer lives in the sidebar, not the main pane: st.chat_input
            # is pinned to the bottom of the viewport regardless of where in
            # the script it's called, so a footer in normal document flow in
            # the main pane ends up floating in a large empty gap above it
            # on any page that doesn't have a full screen of conversation
            # yet -- which is most first-time visits. Plain st.caption
            # instead of the old custom-colored HTML card -- that card's
            # fixed light background/gradient was sized for a wide main-pane
            # banner and would look like a mismatched light box in the
            # narrower sidebar; a native caption is theme-aware for free.
            st.markdown("---")
            st.caption(
                "Built with ❤️ by "
                "[@haramrit09k](https://github.com/haramrit09k) · "
                "[LinkedIn](https://linkedin.com/in/haramrit09k)"
            )

        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Initialize app only if API key is present
        if get_api_key():
            app = get_cached_app(get_api_key())
            if app is None:
                st.error("Failed to initialize application. Please check your API key.")
                return

            last_updated = app.get_knowledge_base_last_updated()
            if last_updated:
                st.caption(
                    f"🗓️ Knowledge base last refreshed: {last_updated}. "
                    "Recent policy changes may not be reflected yet."
                )

            # Clickable example questions -- there's no other onboarding
            # here, so a first-time visitor otherwise faces a blank input.
            # Only shown before the conversation starts, matching how
            # suggested-prompt chips normally behave in a chat UI. Clicking
            # one sends it immediately (chat UI convention), rather than
            # pre-filling an input box -- st.chat_input doesn't support a
            # prefillable value the way the old st.text_input did.
            if not st.session_state.messages:
                st.write("Ask me anything about F-1 visas!")
                st.caption("Not sure where to start? Try one of these:")
                chip_cols = st.columns(len(EXAMPLE_QUESTIONS))
                for i, (chip_label, full_question) in enumerate(EXAMPLE_QUESTIONS):
                    with chip_cols[i]:
                        if st.button(chip_label, key=f"example_{i}", use_container_width=True):
                            st.session_state["_pending_prompt"] = full_question
                            st.rerun()

            # Replay the conversation so far. Each assistant turn stores its
            # full raw answer dict (not just the text), so display_answer()
            # can fully re-render the badge/citation/source-card presentation
            # identically to how it looked the first time, on every rerun.
            # A limit-reached notice (below) has no "answer" dict -- it's
            # not a real app answer, just a plain status message.
            for i, msg in enumerate(st.session_state.messages):
                with st.chat_message(msg["role"]):
                    if msg["role"] == "assistant" and "answer" in msg:
                        app.display_answer(msg["answer"])
                        render_feedback_widget(i, msg["answer"])
                    else:
                        st.markdown(msg["content"])

            prompt = st.chat_input("Ask your F-1 visa question...")
            prompt = prompt or st.session_state.pop("_pending_prompt", None)

            if prompt:
                using_own_key = st.session_state.get("using_own_key", False)
                limited, limit_message = (
                    (False, "") if using_own_key else shared_key_limit_reached()
                )

                # Prior turns only -- this new prompt isn't "history" yet,
                # it's the question being asked right now. Passed to
                # get_answer() so a follow-up like "what about STEM OPT?"
                # can be understood in context (see condense_question()).
                chat_history = list(st.session_state.messages)

                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                if limited:
                    # Blocked before spending a shared-key API call, not
                    # after -- this is the whole point of checking here.
                    with st.chat_message("assistant"):
                        st.warning(limit_message)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": limit_message}
                    )
                else:
                    if not using_own_key:
                        st.session_state["shared_key_query_count"] = (
                            st.session_state.get("shared_key_query_count", 0) + 1
                        )
                        _increment_daily_shared_key_usage()

                    with st.chat_message("assistant"):
                        with st.spinner("🔍 Researching your question..."):
                            answer = app.get_answer(prompt, chat_history=chat_history)
                        app.display_answer(answer)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer["result"], "answer": answer}
                    )

                # The sidebar (which shows the remaining free-question
                # count) renders earlier in this same script pass, before
                # this block -- so without forcing another rerun, it always
                # shows the count as of *before* the question just asked,
                # not reflecting it until some later, unrelated interaction
                # triggers the next rerun. Forcing one here makes the
                # sidebar correct immediately, every time.
                st.rerun()

        else:
            st.error("Please provide an OpenAI API key to use F1rstAid")
            return

    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        st.error("An unexpected error occurred. Please try again later.")


if __name__ == "__main__":
    main()
