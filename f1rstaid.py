from html import escape
import logging
import os
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse
import os.path

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
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

CONDENSE_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=(
        "Given the conversation so far and a follow-up question, rewrite "
        "the follow-up as a standalone question that includes every fact "
        "from the conversation needed to answer it on its own -- dates, "
        "day counts, which OPT phase, or anything else the student already "
        "stated. Do not answer the question yourself. If the follow-up is "
        "already standalone, return it unchanged.\n\n"
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
EXAMPLE_QUESTIONS = [
    "How many days of unemployment am I allowed on OPT?",
    "What is the 60-day grace period after I graduate?",
    "Does cap-gap extend my work authorization?",
    "What documents do I need for a STEM OPT extension?",
]


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

.source-block {
    /* Background here is a fixed light color regardless of Streamlit's
       theme -- so text color inside has to be fixed too, not inherited.
       Left as inherited before, it picked up Streamlit's dark-theme body
       color (near-white), which is invisible on this always-light card. */
    background-color: #ffffff;
    color: #24292f;
    border: 1px solid #e1e4e8;
    margin: 12px 0;
    padding: 16px 20px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}
.source-block h4 {
    color: #0366d6;
    margin: 0 0 12px 0;
    border-bottom: 2px solid #0366d6;
    padding-bottom: 5px;
}
.source-block strong {
    color: #24292f;
}
.source-content {
    margin-left: 10px;
}
.preview-box {
    background-color: #f6f8fa;
    color: #24292f;
    padding: 10px;
    border-radius: 5px;
    margin-top: 10px;
}
.preview-text {
    font-family: monospace;
    font-size: 0.9em;
    line-height: 1.4;
    white-space: pre-wrap;
    color: #24292f;
}
.source-block a {
    color: #0366d6;
    text-decoration: none;
    padding: 2px 4px;
    border-radius: 3px;
    background-color: #f1f8ff;
    word-break: break-all;
}
.source-block a:hover {
    text-decoration: underline;
    background-color: #e1e4e8;
}
.source-reddit { border-left: 3px solid #ff4500; }
.source-official { border-left: 3px solid #0366d6; }

.disclaimer-box {
    font-size: 0.85em;
    color: #6a737d;
    background-color: #f6f8fa;
    border-left: 3px solid #d0d7de;
    padding: 8px 12px;
    border-radius: 4px;
    margin: 8px 0 20px 0;
}

.citation-row {
    font-size: 0.85em;
    color: #57606a;
    margin: 4px 0 16px 0;
}
.citation-chip {
    display: inline-block;
    min-width: 1.4em;
    text-align: center;
    margin: 0 2px;
    padding: 1px 7px;
    border-radius: 10px;
    background-color: #f1f8ff;
    color: #0366d6;
    text-decoration: none;
    font-weight: 600;
}
.citation-chip:hover {
    background-color: #dbedff;
    text-decoration: none;
}

/* Keep long/unbroken source URLs and code from forcing horizontal scroll
   on narrow (mobile) viewports. */
.source-block, .preview-text {
    overflow-wrap: anywhere;
}

.site-footer {
    text-align: center;
    background: linear-gradient(to right, #f8f9fa, #ffffff, #f8f9fa);
    padding: 15px;
    border-top: 1px solid #eee;
    margin-top: 24px;
}
.site-footer span {
    font-size: 14px;
    color: #666;
}
.site-footer a {
    text-decoration: none;
    font-weight: 500;
}

@media (max-width: 640px) {
    .source-block { padding: 12px 14px; }
    .source-content { margin-left: 4px; }
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

    def _is_relevant_question(self, question: str) -> Tuple[bool, str]:
        """Check relevance using fast keyword matching (no API call)."""
        clean_q = question.strip().lower()

        # First check for predefined help questions
        for entry in self.config.GENERIC_HELP_QUESTIONS.values():
            if any(trigger in clean_q for trigger in entry["triggers"]):
                return True, "Help question detected"

        if any(kw in clean_q for kw in _F1_KEYWORDS):
            return True, "F-1 topic detected"

        return False, (
            "This doesn't appear to be an F-1 visa question. "
            "I can help with OPT/CPT eligibility, SEVIS, employment authorization, "
            "maintaining F-1 status, travel requirements, and related topics."
        )

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

        Falls back to the original question on any failure (a bad rewrite,
        or the LLM call itself failing) -- condensation is an enhancement,
        never something that should be the reason a question goes
        unanswered.
        """
        if not chat_history:
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
        display_answer) -- otherwise both groups would restart at "Source 1",
        which wouldn't match the citation numbers shown under the answer.
        Each block also gets an id='source-N' anchor so those citation
        numbers can link straight down to the matching card.
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
                f"<div class='source-block {block_class}' id='source-{i}'>",
                f"<h4>Source {i}</h4>",
                f"<div class='source-content'>",
                f"<p><strong>Type:</strong> {doc_type.upper()}</p>",
                f"<p><strong>Source:</strong> {F1rstAidApp._get_source_link(source, doc_type)}</p>",
                f"<div class='preview-box'>",
                f"<p><strong>Preview:</strong></p>",
                f"<p class='preview-text'>{preview}...</p>",
                "</div>",
                "</div>",
                "</div>",
            ]
            sources.append("\n".join(source_block))

        return "\n\n\n\n".join(sources)

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

            # Citation chips right under the answer, numbered to match the
            # source cards below -- so "which of these sources backed this"
            # is a click instead of a scroll-and-guess. The QA_PROMPT
            # already asks the model to name sources in prose (e.g. "Per
            # USCIS..."); this doesn't change that, it just gives the
            # existing source list a jump-to-anchor from the top.
            total = len(official_sources) + len(community_sources)
            chip_links = " ".join(
                f"<a class='citation-chip' href='#source-{i}'>{i}</a>"
                for i in range(1, total + 1)
            )
            st.markdown(
                f"<div class='citation-row'>📎 Cited sources: {chip_links}</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                "### 📚 Source Documents",
                help="ℹ️ PDF links will open in default PDF viewer",
            )

            if official_sources:
                st.markdown("#### Official Sources")
                st.markdown(
                    self.format_sources(official_sources, start_index=1),
                    unsafe_allow_html=True,
                )

            if community_sources:
                st.markdown("#### Community Experiences (Reddit)")
                st.markdown(
                    self.format_sources(
                        community_sources, start_index=len(official_sources) + 1
                    ),
                    unsafe_allow_html=True,
                )


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
            with st.expander("ℹ️ About F1rstAid", expanded=False):
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

            st.markdown("### 🔑 OpenAI API Key")
            api_key = st.text_input(
                "Enter your OpenAI API key:",
                type="password",
                help="Get your API key from https://platform.openai.com/api-keys",
                key="api_key_input"
            )
            
            if api_key:
                set_api_key(api_key)
                st.success("✅ API key set successfully!")
            elif get_api_key():
                # A key is already available from a local .env or a prior
                # session-state set (e.g. loaded via load_dotenv() at import
                # time) -- the manual field being empty just means the user
                # hasn't (re-)typed one, not that no key exists.
                st.info("✅ Using API key from environment.")
            else:
                st.warning("⚠️ Please enter your OpenAI API key to continue")
                return
            
            st.markdown("""
            ### ℹ️ About API Keys
            1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
            2. Your key is stored securely in session state
            3. Key is never saved or logged
            4. Session expires when you close the browser

            ### 💰 Usage
            - OpenAI charges per API call
            - Check [pricing](https://openai.com/pricing)
            - Monitor usage in your OpenAI account
            """)

            if st.session_state.get("messages"):
                st.markdown("---")
                if st.button("🔄 New conversation", use_container_width=True):
                    st.session_state.messages = []
                    st.rerun()

        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Initialize app only if API key is present
        if get_api_key():
            app = get_cached_app(get_api_key())
            if app is None:
                st.error("Failed to initialize application. Please check your API key.")
                return

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
                for i, example in enumerate(EXAMPLE_QUESTIONS):
                    with chip_cols[i]:
                        if st.button(example, key=f"example_{i}", use_container_width=True):
                            st.session_state["_pending_prompt"] = example
                            st.rerun()

            # Replay the conversation so far. Each assistant turn stores its
            # full raw answer dict (not just the text), so display_answer()
            # can fully re-render the badge/citation/source-card presentation
            # identically to how it looked the first time, on every rerun.
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    if msg["role"] == "assistant":
                        app.display_answer(msg["answer"])
                    else:
                        st.markdown(msg["content"])

            prompt = st.chat_input("Ask your F-1 visa question...")
            prompt = prompt or st.session_state.pop("_pending_prompt", None)

            if prompt:
                # Prior turns only -- this new prompt isn't "history" yet,
                # it's the question being asked right now. Passed to
                # get_answer() so a follow-up like "what about STEM OPT?"
                # can be understood in context (see condense_question()).
                chat_history = list(st.session_state.messages)

                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("🔍 Researching your question..."):
                        answer = app.get_answer(prompt, chat_history=chat_history)
                    app.display_answer(answer)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer["result"], "answer": answer}
                )

        else:
            st.error("Please provide an OpenAI API key to use F1rstAid")
            return

        # Styled footer. Deliberately laid out in normal document flow
        # (not position: fixed) -- a fixed footer on a narrow/mobile
        # viewport, or once the page has more content than fits one
        # screen, ends up overlapping the last bit of real content
        # instead of sitting below it.
        st.markdown("---")
        st.markdown(
            """
            <div class='site-footer'>
                <span>
                    Built with ❤️ by
                    <a href='https://github.com/haramrit09k' target='_blank'
                        style='color: #0366d6;'>
                        @haramrit09k
                    </a>
                    <span style='margin: 0 8px;'>|</span>
                    <a href='https://linkedin.com/in/haramrit09k' target='_blank'
                        style='color: #0077b5;'>
                        LinkedIn
                    </a>
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        st.error("An unexpected error occurred. Please try again later.")


if __name__ == "__main__":
    main()
