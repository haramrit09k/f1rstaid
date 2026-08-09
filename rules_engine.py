"""Deterministic rules for questions that are pure arithmetic/logic over
stated facts, not open-ended synthesis -- routed here instead of the RAG/LLM
path so the answer is exact rather than probabilistic.

Four rule families, each following the same trigger -> extract -> compute
shape:
  - the OPT/STEM-OPT unemployment day cap (90 / 150 aggregate days)
  - the 60-day post-completion/post-OPT grace period (pure date arithmetic)
  - the H-1B cap-gap work-authorization extension (conditional lookup, not
    arithmetic, but still exact rather than something to phrase-match for)
  - the STEM Designated Degree Program List eligibility check (also a fixed
    lookup, added after root-causing a real accuracy gap: retrieval surfaces
    the exact governing sentence -- "the qualifying degree needs to be on
    DHS's STEM Designated Degree Program List" -- but gpt-3.5-turbo still
    abstained instead of applying it, an instruction-following limit the
    QA_PROMPT's "apply a stated rule's direct consequence" carve-out was
    meant to cover but didn't reliably trigger on this phrasing)

match_and_answer() is the single entry point and tries each family in turn,
returning the first non-None match. See the project plan for why this rule
was chosen first and what's deferred to follow-up rules.
"""

import calendar
import logging
from datetime import date, timedelta
from typing import Dict, List, Optional

from langchain_core.documents import Document
from pydantic import BaseModel, Field


def _citation(source: str, snippet: str) -> Document:
    """A fixed, verified source backing a rule -- attached to that rule's
    answers so the UI can show a citation exactly like a RAG answer's, even
    though no retrieval happened. Each of these was checked against a real,
    already-ingested document (not invented from memory -- a bad citation is
    worse than none for a legal-adjacent tool) via a direct FAISS query, and
    the URL and regulation citation confirmed present in that document's
    actual text before being hardcoded here. `rule_based=True` is what
    f1rstaid.py's classify_answer() checks to tell these apart from a real
    RAG answer's retrieved sources -- everything else about the metadata
    (type="web") deliberately matches a normal source so the existing
    source-card rendering needs no special-casing.
    """
    return Document(
        page_content=snippet,
        metadata={"source": source, "type": "web", "rule_based": True},
    )


# Verified via direct FAISS queries against the ingested index (see the
# rules_engine backend-fixes session): each URL is a real, already-scraped
# official source, and each regulation citation is quoted verbatim from that
# source's actual ingested text, not recalled from memory.
_UNEMPLOYMENT_CITATION = _citation(
    "https://studyinthestates.dhs.gov/sevis-help-hub/student-records/fm-student-employment/f-1-optional-practical-training-opt",
    "DHS: governing regulation is 8 CFR 214.2(f)(10) through (13). SEVIS "
    "auto-terminates a student's record after 90 consecutive days of "
    "unemployment during initial OPT (150 days aggregate once on the STEM "
    "OPT extension).",
)
_GRACE_PERIOD_CITATION = _citation(
    "https://travel.state.gov/content/travel/en/us-visas/study/student-visa.html",
    "U.S. Department of State: \"Foreign students in the United States "
    "with F-1 visas must depart the United States within 60 days after "
    "the program end date listed on Form I-20, including any authorized "
    "practical training.\"",
)
_CAP_GAP_CITATION = _citation(
    "https://studyinthestates.dhs.gov/sevis-help-hub/student-records/fm-status/f-1-cap-gap-extension",
    "DHS: governing regulation is 8 CFR 214.2(f)(5)(vi). Students whose "
    "F-1 status ends between April 1 and September 30 with a timely-filed "
    "change-of-status H-1B petition may qualify for the cap-gap extension.",
)
_DEGREE_LIST_CITATION = _citation(
    "https://studyinthestates.dhs.gov/stem-opt-hub/for-students/students-determining-stem-opt-extension-eligibility",
    "DHS: \"The qualifying STEM degree needs to be on DHS's STEM "
    "Designated Degree Program List at the time the student submits their "
    "application for the STEM OPT extension.\"",
)


# --- Rule family 1: OPT/STEM-OPT unemployment day cap ---------------------

# Cheap, conservative pre-check so an LLM extraction call only happens for
# questions plausibly about unemployment-day counting -- mirrors the
# trigger-list pattern already used for _F1_KEYWORDS/GENERIC_HELP_QUESTIONS
# in f1rstaid.py. Kept narrow deliberately: the help-002 bug earlier in this
# project was exactly a trigger list that was too narrow OR too broad in the
# wrong direction, and both failure modes are checked in eval.
_UNEMPLOYMENT_TRIGGERS = [
    "unemployed",
    "unemployment",
    "out of work",
    "no job",
    "haven't found a job",
    "without a job",
    "jobless",
]

UNEMPLOYMENT_CAP_DAYS = {
    "initial_opt": 90,
    "stem_opt_extension": 150,
}


class UnemploymentFields(BaseModel):
    """Facts required to answer an unemployment-day-cap question. The two
    day-count fields are Optional -- the extraction prompt explicitly
    instructs leaving a field None rather than guessing a value the user
    didn't state, since inventing a day count or phase here would be a
    hallucination with a hard-coded veneer of certainty."""

    opt_phase: Optional[str] = Field(
        default=None,
        description=(
            "Which unemployment cap applies to answering this question -- "
            "'initial_opt' (90-day cap) or 'stem_opt_extension' (150-day cap, "
            "which is an AGGREGATE across the initial OPT period and the STEM "
            "extension combined, not an additional 150 days on top of the 90). "
            "IMPORTANT: if the question mentions the STEM OPT extension at "
            "all -- including asking about days remaining once STEM OPT "
            "starts, even if the student hasn't started it yet -- use "
            "'stem_opt_extension', since that 150-day aggregate cap is what "
            "actually governs the total. Only use 'initial_opt' if the STEM "
            "extension isn't mentioned or relevant at all. Leave null if "
            "genuinely not stated or ambiguous."
        ),
    )
    unemployment_days_used: Optional[int] = Field(
        default=None,
        description=(
            "Number of unemployment days the student has already accumulated, "
            "as stated in the question. Leave null if not stated -- never guess."
        ),
    )
    # Placed last deliberately: with a weaker model (gpt-3.5-turbo),
    # verified live that putting this coarser topic-classification field
    # first measurably degraded accuracy on the harder opt_phase reasoning
    # above (a "STEM extension" transition-phrasing question that opt_phase
    # handles correctly on its own started getting misclassified once
    # `applies` came first in the schema). Field order affects a weaker
    # model's effective reasoning budget per field -- let it work through
    # the specific day-count logic before asking the broader applicability
    # question.
    applies: bool = Field(
        description=(
            "True only if this question is actually about the OPT/STEM-OPT "
            "unemployment-day status-maintenance rule (the 90-day cap during "
            "initial post-completion OPT, or the 150-day aggregate cap once "
            "on the STEM OPT extension). The word 'unemployment' or "
            "'unemployed' alone is NOT enough -- set this False for anything "
            "else that merely mentions similar words, such as: state "
            "unemployment INSURANCE/benefits eligibility (a completely "
            "different topic from the status-maintenance day count), "
            "questions about CPT (which has no equivalent unemployment-day "
            "clock -- CPT requires employer authorization before it starts), "
            "or any other tangential mention. When in doubt, set this False "
            "so the question is handled by general knowledge-base search "
            "instead of this narrow rule."
        ),
    )


def _is_trigger_match(question: str) -> bool:
    clean_q = question.strip().lower()
    return any(t in clean_q for t in _UNEMPLOYMENT_TRIGGERS)


def extract_fields(question: str, llm) -> UnemploymentFields:
    """One structured-output call against the fields model."""
    extractor = llm.with_structured_output(UnemploymentFields)
    return extractor.invoke(
        "Extract the OPT phase and unemployment days used from this student's "
        "question, if stated. Never guess a value that isn't clearly present "
        f"in the question -- leave fields null instead.\n\nQuestion: {question}"
    )


def compute_answer(fields: UnemploymentFields) -> str:
    """Pure Python, zero LLM involvement, deterministic."""
    cap = UNEMPLOYMENT_CAP_DAYS[fields.opt_phase]
    used = fields.unemployment_days_used
    phase_label = (
        "the initial 12-month post-completion OPT"
        if fields.opt_phase == "initial_opt"
        else "the STEM OPT extension period (aggregate across initial OPT and the extension)"
    )

    if used > cap:
        return (
            f"Based on what you've described, you have used {used} days of "
            f"unemployment, which exceeds the {cap}-day limit for {phase_label}. "
            "This puts you at risk of falling out of status. Please contact your "
            "DSO immediately to discuss your options."
        )
    remaining = cap - used
    return (
        f"Based on what you've described, you have used {used} of your "
        f"{cap} allowed unemployment days for {phase_label}, leaving {remaining} "
        f"days remaining. This {cap}-day limit is a hard cap -- please consult "
        "your DSO if you're approaching it."
    )


def _general_rule_answer(opt_phase: Optional[str]) -> str:
    """A pure rule-lookup answer (no personal day count needed or invented)."""
    if opt_phase == "initial_opt":
        return (
            "During the initial 12-month post-completion OPT period, you're "
            "allowed a maximum of 90 days of unemployment before you're at risk "
            "of falling out of status."
        )
    if opt_phase == "stem_opt_extension":
        return (
            "Once on the STEM OPT extension, the unemployment cap is 150 days "
            "total, aggregated across your initial OPT and the extension period "
            "combined -- not 150 additional days on top of the initial 90."
        )
    return (
        "F-1 students are allowed a maximum of 90 days of unemployment during "
        "the initial 12-month post-completion OPT period. Once on a STEM OPT "
        "extension, that cap becomes 150 days total, aggregated across both "
        "periods combined -- not 150 additional days on top of the initial 90."
    )


def _match_unemployment(question: str, llm) -> Optional[Dict]:
    """Returns None if this rule doesn't apply (falls through to the
    existing RAG path), a general-rule answer if no personal day count was
    given (a pure "what's the rule" question, always safely answerable
    without guessing), a clarifying-question response if a personal day
    count was given but the OPT phase wasn't (needed to know which cap
    applies), or a full computed answer if both are present.
    """
    if not _is_trigger_match(question):
        return None

    try:
        fields = extract_fields(question, llm)
    except Exception as e:
        logging.error(f"rules_engine unemployment extraction failed: {e}")
        return None

    # Keyword trigger is a cheap pre-filter, not a relevance judgment -- it
    # can't distinguish "unemployment day-count status question" from
    # "unemployment insurance" or "CPT" (different topics that happen to
    # share trigger words). Only the extraction step has enough context to
    # make that call correctly.
    if not fields.applies:
        return None

    # No personal day count stated -- this is a "what's the rule" question,
    # not a request to evaluate a specific situation. Always answerable
    # without asking for anything or inventing a number.
    if fields.unemployment_days_used is None:
        return {
            "result": _general_rule_answer(fields.opt_phase),
            "source_documents": [_UNEMPLOYMENT_CITATION],
        }

    # A personal day count *was* given, so the user wants their specific
    # situation evaluated -- now the OPT phase actually matters, since the
    # cap differs (90 vs 150), and guessing it would risk a wrong answer.
    if fields.opt_phase not in UNEMPLOYMENT_CAP_DAYS:
        return {
            "result": (
                "I can calculate this for you, but I need to know one more "
                "thing: are you on your initial 12-month post-completion OPT, "
                "or on the STEM OPT extension? The unemployment cap differs "
                "(90 days vs. 150 days aggregate)."
            ),
            "source_documents": [_UNEMPLOYMENT_CITATION],
        }

    return {
        "result": compute_answer(fields),
        "source_documents": [_UNEMPLOYMENT_CITATION],
    }


# --- Rule family 2: 60-day grace period ------------------------------------
#
# Pure date arithmetic: 8 CFR 214.2(f)(5)(iv) gives F-1 students 60 days
# after their program of study or authorized OPT/STEM-OPT ends to depart the
# US, transfer schools, or change status -- during which they are NOT
# authorized to work. This is a fixed +60-days calculation, not something
# that should ever vary by phrasing.

_GRACE_PERIOD_TRIGGERS = [
    "grace period",
    "leave the u.s.",
    "leave the us",
    "leave the country",
    "depart the u.s.",
    "depart the us",
    "after i graduate",
    "after graduation",
    "when do i have to leave",
    "how long can i stay after",
]


class GracePeriodFields(BaseModel):
    """The date the grace-period clock starts. Split into month/day/year
    (year Optional) rather than a single date string, since students often
    give a date without a year ("my program ends May 15") -- forcing a full
    ISO date would either force the model to invent a year or fail to
    extract at all."""

    end_month: Optional[int] = Field(
        default=None,
        description=(
            "Month (1-12) that the student's program of study or authorized "
            "OPT/STEM-OPT period ends -- whichever one starts their grace "
            "period. Leave null if not stated."
        ),
    )
    end_day: Optional[int] = Field(
        default=None,
        description="Day of month (1-31) matching end_month. Leave null if not stated.",
    )
    end_year: Optional[int] = Field(
        default=None,
        description=(
            "Year the end date falls in, ONLY if the student explicitly "
            "stated a year. Leave null otherwise -- never guess a year."
        ),
    )
    # Placed last: see the comment on UnemploymentFields.applies -- a weaker
    # model's field-by-field reasoning measurably degrades when a coarse
    # relevance judgment is asked before the specific extraction fields.
    applies: bool = Field(
        description=(
            "True only if this question is about the F-1 60-day grace "
            "period after completing a program of study or after OPT/STEM-"
            "OPT ends. Set False for anything else that merely uses the "
            "phrase 'grace period' in an unrelated sense (e.g. a billing or "
            "payment grace period). When in doubt, set this False so the "
            "question is handled by general knowledge-base search instead."
        ),
    )


def _is_grace_period_trigger_match(question: str) -> bool:
    clean_q = question.strip().lower()
    return any(t in clean_q for t in _GRACE_PERIOD_TRIGGERS)


def extract_grace_period_fields(question: str, llm) -> GracePeriodFields:
    extractor = llm.with_structured_output(GracePeriodFields)
    return extractor.invoke(
        "Extract the program/OPT end date that starts this student's grace "
        "period clock, if stated. Never guess a value that isn't clearly "
        f"present in the question -- leave fields null instead.\n\nQuestion: {question}"
    )


def compute_grace_period_answer(fields: GracePeriodFields) -> str:
    """Pure Python date arithmetic, zero LLM involvement, deterministic."""
    # A non-leap placeholder year is used purely to do the +60-day
    # arithmetic when the student didn't state a year -- Feb 29 never
    # appears in that 60-day window for any date that matters here (the
    # latest a grace period can start and still avoid touching Feb 29 in a
    # non-leap placeholder is irrelevant; the +60 span from any date within
    # a school year lands within +/- a few months, never spanning a leap
    # day introduced only by the placeholder choice).
    placeholder_year = fields.end_year or 2001
    start = date(placeholder_year, fields.end_month, fields.end_day)
    end = start + timedelta(days=60)

    start_str = f"{calendar.month_name[start.month]} {start.day}"
    end_str = f"{calendar.month_name[end.month]} {end.day}"
    if fields.end_year:
        start_str += f", {fields.end_year}"
        # Only append a year to the end date if it's still calculable --
        # i.e. it didn't wrap into the following year via the placeholder.
        end_str += f", {fields.end_year if end.year == placeholder_year else fields.end_year + 1}"

    return (
        f"Based on an end date of {start_str}, your 60-day grace period ends "
        f"on {end_str}. During this period you may prepare to depart the "
        "U.S., transfer to a new school, or apply to change your status, "
        "but you are NOT authorized to work. Confirm the exact date with "
        "your DSO."
    )


def _general_grace_period_answer() -> str:
    return (
        "F-1 students get a 60-day grace period after their program of "
        "study ends (or after their authorized OPT/STEM-OPT period ends, "
        "if they did OPT) to prepare to depart the U.S., transfer to a new "
        "school, or apply to change status. You are not authorized to work "
        "during this grace period."
    )


def _match_grace_period(question: str, llm) -> Optional[Dict]:
    if not _is_grace_period_trigger_match(question):
        return None

    try:
        fields = extract_grace_period_fields(question, llm)
    except Exception as e:
        logging.error(f"rules_engine grace-period extraction failed: {e}")
        return None

    if not fields.applies:
        return None

    if fields.end_month is None or fields.end_day is None:
        return {
            "result": _general_grace_period_answer(),
            "source_documents": [_GRACE_PERIOD_CITATION],
        }

    return {
        "result": compute_grace_period_answer(fields),
        "source_documents": [_GRACE_PERIOD_CITATION],
    }


# --- Rule family 3: H-1B cap-gap extension ---------------------------------
#
# Not arithmetic, but still a fixed, exact lookup rather than something to
# generate: a timely-filed, pending-or-approved change-of-status H-1B
# petition automatically extends F-1 status and any current EAD through
# September 30, regardless of the specific dates involved -- the extension
# ends immediately if/when the petition is denied.

_CAP_GAP_TRIGGERS = [
    "cap-gap",
    "cap gap",
]


class CapGapFields(BaseModel):
    filed_on_time: Optional[bool] = Field(
        default=None,
        description=(
            "True if the student's H-1B petition (requesting a change of "
            "status, NOT consular processing) was filed before their OPT/"
            "EAD expired, while they were still in valid F-1 status. Leave "
            "null if not stated."
        ),
    )
    petition_denied: Optional[bool] = Field(
        default=None,
        description=(
            "True only if the student states their H-1B petition has "
            "already been denied. Leave null/False if pending or approved "
            "or not mentioned -- never assume a denial that wasn't stated."
        ),
    )
    applies: bool = Field(
        description=(
            "True only if this question is about the F-1 cap-gap rule -- "
            "the automatic extension of F-1 status/work authorization while "
            "a timely-filed H-1B change-of-status petition is pending. Set "
            "False for anything else that merely mentions H-1B or a filing "
            "deadline unrelated to this extension. When in doubt, set this "
            "False so the question is handled by general knowledge-base "
            "search instead."
        ),
    )


def _is_cap_gap_trigger_match(question: str) -> bool:
    clean_q = question.strip().lower()
    return any(t in clean_q for t in _CAP_GAP_TRIGGERS)


def extract_cap_gap_fields(question: str, llm) -> CapGapFields:
    extractor = llm.with_structured_output(CapGapFields)
    return extractor.invoke(
        "Extract whether this student's H-1B petition was filed on time "
        "(before OPT/EAD expiration, while in status, requesting change of "
        "status) and whether it's been denied, if stated. Never guess a "
        f"value that isn't clearly present in the question.\n\nQuestion: {question}"
    )


def compute_cap_gap_answer(fields: CapGapFields) -> str:
    if fields.petition_denied:
        return (
            "Your cap-gap extension ended on the date USCIS denied your "
            "H-1B petition. You should already be in your 60-day post-OPT "
            "grace period counted from that denial date -- contact your "
            "DSO immediately to confirm your status and options."
        )

    if fields.filed_on_time is False:
        return (
            "Cap-gap only extends your status and work authorization if "
            "your H-1B petition (requesting a change of status, not "
            "consular processing) was filed before your OPT/EAD expired, "
            "while you were still in valid F-1 status. Based on what "
            "you've described, that doesn't sound like it applies here -- "
            "please confirm your options with your DSO right away."
        )

    return _general_cap_gap_answer()


def _general_cap_gap_answer() -> str:
    return (
        "If your H-1B petition was properly filed while you were in valid "
        "F-1 status and before your OPT (or EAD) expired -- requesting a "
        "change of status effective October 1 -- your F-1 status and "
        "current work authorization are automatically extended under "
        "'cap-gap' through September 30, as long as the petition remains "
        "pending or is approved. If it's denied, the extension ends "
        "immediately on the denial date. Confirm your specific situation "
        "with your DSO."
    )


def _match_cap_gap(question: str, llm) -> Optional[Dict]:
    if not _is_cap_gap_trigger_match(question):
        return None

    try:
        fields = extract_cap_gap_fields(question, llm)
    except Exception as e:
        logging.error(f"rules_engine cap-gap extraction failed: {e}")
        return None

    if not fields.applies:
        return None

    # No personal facts stated at all -- pure "what's the rule" question.
    if fields.filed_on_time is None and fields.petition_denied is None:
        return {
            "result": _general_cap_gap_answer(),
            "source_documents": [_CAP_GAP_CITATION],
        }

    return {
        "result": compute_cap_gap_answer(fields),
        "source_documents": [_CAP_GAP_CITATION],
    }


# --- Rule family 4: STEM Designated Degree Program List eligibility -------
#
# Not arithmetic and not really a "conditional lookup" either -- it's a
# one-step logical consequence of a rule that's already stated verbatim in
# the knowledge base ("the qualifying degree needs to be on DHS's STEM
# Designated Degree Program List"). Routed here anyway because that's
# exactly the case gpt-3.5-turbo was observed abstaining on despite the
# governing sentence being in its top-3 retrieved context -- an
# instruction-following gap, not a missing-information one. See the
# rules_engine module docstring for how this was root-caused.

_DEGREE_LIST_TRIGGERS = [
    "designated degree program",
    "degree program list",
    "stem designated degree",
    "stem designated program",
    "cip code",
]


class DegreeListFields(BaseModel):
    degree_on_list: Optional[bool] = Field(
        default=None,
        description=(
            "True if the student states their degree IS on DHS's STEM "
            "Designated Degree Program List (or has a qualifying/STEM CIP "
            "code). False if they state it is NOT on the list, is not a "
            "STEM field, or they were told it doesn't qualify. Leave null "
            "if the student is just asking how the rule works, without "
            "stating which case they're in."
        ),
    )
    applies: bool = Field(
        description=(
            "True only if this question is about whether a specific degree "
            "qualifies for the STEM OPT extension via DHS's STEM Designated "
            "Degree Program List / CIP code requirement. Set False for "
            "anything else that merely mentions a similar phrase in an "
            "unrelated sense. When in doubt, set this False so the question "
            "is handled by general knowledge-base search instead."
        ),
    )


def _is_degree_list_trigger_match(question: str) -> bool:
    clean_q = question.strip().lower()
    return any(t in clean_q for t in _DEGREE_LIST_TRIGGERS)


def extract_degree_list_fields(question: str, llm) -> DegreeListFields:
    extractor = llm.with_structured_output(DegreeListFields)
    return extractor.invoke(
        "Extract whether the student's degree is on DHS's STEM Designated "
        "Degree Program List, if stated. Never guess a value that isn't "
        f"clearly present in the question.\n\nQuestion: {question}"
    )


def compute_degree_list_answer(fields: DegreeListFields) -> str:
    if fields.degree_on_list is False:
        return (
            "No -- only a degree on DHS's STEM Designated Degree Program "
            "List (identified by its CIP code) qualifies for the STEM OPT "
            "extension. Since your degree isn't on that list, it wouldn't "
            "qualify you on its own. However, if you have an earlier "
            "qualifying STEM degree (bachelor's or higher) from within the "
            "last 10 years, that could still make you eligible. Confirm "
            "your program's CIP code with your DSO, or check DHS's official "
            "list directly."
        )

    if fields.degree_on_list is True:
        return (
            "Since your degree is on DHS's STEM Designated Degree Program "
            "List, you meet the core degree requirement for the STEM OPT "
            "extension. You'll also need a prospective employer enrolled in "
            "E-Verify and a completed Form I-983 -- confirm the rest of "
            "your eligibility with your DSO."
        )

    return _general_degree_list_answer()


def _general_degree_list_answer() -> str:
    return (
        "To qualify for the STEM OPT extension, your degree must be on "
        "DHS's STEM Designated Degree Program List (identified by its CIP "
        "code) at the time you apply -- either your most recent degree, or "
        "a prior qualifying STEM degree (bachelor's or higher) earned "
        "within the last 10 years. If your degree isn't on that list, you "
        "would not be eligible for the STEM extension based on that degree "
        "alone. Check your program's CIP code with your DSO, or see DHS's "
        "official STEM Designated Degree Program List to confirm."
    )


def _match_degree_list(question: str, llm) -> Optional[Dict]:
    if not _is_degree_list_trigger_match(question):
        return None

    try:
        fields = extract_degree_list_fields(question, llm)
    except Exception as e:
        logging.error(f"rules_engine degree-list extraction failed: {e}")
        return None

    if not fields.applies:
        return None

    return {
        "result": compute_degree_list_answer(fields),
        "source_documents": [_DEGREE_LIST_CITATION],
    }


# --- Single entry point -----------------------------------------------------

def match_and_answer(question: str, llm) -> Optional[Dict]:
    """Tries each rule family in turn, returns the first non-None match.
    Returns None if no rule applies, so the caller falls through to the
    existing RAG path unchanged."""
    for matcher in (
        _match_unemployment,
        _match_grace_period,
        _match_cap_gap,
        _match_degree_list,
    ):
        result = matcher(question, llm)
        if result is not None:
            return result
    return None
