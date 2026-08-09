"""Deterministic rules for questions that are pure arithmetic/logic over
stated facts, not open-ended synthesis -- routed here instead of the RAG/LLM
path so the answer is exact rather than probabilistic.

Currently implements one rule family: the OPT/STEM-OPT unemployment day cap
(90 days during initial 12-month post-completion OPT, 150 days aggregate
once on the STEM OPT extension). See the project plan for why this rule was
chosen first and what's deferred to follow-up rules.
"""

import logging
from typing import Dict, Optional

from pydantic import BaseModel, Field

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


def match_and_answer(question: str, llm) -> Optional[Dict]:
    """Single entry point. Returns None if this rule doesn't apply (falls
    through to the existing RAG path), a general-rule answer if no personal
    day count was given (a pure "what's the rule" question, always safely
    answerable without guessing), a clarifying-question response if a
    personal day count was given but the OPT phase wasn't (needed to know
    which cap applies), or a full computed answer if both are present.
    """
    if not _is_trigger_match(question):
        return None

    try:
        fields = extract_fields(question, llm)
    except Exception as e:
        logging.error(f"rules_engine extraction failed: {e}")
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
        return {"result": _general_rule_answer(fields.opt_phase), "source_documents": []}

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
            "source_documents": [],
        }

    return {"result": compute_answer(fields), "source_documents": []}
