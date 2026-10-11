"""Suggested next questions shown as tappable chips under the latest answer.

Typing on a phone is the slowest part of the app, so after an answer we offer
a few related questions. Every suggestion is a question taken verbatim from
eval/dataset.json -- i.e. one the eval harness already checks the pipeline
against -- and a test enforces that, so a chip never sends a question nobody
has verified. Topic detection is plain keyword matching on the question the
student just asked.
"""

from typing import Dict, Iterable, List, Tuple

# label shown on the chip -> exact question text sent when tapped
_Q = {
    "How long is OPT?": "How long is the standard post-completion OPT period for F-1 students?",
    "How long is the STEM extension?": "How long is the STEM OPT extension?",
    "What form for STEM OPT?": "What form must be completed as a formal training plan for a STEM OPT extension?",
    "When to file for STEM OPT?": "Roughly how many days before my current OPT authorization expires should I file for the STEM OPT extension?",
    "Changing employers on STEM OPT?": "If I change employers during my STEM OPT extension, what am I required to do regarding my DSO and Form I-983?",
    "Is my degree STEM-eligible?": "Can I apply for a STEM OPT extension if my degree is not on the DHS STEM Designated Degree Program list?",
    "Unemployment benefits on OPT?": "Can I collect unemployment insurance benefits as an F-1 student on OPT?",
    "Work remotely for a foreign employer?": "Can F-1 students on OPT work fully remotely for an employer based outside the United States?",
    "CPT before one academic year?": "Can I do full-time Curricular Practical Training (CPT) before completing one full academic year?",
    "Does CPT affect OPT?": "Does doing 12 or more months of full-time CPT affect my eligibility for OPT afterward?",
    "Grace period after my program?": "What is the grace period after completing a program of study for an F-1 student who is not continuing on OPT?",
    "Report a change of address?": "How many days do I have to report a change of address to my DSO under SEVIS requirements?",
    "Cap-gap with an H-1B offer?": "I'm on OPT and got an H-1B job offer, but haven't filed a petition yet -- is there a cap-gap for that?",
}

# (keywords that must appear in the student's last question, [chip labels])
# First matching topic wins, so list the specific topics before the broad ones.
_TOPICS: List[Tuple[Tuple[str, ...], List[str]]] = [
    (("unemploy",), ["How long is OPT?", "How long is the STEM extension?", "Unemployment benefits on OPT?"]),
    (("cap-gap", "cap gap", "h-1b", "h1b"), ["Cap-gap with an H-1B offer?", "When to file for STEM OPT?", "How long is OPT?"]),
    (("stem",), ["When to file for STEM OPT?", "What form for STEM OPT?", "Changing employers on STEM OPT?"]),
    (("cpt",), ["CPT before one academic year?", "Does CPT affect OPT?", "How long is OPT?"]),
    (("grace period",), ["Report a change of address?", "How long is OPT?", "Cap-gap with an H-1B offer?"]),
    (("address", "moved", "move "), ["Grace period after my program?", "How long is OPT?"]),
    (("opt",), ["How long is the STEM extension?", "Work remotely for a foreign employer?", "Is my degree STEM-eligible?"]),
]

MAX_SUGGESTIONS = 3


def suggest_follow_ups(last_question: str, already_asked: Iterable[str] = ()) -> List[Tuple[str, str]]:
    """Up to MAX_SUGGESTIONS (chip label, full question) pairs related to
    `last_question`, skipping anything the student already asked. Empty when
    the topic isn't recognised (better no chips than irrelevant ones)."""
    text = (last_question or "").lower()
    asked = {a.strip().lower() for a in already_asked}
    for keywords, labels in _TOPICS:
        if any(k in text for k in keywords):
            picks = [
                (label, _Q[label])
                for label in labels
                if _Q[label].strip().lower() not in asked
            ]
            return picks[:MAX_SUGGESTIONS]
    return []


def all_suggested_questions() -> Dict[str, str]:
    """Label -> question for every suggestion (used by the vetting test)."""
    return dict(_Q)
