"""Cheap, deterministic prompt-injection screening.

Used in three places, all for the same reason -- text from outside the app
(a visitor's question, or a scraped Reddit/web page that ends up in the
knowledge base) must never be able to talk the model out of its role:

  - f1rstaid.py: declines a visitor question that tries to rewrite the
    assistant's instructions, before any LLM call is spent on it.
  - f1rstaid.py: drops retrieved chunks that contain such text before they
    are put in the answer prompt (protects an already-built index).
  - ingest.py: refuses to index such chunks in the first place.

This is a blocklist, so it is one layer, not a guarantee -- a determined
attacker can reword. It exists to remove the cheap, obvious attacks
deterministically; the model-side defenses (untrusted-input framing and the
grounding check in f1rstaid.py) cover what it misses.
"""

import re

_INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}\b(previous|prior|above|earlier|preceding|all|any|your|the|these|those)\b[^.\n]{0,40}\b(instruction|instructions|prompt|prompts|rules|guidelines|context|constraints|restrictions)\b",
    r"\b(ignore|disregard|forget)\b[^.\n]{0,20}\b(the|your|all)\s+(context|documents|sources)\b",
    r"\banswer\b[^.\n]{0,30}\b(from|using|with)\b[^.\n]{0,20}\b(your|own|general|outside)\b[^.\n]{0,20}\b(knowledge|memory|training)\b",
    r"\bso\s+answer\s+(true|yes)\b",
    r"\bset\s+(applies|relevant)\s*(=|to)\s*(true|yes)\b",
    r"\b(you\s+are|you're)\s+now\s+(a|an|in|my|the)\b",
    r"\b(new|updated|revised)\s+(system\s+)?instructions?\s*:",
    r"\b(reveal|print|show|repeat|output|leak)\b[^.\n]{0,30}\b(system\s+prompt|your\s+(prompt|instructions))\b",
    r"\bsystem\s+prompt\b",
    r"\bjailbreak\b",
    r"\bdeveloper\s+mode\b",
    r"\bpretend\s+(to\s+be|you\s+are|that\s+you)\b",
]

_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in _INJECTION_PATTERNS), re.IGNORECASE)


def looks_like_prompt_injection(text: str) -> bool:
    """True if `text` contains a phrase that tries to override the
    assistant's instructions or role. Never raises."""
    return bool(text) and _INJECTION_RE.search(text) is not None
