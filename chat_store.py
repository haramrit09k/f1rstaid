"""Keep a conversation across page refreshes.

Phone browsers routinely discard background tabs, and a Streamlit page
reload starts a brand-new session -- so the whole chat (and the free-question
count) used to vanish. The chat is saved in Redis for 24 hours under a random,
unguessable id that lives in the page URL (?c=...); opening that URL restores
it. Without Redis nothing is saved and the app behaves as before.

Privacy: the saved chat holds the student's questions and the answers shown
(never an API key). Anyone who has the URL can read it until it expires, and
"New chat" deletes it immediately.
"""

import json
import logging
import re
import secrets
from typing import Any, Callable, Dict, List, Optional

from langchain_core.documents import Document

CHAT_TTL_SECONDS = 24 * 3600
MAX_MESSAGES = 40
MAX_DOC_CHARS = 1500
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,48}$")


def new_chat_id() -> str:
    return secrets.token_urlsafe(12)


def is_valid_chat_id(value: Any) -> bool:
    return isinstance(value, str) and bool(_ID_RE.match(value))


def _encode_message(message: Dict) -> Dict:
    out = {"role": message.get("role"), "content": message.get("content", "")}
    answer = message.get("answer")
    if answer:
        encoded = {k: v for k, v in answer.items() if k != "source_documents"}
        encoded["source_documents"] = [
            {"page_content": d.page_content[:MAX_DOC_CHARS], "metadata": d.metadata}
            for d in answer.get("source_documents", [])
        ]
        out["answer"] = encoded
    return out


def _decode_message(raw: Dict) -> Dict:
    message = {"role": raw["role"], "content": raw.get("content", "")}
    answer = raw.get("answer")
    if answer:
        answer = dict(answer)
        answer["source_documents"] = [
            Document(page_content=d["page_content"], metadata=d.get("metadata", {}))
            for d in answer.get("source_documents", [])
        ]
        message["answer"] = answer
    return message


class ChatStore:
    """Redis-backed chat persistence. `get_client` returns a Redis client or
    None (not configured); every method fails soft."""

    def __init__(self, get_client: Callable[[], Any]):
        self._get_client = get_client

    @staticmethod
    def _key(chat_id: str) -> str:
        return f"f1rstaid:chat:{chat_id}"

    def available(self) -> bool:
        return self._get_client() is not None

    def save(self, chat_id: str, messages: List[Dict], shared_key_count: int = 0) -> bool:
        client = self._get_client()
        if client is None or not is_valid_chat_id(chat_id):
            return False
        try:
            payload = {
                "messages": [_encode_message(m) for m in messages[-MAX_MESSAGES:]],
                "shared_key_count": int(shared_key_count),
            }
            client.set(self._key(chat_id), json.dumps(payload, default=str), ex=CHAT_TTL_SECONDS)
            return True
        except Exception as e:
            logging.error(f"Could not save chat: {e}")
            return False

    def load(self, chat_id: str) -> Optional[Dict]:
        client = self._get_client()
        if client is None or not is_valid_chat_id(chat_id):
            return None
        try:
            raw = client.get(self._key(chat_id))
            if raw is None:
                return None
            data = json.loads(raw)
            return {
                "messages": [_decode_message(m) for m in data.get("messages", [])],
                "shared_key_count": int(data.get("shared_key_count", 0)),
            }
        except Exception as e:
            logging.error(f"Could not load chat: {e}")
            return None

    def delete(self, chat_id: str) -> None:
        client = self._get_client()
        if client is None or not is_valid_chat_id(chat_id):
            return
        try:
            client.delete(self._key(chat_id))
        except Exception as e:
            logging.error(f"Could not delete chat: {e}")
