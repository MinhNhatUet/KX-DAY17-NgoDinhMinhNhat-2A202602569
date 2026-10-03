from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model, normalize_provider


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: full history within a thread, no profile files or compaction."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return answer and per-turn heuristic token counts.

        user_id is intentionally not used for memory lookup. Thread ids must
        uniquely identify conversations. No history is shared between threads.
        """
        if self.force_offline or self.langchain_agent is None:
            return self._reply_offline(thread_id, message)
        session = self.sessions.setdefault(thread_id, SessionState())
        prompt = [dict(item) for item in session.messages]
        prompt.append({"role": "user", "content": message})
        # Pass only this thread's complete history. Commit after a successful
        # invocation so retries do not duplicate a failed user turn.
        response = self.langchain_agent.invoke(prompt)
        content = response.content
        if isinstance(content, str):
            answer = content
        else:
            answer = "\n".join(
                block if isinstance(block, str) else block.get("text", "")
                for block in content
                if isinstance(block, str) or isinstance(block, dict) and block.get("type") == "text"
            )
        session.messages.append({"role": "user", "content": message})
        return self._record_reply(session, answer, "live")

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        # Reconstruct facts from this thread only; do not retain a user profile.
        facts: dict[str, str] = {}
        for item in session.messages:
            if item["role"] == "user":
                facts.update(extract_profile_updates(item["content"]))
        question = message.lower()
        recall = "?" in question or bool(re.search(r"nhắc lại|nhớ lại|tóm tắt|tên gì|nghề gì|ở đâu", question))
        if not recall:
            answer = "Mình đã ghi nhận thông tin trong cuộc trò chuyện này."
        else:
            fields = {
                "name": (r"tên|là ai", "Tên"),
                "location": (r"ở đâu|nơi ở|đang ở|còn ở", "Nơi ở hiện tại"),
                "profession": (r"nghề|công việc", "Nghề nghiệp hiện tại"),
                "favorite_drink": (r"đồ uống|uống.*thích", "Đồ uống yêu thích"),
                "favorite_food": (r"món ăn|ăn.*thích", "Món ăn yêu thích"),
                "pet": (r"nuôi|con gì|thú cưng", "Thú cưng"),
                "response_style": (r"style|kiểu trả lời|trả lời.*thế nào|phong cách", "Phong cách trả lời"),
                "interests": (r"quan tâm|kỹ thuật", "Mối quan tâm"),
                "hobbies": (r"sở thích", "Sở thích"),
            }
            selected = [key for key, (pattern, _) in fields.items() if re.search(pattern, question)]
            if not selected and "tóm tắt" in question:
                selected = list(fields)
            parts = [f"{fields[key][1]}: {facts[key]}." for key in selected if key in facts]
            if parts:
                answer = " ".join(parts)
                if any(key not in facts for key in selected):
                    answer += " Những thông tin còn lại chưa có trong cuộc trò chuyện này."
            else:
                answer = "Mình chưa có thông tin đó trong cuộc trò chuyện này."
        return self._record_reply(session, answer, "offline")

    @staticmethod
    def _record_reply(session: SessionState, answer: str, mode: str) -> dict[str, Any]:
        prompt_tokens = sum(estimate_tokens(item["content"]) for item in session.messages)
        output_tokens = estimate_tokens(answer)
        session.prompt_tokens_processed += prompt_tokens
        session.token_usage += output_tokens
        session.messages.append({"role": "assistant", "content": answer})
        return {
            "answer": answer,
            "agent_tokens_only": output_tokens,
            "prompt_tokens_processed": prompt_tokens,
            "mode": mode,
        }

    def _maybe_build_langchain_agent(self):
        """Optionally build a stateless chat model; sessions own all history."""
        if self.force_offline:
            return None
        provider = normalize_provider(self.config.model.provider)
        if provider == "custom" and not self.config.model.base_url:
            return None
        if provider not in ("custom", "ollama") and not self.config.model.api_key:
            return None
        try:
            return build_chat_model(self.config.model)
        except ImportError:
            warnings.warn("Provider SDK unavailable; Baseline will run offline.", RuntimeWarning, stacklevel=2)
            return None
