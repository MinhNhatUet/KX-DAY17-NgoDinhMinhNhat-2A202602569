from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, scored_profile_updates


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: per-thread compact context and durable per-user profile facts.

    The required runtime is deterministic and offline. The optional LangGraph
    tools/middleware integration is intentionally left for a separate extension.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.thread_users: dict[str, str] = {}
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Run the offline memory pipeline and return per-turn token estimates."""
        owner = self.thread_users.get(thread_id)
        if owner is not None and owner != user_id:
            raise ValueError("A thread_id cannot be shared by different users")
        self.thread_users[thread_id] = user_id
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        for key, (value, confidence) in scored_profile_updates(message).items():
            previous = self.profile_store.facts(user_id).get(key)
            # Replacing a different stored value needs stronger evidence than a new
            # fact; response_style merges below, so additions are not replacements.
            replaces = previous and previous != value and key != "response_style"
            required = self.config.profile_confidence_threshold + (0.2 if replaces else 0)
            if confidence < required:
                continue
            if key == "response_style":
                # Style reminders are partial: retain independent preferences,
                # but a new bullet count replaces the old count.
                previous = self.profile_store.facts(user_id).get(key, "")
                parts = previous.split(", ") if previous else []
                if re.search(r"\b\d+ bullet\b", value):
                    parts = [part for part in parts if not re.fullmatch(r"(?:\d+ )?bullet", part)]
                additions = value.split(", ")
                if "bullet" in additions and any(re.fullmatch(r"\d+ bullet", part) for part in parts):
                    additions.remove("bullet")
                value = ", ".join(dict.fromkeys(parts + additions))
            self.profile_store.upsert_fact(user_id, key, value)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        answer = self._offline_response(user_id, thread_id, message)
        output_tokens = estimate_tokens(answer)
        self.compact_memory.append(thread_id, "assistant", answer)
        self.thread_tokens[thread_id] = self.token_usage(thread_id) + output_tokens
        self.thread_prompt_tokens[thread_id] = self.prompt_token_usage(thread_id) + prompt_tokens
        return {
            "answer": answer,
            "agent_tokens_only": output_tokens,
            "prompt_tokens_processed": prompt_tokens,
            "mode": "offline",
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        context = self.compact_memory.context(thread_id)
        return (
            estimate_tokens(self.profile_store.read_text(user_id))
            + estimate_tokens(context["summary"])
            + sum(estimate_tokens(item["content"]) for item in context["messages"])
        )

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        facts = self.profile_store.facts(user_id)
        question = message.lower()
        recall = "?" in question or bool(re.search(r"nhắc lại|nhớ lại|tóm tắt|tên gì|nghề gì|ở đâu", question))
        if not recall:
            return "Mình đã ghi nhận thông tin trong cuộc trò chuyện này."
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
        if not parts:
            return "Mình chưa có thông tin đó trong hồ sơ của bạn."
        if any(key not in facts for key in selected):
            parts.append("Những thông tin còn lại chưa có trong hồ sơ của bạn.")
        style = facts.get("response_style", "")
        if "3 bullet" in style:
            # Group multiple requested facts into exactly three short bullets.
            # If fewer facts were requested, complete with relevant preferences.
            if len(parts) < 3 and "response_style" not in selected:
                parts.append(f"Phong cách trả lời: {style}.")
            if len(parts) < 3:
                parts.append("Trade-off: giữ fact để recall, nén lịch sử để giảm token."
                             if "trade-off" in style else "Thông tin trên lấy từ hồ sơ đã lưu.")
            while len(parts) < 3:
                parts.append("Bạn có thể đính chính khi thông tin thay đổi.")
            groups = [parts[0], parts[1], " ".join(parts[2:])]
            return "\n".join(f"- {part}" for part in groups)
        if "bullet" in style:
            return "\n".join(f"- {part}" for part in parts)
        return " ".join(parts)

    def _maybe_build_langchain_agent(self):
        """No live integration yet; the offline runtime requires no model SDK."""
        return None
