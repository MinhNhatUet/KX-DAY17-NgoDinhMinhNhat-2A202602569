from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Deterministic character heuristic, rounded up for nonempty text."""
    return (len(text.strip()) + 3) // 4


@dataclass
class UserProfileStore:
    """UTF-8 markdown profiles at root_dir/<safe user id>/User.md."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        if not user_id.strip():
            raise ValueError("user_id must not be empty")
        # Preserve ordinary dataset ids. Hash transformed ids to avoid collisions,
        # including case-insensitive filesystems and Windows reserved names.
        reserved = re.fullmatch(r"con|prn|aux|nul|com[1-9]|lpt[1-9]", user_id, re.I)
        if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", user_id) and not reserved:
            slug = user_id
        else:
            label = re.sub(r"[^a-z0-9_-]+", "-", user_id.lower()).strip("-_")[:40] or "user"
            digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]
            slug = f"_{label}-{digest}"
        root = self.root_dir.resolve()
        path = (root / slug / "User.md").resolve()
        if not path.is_relative_to(root):
            raise ValueError("Profile path escapes root_dir")
        return path

    def read_text(self, user_id: str) -> str:
        try:
            return self.path_for(user_id).read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        if not search_text:
            return False
        original = self.read_text(user_id)
        edited = original.replace(search_text, replacement, 1)
        if original == edited:
            return False
        self.write_text(user_id, edited)
        return True

    def file_size(self, user_id: str) -> int:
        try:
            return self.path_for(user_id).stat().st_size
        except FileNotFoundError:
            return 0

    def facts(self, user_id: str) -> dict[str, str]:
        return dict(re.findall(r"^- ([a-z_]+): (.+)$", self.read_text(user_id), re.M))

    def upsert_fact(self, user_id: str, key: str, value: str) -> Path:
        """Replace a keyed fact, keeping unrelated markdown and other facts."""
        if not re.fullmatch(r"[a-z_]+", key):
            raise ValueError("Fact keys must use lowercase letters and underscores")
        value = " ".join(value.split())
        if not value:
            raise ValueError("Fact values must not be empty")
        content = self.read_text(user_id) or "# User profile\n"
        lines = content.splitlines()
        prefix = f"- {key}: "
        positions = [i for i, line in enumerate(lines) if line.startswith(prefix)]
        if positions:
            lines[positions[0]] = prefix + value
            lines = [line for i, line in enumerate(lines) if i not in positions[1:]]
        else:
            lines.append(prefix + value)
        return self.write_text(user_id, "\n".join(lines) + "\n")


def _clean_fact(value: str) -> str:
    value = re.split(
        r",|;|\s+(?:chứ|nhưng|dù|vì|để|trong giai đoạn|mỗi ngày|vài tháng|cho|và đang|và mình)\b",
        value, maxsplit=1, flags=re.I,
    )[0]
    return value.strip(" .!:")


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_HEDGE = re.compile(r"\b(?:hình như|có lẽ|chắc là|dự định|đang cân nhắc|sắp)\b", re.I)
_ASSERT = re.compile(r"nhưng thực ra|nhưng hiện tại|không còn|\b(?:hiện tại|bây giờ|giờ|vẫn|đang)\b", re.I)


def fact_confidence(sentence: str) -> float:
    """Hedged = 0.3, explicit current/correction marker = 0.9, plain = 0.7."""
    # ponytail: hand-tuned lexical scores; calibrate on labelled turns or an LLM judge if needed.
    return 0.3 if _HEDGE.search(sentence) else 0.9 if _ASSERT.search(sentence) else 0.7


def scored_profile_updates(message: str) -> dict[str, tuple[str, float]]:
    """Same facts as extract_profile_updates(), each with its sentence's confidence."""
    scored: dict[str, tuple[str, float]] = {}
    for sentence in _SENTENCE_SPLIT.split(message.strip()):
        confidence = fact_confidence(sentence)
        for key, value in extract_profile_updates(sentence).items():
            scored[key] = (value, confidence)
    return scored


def extract_profile_updates(message: str) -> dict[str, str]:
    """Conservative Vietnamese self-report rules; later assertions win.

    These offline rules cover explicit facts, not arbitrary natural language.
    Temporary news and requests to recall facts are not profile updates.
    """
    updates: dict[str, str] = {}
    for sentence in _SENTENCE_SPLIT.split(message.strip()):
        # A correction may contrast a historical clause with a current one.
        sentence = re.split(r"nhưng thực ra|nhưng hiện tại", sentence, flags=re.I)[-1]
        lower = sentence.lower()
        if not sentence or "?" in sentence:
            continue
        if re.search(r"nhắc lại giúp|nhắc lại (?:tên|style)|nhớ lại xem|thử nhớ lại", lower):
            continue
        if re.search(r"\b(?:nếu|giả sử|đùa|tạm thời|lúc đầu|trước đây|hồi trước)\b", lower):
            continue
        if re.search(r"(?:nhắc lại|nhớ lại|hỏi lại).*(?:là gì|ở đâu|tên gì)", lower):
            continue

        patterns = {
            "name": r"(?:mình tên(?: là)?|tên mình là)\s+([^,.!?;]+)",
            "location": r"(?:mình\s+(?:(?:vẫn|hiện tại|hiện|đang)\s+)*ở|hiện ở|nơi ở hiện tại là)\s+([^,.!?;]+)",
            "favorite_drink": r"đồ uống yêu thích(?: của mình)? là\s+([^,.!?;]+)",
            "favorite_food": r"món ăn yêu thích(?: của mình)? là\s+([^,.!?;]+)",
            "pet": r"mình nuôi\s+(?:một\s+)?(?:bé\s+)?([^,.!?;]+)",
        }
        # Only treat a work location as residence when explicitly a move/stay.
        if re.search(r"vài tháng|chuyển (?:đến|tới|sang)", lower):
            match = re.search(r"mình đang làm việc ở\s+([^,.!?;]+)", sentence, re.I)
            if match:
                updates["location"] = _clean_fact(match[1])
        for key, pattern in patterns.items():
            for match in re.finditer(pattern, sentence, re.I):
                value = _clean_fact(match[1])
                if value and not re.search(r"\b(?:gì|đâu|không|chưa)\b", value, re.I):
                    updates[key] = value

        # Require an explicit current-job assertion, not an incidental job word.
        job = re.search(
            r"(?:mình (?:đang )?làm|và đang làm|giờ chuyển sang|"
            r"nghề nghiệp(?: hiện tại)? (?:thì )?vẫn là|nghề nghiệp hiện tại là)\s+"
            r"([\w -]+?(?:engineer|developer|manager|scientist|designer)|giáo viên|bác sĩ)",
            sentence, re.I,
        )
        if job:
            updates["profession"] = job[1].strip()

        if re.search(r"trả lời|giải thích|style", lower):
            if re.search(r"mình (?:vẫn )?(?:muốn|thích)|hãy trả lời|style trả lời.*(?:giữ nguyên|ngắn)", lower):
                style = []
                if re.search(r"ngắn|gọn", lower):
                    style.append("ngắn gọn")
                bullets = re.search(r"\b(\d+) bullet", lower)
                if bullets:
                    style.append(f"{bullets[1]} bullet")
                elif "bullet" in lower:
                    style.append("bullet")
                for detail in ("rõ ý", "có cấu trúc", "ví dụ thực tế", "ví dụ thực chiến", "trade-off"):
                    if detail in lower:
                        style.append(detail)
                if style:
                    updates["response_style"] = ", ".join(style)

        if re.search(r"mình (?:(?:vẫn|còn) )?(?:thích|đang quan tâm|quan tâm)", lower):
            interests = [term for term in ("Python", "AI", "MLOps", "RAG")
                         if re.search(rf"\b{term}\b", sentence, re.I)]
            if interests:
                updates["interests"] = ", ".join(interests)
            hobbies = [term for term in ("chạy bộ", "lo-fi", "chụp ảnh") if term in lower]
            if hobbies:
                updates["hobbies"] = ", ".join(hobbies)
    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Keep bounded sentence excerpts, favoring explicit facts and user content.

    This is a lossy heuristic, not a semantic LLM summary. Prior summary lines
    are candidates too, so repeated compaction does not blindly discard them.
    """
    if max_items <= 0:
        return ""
    candidates: dict[str, tuple[int, int]] = {}
    index = 0
    for message in messages:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", message["content"]):
            snippet = " ".join(sentence.lstrip("- ").split())
            if not snippet:
                continue
            score = 1 if message["role"] in ("user", "summary") else 0
            if re.search(r"tên|hiện tại|chuyển|thích|trade-off|mục tiêu|nhớ|quan tâm", snippet, re.I):
                score += 2
            snippet = snippet[:180]
            candidates[snippet] = (score, index)
            index += 1
    selected = sorted(candidates, key=lambda item: candidates[item], reverse=True)[:max_items]
    selected.sort(key=lambda item: candidates[item][1])
    return "\n".join(f"- {snippet}" for snippet in selected)


@dataclass
class CompactMemoryManager:
    """Per-thread recent messages plus a bounded rolling summary."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.threshold_tokens <= 0 or self.keep_messages <= 0:
            raise ValueError("threshold_tokens and keep_messages must be positive")

    def append(self, thread_id: str, role: str, content: str) -> None:
        context = self.context(thread_id)
        messages = context["messages"]
        messages.append({"role": role, "content": content})
        total = estimate_tokens(context["summary"]) + sum(estimate_tokens(m["content"]) for m in messages)
        if total <= self.threshold_tokens or len(messages) <= self.keep_messages:
            return
        old_messages = messages[:-self.keep_messages]
        previous = context["summary"]
        inputs = ([{"role": "summary", "content": previous}] if previous else []) + old_messages
        summary = summarize_messages(inputs)
        # Reserve at most a quarter of the threshold for summary, and ensure
        # it is strictly smaller than the material it replaces.
        replaced_tokens = estimate_tokens(previous) + sum(estimate_tokens(m["content"]) for m in old_messages)
        budget = min(self.threshold_tokens // 4, max(0, replaced_tokens - 1))
        context["summary"] = summary[:budget * 4].rstrip()
        context["messages"] = messages[-self.keep_messages:]
        context["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        """Return live thread state; callers should use append to add messages."""
        return self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def compaction_count(self, thread_id: str) -> int:
        return self.context(thread_id)["compactions"]
