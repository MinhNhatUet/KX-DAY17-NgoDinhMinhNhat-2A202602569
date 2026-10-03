"""Unit checks for memory, independent of the unfinished agents."""
import json
from pathlib import Path

import pytest

from memory_store import (
    CompactMemoryManager, UserProfileStore, estimate_tokens,
    extract_profile_updates, summarize_messages,
)


def test_token_estimator():
    assert estimate_tokens("") == estimate_tokens(" \n\t ") == 0
    assert estimate_tokens("x") == 1
    text = "Tiếng Việt có dấu"
    assert estimate_tokens(text) == estimate_tokens(text)
    assert estimate_tokens(text + " dài hơn") >= estimate_tokens(text)
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_profile_disk_roundtrip_and_correction(tmp_path):
    store = UserProfileStore(tmp_path)
    assert store.read_text("dungct") == ""
    assert store.file_size("dungct") == 0
    assert not store.edit_text("dungct", "missing", "replacement")
    content = "# Hồ sơ\nHuế và Huế\n"
    path = store.write_text("dungct", content)
    assert path == tmp_path / "dungct" / "User.md"
    assert store.file_size("dungct") == len(content.encode("utf-8"))
    assert store.edit_text("dungct", "Huế", "Đà Nẵng")
    assert store.read_text("dungct") == "# Hồ sơ\nĐà Nẵng và Huế\n"
    assert not store.edit_text("dungct", "Huế", "Huế")
    assert not store.edit_text("dungct", "", "anything")
    store.upsert_fact("dungct", "location", "Huế")
    store.upsert_fact("dungct", "location", "Đà Nẵng")
    reopened = UserProfileStore(tmp_path)
    assert reopened.facts("dungct") == {"location": "Đà Nẵng"}
    assert reopened.read_text("dungct").count("- location:") == 1
    assert "# Hồ sơ" in reopened.read_text("dungct")
    assert reopened.read_text("other_user") == ""


def test_profile_paths_stay_inside_root(tmp_path):
    store = UserProfileStore(tmp_path)
    ids = ["../../escape", r"..\escape", "C:/outside", "CON", "DungCT", "dungct", "a/b", "a_b"]
    paths = [store.path_for(user) for user in ids]
    assert len({str(path).lower() for path in paths}) == len(ids)
    assert all(path.is_relative_to(tmp_path) for path in paths)
    with pytest.raises(ValueError):
        store.path_for(" ")


@pytest.mark.parametrize("message", [
    "Mình tên gì?", "Mình đang ở Huế phải không?",
    "Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày.",
    "Mình đùa rằng mình làm product manager.",
    "Nếu mình đang ở Hà Nội thì sao.",
    "Trước đây mình làm backend engineer.",
    "Tạm thời mình thích Python trong bài tập này.",
    "Nhắc lại giúp mình tên và style trả lời mình thích trong stress test này.",
])
def test_questions_and_noise_are_not_facts(message):
    assert extract_profile_updates(message) == {}


def test_explicit_facts_and_corrections():
    assert extract_profile_updates("Mình tên là Lan. Bạn tên gì?") == {"name": "Lan"}
    assert extract_profile_updates("Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.") == {
        "location": "Đà Nẵng", "profession": "backend engineer",
    }
    assert extract_profile_updates("Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.") == {
        "profession": "MLOps engineer",
    }
    assert extract_profile_updates("Lúc đầu mình nói hiện ở Huế, nhưng thực ra từ tuần này mình đang làm việc ở Đà Nẵng vài tháng.") == {
        "location": "Đà Nẵng",
    }


def test_compaction_keeps_tail_and_isolates_threads():
    manager = CompactMemoryManager(threshold_tokens=80, keep_messages=2)
    turns = [{"role": "user", "content": f"Lượt {i}: " + "nội dung " * 30} for i in range(8)]
    for turn in turns:
        manager.append("long", **turn)
    context = manager.context("long")
    assert context["messages"] == turns[-2:]
    assert 0 < estimate_tokens(context["summary"]) <= 20
    assert manager.compaction_count("long") > 1
    assert manager.context("new") == {"messages": [], "summary": "", "compactions": 0}
    # A single oversized recent message cannot be shortened without violating
    # the verbatim-tail contract, and must not cause an infinite compact loop.
    manager.append("huge", "user", "x" * 4000)
    assert manager.compaction_count("huge") == 0


def test_summary_is_bounded_and_retains_previous_context():
    summary = summarize_messages([{"role": "user", "content": "Mình quan tâm Python."}])
    next_summary = summarize_messages([
        {"role": "summary", "content": summary},
        {"role": "user", "content": "Một nội dung khác."},
    ])
    assert "Python" in next_summary
    assert len(summarize_messages([{"role": "user", "content": "x" * 10000}])) <= 182
    assert summarize_messages([], max_items=0) == ""


def test_threshold_boundary_and_invalid_settings():
    manager = CompactMemoryManager(3, 1)
    for _ in range(3):
        manager.append("thread", "user", "abcd")
    assert manager.compaction_count("thread") == 0
    manager.append("thread", "user", "e")
    assert manager.compaction_count("thread") == 1
    assert manager.context("thread")["messages"] == [{"role": "user", "content": "e"}]
    for threshold, keep in [(0, 1), (1, 0), (-1, 1)]:
        with pytest.raises(ValueError):
            CompactMemoryManager(threshold, keep)


def test_fixed_datasets_profile_and_prompt_load(tmp_path):
    data = Path(__file__).resolve().parent.parent / "data"
    store = UserProfileStore(tmp_path)
    for filename in ("conversations.json", "advanced_long_context.json"):
        manager = CompactMemoryManager(1000, 4)
        full_load = compact_load = 0
        for conversation in json.loads((data / filename).read_text(encoding="utf-8")):
            history = []
            for index, turn in enumerate(conversation["turns"]):
                for key, value in extract_profile_updates(turn).items():
                    store.upsert_fact(conversation["user_id"], key, value)
                manager.append(conversation["id"], "user", turn)
                history.append(turn)
                context = manager.context(conversation["id"])
                full_load += sum(estimate_tokens(text) for text in history)
                compact_load += estimate_tokens(context["summary"]) + sum(
                    estimate_tokens(m["content"]) for m in context["messages"])
                if filename == "advanced_long_context.json" and index in (8, 9):
                    facts = store.facts(conversation["user_id"])
                    assert facts["location"] == "Đà Nẵng"
                    assert facts["profession"] == "MLOps engineer"
            if filename == "conversations.json":
                assert manager.compaction_count(conversation["id"]) == 0
        if filename == "advanced_long_context.json":
            assert manager.compaction_count("stress-01") > 1
            assert compact_load < full_load
    regular = store.facts("dungct")
    assert regular["name"] == "DũngCT"
    assert regular["location"] == "Huế"
    assert regular["profession"] == "MLOps engineer"
    assert regular["favorite_drink"] == "cà phê sữa đá"
    assert regular["favorite_food"] == "mì Quảng"
    assert "corgi" in regular["pet"]
    stress = store.facts("dungct_stress")
    assert stress["name"] == "DũngCT Stress"
    assert stress["location"] == "Đà Nẵng"
    assert stress["profession"] == "MLOps engineer"
    assert "3 bullet" in stress["response_style"]
