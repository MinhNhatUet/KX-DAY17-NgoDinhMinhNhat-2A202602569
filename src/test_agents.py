from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Isolate all writable state and avoid environment/API-key dependencies."""
    return LabConfig(
        base_dir=tmp_path,
        data_dir=Path(__file__).resolve().parent.parent / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=200,
        compact_keep_messages=4,
        model=ProviderConfig(provider="openai", model_name="offline-test", temperature=0.0),
        judge_model=ProviderConfig(provider="openai", model_name="offline-test", temperature=0.0),
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(make_config(tmp_path).state_dir / "profiles")
    assert store.read_text("u") == ""
    content = "# Hồ sơ\n- location: Huế\n"
    path = store.write_text("u", content)
    assert path.is_file()
    assert path.is_relative_to(tmp_path)
    assert store.read_text("u") == content
    assert store.edit_text("u", "Huế", "Đà Nẵng")
    assert store.read_text("u") == "# Hồ sơ\n- location: Đà Nẵng\n"
    assert store.facts("u")["location"] == "Đà Nẵng"
    assert store.file_size("u") == len(store.read_text("u").encode("utf-8"))


def test_compact_trigger(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    manager = CompactMemoryManager(config.compact_threshold_tokens, config.compact_keep_messages)
    for i in range(20):
        manager.append("long", "user", f"Lượt {i}. " + "Nội dung dài. " * 30)
    assert manager.compaction_count("long") > 0
    assert manager.context("long")["summary"]
    assert len(manager.context("long")["messages"]) == config.compact_keep_messages


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    for agent in (baseline, advanced):
        agent.reply("u", "first", "Mình tên là Lan.")
        assert "Lan" in agent.reply("u", "first", "Mình tên gì?")["answer"]
    assert "Lan" not in baseline.reply("u", "new", "Mình tên gì?")["answer"]
    assert "Lan" in advanced.reply("u", "new", "Mình tên gì?")["answer"]
    restarted = AdvancedAgent(config, force_offline=True)
    assert "Lan" in restarted.reply("u", "after-restart", "Mình tên gì?")["answer"]
    assert "Lan" not in restarted.reply("other", "other-thread", "Mình tên gì?")["answer"]
    assert advanced.memory_file_size("u") > 0


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    for turn in ["Mình tên là Lan."] + ["Bàn về hệ thống. " * 70] * 20:
        baseline.reply("u", "long", turn)
        advanced.reply("u", "long", turn)
    assert advanced.compaction_count("long") > 0
    assert baseline.compaction_count("long") == 0
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long")
    assert advanced.memory_file_size("u") > 0
