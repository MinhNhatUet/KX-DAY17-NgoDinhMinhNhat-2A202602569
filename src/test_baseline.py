"""Baseline behavior, independent of the unfinished Advanced agent."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import agent_baseline
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import estimate_tokens
from model_provider import ProviderConfig


def test_remembers_only_same_thread(tmp_path):
    agent = BaselineAgent(LabConfig(state_dir=tmp_path), force_offline=True)
    agent.reply("same-user", "first", "Mình tên là Lan.")
    assert "Lan" in agent.reply("same-user", "first", "Mình tên gì?")["answer"]
    assert "Lan" not in agent.reply("same-user", "second", "Mình tên gì?")["answer"]
    assert "chưa có" in agent.reply("same-user", "second", "Mình tên gì?")["answer"]
    assert "Lan" in agent.reply("same-user", "first", "Nhắc lại tên mình.")["answer"]
    assert not list(tmp_path.rglob("*"))


def test_existing_profile_is_not_read(tmp_path):
    profile = tmp_path / "profiles" / "same-user" / "User.md"
    profile.parent.mkdir(parents=True)
    profile.write_text("- name: SecretName\n", encoding="utf-8")
    agent = BaselineAgent(LabConfig(state_dir=tmp_path), force_offline=True)
    result = agent.reply("same-user", "new", "Mình tên gì?")
    assert "SecretName" not in result["answer"]
    assert profile.read_text(encoding="utf-8") == "- name: SecretName\n"


def test_corrections_are_local_and_replies_are_deterministic(tmp_path):
    config = LabConfig(state_dir=tmp_path)
    turns = ["Mình ở Huế.", "Mình đang ở Đà Nẵng.", "Hiện tại mình đang ở đâu?"]
    results = []
    for _ in range(2):
        agent = BaselineAgent(config, force_offline=True)
        results.append([agent.reply("u", "t", turn) for turn in turns])
    assert results[0] == results[1]
    assert "Đà Nẵng" in results[0][-1]["answer"]
    assert "Huế" not in results[0][-1]["answer"]


def test_token_counters_accumulate_each_turn_without_compaction(tmp_path):
    agent = BaselineAgent(LabConfig(state_dir=tmp_path, compact_threshold_tokens=1), force_offline=True)
    assert agent.token_usage("missing") == agent.prompt_token_usage("missing") == 0
    assert agent.sessions == {}
    expected_prompt = expected_output = 0
    history = []
    for message in ("Mình tên là Lan.", "Mình tên gì?", "x" * 400):
        prompt = sum(estimate_tokens(text) for text in history) + estimate_tokens(message)
        result = agent.reply("u", "t", message)
        assert result["prompt_tokens_processed"] == prompt
        assert result["agent_tokens_only"] == estimate_tokens(result["answer"])
        expected_prompt += prompt
        expected_output += estimate_tokens(result["answer"])
        history.extend([message, result["answer"]])
        assert agent.prompt_token_usage("t") == expected_prompt
        assert agent.token_usage("t") == expected_output
    assert len(agent.sessions["t"].messages) == 6
    assert agent.compaction_count("t") == 0
    assert agent.prompt_token_usage("missing") == 0


def test_no_builder_in_forced_offline_or_without_credentials(monkeypatch):
    builder = Mock(side_effect=AssertionError("Must not build a model"))
    monkeypatch.setattr(agent_baseline, "build_chat_model", builder)
    config = LabConfig(model=ProviderConfig("openai", "example", 0, api_key="test-key"))
    assert BaselineAgent(config, force_offline=True).langchain_agent is None
    config.model.api_key = None
    assert BaselineAgent(config).langchain_agent is None
    builder.assert_not_called()


def test_live_route_receives_only_current_thread_and_counts_text(monkeypatch):
    model = Mock()
    model.invoke.return_value = SimpleNamespace(content=[{"type": "text", "text": "Xin chào."}])
    builder = Mock(return_value=model)
    monkeypatch.setattr(agent_baseline, "build_chat_model", builder)
    config = LabConfig(model=ProviderConfig("openai", "example", 0, api_key="test-key"))
    agent = BaselineAgent(config)
    result = agent.reply("u", "first", "Mình tên là Lan.")
    agent.reply("u", "first", "Xin chào")
    assert len(model.invoke.call_args.args[0]) == 3
    agent.reply("u", "second", "Mình tên gì?")
    assert model.invoke.call_args.args[0] == [{"role": "user", "content": "Mình tên gì?"}]
    assert result["mode"] == "live"
    assert result["agent_tokens_only"] == estimate_tokens("Xin chào.")
    builder.assert_called_once_with(config.model)
    model.invoke.side_effect = RuntimeError("Request failed")
    before = list(agent.sessions["first"].messages)
    tokens = agent.token_usage("first"), agent.prompt_token_usage("first")
    with pytest.raises(RuntimeError):
        agent.reply("u", "first", "Retry me")
    assert agent.sessions["first"].messages == before
    assert (agent.token_usage("first"), agent.prompt_token_usage("first")) == tokens


def test_missing_sdk_falls_back_to_offline(monkeypatch):
    monkeypatch.setattr(agent_baseline, "build_chat_model", Mock(side_effect=ImportError("missing SDK")))
    config = LabConfig(model=ProviderConfig("ollama", "example", 0))
    with pytest.warns(RuntimeWarning, match="offline"):
        agent = BaselineAgent(config)
    assert agent.reply("u", "t", "Chào bạn")["mode"] == "offline"
