import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import estimate_tokens


def test_prompt_accounting_includes_all_three_layers(tmp_path, monkeypatch):
    agent = AdvancedAgent(LabConfig(state_dir=tmp_path, compact_threshold_tokens=100), force_offline=True)
    original = agent._offline_response
    prompt_costs = []

    def inspect_prompt(user, thread, message):
        context = agent.compact_memory.context(thread)
        profile_cost = estimate_tokens(agent.profile_store.read_text(user))
        summary_cost = estimate_tokens(context['summary'])
        recent_cost = sum(estimate_tokens(item['content']) for item in context['messages'])
        expected = profile_cost + summary_cost + recent_cost
        assert profile_cost > 0
        assert agent._estimate_prompt_context_tokens(user, thread) == expected
        prompt_costs.append(expected)
        return original(user, thread, message)

    monkeypatch.setattr(agent, '_offline_response', inspect_prompt)
    outputs = []
    for turn in ['Mình tên là Lan.'] + ['Nội dung dài ' * 50] * 10:
        result = agent.reply('u', 't', turn)
        assert result['prompt_tokens_processed'] == prompt_costs[-1]
        assert result['agent_tokens_only'] == estimate_tokens(result['answer'])
        outputs.append(result['agent_tokens_only'])
    assert agent.prompt_token_usage('t') == sum(prompt_costs)
    assert agent.token_usage('t') == sum(outputs)
    assert agent.compact_memory.context('t')['summary']
    assert agent.token_usage('missing') == agent.prompt_token_usage('missing') == 0


def test_correction_and_recall_do_not_duplicate_or_overwrite_facts(tmp_path):
    agent = AdvancedAgent(LabConfig(state_dir=tmp_path), force_offline=True)
    for message in ['Mình ở Huế.', 'Mình đang ở Đà Nẵng.', 'Mình làm MLOps engineer.']:
        agent.reply('u', 't', message)
    before = agent.profile_store.read_text('u')
    answer = agent.reply('u', 'new', 'Hiện tại mình làm nghề gì và ở đâu?')['answer']
    assert 'Đà Nẵng' in answer and 'MLOps engineer' in answer
    assert 'Huế' not in answer
    assert before.count('- location:') == 1
    assert agent.profile_store.read_text('u') == before
    with pytest.raises(ValueError, match='thread_id'):
        agent.reply('other', 't', 'Mình tên là Nam.')
    assert agent.memory_file_size('other') == 0


def test_offline_never_builds_a_model(tmp_path, monkeypatch):
    builder = Mock(side_effect=AssertionError('No live model allowed'))
    monkeypatch.setattr(AdvancedAgent, '_maybe_build_langchain_agent', builder)
    agent = AdvancedAgent(LabConfig(state_dir=tmp_path), force_offline=True)
    assert agent.reply('u', 't', 'Mình tên gì?')['mode'] == 'offline'
    builder.assert_not_called()


def test_all_dataset_recall_questions_in_fresh_threads(tmp_path):
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    for filename in ('conversations.json', 'advanced_long_context.json'):
        config = LabConfig(state_dir=tmp_path / filename)
        agent = AdvancedAgent(config, force_offline=True)
        baseline = BaselineAgent(config, force_offline=True)
        conversations = json.loads((data_dir / filename).read_text(encoding='utf-8'))
        for conversation in conversations:
            user, thread = conversation['user_id'], conversation['id']
            for turn in conversation['turns']:
                agent.reply(user, thread, turn)
                baseline.reply(user, thread, turn)
            profile_before = agent.profile_store.read_text(user)
            for index, question in enumerate(conversation['recall_questions']):
                result = agent.reply(user, f'{thread}-recall-{index}', question['question'])
                assert all(fact in result['answer'] for fact in question['expected_contains']), result['answer']
                if filename == 'advanced_long_context.json':
                    assert len(result['answer'].splitlines()) == 3
                    assert all(line.startswith('- ') for line in result['answer'].splitlines())
            assert agent.profile_store.read_text(user) == profile_before
        if filename == 'advanced_long_context.json':
            assert agent.memory_file_size(user) > 0
            assert agent.compaction_count(thread) > 1
            assert agent.prompt_token_usage(thread) < baseline.prompt_token_usage(thread)
            assert 'trade-off' in agent.profile_store.facts(user)['response_style']
