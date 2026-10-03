import copy
import json
from pathlib import Path

import pytest

import benchmark
from benchmark import heuristic_quality, load_conversations, recall_points, run_agent_benchmark
from config import LabConfig


@pytest.mark.parametrize("answer,expected,recall,quality", [
    ("", ["A"], 0, 0),
    ("Python", ["Python", "AI", "MLOps"], 0.5, 1 / 3),
    ("PYTHON và ai", ["Python", "AI"], 1, 1),
    ("Anything", [], 0, 0),
])
def test_scoring(answer, expected, recall, quality):
    assert recall_points(answer, expected) == recall
    assert heuristic_quality(answer, expected) == quality


def test_dataset_validation(tmp_path):
    path = tmp_path / "data.json"
    path.write_text('[{"id": "broken"}]', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid conversation"):
        load_conversations(path)
    path.write_text('{}', encoding="utf-8")
    with pytest.raises(ValueError, match="list of conversations"):
        load_conversations(path)
    path.write_text('[]', encoding="utf-8")
    assert load_conversations(path) == []


class RecordingAgent:
    def __init__(self):
        self.calls = []

    def reply(self, user, thread, message):
        self.calls.append((user, thread, message))
        return {"answer": "Lan", "agent_tokens_only": 2, "prompt_tokens_processed": 5}

    def memory_file_size(self, user):
        return 100 + len(self.calls)

    def compaction_count(self, thread):
        return sum(call[1] == thread for call in self.calls)


def test_fair_order_fresh_recall_threads_and_accounting(tmp_path):
    conversations = [
        {"id": "c1", "user_id": "u", "turns": ["one", "two"],
         "recall_questions": [{"question": "q1", "expected_contains": ["Lan"]},
                              {"question": "q2", "expected_contains": ["Lan", "AI"]}]},
        {"id": "c2", "user_id": "u", "turns": ["three"],
         "recall_questions": [{"question": "q3", "expected_contains": ["missing"]}]},
    ]
    original = copy.deepcopy(conversations)
    a, b = RecordingAgent(), RecordingAgent()
    row = run_agent_benchmark("A", a, conversations, LabConfig(state_dir=tmp_path))
    run_agent_benchmark("B", b, conversations, LabConfig(state_dir=tmp_path))
    assert a.calls == b.calls
    assert conversations == original
    assert [call[2] for call in a.calls] == ["one", "two", "q1", "q2", "three", "q3"]
    assert a.calls[0][1] == a.calls[1][1]
    assert a.calls[2][1] == a.calls[3][1]
    assert len({a.calls[i][1] for i in (0, 2, 4, 5)}) == 4
    assert row.agent_tokens_only == 12
    assert row.prompt_tokens_processed == 30
    assert row.memory_growth_bytes == 6  # shared user counted once, net change
    assert row.compactions == 6
    assert row.recall_score == 0.5
    assert row.response_quality == 0.5


def test_empty_suite(tmp_path):
    row = run_agent_benchmark("Empty", RecordingAgent(), [], LabConfig(state_dir=tmp_path))
    assert row.agent_tokens_only == row.prompt_tokens_processed == 0
    assert row.recall_score == row.response_quality == 0
    assert row.memory_growth_bytes == row.compactions == 0


def test_main_two_suites_repeatable_and_preserves_existing_state(tmp_path, monkeypatch, capsys):
    data_dir = Path(__file__).resolve().parent.parent / "data"
    config = LabConfig(data_dir=data_dir, state_dir=tmp_path)
    existing = tmp_path / "profiles" / "dungct" / "User.md"
    existing.parent.mkdir(parents=True)
    existing.write_text("Existing unrelated profile", encoding="utf-8")
    before = {path.name: path.read_bytes() for path in data_dir.glob('*.json')}
    monkeypatch.setattr(benchmark, "load_config", lambda root: config)
    benchmark.main()
    first = capsys.readouterr().out
    benchmark.main()
    second = capsys.readouterr().out
    assert first == second
    assert first.count("## Standard Benchmark") == 1
    assert first.count("## Long-Context Stress Benchmark") == 1
    assert first.count("| Baseline |") == first.count("| Advanced |") == 2
    assert first.count("| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |") == 2
    assert existing.read_text(encoding="utf-8") == "Existing unrelated profile"
    assert {path.name: path.read_bytes() for path in data_dir.glob('*.json')} == before
    assert len(list((tmp_path / "benchmarks").glob("run-*"))) == 2
