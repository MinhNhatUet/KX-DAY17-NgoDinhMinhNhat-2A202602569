from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read and validate the shared UTF-8 dataset without changing it."""
    conversations = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(conversations, list):
        raise ValueError(f"{path}: expected a list of conversations")
    for index, conversation in enumerate(conversations):
        valid = (
            isinstance(conversation, dict)
            and all(isinstance(conversation.get(key), str) and conversation[key] for key in ("id", "user_id"))
            and isinstance(conversation.get("turns"), list)
            and all(isinstance(turn, str) for turn in conversation["turns"])
            and isinstance(conversation.get("recall_questions"), list)
        )
        if not valid:
            raise ValueError(f"{path}: invalid conversation at index {index}")
        for question in conversation["recall_questions"]:
            if not (
                isinstance(question, dict)
                and isinstance(question.get("question"), str)
                and isinstance(question.get("expected_contains"), list)
                and all(isinstance(fact, str) and fact.strip() for fact in question["expected_contains"])
            ):
                raise ValueError(f"{path}: invalid recall question in conversation {index}")
    return conversations


def _coverage(answer: str, expected: list[str]) -> float:
    if not expected:
        return 0.0
    normalized = answer.casefold()
    return sum(fact.casefold() in normalized for fact in expected) / len(expected)


def recall_points(answer: str, expected: list[str]) -> float:
    """No facts = 0, some but not all = 0.5, all expected facts = 1."""
    coverage = _coverage(answer, expected)
    return 1.0 if coverage == 1.0 else 0.5 if coverage > 0 else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Offline quality proxy: fraction of expected facts mentioned, in [0, 1].

    This measures factual coverage only, not fluency, truth or contradictions.
    Empty reference lists receive zero rather than vacuous full credit.
    """
    return _coverage(answer, expected)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Run training then recall per conversation, using one fresh recall thread.

    Pass a newly constructed agent for each suite. All successful turns,
    including recall, contribute to token totals. Memory growth is net byte
    change over distinct users, not a sum of repeated snapshots.
    """
    users = {conversation["user_id"] for conversation in conversations}
    size = getattr(agent, "memory_file_size", lambda user_id: 0)
    initial_bytes = sum(size(user) for user in users)
    output_tokens = prompt_tokens = compactions = 0
    recall_scores: list[float] = []
    qualities: list[float] = []
    for index, conversation in enumerate(conversations):
        prefix = f"benchmark:{index}:{conversation['id']}"
        thread, recall_thread = f"{prefix}:chat", f"{prefix}:recall"
        user = conversation["user_id"]
        counts_before = sum(agent.compaction_count(key) for key in (thread, recall_thread))
        for turn in conversation["turns"]:
            result = agent.reply(user, thread, turn)
            output_tokens += result["agent_tokens_only"]
            prompt_tokens += result["prompt_tokens_processed"]
        # Same ids, order and inputs for both agents. Recall never uses chat.
        for question in conversation["recall_questions"]:
            result = agent.reply(user, recall_thread, question["question"])
            output_tokens += result["agent_tokens_only"]
            prompt_tokens += result["prompt_tokens_processed"]
            recall_scores.append(recall_points(result["answer"], question["expected_contains"]))
            qualities.append(heuristic_quality(result["answer"], question["expected_contains"]))
        compactions += sum(agent.compaction_count(key) for key in (thread, recall_thread)) - counts_before
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=output_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
        response_quality=sum(qualities) / len(qualities) if qualities else 0.0,
        memory_growth_bytes=sum(size(user) for user in users) - initial_bytes,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = ["Agent", "Agent tokens only", "Prompt tokens processed", "Cross-session recall",
               "Response quality", "Memory growth (bytes)", "Compactions"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] + ["---:"] * 6) + " |"]
    for row in rows:
        values = [row.agent_name.replace("|", "\\|").replace("\n", " "), str(row.agent_tokens_only),
                  str(row.prompt_tokens_processed), f"{row.recall_score:.2%}",
                  f"{row.response_quality:.2%}", str(row.memory_growth_bytes), str(row.compactions)]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)
    suites = [
        ("Standard Benchmark", "standard", load_conversations(config.data_dir / "conversations.json")),
        ("Long-Context Stress Benchmark", "stress", load_conversations(config.data_dir / "advanced_long_context.json")),
    ]
    # Preserve existing profiles and leave inspectable artifacts from each run.
    runs_dir = config.state_dir / "benchmarks"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(mkdtemp(prefix="run-", dir=runs_dir))
    for title, suite_name, conversations in suites:
        suite_config = replace(config, state_dir=run_dir / suite_name)
        agents = [
            ("Baseline", BaselineAgent(suite_config, force_offline=True)),
            ("Advanced", AdvancedAgent(suite_config, force_offline=True)),
        ]
        rows = [run_agent_benchmark(name, agent, conversations, suite_config) for name, agent in agents]
        print(f"## {title}\n")
        print(format_rows(rows))
        print()


if __name__ == "__main__":
    main()
