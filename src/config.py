from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

from model_provider import ProviderConfig, normalize_provider

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "custom": "local-model",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-sonnet-4-5",
    "ollama": "llama3.2",
    "openrouter": "openai/gpt-4o-mini",
}


@dataclass
class LabConfig:
    """Shared paths, compact-memory limits, and live model settings."""

    base_dir: Path = REPO_ROOT
    data_dir: Path = REPO_ROOT / "data"
    state_dir: Path = REPO_ROOT / "state"
    compact_threshold_tokens: int = 1000
    compact_keep_messages: int = 4
    profile_confidence_threshold: float = 0.6
    model: ProviderConfig = field(default_factory=lambda: ProviderConfig("openai", "gpt-4o-mini", 0.0))
    judge_model: ProviderConfig = field(default_factory=lambda: ProviderConfig("openai", "gpt-4o-mini", 0.0))


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


def _provider_config(provider: str) -> ProviderConfig:
    prefix = provider.upper()
    return ProviderConfig(
        provider=provider,
        model_name=DEFAULT_MODELS[provider],
        temperature=0.0,
        api_key=os.getenv(f"{prefix}_API_KEY") or None,
        base_url=(None if provider == "gemini" else
                  os.getenv(f"{prefix}_BASE_URL") or
                  ("http://localhost:11434" if provider == "ollama" else None)),
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Read root/.env without overriding process env; never instantiate a model."""
    root = (base_dir or REPO_ROOT).resolve()
    load_dotenv(root / ".env", override=False)
    provider = normalize_provider(os.getenv("LLM_PROVIDER", "openai"))
    model = _provider_config(provider)
    model.model_name = os.getenv("LLM_MODEL") or model.model_name
    model.temperature = float(os.getenv("LLM_TEMPERATURE", "0"))

    judge_provider = normalize_provider(os.getenv("JUDGE_PROVIDER", provider))
    judge = replace(model) if judge_provider == provider else _provider_config(judge_provider)
    judge.model_name = os.getenv("JUDGE_MODEL") or judge.model_name
    judge.temperature = float(os.getenv("JUDGE_TEMPERATURE", "0"))
    judge.api_key = os.getenv("JUDGE_API_KEY") or judge.api_key
    judge.base_url = os.getenv("JUDGE_BASE_URL") or judge.base_url

    threshold = _positive_int("COMPACT_THRESHOLD_TOKENS", 1000)
    keep_messages = _positive_int("COMPACT_KEEP_MESSAGES", 4)
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=threshold,
        compact_keep_messages=keep_messages,
        model=model,
        judge_model=judge,
    )
