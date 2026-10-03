from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Configuration shared by all six supported model providers."""

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Normalize known aliases and reject unknown providers early."""
    provider = value.strip().lower()
    aliases = {
        "anthorpic": "anthropic",
        "google": "gemini",
        "google-genai": "gemini",
        "openai-compatible": "custom",
    }
    provider = aliases.get(provider, provider)
    supported = ("openai", "custom", "gemini", "anthropic", "ollama", "openrouter")
    if provider not in supported:
        raise ValueError(f"Unknown provider {value!r}; expected one of: {', '.join(supported)}")
    return provider


def build_chat_model(config: ProviderConfig):
    """Build a live model lazily; offline callers need no provider SDK or key."""
    provider = normalize_provider(config.provider)
    if provider == "custom" and not config.base_url:
        raise ValueError("The custom provider requires CUSTOM_BASE_URL (or JUDGE_BASE_URL).")
    if provider == "gemini" and config.base_url:
        raise ValueError("Custom base_url is not supported for Gemini in this lab.")
    kwargs = {"model": config.model_name, "temperature": config.temperature}
    if provider != "ollama" and config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if provider == "custom" and not config.api_key:
        # Local OpenAI-compatible servers may not require authentication.
        kwargs["api_key"] = "not-needed"

    if provider in ("openai", "custom"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(**kwargs)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(**kwargs)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(**kwargs)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(**kwargs)
    from langchain_openrouter import ChatOpenRouter
    return ChatOpenRouter(**kwargs)
