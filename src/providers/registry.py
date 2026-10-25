"""Provider metadata registry for MiniCode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic.alias_generators import to_snake


@dataclass(frozen=True)
class ProviderSpec:
    """Metadata describing an LLM provider."""

    name: str
    keywords: tuple[str, ...] = ()
    env_key: str = ""
    display_name: str = ""
    litellm_prefix: str = ""
    skip_prefixes: tuple[str, ...] = ()
    env_extras: tuple[tuple[str, str], ...] = ()
    is_gateway: bool = False
    is_local: bool = False
    detect_by_key_prefix: str = ""
    detect_by_base_keyword: str = ""
    default_api_base: str = ""
    strip_model_prefix: bool = False
    model_overrides: tuple[tuple[str, dict[str, Any]], ...] = ()
    supports_prompt_caching: bool = False

    @property
    def label(self) -> str:
        """Return the display label for the provider."""
        return self.display_name or self.name.title()


PROVIDERS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        name="openai",
        keywords=("openai", "gpt"),
        env_key="OPENAI_API_KEY",
        display_name="OpenAI",
        litellm_prefix="",
        skip_prefixes=(),
        env_extras=(),
        is_gateway=False,
        is_local=False,
        detect_by_key_prefix="",
        detect_by_base_keyword="",
        default_api_base="",
        strip_model_prefix=False,
        model_overrides=(),
        supports_prompt_caching=False,
    ),
    ProviderSpec(
        name="anthropic",
        keywords=("anthropic", "claude"),
        env_key="ANTHROPIC_API_KEY",
        display_name="Anthropic",
        litellm_prefix="",
        skip_prefixes=(),
        env_extras=(),
        is_gateway=False,
        is_local=False,
        detect_by_key_prefix="",
        detect_by_base_keyword="",
        default_api_base="",
        strip_model_prefix=False,
        model_overrides=(),
        supports_prompt_caching=True,
    ),
)


def find_by_name(name: str) -> ProviderSpec | None:
    """Find a provider by its canonical name.

    Args:
        name: Provider name.

    Returns:
        The matching provider specification, or None if not found.
    """
    normalized = to_snake(name.replace("-", "_"))
    for spec in PROVIDERS:
        if spec.name == normalized:
            return spec
    return None


def find_by_model(model: str) -> ProviderSpec | None:
    """Match a provider by model name.

    Args:
        model: Model identifier.

    Returns:
        The first matching provider specification, or None.
    """
    model_lower = model.lower()
    for spec in PROVIDERS:
        if spec.is_gateway or spec.is_local:
            continue
        if any(kw in model_lower for kw in spec.keywords):
            return spec
    return None


def find_gateway(
    provider_name: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> ProviderSpec | None:
    """Detect a gateway or local provider.

    Args:
        provider_name: Explicit configured provider name.
        api_key: Provider API key.
        api_base: Provider API base URL.

    Returns:
        The detected gateway/local provider, or None.
    """
    # 1. Direct match by config key
    if provider_name:
        spec = find_by_name(provider_name)
        if spec and (spec.is_gateway or spec.is_local):
            return spec

    # 2. Auto-detect by api_key prefix / api_base keyword
    for spec in PROVIDERS:
        if spec.detect_by_key_prefix and api_key and api_key.startswith(spec.detect_by_key_prefix):
            return spec
        if spec.detect_by_base_keyword and api_base and spec.detect_by_base_keyword in api_base:
            return spec

    return None
