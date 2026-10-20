"""Pydantic configuration schema for MiniCode."""

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class AgentDefaults(BaseModel):
    """Default agent configuration."""

    workspace: str = "~/.minicode/workspace"
    model: str = "gpt-4o"
    max_tokens: int = 8192
    temperature: float = 0.7
    max_tool_iterations: int = 20
    # Reasoning effort forwarded to reasoning-capable models (None = unset).
    reasoning_effort: str | None = None  # low / medium / high — enables LLM thinking mode


class AgentsConfig(BaseModel):
    """Agent configuration."""

    defaults: AgentDefaults = Field(default_factory=AgentDefaults)


class ProviderConfig(BaseModel):
    """LLM provider configuration."""

    api_key: str = ""
    api_base: str | None = None
    extra_headers: dict[str, str] | None = None


class ProvidersConfig(BaseModel):
    """Configuration for LLM providers."""

    anthropic: ProviderConfig = Field(default_factory=ProviderConfig)
    openai: ProviderConfig = Field(default_factory=ProviderConfig)


class MiniCodeConfig(BaseSettings):
    """Root configuration for MiniCode."""

    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)

    @property
    def workspace_path(self) -> Path:
        """Get expanded workspace path."""
        return Path(self.agents.defaults.workspace).expanduser()

    def _match_provider(
        self, model: str | None = None
    ) -> tuple[ProviderConfig | None, str | None]:
        """Match provider config with its registry name.

        Args:
            model: Model identifier. Defaults to the configured
                default model.

        Returns:
            Tuple of matched ProviderConfig and registry provider
            name, or (None, None) when nothing matches.
        """
        from providers.registry import PROVIDERS

        model_lower = (model or self.agents.defaults.model).lower()
        for spec in PROVIDERS:
            config = getattr(self.providers, spec.name, None)
            if (
                config is not None
                and any(kw in model_lower for kw in spec.keywords)
                and config.api_key
            ):
                return config, spec.name
        return None, None

    def get_provider(
        self, model: str | None = None
    ) -> ProviderConfig | None:
        """Get the provider config matched by model.

        Args:
            model: Model identifier. Defaults to the configured
                default model.

        Returns:
            Matched ProviderConfig, or None when nothing matches.
        """
        config, _ = self._match_provider(model)
        return config

    def get_provider_name(self, model: str | None = None) -> str | None:
        """Get the registry name of the provider matched by model.

        Args:
            model: Model identifier. Defaults to the configured
                default model.

        Returns:
            Registry provider name, or None when nothing matches.
        """
        _, name = self._match_provider(model)
        return name

    def get_api_key(self, model: str | None = None) -> str | None:
        """Get the API key of the provider matched by model.

        Args:
            model: Model identifier. Defaults to the configured
                default model.

        Returns:
            API key, or None when nothing matches.
        """
        config = self.get_provider(model)
        return config.api_key if config else None

    def get_api_base(self, model: str | None = None) -> str | None:
        """Get the API base URL of the provider matched by model.

        Args:
            model: Model identifier. Defaults to the configured
                default model.

        Returns:
            Explicit api_base, registry default_api_base, or None.
        """
        from providers.registry import find_by_name

        config, name = self._match_provider(model)
        if config and config.api_base:
            return config.api_base
        if name:
            spec = find_by_name(name)
            if spec and spec.is_gateway and spec.default_api_base:
                return spec.default_api_base
        return None
