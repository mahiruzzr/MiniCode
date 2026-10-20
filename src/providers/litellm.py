"""LiteLLM-backed LLM provider for MiniCode."""

import json
import os
from typing import Any

import litellm
from litellm import acompletion

from .base import LLMProvider, LLMResponse, ToolCallRequest
from .registry import find_by_model, find_gateway


# Standard OpenAI chat-completion message keys; extras (e.g. reasoning_content) are stripped for strict providers.
# thinking-enabled models (Kimi k2.5, DeepSeek-R1, etc.).
_ALLOWED_MSG_KEYS = frozenset({"role", "content", "tool_calls", "tool_call_id", "name", "reasoning_content", "thinking_blocks"})


class LiteLLMProvider(LLMProvider):
    """Execute LLM requests through LiteLLM.

    Translates between MiniCode's provider interface and LiteLLM:
    provider routing comes from the registry metadata, requests go
    out via LiteLLM, and responses are normalized back to
    MiniCode's LLMResponse.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "gpt-4o",
        extra_headers: dict[str, str] | None = None,
        provider_name: str | None = None,
    ) -> None:
        """Initialize the LiteLLM provider.

        Args:
            api_key: Provider API key, if any.
            api_base: Custom API base URL, if any.
            default_model: Model used when chat() gets none.
            extra_headers: Extra headers forwarded to LiteLLM.
            provider_name: Explicit provider name for gateway
                detection.
        """
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}
        self._gateway = find_gateway(provider_name, api_key, api_base)
        if api_key:
            self._setup_env(api_key, api_base, default_model)
        if api_base:
            litellm.api_base = api_base
        litellm.suppress_debug_info = True
        litellm.drop_params = True

    def _setup_env(
        self,
        api_key: str,
        api_base: str | None,
        model: str,
    ) -> None:
        """Expose credentials through environment variables.

        Args:
            api_key: Provider API key.
            api_base: Custom API base URL, if any.
            model: Default model used to detect the provider.
        """
        spec = self._gateway or find_by_model(model)
        if not spec:
            return
        if self._gateway:
            os.environ[spec.env_key] = api_key
        else:
            os.environ.setdefault(spec.env_key, api_key)
        effective_base = api_base or spec.default_api_base
        for env_name, template in spec.env_extras:
            resolved = template.replace("{api_key}", api_key)
            resolved = resolved.replace("{api_base}", effective_base)
            os.environ.setdefault(env_name, resolved)

    def _resolve_model(self, model: str) -> str:
        """Resolve model name by applying provider/gateway prefixes."""
        if self._gateway:
            # Gateway mode: apply gateway prefix, skip provider-specific prefixes
            prefix = self._gateway.litellm_prefix
            if self._gateway.strip_model_prefix:
                model = model.split("/")[-1]
            if prefix and not model.startswith(f"{prefix}/"):
                model = f"{prefix}/{model}"
            return model

        # Standard mode: auto-prefix for known providers
        spec = find_by_model(model)
        if spec and spec.litellm_prefix:
            model = self._canonicalize_explicit_prefix(model, spec.name, spec.litellm_prefix)
            if not any(model.startswith(s) for s in spec.skip_prefixes):
                model = f"{spec.litellm_prefix}/{model}"

        return model

    @staticmethod
    def _canonicalize_explicit_prefix(model: str, spec_name: str, canonical_prefix: str) -> str:
        """Normalize explicit provider prefixes like `github-copilot/...`."""
        if "/" not in model:
            return model
        prefix, remainder = model.split("/", 1)
        if prefix.lower().replace("-", "_") != spec_name:
            return model
        return f"{canonical_prefix}/{remainder}"

    def _apply_model_overrides(self, model: str, kwargs: dict[str, Any]) -> None:
        """Apply model-specific parameter overrides from the registry."""
        model_lower = model.lower()
        spec = find_by_model(model)
        if spec:
            for pattern, overrides in spec.model_overrides:
                if pattern in model_lower:
                    kwargs.update(overrides)
                    return

    @staticmethod
    def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip non-standard keys and ensure assistant messages have a content key."""
        sanitized = []
        for msg in messages:
            clean = {k: v for k, v in msg.items() if k in _ALLOWED_MSG_KEYS}
            # Strict providers require "content" even when assistant only has tool_calls
            if clean.get("role") == "assistant" and "content" not in clean:
                clean["content"] = None
            sanitized.append(clean)
        return sanitized

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Send a chat completion request via LiteLLM.

        Args:
            messages: Message dicts with role and content.
            tools: Optional tool definitions in OpenAI format.
            model: Model identifier, defaults to default_model.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.
            reasoning_effort: Reasoning effort hint forwarded to LiteLLM.
            tool_choice: Tool selection policy forwarded to LiteLLM.
                Defaults to "auto" when tools are provided.

        Returns:
            Normalized LLMResponse for the agent layer.
        """
        resolved = self._resolve_model(model or self.default_model)

        # Clamp max_tokens to at least 1 — negative or zero values cause
        # LiteLLM to reject the request with "max_tokens must be at least 1".
        max_tokens = max(1, max_tokens)

        kwargs: dict[str, Any] = {
            "model": resolved,
            "messages": self._sanitize_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        self._apply_model_overrides(resolved, kwargs)
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
            kwargs["drop_params"] = True
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        try:
            response = await acompletion(**kwargs)
            return self._parse_response(response)
        except Exception as error:
            return LLMResponse(
                content=f"Error calling LLM: {str(error)}",
                finish_reason="error",
            )

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse a LiteLLM response into a MiniCode response.

        Args:
            response: Raw response object returned by LiteLLM.

        Returns:
            Normalized LLMResponse for the agent layer.
        """
        choice = response.choices[0]
        message = choice.message
        tool_calls: list[ToolCallRequest] = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for call in message.tool_calls:
                arguments = call.function.arguments
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": arguments}
                tool_calls.append(
                    ToolCallRequest(
                        id=call.id,
                        name=call.function.name,
                        arguments=arguments,
                    )
                )
        usage: dict[str, int] = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        reasoning_content = getattr(message, "reasoning_content", None) or None
        thinking_blocks = getattr(message, "thinking_blocks", None) or None
        
        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
        )

    def get_default_model(self) -> str:
        """Get the provider's default model."""
        return self.default_model
