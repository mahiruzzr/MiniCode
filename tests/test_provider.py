"""Tests for provider types, registry lookup, and the LiteLLM adapter."""

from types import SimpleNamespace
from typing import Any

import pytest

import providers.litellm as litellm_module
from providers.base import LLMProvider, LLMResponse, ToolCallRequest
from providers.litellm import LiteLLMProvider
from providers.registry import (
    PROVIDERS,
    ProviderSpec,
    find_by_model,
    find_by_name,
    find_gateway,
)


def test_tool_call_request_fields() -> None:
    """Tool calls carry id, name, and arguments."""
    call = ToolCallRequest(id="c1", name="bash", arguments={"command": "ls"})
    assert (call.id, call.name, call.arguments) == ("c1", "bash", {"command": "ls"})


def test_has_tool_calls() -> None:
    """Presence flag follows the call list."""
    assert LLMResponse(content="hi").has_tool_calls is False
    response = LLMResponse(
        content=None, tool_calls=[ToolCallRequest(id="c1", name="bash", arguments={})]
    )
    assert response.has_tool_calls is True


def test_provider_is_abstract() -> None:
    """LLMProvider cannot run without chat and default model."""
    with pytest.raises(TypeError):
        LLMProvider()  # type: ignore[abstract]


def test_registry_holds_openai_and_anthropic() -> None:
    """Registry lists the two supported providers with labels."""
    assert [p.name for p in PROVIDERS] == ["openai", "anthropic"]
    assert PROVIDERS[0].label == "OpenAI"
    assert ProviderSpec(name="custom").label == "Custom"


def test_find_by_name_exact() -> None:
    """Canonical names match exactly."""
    assert find_by_name("openai").name == "openai"
    assert find_by_name("anthropic").name == "anthropic"
    assert find_by_name("unknown") is None


def test_find_by_model_keywords() -> None:
    """Model keywords route case-insensitively in registry order."""
    assert find_by_model("gpt-4o").name == "openai"
    assert find_by_model("GPT-4O").name == "openai"
    assert find_by_model("claude-sonnet-4").name == "anthropic"
    assert find_by_model("openai/gpt-4o").name == "openai"
    assert find_by_model("deepseek-chat") is None
    assert find_by_model("unknown-model-zzz") is None


def test_find_gateway_empty_registry() -> None:
    """No gateway or local provider is registered yet."""
    assert find_gateway() is None
    assert find_gateway(provider_name="openrouter") is None
    assert find_gateway(provider_name="openai") is None
    assert find_gateway(api_key="sk-or-abc") is None
    assert find_gateway(api_base="https://openrouter.ai/api/v1") is None


def _message(content: Any = "hi", tool_calls: Any = None, reasoning: Any = None) -> SimpleNamespace:
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    if reasoning is not None:
        msg.reasoning_content = reasoning
    return msg


def _response(
    content: Any = "hi",
    tool_calls: Any = None,
    finish: Any = "stop",
    usage: Any = None,
    reasoning: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish, message=_message(content, tool_calls, reasoning))],
        usage=usage,
    )


def _tool_call(call_id: str, name: str, arguments: Any) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=arguments)
    )


def test_litellm_defaults_and_interface() -> None:
    """Defaults match the documented contract."""
    provider = LiteLLMProvider()
    assert isinstance(provider, LLMProvider)
    assert provider.get_default_model() == "gpt-4o"
    assert provider.extra_headers == {}
    assert provider.api_key is None


def test_resolve_model_passthrough() -> None:
    """Empty prefixes keep model names untouched."""
    provider = LiteLLMProvider()
    assert provider._resolve_model("gpt-4o") == "gpt-4o"
    assert provider._resolve_model("claude-sonnet-4") == "claude-sonnet-4"
    assert provider._resolve_model("some-unknown-model") == "some-unknown-model"


def test_cache_control_supported_by_anthropic_only() -> None:
    """Explicit cache_control is enabled only for providers that support it."""
    provider = LiteLLMProvider()
    assert provider._supports_cache_control("claude-sonnet-4") is True
    assert provider._supports_cache_control("gpt-4o") is False
    assert provider._supports_cache_control("unknown-model-zzz") is False


def test_cache_control_supported_by_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gateway support follows the gateway's prompt caching flag."""
    spec = ProviderSpec(name="gateway", is_gateway=True, supports_prompt_caching=True)
    monkeypatch.setattr(litellm_module, "find_gateway", lambda *args, **kwargs: spec)
    provider = LiteLLMProvider()
    assert provider._supports_cache_control("gpt-4o") is True


def test_apply_cache_control_wraps_system_text_and_last_tool() -> None:
    """System text becomes a cached text block and only the last tool is marked."""
    provider = LiteLLMProvider()
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hi"},
    ]
    tools = [
        {"type": "function", "function": {"name": "bash"}},
        {"type": "function", "function": {"name": "read"}},
    ]
    new_messages, new_tools = provider._apply_cache_control(messages, tools)
    assert messages[0]["content"] == "rules"
    assert new_messages[0]["content"] == [
        {"type": "text", "text": "rules", "cache_control": {"type": "ephemeral"}}
    ]
    assert new_messages[1] == {"role": "user", "content": "hi"}
    assert new_tools is not None
    assert "cache_control" not in tools[-1]
    assert "cache_control" not in new_tools[0]
    assert new_tools[1]["cache_control"] == {"type": "ephemeral"}


def test_apply_cache_control_updates_last_system_block() -> None:
    """List content marks only the last system block and leaves the input alone."""
    provider = LiteLLMProvider()
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "one"},
                {"type": "text", "text": "two"},
            ],
        }
    ]
    new_messages, new_tools = provider._apply_cache_control(messages, None)
    assert new_tools is None
    assert new_messages[0]["content"] == [
        {"type": "text", "text": "one"},
        {"type": "text", "text": "two", "cache_control": {"type": "ephemeral"}},
    ]
    assert messages[0]["content"][1] == {"type": "text", "text": "two"}


def test_setup_env_uses_registry_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """API keys land in the registry-declared env var without clobbering."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    LiteLLMProvider(api_key="sk-test", default_model="gpt-4o")
    import os

    assert os.environ["OPENAI_API_KEY"] == "sk-test"


def test_parse_content_and_finish() -> None:
    """Content, finish reason, and empty usage map over."""
    provider = LiteLLMProvider()
    parsed = provider._parse_response(_response(content="hello", finish="stop"))
    assert isinstance(parsed, LLMResponse)
    assert parsed.content == "hello"
    assert parsed.tool_calls == []
    assert parsed.finish_reason == "stop"
    assert parsed.usage == {}
    assert parsed.reasoning_content is None


def test_parse_tool_calls() -> None:
    """JSON string arguments decode; broken JSON degrades gracefully."""
    provider = LiteLLMProvider()
    good = _tool_call("c1", "bash", '{"command": "ls"}')
    bad = _tool_call("c2", "bash", "{oops")
    parsed = provider._parse_response(
        _response(content=None, tool_calls=[good, bad], finish="tool_calls")
    )
    assert parsed.tool_calls[0].arguments == {"command": "ls"}
    assert parsed.tool_calls[1].arguments == {"raw": "{oops"}
    assert parsed.has_tool_calls is True


def test_parse_tool_calls_generates_short_alphanumeric_ids() -> None:
    """Parsed tool calls get fresh 9-character IDs and preserve other fields."""
    provider = LiteLLMProvider()
    parsed = provider._parse_response(
        _response(
            content=None,
            tool_calls=[_tool_call("provider-id", "bash", '{"command": "ls"}')],
            finish="tool_calls",
        )
    )
    call = parsed.tool_calls[0]
    assert call.id != "provider-id"
    assert len(call.id) == 9
    assert call.id.isascii()
    assert call.id.isalnum()
    assert call.name == "bash"
    assert call.arguments == {"command": "ls"}
    assert parsed.content is None
    assert parsed.finish_reason == "tool_calls"
    assert parsed.usage == {}


def test_parse_usage_and_reasoning() -> None:
    """Usage counters and reasoning content carry over."""
    provider = LiteLLMProvider()
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    parsed = provider._parse_response(_response(usage=usage, reasoning="because"))
    assert parsed.usage == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    assert parsed.reasoning_content == "because"


@pytest.mark.asyncio
async def test_chat_error_becomes_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Transport failures return an error LLMResponse instead of raising."""

    async def boom(**kwargs: Any) -> Any:
        raise RuntimeError("down")

    monkeypatch.setattr(litellm_module, "acompletion", boom)
    result = await LiteLLMProvider().chat([{"role": "user", "content": "hi"}])
    assert result.finish_reason == "error"
    assert result.content.startswith("Error calling LLM:")


@pytest.mark.asyncio
async def test_chat_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model, tools, and tool_choice reach LiteLLM intact."""
    seen: dict[str, Any] = {}

    async def fake_call(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return _response(content="ok")

    monkeypatch.setattr(litellm_module, "acompletion", fake_call)
    result = await LiteLLMProvider().chat(
        [{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "bash"}}],
        model="gpt-4o",
    )
    assert result.content == "ok"
    assert seen["model"] == "gpt-4o"
    assert seen["tool_choice"] == "auto"
    assert len(seen["tools"]) == 1


async def _capture_chat(
    monkeypatch: pytest.MonkeyPatch,
    **chat_kwargs: Any,
) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    async def fake_call(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return _response(content="ok")

    monkeypatch.setattr(litellm_module, "acompletion", fake_call)
    await LiteLLMProvider().chat(
        [{"role": "user", "content": "hi"}],
        model="gpt-4o",
        **chat_kwargs,
    )
    return seen


@pytest.mark.asyncio
async def test_chat_injects_cache_control_for_anthropic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anthropic requests carry cached system content and tool definitions."""
    seen: dict[str, Any] = {}

    async def fake_call(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return _response(content="ok")

    monkeypatch.setattr(litellm_module, "acompletion", fake_call)
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hi"},
    ]
    tools = [{"type": "function", "function": {"name": "bash"}}]
    result = await LiteLLMProvider().chat(
        messages,
        tools=tools,
        model="claude-sonnet-4",
    )
    assert result.content == "ok"
    assert seen["messages"][0]["content"] == [
        {"type": "text", "text": "rules", "cache_control": {"type": "ephemeral"}}
    ]
    assert seen["tools"][0]["cache_control"] == {"type": "ephemeral"}
    assert messages[0]["content"] == "rules"
    assert "cache_control" not in tools[0]


@pytest.mark.asyncio
async def test_chat_skips_cache_control_for_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI requests keep their original system content and tools."""
    seen: dict[str, Any] = {}

    async def fake_call(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return _response(content="ok")

    monkeypatch.setattr(litellm_module, "acompletion", fake_call)
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hi"},
    ]
    tools = [{"type": "function", "function": {"name": "bash"}}]
    result = await LiteLLMProvider().chat(messages, tools=tools, model="gpt-4o")
    assert result.content == "ok"
    assert seen["messages"][0]["content"] == "rules"
    assert "cache_control" not in seen["tools"][0]


@pytest.mark.asyncio
async def test_chat_tool_choice_none_keeps_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """None keeps the pre-existing auto policy when tools are sent."""
    seen = await _capture_chat(
        monkeypatch,
        tools=[{"type": "function", "function": {"name": "bash"}}],
    )
    assert seen["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_chat_tool_choice_string_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A string policy reaches acompletion untouched."""
    seen = await _capture_chat(
        monkeypatch,
        tools=[{"type": "function", "function": {"name": "bash"}}],
        tool_choice="required",
    )
    assert seen["tool_choice"] == "required"


@pytest.mark.asyncio
async def test_chat_tool_choice_dict_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dict policy reaches acompletion with every field intact."""
    choice = {"type": "function", "function": {"name": "bash"}}
    seen = await _capture_chat(
        monkeypatch,
        tools=[{"type": "function", "function": {"name": "bash"}}],
        tool_choice=choice,
    )
    assert seen["tool_choice"] == choice


@pytest.mark.asyncio
async def test_chat_tool_choice_ignored_without_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tool_choice is not sent when no tools are defined."""
    seen = await _capture_chat(monkeypatch, tool_choice="required")
    assert "tool_choice" not in seen
