"""Tests for provider types, registry lookup, and the LiteLLM adapter."""

from collections.abc import AsyncIterator
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


def _stream_chunk(
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish: str | None = None,
    usage: Any = None,
    reasoning: str | None = None,
) -> SimpleNamespace:
    """Create a LiteLLM-style streaming chunk for tests."""
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    if reasoning is not None:
        delta.reasoning_content = reasoning
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish)],
        usage=usage,
    )


def _stream_tool_call(
    index: int,
    call_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> SimpleNamespace:
    """Create a partial tool call in a LiteLLM-style chunk."""
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_stream(
    monkeypatch: pytest.MonkeyPatch,
    chunks: list[Any],
) -> tuple[dict[str, Any], list[Any]]:
    """Mock acompletion with an async stream and record consumed chunks."""
    seen: dict[str, Any] = {}
    consumed: list[Any] = []

    async def stream() -> AsyncIterator[Any]:
        for chunk in chunks:
            consumed.append(chunk)
            yield chunk

    async def fake_call(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return stream()

    monkeypatch.setattr(litellm_module, "acompletion", fake_call)
    return seen, consumed


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


def test_sanitize_replaces_empty_string_by_role_and_tool_calls() -> None:
    """Empty string content becomes None for tool-call turns, "(empty)" otherwise."""
    cleaned = LLMProvider._sanitize_empty_content([
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "user", "content": ""},
    ])
    assert cleaned[0]["content"] is None
    assert cleaned[1]["content"] == "(empty)"


def test_sanitize_drops_empty_text_blocks_in_list_content() -> None:
    """Empty text-ish blocks are removed while real content survives."""
    cleaned = LLMProvider._sanitize_empty_content([
        {"role": "user", "content": [
            {"type": "text", "text": ""},
            {"type": "input_text", "text": None},
            {"type": "output_text", "text": ""},
            {"type": "text", "text": "keep me"},
        ]},
    ])
    assert cleaned[0]["content"] == [{"type": "text", "text": "keep me"}]


def test_sanitize_strips_meta_fields_but_keeps_content() -> None:
    """Internal _meta fields are removed; sibling fields stay untouched."""
    cleaned = LLMProvider._sanitize_empty_content([
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://x/y.png"}, "_meta": {"src": "paste"}},
            {"type": "text", "text": "caption"},
        ]},
    ])
    assert cleaned[0]["content"][0] == {
        "type": "image_url", "image_url": {"url": "https://x/y.png"},
    }
    assert cleaned[0]["content"][1] == {"type": "text", "text": "caption"}


def test_sanitize_empty_list_falls_back_by_role() -> None:
    """Lists emptied by cleanup follow the same None/"(empty)" rule as strings."""
    cleaned = LLMProvider._sanitize_empty_content([
        {"role": "assistant", "content": [{"type": "text", "text": ""}], "tool_calls": [{"id": "c1"}]},
        {"role": "user", "content": [{"type": "text", "text": ""}]},
    ])
    assert cleaned[0]["content"] is None
    assert cleaned[1]["content"] == "(empty)"


def test_sanitize_wraps_dict_content_in_list() -> None:
    """A bare dict content is normalized to a one-element list."""
    blob = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}
    cleaned = LLMProvider._sanitize_empty_content([{"role": "user", "content": blob}])
    assert cleaned[0]["content"] == [blob]


def test_sanitize_passes_unmodified_messages_through() -> None:
    """Messages that need no edits keep their content and identity."""
    messages = [
        {"role": "system", "content": "You are MiniCode."},
        {"role": "user", "content": "hello"},
        {"role": "user", "content": None},
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
    ]
    cleaned = LLMProvider._sanitize_empty_content(messages)
    assert cleaned == messages
    assert all(cleaned[i] is messages[i] for i in range(len(messages)))

    # Modified messages are shallow copies; the originals stay untouched.
    original = {"role": "user", "content": ""}
    LLMProvider._sanitize_empty_content([original])
    assert original["content"] == ""


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


@pytest.mark.asyncio
async def test_chat_stream_emits_text_deltas_and_returns_complete_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Text chunks stream in order and metadata is retained."""
    chunks = [
        _stream_chunk(content="Hel", reasoning="think"),
        _stream_chunk(content="lo", reasoning="ing"),
        _stream_chunk(
            content="!",
            finish="stop",
            usage=SimpleNamespace(
                prompt_tokens=2,
                completion_tokens=3,
                total_tokens=5,
            ),
        ),
    ]
    seen, consumed = _mock_stream(monkeypatch, chunks)
    deltas: list[str] = []

    async def on_delta(delta: str) -> None:
        deltas.append(delta)

    result = await LiteLLMProvider().chat_stream(
        [{"role": "user", "content": "hi"}],
        on_content_delta=on_delta,
    )

    assert seen["stream"] is True
    assert deltas == ["Hel", "lo", "!"]
    assert result.content == "Hello!"
    assert result.finish_reason == "stop"
    assert result.usage == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
    }
    assert result.reasoning_content == "thinking"
    assert consumed == chunks


@pytest.mark.asyncio
async def test_chat_stream_consumes_all_chunks_without_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing callback does not stop stream consumption."""
    chunks = [_stream_chunk(content="one"), _stream_chunk(content=" two")]
    seen, consumed = _mock_stream(monkeypatch, chunks)

    result = await LiteLLMProvider().chat_stream([{"role": "user", "content": "hi"}])

    assert seen["stream"] is True
    assert consumed == chunks
    assert result.content == "one two"


@pytest.mark.asyncio
async def test_chat_stream_merges_tool_call_fragments_in_index_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fragmented tool fields and arguments combine by call index."""
    chunks = [
        _stream_chunk(
            tool_calls=[
                _stream_tool_call(
                    1, call_id="call-b", name="bash", arguments='{"command":"'
                ),
                _stream_tool_call(
                    0, call_id="call-a", name="read_", arguments='{"path":'
                ),
            ]
        ),
        _stream_chunk(
            tool_calls=[
                _stream_tool_call(0, name="file", arguments='"a.py"}'),
                _stream_tool_call(1, name="_tool", arguments='echo hi"}'),
            ],
            finish="tool_calls",
        ),
    ]
    _mock_stream(monkeypatch, chunks)

    result = await LiteLLMProvider().chat_stream(
        [{"role": "user", "content": "read and run"}]
    )

    assert result.content is None
    assert result.finish_reason == "tool_calls"
    assert [(call.id, call.name, call.arguments) for call in result.tool_calls] == [
        ("call-a", "read_file", {"path": "a.py"}),
        ("call-b", "bash_tool", {"command": "echo hi"}),
    ]


@pytest.mark.asyncio
async def test_chat_stream_error_returns_error_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Iteration failures become the provider error response."""

    async def stream() -> AsyncIterator[Any]:
        yield _stream_chunk(content="partial")
        raise RuntimeError("stream broke")

    async def fake_call(**kwargs: Any) -> Any:
        return stream()

    monkeypatch.setattr(litellm_module, "acompletion", fake_call)
    result = await LiteLLMProvider().chat_stream([{"role": "user", "content": "hi"}])

    assert result.finish_reason == "error"
    assert result.content == "Error calling LLM: stream broke"


@pytest.mark.asyncio
async def test_chat_stream_callback_error_is_not_reported_as_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Callback failures surface as an error response."""
    _mock_stream(monkeypatch, [_stream_chunk(content="hello")])

    async def on_delta(delta: str) -> None:
        raise RuntimeError("callback failed")

    result = await LiteLLMProvider().chat_stream(
        [{"role": "user", "content": "hi"}],
        on_content_delta=on_delta,
    )

    assert result.finish_reason == "error"
    assert result.content == "Error calling LLM: callback failed"


@pytest.mark.asyncio
async def test_chat_stream_request_matches_chat_with_stream_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stream requests retain the non-streaming request options."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(litellm_module.litellm, "api_base", None)
    provider = LiteLLMProvider(
        api_key="sk-test",
        api_base="https://api.example.test",
        extra_headers={"X-Test": "yes"},
    )
    calls: list[dict[str, Any]] = []
    chunks = [_stream_chunk(content="ok", finish="stop")]

    async def stream() -> AsyncIterator[Any]:
        for chunk in chunks:
            yield chunk

    async def fake_call(**kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        if kwargs.get("stream"):
            return stream()
        return _response(content="ok")

    monkeypatch.setattr(litellm_module, "acompletion", fake_call)
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hi"},
    ]
    tools = [{"type": "function", "function": {"name": "bash"}}]
    request = {
        "messages": messages,
        "tools": tools,
        "model": "claude-sonnet-4",
        "max_tokens": 0,
        "temperature": 0.3,
        "reasoning_effort": "high",
        "tool_choice": "required",
    }

    await provider.chat_stream(**request)
    await provider.chat(**request)

    stream_kwargs = calls[0]
    assert stream_kwargs.pop("stream") is True
    assert stream_kwargs == calls[1]
    assert stream_kwargs["max_tokens"] == 1
    assert stream_kwargs["api_key"] == "sk-test"
    assert stream_kwargs["api_base"] == "https://api.example.test"
    assert stream_kwargs["extra_headers"] == {"X-Test": "yes"}
    assert stream_kwargs["reasoning_effort"] == "high"
    assert stream_kwargs["tool_choice"] == "required"
