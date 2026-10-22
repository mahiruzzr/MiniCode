"""Tests for Micro-compact: stale tool results are omitted at send time.

Pure-function tests exercise ``AgentLoop._microcompact`` directly. Integration
tests assert the exact payloads a FakeProvider receives during real runs, and
that sessions keep the raw tool outputs on disk.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from agent.loop import (
    _COMPACTABLE_TOOLS,
    _MICROCOMPACT_KEEP_RECENT,
    _MICROCOMPACT_MIN_CHARS,
    AgentLoop,
)
from providers.base import LLMResponse
from providers.litellm import LiteLLMProvider
from session.manager import SessionManager
from test_agent import (
    _append_tool_exchange,
    _assert_tool_calls_paired,
    _call,
    _runner,
)

_FULL = "x" * 600
_BIG_CMD = "printf '%0600d' 0"
_BIG_OUT = "0" * 600


def _placeholder(name: str) -> str:
    return f"[{name} result omitted from context]"


def _tool_message(name: str, content: Any, call_id: str = "c0", **extra: Any) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": content,
    }
    message.update(extra)
    return message


def _recent_results(count: int = _MICROCOMPACT_KEEP_RECENT) -> list[dict[str, Any]]:
    return [_tool_message("bash", _FULL, call_id=f"r{i}") for i in range(count)]


def _exchange(call_id: str, name: str, content: Any) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": "{}"},
                }
            ],
        },
        _tool_message(name, content, call_id=call_id),
    ]


def _bash_rounds(rounds: int) -> list[LLMResponse]:
    return [
        LLMResponse(
            content="",
            tool_calls=[_call(f"c{i}", "bash", {"command": _BIG_CMD})],
        )
        for i in range(rounds)
    ]


@pytest.mark.parametrize("total", [0, 1, 9, 10, 11, 20])
def test_keep_count_boundaries(total: int) -> None:
    """Only whitelist results older than the newest ten are compacted."""
    messages = [_tool_message("bash", _FULL, call_id=f"c{i}") for i in range(total)]
    result = AgentLoop._microcompact(messages)
    stale = max(0, total - _MICROCOMPACT_KEEP_RECENT)
    assert len(result) == total
    for i, message in enumerate(result):
        expected = _placeholder("bash") if i < stale else _FULL
        assert message["content"] == expected
        assert message["tool_call_id"] == f"c{i}"
        assert message["name"] == "bash"


def test_keep_count_includes_short_and_non_string_results() -> None:
    """Short and non-string results still occupy keep slots in the count."""
    with_short = [_tool_message("bash", "y" * 600)] + [
        _tool_message("bash", "short", call_id=f"s{i}") for i in range(10)
    ]
    result = AgentLoop._microcompact(with_short)
    assert result[0]["content"] == _placeholder("bash")
    assert [m["content"] for m in result[1:]] == ["short"] * 10

    with_non_string = [
        _tool_message("bash", "y" * 600),
        _tool_message("bash", 12345),
    ] + [_tool_message("bash", "short", call_id=f"t{i}") for i in range(9)]
    result = AgentLoop._microcompact(with_non_string)
    assert result[0]["content"] == _placeholder("bash")
    assert result[1]["content"] == 12345


@pytest.mark.parametrize(
    ("length", "compacted"),
    [(0, False), (1, False), (499, False), (500, True), (501, True), (100_000, True)],
)
def test_min_chars_boundary(length: int, compacted: bool) -> None:
    """499 characters stay; exactly 500 is the first compacted length."""
    target = "a" * length
    messages = [_tool_message("read", target)] + _recent_results()
    result = AgentLoop._microcompact(messages)
    expected = _placeholder("read") if compacted else target
    assert result[0]["content"] == expected


@pytest.mark.parametrize(
    "content",
    ["", 12345, None, ["a"], {"k": "v"}, ("t",)],
    ids=["empty-string", "int", "none", "list", "dict", "tuple"],
)
def test_contents_that_are_not_long_strings_are_skipped(content: Any) -> None:
    """Non-string and empty contents are left exactly as they are."""
    messages = [_tool_message("grep", content)] + _recent_results()
    snapshot = copy.deepcopy(messages)
    result = AgentLoop._microcompact(messages)
    assert result[0]["content"] == content
    assert result[0] == snapshot[0]
    assert result is messages


def test_missing_content_key_is_skipped() -> None:
    """A tool message without a content key is not touched or given one."""
    target: dict[str, Any] = {"role": "tool", "tool_call_id": "c0", "name": "glob"}
    messages = [target] + _recent_results()
    result = AgentLoop._microcompact(messages)
    assert "content" not in result[0]
    assert result is messages


@pytest.mark.parametrize("name", sorted(_COMPACTABLE_TOOLS))
def test_each_compactable_tool_is_compacted(name: str) -> None:
    """Every whitelisted tool gets its own named placeholder."""
    messages = [_tool_message(name, _FULL)] + _recent_results()
    result = AgentLoop._microcompact(messages)
    assert result[0]["content"] == _placeholder(name)


@pytest.mark.parametrize(
    "name", ["write", "edit", "apply_patch", "skill", "memory", "unknown-tool"]
)
def test_non_compactable_tools_are_untouched(name: str) -> None:
    """Results outside the whitelist keep their content even when it is long."""
    messages = [_tool_message(name, "z" * 600)] + _recent_results(11)
    result = AgentLoop._microcompact(messages)
    assert result[0]["content"] == "z" * 600
    assert result[1]["content"] == _placeholder("bash")
    assert all(m["content"] == _FULL for m in result[2:])


def test_compactable_tools_match_registered_tools(
    isolated_home: Path, tmp_path: Path
) -> None:
    """The whitelist is exactly the agreed subset of real registered tools."""
    runner, _ = _runner([], tmp_path)
    registered = {d["function"]["name"] for d in runner.tools.get_definitions()}
    assert registered == {
        "apply_patch",
        "bash",
        "edit",
        "glob",
        "grep",
        "memory",
        "read",
        "skill",
        "update_plan",
        "write",
    }
    assert _COMPACTABLE_TOOLS == frozenset({"bash", "read", "grep", "glob"})
    assert _COMPACTABLE_TOOLS <= registered


def test_only_content_of_compacted_messages_changes() -> None:
    """Compaction rewrites content and preserves every other field."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "thinking",
            "tool_calls": [
                {
                    "id": "c0",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                }
            ],
            "reasoning_content": "why",
        },
        _tool_message("bash", _FULL, call_id="c0", custom="kept", timestamp="t0"),
        _tool_message("read", "short", call_id="c1"),
    ] + _recent_results()
    snapshot = copy.deepcopy(messages)
    result = AgentLoop._microcompact(messages)
    for i, message in enumerate(result):
        assert message["role"] == snapshot[i]["role"]
        for key in (
            "tool_call_id",
            "name",
            "tool_calls",
            "reasoning_content",
            "custom",
            "timestamp",
        ):
            if key in snapshot[i]:
                assert message[key] == snapshot[i][key]
        if i == 3:
            assert message["content"] == _placeholder("bash")
            assert message["content"] != snapshot[i]["content"]
        else:
            assert message == snapshot[i]
    assert result[2]["tool_calls"][0]["id"] == "c0"
    assert result[2]["reasoning_content"] == "why"


def test_input_messages_are_never_mutated() -> None:
    """The caller's message list and dicts stay byte-for-byte identical."""
    messages = [
        _tool_message("read" if i % 2 == 0 else "bash", _FULL, call_id=f"c{i}")
        for i in range(12)
    ]
    snapshot = copy.deepcopy(messages)
    result = AgentLoop._microcompact(messages)
    assert messages == snapshot
    assert result is not messages
    assert result[0]["content"] == _placeholder("read")
    assert result[1]["content"] == _placeholder("bash")


def test_copy_on_write_semantics() -> None:
    """Unchanged inputs come back as the same object; compaction copies."""
    at_limit = [_tool_message("bash", _FULL, call_id=f"c{i}") for i in range(10)]
    assert AgentLoop._microcompact(at_limit) is at_limit

    short_stale = [_tool_message("bash", "short", call_id=f"s{i}") for i in range(11)]
    assert AgentLoop._microcompact(short_stale) is short_stale

    compactable = [_tool_message("bash", _FULL, call_id=f"c{i}") for i in range(11)]
    result = AgentLoop._microcompact(compactable)
    assert result is not compactable
    assert compactable[0]["content"] == _FULL
    assert result[0]["content"] == _placeholder("bash")


def test_repeated_compaction_is_stable_and_does_not_rewrap() -> None:
    """Re-running compaction is deterministic and never nests placeholders."""
    messages = [_tool_message("bash", _FULL)] + _recent_results()
    first = AgentLoop._microcompact(messages)
    second = AgentLoop._microcompact(messages)
    assert first == second

    again = AgentLoop._microcompact(first)
    assert again is first
    assert again[0]["content"] == _placeholder("bash")
    assert again[0]["content"].count("omitted from context") == 1
    assert messages[0]["content"] == _FULL


def test_mixed_message_roles_and_tools() -> None:
    """Only stale, long, whitelisted results change in a mixed list."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c3",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        },
        _tool_message("read", "r" * 600, call_id="c3"),
        _tool_message("write", "w" * 600, call_id="c4"),
        _tool_message("grep", "abc", call_id="c5"),
        _tool_message("bash", 42, call_id="c6"),
    ] + [_tool_message("glob", "g" * 600, call_id=f"r{i}") for i in range(10)]
    snapshot = copy.deepcopy(messages)
    result = AgentLoop._microcompact(messages)
    assert len(result) == len(messages)
    assert result[3]["content"] == _placeholder("read")
    assert result[4]["content"] == "w" * 600
    assert result[5]["content"] == "abc"
    assert result[6]["content"] == 42
    assert [m["content"] for m in result[7:]] == ["g" * 600] * 10
    for i in (0, 1, 2, 4, 5, 6):
        assert result[i] == snapshot[i]


@pytest.mark.parametrize(
    ("content", "compacted"),
    [
        ("字" * 499, False),
        ("字" * 500, True),
        ("😀" * 499, False),
        ("😀" * 500, True),
        ("x" * 499 + "\n", True),
        ("\n" * 600, True),
        (" " * 600, True),
        ("héllo wörld 你好 مرحبا שלום\n\t" * 30, True),
        ("\x1b[31mred\x1b[0m" * 50, True),
    ],
    ids=[
        "cjk-499",
        "cjk-500",
        "emoji-499",
        "emoji-500",
        "newline-500",
        "newlines",
        "spaces",
        "multilingual",
        "ansi",
    ],
)
def test_unicode_and_special_contents(content: str, compacted: bool) -> None:
    """Length is measured in characters; unicode is handled verbatim."""
    messages = [_tool_message("bash", content)] + _recent_results()
    result = AgentLoop._microcompact(messages)
    expected = _placeholder("bash") if compacted else content
    assert result[0]["content"] == expected


def test_single_call_structure_preserved() -> None:
    """A compacted single result keeps its id, name, and pairing."""
    messages = _exchange("c1", "bash", "z" * 600)
    messages += [msg for i in range(10) for msg in _exchange(f"r{i}", "bash", _FULL)]
    snapshot = copy.deepcopy(messages)
    result = AgentLoop._microcompact(messages)
    assert len(result) == len(messages)
    assert result[0] == snapshot[0]
    assert result[1]["content"] == _placeholder("bash")
    assert result[1]["tool_call_id"] == "c1"
    assert result[1]["name"] == "bash"
    assert all(m["content"] == _FULL for m in result[3::2])
    _assert_tool_calls_paired(result)
    assert result[2] == snapshot[2]


def test_parallel_calls_structure_preserved() -> None:
    """Sibling results of one assistant message compact without losing ids."""
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                },
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "grep", "arguments": "{}"},
                },
            ],
        },
        _tool_message("bash", "a" * 600, call_id="c1"),
        _tool_message("grep", "b" * 600, call_id="c2"),
    ]
    messages += [msg for i in range(10) for msg in _exchange(f"r{i}", "bash", _FULL)]
    snapshot = copy.deepcopy(messages)
    result = AgentLoop._microcompact(messages)
    assert len(result) == len(messages)
    assert result[0] == snapshot[0]
    assert result[1]["content"] == _placeholder("bash")
    assert result[2]["content"] == _placeholder("grep")
    assert result[1]["tool_call_id"] == "c1"
    assert result[2]["tool_call_id"] == "c2"
    _assert_tool_calls_paired(result)


def test_consecutive_rounds_structure_preserved() -> None:
    """Across twelve rounds, only the two oldest results are rewritten."""
    messages = [msg for i in range(12) for msg in _exchange(f"c{i}", "bash", _FULL)]
    snapshot = copy.deepcopy(messages)
    result = AgentLoop._microcompact(messages)
    assert len(result) == len(messages)
    assert [m["content"] for m in result[1::2]] == (
        [_placeholder("bash")] * 2 + [_FULL] * 10
    )
    assert all(result[i] == snapshot[i] for i in range(0, len(result), 2))
    _assert_tool_calls_paired(result)


def test_malformed_histories_are_not_further_damaged() -> None:
    """Missing results, duplicate ids, and orphan results are left in place."""
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c0",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                },
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "grep", "arguments": "{}"},
                },
            ],
        },
        _tool_message("bash", "a" * 600, call_id="c0"),
        _tool_message("bash", "b" * 600, call_id="c0"),
        _tool_message("bash", "d" * 600, call_id="zzz"),
    ] + _recent_results()
    snapshot = copy.deepcopy(messages)
    result = AgentLoop._microcompact(messages)
    assert len(result) == len(messages)
    assert [m["tool_call_id"] for m in result[1:4]] == ["c0", "c0", "zzz"]
    assert all(m["content"] == _placeholder("bash") for m in result[1:4])
    assert messages == snapshot
    assert result[0] == snapshot[0]


def test_placeholder_text_is_named_nonempty_and_not_an_error() -> None:
    """Placeholders carry the tool name and cannot trip the Error heuristic."""
    for name in sorted(_COMPACTABLE_TOOLS):
        text = _placeholder(name)
        assert text
        assert name in text
        assert not text.lstrip().startswith("Error")
        assert len(text) < _MICROCOMPACT_MIN_CHARS
    assert _placeholder("read") == "[read result omitted from context]"


def test_provider_sanitizer_accepts_placeholder_messages() -> None:
    """The LiteLLM message whitelist keeps placeholder tool messages intact."""
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "read",
            "content": _placeholder("read"),
        },
    ]
    sanitized = LiteLLMProvider._sanitize_messages(messages)
    assert sanitized[0]["tool_calls"] == messages[0]["tool_calls"]
    assert sanitized[1]["content"] == _placeholder("read")
    assert sanitized[1]["tool_call_id"] == "c1"
    assert sanitized[1]["name"] == "read"


@pytest.mark.asyncio
async def test_first_request_keeps_results_within_keep_limit(
    isolated_home: Path, tmp_path: Path
) -> None:
    """Exactly ten results in history are sent verbatim."""
    runner, provider = _runner([LLMResponse(content="done")], tmp_path)
    session = runner.sessions.create_session()
    for i in range(_MICROCOMPACT_KEEP_RECENT):
        _append_tool_exchange(session, f"c{i}", "bash", _BIG_OUT)
    runner.sessions.save(session)

    assert await runner.run("go", session_id=session.session_id) == "done"

    tools = [m for m in provider.seen[0] if m.get("role") == "tool"]
    assert [t["content"] for t in tools] == [_BIG_OUT] * _MICROCOMPACT_KEEP_RECENT


@pytest.mark.asyncio
async def test_provider_sees_progressive_compaction_across_rounds(
    isolated_home: Path, tmp_path: Path
) -> None:
    """Every request compacts exactly the results older than the newest ten."""
    script = _bash_rounds(12) + [LLMResponse(content="done")]
    runner, provider = _runner(script, tmp_path)

    assert await runner.run("go", session_id="mc-sweep") == "done"

    assert len(provider.seen) == 13
    for k, request in enumerate(provider.seen):
        tools = [m for m in request if m.get("role") == "tool"]
        assert len(tools) == k
        stale = max(0, k - _MICROCOMPACT_KEEP_RECENT)
        expected = [
            _placeholder("bash") if i < stale else _BIG_OUT for i in range(k)
        ]
        assert [t["content"] for t in tools] == expected
        _assert_tool_calls_paired(request)


@pytest.mark.asyncio
async def test_compaction_happens_at_send_time_not_tool_completion(
    isolated_home: Path, tmp_path: Path
) -> None:
    """Session history stays raw while every provider request is transformed."""
    script = _bash_rounds(12) + [LLMResponse(content="done")]
    runner, provider = _runner(script, tmp_path)
    observed: list[int] = []

    def on_result(name: str, output: str, is_error: bool) -> None:
        session = runner.sessions.get_session("mc-send")
        tools = [m for m in session.messages if m.get("role") == "tool"]
        observed.append(len(tools))
        assert all(m["content"] == _BIG_OUT for m in tools)

    reply = await runner.run("go", session_id="mc-send", on_tool_result=on_result)
    assert reply == "done"
    assert observed == list(range(12))

    last_tools = [m for m in provider.seen[-1] if m.get("role") == "tool"]
    assert last_tools[0]["content"] == _placeholder("bash")
    assert last_tools[1]["content"] == _placeholder("bash")

    raw = (runner.sessions.sessions_dir / "mc-send.jsonl").read_text(encoding="utf-8")
    assert _placeholder("bash") not in raw
    assert _BIG_OUT in raw


@pytest.mark.asyncio
async def test_parallel_tool_rounds_compact_correctly(
    isolated_home: Path, tmp_path: Path
) -> None:
    """Sibling calls in one assistant message survive compaction as pairs."""
    script = [
        LLMResponse(
            content="",
            tool_calls=[
                _call(f"c{i}a", "bash", {"command": _BIG_CMD}),
                _call(f"c{i}b", "bash", {"command": _BIG_CMD}),
            ],
        )
        for i in range(6)
    ]
    script.append(LLMResponse(content="done"))
    runner, provider = _runner(script, tmp_path)

    assert await runner.run("go", session_id="mc-parallel") == "done"

    last = provider.seen[-1]
    tools = [m for m in last if m.get("role") == "tool"]
    assert len(last) == 2 + 6 + 12
    assert len(tools) == 12
    assert [t["content"] for t in tools[:2]] == [_placeholder("bash")] * 2
    assert all(t["content"] == _BIG_OUT for t in tools[2:])
    assert {t["tool_call_id"] for t in tools} == {
        f"c{i}{side}" for i in range(6) for side in "ab"
    }
    _assert_tool_calls_paired(last)


@pytest.mark.asyncio
async def test_max_iterations_fallback_with_compaction(
    isolated_home: Path, tmp_path: Path
) -> None:
    """The iteration budget still ends the run while compaction is active."""
    runner, provider = _runner(_bash_rounds(12), tmp_path, max_iterations=12)

    result = await runner.run("go", session_id="mc-max")

    assert result == "I've completed processing but have no response to give."
    assert len(provider.seen) == 12
    tools = [m for m in provider.seen[-1] if m.get("role") == "tool"]
    assert len(tools) == 11
    assert tools[0]["content"] == _placeholder("bash")
    assert all(t["content"] == _BIG_OUT for t in tools[1:])

    loaded = SessionManager().get_session("mc-max")
    stored = [m for m in loaded.messages if m.get("role") == "tool"]
    assert len(stored) == 12
    assert all(m["content"] == _BIG_OUT for m in stored)


@pytest.mark.asyncio
async def test_reload_session_compacts_on_first_request(
    isolated_home: Path, tmp_path: Path
) -> None:
    """A reloaded session keeps raw results and compacts again on send."""
    runner1, _ = _runner(
        _bash_rounds(12) + [LLMResponse(content="first")], tmp_path
    )
    assert await runner1.run("go", session_id="mc-reload") == "first"

    runner2, provider2 = _runner([LLMResponse(content="second")], tmp_path)
    assert await runner2.run("next", session_id="mc-reload") == "second"

    tools = [m for m in provider2.seen[0] if m.get("role") == "tool"]
    assert len(tools) == 12
    assert tools[0]["content"] == _placeholder("bash")
    assert tools[1]["content"] == _placeholder("bash")
    assert all(t["content"] == _BIG_OUT for t in tools[2:])
    _assert_tool_calls_paired(provider2.seen[0])

    raw = (runner2.sessions.sessions_dir / "mc-reload.jsonl").read_text(
        encoding="utf-8"
    )
    assert _placeholder("bash") not in raw


@pytest.mark.asyncio
async def test_short_results_are_never_compacted_end_to_end(
    isolated_home: Path, tmp_path: Path
) -> None:
    """Fifteen short results: nothing compacted, request and session intact."""
    runner, provider = _runner([LLMResponse(content="done")], tmp_path)
    session = runner.sessions.create_session()
    for i in range(15):
        _append_tool_exchange(session, f"c{i}", "bash", "x")
    runner.sessions.save(session)

    assert await runner.run("go", session_id=session.session_id) == "done"

    tools = [m for m in provider.seen[0] if m.get("role") == "tool"]
    assert [t["content"] for t in tools] == ["x"] * 15
    loaded = SessionManager().get_session(session.session_id)
    stored = [m for m in loaded.messages if m.get("role") == "tool"]
    assert [m["content"] for m in stored] == ["x"] * 15
