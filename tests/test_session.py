"""Tests for Session persistence and history."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from session.manager import Session, SessionManager


def _manager(isolated_home) -> SessionManager:
    return SessionManager()


def test_create_and_get(isolated_home) -> None:
    """Sessions get unique ids and round-trip through the cache."""
    manager = _manager(isolated_home)
    first = manager.create_session()
    second = manager.create_session()
    assert first.session_id and first.session_id != second.session_id
    assert first.last_consolidated == 0
    assert manager.get_session(first.session_id) is first
    assert manager.get_session("missing") is None


def test_explicit_id_and_collision(isolated_home) -> None:
    """Explicit ids stick; collisions regenerate without overwrite."""
    manager = _manager(isolated_home)
    assert manager.create_session(session_id="fixed").session_id == "fixed"
    other = manager.create_session(session_id="fixed")
    assert other.session_id != "fixed"
    assert manager.get_session("fixed").session_id == "fixed"


def test_save_reload_roundtrip(isolated_home) -> None:
    """Messages, metadata, and consolidation offset survive disk reload."""
    manager = _manager(isolated_home)
    session = manager.create_session(metadata={"k": "v"})
    session.add_message("user", "old question")
    session.add_message("assistant", "old answer")
    session.add_message("user", "hi")
    session.last_consolidated = 2
    manager.save(session)

    reloaded = SessionManager().get_session(session.session_id)

    assert reloaded.metadata == {"k": "v"}
    assert reloaded.last_consolidated == 2
    assert [message["content"] for message in reloaded.messages] == [
        "old question",
        "old answer",
        "hi",
    ]
    assert reloaded.get_history() == [{"role": "user", "content": "hi"}]


def test_history_keeps_tool_metadata(isolated_home) -> None:
    """Tool-call fields replay; internal timestamps do not."""
    session = Session(session_id="hist")
    session.add_message("user", "go")
    session.messages.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1"}],
            "timestamp": "t",
        }
    )
    session.messages.append(
        {
            "role": "tool",
            "content": "out",
            "tool_call_id": "c1",
            "name": "bash",
            "timestamp": "t",
        }
    )
    history = session.get_history()
    assert history[1]["tool_calls"] == [{"id": "c1"}]
    assert history[2]["tool_call_id"] == "c1"
    assert history[2]["name"] == "bash"
    assert all("timestamp" not in m for m in history)


def test_delete_and_list(isolated_home) -> None:
    """Delete removes disk and cache entries."""
    manager = _manager(isolated_home)
    session = manager.create_session()
    assert any(s["session_id"] == session.session_id for s in manager.list_sessions())
    assert manager.delete(session.session_id) is True
    assert manager.delete(session.session_id) is False
    assert manager.list_sessions() == []


def _assistant(call_ids: tuple[str, ...]) -> dict[str, Any]:
    """Build an assistant message declaring the given tool calls."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "bash", "arguments": "{}"},
            }
            for call_id in call_ids
        ],
    }


def _result(call_id: str) -> dict[str, Any]:
    """Build the tool result answering one tool call."""
    return {
        "role": "tool",
        "content": f"out-{call_id}",
        "tool_call_id": call_id,
        "name": "bash",
    }


def _assert_window_paired(window: list[dict[str, Any]]) -> None:
    """Assert every result is declared and every declaration is answered."""
    declared: set[str] = set()
    answered: set[str] = set()
    for msg in window:
        if msg["role"] == "assistant":
            declared.update(tc["id"] for tc in msg.get("tool_calls", []))
        elif msg["role"] == "tool":
            assert msg["tool_call_id"] in declared
            answered.add(msg["tool_call_id"])
    assert answered == declared


@pytest.mark.parametrize(
    ("total", "limit", "first_index", "expected_length"),
    [(10, 10, 0, 10), (5, 10, 0, 5), (25, 10, 16, 9)],
    ids=["at-limit", "under-limit", "starts-on-assistant"],
)
def test_history_without_tool_calls_starts_at_first_user(
    isolated_home: Path, total: int, limit: int, first_index: int, expected_length: int
) -> None:
    """A clipped plain history begins at its first user message when available."""
    session = Session(session_id="plain")
    for i in range(total):
        session.add_message("user" if i % 2 == 0 else "assistant", f"m{i}")

    history = session.get_history(max_messages=limit)

    assert len(history) == expected_length
    assert [m["content"] for m in history] == [
        f"m{i}" for i in range(first_index, total)
    ]
    assert history[0]["role"] == "user"
    assert all("tool_calls" not in message for message in history)


def test_history_boundary_between_normal_messages(isolated_home) -> None:
    """A boundary on a normal message keeps the exact window size."""
    session = Session(session_id="clean-boundary")
    for i in range(4):
        session.messages.append({"role": "user", "content": f"u{i}"})
    session.messages.append(_assistant(("a1",)))
    session.messages.append(_result("a1"))
    for i in range(4):
        session.messages.append({"role": "user", "content": f"tail{i}"})

    history = session.get_history(max_messages=4)

    assert len(history) == 4
    assert [m["content"] for m in history] == ["tail0", "tail1", "tail2", "tail3"]
    _assert_window_paired(history)


def test_history_boundary_inside_single_tool_group(isolated_home: Path) -> None:
    """A clipped history drops a result whose tool call is outside the window."""
    session = Session(session_id="single-group")
    session.messages.append(_assistant(("c1",)))
    session.messages.append(_result("c1"))
    session.messages.append({"role": "assistant", "content": "done"})

    history = session.get_history(max_messages=2)

    assert history == [{"role": "assistant", "content": "done"}]


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_history_boundary_inside_parallel_group(
    isolated_home: Path, limit: int
) -> None:
    """A window containing only parallel results drops every orphan result."""
    session = Session(session_id="parallel")
    session.messages.append({"role": "user", "content": "u0"})
    session.messages.append(_assistant(("p1", "p2", "p3")))
    session.messages.append(_result("p1"))
    session.messages.append(_result("p2"))
    session.messages.append(_result("p3"))

    history = session.get_history(max_messages=limit)

    assert history == []


def test_history_parallel_group_intact_when_calls_included(isolated_home) -> None:
    """A window that reaches back to the assistant keeps the group paired."""
    session = Session(session_id="parallel-4")
    session.messages.append({"role": "user", "content": "u0"})
    session.messages.append(_assistant(("p1", "p2", "p3")))
    session.messages.append(_result("p1"))
    session.messages.append(_result("p2"))
    session.messages.append(_result("p3"))

    history = session.get_history(max_messages=4)

    assert history[0]["role"] == "assistant"
    assert [tc["id"] for tc in history[0]["tool_calls"]] == ["p1", "p2", "p3"]
    assert [m["tool_call_id"] for m in history[1:]] == ["p1", "p2", "p3"]
    _assert_window_paired(history)


@pytest.mark.parametrize(
    ("limit", "expected_roles", "expected_tool_ids"),
    [
        (
            9,
            [
                "user",
                "assistant",
                "tool",
                "assistant",
                "tool",
                "assistant",
                "tool",
                "tool",
                "assistant",
            ],
            ["r0", "r1", "r2a", "r2b"],
        ),
        (
            8,
            [
                "assistant",
                "tool",
                "assistant",
                "tool",
                "assistant",
                "tool",
                "tool",
                "assistant",
            ],
            ["r0", "r1", "r2a", "r2b"],
        ),
        (
            7,
            ["assistant", "tool", "assistant", "tool", "tool", "assistant"],
            ["r1", "r2a", "r2b"],
        ),
        (
            6,
            ["assistant", "tool", "assistant", "tool", "tool", "assistant"],
            ["r1", "r2a", "r2b"],
        ),
        (5, ["assistant", "tool", "tool", "assistant"], ["r2a", "r2b"]),
    ],
)
def test_history_multiple_tool_rounds_keep_legal_boundaries(
    isolated_home: Path,
    limit: int,
    expected_roles: list[str],
    expected_tool_ids: list[str],
) -> None:
    """Clipping removes orphan results and preserves later tool exchanges."""
    session = Session(session_id="rounds")
    for i in range(4):
        session.messages.append({"role": "user", "content": f"u{i}"})
    session.messages.append(_assistant(("r0",)))
    session.messages.append(_result("r0"))
    session.messages.append(_assistant(("r1",)))
    session.messages.append(_result("r1"))
    session.messages.append(_assistant(("r2a", "r2b")))
    session.messages.append(_result("r2a"))
    session.messages.append(_result("r2b"))
    session.messages.append({"role": "assistant", "content": "done"})

    history = session.get_history(max_messages=limit)

    assert [message["role"] for message in history] == expected_roles
    declared_ids = [
        call["id"] for message in history for call in message.get("tool_calls", [])
    ]
    result_ids = [
        message["tool_call_id"] for message in history if message["role"] == "tool"
    ]
    assert declared_ids == expected_tool_ids
    assert result_ids == expected_tool_ids
    _assert_window_paired(history)


@pytest.mark.parametrize(
    ("limit", "expected_contents"),
    [
        (6, ["u0", "", "out-c1", "out-c2", "done", "u5"]),
        (10, ["u0", "", "out-c1", "out-c2", "done", "u5"]),
        (5, ["u5"]),
    ],
    ids=["at-limit", "under-limit", "starts-after-user"],
)
def test_history_at_under_and_over_limit(
    isolated_home: Path, limit: int, expected_contents: list[str]
) -> None:
    """The requested limit clips history, then aligns it to a user turn."""
    session = Session(session_id="limits")
    session.messages.append({"role": "user", "content": "u0"})
    session.messages.append(_assistant(("c1", "c2")))
    session.messages.append(_result("c1"))
    session.messages.append(_result("c2"))
    session.messages.append({"role": "assistant", "content": "done"})
    session.messages.append({"role": "user", "content": "u5"})

    history = session.get_history(max_messages=limit)

    assert [message["content"] for message in history] == expected_contents
    _assert_window_paired(history)


def test_history_default_window_aligns_tool_call_boundary(
    isolated_home: Path,
) -> None:
    """The default 500-message window drops its leading orphan tool result."""
    session = Session(session_id="scale")
    for i in range(299):
        session.messages.append(_assistant((f"c{i}",)))
        session.messages.append(_result(f"c{i}"))
    session.messages.append({"role": "assistant", "content": "tail"})

    history = session.get_history()

    assert len(history) == 499
    assert history[0]["role"] == "assistant"
    assert history[0]["tool_calls"][0]["id"] == "c50"
    assert history[-1] == {"role": "assistant", "content": "tail"}
    result_ids = [
        message["tool_call_id"] for message in history if message["role"] == "tool"
    ]
    assert result_ids == [f"c{i}" for i in range(50, 299)]
    _assert_window_paired(history)


def test_history_uses_unconsolidated_messages_without_mutation(
    isolated_home: Path,
) -> None:
    """History starts at the consolidation offset without changing stored data."""
    session = Session(session_id="consolidated")
    session.messages = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "current question"},
        _assistant(("current-call",)),
        _result("current-call"),
        {"role": "assistant", "content": "current answer"},
    ]
    session.last_consolidated = 2
    original_messages = deepcopy(session.messages)

    history = session.get_history()

    assert [message["content"] for message in history] == [
        "current question",
        "",
        "out-current-call",
        "current answer",
    ]
    _assert_window_paired(history)
    assert session.messages == original_messages
    assert session.last_consolidated == 2


def test_clear_resets_messages_and_consolidation_offset(isolated_home: Path) -> None:
    """Clearing a session resets both its messages and history offset."""
    session = Session(session_id="clear")
    session.messages = [{"role": "user", "content": "old"}]
    session.last_consolidated = 1

    session.clear()

    assert session.messages == []
    assert session.last_consolidated == 0
