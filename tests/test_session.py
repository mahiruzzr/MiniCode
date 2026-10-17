"""Tests for Session persistence and history."""

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
    """Messages and metadata survive disk reload."""
    manager = _manager(isolated_home)
    session = manager.create_session(metadata={"k": "v"})
    session.add_message("user", "hi")
    manager.save(session)
    reloaded = SessionManager().get_session(session.session_id)
    assert reloaded.metadata == {"k": "v"}
    assert [m["role"] for m in reloaded.messages] == ["user"]
    assert reloaded.messages[0]["content"] == "hi"


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
    ("total", "limit", "expected"),
    [(10, 10, 10), (5, 10, 5), (25, 10, 10)],
    ids=["at-limit", "under-limit", "over-limit"],
)
def test_history_without_tool_calls_keeps_window(
    isolated_home, total: int, limit: int, expected: int
) -> None:
    """A conversation without tool calls keeps the plain last-N window."""
    session = Session(session_id="plain")
    for i in range(total):
        session.add_message("user" if i % 2 == 0 else "assistant", f"m{i}")

    history = session.get_history(max_messages=limit)

    assert len(history) == expected
    assert [m["content"] for m in history] == [
        f"m{i}" for i in range(total - expected, total)
    ]
    assert all("tool_calls" not in m for m in history)


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


def test_history_boundary_inside_single_tool_group(isolated_home) -> None:
    """Plain clipping starts the window on the result without its call."""
    session = Session(session_id="single-group")
    session.messages.append({"role": "user", "content": "u0"})
    session.messages.append(_assistant(("c1",)))
    session.messages.append(_result("c1"))
    session.messages.append({"role": "user", "content": "u3"})

    history = session.get_history(max_messages=2)

    assert [m["role"] for m in history] == ["tool", "user"]
    assert [m["content"] for m in history] == ["out-c1", "u3"]


@pytest.mark.parametrize(
    ("limit", "roles", "contents"),
    [
        (1, ["tool"], ["out-p3"]),
        (2, ["tool", "tool"], ["out-p2", "out-p3"]),
        (3, ["tool", "tool", "tool"], ["out-p1", "out-p2", "out-p3"]),
    ],
)
def test_history_boundary_inside_parallel_group(
    isolated_home, limit: int, roles: list[str], contents: list[str]
) -> None:
    """Plain clipping may start anywhere, even inside a parallel group."""
    session = Session(session_id="parallel")
    session.messages.append({"role": "user", "content": "u0"})
    session.messages.append(_assistant(("p1", "p2", "p3")))
    session.messages.append(_result("p1"))
    session.messages.append(_result("p2"))
    session.messages.append(_result("p3"))

    history = session.get_history(max_messages=limit)

    assert [m["role"] for m in history] == roles
    assert [m["content"] for m in history] == contents


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
    ("limit", "roles", "first_call"),
    [
        (9, ["tool", "assistant", "tool", "assistant", "tool", "tool",
             "user", "user", "user"], "r0"),
        (8, ["assistant", "tool", "assistant", "tool", "tool",
             "user", "user", "user"], "r1"),
        (7, ["tool", "assistant", "tool", "tool", "user", "user", "user"],
         "r1"),
        (6, ["assistant", "tool", "tool", "user", "user", "user"], "r2a"),
        (5, ["tool", "tool", "user", "user", "user"], "r2a"),
    ],
)
def test_history_multiple_tool_rounds(
    isolated_home, limit: int, roles: list[str], first_call: str
) -> None:
    """Plain last-N clipping can start a window inside a tool round."""
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
    for i in range(3):
        session.messages.append({"role": "user", "content": f"tail{i}"})

    history = session.get_history(max_messages=limit)

    assert [m["role"] for m in history] == roles
    leading = history[0]
    if leading["role"] == "assistant":
        assert leading["tool_calls"][0]["id"] == first_call
    else:
        assert leading["tool_call_id"] == first_call


@pytest.mark.parametrize("limit", [8, 6])
def test_history_round_window_pairing(isolated_home, limit: int) -> None:
    """Windows anchored on an assistant call keep results answered."""
    session = Session(session_id=f"rounds-paired-{limit}")
    for i in range(4):
        session.messages.append({"role": "user", "content": f"u{i}"})
    session.messages.append(_assistant(("r0",)))
    session.messages.append(_result("r0"))
    session.messages.append(_assistant(("r1",)))
    session.messages.append(_result("r1"))
    session.messages.append(_assistant(("r2a", "r2b")))
    session.messages.append(_result("r2a"))
    session.messages.append(_result("r2b"))
    for i in range(3):
        session.messages.append({"role": "user", "content": f"tail{i}"})

    history = session.get_history(max_messages=limit)

    assert history[0]["role"] == "assistant"
    _assert_window_paired(history)


@pytest.mark.parametrize(
    ("limit", "expected_contents"),
    [
        (6, ["u0", "", "out-c1", "out-c2", "done", "u5"]),
        (10, ["u0", "", "out-c1", "out-c2", "done", "u5"]),
        (5, ["", "out-c1", "out-c2", "done", "u5"]),
    ],
    ids=["at-limit", "under-limit", "over-limit"],
)
def test_history_at_under_and_over_limit(
    isolated_home, limit: int, expected_contents: list[str]
) -> None:
    """Equal, smaller, and larger histories all clip safely."""
    session = Session(session_id="limits")
    session.messages.append({"role": "user", "content": "u0"})
    session.messages.append(_assistant(("c1", "c2")))
    session.messages.append(_result("c1"))
    session.messages.append(_result("c2"))
    session.messages.append({"role": "assistant", "content": "done"})
    session.messages.append({"role": "user", "content": "u5"})

    history = session.get_history(max_messages=limit)

    assert [m["content"] for m in history] == expected_contents
    _assert_window_paired(history)


def test_history_plain_truncation_window(isolated_home) -> None:
    """The default window is exactly last-50, even splitting a round."""
    session = Session(session_id="scale")
    for i in range(299):
        session.messages.append(_assistant((f"c{i}",)))
        session.messages.append(_result(f"c{i}"))
    session.messages.append({"role": "assistant", "content": "tail"})

    history = session.get_history()

    assert len(history) == 50
    assert history[0]["role"] == "tool"
    assert history[0]["tool_call_id"] == "c274"
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"] == "tail"
    tool_ids = [m["tool_call_id"] for m in history if m["role"] == "tool"]
    assert len(tool_ids) == 25
    assert tool_ids[0] == "c274"
    assert tool_ids[-1] == "c298"
    # c274's result enters the window without its declaring call: plain
    # last-N clipping can split a round (see defect report).
    declared = {
        tc["id"]
        for m in history
        if m["role"] == "assistant"
        for tc in m.get("tool_calls", [])
    }
    assert "c274" not in declared
