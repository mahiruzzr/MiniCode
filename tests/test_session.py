"""Tests for Session persistence and history."""

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
