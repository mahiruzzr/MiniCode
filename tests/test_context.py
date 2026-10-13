"""Tests for ContextBuilder message assembly."""

from pathlib import Path

import pytest

from agent.context import ContextBuilder

BASE_START = "You are MiniCode, a coding agent working in a workspace directory."


def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    return home


def test_build_system_prompt_returns_string(tmp_path: Path) -> None:
    """System prompt builds to a non-empty string."""
    prompt = ContextBuilder(tmp_path).build_system_prompt()
    assert isinstance(prompt, str)
    assert prompt.strip()


def test_default_system_prompt_enters_system_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Base prompt becomes the system message content start."""
    _isolated_home(tmp_path, monkeypatch)
    builder = ContextBuilder(tmp_path)
    messages = builder.build_messages(history=[], current_message="hi")
    assert messages[0] == {"role": "system", "content": builder.build_system_prompt()}
    assert messages[0]["content"].startswith(BASE_START)


def test_memory_preamble_appended_after_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MEMORY.md content lands after the base prompt."""
    _isolated_home(tmp_path, monkeypatch)
    mem = tmp_path / ".minicode" / "memory"
    mem.mkdir(parents=True)
    (mem / "MEMORY.md").write_text("durable fact", encoding="utf-8")
    prompt = ContextBuilder(tmp_path).build_system_prompt()
    assert prompt.startswith(BASE_START)
    assert "durable fact" in prompt
    assert prompt.index(BASE_START) < prompt.index("durable fact")


def test_memory_snapshot_frozen_after_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MEMORY.md edits after construction do not change the prompt."""
    _isolated_home(tmp_path, monkeypatch)
    mem = tmp_path / ".minicode" / "memory"
    mem.mkdir(parents=True)
    memory_file = mem / "MEMORY.md"
    memory_file.write_text("fact-v1", encoding="utf-8")
    builder = ContextBuilder(tmp_path)
    assert "fact-v1" in builder.build_system_prompt()
    memory_file.write_text("fact-v2", encoding="utf-8")
    rebuilt = builder.build_system_prompt()
    assert "fact-v1" in rebuilt
    assert "fact-v2" not in rebuilt


def test_instruction_snapshot_frozen_after_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AGENTS.md and user-global edits after construction are ignored."""
    home = _isolated_home(tmp_path, monkeypatch)
    agents_file = tmp_path / "AGENTS.md"
    agents_file.write_text("rules-v1", encoding="utf-8")
    user_dir = home / ".minicode"
    user_dir.mkdir()
    global_file = user_dir / "AGENTS.md"
    global_file.write_text("global-v1", encoding="utf-8")
    builder = ContextBuilder(tmp_path)
    assert "rules-v1" in builder.build_system_prompt()
    assert "global-v1" in builder.build_system_prompt()
    agents_file.write_text("rules-v2", encoding="utf-8")
    global_file.write_text("global-v2", encoding="utf-8")
    rebuilt = builder.build_system_prompt()
    assert "rules-v1" in rebuilt
    assert "global-v1" in rebuilt
    assert "rules-v2" not in rebuilt
    assert "global-v2" not in rebuilt


def test_skills_snapshot_shared_with_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discovery happens once; prompt and tool share the registry."""
    _isolated_home(tmp_path, monkeypatch)
    skill_dir = tmp_path / ".minicode" / "skills" / "review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review code changes.\n---\n\nChecklist here.\n",
        encoding="utf-8",
    )
    builder = ContextBuilder(tmp_path)
    assert "Review code changes." in builder.build_system_prompt()
    assert "Checklist here." not in builder.build_system_prompt()
    late_dir = tmp_path / ".minicode" / "skills" / "late"
    late_dir.mkdir()
    (late_dir / "SKILL.md").write_text(
        "---\nname: late\ndescription: Late skill.\n---\n\nBody.\n",
        encoding="utf-8",
    )
    assert "Late skill." not in builder.build_system_prompt()
    assert builder.skills.get("review") is not None
    assert builder.skills.get("late") is None


def test_no_workspace_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The base prompt never leaks the workspace path."""
    _isolated_home(tmp_path, monkeypatch)
    assert str(tmp_path.resolve()) not in ContextBuilder(tmp_path).build_system_prompt()


def test_no_nonexistent_tool_names(tmp_path: Path) -> None:
    """Base prompt names no unimplemented tools or frameworks."""
    prompt = ContextBuilder(tmp_path).build_system_prompt()
    for banned in (
        "apply_patch",
        "update_plan",
        "ask_user",
        "spawn_agent",
        "web_search",
        "web_fetch",
    ):
        assert banned not in prompt


def test_build_messages_shape(tmp_path: Path) -> None:
    """Messages are system first, then history, then current user."""
    builder = ContextBuilder(tmp_path)
    history = [{"role": "user", "content": "old"}]
    messages = builder.build_messages(history=history, current_message="new")
    assert [m["role"] for m in messages] == ["system", "user", "user"]
    assert messages[0]["content"] == builder.build_system_prompt()
    assert messages[1] == {"role": "user", "content": "old"}
    assert messages[2] == {"role": "user", "content": "new"}


def test_build_messages_current_user_once(tmp_path: Path) -> None:
    """Current user message appears exactly once with correct content."""
    messages = ContextBuilder(tmp_path).build_messages(
        history=[], current_message="hello"
    )
    users = [m for m in messages if m["role"] == "user"]
    assert users == [{"role": "user", "content": "hello"}]


def test_add_assistant_message_with_tool_calls(tmp_path: Path) -> None:
    """Assistant message keeps content and tool calls."""
    tool_calls = [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "bash", "arguments": "{}"},
        }
    ]
    messages = ContextBuilder(tmp_path).add_assistant_message(
        [], "working", tool_calls
    )
    assert messages == [
        {"role": "assistant", "content": "working", "tool_calls": tool_calls}
    ]


def test_add_assistant_message_reasoning_content(tmp_path: Path) -> None:
    """Reasoning content is kept only when provided."""
    builder = ContextBuilder(tmp_path)
    with_reasoning = builder.add_assistant_message(
        [], "working", None, reasoning_content="thinking"
    )
    assert with_reasoning[0]["reasoning_content"] == "thinking"
    without_reasoning = builder.add_assistant_message([], "working")
    assert "reasoning_content" not in without_reasoning[0]
    assert "tool_calls" not in without_reasoning[0]


def test_add_tool_result_shape(tmp_path: Path) -> None:
    """Tool result carries id, name, and content under role tool."""
    messages = ContextBuilder(tmp_path).add_tool_result(
        [], "c1", "bash", "output"
    )
    assert messages == [
        {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "bash",
            "content": "output",
        }
    ]
