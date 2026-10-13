"""Tests for agent memory: instructions, MEMORY.md, and MemoryTool."""

from pathlib import Path

import pytest

from agent.context import ContextBuilder
from agent.memory import (
    MemoryTool,
    memory_dir,
    memory_index,
    project_instructions,
    system_preamble,
    user_global_instructions,
)


def test_memory_dir_uses_minicode_namespace(tmp_path: Path) -> None:
    """Memory lives under <workspace>/.minicode/memory."""
    assert memory_dir(tmp_path) == tmp_path / ".minicode" / "memory"


def test_project_instruction_discovery(tmp_path: Path) -> None:
    """AGENTS.md beats CLAUDE.md in one directory; DEEPCODE.md is ignored."""
    (tmp_path / "CLAUDE.md").write_text("claude", encoding="utf-8")
    (tmp_path / "DEEPCODE.md").write_text("deep", encoding="utf-8")
    assert "claude" in project_instructions(tmp_path)
    (tmp_path / "AGENTS.md").write_text("agents", encoding="utf-8")
    found = project_instructions(tmp_path)
    assert "agents" in found
    assert "claude" not in found
    assert "deep" not in found


def test_nested_precedence_root_before_workspace(tmp_path: Path) -> None:
    """Repo-root instructions come first, nearer ones last."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "AGENTS.md").write_text("root-rules", encoding="utf-8")
    nested = tmp_path / "pkg" / "sub"
    nested.mkdir(parents=True)
    (nested / "AGENTS.md").write_text("near-rules", encoding="utf-8")
    found = project_instructions(nested)
    assert found.index("root-rules") < found.index("near-rules")


def test_no_instructions_is_empty(tmp_path: Path) -> None:
    """Workspaces without instruction files inject nothing."""
    assert project_instructions(tmp_path) == ""


def test_global_instructions_native_first(tmp_path: Path) -> None:
    """~/.minicode/AGENTS.md wins; missing home injects nothing."""
    native = tmp_path / ".minicode"
    native.mkdir()
    (native / "AGENTS.md").write_text("native-rules", encoding="utf-8")
    compat = tmp_path / ".claude"
    compat.mkdir()
    (compat / "CLAUDE.md").write_text("compat-rules", encoding="utf-8")
    found = user_global_instructions(tmp_path)
    assert "native-rules" in found
    assert "compat-rules" not in found
    assert user_global_instructions(tmp_path / "empty-home") == ""


def test_memory_index_only_loads_index(tmp_path: Path) -> None:
    """Only MEMORY.md is auto-loaded; other notes stay on demand."""
    mem = memory_dir(tmp_path)
    mem.mkdir(parents=True)
    (mem / "MEMORY.md").write_text(" durable fact ", encoding="utf-8")
    (mem / "decisions.md").write_text("other note", encoding="utf-8")
    found = memory_index(tmp_path)
    assert "durable fact" in found
    assert "other note" not in found
    assert memory_index(tmp_path / "no-mem") == ""


def test_system_preamble_always_states_tool(tmp_path: Path) -> None:
    """Preamble is never empty: the memory tool usage always applies."""
    found = system_preamble(tmp_path, home=tmp_path / "no-home")
    assert "memory` tool" in found


def test_injection_cap(tmp_path: Path) -> None:
    """Oversized MEMORY.md is truncated with a marker."""
    mem = memory_dir(tmp_path)
    mem.mkdir(parents=True)
    (mem / "MEMORY.md").write_text("x" * 9000, encoding="utf-8")
    found = memory_index(tmp_path)
    assert "…[truncated]" in found
    assert len(found) < 9000


@pytest.mark.asyncio
async def test_memory_tool_crud(tmp_path: Path) -> None:
    """list/read/write/append/delete round-trip."""
    tool = MemoryTool(workspace=str(tmp_path))
    assert tool.name == "memory"
    assert await tool.execute(action="list") == "(no memory yet)"
    assert await tool.execute(action="write", name="n.md", content="a") == (
        "Saved memory: n.md (1 chars)."
    )
    assert await tool.execute(action="read", name="n.md") == "a"
    assert await tool.execute(action="append", name="n.md", content="b") == (
        "Saved memory: n.md (3 chars)."
    )
    assert await tool.execute(action="read", name="n.md") == "a\nb"
    assert "n.md" in await tool.execute(action="list")
    assert await tool.execute(action="delete", name="n.md") == "Deleted memory: n.md."
    assert await tool.execute(action="list") == "(no memory yet)"


@pytest.mark.asyncio
async def test_memory_tool_rejections(tmp_path: Path) -> None:
    """Traversal, missing notes, and bad input fail as data."""
    tool = MemoryTool(workspace=str(tmp_path))
    assert await tool.execute(action="read", name="../evil") == (
        "Error: invalid memory name: '../evil' (use a plain file name)."
    )
    assert await tool.execute(action="read", name="sub/note") == (
        "Error: invalid memory name: 'sub/note' (use a plain file name)."
    )
    assert await tool.execute(action="read", name="ghost.md") == (
        "Error: no such memory: ghost.md"
    )
    assert await tool.execute(action="write", name="n.md", content="  ") == (
        "Error: content is required for write/append."
    )
    assert await tool.execute(action="dance", name="n.md") == (
        "Error: unknown action: dance"
    )


def test_context_builder_injects_memory(tmp_path: Path) -> None:
    """ContextBuilder appends the memory preamble to the system prompt."""
    mem = memory_dir(tmp_path)
    mem.mkdir(parents=True)
    (mem / "MEMORY.md").write_text("project uses uv", encoding="utf-8")
    prompt = ContextBuilder(tmp_path).build_system_prompt()
    assert "project uses uv" in prompt
    assert "memory` tool" in prompt
