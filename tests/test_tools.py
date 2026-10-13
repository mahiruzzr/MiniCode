"""Tests for the Tool contract, registry, and the six tools."""

from typing import Any

import pytest

from tools.base import Tool
from tools.files import EditTool, ReadTool, WriteTool
from tools.registry import ToolRegistry
from tools.search import GlobTool, GrepTool
from tools.shell import BashTool


class EchoTool(Tool):
    """Minimal concrete tool for contract tests."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echo back the input."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return str(kwargs.get("text", ""))


class FakeTool(Tool):
    """Configurable stand-in for registry tests."""

    def __init__(self, name: str = "fake", behavior: str = "ok") -> None:
        self._name = name
        self._behavior = behavior

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Fake tool."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        if self._behavior == "boom":
            raise RuntimeError("boom")
        return "fake-result"


class StrictTool(FakeTool):
    """Fake tool with a required parameter."""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }


def test_tool_is_abstract() -> None:
    """Tool requires name, description, parameters, and execute."""
    with pytest.raises(TypeError):
        Tool()  # type: ignore[abstract]


def test_to_schema_shape() -> None:
    """Schemas follow the OpenAI function format."""
    schema = EchoTool().to_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "echo"
    assert schema["function"]["parameters"]["required"] == ["text"]


def test_validate_params() -> None:
    """Missing or mistyped params produce errors."""
    tool = EchoTool()
    assert tool.validate_params({"text": "hi"}) == []
    assert tool.validate_params({}) != []
    assert tool.validate_params({"text": 1}) != []


def test_register_get_definitions() -> None:
    """Registration feeds lookup and LLM schemas."""
    registry = ToolRegistry()
    registry.register(FakeTool())
    assert registry.get("fake").name == "fake"
    assert registry.get("missing") is None
    definitions = registry.get_definitions()
    assert len(definitions) == 1
    assert definitions[0]["function"]["name"] == "fake"


@pytest.mark.asyncio
async def test_registry_execute_success() -> None:
    """Known tools run and return strings."""
    registry = ToolRegistry()
    registry.register(FakeTool())
    assert await registry.execute("fake", {}) == "fake-result"


@pytest.mark.asyncio
async def test_registry_execute_unknown_tool() -> None:
    """Unknown names become error strings, not exceptions."""
    result = await ToolRegistry().execute("nope", {})
    assert result.startswith("Error:")
    assert "nope" in result


@pytest.mark.asyncio
async def test_registry_execute_exception() -> None:
    """Tool crashes become error strings, not exceptions."""
    registry = ToolRegistry()
    registry.register(FakeTool(behavior="boom"))
    result = await registry.execute("fake", {})
    assert result.startswith("Error")
    assert "boom" in result


@pytest.mark.asyncio
async def test_registry_execute_invalid_params() -> None:
    """Schema violations become error strings, not exceptions."""
    registry = ToolRegistry()
    registry.register(StrictTool(name="strict"))
    result = await registry.execute("strict", {})
    assert result.startswith("Error:")


def test_shell_identity(tmp_path) -> None:
    """Bash tool takes a single command."""
    tool = BashTool(workspace=str(tmp_path))
    assert tool.name == "bash"
    assert tool.parameters["required"] == ["command"]
    assert "command" in tool.parameters["properties"]


@pytest.mark.asyncio
async def test_shell_execute_success(tmp_path) -> None:
    """Commands run in the workspace and return output."""
    result = await BashTool(workspace=str(tmp_path)).execute(command="printf 'hello'")
    assert result == "hello"


@pytest.mark.asyncio
async def test_shell_execute_empty_command(tmp_path) -> None:
    """Blank input is rejected without spawning a shell."""
    result = await BashTool(workspace=str(tmp_path)).execute(command="   ")
    assert result.startswith("Error:")


@pytest.mark.asyncio
async def test_shell_execute_timeout(tmp_path) -> None:
    """Overrunning commands are killed and reported."""
    result = await BashTool(workspace=str(tmp_path)).execute(command="sleep 5", timeout=1)
    assert result.startswith("Error:")
    assert "timed out" in result


@pytest.mark.asyncio
async def test_write_then_read(tmp_path) -> None:
    """Written content reads back with line numbers."""
    ws = str(tmp_path)
    result = await WriteTool(workspace=ws).execute(file_path="a.txt", content="line1\nline2\n")
    assert result.startswith("Wrote")
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "line1\nline2\n"
    content = await ReadTool(workspace=ws).execute(file_path="a.txt")
    assert "1: line1" in content
    assert "2: line2" in content


@pytest.mark.asyncio
async def test_read_missing_file(tmp_path) -> None:
    """Missing files report an error."""
    result = await ReadTool(workspace=str(tmp_path)).execute(file_path="missing.txt")
    assert result.startswith("Error:")


@pytest.mark.asyncio
async def test_edit_replaces_text(tmp_path) -> None:
    """Exact matches are replaced and reported."""
    ws = str(tmp_path)
    await WriteTool(workspace=ws).execute(file_path="b.txt", content="foo bar\n")
    result = await EditTool(workspace=ws).execute(
        file_path="b.txt", old_string="foo", new_string="baz"
    )
    assert result.startswith("Edited")
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "baz bar\n"


@pytest.mark.asyncio
async def test_edit_missing_file(tmp_path) -> None:
    """Editing a missing file points at write instead."""
    result = await EditTool(workspace=str(tmp_path)).execute(
        file_path="missing.txt", old_string="a", new_string="b"
    )
    assert result.startswith("Error:")


@pytest.mark.asyncio
async def test_write_outside_workspace_rejected(tmp_path) -> None:
    """Writes cannot escape the workspace."""
    outside = tmp_path.parent / "outside.txt"
    result = await WriteTool(workspace=str(tmp_path)).execute(
        file_path=str(outside), content="x"
    )
    assert result.startswith("Error:")
    assert not outside.exists()


@pytest.fixture
def search_workspace(tmp_path):
    """Workspace holding searchable files."""
    (tmp_path / "notes.txt").write_text("needle here\nnothing else\n", encoding="utf-8")
    (tmp_path / "skip.log").write_text("needle too\n", encoding="utf-8")
    return str(tmp_path)


@pytest.mark.asyncio
async def test_grep_finds_match(search_workspace: str) -> None:
    """Matching lines come back with file and line info."""
    result = await GrepTool(workspace=search_workspace).execute(pattern="needle")
    assert "notes.txt" in result
    assert "needle here" in result


@pytest.mark.asyncio
async def test_grep_no_matches(search_workspace: str) -> None:
    """Empty results say so plainly."""
    assert await GrepTool(workspace=search_workspace).execute(pattern="zzz-nope") == "No matches."


@pytest.mark.asyncio
async def test_grep_invalid_regex(search_workspace: str) -> None:
    """Broken patterns report an error."""
    result = await GrepTool(workspace=search_workspace).execute(pattern="(")
    assert result.startswith("Error:")


@pytest.mark.asyncio
async def test_glob_finds_files(search_workspace: str) -> None:
    """Glob lists matching files workspace-relative."""
    result = await GlobTool(workspace=search_workspace).execute(pattern="*.txt")
    assert "notes.txt" in result


@pytest.mark.asyncio
async def test_glob_no_files(search_workspace: str) -> None:
    """Empty globs say so plainly."""
    result = await GlobTool(workspace=search_workspace).execute(pattern="*.zzz")
    assert result == "No files matched."
