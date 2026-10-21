"""Tests for the Tool contract, registry, and the built-in tools."""

from typing import Any

import pytest

from tools.base import Tool
from tools.diagnostics import Diagnostic
from tools.files import EditTool, ReadTool, WriteTool
from tools.patch import ApplyPatchTool, PatchError, parse_patch
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


def _no_diagnostics(_path: str) -> list[Diagnostic]:
    """Diagnostics stub that reports nothing."""
    return []


def _patch_tool(workspace: str) -> ApplyPatchTool:
    """Build an apply_patch tool with diagnostics switched off."""
    return ApplyPatchTool(workspace=workspace, diagnostics=_no_diagnostics)


class RecordingDiagnostics:
    """Diagnostics stub that records the paths it was asked to check."""

    def __init__(self, diagnostics: list[Diagnostic] | None = None) -> None:
        self.paths: list[str] = []
        self._diagnostics = diagnostics if diagnostics is not None else []

    def __call__(self, path: str) -> list[Diagnostic]:
        self.paths.append(path)
        return self._diagnostics


def test_apply_patch_identity(tmp_path) -> None:
    """The tool takes a single patch argument."""
    tool = _patch_tool(str(tmp_path))
    assert tool.name == "apply_patch"
    assert tool.parameters["required"] == ["patch"]
    assert "patch" in tool.parameters["properties"]


def test_parse_patch_reads_every_directive() -> None:
    """Add, Update, Delete, and Move parse into ordered file operations."""
    ops = parse_patch(
        "*** Begin Patch\n"
        "*** Add File: new.py\n"
        "+print(1)\n"
        "*** Update File: old.py\n"
        "*** Move to: moved.py\n"
        "@@ def f():\n"
        " def f():\n"
        "-    return 1\n"
        "+    return 2\n"
        "*** Delete File: gone.py\n"
        "*** End Patch"
    )
    assert [op.kind for op in ops] == ["add", "update", "delete"]
    assert ops[0].path == "new.py"
    assert ops[0].add_content == "print(1)"
    assert ops[1].path == "old.py"
    assert ops[1].move_to == "moved.py"
    assert ops[1].hunks[0].before == "def f():\n    return 1"
    assert ops[1].hunks[0].after == "def f():\n    return 2"
    assert ops[2].path == "gone.py"


def test_parse_patch_requires_begin_sentinel() -> None:
    """A patch without the opening sentinel is rejected."""
    with pytest.raises(PatchError, match="must start with"):
        parse_patch("*** Add File: x\n+hi\n")


def test_parse_patch_rejects_unknown_directive() -> None:
    """Unknown directives are reported instead of ignored."""
    with pytest.raises(PatchError, match="unexpected patch directive"):
        parse_patch("*** Begin Patch\n*** Frobnicate File: x\n*** End Patch")


def test_parse_patch_rejects_update_without_changes() -> None:
    """An update holding only context lines has nothing to apply."""
    with pytest.raises(PatchError, match="no change hunks"):
        parse_patch("*** Begin Patch\n*** Update File: x\n@@\n keep me\n*** End Patch")


def test_parse_patch_rejects_patch_without_operations() -> None:
    """A patch with no directives carries no work."""
    with pytest.raises(PatchError, match="no file operations"):
        parse_patch("*** Begin Patch\n*** End Patch")


def test_parse_patch_splits_hunks_on_at_marker() -> None:
    """Every '@@' line starts a new hunk."""
    ops = parse_patch(
        "*** Begin Patch\n"
        "*** Update File: x\n"
        "@@\n"
        "-a\n"
        "+b\n"
        "@@\n"
        "-c\n"
        "+d\n"
        "*** End Patch"
    )
    assert [(hunk.before, hunk.after) for hunk in ops[0].hunks] == [
        ("a", "b"),
        ("c", "d"),
    ]


def test_parse_patch_keeps_at_header_out_of_the_match() -> None:
    """The '@@' header separates hunks; it is not a match anchor."""
    ops = parse_patch(
        "*** Begin Patch\n"
        "*** Update File: x\n"
        "@@ totally wrong anchor\n"
        "-a\n"
        "+b\n"
        "*** End Patch"
    )
    assert ops[0].hunks[0].before == "a"


def test_parse_patch_normalizes_line_endings() -> None:
    """CRLF patches parse the same as LF patches."""
    crlf = parse_patch(
        "*** Begin Patch\r\n*** Add File: x\r\n+hello\r\n*** End Patch\r\n"
    )
    lf = parse_patch("*** Begin Patch\n*** Add File: x\n+hello\n*** End Patch")
    assert crlf[0].add_content == lf[0].add_content == "hello"


def test_parse_patch_keeps_directive_like_text_inside_add_body() -> None:
    """Prefixed add lines keep their text, even when it looks like a directive."""
    ops = parse_patch(
        "*** Begin Patch\n"
        "*** Add File: x\n"
        "+before\n"
        "+*** not a directive\n"
        "+after\n"
        "*** End Patch"
    )
    assert ops[0].add_content == "before\n*** not a directive\nafter"


def test_parse_patch_rejects_raw_directive_line_in_add_body() -> None:
    """An unprefixed '*** ' line inside an add body is read as a directive.

    Known limitation: add bodies must prefix each line with '+' or ' ' so
    the parser can tell content from directives.
    """
    with pytest.raises(PatchError, match="unexpected patch directive"):
        parse_patch(
            "*** Begin Patch\n"
            "*** Add File: r.txt\n"
            "+before\n"
            "*** raw line\n"
            "*** End Patch"
        )


@pytest.mark.asyncio
async def test_apply_patch_adds_file(tmp_path) -> None:
    """An Add directive creates the file and its parent directories."""
    ws = str(tmp_path)
    result = await _patch_tool(ws).execute(
        patch=(
            "*** Begin Patch\n"
            "*** Add File: sub/new.txt\n"
            "+line1\n"
            "+line2\n"
            "*** End Patch"
        )
    )
    assert result.startswith("Applied patch: 1 change(s)")
    assert (tmp_path / "sub" / "new.txt").read_text(encoding="utf-8") == "line1\nline2"


@pytest.mark.asyncio
async def test_apply_patch_updates_file_by_context(tmp_path) -> None:
    """An Update hunk replaces the span its context selects."""
    ws = str(tmp_path)
    (tmp_path / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    result = await _patch_tool(ws).execute(
        patch=(
            "*** Begin Patch\n"
            "*** Update File: a.py\n"
            "@@ def f():\n"
            " def f():\n"
            "-    return 1\n"
            "+    return 2\n"
            "*** End Patch"
        )
    )
    assert result.startswith("Applied patch: 1 change(s)")
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "def f():\n    return 2\n"


@pytest.mark.asyncio
async def test_apply_patch_deletes_file(tmp_path) -> None:
    """A Delete directive removes the file."""
    ws = str(tmp_path)
    (tmp_path / "d.txt").write_text("x\n", encoding="utf-8")
    result = await _patch_tool(ws).execute(
        patch="*** Begin Patch\n*** Delete File: d.txt\n*** End Patch"
    )
    assert "d.txt (deleted)" in result
    assert not (tmp_path / "d.txt").exists()


@pytest.mark.asyncio
async def test_apply_patch_moves_file(tmp_path) -> None:
    """A Move directive renames the file and keeps the edited content."""
    ws = str(tmp_path)
    (tmp_path / "old.py").write_text("body\n", encoding="utf-8")
    result = await _patch_tool(ws).execute(
        patch=(
            "*** Begin Patch\n"
            "*** Update File: old.py\n"
            "*** Move to: new.py\n"
            "@@\n"
            " body\n"
            "+more\n"
            "*** End Patch"
        )
    )
    assert "new.py" in result
    assert not (tmp_path / "old.py").exists()
    assert (tmp_path / "new.py").read_text(encoding="utf-8") == "body\nmore\n"


@pytest.mark.asyncio
async def test_apply_patch_changes_multiple_files(tmp_path) -> None:
    """One patch can add and update different files in a single call."""
    ws = str(tmp_path)
    (tmp_path / "m1.txt").write_text("one\n", encoding="utf-8")
    result = await _patch_tool(ws).execute(
        patch=(
            "*** Begin Patch\n"
            "*** Add File: m2.txt\n"
            "+two\n"
            "*** Update File: m1.txt\n"
            "@@\n"
            "-one\n"
            "+ONE\n"
            "*** End Patch"
        )
    )
    assert result.startswith("Applied patch: 2 change(s)")
    assert (tmp_path / "m1.txt").read_text(encoding="utf-8") == "ONE\n"
    assert (tmp_path / "m2.txt").read_text(encoding="utf-8") == "two"


@pytest.mark.asyncio
async def test_apply_patch_rejects_path_outside_workspace(tmp_path) -> None:
    """Paths that escape the workspace are refused."""
    ws = str(tmp_path)
    result = await _patch_tool(ws).execute(
        patch="*** Begin Patch\n*** Add File: ../escape.txt\n+bad\n*** End Patch"
    )
    assert result.startswith("Error:")
    assert "outside the workspace" in result
    assert not (tmp_path.parent / "escape.txt").exists()


@pytest.mark.asyncio
async def test_apply_patch_rejects_move_outside_workspace(tmp_path) -> None:
    """A move target outside the workspace is refused and the file stays put."""
    ws = str(tmp_path)
    (tmp_path / "m.py").write_text("x\n", encoding="utf-8")
    result = await _patch_tool(ws).execute(
        patch=(
            "*** Begin Patch\n"
            "*** Update File: m.py\n"
            "*** Move to: ../out.py\n"
            "@@\n"
            " x\n"
            "+y\n"
            "*** End Patch"
        )
    )
    assert result.startswith("Error:")
    assert "refusing to move" in result
    assert not (tmp_path.parent / "out.py").exists()
    assert (tmp_path / "m.py").read_text(encoding="utf-8") == "x\n"


@pytest.mark.asyncio
async def test_apply_patch_rejects_adding_existing_file(tmp_path) -> None:
    """Adding over an existing file points at the update directive."""
    ws = str(tmp_path)
    (tmp_path / "e.txt").write_text("x\n", encoding="utf-8")
    result = await _patch_tool(ws).execute(
        patch="*** Begin Patch\n*** Add File: e.txt\n+y\n*** End Patch"
    )
    assert result.startswith("Error:")
    assert "already exists" in result
    assert (tmp_path / "e.txt").read_text(encoding="utf-8") == "x\n"


@pytest.mark.asyncio
async def test_apply_patch_rejects_deleting_missing_file(tmp_path) -> None:
    """Deleting a file that is not there reports the miss."""
    result = await _patch_tool(str(tmp_path)).execute(
        patch="*** Begin Patch\n*** Delete File: gone.txt\n*** End Patch"
    )
    assert result.startswith("Error:")
    assert "file not found" in result


@pytest.mark.asyncio
async def test_apply_patch_rejects_updating_missing_file(tmp_path) -> None:
    """Updating a file that is not there points at the add directive."""
    result = await _patch_tool(str(tmp_path)).execute(
        patch="*** Begin Patch\n*** Update File: gone.txt\n@@\n-a\n+b\n*** End Patch"
    )
    assert result.startswith("Error:")
    assert "file not found" in result


@pytest.mark.asyncio
async def test_apply_patch_reports_unmatched_hunk(tmp_path) -> None:
    """A hunk that matches nothing leaves the file untouched."""
    ws = str(tmp_path)
    (tmp_path / "u.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    result = await _patch_tool(ws).execute(
        patch="*** Begin Patch\n*** Update File: u.txt\n@@\n-nope\n+yes\n*** End Patch"
    )
    assert result.startswith("Error:")
    assert "Could not find old_string" in result
    assert (tmp_path / "u.txt").read_text(encoding="utf-8") == "alpha\nbeta\n"


@pytest.mark.asyncio
async def test_apply_patch_reports_ambiguous_hunk(tmp_path) -> None:
    """A hunk matching several places asks for more context."""
    ws = str(tmp_path)
    (tmp_path / "m.txt").write_text("dup\nx\ndup\n", encoding="utf-8")
    result = await _patch_tool(ws).execute(
        patch="*** Begin Patch\n*** Update File: m.txt\n@@\n-dup\n+unique\n*** End Patch"
    )
    assert result.startswith("Error:")
    assert "multiple matches" in result
    assert (tmp_path / "m.txt").read_text(encoding="utf-8") == "dup\nx\ndup\n"


@pytest.mark.asyncio
async def test_apply_patch_rejects_hunk_without_anchor(tmp_path) -> None:
    """A hunk made of added lines only has nothing to match on.

    Known limitation: the matcher needs at least one context or removed
    line to anchor the span, so the hunk is refused instead of applied.
    """
    ws = str(tmp_path)
    (tmp_path / "p.txt").write_text("alpha\n", encoding="utf-8")
    result = await _patch_tool(ws).execute(
        patch="*** Begin Patch\n*** Update File: p.txt\n@@\n+brand new\n*** End Patch"
    )
    assert result.startswith("Error:")
    assert "old_string cannot be empty" in result
    assert (tmp_path / "p.txt").read_text(encoding="utf-8") == "alpha\n"


@pytest.mark.asyncio
async def test_apply_patch_rejects_blank_patch(tmp_path) -> None:
    """A blank patch never reaches the parser."""
    result = await _patch_tool(str(tmp_path)).execute(patch="   \n  ")
    assert result == "Error: patch is empty."


@pytest.mark.asyncio
async def test_apply_patch_skips_writes_when_validation_fails(tmp_path) -> None:
    """A later failing operation aborts the patch before anything is written."""
    ws = str(tmp_path)
    result = await _patch_tool(ws).execute(
        patch=(
            "*** Begin Patch\n"
            "*** Add File: first.txt\n"
            "+ok\n"
            "*** Update File: missing.txt\n"
            "@@\n"
            "-a\n"
            "+b\n"
            "*** End Patch"
        )
    )
    assert result.startswith("Error:")
    assert not (tmp_path / "first.txt").exists()


@pytest.mark.asyncio
async def test_apply_patch_keeps_earlier_edits_when_validation_fails(tmp_path) -> None:
    """A failed operation also rolls back earlier edits of the same patch."""
    ws = str(tmp_path)
    (tmp_path / "keep.txt").write_text("x\n", encoding="utf-8")
    result = await _patch_tool(ws).execute(
        patch=(
            "*** Begin Patch\n"
            "*** Update File: keep.txt\n"
            "@@\n"
            "-x\n"
            "+y\n"
            "*** Add File: ../escape.txt\n"
            "+bad\n"
            "*** End Patch"
        )
    )
    assert result.startswith("Error:")
    assert (tmp_path / "keep.txt").read_text(encoding="utf-8") == "x\n"


@pytest.mark.asyncio
async def test_apply_patch_checks_diagnostics_of_written_files(tmp_path) -> None:
    """Diagnostics run against every file the patch writes."""
    ws = str(tmp_path)
    recorder = RecordingDiagnostics()
    tool = ApplyPatchTool(workspace=ws, diagnostics=recorder)
    await tool.execute(
        patch="*** Begin Patch\n*** Add File: diag.py\n+x = 1\n*** End Patch"
    )
    assert recorder.paths == [str(tmp_path / "diag.py")]


@pytest.mark.asyncio
async def test_apply_patch_checks_diagnostics_of_move_destination(tmp_path) -> None:
    """A move reports diagnostics for the new path."""
    ws = str(tmp_path)
    (tmp_path / "src.py").write_text("x\n", encoding="utf-8")
    recorder = RecordingDiagnostics()
    tool = ApplyPatchTool(workspace=ws, diagnostics=recorder)
    await tool.execute(
        patch=(
            "*** Begin Patch\n"
            "*** Update File: src.py\n"
            "*** Move to: dst.py\n"
            "@@\n"
            " x\n"
            "+y\n"
            "*** End Patch"
        )
    )
    assert recorder.paths == [str(tmp_path / "dst.py")]


@pytest.mark.asyncio
async def test_apply_patch_reports_diagnostics_in_result(tmp_path) -> None:
    """Diagnostics findings are appended to the tool result."""
    ws = str(tmp_path)
    recorder = RecordingDiagnostics(
        [
            Diagnostic(
                line=1,
                column=1,
                severity="error",
                message="undefined name",
                source="py_compile",
            )
        ]
    )
    tool = ApplyPatchTool(workspace=ws, diagnostics=recorder)
    result = await tool.execute(
        patch="*** Begin Patch\n*** Add File: bad.py\n+x = 1\n*** End Patch"
    )
    assert "Diagnostics detected" in result
    assert "undefined name" in result
