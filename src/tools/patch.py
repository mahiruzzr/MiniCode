"""Apply Add, Update, Delete, and Move operations from a multi-file patch.

A patch uses this envelope; ``@@`` lines separate update hunks::

    *** Begin Patch
    *** Add File: pkg/new.py
    +def f():
    +    return 1
    *** Update File: pkg/existing.py
    *** Move to: pkg/renamed.py
    @@ def existing():
     def existing():
    -    return old()
    +    return new()
    *** Delete File: pkg/dead.py
    *** End Patch

Update hunks locate their span by surrounding context instead of line
numbers, reusing the fuzzy matching engine behind ``edit``. The context and
removed lines form the match text; the context and added lines form the
replacement.

All paths must resolve inside the configured workspace. ``execute`` validates
and stages every operation before writing, so parse and validation failures
leave all files untouched. An I/O error during the commit phase can leave a
partial patch; that case is reported explicitly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import Tool
from .diagnostics import Diagnostic, format_diagnostics, run_diagnostics
from .files import _resolve, _within
from .replace import ReplaceError, replace

_BEGIN = "*** Begin Patch"
_END = "*** End Patch"
_ADD = "*** Add File: "
_UPDATE = "*** Update File: "
_DELETE = "*** Delete File: "
_MOVE = "*** Move to: "


class PatchError(ValueError):
    """Raised for a malformed patch envelope or update hunk."""


@dataclass(frozen=True)
class Hunk:
    """An update hunk expressed as match and replacement text.

    ``before`` contains context and removed lines; ``after`` contains context
    and added lines.
    """

    before: str
    after: str


@dataclass(frozen=True)
class FileOp:
    """A file operation parsed from a patch envelope.

    ``kind`` is ``"add"``, ``"update"``, or ``"delete"``. Adds use
    ``add_content``; updates use ``hunks`` and may set ``move_to``.
    """

    kind: str
    path: str
    move_to: str | None = None
    add_content: str | None = None
    hunks: list[Hunk] = field(default_factory=list)


def _hunk_from_lines(lines: list[tuple[str, str]]) -> Hunk | None:
    """Build a hunk from classified patch-body lines.

    Args:
        lines: ``(prefix, text)`` pairs, where prefix is ``" "`` for context,
            ``"-"`` for a removed line, or ``"+"`` for an added line.

    Returns:
        A hunk containing the match and replacement text, or ``None`` when
        the lines contain no changes.

    """
    before: list[str] = []
    after: list[str] = []
    changed = False
    for op, text in lines:
        if op == " ":
            before.append(text)
            after.append(text)
        elif op == "-":
            before.append(text)
            changed = True
        elif op == "+":
            after.append(text)
            changed = True
    if not changed:
        return None
    return Hunk(before="\n".join(before), after="\n".join(after))


def _classify(line: str) -> tuple[str, str]:
    """Split a patch-body line into its prefix and text.

    Empty lines are treated as empty context lines. Lines without a recognized
    prefix are also treated as context, which tolerates patches that omit the
    leading space on unchanged lines.
    """
    if line == "":
        return " ", ""
    head, rest = line[0], line[1:]
    if head in (" ", "-", "+"):
        return head, rest
    return " ", line


def parse_patch(text: str) -> list[FileOp]:
    """Parse a patch envelope into file operations.

    The opening ``*** Begin Patch`` sentinel is required. The closing sentinel
    is optional; when present, the last occurrence marks the end of the patch.
    ``@@`` lines separate update hunks and are not included in their match
    text.

    Args:
        text: Patch text. CRLF and CR line endings are normalized to LF.

    Returns:
        File operations in the order they appear in the patch.

    Raises:
        PatchError: If the opening sentinel is missing, a directive is
            unknown, an update contains no changes, or the patch contains no
            file operations.

    """
    raw = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = raw.split("\n")

    try:
        start = next(
            i for i, line in enumerate(lines) if line.strip() == _BEGIN
        )
    except StopIteration:
        raise PatchError(f"patch must start with '{_BEGIN}'")
    end = next(
        (i for i in range(len(lines) - 1, -1, -1) if lines[i].strip() == _END),
        len(lines),
    )
    body = lines[start + 1 : end]

    ops: list[FileOp] = []
    i = 0
    line_count = len(body)
    while i < line_count:
        line = body[i]
        if line.startswith(_ADD):
            path = line[len(_ADD) :].strip()
            i += 1
            added: list[str] = []
            while i < line_count and not body[i].startswith("*** "):
                op, body_text = _classify(body[i])
                # Add bodies normally use "+", but context lines are accepted.
                added.append(body_text if op in ("+", " ") else body[i])
                i += 1
            ops.append(
                FileOp(kind="add", path=path, add_content="\n".join(added))
            )
            continue

        if line.startswith(_UPDATE):
            path = line[len(_UPDATE) :].strip()
            i += 1
            move_to: str | None = None
            if i < line_count and body[i].startswith(_MOVE):
                move_to = body[i][len(_MOVE) :].strip()
                i += 1
            hunk_lines: list[tuple[str, str]] = []
            hunks: list[Hunk] = []

            def _flush_hunk() -> None:
                if hunk_lines:
                    hunk = _hunk_from_lines(hunk_lines)
                    if hunk is not None:
                        hunks.append(hunk)

            while i < line_count and not body[i].startswith("*** "):
                current_line = body[i]
                if current_line.startswith("@@"):
                    _flush_hunk()
                    hunk_lines = []
                    i += 1
                    continue
                hunk_lines.append(_classify(current_line))
                i += 1
            _flush_hunk()
            if not hunks:
                raise PatchError(
                    f"update for {path!r} has no change hunks (only context?)."
                )
            ops.append(
                FileOp(kind="update", path=path, move_to=move_to, hunks=hunks)
            )
            continue

        if line.startswith(_DELETE):
            path = line[len(_DELETE) :].strip()
            ops.append(FileOp(kind="delete", path=path))
            i += 1
            continue

        if line.strip() == "":
            i += 1
            continue

        raise PatchError(f"unexpected patch directive: {line!r}")

    if not ops:
        raise PatchError("patch contains no file operations.")
    return ops


@dataclass
class _PlannedWrite:
    """A validated write queued for the commit phase."""

    target: Path
    content: str
    display: str


@dataclass
class _PlannedDelete:
    """A validated deletion queued to run after writes."""

    target: Path
    display: str


class ApplyPatchTool(Tool):
    """Apply multi-file patches with context-based fuzzy hunk matching."""

    def __init__(
        self,
        workspace: str,
        diagnostics: Callable[[str], list[Diagnostic]] = run_diagnostics,
    ) -> None:
        """Initialize the tool.

        Args:
            workspace: Root directory that every patch path must stay within.
            diagnostics: Callable that returns diagnostics for a written file.

        """
        self._workspace = str(workspace)
        self._diagnostics = diagnostics

    @property
    def name(self) -> str:
        """Return the name used by the agent to call this tool."""
        return "apply_patch"

    @property
    def description(self) -> str:
        """Return the tool description shown to the model."""
        return (
            "Apply a multi-file patch (Add/Update/Delete/Move) atomically. "
            "Hunks locate their span by surrounding context, so they tolerate "
            "small drift. Prefer this over several edit calls when a change "
            "spans files or must land all-or-nothing."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        """Return the JSON schema for the tool's ``patch`` argument."""
        return {
            "type": "object",
            "properties": {
                "patch": {
                    "type": "string",
                    "description": (
                        "A patch in the '*** Begin Patch' / '*** End Patch' "
                        "envelope with '*** Add File:', '*** Update File:' "
                        "(with @@ hunks of space/-/+ lines), "
                        "'*** Delete File:', and optional '*** Move to:' "
                        "directives."
                    ),
                }
            },
            "required": ["patch"],
        }

    async def execute(self, **kwargs: Any) -> str:
        """Apply the patch and return a summary, diagnostics, or an error.

        Handled patch failures are returned as ``"Error: ..."`` strings so the
        model can revise the patch and retry.
        """
        patch_text = kwargs.get("patch", "")
        if not isinstance(patch_text, str) or not patch_text.strip():
            return "Error: patch is empty."
        try:
            ops = parse_patch(patch_text)
        except PatchError as exc:
            return f"Error: {exc}"

        # Validate and stage all operations before writing to disk.
        writes: list[_PlannedWrite] = []
        deletes: list[_PlannedDelete] = []
        planned_paths: set[Path] = set()
        for op in ops:
            target = _resolve(self._workspace, op.path)
            if not _within(self._workspace, target):
                return (
                    "Error: refusing to touch a path outside the workspace: "
                    f"{op.path}."
                )

            if op.kind == "add":
                if target.exists():
                    return (
                        f"Error: cannot add {op.path}: it already exists. "
                        "Use an Update File hunk to change it."
                    )
                writes.append(
                    _PlannedWrite(target, op.add_content or "", op.path)
                )
                planned_paths.add(target)
                continue

            if op.kind == "delete":
                if not target.exists():
                    return f"Error: cannot delete {op.path}: file not found."
                deletes.append(_PlannedDelete(target, op.path))
                continue

            if not target.exists():
                return (
                    f"Error: cannot update {op.path}: file not found. Use "
                    "Add File to create it."
                )
            try:
                content = target.read_text(encoding="utf-8")
            except OSError as exc:
                return f"Error: could not read {op.path}: {exc}"
            for hunk in op.hunks:
                try:
                    content = replace(content, hunk.before, hunk.after)
                except ReplaceError as exc:
                    return f"Error: in {op.path}: {exc}"
                except ValueError as exc:
                    return f"Error: in {op.path}: {exc}"
            if op.move_to:
                dest = _resolve(self._workspace, op.move_to)
                if not _within(self._workspace, dest):
                    return (
                        f"Error: refusing to move {op.path} outside the "
                        f"workspace: {op.move_to}."
                    )
                writes.append(_PlannedWrite(dest, content, op.move_to))
                deletes.append(_PlannedDelete(target, op.path))
                planned_paths.add(dest)
            else:
                writes.append(_PlannedWrite(target, content, op.path))
                planned_paths.add(target)

        # Write first, then delete; skip deletions that overlap a planned write.
        touched: list[str] = []
        try:
            for write_op in writes:
                write_op.target.parent.mkdir(parents=True, exist_ok=True)
                write_op.target.write_text(write_op.content, encoding="utf-8")
                touched.append(write_op.display)
            for delete_op in deletes:
                if delete_op.target not in planned_paths:
                    delete_op.target.unlink()
                    touched.append(f"{delete_op.display} (deleted)")
        except OSError as exc:
            return f"Error: patch partially failed while writing: {exc}"

        reports: list[str] = []
        for write_op in writes:
            report = format_diagnostics(self._diagnostics(str(write_op.target)))
            if report:
                reports.append(f"{write_op.display}:\n{report}")
        summary = f"Applied patch: {len(touched)} change(s) — " + ", ".join(
            touched
        )
        if reports:
            return summary + "\n\n" + "\n\n".join(reports)
        return summary
