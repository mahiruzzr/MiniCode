"""Project instructions and persistent cross-session memory.

Project instructions are loaded from the repository and user-global
locations. MEMORY.md is injected into the system prompt, while other
notes are accessed through MemoryTool.
"""

from pathlib import Path
from typing import Any

from tools.base import Tool

_MEMORY_SUBDIR = ".minicode/memory"
_INDEX_FILE = "MEMORY.md"
_PROJECT_FILES = ("AGENTS.md", "MINICODE.md", "CLAUDE.md")
# Markers that identify the enclosing project root when the workspace is a
# subdirectory: walk up until a directory holding one of these is found.
_PROJECT_ROOT_MARKERS = (".git",)
# User-level standing instructions that apply across every project (lowest
# precedence). Native first, then Claude Code interop.
_USER_GLOBAL_FILES = ((".minicode", "AGENTS.md"), (".claude", "CLAUDE.md"))
_MAX_INJECT_CHARS = 8000  # keep the preamble bounded; the tool reads the rest


def memory_dir(workspace: str | Path) -> Path:
    """Return the persistent memory directory for a workspace."""
    return Path(workspace) / _MEMORY_SUBDIR


def _read_capped(path: Path, cap: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[:cap] + "\n…[truncated]" if len(text) > cap else text


def _find_project_root(start: Path) -> Path | None:
    """Find the nearest ancestor containing a project root marker.

    Returns None when no marker is found.
    """
    for directory in (start, *start.parents):
        if any((directory / marker).exists() for marker in _PROJECT_ROOT_MARKERS):
            return directory
    return None


def project_instructions(workspace: str | Path) -> str:
    """Load project instructions from the repo root to the workspace.

    Each directory contributes at most one file, with AGENTS.md preferred
    over MINICODE.md, and MINICODE.md preferred over CLAUDE.md. A workspace
    outside a repository uses only its own file.
    """
    workspace = Path(workspace).resolve()
    root = _find_project_root(workspace)
    if root is None or root == workspace:
        search_dirs = [workspace]
    else:
        chain, cursor = [workspace], workspace
        while cursor != root and cursor.parent != cursor:
            cursor = cursor.parent
            chain.append(cursor)
        search_dirs = list(reversed(chain))  # root first → workspace last

    blocks: list[str] = []
    remaining = _MAX_INJECT_CHARS
    for directory in search_dirs:
        for name in _PROJECT_FILES:
            candidate = directory / name
            if candidate.is_file():
                body = _read_capped(candidate, remaining).strip()
                if body:
                    label = name if directory == workspace else f"{directory}/{name}"
                    blocks.append(f"## Project instructions (from {label})\n\n{body}")
                    remaining -= len(body)
                break  # one file per directory: AGENTS.md > MINICODE.md > CLAUDE.md
        if remaining <= 0:
            break
    return "\n\n".join(blocks)


def user_global_instructions(home: str | Path | None = None) -> str:
    """Load user-global instructions from the home directory."""
    base = Path(home) if home is not None else Path.home()
    for subdir, name in _USER_GLOBAL_FILES:
        candidate = base / subdir / name
        if candidate.is_file():
            body = _read_capped(candidate, _MAX_INJECT_CHARS).strip()
            if body:
                return f"## User instructions (from ~/{subdir}/{name})\n\n{body}"
    return ""


def memory_index(workspace: str | Path) -> str:
    """Load the workspace MEMORY.md index for prompt injection."""
    index = memory_dir(workspace) / _INDEX_FILE
    if index.is_file():
        body = _read_capped(index, _MAX_INJECT_CHARS)
        if body.strip():
            return f"## Memory (from {_MEMORY_SUBDIR}/{_INDEX_FILE})\n\n{body.strip()}"
    return ""


_MEMORY_USAGE = (
    "You have a `memory` tool for persistent notes under "
    f"`{_MEMORY_SUBDIR}/`. When you learn a durable fact — a project "
    "convention, an architectural decision, a gotcha, or a user preference — "
    f"record it so future sessions benefit, and keep `{_INDEX_FILE}` as a "
    "short index of what you know. Read a note before relying on it; it "
    "reflects a past session and may be stale."
)


def system_preamble(workspace: str | Path, home: str | Path | None = None) -> str:
    """Build the memory-related system-prompt section.

    Order: user-global instructions, project instructions, memory index,
    then memory tool guidance.
    """
    parts = [
        user_global_instructions(home),
        project_instructions(workspace),
        memory_index(workspace),
        _MEMORY_USAGE,
    ]
    return "\n\n".join(p for p in parts if p)


class MemoryTool(Tool):
    """Read/write persistent memory notes under ``<workspace>/.minicode/memory``."""

    def __init__(self, workspace: str | Path):
        self._dir = memory_dir(workspace)

    @property
    def name(self) -> str:
        return "memory"

    @property
    def description(self) -> str:
        return (
            "Persistent notes that survive across sessions, stored under "
            ".minicode/memory/. actions: list | read | write | append | "
            "delete. Keep MEMORY.md as the index of what you know."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "read", "write", "append", "delete"],
                    "description": "The memory operation to perform.",
                },
                "name": {
                    "type": "string",
                    "description": "Memory file name (e.g. MEMORY.md, decisions.md). "
                    "Required for all actions except list.",
                },
                "content": {
                    "type": "string",
                    "description": "Text to write or append (for write/append).",
                },
            },
            "required": ["action"],
        }

    def _resolve(self, name: str) -> Path | None:
        """Resolve ``name`` inside the memory dir, or None if it escapes."""
        if not name or name != Path(name).name:
            return None  # no subdirs / traversal — a flat notes namespace
        return self._dir / name

    async def execute(self, **kwargs: Any) -> str:
        """Execute a memory operation and return its result."""
        action = str(kwargs.get("action", "")).lower()
        name = kwargs.get("name") or ""
        content = kwargs.get("content") or ""

        if action == "list":
            if not self._dir.is_dir():
                return "(no memory yet)"
            names = sorted(p.name for p in self._dir.iterdir() if p.is_file())
            return "\n".join(names) if names else "(no memory yet)"

        target = self._resolve(name)
        if target is None:
            return f"Error: invalid memory name: {name!r} (use a plain file name)."

        if action == "read":
            if not target.is_file():
                return f"Error: no such memory: {name}"
            return target.read_text(encoding="utf-8", errors="replace")

        if action in ("write", "append"):
            if not content.strip():
                return "Error: content is required for write/append."
            try:
                self._dir.mkdir(parents=True, exist_ok=True)
                if action == "append" and target.is_file():
                    existing = target.read_text(encoding="utf-8", errors="replace")
                    content = existing.rstrip() + "\n" + content
                target.write_text(content, encoding="utf-8")
            except OSError as exc:
                return f"Error: could not write memory {name}: {exc}"
            return f"Saved memory: {name} ({len(content)} chars)."

        if action == "delete":
            if not target.is_file():
                return f"Error: no such memory: {name}"
            try:
                target.unlink()
            except OSError as exc:
                return f"Error: could not delete memory {name}: {exc}"
            return f"Deleted memory: {name}."

        return f"Error: unknown action: {action}"
