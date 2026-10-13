"""TUI input history persistence."""

import json
from pathlib import Path

from config.loader import minicode_home

_MAX_HISTORY = 200
_HISTORY_FILENAME = "history.json"


def _history_path() -> Path:
    """Return the history file path under the user home directory."""
    return minicode_home() / _HISTORY_FILENAME


def load_history_entries() -> list[str]:
    """Load saved input history entries.

    Returns:
        Saved entries as strings, newest last, at most the most recent
        200. Empty list when the file is missing or unreadable.
    """
    try:
        raw = _history_path().read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    entries = data.get("entries", []) if isinstance(data, dict) else []
    if not isinstance(entries, list):
        return []
    return [str(entry) for entry in entries[-_MAX_HISTORY:]]


def save_history_entries(entries: list[str]) -> None:
    """Save input history entries, keeping only the most recent 200.

    Args:
        entries: History entries in order, newest last.
    """
    path = _history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    capped = list(entries[-_MAX_HISTORY:])
    path.write_text(
        json.dumps({"entries": capped}, indent=2) + "\n",
        encoding="utf-8",
    )
