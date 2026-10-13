"""Tests for TUI input history persistence."""

import json
from pathlib import Path

import pytest

from tui.history import load_history_entries, save_history_entries


def test_load_missing_file_returns_empty(isolated_home: Path) -> None:
    """No history file means empty history."""
    assert load_history_entries() == []


def test_save_then_load_roundtrip(isolated_home: Path) -> None:
    """Saved entries load back in order."""
    save_history_entries(["hello", "world"])
    assert load_history_entries() == ["hello", "world"]


def test_corrupt_json_returns_empty(
    isolated_home: Path, tmp_path: Path
) -> None:
    """Garbage file content is treated as empty history."""
    path = tmp_path / "home" / "history.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert load_history_entries() == []


def test_non_list_entries_returns_empty(
    isolated_home: Path, tmp_path: Path
) -> None:
    """A non-list entries value is treated as empty history."""
    path = tmp_path / "home" / "history.json"
    path.write_text(json.dumps({"entries": "nope"}), encoding="utf-8")
    assert load_history_entries() == []


def test_non_string_entries_become_strings(isolated_home: Path) -> None:
    """Non-string entries load back as strings."""
    save_history_entries(["ok", 42, None])  # type: ignore[list-item]
    assert load_history_entries() == ["ok", "42", "None"]


def test_save_caps_at_max_history(isolated_home: Path) -> None:
    """Only the most recent 200 entries are kept."""
    entries = [f"cmd-{i}" for i in range(250)]
    save_history_entries(entries)
    loaded = load_history_entries()
    assert loaded == entries[-200:]
    assert loaded[0] == "cmd-50"


def test_save_creates_parent_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Saving creates missing parent directories."""
    home = tmp_path / "deep" / "nested"
    assert not home.exists()
    monkeypatch.setenv("MINICODE_HOME", str(home))
    save_history_entries(["x"])
    assert load_history_entries() == ["x"]
