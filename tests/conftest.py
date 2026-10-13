"""Pytest configuration for MiniCode tests."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate MINICODE_HOME in a temp dir."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("MINICODE_HOME", str(home))
    return home
