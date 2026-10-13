"""Tests for config loading, layering, and schema."""

import json
from pathlib import Path

import pytest

from config.loader import (
    ConfigError,
    default_config_path,
    home_config_path,
    load_config,
)
from config.schema import MiniCodeConfig


def _write(path: Path, data: object) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_defaults_when_no_config(
    isolated_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing files fall back to schema defaults."""
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    config = load_config()
    assert config.agents.defaults.model == "gpt-4o"
    assert config.providers.openai.api_key == ""


def test_single_config_path(tmp_path: Path, isolated_home: Path) -> None:
    """Explicit path loads exactly that file."""
    path = _write(tmp_path / "c.json", {"agents": {"defaults": {"model": "claude-x"}}})
    config = load_config(path)
    assert config.agents.defaults.model == "claude-x"


def test_project_overrides_user(
    isolated_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Project layer wins; untouched nested keys survive."""
    _write(
        isolated_home / "minicode_config.json",
        {"agents": {"defaults": {"model": "gpt-4o", "temperature": 0.2}}},
    )
    proj = tmp_path / "proj"
    proj.mkdir()
    _write(proj / "minicode_config.json", {"agents": {"defaults": {"model": "gpt-5"}}})
    monkeypatch.chdir(proj)
    config = load_config()
    assert config.agents.defaults.model == "gpt-5"
    assert config.agents.defaults.temperature == 0.2


def test_env_interpolation(
    tmp_path: Path, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """${VAR} expands from the environment."""
    monkeypatch.setenv("MINICODE_TEST_KEY", "secret-1")
    path = _write(tmp_path / "c.json", {"providers": {"openai": {"api_key": "${MINICODE_TEST_KEY}"}}})
    assert load_config(path).providers.openai.api_key == "secret-1"


def test_missing_env_raises(
    tmp_path: Path, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unset variables fail loudly with the variable name."""
    monkeypatch.delenv("MINICODE_TEST_MISSING", raising=False)
    path = _write(tmp_path / "c.json", {"x": "${MINICODE_TEST_MISSING}"})
    with pytest.raises(ConfigError, match="MINICODE_TEST_MISSING"):
        load_config(path)


def test_invalid_json_raises(tmp_path: Path, isolated_home: Path) -> None:
    """Malformed JSON is a ConfigError, not a silent default."""
    path = tmp_path / "bad.json"
    path.write_text("{oops", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_non_object_root_raises(tmp_path: Path, isolated_home: Path) -> None:
    """Top-level arrays are rejected."""
    path = _write(tmp_path / "arr.json", [1, 2])
    with pytest.raises(ConfigError):
        load_config(path)


def test_path_helpers_respect_home(
    isolated_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Home and upward project discovery work."""
    assert home_config_path() == isolated_home / "minicode_config.json"
    proj = tmp_path / "proj"
    deep = proj / "src" / "foo"
    deep.mkdir(parents=True)
    _write(proj / "minicode_config.json", {})
    monkeypatch.chdir(deep)
    assert default_config_path() == proj / "minicode_config.json"


def test_agent_defaults() -> None:
    """Schema ships working coding-agent defaults."""
    defaults = MiniCodeConfig().agents.defaults
    assert defaults.workspace == "~/.minicode/workspace"
    assert defaults.model == "gpt-4o"
    assert defaults.max_tokens == 8192
    assert defaults.temperature == 0.7
    assert defaults.max_tool_iterations == 20


def test_workspace_path_expands_home() -> None:
    """workspace_path resolves the home directory."""
    config = MiniCodeConfig()
    assert config.workspace_path == Path("~/.minicode/workspace").expanduser()


def test_provider_entries_exist() -> None:
    """Only the registered providers are configured."""
    providers = MiniCodeConfig().providers
    assert providers.openai is not None
    assert providers.anthropic is not None


def test_no_key_matches_nothing() -> None:
    """Without API keys no provider is ever faked."""
    config = MiniCodeConfig()
    assert config.get_provider_name("gpt-4o") is None
    assert config.get_provider("gpt-4o") is None
    assert config.get_api_key("gpt-4o") is None


def test_keyed_model_matching() -> None:
    """Keyword matching follows the provider registry."""
    config = MiniCodeConfig()
    config.providers.openai.api_key = "sk-test-openai"
    config.providers.anthropic.api_key = "sk-test-anthropic"
    assert config.get_provider_name("gpt-4o") == "openai"
    assert config.get_provider_name("claude-sonnet-4") == "anthropic"
    assert config.get_api_key("gpt-4o") == "sk-test-openai"
    assert config.get_provider_name("unknown-model-zzz") is None


def test_api_base_explicit_wins() -> None:
    """Explicit api_base beats registry defaults."""
    config = MiniCodeConfig()
    config.providers.openai.api_key = "sk-test-openai"
    assert config.get_api_base("gpt-4o") is None
    config.providers.openai.api_base = "https://proxy.example.com/v1"
    assert config.get_api_base("gpt-4o") == "https://proxy.example.com/v1"
