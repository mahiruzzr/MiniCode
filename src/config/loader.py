"""Read-only configuration loader for MiniCode."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .schema import MiniCodeConfig


_DEFAULT_CONFIG_FILENAME = "minicode_config.json"
_MINICODE_HOME_ENV = "MINICODE_HOME"

_ENV_REF_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(ValueError):
    """Configuration error that requires user correction."""


def minicode_home() -> Path:
    """Return MiniCode's user home directory.

    Returns:
        MINICODE_HOME when set, otherwise ~/.minicode.
    """
    override = os.environ.get(_MINICODE_HOME_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".minicode").expanduser().resolve()


def home_config_path() -> Path:
    """Return the user-level config file path.

    Returns:
        Path to minicode_config.json under minicode_home().
    """
    return minicode_home() / _DEFAULT_CONFIG_FILENAME


def default_config_path() -> Path:
    """Return the default project config file path.

    Returns:
        Config file path resolved from the current directory.
    """
    return _resolve_workspace_path() / _DEFAULT_CONFIG_FILENAME


def _resolve_workspace_path(start: Path | None = None) -> Path:
    """Find the nearest directory holding a config file.

    Args:
        start: Directory to start searching from. Defaults to the
            current working directory.

    Returns:
        First ancestor directory holding the config file, or the
        start directory when none is found.
    """
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / _DEFAULT_CONFIG_FILENAME).is_file():
            return candidate
    return current


def _load_raw(path: Path) -> dict[str, Any]:
    """Load raw config data from a JSON file.

    Args:
        path: Config file path.

    Returns:
        Parsed object, or {} when the file does not exist.

    Raises:
        ConfigError: When the file is not valid JSON or its root
            is not an object.
    """
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
    except json.JSONDecodeError as error:
        raise ConfigError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(data, dict):
        raise ConfigError(f"Config root in {path} must be an object")
    return data


def _deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """Merge the override layer into the base layer recursively.

    Args:
        base: Lower-priority config layer.
        override: Higher-priority config layer.

    Returns:
        New merged dict; inputs are not mutated.
    """
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _resolve_env_refs(
    value: Any,
    *,
    path: str = "",
) -> Any:
    """Resolve ${ENV_VAR} references recursively.

    Args:
        value: Config value, possibly nested.
        path: Config location used in error messages.

    Returns:
        Value with all references replaced.

    Raises:
        ConfigError: When a referenced variable is not set.
    """
    if isinstance(value, str):

        def _replace(match: re.Match[str]) -> str:
            name = match.group(1)
            resolved = os.environ.get(name)
            if resolved is None:
                location = f" at {path}" if path else ""
                raise ConfigError(
                    f"Environment variable '{name}' referenced in minicode_config.json{location} is not set"
                )
            return resolved
         
        return _ENV_REF_PATTERN.sub(_replace, value)

    if isinstance(value, dict):
        return {
            key: _resolve_env_refs(item, path=f"{path}.{key}" if path else key)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_env_refs(item, path=f"{path}[{i}]") for i, item in enumerate(value)
        ]
    return value


def load_config(
    config_path: str | Path | None = None,
) -> MiniCodeConfig:
    """Load MiniCode configuration from disk.

    Args:
        config_path: Single config file to load. When omitted, the
            user config and project config are merged with the
            project layer taking precedence.

    Returns:
        Validated MiniCodeConfig, using schema defaults when no
        config file exists.
    """
    if config_path is not None:
        raw = _load_raw(Path(config_path).expanduser().resolve())
    else:
        base = _load_raw(home_config_path())  # user-level, cwd-independent
        project = _load_raw(default_config_path())  # cwd-scoped override
        raw = _deep_merge(base, project)

    raw = _resolve_env_refs(raw)
    return MiniCodeConfig.model_validate(raw)
