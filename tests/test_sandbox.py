"""Tests for sandbox command wrapping and configuration."""

from pathlib import Path

import pytest

import sandbox
from sandbox import (
    SandboxPolicy,
    build_exec_command,
    sandbox_enabled,
    wrap_argv_command,
)


@pytest.mark.parametrize("value", ["0", "false", "off", "no", "FALSE"])
def test_minicode_sandbox_env_disables(monkeypatch, value: str) -> None:
    """The MiniCode environment variable disables command wrapping."""
    monkeypatch.setenv("MINICODE_SANDBOX", value)
    assert sandbox_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", ""])
def test_minicode_sandbox_env_enables(monkeypatch, value: str) -> None:
    """Truthy values leave sandboxing enabled."""
    monkeypatch.setenv("MINICODE_SANDBOX", value)
    assert sandbox_enabled() is True


def test_sandbox_enabled_by_default(monkeypatch) -> None:
    """Sandboxing is enabled when the environment variable is absent."""
    monkeypatch.delenv("MINICODE_SANDBOX", raising=False)
    assert sandbox_enabled() is True


def test_build_exec_command_honors_minicode_env(tmp_path, monkeypatch) -> None:
    """Disabling sandboxing returns the bare shell command."""
    monkeypatch.setenv("MINICODE_SANDBOX", "off")
    wrapped = build_exec_command(command="echo hi", workspace=tmp_path)
    assert wrapped.backend == "disabled"
    assert wrapped.argv == ["/bin/bash", "-c", "echo hi"]
    assert wrapped.cleanup_path is None


def test_seatbelt_profile_uses_minicode_prefix(tmp_path, monkeypatch) -> None:
    """Seatbelt profiles use the MiniCode temp prefix and are cleanable."""
    monkeypatch.setattr(sandbox, "sandbox_backend", lambda: "seatbelt")
    policy = SandboxPolicy.for_workspace(tmp_path)
    wrapped = wrap_argv_command(["echo", "hi"], policy)

    assert wrapped.backend == "seatbelt"
    assert wrapped.cleanup_path is not None
    profile = Path(wrapped.cleanup_path)
    assert profile.name.startswith("minicode-sb-")
    assert profile.suffix == ".sbpl"
    assert "(allow file-read*)" in profile.read_text(encoding="utf-8")

    wrapped.cleanup()
    assert not profile.exists()


def test_bwrap_wrapping_preserves_workspace_and_network_policy(
    tmp_path, monkeypatch
) -> None:
    """Bubblewrap gets the workspace bind and network isolation flag."""
    monkeypatch.setattr(sandbox, "sandbox_backend", lambda: "bwrap")
    wrapped = build_exec_command(
        command="echo hi",
        workspace=tmp_path,
        allow_network=False,
    )

    assert wrapped.backend == "bwrap"
    assert "--bind" in wrapped.argv
    assert str(tmp_path) in wrapped.argv
    assert "--unshare-net" in wrapped.argv
    assert wrapped.argv[-3:] == ["/bin/bash", "-c", "echo hi"]
