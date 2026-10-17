"""Tests for the minimal interactive CLI."""

import builtins
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from agent import AgentLoop
from providers.base import LLMProvider, LLMResponse, ToolCallRequest
from tui.cli import Cli, summarize_call, summarize_result
from tui.history import load_history_entries, save_history_entries


class FakeProvider(LLMProvider):
    """Scripted provider for CLI tests."""

    def __init__(self, script: list[LLMResponse | Exception]) -> None:
        super().__init__()
        self.script = list(script)
        self.seen: list[list[dict[str, Any]]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        self.seen.append(list(messages))
        next_item = self.script.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return next_item

    def get_default_model(self) -> str:
        return "fake-model"


def _make_cli(
    script: list[LLMResponse | Exception],
    workspace: Path,
    session_id: str = "cli-test",
) -> tuple[Cli, FakeProvider, Console]:
    """Build a CLI with a scripted provider and recording console."""
    provider = FakeProvider(script)
    agent = AgentLoop(provider=provider, workspace=workspace)
    console = Console(record=True, width=100)
    return Cli(agent=agent, console=console, session_id=session_id), provider, console


@pytest.mark.asyncio
async def test_agent_message_renders_markdown(
    isolated_home: Path, tmp_path: Path
) -> None:
    """Full non-streaming reply renders as Markdown."""
    cli, _, console = _make_cli([LLMResponse(content="# Title\n\n**bold**")], tmp_path)
    assert await cli.run_once("hi") == "# Title\n\n**bold**"
    output = console.export_text()
    assert "Title" in output
    assert "bold" in output


@pytest.mark.asyncio
async def test_tool_callbacks_display(isolated_home: Path, tmp_path: Path) -> None:
    """Tool start and completion render the tool name and success mark."""
    cli, _, console = _make_cli(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(id="c1", name="bash", arguments={"command": "ls"})
                ],
            ),
            LLMResponse(content="done"),
        ],
        tmp_path,
    )
    assert await cli.run_once("go") == "done"
    output = console.export_text()
    assert "bash" in output
    assert "●" in output
    assert "✓" in output
    assert "⎿" in output


@pytest.mark.asyncio
async def test_tool_error_displays_failure(
    isolated_home: Path, tmp_path: Path
) -> None:
    """Failed tools render the error mark and failure text."""
    cli, _, console = _make_cli(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(id="c9", name="nope", arguments={})
                ],
            ),
            LLMResponse(content="recovered"),
        ],
        tmp_path,
    )
    assert await cli.run_once("go") == "recovered"
    output = console.export_text()
    assert "✗" in output
    assert "failed" in output


def test_tool_result_first_line_capped_at_100(
    isolated_home: Path, tmp_path: Path
) -> None:
    """Only the first result line renders, capped at 100 chars."""
    cli, _, console = _make_cli([LLMResponse(content="unused")], tmp_path)
    cli._on_tool_completed("bash", "x" * 150 + "\nsecond line", False)
    output = console.export_text()
    assert "x" * 100 in output
    assert "x" * 101 not in output
    assert "…" in output
    assert "second line" not in output


def test_tool_started_shows_short_detail(
    isolated_home: Path, tmp_path: Path
) -> None:
    """Start detail comes from the shared call summary, capped at 80."""
    cli, _, console = _make_cli([LLMResponse(content="unused")], tmp_path)
    cli._on_tool_started("bash", {"command": "y" * 120})
    output = console.export_text()
    assert "bash" in output
    assert "y" * 80 in output
    assert "y" * 81 not in output
    assert "…" in output


def test_summarize_call_preference_order() -> None:
    """Preferred keys win and long values cap at 80 chars."""
    assert summarize_call("bash", {"command": "ls", "file_path": "a.txt"}) == "ls"
    assert summarize_call("read", {"file_path": "a.txt"}) == "a.txt"
    assert summarize_call("grep", {"pattern": "needle"}) == "needle"
    assert summarize_call("bash", {"command": "x" * 120}) == "x" * 80 + "…"


def test_summarize_call_empty_and_fallback() -> None:
    """Empty inputs give empty strings; unknown keys fall back."""
    assert summarize_call("bash", None) == ""
    assert summarize_call("bash", {}) == ""
    assert summarize_call("bash", "nope") == ""  # type: ignore[arg-type]
    assert summarize_call("custom", {"other": 42}) == "42"


def test_summarize_result_collapses_and_defensive() -> None:
    """Results collapse past the limit; any value becomes a string."""
    assert summarize_result("  hi  ") == "hi"
    assert summarize_result(None) == "None"
    assert summarize_result(42) == "42"
    collapsed = summarize_result("y" * 500)
    assert len(collapsed) < 500
    assert collapsed.endswith(" …")
    assert summarize_result("ok", limit=10) == "ok"


@pytest.mark.asyncio
async def test_multi_round_keeps_session(
    isolated_home: Path, tmp_path: Path
) -> None:
    """Two turns share one session and both replies return."""
    cli, provider, _ = _make_cli(
        [LLMResponse(content="first"), LLMResponse(content="second")], tmp_path
    )
    assert await cli.run_once("one") == "first"
    assert await cli.run_once("two") == "second"
    assert len(provider.seen) == 2
    session = cli.agent.sessions.get_session(cli.session_id)
    assert session is not None
    contents = [m["content"] for m in session.messages]
    assert "one" in contents
    assert "two" in contents


@pytest.mark.asyncio
async def test_error_handling_calls_on_error(
    isolated_home: Path, tmp_path: Path
) -> None:
    """Provider exceptions surface via _on_error without raising."""
    cli, _, console = _make_cli([RuntimeError("down")], tmp_path)
    assert await cli.run_once("hi") == ""
    output = console.export_text()
    assert "error" in output.lower()
    assert "down" in output


def test_run_exits_cleanly_and_saves_history(
    isolated_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EOF and Ctrl+C exit the loop; inputs persist via history.py."""
    cli, _, console = _make_cli([LLMResponse(content="hi")], tmp_path)
    inputs = iter(["hello", EOFError()])

    def _fake_next(prompt: str = "") -> str:
        item = next(inputs)
        if isinstance(item, Exception):
            raise item
        return str(item)

    monkeypatch.setattr(builtins, "input", _fake_next)
    cli.run()
    assert load_history_entries() == ["hello"]
    assert "hi" in console.export_text()

    cli2, _, _ = _make_cli([LLMResponse(content="unused")], tmp_path)

    def _raise_interrupt(_prompt: str = "") -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(builtins, "input", _raise_interrupt)
    cli2.run()


def test_cli_restores_saved_input_history_to_readline(
    isolated_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Saved entries are available to the terminal's up-arrow history."""
    class FakeReadline:
        def __init__(self) -> None:
            self.entries: list[str] = []
            self.auto_history: bool | None = None

        def set_auto_history(self, enabled: bool) -> None:
            self.auto_history = enabled

        def clear_history(self) -> None:
            self.entries.clear()

        def add_history(self, entry: str) -> None:
            self.entries.append(entry)

    readline = FakeReadline()
    monkeypatch.setattr("tui.cli._readline", readline)
    save_history_entries(["older command", "newer command"])

    cli, _, _ = _make_cli([LLMResponse(content="unused")], tmp_path)

    assert readline.auto_history is False
    assert readline.entries == ["older command", "newer command"]

    inputs = iter(["new command", EOFError()])

    def fake_input(_prompt: str = "") -> str:
        item = next(inputs)
        if isinstance(item, Exception):
            raise item
        return str(item)

    monkeypatch.setattr(builtins, "input", fake_input)
    cli.run()

    assert readline.entries == ["older command", "newer command", "new command"]
    assert load_history_entries() == readline.entries
