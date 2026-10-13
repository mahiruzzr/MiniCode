"""Regression tests for dynamic text rendered through Rich markup."""

import pytest
from rich.console import Console

from tui.cli import Cli


def _renderer() -> Cli:
    """Build a CLI shell for exercising render callbacks only."""
    cli = object.__new__(Cli)
    cli.console = Console(record=True, width=100)
    cli._line_open = False
    return cli


@pytest.mark.parametrize("render", ["tool detail", "error message"])
def test_bracketed_dynamic_text_is_displayed_literally(render: str) -> None:
    """Tool details and errors preserve square brackets in their output."""
    cli = _renderer()
    text = "[not-a-rich-style]"
    if render == "tool detail":
        cli._on_tool_started("bash", {"command": text})
    else:
        cli._on_error(text)

    assert text in cli.console.export_text()


def test_tool_names_are_displayed_literally() -> None:
    """A dynamic tool name cannot inject Rich formatting."""
    cli = _renderer()
    name = "[bold]bash[/bold]"

    cli._on_tool_started(name, {"command": "ls"})

    assert name in cli.console.export_text()


def test_tool_result_names_and_content_are_displayed_literally() -> None:
    """Completion names and result previews retain markup-looking text."""
    cli = _renderer()
    name = "[italic]bash[/italic]"
    result = "[bold]result[/bold]"

    cli._on_tool_completed(name, result, False)

    output = cli.console.export_text()
    assert name in output
    assert result in output
