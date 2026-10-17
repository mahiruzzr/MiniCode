"""Minimal non-streaming interactive CLI for MiniCode."""

from __future__ import annotations

import asyncio
from typing import Any

try:
    import readline as _readline
except ImportError:  # pragma: no cover - unavailable on some platforms
    _readline = None

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape

from agent import AgentLoop
from config.loader import load_config
from providers.litellm import LiteLLMProvider
from tui import theme
from tui.history import load_history_entries, save_history_entries


def summarize_call(name: str, arguments: dict[str, Any] | None) -> str:
    """One-line argument summary for a tool call (pure, best-effort).

    Picks the most informative argument by a preference order shared by all
    tools (command for bash, path for file tools, pattern for search), so
    frontends render consistent cards without per-tool special cases.
    """
    if not isinstance(arguments, dict) or not arguments:
        return ""
    for key in ("command", "file_path", "pattern", "prompt", "patch", "text"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            first_line = value.strip().splitlines()[0]
            return first_line[:80] + ("…" if len(first_line) > 80 else "")
    value = next(iter(arguments.values()))
    text = str(value).strip().splitlines()[0] if value is not None else ""
    return text[:80] + ("…" if len(text) > 80 else "")


def summarize_result(result: Any, limit: int = 400) -> str:
    """A compact, single-block preview of a tool result for display.

    Collapses to the leading ``limit`` characters with an elision marker.
    Pure and defensive: any value becomes a string, never raises.
    """
    text = result if isinstance(result, str) else str(result)
    text = text.strip()
    if len(text) > limit:
        return text[:limit].rstrip() + " …"
    return text


class Cli:
    """Minimal non-streaming interactive CLI."""

    def __init__(
        self,
        agent: AgentLoop,
        console: Console | None = None,
        session_id: str | None = None,
    ) -> None:
        """Initialize the CLI.

        Args:
            agent: Configured agent loop to run each turn.
            console: Rich console. A new one is created when omitted.
            session_id: Session to reuse across turns. A new session
                is created when omitted so multi-round history works.
        """
        self.agent = agent
        self.console = console if console is not None else Console()
        self._line_open = False
        self._history = load_history_entries()
        self._restore_readline_history()
        if session_id is None:
            session_id = agent.sessions.create_session().session_id
        self.session_id = session_id

    def _restore_readline_history(self) -> None:
        """Load persisted entries into the terminal's interactive history."""
        if _readline is None:
            return
        if hasattr(_readline, "set_auto_history"):
            _readline.set_auto_history(False)
        _readline.clear_history()
        for entry in self._history:
            _readline.add_history(entry)

    def _close_line(self) -> None:
        if self._line_open:
            self.console.print()
            self._line_open = False

    def _on_agent_message(self, text: str | None) -> None:
        """Render the full non-streaming agent reply as Markdown.

        Args:
            text: Complete reply text for this turn.
        """
        self._close_line()
        self.console.print(Markdown(text or ""))
        self.console.print()

    def _on_tool_started(self, name: str, tool_input: dict[str, Any]) -> None:
        """Show the tool name and a short detail line.

        Args:
            name: Tool name.
            tool_input: Tool arguments from the agent loop.
        """
        self._close_line()
        detail = summarize_call(name, tool_input)
        label = (
            f"[{theme.TOOL_RUNNING_STYLE}]{theme.TOOL_BULLET}[/] "
            f"[bold]{escape(name)}[/]"
        )
        if detail:
            label += f"[{theme.TOOL_DETAIL_STYLE}]({escape(detail)})[/]"
        self.console.print(label, highlight=False)

    def _on_tool_completed(self, name: str, result: str, is_error: bool) -> None:
        """Show the success or failure mark plus the first result line.

        Args:
            name: Tool name.
            result: Tool output text.
            is_error: Whether the tool reported an error.
        """
        self._close_line()
        if is_error:
            mark = f"[{theme.TOOL_ERR_STYLE}]{theme.DONE_ERR} {escape(name)} failed[/]"
        else:
            mark = (
                f"[{theme.TOOL_OK_STYLE}]{theme.DONE_OK}[/] "
                f"[{theme.META_STYLE}]{escape(name)}[/]"
            )
        preview = summarize_result(result)
        first_line = preview.splitlines()[0] if preview else ""
        if first_line:
            snippet = first_line[:100] + ("…" if len(first_line) > 100 else "")
            mark += f"  [{theme.TOOL_DETAIL_STYLE}]{escape(snippet)}[/]"
        self.console.print(f"{theme.TOOL_RESULT_PREFIX} {mark}", highlight=False)

    def _on_error(self, message: str) -> None:
        """Show an error in the shared error style.

        Args:
            message: Error message text.
        """
        self._close_line()
        self.console.print(f"[{theme.ERROR_STYLE}]error:[/] {escape(message)}")

    async def run_once(self, text: str) -> str:
        """Run one non-streaming turn and render the result.

        Args:
            text: User input for this turn.

        Returns:
            The agent reply, or "" when the turn failed.
        """
        try:
            reply = await self.agent.run(
                text,
                session_id=self.session_id,
                on_tool_start=self._on_tool_started,
                on_tool_result=self._on_tool_completed,
            )
        except Exception as exc:  # noqa: BLE001 - surface any failure to the CLI
            self._on_error(str(exc))
            return ""
        self._on_agent_message(reply)
        return reply

    def run(self) -> None:
        """Start the read-eval loop (one event loop per session)."""
        asyncio.run(self.arun())

    async def arun(self) -> None:
        """Read-eval loop until Ctrl+C or EOF.

        A single event loop serves the whole session so LiteLLM's
        background logging worker never sees a loop change (which
        otherwise logs "event loop changed... carried N pending...").
        """
        while True:
            try:
                line = await asyncio.to_thread(input, theme.PROMPT)
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                break
            if not line.strip():
                continue
            self._history.append(line)
            save_history_entries(self._history)
            if _readline is not None:
                _readline.add_history(line)
            self.console.print()
            await self.run_once(line)


def main() -> None:
    """Build the default agent from disk config and start the REPL."""
    config = load_config()
    model = config.agents.defaults.model
    provider_config = config.get_provider(model)
    provider = LiteLLMProvider(
        api_key=config.get_api_key(model),
        api_base=config.get_api_base(model),
        default_model=model,
        extra_headers=provider_config.extra_headers if provider_config else None,
        provider_name=config.get_provider_name(model),
    )
    workspace = config.workspace_path
    workspace.mkdir(parents=True, exist_ok=True)
    agent = AgentLoop(
        provider=provider,
        workspace=workspace,
        model=model,
        max_iterations=config.agents.defaults.max_tool_iterations,
        reasoning_effort=config.agents.defaults.reasoning_effort,
    )
    Cli(agent=agent).run()


if __name__ == "__main__":
    main()
