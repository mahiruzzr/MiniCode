"""Context builder for assembling agent prompts."""

from pathlib import Path
from typing import Any

from .memory import system_preamble
from .skills import SkillRegistry, discover_skills, skills_preamble

_BASE_PROMPT = (
    "You are MiniCode, a coding agent working in a workspace directory. "
    "The write, edit, and apply_patch tools reject paths outside it; bash "
    "runs with the workspace as its working directory.\n\n"
    "Use grep or glob to locate code, then read the complete region you "
    "intend to change. read is paginated, so keep reading until you have "
    "enough context. Parallelize independent reads and searches; wait for "
    "their results before dependent steps. Keep the change minimal and "
    "consistent with the surrounding code; do not refactor, rename, or "
    "reformat unrelated code.\n\n"
    "Use edit for a focused change, write only for a new file or a full "
    "rewrite, and apply_patch when one change spans several files or must "
    "land atomically. Run shell commands non-interactively; do not start "
    "commands that wait for input.\n\n"
    "Verify with the narrowest check that can fail: a focused test, type "
    "check, build, or run. Then run the broader suite when it is cheap and "
    "relevant. If a command fails, read the error and change your approach; "
    "do not repeat an identical failing call. When write, edit, or "
    "apply_patch reports diagnostics, fix reported errors before continuing."
    "\n\n"
    "Act with tools instead of narrating the next step. A turn ends when "
    "you answer without a tool call, and each request allows only a limited "
    "number of tool rounds. Finish with a short summary of what changed, "
    "where, and what you ran. Report checks you did not run and any "
    "remaining blocker. Never claim a check passed unless you ran it."
)


class ContextBuilder:
    """Build the session-stable system prompt and per-turn messages."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.skills: SkillRegistry = discover_skills(workspace)
        parts = (
            _BASE_PROMPT,
            system_preamble(workspace),
            skills_preamble(self.skills),
        )
        self._system_prompt = "\n\n".join(part for part in parts if part)

    def build_system_prompt(self) -> str:
        """Return the session-stable system prompt."""
        return self._system_prompt

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
    ) -> list[dict[str, Any]]:
        """Return system, history, and current user messages."""
        return [
            {"role": "system", "content": self.build_system_prompt()},
            *history,
            {"role": "user", "content": current_message},
        ]

    def add_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> list[dict[str, Any]]:
        """Append a tool result and return the message list."""
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": result,
            }
        )
        return messages

    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
        thinking_blocks: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Append an assistant message and return the message list."""
        msg: dict[str, Any] = {"role": "assistant", "content": content or ""}

        if tool_calls:
            msg["tool_calls"] = tool_calls
        if reasoning_content:
            msg["reasoning_content"] = reasoning_content
        if thinking_blocks:
            msg["thinking_blocks"] = thinking_blocks

        messages.append(msg)
        return messages
