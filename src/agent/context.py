"""Context builder for assembling agent prompts."""

from pathlib import Path
from typing import Any

from .memory import system_preamble
from .skills import SkillRegistry, discover_skills, skills_preamble


class ContextBuilder:
    """Builds the context (system prompt + messages) for the agent."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.memory_preamble = system_preamble(workspace)
        self.skills: SkillRegistry = discover_skills(workspace)

    def build_system_prompt(self) -> str:
        """Build MiniCode's system prompt."""
        parts = [
            (
                "You are MiniCode, a coding agent working in a workspace "
                "directory. All file tools are fenced to the workspace and "
                "reject paths outside it. Locate the relevant code with grep "
                "or glob, read the files before changing them, then modify: "
                "use edit for targeted changes, giving old_string enough "
                "context to be unique, reserve write for new files or full "
                "rewrites, and use apply_patch when one change spans several "
                "files or several places that must land together. Run "
                "commands and tests with bash, preferring non-interactive "
                "commands. A turn allows a bounded number of tool rounds, so "
                "make each call move the task forward instead of browsing. "
                "After a write, edit, or apply_patch the result may contain a "
                "'Diagnostics detected' block; fix every reported error "
                "before moving on. When the task is done, reply with a short "
                "summary of what changed."
            ),
            self.memory_preamble,
            skills_preamble(self.skills),
        ]

        return "\n\n".join(part for part in parts if part)

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
    ) -> list[dict[str, Any]]:
        """Build the complete message list for an LLM call.

        Args:
            history: Previous conversation messages.
            current_message: The new user message.

        Returns:
            List of messages including system prompt.
        """
        messages = []
        messages.append(
            {"role": "system", "content": self.build_system_prompt()}
        )
        messages.extend(history)
        messages.append({"role": "user", "content": current_message})
        return messages

    def add_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> list[dict[str, Any]]:
        """Add a tool result to the message list.

        Args:
            messages: Current message list.
            tool_call_id: ID of the tool call.
            tool_name: Name of the tool.
            result: Tool execution result.

        Returns:
            Updated message list.
        """
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
        thinking_blocks: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Add an assistant message to the message list.

        Args:
            messages: Current message list.
            content: Message content.
            tool_calls: Optional tool calls.
            reasoning_content: Thinking model output.

        Returns:
            Updated message list.
        """
        msg: dict[str, Any] = {"role": "assistant", "content": content or ""}

        if tool_calls:
            msg["tool_calls"] = tool_calls

        if reasoning_content:
            msg["reasoning_content"] = reasoning_content

        if thinking_blocks:
            msg["thinking_blocks"] = thinking_blocks

        messages.append(msg)
        return messages
