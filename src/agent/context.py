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
            "You are MiniCode, a coding agent working in a workspace directory. "
            "Navigate with grep/glob, inspect with read, make targeted changes with "
            "edit (or write for new files). Run commands and tests with bash. "
            "For multi-step tasks, work through them in order and keep track of "
            "what remains. After a write or edit, check the tool result for reported "
            "errors and fix them. When the task is done, reply with a short summary.",
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

        messages.append(msg)
        return messages
