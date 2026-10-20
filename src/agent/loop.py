"""Agent loop: the core processing engine."""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent.context import ContextBuilder
from providers.base import LLMProvider
from session.manager import SessionManager
from tools import (    
    BashTool,
    EditTool,
    GlobTool,
    GrepTool,
    ReadTool,
    ToolRegistry,
    WriteTool,
)

from .memory import MemoryTool
from .skills import SkillTool

logger = logging.getLogger(__name__)

_MICROCOMPACT_KEEP_RECENT = 10
_MICROCOMPACT_MIN_CHARS = 500
_COMPACTABLE_TOOLS = frozenset({"bash", "read", "grep", "glob"})


class AgentLoop:
    """The agent loop is the core processing engine.

    It:
    1. Takes the user content and session
    2. Builds messages with history
    3. Calls the LLM
    4. Executes tool calls
    5. Returns the final response
    """

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        session_manager: SessionManager | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        """Initialize the agent loop.

        Args:
            provider: Already-configured LLM provider to call.
            workspace: Agent working directory for tools.
            model: Model identifier. Defaults to the provider default.
            max_iterations: Maximum provider calls per run.
            session_manager: Session manager. A new one is created
                when omitted.
            reasoning_effort: Reasoning effort hint forwarded to the
                provider on every chat call.
        """
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.reasoning_effort = reasoning_effort
        self.sessions = session_manager or SessionManager()
        self.context = ContextBuilder(workspace)
        self.tools = ToolRegistry()
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        self.tools.register(ReadTool(workspace=self.workspace))
        self.tools.register(WriteTool(workspace=self.workspace))
        self.tools.register(EditTool(workspace=self.workspace))
        self.tools.register(BashTool(workspace=self.workspace))
        self.tools.register(GrepTool(workspace=self.workspace))
        self.tools.register(GlobTool(workspace=self.workspace))
        self.tools.register(SkillTool(self.context.skills))
        self.tools.register(MemoryTool(workspace=self.workspace))

    @staticmethod
    def _microcompact(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compactable_indices: list[int] = []
        for idx, msg in enumerate(messages):
            if msg.get("role") == "tool" and msg.get("name") in _COMPACTABLE_TOOLS:
                compactable_indices.append(idx)

        if len(compactable_indices) <= _MICROCOMPACT_KEEP_RECENT:
            return messages

        stale = compactable_indices[
            : len(compactable_indices) - _MICROCOMPACT_KEEP_RECENT
        ]
        updated: list[dict[str, Any]] | None = None
        for idx in stale:
            msg = messages[idx]
            content = msg.get("content")
            if not isinstance(content, str) or len(content) < _MICROCOMPACT_MIN_CHARS:
                continue
            name = msg.get("name", "tool")
            summary = f"[{name} result omitted from context]"
            if updated is None:
                updated = [dict(m) for m in messages]
            updated[idx]["content"] = summary

        return updated if updated is not None else messages

    async def run(
        self,
        content: str,
        session_id: str | None = None,
        *,
        on_tool_start: Callable[[str, dict[str, Any]], None] | None = None,
        on_tool_result: Callable[[str, str, bool], None] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> str:
        """Process a user message and return the final response.

        Args:
            content: The user message content.
            session_id: Existing session identifier. A new session is
                created when omitted or unknown.
            on_tool_start: Optional callback invoked with
                (tool_name, tool_input) before each tool executes.
            on_tool_result: Optional callback invoked with
                (tool_name, output, is_error) after each tool executes.
            tool_choice: Tool selection policy forwarded to the
                provider on every round. Provider defaults apply when
                omitted.

        Returns:
            The final assistant response.
        """
        # Resolve the session this run belongs to.
        if session_id is None:
            session = self.sessions.create_session()
        else:
            session = self.sessions.get_session(session_id)
            if session is None:
                session = self.sessions.create_session(session_id=session_id)

        history = session.get_history()
        messages = self.context.build_messages(
            history=history,
            current_message=content,
        )
        session.add_message("user", content)

        iteration = 0
        final_content = None

        # Each iteration is one provider round, even with multiple tool calls.
        while iteration < self.max_iterations:
            iteration += 1

            response = await self.provider.chat(
                messages=self._microcompact(messages),
                tools=self.tools.get_definitions(),
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                tool_choice=tool_choice,
            )

            if response.has_tool_calls:
                # Record the assistant turn that requested tools.
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            # Must be JSON string
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages,
                    response.content,
                    tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                session.add_message(
                    "assistant",
                    messages[-1]["content"],
                    tool_calls=tool_call_dicts,
                )

                # Execute each call and keep its result.
                for tool_call in response.tool_calls:
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")
                    if on_tool_start:
                        on_tool_start(tool_call.name, tool_call.arguments)
                    result = await self.tools.execute(
                        tool_call.name, tool_call.arguments
                    )
                    is_error = result.lstrip().startswith("Error")
                    if on_tool_result:
                        on_tool_result(tool_call.name, result, is_error)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
                    session.add_message(
                        "tool",
                        result,
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                    )
            else:
                final_content = response.content
                break

        if final_content is None:
            final_content = "I've completed processing but have no response to give."
        session.add_message("assistant", final_content)

        self.sessions.save(session)
        return final_content
