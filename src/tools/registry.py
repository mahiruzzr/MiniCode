"""Registry and executor for agent tools."""

from typing import Any

from .base import Tool


class ToolRegistry:
    """Registry and executor for agent tools."""

    def __init__(self) -> None:
        """Initialize an empty tool registry."""
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool by name.

        Args:
            tool: Tool instance to register.
        """
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get a tool by name.

        Args:
            name: Tool name.

        Returns:
            Tool instance, or None if not found.
        """
        return self._tools.get(name)

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool schemas for the LLM layer.

        Returns:
            List of tool schemas in OpenAI function format.
        """
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(self, name: str, params: dict[str, Any]) -> str:
        """Execute a tool by name.

        Args:
            name: Tool name.
            params: Tool parameters.

        Returns:
            Tool result or error message.
        """
        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found"

        try:
            errors = tool.validate_params(params)
            if errors:
                return f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors)
            return await tool.execute(**params)
        except Exception as e:
            return f"Error executing {name}: {str(e)}"
