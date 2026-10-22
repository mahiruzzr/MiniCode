"""Maintain the agent's TODO plan for a multi-step task.

The agent supplies a non-empty, complete list of steps, each with a
``step`` label and a ``status`` of ``pending``, ``in_progress``, or
``completed``. Calling the tool again replaces the previous plan. At most
one step may be ``in_progress`` at a time.

The plan is kept in memory and is not persisted; this tool does not read
or modify the workspace. A successful update returns a rendered checklist,
while invalid input returns an error message.
"""

from __future__ import annotations

from typing import Any

from .base import Tool

_STATUSES = ("pending", "in_progress", "completed")
_GLYPH = {"pending": "☐", "in_progress": "▶", "completed": "☑"}


class UpdatePlanTool(Tool):
    """An in-memory TODO plan with at most one in-progress step."""

    def __init__(self) -> None:
        self._plan: list[dict[str, str]] = []

    @property
    def name(self) -> str:
        return "update_plan"

    @property
    def description(self) -> str:
        return (
            "Maintain a short TODO plan for a multi-step task. Provide the "
            "full list of steps (each with status pending | in_progress | "
            "completed) and an optional one-line explanation; call again to "
            "update progress. At most one step may be in_progress at a time. "
            "Use it to plan and track your own work on non-trivial tasks."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string",
                    "description": (
                        "Optional one-line note about this plan update."
                    ),
                },
                "plan": {
                    "type": "array",
                    "description": (
                        "The full list of steps (replaces the previous plan)."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {
                                "type": "string",
                                "description": "Task step text.",
                            },
                            "status": {
                                "type": "string",
                                "enum": list(_STATUSES),
                                "description": "Step status.",
                            },
                        },
                        "required": ["step", "status"],
                    },
                },
            },
            "required": ["plan"],
        }

    @property
    def plan(self) -> list[dict[str, str]]:
        """Return a shallow copy of the current plan."""
        return list(self._plan)

    async def execute(self, **kwargs: Any) -> Any:
        """Validate and store a plan, returning its checklist or an error.

        Invalid input leaves the previously stored plan unchanged.
        """
        raw = kwargs.get("plan")
        if not isinstance(raw, list) or not raw:
            return (
                "Error: 'plan' must be a non-empty list of {step, status} "
                "items."
            )

        steps: list[dict[str, str]] = []
        in_progress = 0
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                return (
                    f"Error: plan[{i}] must be an object with 'step' and "
                    "'status'."
                )
            step = str(item.get("step", "")).strip()
            status = str(item.get("status", "")).strip()
            if not step:
                return f"Error: plan[{i}].step is required."
            if status not in _STATUSES:
                return (
                    f"Error: plan[{i}].status must be one of "
                    f"{', '.join(_STATUSES)} (got {status!r})."
                )
            if status == "in_progress":
                in_progress += 1
            steps.append({"step": step, "status": status})
        if in_progress > 1:
            return "Error: at most one step may be in_progress at a time."

        self._plan = steps
        return self._render(str(kwargs.get("explanation") or "").strip())

    def _render(self, explanation: str) -> str:
        """Render the stored plan with a completion count."""
        done = sum(1 for s in self._plan if s["status"] == "completed")
        head = f"Plan updated ({done}/{len(self._plan)} done)"
        if explanation:
            head += f" — {explanation}"
        body = "\n".join(
            f"{_GLYPH[s['status']]} {s['step']}" for s in self._plan
        )
        return f"{head}\n{body}"
