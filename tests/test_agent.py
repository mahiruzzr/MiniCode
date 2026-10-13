"""Tests for the AgentLoop data flow with a fake provider."""

from pathlib import Path
from typing import Any

import pytest

from agent import AgentLoop
from providers.base import LLMProvider, LLMResponse, ToolCallRequest
from session.manager import SessionManager


class FakeProvider(LLMProvider):
    """Scripted provider that records every request."""

    def __init__(self, script: list[LLMResponse]) -> None:
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
    ) -> LLMResponse:
        self.seen.append(list(messages))
        return self.script.pop(0)

    def get_default_model(self) -> str:
        return "fake-model"


def _call(call_id: str, name: str, arguments: dict[str, Any]) -> ToolCallRequest:
    return ToolCallRequest(id=call_id, name=name, arguments=arguments)


def _runner(script: list[LLMResponse], workspace: Path, **kwargs: Any) -> tuple[AgentLoop, FakeProvider]:
    provider = FakeProvider(script)
    return AgentLoop(provider=provider, workspace=workspace, **kwargs), provider


def test_default_tools_include_memory_and_skill(
    isolated_home: Path, tmp_path: Path
) -> None:
    """AgentLoop exposes memory and skill alongside the six base tools."""
    runner, _ = _runner([LLMResponse(content="hi")], tmp_path)
    names = sorted(d["function"]["name"] for d in runner.tools.get_definitions())
    assert names == ["bash", "edit", "glob", "grep", "memory", "read", "skill", "write"]


@pytest.mark.asyncio
async def test_plain_final_response(isolated_home: Path, tmp_path: Path) -> None:
    """No tool calls means one round and a direct answer."""
    runner, provider = _runner([LLMResponse(content="hi")], tmp_path)
    assert await runner.run("hello") == "hi"
    assert len(provider.seen) == 1


@pytest.mark.asyncio
async def test_single_tool_call(isolated_home: Path, tmp_path: Path) -> None:
    """One tool round executes and feeds the next iteration."""
    runner, provider = _runner(
        [
            LLMResponse(
                content="",
                tool_calls=[_call("c1", "bash", {"command": "printf 'x'"})],
            ),
            LLMResponse(content="done"),
        ],
        tmp_path,
    )
    assert await runner.run("go") == "done"
    assert len(provider.seen) == 2
    second = provider.seen[1]
    assert second[-2]["tool_calls"][0]["id"] == "c1"
    assert second[-1] == {
        "role": "tool",
        "tool_call_id": "c1",
        "name": "bash",
        "content": "x",
    }


@pytest.mark.asyncio
async def test_multi_calls_share_one_iteration(
    isolated_home: Path, tmp_path: Path
) -> None:
    """Several calls in one response cost a single chat round."""
    runner, provider = _runner(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    _call("c1", "bash", {"command": "printf 'a'"}),
                    _call("c2", "bash", {"command": "printf 'b'"}),
                ],
            ),
            LLMResponse(content="both"),
        ],
        tmp_path,
    )
    assert await runner.run("go") == "both"
    assert len(provider.seen) == 2


@pytest.mark.asyncio
async def test_max_iterations_fallback(isolated_home: Path, tmp_path: Path) -> None:
    """Exhausted budgets end after max_iterations rounds with a fallback."""
    script = [
        LLMResponse(
            content="",
            tool_calls=[_call(f"c{i}", "bash", {"command": "printf 'z'"})],
        )
        for i in range(10)
    ]
    runner, provider = _runner(script, tmp_path, max_iterations=2)
    result = await runner.run("go")
    assert result == "I've completed processing but have no response to give."
    assert len(provider.seen) == 2


@pytest.mark.asyncio
async def test_persistence_and_reload(isolated_home: Path, tmp_path: Path) -> None:
    """Full turn history reloads with tool metadata and no system."""
    runner, provider = _runner(
        [
            LLMResponse(
                content="",
                tool_calls=[_call("c9", "bash", {"command": "printf 'p'"})],
            ),
            LLMResponse(content="persisted"),
        ],
        tmp_path,
    )
    assert await runner.run("save me", session_id="loop1") == "persisted"
    loaded = SessionManager().get_session("loop1")
    roles = [m["role"] for m in loaded.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert all(m["role"] != "system" for m in loaded.messages)
    history = loaded.get_history()
    assert history[1]["tool_calls"][0]["id"] == "c9"
    assert history[2]["tool_call_id"] == "c9"
    assert history[2]["name"] == "bash"


@pytest.mark.asyncio
async def test_system_first_in_working_messages(
    isolated_home: Path, tmp_path: Path
) -> None:
    """Provider requests start with the built system prompt."""
    runner, provider = _runner([LLMResponse(content="ok")], tmp_path)
    await runner.run("hi")
    first = provider.seen[0]
    assert first[0]["role"] == "system"
    assert first[0]["content"].startswith(runner.context.build_system_prompt())
    assert str(tmp_path.resolve()) not in first[0]["content"]
    assert first[1] == {"role": "user", "content": "hi"}
