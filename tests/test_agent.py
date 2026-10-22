"""Tests for the AgentLoop data flow with a fake provider."""

from pathlib import Path
from typing import Any

import pytest

from agent import AgentLoop
from providers.base import LLMProvider, LLMResponse, ToolCallRequest
from session.manager import Session, SessionManager


class FakeProvider(LLMProvider):
    """Scripted provider that records every request."""

    def __init__(self, script: list[LLMResponse]) -> None:
        super().__init__()
        self.script = list(script)
        self.seen: list[list[dict[str, Any]]] = []
        self.reasoning_efforts: list[str | None] = []
        self.tool_choices: list[str | dict[str, Any] | None] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.reasoning_efforts.append(reasoning_effort)
        self.tool_choices.append(tool_choice)
        self.seen.append(list(messages))
        return self.script.pop(0)

    def get_default_model(self) -> str:
        return "fake-model"


def _call(call_id: str, name: str, arguments: dict[str, Any]) -> ToolCallRequest:
    return ToolCallRequest(id=call_id, name=name, arguments=arguments)


def _runner(script: list[LLMResponse], workspace: Path, **kwargs: Any) -> tuple[AgentLoop, FakeProvider]:
    provider = FakeProvider(script)
    return AgentLoop(provider=provider, workspace=workspace, **kwargs), provider


def _append_tool_exchange(session: Session, call_id: str, name: str, content: Any) -> None:
    """Append an assistant tool call and its raw tool result to a session."""
    session.add_message(
        "assistant",
        "",
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
        ],
    )
    session.messages.append(
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": content,
        }
    )


def _assert_tool_calls_paired(messages: list[dict[str, Any]]) -> None:
    """Check every assistant tool_calls id is answered by its tool result."""
    declared: set[str] = set()
    answered: set[str] = set()
    for msg in messages:
        if msg.get("role") == "assistant":
            declared.update(tc["id"] for tc in msg.get("tool_calls") or [])
        elif msg.get("role") == "tool":
            assert msg["tool_call_id"] in declared
            answered.add(msg["tool_call_id"])
    assert answered == declared


def test_default_tools_include_plan_memory_and_skill(
    isolated_home: Path, tmp_path: Path
) -> None:
    """AgentLoop exposes plan, apply_patch, memory, and skill with base tools."""
    runner, _ = _runner([LLMResponse(content="hi")], tmp_path)
    names = sorted(d["function"]["name"] for d in runner.tools.get_definitions())
    assert names == [
        "apply_patch",
        "bash",
        "edit",
        "glob",
        "grep",
        "memory",
        "read",
        "skill",
        "update_plan",
        "write",
    ]


@pytest.mark.asyncio
async def test_apply_patch_tool_is_registered_and_executes(
    isolated_home: Path, tmp_path: Path
) -> None:
    """apply_patch is exposed, receives the workspace, and dispatches normally."""
    patch = (
        "*** Begin Patch\n"
        "*** Add File: generated.txt\n"
        "+from patch\n"
        "*** End Patch"
    )
    runner, provider = _runner(
        [
            LLMResponse(
                content="",
                tool_calls=[_call("p1", "apply_patch", {"patch": patch})],
            ),
            LLMResponse(content="patched"),
        ],
        tmp_path,
    )
    definitions = {
        d["function"]["name"]: d["function"] for d in runner.tools.get_definitions()
    }
    assert "apply_patch" in definitions
    assert definitions["apply_patch"]["parameters"]["required"] == ["patch"]

    assert await runner.run("go") == "patched"
    assert (tmp_path / "generated.txt").read_text(encoding="utf-8") == "from patch"
    result = provider.seen[1][-1]
    assert result["name"] == "apply_patch"
    assert result["content"].startswith("Applied patch: 1 change(s)")


@pytest.mark.asyncio
async def test_update_plan_tool_is_registered_and_executes(
    isolated_home: Path, tmp_path: Path
) -> None:
    """update_plan is exposed and dispatches through the registry."""
    plan = [
        {"step": "inspect", "status": "completed"},
        {"step": "integrate", "status": "in_progress"},
        {"step": "test", "status": "pending"},
    ]
    runner, provider = _runner(
        [
            LLMResponse(
                content="",
                tool_calls=[_call("u1", "update_plan", {"plan": plan})],
            ),
            LLMResponse(content="planned"),
        ],
        tmp_path,
    )
    definitions = {
        d["function"]["name"]: d["function"] for d in runner.tools.get_definitions()
    }
    assert "update_plan" in definitions
    assert definitions["update_plan"]["parameters"]["required"] == ["plan"]

    assert await runner.run("go") == "planned"
    result = provider.seen[1][-1]
    assert result["name"] == "update_plan"
    assert result["content"].startswith("Plan updated (1/3 done)")
    assert "☑ inspect" in result["content"]
    assert "▶ integrate" in result["content"]
    assert "☐ test" in result["content"]


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
async def test_reasoning_fields_flow_to_provider(
    isolated_home: Path, tmp_path: Path
) -> None:
    """reasoning_effort reaches the provider; thinking fields ride in-turn."""
    runner, provider = _runner(
        [
            LLMResponse(
                content="",
                tool_calls=[_call("c1", "bash", {"command": "true"})],
                reasoning_content="why",
                thinking_blocks=[{"type": "thinking", "thinking": "why"}],
            ),
            LLMResponse(content="done"),
        ],
        tmp_path,
        reasoning_effort="high",
    )
    assert await runner.run("go", session_id="flow1") == "done"
    assert provider.reasoning_efforts == ["high", "high"]

    # Second provider round carries the assistant thinking fields.
    assert provider.seen[1][2]["reasoning_content"] == "why"
    assert provider.seen[1][2]["thinking_blocks"][0]["thinking"] == "why"


@pytest.mark.asyncio
async def test_tool_choice_flows_to_provider_on_every_round(
    isolated_home: Path, tmp_path: Path
) -> None:
    """A caller-supplied tool_choice reaches every provider round unchanged."""
    runner, provider = _runner(
        [
            LLMResponse(
                content="",
                tool_calls=[_call("c1", "bash", {"command": "true"})],
            ),
            LLMResponse(content="done"),
        ],
        tmp_path,
    )
    choice = {"type": "function", "function": {"name": "bash"}}
    assert await runner.run("go", tool_choice=choice) == "done"
    assert provider.tool_choices == [choice, choice]


@pytest.mark.asyncio
async def test_tool_choice_defaults_to_none(
    isolated_home: Path, tmp_path: Path
) -> None:
    """Omitting tool_choice leaves provider defaults untouched."""
    runner, provider = _runner([LLMResponse(content="ok")], tmp_path)
    assert await runner.run("hi") == "ok"
    assert provider.tool_choices == [None]


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


@pytest.mark.asyncio
async def test_microcompact_applied_before_provider_request(
    isolated_home: Path, tmp_path: Path
) -> None:
    """Old long compactable results from history shrink; recent ones stay."""
    runner, provider = _runner([LLMResponse(content="done")], tmp_path)
    session = runner.sessions.create_session()

    long_read = "r" * 600
    long_grep = "g" * 700
    long_write = "w" * 800
    long_glob = "n" * 900
    _append_tool_exchange(session, "c1", "bash", "tiny")
    _append_tool_exchange(session, "c2", "bash", 12345)
    _append_tool_exchange(session, "c3", "read", long_read)
    _append_tool_exchange(session, "c4", "grep", long_grep)
    _append_tool_exchange(session, "c5", "write", long_write)
    for i in range(10):
        _append_tool_exchange(session, f"c{i + 6}", "glob", long_glob)
    runner.sessions.save(session)

    assert await runner.run("go", session_id=session.session_id) == "done"

    tools = [m for m in provider.seen[0] if m.get("role") == "tool"]
    assert [t["content"] for t in tools[:5]] == [
        "tiny",
        12345,
        "[read result omitted from context]",
        "[grep result omitted from context]",
        long_write,
    ]
    assert all(t["content"] == long_glob for t in tools[5:])
    assert tools[2]["tool_call_id"] == "c3"
    assert tools[2]["name"] == "read"
    assert tools[3]["tool_call_id"] == "c4"
    assert tools[3]["name"] == "grep"
    _assert_tool_calls_paired(provider.seen[0])

    loaded = SessionManager().get_session(session.session_id)
    stored = [m for m in loaded.messages if m.get("role") == "tool"]
    assert stored[1]["content"] == 12345
    assert stored[2]["content"] == long_read
    assert stored[3]["content"] == long_grep
    assert stored[4]["content"] == long_write


@pytest.mark.asyncio
async def test_microcompact_progresses_across_tool_rounds(
    isolated_home: Path, tmp_path: Path
) -> None:
    """Later requests compact outputs accumulated earlier in the same run."""
    full = "0" * 600
    big = "printf '%0600d' 0"
    script = [
        LLMResponse(content="", tool_calls=[_call(f"c{i}", "bash", {"command": big})])
        for i in range(12)
    ]
    script.append(LLMResponse(content="done"))
    runner, provider = _runner(script, tmp_path)

    assert await runner.run("go", session_id="mc-run") == "done"

    with_11 = [m for m in provider.seen[11] if m.get("role") == "tool"]
    assert with_11[0]["content"] == "[bash result omitted from context]"
    assert all(m["content"] == full for m in with_11[1:])

    with_12 = [m for m in provider.seen[12] if m.get("role") == "tool"]
    assert with_12[0]["content"] == "[bash result omitted from context]"
    assert with_12[1]["content"] == "[bash result omitted from context]"
    assert all(m["content"] == full for m in with_12[2:])
    _assert_tool_calls_paired(provider.seen[12])

    loaded = SessionManager().get_session("mc-run")
    stored = [m for m in loaded.messages if m.get("role") == "tool"]
    assert all(m["content"] == full for m in stored)


@pytest.mark.asyncio
async def test_microcompact_keeps_full_recent_window_in_request(
    isolated_home: Path, tmp_path: Path
) -> None:
    """Exactly ten long compactable results pass through to the provider intact."""
    runner, provider = _runner([LLMResponse(content="done")], tmp_path)
    session = runner.sessions.create_session()
    originals = []
    for i in range(10):
        content = f"result-{i}-" + "x" * 600
        originals.append(content)
        _append_tool_exchange(session, f"c{i}", "bash", content)
    runner.sessions.save(session)

    assert await runner.run("go", session_id=session.session_id) == "done"

    tools = [m for m in provider.seen[0] if m.get("role") == "tool"]
    assert [t["content"] for t in tools] == originals
    assert not any("result omitted" in str(t["content"]) for t in tools)
    _assert_tool_calls_paired(provider.seen[0])


@pytest.mark.asyncio
async def test_microcompact_parallel_calls_in_one_round(
    isolated_home: Path, tmp_path: Path
) -> None:
    """A single round with twelve parallel results compacts only the stale two."""
    full = "0" * 600
    big = "printf '%0600d' 0"
    calls = [_call(f"c{i}", "bash", {"command": big}) for i in range(12)]
    runner, provider = _runner(
        [LLMResponse(content="", tool_calls=calls), LLMResponse(content="done")],
        tmp_path,
    )

    assert await runner.run("go", session_id="mc-parallel") == "done"

    sent = [m for m in provider.seen[1] if m.get("role") == "tool"]
    assert len(sent) == 12
    assert [m["content"] for m in sent[:2]] == [
        "[bash result omitted from context]",
        "[bash result omitted from context]",
    ]
    assert all(m["content"] == full for m in sent[2:])
    _assert_tool_calls_paired(provider.seen[1])

    loaded = SessionManager().get_session("mc-parallel")
    stored = [m for m in loaded.messages if m.get("role") == "tool"]
    assert len(stored) == 12
    assert all(m["content"] == full for m in stored)


@pytest.mark.asyncio
async def test_microcompact_is_request_scoped_and_non_destructive(
    isolated_home: Path, tmp_path: Path
) -> None:
    """Callbacks and earlier requests keep originals after later compaction."""
    full = "0" * 600
    big = "printf '%0600d' 0"
    script = [
        LLMResponse(content="", tool_calls=[_call(f"c{i}", "bash", {"command": big})])
        for i in range(12)
    ]
    script.append(LLMResponse(content="done"))
    runner, provider = _runner(script, tmp_path)
    observed: list[str] = []

    def capture(name: str, result: str, is_error: bool) -> None:
        observed.append(result)

    assert (
        await runner.run("go", session_id="mc-scope", on_tool_result=capture)
        == "done"
    )

    assert observed == [full] * 12
    early = [m for m in provider.seen[1] if m.get("role") == "tool"]
    assert early[0]["content"] == full
    late = [m for m in provider.seen[12] if m.get("role") == "tool"]
    assert late[0]["content"] == "[bash result omitted from context]"
    assert late[1]["content"] == "[bash result omitted from context]"
    assert all(m["content"] == full for m in late[2:])

    loaded = SessionManager().get_session("mc-scope")
    stored = [m for m in loaded.messages if m.get("role") == "tool"]
    assert all(m["content"] == full for m in stored)


@pytest.mark.asyncio
async def test_microcompact_reload_recompacts_from_stored_history(
    isolated_home: Path, tmp_path: Path
) -> None:
    """A fresh loop reloads full results and re-applies compaction per request."""
    full = "0" * 600
    big = "printf '%0600d' 0"
    script = [
        LLMResponse(content="", tool_calls=[_call(f"c{i}", "bash", {"command": big})])
        for i in range(12)
    ]
    script.append(LLMResponse(content="done"))
    runner, _ = _runner(script, tmp_path)
    assert await runner.run("first", session_id="mc-reload") == "done"

    fresh_runner, fresh_provider = _runner([LLMResponse(content="again")], tmp_path)
    assert await fresh_runner.run("second", session_id="mc-reload") == "again"

    first_tools = [m for m in fresh_provider.seen[0] if m.get("role") == "tool"]
    assert len(first_tools) == 12
    assert [m["content"] for m in first_tools[:2]] == [
        "[bash result omitted from context]",
        "[bash result omitted from context]",
    ]
    assert all(m["content"] == full for m in first_tools[2:])
    _assert_tool_calls_paired(fresh_provider.seen[0])

    loaded = SessionManager().get_session("mc-reload")
    stored = [m for m in loaded.messages if m.get("role") == "tool"]
    assert len(stored) == 12
    assert all(m["content"] == full for m in stored)


@pytest.mark.asyncio
async def test_microcompact_keeps_flow_on_iteration_budget_fallback(
    isolated_home: Path, tmp_path: Path
) -> None:
    """Compaction still applies when the loop exhausts max_iterations."""
    full = "0" * 600
    big = "printf '%0600d' 0"
    calls = [_call(f"c{i}", "bash", {"command": big}) for i in range(12)]
    script = [
        LLMResponse(content="", tool_calls=calls),
        LLMResponse(
            content="",
            tool_calls=[_call("c12", "bash", {"command": "printf 'z'"})],
        ),
    ]
    runner, provider = _runner(script, tmp_path, max_iterations=2)

    result = await runner.run("go", session_id="mc-budget")

    assert result == "I've completed processing but have no response to give."
    sent = [m for m in provider.seen[1] if m.get("role") == "tool"]
    assert [m["content"] for m in sent[:2]] == [
        "[bash result omitted from context]",
        "[bash result omitted from context]",
    ]
    assert all(m["content"] == full for m in sent[2:])
    loaded = SessionManager().get_session("mc-budget")
    stored = [m for m in loaded.messages if m.get("role") == "tool"]
    assert [m["content"] for m in stored[:12]] == [full] * 12
    assert stored[12]["content"] == "z"
