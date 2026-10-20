# MiniCode

MiniCode is a small, non-streaming coding-agent CLI written in Python. You describe a task at the `> ` prompt; it sends the request and conversation history to an LLM, runs requested tools with a configured workspace as their working directory, and prints the final answer as Markdown.

## Features

- **Interactive REPL** — Markdown-rendered replies, tool status cards, and clean exit with `Ctrl+C` or `Ctrl+D`.
- **Agentic tool loop** — up to 20 model/tool iterations per turn, using nine built-in tools: `bash`, `read`, `write`, `edit`, `apply_patch`, `grep`, `glob`, `skill`, and `memory`. Path-based file tools resolve inside the workspace; `bash` runs with the workspace as its working directory.
- **Multi-file patches** — `apply_patch` applies Add, Update, Delete, and Move operations as one validated patch; hunks are matched by surrounding context rather than line numbers.
- **Post-write diagnostics** — after `write`, `edit`, or `apply_patch`, MiniCode runs `py_compile` for Python and optionally `ruff` and `node --check` when those tools are installed.
- **LiteLLM providers** — OpenAI and Anthropic models, plus OpenAI-compatible gateways through a custom `api_base`.
- **Project context** — loads `AGENTS.md`, `MINICODE.md`, or `CLAUDE.md` from the repository root down to the workspace, and keeps persistent notes in the workspace's `.minicode/memory/` directory.
- **Skills** — discovers `SKILL.md` playbooks from `.minicode/skills/` and `.claude/skills/` (project and user levels) and loads their full instructions on demand.
- **Persistent data** — transcripts are written as JSONL under the MiniCode home, and the last 200 CLI inputs are restored as terminal history.

## Requirements

- Python 3.10 or newer, plus `pip` and `git`.
- A Unix-like shell: Linux, macOS, or Windows via WSL. The `bash` tool executes `/bin/bash`.
- An API key for OpenAI, Anthropic, or an OpenAI-compatible gateway.
- Optional: [`ripgrep`](https://github.com/BurntSushi/ripgrep) for faster `grep`, and [`ruff`](https://docs.astral.sh/ruff/) or [`node`](https://nodejs.org/) for richer diagnostics.

## Installation

```bash
git clone <repository-url> MiniCode
cd MiniCode

python3 -m venv .venv
source .venv/bin/activate

pip install -e .
```

## Configuration

Create `~/.minicode/minicode_config.json`:

```json
{
  "agents": {
    "defaults": {
      "model": "gpt-4o",
      "workspace": "~/.minicode/workspace"
    }
  },
  "providers": {
    "openai": {
      "api_key": "${OPENAI_API_KEY}"
    }
  }
}
```

- Set `OPENAI_API_KEY` in your environment, or replace the placeholder with your key.
- The model name selects the provider: names containing `openai` or `gpt` use `providers.openai`; names containing `anthropic` or `claude` use `providers.anthropic`. The matching provider must have a non-empty `api_key`.
- For an OpenAI-compatible gateway, use a model name containing `openai` or `gpt` and set `providers.openai.api_base` to the full base URL, for example `http://127.0.0.1:8317/v1`.
- A `minicode_config.json` found in the current directory or any parent is merged over the user config, so projects can override the model or workspace.
- `workspace` is the directory MiniCode opens for tools; set it to the project you want it to work on. It defaults to `~/.minicode/workspace` and is created if missing.
- Set `MINICODE_HOME` to use a different directory for config, sessions, and history.

Only these agent settings are currently used by the CLI:

| Setting | Default | Purpose |
|---|---|---|
| `model` | `gpt-4o` | Model sent to LiteLLM |
| `workspace` | `~/.minicode/workspace` | Working directory for `bash` and root for path-based file tools |
| `max_tool_iterations` | `20` | Maximum model/tool iterations per turn |
| `reasoning_effort` | unset | Optional provider-specific reasoning hint |

## Usage

```bash
python -m tui.cli
```

Type a task and press Enter:

```text
> Find the retry logic and summarize how it works.
```

Tool calls are shown as they run; they are not sandboxed, so review the cards and use a dedicated workspace. The final response is printed as Markdown. Each launch starts a new conversation; transcripts are stored under `~/.minicode/sessions/`.

## Testing

```bash
pip install pytest pytest-asyncio
python -m pytest tests/ -q
```

Tests use an isolated temporary home directory, so they do not touch your real `~/.minicode`.
