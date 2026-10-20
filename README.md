# MiniCode

MiniCode is a minimal, non-streaming coding-agent CLI written in Python — a graduation project that is small on purpose. You type a message at a `> ` prompt; an agent loop sends it (plus conversation history) to an LLM, executes requested tool calls inside a fenced workspace directory, and renders the final reply as Markdown.

## Features

- **Interactive REPL** — `> ` prompt loop with Markdown-rendered replies, tool status cards, and clean exit on `Ctrl+C` or `Ctrl+D`.
- **Agentic tool loop** — up to 20 tool rounds per turn, with 8 built-in tools fenced to the workspace: `bash`, `read`, `write`, `edit`, `grep`, `glob`, `skill`, `memory`.
- **Post-write diagnostics** — after every `write`/`edit`, fast syntax checks run (`py_compile` always; `ruff` and `node --check` when installed) and problems are reported back immediately.
- **LiteLLM providers** — OpenAI and Anthropic models matched by model-name keywords, plus any OpenAI-compatible gateway via a custom `api_base`.
- **Persistent sessions** — conversation history is stored as JSONL in the MiniCode home and reused across turns; the last 200 inputs are restored into terminal history.

## Requirements

- **Python 3.10 or newer** (developed/tested on 3.12), plus `pip` and `git`.
- **A Unix-like shell** — Linux, macOS, or Windows via **WSL**; the `bash` tool executes `/bin/bash`.
- **An LLM API key** for OpenAI, Anthropic, or an OpenAI-compatible gateway.
- Optional: [`ripgrep`](https://github.com/BurntSushi/ripgrep) for a faster `grep`; [`ruff`](https://docs.astral.sh/ruff/) and/or [`node`](https://nodejs.org/) for richer diagnostics.

## Installation

```bash
# Clone the repository and enter it
git clone <repository-url> MiniCode
cd MiniCode

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1

# Install the project (editable) and the test dependencies
pip install -e .
pip install pytest pytest-asyncio

# Sanity check: package imports
python -c "import tui.cli; print('import ok')"
```

## Configuration

Create `~/.minicode/minicode_config.json` (the MiniCode home; override with `MINICODE_HOME`):

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
      "api_key": "sk-your-key-here"
    }
  }
}
```

A `minicode_config.json` found upward from the current directory deep-merges over this file. Anthropic and gateway examples, model-to-provider matching, `${ENV_VAR}` interpolation, and all settings: see [docs/configuration.md](docs/configuration.md).

## Running

```bash
python -m tui.cli
```

Type a message at the `> ` prompt; the agent may call tools before printing its Markdown reply. All tool file access is confined to `workspace` — point it at a dedicated directory and review the tool calls shown in the CLI. Exit with `Ctrl+C` or `Ctrl+D`. An example session is in [docs/usage.md](docs/usage.md).

## Testing

```bash
python -m pytest tests/ -q              # full suite
python -m pytest tests/test_cli.py -v   # single module, verbose
```

Tests use an isolated temp home (`isolated_home` fixture), so they never touch your real `~/.minicode`.

## Documentation

- [docs/configuration.md](docs/configuration.md) — configuration layers, provider examples, settings reference.
- [docs/usage.md](docs/usage.md) — built-in tools, example session, and practical prompts.
- [docs/troubleshooting.md](docs/troubleshooting.md) — common errors and fixes.
- [docs/contributing.md](docs/contributing.md) — contribution workflow, testing notes, and project layout.
