# MiniCode

MiniCode is a minimal, non-streaming coding-agent CLI written in Python. You type a message at a `> ` prompt, an agent loop sends it (plus conversation history) to an LLM, executes any requested tool calls inside a fenced workspace directory, and renders the final reply as Markdown. Then the prompt comes back for the next message.

It is a graduation project: small on purpose — one agent loop, one CLI, no fullscreen UI, no slash commands, no permissions system.

## Features

- **Interactive REPL** — `> ` prompt loop with Markdown-rendered replies, tool status cards, blank line between turns, and clean exit on `Ctrl+C` or `Ctrl+D` (EOF).
- **Agentic tool loop** — the agent can call tools over up to 20 rounds per turn (`max_tool_iterations`), with per-call logging.
- **8 built-in tools**, all fenced to the workspace directory:

  | Tool | What it does |
  |------|--------------|
  | `bash` | Runs a shell command in the workspace, returns combined output (truncated with spill-to-file for large output) |
  | `read` | Reads a file with line numbers; lists directory entries |
  | `write` | Creates/overwrites a file (creates parent dirs) |
  | `edit` | Replaces `old_string` with `new_string` using fuzzy matching tolerant of whitespace drift |
  | `grep` | Regex search over file contents (uses `rg` when installed, pure-Python fallback otherwise) |
  | `glob` | Lists files matching a glob pattern |
  | `skill` | Loads a pre-authored skill playbook by name |
  | `memory` | Lists/reads/writes/appends/deletes persistent memory files |

- **Post-write diagnostics** — after every `write`/`edit`, fast syntax checkers run (`py_compile` for Python always; `ruff` and `node --check` when installed) and problems are reported back immediately.
- **LLM providers via LiteLLM** — OpenAI and Anthropic models, matched by model-name keywords, plus any OpenAI-compatible gateway via a custom `api_base`.
- **Persistent sessions** — conversation history is stored as JSONL under the MiniCode home directory and reused across turns.
- **Input history** — previous inputs are persisted (last 200) and restored into terminal up-arrow history.

## Requirements

- **Python 3.10 or newer** (`pyproject.toml`: `requires-python = ">=3.10"`; developed/tested on 3.12).
- **pip** and **git**.
- **A Unix-like shell**: the `bash` tool executes `/bin/bash`, so run on Linux, macOS, or Windows via **WSL**.
- **An LLM API key** for OpenAI, Anthropic, or an OpenAI-compatible gateway (see Configuration).
- Optional but recommended:
  - [`ripgrep` (`rg`)](https://github.com/BurntSushi/ripgrep) — faster `grep` tool; otherwise a built-in Python search is used.
  - [`ruff`](https://docs.astral.sh/ruff/) and/or [`node`](https://nodejs.org/) — richer post-edit diagnostics.

## Installation

```bash
# 1. Clone the repository and enter it
git clone <repository-url> MiniCode
cd MiniCode

# 2. Create and activate a virtual environment
python3 -m venv .venv

# Linux / macOS / WSL:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

# 3. Install the project (editable) with its runtime dependencies
pip install -e .

# 4. Install test dependencies (needed only to run the test suite)
pip install pytest pytest-asyncio

# 5. Sanity check: package imports and tests pass
python -c "import tui.cli; print('import ok')"
python -m pytest tests/ -q
```

Runtime dependencies installed by `pip install -e .` are `litellm`, `pydantic`, `pydantic-settings`, `pyyaml`, and `rich` (see `pyproject.toml`).

## Configuration

MiniCode reads JSON configuration from two layers (see `src/config/loader.py`):

1. **User layer** (required): `$MINICODE_HOME/minicode_config.json`, defaulting to `~/.minicode/minicode_config.json`.
2. **Project layer** (optional): a `minicode_config.json` found by searching upward from the current working directory. It deep-merges over the user layer, so per-project model/key overrides are possible.

Values support `${ENV_VAR}` interpolation, which fails loudly if the variable is unset. Malformed JSON or a non-object root raises a `ConfigError` instead of silently falling back.

### Minimal example (OpenAI)

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
      "api_key": "sk-your-key-here"
    }
  }
}
```

### Anthropic example

```json
{
  "agents": {
    "defaults": {
      "model": "claude-sonnet-4"
    }
  },
  "providers": {
    "anthropic": {
      "api_key": "sk-ant-your-key-here"
    }
  }
}
```

### OpenAI-compatible gateway example

```json
{
  "agents": {
    "defaults": {
      "model": "openai/my-custom-model"
    }
  },
  "providers": {
    "openai": {
      "api_key": "gateway-key-here",
      "api_base": "http://127.0.0.1:8317/v1"
    }
  }
}
```

### How model → provider matching works

The model string only needs to **contain** a provider keyword (`src/providers/registry.py`):

| Model contains | Provider section used |
|----------------|-----------------------|
| `openai` or `gpt` (e.g. `gpt-4o`, `openai/my-model`) | `providers.openai` |
| `anthropic` or `claude` (e.g. `claude-sonnet-4`) | `providers.anthropic` |

A provider only matches when its `api_key` is non-empty. The `openai/` prefix style plus a custom `api_base` is how LiteLLM routes to a gateway while MiniCode still attributes the key to the `openai` section. `api_base` must be the full base URL including the version path (e.g. ending in `/v1` for OpenAI-compatible servers).

### Settings reference

| Setting | Required | Default | Notes |
|---------|----------|---------|-------|
| `agents.defaults.model` | Yes | `"gpt-4o"` | Matched against provider keywords as above |
| `agents.defaults.workspace` | No | `"~/.minicode/workspace"` | All tool file access is fenced inside it; created on startup |
| `agents.defaults.max_tokens` | No | `8192` | Schema default (provider calls use their own default) |
| `agents.defaults.temperature` | No | `0.7` | Schema default |
| `agents.defaults.max_tool_iterations` | No | `20` | Max LLM rounds per turn; then a fallback message is returned |
| `providers.<name>.api_key` | Yes | `""` | Empty key never matches — requests would fail authentication |
| `providers.<name>.api_base` | No | `null` | Custom endpoint for gateways/proxies |
| `providers.<name>.extra_headers` | No | `null` | Extra HTTP headers forwarded to the provider |

To keep secrets out of files entirely, combine interpolation with environment variables:

```json
{ "providers": { "openai": { "api_key": "${OPENAI_API_KEY}" } } }
```

## Running MiniCode

From a clean setup (after Installation and Configuration):

```bash
python -m tui.cli
```

You should see a `> ` prompt. First startup creates the workspace directory plus `sessions/` storage under the MiniCode home. To use an isolated home (e.g. for testing), set `MINICODE_HOME`:

```bash
MINICODE_HOME=/tmp/minicode-home python -m tui.cli
```

## Usage

Type a message and press Enter. Empty lines are ignored. The agent may call tools (shown as `● <tool>` lines with `⎿ ✓/✗` result cards) before printing its Markdown reply. History carries over between turns in the same process.

```text
> What does the login function do? Start by finding it.

● grep(pattern)
  ⎿ ✓ grep  src/auth.py:42: def login(...
(multiline Markdown reply here)

> Read src/auth.py lines 30-70

● read(src/auth.py)
  ⎿ ✓ read
(auth.py excerpt rendered here)

> Add input validation to login and run the test suite

● read(src/auth.py)
  ⎿ ✓ read
● edit(src/auth.py)
  ⎿ ✓ edit
● bash(python -m pytest tests/ -q)
  ⎿ ✓ bash
(summary of the change here)

> ^D
```

Practical examples:

- **Ask about code** — `Where is retry logic implemented? Search the codebase and summarize.`
- **Read files** — the agent uses `read` itself, but you can point it: `Read src/agent/loop.py and explain the iteration budget.`
- **Request changes** — `Fix the off-by-one in paginate() in src/api.py, then run the relevant tests.` (writes/edits run syntax diagnostics automatically; test runs go through `bash`.)
- **Run commands** — `Run the linter and report any errors.` (confined to the workspace; interactive scaffolds that would hang are refused with a non-interactive hint.)
- **Exit** — `Ctrl+C` or `Ctrl+D`.

## Project Structure

```text
MiniCode/
├── pyproject.toml        # package metadata, runtime deps, pytest config
├── src/                  # installed packages (setuptools find: where=["src"])
│   ├── agent/            # AgentLoop (tool loop), context builder, memory, skills
│   ├── config/           # loader.py (layered JSON + env interpolation), schema.py
│   ├── providers/        # LLMProvider ABC, LiteLLM adapter, provider registry
│   ├── session/          # JSONL session store (one file per session)
│   ├── tools/            # Tool ABC + registry, files/search/shell implementations,
│   │                     # fuzzy replace engine, post-edit diagnostics
│   └── tui/              # cli.py (REPL entry, `python -m tui.cli`),
│                         # theme.py (styles/symbols), history.py (input history)
└── tests/                # pytest suite, incl. isolated-home fixture (conftest.py)
```

Runtime state lives outside the repo in the MiniCode home (`~/.minicode` unless `MINICODE_HOME` is set):

- `minicode_config.json` — your configuration (never committed; see Security Notes)
- `sessions/*.jsonl` — persisted conversations
- `history.json` — last 200 CLI inputs
- `workspace/` — default agent working directory

## Development and Testing

```bash
# Run the full suite from the repo root
python -m pytest tests/ -q

# Run a single module verbosely
python -m pytest tests/test_cli.py -v
```

- Tests use `asyncio_mode = "auto"` (requires `pytest-asyncio`) and an `isolated_home` fixture that points `MINICODE_HOME` at a temp dir, so the suite never touches your real `~/.minicode`.
- Code style follows PEP 8 with Google-style docstrings; keep lines reasonably short and add tests for new behavior.

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `ConfigError: Invalid JSON ...` on startup | `minicode_config.json` has a syntax error — validate it with `python -m json.tool <file>` |
| `ConfigError: ... is not set` | A `${VAR}` reference has no matching environment variable — export it first |
| Replies like `Error calling LLM: ...401/403...` | Missing or wrong `api_key` for the matched provider; check the model→provider keyword table |
| Requests go to the wrong server / model not found | `api_base` must be the full base URL (e.g. include `/v1`); the model name must contain `openai`/`gpt` or `anthropic`/`claude` so a provider section matches |
| `bash` tool fails immediately | It executes `/bin/bash` — use Linux, macOS, or WSL |
| `grep` feels slow | Install `ripgrep` (`rg`); the tool uses it automatically when present |
| `ModuleNotFoundError: tui` when launching | The package isn't installed — rerun `pip install -e .` in the activated venv |
| `pytest: asyncio_mode unknown` or async test errors | Install the test extras: `pip install pytest pytest-asyncio` |
| Hanging `npm init` / scaffolding commands | Intentional: the `bash` tool refuses interactive scaffolds; rerun with the suggested non-interactive flag (e.g. `npm init -y`) |

## Security Notes

- `minicode_config.json` holds **secrets** (provider API keys). The shipped `.gitignore` excludes it along with `.minicode/`, `.env` files, and the personal `.opencode/`/`.trellis/` directories — still, never place real keys in a config file inside the repo, and double-check `git status` before committing.
- Prefer `${ENV_VAR}` interpolation (e.g. `"api_key": "${OPENAI_API_KEY}"`) so keys live in the environment, not on disk next to code.
- The agent executes shell commands and edits files — confine it by pointing `workspace` at a dedicated directory and reviewing tool calls in the CLI output.
- LiteLLM forwards `extra_headers` verbatim; only add headers you understand.

## Contributing

1. Create a branch from a clean tree: `git checkout -b <short-description>`.
2. Make focused changes with Google-style docstrings; don't add unneeded modules or helpers.
3. Run the suite (`python -m pytest tests/ -q`) — all tests must pass.
4. Commit with a concise message and open a pull request describing what changed and which tests cover it.
