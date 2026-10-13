"""Visual constants for the MiniCode TUI.

Keep display symbols, prefixes, and Rich styles in one place.
Rendering logic belongs in the TUI renderer, not in this module.
"""

from __future__ import annotations


# -- brand ------------------------------------------------------------------
BRAND = "✳ MiniCode"
ACCENT = "cyan"
DIM = "grey58"


# -- user input -------------------------------------------------------------
PROMPT = "> "
PROMPT_STYLE = f"bold {ACCENT}"
CONTINUATION = "… "


# -- conversation -----------------------------------------------------------
USER_STYLE = "bold"
ASSISTANT_STYLE = "default"
THINKING_STYLE = DIM


# -- tool execution ---------------------------------------------------------
TOOL_BULLET = "●"
TOOL_RESULT_PREFIX = "  ⎿ "

TOOL_RUNNING_STYLE = ACCENT
TOOL_OK_STYLE = "green"
TOOL_ERR_STYLE = "red"
TOOL_DETAIL_STYLE = DIM


# -- status and messages ----------------------------------------------------
META_STYLE = DIM
ERROR_STYLE = "bold red"

DONE_OK = "✓"
DONE_ERR = "✗"
