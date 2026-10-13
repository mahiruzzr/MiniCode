"""Discover and load SKILL.md-based agent skills.

Skill metadata is summarized in the system prompt, while full instructions
are loaded on demand through the skill tool.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from tools.base import Tool

_SKILL_FILE = "SKILL.md"
# Project roots take precedence over user roots.
# MiniCode is checked before Claude Code at each level.
_PROJECT_ROOTS = (".minicode/skills", ".claude/skills")
_USER_ROOTS = (".minicode/skills", ".claude/skills")  # resolved under $HOME
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
# Bound skill content returned to the model.
_MAX_BODY_CHARS = 16000


class SkillError(ValueError):
    """A SKILL.md is malformed."""


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    instructions: str
    allowed_tools: tuple[str, ...] = ()
    directory: str = ""  # Parent directory of SKILL.md.
    source: str = ""  # Discovery root for diagnostics.

    @property
    def summary_line(self) -> str:
        """Return the skill's compact system-prompt entry."""
        return f"- **{self.name}**: {self.description}"


def _coerce_tools(value: Any) -> tuple[str, ...]:
    """Normalize allowed-tools metadata to a tuple."""
    if not value:
        return ()
    if isinstance(value, str):
        return tuple(t.strip() for t in value.split(",") if t.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(t).strip() for t in value if str(t).strip())
    return ()


def parse_skill_md(path: str | Path, *, source: str = "") -> Skill:
    """Parse a SKILL.md into a Skill.

    Raises SkillError for invalid frontmatter or missing required metadata.
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise SkillError(f"cannot read {p}: {exc}") from exc

    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise SkillError(f"{p}: SKILL.md must open with a YAML frontmatter block")

    import yaml

    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise SkillError(f"{p}: invalid YAML frontmatter: {exc}") from exc
    if not isinstance(meta, dict):
        raise SkillError(f"{p}: frontmatter must be a mapping")

    # Use the folder name when name is omitted.
    name = str(meta.get("name") or p.parent.name).strip()
    description = str(meta.get("description") or "").strip()
    if not name:
        raise SkillError(f"{p}: skill needs a name")
    if not description:
        raise SkillError(f"{p}: skill {name!r} needs a description")

    body = text[match.end() :].strip()
    allowed = _coerce_tools(meta.get("allowed-tools", meta.get("allowed_tools")))
    return Skill(
        name=name,
        description=description,
        instructions=body,
        allowed_tools=allowed,
        directory=str(p.parent),
        source=source,
    )


class SkillRegistry:
    """Registry of discovered skills keyed by name."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self.errors: list[str] = []

    def add(self, skill: Skill) -> bool:
        """Add a skill unless the name is already registered."""
        # Discovery order determines precedence.
        if skill.name in self._skills:
            return False
        self._skills[skill.name] = skill
        return True

    def get(self, name: str) -> Skill | None:
        """Return the skill with ``name``, if any."""
        return self._skills.get(name)

    def all(self) -> list[Skill]:
        """Return all skills, sorted by name."""
        return sorted(self._skills.values(), key=lambda s: s.name)

    def names(self) -> list[str]:
        """Return all skill names, sorted."""
        return sorted(self._skills)

    def __len__(self) -> int:
        return len(self._skills)

    def __bool__(self) -> bool:
        return bool(self._skills)


def _iter_skill_files(root: Path) -> Iterator[Path]:
    """Yield discovered SKILL.md files under a skill root."""
    if not root.is_dir():
        return
    for child in sorted(root.iterdir()):
        if child.is_dir():
            skill_file = child / _SKILL_FILE
            if skill_file.is_file():
                yield skill_file


def discover_skills(
    workspace: str | Path,
    *,
    home: str | Path | None = None,
    include_user: bool = True,
) -> SkillRegistry:
    """Discover project and user skills.

    Project skills are discovered before user skills, so duplicate names
    resolve to the first discovered skill. Invalid skills are recorded in
    the registry instead of aborting discovery.
    ``home`` overrides ``$HOME`` for tests.
    """
    registry = SkillRegistry()
    ws = Path(workspace)
    roots: list[tuple[Path, str]] = [
        (ws / rel, f"project:{rel}") for rel in _PROJECT_ROOTS
    ]
    if include_user:
        home_dir = Path(home) if home is not None else Path.home()
        roots += [(home_dir / rel, f"user:{rel}") for rel in _USER_ROOTS]
    for root, source in roots:
        for skill_file in _iter_skill_files(root):
            try:
                registry.add(parse_skill_md(skill_file, source=source))
            except SkillError as exc:
                registry.errors.append(str(exc))
    return registry


class SkillTool(Tool):
    """Load a skill's full instructions on demand (progressive disclosure)."""

    def __init__(self, registry: SkillRegistry):
        self._registry = registry

    @property
    def name(self) -> str:
        return "skill"

    @property
    def description(self) -> str:
        return (
            "Load the full instructions for a named skill before doing that "
            "kind of task. Skills are pre-authored playbooks; the available "
            "ones are listed in the system prompt. Returns the skill's steps "
            "for you to follow with your normal tools."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "The skill to load (one of the available skills "
                        "listed in the system prompt)."
                    ),
                }
            },
            "required": ["name"],
        }

    async def execute(self, **kwargs: Any) -> str:
        """Load a skill's full instructions by name."""
        name = str(kwargs.get("name") or "").strip()
        skill = self._registry.get(name)
        if skill is None:
            available = ", ".join(self._registry.names()) or "(none)"
            return f"Error: no skill named {name!r}. Available skills: {available}"

        body = skill.instructions[:_MAX_BODY_CHARS]
        if len(skill.instructions) > _MAX_BODY_CHARS:
            body += "\n…[truncated]"
        header = f"# Skill: {skill.name}\n{skill.description}\n"
        if skill.allowed_tools:
            header += f"\nIntended tools: {', '.join(skill.allowed_tools)}\n"
        if skill.directory:
            header += (
                f"\nSkill files are in `{skill.directory}` — read any bundled "
                "resources with the read tool.\n"
            )
        return f"{header}\n{body}"


def skills_preamble(registry: SkillRegistry) -> str:
    """Build the compact system-prompt summary for progressive disclosure."""
    if not registry:
        return ""
    lines = "\n".join(s.summary_line for s in registry.all())
    return (
        "## Available skills\n\n"
        "Pre-authored playbooks for specific tasks. When a task matches one, "
        "call the `skill` tool with its name to load the full instructions "
        "BEFORE starting, then follow them.\n\n" + lines
    )
