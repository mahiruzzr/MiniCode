"""Tests for skills discovery, parsing, registry, and SkillTool."""

from pathlib import Path

import pytest

from agent.context import ContextBuilder
from agent.skills import (
    Skill,
    SkillError,
    SkillRegistry,
    SkillTool,
    discover_skills,
    parse_skill_md,
    skills_preamble,
)
from tools import ToolRegistry

SKILL_MD = """---
name: review
description: Review code changes.
allowed-tools:
  - read
  - grep
---

# Review playbook

Checklist here.
"""


def _write_skill(root: Path, folder: str, text: str = SKILL_MD) -> Path:
    skill_dir = root / folder
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        text.replace("name: review", f"name: {folder}", 1), encoding="utf-8"
    )
    return skill_file


def test_parse_valid_skill(tmp_path: Path) -> None:
    """Frontmatter, body, tools, directory, and source survive parsing."""
    skill_file = _write_skill(tmp_path, "review")
    skill = parse_skill_md(skill_file, source="project:.minicode/skills")
    assert isinstance(skill, Skill)
    assert skill.name == "review"
    assert skill.description == "Review code changes."
    assert "Checklist here." in skill.instructions
    assert skill.allowed_tools == ("read", "grep")
    assert skill.directory == str(skill_file.parent)
    assert skill.source == "project:.minicode/skills"


def test_parse_allows_comma_separated_tools(tmp_path: Path) -> None:
    """allowed-tools also accepts a comma-separated string."""
    skill_file = _write_skill(
        tmp_path, "fmt", SKILL_MD.replace("allowed-tools:\n  - read\n  - grep", 'allowed-tools: "read, grep"')
    )
    assert parse_skill_md(skill_file).allowed_tools == ("read", "grep")


def test_parse_malformed_skills_raise(tmp_path: Path) -> None:
    """Missing frontmatter or required fields raise SkillError."""
    no_frontmatter = tmp_path / "plain" / "SKILL.md"
    no_frontmatter.parent.mkdir(parents=True)
    no_frontmatter.write_text("no frontmatter here", encoding="utf-8")
    with pytest.raises(SkillError):
        parse_skill_md(no_frontmatter)

    no_desc = tmp_path / "nodesc" / "SKILL.md"
    no_desc.parent.mkdir(parents=True)
    no_desc.write_text("---\nname: nodesc\n---\nbody\n", encoding="utf-8")
    with pytest.raises(SkillError):
        parse_skill_md(no_desc)

    bad_yaml = tmp_path / "badyaml" / "SKILL.md"
    bad_yaml.parent.mkdir(parents=True)
    bad_yaml.write_text("---\nname: [unclosed\n---\nbody\n", encoding="utf-8")
    with pytest.raises(SkillError):
        parse_skill_md(bad_yaml)

    not_mapping = tmp_path / "notmap" / "SKILL.md"
    not_mapping.parent.mkdir(parents=True)
    not_mapping.write_text("---\njust a string\n---\nbody\n", encoding="utf-8")
    with pytest.raises(SkillError):
        parse_skill_md(not_mapping)


def test_project_discovery_both_roots(tmp_path: Path) -> None:
    """Project .minicode and .claude skill dirs are both discovered."""
    _write_skill(tmp_path / ".minicode" / "skills", "native")
    _write_skill(tmp_path / ".claude" / "skills", "compat")
    registry = discover_skills(tmp_path, home=tmp_path / "no-home", include_user=False)
    assert registry.names() == ["compat", "native"]


def test_user_discovery(tmp_path: Path) -> None:
    """User-level native and interop dirs are discovered with sources."""
    home = tmp_path / "home"
    _write_skill(home / ".minicode" / "skills", "mine")
    _write_skill(home / ".claude" / "skills", "theirs")
    registry = discover_skills(tmp_path, home=home)
    assert registry.names() == ["mine", "theirs"]
    assert registry.get("mine").source == "user:.minicode/skills"
    assert registry.get("theirs").source == "user:.claude/skills"


def test_precedence_project_over_user(tmp_path: Path) -> None:
    """Project skills win over same-named user skills."""
    ws_native = tmp_path / "ws" / ".minicode" / "skills"
    _write_skill(ws_native, "shared",
                 SKILL_MD.replace("description: Review code changes.",
                                  "description: Project version."))
    home = tmp_path / "home"
    _write_skill(home / ".minicode" / "skills", "shared",
                 SKILL_MD.replace("description: Review code changes.",
                                  "description: User version."))
    registry = discover_skills(tmp_path / "ws", home=home)
    assert registry.get("shared").description == "Project version."


def test_duplicate_add_returns_false() -> None:
    """First-added skill keeps the name."""
    registry = SkillRegistry()
    first = Skill(name="dup", description="first", instructions="a")
    assert registry.add(first) is True
    assert registry.add(Skill(name="dup", description="second", instructions="b")) is False
    assert registry.get("dup").description == "first"


def test_registry_collects_errors_not_crashes(tmp_path: Path) -> None:
    """Malformed files are surfaced in errors; valid ones still load."""
    root = tmp_path / ".minicode" / "skills"
    _write_skill(root, "good")
    broken = root / "broken"
    broken.mkdir(parents=True)
    (broken / "SKILL.md").write_text("no frontmatter", encoding="utf-8")
    registry = discover_skills(tmp_path, home=tmp_path / "no-home", include_user=False)
    assert registry.names() == ["good"]
    assert len(registry.errors) == 1


def test_empty_discovery(tmp_path: Path) -> None:
    """No skills means an empty registry and an empty preamble."""
    registry = discover_skills(tmp_path, home=tmp_path / "no-home")
    assert len(registry) == 0
    assert not registry
    assert skills_preamble(registry) == ""


def test_summary_discloses_names_only(tmp_path: Path) -> None:
    """System prompt gets names and descriptions, never full bodies."""
    _write_skill(tmp_path / ".minicode" / "skills", "review")
    registry = discover_skills(tmp_path, home=tmp_path / "no-home", include_user=False)
    preamble = skills_preamble(registry)
    assert "- **review**: Review code changes." in preamble
    assert "Checklist here." not in preamble
    assert "`skill` tool" in preamble


@pytest.mark.asyncio
async def test_skill_tool_full_load(tmp_path: Path) -> None:
    """SkillTool returns header plus full instructions."""
    _write_skill(tmp_path / ".minicode" / "skills", "review")
    registry = discover_skills(tmp_path, home=tmp_path / "no-home", include_user=False)
    tool = SkillTool(registry)
    assert tool.name == "skill"
    assert tool.parameters["required"] == ["name"]
    result = await tool.execute(name="review")
    assert result.startswith("# Skill: review\nReview code changes.\n")
    assert "Checklist here." in result
    assert "Intended tools: read, grep" in result


@pytest.mark.asyncio
async def test_skill_tool_through_registry(tmp_path: Path) -> None:
    """SkillTool works through MiniCode's ToolRegistry validation."""
    _write_skill(tmp_path / ".minicode" / "skills", "review")
    registry = discover_skills(tmp_path, home=tmp_path / "no-home", include_user=False)
    tools = ToolRegistry()
    tools.register(SkillTool(registry))
    result = await tools.execute("skill", {"name": "review"})
    assert "Checklist here." in result
    assert (await tools.execute("skill", {})).startswith("Error:")


@pytest.mark.asyncio
async def test_skill_tool_truncates_long_bodies(tmp_path: Path) -> None:
    """Bodies over the cap are truncated with a marker."""
    long_skill = SKILL_MD.replace("Checklist here.", "x" * 20000)
    _write_skill(tmp_path / ".minicode" / "skills", "long", long_skill)
    registry = discover_skills(tmp_path, home=tmp_path / "no-home", include_user=False)
    result = await SkillTool(registry).execute(name="long")
    assert "…[truncated]" in result
    assert len(result) < 20000


@pytest.mark.asyncio
async def test_skill_tool_unknown_skill(tmp_path: Path) -> None:
    """Unknown names list what is available."""
    _write_skill(tmp_path / ".minicode" / "skills", "review")
    registry = discover_skills(tmp_path, home=tmp_path / "no-home", include_user=False)
    result = await SkillTool(registry).execute(name="nope")
    assert result == "Error: no skill named 'nope'. Available skills: review"


def test_context_builder_shares_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ContextBuilder discovers once and injects only the summary."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _write_skill(tmp_path / ".minicode" / "skills", "review")
    builder = ContextBuilder(tmp_path)
    assert builder.skills.get("review").name == "review"
    prompt = builder.build_system_prompt()
    assert "- **review**: Review code changes." in prompt
    assert "Checklist here." not in prompt
