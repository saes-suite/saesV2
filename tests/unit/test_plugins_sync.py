"""Cross-repository drift checks between `saesV2` and `saes-plugins` skills."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import saesV2
import saesV2.domain.exceptions as saes_exceptions

SKILLS_DIR = Path(__file__).resolve().parents[3] / "saes-plugins" / "plugins" / "saes" / "skills"


def _require_skills_dir() -> Path:
    if not SKILLS_DIR.is_dir():
        pytest.skip("saes-plugins sibling repository not present")
    return SKILLS_DIR


def _skill_files() -> list[Path]:
    skills_dir = _require_skills_dir()
    files = sorted(skills_dir.glob("*/SKILL.md"))
    assert files, f"No SKILL.md files found under {skills_dir}"
    return files


def test_documented_top_level_and_statistics_symbols_exist() -> None:
    """Verify every `saesV2.<symbol>` and `saesV2.statistics.<symbol>` in skills exists."""
    top_level_refs: set[str] = set()
    stats_refs: set[str] = set()

    for skill_file in _skill_files():
        text = skill_file.read_text(encoding="utf-8")
        top_level_refs.update(re.findall(r"\bsaesV2\.([a-zA-Z_][a-zA-Z0-9_]*)", text))
        stats_refs.update(re.findall(r"\bsaesV2\.statistics\.([a-zA-Z_][a-zA-Z0-9_]*)", text))

    assert top_level_refs, "Expected at least one saesV2.<symbol> reference in SKILL.md files"
    assert stats_refs, "Expected at least one saesV2.statistics.<symbol> reference in SKILL.md files"

    missing_top = sorted(sym for sym in top_level_refs if not hasattr(saesV2, sym))
    assert not missing_top, f"Skills reference missing saesV2 symbols: {missing_top}"

    missing_stats = sorted(sym for sym in stats_refs if not hasattr(saesV2.statistics, sym))
    assert not missing_stats, f"Skills reference missing saesV2.statistics symbols: {missing_stats}"


def test_documented_exceptions_exist() -> None:
    """Verify every exception class documented in `saes-python-api/SKILL.md` exists."""
    api_skill = _require_skills_dir() / "saes-python-api" / "SKILL.md"
    text = api_skill.read_text(encoding="utf-8")

    match = re.search(r"## Exceptions.*?(?=\n## |\Z)", text, flags=re.DOTALL)
    assert match, "Exceptions section not found in saes-python-api/SKILL.md"

    documented_errors = set(re.findall(r"\b([A-Z][a-zA-Z0-9]*Error)\b", match.group(0)))
    assert documented_errors, "No exception classes found in Exceptions section"

    missing = sorted(err for err in documented_errors if not hasattr(saes_exceptions, err))
    assert not missing, f"Skills reference missing exceptions in saesV2.domain.exceptions: {missing}"


def test_minimum_saesv2_version_matches_package() -> None:
    """Verify `saesV2.__version__` satisfies the floor declared in `saes-python-api/SKILL.md`."""
    api_skill = _require_skills_dir() / "saes-python-api" / "SKILL.md"
    text = api_skill.read_text(encoding="utf-8")

    match = re.search(r"Target library:\s*`saesV2\s*>=\s*(\d+\.\d+\.\d+)`", text)
    assert match, "Target library floor (`saesV2 >= X.Y.Z`) not found in saes-python-api/SKILL.md"

    floor_tuple = tuple(int(part) for part in match.group(1).split("."))
    current_tuple = tuple(int(part) for part in saesV2.__version__.split(".")[:3])
    assert current_tuple >= floor_tuple, (
        f"saesV2 {saesV2.__version__} is older than plugin minimum {match.group(1)}"
    )
