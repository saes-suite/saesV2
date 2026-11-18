"""Dynamic package version resolution for saesV2."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path


def resolve_version() -> str:
    """Resolve package version from installed metadata or version.txt fallback."""
    try:
        return _pkg_version("saesV2")
    except PackageNotFoundError:
        version_file = Path(__file__).resolve().parents[2] / "version.txt"
        if version_file.exists():
            return version_file.read_text(encoding="utf-8").strip().replace("-SNAPSHOT", "")
        return "0.2.0"


__version__ = resolve_version()
