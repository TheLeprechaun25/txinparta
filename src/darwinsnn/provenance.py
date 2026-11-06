"""Record what produced a result: code versions, seed, machine."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy

REPO = Path(__file__).resolve().parents[2]


def git_state(path: Path) -> dict | None:
    """Commit hash and whether the working tree has uncommitted changes.

    Uses the git binary when it exists. Otherwise (e.g. on cluster compute
    nodes without git) it reads the commit from the .git files, and "dirty"
    is reported as None (unknown).
    """
    def git(*args):
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    try:
        commit = git("rev-parse", "HEAD")
        if commit:
            return {"commit": commit, "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        pass
    commit = read_head(path)
    return {"commit": commit, "dirty": None} if commit else None


def read_head(path: Path) -> str | None:
    """Commit of HEAD read straight from the .git directory (no git binary needed)."""
    git_dir = path / ".git"
    if git_dir.is_file():   # submodule / worktree: ".git" is a file "gitdir: <path>"
        git_dir = (path / git_dir.read_text().split(":", 1)[1].strip()).resolve()
    try:
        head = (git_dir / "HEAD").read_text().strip()
    except OSError:
        return None
    if not head.startswith("ref:"):
        return head                                   # detached HEAD
    ref = head.split(" ", 1)[1]
    if (git_dir / ref).exists():
        return (git_dir / ref).read_text().strip()
    packed = git_dir / "packed-refs"
    if packed.exists():
        for line in packed.read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    return None


def provenance(seed: int, simulator_backend: str) -> dict:
    info = {
        "time_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": sys.argv,
        "seed": seed,
        "backend": simulator_backend,
        "host": platform.node(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "darwinsnn": git_state(REPO),
    }
    if simulator_backend == "pultsito":
        info["pultsito"] = git_state(REPO / "third_party" / "pultsito")
    return info
