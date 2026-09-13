from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from ll009ag_common import CE, hf, rel


def _git_executable() -> Path:
    found = shutil.which("git")
    if not found:
        raise CE("git executable not found")
    p = Path(found)
    try:
        p = p.resolve(strict=True)
    except FileNotFoundError as e:
        raise CE(f"git executable disappeared: {found}") from e
    if not p.is_file():
        raise CE(f"git executable is not a regular file: {p}")
    return p


def _run(git: Path, repo: Path, args: list[str], *, ok: tuple[int, ...] = (0,)) -> subprocess.CompletedProcess[str]:
    try:
        cp = subprocess.run(
            [str(git), "-C", str(repo), *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError as e:
        raise CE(f"cannot execute git: {e}") from e
    if cp.returncode not in ok:
        detail = cp.stderr.strip() or cp.stdout.strip() or f"exit {cp.returncode}"
        raise CE(f"git {' '.join(args)} failed: {detail}")
    return cp


def _oid(v: str, label: str, object_format: str) -> str:
    v = v.strip().lower()
    expected = {"sha1": 40, "sha256": 64}.get(object_format)
    if expected is None:
        raise CE(f"unsupported git object format: {object_format!r}")
    if len(v) != expected:
        raise CE(f"{label} must be {expected}-hex Git {object_format} object ID")
    try:
        int(v, 16)
    except ValueError as e:
        raise CE(f"{label} must be hexadecimal") from e
    return v


def repo_relative(repo: Path, path: Path, label: str) -> str:
    repo = repo.resolve(strict=True)
    if path.is_symlink():
        raise CE(f"{label} may not be a symlink: {path}")
    try:
        q = path.resolve(strict=True)
    except FileNotFoundError as e:
        raise CE(f"{label} missing: {path}") from e
    try:
        relative = q.relative_to(repo)
    except ValueError as e:
        raise CE(f"{label} is outside repository root: {q}") from e
    return rel(relative.as_posix(), label)


def capture_git_subject(repo: Path, expected_head: str, tracked_paths: list[str]) -> dict[str, Any]:
    """Prove a declared campaign subject is the actual clean Git subject for protected inputs."""
    repo = repo.resolve(strict=True)
    if not repo.is_dir():
        raise CE(f"repository root is not a directory: {repo}")
    git = _git_executable()

    top = _run(git, repo, ["rev-parse", "--show-toplevel"]).stdout.strip()
    try:
        top_path = Path(top).resolve(strict=True)
    except FileNotFoundError as e:
        raise CE(f"git reported missing repository root: {top}") from e
    if top_path != repo:
        raise CE(f"repo-root mismatch: supplied {repo}, git top-level {top_path}")

    object_format = _run(git, repo, ["rev-parse", "--show-object-format"]).stdout.strip().lower()
    if object_format not in {"sha1", "sha256"}:
        raise CE(f"unsupported git object format: {object_format!r}")

    head = _oid(_run(git, repo, ["rev-parse", "--verify", "HEAD^{commit}"]).stdout, "actual HEAD", object_format)
    expected = _oid(expected_head, "declared repo head", object_format)
    if head != expected:
        raise CE(f"declared repo head mismatch: declared {expected}, actual {head}")
    tree = _oid(_run(git, repo, ["rev-parse", "--verify", "HEAD^{tree}"]).stdout, "HEAD tree", object_format)

    paths = [rel(x, "git tracked campaign path") for x in tracked_paths]
    if not paths or len(paths) != len(set(paths)):
        raise CE("git tracked campaign paths must be a non-empty unique list")

    for path in paths:
        _run(git, repo, ["ls-files", "--error-unmatch", "--", path])

    diff = _run(git, repo, ["diff", "--quiet", "HEAD", "--", *paths], ok=(0, 1))
    if diff.returncode == 1:
        raise CE("protected campaign files differ from declared Git HEAD")

    git_version = _run(git, repo, ["--version"]).stdout.strip()
    if not git_version.startswith("git version "):
        raise CE(f"unexpected git version response: {git_version!r}")

    return {
        "object_format": object_format,
        "head_commit_oid": head,
        "head_tree_oid": tree,
        "git_version": git_version,
        "git_executable_path": str(git),
        "git_executable_sha256": hf(git),
        "protected_paths_tracked": paths,
        "protected_paths_clean_against_head": True,
    }
