#!/usr/bin/env python3
"""Minimal #681-conformant subprocess executor for Workbench-class programs.

This module performs no Nix discovery, no version parsing, no CPU diagnostics, no
scientific interpretation, and grants no authority. It only creates the qualified
root-main-program entry boundary, executes one exact absolute program, retains the
process streams/status, verifies program bytes before and after execution, and
confirms isolation-root cleanup before returning.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}\Z")

FIXED_ENTRY_ENVIRONMENT: Mapping[str, str] = MappingProxyType({
    "LANG": "C",
    "LC_ALL": "C",
    "TZ": "UTC0",
    "OMP_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "PATH": "",
})

ROOT_ROLES = (
    "cwd",
    "home",
    "xdg_config_home",
    "xdg_cache_home",
    "xdg_data_home",
    "xdg_state_home",
    "xdg_runtime_dir",
    "xdg_config_dirs",
    "xdg_data_dirs",
    "tmpdir",
)

DYNAMIC_ENTRY_BINDINGS: Mapping[str, str] = MappingProxyType({
    "PWD": "cwd",
    "HOME": "home",
    "XDG_CONFIG_HOME": "xdg_config_home",
    "XDG_CACHE_HOME": "xdg_cache_home",
    "XDG_DATA_HOME": "xdg_data_home",
    "XDG_STATE_HOME": "xdg_state_home",
    "XDG_RUNTIME_DIR": "xdg_runtime_dir",
    "XDG_CONFIG_DIRS": "xdg_config_dirs",
    "XDG_DATA_DIRS": "xdg_data_dirs",
    "TMPDIR": "tmpdir",
    "TMP": "tmpdir",
    "TEMP": "tmpdir",
})

DIRECTORY_MODE = 0o700
PROCESS_UMASK = 0o077


class ExecutionContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class IsolatedInvocationResultV1:
    """Authority-free result of one isolated process invocation."""

    executable: Path
    argv: tuple[str, ...]
    exit_code: int
    stdout: bytes
    stderr: bytes
    pre_executable_sha256: str
    post_executable_sha256: str
    entry_environment: Mapping[str, str]
    entry_environment_sha256: str
    isolation_roots: Mapping[str, Path]
    cleanup_confirmed: bool
    stdin_policy: str = "devnull"
    close_fds: bool = True
    pass_fds: tuple[int, ...] = ()
    umask_octal: str = "0077"


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _strict_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ExecutionContractError(f"{label}: expected sha256:<64 lowercase hex>")
    return value


def _strict_argument(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ExecutionContractError(f"{label}: string required")
    if "\x00" in value:
        raise ExecutionContractError(f"{label}: NUL forbidden")
    return value


def _open_regular(path: Path, label: str) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ExecutionContractError(f"{label}: unable to open exact regular file") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ExecutionContractError(f"{label}: opened object is not a regular file")
        return fd
    except Exception:
        os.close(fd)
        raise


def digest_regular_file(path: Path, label: str) -> str:
    fd = _open_regular(path, label)
    hasher = hashlib.sha256()
    try:
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    finally:
        os.close(fd)
    return "sha256:" + hasher.hexdigest()


def resolve_exact_executable(path: Path) -> Path:
    if not isinstance(path, Path):
        path = Path(path)
    if not path.is_absolute():
        raise ExecutionContractError("program: absolute executable path required")
    if path.is_symlink():
        raise ExecutionContractError("program: main executable symlink forbidden")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ExecutionContractError("program: executable not found") from exc
    if resolved != path or not resolved.is_file():
        raise ExecutionContractError("program: exact regular absolute executable required")
    _open_fd = _open_regular(resolved, "program")
    os.close(_open_fd)
    return resolved


def _resolve_parent(parent: Path) -> Path:
    expanded = parent.expanduser()
    if expanded.is_symlink():
        raise ExecutionContractError("isolation parent: direct symlink forbidden")
    try:
        resolved = expanded.resolve(strict=True)
    except OSError as exc:
        raise ExecutionContractError("isolation parent: directory not found") from exc
    if not resolved.is_dir():
        raise ExecutionContractError("isolation parent: directory required")
    return resolved


def create_isolation_roots(parent: Path, stem: str) -> tuple[Path, dict[str, Path]]:
    parent = _resolve_parent(parent)
    stem = _strict_argument(stem, "isolation stem")
    if not stem or "/" in stem:
        raise ExecutionContractError("isolation stem: non-empty path component required")

    base = Path(tempfile.mkdtemp(prefix=f".{stem}.isolation-", dir=parent))
    try:
        os.chmod(base, DIRECTORY_MODE)
        if stat.S_IMODE(base.stat().st_mode) != DIRECTORY_MODE:
            raise ExecutionContractError("isolation base: mode 0700 required")
        roots: dict[str, Path] = {}
        for role in ROOT_ROLES:
            root = base / role
            root.mkdir(mode=DIRECTORY_MODE)
            resolved = root.resolve(strict=True)
            if resolved != root:
                raise ExecutionContractError(f"isolation root {role}: exact path required")
            if stat.S_IMODE(resolved.stat().st_mode) != DIRECTORY_MODE:
                raise ExecutionContractError(f"isolation root {role}: mode 0700 required")
            if any(resolved.iterdir()):
                raise ExecutionContractError(f"isolation root {role}: fresh empty directory required")
            roots[role] = resolved
        if len({str(path) for path in roots.values()}) != len(ROOT_ROLES):
            raise ExecutionContractError("isolation roots: distinct paths required")
        return base, roots
    except Exception as exc:
        try:
            if os.path.lexists(base):
                shutil.rmtree(base)
        except OSError as cleanup_exc:
            raise ExecutionContractError(
                "isolation roots: construction failed and cleanup not confirmed"
            ) from cleanup_exc
        if os.path.lexists(base):
            raise ExecutionContractError(
                "isolation roots: construction failed and cleanup not confirmed"
            ) from exc
        raise


def build_entry_environment(roots: Mapping[str, Path]) -> dict[str, str]:
    if set(roots) != set(ROOT_ROLES):
        raise ExecutionContractError("entry environment: exact isolation root roles required")
    environment = dict(FIXED_ENTRY_ENVIRONMENT)
    for variable, role in DYNAMIC_ENTRY_BINDINGS.items():
        path = roots[role]
        if not path.is_absolute():
            raise ExecutionContractError(f"entry environment {variable}: absolute path required")
        environment[variable] = str(path)
    expected_keys = set(FIXED_ENTRY_ENVIRONMENT) | set(DYNAMIC_ENTRY_BINDINGS)
    if set(environment) != expected_keys:
        raise ExecutionContractError("entry environment: exact variable set required")
    return environment


def _cleanup_isolation_base(base: Path) -> None:
    try:
        if os.path.lexists(base):
            shutil.rmtree(base)
    except OSError as exc:
        raise ExecutionContractError("isolation cleanup: cleanup not confirmed") from exc
    if os.path.lexists(base):
        raise ExecutionContractError("isolation cleanup: cleanup not confirmed")


def invoke_isolated(
    executable: Path,
    arguments: Sequence[str],
    *,
    expected_executable_sha256: str,
    isolation_parent: Path,
    stem: str = "workbench",
) -> IsolatedInvocationResultV1:
    """Execute one exact absolute program under the v1 isolated entry contract.

    A non-zero child exit status is returned as process evidence rather than being
    converted into a contract failure. Contract failures include wrong or changing
    executable bytes, invalid process-boundary inputs, launch failure, or unconfirmed
    cleanup.
    """

    expected_digest = _strict_sha256(expected_executable_sha256, "program digest")
    program = resolve_exact_executable(executable)
    tail = tuple(_strict_argument(value, f"argument[{index}]") for index, value in enumerate(arguments))
    argv = (str(program), *tail)

    pre_digest = digest_regular_file(program, "program")
    if pre_digest != expected_digest:
        raise ExecutionContractError("program: pre-execution byte identity mismatch")

    base: Path | None = None
    result: IsolatedInvocationResultV1 | None = None
    try:
        base, roots = create_isolation_roots(isolation_parent, stem)
        environment = build_entry_environment(roots)
        environment_digest = digest_bytes(canonical_json_bytes(environment))

        previous_umask = os.umask(PROCESS_UMASK)
        try:
            try:
                process = subprocess.run(
                    list(argv),
                    cwd=roots["cwd"],
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=False,
                    close_fds=True,
                    pass_fds=(),
                    check=False,
                )
            except OSError as exc:
                raise ExecutionContractError("program: isolated process launch failed") from exc
        finally:
            os.umask(previous_umask)

        post_digest = digest_regular_file(program, "program")
        if post_digest != expected_digest or post_digest != pre_digest:
            raise ExecutionContractError("program: byte identity changed during execution")

        result = IsolatedInvocationResultV1(
            executable=program,
            argv=argv,
            exit_code=process.returncode,
            stdout=bytes(process.stdout),
            stderr=bytes(process.stderr),
            pre_executable_sha256=pre_digest,
            post_executable_sha256=post_digest,
            entry_environment=MappingProxyType(dict(environment)),
            entry_environment_sha256=environment_digest,
            isolation_roots=MappingProxyType(dict(roots)),
            cleanup_confirmed=False,
        )
    finally:
        if base is not None:
            _cleanup_isolation_base(base)

    if result is None:
        raise ExecutionContractError("program: isolated invocation incomplete")
    return replace(result, cleanup_confirmed=True)
