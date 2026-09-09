#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import verify_workbench_invocation_isolation_profile as isolation_verifier

SCHEMA = "symthaea-workbench-isolated-execution-observation-v1"
STATUS = "observed-isolated-execution-unqualified"
ROOT_ROLES = (
    "cwd", "home", "xdg_config_home", "xdg_cache_home", "xdg_data_home",
    "xdg_state_home", "xdg_runtime_dir", "xdg_config_dirs", "xdg_data_dirs", "tmpdir",
)
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}\Z")
STEM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
AUTHORITY = {
    "executor_mechanism_qualified": False,
    "real_workbench_invocation_verified": False,
    "same_host_repeatability_established": False,
    "path_equivalence_established": False,
    "cross_cpu_equivalence_established": False,
    "workbench_execution_qualified": False,
    "scientific_execution_qualified": False,
    "transform_executed": False,
    "atlas_correctness_established": False,
    "fmq010_established": False,
    "neural_alignment_established": False,
    "consciousness_evidence": False,
}


class ExecutionError(ValueError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def digest_regular_file(path: Path, label: str) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ExecutionError(f"{label}: unable to open regular file") from exc
    h = hashlib.sha256()
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ExecutionError(f"{label}: opened object is not a regular file")
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    finally:
        os.close(fd)
    return "sha256:" + h.hexdigest()


def strict_expected_sha256(value: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ExecutionError("expected executable SHA-256: canonical sha256:<64 lowercase hex> required")
    return value


def resolve_executable(path: Path) -> Path:
    supplied = path.expanduser()
    if not supplied.is_absolute():
        raise ExecutionError("executable: absolute path required")
    if supplied.is_symlink():
        raise ExecutionError("executable: direct symlink forbidden")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise ExecutionError("executable: file not found") from exc
    if resolved != supplied or not resolved.is_file():
        raise ExecutionError("executable: exact regular file required")
    if not os.access(resolved, os.X_OK):
        raise ExecutionError("executable: execute permission required")
    return resolved


def verify_profiles(isolation_profile_path: Path, parent_profile_path: Path) -> dict[str, Any]:
    return isolation_verifier.verify_profile(
        isolation_verifier.load(isolation_profile_path),
        isolation_verifier.load(parent_profile_path),
    )


def create_isolation_roots(parent: Path, stem: str) -> tuple[Path, dict[str, Path], dict[str, dict[str, Any]]]:
    if not isinstance(stem, str) or not STEM_RE.fullmatch(stem):
        raise ExecutionError("isolation stem: canonical 1..64 character token required")
    try:
        parent = parent.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ExecutionError("scratch parent: existing directory required") from exc
    if not parent.is_dir():
        raise ExecutionError("scratch parent: directory required")
    base = Path(tempfile.mkdtemp(prefix=f".{stem}.isolation-", dir=parent))
    try:
        os.chmod(base, 0o700)
        roots: dict[str, Path] = {}
        lifecycle: dict[str, dict[str, Any]] = {}
        for role in ROOT_ROLES:
            p = base / role
            p.mkdir(mode=0o700)
            resolved = p.resolve(strict=True)
            mode = stat.S_IMODE(resolved.stat().st_mode)
            empty = not any(resolved.iterdir())
            if mode != 0o700 or not empty:
                raise ExecutionError(f"isolation root {role}: fresh private empty directory required")
            roots[role] = resolved
            lifecycle[role] = {
                "path": str(resolved),
                "directory_mode_octal": "0700",
                "empty_before": True,
                "reuse_allowed": False,
            }
        if len({str(p) for p in roots.values()}) != len(ROOT_ROLES):
            raise ExecutionError("isolation roots: all role roots must be distinct")
        return base, roots, lifecycle
    except Exception as exc:
        try:
            if os.path.lexists(base):
                shutil.rmtree(base)
        except OSError as cleanup_exc:
            raise ExecutionError("isolation roots: construction cleanup could not be confirmed") from cleanup_exc
        if os.path.lexists(base):
            raise ExecutionError("isolation roots: construction cleanup could not be confirmed") from exc
        if isinstance(exc, ExecutionError):
            raise
        raise ExecutionError("isolation roots: construction failed") from exc


def entry_environment(profile: dict[str, Any], roots: dict[str, Path]) -> dict[str, str]:
    contract = profile["process_environment"]
    fixed = dict(contract["fixed"])
    dynamic = contract["dynamic_bindings"]
    values = {
        "invocation.cwd": roots["cwd"],
        "invocation.home": roots["home"],
        "invocation.xdg_config_home": roots["xdg_config_home"],
        "invocation.xdg_cache_home": roots["xdg_cache_home"],
        "invocation.xdg_data_home": roots["xdg_data_home"],
        "invocation.xdg_state_home": roots["xdg_state_home"],
        "invocation.xdg_runtime_dir": roots["xdg_runtime_dir"],
        "invocation.xdg_config_dirs": roots["xdg_config_dirs"],
        "invocation.xdg_data_dirs": roots["xdg_data_dirs"],
        "invocation.tmpdir": roots["tmpdir"],
    }
    env = fixed
    for name, binding in dynamic.items():
        if binding not in values:
            raise ExecutionError(f"process environment: unsupported dynamic binding {binding}")
        env[name] = str(values[binding])
    if set(env) != set(fixed) | set(dynamic):
        raise ExecutionError("process environment: exact entry key set required")
    if env["PWD"] != str(roots["cwd"]):
        raise ExecutionError("process environment: PWD must equal cwd")
    if env["TMPDIR"] != env["TMP"] or env["TMPDIR"] != env["TEMP"]:
        raise ExecutionError("process environment: temp aliases must share one root")
    return env


def _cleanup_base(base: Path) -> None:
    try:
        if os.path.lexists(base):
            shutil.rmtree(base)
    except OSError as exc:
        raise ExecutionError("isolation cleanup could not be confirmed") from exc
    if os.path.lexists(base):
        raise ExecutionError("isolation cleanup could not be confirmed")


def observe_isolated_execution(
    executable: Path,
    args: list[str],
    expected_executable_sha256: str,
    isolation_profile_path: Path,
    parent_profile_path: Path,
    scratch_parent: Path,
    stem: str = "workbench",
) -> dict[str, Any]:
    if not isinstance(args, list) or any(not isinstance(arg, str) for arg in args):
        raise ExecutionError("argv: list[str] required")
    if any("\x00" in arg for arg in args):
        raise ExecutionError("argv: NUL forbidden")
    expected = strict_expected_sha256(expected_executable_sha256)
    program = resolve_executable(executable)
    profile = verify_profiles(isolation_profile_path, parent_profile_path)
    pre_sha = digest_regular_file(program, "executable")
    if pre_sha != expected:
        raise ExecutionError("executable: pre-execution byte identity mismatch")

    base: Path | None = None
    previous_umask: int | None = None
    observation: dict[str, Any] | None = None
    try:
        base, roots, lifecycle = create_isolation_roots(scratch_parent, stem)
        env = entry_environment(profile, roots)
        argv = [str(program), *args]
        previous_umask = os.umask(0o077)
        try:
            try:
                proc = subprocess.run(
                    argv,
                    cwd=roots["cwd"],
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=False,
                    close_fds=True,
                    pass_fds=(),
                    check=False,
                )
            except OSError as exc:
                raise ExecutionError("isolated process: launch failed") from exc
        finally:
            os.umask(previous_umask)
            previous_umask = None
        post_sha = digest_regular_file(program, "executable")
        if post_sha != expected:
            raise ExecutionError("executable: post-execution byte identity mismatch")
        stdout = bytes(proc.stdout)
        stderr = bytes(proc.stderr)
        observation = {
            "schema": SCHEMA,
            "status": STATUS,
            "executable": {
                "path": str(program),
                "expected_sha256": expected,
                "pre_sha256": pre_sha,
                "post_sha256": post_sha,
            },
            "command": {
                "argv": argv,
                "exit_code": int(proc.returncode),
                "cwd": str(roots["cwd"]),
                "stdin": "devnull",
                "close_fds": True,
                "pass_fds": [],
                "umask_octal": "0077",
            },
            "environment": {
                "stage": profile["process_environment"]["stage"],
                "inherit_host_environment": False,
                "entries": env,
                "entry_environment_sha256": digest_bytes(canonical_json_bytes(env)),
            },
            "isolation": {
                "roots": lifecycle,
                "cleanup_confirmed": False,
            },
            "stdout": stdout,
            "stderr": stderr,
            "stdout_sha256": digest_bytes(stdout),
            "stderr_sha256": digest_bytes(stderr),
            "authority": dict(AUTHORITY),
        }
    finally:
        if previous_umask is not None:
            os.umask(previous_umask)
        if base is not None:
            _cleanup_base(base)

    if observation is None:
        raise ExecutionError("isolated execution: no observation produced")
    observation["isolation"]["cleanup_confirmed"] = True
    for item in observation["isolation"]["roots"].values():
        item["cleanup_confirmed"] = True
    return observation


def require_success(observation: dict[str, Any]) -> dict[str, Any]:
    if observation.get("schema") != SCHEMA or observation.get("status") != STATUS:
        raise ExecutionError("execution observation: schema/status mismatch")
    if observation.get("command", {}).get("exit_code") != 0:
        raise ExecutionError("isolated process returned nonzero exit status")
    exe = observation.get("executable", {})
    expected = exe.get("expected_sha256")
    if exe.get("pre_sha256") != expected or exe.get("post_sha256") != expected:
        raise ExecutionError("execution observation: executable byte identity mismatch")
    if observation.get("isolation", {}).get("cleanup_confirmed") is not True:
        raise ExecutionError("execution observation: cleanup must be confirmed")
    return observation
