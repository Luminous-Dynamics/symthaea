#!/usr/bin/env python3
"""Observation-only Workbench -version invocation under the qualified isolation profile."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import verify_workbench_invocation_isolation_profile as isolation_verifier

SCHEMA = "symthaea-workbench-isolated-version-invocation-receipt-v1"
CLOSURE_SCHEMA = "symthaea-workbench-nix-closure-capture-receipt-v1"
CLOSURE_STATUS = "observed-normalized-unqualified"
STATUS_SUCCESS = "observed-isolated-version-invocation-unqualified"
STATUS_FAILURE = "isolated-version-invocation-incomplete-unqualified"
STORE_PATH_RE = re.compile(r"^/nix/store/[0123456789abcdfghijklmnpqrsvwxyz]{32}-[^/\r\n]+\Z")
VERSION_LINE_RE = re.compile(r"(?m)^Version:[ \t]+([0-9]+\.[0-9]+\.[0-9]+)[ \t]*\r?$")
ROOT_ROLES = (
    "cwd", "home", "xdg_config_home", "xdg_cache_home", "xdg_data_home",
    "xdg_state_home", "xdg_runtime_dir", "xdg_config_dirs", "xdg_data_dirs", "tmpdir",
)
AUTHORITY = {
    "program_membership_verified": False,
    "invocation_profile_verified": False,
    "invocation_executed": False,
    "version_output_bound": False,
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


class CaptureError(ValueError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise CaptureError(f"JSON object: duplicate key: {key}")
        out[key] = value
    return out


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)


def strict_store_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not STORE_PATH_RE.fullmatch(value):
        raise CaptureError(f"{label}: canonical /nix/store path required")
    return value


def derive_untrusted_root(closure_receipt_dir: Path) -> tuple[str, str]:
    receipt_path = closure_receipt_dir / "receipt.json"
    receipt = load_json(receipt_path)
    if not isinstance(receipt, dict) or receipt.get("schema") != CLOSURE_SCHEMA:
        raise CaptureError("closure receipt: schema mismatch")
    if receipt.get("status") != CLOSURE_STATUS:
        raise CaptureError("closure receipt: complete normalized observation required")
    root = strict_store_path(receipt.get("root"), "closure receipt root")
    capture_digest = receipt.get("capture_digest")
    if not isinstance(capture_digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", capture_digest):
        raise CaptureError("closure receipt: canonical capture digest required")
    return root, capture_digest


def cpu_diagnostics() -> dict[str, str]:
    fields: dict[str, str] = {}
    flags: list[str] = []
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        first: dict[str, str] = {}
        for line in cpuinfo.read_text(encoding="utf-8", errors="strict").splitlines():
            if not line.strip():
                if first:
                    break
                continue
            if ":" in line:
                key, value = line.split(":", 1)
                first[key.strip()] = value.strip()
        fields["cpu_vendor"] = first.get("vendor_id", "")
        fields["cpu_family"] = first.get("cpu family", "")
        fields["cpu_model"] = first.get("model", "")
        fields["cpu_stepping"] = first.get("stepping", "")
        flags = sorted(set(first.get("flags", "").split()))
    for key in ("cpu_vendor", "cpu_family", "cpu_model", "cpu_stepping"):
        if not fields.get(key):
            raise CaptureError(f"diagnostics: missing {key}")
    if not flags:
        raise CaptureError("diagnostics: CPU flags unavailable")
    fields["cpu_flags_digest"] = digest_bytes(("\n".join(flags) + "\n").encode("ascii"))
    fields["kernel_release"] = platform.release()
    if not fields["kernel_release"]:
        raise CaptureError("diagnostics: kernel release unavailable")
    return fields


def create_isolation_roots(parent: Path, stem: str) -> tuple[Path, dict[str, Path], dict[str, dict[str, Any]]]:
    base = Path(tempfile.mkdtemp(prefix=f".{stem}.isolation-", dir=parent))
    try:
        os.chmod(base, 0o700)
        roots: dict[str, Path] = {}
        lifecycle: dict[str, dict[str, Any]] = {}
        for role in ROOT_ROLES:
            path = base / role
            path.mkdir(mode=0o700)
            resolved = path.resolve(strict=True)
            mode = stat.S_IMODE(resolved.stat().st_mode)
            empty = not any(resolved.iterdir())
            if mode != 0o700 or not empty:
                raise CaptureError(f"isolation root {role}: fresh private empty directory required")
            roots[role] = resolved
            lifecycle[role] = {
                "path": str(resolved),
                "directory_mode_octal": "0700",
                "empty_before": True,
                "reuse_allowed": False,
            }
        if len({str(path) for path in roots.values()}) != len(roots):
            raise CaptureError("isolation roots: distinct role roots required")
        return base, roots, lifecycle
    except Exception as exc:
        try:
            if os.path.lexists(base):
                shutil.rmtree(base)
        except OSError as cleanup_exc:
            raise CaptureError(
                "isolation roots: construction failed and cleanup could not be confirmed"
            ) from cleanup_exc
        if os.path.lexists(base):
            raise CaptureError(
                "isolation roots: construction failed and cleanup could not be confirmed"
            ) from exc
        raise


def entry_environment(profile: dict[str, Any], roots: dict[str, Path]) -> dict[str, str]:
    env_contract = profile["process_environment"]
    fixed = dict(env_contract["fixed"])
    dynamic = env_contract["dynamic_bindings"]
    binding_values = {
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
        try:
            path = binding_values[binding]
        except KeyError as exc:
            raise CaptureError(f"process environment: unsupported dynamic binding {binding}") from exc
        env[name] = str(path)
    if set(env) != set(fixed) | set(dynamic):
        raise CaptureError("process environment: exact entry key set required")
    return env


def parse_version(stdout: bytes, stderr: bytes, expected: str) -> str:
    try:
        text = (stdout + stderr).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CaptureError("Workbench -version output: UTF-8 required") from exc
    matches = VERSION_LINE_RE.findall(text)
    if matches != [expected]:
        raise CaptureError(f"Workbench -version output: exact Version line for {expected} required")
    return matches[0]


def run_version(
    executable: Path,
    profile: dict[str, Any],
    parent_dir: Path,
    stem: str,
) -> dict[str, Any]:
    base: Path | None = None
    try:
        base, roots, lifecycle = create_isolation_roots(parent_dir, stem)
        env = entry_environment(profile, roots)
        argv = [str(executable), "-version"]
        pre_sha = digest_file(executable)
        previous_umask = os.umask(0o077)
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
        finally:
            os.umask(previous_umask)
        post_sha = digest_file(executable)
        return {
            "argv": argv,
            "exit_code": proc.returncode,
            "cwd": str(roots["cwd"]),
            "stdin": "devnull",
            "close_fds": True,
            "pass_fds": [],
            "umask_octal": "0077",
            "environment": env,
            "roots": lifecycle,
            "pre_executable_sha256": pre_sha,
            "post_executable_sha256": post_sha,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "_base": base,
        }
    except Exception:
        if base is not None and base.exists():
            shutil.rmtree(base)
        raise


def publish_capture(staging: Path, final_out: Path) -> None:
    try:
        final_out.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise CaptureError("output: destination appeared before publish") from exc

    published = False
    try:
        for child in sorted(staging.iterdir(), key=lambda path: path.name):
            os.rename(child, final_out / child.name)
        staging.rmdir()
        published = True
    finally:
        if not published and os.path.lexists(final_out):
            shutil.rmtree(final_out)


def capture(
    closure_receipt_dir: Path,
    isolation_profile_path: Path,
    parent_profile_path: Path,
    out_dir: Path,
) -> int:
    requested = out_dir.expanduser()
    if not requested.name or requested.name in {".", ".."}:
        raise CaptureError("output: canonical destination name required")
    out_parent = requested.parent.resolve(strict=True)
    final_out = out_parent / requested.name
    if os.path.lexists(final_out):
        raise CaptureError("output: existing destination forbidden")
    root, closure_capture_digest = derive_untrusted_root(closure_receipt_dir)

    isolation_profile = isolation_verifier.verify_profile(
        isolation_verifier.load(isolation_profile_path),
        isolation_verifier.load(parent_profile_path),
    )
    parent_profile = isolation_verifier.load(parent_profile_path)
    expected_version = parent_profile["nixpkgs_package"]["version"]
    executable = Path(root) / isolation_profile["program_binding"]["relative_main_program"]
    if not executable.is_file() or executable.is_symlink():
        raise CaptureError("program: regular root main program required")

    temp: Path | None = Path(tempfile.mkdtemp(prefix=f".{final_out.name}.capture-", dir=out_parent))
    os.chmod(temp, 0o700)
    isolation_base: Path | None = None
    try:
        result = run_version(executable, isolation_profile, out_parent, final_out.name)
        isolation_base = result.pop("_base")
        raw = temp / "raw"
        raw.mkdir(mode=0o700)
        stdout = result.pop("stdout")
        stderr = result.pop("stderr")
        stdout_path = raw / "wb-version.stdout"
        stderr_path = raw / "wb-version.stderr"
        stdout_path.write_bytes(stdout)
        stderr_path.write_bytes(stderr)
        os.chmod(stdout_path, 0o600)
        os.chmod(stderr_path, 0o600)

        version: str | None = None
        success = False
        if (
            result["exit_code"] == 0
            and result["pre_executable_sha256"] == result["post_executable_sha256"]
        ):
            try:
                version = parse_version(stdout, stderr, expected_version)
                success = True
            except CaptureError:
                success = False

        if isolation_base.exists():
            shutil.rmtree(isolation_base)
        cleanup_confirmed = not os.path.lexists(isolation_base)
        isolation_base = None
        for lifecycle in result["roots"].values():
            lifecycle["cleanup_confirmed"] = cleanup_confirmed
        if not cleanup_confirmed:
            success = False

        stream_stdout = {
            "path": "raw/wb-version.stdout",
            "byte_length": len(stdout),
            "sha256": digest_bytes(stdout),
        }
        stream_stderr = {
            "path": "raw/wb-version.stderr",
            "byte_length": len(stderr),
            "sha256": digest_bytes(stderr),
        }
        combined = stdout + stderr
        receipt: dict[str, Any] = {
            "schema": SCHEMA,
            "status": STATUS_SUCCESS if success else STATUS_FAILURE,
            "closure_capture_digest": closure_capture_digest,
            "root": root,
            "executable": {
                "path": str(executable),
                "pre_sha256": result["pre_executable_sha256"],
                "post_sha256": result["post_executable_sha256"],
            },
            "implementations": {
                "producer_sha256": digest_file(Path(__file__)),
                "isolation_verifier_sha256": digest_file(Path(isolation_verifier.__file__).resolve()),
            },
            "profiles": {
                "execution_capsule_sha256": digest_file(parent_profile_path),
                "invocation_isolation_sha256": digest_file(isolation_profile_path),
            },
            "command": {
                key: result[key]
                for key in ("argv", "exit_code", "cwd", "stdin", "close_fds", "pass_fds", "umask_octal")
            },
            "environment": {
                "stage": isolation_profile["process_environment"]["stage"],
                "inherit_host_environment": False,
                "entries": result["environment"],
                "entry_environment_sha256": digest_bytes(canonical_json_bytes(result["environment"])),
            },
            "isolation": {
                "roots": result["roots"],
                "cleanup_confirmed": cleanup_confirmed,
            },
            "diagnostics": cpu_diagnostics(),
            "stdout": stream_stdout,
            "stderr": stream_stderr,
            "version_output": {
                "aggregation": "stdout-then-stderr-v1",
                "byte_length": len(combined),
                "sha256": digest_bytes(combined),
                "parsed_version": version,
            },
            "authority": dict(AUTHORITY),
            "capture_digest": "",
        }
        receipt["capture_digest"] = digest_bytes(
            canonical_json_bytes({k: v for k, v in receipt.items() if k != "capture_digest"})
        )
        (temp / "receipt.json").write_bytes(canonical_json_bytes(receipt) + b"\n")
        os.chmod(temp / "receipt.json", 0o600)
        publish_capture(temp, final_out)
        temp = None
        return 0 if success else 2
    finally:
        if isolation_base is not None and isolation_base.exists():
            shutil.rmtree(isolation_base)
        if temp is not None and temp.exists():
            shutil.rmtree(temp)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--closure-receipt-dir", required=True, type=Path)
    parser.add_argument("--isolation-profile", required=True, type=Path)
    parser.add_argument("--parent-profile", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        return capture(
            args.closure_receipt_dir,
            args.isolation_profile,
            args.parent_profile,
            args.out,
        )
    except (CaptureError, OSError, json.JSONDecodeError, isolation_verifier.ContractError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
