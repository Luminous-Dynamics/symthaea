#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import capture_hcpmmp1_neuromaps_lineage_b_qualified_run_v2 as qualified_run
import capture_hcpmmp1_neuromaps_lineage_b_verified_run as run_capture
import verify_workbench_nix_closure_capture as closure_capture_verifier
import workbench_nix_closure_identity as closure_identity

SCHEMA = "symthaea-hcpmmp1-lineage-b-local-closure-readmission-v1"
STATUS = "local-qualified-workbench-closure-readmitted"
EXPECTED_NIX_VERSION = "2.33.6"
EXPECTED_ROOT = "/nix/store/g4bn2ilmgr5xpsazz6jw8bn8n0398lna-connectome-workbench-2.1.0"
EXPECTED_PROGRAM_SHA256 = "sha256:ad461ffeef56a0d807617e41ca65e38a2c25f1ec47abaea560a0bf843613db88"
EXPECTED_VERSION_OUTPUT_SHA256 = "sha256:d4e353408c9d76bc7c4ffe476e4161e0dc443add050ad499baa2cb93c84630c5"
EXPECTED_CLOSURE_DIGEST = "sha256:4b9820c088e3ab1481833c1659c449b0acc4b168f172d8f43abf383fae6b8a6a"
EXPECTED_VERIFICATION_FILE_SHA256 = qualified_run.QUALIFIED_VERIFICATION_FILE_SHA256
EXPECTED_PROFILE_FILE_SHA256 = qualified_run.QUALIFIED_PROFILE_FILE_SHA256
PROGRAM_RELATIVE_PATH = "bin/wb_command"
NIX_VERSION_RE = re.compile(r"^nix \(Nix\) ([0-9]+\.[0-9]+\.[0-9]+)\r?\n?$", re.ASCII)
AUTHORITY = {
    "qualified_workbench_trust_package_bound": True,
    "local_nix_version_verified": True,
    "local_closure_metadata_matches": True,
    "local_closure_contents_verified": True,
    "local_program_bytes_match": True,
    "local_closure_readmitted": True,
    "workbench_execution_qualified": False,
    "scientific_execution_qualified": False,
    "transform_executed": False,
    "same_host_repeatability_established": False,
    "path_equivalence_established": False,
    "cross_cpu_equivalence_established": False,
    "atlas_correctness_established": False,
    "fmq010_established": False,
    "neural_alignment_established": False,
    "consciousness_evidence": False,
}


class ReadmissionError(ValueError):
    pass


def digest_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def digest_regular_file(path: Path, label: str) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ReadmissionError(f"{label}: unable to open regular file") from exc
    h = hashlib.sha256()
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ReadmissionError(f"{label}: opened object is not regular")
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    finally:
        os.close(fd)
    return "sha256:" + h.hexdigest()


def bind_qualified_trust_package() -> dict[str, Any]:
    try:
        profile_raw = qualified_run.PROFILE_PATH.read_bytes()
        verification_raw = qualified_run.VERIFICATION_PATH.read_bytes()
    except OSError as exc:
        raise ReadmissionError("qualified trust package: retained files unavailable") from exc
    if digest_bytes(profile_raw) != EXPECTED_PROFILE_FILE_SHA256:
        raise ReadmissionError("qualified trust package: profile root mismatch")
    if digest_bytes(verification_raw) != EXPECTED_VERIFICATION_FILE_SHA256:
        raise ReadmissionError("qualified trust package: verification root mismatch")
    try:
        profile = run_capture.verify_profile_bytes(profile_raw)
        verification = run_capture.verify_retained_workbench_bytes(profile, verification_raw)
    except (run_capture.CaptureError, run_capture.ContractError, json.JSONDecodeError) as exc:
        raise ReadmissionError(str(exc)) from exc
    expected = {
        "root": EXPECTED_ROOT,
        "program_content_sha256": EXPECTED_PROGRAM_SHA256,
        "version_output_sha256": EXPECTED_VERSION_OUTPUT_SHA256,
        "closure_digest": EXPECTED_CLOSURE_DIGEST,
        "qualification_platform": "x86_64-linux",
        "version": "2.1.0",
    }
    for key, value in expected.items():
        if verification.get(key) != value:
            raise ReadmissionError(f"qualified trust package: unexpected {key}")
    return verification


def resolve_nix_executable() -> Path:
    raw = shutil.which("nix")
    if not raw:
        raise ReadmissionError("nix executable not found on PATH")
    try:
        resolved = Path(raw).resolve(strict=True)
    except OSError as exc:
        raise ReadmissionError("nix executable cannot be resolved") from exc
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ReadmissionError("nix executable must resolve to executable regular file")
    return resolved


def parse_nix_version(stdout: bytes) -> str:
    try:
        text = stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReadmissionError("nix --version: UTF-8 required") from exc
    match = NIX_VERSION_RE.fullmatch(text)
    if not match or match.group(1) != EXPECTED_NIX_VERSION:
        raise ReadmissionError(f"nix --version: exact Nix {EXPECTED_NIX_VERSION} required")
    return match.group(1)


def _run(argv: list[str]) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            close_fds=True,
            pass_fds=(),
            check=False,
        )
    except OSError as exc:
        raise ReadmissionError(f"command launch failed: {argv[1] if len(argv) > 1 else argv[0]}") from exc


def compile_local_identity(path_info_stdout: bytes, root: str = EXPECTED_ROOT) -> dict[str, Any]:
    try:
        entries = closure_capture_verifier.parse_path_info_v2(path_info_stdout)
        identity = closure_identity.compile_identity(root, entries)
    except (closure_capture_verifier.VerificationError, closure_identity.ContractError) as exc:
        raise ReadmissionError(str(exc)) from exc
    if identity["closure_digest"] != EXPECTED_CLOSURE_DIGEST:
        raise ReadmissionError("local closure: qualified closure digest mismatch")
    return identity


def verify_local_closure(
    *,
    run_command: Callable[[list[str]], subprocess.CompletedProcess[bytes]] = _run,
    nix_executable: Path | None = None,
) -> dict[str, Any]:
    verification = bind_qualified_trust_package()
    nix = nix_executable or resolve_nix_executable()
    nix = nix.resolve(strict=True)

    version_cmd = [str(nix), "--version"]
    version_result = run_command(version_cmd)
    if type(version_result.returncode) is not int or version_result.returncode != 0:
        raise ReadmissionError("nix --version failed")
    nix_version = parse_nix_version(bytes(version_result.stdout))

    root = verification["root"]
    if root != EXPECTED_ROOT:
        raise ReadmissionError("local closure: exact qualified root required")
    program = Path(root) / PROGRAM_RELATIVE_PATH
    if program.is_symlink():
        raise ReadmissionError("local closure: qualified main program symlink forbidden")
    try:
        resolved_program = program.resolve(strict=True)
    except OSError as exc:
        raise ReadmissionError("local closure: qualified main program unavailable") from exc
    if resolved_program != program or not resolved_program.is_file():
        raise ReadmissionError("local closure: exact regular main program required")
    program_before = digest_regular_file(program, "qualified main program")
    if program_before != EXPECTED_PROGRAM_SHA256:
        raise ReadmissionError("local closure: main program bytes differ from qualified root")

    path_info_cmd = [
        str(nix), "--offline", "path-info", "--json", "--json-format", "2", "--recursive", root,
    ]
    path_info = run_command(path_info_cmd)
    if type(path_info.returncode) is not int or path_info.returncode != 0:
        raise ReadmissionError("local closure: recursive path-info failed")
    identity = compile_local_identity(bytes(path_info.stdout), root)

    verify_cmd = [str(nix), "--offline", "store", "verify", "--recursive", "--no-trust", root]
    content_verify = run_command(verify_cmd)
    if type(content_verify.returncode) is not int or content_verify.returncode != 0:
        raise ReadmissionError("local closure: recursive NAR-content verification failed")

    program_after = digest_regular_file(program, "qualified main program")
    if program_after != program_before or program_after != EXPECTED_PROGRAM_SHA256:
        raise ReadmissionError("local closure: main program bytes changed during readmission")

    return {
        "schema": SCHEMA,
        "status": STATUS,
        "qualified_workbench": {
            "root": root,
            "program_content_sha256": EXPECTED_PROGRAM_SHA256,
            "version_output_sha256": EXPECTED_VERSION_OUTPUT_SHA256,
            "closure_digest": identity["closure_digest"],
            "closure_entry_count": identity["entry_count"],
        },
        "trust_package": {
            "profile_file_sha256": EXPECTED_PROFILE_FILE_SHA256,
            "verification_file_sha256": EXPECTED_VERIFICATION_FILE_SHA256,
        },
        "local_nix": {
            "executable": str(nix),
            "version": nix_version,
            "version_stdout_sha256": digest_bytes(bytes(version_result.stdout)),
            "version_stderr_sha256": digest_bytes(bytes(version_result.stderr)),
        },
        "commands": {
            "path_info": {
                "argv": path_info_cmd,
                "stdout_sha256": digest_bytes(bytes(path_info.stdout)),
                "stderr_sha256": digest_bytes(bytes(path_info.stderr)),
                "exit_code": path_info.returncode,
            },
            "store_verify": {
                "argv": verify_cmd,
                "stdout_sha256": digest_bytes(bytes(content_verify.stdout)),
                "stderr_sha256": digest_bytes(bytes(content_verify.stderr)),
                "exit_code": content_verify.returncode,
            },
        },
        "program": {
            "path": str(program),
            "pre_sha256": program_before,
            "post_sha256": program_after,
        },
        "authority": dict(AUTHORITY),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-admit the exact qualified Lineage-B Workbench closure on the current local Nix store without executing Workbench.")
    parser.parse_args(argv)
    try:
        result = verify_local_closure()
    except (ReadmissionError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
