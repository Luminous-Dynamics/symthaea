#!/usr/bin/env python3
"""Build a portable (never hosted-authoritative) execution receipt for #3491."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "symthaea.lqcd.portable-execution-receipt.v1"
DOMAIN = b"symthaea.lqcd.portable-execution-receipt.v1\0"
SEM_DOMAIN = b"symthaea.lqcd.execution-semantics.v1\0"
AUTHORITY = "PortableExecutionCandidate"
PROVIDER = "portable"
SEM_FIELDS = (
    "subject_sha", "subject_tree_sha", "base_sha", "base_tree_sha",
    "verifier_sha", "verifier_tree_sha", "qualification_profile_id",
    "recipe_semantics_sha256", "command_argv", "cargo_lock_sha256",
    "manifest_set_sha256", "rust_toolchain_sha256", "rustc_identity",
    "cargo_identity", "clippy_identity", "target_triple",
    "numerical_profile_id", "environment_equivalence_id",
)

def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()

def sha(domain, value):
    return sha256_bytes(domain + canonical(value))

def file_sha(path):
    return sha256_bytes(Path(path).read_bytes())

def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()

def clean():
    return subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], text=True
    ) == ""

def identity(argv):
    p = subprocess.run(argv, check=True, capture_output=True, text=True)
    return p.stdout.strip()

def manifest_set(paths):
    rows = []
    for raw in sorted(set(paths)):
        p = Path(raw)
        rows.append({"path": p.as_posix(), "sha256": file_sha(p)})
    return sha256_bytes(canonical(rows))

def main():
    ap = argparse.ArgumentParser()
    for name in ("subject-sha", "subject-tree-sha", "base-sha", "base-tree-sha",
                 "verifier-sha", "verifier-tree-sha", "profile-id",
                 "recipe-sha256", "numerical-profile-id",
                 "environment-equivalence-id", "provider-instance",
                 "environment-profile-sha256", "output"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--manifest", action="append", default=[])
    ap.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not command:
        raise SystemExit("missing command after --")

    if git("rev-parse", "HEAD") != a.subject_sha:
        raise SystemExit("subject SHA mismatch")
    if git("rev-parse", "HEAD^{tree}") != a.subject_tree_sha:
        raise SystemExit("subject tree mismatch")
    if not clean():
        raise SystemExit("working tree dirty before execution")

    cargo_lock = file_sha("Cargo.lock")
    toolchain = file_sha("rust-toolchain.toml")
    manifests = manifest_set(a.manifest or ["Cargo.toml"])
    rustc = identity(["rustc", "-Vv"])
    cargo = identity(["cargo", "-V"])
    clippy = identity(["cargo", "clippy", "-V"])
    target = ""
    for line in rustc.splitlines():
        if line.startswith("host: "):
            target = line[6:].strip()
            break
    if not target:
        raise SystemExit("rustc -Vv missing host target")

    started = time.time_ns()
    p = subprocess.run(command, capture_output=True)
    finished = time.time_ns()

    if not clean():
        raise SystemExit("working tree mutated by execution")
    if git("rev-parse", "HEAD") != a.subject_sha:
        raise SystemExit("subject SHA changed after execution")
    if git("rev-parse", "HEAD^{tree}") != a.subject_tree_sha:
        raise SystemExit("subject tree changed after execution")

    receipt = {
        "schema": SCHEMA,
        "authority_class": AUTHORITY,
        "provider_kind": PROVIDER,
        "provider_instance": a.provider_instance,
        "subject_sha": a.subject_sha,
        "subject_tree_sha": a.subject_tree_sha,
        "base_sha": a.base_sha,
        "base_tree_sha": a.base_tree_sha,
        "verifier_sha": a.verifier_sha,
        "verifier_tree_sha": a.verifier_tree_sha,
        "qualification_profile_id": a.profile_id,
        "recipe_semantics_sha256": a.recipe_sha256,
        "command_argv": command,
        "cargo_lock_sha256": cargo_lock,
        "manifest_set_sha256": manifests,
        "rust_toolchain_sha256": toolchain,
        "rustc_identity": rustc,
        "cargo_identity": cargo,
        "clippy_identity": clippy,
        "target_triple": target,
        "numerical_profile_id": a.numerical_profile_id,
        "environment_equivalence_id": a.environment_equivalence_id,
        "clean_before": True,
        "clean_after": True,
        "pre_tree_sha": a.subject_tree_sha,
        "post_tree_sha": a.subject_tree_sha,
        "dependency_resolution": "locked",
        "environment_profile_sha256": a.environment_profile_sha256,
        "os_kernel_arch": f"{platform.system()}/{platform.release()}/{platform.machine()}",
        "started_ns": started,
        "finished_ns": finished,
        "gates": [{
            "gate_id": "subject-command",
            "exit_status": p.returncode,
            "stdout_sha256": sha256_bytes(p.stdout),
            "stderr_sha256": sha256_bytes(p.stderr),
        }],
    }
    receipt["semantic_sha256"] = sha(
        SEM_DOMAIN, {k: receipt[k] for k in SEM_FIELDS}
    )
    receipt["receipt_sha256"] = sha(DOMAIN, receipt)
    Path(a.output).write_bytes(canonical(receipt) + b"\n")
    print(json.dumps({
        "authority_class": AUTHORITY,
        "exit_status": p.returncode,
        "receipt_sha256": receipt["receipt_sha256"],
        "semantic_sha256": receipt["semantic_sha256"],
    }, sort_keys=True, separators=(",", ":")))
    raise SystemExit(p.returncode)

if __name__ == "__main__":
    main()
