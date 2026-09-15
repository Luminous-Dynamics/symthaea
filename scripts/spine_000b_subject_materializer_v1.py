#!/usr/bin/env python3
"""SPINE-000B-M1A real subject materializer / drift verifier.

This tool prepares subject identity only. It must run from a clean checkout and writes
all outputs outside the repository so materialization itself cannot dirty the subject.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

ENCODER_PATH = Path("scripts/spine_000b_subject_manifest_v1.py")

CORE_BOUND_FILES = [
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    "flake.lock",
    "src/cognitive_loop/subsystem_trait.rs",
    "src/cognitive_loop/cycle_phase_dynamics/mod.rs",
    "src/cognitive_loop/cycle_phase_output/mod.rs",
    "src/cognitive_loop/helpers/feedback_helpers.rs",
    "docs/research/SPINE_000B_RUNTIME_SUBJECT_MANIFEST_V1.md",
    "scripts/spine_000b_subject_manifest_v1.py",
    "docs/research/SPINE_000B_MANAGER_ID_REGISTRY_V1.json",
    "scripts/verify_spine_000b_manager_id_registry_v1.py",
]


def fail(msg: str) -> None:
    raise ValueError(msg)


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def repo_root() -> Path:
    return Path(run("git", "rev-parse", "--show-toplevel")).resolve()


def load_encoder(root: Path):
    path = root / ENCODER_PATH
    spec = importlib.util.spec_from_file_location("spine_m1_encoder_runtime", path)
    if spec is None or spec.loader is None:
        fail("could not load M1 canonical encoder")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def status_porcelain(root: Path) -> str:
    return subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True
    ).strip()


def git_identity(root: Path) -> tuple[str, str]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=root, text=True).strip()
    return head, tree


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def observed_toolchain(root: Path) -> tuple[str, str, str]:
    rustc = subprocess.check_output(["rustc", "--version"], cwd=root, text=True).strip()
    cargo = subprocess.check_output(["cargo", "--version"], cwd=root, text=True).strip()
    vv = subprocess.check_output(["rustc", "-vV"], cwd=root, text=True)
    host = None
    for line in vv.splitlines():
        if line.startswith("host: "):
            host = line.removeprefix("host: ").strip()
            break
    if not host:
        fail("could not derive rustc host target")
    return rustc, cargo, host


def load_config(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "runtime_profile_id", "default_features", "features", "protocol_ids",
        "workload_id", "workload_digest", "seeds", "cycle_start", "cycle_count",
        "stopping_rule_id", "capacities", "qualification_policy_digest",
        "additional_bound_files",
    }
    if set(data) != required:
        fail(f"config fields drifted missing={sorted(required-set(data))} extra={sorted(set(data)-required)}")
    return data


def validate_features(root: Path, features: list[str]) -> None:
    cargo = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    known = set(cargo.get("features", {}).keys())
    if len(features) != len(set(features)):
        fail("duplicate configured feature")
    for feature in features:
        if not isinstance(feature, str) or not feature or feature == "default":
            fail(f"invalid explicit feature {feature!r}")
        if feature not in known:
            fail(f"unknown Cargo feature: {feature}")


def canonical_bound_paths(root: Path, encoder, config: dict) -> list[str]:
    extras = config["additional_bound_files"]
    if not isinstance(extras, list):
        fail("additional_bound_files must be list")
    paths = CORE_BOUND_FILES + extras
    if len(paths) != len(set(paths)):
        fail("duplicate bound path")
    normalized = [encoder.canonical_path(p) for p in paths]
    for rel in normalized:
        path = root / rel
        if not path.is_file():
            fail(f"required bound file missing: {rel}")
    return sorted(normalized)


def ensure_output_outside_repo(root: Path, out_dir: Path) -> Path:
    resolved = out_dir.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        pass
    else:
        fail("M1A output directory must be outside the repository")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def build_manifest(root: Path, encoder, config: dict, bound_paths: list[str]) -> dict:
    if status_porcelain(root):
        fail("qualified subject worktree is dirty")
    head, tree = git_identity(root)
    rustc, cargo, target = observed_toolchain(root)
    validate_features(root, config["features"])
    bound_files = [{"path": rel, "sha256": sha256_file(root / rel)} for rel in bound_paths]
    manifest = {
        "repository": "Luminous-Dynamics/symthaea",
        "git_head": head,
        "git_tree": tree,
        "clean_worktree_required": True,
        "runtime_profile_id": config["runtime_profile_id"],
        "target_triple": target,
        "rustc_version": rustc,
        "cargo_version": cargo,
        "default_features": config["default_features"],
        "features": config["features"],
        "bound_files": bound_files,
        "protocol_ids": config["protocol_ids"],
        "workload_id": config["workload_id"],
        "workload_digest": config["workload_digest"],
        "seeds": config["seeds"],
        "cycle_start": config["cycle_start"],
        "cycle_count": config["cycle_count"],
        "stopping_rule_id": config["stopping_rule_id"],
        "capacities": config["capacities"],
        "qualification_policy_digest": config["qualification_policy_digest"],
    }
    encoder.validate_manifest(manifest)
    return manifest


def verify_checkout_against_manifest(root: Path, encoder, manifest: dict) -> None:
    encoder.validate_manifest(manifest)
    if status_porcelain(root):
        fail("subject worktree drifted/dirty")
    head, tree = git_identity(root)
    if head != manifest["git_head"] or tree != manifest["git_tree"]:
        fail("Git HEAD/tree drifted from subject manifest")
    rustc, cargo, target = observed_toolchain(root)
    if rustc != manifest["rustc_version"] or cargo != manifest["cargo_version"] or target != manifest["target_triple"]:
        fail("observed toolchain/target drifted from subject manifest")
    validate_features(root, manifest["features"])
    for entry in manifest["bound_files"]:
        rel = encoder.canonical_path(entry["path"])
        path = root / rel
        if not path.is_file():
            fail(f"bound file disappeared: {rel}")
        if sha256_file(path) != entry["sha256"]:
            fail(f"bound file drifted: {rel}")


def materialize(config_path: Path, out_dir: Path) -> int:
    root = repo_root()
    encoder = load_encoder(root)
    config = load_config(config_path)
    out = ensure_output_outside_repo(root, out_dir)
    paths = canonical_bound_paths(root, encoder, config)

    pre_head, pre_tree = git_identity(root)
    pre_status = status_porcelain(root)
    if pre_status:
        fail("qualified subject worktree is dirty before materialization")

    manifest = build_manifest(root, encoder, config, paths)
    canonical = encoder.canonical_bytes(manifest)
    digest = hashlib.sha256(canonical).hexdigest()

    # Re-read every bound source and all checkout identity after encoding.
    verify_checkout_against_manifest(root, encoder, manifest)
    post_head, post_tree = git_identity(root)
    post_status = status_porcelain(root)
    if (pre_head, pre_tree, pre_status) != (post_head, post_tree, post_status):
        fail("checkout identity changed during subject materialization")

    (out / "subject-manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (out / "subject-manifest.bin").write_bytes(canonical)
    (out / "subject-manifest.sha256").write_text(digest + "\n", encoding="ascii")
    receipt = {
        "schema": "symthaea.spine.000b.subject-materialization-receipt.v1",
        "authority": "measurement-only",
        "subject_manifest_sha256": digest,
        "git_head": post_head,
        "git_tree": post_tree,
        "bound_file_count": len(paths),
        "runtime_evidence_claimed": False,
        "causal_load_claimed": False,
    }
    (out / "materialization-receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(f"subject_manifest_sha256={digest}")
    print(f"bound_file_count={len(paths)}")
    print("authority=measurement-only")
    return 0


def verify(manifest_path: Path, bin_path: Path | None, digest_path: Path | None) -> int:
    root = repo_root()
    encoder = load_encoder(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_checkout_against_manifest(root, encoder, manifest)
    canonical = encoder.canonical_bytes(manifest)
    digest = hashlib.sha256(canonical).hexdigest()
    if bin_path is not None and bin_path.read_bytes() != canonical:
        fail("materialized canonical binary does not match manifest semantics")
    if digest_path is not None and digest_path.read_text(encoding="ascii").strip() != digest:
        fail("materialized manifest digest does not match canonical bytes")
    print("SPINE-000B-M1A drift verification: PASS")
    print(f"subject_manifest_sha256={digest}")
    print("authority=measurement-only")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("materialize")
    m.add_argument("--config", required=True, type=Path)
    m.add_argument("--out-dir", required=True, type=Path)
    v = sub.add_parser("verify")
    v.add_argument("--manifest", required=True, type=Path)
    v.add_argument("--bin", dest="bin_path", type=Path)
    v.add_argument("--digest", dest="digest_path", type=Path)
    args = parser.parse_args()
    if args.cmd == "materialize":
        return materialize(args.config, args.out_dir)
    return verify(args.manifest, args.bin_path, args.digest_path)


if __name__ == "__main__":
    raise SystemExit(main())
