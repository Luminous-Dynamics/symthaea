#!/usr/bin/env python3
"""Qualification verifier for SPINE-000B-M1 subject-manifest v1."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

ENCODER = Path("scripts/spine_000b_subject_manifest_v1.py")
VECTORS = Path("tests/fixtures/spine_000b_subject_manifest_v1_vectors.json")
DOC = Path("docs/research/SPINE_000B_RUNTIME_SUBJECT_MANIFEST_V1.md")
RUST_TOOLCHAIN = Path("rust-toolchain.toml")
FLAKE_LOCK = Path("flake.lock")


def fail(msg: str) -> None:
    raise ValueError(msg)


def load_encoder():
    spec = importlib.util.spec_from_file_location("spine_m1_encoder", ENCODER)
    if spec is None or spec.loader is None:
        fail("could not load M1 encoder")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def must_reject(fn, label: str) -> None:
    try:
        fn()
    except (ValueError, TypeError, UnicodeError):
        return
    fail(f"negative control did not reject: {label}")


def main() -> int:
    m1 = load_encoder()
    frozen = json.loads(VECTORS.read_text(encoding="utf-8"))
    generated = m1.sample_vectors()
    if generated != frozen:
        fail("checked M1 vectors drifted from reference encoder")

    if frozen["base"] != frozen["reordered"]:
        fail("canonical list ordering is not stable")
    if frozen["base"]["sha256"] == frozen["mutated_capacity"]["sha256"]:
        fail("observer capacity mutation did not change digest")
    if frozen["base"]["sha256"] == frozen["mutated_hash"]["sha256"]:
        fail("bound source hash mutation did not change digest")

    base = m1.sample_manifest()

    dup_feature = copy.deepcopy(base)
    dup_feature["features"].append(dup_feature["features"][0])
    must_reject(lambda: m1.canonical_bytes(dup_feature), "duplicate feature")

    dup_protocol = copy.deepcopy(base)
    dup_protocol["protocol_ids"].append(dup_protocol["protocol_ids"][0])
    must_reject(lambda: m1.canonical_bytes(dup_protocol), "duplicate protocol")

    dup_seed = copy.deepcopy(base)
    dup_seed["seeds"].append(copy.deepcopy(dup_seed["seeds"][0]))
    must_reject(lambda: m1.canonical_bytes(dup_seed), "duplicate seed")

    dup_path = copy.deepcopy(base)
    dup_path["bound_files"].append(copy.deepcopy(dup_path["bound_files"][0]))
    must_reject(lambda: m1.canonical_bytes(dup_path), "duplicate bound path")

    for bad in ("/tmp/Cargo.lock", "../Cargo.lock", "src//x.rs", "src\\x.rs", "./Cargo.lock"):
        mutant = copy.deepcopy(base)
        mutant["bound_files"][0]["path"] = bad
        must_reject(lambda mutant=mutant: m1.canonical_bytes(mutant), f"noncanonical path {bad}")

    dirty = copy.deepcopy(base)
    dirty["clean_worktree_required"] = False
    must_reject(lambda: m1.canonical_bytes(dirty), "dirty subject policy")

    zero_cycles = copy.deepcopy(base)
    zero_cycles["cycle_count"] = 0
    must_reject(lambda: m1.canonical_bytes(zero_cycles), "zero cycle count")

    zero_capacity = copy.deepcopy(base)
    zero_capacity["capacities"]["guard"] = 0
    must_reject(lambda: m1.canonical_bytes(zero_capacity), "zero guard capacity")

    # Mutations that must change canonical identity.
    mutations = []
    x = copy.deepcopy(base); x["rustc_version"] += " changed"; mutations.append(("rustc", x))
    x = copy.deepcopy(base); x["cargo_version"] += " changed"; mutations.append(("cargo", x))
    x = copy.deepcopy(base); x["features"].append("identity"); mutations.append(("feature", x))
    x = copy.deepcopy(base); x["workload_digest"] = "99" * 32; mutations.append(("workload", x))
    x = copy.deepcopy(base); x["seeds"][0]["value"] += 1; mutations.append(("seed", x))
    x = copy.deepcopy(base); x["cycle_count"] += 1; mutations.append(("cycle_count", x))
    x = copy.deepcopy(base); x["stopping_rule_id"] += "-changed"; mutations.append(("stopping_rule", x))
    x = copy.deepcopy(base); x["default_features"] = True; mutations.append(("default_features", x))
    x = copy.deepcopy(base); x["git_tree"] = "aa" * 20; mutations.append(("git_tree", x))
    for label, mutant in mutations:
        if m1.digest_hex(mutant) == m1.digest_hex(base):
            fail(f"identity mutation did not change digest: {label}")

    toolchain = RUST_TOOLCHAIN.read_text(encoding="utf-8")
    if 'channel = "1.96.0"' not in toolchain:
        fail("pinned Rust 1.96.0 declaration drifted")
    if not FLAKE_LOCK.is_file():
        fail("flake.lock missing")
    doc = DOC.read_text(encoding="utf-8")
    for token in (
        "measurement-only", "symthaea.spine.000b.subject-manifest.v1",
        "clean worktree", "observer capacity", "Python", "Rust",
    ):
        if token not in doc:
            fail(f"M1 contract missing boundary token: {token}")

    print("SPINE-000B-M1 subject manifest verifier: PASS")
    print(f"base_sha256={frozen['base']['sha256']}")
    print("canonical_reordering=true")
    print("authority=measurement-only")
    print("runtime_evidence_claimed=false")
    print("causal_load_claimed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
