#!/usr/bin/env python3
"""Independent stdlib oracle for #3491 portable execution-receipt semantics."""

import copy
import hashlib
import json
import re

SCHEMA = "symthaea.lqcd.portable-execution-receipt.v1"
DOMAIN = b"symthaea.lqcd.portable-execution-receipt.v1\0"
SEM_DOMAIN = b"symthaea.lqcd.execution-semantics.v1\0"
PORTABLE = "PortableExecutionCandidate"
HOSTED = "HostedExactHeadQualified"
H40 = re.compile(r"^[0-9a-f]{40}$")
H64 = re.compile(r"^[0-9a-f]{64}$")

SEM_FIELDS = (
    "subject_sha", "subject_tree_sha", "base_sha", "base_tree_sha",
    "verifier_sha", "verifier_tree_sha", "qualification_profile_id",
    "recipe_semantics_sha256", "command_argv", "cargo_lock_sha256",
    "manifest_set_sha256", "rust_toolchain_sha256", "rustc_identity",
    "cargo_identity", "clippy_identity", "target_triple",
    "numerical_profile_id", "environment_equivalence_id",
)

SHIFT = (
    ("subject_sha", "SubjectShift"), ("subject_tree_sha", "SubjectShift"),
    ("base_sha", "SubjectShift"), ("base_tree_sha", "SubjectShift"),
    ("verifier_sha", "RecipeShift"), ("verifier_tree_sha", "RecipeShift"),
    ("qualification_profile_id", "RecipeShift"),
    ("recipe_semantics_sha256", "RecipeShift"), ("command_argv", "RecipeShift"),
    ("cargo_lock_sha256", "DependencyShift"),
    ("manifest_set_sha256", "DependencyShift"),
    ("rust_toolchain_sha256", "ToolchainShift"),
    ("rustc_identity", "ToolchainShift"), ("cargo_identity", "ToolchainShift"),
    ("clippy_identity", "ToolchainShift"), ("target_triple", "TargetShift"),
    ("numerical_profile_id", "NumericalProfileShift"),
    ("environment_equivalence_id", "EnvironmentEquivalenceUnknown"),
)

def canon(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def sha(domain, x):
    return hashlib.sha256(domain + canon(x)).hexdigest()

def h(label):
    return hashlib.sha256(label.encode()).hexdigest()

def project(r):
    return {k: copy.deepcopy(r[k]) for k in SEM_FIELDS}

def semantic_sha(r):
    return sha(SEM_DOMAIN, project(r))

def receipt_sha(r):
    x = copy.deepcopy(r)
    x.pop("receipt_sha256", None)
    x.pop("semantic_sha256", None)
    return sha(DOMAIN, x)

def validate(r):
    required = set(SEM_FIELDS) | {
        "schema", "authority_class", "provider_kind", "provider_instance",
        "clean_before", "clean_after", "pre_tree_sha", "post_tree_sha",
        "dependency_resolution", "environment_profile_sha256", "gates",
        "receipt_sha256", "semantic_sha256",
    }
    missing = sorted(required - set(r))
    if missing:
        raise ValueError(f"missing:{missing}")
    if r["schema"] != SCHEMA:
        raise ValueError("schema")
    if r["provider_kind"] not in {"portable", "github-hosted"}:
        raise ValueError("provider_kind")
    if r["provider_kind"] == "portable" and r["authority_class"] == HOSTED:
        raise ValueError("portable_self_promotion")
    if r["authority_class"] == HOSTED and r["provider_kind"] != "github-hosted":
        raise ValueError("hosted_provider_required")
    for k in ("subject_sha", "subject_tree_sha", "base_sha", "base_tree_sha",
              "verifier_sha", "verifier_tree_sha", "pre_tree_sha", "post_tree_sha"):
        if not isinstance(r[k], str) or not H40.fullmatch(r[k]):
            raise ValueError(f"hex40:{k}")
    for k in ("recipe_semantics_sha256", "cargo_lock_sha256",
              "manifest_set_sha256", "rust_toolchain_sha256",
              "environment_profile_sha256"):
        if not isinstance(r[k], str) or not H64.fullmatch(r[k]):
            raise ValueError(f"hex64:{k}")
    if r["clean_before"] is not True:
        raise ValueError("dirty_before")
    if r["clean_after"] is not True:
        raise ValueError("dirty_after")
    if r["pre_tree_sha"] != r["subject_tree_sha"]:
        raise ValueError("pre_tree_mismatch")
    if r["post_tree_sha"] != r["subject_tree_sha"]:
        raise ValueError("post_tree_mismatch")
    if r["dependency_resolution"] != "locked":
        raise ValueError("unlocked")
    for k in ("qualification_profile_id", "rustc_identity", "cargo_identity",
              "clippy_identity", "target_triple", "numerical_profile_id",
              "environment_equivalence_id", "provider_instance"):
        if not isinstance(r[k], str) or not r[k]:
            raise ValueError(f"empty:{k}")
    if not isinstance(r["command_argv"], list) or not r["command_argv"]:
        raise ValueError("command")
    if not isinstance(r["gates"], list) or not r["gates"]:
        raise ValueError("gates")
    seen = set()
    for g in r["gates"]:
        if g.get("gate_id") in seen:
            raise ValueError("duplicate_gate")
        seen.add(g.get("gate_id"))
        if not isinstance(g.get("exit_status"), int) or isinstance(g.get("exit_status"), bool):
            raise ValueError("gate_status")
        if not 0 <= g["exit_status"] <= 255:
            raise ValueError("gate_status")
    if r["semantic_sha256"] != semantic_sha(r):
        raise ValueError("semantic_digest")
    if r["receipt_sha256"] != receipt_sha(r):
        raise ValueError("receipt_digest")
    return r

def seal(r):
    x = copy.deepcopy(r)
    x["semantic_sha256"] = semantic_sha(x)
    x["receipt_sha256"] = receipt_sha(x)
    return validate(x)

def seal_unchecked(r):
    x = copy.deepcopy(r)
    x["semantic_sha256"] = semantic_sha(x)
    x["receipt_sha256"] = receipt_sha(x)
    return x

def classify(a, b):
    validate(a); validate(b)
    if a["semantic_sha256"] == b["semantic_sha256"]:
        return "ExecutionSemanticsEquivalent"
    for k, label in SHIFT:
        if a[k] != b[k]:
            return label
    raise AssertionError("unclassified semantic change")

def rust_qualified(r):
    validate(r)
    return (r["authority_class"] == HOSTED and
            r["provider_kind"] == "github-hosted" and
            all(g["exit_status"] == 0 for g in r["gates"]))

def fixture():
    return {
        "schema": SCHEMA, "authority_class": PORTABLE, "provider_kind": "portable",
        "provider_instance": "nix-local-a",
        "subject_sha": "4bad8af72ff775e7c869b6df83faba718a339a36",
        "subject_tree_sha": "4884e38d1436427e406f8ae8f3cdba326765a36d",
        "base_sha": "a81955b838de5fcc46c7b277dfd1fe11bcafb87f",
        "base_tree_sha": "1111111111111111111111111111111111111111",
        "verifier_sha": "1af61ddee9131800a3dda4f89a307db552062f7f",
        "verifier_tree_sha": "2222222222222222222222222222222222222222",
        "qualification_profile_id": "lqcd-particle-physics-fast-v1",
        "recipe_semantics_sha256": h("recipe-v1"),
        "command_argv": ["cargo","clippy","-p","symthaea-particle-physics",
                         "--locked","--all-targets","--","-D","warnings"],
        "cargo_lock_sha256": h("lock"), "manifest_set_sha256": h("manifests"),
        "rust_toolchain_sha256": h("toolchain"),
        "rustc_identity": "rustc-1.96.0-fixture",
        "cargo_identity": "cargo-1.96.0-fixture",
        "clippy_identity": "clippy-1.96.0-fixture",
        "target_triple": "x86_64-unknown-linux-gnu",
        "numerical_profile_id": "non-numerical-clippy-v1",
        "environment_equivalence_id": "nix-lqcd-fast-env-v1",
        "clean_before": True, "clean_after": True,
        "pre_tree_sha": "4884e38d1436427e406f8ae8f3cdba326765a36d",
        "post_tree_sha": "4884e38d1436427e406f8ae8f3cdba326765a36d",
        "dependency_resolution": "locked",
        "environment_profile_sha256": h("env"),
        "gates": [{"gate_id": "clippy", "exit_status": 1}],
    }

def rejected(r, needle):
    try:
        validate(r)
    except ValueError as e:
        assert needle in str(e), (needle, str(e))
        return
    raise AssertionError(f"expected rejection:{needle}")

def main():
    p = seal(fixture())
    assert not rust_qualified(p)

    same = copy.deepcopy(p)
    same["provider_instance"] = "nix-local-b"
    same = seal(same)
    assert p["receipt_sha256"] != same["receipt_sha256"]
    assert p["semantic_sha256"] == same["semantic_sha256"]
    assert classify(p, same) == "ExecutionSemanticsEquivalent"

    shifts = {}
    cases = {
        "rustc_identity": ("-changed", "ToolchainShift"),
        "recipe_semantics_sha256": (h("recipe-v2"), "RecipeShift"),
        "cargo_lock_sha256": (h("lock-v2"), "DependencyShift"),
        "target_triple": ("aarch64-unknown-linux-gnu", "TargetShift"),
        "numerical_profile_id": ("profile-v2", "NumericalProfileShift"),
        "subject_sha": ("3333333333333333333333333333333333333333", "SubjectShift"),
        "environment_equivalence_id": ("unknown-v2", "EnvironmentEquivalenceUnknown"),
    }
    for k, (v, expected) in cases.items():
        q = copy.deepcopy(p)
        q[k] = q[k] + v if k == "rustc_identity" else v
        q = seal(q)
        got = classify(p, q)
        assert got == expected
        shifts[k] = got

    host = copy.deepcopy(p)
    host["authority_class"] = HOSTED
    host["provider_kind"] = "github-hosted"
    host["provider_instance"] = "synthetic-host"
    host["gates"][0]["exit_status"] = 0
    host = seal(host)
    assert rust_qualified(host)

    negatives = []
    for label, mutate, needle in [
        ("dirty", lambda x: x.update(clean_before=False), "dirty_before"),
        ("post-mutation", lambda x: x.update(post_tree_sha="4"*40), "post_tree_mismatch"),
        ("unlocked", lambda x: x.update(dependency_resolution="unlocked"), "unlocked"),
        ("missing-rustc", lambda x: x.update(rustc_identity=""), "empty:rustc_identity"),
        ("self-promotion", lambda x: x.update(authority_class=HOSTED), "portable_self_promotion"),
        ("pre-mismatch", lambda x: x.update(pre_tree_sha="5"*40), "pre_tree_mismatch"),
        ("bad-status", lambda x: x["gates"][0].update(exit_status=999), "gate_status"),
        ("duplicate-gate", lambda x: x["gates"].append(copy.deepcopy(x["gates"][0])), "duplicate_gate"),
    ]:
        q = copy.deepcopy(p); mutate(q); q = seal_unchecked(q)
        rejected(q, needle); negatives.append(label)

    print(json.dumps({
        "schema": SCHEMA,
        "portable_receipt_sha256": p["receipt_sha256"],
        "portable_semantic_sha256": p["semantic_sha256"],
        "same_semantics_receipt_sha256": same["receipt_sha256"],
        "same_semantics_classification": classify(p, same),
        "shift_classifications": shifts,
        "negative_controls": negatives,
        "portable_can_satisfy_rust_qualified": rust_qualified(p),
        "synthetic_hosted_can_satisfy_rust_qualified": rust_qualified(host),
        "real_rust_execution_performed": False,
        "real_beta6_campaign_authorized": False,
    }, sort_keys=True, separators=(",", ":")))

if __name__ == "__main__":
    main()
