#!/usr/bin/env python3
"""Fail-closed exact-tree integration preflight for WCARE-46.

MeasurementOnly. Proves lineage/tree integration facts only; it never promotes
child-verifier execution or downstream authentication claims.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

PROTOCOL = "wcare46-exact-child-tree-integration-v1"
INTEGRATION = "7e7aaf0314ea2c15c663351f463afe589c35cf4f"
W45 = "a53595ad9ef7ebd16eac8e047f70666323083f86"
W42 = "4bb84790bab3dc6a6d2e30d0ef081950a3954717"
W43 = "9e4bfd621e3c48ba6335f95b3dd1fd7b36dccf25"

W42_BLOBS = {
    "crates/domains/symthaea-wisdom/tests/wcare42_builder_attestation_prelock_integrity.rs": "bb509be90791f738b387b7b036069033c493c018",
    "docs/release/evidence/WCARE42_BUILDER_ATTESTATION_GOLDEN_VECTOR_V1.json": "dc42f404537795f37a9bf178d3f30fd52c74a355",
    "docs/release/evidence/WCARE42_BUILDER_ATTESTATION_RESULT_SCHEMA_V1.json": "b2d6b4b61af46c8924cdb95b2f958df3d3d7ab96",
    "docs/release/evidence/WCARE42_BUILDER_ATTESTATION_VERIFIER_PROTOCOL_V1.md": "1cd6e3f5f2f4edcf4fccf45b35c4c68bdb9c12a5",
    "scripts/wcare42-integrity.sh": "ecb602b818619c4e091b4878943d5b760931f327",
    "scripts/wcare42-qualify.sh": "8a80d1a74e409503a72b4607425cc61d8072ca2e",
    "tools/wcare42_builder_attestation_verifier/Cargo.toml": "5410040e5616241dd4ba581af8f297675d083830",
    "tools/wcare42_builder_attestation_verifier/src/main.rs": "1c300a455f054d118e55556aac81b623824629bc",
    "tools/wcare42_builder_attestation_verifier/tests/golden.rs": "666242f74fb302f9be2b62fa4b3050f3ee9ffefd",
}

W43_BLOBS = {
    "crates/domains/symthaea-wisdom/tests/wcare43_rfc3161_preregistration_integrity.rs": "0cb86c600aacef208a724b820ffc06001ce7f770",
    "docs/release/evidence/WCARE43_RFC3161_BACKEND_POLICY_SCHEMA_V1.json": "2e04e221a5d89bd2da3ea4244649898b66e47a74",
    "docs/release/evidence/WCARE43_RFC3161_PREREGISTRATION_PROTOCOL_V1.md": "76ff1774ee510c096649b82cfc3857deaf921e12",
    "docs/release/evidence/WCARE43_RFC3161_RESULT_SCHEMA_V1.json": "f504e2fed24a0637621f7ab757e580f33b321a98",
    "docs/release/evidence/fixtures/wcare43/a.final.json": "8cec0486532180e6f4fe0f58920c1f318b5154ac",
    "docs/release/evidence/fixtures/wcare43/b.final.json": "ba7cce8fbfece6812fa44816942ed66d3ed7e55f",
    "docs/release/evidence/fixtures/wcare43/fixture_manifest.json": "ec601cc591365dd6f9026894c477ae5a7180c97a",
    "docs/release/evidence/fixtures/wcare43/synthetic_plan.json": "a6569381df8e320c286833a9830075376482206a",
    "docs/release/evidence/fixtures/wcare43/synthetic_response.tsr.b64": "0add55553bbe1927635b747b184523b0e87e576f",
    "docs/release/evidence/fixtures/wcare43/synthetic_root.pem": "901a385c49aa56a5606324319a38956cba414fc1",
    "docs/release/evidence/fixtures/wcare43/synthetic_tsa.pem": "d4ca7537a5baebe978e8cc3744763716a5bb800e",
    "docs/release/evidence/fixtures/wcare43/synthetic_wcare40_result.json": "8a48ce8597167d4de32b67806217e6a0525267cd",
    "scripts/wcare43-integrity.sh": "811c8c1bfa7686858a965dc617cd4051205a7b4b",
    "scripts/wcare43_rfc3161_verify.py": "e89b66a24cb51bed589911b082ea6fef5ffd369b",
    "scripts/wcare43_selftest.py": "7906ce87f7b121464b42dbf35584ddbc9a50ae4d",
}
ALL_BLOBS = W42_BLOBS | W43_BLOBS
LOCK_PATH = "tools/wcare42_builder_attestation_verifier/Cargo.lock"


def run(root: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def base(head: str) -> dict:
    return {
        "authority": "MeasurementOnly",
        "protocol_version": PROTOCOL,
        "stage": "IntegrationPreflight",
        "classification": "INVALID_PROTOCOL",
        "detail": "uninitialized",
        "head": head,
        "integration_commit": INTEGRATION,
        "wcare45_parent": W45,
        "wcare42_head": W42,
        "wcare43_head": W43,
        "integration_parent_verified": False,
        "lineage_convergence_verified": False,
        "integration_diff_exact": False,
        "imported_path_count": len(ALL_BLOBS),
        "wcare42_imported_path_count": len(W42_BLOBS),
        "wcare43_imported_path_count": len(W43_BLOBS),
        "wcare42_exact_tree_present": False,
        "wcare43_exact_tree_present": False,
        "wcare42_standalone_lock_present": False,
        "wcare42_executable_qualification_established": False,
        "wcare43_external_execution_lineage_established": False,
        "child_verifier_execution_established": False,
        "builder_authentication_established": False,
        "preregistration_temporal_precedence_established": False,
        "authenticated_preregistered_replication_established": False,
        "runtime_authority_granted": False,
    }


def emit(out: dict, code: int) -> int:
    print(json.dumps(out, sort_keys=True, separators=(",", ":")))
    return code


def blob_at(root: Path, revision: str, path: str) -> str | None:
    result = run(root, "git", "rev-parse", f"{revision}:{path}")
    return result.stdout.strip() if result.returncode == 0 else None


def exact_blob_census(root: Path, revision: str, expected: dict[str, str]) -> bool:
    return all(blob_at(root, revision, path) == sha for path, sha in expected.items())


def main() -> int:
    root_result = subprocess.run(["git", "rev-parse", "--show-toplevel"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if root_result.returncode != 0:
        return emit(base("0" * 40) | {"detail": "not_in_git_worktree"}, 4)
    root = Path(root_result.stdout.strip())
    head_result = run(root, "git", "rev-parse", "HEAD")
    head = head_result.stdout.strip() if head_result.returncode == 0 else "0" * 40
    out = base(head)

    parents = run(root, "git", "show", "-s", "--format=%P", INTEGRATION)
    if parents.returncode != 0 or parents.stdout.strip().split() != [W45]:
        out["detail"] = "integration_parent_mismatch"
        return emit(out, 4)
    out["integration_parent_verified"] = True

    for ancestor in [INTEGRATION, W45, W42, W43]:
        check = run(root, "git", "merge-base", "--is-ancestor", ancestor, "HEAD")
        if check.returncode != 0:
            out["detail"] = f"required_ancestor_missing:{ancestor}"
            return emit(out, 4)
    out["lineage_convergence_verified"] = True

    diff = run(root, "git", "diff", "--name-status", W45, INTEGRATION)
    if diff.returncode != 0:
        out["detail"] = "integration_diff_unreadable"
        return emit(out, 4)
    observed: dict[str, str] = {}
    for line in diff.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            out["detail"] = "integration_diff_noncanonical"
            return emit(out, 4)
        status, path = parts
        observed[path] = status
    if set(observed) != set(ALL_BLOBS) or any(status != "A" for status in observed.values()):
        out["detail"] = "integration_diff_not_exact_24_additions"
        return emit(out, 4)

    if not exact_blob_census(root, INTEGRATION, ALL_BLOBS):
        out["detail"] = "integration_commit_blob_census_mismatch"
        return emit(out, 4)
    out["integration_diff_exact"] = True

    out["wcare42_exact_tree_present"] = exact_blob_census(root, "HEAD", W42_BLOBS)
    out["wcare43_exact_tree_present"] = exact_blob_census(root, "HEAD", W43_BLOBS)
    if not out["wcare42_exact_tree_present"]:
        out["detail"] = "wcare42_exact_tree_drift"
        return emit(out, 4)
    if not out["wcare43_exact_tree_present"]:
        out["detail"] = "wcare43_exact_tree_drift"
        return emit(out, 4)

    lock = run(root, "git", "cat-file", "-e", f"HEAD:{LOCK_PATH}")
    out["wcare42_standalone_lock_present"] = lock.returncode == 0

    blockers = []
    if not out["wcare42_standalone_lock_present"]:
        blockers.append("wcare42_standalone_lock_missing")
    blockers.append("child_verifiers_not_reexecuted_by_wcare46_preflight")

    out["classification"] = "CHILD_EXECUTION_INDETERMINATE"
    out["detail"] = ";".join(blockers)
    return emit(out, 3)


if __name__ == "__main__":
    raise SystemExit(main())