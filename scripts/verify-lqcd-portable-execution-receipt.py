#!/usr/bin/env python3
"""Independent verifier for #3491 portable execution receipts."""

import argparse
import hashlib
import json
import re
from pathlib import Path

SCHEMA = "symthaea.lqcd.portable-execution-receipt.v1"
DOMAIN = b"symthaea.lqcd.portable-execution-receipt.v1\0"
SEM_DOMAIN = b"symthaea.lqcd.execution-semantics.v1\0"
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

def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def sha(domain, x):
    return hashlib.sha256(domain + canonical(x)).hexdigest()

def fail(msg):
    raise SystemExit("INVALID:" + msg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("receipt")
    a = ap.parse_args()
    try:
        r = json.loads(Path(a.receipt).read_text())
    except Exception as e:
        fail("json:" + type(e).__name__)

    required = set(SEM_FIELDS) | {
        "schema", "authority_class", "provider_kind", "provider_instance",
        "clean_before", "clean_after", "pre_tree_sha", "post_tree_sha",
        "dependency_resolution", "environment_profile_sha256",
        "os_kernel_arch", "started_ns", "finished_ns", "gates",
        "semantic_sha256", "receipt_sha256",
    }
    miss = sorted(required - set(r))
    if miss: fail("missing:" + ",".join(miss))
    if r["schema"] != SCHEMA: fail("schema")
    if r["authority_class"] != "PortableExecutionCandidate": fail("authority")
    if r["provider_kind"] != "portable": fail("provider")
    for k in ("subject_sha","subject_tree_sha","base_sha","base_tree_sha",
              "verifier_sha","verifier_tree_sha","pre_tree_sha","post_tree_sha"):
        if not isinstance(r[k], str) or not H40.fullmatch(r[k]): fail("hex40:"+k)
    for k in ("recipe_semantics_sha256","cargo_lock_sha256","manifest_set_sha256",
              "rust_toolchain_sha256","environment_profile_sha256",
              "semantic_sha256","receipt_sha256"):
        if not isinstance(r[k], str) or not H64.fullmatch(r[k]): fail("hex64:"+k)
    if r["clean_before"] is not True or r["clean_after"] is not True: fail("clean")
    if r["pre_tree_sha"] != r["subject_tree_sha"]: fail("pre-tree")
    if r["post_tree_sha"] != r["subject_tree_sha"]: fail("post-tree")
    if r["dependency_resolution"] != "locked": fail("resolution")
    if not isinstance(r["command_argv"], list) or not r["command_argv"]: fail("command")
    if not isinstance(r["gates"], list) or len(r["gates"]) != 1: fail("gates")
    g = r["gates"][0]
    if g.get("gate_id") != "subject-command": fail("gate-id")
    if not isinstance(g.get("exit_status"), int) or isinstance(g.get("exit_status"), bool):
        fail("exit-status")
    if not 0 <= g["exit_status"] <= 255: fail("exit-status")
    for k in ("stdout_sha256","stderr_sha256"):
        if not isinstance(g.get(k), str) or not H64.fullmatch(g[k]): fail("gate-"+k)
    sem = sha(SEM_DOMAIN, {k: r[k] for k in SEM_FIELDS})
    if r["semantic_sha256"] != sem: fail("semantic-digest")
    x = dict(r)
    claimed = x.pop("receipt_sha256")
    if claimed != sha(DOMAIN, x): fail("receipt-digest")
    print(json.dumps({
        "authority_class": r["authority_class"],
        "portable_can_satisfy_rust_qualified": False,
        "receipt_sha256": claimed,
        "semantic_sha256": sem,
        "subject_exit_status": g["exit_status"],
        "valid": True,
    }, sort_keys=True, separators=(",", ":")))

if __name__ == "__main__":
    main()
