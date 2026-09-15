#!/usr/bin/env python3
"""Independent comparator for portable v1 vs hosted focused-positive v2 receipts.

The comparator verifies the portable self-digests and an externally supplied
SHA-256 for the hosted positive-receipt file. It may establish agreement on the
authority spine shared by both schemas. It MUST NOT emit
CrossProviderCorroborated because hosted v2 lacks normalized extended bindings
required by #3491.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

PORTABLE_SCHEMA = "symthaea.lqcd.portable-execution-receipt.v1"
HOSTED_SCHEMA = "symthaea.focused-positive-receipt.v2"
PORTABLE_DOMAIN = b"symthaea.lqcd.portable-execution-receipt.v1\0"
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

def result(classification, *, valid=True, core=False, missing=None):
    out = {
        "valid": valid,
        "classification": classification,
        "core_equivalent": core,
        "cross_provider_corroborated": False,
    }
    if missing is not None:
        out["missing_hosted_bindings"] = missing
    return out

def nonempty(v):
    return isinstance(v, str) and bool(v)

def validate_portable(r):
    if r.get("schema") != PORTABLE_SCHEMA:
        return "PortableSchemaInvalid"
    if r.get("authority_class") != "PortableExecutionCandidate":
        return "PortableAuthorityInvalid"
    if r.get("provider_kind") != "portable":
        return "PortableProviderInvalid"
    for k in ("subject_sha", "subject_tree_sha", "base_sha", "base_tree_sha",
              "verifier_sha", "verifier_tree_sha", "pre_tree_sha", "post_tree_sha"):
        if not isinstance(r.get(k), str) or not H40.fullmatch(r[k]):
            return "PortableIdentityInvalid:" + k
    for k in ("recipe_semantics_sha256", "cargo_lock_sha256",
              "manifest_set_sha256", "rust_toolchain_sha256",
              "environment_profile_sha256", "semantic_sha256", "receipt_sha256"):
        if not isinstance(r.get(k), str) or not H64.fullmatch(r[k]):
            return "PortableDigestInvalid:" + k
    for k in ("qualification_profile_id", "rustc_identity", "cargo_identity",
              "clippy_identity", "target_triple", "numerical_profile_id",
              "environment_equivalence_id"):
        if not nonempty(r.get(k)):
            return "PortableFieldMissing:" + k
    if r.get("clean_before") is not True or r.get("clean_after") is not True:
        return "PortableDirtySubject"
    if r["pre_tree_sha"] != r["subject_tree_sha"] or r["post_tree_sha"] != r["subject_tree_sha"]:
        return "PortableSubjectTreeMismatch"
    if r.get("dependency_resolution") != "locked":
        return "PortableUnlockedResolution"
    if not isinstance(r.get("command_argv"), list) or not r["command_argv"]:
        return "PortableCommandMissing"
    gates = r.get("gates")
    if not isinstance(gates, list) or not gates:
        return "PortableGatesMissing"
    sem = sha(SEM_DOMAIN, {k: r[k] for k in SEM_FIELDS})
    if r["semantic_sha256"] != sem:
        return "PortableSemanticDigestMismatch"
    x = dict(r)
    claimed = x.pop("receipt_sha256")
    if claimed != sha(PORTABLE_DOMAIN, x):
        return "PortableReceiptDigestMismatch"
    return None

def validate_hosted(r, raw_bytes, expected_sha):
    if hashlib.sha256(raw_bytes).hexdigest() != expected_sha:
        return "HostedArtifactDigestMismatch"
    if r.get("schema_version") != HOSTED_SCHEMA:
        return "HostedSchemaInvalid"
    q = r.get("qualification_profile") or {}
    v = r.get("verifier_authority") or {}
    s = r.get("subject") or {}
    t = r.get("toolchain") or {}
    if not nonempty(q.get("profile_id")):
        return "HostedProfileMissing:profile_id"
    if not isinstance(q.get("recipe_semantics_sha256"), str) or not H64.fullmatch(q["recipe_semantics_sha256"]):
        return "HostedProfileDigestInvalid"
    req = q.get("required_gates")
    if not isinstance(req, list) or not req or any(not nonempty(x) for x in req):
        return "HostedRequiredGatesInvalid"
    for obj, prefix in ((v, "HostedVerifier"), (s, "HostedSubject")):
        for k in ("checked_out_commit_sha", "checked_out_tree_sha"):
            if not isinstance(obj.get(k), str) or not H40.fullmatch(obj[k]):
                return prefix + "IdentityInvalid:" + k
    if not H40.fullmatch(v.get("expected_commit_sha", "")) or v["expected_commit_sha"] != v["checked_out_commit_sha"]:
        return "HostedVerifierExpectedCommitMismatch"
    if not H40.fullmatch(v.get("expected_tree_sha", "")) or v["expected_tree_sha"] != v["checked_out_tree_sha"]:
        return "HostedVerifierExpectedTreeMismatch"
    if not H40.fullmatch(s.get("expected_head_sha", "")) or s["expected_head_sha"] != s["checked_out_commit_sha"]:
        return "HostedExpectedHeadMismatch"
    base = s.get("qualification_base_sha")
    if base is not None and (not isinstance(base, str) or not H40.fullmatch(base)):
        return "HostedBaseInvalid"
    if v.get("source_state") != "exact_verifier_clean":
        return "HostedVerifierNotClean"
    if s.get("source_state") != "exact_raw_head_clean":
        return "HostedSubjectNotClean"
    for k in ("rustc", "cargo", "clippy"):
        if not nonempty(t.get(k)):
            return "HostedToolchainMissing:" + k
    if not isinstance(r.get("attempt_sha256"), str) or not H64.fullmatch(r["attempt_sha256"]):
        return "HostedAttemptDigestInvalid"
    gates = r.get("gates")
    if not isinstance(gates, dict) or not gates:
        return "HostedGatesMissing"
    for gate in req:
        if gates.get(gate) not in ("PASS", "NOT_APPLICABLE"):
            return "HostedRequiredGateNotPass:" + gate
    return None

def compare(p, h):
    q = h["qualification_profile"]
    v = h["verifier_authority"]
    s = h["subject"]
    t = h["toolchain"]
    comparisons = [
        ("SubjectShift", p["subject_sha"], s["checked_out_commit_sha"]),
        ("SubjectTreeShift", p["subject_tree_sha"], s["checked_out_tree_sha"]),
        ("BaseShift", p["base_sha"], s.get("qualification_base_sha")),
        ("VerifierShift", p["verifier_sha"], v["checked_out_commit_sha"]),
        ("VerifierTreeShift", p["verifier_tree_sha"], v["checked_out_tree_sha"]),
        ("ProfileShift", p["qualification_profile_id"], q["profile_id"]),
        ("RecipeShift", p["recipe_semantics_sha256"], q["recipe_semantics_sha256"]),
        ("RustcShift", p["rustc_identity"], t["rustc"]),
        ("CargoShift", p["cargo_identity"], t["cargo"]),
        ("ClippyShift", p["clippy_identity"], t["clippy"]),
    ]
    for label, a, b in comparisons:
        if a != b:
            return result(label)
    missing = [
        "base_tree_sha",
        "normalized_target_triple",
        "numerical_profile_id",
        "environment_equivalence_id",
        "portable_compatible_subject_input_digest",
    ]
    return result("CoreEquivalentExtendedBindingUnavailable", core=True, missing=missing)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("portable")
    ap.add_argument("hosted")
    ap.add_argument("--hosted-sha256", required=True)
    a = ap.parse_args()
    p = json.loads(Path(a.portable).read_text())
    raw = Path(a.hosted).read_bytes()
    h = json.loads(raw)
    err = validate_portable(p)
    if err:
        print(json.dumps(result(err, valid=False), sort_keys=True, separators=(",", ":")))
        raise SystemExit(1)
    if not H64.fullmatch(a.hosted_sha256):
        print(json.dumps(result("HostedExpectedDigestInvalid", valid=False), sort_keys=True, separators=(",", ":")))
        raise SystemExit(1)
    err = validate_hosted(h, raw, a.hosted_sha256)
    if err:
        print(json.dumps(result(err, valid=False), sort_keys=True, separators=(",", ":")))
        raise SystemExit(1)
    print(json.dumps(compare(p, h), sort_keys=True, separators=(",", ":")))

if __name__ == "__main__":
    main()
