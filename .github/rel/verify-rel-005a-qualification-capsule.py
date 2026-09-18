#!/usr/bin/env python3
"""Offline verifier for an REL-005A qualification capsule.

This verifier has no network or GitHub dependency. It validates the deterministic
tar envelope and the internal authority-chain consistency of the capsule bytes.
It does not independently establish the scientific premises or reproduce the
underlying experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import tarfile
from typing import Any

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_RE = re.compile(r"^[0-9a-f]{40}$")

INPUT_MEMBERS = {
    "predicate-contract-receipt.json",
    "execution-v3-receipt.json",
    "observation-seal-v3.json",
    "comparison-only-qualification-receipt.json",
    "qualification-input-manifest.json",
    "qualification-assembly-receipt.json",
    "qualification-only.json",
}
MANIFEST_MEMBER = "qualification-capsule-manifest.json"
ALL_MEMBERS = INPUT_MEMBERS | {MANIFEST_MEMBER}
FORBIDDEN_KEYS = {"predicates", "threshold_table", "observed", "expected"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(data: bytes, name: str) -> dict[str, Any]:
    value = json.loads(data.decode("utf-8"), object_pairs_hook=no_duplicate_pairs)
    require(isinstance(value, dict), f"{name}: expected JSON object")
    return value


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(key)
            keys |= walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            keys |= walk_keys(child)
    return keys


def valid_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def valid_git(value: Any) -> bool:
    return isinstance(value, str) and GIT_RE.fullmatch(value) is not None


def read_capsule(path: pathlib.Path) -> tuple[dict[str, bytes], list[tarfile.TarInfo]]:
    require(path.is_file(), "capsule file missing")
    with tarfile.open(path, mode="r:") as tf:
        members = tf.getmembers()
        names = [member.name for member in members]
        require(len(names) == len(set(names)), "duplicate tar member")
        require(set(names) == ALL_MEMBERS, f"tar member census mismatch: {sorted(names)}")
        require(names == sorted(names, key=lambda s: s.encode()), "tar members not bytewise sorted")
        data: dict[str, bytes] = {}
        for member in members:
            require(member.isfile(), f"{member.name}: non-file tar member")
            require("/" not in member.name and member.name not in {".", ".."}, f"{member.name}: nested/path member")
            require(member.mtime == 0, f"{member.name}: mtime mismatch")
            require(member.uid == 0 and member.gid == 0, f"{member.name}: uid/gid mismatch")
            require(member.uname == "" and member.gname == "", f"{member.name}: owner name mismatch")
            require(member.mode == 0o644, f"{member.name}: file mode mismatch")
            require(not member.pax_headers, f"{member.name}: PAX metadata present")
            handle = tf.extractfile(member)
            require(handle is not None, f"{member.name}: cannot read member")
            payload = handle.read()
            require(len(payload) == member.size, f"{member.name}: tar size mismatch")
            data[member.name] = payload
        return data, members


def verify(path: pathlib.Path) -> dict[str, Any]:
    data, _ = read_capsule(path)
    values = {name: parse_json(payload, name) for name, payload in data.items()}

    manifest = values[MANIFEST_MEMBER]
    require(manifest.get("schema") == "symthaea.rel.qualification-capsule-manifest.v1", "capsule manifest schema")
    require(manifest.get("authority") == "AttestationInputOnly", "capsule manifest authority")
    require(manifest.get("relation") == "REL-005A", "capsule manifest relation")
    require(manifest.get("canonicalization") == {
        "compression": "none",
        "sort": "bytewise basename ascending",
        "mtime": 0,
        "uid": 0,
        "gid": 0,
        "uname": "",
        "gname": "",
        "file_mode": "0644",
        "pax_headers": False,
    }, "capsule canonicalization contract mismatch")
    require(manifest.get("claims") == {
        "attestation_created": False,
        "qualification_completed": False,
        "rel_005a_qualified": False,
        "scientific_pass": False,
        "scientific_fail": False,
    }, "capsule manifest claim boundary mismatch")

    entries = manifest.get("files")
    require(isinstance(entries, list), "capsule manifest files must be list")
    entry_names = [entry.get("basename") for entry in entries]
    expected_names = sorted(INPUT_MEMBERS, key=lambda s: s.encode())
    require(entry_names == expected_names, "capsule manifest file order/census mismatch")
    for entry in entries:
        name = entry["basename"]
        require(type(entry.get("byte_length")) is int, f"{name}: byte_length invalid")
        require(entry["byte_length"] == len(data[name]), f"{name}: manifest length mismatch")
        require(valid_sha(entry.get("sha256")), f"{name}: manifest sha format")
        require(entry["sha256"] == sha_bytes(data[name]), f"{name}: manifest sha mismatch")

    p = values["predicate-contract-receipt.json"]
    e = values["execution-v3-receipt.json"]
    s = values["observation-seal-v3.json"]
    c = values["comparison-only-qualification-receipt.json"]
    m = values["qualification-input-manifest.json"]
    a = values["qualification-assembly-receipt.json"]
    q = values["qualification-only.json"]

    require(p.get("schema") == "symthaea.rel.predicate-contract-static-receipt.v1", "predicate schema")
    require(p.get("authority") == "PredicateContractOnly", "predicate authority")
    require(p.get("predicate_count") == 41, "predicate count")
    require(p.get("source_grounded") is True and p.get("predicate_contract_static_valid") is True, "predicate source grounding")

    predicate_head = p.get("subject_head")
    execution_head = e.get("subject_head")
    frozen_subject = p.get("frozen_scientific_subject")
    frozen_blob = p.get("frozen_test_blob")
    for value, label in ((predicate_head, "predicate head"), (execution_head, "execution head"),
                         (frozen_subject, "frozen subject"), (frozen_blob, "frozen blob")):
        require(valid_git(value), f"{label}: format")

    require(e.get("schema") == "symthaea.rel.execution-only-receipt.v3", "execution schema")
    require(e.get("authority") == "ExecutionOnly", "execution authority")
    require(e.get("predicate_contract_parent") == predicate_head, "execution predicate binding")
    require(e.get("frozen_scientific_subject") == frozen_subject and e.get("frozen_test_blob") == frozen_blob, "execution frozen identity")
    require(e.get("result") == "EXECUTION_OK" and str(e.get("measurement_exit_code")) == "0", "execution result")
    require(e.get("observation_present") is True and valid_sha(e.get("observation_sha256")), "execution observation binding")
    observation_sha = e["observation_sha256"]

    require(s.get("schema") == "symthaea.rel.observation-seal.v3", "seal schema")
    require(s.get("authority") == "ObservationSeal" and s.get("observation_status") == "sealed", "seal authority/status")
    require(s.get("adjudication") == "not_run" and s.get("scientific_result") == "not_run", "seal exceeds authority")
    require(s.get("execution_subject_head") == execution_head and s.get("predicate_contract_head") == predicate_head, "seal subject binding")
    require(s.get("frozen_scientific_subject") == frozen_subject and s.get("frozen_test_blob") == frozen_blob, "seal frozen identity")
    require(s.get("observation_sha256") == observation_sha and valid_sha(s.get("chain_commitment_sha256")), "seal commitment")
    seal_commitment = s["chain_commitment_sha256"]

    require(c.get("schema") == "symthaea.rel.comparison-only-qualification-receipt.v1", "comparison schema")
    require(c.get("authority") == "ComparisonOnly", "comparison authority")
    require(c.get("predicate_contract_head") == predicate_head and c.get("execution_subject_head") == execution_head, "comparison subject binding")
    require(c.get("frozen_scientific_subject") == frozen_subject and c.get("frozen_test_blob") == frozen_blob, "comparison frozen identity")
    require(c.get("observation_sha256") == observation_sha and c.get("seal_chain_commitment_sha256") == seal_commitment, "comparison evidence binding")
    require(valid_sha(c.get("full_comparison_sha256")), "full comparison hash")
    require(c.get("predicate_count") == 41, "comparison predicate count")
    passed = c.get("passed_count"); failed = c.get("failed_count"); ids = c.get("failed_predicate_ids")
    require(type(passed) is int and type(failed) is int and passed >= 0 and failed >= 0 and passed + failed == 41, "comparison counts")
    require(isinstance(ids, list) and all(isinstance(x, str) for x in ids), "failed predicate IDs")
    require(len(ids) == len(set(ids)) == failed, "failed predicate ID count/uniqueness")
    allowed_ids = [f"REL005A-P{i:03d}" for i in range(1, 42)]
    require(all(x in set(allowed_ids) for x in ids), "unknown failed predicate ID")
    require(ids == sorted(ids, key=allowed_ids.index), "failed predicate IDs not canonical")
    result = c.get("comparison_result")
    require(result in {"ALL_PREDICATES_PASS", "PREDICATE_FAILURES"}, "comparison result")
    if result == "ALL_PREDICATES_PASS":
        require(passed == 41 and failed == 0, "all-pass counts")
    else:
        require(failed > 0, "predicate-failure result lacks failures")

    require(m.get("schema") == "symthaea.rel.qualification-input-manifest.v2", "projection manifest schema")
    require(m.get("authority") == "EvidenceProjectionOnly", "projection manifest authority")
    require(m.get("predicate_contract_head") == predicate_head and m.get("execution_subject_head") == execution_head, "projection subject binding")
    require(m.get("observation_sha256") == observation_sha and m.get("seal_chain_commitment_sha256") == seal_commitment, "projection evidence binding")
    require(m.get("comparison_result") == result, "projection result binding")
    require(m.get("raw_observation_included") is False and m.get("raw_execution_logs_included") is False and m.get("detailed_predicate_values_included") is False, "projection leakage flags")
    projected_names = ["predicate-contract-receipt.json", "execution-v3-receipt.json", "observation-seal-v3.json", "comparison-only-qualification-receipt.json"]
    pentries = m.get("files")
    require(isinstance(pentries, list) and [x.get("basename") for x in pentries] == projected_names, "projection manifest census/order")
    for entry in pentries:
        name = entry["basename"]
        require(entry.get("byte_length") == len(data[name]), f"{name}: projection length")
        require(entry.get("sha256") == sha_bytes(data[name]), f"{name}: projection hash")

    require(a.get("schema") == "symthaea.rel.qualification-assembly-receipt.v3", "assembly schema")
    require(a.get("authority") == "QualificationAssemblyOnly", "assembly authority")
    require(valid_git(a.get("assembly_contract_head")), "assembly contract head")
    require(a.get("full_comparison_sha256") == c.get("full_comparison_sha256"), "assembly comparison hash")
    require(a.get("transport_identity_scientific_authority") is False, "transport promoted to scientific authority")
    require(a.get("inner_content_commitments_verified") is True, "assembly inner commitments not verified")
    require(valid_sha(a.get("authority_extraction_receipt_sha256")), "extraction receipt hash")
    require(a.get("claims", {}).get("receipt_extraction_performed") is True, "receipt extraction not claimed")
    require(a.get("claims", {}).get("projection_performed") is True, "projection not claimed")
    require(a.get("claims", {}).get("projection_raw_observation_visibility") is False, "projection saw raw observation")
    require(a.get("claims", {}).get("projection_execution_log_visibility") is False, "projection saw logs")
    require(a.get("claims", {}).get("qualification_completed") is False, "assembly exceeds qualification authority")

    for name in INPUT_MEMBERS - {"qualification-only.json"}:
        leaked = FORBIDDEN_KEYS & walk_keys(values[name])
        require(not leaked, f"{name}: forbidden detailed keys leaked: {sorted(leaked)}")

    require(q.get("schema") == "symthaea.rel.qualification-only.v1", "qualification schema")
    require(q.get("authority") == "QualificationOnly", "qualification authority")
    require(q.get("qualification_completed") is True, "qualification incomplete")
    for key, expected in (("comparison_result", result), ("predicate_count", 41),
                          ("passed_count", passed), ("failed_count", failed),
                          ("failed_predicate_ids", ids)):
        require(q.get(key) == expected, f"qualification {key} mismatch")
    scientific_pass = q.get("scientific_pass") is True
    scientific_fail = q.get("scientific_fail") is True
    require(scientific_pass ^ scientific_fail, "scientific pass/fail must be XOR")
    if result == "ALL_PREDICATES_PASS":
        require(scientific_pass and not scientific_fail and q.get("rel_005a_qualified") is True, "all-pass qualification mapping")
    else:
        require(scientific_fail and not scientific_pass and q.get("rel_005a_qualified") is False, "predicate-failure qualification mapping")

    return {
        "schema": "symthaea.rel.qualification-capsule-offline-verification.v1",
        "authority": "OfflineVerificationOnly",
        "capsule_sha256": sha_bytes(path.read_bytes()),
        "capsule_byte_length": path.stat().st_size,
        "member_count": len(data),
        "canonical_tar_verified": True,
        "manifest_hashes_verified": True,
        "authority_chain_consistent": True,
        "detailed_metric_keys_absent": True,
        "qualification_completed": True,
        "comparison_result": result,
        "scientific_pass": scientific_pass,
        "scientific_fail": scientific_fail,
        "rel_005a_qualified": q.get("rel_005a_qualified"),
        "limitations": [
            "does not independently reproduce the scientific experiment",
            "does not by itself verify GitHub/Sigstore provenance",
            "does not establish that the frozen predicates are scientifically sufficient",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capsule", type=pathlib.Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.capsule), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
