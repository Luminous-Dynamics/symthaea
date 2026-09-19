#!/usr/bin/env python3
"""Offline verifier for transparent REL-005A qualification capsule v3.

Authority: OfflineVerificationOnly.

This verifier strengthens internal capsule verification without becoming a new
scientific authority. It verifies exact canonical USTAR bytes, the capsule and
projected manifests, frozen authority identities and nonclaims, assembly/audit
hash links, the frozen REL-005A failed-predicate namespace, metric-free
information flow, and the already-frozen QualificationOnly mapping.

It does not independently reproduce the experiment, judge predicate
sufficiency, or establish Sigstore provenance.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import pathlib
import re
import tarfile
from typing import Any

SHA = re.compile(r"^[0-9a-f]{64}$")
GIT = re.compile(r"^[0-9a-f]{40}$")
ARTIFACT_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")

PREDICATE_HEAD = "43bf1d588447f602ce0f1549986bb558a839762c"
EXECUTION_HEAD = "6931639e53060809f9d459196a329c4fe983be7b"
FROZEN_SUBJECT = "7f5826675b44dd0d3f702f62bc9818f7990e4a01"
FROZEN_BLOB = "787ea051ae0bf8e5667b6925481afd46c15d0bc4"
OBS_SHA = "e271a5c2e6b51fda15cd6c3209e52859296f44ecd790b9c5e72f769d4aec1c4d"
SEAL_CHAIN = "a3a288caa9221ec5859b5ac81fcc55f61b4ff43d5e52ad096021c99ea6bf31ab"

COMPARISON_HEAD = "a47efccf80fa66bacbdfa9930f59c8439be7ee38"
COMPARISON_RUN = 35349750595
PROJECTION_HEAD = "07af7c4bdfdda377aa8175efe7a3233ce90e889e"
PROJECTION_RUN = 35324485955
ASSEMBLY_HEAD = "3dc01663ff8b2f9f6eb0568f8649ccdd999ddf29"
PREDICATE_RUN = 35278498517
SEAL_RUN = 35284772522
SEAL_JOB = 105414583527

COMPARISON_ARTIFACT_NAME = (
    "rel-005a-comparison-v3-a47efccf80fa66bacbdfa9930f59c8439be7ee38"
)
PROJECTION_ARTIFACT_NAME = (
    "rel-005a-qualification-projection-contract-"
    "07af7c4bdfdda377aa8175efe7a3233ce90e889e"
)

PREDICATE_IDS = {f"REL005A-P{i:03d}" for i in range(1, 42)}
EXECUTION_REPLAY_BASENAMES = {
    "cargo-version.txt",
    "clippy-version.txt",
    "execution-exit-code.txt",
    "execution-v3-receipt.json",
    "execution-v3-static-audit.json",
    "execution.stderr.log",
    "execution.stdout.log",
    "rel-005a-execution-v3-observation.json",
    "rustc-version.txt",
    "rustfmt-version.txt",
}

INPUT = {
    "predicate-contract-receipt.json",
    "execution-v3-receipt.json",
    "observation-seal-v3.json",
    "comparison-only-qualification-receipt.json",
    "qualification-input-manifest.json",
    "qualification-assembly-receipt.json",
    "qualification-only.json",
    "sealed-manifest-replay-receipt.json",
    "authority-receipt-extraction.json",
}
MANIFEST = "qualification-capsule-manifest.json"
ALL = INPUT | {MANIFEST}
PROJECTED_FILES = [
    "predicate-contract-receipt.json",
    "execution-v3-receipt.json",
    "observation-seal-v3.json",
    "comparison-only-qualification-receipt.json",
]
FORBIDDEN_DETAIL_KEYS = {"predicates", "threshold_table", "observed", "expected"}

CANON = {
    "compression": "none",
    "format": "ustar",
    "sort": "bytewise basename ascending",
    "mtime": 0,
    "uid": 0,
    "gid": 0,
    "uname": "",
    "gname": "",
    "file_mode": "0644",
    "pax_headers": False,
}


def req(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def nodup(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        req(key not in out, f"duplicate JSON key: {key}")
        out[key] = value
    return out


def parse(data: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=nodup)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name}: invalid UTF-8 JSON") from exc
    req(isinstance(value, dict), f"{name}: expected JSON object")
    return value


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(value: Any) -> bool:
    return isinstance(value, str) and SHA.fullmatch(value) is not None


def git(value: Any) -> bool:
    return isinstance(value, str) and GIT.fullmatch(value) is not None


def artifact_digest(value: Any) -> bool:
    return isinstance(value, str) and ARTIFACT_DIGEST.fullmatch(value) is not None


def all_claims_false(value: Any, label: str) -> None:
    req(isinstance(value, dict), f"{label}: claims missing")
    req(value, f"{label}: claims empty")
    for key, claim in value.items():
        req(type(claim) is bool, f"{label}: non-boolean claim {key}")
        req(claim is False, f"{label}: overclaim {key}")


def claims_true_only(value: Any, allowed_true: set[str], label: str) -> None:
    req(isinstance(value, dict), f"{label}: claims missing")
    for key, claim in value.items():
        req(type(claim) is bool, f"{label}: non-boolean claim {key}")
        if key in allowed_true:
            req(claim is True, f"{label}: expected true claim {key}")
        else:
            req(claim is False, f"{label}: overclaim {key}")
    req(allowed_true <= set(value), f"{label}: missing required true claim")


def recursive_keys(value: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            out.add(key)
            out |= recursive_keys(child)
    elif isinstance(value, list):
        for child in value:
            out |= recursive_keys(child)
    return out


def canonical_tar_bytes(raw: dict[str, bytes]) -> bytes:
    req(set(raw) == ALL, "canonical tar input census mismatch")
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        for name in sorted(raw, key=lambda s: s.encode()):
            data = raw[name]
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mode = 0o644
            tf.addfile(info, io.BytesIO(data))
    return stream.getvalue()


def read_tar_bytes(data: bytes) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tf:
            members = tf.getmembers()
            names = [m.name for m in members]
            req(len(names) == len(set(names)), "duplicate tar member")
            req(set(names) == ALL, f"tar census mismatch: {sorted(names)}")
            req(names == sorted(names, key=lambda s: s.encode()), "tar member order not canonical")
            for member in members:
                req(member.isfile(), f"{member.name}: non-file member")
                req(
                    member.mtime == 0
                    and member.uid == 0
                    and member.gid == 0
                    and member.uname == ""
                    and member.gname == "",
                    f"{member.name}: metadata mismatch",
                )
                req(member.mode == 0o644, f"{member.name}: mode mismatch")
                extracted = tf.extractfile(member)
                req(extracted is not None, f"{member.name}: unreadable")
                out[member.name] = extracted.read()
    except tarfile.TarError as exc:
        raise ValueError("invalid tar archive") from exc
    req(data == canonical_tar_bytes(out), "capsule bytes are not exact canonical USTAR encoding")
    return out


def read_tar(path: pathlib.Path) -> tuple[bytes, dict[str, bytes]]:
    req(path.is_file(), "capsule missing")
    data = path.read_bytes()
    return data, read_tar_bytes(data)


def verify_manifest(raw: dict[str, bytes], docs: dict[str, dict[str, Any]]) -> None:
    manifest = docs[MANIFEST]
    req(manifest.get("schema") == "symthaea.rel.qualification-capsule-manifest.v2", "capsule manifest schema")
    req(manifest.get("authority") == "AttestationInputOnly", "capsule manifest authority")
    req(manifest.get("relation") == "REL-005A", "capsule manifest relation")
    req(manifest.get("canonicalization") == CANON, "capsule manifest canonicalization mismatch")
    req(
        manifest.get("audit_receipts_included")
        == ["sealed-manifest-replay-receipt.json", "authority-receipt-extraction.json"],
        "audit census declaration",
    )
    req(
        manifest.get("raw_observation_included") is False
        and manifest.get("execution_logs_included") is False
        and manifest.get("detailed_predicate_values_included") is False,
        "capsule leakage declaration",
    )
    all_claims_false(manifest.get("claims"), "capsule manifest")

    entries = manifest.get("files")
    req(isinstance(entries, list), "manifest files invalid")
    names = [entry.get("basename") for entry in entries if isinstance(entry, dict)]
    req(len(entries) == len(INPUT), "capsule manifest entry count")
    req(names == sorted(INPUT, key=lambda s: s.encode()), "manifest member order/census")
    req(len(names) == len(set(names)), "duplicate capsule manifest basename")
    for entry in entries:
        req(isinstance(entry, dict), "capsule manifest entry not object")
        name = entry.get("basename")
        req(name in INPUT, "capsule manifest unexpected member")
        req(type(entry.get("byte_length")) is int and entry["byte_length"] >= 0, f"{name}: invalid byte_length")
        req(entry["byte_length"] == len(raw[name]), f"{name}: length mismatch")
        req(entry.get("sha256") == digest(raw[name]), f"{name}: hash mismatch")


def verify_projected_manifest(raw: dict[str, bytes], docs: dict[str, dict[str, Any]]) -> None:
    manifest = docs["qualification-input-manifest.json"]
    comparison = docs["comparison-only-qualification-receipt.json"]
    execution = docs["execution-v3-receipt.json"]
    seal = docs["observation-seal-v3.json"]

    req(manifest.get("schema") == "symthaea.rel.qualification-input-manifest.v2", "qualification input manifest schema")
    req(manifest.get("authority") == "EvidenceProjectionOnly", "qualification input manifest authority")
    req(manifest.get("predicate_contract_head") == PREDICATE_HEAD, "qualification input predicate head")
    req(manifest.get("execution_subject_head") == EXECUTION_HEAD, "qualification input execution head")
    req(manifest.get("observation_sha256") == OBS_SHA == execution.get("observation_sha256"), "qualification input observation binding")
    req(manifest.get("seal_chain_commitment_sha256") == SEAL_CHAIN == seal.get("chain_commitment_sha256"), "qualification input seal binding")
    req(manifest.get("comparison_result") == comparison.get("comparison_result"), "qualification input comparison result")
    req(
        manifest.get("raw_observation_included") is False
        and manifest.get("raw_execution_logs_included") is False
        and manifest.get("detailed_predicate_values_included") is False,
        "qualification input leakage flags",
    )

    entries = manifest.get("files")
    req(isinstance(entries, list) and len(entries) == 4, "qualification input file census")
    req([e.get("basename") for e in entries if isinstance(e, dict)] == PROJECTED_FILES, "qualification input file order/census")
    for entry in entries:
        req(isinstance(entry, dict), "qualification input manifest entry not object")
        name = entry.get("basename")
        req(name in PROJECTED_FILES, "qualification input unexpected file")
        req(type(entry.get("byte_length")) is int and entry["byte_length"] == len(raw[name]), f"{name}: projected byte length mismatch")
        req(entry.get("sha256") == digest(raw[name]), f"{name}: projected hash mismatch")


def verify_authorities(raw: dict[str, bytes], docs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    predicate = docs["predicate-contract-receipt.json"]
    execution = docs["execution-v3-receipt.json"]
    seal = docs["observation-seal-v3.json"]
    comparison = docs["comparison-only-qualification-receipt.json"]
    replay = docs["sealed-manifest-replay-receipt.json"]
    extraction = docs["authority-receipt-extraction.json"]
    assembly = docs["qualification-assembly-receipt.json"]
    qualification = docs["qualification-only.json"]

    req(predicate.get("schema") == "symthaea.rel.predicate-contract-static-receipt.v1", "predicate schema")
    req(predicate.get("authority") == "PredicateContractOnly", "predicate authority")
    req(predicate.get("subject_head") == PREDICATE_HEAD, "predicate head")
    req(predicate.get("predicate_count") == 41, "predicate count")
    req(predicate.get("source_grounded") is True and predicate.get("predicate_contract_static_valid") is True, "predicate admissibility")
    req(predicate.get("frozen_scientific_subject") == FROZEN_SUBJECT, "predicate frozen subject")
    req(predicate.get("frozen_test_blob") == FROZEN_BLOB, "predicate frozen blob")
    all_claims_false(predicate.get("claims"), "predicate")

    req(execution.get("schema") == "symthaea.rel.execution-only-receipt.v3", "execution schema")
    req(execution.get("authority") == "ExecutionOnly", "execution authority")
    req(execution.get("subject_head") == EXECUTION_HEAD, "execution head")
    req(execution.get("predicate_contract_parent") == PREDICATE_HEAD, "execution predicate head")
    req(execution.get("frozen_scientific_subject") == FROZEN_SUBJECT, "execution frozen subject")
    req(execution.get("frozen_test_blob") == FROZEN_BLOB, "execution frozen blob")
    req(execution.get("result") == "EXECUTION_OK", "execution result")
    req(str(execution.get("measurement_exit_code")) == "0", "execution exit")
    req(execution.get("observation_present") is True, "execution observation missing")
    req(execution.get("observation_sha256") == OBS_SHA, "execution observation hash")
    all_claims_false(execution.get("claims"), "execution")

    req(seal.get("schema") == "symthaea.rel.observation-seal.v3", "seal schema")
    req(seal.get("authority") == "ObservationSeal", "seal authority")
    req(seal.get("observation_status") == "sealed", "seal status")
    req(seal.get("adjudication") == "not_run" and seal.get("scientific_result") == "not_run", "seal authority boundary")
    req(seal.get("execution_subject_head") == EXECUTION_HEAD, "seal execution head")
    req(seal.get("predicate_contract_head") == PREDICATE_HEAD, "seal predicate head")
    req(seal.get("frozen_scientific_subject") == FROZEN_SUBJECT, "seal frozen subject")
    req(seal.get("frozen_test_blob") == FROZEN_BLOB, "seal frozen blob")
    req(seal.get("observation_sha256") == OBS_SHA, "seal observation hash")
    req(seal.get("chain_commitment_sha256") == SEAL_CHAIN, "seal chain")
    claims_true_only(seal.get("claims"), {"observation_sealed"}, "seal")

    req(comparison.get("schema") == "symthaea.rel.comparison-only-qualification-receipt.v1", "comparison schema")
    req(comparison.get("authority") == "ComparisonOnly", "comparison authority")
    req(comparison.get("predicate_contract_head") == PREDICATE_HEAD, "comparison predicate head")
    req(comparison.get("execution_subject_head") == EXECUTION_HEAD, "comparison execution head")
    req(comparison.get("frozen_scientific_subject") == FROZEN_SUBJECT, "comparison frozen subject")
    req(comparison.get("frozen_test_blob") == FROZEN_BLOB, "comparison frozen blob")
    req(comparison.get("observation_sha256") == OBS_SHA, "comparison observation hash")
    req(comparison.get("seal_chain_commitment_sha256") == SEAL_CHAIN, "comparison seal chain")
    req(sha(comparison.get("full_comparison_sha256")), "full comparison SHA format")
    claims_true_only(comparison.get("claims"), {"comparison_only_adjudicated"}, "comparison")

    count = comparison.get("predicate_count")
    passed = comparison.get("passed_count")
    failed = comparison.get("failed_count")
    failed_ids = comparison.get("failed_predicate_ids")
    result = comparison.get("comparison_result")
    req(type(count) is int and count == 41, "comparison predicate count")
    req(type(passed) is int and type(failed) is int and passed >= 0 and failed >= 0 and passed + failed == 41, "comparison counts")
    req(isinstance(failed_ids, list) and len(failed_ids) == failed and len(failed_ids) == len(set(failed_ids)), "comparison failed IDs")
    req(all(isinstance(pid, str) and pid in PREDICATE_IDS for pid in failed_ids), "failed predicate outside frozen namespace")
    if result == "ALL_PREDICATES_PASS":
        req(passed == 41 and failed == 0 and failed_ids == [], "all-pass counts")
    elif result == "PREDICATE_FAILURES":
        req(failed > 0, "predicate-failure result without failed predicates")
    else:
        raise ValueError("unsupported comparison result")

    req(replay.get("schema") == "symthaea.rel.sealed-manifest-replay-receipt.v1", "replay schema")
    req(replay.get("authority") == "SealedManifestReplayOnly", "replay authority")
    req(replay.get("execution_subject_head") == EXECUTION_HEAD, "replay execution head")
    req(sha(replay.get("manifest_sha256")), "replay manifest hash format")
    req(replay.get("manifest_entry_count") == 10, "replay manifest entry count")
    req(
        replay.get("exact_file_census_verified") is True
        and replay.get("byte_lengths_verified") is True
        and replay.get("sha256_commitments_verified") is True,
        "replay incomplete",
    )
    req(
        replay.get("scientific_observation_fields_parsed") is False
        and replay.get("execution_logs_parsed") is False,
        "replay exceeded authority",
    )
    replay_names = replay.get("verified_basenames")
    req(isinstance(replay_names, list) and len(replay_names) == 10 and len(set(replay_names)) == 10, "replay basename census")
    req(set(replay_names) == EXECUTION_REPLAY_BASENAMES, "replay basenames mismatch")
    all_claims_false(replay.get("claims"), "replay")

    req(extraction.get("schema") == "symthaea.rel.authority-receipt-extraction-receipt.v2", "extraction schema")
    req(extraction.get("authority") == "AuthorityReceiptExtractionOnly", "extraction authority")
    req(sha(extraction.get("pipeline_source_sha256")), "pipeline source hash format")
    req(extraction.get("observation_sha256_verified") == OBS_SHA, "extraction observation binding")
    req(extraction.get("seal_chain_commitment_sha256_verified") == SEAL_CHAIN, "extraction seal binding")
    req(extraction.get("predicate_receipt_sha256") == digest(raw["predicate-contract-receipt.json"]), "extraction predicate hash")
    req(extraction.get("execution_receipt_sha256") == digest(raw["execution-v3-receipt.json"]), "extraction execution hash")
    req(extraction.get("seal_receipt_sha256") == digest(raw["observation-seal-v3.json"]), "extraction seal hash")
    req(
        extraction.get("raw_observation_output") is False
        and extraction.get("execution_logs_output") is False
        and extraction.get("scientific_observation_fields_parsed") is False,
        "extraction exceeded authority",
    )
    all_claims_false(extraction.get("claims"), "extraction")

    req(assembly.get("schema") == "symthaea.rel.qualification-assembly-receipt.v3", "assembly schema")
    req(assembly.get("authority") == "QualificationAssemblyOnly", "assembly authority")
    req(assembly.get("assembly_contract_head") == ASSEMBLY_HEAD, "assembly contract head")
    req(assembly.get("comparison_subject_head") == COMPARISON_HEAD, "assembly comparison head")
    req(assembly.get("comparison_source_run_id") == COMPARISON_RUN, "assembly comparison run")
    req(type(assembly.get("comparison_source_job_id")) is int and assembly["comparison_source_job_id"] > 0, "assembly comparison job")
    req(type(assembly.get("comparison_artifact_id")) is int and assembly["comparison_artifact_id"] > 0, "assembly comparison artifact id")
    req(assembly.get("comparison_artifact_name") == COMPARISON_ARTIFACT_NAME, "assembly comparison artifact name")
    req(artifact_digest(assembly.get("comparison_artifact_digest")), "assembly comparison artifact digest")
    req(assembly.get("full_comparison_sha256") == comparison.get("full_comparison_sha256"), "assembly full comparison hash")
    req(assembly.get("projection_contract_head") == PROJECTION_HEAD, "assembly projection head")
    req(assembly.get("projection_contract_run_id") == PROJECTION_RUN, "assembly projection run")
    req(type(assembly.get("projection_contract_source_job_id")) is int and assembly["projection_contract_source_job_id"] > 0, "assembly projection job")
    req(type(assembly.get("projection_contract_artifact_id")) is int and assembly["projection_contract_artifact_id"] > 0, "assembly projection artifact id")
    req(assembly.get("projection_contract_artifact_name") == PROJECTION_ARTIFACT_NAME, "assembly projection artifact name")
    req(artifact_digest(assembly.get("projection_contract_artifact_digest")), "assembly projection artifact digest")
    req(sha(assembly.get("projection_contract_receipt_sha256")), "assembly projection receipt hash")
    req(assembly.get("predicate_source_run_id") == PREDICATE_RUN, "assembly predicate run")
    req(type(assembly.get("predicate_artifact_id")) is int and assembly["predicate_artifact_id"] > 0, "assembly predicate artifact id")
    req(artifact_digest(assembly.get("predicate_artifact_digest")), "assembly predicate artifact digest")
    req(assembly.get("seal_source_run_id") == SEAL_RUN, "assembly seal run")
    req(assembly.get("seal_source_job_id") == SEAL_JOB, "assembly seal job")
    req(type(assembly.get("seal_artifact_id")) is int and assembly["seal_artifact_id"] > 0, "assembly seal artifact id")
    req(artifact_digest(assembly.get("seal_artifact_digest")), "assembly seal artifact digest")
    req(assembly.get("predicate_receipt_sha256") == digest(raw["predicate-contract-receipt.json"]), "assembly predicate receipt hash")
    req(assembly.get("execution_receipt_sha256") == digest(raw["execution-v3-receipt.json"]), "assembly execution receipt hash")
    req(assembly.get("seal_receipt_sha256") == digest(raw["observation-seal-v3.json"]), "assembly seal receipt hash")
    req(assembly.get("authority_extraction_receipt_sha256") == digest(raw["authority-receipt-extraction.json"]), "assembly extraction receipt hash")
    req(assembly.get("sealed_manifest_replay_receipt_sha256") == digest(raw["sealed-manifest-replay-receipt.json"]), "assembly replay receipt hash")
    req(assembly.get("transport_identity_scientific_authority") is False, "assembly transport authority")
    req(assembly.get("inner_content_commitments_verified") is True, "assembly inner commitments")
    claims_true_only(
        assembly.get("claims"),
        {"receipt_extraction_performed", "projection_performed"},
        "assembly",
    )

    req(qualification.get("schema") == "symthaea.rel.qualification-only.v1", "qualification schema")
    req(qualification.get("authority") == "QualificationOnly", "qualification authority")
    req(qualification.get("predicate_contract_head") == PREDICATE_HEAD, "qualification predicate head")
    req(qualification.get("execution_subject_head") == EXECUTION_HEAD, "qualification execution head")
    req(qualification.get("frozen_scientific_subject") == FROZEN_SUBJECT, "qualification frozen subject")
    req(qualification.get("frozen_test_blob") == FROZEN_BLOB, "qualification frozen blob")
    for key in ("comparison_result", "predicate_count", "passed_count", "failed_count", "failed_predicate_ids"):
        req(qualification.get(key) == comparison.get(key), f"qualification {key} mismatch")
    req(qualification.get("qualification_completed") is True, "qualification not completed")
    scientific_pass = qualification.get("scientific_pass") is True
    scientific_fail = qualification.get("scientific_fail") is True
    req(scientific_pass ^ scientific_fail, "scientific PASS/FAIL not XOR")
    if scientific_pass:
        req(result == "ALL_PREDICATES_PASS", "PASS mapping comparison mismatch")
        req(qualification.get("rel_005a_qualified") is True, "PASS mapping qualification mismatch")
    else:
        req(result == "PREDICATE_FAILURES", "FAIL mapping comparison mismatch")
        req(qualification.get("rel_005a_qualified") is False, "FAIL mapping qualification mismatch")

    return {
        "comparison_result": result,
        "scientific_pass": scientific_pass,
        "scientific_fail": scientific_fail,
    }


def verify_bytes(data: bytes) -> dict[str, Any]:
    raw = read_tar_bytes(data)
    docs = {name: parse(raw[name], name) for name in ALL}
    verify_manifest(raw, docs)
    verify_projected_manifest(raw, docs)
    authority = verify_authorities(raw, docs)

    leaked: set[str] = set()
    for name in INPUT:
        leaked |= FORBIDDEN_DETAIL_KEYS & recursive_keys(docs[name])
    req(not leaked, f"forbidden detailed keys leaked: {sorted(leaked)}")

    return {
        "schema": "symthaea.rel.offline-qualification-capsule-verification.v3",
        "authority": "OfflineVerificationOnly",
        "capsule_sha256": digest(data),
        "member_count": len(ALL),
        "canonical_ustar_bytes_verified": True,
        "manifest_hashes_verified": True,
        "manifest_canonicalization_verified": True,
        "projected_manifest_hashes_verified": True,
        "audit_receipt_hash_links_verified": True,
        "upstream_authority_claims_verified": True,
        "assembly_provenance_bindings_verified": True,
        "failed_predicate_namespace_verified": True,
        "metric_leakage_absent": True,
        "authority_chain_consistent": True,
        "comparison_result": authority["comparison_result"],
        "qualification_mapping_consistent": True,
        "does_not_establish": [
            "independent replication",
            "predicate sufficiency",
            "scientific truth beyond the represented authority chain",
            "Sigstore provenance",
        ],
    }


def verify(path: pathlib.Path) -> dict[str, Any]:
    data, _ = read_tar(path)
    return verify_bytes(data)


def make_capsule_manifest(raw: dict[str, bytes]) -> dict[str, Any]:
    return {
        "schema": "symthaea.rel.qualification-capsule-manifest.v2",
        "authority": "AttestationInputOnly",
        "relation": "REL-005A",
        "canonicalization": copy.deepcopy(CANON),
        "files": [
            {
                "basename": name,
                "byte_length": len(raw[name]),
                "sha256": digest(raw[name]),
            }
            for name in sorted(INPUT, key=lambda s: s.encode())
        ],
        "audit_receipts_included": [
            "sealed-manifest-replay-receipt.json",
            "authority-receipt-extraction.json",
        ],
        "raw_observation_included": False,
        "execution_logs_included": False,
        "detailed_predicate_values_included": False,
        "claims": {
            "attestation_created": False,
            "qualification_completed": False,
            "rel_005a_qualified": False,
            "scientific_pass": False,
            "scientific_fail": False,
        },
    }


def synthetic_docs(result: str = "ALL_PREDICATES_PASS") -> dict[str, dict[str, Any]]:
    req(result in {"ALL_PREDICATES_PASS", "PREDICATE_FAILURES"}, "synthetic result")
    failed_ids = [] if result == "ALL_PREDICATES_PASS" else ["REL005A-P001"]
    failed = len(failed_ids)
    passed = 41 - failed

    false5 = {
        "observation_sealed": False,
        "comparison_only_adjudicated": False,
        "rel_005a_qualified": False,
        "scientific_pass": False,
        "scientific_fail": False,
    }
    predicate = {
        "schema": "symthaea.rel.predicate-contract-static-receipt.v1",
        "authority": "PredicateContractOnly",
        "subject_head": PREDICATE_HEAD,
        "predicate_count": 41,
        "source_grounded": True,
        "predicate_contract_static_valid": True,
        "frozen_scientific_subject": FROZEN_SUBJECT,
        "frozen_test_blob": FROZEN_BLOB,
        "claims": copy.deepcopy(false5),
    }
    execution = {
        "schema": "symthaea.rel.execution-only-receipt.v3",
        "authority": "ExecutionOnly",
        "subject_head": EXECUTION_HEAD,
        "predicate_contract_parent": PREDICATE_HEAD,
        "frozen_scientific_subject": FROZEN_SUBJECT,
        "frozen_test_blob": FROZEN_BLOB,
        "result": "EXECUTION_OK",
        "measurement_exit_code": "0",
        "observation_present": True,
        "observation_sha256": OBS_SHA,
        "claims": copy.deepcopy(false5),
    }
    seal = {
        "schema": "symthaea.rel.observation-seal.v3",
        "authority": "ObservationSeal",
        "observation_status": "sealed",
        "adjudication": "not_run",
        "scientific_result": "not_run",
        "execution_subject_head": EXECUTION_HEAD,
        "predicate_contract_head": PREDICATE_HEAD,
        "frozen_scientific_subject": FROZEN_SUBJECT,
        "frozen_test_blob": FROZEN_BLOB,
        "observation_sha256": OBS_SHA,
        "chain_commitment_sha256": SEAL_CHAIN,
        "claims": {
            "observation_sealed": True,
            "comparison_only_adjudicated": False,
            "rel_005a_qualified": False,
            "scientific_pass": False,
            "scientific_fail": False,
        },
    }
    comparison = {
        "schema": "symthaea.rel.comparison-only-qualification-receipt.v1",
        "authority": "ComparisonOnly",
        "predicate_contract_head": PREDICATE_HEAD,
        "execution_subject_head": EXECUTION_HEAD,
        "frozen_scientific_subject": FROZEN_SUBJECT,
        "frozen_test_blob": FROZEN_BLOB,
        "observation_sha256": OBS_SHA,
        "seal_chain_commitment_sha256": SEAL_CHAIN,
        "predicate_count": 41,
        "passed_count": passed,
        "failed_count": failed,
        "failed_predicate_ids": failed_ids,
        "comparison_result": result,
        "full_comparison_sha256": "55" * 32,
        "claims": {
            "comparison_only_adjudicated": True,
            "rel_005a_qualified": False,
            "scientific_pass": False,
            "scientific_fail": False,
        },
    }

    raw4 = {
        "predicate-contract-receipt.json": json_bytes(predicate),
        "execution-v3-receipt.json": json_bytes(execution),
        "observation-seal-v3.json": json_bytes(seal),
        "comparison-only-qualification-receipt.json": json_bytes(comparison),
    }
    projected_manifest = {
        "schema": "symthaea.rel.qualification-input-manifest.v2",
        "authority": "EvidenceProjectionOnly",
        "predicate_contract_head": PREDICATE_HEAD,
        "execution_subject_head": EXECUTION_HEAD,
        "observation_sha256": OBS_SHA,
        "seal_chain_commitment_sha256": SEAL_CHAIN,
        "comparison_result": result,
        "raw_observation_included": False,
        "raw_execution_logs_included": False,
        "detailed_predicate_values_included": False,
        "files": [
            {
                "basename": name,
                "byte_length": len(raw4[name]),
                "sha256": digest(raw4[name]),
            }
            for name in PROJECTED_FILES
        ],
    }
    replay = {
        "schema": "symthaea.rel.sealed-manifest-replay-receipt.v1",
        "authority": "SealedManifestReplayOnly",
        "execution_subject_head": EXECUTION_HEAD,
        "manifest_sha256": "66" * 32,
        "manifest_entry_count": 10,
        "exact_file_census_verified": True,
        "byte_lengths_verified": True,
        "sha256_commitments_verified": True,
        "scientific_observation_fields_parsed": False,
        "execution_logs_parsed": False,
        "verified_basenames": sorted(EXECUTION_REPLAY_BASENAMES),
        "claims": {
            "comparison_only_adjudicated": False,
            "qualification_completed": False,
            "rel_005a_qualified": False,
            "scientific_pass": False,
            "scientific_fail": False,
        },
    }
    extraction = {
        "schema": "symthaea.rel.authority-receipt-extraction-receipt.v2",
        "authority": "AuthorityReceiptExtractionOnly",
        "pipeline_source_sha256": "77" * 32,
        "observation_sha256_verified": OBS_SHA,
        "seal_chain_commitment_sha256_verified": SEAL_CHAIN,
        "predicate_receipt_sha256": digest(raw4["predicate-contract-receipt.json"]),
        "execution_receipt_sha256": digest(raw4["execution-v3-receipt.json"]),
        "seal_receipt_sha256": digest(raw4["observation-seal-v3.json"]),
        "raw_observation_output": False,
        "execution_logs_output": False,
        "scientific_observation_fields_parsed": False,
        "claims": {
            "comparison_only_adjudicated": False,
            "qualification_completed": False,
            "rel_005a_qualified": False,
            "scientific_pass": False,
            "scientific_fail": False,
        },
    }
    qualification = {
        "schema": "symthaea.rel.qualification-only.v1",
        "authority": "QualificationOnly",
        "predicate_contract_head": PREDICATE_HEAD,
        "execution_subject_head": EXECUTION_HEAD,
        "frozen_scientific_subject": FROZEN_SUBJECT,
        "frozen_test_blob": FROZEN_BLOB,
        "predicate_count": 41,
        "comparison_result": result,
        "passed_count": passed,
        "failed_count": failed,
        "failed_predicate_ids": failed_ids,
        "qualification_completed": True,
        "rel_005a_qualified": result == "ALL_PREDICATES_PASS",
        "scientific_pass": result == "ALL_PREDICATES_PASS",
        "scientific_fail": result == "PREDICATE_FAILURES",
    }

    raw_replay = json_bytes(replay)
    raw_extraction = json_bytes(extraction)
    assembly = {
        "schema": "symthaea.rel.qualification-assembly-receipt.v3",
        "authority": "QualificationAssemblyOnly",
        "assembly_contract_head": ASSEMBLY_HEAD,
        "comparison_subject_head": COMPARISON_HEAD,
        "comparison_source_run_id": COMPARISON_RUN,
        "comparison_source_job_id": 123456789,
        "comparison_artifact_id": 222222222,
        "comparison_artifact_name": COMPARISON_ARTIFACT_NAME,
        "comparison_artifact_digest": "sha256:" + "88" * 32,
        "full_comparison_sha256": comparison["full_comparison_sha256"],
        "projection_contract_head": PROJECTION_HEAD,
        "projection_contract_run_id": PROJECTION_RUN,
        "projection_contract_source_job_id": 333333333,
        "projection_contract_artifact_id": 444444444,
        "projection_contract_artifact_name": PROJECTION_ARTIFACT_NAME,
        "projection_contract_artifact_digest": "sha256:" + "99" * 32,
        "projection_contract_receipt_sha256": "aa" * 32,
        "predicate_source_run_id": PREDICATE_RUN,
        "predicate_artifact_id": 555555555,
        "predicate_artifact_digest": "sha256:" + "bb" * 32,
        "seal_source_run_id": SEAL_RUN,
        "seal_source_job_id": SEAL_JOB,
        "seal_artifact_id": 666666666,
        "seal_artifact_digest": "sha256:" + "cc" * 32,
        "predicate_receipt_sha256": digest(raw4["predicate-contract-receipt.json"]),
        "execution_receipt_sha256": digest(raw4["execution-v3-receipt.json"]),
        "seal_receipt_sha256": digest(raw4["observation-seal-v3.json"]),
        "authority_extraction_receipt_sha256": digest(raw_extraction),
        "sealed_manifest_replay_receipt_sha256": digest(raw_replay),
        "transport_identity_scientific_authority": False,
        "inner_content_commitments_verified": True,
        "claims": {
            "receipt_extraction_performed": True,
            "projection_performed": True,
            "projection_raw_observation_visibility": False,
            "projection_execution_log_visibility": False,
            "source_extraction_raw_observation_output": False,
            "source_extraction_execution_log_output": False,
            "qualification_completed": False,
            "rel_005a_qualified": False,
            "scientific_pass": False,
            "scientific_fail": False,
        },
    }

    return {
        "predicate-contract-receipt.json": predicate,
        "execution-v3-receipt.json": execution,
        "observation-seal-v3.json": seal,
        "comparison-only-qualification-receipt.json": comparison,
        "qualification-input-manifest.json": projected_manifest,
        "qualification-assembly-receipt.json": assembly,
        "qualification-only.json": qualification,
        "sealed-manifest-replay-receipt.json": replay,
        "authority-receipt-extraction.json": extraction,
    }


def rebind_synthetic_dependencies(docs: dict[str, dict[str, Any]]) -> None:
    """Recompute synthetic downstream byte commitments after an upstream mutation."""
    raw4 = {
        name: json_bytes(docs[name])
        for name in PROJECTED_FILES
    }
    projected = docs["qualification-input-manifest.json"]
    projected["predicate_contract_head"] = docs["predicate-contract-receipt.json"]["subject_head"]
    projected["execution_subject_head"] = docs["execution-v3-receipt.json"]["subject_head"]
    projected["observation_sha256"] = docs["execution-v3-receipt.json"]["observation_sha256"]
    projected["seal_chain_commitment_sha256"] = docs["observation-seal-v3.json"]["chain_commitment_sha256"]
    projected["comparison_result"] = docs["comparison-only-qualification-receipt.json"]["comparison_result"]
    projected["files"] = [
        {
            "basename": name,
            "byte_length": len(raw4[name]),
            "sha256": digest(raw4[name]),
        }
        for name in PROJECTED_FILES
    ]

    extraction = docs["authority-receipt-extraction.json"]
    extraction["predicate_receipt_sha256"] = digest(raw4["predicate-contract-receipt.json"])
    extraction["execution_receipt_sha256"] = digest(raw4["execution-v3-receipt.json"])
    extraction["seal_receipt_sha256"] = digest(raw4["observation-seal-v3.json"])

    assembly = docs["qualification-assembly-receipt.json"]
    assembly["full_comparison_sha256"] = docs["comparison-only-qualification-receipt.json"]["full_comparison_sha256"]
    assembly["predicate_receipt_sha256"] = digest(raw4["predicate-contract-receipt.json"])
    assembly["execution_receipt_sha256"] = digest(raw4["execution-v3-receipt.json"])
    assembly["seal_receipt_sha256"] = digest(raw4["observation-seal-v3.json"])
    assembly["authority_extraction_receipt_sha256"] = digest(json_bytes(extraction))
    assembly["sealed_manifest_replay_receipt_sha256"] = digest(
        json_bytes(docs["sealed-manifest-replay-receipt.json"])
    )

    comparison = docs["comparison-only-qualification-receipt.json"]
    qualification = docs["qualification-only.json"]
    for key in ("comparison_result", "predicate_count", "passed_count", "failed_count", "failed_predicate_ids"):
        qualification[key] = copy.deepcopy(comparison[key])
    qualification["qualification_completed"] = True
    qualification["rel_005a_qualified"] = comparison["comparison_result"] == "ALL_PREDICATES_PASS"
    qualification["scientific_pass"] = comparison["comparison_result"] == "ALL_PREDICATES_PASS"
    qualification["scientific_fail"] = comparison["comparison_result"] == "PREDICATE_FAILURES"


def build_capsule_bytes(
    docs: dict[str, dict[str, Any]], *, rebind_dependencies: bool = False
) -> bytes:
    req(set(docs) == INPUT, "synthetic input census")
    if rebind_dependencies:
        rebind_synthetic_dependencies(docs)
    raw = {name: json_bytes(docs[name]) for name in INPUT}
    raw[MANIFEST] = json_bytes(make_capsule_manifest(raw))
    return canonical_tar_bytes(raw)


def must_reject(data: bytes, expected_fragment: str) -> None:
    try:
        verify_bytes(data)
    except ValueError as exc:
        req(expected_fragment in str(exc), f"wrong rejection: {exc!s}")
    else:
        raise ValueError(f"mutation accepted: expected {expected_fragment}")


def self_test() -> dict[str, Any]:
    base_docs = synthetic_docs("ALL_PREDICATES_PASS")
    valid = build_capsule_bytes(base_docs)
    receipt = verify_bytes(valid)
    req(receipt["qualification_mapping_consistent"] is True, "valid synthetic capsule failed")

    must_reject(valid + b"\x00", "exact canonical USTAR")

    changed = copy.deepcopy(base_docs)
    capsule = build_capsule_bytes(changed)
    raw = read_tar_bytes(capsule)
    manifest = parse(raw[MANIFEST], MANIFEST)
    manifest["canonicalization"]["mtime"] = 1
    raw[MANIFEST] = json_bytes(manifest)
    must_reject(canonical_tar_bytes(raw), "canonicalization mismatch")

    changed = copy.deepcopy(base_docs)
    changed["predicate-contract-receipt.json"]["claims"]["scientific_pass"] = True
    must_reject(build_capsule_bytes(changed, rebind_dependencies=True), "predicate: overclaim")

    changed = synthetic_docs("PREDICATE_FAILURES")
    changed["comparison-only-qualification-receipt.json"]["failed_predicate_ids"] = ["REL005A-P999"]
    must_reject(build_capsule_bytes(changed, rebind_dependencies=True), "frozen namespace")

    changed = copy.deepcopy(base_docs)
    changed["comparison-only-qualification-receipt.json"]["threshold_table"] = {"POSITIVE_TOL": 0.00001}
    must_reject(build_capsule_bytes(changed, rebind_dependencies=True), "forbidden detailed keys leaked")

    changed = copy.deepcopy(base_docs)
    changed["qualification-input-manifest.json"]["files"][0]["sha256"] = "00" * 32
    must_reject(build_capsule_bytes(changed), "projected hash mismatch")

    changed = copy.deepcopy(base_docs)
    changed["qualification-assembly-receipt.json"]["authority_extraction_receipt_sha256"] = "00" * 32
    must_reject(build_capsule_bytes(changed), "assembly extraction receipt hash")

    return {
        "schema": "symthaea.rel.offline-qualification-capsule-verification-self-test.v3",
        "authority": "OfflineVerificationOnly",
        "valid_capsule_accepted": True,
        "adversarial_mutations_rejected": 7,
        "canonical_ustar_mutation_rejected": True,
        "canonicalization_declaration_mutation_rejected": True,
        "upstream_overclaim_rejected": True,
        "failed_predicate_namespace_mutation_rejected": True,
        "metric_leakage_mutation_rejected": True,
        "projected_manifest_tamper_rejected": True,
        "assembly_audit_hash_tamper_rejected": True,
        "claims": {
            "independent_replication_established": False,
            "predicate_sufficiency_established": False,
            "scientific_truth_established": False,
            "sigstore_provenance_established": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capsule", type=pathlib.Path, nargs="?")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return
    req(args.capsule is not None, "capsule path required unless --self-test")
    print(json.dumps(verify(args.capsule), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
