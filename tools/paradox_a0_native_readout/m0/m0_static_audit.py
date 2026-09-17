#!/usr/bin/env python3
"""Static audit for PARADOX-A0R M0. Standard library only; no production execution."""

from __future__ import annotations

import hashlib
import json
import struct
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "m0_static_contract.json"

EXPECTED_A0 = "8cc2651576f0068a1ccac3ef21c8a0a0eb3c2afb"
EXPECTED_G2B = "09d83a1d1fddbbbd30e4eba7cc95946c8eab871f"
EXPECTED_PRODUCTION = "eb73527d05a913e79d1f05135ad6b06c1da8e2ee"


class AuditFailure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def _check_value(value):
    if isinstance(value, str):
        require(unicodedata.normalize("NFC", value) == value, "non-NFC string")
    elif isinstance(value, float):
        raise AuditFailure("floating-point number in scientific commitment object")
    elif isinstance(value, list):
        for item in value:
            _check_value(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            require(isinstance(key, str), "non-string JSON key")
            require(unicodedata.normalize("NFC", key) == key, "non-NFC JSON key")
            _check_value(item)
    elif value is not None and not isinstance(value, (bool, int)):
        raise AuditFailure(f"unsupported canonical value: {type(value)!r}")


def canonical_json(value) -> bytes:
    _check_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def frame(payload: bytes) -> bytes:
    return struct.pack(">Q", len(payload)) + payload


def object_commit(domain: str, value) -> str:
    payload = frame(domain.encode("utf-8")) + frame(canonical_json(value))
    return hashlib.sha256(payload).hexdigest()


def verify_object_commit(domain: str, value, declared: str) -> None:
    require(object_commit(domain, value) == declared, "commitment derivation mismatch")


def row_commit(metadata, feature_bundle_digest_hex: str) -> str:
    require(len(feature_bundle_digest_hex) == 64, "feature digest width")
    try:
        feature_digest = bytes.fromhex(feature_bundle_digest_hex)
    except ValueError as exc:
        raise AuditFailure("feature digest encoding") from exc
    domain = b"SYMT-PARADOX-A0R-M0-ROW-V1"
    payload = frame(domain) + frame(canonical_json(metadata)) + frame(feature_digest)
    return hashlib.sha256(payload).hexdigest()


def dataset_commit(header, ordered_rows) -> str:
    ids = [row_id for row_id, _ in ordered_rows]
    require(len(ids) == len(set(ids)), "duplicate opaque measurement id")
    row_bytes = b""
    for _, digest in ordered_rows:
        require(len(digest) == 64, "row digest width")
        row_bytes += bytes.fromhex(digest)
    domain = b"SYMT-PARADOX-A0R-M0-DATASET-V1"
    payload = frame(domain) + frame(canonical_json(header)) + frame(row_bytes)
    return hashlib.sha256(payload).hexdigest()


def evidence_binding_commit(dataset_sha, fixture_sha, label_sha, transform_sha, split_sha) -> str:
    value = {
        "dataset_sha256": dataset_sha,
        "fixture_manifest_sha256": fixture_sha,
        "label_manifest_sha256": label_sha,
        "transform_manifest_sha256": transform_sha,
        "split_manifest_sha256": split_sha,
    }
    return object_commit("SYMT-PARADOX-A0R-M0-EVIDENCE-BINDING-V1", value)


def validate_feature_receipt(contract, receipt) -> None:
    forbidden = set(contract["feature_receipt"]["forbidden_fields"])
    require(not (forbidden & set(receipt)), "forbidden semantic/probe field in feature receipt")


def validate_split_builder_input(contract, value) -> None:
    forbidden = set(contract["planes"]["S"]["must_not_read"])
    require(not (forbidden & set(value)), "feature/probe value reached split plane")


def validate_technical_pair(a, b) -> None:
    require(a["opaque_measurement_id"] == b["opaque_measurement_id"], "technical pair identity mismatch")
    require(a["channel_digest"] == b["channel_digest"], "technical replicate mismatch")
    require(a["technical_replicate_index"] != b["technical_replicate_index"], "technical replicate index reused")


def expect_reject(name, fn) -> None:
    try:
        fn()
    except (AuditFailure, ValueError):
        return
    raise AuditFailure(f"negative control did not reject: {name}")


def main() -> int:
    raw = CONTRACT_PATH.read_bytes()
    contract = json.loads(raw)
    _check_value(contract)

    require(contract["schema_version"] == "PARADOX-A0R-M0-STATIC-CONTRACT-V1", "schema drift")
    require(contract["authority"] == "DevelopmentOnly / Representational MeasurementOnly / ProtocolHardening", "authority drift")
    ancestry = contract["ancestry"]
    require(ancestry["a0_static_subject_sha"] == EXPECTED_A0, "A0 subject drift")
    require(ancestry["g2b_subject_sha"] == EXPECTED_G2B, "G2b drift")
    require(ancestry["production_subject_sha"] == EXPECTED_PRODUCTION, "production drift")
    require(ancestry["a0_static_status_required"] == "QUALIFIED_PASS", "A0 prerequisite drift")
    qualification = ancestry["a0_static_qualification"]
    require(qualification["pull_request"] == 3652, "A0 qualifier PR drift")
    require(qualification["run_id"] == 35151014357, "A0 qualifier run drift")
    require(qualification["job_id"] == 104979041434, "A0 qualifier job drift")
    require(qualification["runner_sha"] == "49384f96d4feca186deb5bdc254c88aa6343d31a", "A0 qualifier runner drift")
    require(qualification["runner_base_sha"] == "4bad8af72ff775e7c869b6df83faba718a339a36", "A0 qualifier base drift")

    require(all(value is False for value in contract["execution_authority"].values()), "static subject gained execution authority")
    require(contract["feature_receipt"]["cfc_vector_length"] == 256, "CfC dimension drift")
    require(contract["feature_receipt"]["technical_replicates_per_measurement"] == 2, "technical repeat count drift")
    require(contract["feature_receipt"]["technical_replicates_increase_n"] is False, "technical repeats inflate n")
    require(contract["split_firewall"]["feature_values_can_affect_split"] is False, "feature-dependent split")
    require(contract["invalidity_policy"]["label_aware_retry_allowed"] is False, "label-aware retry")
    require(contract["invalidity_policy"]["all_attempts_retained"] is True, "attempt retention disabled")
    require(contract["commitment_derivation_registry"]["hardcoded_digest_only_comparison_forbidden_when_source_available"] is True, "hard-coded digest-only audit allowed")
    require(len(contract["mutation_controls"]) == 14 and len(set(contract["mutation_controls"])) == 14, "mutation-control census drift")

    f_forbidden = set(contract["planes"]["F"]["must_not_read"])
    require({"target_label", "expected_response", "oracle_output", "capability_atom_label", "score"} <= f_forbidden, "F-plane semantic firewall weakened")
    s_forbidden = set(contract["planes"]["S"]["must_not_read"])
    require({"raw_feature_bytes", "feature_vector", "feature_digest", "channel_digest", "probe_predictions"} <= s_forbidden, "S-plane feature firewall weakened")

    source = {"fixture": "opaque-1", "transform": "opaque-t0"}
    source_domain = contract["commitments"]["manifest_domains"]["fixture_manifest_sha256"]
    source_digest = object_commit(source_domain, source)
    changed_source = {"fixture": "opaque-2", "transform": "opaque-t0"}
    expect_reject("SOURCE_CHANGED_DIGEST_STALE", lambda: verify_object_commit(source_domain, changed_source, source_digest))
    expect_reject("DIGEST_CHANGED_SOURCE_STABLE", lambda: verify_object_commit(source_domain, source, "0" * 64))

    f1 = hashlib.sha256(b"feature-1").hexdigest()
    f2 = hashlib.sha256(b"feature-2").hexdigest()
    r1 = row_commit({"opaque_measurement_id": "m1"}, f1)
    r2 = row_commit({"opaque_measurement_id": "m2"}, f2)
    header = {"schema": "synthetic-v1", "row_count": 2}
    ds = dataset_commit(header, [("m1", r1), ("m2", r2)])
    require(dataset_commit(header, [("m2", r2), ("m1", r1)]) != ds, "COMMITTED_ROW_ORDER_CHANGED not detected")
    expect_reject("DUPLICATE_ROW_INSERTED", lambda: dataset_commit({"schema": "synthetic-v1", "row_count": 3}, [("m1", r1), ("m1", r1), ("m2", r2)]))
    require(dataset_commit({"schema": "synthetic-v1", "row_count": 1}, [("m1", r1)]) != ds, "ROW_DELETED not detected")

    fixture_sha = object_commit("fixture", {"f": 1})
    label_sha = object_commit("label", {"l": 1})
    transform_sha = object_commit("transform", {"t": 1})
    split_sha = object_commit("split", {"seed": "s1"})
    binding = evidence_binding_commit(ds, fixture_sha, label_sha, transform_sha, split_sha)
    swapped = evidence_binding_commit(ds, label_sha, fixture_sha, transform_sha, split_sha)
    require(binding != swapped, "MANIFEST_DATASET_CROSS_BINDING_SWAP not detected")
    require(object_commit("split", {"seed": "s2"}) != split_sha, "SPLIT_SEED_OR_TIEBREAK_CHANGED not detected")

    parsed_a = json.loads('{"a":1,\r\n"b":"x"}')
    parsed_b = json.loads(' { "b" : "x", "a" : 1 }\n')
    require(canonical_json(parsed_a) == canonical_json(parsed_b), "CANONICALIZATION_ALTERNATIVE not normalized")

    substituted_label = object_commit("label", {"l": 2})
    require(evidence_binding_commit(ds, fixture_sha, substituted_label, transform_sha, split_sha) != binding, "LABEL_OR_TRANSFORM_SUBSTITUTION_AFTER_FEATURE_SEAL not detected")
    prediction = {"id": "p1", "prediction": "sealed"}
    prediction_digest = object_commit("prediction", prediction)
    expect_reject("PREDICTION_BUNDLE_MUTATED_AFTER_SEAL", lambda: verify_object_commit("prediction", {"id": "p1", "prediction": "changed"}, prediction_digest))

    expect_reject("FORBIDDEN_LABEL_FIELD_IN_FEATURE_RECEIPT", lambda: validate_feature_receipt(contract, {"opaque_measurement_id": "m1", "target_label": 1}))
    expect_reject("FEATURE_VALUE_PRESENT_IN_SPLIT_INPUT", lambda: validate_split_builder_input(contract, {"fixture_manifest": [], "feature_digest": f1}))
    expect_reject("DUPLICATE_OPAQUE_MEASUREMENT_ID", lambda: dataset_commit(header, [("m1", r1), ("m1", r2)]))
    expect_reject(
        "TECHNICAL_REPLICATE_MISMATCH",
        lambda: validate_technical_pair(
            {"opaque_measurement_id": "m1", "technical_replicate_index": 0, "channel_digest": f1},
            {"opaque_measurement_id": "m1", "technical_replicate_index": 1, "channel_digest": f2},
        ),
    )

    result = {
        "schema": "PARADOX-A0R-M0-STATIC-AUDIT-RECEIPT-V1",
        "status": "PASS",
        "contract_sha256": hashlib.sha256(raw).hexdigest(),
        "mutation_controls_executed": 14,
        "execution_authority": "STATIC_ONLY",
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditFailure as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
