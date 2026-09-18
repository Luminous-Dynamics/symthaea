#!/usr/bin/env python3
"""Deterministic static audit for PARADOX A0-R F0 V4."""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import unicodedata

ROOT = pathlib.Path(__file__).resolve().parents[3]
HERE = pathlib.Path(__file__).resolve().parent
CONTRACT = HERE / "f0_transport_contract.json"
WORKER = ROOT / "src" / "bin" / "paradox_a0r_f0_worker.rs"
PAIR = HERE / "f0_pair_audit.py"
M0 = "306b0471a0854d2e77c1d50438fc493806fe1564"
SUPERSEDED_V3 = "1760cf4a331793814ec47552a86565d8f6163646"
CONFIG_SHA = "2cd99b06cf2cf11cfc0612c0db818355099b48b04bd687ca66a31a08680be1f9"
CARGO_BLOB = "10f004ce824af40c069cc630ad867d9d2cfabd10"
LOCK_BLOB = "1d5b6c3dffb474fabef0f8237c6741b4f252bc62"
ALLOWED_DIFF = {
    "src/bin/paradox_a0r_f0_worker.rs",
    "tools/paradox_a0_native_readout/f0/PROTOCOL.md",
    "tools/paradox_a0_native_readout/f0/f0_pair_audit.py",
    "tools/paradox_a0_native_readout/f0/f0_static_audit.py",
    "tools/paradox_a0_native_readout/f0/f0_transport_contract.json",
}


class AuditFailure(RuntimeError):
    pass


def require(condition: bool, code: str) -> None:
    if not condition:
        raise AuditFailure(code)


def git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return proc.stdout.strip()


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate contract key:{key}")
        result[key] = value
    return result


def reject_float(_: str):
    raise AuditFailure("float token in frozen contract")


def reject_constant(token: str):
    raise AuditFailure(f"nonfinite token in frozen contract:{token}")


def check_nfc(value):
    if isinstance(value, str):
        require(unicodedata.normalize("NFC", value) == value, "non-NFC contract string")
    elif isinstance(value, list):
        for item in value:
            check_nfc(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            check_nfc(key)
            check_nfc(item)


def main() -> None:
    contract = json.loads(
        CONTRACT.read_text(encoding="utf-8"),
        object_pairs_hook=strict_pairs,
        parse_float=reject_float,
        parse_constant=reject_constant,
    )
    check_nfc(contract)
    worker = WORKER.read_text(encoding="utf-8")
    pair = PAIR.read_text(encoding="utf-8")
    controls: list[str] = []

    require(contract["schema_version"] == "PARADOX-A0R-F0-TRANSPORT-CONTRACT-V4", "contract schema drift")
    require(contract["authority"] == "DevelopmentOnly / Representational MeasurementOnly / TransportOnly", "authority drift")
    require(contract["supersedes_unqualified_subject"] == SUPERSEDED_V3, "V3 supersession drift")
    require(contract["ancestry"]["m0_static_subject_sha"] == M0, "M0 ancestry drift")
    require(contract["worker"]["request_schema"] == "PARADOX-A0R-F0-WORKER-REQUEST-V4", "request schema drift")
    require(contract["worker"]["response_schema"] == "PARADOX-A0R-F0-WORKER-RESPONSE-V3", "response compatibility drift")
    controls.append("authority_lineage_and_wire_schema_bound")

    projection = contract["runner_config_projection"]["canonical_json"].encode()
    require(hashlib.sha256(projection).hexdigest() == CONFIG_SHA, "config projection derivation mismatch")
    require(contract["runner_config_projection"]["sha256"] == CONFIG_SHA, "config projection declaration mismatch")
    controls.append("config_projection_bound")

    require(git("rev-parse", "HEAD^") == M0, "F0 parent is not exact M0")
    require(git("rev-list", "--count", f"{M0}..HEAD") == "1", "F0 is not one commit")
    changed = set(git("diff", "--name-only", M0, "HEAD").splitlines())
    require(changed == ALLOWED_DIFF, "F0 five-path census mismatch")
    require("Cargo.toml" not in changed and "Cargo.lock" not in changed, "dependency manifest changed")
    controls.append("single_commit_exact_five_path_surface_no_manifest_change")

    require(blob_sha((ROOT / "Cargo.toml").read_bytes()) == CARGO_BLOB, "Cargo.toml drift")
    require(blob_sha((ROOT / "Cargo.lock").read_bytes()) == LOCK_BLOB, "Cargo.lock drift")
    for binding in contract["source_bindings"]:
        require(blob_sha((ROOT / binding["path"]).read_bytes()) == binding["blob_sha"], f"source binding mismatch:{binding['path']}")
    controls.append("production_manifest_and_lock_sources_unchanged")

    dependency = contract["dependency_plane"]
    require(dependency["target_kind"] == "auto-discovered root binary", "target kind drift")
    require(dependency["target_path"] == "src/bin/paradox_a0r_f0_worker.rs", "target path drift")
    require(dependency["cargo_toml_must_remain_unchanged"] is True, "Cargo manifest mutation allowed")
    require(dependency["cargo_lock_must_remain_unchanged"] is True, "lock mutation allowed")
    require(dependency["root_dev_dependency_graph_allowed"] is False, "dev graph allowed")
    require(dependency["new_external_dependency_allowed"] is False, "new external dependency allowed")
    controls.append("dependency_plane_isolated")

    required_worker = [
        "#[derive(Debug, Deserialize)]",
        "#[serde(deny_unknown_fields)]",
        "let wire: RequestWire = serde_json::from_str(&line)",
        "MEASUREMENT_CYCLE_NOT_FINAL",
        "event_stream_digest(&request.agent_visible_events)",
        "CognitiveLoopService::new(frozen_config())",
        "service.cycle(event)",
        "config.enable_recurrent_dim_masking = false",
        "config.enable_spectral_entropy_masking = false",
        "config.effective_dim_fraction_override = None",
        "verify_sha256_implementation()?",
        "SHA256_SELF_TEST_EMPTY_FAILED",
        "SHA256_SELF_TEST_ABC_FAILED",
        "fn sha256_digest(input: &[u8]) -> [u8; 32]",
        "PARADOX-A0R-F0-PROVENANCE-ENVELOPE-V2",
        "canonical_scientific_payload",
        "canonical_provenance_envelope",
    ]
    for fragment in required_worker:
        require(fragment in worker, f"worker gate missing:{fragment}")
    require("use sha2" not in worker, "external sha2 dependency returned")
    require("let value: serde_json::Value" not in worker, "generic Value request decoding returned")
    require(".cycle_with_hv(" not in worker, "forbidden cycle_with_hv path")
    require("std::fs" not in worker and "File::open" not in worker, "worker filesystem sidecar surface")
    require("std::env::var(" not in worker and "std::env::vars" not in worker, "worker environment-value read surface")
    controls.extend([
        "typed_duplicate_rejecting_request_decode",
        "strict_opaque_and_replica_request_types",
        "final_cycle_only",
        "ordered_event_commitment",
        "stdin_only_no_sidecar_reads",
        "normal_text_cycle_only",
        "self_contained_sha256_with_startup_vectors",
    ])

    frozen = worker[worker.index("fn frozen_config()") : worker.index("fn execute_request")]
    require(not any(token in frozen for token in ("request.", "opaque_", "technical_pair_id")), "provenance influences config")
    science_seal = worker.index("let scientific_payload_sha256 =")
    uuid_generation = worker.index("Uuid::new_v4()")
    require(uuid_generation > science_seal, "service UUID precedes scientific seal")
    science_prefix = worker[worker.index("fn execute_request") : science_seal]
    require(not any(
        token in science_prefix
        for token in (
            "opaque_measurement_id", "opaque_base_fixture_id", "opaque_transform_id",
            "technical_pair_id", "technical_replicate_index", "service_instance_id",
        )
    ), "provenance influences scientific path")
    controls.extend([
        "provenance_not_config_input",
        "science_sealed_before_service_identity",
        "provenance_not_science_input",
    ])

    required_pair = [
        'WORKER_SCHEMA = "PARADOX-A0R-F0-WORKER-RESPONSE-V3"',
        "object_pairs_hook=unique_object",
        "parse_float=reject_float_token",
        "parse_constant=reject_constant",
        "JSON_DUPLICATE_KEY",
        "JSON_FLOAT_TOKEN_FORBIDDEN",
        "JSON_NONFINITE_CONSTANT_FORBIDDEN",
        "JSON_STRING_NOT_NFC",
        "type(value) is not int",
        "type(value) is not bool",
        "type(value) is not str",
        'struct.unpack("<f"',
        "NONFINITE_COUNT_MISMATCH:RECURRENT",
        "RECURRENT_ALL_ZERO_MISMATCH",
        "CONFIG_DIAGNOSTIC_DRIFT:RECURRENT_MASK",
        "INVALIDITY_REASON_MISMATCH",
        "EXECUTION_BINDING_MISMATCH:",
        "BINDING_MEASUREMENT_CYCLE_NOT_FINAL",
        "PEER_BINDING_MISMATCH:",
        "RAW_FEATURE_BUNDLE_DIGEST_MISMATCH",
        "SCIENTIFIC_PAYLOAD_DIGEST_MISMATCH",
        "PROVENANCE_RECEIPT_DIGEST_MISMATCH",
        '"statistical_n_increment": 0',
    ]
    for fragment in required_pair:
        require(fragment in pair, f"pair verifier gate missing:{fragment}")
    compile(pair, str(PAIR), "exec")
    controls.extend([
        "strict_json_duplicate_and_numeric_token_rejection",
        "strict_json_exact_types_and_nfc",
        "independent_f32_diagnostic_reconstruction",
        "independent_config_and_validity_reconstruction",
        "external_execution_binding_gate",
        "strict_pair_invariant_provenance",
        "raw_science_provenance_commitments_recomputed",
        "technical_repeats_n_zero",
    ])

    pair_fields = contract["technical_pair_receipt"]["peer_invariant_provenance_fields"]
    pair_source = pair[pair.index("PAIR_INVARIANT_FIELDS") : pair.index("FORBIDDEN_KEY_FRAGMENTS")]
    for field in pair_fields:
        require(f'"{field}"' in pair_source, f"pair invariant field absent:{field}")
    binding_fields = contract["execution_binding"]["required_fields"]
    binding_source = pair[pair.index("BINDING_KEYS") : pair.index("PAIR_INVARIANT_FIELDS")]
    for field in binding_fields:
        require(f'"{field}"' in binding_source, f"execution binding field absent:{field}")
    controls.append("contract_and_verifier_field_sets_match")

    runtime_controls = contract["qualification_runtime_controls"]
    require(len(runtime_controls) == 34 and len(set(runtime_controls)) == 34, "runtime control census drift")
    controls.append("thirty_four_runtime_controls_frozen")

    receipt = {
        "schema": "PARADOX-A0R-F0-V4-STATIC-AUDIT-RECEIPT-V1",
        "status": "PASS",
        "authority": "STATIC_ONLY",
        "m0_parent_sha": M0,
        "f0_head_sha": git("rev-parse", "HEAD"),
        "supersedes_unqualified_subject": SUPERSEDED_V3,
        "changed_paths": sorted(changed),
        "cargo_toml_blob_sha": CARGO_BLOB,
        "cargo_lock_blob_sha": LOCK_BLOB,
        "config_projection_sha256": CONFIG_SHA,
        "source_bindings_checked": len(contract["source_bindings"]),
        "static_controls_executed": len(controls),
        "runtime_controls_preregistered": len(runtime_controls),
        "claim_ceiling": "F0_IMPLEMENTED_UNQUALIFIED only; no dataset, decodability, behavior, causal-use, metacognition, ontology-repair, consciousness, sentience or phenomenology claim",
    }
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
