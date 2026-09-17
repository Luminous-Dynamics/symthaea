#!/usr/bin/env python3
"""Deterministic static audit for the frozen PARADOX A0-R F0 transport subject."""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[3]
HERE = pathlib.Path(__file__).resolve().parent
CONTRACT = HERE / "f0_transport_contract.json"
WORKER = ROOT / "examples" / "paradox_a0r_f0_worker.rs"
PAIR = HERE / "f0_pair_audit.py"
M0 = "306b0471a0854d2e77c1d50438fc493806fe1564"
CONFIG_SHA = "2cd99b06cf2cf11cfc0612c0db818355099b48b04bd687ca66a31a08680be1f9"
LOCK_BLOB = "1d5b6c3dffb474fabef0f8237c6741b4f252bc62"
ALLOWED_DIFF = {
    "Cargo.toml",
    "examples/paradox_a0r_f0_worker.rs",
    "tools/paradox_a0_native_readout/f0/PROTOCOL.md",
    "tools/paradox_a0_native_readout/f0/f0_pair_audit.py",
    "tools/paradox_a0_native_readout/f0/f0_static_audit.py",
    "tools/paradox_a0_native_readout/f0/f0_transport_contract.json",
}


def fail(code: str) -> None:
    raise SystemExit(code)


def git(*args: str) -> str:
    p = subprocess.run(["git", *args], cwd=ROOT, check=True, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.stdout.strip()


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def frame(name: str, value: bytes) -> bytes:
    key = name.encode()
    return len(key).to_bytes(4, "little") + key + len(value).to_bytes(8, "little") + value


def domain_hash(domain: bytes, payload: bytes) -> str:
    h = hashlib.sha256()
    h.update(len(domain).to_bytes(8, "little"))
    h.update(domain)
    h.update(len(payload).to_bytes(8, "little"))
    h.update(payload)
    return h.hexdigest()


def synthetic_provenance(science_hash: str, ids: tuple[str, str, str, str],
                         replicate: int, service: str) -> bytes:
    measurement, base, transform, pair = ids
    values = [
        ("schema", "PARADOX-A0R-F0-PROVENANCE-ENVELOPE-V1"),
        ("production_subject_sha", "e" * 40),
        ("g2b_subject_sha", "0" * 40),
        ("a0_subject_sha", "8" * 40),
        ("m0_subject_sha", M0),
        ("f0_subject_sha", "f" * 40),
        ("opaque_measurement_id", measurement),
        ("opaque_base_fixture_id", base),
        ("opaque_transform_id", transform),
        ("technical_pair_id", pair),
        ("service_instance_id", service),
        ("runner_config_projection_sha256", CONFIG_SHA),
        ("executable_sha256", "1" * 64),
        ("environment_capsule_sha256", "2" * 64),
        ("source_binding_sha256", "3" * 64),
        ("scientific_payload_sha256", science_hash),
    ]
    out = b"".join(frame(k, v.encode()) for k, v in values)
    return out + frame("technical_replicate_index", replicate.to_bytes(8, "little"))


def main() -> None:
    c = json.loads(CONTRACT.read_text())
    worker = WORKER.read_text()
    pair = PAIR.read_text()
    cargo = (ROOT / "Cargo.toml").read_text()
    controls: list[str] = []

    if c["schema_version"] != "PARADOX-A0R-F0-TRANSPORT-CONTRACT-V1":
        fail("CONTRACT_SCHEMA_MISMATCH")
    if c["authority"] != "DevelopmentOnly / Representational MeasurementOnly / TransportOnly":
        fail("AUTHORITY_MISMATCH")
    if c["ancestry"]["m0_static_subject_sha"] != M0:
        fail("M0_BINDING_MISMATCH")
    if c["ancestry"]["m0_qualification"]["status"] != "QUALIFIED_PASS":
        fail("M0_NOT_QUALIFIED")
    controls.append("qualified_m0_bound")

    projection = c["runner_config_projection"]["canonical_json"].encode()
    if hashlib.sha256(projection).hexdigest() != CONFIG_SHA:
        fail("CONFIG_PROJECTION_DIGEST_MISMATCH")
    if c["runner_config_projection"]["sha256"] != CONFIG_SHA:
        fail("CONFIG_PROJECTION_DECLARATION_MISMATCH")
    controls.append("config_projection_bound")

    if git("rev-parse", "HEAD^") != M0:
        fail("F0_PARENT_NOT_EXACT_M0")
    changed = set(git("diff", "--name-only", M0, "HEAD").splitlines())
    if changed != ALLOWED_DIFF:
        fail("F0_SOURCE_CENSUS_MISMATCH")
    controls.append("exact_six_path_surface")

    for binding in c["source_bindings"]:
        if blob_sha((ROOT / binding["path"]).read_bytes()) != binding["blob_sha"]:
            fail("BOUND_SOURCE_BYTES_MISMATCH:" + binding["path"])
    if blob_sha((ROOT / "Cargo.lock").read_bytes()) != LOCK_BLOB:
        fail("CARGO_LOCK_DRIFT")
    controls.append("frozen_public_source_and_lock_bytes")

    if "sha2 = { workspace = true }" not in cargo:
        fail("SHA2_DEV_DEPENDENCY_MISSING")
    if 'name = "paradox_a0r_f0_worker"' not in cargo:
        fail("F0_EXAMPLE_REGISTRATION_MISSING")
    controls.append("research_target_registered")

    must_have = [
        "std::env::args_os().len() != 1",
        "reject_forbidden_keys(&value)?",
        "CognitiveLoopService::new(frozen_config())",
        "service.cycle(event)",
        "config.genesis_phrase = Some(GENESIS.to_owned())",
        "config.enable_recurrent_dim_masking = false",
        "config.enable_spectral_entropy_masking = false",
        "config.effective_dim_fraction_override = None",
        "canonical_scientific_payload",
        "canonical_provenance_envelope",
        "technical_pair_id",
    ]
    for fragment in must_have:
        if fragment not in worker:
            fail("WORKER_REQUIRED_FRAGMENT_MISSING:" + fragment)
    if ".cycle_with_hv(" in worker:
        fail("HV_CYCLE_PATH_FORBIDDEN")
    if "std::fs" in worker or "File::open" in worker:
        fail("WORKER_FILESYSTEM_SIDECHANNEL")
    if "std::env::vars" in worker or "std::env::var(" in worker:
        fail("WORKER_ENVIRONMENT_READ_SURFACE")
    controls.extend([
        "stdin_request_surface_only",
        "recursive_semantic_key_rejection",
        "fresh_service_per_request",
        "normal_text_cycle_only",
        "worker_does_not_read_sidecar_files",
        "worker_does_not_read_environment_values",
    ])

    frozen = worker[worker.index("fn frozen_config()") : worker.index("fn execute_request")]
    if "request." in frozen or "opaque_" in frozen or "technical_pair_id" in frozen:
        fail("OPAQUE_ID_CONFIG_INFLUENCE")
    prefix = worker[worker.index("fn execute_request") : worker.index("let scientific_payload_bytes")]
    if any(term in prefix for term in (
        "opaque_measurement_id", "opaque_base_fixture_id",
        "opaque_transform_id", "technical_pair_id"
    )):
        fail("OPAQUE_ID_SCIENTIFIC_PATH_INFLUENCE")
    controls.extend(["opaque_ids_not_config_inputs", "opaque_ids_not_science_inputs"])

    if set(c["scientific_payload"]["must_not_contain"]) & set(c["scientific_payload"]["fields"]):
        fail("SCIENCE_PROVENANCE_SCHEMA_COLLISION")
    controls.append("science_provenance_schema_separated")

    science = domain_hash(b"PARADOX-A0R-F0-SCIENTIFIC-V1", b"fixed-scientific-bytes")
    p_a = synthetic_provenance(science, ("m1", "b1", "t1", "p1"), 0, "s1")
    p_b = synthetic_provenance(science, ("m2", "b2", "t2", "p2"), 0, "s1")
    if domain_hash(b"PARADOX-A0R-F0-RECEIPT-V1", p_a) == domain_hash(
        b"PARADOX-A0R-F0-RECEIPT-V1", p_b
    ):
        fail("ID_RENAME_PROVENANCE_INVARIANT")
    controls.append("id_rename_changes_provenance_not_science")

    p0 = synthetic_provenance(science, ("m", "b", "t", "pair"), 0, "service-A")
    p1 = synthetic_provenance(science, ("m", "b", "t", "pair"), 1, "service-B")
    if domain_hash(b"PARADOX-A0R-F0-RECEIPT-V1", p0) == domain_hash(
        b"PARADOX-A0R-F0-RECEIPT-V1", p1
    ):
        fail("REPLICATE_PROVENANCE_NOT_DISTINCT")
    controls.append("technical_replicates_provenance_distinct")

    pair_need = [
        "RAW_FEATURE_BUNDLE_DIGEST_MISMATCH",
        "SCIENTIFIC_PAYLOAD_DIGEST_MISMATCH",
        "PROVENANCE_RECEIPT_DIGEST_MISMATCH",
        "REPLICATE_INDEX_SET_INVALID",
        "SERVICE_INSTANCE_REUSE",
        "SCIENTIFIC_PAYLOAD_MISMATCH",
        "INVALID_MEASUREMENT_NONDETERMINISTIC",
        '"statistical_n_increment": 0',
    ]
    for fragment in pair_need:
        if fragment not in pair:
            fail("PAIR_AUDITOR_GATE_MISSING:" + fragment)
    controls.extend([
        "pair_recomputes_raw_commitment",
        "pair_recomputes_science_commitment",
        "pair_recomputes_provenance_commitment",
        "pair_requires_indices_0_1",
        "pair_rejects_service_reuse",
        "pair_rejects_science_mismatch",
        "technical_repeats_n_zero",
    ])

    negative_controls = [
        "top_level_semantic_field", "nested_semantic_field",
        "semantic_sidecar_argument", "semantic_environment_exposure",
        "condition_specific_branch_surface", "opaque_id_as_seed_or_config",
        "opaque_id_as_cycle_selector", "plan_order_row_independence",
        "opaque_id_rename_noninterference", "semantic_sidecar_substitution",
        "split_membership_substitution", "technical_service_reuse",
        "technical_scientific_byte_mismatch", "recurrent_dimension_drift",
        "nonfinite_recurrent_value", "masking_config_drift",
        "postseal_feature_mutation", "unretained_failed_attempt",
        "label_aware_retry", "unbound_execution_identity",
        "cycle_with_hv_path_substitution", "external_text_preencoding_substitution",
    ]
    if len(negative_controls) != 22:
        fail("NEGATIVE_CONTROL_CENSUS_DRIFT")

    print(json.dumps({
        "schema": "PARADOX-A0R-F0-STATIC-AUDIT-RECEIPT-V1",
        "status": "PASS",
        "authority": "STATIC_ONLY",
        "m0_parent_sha": M0,
        "f0_head_sha": git("rev-parse", "HEAD"),
        "changed_paths": sorted(changed),
        "config_projection_sha256": CONFIG_SHA,
        "source_bindings_checked": len(c["source_bindings"]),
        "static_controls_executed": len(controls),
        "negative_controls_preregistered": len(negative_controls),
        "negative_control_ids": negative_controls,
        "claim_ceiling": "F0_IMPLEMENTED_UNQUALIFIED; no feature-acquisition, decodability, behavior, causal use, metacognition, ontology repair, consciousness, sentience or phenomenology claim",
    }, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
