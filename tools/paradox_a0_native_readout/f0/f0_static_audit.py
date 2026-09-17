#!/usr/bin/env python3
"""Deterministic static audit for PARADOX A0-R F0 V2."""
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
SUPERSEDED_V1 = "876ff2d308fd10ed0873f912079e2aa02a58c834"
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
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout.strip()


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(data)).encode() + b"\0" + data
    ).hexdigest()


def main() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    worker = WORKER.read_text(encoding="utf-8")
    pair = PAIR.read_text(encoding="utf-8")
    cargo = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    controls: list[str] = []

    if contract["schema_version"] != "PARADOX-A0R-F0-TRANSPORT-CONTRACT-V2":
        fail("CONTRACT_SCHEMA_MISMATCH")
    if contract["authority"] != "DevelopmentOnly / Representational MeasurementOnly / TransportOnly":
        fail("AUTHORITY_MISMATCH")
    if contract["supersedes_unqualified_subject"] != SUPERSEDED_V1:
        fail("SUPERSEDED_V1_BINDING_MISMATCH")
    if contract["ancestry"]["m0_static_subject_sha"] != M0:
        fail("M0_BINDING_MISMATCH")
    controls.append("authority_and_lineage_bound")

    projection = contract["runner_config_projection"]["canonical_json"].encode()
    if hashlib.sha256(projection).hexdigest() != CONFIG_SHA:
        fail("CONFIG_PROJECTION_DIGEST_MISMATCH")
    if contract["runner_config_projection"]["sha256"] != CONFIG_SHA:
        fail("CONFIG_PROJECTION_DECLARATION_MISMATCH")
    controls.append("config_projection_bound")

    if git("rev-parse", "HEAD^") != M0:
        fail("F0_PARENT_NOT_EXACT_M0")
    changed = set(git("diff", "--name-only", M0, "HEAD").splitlines())
    if changed != ALLOWED_DIFF:
        fail("F0_SOURCE_CENSUS_MISMATCH")
    if git("rev-list", "--count", f"{M0}..HEAD") != "1":
        fail("F0_NOT_SINGLE_COMMIT")
    controls.append("single_commit_exact_six_path_surface")

    for binding in contract["source_bindings"]:
        if blob_sha((ROOT / binding["path"]).read_bytes()) != binding["blob_sha"]:
            fail(f"BOUND_SOURCE_BYTES_MISMATCH:{binding['path']}")
    if blob_sha((ROOT / "Cargo.lock").read_bytes()) != LOCK_BLOB:
        fail("CARGO_LOCK_DRIFT")
    controls.append("production_sources_and_lock_unchanged")

    if "sha2 = { workspace = true }" not in cargo:
        fail("SHA2_DEV_DEPENDENCY_MISSING")
    if 'name = "paradox_a0r_f0_worker"' not in cargo:
        fail("F0_EXAMPLE_REGISTRATION_MISSING")
    controls.append("research_target_registered")

    required_worker = [
        'const REQUEST_SCHEMA: &str = "PARADOX-A0R-F0-WORKER-REQUEST-V2"',
        "reject_forbidden_keys(&value)?",
        'require_opaque_id(obj, "opaque_measurement_id")?',
        'require_opaque_id(obj, "opaque_base_fixture_id")?',
        'require_opaque_id(obj, "opaque_transform_id")?',
        'require_opaque_id(obj, "technical_pair_id")?',
        "if technical_replicate_index > 1",
        "CognitiveLoopService::new(frozen_config())",
        "service.cycle(event)",
        "config.enable_recurrent_dim_masking = false",
        "config.enable_spectral_entropy_masking = false",
        "config.effective_dim_fraction_override = None",
        "canonical_scientific_payload",
        "canonical_provenance_envelope",
    ]
    for fragment in required_worker:
        if fragment not in worker:
            fail(f"WORKER_REQUIRED_FRAGMENT_MISSING:{fragment}")
    if ".cycle_with_hv(" in worker:
        fail("HV_CYCLE_PATH_FORBIDDEN")
    if "std::fs" in worker or "File::open" in worker:
        fail("WORKER_FILESYSTEM_READ_SURFACE")
    if "std::env::var(" in worker or "std::env::vars" in worker:
        fail("WORKER_ENVIRONMENT_READ_SURFACE")
    controls.extend([
        "stdin_only_no_sidecar_surface",
        "recursive_semantic_key_rejection",
        "strict_four_opaque_ids",
        "replicate_index_0_or_1_only",
        "fresh_service_per_request",
        "normal_text_cycle_only",
        "no_filesystem_sidecars",
        "no_environment_value_reads",
    ])

    frozen = worker[worker.index("fn frozen_config()") : worker.index("fn execute_request")]
    if any(token in frozen for token in ("request.", "opaque_", "technical_pair_id")):
        fail("PROVENANCE_CONFIG_INFLUENCE")
    science_seal = worker.index("let scientific_payload_sha256 =")
    uuid_generation = worker.index("Uuid::new_v4()")
    if uuid_generation <= science_seal:
        fail("SERVICE_ID_PRECEDES_SCIENCE_SEAL")
    science_prefix = worker[worker.index("fn execute_request") : science_seal]
    if any(
        token in science_prefix
        for token in (
            "opaque_measurement_id",
            "opaque_base_fixture_id",
            "opaque_transform_id",
            "technical_pair_id",
            "technical_replicate_index",
            "service_instance_id",
        )
    ):
        fail("PROVENANCE_INFLUENCES_SCIENTIFIC_PATH")
    controls.extend([
        "provenance_not_config_input",
        "science_sealed_before_service_identity",
        "provenance_not_science_input",
    ])

    required_pair = [
        'struct.unpack("<f"',
        "NONFINITE_COUNT_MISMATCH:RECURRENT",
        "NONFINITE_COUNT_MISMATCH:THOUGHT",
        "RECURRENT_ALL_ZERO_MISMATCH",
        "CONFIG_DIAGNOSTIC_DRIFT:RECURRENT_MASK",
        "CONFIG_DIAGNOSTIC_DRIFT:SPECTRAL_MASK",
        "CONFIG_DIAGNOSTIC_DRIFT:EFFECTIVE_DIM_OVERRIDE",
        "INVALIDITY_REASON_MISMATCH",
        "MEASUREMENT_VALIDITY_MISMATCH",
        "FIXED_PROVENANCE_MISMATCH:PRODUCTION",
        "FIXED_PROVENANCE_MISMATCH:G2B",
        "FIXED_PROVENANCE_MISMATCH:A0",
        "FIXED_PROVENANCE_MISMATCH:M0",
        "REPLICATE_INDEX_INVALID",
        "SERVICE_INSTANCE_REUSE",
        "PEER_BINDING_MISMATCH:",
        "RAW_FEATURE_BUNDLE_DIGEST_MISMATCH",
        "SCIENTIFIC_PAYLOAD_DIGEST_MISMATCH",
        "PROVENANCE_RECEIPT_DIGEST_MISMATCH",
        '"statistical_n_increment": 0',
    ]
    for fragment in required_pair:
        if fragment not in pair:
            fail(f"PAIR_AUDITOR_GATE_MISSING:{fragment}")
    controls.extend([
        "independent_f32_diagnostic_reconstruction",
        "independent_config_diagnostic_gate",
        "independent_validity_reconstruction",
        "fixed_lineage_provenance_gate",
        "strict_pair_invariant_provenance",
        "raw_science_provenance_commitments_recomputed",
        "technical_repeats_n_zero",
    ])

    peer_fields = contract["technical_pair_receipt"]["peer_invariant_provenance_fields"]
    pair_source = pair[pair.index("PAIR_INVARIANT_FIELDS") : pair.index("FORBIDDEN_KEY_FRAGMENTS")]
    for field in peer_fields:
        if f'"{field}"' not in pair_source:
            fail(f"PAIR_INVARIANT_FIELD_MISSING:{field}")
    controls.append("contract_pair_invariants_match_auditor")

    runtime_controls = contract["qualification_runtime_controls"]
    if len(runtime_controls) != 22 or len(set(runtime_controls)) != 22:
        fail("RUNTIME_CONTROL_CENSUS_DRIFT")
    controls.append("twenty_two_runtime_controls_frozen")

    print(
        json.dumps(
            {
                "schema": "PARADOX-A0R-F0-V2-STATIC-AUDIT-RECEIPT-V1",
                "status": "PASS",
                "authority": "STATIC_ONLY",
                "m0_parent_sha": M0,
                "f0_head_sha": git("rev-parse", "HEAD"),
                "supersedes_unqualified_subject": SUPERSEDED_V1,
                "changed_paths": sorted(changed),
                "config_projection_sha256": CONFIG_SHA,
                "source_bindings_checked": len(contract["source_bindings"]),
                "static_controls_executed": len(controls),
                "runtime_controls_preregistered": len(runtime_controls),
                "claim_ceiling": (
                    "F0_IMPLEMENTED_UNQUALIFIED only; no dataset collection, "
                    "decodability, behavior, causal-use, metacognition, ontology-repair, "
                    "consciousness, sentience or phenomenology claim"
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
