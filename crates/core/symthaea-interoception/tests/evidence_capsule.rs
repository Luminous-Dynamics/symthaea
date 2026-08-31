use symthaea_interoception::{
    ArtifactDigest, EvidenceCapsuleManifest, ForecastBasisId, EVIDENCE_CAPSULE_SCHEMA_VERSION,
    INTEROCEPTIVE_MODEL_SEMANTICS_VERSION, INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION,
};

fn digest(ch: char) -> String {
    std::iter::repeat_n(ch, 64).collect()
}

fn valid_manifest() -> EvidenceCapsuleManifest {
    EvidenceCapsuleManifest {
        schema_version: EVIDENCE_CAPSULE_SCHEMA_VERSION,
        model_semantics_version: INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
        source_commit: "0123456789abcdef0123456789abcdef01234567".into(),
        cargo_lock_sha256: digest('a'),
        flake_lock_sha256: Some(digest('b')),
        rust_toolchain_sha256: Some(digest('c')),
        rustc_vv: "rustc 1.96.0\nhost: x86_64-unknown-linux-gnu".into(),
        cargo_vv: "cargo 1.96.0".into(),
        target_triple: "x86_64-unknown-linux-gnu".into(),
        architecture: "x86_64".into(),
        experiment_id: "native-interoception-v0.1-smoke".into(),
        preregistration_sha256: digest('0'),
        forecast_basis: ForecastBasisId::DynamicsAwareConstantDrive,
        experiment_config_sha256: digest('d'),
        input_sequence_sha256: digest('e'),
        snapshot_schema_version: INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION,
        evidence_plane_sha256: digest('f'),
        artifacts: vec![ArtifactDigest::new("snapshots.jsonl", digest('1'))],
    }
}

#[test]
fn valid_evidence_capsule_passes_validation_and_round_trip() {
    let manifest = valid_manifest();
    manifest.validate().expect("valid evidence manifest");

    let encoded = serde_json::to_vec(&manifest).expect("serialize evidence manifest");
    let decoded: EvidenceCapsuleManifest =
        serde_json::from_slice(&encoded).expect("deserialize evidence manifest");
    assert_eq!(decoded, manifest);
}

#[test]
fn evidence_capsule_rejects_ambiguous_or_malformed_identity() {
    let mut manifest = valid_manifest();
    manifest.source_commit = "NOT-A-GIT-SHA".into();
    manifest.cargo_lock_sha256 = "ABCDEF".into();
    manifest.preregistration_sha256 = "not-a-digest".into();
    manifest
        .artifacts
        .push(ArtifactDigest::new("snapshots.jsonl", digest('2')));

    let errors = manifest
        .validate()
        .expect_err("manifest must fail validation");
    assert!(errors.iter().any(|error| error.contains("source_commit")));
    assert!(errors
        .iter()
        .any(|error| error.contains("cargo_lock_sha256")));
    assert!(errors
        .iter()
        .any(|error| error.contains("preregistration_sha256")));
    assert!(errors
        .iter()
        .any(|error| error.contains("duplicate artifact name")));
}

#[test]
fn evidence_capsule_requires_exact_snapshot_and_semantics_versions() {
    let mut manifest = valid_manifest();
    manifest.snapshot_schema_version = INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION + 1;
    manifest.model_semantics_version = INTEROCEPTIVE_MODEL_SEMANTICS_VERSION + 1;

    let errors = manifest.validate().expect_err("version mismatch must fail");
    assert!(errors
        .iter()
        .any(|error| error.contains("snapshot schema version mismatch")));
    assert!(errors
        .iter()
        .any(|error| error.contains("model semantics version mismatch")));
}
