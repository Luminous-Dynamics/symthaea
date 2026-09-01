use symthaea_interoception::{
    ArtifactDigest, EvidenceCapsuleManifest, ForecastBasisId, GateStatus,
    QualificationEvidenceBundle, QualificationGateReceipt, QualificationReceipt,
    EVIDENCE_CAPSULE_SCHEMA_VERSION, INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
    INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION, QUALIFICATION_EVIDENCE_BUNDLE_SCHEMA_VERSION,
    QUALIFICATION_RECEIPT_SCHEMA_VERSION, REQUIRED_QUALIFICATION_GATES,
};

const SOURCE_A: &str = "0123456789abcdef0123456789abcdef01234567";
const SOURCE_B: &str = "89abcdef0123456789abcdef0123456789abcdef";

fn digest(ch: char) -> String {
    std::iter::repeat_n(ch, 64).collect()
}

fn qualification(source_commit: &str, status: GateStatus) -> QualificationReceipt {
    QualificationReceipt {
        schema_version: QUALIFICATION_RECEIPT_SCHEMA_VERSION,
        model_semantics_version: INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
        source_commit: source_commit.into(),
        gates: REQUIRED_QUALIFICATION_GATES
            .iter()
            .map(|gate| QualificationGateReceipt::new(*gate, status, format!("evidence:{gate}")))
            .collect(),
    }
}

fn evidence(source_commit: &str) -> EvidenceCapsuleManifest {
    EvidenceCapsuleManifest {
        schema_version: EVIDENCE_CAPSULE_SCHEMA_VERSION,
        model_semantics_version: INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
        source_commit: source_commit.into(),
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

fn bundle(source_commit: &str, status: GateStatus) -> QualificationEvidenceBundle {
    QualificationEvidenceBundle {
        schema_version: QUALIFICATION_EVIDENCE_BUNDLE_SCHEMA_VERSION,
        model_semantics_version: INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
        source_commit: source_commit.into(),
        qualification: qualification(source_commit, status),
        evidence: evidence(source_commit),
    }
}

#[test]
fn matching_passed_qualification_and_evidence_form_one_qualified_lineage() {
    let bundle = bundle(SOURCE_A, GateStatus::Passed);
    bundle.validate().expect("matching bundle must validate");
    assert!(bundle.is_qualified());

    let digest = bundle.sha256().expect("bundle digest");
    assert_eq!(digest.len(), 64);

    let encoded = bundle.canonical_json().expect("canonical bundle json");
    let decoded: QualificationEvidenceBundle =
        serde_json::from_slice(&encoded).expect("deserialize qualification bundle");
    assert_eq!(decoded, bundle);
    assert_eq!(decoded.sha256().expect("decoded digest"), digest);
}

#[test]
fn pending_required_gate_keeps_matching_bundle_unqualified() {
    let bundle = bundle(SOURCE_A, GateStatus::Pending);
    bundle.validate().expect("pending bundle remains structurally valid");
    assert!(!bundle.is_qualified());
}

#[test]
fn cross_paired_valid_artifacts_from_different_source_heads_are_rejected() {
    let mut bundle = bundle(SOURCE_A, GateStatus::Passed);
    bundle.evidence = evidence(SOURCE_B);

    assert!(bundle.qualification.validate().is_ok());
    assert!(bundle.evidence.validate().is_ok());

    let errors = bundle
        .validate()
        .expect_err("cross-paired source lineages must fail");
    assert!(errors.iter().any(|error| {
        error.contains("source_commit does not match evidence capsule")
            || error.contains("source commits differ")
    }));
    assert!(!bundle.is_qualified());
}

#[test]
fn bundle_digest_changes_when_bound_qualification_evidence_changes() {
    let left = bundle(SOURCE_A, GateStatus::Passed);
    let mut right = left.clone();
    right.qualification.gates[0].evidence = "evidence:local_fmt:alternate".into();

    assert_ne!(
        left.sha256().expect("left digest"),
        right.sha256().expect("right digest")
    );
}
