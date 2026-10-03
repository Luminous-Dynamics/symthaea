use symthaea_evidence_plane::execution_lineage::{
    EvidenceLineageDecision, EvidenceLineageGuardV1, ExecutionLineageV1, RepositorySourceSnapshotId,
};

fn fixture() -> ExecutionLineageV1 {
    ExecutionLineageV1::from_raw_entries(
        "luminous-dynamics/symthaea".into(),
        "a".repeat(40),
        "b".repeat(40),
        "c".repeat(64),
        vec![("Cargo.lock".into(), "sha256:0011223344556677".into())],
        vec![("rustc".into(), "1.96.0".into())],
        "x86_64-unknown-linux-gnu".into(),
        "x86_64-unknown-linux-gnu".into(),
        Some("nix:fixture".into()),
        vec!["default".into()],
        "/workspace".into(),
        vec!["cargo".into(), "test".into()],
        vec![("RUST_BACKTRACE".into(), "0".into())],
        vec![("fixture".into(), "blake3:8899aabbccddeeff".into())],
    )
    .expect("fixture must be valid")
}

#[test]
fn public_execution_lineage_namespace_exposes_validated_identity() {
    let lineage = fixture();
    assert_eq!(
        lineage.repository_source_snapshot_id,
        RepositorySourceSnapshotId::parse(&"c".repeat(64)).unwrap()
    );
    assert_eq!(lineage.validated_digest().unwrap().len(), 64);
}

#[test]
fn workload_identity_is_stable_across_environment_only_changes() {
    let base = fixture();
    let workload = base.workload_digest().unwrap();
    let environment = base.environment_digest().unwrap();

    let mut changed = base.clone();
    changed
        .toolchain_versions
        .insert("cargo".into(), "1.96.0".into());
    changed.host_triple = "aarch64-unknown-linux-gnu".into();
    changed
        .allowed_env
        .insert("RUSTFLAGS".into(), "-Copt-level=3".into());

    assert_eq!(changed.workload_digest().unwrap(), workload);
    assert_ne!(changed.environment_digest().unwrap(), environment);
    assert_ne!(
        changed.validated_digest().unwrap(),
        base.validated_digest().unwrap()
    );
}

#[test]
fn guard_refuses_cross_execution_evidence_after_commit() {
    let base = fixture();
    let mut changed = base.clone();
    changed.source_revision = "d".repeat(40);

    let mut guard = EvidenceLineageGuardV1::prepare(&base).unwrap();
    guard.commit_evidence(&base).unwrap();

    assert_eq!(
        guard.try_check(&changed).unwrap(),
        EvidenceLineageDecision::RefuseMixedLineageAfterEvidence
    );
    assert!(guard.commit_evidence(&changed).is_err());
}

#[test]
fn serde_rejects_duplicate_map_keys_before_canonicalization() {
    let text = r#"{
        "source_repository":"luminous-dynamics/symthaea",
        "source_revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "source_tree":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "repository_source_snapshot_id":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        "lock_digests":{
            "Cargo.lock":"sha256:0011223344556677",
            "Cargo.lock":"blake3:8899aabbccddeeff"
        },
        "toolchain_versions":{"rustc":"1.96.0"},
        "host_triple":"x86_64-unknown-linux-gnu",
        "target_triple":"x86_64-unknown-linux-gnu",
        "nix_identity":"nix:fixture",
        "feature_flags":["default"],
        "cwd":"/workspace",
        "argv":["cargo","test"],
        "allowed_env":{"RUST_BACKTRACE":"0"},
        "immutable_input_digests":{"fixture":"blake3:8899aabbccddeeff"}
    }"#;

    assert!(serde_json::from_str::<ExecutionLineageV1>(text).is_err());
}

#[test]
fn serde_rejects_duplicate_feature_members_before_set_canonicalization() {
    let text = r#"{
        "source_repository":"luminous-dynamics/symthaea",
        "source_revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "source_tree":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "repository_source_snapshot_id":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        "lock_digests":{"Cargo.lock":"sha256:0011223344556677"},
        "toolchain_versions":{"rustc":"1.96.0"},
        "host_triple":"x86_64-unknown-linux-gnu",
        "target_triple":"x86_64-unknown-linux-gnu",
        "nix_identity":"nix:fixture",
        "feature_flags":["default","default"],
        "cwd":"/workspace",
        "argv":["cargo","test"],
        "allowed_env":{"RUST_BACKTRACE":"0"},
        "immutable_input_digests":{"fixture":"blake3:8899aabbccddeeff"}
    }"#;

    assert!(serde_json::from_str::<ExecutionLineageV1>(text).is_err());
}
