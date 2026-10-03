use symthaea_evidence_plane::execution_lineage::{
    qualify_lineage_perturbation, EvidenceLineageCommitError, EvidenceLineageDecision,
    EvidenceLineageGuardV1, ExecutionLineageDriftFieldV1, ExecutionLineageV1,
    LineagePerturbationResult,
    RepositorySourceSnapshotId,
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
fn public_execution_lineage_namespace_exposes_checked_admission_paths() {
    let base = fixture();
    let mut changed = base.clone();
    changed.source_revision = "d".repeat(40);

    let guard = EvidenceLineageGuardV1::prepare(&base).unwrap();
    assert_eq!(
        guard.try_check(&changed).unwrap(),
        EvidenceLineageDecision::ReprepareBeforeEvidence
    );

    assert_eq!(
        qualify_lineage_perturbation(&base, &changed, false),
        LineagePerturbationResult::UnexpectedCollateralChange
    );
    assert_eq!(
        symthaea_evidence_plane::execution_lineage::try_qualify_lineage_perturbation(&base, &changed, true)
            .unwrap(),
        LineagePerturbationResult::ExpectedDependencyChanged
    );
    assert_eq!(base.validated_digest().unwrap(), base.digest());
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
fn every_canonical_lineage_field_changes_identity_and_drift_report() {
    let base = fixture();

    let cases: Vec<(&str, ExecutionLineageV1, ExecutionLineageDriftFieldV1)> = vec![
        {
            let mut lineage = base.clone();
            lineage.source_repository = "another-org/symthaea".into();
            (
                "source_repository",
                lineage,
                ExecutionLineageDriftFieldV1::SourceRepository,
            )
        },
        {
            let mut lineage = base.clone();
            lineage.source_revision = "d".repeat(40);
            (
                "source_revision",
                lineage,
                ExecutionLineageDriftFieldV1::SourceRevision,
            )
        },
        {
            let mut lineage = base.clone();
            lineage.source_tree = "e".repeat(40);
            (
                "source_tree",
                lineage,
                ExecutionLineageDriftFieldV1::SourceTree,
            )
        },
        {
            let mut lineage = base.clone();
            lineage.repository_source_snapshot_id =
                RepositorySourceSnapshotId::parse(&"f".repeat(64)).unwrap();
            (
                "repository_source_snapshot_id",
                lineage,
                ExecutionLineageDriftFieldV1::RepositorySourceSnapshotId,
            )
        },
        {
            let mut lineage = base.clone();
            lineage
                .lock_digests
                .insert("Cargo.lock".into(), "sha256:1122334455667788".into());
            (
                "lock_digests",
                lineage,
                ExecutionLineageDriftFieldV1::LockDigests,
            )
        },
        {
            let mut lineage = base.clone();
            lineage
                .toolchain_versions
                .insert("cargo".into(), "1.96.0".into());
            (
                "toolchain_versions",
                lineage,
                ExecutionLineageDriftFieldV1::ToolchainVersions,
            )
        },
        {
            let mut lineage = base.clone();
            lineage.host_triple = "aarch64-unknown-linux-gnu".into();
            (
                "host_triple",
                lineage,
                ExecutionLineageDriftFieldV1::HostTriple,
            )
        },
        {
            let mut lineage = base.clone();
            lineage.target_triple = "wasm32-unknown-unknown".into();
            (
                "target_triple",
                lineage,
                ExecutionLineageDriftFieldV1::TargetTriple,
            )
        },
        {
            let mut lineage = base.clone();
            lineage.nix_identity = None;
            (
                "nix_identity",
                lineage,
                ExecutionLineageDriftFieldV1::NixIdentity,
            )
        },
        {
            let mut lineage = base.clone();
            lineage.feature_flags.insert("research".into());
            (
                "feature_flags",
                lineage,
                ExecutionLineageDriftFieldV1::FeatureFlags,
            )
        },
        {
            let mut lineage = base.clone();
            lineage.cwd = "/workspace/changed".into();
            (
                "working_directory",
                lineage,
                ExecutionLineageDriftFieldV1::WorkingDirectory,
            )
        },
        {
            let mut lineage = base.clone();
            lineage.argv.push("--nocapture".into());
            (
                "command_argv",
                lineage,
                ExecutionLineageDriftFieldV1::CommandArgv,
            )
        },
        {
            let mut lineage = base.clone();
            lineage
                .allowed_env
                .insert("RUSTFLAGS".into(), "-Copt-level=3".into());
            (
                "allowed_environment",
                lineage,
                ExecutionLineageDriftFieldV1::AllowedEnvironment,
            )
        },
        {
            let mut lineage = base.clone();
            lineage
                .immutable_input_digests
                .insert("dataset.bin".into(), "blake3:1122334455667788".into());
            (
                "immutable_input_digests",
                lineage,
                ExecutionLineageDriftFieldV1::ImmutableInputDigests,
            )
        },
    ];

    for (name, changed, expected_field) in cases {
        assert!(changed.validate().is_ok(), "{name} fixture must remain valid");
        assert_ne!(
            base.digest(),
            changed.digest(),
            "{name} must be identity-material"
        );
        let report =
            symthaea_evidence_plane::execution_lineage::ExecutionLineageDriftV1::between(
                &base, &changed,
            )
            .unwrap()
            .expect("changed lineage must report drift");
        assert_eq!(
            report.changed_fields,
            vec![expected_field],
            "{name} drift must be isolated"
        );
    }
}

#[test]
fn invalid_current_lineage_cannot_advance_guard_phase() {
    let base = fixture();
    let mut invalid = base.clone();
    invalid.source_revision = "not-a-git-object".into();

    let mut guard = EvidenceLineageGuardV1::prepare(&base).unwrap();
    let error = guard.commit_evidence(&invalid).unwrap_err();

    assert!(matches!(
        error,
        EvidenceLineageCommitError::InvalidCurrentLineage(_)
    ));
    assert_eq!(
        guard.try_check(&base).unwrap(),
        EvidenceLineageDecision::Stable
    );
    assert_eq!(
        guard.try_check(&invalid).unwrap_err().to_string(),
        "invalid or non-canonical Git object identity for source_revision: \"not-a-git-object\""
    );
}

#[test]
fn drift_report_rejects_invalid_lineage() {
    let prepared = fixture();
    let mut observed = prepared.clone();
    observed.source_revision = "not-a-git-object".into();

    assert_eq!(
        symthaea_evidence_plane::execution_lineage::ExecutionLineageDriftV1::between(
            &prepared,
            &observed,
        )
        .unwrap_err(),
        "invalid or non-canonical Git object identity for source_revision: \"not-a-git-object\""
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
fn public_perturbation_classifier_reports_declared_and_collateral_drift() {
    let base = fixture();
    let mut environment = base.clone();
    environment.host_triple = "aarch64-unknown-linux-gnu".into();
    let mut workload = base.clone();
    workload.source_revision = "d".repeat(40);

    assert_eq!(
        qualify_lineage_perturbation(&base, &base, false),
        LineagePerturbationResult::InvariantPreserved
    );
    assert_eq!(
        qualify_lineage_perturbation(&base, &workload, true),
        LineagePerturbationResult::ExpectedDependencyChanged
    );
    assert_eq!(
        qualify_lineage_perturbation(&base, &environment, false),
        LineagePerturbationResult::UnexpectedCollateralChange
    );
}

#[test]
fn serde_rejects_unknown_wire_fields() {
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
        "feature_flags":["default"],
        "cwd":"/workspace",
        "argv":["cargo","test"],
        "allowed_env":{"RUST_BACKTRACE":"0"},
        "immutable_input_digests":{"fixture":"blake3:8899aabbccddeeff"},
        "unsupported_field":"must-fail"
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
