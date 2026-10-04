use symthaea_evidence_plane::execution_lineage::{
    EvidenceLineageCommitError, EvidenceLineageDecision, EvidenceLineageGuardV1,
    ExecutionLineageDriftFieldV1, ExecutionLineageV1, LineagePerturbationResult,
    RepositorySourceSnapshotId, qualify_lineage_perturbation,
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
        symthaea_evidence_plane::execution_lineage::try_qualify_lineage_perturbation(
            &base, &changed, true,
        )
        .unwrap(),
        LineagePerturbationResult::ExpectedDependencyChanged
    );
    assert_eq!(base.validated_digest().unwrap(), base.digest());
}

#[test]
fn public_perturbation_classifier_covers_both_invariant_outcomes() {
    let base = fixture();

    assert_eq!(
        qualify_lineage_perturbation(&base, &base, false),
        LineagePerturbationResult::InvariantPreserved
    );
    assert_eq!(
        qualify_lineage_perturbation(&base, &base, true),
        LineagePerturbationResult::UnexpectedInvariance
    );
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
fn environment_identity_is_stable_across_workload_only_changes() {
    let base = fixture();
    let environment = base.environment_digest().unwrap();

    let mut changed = base.clone();
    changed.source_revision = "d".repeat(40);
    changed.repository_source_snapshot_id =
        RepositorySourceSnapshotId::parse(&"f".repeat(64)).unwrap();
    changed.argv.push("--nocapture".into());
    changed
        .immutable_input_digests
        .insert("dataset.bin".into(), "blake3:1122334455667788".into());

    assert_eq!(changed.environment_digest().unwrap(), environment);
    assert_ne!(
        changed.workload_digest().unwrap(),
        base.workload_digest().unwrap()
    );
    assert_ne!(
        changed.validated_digest().unwrap(),
        base.validated_digest().unwrap()
    );
}

#[test]
fn every_canonical_lineage_field_changes_only_its_declared_projection() {
    let base = fixture();
    let workload = base.workload_digest().unwrap();
    let environment = base.environment_digest().unwrap();

    let cases: Vec<(
        &str,
        Box<dyn Fn(&mut ExecutionLineageV1)>,
        bool,
    )> = vec![
        (
            "source_repository",
            Box::new(|lineage| lineage.source_repository = "another-org/symthaea".into()),
            true,
        ),
        (
            "source_revision",
            Box::new(|lineage| lineage.source_revision = "d".repeat(40)),
            true,
        ),
        (
            "source_tree",
            Box::new(|lineage| lineage.source_tree = "e".repeat(40)),
            true,
        ),
        (
            "repository_source_snapshot_id",
            Box::new(|lineage| {
                lineage.repository_source_snapshot_id =
                    RepositorySourceSnapshotId::parse(&"f".repeat(64)).unwrap();
            }),
            true,
        ),
        (
            "lock_digests",
            Box::new(|lineage| {
                lineage
                    .lock_digests
                    .insert("flake.lock".into(), "sha256:1122334455667788".into());
            }),
            true,
        ),
        (
            "feature_flags",
            Box::new(|lineage| {
                lineage.feature_flags.insert("research".into());
            }),
            true,
        ),
        (
            "working_directory",
            Box::new(|lineage| lineage.cwd = "/workspace/changed".into()),
            true,
        ),
        (
            "command_argv",
            Box::new(|lineage| lineage.argv.push("--nocapture".into())),
            true,
        ),
        (
            "immutable_input_digests",
            Box::new(|lineage| {
                lineage
                    .immutable_input_digests
                    .insert("dataset.bin".into(), "blake3:1122334455667788".into());
            }),
            true,
        ),
        (
            "toolchain_versions",
            Box::new(|lineage| {
                lineage
                    .toolchain_versions
                    .insert("cargo".into(), "1.96.0".into());
            }),
            false,
        ),
        (
            "host_triple",
            Box::new(|lineage| lineage.host_triple = "aarch64-unknown-linux-gnu".into()),
            false,
        ),
        (
            "target_triple",
            Box::new(|lineage| lineage.target_triple = "wasm32-unknown-unknown".into()),
            false,
        ),
        (
            "nix_identity",
            Box::new(|lineage| lineage.nix_identity = None),
            false,
        ),
        (
            "allowed_environment",
            Box::new(|lineage| {
                lineage
                    .allowed_env
                    .insert("RUSTFLAGS".into(), "-Copt-level=3".into());
            }),
            false,
        ),
    ];

    for (name, mutate, affects_workload) in cases {
        let mut changed = base.clone();
        mutate(&mut changed);

        assert_ne!(
            changed.validated_digest().unwrap(),
            base.validated_digest().unwrap(),
            "{name} must change the full lineage identity",
        );

        if affects_workload {
            assert_ne!(
                changed.workload_digest().unwrap(),
                workload,
                "{name} must change workload identity",
            );
            assert_eq!(
                changed.environment_digest().unwrap(),
                environment,
                "{name} must not change environment identity",
            );
        } else {
            assert_eq!(
                changed.workload_digest().unwrap(),
                workload,
                "{name} must not change workload identity",
            );
            assert_ne!(
                changed.environment_digest().unwrap(),
                environment,
                "{name} must change environment identity",
            );
        }
    }
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
        assert!(
            changed.validate().is_ok(),
            "{name} fixture must remain valid"
        );
        assert_ne!(
            base.digest(),
            changed.digest(),
            "{name} must be identity-material"
        );
        let report = symthaea_evidence_plane::execution_lineage::ExecutionLineageDriftV1::between(
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
fn public_checked_paths_reject_invalid_lineage() {
    let base = fixture();
    let mut invalid = base.clone();
    invalid.source_revision = "not-a-git-object".into();

    assert_eq!(
        invalid.validated_digest().unwrap_err(),
        "invalid or non-canonical Git object identity for source_revision: \"not-a-git-object\""
    );
    assert_eq!(
        symthaea_evidence_plane::execution_lineage::try_qualify_lineage_perturbation(
            &base, &invalid, true,
        )
        .unwrap_err(),
        "invalid or non-canonical Git object identity for source_revision: \"not-a-git-object\""
    );
    assert_eq!(
        EvidenceLineageGuardV1::prepare(&invalid).unwrap_err(),
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
            &prepared, &observed,
        )
        .unwrap_err(),
        "invalid or non-canonical Git object identity for source_revision: \"not-a-git-object\""
    );
}

#[test]
fn invalid_post_commit_lineage_cannot_bypass_committed_guard_state() {
    let base = fixture();
    let mut invalid = base.clone();
    invalid.source_revision = "not-a-git-object".into();
    let mut changed = base.clone();
    changed.source_revision = "d".repeat(40);

    let mut guard = EvidenceLineageGuardV1::prepare(&base).unwrap();
    guard.commit_evidence(&base).unwrap();

    assert!(matches!(
        guard.commit_evidence(&invalid),
        Err(EvidenceLineageCommitError::InvalidCurrentLineage(_))
    ));
    assert_eq!(
        guard.try_check(&base).unwrap(),
        EvidenceLineageDecision::Stable
    );
    assert_eq!(
        guard.try_check(&changed).unwrap(),
        EvidenceLineageDecision::RefuseMixedLineageAfterEvidence
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
fn serde_reordering_does_not_change_lineage_identity() {
    let first = r#"{
        "source_repository":"luminous-dynamics/symthaea",
        "source_revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "source_tree":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "repository_source_snapshot_id":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        "lock_digests":{
            "Cargo.toml":"sha256:8899aabbccddeeff",
            "Cargo.lock":"sha256:0011223344556677"
        },
        "toolchain_versions":{
            "cargo":"1.96.0",
            "rustc":"1.96.0"
        },
        "host_triple":"x86_64-unknown-linux-gnu",
        "target_triple":"x86_64-unknown-linux-gnu",
        "nix_identity":"nix:fixture",
        "feature_flags":["research","default"],
        "cwd":"/workspace",
        "argv":["cargo","test"],
        "allowed_env":{
            "RUSTFLAGS":"-Copt-level=3",
            "RUST_BACKTRACE":"0"
        },
        "immutable_input_digests":{
            "z":"blake3:8899aabbccddeeff",
            "a":"blake3:0011223344556677"
        }
    }"#;

    let second = r#"{
        "immutable_input_digests":{
            "a":"blake3:0011223344556677",
            "z":"blake3:8899aabbccddeeff"
        },
        "argv":["cargo","test"],
        "cwd":"/workspace",
        "feature_flags":["default","research"],
        "nix_identity":"nix:fixture",
        "target_triple":"x86_64-unknown-linux-gnu",
        "host_triple":"x86_64-unknown-linux-gnu",
        "toolchain_versions":{
            "rustc":"1.96.0",
            "cargo":"1.96.0"
        },
        "lock_digests":{
            "Cargo.lock":"sha256:0011223344556677",
            "Cargo.toml":"sha256:8899aabbccddeeff"
        },
        "repository_source_snapshot_id":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        "source_tree":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "source_revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "source_repository":"luminous-dynamics/symthaea",
        "allowed_env":{
            "RUST_BACKTRACE":"0",
            "RUSTFLAGS":"-Copt-level=3"
        }
    }"#;

    let first: ExecutionLineageV1 = serde_json::from_str(first).unwrap();
    let second: ExecutionLineageV1 = serde_json::from_str(second).unwrap();

    assert_eq!(first, second);
    assert_eq!(first.digest(), second.digest());
    assert_eq!(
        first.workload_digest().unwrap(),
        second.workload_digest().unwrap()
    );
    assert_eq!(
        first.environment_digest().unwrap(),
        second.environment_digest().unwrap()
    );
}

#[test]
fn serde_rejects_duplicate_keys_in_every_named_map_family() {
    let cases = [
        (
            "toolchain_versions",
            "\"rustc\":\"1.96.0\",\"rustc\":\"1.97.0\"",
        ),
        (
            "allowed_env",
            "\"RUST_BACKTRACE\":\"0\",\"RUST_BACKTRACE\":\"1\"",
        ),
        (
            "immutable_input_digests",
            "\"fixture\":\"blake3:8899aabbccddeeff\",\"fixture\":\"blake3:0011223344556677\"",
        ),
    ];

    for (field, entries) in cases {
        let immutable_inputs = if field == "immutable_input_digests" {
            ""
        } else {
            r#","immutable_input_digests":{"fixture":"blake3:8899aabbccddeeff"}"#
        };
        let text = format!(
            r#"{{"source_repository":"luminous-dynamics/symthaea","source_revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","source_tree":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","repository_source_snapshot_id":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","lock_digests":{{"Cargo.lock":"sha256:0011223344556677"}},"{field}":{{{entries}}},"host_triple":"x86_64-unknown-linux-gnu","target_triple":"x86_64-unknown-linux-gnu","nix_identity":"nix:fixture","feature_flags":["default"],"cwd":"/workspace","argv":["cargo","test"],"allowed_env":{{"RUST_BACKTRACE":"0"}}{immutable_inputs}}}"#
        );

        let error = serde_json::from_str::<ExecutionLineageV1>(&text).unwrap_err();
        assert!(
            error.to_string().contains("duplicate map key"),
            "duplicate {field} entries must fail in the inner map visitor: {error}"
        );
    }
}

#[test]
fn public_raw_constructor_rejects_duplicate_named_maps() {
    let toolchain = ExecutionLineageV1::from_raw_entries(
        "luminous-dynamics/symthaea".into(),
        "a".repeat(40),
        "b".repeat(40),
        "c".repeat(64),
        vec![("Cargo.lock".into(), "sha256:0011223344556677".into())],
        vec![
            ("rustc".into(), "1.96.0".into()),
            ("rustc".into(), "1.97.0".into()),
        ],
        "x86_64-unknown-linux-gnu".into(),
        "x86_64-unknown-linux-gnu".into(),
        Some("nix:fixture".into()),
        vec!["default".into()],
        "/workspace".into(),
        vec!["cargo".into(), "test".into()],
        vec![("RUST_BACKTRACE".into(), "0".into())],
        vec![("fixture".into(), "blake3:8899aabbccddeeff".into())],
    );
    assert!(toolchain.unwrap_err().contains("duplicate name"));

    let environment = ExecutionLineageV1::from_raw_entries(
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
        vec![
            ("RUST_BACKTRACE".into(), "0".into()),
            ("RUST_BACKTRACE".into(), "1".into()),
        ],
        vec![("fixture".into(), "blake3:8899aabbccddeeff".into())],
    );
    assert!(environment.unwrap_err().contains("duplicate name"));

    let immutable_inputs = ExecutionLineageV1::from_raw_entries(
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
        vec![
            ("fixture".into(), "blake3:8899aabbccddeeff".into()),
            ("fixture".into(), "blake3:0011223344556677".into()),
        ],
    );
    assert!(immutable_inputs.unwrap_err().contains("duplicate name"));
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
fn public_serde_serialization_rejects_invalid_mutated_lineage() {
    let mut lineage = fixture();
    lineage.source_revision = "not-a-git-object".into();

    let error = serde_json::to_string(&lineage).unwrap_err();
    assert_eq!(
        error.to_string(),
        "invalid or non-canonical Git object identity for source_revision: \"not-a-git-object\""
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
