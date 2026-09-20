#![allow(dead_code)]

#[path = "../src/cargo_adapter_semantics.rs"]
mod cargo_adapter_semantics;
#[path = "../src/cargo_context.rs"]
mod cargo_context;
#[path = "../src/cargo_context_verify.rs"]
mod cargo_context_verify;
#[path = "../src/repository_snapshot_receipt.rs"]
mod repository_snapshot_receipt;
#[path = "../src/repository_snapshot_diff.rs"]
mod repository_snapshot_diff;
#[path = "../src/repository_effect_policy.rs"]
mod repository_effect_policy;
#[path = "../src/cargo_execution_contract.rs"]
mod cargo_execution_contract;
#[path = "../src/cargo_runtime_binding.rs"]
mod cargo_runtime_binding;
#[path = "../src/repository_materialization_receipt.rs"]
mod repository_materialization_receipt;
#[path = "../src/tool_executable_attestation.rs"]
mod tool_executable_attestation;
#[path = "../src/pre_spawn_freshness.rs"]
mod pre_spawn_freshness;
#[path = "../src/prepared_cargo_intent.rs"]
mod prepared_cargo_intent;

#[cfg(test)]
mod contract_tests {
    use super::*;
    use cargo_adapter_semantics::{
        BoundInputPolicy, CapturePolicy, CargoAdapterSemanticsReceipt, CargoAdapterSemanticsSpec,
        CargoHomePolicy, DescendantPolicy, EnvironmentInheritancePolicy, EnvironmentPolicy,
        EphemeralDirectoryPolicy, NetworkPolicy, PlatformFamily, PointOfNoReturnPolicy,
        ProcessTreePolicy, RepositoryAccess, SandboxBackend, StdinPolicy, WorkingDirectoryPolicy,
    };
    use cargo_context::{
        CargoBuildContextDocument, CargoBuildContextSpec, CargoOperation, FeatureSelection,
        PackageSelection, TargetSelection, ToolchainIdentity,
    };
    use cargo_runtime_binding::{
        BoundRuntimeInput, CargoHomeEvidence, CargoRuntimeBindingReceipt, CargoRuntimeBindingSpec,
        EphemeralDirectoryEvidence, RealizedAdapterConfiguration, RuntimeContextRealization,
        RuntimeEnvironmentValue, RuntimePreparationEvidence,
    };
    use prepared_cargo_intent::{GitWorktreeStateBinding, PreparedCargoIntentSpec};
    use repository_effect_policy::{EffectMode, EffectPolicySpec};
    use repository_materialization_receipt::{
        RepositoryMaterializationReceipt, RepositoryMaterializationSpec,
    };
    use repository_snapshot_receipt::{
        EntryKind, SnapshotEntry, SnapshotScope, SourceClass, UnknownSurface,
        ValidatedSnapshotReceipt,
    };
    use std::collections::BTreeMap;
    use tool_executable_attestation::{
        ExecutableProbeEvidence, ToolExecutableAttestationReceipt, ToolExecutableAttestationSpec,
    };

    fn digest(seed: char) -> String {
        assert!(seed.is_ascii(), "fixture digest seed must be ASCII");
        format!("{:02x}", u32::from(seed)).repeat(32)
    }

    fn source() -> ValidatedSnapshotReceipt {
        let mut source = ValidatedSnapshotReceipt {
            snapshot_id: String::new(),
            schema: "symthaea.repository-source-snapshot.v1".into(),
            git_head: "a".repeat(40),
            git_head_tree: "b".repeat(40),
            git_version: "git version 2.50.0".into(),
            scope: SnapshotScope {
                tracked_worktree: true,
                untracked_non_ignored: true,
                explicit_ignored_inputs: vec![],
                ignored_policy: "git_exclude_standard_plus_explicit_ignored_inputs".into(),
                symlink_policy: "hash_link_target_bytes_do_not_follow_referent".into(),
                submodule_policy: "record_gitlink_index_identity_mark_contents_unknown".into(),
                external_input_policy: "not_captured".into(),
            },
            entries: vec![SnapshotEntry {
                path: "src/lib.rs".into(),
                source_class: SourceClass::Tracked,
                kind: EntryKind::File,
                content_sha256: Some(digest('c')),
                size_bytes: Some(12),
                executable: Some(false),
                index_mode: Some("100644".into()),
                index_blob: Some("d".repeat(40)),
            }],
            unknown_surfaces: vec![UnknownSurface {
                kind: "environment_network_and_external_build_inputs_not_captured".into(),
                subject: None,
            }],
        };
        source.snapshot_id = source.computed_snapshot_id().unwrap();
        source.validate().unwrap();
        source
    }

    fn context() -> CargoBuildContextDocument {
        cargo_context::build_document(CargoBuildContextSpec {
            schema: "symthaea.cargo-build-context.v1".into(),
            operation: CargoOperation::Check,
            manifest_path: "Cargo.toml".into(),
            package_selection: PackageSelection {
                workspace: false,
                packages: vec!["symthaea".into()],
                exclude: vec![],
            },
            target_selection: TargetSelection {
                lib: true,
                bins: vec![],
                examples: vec![],
                tests: vec![],
                benches: vec![],
                all_targets: false,
            },
            feature_selection: FeatureSelection {
                requested: vec!["code_generation".into()],
                no_default_features: false,
                all_features: false,
            },
            target_triples: vec!["x86_64-unknown-linux-gnu".into()],
            profile: "dev".into(),
            toolchain: ToolchainIdentity {
                cargo_version: "cargo 1.96.0".into(),
                rustc_version: "rustc 1.96.0".into(),
                host_triple: "x86_64-unknown-linux-gnu".into(),
                toolchain_name: Some("1.96.0".into()),
            },
            cargo_config_sha256: Some(digest('4')),
            rustflags_sha256: Some(digest('5')),
            rustdocflags_sha256: None,
            environment_fingerprints: BTreeMap::from([("RUSTFLAGS".into(), digest('5'))]),
            raw_argv: vec![
                "cargo".into(),
                "check".into(),
                "--lib".into(),
                "--features".into(),
                "code_generation".into(),
            ],
        })
        .unwrap()
    }

    fn semantics() -> CargoAdapterSemanticsReceipt {
        cargo_adapter_semantics::build_receipt(CargoAdapterSemanticsSpec {
            schema: "symthaea.cargo-adapter-semantics-input.v1".into(),
            adapter_implementation_sha256: digest('1'),
            platform: PlatformFamily::Linux,
            sandbox_backend: SandboxBackend {
                name: "bubblewrap".into(),
                implementation_sha256: digest('2'),
            },
            repository_access: RepositoryAccess::ReadOnlySource,
            working_directory: WorkingDirectoryPolicy::RepositoryRoot,
            network: NetworkPolicy::Denied,
            environment: EnvironmentPolicy {
                inheritance: EnvironmentInheritancePolicy::ClearThenAllowlist,
                allowed_keys: vec!["RUST_BACKTRACE".into()],
            },
            home: EphemeralDirectoryPolicy::EphemeralEmpty,
            cargo_home: CargoHomePolicy::ReadOnlyPrefetched,
            target_dir: EphemeralDirectoryPolicy::EphemeralEmpty,
            temp_dir: EphemeralDirectoryPolicy::EphemeralEmpty,
            external_inputs: BoundInputPolicy::ReadOnlyBoundClosures,
            descendants: DescendantPolicy::SameSandbox,
            stdin: StdinPolicy::Null,
            stdout: CapturePolicy::ExactBytesNoTruncation,
            stderr: CapturePolicy::ExactBytesNoTruncation,
            wall_clock_timeout_ms: 1_200_000,
            termination_grace_ms: 5_000,
            process_tree: ProcessTreePolicy::TerminateThenKillEntireSandboxTree,
            point_of_no_return: PointOfNoReturnPolicy::AfterEffectAdmissionBeforeSpawn,
        })
        .unwrap()
    }

    fn effect_policy() -> EffectPolicySpec {
        EffectPolicySpec {
            schema: "symthaea.repository-effect-policy.v1".into(),
            mode: EffectMode::ReadOnlySource,
            expected_diff_id: None,
        }
    }

    fn dir(seed: char) -> EphemeralDirectoryEvidence {
        EphemeralDirectoryEvidence {
            instance_id: digest(seed),
            empty_at_start_receipt_sha256: digest(char::from_u32(seed as u32 + 1).unwrap()),
            setup_receipt_sha256: digest(char::from_u32(seed as u32 + 2).unwrap()),
        }
    }

    fn runtime() -> CargoRuntimeBindingReceipt {
        let source = source();
        let context = context();
        let semantics = semantics();
        let mut policy = effect_policy();
        let policy_id = repository_effect_policy::validate_and_identify_policy(&mut policy).unwrap();
        cargo_runtime_binding::build_receipt(
            CargoRuntimeBindingSpec {
                schema: "symthaea.cargo-runtime-binding-input.v1".into(),
                realized: RealizedAdapterConfiguration {
                    platform: PlatformFamily::Linux,
                    adapter_implementation_sha256: digest('1'),
                    sandbox_backend: SandboxBackend {
                        name: "bubblewrap".into(),
                        implementation_sha256: digest('2'),
                    },
                    repository_access: RepositoryAccess::ReadOnlySource,
                    working_directory: WorkingDirectoryPolicy::RepositoryRoot,
                    network: NetworkPolicy::Denied,
                    environment_inheritance: EnvironmentInheritancePolicy::ClearThenAllowlist,
                    home: EphemeralDirectoryPolicy::EphemeralEmpty,
                    cargo_home: CargoHomePolicy::ReadOnlyPrefetched,
                    target_dir: EphemeralDirectoryPolicy::EphemeralEmpty,
                    temp_dir: EphemeralDirectoryPolicy::EphemeralEmpty,
                    external_inputs: BoundInputPolicy::ReadOnlyBoundClosures,
                    descendants: DescendantPolicy::SameSandbox,
                    stdin: StdinPolicy::Null,
                    stdout: CapturePolicy::ExactBytesNoTruncation,
                    stderr: CapturePolicy::ExactBytesNoTruncation,
                    wall_clock_timeout_ms: 1_200_000,
                    termination_grace_ms: 5_000,
                    process_tree: ProcessTreePolicy::TerminateThenKillEntireSandboxTree,
                    point_of_no_return: PointOfNoReturnPolicy::AfterEffectAdmissionBeforeSpawn,
                },
                context: RuntimeContextRealization {
                    toolchain: context.context.toolchain.clone(),
                    cargo_config_sha256: context.context.cargo_config_sha256.clone(),
                    rustflags_sha256: context.context.rustflags_sha256.clone(),
                    rustdocflags_sha256: context.context.rustdocflags_sha256.clone(),
                    environment_fingerprints: context.context.environment_fingerprints.clone(),
                    configuration_setup_receipt_sha256: digest('6'),
                },
                evidence: RuntimePreparationEvidence {
                    runtime_instance_id: digest('7'),
                    sandbox_setup_receipt_sha256: digest('8'),
                    repository_materialization_id: digest('9'),
                    repository_setup_receipt_sha256: digest('a'),
                    working_directory_subject_id: digest('9'),
                    network_setup_receipt_sha256: digest('b'),
                    environment_clear_receipt_sha256: digest('c'),
                    ambient_environment: BTreeMap::from([(
                        "RUST_BACKTRACE".into(),
                        RuntimeEnvironmentValue::Present {
                            value_sha256: digest('d'),
                        },
                    )]),
                    home: dir('e'),
                    cargo_home: CargoHomeEvidence::ReadOnlyPrefetched {
                        content_id: digest('h'),
                        setup_receipt_sha256: digest('i'),
                    },
                    target_dir: dir('j'),
                    temp_dir: dir('m'),
                    bound_inputs: vec![
                        BoundRuntimeInput {
                            role: "rustc_executable".into(),
                            content_id: digest('q'),
                            read_only_setup_receipt_sha256: digest('r'),
                        },
                        BoundRuntimeInput {
                            role: "cargo_home_prefetch".into(),
                            content_id: digest('h'),
                            read_only_setup_receipt_sha256: digest('s'),
                        },
                        BoundRuntimeInput {
                            role: "cargo_executable".into(),
                            content_id: digest('t'),
                            read_only_setup_receipt_sha256: digest('u'),
                        },
                    ],
                    io_setup_receipt_sha256: digest('v'),
                    process_tree_setup_receipt_sha256: digest('w'),
                },
            },
            source,
            context,
            semantics,
            policy_id,
            policy,
        )
        .unwrap()
    }

    fn materialization() -> RepositoryMaterializationReceipt {
        repository_materialization_receipt::build_receipt(
            RepositoryMaterializationSpec {
                schema: "symthaea.repository-materialization-input.v1".into(),
                runtime_instance_id: digest('7'),
                kind: RepositoryAccess::ReadOnlySource,
                materialization_id: digest('9'),
                working_directory_subject_id: digest('9'),
                repository_setup_receipt_sha256: digest('a'),
                derivation_receipt_sha256: digest('x'),
                staged_content_id: None,
            },
            source(),
        )
        .unwrap()
    }

    fn tools() -> ToolExecutableAttestationReceipt {
        tool_executable_attestation::build_receipt(
            ToolExecutableAttestationSpec {
                schema: "symthaea.tool-executable-attestation-input.v1".into(),
                cargo: ExecutableProbeEvidence {
                    executable_content_id: digest('t'),
                    probe_transcript_sha256: digest('1'),
                    probe_implementation_sha256: digest('2'),
                    reported_version: "cargo 1.96.0".into(),
                    reported_host_triple: "x86_64-unknown-linux-gnu".into(),
                },
                rustc: ExecutableProbeEvidence {
                    executable_content_id: digest('q'),
                    probe_transcript_sha256: digest('3'),
                    probe_implementation_sha256: digest('2'),
                    reported_version: "rustc 1.96.0".into(),
                    reported_host_triple: "x86_64-unknown-linux-gnu".into(),
                },
            },
            context(),
        )
        .unwrap()
    }

    fn intent_spec() -> PreparedCargoIntentSpec {
        PreparedCargoIntentSpec {
            schema: "symthaea.prepared-cargo-execution-intent-input.v1".into(),
            plan_id: Some(digest('3')),
            transaction_id: Some(digest('4')),
        }
    }

    fn build() -> anyhow::Result<prepared_cargo_intent::PreparedCargoExecutionIntent> {
        prepared_cargo_intent::build_prepared_intent(
            intent_spec(),
            source(),
            context(),
            semantics(),
            effect_policy(),
            runtime(),
            materialization(),
            tools(),
        )
    }

    #[test]
    fn prepared_intent_derives_all_engineering_subjects() {
        let prepared = build().unwrap();
        assert_eq!(prepared.base_intent.build_context_id, context().context_id);
        assert_eq!(prepared.base_intent.invocation_id, context().invocation_id);
        assert_eq!(prepared.runtime_binding_id, runtime().runtime_binding_id);
        assert_eq!(
            prepared.materialization_receipt_id,
            materialization().materialization_receipt_id
        );
        assert_eq!(prepared.tool_attestation_id, tools().tool_attestation_id);
        assert_eq!(prepared.base_intent.git_worktree_state_before, None);
        assert_eq!(
            prepared.git_worktree_state_binding,
            GitWorktreeStateBinding::NotConverged
        );
    }

    #[test]
    fn input_schema_rejects_caller_minted_engineering_ids() {
        let json = format!(
            "{{\"schema\":\"symthaea.prepared-cargo-execution-intent-input.v1\",\"plan_id\":\"{}\",\"build_context_id\":\"{}\"}}",
            digest('3'),
            digest('9')
        );
        assert!(serde_json::from_str::<PreparedCargoIntentSpec>(&json).is_err());
    }

    #[test]
    fn mutated_runtime_receipt_rejects_before_intent_minting() {
        let mut bad = runtime();
        bad.context_id = digest('0');
        assert!(
            prepared_cargo_intent::build_prepared_intent(
                intent_spec(),
                source(),
                context(),
                semantics(),
                effect_policy(),
                bad,
                materialization(),
                tools(),
            )
            .is_err()
        );
    }

    #[test]
    fn materialization_or_executable_substitution_rejects() {
        let mut bad_materialization = materialization();
        bad_materialization.materialization_id = digest('0');
        assert!(
            prepared_cargo_intent::build_prepared_intent(
                intent_spec(),
                source(),
                context(),
                semantics(),
                effect_policy(),
                runtime(),
                bad_materialization,
                tools(),
            )
            .is_err()
        );

        let mut bad_tools = tools();
        bad_tools.cargo.executable_content_id = digest('0');
        assert!(
            prepared_cargo_intent::build_prepared_intent(
                intent_spec(),
                source(),
                context(),
                semantics(),
                effect_policy(),
                runtime(),
                materialization(),
                bad_tools,
            )
            .is_err()
        );
    }

    #[test]
    fn plan_or_transaction_change_changes_prepared_intent_identity() {
        let a = build().unwrap();
        let mut changed = intent_spec();
        changed.plan_id = Some(digest('5'));
        let b = prepared_cargo_intent::build_prepared_intent(
            changed,
            source(),
            context(),
            semantics(),
            effect_policy(),
            runtime(),
            materialization(),
            tools(),
        )
        .unwrap();
        assert_ne!(a.prepared_intent_id, b.prepared_intent_id);
    }
}
