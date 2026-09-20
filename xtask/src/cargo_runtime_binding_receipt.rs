use anyhow::{Context, bail};

use crate::cargo_adapter_semantics::CargoAdapterSemanticsReceipt;
use crate::cargo_context::CargoBuildContextDocument;
use crate::cargo_runtime_binding::{
    CargoRuntimeBindingReceipt, CargoRuntimeBindingSpec, build_receipt,
};
use crate::repository_effect_policy::{EffectPolicySpec, validate_and_identify_policy};
use crate::repository_snapshot_receipt::ValidatedSnapshotReceipt;

const RECEIPT_SCHEMA: &str = "symthaea.cargo-runtime-binding.v1";
const INPUT_SCHEMA: &str = "symthaea.cargo-runtime-binding-input.v1";

/// Revalidate a persisted runtime-binding receipt by projecting it back into
/// the original constructor input and rebuilding it through the canonical
/// runtime-binding implementation.
///
/// This module deliberately owns no second normalization or identity formula.
/// Any future change to runtime-binding semantics must therefore be expressed
/// in `cargo_runtime_binding::build_receipt`, and stored receipts inherit that
/// exact interpretation when reloaded.
pub(crate) fn verify_stored(
    stored: CargoRuntimeBindingReceipt,
    source: ValidatedSnapshotReceipt,
    context: CargoBuildContextDocument,
    semantics: CargoAdapterSemanticsReceipt,
    mut effect_policy: EffectPolicySpec,
) -> anyhow::Result<CargoRuntimeBindingReceipt> {
    if stored.schema != RECEIPT_SCHEMA {
        bail!(
            "unsupported Cargo runtime-binding receipt schema: {}",
            stored.schema
        );
    }
    validate_digest("runtime_binding_id", &stored.runtime_binding_id)?;

    let effect_policy_id = validate_and_identify_policy(&mut effect_policy)?;
    let spec = CargoRuntimeBindingSpec {
        schema: INPUT_SCHEMA.into(),
        realized: stored.realized.clone(),
        context: stored.context.clone(),
        evidence: stored.evidence.clone(),
    };

    let rebuilt = build_receipt(
        spec,
        source,
        context,
        semantics,
        effect_policy_id,
        effect_policy,
    )
    .context("rebuild persisted Cargo runtime-binding receipt")?;

    if rebuilt != stored {
        bail!(
            "stored Cargo runtime-binding receipt is not the canonical rebuild for its validated upstream subjects"
        );
    }
    Ok(rebuilt)
}

fn validate_digest(name: &str, value: &str) -> anyhow::Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{name} must be a 64-character SHA-256 hex digest");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use crate::cargo_adapter_semantics::{
        BoundInputPolicy, CapturePolicy, CargoAdapterSemanticsSpec, CargoHomePolicy,
        DescendantPolicy, EnvironmentInheritancePolicy, EnvironmentPolicy,
        EphemeralDirectoryPolicy, NetworkPolicy, PlatformFamily, PointOfNoReturnPolicy,
        ProcessTreePolicy, RepositoryAccess, SandboxBackend, StdinPolicy,
        WorkingDirectoryPolicy, build_receipt as build_semantics,
    };
    use crate::cargo_context::{
        CargoBuildContextSpec, CargoOperation, FeatureSelection, PackageSelection,
        TargetSelection, ToolchainIdentity, build_document,
    };
    use crate::cargo_runtime_binding::{
        BoundRuntimeInput, CargoHomeEvidence, EphemeralDirectoryEvidence,
        RealizedAdapterConfiguration, RuntimeContextRealization, RuntimeEnvironmentValue,
        RuntimePreparationEvidence,
    };
    use crate::repository_effect_policy::EffectMode;
    use crate::repository_snapshot_receipt::{
        EntryKind, SnapshotEntry, SnapshotScope, SourceClass, UnknownSurface,
    };

    const SOURCE_UNKNOWN_CLOSED_BY_RUNTIME: &str =
        "environment_network_and_external_build_inputs_not_captured";

    fn digest(seed: char) -> String {
        assert!(seed.is_ascii());
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
                kind: SOURCE_UNKNOWN_CLOSED_BY_RUNTIME.into(),
                subject: None,
            }],
        };
        source.snapshot_id = source.computed_snapshot_id().unwrap();
        source.validate().unwrap();
        source
    }

    fn context() -> CargoBuildContextDocument {
        build_document(CargoBuildContextSpec {
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
        build_semantics(CargoAdapterSemanticsSpec {
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

    fn policy() -> EffectPolicySpec {
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

    fn runtime_spec() -> CargoRuntimeBindingSpec {
        CargoRuntimeBindingSpec {
            schema: INPUT_SCHEMA.into(),
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
                toolchain: context().context.toolchain,
                cargo_config_sha256: Some(digest('4')),
                rustflags_sha256: Some(digest('5')),
                rustdocflags_sha256: None,
                environment_fingerprints: BTreeMap::from([("RUSTFLAGS".into(), digest('5'))]),
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
        }
    }

    fn canonical_receipt() -> CargoRuntimeBindingReceipt {
        let mut policy = policy();
        let policy_id = validate_and_identify_policy(&mut policy).unwrap();
        build_receipt(
            runtime_spec(),
            source(),
            context(),
            semantics(),
            policy_id,
            policy,
        )
        .unwrap()
    }

    #[test]
    fn canonical_runtime_receipt_reloads_through_same_constructor() {
        let stored = canonical_receipt();
        assert_eq!(
            verify_stored(
                stored.clone(),
                source(),
                context(),
                semantics(),
                policy(),
            )
            .unwrap(),
            stored
        );
    }

    #[test]
    fn stale_runtime_id_rejects_even_when_payload_is_otherwise_valid() {
        let mut stored = canonical_receipt();
        stored.runtime_binding_id = digest('z');
        assert!(
            verify_stored(stored, source(), context(), semantics(), policy()).is_err()
        );
    }

    #[test]
    fn payload_mutation_with_old_identity_rejects() {
        let mut stored = canonical_receipt();
        stored.evidence.network_setup_receipt_sha256 = digest('y');
        assert!(
            verify_stored(stored, source(), context(), semantics(), policy()).is_err()
        );
    }

    #[test]
    fn source_substitution_rejects() {
        let stored = canonical_receipt();
        let mut other = source();
        other.entries[0].content_sha256 = Some(digest('x'));
        other.snapshot_id = other.computed_snapshot_id().unwrap();
        other.validate().unwrap();
        assert!(verify_stored(stored, other, context(), semantics(), policy()).is_err());
    }

    #[test]
    fn semantics_substitution_rejects() {
        let stored = canonical_receipt();
        let mut alternate = semantics();
        alternate.adapter_implementation_sha256 = digest('x');
        let alternate = build_semantics(CargoAdapterSemanticsSpec {
            schema: "symthaea.cargo-adapter-semantics-input.v1".into(),
            adapter_implementation_sha256: alternate.adapter_implementation_sha256,
            platform: alternate.platform,
            sandbox_backend: alternate.sandbox_backend,
            repository_access: alternate.repository_access,
            working_directory: alternate.working_directory,
            network: alternate.network,
            environment: alternate.environment,
            home: alternate.home,
            cargo_home: alternate.cargo_home,
            target_dir: alternate.target_dir,
            temp_dir: alternate.temp_dir,
            external_inputs: alternate.external_inputs,
            descendants: alternate.descendants,
            stdin: alternate.stdin,
            stdout: alternate.stdout,
            stderr: alternate.stderr,
            wall_clock_timeout_ms: alternate.wall_clock_timeout_ms,
            termination_grace_ms: alternate.termination_grace_ms,
            process_tree: alternate.process_tree,
            point_of_no_return: alternate.point_of_no_return,
        })
        .unwrap();
        assert!(verify_stored(stored, source(), context(), alternate, policy()).is_err());
    }
}
