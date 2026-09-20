#![allow(dead_code)]

#[path = "../src/cargo_adapter_semantics.rs"]
mod cargo_adapter_semantics;
#[path = "../src/cargo_context.rs"]
mod cargo_context;
#[path = "../src/cargo_context_verify.rs"]
mod cargo_context_verify;
#[path = "../src/cargo_execution_contract.rs"]
mod cargo_execution_contract;
#[path = "../src/cargo_runtime_binding.rs"]
mod cargo_runtime_binding;
#[path = "../src/git_worktree_state_receipt.rs"]
mod git_worktree_state_receipt;
#[path = "../src/pre_spawn_freshness.rs"]
mod pre_spawn_freshness;
#[path = "../src/pre_spawn_freshness_v2.rs"]
mod pre_spawn_freshness_v2;
#[path = "../src/prepared_cargo_intent_v2.rs"]
mod prepared_cargo_intent_v2;
#[path = "../src/prepared_cargo_intent_v2_receipt.rs"]
mod prepared_cargo_intent_v2_receipt;
#[path = "../src/prepared_cargo_intent.rs"]
mod prepared_cargo_intent;
#[path = "../src/prepared_cargo_intent_receipt.rs"]
mod prepared_cargo_intent_receipt;
#[path = "../src/repository_materialization_receipt.rs"]
mod repository_materialization_receipt;
#[path = "../src/repository_snapshot_receipt.rs"]
mod repository_snapshot_receipt;
#[path = "../src/repository_snapshot_diff.rs"]
mod repository_snapshot_diff;
#[path = "../src/repository_effect_policy.rs"]
mod repository_effect_policy;
#[path = "../src/tool_executable_attestation.rs"]
mod tool_executable_attestation;

use std::collections::BTreeMap;

use cargo_adapter_semantics::{
    BoundInputPolicy, CapturePolicy, CargoHomePolicy, DescendantPolicy,
    EnvironmentInheritancePolicy, EphemeralDirectoryPolicy, NetworkPolicy, PlatformFamily,
    PointOfNoReturnPolicy, ProcessTreePolicy, RepositoryAccess, SandboxBackend, StdinPolicy,
    WorkingDirectoryPolicy,
};
use cargo_context::ToolchainIdentity;
use cargo_execution_contract::{build_intent, CargoExecutionIntentSpec};
use cargo_runtime_binding::{
    BoundRuntimeInput, CargoHomeEvidence, CargoRuntimeBindingReceipt, EphemeralDirectoryEvidence,
    RealizedAdapterConfiguration, RuntimeContextRealization, RuntimePreparationEvidence,
};
use git_worktree_state_receipt::{
    FsmonitorEnvironment, GitIndexFlags, IgnoreEnvironment, SparseCheckoutState,
    ValidatedGitWorktreeState,
};
use pre_spawn_freshness::{
    build_receipt as build_v1, expected_projection, PreSpawnFreshnessObservation,
};
use pre_spawn_freshness_v2::{
    build_receipt_v2, expected_projection_v2, PreSpawnFreshnessObservationV2,
};
use prepared_cargo_intent_v2::{
    PreparedCargoExecutionIntentV2, SourceGitStateObservationModel,
};
use repository_materialization_receipt::RepositoryMaterializationReceipt;
use serde::Serialize;
use sha2::{Digest, Sha256};
use tool_executable_attestation::{ExecutableProbeEvidence, ToolExecutableAttestationReceipt};

const PREPARED_SCHEMA: &str = "symthaea.prepared-cargo-execution-intent.v2";
const PREPARED_DOMAIN: &[u8] = b"symthaea.prepared-cargo-execution-intent.v2\0";
const INTENT_INPUT_SCHEMA: &str = "symthaea.cargo-execution-intent-input.v1";

#[derive(Serialize)]
struct PreparedIdentity<'a> {
    schema: &'static str,
    base_intent_id: &'a str,
    runtime_binding_id: &'a str,
    materialization_receipt_id: &'a str,
    tool_attestation_id: &'a str,
    git_worktree_state_id: &'a str,
    source_git_state_observation: SourceGitStateObservationModel,
}

fn digest(seed: char) -> String {
    assert!(seed.is_ascii());
    format!("{:02x}", u32::from(seed)).repeat(32)
}

fn domain_sha256(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").unwrap();
    }
    out
}

fn dir(id: char) -> EphemeralDirectoryEvidence {
    EphemeralDirectoryEvidence {
        instance_id: digest(id),
        empty_at_start_receipt_sha256: digest(char::from_u32(id as u32 + 1).unwrap()),
        setup_receipt_sha256: digest(char::from_u32(id as u32 + 2).unwrap()),
    }
}

fn runtime() -> CargoRuntimeBindingReceipt {
    CargoRuntimeBindingReceipt {
        runtime_binding_id: digest('1'),
        schema: "symthaea.cargo-runtime-binding.v1".into(),
        repository_source_before: digest('2'),
        context_id: digest('3'),
        invocation_id: digest('4'),
        adapter_semantics_id: digest('5'),
        effect_policy_id: digest('6'),
        realized: RealizedAdapterConfiguration {
            platform: PlatformFamily::Linux,
            adapter_implementation_sha256: digest('7'),
            sandbox_backend: SandboxBackend {
                name: "bubblewrap".into(),
                implementation_sha256: digest('8'),
            },
            repository_access: RepositoryAccess::IsolatedStagingTree,
            working_directory: WorkingDirectoryPolicy::StagingRoot,
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
            toolchain: ToolchainIdentity {
                cargo_version: "cargo 1.96.0".into(),
                rustc_version: "rustc 1.96.0".into(),
                host_triple: "x86_64-unknown-linux-gnu".into(),
                toolchain_name: Some("1.96.0".into()),
            },
            cargo_config_sha256: Some(digest('9')),
            rustflags_sha256: None,
            rustdocflags_sha256: None,
            environment_fingerprints: BTreeMap::new(),
            configuration_setup_receipt_sha256: digest('c'),
        },
        evidence: RuntimePreparationEvidence {
            runtime_instance_id: digest('d'),
            sandbox_setup_receipt_sha256: digest('e'),
            repository_materialization_id: digest('f'),
            repository_setup_receipt_sha256: digest('a'),
            working_directory_subject_id: digest('f'),
            network_setup_receipt_sha256: digest('b'),
            environment_clear_receipt_sha256: digest('c'),
            ambient_environment: BTreeMap::new(),
            home: dir('g'),
            cargo_home: CargoHomeEvidence::ReadOnlyPrefetched {
                content_id: digest('h'),
                setup_receipt_sha256: digest('i'),
            },
            target_dir: dir('j'),
            temp_dir: dir('k'),
            bound_inputs: vec![
                BoundRuntimeInput {
                    role: "cargo_executable".into(),
                    content_id: digest('l'),
                    read_only_setup_receipt_sha256: digest('m'),
                },
                BoundRuntimeInput {
                    role: "rustc_executable".into(),
                    content_id: digest('n'),
                    read_only_setup_receipt_sha256: digest('o'),
                },
            ],
            io_setup_receipt_sha256: digest('p'),
            process_tree_setup_receipt_sha256: digest('q'),
        },
    }
}

fn materialization() -> RepositoryMaterializationReceipt {
    RepositoryMaterializationReceipt {
        materialization_receipt_id: digest('r'),
        schema: "symthaea.repository-materialization.v1".into(),
        source_snapshot_id: digest('2'),
        runtime_instance_id: digest('d'),
        kind: RepositoryAccess::IsolatedStagingTree,
        materialization_id: digest('f'),
        working_directory_subject_id: digest('f'),
        repository_setup_receipt_sha256: digest('a'),
        derivation_receipt_sha256: digest('s'),
        staged_content_id: Some(digest('z')),
    }
}

fn tools() -> ToolExecutableAttestationReceipt {
    ToolExecutableAttestationReceipt {
        tool_attestation_id: digest('t'),
        schema: "symthaea.tool-executable-attestation.v1".into(),
        context_id: digest('3'),
        invocation_id: digest('4'),
        cargo: ExecutableProbeEvidence {
            executable_content_id: digest('l'),
            probe_transcript_sha256: digest('u'),
            probe_implementation_sha256: digest('v'),
            reported_version: "cargo 1.96.0".into(),
            reported_host_triple: "x86_64-unknown-linux-gnu".into(),
        },
        rustc: ExecutableProbeEvidence {
            executable_content_id: digest('n'),
            probe_transcript_sha256: digest('w'),
            probe_implementation_sha256: digest('v'),
            reported_version: "rustc 1.96.0".into(),
            reported_host_triple: "x86_64-unknown-linux-gnu".into(),
        },
    }
}

fn git_state() -> ValidatedGitWorktreeState {
    let mut state = ValidatedGitWorktreeState {
        state_id: String::new(),
        schema: "symthaea.git-worktree-state.v1".into(),
        repository_source_snapshot_id: digest('2'),
        git_version: "git version 2.50.0".into(),
        index_flags: vec![],
        sparse_checkout: SparseCheckoutState {
            enabled: false,
            cone_mode_configured: false,
            sparse_index_configured: false,
            specification_sha256: None,
        },
        ignore_environment: IgnoreEnvironment {
            info_exclude_present: false,
            info_exclude_sha256: None,
            global_excludes_configured: false,
            global_excludes_present: false,
            global_excludes_sha256: None,
        },
        fsmonitor_environment: FsmonitorEnvironment {
            configured: false,
            config_value_sha256: None,
        },
    };
    state.state_id = state.computed_state_id().unwrap();
    state.validate().unwrap();
    state
}

#[allow(clippy::too_many_arguments)]
fn prepared_for(
    git_id: String,
    source: String,
    context: String,
    invocation: String,
    effect_policy: String,
    semantics: String,
    runtime_id: String,
    materialization_id: String,
    tool_id: String,
) -> PreparedCargoExecutionIntentV2 {
    let base_intent = build_intent(
        CargoExecutionIntentSpec {
            schema: INTENT_INPUT_SCHEMA.into(),
            git_worktree_state_before: Some(git_id.clone()),
            build_context_id: context,
            invocation_id: invocation,
            plan_id: Some(digest('A')),
            transaction_id: Some(digest('B')),
            adapter_semantics_digest: semantics,
        },
        &source,
        &effect_policy,
    )
    .unwrap();
    let observation_model = SourceGitStateObservationModel::SequentialNotAtomic;
    let identity = PreparedIdentity {
        schema: PREPARED_SCHEMA,
        base_intent_id: &base_intent.intent_id,
        runtime_binding_id: &runtime_id,
        materialization_receipt_id: &materialization_id,
        tool_attestation_id: &tool_id,
        git_worktree_state_id: &git_id,
        source_git_state_observation: observation_model,
    };
    let prepared_intent_id = domain_sha256(
        PREPARED_DOMAIN,
        &serde_json::to_vec(&identity).unwrap(),
    );
    PreparedCargoExecutionIntentV2 {
        prepared_intent_id,
        schema: PREPARED_SCHEMA.into(),
        base_intent,
        runtime_binding_id: runtime_id,
        materialization_receipt_id: materialization_id,
        tool_attestation_id: tool_id,
        git_worktree_state_id: git_id,
        source_git_state_observation: observation_model,
    }
}

fn prepared() -> PreparedCargoExecutionIntentV2 {
    let git = git_state();
    prepared_for(
        git.state_id,
        digest('2'),
        digest('3'),
        digest('4'),
        digest('6'),
        digest('5'),
        digest('1'),
        digest('r'),
        digest('t'),
    )
}

fn observation() -> PreSpawnFreshnessObservationV2 {
    let prepared = prepared();
    let expected = expected_projection_v2(
        &prepared,
        &runtime(),
        &materialization(),
        &tools(),
        git_state(),
    )
    .unwrap();
    PreSpawnFreshnessObservationV2 {
        schema: "symthaea.pre-spawn-freshness-observation.v2".into(),
        probe_implementation_sha256: digest('C'),
        probe_transcript_sha256: digest('D'),
        observed: expected,
    }
}

fn rebuild_git(mut state: ValidatedGitWorktreeState) -> ValidatedGitWorktreeState {
    state.state_id = state.computed_state_id().unwrap();
    state.validate().unwrap();
    state
}

#[test]
fn exact_complete_prepared_subject_produces_v2_receipt() {
    let receipt = build_receipt_v2(
        observation(),
        prepared(),
        runtime(),
        materialization(),
        tools(),
        git_state(),
    )
    .unwrap();
    assert_eq!(receipt.schema, "symthaea.pre-spawn-freshness.v2");
    assert_eq!(receipt.prepared_intent_id, prepared().prepared_intent_id);
    assert_eq!(receipt.runtime_binding_id, digest('1'));
    assert_eq!(receipt.materialization_receipt_id, digest('r'));
    assert_eq!(receipt.tool_attestation_id, digest('t'));
    assert_eq!(receipt.git_worktree_state_id, git_state().state_id);
    assert_eq!(
        receipt.observed.source_git_state_observation,
        SourceGitStateObservationModel::SequentialNotAtomic
    );
}

#[test]
fn prepared_constituent_substitution_rejects() {
    let git = git_state();
    for prepared in [
        prepared_for(
            git.state_id.clone(),
            digest('2'),
            digest('3'),
            digest('4'),
            digest('6'),
            digest('5'),
            digest('0'),
            digest('r'),
            digest('t'),
        ),
        prepared_for(
            git.state_id.clone(),
            digest('2'),
            digest('3'),
            digest('4'),
            digest('6'),
            digest('5'),
            digest('1'),
            digest('0'),
            digest('t'),
        ),
        prepared_for(
            git.state_id.clone(),
            digest('2'),
            digest('3'),
            digest('4'),
            digest('6'),
            digest('5'),
            digest('1'),
            digest('r'),
            digest('0'),
        ),
    ] {
        assert!(
            expected_projection_v2(
                &prepared,
                &runtime(),
                &materialization(),
                &tools(),
                git.clone(),
            )
            .is_err()
        );
    }
}

#[test]
fn prepared_base_subject_substitution_rejects() {
    let git = git_state();
    let cases = [
        (
            digest('0'),
            digest('3'),
            digest('4'),
            digest('6'),
            digest('5'),
        ),
        (
            digest('2'),
            digest('0'),
            digest('4'),
            digest('6'),
            digest('5'),
        ),
        (
            digest('2'),
            digest('3'),
            digest('0'),
            digest('6'),
            digest('5'),
        ),
        (
            digest('2'),
            digest('3'),
            digest('4'),
            digest('0'),
            digest('5'),
        ),
        (
            digest('2'),
            digest('3'),
            digest('4'),
            digest('6'),
            digest('0'),
        ),
    ];
    for (source, context, invocation, effect, semantics) in cases {
        let prepared = prepared_for(
            git.state_id.clone(),
            source,
            context,
            invocation,
            effect,
            semantics,
            digest('1'),
            digest('r'),
            digest('t'),
        );
        assert!(
            expected_projection_v2(
                &prepared,
                &runtime(),
                &materialization(),
                &tools(),
                git.clone(),
            )
            .is_err()
        );
    }
}

#[test]
fn git_state_bound_to_another_source_rejects() {
    let mut drifted = git_state();
    drifted.repository_source_snapshot_id = digest('0');
    let drifted = rebuild_git(drifted);
    assert!(
        expected_projection_v2(
            &prepared(),
            &runtime(),
            &materialization(),
            &tools(),
            drifted,
        )
        .is_err()
    );
}

#[test]
fn valid_git_operational_drift_rejects_even_when_source_id_is_unchanged() {
    let base = git_state();

    let mut flags = base.clone();
    flags.index_flags = vec![GitIndexFlags {
        path: "src/lib.rs".into(),
        skip_worktree: true,
        assume_unchanged: false,
        fsmonitor_valid: false,
    }];

    let mut sparse = base.clone();
    sparse.sparse_checkout.enabled = true;
    sparse.sparse_checkout.cone_mode_configured = true;
    sparse.sparse_checkout.specification_sha256 = Some(digest('E'));

    let mut ignored = base.clone();
    ignored.ignore_environment.info_exclude_present = true;
    ignored.ignore_environment.info_exclude_sha256 = Some(digest('F'));

    let mut fsmonitor = base;
    fsmonitor.fsmonitor_environment.configured = true;
    fsmonitor.fsmonitor_environment.config_value_sha256 = Some(digest('G'));

    for drifted in [flags, sparse, ignored, fsmonitor] {
        let drifted = rebuild_git(drifted);
        assert_eq!(drifted.repository_source_snapshot_id, digest('2'));
        assert_ne!(drifted.state_id, git_state().state_id);
        assert!(
            expected_projection_v2(
                &prepared(),
                &runtime(),
                &materialization(),
                &tools(),
                drifted,
            )
            .is_err()
        );
    }
}

#[test]
fn observed_projection_tampering_rejects() {
    let mut observed = observation();
    observed.observed.git_worktree_state_id = digest('0');
    assert!(
        build_receipt_v2(
            observed,
            prepared(),
            runtime(),
            materialization(),
            tools(),
            git_state(),
        )
        .is_err()
    );
}

#[test]
fn v1_remains_unchanged_and_v2_has_a_distinct_identity() {
    let v1_observation = PreSpawnFreshnessObservation {
        schema: "symthaea.pre-spawn-freshness-observation.v1".into(),
        probe_implementation_sha256: digest('C'),
        probe_transcript_sha256: digest('D'),
        observed: expected_projection(&runtime(), &materialization(), &tools()).unwrap(),
    };
    let v1 = build_v1(v1_observation, runtime(), materialization(), tools()).unwrap();
    let v2 = build_receipt_v2(
        observation(),
        prepared(),
        runtime(),
        materialization(),
        tools(),
        git_state(),
    )
    .unwrap();
    assert_eq!(v1.schema, "symthaea.pre-spawn-freshness.v1");
    assert_eq!(v2.schema, "symthaea.pre-spawn-freshness.v2");
    assert_ne!(v1.freshness_receipt_id, v2.freshness_receipt_id);
}

#[test]
fn probe_transcript_is_v2_identity_material() {
    let a = build_receipt_v2(
        observation(),
        prepared(),
        runtime(),
        materialization(),
        tools(),
        git_state(),
    )
    .unwrap();
    let mut other = observation();
    other.probe_transcript_sha256 = digest('H');
    let b = build_receipt_v2(
        other,
        prepared(),
        runtime(),
        materialization(),
        tools(),
        git_state(),
    )
    .unwrap();
    assert_ne!(a.freshness_receipt_id, b.freshness_receipt_id);
}

#[test]
fn observation_schema_rejects_unknown_fields() {
    let mut value = serde_json::to_value(observation()).unwrap();
    value.as_object_mut().unwrap().insert(
        "expected_git_state_id".into(),
        serde_json::json!(digest('0')),
    );
    assert!(serde_json::from_value::<PreSpawnFreshnessObservationV2>(value).is_err());
}
