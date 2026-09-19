use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

use crate::cargo_adapter_semantics::CargoAdapterSemanticsReceipt;
use crate::cargo_execution_contract::{
    CargoExecutionIntent, CargoExecutionIntentSpec, build_intent,
};
use crate::repository_effect_policy::{EffectPolicySpec, validate_and_identify_policy};
use crate::repository_snapshot_receipt::ValidatedSnapshotReceipt;

const INTENT_INPUT_SCHEMA_V2: &str = "symthaea.cargo-execution-intent-input.v2";
const LEGACY_INTENT_INPUT_SCHEMA_V1: &str = "symthaea.cargo-execution-intent-input.v1";

/// User-facing v2 intent input deliberately contains no adapter-semantics
/// digest field. The adapter identity is derived only from a separately
/// validated `CargoAdapterSemanticsReceipt`.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct CargoExecutionIntentSpecV2 {
    pub schema: String,
    #[serde(default)]
    pub git_worktree_state_before: Option<String>,
    pub build_context_id: String,
    pub invocation_id: String,
    #[serde(default)]
    pub plan_id: Option<String>,
    #[serde(default)]
    pub transaction_id: Option<String>,
}

pub(crate) fn run(
    spec_path: &Path,
    pre_snapshot_path: &Path,
    effect_policy_path: &Path,
    adapter_semantics_path: &Path,
    output: Option<PathBuf>,
) -> anyhow::Result<()> {
    // Keep the legacy v1 file-oriented helper reachable while frozen test
    // lineages still depend on its underlying contract. It is intentionally
    // not exposed as a CLI path anymore.
    let _legacy_run_intent_compat: fn(
        &Path,
        &Path,
        &Path,
        Option<PathBuf>,
    ) -> anyhow::Result<()> = crate::cargo_execution_contract::run_intent;

    let bytes = fs::read(spec_path)
        .with_context(|| format!("read Cargo execution intent v2 spec {}", spec_path.display()))?;
    let spec: CargoExecutionIntentSpecV2 = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse Cargo execution intent v2 spec {}", spec_path.display()))?;

    let pre = ValidatedSnapshotReceipt::load(pre_snapshot_path)?;
    let policy_id = load_effect_policy_id(effect_policy_path)?;
    let semantics = CargoAdapterSemanticsReceipt::load(adapter_semantics_path)?;

    let intent = build_intent_from_semantics(spec, &pre.snapshot_id, &policy_id, semantics)?;
    write_json("Cargo execution intent", &intent, output)
}

pub(crate) fn build_intent_from_semantics(
    spec: CargoExecutionIntentSpecV2,
    repository_source_before: &str,
    effect_policy_id: &str,
    mut semantics: CargoAdapterSemanticsReceipt,
) -> anyhow::Result<CargoExecutionIntent> {
    if spec.schema != INTENT_INPUT_SCHEMA_V2 {
        bail!("unsupported Cargo execution intent v2 input schema: {}", spec.schema);
    }

    // Revalidate even when the caller already loaded the receipt. This keeps the
    // constructor safe for direct in-crate use and prevents a mutated receipt
    // object from becoming an intent authority claim.
    semantics.validate()?;

    let legacy = CargoExecutionIntentSpec {
        schema: LEGACY_INTENT_INPUT_SCHEMA_V1.into(),
        git_worktree_state_before: spec.git_worktree_state_before,
        build_context_id: spec.build_context_id,
        invocation_id: spec.invocation_id,
        plan_id: spec.plan_id,
        transaction_id: spec.transaction_id,
        adapter_semantics_digest: semantics.adapter_semantics_id,
    };

    build_intent(legacy, repository_source_before, effect_policy_id)
}

fn load_effect_policy_id(path: &Path) -> anyhow::Result<String> {
    let bytes = fs::read(path)
        .with_context(|| format!("read repository effect policy {}", path.display()))?;
    let mut policy: EffectPolicySpec = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse repository effect policy {}", path.display()))?;
    validate_and_identify_policy(&mut policy)
}

fn write_json<T: Serialize>(label: &str, value: &T, output: Option<PathBuf>) -> anyhow::Result<()> {
    let mut rendered = serde_json::to_string_pretty(value)?;
    rendered.push('\n');
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create {label} output directory {}", parent.display()))?;
        }
        fs::write(&path, rendered)
            .with_context(|| format!("write {label} {}", path.display()))?;
        println!("{label} written to {}", path.display());
    } else {
        print!("{rendered}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cargo_adapter_semantics::{
        BoundInputPolicy, CapturePolicy, CargoAdapterSemanticsSpec, CargoHomePolicy,
        DescendantPolicy, EnvironmentInheritancePolicy, EnvironmentPolicy,
        EphemeralDirectoryPolicy, NetworkPolicy, PlatformFamily, PointOfNoReturnPolicy,
        ProcessTreePolicy, RepositoryAccess, SandboxBackend, StdinPolicy,
        WorkingDirectoryPolicy, build_receipt,
    };

    fn digest(byte: char) -> String {
        byte.to_string().repeat(64)
    }

    fn intent_spec() -> CargoExecutionIntentSpecV2 {
        CargoExecutionIntentSpecV2 {
            schema: INTENT_INPUT_SCHEMA_V2.into(),
            git_worktree_state_before: Some(digest('9')),
            build_context_id: digest('b'),
            invocation_id: digest('c'),
            plan_id: Some(digest('e')),
            transaction_id: Some(digest('f')),
        }
    }

    fn semantics_spec() -> CargoAdapterSemanticsSpec {
        CargoAdapterSemanticsSpec {
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
        }
    }

    #[test]
    fn v2_input_rejects_arbitrary_adapter_digest_field() {
        let json = format!(
            "{{\"schema\":\"{INTENT_INPUT_SCHEMA_V2}\",\"git_worktree_state_before\":\"{}\",\"build_context_id\":\"{}\",\"invocation_id\":\"{}\",\"plan_id\":\"{}\",\"transaction_id\":\"{}\",\"adapter_semantics_digest\":\"{}\"}}",
            digest('9'),
            digest('b'),
            digest('c'),
            digest('e'),
            digest('f'),
            digest('1')
        );
        assert!(serde_json::from_str::<CargoExecutionIntentSpecV2>(&json).is_err());
    }

    #[test]
    fn validated_semantics_identity_is_bound_into_intent() {
        let semantics = build_receipt(semantics_spec()).unwrap();
        let semantics_id = semantics.adapter_semantics_id.clone();
        let intent = build_intent_from_semantics(
            intent_spec(),
            &digest('a'),
            &digest('d'),
            semantics,
        )
        .unwrap();
        assert_eq!(intent.adapter_semantics_digest, semantics_id);
    }

    #[test]
    fn semantics_substitution_changes_intent_identity() {
        let denied = build_receipt(semantics_spec()).unwrap();
        let mut alternate = semantics_spec();
        alternate.network = NetworkPolicy::LoopbackOnly;
        let loopback = build_receipt(alternate).unwrap();

        let a = build_intent_from_semantics(
            intent_spec(),
            &digest('a'),
            &digest('d'),
            denied,
        )
        .unwrap();
        let b = build_intent_from_semantics(
            intent_spec(),
            &digest('a'),
            &digest('d'),
            loopback,
        )
        .unwrap();
        assert_ne!(a.intent_id, b.intent_id);
    }

    #[test]
    fn mutated_semantics_receipt_is_rejected_before_intent_minting() {
        let mut semantics = build_receipt(semantics_spec()).unwrap();
        semantics.wall_clock_timeout_ms += 1;
        assert!(
            build_intent_from_semantics(
                intent_spec(),
                &digest('a'),
                &digest('d'),
                semantics,
            )
            .is_err()
        );
    }
}
