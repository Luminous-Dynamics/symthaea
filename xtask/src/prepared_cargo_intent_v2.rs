use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::cargo_adapter_semantics::CargoAdapterSemanticsReceipt;
use crate::cargo_context::CargoBuildContextDocument;
use crate::cargo_execution_contract::{
    CargoExecutionIntent, CargoExecutionIntentSpec, build_intent,
};
use crate::cargo_runtime_binding::CargoRuntimeBindingReceipt;
use crate::git_worktree_state_receipt::{ValidatedGitWorktreeState, verify_for_source};
use crate::prepared_cargo_intent::{
    PreparedCargoExecutionIntent, PreparedCargoIntentSpec, build_prepared_intent,
};
use crate::prepared_cargo_intent_receipt::verify_local as verify_prepared_v1;
use crate::repository_effect_policy::EffectPolicySpec;
use crate::repository_materialization_receipt::RepositoryMaterializationReceipt;
use crate::repository_snapshot_receipt::ValidatedSnapshotReceipt;
use crate::tool_executable_attestation::ToolExecutableAttestationReceipt;

const INPUT_SCHEMA: &str = "symthaea.prepared-cargo-execution-intent-input.v2";
const RECEIPT_SCHEMA: &str = "symthaea.prepared-cargo-execution-intent.v2";
const HASH_DOMAIN: &[u8] = b"symthaea.prepared-cargo-execution-intent.v2\0";
const V1_INPUT_SCHEMA: &str = "symthaea.prepared-cargo-execution-intent-input.v1";
const LEGACY_INTENT_INPUT_SCHEMA: &str = "symthaea.cargo-execution-intent-input.v1";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct PreparedCargoIntentSpecV2 {
    pub schema: String,
    #[serde(default)]
    pub plan_id: Option<String>,
    #[serde(default)]
    pub transaction_id: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SourceGitStateObservationModel {
    SequentialNotAtomic,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct PreparedCargoExecutionIntentV2 {
    pub prepared_intent_id: String,
    pub schema: String,
    pub base_intent: CargoExecutionIntent,
    pub runtime_binding_id: String,
    pub materialization_receipt_id: String,
    pub tool_attestation_id: String,
    pub git_worktree_state_id: String,
    pub source_git_state_observation: SourceGitStateObservationModel,
}

#[derive(Serialize)]
struct PreparedIntentIdentity<'a> {
    schema: &'static str,
    base_intent_id: &'a str,
    runtime_binding_id: &'a str,
    materialization_receipt_id: &'a str,
    tool_attestation_id: &'a str,
    git_worktree_state_id: &'a str,
    source_git_state_observation: SourceGitStateObservationModel,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_prepared_intent_v2(
    mut spec: PreparedCargoIntentSpecV2,
    source: ValidatedSnapshotReceipt,
    context: CargoBuildContextDocument,
    semantics: CargoAdapterSemanticsReceipt,
    effect_policy: EffectPolicySpec,
    runtime: CargoRuntimeBindingReceipt,
    materialization: RepositoryMaterializationReceipt,
    tool_attestation: ToolExecutableAttestationReceipt,
    git_state: ValidatedGitWorktreeState,
) -> anyhow::Result<PreparedCargoExecutionIntentV2> {
    if spec.schema != INPUT_SCHEMA {
        bail!("unsupported prepared Cargo intent v2 input schema: {}", spec.schema);
    }
    normalize_optional_digest("plan_id", &mut spec.plan_id)?;
    normalize_optional_digest("transaction_id", &mut spec.transaction_id)?;

    let prepared_v1 = build_prepared_intent(
        PreparedCargoIntentSpec {
            schema: V1_INPUT_SCHEMA.into(),
            plan_id: spec.plan_id,
            transaction_id: spec.transaction_id,
        },
        source,
        context,
        semantics,
        effect_policy,
        runtime,
        materialization,
        tool_attestation,
    )?;
    bind_verified_git_state(prepared_v1, git_state)
}

fn bind_verified_git_state(
    prepared_v1: PreparedCargoExecutionIntent,
    git_state: ValidatedGitWorktreeState,
) -> anyhow::Result<PreparedCargoExecutionIntentV2> {
    let prepared_v1 = verify_prepared_v1(prepared_v1)?;
    let git_state = verify_for_source(
        git_state,
        &prepared_v1.base_intent.repository_source_before,
    )?;

    let old_base = &prepared_v1.base_intent;
    let new_base = build_intent(
        CargoExecutionIntentSpec {
            schema: LEGACY_INTENT_INPUT_SCHEMA.into(),
            git_worktree_state_before: Some(git_state.state_id.clone()),
            build_context_id: old_base.build_context_id.clone(),
            invocation_id: old_base.invocation_id.clone(),
            plan_id: old_base.plan_id.clone(),
            transaction_id: old_base.transaction_id.clone(),
            adapter_semantics_digest: old_base.adapter_semantics_digest.clone(),
        },
        &old_base.repository_source_before,
        &old_base.effect_policy_id,
    )?;

    if new_base.repository_source_before != old_base.repository_source_before
        || new_base.build_context_id != old_base.build_context_id
        || new_base.invocation_id != old_base.invocation_id
        || new_base.effect_policy_id != old_base.effect_policy_id
        || new_base.plan_id != old_base.plan_id
        || new_base.transaction_id != old_base.transaction_id
        || new_base.adapter_semantics_digest != old_base.adapter_semantics_digest
    {
        bail!("Git-state binding changed a non-Git Cargo execution-intent field");
    }

    let source_git_state_observation = SourceGitStateObservationModel::SequentialNotAtomic;
    let identity = PreparedIntentIdentity {
        schema: RECEIPT_SCHEMA,
        base_intent_id: &new_base.intent_id,
        runtime_binding_id: &prepared_v1.runtime_binding_id,
        materialization_receipt_id: &prepared_v1.materialization_receipt_id,
        tool_attestation_id: &prepared_v1.tool_attestation_id,
        git_worktree_state_id: &git_state.state_id,
        source_git_state_observation,
    };
    let bytes = serde_json::to_vec(&identity)
        .context("serialize Git-bound prepared Cargo execution intent")?;
    let prepared_intent_id = domain_sha256(HASH_DOMAIN, &bytes);

    Ok(PreparedCargoExecutionIntentV2 {
        prepared_intent_id,
        schema: RECEIPT_SCHEMA.into(),
        base_intent: new_base,
        runtime_binding_id: prepared_v1.runtime_binding_id,
        materialization_receipt_id: prepared_v1.materialization_receipt_id,
        tool_attestation_id: prepared_v1.tool_attestation_id,
        git_worktree_state_id: git_state.state_id,
        source_git_state_observation,
    })
}

fn normalize_optional_digest(name: &str, value: &mut Option<String>) -> anyhow::Result<()> {
    if let Some(value) = value {
        if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            bail!("{name} must be a 64-character SHA-256 hex digest");
        }
        value.make_ascii_lowercase();
    }
    Ok(())
}

fn domain_sha256(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cargo_execution_contract::build_intent;
    use crate::git_worktree_state_receipt::{
        FsmonitorEnvironment, IgnoreEnvironment, SparseCheckoutState,
    };
    use crate::prepared_cargo_intent::{
        GitWorktreeStateBinding, PreparedCargoExecutionIntent,
    };

    const V1_RECEIPT_SCHEMA: &str = "symthaea.prepared-cargo-execution-intent.v1";
    const V1_HASH_DOMAIN: &[u8] = b"symthaea.prepared-cargo-execution-intent.v1\0";

    #[derive(Serialize)]
    struct V1Identity<'a> {
        schema: &'static str,
        base_intent_id: &'a str,
        runtime_binding_id: &'a str,
        materialization_receipt_id: &'a str,
        tool_attestation_id: &'a str,
        git_worktree_state_binding: &'a GitWorktreeStateBinding,
    }

    fn digest(seed: char) -> String {
        assert!(seed.is_ascii());
        format!("{:02x}", u32::from(seed)).repeat(32)
    }

    fn prepared_v1() -> PreparedCargoExecutionIntent {
        let base_intent = build_intent(
            CargoExecutionIntentSpec {
                schema: LEGACY_INTENT_INPUT_SCHEMA.into(),
                git_worktree_state_before: None,
                build_context_id: digest('b'),
                invocation_id: digest('c'),
                plan_id: Some(digest('d')),
                transaction_id: Some(digest('e')),
                adapter_semantics_digest: digest('f'),
            },
            &digest('a'),
            &digest('1'),
        )
        .unwrap();
        let runtime_binding_id = digest('2');
        let materialization_receipt_id = digest('3');
        let tool_attestation_id = digest('4');
        let git_worktree_state_binding = GitWorktreeStateBinding::NotConverged;
        let identity = V1Identity {
            schema: V1_RECEIPT_SCHEMA,
            base_intent_id: &base_intent.intent_id,
            runtime_binding_id: &runtime_binding_id,
            materialization_receipt_id: &materialization_receipt_id,
            tool_attestation_id: &tool_attestation_id,
            git_worktree_state_binding: &git_worktree_state_binding,
        };
        let prepared_intent_id = domain_sha256(
            V1_HASH_DOMAIN,
            &serde_json::to_vec(&identity).unwrap(),
        );
        PreparedCargoExecutionIntent {
            prepared_intent_id,
            schema: V1_RECEIPT_SCHEMA.into(),
            base_intent,
            runtime_binding_id,
            materialization_receipt_id,
            tool_attestation_id,
            git_worktree_state_binding,
        }
    }

    fn git_state() -> ValidatedGitWorktreeState {
        let mut state = ValidatedGitWorktreeState {
            state_id: digest('0'),
            schema: "symthaea.git-worktree-state.v1".into(),
            repository_source_snapshot_id: digest('a'),
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

    #[test]
    fn git_state_binding_changes_only_git_dimension_of_base_intent() {
        let old = prepared_v1();
        let state = git_state();
        let expected_state_id = state.state_id.clone();
        let upgraded = bind_verified_git_state(old.clone(), state).unwrap();

        assert_eq!(
            upgraded.base_intent.git_worktree_state_before.as_deref(),
            Some(expected_state_id.as_str())
        );
        assert_eq!(upgraded.git_worktree_state_id, expected_state_id);
        assert_ne!(upgraded.base_intent.intent_id, old.base_intent.intent_id);
        assert_eq!(
            upgraded.base_intent.repository_source_before,
            old.base_intent.repository_source_before
        );
        assert_eq!(upgraded.base_intent.build_context_id, old.base_intent.build_context_id);
        assert_eq!(upgraded.base_intent.invocation_id, old.base_intent.invocation_id);
        assert_eq!(upgraded.base_intent.effect_policy_id, old.base_intent.effect_policy_id);
        assert_eq!(
            upgraded.source_git_state_observation,
            SourceGitStateObservationModel::SequentialNotAtomic
        );
    }

    #[test]
    fn source_substitution_rejects() {
        let mut state = git_state();
        state.repository_source_snapshot_id = digest('9');
        state.state_id = state.computed_state_id().unwrap();
        assert!(bind_verified_git_state(prepared_v1(), state).is_err());
    }

    #[test]
    fn mutated_v1_prepared_subject_rejects_before_upgrade() {
        let mut prepared = prepared_v1();
        prepared.runtime_binding_id = digest('9');
        assert!(bind_verified_git_state(prepared, git_state()).is_err());
    }

    #[test]
    fn different_git_state_changes_v2_identity() {
        let a = bind_verified_git_state(prepared_v1(), git_state()).unwrap();
        let mut alternate = git_state();
        alternate.fsmonitor_environment.configured = true;
        alternate.fsmonitor_environment.config_value_sha256 = Some(digest('8'));
        alternate.state_id = alternate.computed_state_id().unwrap();
        alternate.validate().unwrap();
        let b = bind_verified_git_state(prepared_v1(), alternate).unwrap();
        assert_ne!(a.prepared_intent_id, b.prepared_intent_id);
        assert_ne!(a.git_worktree_state_id, b.git_worktree_state_id);
    }

    #[test]
    fn input_schema_rejects_caller_minted_git_state_id() {
        let json = format!(
            "{{\"schema\":\"{INPUT_SCHEMA}\",\"plan_id\":\"{}\",\"git_worktree_state_id\":\"{}\"}}",
            digest('d'),
            digest('5')
        );
        assert!(serde_json::from_str::<PreparedCargoIntentSpecV2>(&json).is_err());
    }
}
