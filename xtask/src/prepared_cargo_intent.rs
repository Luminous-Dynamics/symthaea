use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::cargo_adapter_semantics::CargoAdapterSemanticsReceipt;
use crate::cargo_context::CargoBuildContextDocument;
use crate::cargo_context_verify::verify_bytes as verify_context_bytes;
use crate::cargo_execution_contract::{CargoExecutionIntent, CargoExecutionIntentSpec, build_intent};
use crate::cargo_runtime_binding::{
    CargoRuntimeBindingReceipt, CargoRuntimeBindingSpec, build_receipt as rebuild_runtime_binding,
};
use crate::pre_spawn_freshness::expected_projection;
use crate::repository_effect_policy::{EffectPolicySpec, validate_and_identify_policy};
use crate::repository_materialization_receipt::{
    RepositoryMaterializationReceipt, verify_stored as verify_materialization,
};
use crate::repository_snapshot_receipt::ValidatedSnapshotReceipt;
use crate::tool_executable_attestation::{
    ToolExecutableAttestationReceipt, verify_stored as verify_tool_attestation,
};

const INPUT_SCHEMA: &str = "symthaea.prepared-cargo-execution-intent-input.v1";
const RECEIPT_SCHEMA: &str = "symthaea.prepared-cargo-execution-intent.v1";
const HASH_DOMAIN: &[u8] = b"symthaea.prepared-cargo-execution-intent.v1\0";
const LEGACY_INTENT_INPUT_SCHEMA: &str = "symthaea.cargo-execution-intent-input.v1";
const RUNTIME_INPUT_SCHEMA: &str = "symthaea.cargo-runtime-binding-input.v1";

/// V1 deliberately contains no engineering subject IDs. Every repository,
/// Cargo, adapter, runtime, materialization, and executable identity is derived
/// from a separately validated owning document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct PreparedCargoIntentSpec {
    pub schema: String,
    #[serde(default)]
    pub plan_id: Option<String>,
    #[serde(default)]
    pub transaction_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum GitWorktreeStateBinding {
    /// The exact #4562 receipt is not yet present in this source lineage. V1
    /// therefore refuses to mint a free-form Git-state digest.
    NotConverged,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct PreparedCargoExecutionIntent {
    pub prepared_intent_id: String,
    pub schema: String,
    pub base_intent: CargoExecutionIntent,
    pub runtime_binding_id: String,
    pub materialization_receipt_id: String,
    pub tool_attestation_id: String,
    pub git_worktree_state_binding: GitWorktreeStateBinding,
}

#[derive(Serialize)]
struct PreparedIntentIdentity<'a> {
    schema: &'static str,
    base_intent_id: &'a str,
    runtime_binding_id: &'a str,
    materialization_receipt_id: &'a str,
    tool_attestation_id: &'a str,
    git_worktree_state_binding: &'a GitWorktreeStateBinding,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_prepared_intent(
    mut spec: PreparedCargoIntentSpec,
    mut source: ValidatedSnapshotReceipt,
    context: CargoBuildContextDocument,
    mut semantics: CargoAdapterSemanticsReceipt,
    mut effect_policy: EffectPolicySpec,
    runtime: CargoRuntimeBindingReceipt,
    materialization: RepositoryMaterializationReceipt,
    tool_attestation: ToolExecutableAttestationReceipt,
) -> anyhow::Result<PreparedCargoExecutionIntent> {
    if spec.schema != INPUT_SCHEMA {
        bail!("unsupported prepared Cargo intent input schema: {}", spec.schema);
    }
    normalize_optional_digest("plan_id", &mut spec.plan_id)?;
    normalize_optional_digest("transaction_id", &mut spec.transaction_id)?;

    source.validate()?;
    let context_bytes =
        serde_json::to_vec(&context).context("serialize Cargo context for prepared-intent validation")?;
    let context = verify_context_bytes(&context_bytes)?;
    semantics.validate()?;
    let effect_policy_id = validate_and_identify_policy(&mut effect_policy)?;

    let runtime_spec = CargoRuntimeBindingSpec {
        schema: RUNTIME_INPUT_SCHEMA.into(),
        realized: runtime.realized.clone(),
        context: runtime.context.clone(),
        evidence: runtime.evidence.clone(),
    };
    let rebuilt_runtime = rebuild_runtime_binding(
        runtime_spec,
        source.clone(),
        context.clone(),
        semantics.clone(),
        effect_policy_id.clone(),
        effect_policy.clone(),
    )?;
    if rebuilt_runtime != runtime {
        bail!("runtime binding is not canonical for the validated prepared-intent subjects");
    }

    let verified_materialization = verify_materialization(materialization, source.clone())?;
    let verified_tools = verify_tool_attestation(tool_attestation, context.clone())?;

    // Reuse one canonical cross-link implementation instead of defining a
    // second set of materialization/tool/runtime consistency rules here.
    expected_projection(&rebuilt_runtime, &verified_materialization, &verified_tools)?;

    let legacy = CargoExecutionIntentSpec {
        schema: LEGACY_INTENT_INPUT_SCHEMA.into(),
        git_worktree_state_before: None,
        build_context_id: context.context_id,
        invocation_id: context.invocation_id,
        plan_id: spec.plan_id,
        transaction_id: spec.transaction_id,
        adapter_semantics_digest: semantics.adapter_semantics_id,
    };
    let base_intent = build_intent(legacy, &source.snapshot_id, &effect_policy_id)?;

    if base_intent.build_context_id != rebuilt_runtime.context_id
        || base_intent.invocation_id != rebuilt_runtime.invocation_id
        || base_intent.repository_source_before != rebuilt_runtime.repository_source_before
        || base_intent.adapter_semantics_digest != rebuilt_runtime.adapter_semantics_id
        || base_intent.effect_policy_id != rebuilt_runtime.effect_policy_id
    {
        bail!("canonical base intent does not match the validated runtime binding");
    }

    let git_worktree_state_binding = GitWorktreeStateBinding::NotConverged;
    let identity = PreparedIntentIdentity {
        schema: RECEIPT_SCHEMA,
        base_intent_id: &base_intent.intent_id,
        runtime_binding_id: &rebuilt_runtime.runtime_binding_id,
        materialization_receipt_id: &verified_materialization.materialization_receipt_id,
        tool_attestation_id: &verified_tools.tool_attestation_id,
        git_worktree_state_binding: &git_worktree_state_binding,
    };
    let canonical = serde_json::to_vec(&identity)
        .context("serialize prepared Cargo execution intent identity")?;
    let prepared_intent_id = domain_sha256(HASH_DOMAIN, &canonical);

    Ok(PreparedCargoExecutionIntent {
        prepared_intent_id,
        schema: RECEIPT_SCHEMA.into(),
        base_intent,
        runtime_binding_id: rebuilt_runtime.runtime_binding_id,
        materialization_receipt_id: verified_materialization.materialization_receipt_id,
        tool_attestation_id: verified_tools.tool_attestation_id,
        git_worktree_state_binding,
    })
}

fn normalize_optional_digest(name: &str, value: &mut Option<String>) -> anyhow::Result<()> {
    if let Some(value) = value {
        validate_digest(name, value)?;
        value.make_ascii_lowercase();
    }
    Ok(())
}

fn validate_digest(name: &str, value: &str) -> anyhow::Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{name} must be a 64-character SHA-256 hex digest");
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
