use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::cargo_runtime_binding::CargoRuntimeBindingReceipt;
use crate::git_worktree_state_receipt::{ValidatedGitWorktreeState, verify_for_source};
use crate::pre_spawn_freshness::{FreshnessProjection, expected_projection};
use crate::prepared_cargo_intent_v2::{
    PreparedCargoExecutionIntentV2, SourceGitStateObservationModel,
};
use crate::prepared_cargo_intent_v2_receipt::verify_local as verify_prepared_intent;
use crate::repository_materialization_receipt::RepositoryMaterializationReceipt;
use crate::tool_executable_attestation::ToolExecutableAttestationReceipt;

const OBSERVATION_SCHEMA: &str = "symthaea.pre-spawn-freshness-observation.v2";
const RECEIPT_SCHEMA: &str = "symthaea.pre-spawn-freshness.v2";
const HASH_DOMAIN: &[u8] = b"symthaea.pre-spawn-freshness.v2\0";

/// Complete immediate pre-spawn projection for one Git-bound prepared Cargo intent.
///
/// The v1 runtime/materialization/tool projection is retained without reinterpretation.
/// V2 adds the exact Git operational-state subject that the prepared intent admitted.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct FreshnessProjectionV2 {
    pub runtime: FreshnessProjection,
    pub git_worktree_state_id: String,
    pub source_git_state_observation: SourceGitStateObservationModel,
}

/// Immediate observation produced after admission/persistence and before process spawn.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct PreSpawnFreshnessObservationV2 {
    pub schema: String,
    pub probe_implementation_sha256: String,
    pub probe_transcript_sha256: String,
    pub observed: FreshnessProjectionV2,
}

/// Content-addressed proof that one freshly observed execution subject exactly matched
/// the complete Git-bound prepared Cargo subject immediately before spawn.
///
/// The receipt preserves [`SourceGitStateObservationModel::SequentialNotAtomic`]:
/// matching source and Git sensors are not represented as an atomic filesystem/Git
/// transaction. Isolation and the effect-entry boundary own stronger race claims.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct PreSpawnFreshnessReceiptV2 {
    pub freshness_receipt_id: String,
    pub schema: String,
    pub prepared_intent_id: String,
    pub runtime_binding_id: String,
    pub materialization_receipt_id: String,
    pub tool_attestation_id: String,
    pub git_worktree_state_id: String,
    pub probe_implementation_sha256: String,
    pub probe_transcript_sha256: String,
    pub observed: FreshnessProjectionV2,
}

#[derive(Serialize)]
struct FreshnessIdentityV2<'a> {
    schema: &'static str,
    prepared_intent_id: &'a str,
    runtime_binding_id: &'a str,
    materialization_receipt_id: &'a str,
    tool_attestation_id: &'a str,
    git_worktree_state_id: &'a str,
    probe_implementation_sha256: &'a str,
    probe_transcript_sha256: &'a str,
    observed: &'a FreshnessProjectionV2,
}

/// Build a v2 immediate-freshness receipt from the exact prepared subject and a fresh
/// Git operational-state observation.
///
/// There is intentionally no caller-supplied expected Git id. The expected id comes
/// from the verified prepared intent; the current id comes from the validated fresh
/// Git receipt.
pub(crate) fn build_receipt_v2(
    mut observation: PreSpawnFreshnessObservationV2,
    prepared_intent: PreparedCargoExecutionIntentV2,
    runtime: CargoRuntimeBindingReceipt,
    materialization: RepositoryMaterializationReceipt,
    tool_attestation: ToolExecutableAttestationReceipt,
    git_state: ValidatedGitWorktreeState,
) -> anyhow::Result<PreSpawnFreshnessReceiptV2> {
    if observation.schema != OBSERVATION_SCHEMA {
        bail!(
            "unsupported pre-spawn freshness v2 observation schema: {}",
            observation.schema
        );
    }
    normalize_digest(
        "probe_implementation_sha256",
        &mut observation.probe_implementation_sha256,
    )?;
    normalize_digest(
        "probe_transcript_sha256",
        &mut observation.probe_transcript_sha256,
    )?;

    let prepared_intent = verify_prepared_intent(prepared_intent)?;
    let expected = expected_projection_v2(
        &prepared_intent,
        &runtime,
        &materialization,
        &tool_attestation,
        git_state,
    )?;
    if observation.observed != expected {
        bail!(
            "pre-spawn freshness v2 observation does not equal the complete prepared Cargo subject"
        );
    }

    let identity = FreshnessIdentityV2 {
        schema: RECEIPT_SCHEMA,
        prepared_intent_id: &prepared_intent.prepared_intent_id,
        runtime_binding_id: &runtime.runtime_binding_id,
        materialization_receipt_id: &materialization.materialization_receipt_id,
        tool_attestation_id: &tool_attestation.tool_attestation_id,
        git_worktree_state_id: &expected.git_worktree_state_id,
        probe_implementation_sha256: &observation.probe_implementation_sha256,
        probe_transcript_sha256: &observation.probe_transcript_sha256,
        observed: &expected,
    };
    let canonical = serde_json::to_vec(&identity)
        .context("serialize pre-spawn freshness v2 identity")?;
    let freshness_receipt_id = domain_sha256(HASH_DOMAIN, &canonical);

    Ok(PreSpawnFreshnessReceiptV2 {
        freshness_receipt_id,
        schema: RECEIPT_SCHEMA.into(),
        prepared_intent_id: prepared_intent.prepared_intent_id,
        runtime_binding_id: runtime.runtime_binding_id,
        materialization_receipt_id: materialization.materialization_receipt_id,
        tool_attestation_id: tool_attestation.tool_attestation_id,
        git_worktree_state_id: expected.git_worktree_state_id.clone(),
        probe_implementation_sha256: observation.probe_implementation_sha256,
        probe_transcript_sha256: observation.probe_transcript_sha256,
        observed: expected,
    })
}

/// Reconstruct the exact v2 projection that a fresh observation must equal.
pub(crate) fn expected_projection_v2(
    prepared_intent: &PreparedCargoExecutionIntentV2,
    runtime: &CargoRuntimeBindingReceipt,
    materialization: &RepositoryMaterializationReceipt,
    tool_attestation: &ToolExecutableAttestationReceipt,
    git_state: ValidatedGitWorktreeState,
) -> anyhow::Result<FreshnessProjectionV2> {
    let prepared_intent = verify_prepared_intent(prepared_intent.clone())?;
    let runtime_projection = expected_projection(runtime, materialization, tool_attestation)?;

    require_equal(
        "prepared runtime binding",
        &prepared_intent.runtime_binding_id,
        &runtime.runtime_binding_id,
    )?;
    require_equal(
        "prepared repository materialization",
        &prepared_intent.materialization_receipt_id,
        &materialization.materialization_receipt_id,
    )?;
    require_equal(
        "prepared tool attestation",
        &prepared_intent.tool_attestation_id,
        &tool_attestation.tool_attestation_id,
    )?;

    let base = &prepared_intent.base_intent;
    for (name, prepared, current) in [
        (
            "repository source",
            base.repository_source_before.as_str(),
            runtime_projection.repository_source_before.as_str(),
        ),
        (
            "Cargo context",
            base.build_context_id.as_str(),
            runtime_projection.context_id.as_str(),
        ),
        (
            "Cargo invocation",
            base.invocation_id.as_str(),
            runtime_projection.invocation_id.as_str(),
        ),
        (
            "repository effect policy",
            base.effect_policy_id.as_str(),
            runtime_projection.effect_policy_id.as_str(),
        ),
        (
            "adapter semantics",
            base.adapter_semantics_digest.as_str(),
            runtime_projection.adapter_semantics_id.as_str(),
        ),
    ] {
        require_equal(name, prepared, current)?;
    }

    let git_state = verify_for_source(git_state, &base.repository_source_before)?;
    require_equal(
        "fresh Git worktree state",
        &git_state.state_id,
        &prepared_intent.git_worktree_state_id,
    )?;

    Ok(FreshnessProjectionV2 {
        runtime: runtime_projection,
        git_worktree_state_id: git_state.state_id,
        source_git_state_observation: prepared_intent.source_git_state_observation,
    })
}

fn require_equal(name: &str, actual: &str, expected: &str) -> anyhow::Result<()> {
    if actual != expected {
        bail!("pre-spawn freshness v2 {name} does not match the prepared Cargo subject");
    }
    Ok(())
}

fn normalize_digest(name: &str, value: &mut String) -> anyhow::Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{name} must be a 64-character SHA-256 hex digest");
    }
    value.make_ascii_lowercase();
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
