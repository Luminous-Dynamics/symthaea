use anyhow::bail;

use crate::cargo_runtime_binding::CargoRuntimeBindingReceipt;
use crate::git_worktree_state_receipt::ValidatedGitWorktreeState;
use crate::pre_spawn_freshness_v2::{
    PreSpawnFreshnessObservationV2, PreSpawnFreshnessReceiptV2, build_receipt_v2,
};
use crate::prepared_cargo_intent_v2::PreparedCargoExecutionIntentV2;
use crate::repository_materialization_receipt::RepositoryMaterializationReceipt;
use crate::tool_executable_attestation::ToolExecutableAttestationReceipt;

const OBSERVATION_SCHEMA: &str = "symthaea.pre-spawn-freshness-observation.v2";
const RECEIPT_SCHEMA: &str = "symthaea.pre-spawn-freshness.v2";

/// Rebuild a stored v2 freshness receipt against independently supplied current
/// subjects and require complete canonical equality.
///
/// In particular, `fresh_git_state` must be a current observation from the
/// immediate pre-spawn boundary. Reusing historical Git evidence would verify
/// historical consistency rather than currentness.
pub(crate) fn verify_stored_v2(
    mut stored: PreSpawnFreshnessReceiptV2,
    prepared_intent: PreparedCargoExecutionIntentV2,
    runtime: CargoRuntimeBindingReceipt,
    materialization: RepositoryMaterializationReceipt,
    tool_attestation: ToolExecutableAttestationReceipt,
    fresh_git_state: ValidatedGitWorktreeState,
) -> anyhow::Result<PreSpawnFreshnessReceiptV2> {
    if stored.schema != RECEIPT_SCHEMA {
        bail!("unsupported pre-spawn freshness v2 receipt schema: {}", stored.schema);
    }

    for (name, value) in [
        ("freshness_receipt_id", &mut stored.freshness_receipt_id),
        ("prepared_intent_id", &mut stored.prepared_intent_id),
        ("runtime_binding_id", &mut stored.runtime_binding_id),
        (
            "materialization_receipt_id",
            &mut stored.materialization_receipt_id,
        ),
        ("tool_attestation_id", &mut stored.tool_attestation_id),
        ("git_worktree_state_id", &mut stored.git_worktree_state_id),
        (
            "probe_implementation_sha256",
            &mut stored.probe_implementation_sha256,
        ),
        ("probe_transcript_sha256", &mut stored.probe_transcript_sha256),
    ] {
        normalize_digest(name, value)?;
    }

    let observation = PreSpawnFreshnessObservationV2 {
        schema: OBSERVATION_SCHEMA.into(),
        probe_implementation_sha256: stored.probe_implementation_sha256.clone(),
        probe_transcript_sha256: stored.probe_transcript_sha256.clone(),
        observed: stored.observed.clone(),
    };
    let rebuilt = build_receipt_v2(
        observation,
        prepared_intent,
        runtime,
        materialization,
        tool_attestation,
        fresh_git_state,
    )?;

    if rebuilt != stored {
        bail!(
            "stored pre-spawn freshness v2 receipt is not the canonical rebuild for the supplied current subjects"
        );
    }

    Ok(rebuilt)
}

fn normalize_digest(name: &str, value: &mut String) -> anyhow::Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{name} must be a 64-character SHA-256 hex digest");
    }
    value.make_ascii_lowercase();
    Ok(())
}
