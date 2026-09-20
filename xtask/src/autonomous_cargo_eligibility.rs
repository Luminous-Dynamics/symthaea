use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::autonomous_cargo_semantics::{
    AutonomousCargoSemanticsProfileReceipt, verify_stored as verify_autonomous_profile,
};
use crate::cargo_adapter_semantics::{
    CargoAdapterSemanticsReceipt, RepositoryAccess, WorkingDirectoryPolicy,
};
use crate::cargo_context::CargoBuildContextDocument;
use crate::cargo_context_verify::verify_bytes as verify_context_bytes;
use crate::cargo_runtime_binding::CargoRuntimeBindingReceipt;
use crate::cargo_runtime_binding_receipt::verify_stored as verify_runtime_binding;
use crate::git_worktree_state_receipt::{
    ValidatedGitWorktreeState, verify_for_source as verify_git_state,
};
use crate::pre_spawn_freshness::expected_projection;
use crate::prepared_cargo_intent_v2::PreparedCargoExecutionIntentV2;
use crate::prepared_cargo_intent_v2_receipt::verify_local as verify_prepared_intent;
use crate::repository_effect_policy::{
    EffectMode, EffectPolicySpec, validate_and_identify_policy,
};
use crate::repository_materialization_receipt::{
    RepositoryMaterializationReceipt, verify_stored as verify_materialization,
};
use crate::repository_snapshot_receipt::ValidatedSnapshotReceipt;
use crate::tool_executable_attestation::{
    ToolExecutableAttestationReceipt, verify_stored as verify_tool_attestation,
};

const RECEIPT_SCHEMA: &str = "symthaea.autonomous-cargo-eligibility.v1";
const HASH_DOMAIN: &[u8] = b"symthaea.autonomous-cargo-eligibility.v1\0";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct AutonomousCargoEligibilityReceipt {
    pub eligibility_id: String,
    pub schema: String,
    pub prepared_intent_id: String,
    pub autonomous_semantics_profile_id: String,
    pub runtime_binding_id: String,
    pub materialization_receipt_id: String,
    pub tool_attestation_id: String,
    pub git_worktree_state_id: String,
    pub source_snapshot_id: String,
    pub context_id: String,
    pub invocation_id: String,
    pub effect_policy_id: String,
    pub runtime_instance_id: String,
    pub materialization_id: String,
    pub staged_content_id: String,
}

#[derive(Serialize)]
struct EligibilityIdentity<'a> {
    schema: &'static str,
    prepared_intent_id: &'a str,
    autonomous_semantics_profile_id: &'a str,
    runtime_binding_id: &'a str,
    materialization_receipt_id: &'a str,
    tool_attestation_id: &'a str,
    git_worktree_state_id: &'a str,
    source_snapshot_id: &'a str,
    context_id: &'a str,
    invocation_id: &'a str,
    effect_policy_id: &'a str,
    runtime_instance_id: &'a str,
    materialization_id: &'a str,
    staged_content_id: &'a str,
}

#[derive(Debug)]
struct JoinView<'a> {
    prepared_source: &'a str,
    prepared_context: &'a str,
    prepared_invocation: &'a str,
    prepared_effect_policy: &'a str,
    prepared_semantics: &'a str,
    prepared_runtime: &'a str,
    prepared_materialization: &'a str,
    prepared_tools: &'a str,
    prepared_git: &'a str,
    profile_semantics: &'a str,
    runtime_receipt: &'a str,
    runtime_source: &'a str,
    runtime_context: &'a str,
    runtime_invocation: &'a str,
    runtime_effect_policy: &'a str,
    runtime_semantics: &'a str,
    runtime_access: RepositoryAccess,
    runtime_working_directory: WorkingDirectoryPolicy,
    runtime_instance: &'a str,
    runtime_materialization: &'a str,
    runtime_working_subject: &'a str,
    runtime_cargo_executable: &'a str,
    runtime_rustc_executable: &'a str,
    materialization_receipt: &'a str,
    materialization_source: &'a str,
    materialization_instance: &'a str,
    materialization_kind: RepositoryAccess,
    materialization_subject: &'a str,
    materialization_working_subject: &'a str,
    tool_receipt: &'a str,
    tool_context: &'a str,
    tool_invocation: &'a str,
    tool_cargo_executable: &'a str,
    tool_rustc_executable: &'a str,
    git_state: &'a str,
    git_source: &'a str,
    expected_source: &'a str,
    expected_context: &'a str,
    expected_invocation: &'a str,
    expected_effect_policy: &'a str,
    expected_semantics: &'a str,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_eligibility(
    prepared_intent: PreparedCargoExecutionIntentV2,
    autonomous_profile: AutonomousCargoSemanticsProfileReceipt,
    mut source: ValidatedSnapshotReceipt,
    context: CargoBuildContextDocument,
    mut semantics: CargoAdapterSemanticsReceipt,
    mut effect_policy: EffectPolicySpec,
    runtime: CargoRuntimeBindingReceipt,
    materialization: RepositoryMaterializationReceipt,
    tool_attestation: ToolExecutableAttestationReceipt,
    git_state: ValidatedGitWorktreeState,
) -> anyhow::Result<AutonomousCargoEligibilityReceipt> {
    source.validate()?;
    let context_bytes = serde_json::to_vec(&context)
        .context("serialize Cargo context for autonomous eligibility")?;
    let context = verify_context_bytes(&context_bytes)?;
    semantics.validate()?;
    let effect_policy_id = validate_and_identify_policy(&mut effect_policy)?;
    if effect_policy.mode != EffectMode::ReadOnlySource {
        bail!("autonomous Cargo eligibility v1 requires read_only_source effect policy");
    }

    let autonomous_profile = verify_autonomous_profile(autonomous_profile, semantics.clone())?;
    let runtime = verify_runtime_binding(
        runtime,
        source.clone(),
        context.clone(),
        semantics.clone(),
        effect_policy.clone(),
    )?;
    let materialization = verify_materialization(materialization, source.clone())?;
    let tool_attestation = verify_tool_attestation(tool_attestation, context.clone())?;
    let git_state = verify_git_state(git_state, &source.snapshot_id)?;
    let prepared_intent = verify_prepared_intent(prepared_intent)?;

    // Reuse the established runtime/materialization/tool projection so this
    // eligibility layer cannot drift into a second interpretation of those
    // cross-links.
    expected_projection(&runtime, &materialization, &tool_attestation)?;

    let runtime_cargo_executable = bound_input_content(&runtime, "cargo_executable")?;
    let runtime_rustc_executable = bound_input_content(&runtime, "rustc_executable")?;
    let staged_content_id = materialization
        .staged_content_id
        .clone()
        .context("autonomous Cargo eligibility requires staged_content_id")?;

    let join = JoinView {
        prepared_source: &prepared_intent.base_intent.repository_source_before,
        prepared_context: &prepared_intent.base_intent.build_context_id,
        prepared_invocation: &prepared_intent.base_intent.invocation_id,
        prepared_effect_policy: &prepared_intent.base_intent.effect_policy_id,
        prepared_semantics: &prepared_intent.base_intent.adapter_semantics_digest,
        prepared_runtime: &prepared_intent.runtime_binding_id,
        prepared_materialization: &prepared_intent.materialization_receipt_id,
        prepared_tools: &prepared_intent.tool_attestation_id,
        prepared_git: &prepared_intent.git_worktree_state_id,
        profile_semantics: &autonomous_profile.adapter_semantics_id,
        runtime_receipt: &runtime.runtime_binding_id,
        runtime_source: &runtime.repository_source_before,
        runtime_context: &runtime.context_id,
        runtime_invocation: &runtime.invocation_id,
        runtime_effect_policy: &runtime.effect_policy_id,
        runtime_semantics: &runtime.adapter_semantics_id,
        runtime_access: runtime.realized.repository_access.clone(),
        runtime_working_directory: runtime.realized.working_directory.clone(),
        runtime_instance: &runtime.evidence.runtime_instance_id,
        runtime_materialization: &runtime.evidence.repository_materialization_id,
        runtime_working_subject: &runtime.evidence.working_directory_subject_id,
        runtime_cargo_executable,
        runtime_rustc_executable,
        materialization_receipt: &materialization.materialization_receipt_id,
        materialization_source: &materialization.source_snapshot_id,
        materialization_instance: &materialization.runtime_instance_id,
        materialization_kind: materialization.kind.clone(),
        materialization_subject: &materialization.materialization_id,
        materialization_working_subject: &materialization.working_directory_subject_id,
        tool_receipt: &tool_attestation.tool_attestation_id,
        tool_context: &tool_attestation.context_id,
        tool_invocation: &tool_attestation.invocation_id,
        tool_cargo_executable: &tool_attestation.cargo.executable_content_id,
        tool_rustc_executable: &tool_attestation.rustc.executable_content_id,
        git_state: &git_state.state_id,
        git_source: &git_state.repository_source_snapshot_id,
        expected_source: &source.snapshot_id,
        expected_context: &context.context_id,
        expected_invocation: &context.invocation_id,
        expected_effect_policy: &effect_policy_id,
        expected_semantics: &semantics.adapter_semantics_id,
    };
    validate_join(&join)?;

    let identity = EligibilityIdentity {
        schema: RECEIPT_SCHEMA,
        prepared_intent_id: &prepared_intent.prepared_intent_id,
        autonomous_semantics_profile_id: &autonomous_profile.autonomous_semantics_profile_id,
        runtime_binding_id: &runtime.runtime_binding_id,
        materialization_receipt_id: &materialization.materialization_receipt_id,
        tool_attestation_id: &tool_attestation.tool_attestation_id,
        git_worktree_state_id: &git_state.state_id,
        source_snapshot_id: &source.snapshot_id,
        context_id: &context.context_id,
        invocation_id: &context.invocation_id,
        effect_policy_id: &effect_policy_id,
        runtime_instance_id: &runtime.evidence.runtime_instance_id,
        materialization_id: &materialization.materialization_id,
        staged_content_id: &staged_content_id,
    };
    let bytes = serde_json::to_vec(&identity)
        .context("serialize autonomous Cargo eligibility identity")?;
    let eligibility_id = domain_sha256(HASH_DOMAIN, &bytes);

    Ok(AutonomousCargoEligibilityReceipt {
        eligibility_id,
        schema: RECEIPT_SCHEMA.into(),
        prepared_intent_id: prepared_intent.prepared_intent_id,
        autonomous_semantics_profile_id: autonomous_profile.autonomous_semantics_profile_id,
        runtime_binding_id: runtime.runtime_binding_id,
        materialization_receipt_id: materialization.materialization_receipt_id,
        tool_attestation_id: tool_attestation.tool_attestation_id,
        git_worktree_state_id: git_state.state_id,
        source_snapshot_id: source.snapshot_id,
        context_id: context.context_id,
        invocation_id: context.invocation_id,
        effect_policy_id,
        runtime_instance_id: runtime.evidence.runtime_instance_id,
        materialization_id: materialization.materialization_id,
        staged_content_id,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_stored(
    stored: AutonomousCargoEligibilityReceipt,
    prepared_intent: PreparedCargoExecutionIntentV2,
    autonomous_profile: AutonomousCargoSemanticsProfileReceipt,
    source: ValidatedSnapshotReceipt,
    context: CargoBuildContextDocument,
    semantics: CargoAdapterSemanticsReceipt,
    effect_policy: EffectPolicySpec,
    runtime: CargoRuntimeBindingReceipt,
    materialization: RepositoryMaterializationReceipt,
    tool_attestation: ToolExecutableAttestationReceipt,
    git_state: ValidatedGitWorktreeState,
) -> anyhow::Result<AutonomousCargoEligibilityReceipt> {
    if stored.schema != RECEIPT_SCHEMA {
        bail!("unsupported autonomous Cargo eligibility schema: {}", stored.schema);
    }
    validate_digest("eligibility_id", &stored.eligibility_id)?;
    let rebuilt = build_eligibility(
        prepared_intent,
        autonomous_profile,
        source,
        context,
        semantics,
        effect_policy,
        runtime,
        materialization,
        tool_attestation,
        git_state,
    )?;
    if rebuilt != stored {
        bail!(
            "stored autonomous Cargo eligibility receipt is not the canonical rebuild for its validated subjects"
        );
    }
    Ok(rebuilt)
}

fn bound_input_content<'a>(
    runtime: &'a CargoRuntimeBindingReceipt,
    role: &str,
) -> anyhow::Result<&'a str> {
    runtime
        .evidence
        .bound_inputs
        .iter()
        .find(|input| input.role == role)
        .map(|input| input.content_id.as_str())
        .with_context(|| format!("validated runtime binding is missing required input {role}"))
}

fn validate_join(join: &JoinView<'_>) -> anyhow::Result<()> {
    for (name, actual, expected) in [
        ("prepared source", join.prepared_source, join.expected_source),
        ("runtime source", join.runtime_source, join.expected_source),
        ("materialization source", join.materialization_source, join.expected_source),
        ("Git-state source", join.git_source, join.expected_source),
        ("prepared context", join.prepared_context, join.expected_context),
        ("runtime context", join.runtime_context, join.expected_context),
        ("tool context", join.tool_context, join.expected_context),
        ("prepared invocation", join.prepared_invocation, join.expected_invocation),
        ("runtime invocation", join.runtime_invocation, join.expected_invocation),
        ("tool invocation", join.tool_invocation, join.expected_invocation),
        (
            "prepared effect policy",
            join.prepared_effect_policy,
            join.expected_effect_policy,
        ),
        ("runtime effect policy", join.runtime_effect_policy, join.expected_effect_policy),
        ("prepared semantics", join.prepared_semantics, join.expected_semantics),
        ("profile semantics", join.profile_semantics, join.expected_semantics),
        ("runtime semantics", join.runtime_semantics, join.expected_semantics),
        ("prepared runtime", join.prepared_runtime, join.runtime_receipt),
        (
            "prepared materialization",
            join.prepared_materialization,
            join.materialization_receipt,
        ),
        ("prepared tools", join.prepared_tools, join.tool_receipt),
        ("prepared Git state", join.prepared_git, join.git_state),
    ] {
        if actual != expected {
            bail!("autonomous Cargo eligibility cross-link mismatch: {name}");
        }
    }

    if join.runtime_access != RepositoryAccess::IsolatedStagingTree
        || join.runtime_working_directory != WorkingDirectoryPolicy::StagingRoot
        || join.materialization_kind != RepositoryAccess::IsolatedStagingTree
    {
        bail!("autonomous Cargo eligibility requires one isolated staging runtime");
    }
    if join.runtime_instance != join.materialization_instance {
        bail!("runtime instance does not match repository materialization instance");
    }
    if join.runtime_materialization != join.materialization_subject
        || join.runtime_working_subject != join.materialization_working_subject
        || join.materialization_subject != join.materialization_working_subject
    {
        bail!("runtime working/materialization subject does not match staged materialization");
    }
    if join.runtime_cargo_executable != join.tool_cargo_executable
        || join.runtime_rustc_executable != join.tool_rustc_executable
    {
        bail!("runtime executable inputs do not match executable attestation");
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

#[cfg(test)]
mod tests {
    use super::*;

    fn view<'a>(value: &'a str) -> JoinView<'a> {
        JoinView {
            prepared_source: value,
            prepared_context: value,
            prepared_invocation: value,
            prepared_effect_policy: value,
            prepared_semantics: value,
            prepared_runtime: "runtime",
            prepared_materialization: "materialization-receipt",
            prepared_tools: "tool-receipt",
            prepared_git: "git-state",
            profile_semantics: value,
            runtime_receipt: "runtime",
            runtime_source: value,
            runtime_context: value,
            runtime_invocation: value,
            runtime_effect_policy: value,
            runtime_semantics: value,
            runtime_access: RepositoryAccess::IsolatedStagingTree,
            runtime_working_directory: WorkingDirectoryPolicy::StagingRoot,
            runtime_instance: "instance",
            runtime_materialization: "materialization",
            runtime_working_subject: "materialization",
            runtime_cargo_executable: "cargo",
            runtime_rustc_executable: "rustc",
            materialization_receipt: "materialization-receipt",
            materialization_source: value,
            materialization_instance: "instance",
            materialization_kind: RepositoryAccess::IsolatedStagingTree,
            materialization_subject: "materialization",
            materialization_working_subject: "materialization",
            tool_receipt: "tool-receipt",
            tool_context: value,
            tool_invocation: value,
            tool_cargo_executable: "cargo",
            tool_rustc_executable: "rustc",
            git_state: "git-state",
            git_source: value,
            expected_source: value,
            expected_context: value,
            expected_invocation: value,
            expected_effect_policy: value,
            expected_semantics: value,
        }
    }

    #[test]
    fn exact_isolated_staging_join_accepts() {
        assert!(validate_join(&view("same")).is_ok());
    }

    #[test]
    fn prepared_runtime_substitution_rejects() {
        let mut candidate = view("same");
        candidate.prepared_runtime = "other-runtime";
        assert!(validate_join(&candidate).is_err());
    }

    #[test]
    fn direct_worktree_runtime_rejects() {
        let mut candidate = view("same");
        candidate.runtime_access = RepositoryAccess::ReadOnlySource;
        candidate.runtime_working_directory = WorkingDirectoryPolicy::RepositoryRoot;
        assert!(validate_join(&candidate).is_err());
    }

    #[test]
    fn runtime_materialization_instance_substitution_rejects() {
        let mut candidate = view("same");
        candidate.materialization_instance = "other";
        assert!(validate_join(&candidate).is_err());
    }

    #[test]
    fn executable_substitution_rejects() {
        let mut candidate = view("same");
        candidate.tool_cargo_executable = "other-cargo";
        assert!(validate_join(&candidate).is_err());
    }

    #[test]
    fn source_substitution_rejects() {
        let mut candidate = view("same");
        candidate.git_source = "other-source";
        assert!(validate_join(&candidate).is_err());
    }
}
