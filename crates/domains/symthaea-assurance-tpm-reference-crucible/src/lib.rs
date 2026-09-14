// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! End-to-end TPM/reference-integrity assurance composition.
//!
//! This crate recomputes every lower qualification theorem and then checks the
//! cross-layer bindings that individual assurance crates cannot establish alone.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_assurance_reference_integrity::{
    evaluate_reference_integrity, MeasurementExtractionVerificationReceipt,
    NormalizedMeasurementSet, ReferenceIntegrityError, ReferenceIntegrityEvaluationReport,
    ReferenceIntegrityManifest, ReferenceIntegrityPolicy, ReferenceIntegrityStatus,
    RimSignatureVerificationReceipt,
};
use symthaea_assurance_reference_integrity_lineage::{
    qualify_reference_integrity_lineage, ReferenceIntegrityLineageEntry,
    ReferenceIntegrityLineageError, ReferenceIntegrityLineagePolicy,
    ReferenceIntegrityLineageReport,
};
use symthaea_assurance_tpm2_attestation_possession::{
    qualify_attestation_possession, AttestationChallenge, AttestationKeyBinding,
    AttestationPossessionError, AttestationPossessionPolicy, QuoteVerificationReceipt,
    Tpm2QuoteArtifacts,
};
use symthaea_assurance_tpm2_measured_boot_replay::{
    qualify_measured_boot_replay, MeasuredBootLogBundle, MeasuredBootReplayError,
    MeasuredBootReplayPolicy, MeasuredBootReplayVerificationReceipt,
};
use symthaea_assurance_tpm2_platform_qualification::{
    qualify_tpm2_platform, Tpm2PlatformQualificationError, Tpm2PlatformQualificationPolicy,
    Tpm2RuntimeSubjectEvidence,
};
use symthaea_assurance_tpm2_tools_adapter::{
    Tpm2NvCounterObservation, Tpm2ReadHierarchy, Tpm2ToolsAdapterPolicy, Tpm2ToolsExecutor,
};

const REPORT_DIGEST_SCHEMA: &[u8] = b"symthaea-tpm-reference-chain-report-v1\0";
const CHAIN_POLICY_DIGEST_SCHEMA: &[u8] = b"symthaea-tpm-reference-chain-policy-v1\0";
const LOWER_POLICY_BUNDLE_DIGEST_SCHEMA: &[u8] =
    b"symthaea-tpm-reference-lower-policy-bundle-v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TpmReferenceChainPolicy {
    pub schema_version: String,
    pub campaign_id: String,
    pub expected_adapter_id: String,
    pub expected_trust_store_ref: String,
    pub expected_runtime_subject_digest: String,
    /// Canonical digest of every lower assurance policy used by this campaign.
    pub expected_lower_policy_bundle_digest: String,
    pub expected_reference_policy_id: String,
    pub expected_lineage_policy_id: String,
    pub max_platform_age_ms: u64,
    pub max_possession_age_ms: u64,
    pub max_replay_age_ms: u64,
    pub max_reference_age_ms: u64,
    pub evidence_refs: Vec<String>,
}

impl TpmReferenceChainPolicy {
    pub fn validate(&self) -> bool {
        canonical_text(&self.schema_version)
            && canonical_text(&self.campaign_id)
            && canonical_text(&self.expected_adapter_id)
            && canonical_text(&self.expected_trust_store_ref)
            && valid_digest(&self.expected_runtime_subject_digest)
            && valid_digest(&self.expected_lower_policy_bundle_digest)
            && canonical_text(&self.expected_reference_policy_id)
            && canonical_text(&self.expected_lineage_policy_id)
            && self.max_platform_age_ms > 0
            && self.max_possession_age_ms > 0
            && self.max_replay_age_ms > 0
            && self.max_reference_age_ms > 0
            && nonempty_refs(&self.evidence_refs)
    }

    /// Canonical identity of the top-level campaign contract itself.
    pub fn policy_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(CHAIN_POLICY_DIGEST_SCHEMA);
        for value in [
            self.schema_version.as_str(),
            self.campaign_id.as_str(),
            self.expected_adapter_id.as_str(),
            self.expected_trust_store_ref.as_str(),
            self.expected_runtime_subject_digest.as_str(),
            self.expected_lower_policy_bundle_digest.as_str(),
            self.expected_reference_policy_id.as_str(),
            self.expected_lineage_policy_id.as_str(),
        ] {
            push_field(&mut hasher, value);
        }
        for value in [
            self.max_platform_age_ms,
            self.max_possession_age_ms,
            self.max_replay_age_ms,
            self.max_reference_age_ms,
        ] {
            push_field(&mut hasher, &value.to_string());
        }
        push_sorted_refs(&mut hasher, "chain-policy-evidence", &self.evidence_refs);
        format!("blake3:{}", hasher.finalize().to_hex())
    }
}

#[derive(Debug, Clone)]
pub struct TpmReferenceChainInputs {
    pub adapter_policy: Tpm2ToolsAdapterPolicy,
    pub runtime_subject: Tpm2RuntimeSubjectEvidence,
    pub before_counter: Tpm2NvCounterObservation,
    pub after_counter: Tpm2NvCounterObservation,
    pub platform_policy: Tpm2PlatformQualificationPolicy,
    pub platform_qualified_at_ms: u64,

    pub possession_policy: AttestationPossessionPolicy,
    pub challenge: AttestationChallenge,
    pub attestation_key: AttestationKeyBinding,
    pub quote: Tpm2QuoteArtifacts,
    pub quote_verification: QuoteVerificationReceipt,
    pub possession_qualified_at_ms: u64,

    pub replay_policy: MeasuredBootReplayPolicy,
    pub log_bundle: MeasuredBootLogBundle,
    pub replay_verification: MeasuredBootReplayVerificationReceipt,
    pub replay_qualified_at_ms: u64,

    pub reference_policy: ReferenceIntegrityPolicy,
    pub reference_manifest: ReferenceIntegrityManifest,
    pub reference_signature: RimSignatureVerificationReceipt,
    pub measurements: NormalizedMeasurementSet,
    pub extraction: MeasurementExtractionVerificationReceipt,
    pub reference_evaluated_at_ms: u64,

    pub lineage_policy: ReferenceIntegrityLineagePolicy,
    pub lineage_entries: Vec<ReferenceIntegrityLineageEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TpmReferenceChainReport {
    pub schema_version: String,
    pub campaign_id: String,
    pub chain_policy_digest: String,
    pub lower_policy_bundle_digest: String,
    pub before_observation_digest: String,
    pub after_observation_digest: String,
    pub platform_qualification_digest: String,
    pub possession_digest: String,
    pub replay_record_digest: String,
    pub reference_report_digest: String,
    pub lineage_digest: String,
    pub current_manifest_digest: String,
    pub current_signature_receipt_digest: String,
    pub qualified_at_ms: u64,
    pub evidence_refs: Vec<String>,
}

impl TpmReferenceChainReport {
    pub fn report_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(REPORT_DIGEST_SCHEMA);
        for value in [
            self.schema_version.as_str(),
            self.campaign_id.as_str(),
            self.chain_policy_digest.as_str(),
            self.lower_policy_bundle_digest.as_str(),
            self.before_observation_digest.as_str(),
            self.after_observation_digest.as_str(),
            self.platform_qualification_digest.as_str(),
            self.possession_digest.as_str(),
            self.replay_record_digest.as_str(),
            self.reference_report_digest.as_str(),
            self.lineage_digest.as_str(),
            self.current_manifest_digest.as_str(),
            self.current_signature_receipt_digest.as_str(),
        ] {
            push_field(&mut hasher, value);
        }
        push_field(&mut hasher, &self.qualified_at_ms.to_string());
        let mut refs = self.evidence_refs.clone();
        refs.sort();
        for reference in refs {
            push_field(&mut hasher, &reference);
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TpmReferenceChainError {
    InvalidPolicy,
    LowerPolicyBundleMismatch,
    Platform(Tpm2PlatformQualificationError),
    PlatformAdapterMismatch,
    PlatformTrustStoreMismatch,
    RuntimeSubjectMismatch,
    Possession(AttestationPossessionError),
    Replay(MeasuredBootReplayError),
    Reference(ReferenceIntegrityError),
    ReferenceNotApproved(ReferenceIntegrityStatus),
    Lineage(ReferenceIntegrityLineageError),
    ReferencePolicyMismatch,
    LineagePolicyMismatch,
    ReferenceLineageTipMismatch,
    ReferenceLineageSignatureSplice,
    ReferenceLineageRevisionMismatch,
    MeasurementSourceArtifactMismatch,
    EvidenceFromFuture(&'static str),
    StaleEvidence(&'static str),
}

#[allow(clippy::too_many_arguments)]
pub fn qualify_tpm_reference_chain(
    policy: &TpmReferenceChainPolicy,
    inputs: &TpmReferenceChainInputs,
    qualified_at_ms: u64,
    executor: &impl Tpm2ToolsExecutor,
) -> Result<TpmReferenceChainReport, TpmReferenceChainError> {
    if !policy.validate() {
        return Err(TpmReferenceChainError::InvalidPolicy);
    }

    let chain_policy_digest = policy.policy_digest();
    let policy_bundle_digest = lower_policy_bundle_digest(inputs);
    if policy_bundle_digest != policy.expected_lower_policy_bundle_digest {
        return Err(TpmReferenceChainError::LowerPolicyBundleMismatch);
    }

    let platform = qualify_tpm2_platform(
        &inputs.platform_policy,
        &inputs.adapter_policy,
        &inputs.runtime_subject,
        &inputs.before_counter,
        &inputs.after_counter,
        inputs.platform_qualified_at_ms,
        executor,
    )
    .map_err(TpmReferenceChainError::Platform)?;

    if platform.adapter_id != policy.expected_adapter_id {
        return Err(TpmReferenceChainError::PlatformAdapterMismatch);
    }
    if platform.trust_store_ref != policy.expected_trust_store_ref {
        return Err(TpmReferenceChainError::PlatformTrustStoreMismatch);
    }
    if platform.runtime_subject_digest != policy.expected_runtime_subject_digest {
        return Err(TpmReferenceChainError::RuntimeSubjectMismatch);
    }
    ensure_fresh(
        qualified_at_ms,
        platform.qualified_at_ms,
        policy.max_platform_age_ms,
        "platform",
    )?;

    let possession = qualify_attestation_possession(
        &inputs.possession_policy,
        &platform,
        &inputs.challenge,
        &inputs.attestation_key,
        &inputs.quote,
        &inputs.quote_verification,
        inputs.possession_qualified_at_ms,
    )
    .map_err(TpmReferenceChainError::Possession)?;
    ensure_fresh(
        qualified_at_ms,
        possession.qualified_at_ms,
        policy.max_possession_age_ms,
        "possession",
    )?;

    let replay = qualify_measured_boot_replay(
        &inputs.replay_policy,
        &possession,
        &inputs.quote,
        &inputs.log_bundle,
        &inputs.replay_verification,
        inputs.replay_qualified_at_ms,
    )
    .map_err(TpmReferenceChainError::Replay)?;
    ensure_fresh(
        qualified_at_ms,
        replay.qualified_at_ms,
        policy.max_replay_age_ms,
        "replay",
    )?;

    if inputs.measurements.source_event_log_digest != inputs.log_bundle.event_log_digest
        || inputs.measurements.source_final_events_digest != inputs.log_bundle.final_events_digest
    {
        return Err(TpmReferenceChainError::MeasurementSourceArtifactMismatch);
    }

    let reference = evaluate_reference_integrity(
        &inputs.reference_policy,
        &replay,
        &inputs.reference_manifest,
        &inputs.reference_signature,
        &inputs.measurements,
        &inputs.extraction,
        inputs.reference_evaluated_at_ms,
    )
    .map_err(TpmReferenceChainError::Reference)?;
    if reference.policy_id != policy.expected_reference_policy_id {
        return Err(TpmReferenceChainError::ReferencePolicyMismatch);
    }
    if reference.status != ReferenceIntegrityStatus::Approved {
        return Err(TpmReferenceChainError::ReferenceNotApproved(reference.status));
    }
    ensure_fresh(
        qualified_at_ms,
        reference.evaluated_at_ms,
        policy.max_reference_age_ms,
        "reference",
    )?;

    let lineage = qualify_reference_integrity_lineage(
        &inputs.lineage_policy,
        &inputs.lineage_entries,
    )
    .map_err(TpmReferenceChainError::Lineage)?;
    if lineage.policy_id != policy.expected_lineage_policy_id {
        return Err(TpmReferenceChainError::LineagePolicyMismatch);
    }
    cross_check_reference_lineage(&reference, &lineage, &inputs.reference_manifest)?;

    let mut evidence_refs = policy.evidence_refs.clone();
    evidence_refs.extend([
        format!("chain-policy:{chain_policy_digest}"),
        format!("lower-policy-bundle:{policy_bundle_digest}"),
        format!("platform:{}", platform.qualification_digest()),
        format!("possession:{}", possession.possession_digest()),
        format!("replay:{}", replay.replay_record_digest()),
        format!("reference:{}", reference.report_digest()),
        format!("lineage:{}", lineage.lineage_digest()),
    ]);

    Ok(TpmReferenceChainReport {
        schema_version: "1".into(),
        campaign_id: policy.campaign_id.clone(),
        chain_policy_digest,
        lower_policy_bundle_digest: policy_bundle_digest,
        before_observation_digest: inputs.before_counter.observation_digest(),
        after_observation_digest: inputs.after_counter.observation_digest(),
        platform_qualification_digest: platform.qualification_digest(),
        possession_digest: possession.possession_digest(),
        replay_record_digest: replay.replay_record_digest(),
        reference_report_digest: reference.report_digest(),
        lineage_digest: lineage.lineage_digest(),
        current_manifest_digest: reference.manifest_digest,
        current_signature_receipt_digest: reference.manifest_signature_receipt_digest,
        qualified_at_ms,
        evidence_refs,
    })
}

/// Canonical content digest of every lower policy that can change campaign semantics.
///
/// IDs alone are insufficient: a policy can retain the same ID while weakening
/// freshness, signer, tool, measurement, or allow/deny requirements. This digest
/// makes those semantics part of the end-to-end campaign contract.
pub fn lower_policy_bundle_digest(inputs: &TpmReferenceChainInputs) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(LOWER_POLICY_BUNDLE_DIGEST_SCHEMA);

    push_field(&mut hasher, "adapter-policy");
    let adapter = &inputs.adapter_policy;
    for value in [
        adapter.schema_version.as_str(),
        adapter.adapter_id.as_str(),
        adapter.logical_store_id.as_str(),
        adapter.trust_store_ref.as_str(),
        adapter.counter_epoch.as_str(),
        adapter.expected_nv_name.as_str(),
        adapter.tcti.as_str(),
        adapter.nvreadpublic_path.as_str(),
        adapter.nvread_path.as_str(),
        adapter.expected_nvreadpublic_blake3.as_str(),
        adapter.expected_nvread_blake3.as_str(),
    ] {
        push_field(&mut hasher, value);
    }
    push_field(&mut hasher, &format!("{:#x}", adapter.nv_index));
    push_field(
        &mut hasher,
        match adapter.read_hierarchy {
            Tpm2ReadHierarchy::Owner => "owner",
            Tpm2ReadHierarchy::Platform => "platform",
            Tpm2ReadHierarchy::Index => "index",
        },
    );
    push_field(&mut hasher, &adapter.minimum_counter_value.to_string());
    push_sorted_refs(&mut hasher, "adapter-evidence", &adapter.evidence_refs);

    push_field(&mut hasher, "platform-policy");
    let platform = &inputs.platform_policy;
    for value in [
        platform.schema_version.as_str(),
        platform.qualification_id.as_str(),
        platform.getcap_path.as_str(),
        platform.expected_getcap_blake3.as_str(),
    ] {
        push_field(&mut hasher, value);
    }
    for value in [
        platform.expected_family_indicator,
        platform.expected_specification_revision,
        platform.expected_manufacturer,
        platform.expected_firmware_version_1,
        platform.expected_firmware_version_2,
    ] {
        push_field(&mut hasher, &value.to_string());
    }
    push_sorted_refs(&mut hasher, "platform-evidence", &platform.evidence_refs);

    push_field(&mut hasher, "possession-policy");
    let possession = &inputs.possession_policy;
    for value in [
        possession.schema_version.as_str(),
        possession.policy_id.as_str(),
        possession.expected_platform_qualification_digest.as_str(),
        possession.expected_ak_binding_digest.as_str(),
        possession.expected_pcr_selection_digest.as_str(),
        possession.expected_verifier_ref.as_str(),
        possession.expected_verification_tool_digest.as_str(),
    ] {
        push_field(&mut hasher, value);
    }
    push_field(&mut hasher, &possession.max_challenge_lifetime_ms.to_string());
    push_field(&mut hasher, &possession.max_quote_to_verification_ms.to_string());
    push_field(&mut hasher, if possession.require_fixed_tpm { "1" } else { "0" });
    push_field(
        &mut hasher,
        if possession.require_restricted_signing { "1" } else { "0" },
    );
    push_sorted_refs(&mut hasher, "possession-evidence", &possession.evidence_refs);

    push_field(&mut hasher, "replay-policy");
    let replay = &inputs.replay_policy;
    for value in [
        replay.schema_version.as_str(),
        replay.policy_id.as_str(),
        replay.expected_possession_digest.as_str(),
        replay.expected_platform_qualification_digest.as_str(),
        replay.expected_pcr_selection_digest.as_str(),
        replay.expected_boot_session_id.as_str(),
        replay.expected_verifier_ref.as_str(),
        replay.expected_replay_tool_digest.as_str(),
    ] {
        push_field(&mut hasher, value);
    }
    push_field(&mut hasher, if replay.require_final_events { "1" } else { "0" });
    push_field(&mut hasher, &replay.max_possession_to_replay_ms.to_string());
    push_field(&mut hasher, &replay.max_log_to_verification_ms.to_string());
    push_sorted_refs(&mut hasher, "replay-evidence", &replay.evidence_refs);

    push_field(&mut hasher, "reference-policy");
    let reference = &inputs.reference_policy;
    for value in [
        reference.schema_version.as_str(),
        reference.policy_id.as_str(),
        reference.expected_replay_record_digest.as_str(),
        reference.expected_manifest_digest.as_str(),
        reference.expected_subject_class_ref.as_str(),
        reference.expected_release_ref.as_str(),
        reference.expected_manifest_verifier_ref.as_str(),
        reference.expected_manifest_verification_tool_digest.as_str(),
        reference.expected_extraction_verifier_ref.as_str(),
        reference.expected_extraction_tool_digest.as_str(),
    ] {
        push_field(&mut hasher, value);
    }
    push_field(
        &mut hasher,
        if reference.allow_unknown_noncritical { "1" } else { "0" },
    );
    push_field(&mut hasher, &reference.max_replay_to_evaluation_ms.to_string());
    push_field(
        &mut hasher,
        &reference.max_manifest_verification_age_ms.to_string(),
    );
    push_field(
        &mut hasher,
        &reference.max_extraction_verification_age_ms.to_string(),
    );
    push_sorted_refs(&mut hasher, "reference-evidence", &reference.evidence_refs);

    push_field(&mut hasher, "lineage-policy");
    let lineage = &inputs.lineage_policy;
    for value in [
        lineage.schema_version.as_str(),
        lineage.policy_id.as_str(),
        lineage.expected_manifest_id.as_str(),
        lineage.expected_issuer_ref.as_str(),
        lineage.expected_subject_class_ref.as_str(),
        lineage.expected_signature_verifier_ref.as_str(),
        lineage.expected_signature_verification_tool_digest.as_str(),
        lineage.expected_tip_manifest_digest.as_str(),
    ] {
        push_field(&mut hasher, value);
    }
    push_field(&mut hasher, &lineage.expected_tip_revision.to_string());
    let mut signer_rules = lineage.signer_rules.iter().collect::<Vec<_>>();
    signer_rules.sort_by(|left, right| left.rule_id.cmp(&right.rule_id));
    for rule in signer_rules {
        push_field(&mut hasher, "signer-rule");
        push_field(&mut hasher, &rule.rule_id);
        push_field(&mut hasher, &rule.signer_ref);
        push_field(&mut hasher, &rule.signer_key_digest);
        push_field(&mut hasher, &rule.first_revision.to_string());
        push_field(
            &mut hasher,
            &rule.last_revision.map(|value| value.to_string()).unwrap_or_default(),
        );
        push_sorted_refs(&mut hasher, "signer-evidence", &rule.evidence_refs);
    }
    push_sorted_refs(&mut hasher, "lineage-evidence", &lineage.evidence_refs);

    format!("blake3:{}", hasher.finalize().to_hex())
}

fn cross_check_reference_lineage(
    reference: &ReferenceIntegrityEvaluationReport,
    lineage: &ReferenceIntegrityLineageReport,
    manifest: &ReferenceIntegrityManifest,
) -> Result<(), TpmReferenceChainError> {
    if lineage.tip_manifest_digest != reference.manifest_digest
        || lineage.manifest_digests.last() != Some(&reference.manifest_digest)
    {
        return Err(TpmReferenceChainError::ReferenceLineageTipMismatch);
    }
    if lineage.signature_receipt_digests.last()
        != Some(&reference.manifest_signature_receipt_digest)
    {
        return Err(TpmReferenceChainError::ReferenceLineageSignatureSplice);
    }
    if lineage.tip_revision != manifest.revision
        || lineage.manifest_id != manifest.manifest_id
        || lineage.subject_class_ref != manifest.subject_class_ref
    {
        return Err(TpmReferenceChainError::ReferenceLineageRevisionMismatch);
    }
    Ok(())
}

fn ensure_fresh(
    now_ms: u64,
    evidence_at_ms: u64,
    max_age_ms: u64,
    label: &'static str,
) -> Result<(), TpmReferenceChainError> {
    if evidence_at_ms > now_ms {
        return Err(TpmReferenceChainError::EvidenceFromFuture(label));
    }
    if now_ms - evidence_at_ms > max_age_ms {
        return Err(TpmReferenceChainError::StaleEvidence(label));
    }
    Ok(())
}

fn push_sorted_refs(hasher: &mut blake3::Hasher, label: &str, refs: &[String]) {
    let mut refs = refs.to_vec();
    refs.sort();
    for reference in refs {
        push_field(hasher, label);
        push_field(hasher, &reference);
    }
}

fn canonical_text(value: &str) -> bool {
    !value.is_empty() && value.trim() == value
}

fn nonempty_refs(values: &[String]) -> bool {
    !values.is_empty() && values.iter().all(|value| canonical_text(value))
}

fn valid_digest(value: &str) -> bool {
    value.strip_prefix("blake3:").is_some_and(|digest| {
        digest.len() == 64
            && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
            && !digest.bytes().any(|byte| byte.is_ascii_uppercase())
    })
}

fn push_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(label: &str) -> String {
        format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex())
    }

    fn chain_policy() -> TpmReferenceChainPolicy {
        TpmReferenceChainPolicy {
            schema_version: "1".into(),
            campaign_id: "campaign:policy-digest".into(),
            expected_adapter_id: "adapter:1".into(),
            expected_trust_store_ref: "trust-store:1".into(),
            expected_runtime_subject_digest: d("runtime"),
            expected_lower_policy_bundle_digest: d("lower-policy-bundle"),
            expected_reference_policy_id: "reference-policy:1".into(),
            expected_lineage_policy_id: "lineage-policy:1".into(),
            max_platform_age_ms: 1_000,
            max_possession_age_ms: 1_000,
            max_replay_age_ms: 1_000,
            max_reference_age_ms: 1_000,
            evidence_refs: vec!["review:campaign-policy".into()],
        }
    }

    #[test]
    fn top_level_policy_digest_changes_when_freshness_semantics_change() {
        let policy = chain_policy();
        let original = policy.policy_digest();
        let mut weakened = policy.clone();
        weakened.max_reference_age_ms = 10_000;
        assert_ne!(original, weakened.policy_digest());
    }
}
