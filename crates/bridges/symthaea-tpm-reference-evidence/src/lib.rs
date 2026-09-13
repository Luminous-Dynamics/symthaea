// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed candidate-evidence bindings for TPM/reference assurance obligations.
//!
//! Candidate evidence is not proof, readiness, or physical authority. The bridge
//! preserves independent verification boundaries and emits canonical
//! `SafetyEvidenceReceipt`s only after explicit review.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_assurance_tpm_reference_crucible::TpmReferenceChainReport;
use symthaea_domain_awareness_evidence::{ArtifactBinding, IndependentVerification};
use symthaea_formal_safety::{
    DomainAwarenessTpmObligation, EvidenceKind, SafetyEvidenceReceipt,
};

const EVIDENCE_POLICY_DIGEST_SCHEMA: &[u8] =
    b"symthaea-tpm-reference-evidence-policy-v1\0";
const CAMPAIGN_VERIFICATION_DIGEST_SCHEMA: &[u8] =
    b"symthaea-tpm-reference-campaign-verification-v2\0";
const CANDIDATE_DIGEST_SCHEMA: &[u8] = b"symthaea-tpm-reference-candidate-evidence-v4\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TpmReferenceEvidenceBindings {
    pub campaign: ArtifactBinding,
    pub platform: ArtifactBinding,
    pub before_counter: ArtifactBinding,
    pub after_counter: ArtifactBinding,
    pub possession: ArtifactBinding,
    pub replay: ArtifactBinding,
    pub reference: ArtifactBinding,
    pub current_manifest: ArtifactBinding,
    pub current_manifest_signature: ArtifactBinding,
    pub lineage: ArtifactBinding,
}

impl TpmReferenceEvidenceBindings {
    fn validate_for(&self, report: &TpmReferenceChainReport) -> bool {
        [
            &self.campaign,
            &self.platform,
            &self.before_counter,
            &self.after_counter,
            &self.possession,
            &self.replay,
            &self.reference,
            &self.current_manifest,
            &self.current_manifest_signature,
            &self.lineage,
        ]
        .into_iter()
        .all(ArtifactBinding::validate)
            && self.campaign.evidence_digest == report.report_digest()
            && self.platform.evidence_digest == report.platform_qualification_digest
            && self.before_counter.evidence_digest == report.before_observation_digest
            && self.after_counter.evidence_digest == report.after_observation_digest
            && self.possession.evidence_digest == report.possession_digest
            && self.replay.evidence_digest == report.replay_record_digest
            && self.reference.evidence_digest == report.reference_report_digest
            && self.current_manifest.evidence_digest == report.current_manifest_digest
            && self.current_manifest_signature.evidence_digest
                == report.current_signature_receipt_digest
            && self.lineage.evidence_digest == report.lineage_digest
    }
}

/// Reviewed admission policy for end-to-end TPM/reference campaign evidence.
///
/// The campaign verifier and verification tool are externally pinned here rather
/// than being self-selected by the verification receipt. The exact top-level and
/// lower-level policy identities are also pinned before candidate evidence exists.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TpmReferenceEvidencePolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_chain_policy_digest: String,
    pub expected_lower_policy_bundle_digest: String,
    pub expected_campaign_verifier_ref: String,
    pub expected_campaign_verification_tool_digest: String,
    pub max_campaign_verification_lag_ms: u64,
    pub evidence_refs: Vec<String>,
}

impl TpmReferenceEvidencePolicy {
    pub fn validate(&self) -> bool {
        canonical_text(&self.schema_version)
            && canonical_text(&self.policy_id)
            && valid_digest(&self.expected_chain_policy_digest)
            && valid_digest(&self.expected_lower_policy_bundle_digest)
            && canonical_text(&self.expected_campaign_verifier_ref)
            && valid_digest(&self.expected_campaign_verification_tool_digest)
            && self.max_campaign_verification_lag_ms > 0
            && nonempty_refs(&self.evidence_refs)
    }

    pub fn policy_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(EVIDENCE_POLICY_DIGEST_SCHEMA);
        for value in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.expected_chain_policy_digest.as_str(),
            self.expected_lower_policy_bundle_digest.as_str(),
            self.expected_campaign_verifier_ref.as_str(),
            self.expected_campaign_verification_tool_digest.as_str(),
        ] {
            push_field(&mut hasher, value);
        }
        push_field(
            &mut hasher,
            &self.max_campaign_verification_lag_ms.to_string(),
        );
        let mut refs = self.evidence_refs.clone();
        refs.sort();
        for reference in refs {
            push_field(&mut hasher, &reference);
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }
}

/// Independent verification that a serialized campaign report was reproduced from
/// the underlying qualification inputs and both reviewed policy identities were
/// checked during that recomputation.
///
/// This is intentionally a separate boundary from the later obligation-specific
/// safety review. A structurally plausible `TpmReferenceChainReport` alone is not
/// eligible to mint DA-038..DA-042 candidates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TpmReferenceCampaignVerification {
    pub schema_version: String,
    pub verification_id: String,
    pub report_digest: String,
    pub chain_policy_digest: String,
    pub lower_policy_bundle_digest: String,
    pub verifier_ref: String,
    pub verification_tool_ref: String,
    pub verification_tool_digest: String,
    pub verified_at_ms: u64,
    pub chain_recomputed: bool,
    pub chain_policy_digest_checked: bool,
    pub policy_bundle_pin_checked: bool,
    pub evidence_refs: Vec<String>,
}

impl TpmReferenceCampaignVerification {
    pub fn validate_for(
        &self,
        policy: &TpmReferenceEvidencePolicy,
        report: &TpmReferenceChainReport,
    ) -> bool {
        if !policy.validate() || self.verified_at_ms < report.qualified_at_ms {
            return false;
        }
        let lag = self.verified_at_ms - report.qualified_at_ms;
        canonical_text(&self.schema_version)
            && canonical_text(&self.verification_id)
            && self.report_digest == report.report_digest()
            && self.chain_policy_digest == report.chain_policy_digest
            && self.chain_policy_digest == policy.expected_chain_policy_digest
            && self.lower_policy_bundle_digest == report.lower_policy_bundle_digest
            && self.lower_policy_bundle_digest == policy.expected_lower_policy_bundle_digest
            && self.verifier_ref == policy.expected_campaign_verifier_ref
            && canonical_text(&self.verification_tool_ref)
            && self.verification_tool_digest
                == policy.expected_campaign_verification_tool_digest
            && lag <= policy.max_campaign_verification_lag_ms
            && self.chain_recomputed
            && self.chain_policy_digest_checked
            && self.policy_bundle_pin_checked
            && nonempty_refs(&self.evidence_refs)
    }

    pub fn verification_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(CAMPAIGN_VERIFICATION_DIGEST_SCHEMA);
        for value in [
            self.schema_version.as_str(),
            self.verification_id.as_str(),
            self.report_digest.as_str(),
            self.chain_policy_digest.as_str(),
            self.lower_policy_bundle_digest.as_str(),
            self.verifier_ref.as_str(),
            self.verification_tool_ref.as_str(),
            self.verification_tool_digest.as_str(),
        ] {
            push_field(&mut hasher, value);
        }
        push_field(&mut hasher, &self.verified_at_ms.to_string());
        for value in [
            self.chain_recomputed,
            self.chain_policy_digest_checked,
            self.policy_bundle_pin_checked,
        ] {
            push_field(&mut hasher, if value { "1" } else { "0" });
        }
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TpmReferenceCandidateEvidence {
    pub candidate_id: String,
    pub obligation: DomainAwarenessTpmObligation,
    /// Durable reference to the end-to-end qualification campaign.
    pub evidence_ref: String,
    /// Digest over the exact campaign, admission policy, verification, and facets.
    pub evidence_digest: String,
    pub campaign_digest: String,
    pub evidence_policy_digest: String,
    pub campaign_verification_digest: String,
    /// Campaign-level verifier that reproduced the full chain.
    pub campaign_verifier_ref: String,
    /// Time at which the full campaign recomputation was independently verified.
    pub campaign_verified_at_ms: u64,
    pub facet_refs: Vec<String>,
    pub facet_digests: Vec<String>,
    pub observed_at_ms: u64,
    pub rationale: String,
}

impl TpmReferenceCandidateEvidence {
    pub fn validate(&self) -> bool {
        canonical_text(&self.candidate_id)
            && canonical_text(&self.evidence_ref)
            && valid_digest(&self.evidence_digest)
            && valid_digest(&self.campaign_digest)
            && valid_digest(&self.evidence_policy_digest)
            && valid_digest(&self.campaign_verification_digest)
            && canonical_text(&self.campaign_verifier_ref)
            && self.campaign_verified_at_ms >= self.observed_at_ms
            && !self.facet_refs.is_empty()
            && self.facet_refs.len() == self.facet_digests.len()
            && self.facet_refs.iter().all(|value| canonical_text(value))
            && self.facet_digests.iter().all(|value| valid_digest(value))
            && canonical_text(&self.rationale)
            && self.evidence_digest == self.recompute_digest()
    }

    pub const fn evidence_kind(&self) -> EvidenceKind {
        self.obligation.expected_evidence()
    }

    pub fn obligation_key(&self) -> String {
        self.obligation.stable_key()
    }

    /// Convert independently reviewed candidate evidence into a strict receipt.
    ///
    /// The obligation-specific verifier must be distinct from the campaign-level
    /// verifier that reproduced the full chain, and the review cannot predate that
    /// campaign verification. This still does not change workflow discharge state.
    pub fn verify(
        &self,
        verification: &IndependentVerification,
    ) -> Result<SafetyEvidenceReceipt, TpmReferenceEvidenceError> {
        if !self.validate() {
            return Err(TpmReferenceEvidenceError::InvalidCandidate);
        }
        if !verification.validate() || verification.verified_at_ms < self.campaign_verified_at_ms {
            return Err(TpmReferenceEvidenceError::InvalidVerification);
        }
        if verification.verifier_ref == self.campaign_verifier_ref {
            return Err(TpmReferenceEvidenceError::VerificationNotIndependent);
        }
        Ok(SafetyEvidenceReceipt {
            receipt_id: verification.receipt_id.clone(),
            obligation_key: self.obligation_key(),
            evidence_kind: self.evidence_kind(),
            evidence_ref: self.evidence_ref.clone(),
            evidence_digest: self.evidence_digest.clone(),
            verifier_ref: verification.verifier_ref.clone(),
            verified_at_ms: verification.verified_at_ms,
        })
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }

    fn recompute_digest(&self) -> String {
        candidate_digest(
            self.obligation,
            &self.evidence_ref,
            &self.campaign_digest,
            &self.evidence_policy_digest,
            &self.campaign_verification_digest,
            &self.campaign_verifier_ref,
            self.campaign_verified_at_ms,
            self.observed_at_ms,
            &self.rationale,
            &self.facet_refs,
            &self.facet_digests,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TpmReferenceEvidenceError {
    InvalidPolicy,
    InvalidReport,
    InvalidCampaignVerification,
    BindingMismatch,
    InvalidCandidate,
    InvalidVerification,
    VerificationNotIndependent,
}

pub fn tpm_reference_candidates(
    policy: &TpmReferenceEvidencePolicy,
    report: &TpmReferenceChainReport,
    campaign_verification: &TpmReferenceCampaignVerification,
    bindings: &TpmReferenceEvidenceBindings,
) -> Result<Vec<TpmReferenceCandidateEvidence>, TpmReferenceEvidenceError> {
    if !policy.validate() {
        return Err(TpmReferenceEvidenceError::InvalidPolicy);
    }
    if !valid_report(report) {
        return Err(TpmReferenceEvidenceError::InvalidReport);
    }
    if !campaign_verification.validate_for(policy, report) {
        return Err(TpmReferenceEvidenceError::InvalidCampaignVerification);
    }
    if !bindings.validate_for(report) {
        return Err(TpmReferenceEvidenceError::BindingMismatch);
    }

    let campaign_digest = report.report_digest();
    let evidence_policy_digest = policy.policy_digest();
    let campaign_verification_digest = campaign_verification.verification_digest();
    let observed_at_ms = report.qualified_at_ms;
    let campaign_ref = bindings.campaign.evidence_ref.clone();

    let specs = [
        (
            DomainAwarenessTpmObligation::HardwareTrustRootSubjectBindingIsExact,
            vec![&bindings.platform, &bindings.before_counter, &bindings.after_counter],
            "end-to-end qualification binds the exact reviewed TPM/platform/runtime subject and both monotonic counter observations",
        ),
        (
            DomainAwarenessTpmObligation::FreshAttestationKeyPossessionRequired,
            vec![&bindings.possession],
            "end-to-end qualification includes fresh nonce-bound possession of the reviewed attestation key",
        ),
        (
            DomainAwarenessTpmObligation::MeasuredBootReplayMustMatchFreshQuote,
            vec![&bindings.replay],
            "end-to-end qualification includes ordered measured-boot replay to the exact fresh quoted PCR state",
        ),
        (
            DomainAwarenessTpmObligation::ReferenceIntegrityApprovalRequired,
            vec![
                &bindings.reference,
                &bindings.current_manifest,
                &bindings.current_manifest_signature,
            ],
            "end-to-end qualification includes approved measured state under the exact current signed reference manifest",
        ),
        (
            DomainAwarenessTpmObligation::ReferenceIntegrityLineageMustBeCurrentAndAuthorized,
            vec![
                &bindings.lineage,
                &bindings.current_manifest,
                &bindings.current_manifest_signature,
            ],
            "end-to-end qualification binds the exact current manifest and signature receipt to the authorized signed reference lineage tip",
        ),
    ];

    specs
        .into_iter()
        .map(|(obligation, facets, rationale)| {
            let facet_refs = facets
                .iter()
                .map(|binding| binding.evidence_ref.clone())
                .collect::<Vec<_>>();
            let facet_digests = facets
                .iter()
                .map(|binding| binding.evidence_digest.clone())
                .collect::<Vec<_>>();
            let candidate = TpmReferenceCandidateEvidence {
                candidate_id: format!(
                    "{}:{}:{}",
                    obligation.code(), report.campaign_id, observed_at_ms
                ),
                obligation,
                evidence_ref: campaign_ref.clone(),
                evidence_digest: candidate_digest(
                    obligation,
                    &campaign_ref,
                    &campaign_digest,
                    &evidence_policy_digest,
                    &campaign_verification_digest,
                    &campaign_verification.verifier_ref,
                    campaign_verification.verified_at_ms,
                    observed_at_ms,
                    rationale,
                    &facet_refs,
                    &facet_digests,
                ),
                campaign_digest: campaign_digest.clone(),
                evidence_policy_digest: evidence_policy_digest.clone(),
                campaign_verification_digest: campaign_verification_digest.clone(),
                campaign_verifier_ref: campaign_verification.verifier_ref.clone(),
                campaign_verified_at_ms: campaign_verification.verified_at_ms,
                facet_refs,
                facet_digests,
                observed_at_ms,
                rationale: rationale.into(),
            };
            candidate
                .validate()
                .then_some(candidate)
                .ok_or(TpmReferenceEvidenceError::InvalidCandidate)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn candidate_digest(
    obligation: DomainAwarenessTpmObligation,
    campaign_ref: &str,
    campaign_digest: &str,
    evidence_policy_digest: &str,
    campaign_verification_digest: &str,
    campaign_verifier_ref: &str,
    campaign_verified_at_ms: u64,
    observed_at_ms: u64,
    rationale: &str,
    facet_refs: &[String],
    facet_digests: &[String],
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CANDIDATE_DIGEST_SCHEMA);
    push_field(&mut hasher, obligation.code());
    push_field(&mut hasher, &obligation.stable_key());
    push_field(&mut hasher, campaign_ref);
    push_field(&mut hasher, campaign_digest);
    push_field(&mut hasher, evidence_policy_digest);
    push_field(&mut hasher, campaign_verification_digest);
    push_field(&mut hasher, campaign_verifier_ref);
    push_field(&mut hasher, &campaign_verified_at_ms.to_string());
    push_field(&mut hasher, &observed_at_ms.to_string());
    push_field(&mut hasher, rationale);
    for (reference, digest) in facet_refs.iter().zip(facet_digests) {
        push_field(&mut hasher, reference);
        push_field(&mut hasher, digest);
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn valid_report(report: &TpmReferenceChainReport) -> bool {
    canonical_text(&report.schema_version)
        && canonical_text(&report.campaign_id)
        && valid_digest(&report.chain_policy_digest)
        && valid_digest(&report.lower_policy_bundle_digest)
        && valid_digest(&report.before_observation_digest)
        && valid_digest(&report.after_observation_digest)
        && valid_digest(&report.platform_qualification_digest)
        && valid_digest(&report.possession_digest)
        && valid_digest(&report.replay_record_digest)
        && valid_digest(&report.reference_report_digest)
        && valid_digest(&report.lineage_digest)
        && valid_digest(&report.current_manifest_digest)
        && valid_digest(&report.current_signature_receipt_digest)
        && nonempty_refs(&report.evidence_refs)
        && valid_digest(&report.report_digest())
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
    use symthaea_formal_safety::{SafetyCase, SafetyCaseTemplate};

    fn d(label: &str) -> String {
        format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex())
    }

    fn report() -> TpmReferenceChainReport {
        TpmReferenceChainReport {
            schema_version: "1".into(),
            campaign_id: "campaign:1".into(),
            chain_policy_digest: d("chain-policy"),
            lower_policy_bundle_digest: d("lower-policy-bundle"),
            before_observation_digest: d("counter-before"),
            after_observation_digest: d("counter-after"),
            platform_qualification_digest: d("platform"),
            possession_digest: d("possession"),
            replay_record_digest: d("replay"),
            reference_report_digest: d("reference"),
            lineage_digest: d("lineage"),
            current_manifest_digest: d("manifest"),
            current_signature_receipt_digest: d("manifest-signature"),
            qualified_at_ms: 10_000,
            evidence_refs: vec!["campaign:evidence".into()],
        }
    }

    fn policy(report: &TpmReferenceChainReport) -> TpmReferenceEvidencePolicy {
        TpmReferenceEvidencePolicy {
            schema_version: "1".into(),
            policy_id: "policy:tpm-reference-evidence:1".into(),
            expected_chain_policy_digest: report.chain_policy_digest.clone(),
            expected_lower_policy_bundle_digest: report.lower_policy_bundle_digest.clone(),
            expected_campaign_verifier_ref: "verifier:campaign-recompute".into(),
            expected_campaign_verification_tool_digest: d("tpm-reference-recompute-tool"),
            max_campaign_verification_lag_ms: 1_000,
            evidence_refs: vec!["review:evidence-admission-policy".into()],
        }
    }

    fn campaign_verification(report: &TpmReferenceChainReport) -> TpmReferenceCampaignVerification {
        TpmReferenceCampaignVerification {
            schema_version: "1".into(),
            verification_id: "verification:campaign:1".into(),
            report_digest: report.report_digest(),
            chain_policy_digest: report.chain_policy_digest.clone(),
            lower_policy_bundle_digest: report.lower_policy_bundle_digest.clone(),
            verifier_ref: "verifier:campaign-recompute".into(),
            verification_tool_ref: "tool:tpm-reference-recompute".into(),
            verification_tool_digest: d("tpm-reference-recompute-tool"),
            verified_at_ms: 10_500,
            chain_recomputed: true,
            chain_policy_digest_checked: true,
            policy_bundle_pin_checked: true,
            evidence_refs: vec!["audit:campaign-recompute".into()],
        }
    }

    fn b(reference: &str, digest: String) -> ArtifactBinding {
        ArtifactBinding {
            evidence_ref: reference.into(),
            evidence_digest: digest,
        }
    }

    fn bindings(report: &TpmReferenceChainReport) -> TpmReferenceEvidenceBindings {
        TpmReferenceEvidenceBindings {
            campaign: b("artifact:campaign", report.report_digest()),
            platform: b("artifact:platform", report.platform_qualification_digest.clone()),
            before_counter: b("artifact:counter-before", report.before_observation_digest.clone()),
            after_counter: b("artifact:counter-after", report.after_observation_digest.clone()),
            possession: b("artifact:possession", report.possession_digest.clone()),
            replay: b("artifact:replay", report.replay_record_digest.clone()),
            reference: b("artifact:reference", report.reference_report_digest.clone()),
            current_manifest: b("artifact:manifest", report.current_manifest_digest.clone()),
            current_manifest_signature: b(
                "artifact:manifest-signature",
                report.current_signature_receipt_digest.clone(),
            ),
            lineage: b("artifact:lineage", report.lineage_digest.clone()),
        }
    }

    #[test]
    fn exact_verified_chain_emits_five_atomic_candidates() {
        let report = report();
        let policy = policy(&report);
        let verification = campaign_verification(&report);
        let candidates = tpm_reference_candidates(
            &policy,
            &report,
            &verification,
            &bindings(&report),
        )
        .unwrap();
        assert_eq!(candidates.len(), 5);
        assert_eq!(
            candidates
                .iter()
                .map(|value| value.obligation.code())
                .collect::<Vec<_>>(),
            vec!["DA-038", "DA-039", "DA-040", "DA-041", "DA-042"]
        );
        assert!(candidates.iter().all(TpmReferenceCandidateEvidence::validate));
        assert!(candidates.iter().all(|value| !value.grants_physical_authority()));
        assert!(!verification.grants_physical_authority());
        assert_ne!(candidates[0].evidence_digest, candidates[1].evidence_digest);
    }

    #[test]
    fn structurally_plausible_report_without_recomputation_verification_is_ineligible() {
        let report = report();
        let policy = policy(&report);
        let mut verification = campaign_verification(&report);
        verification.chain_recomputed = false;
        assert_eq!(
            tpm_reference_candidates(&policy, &report, &verification, &bindings(&report)),
            Err(TpmReferenceEvidenceError::InvalidCampaignVerification)
        );
    }

    #[test]
    fn top_level_chain_policy_identity_must_match_reviewed_policy() {
        let report = report();
        let policy = policy(&report);
        let mut verification = campaign_verification(&report);
        verification.chain_policy_digest = d("different-chain-policy");
        assert_eq!(
            tpm_reference_candidates(&policy, &report, &verification, &bindings(&report)),
            Err(TpmReferenceEvidenceError::InvalidCampaignVerification)
        );
    }

    #[test]
    fn campaign_verifier_and_tool_are_policy_bound() {
        let report = report();
        let policy = policy(&report);
        let mut verification = campaign_verification(&report);
        verification.verifier_ref = "verifier:self-selected".into();
        assert_eq!(
            tpm_reference_candidates(&policy, &report, &verification, &bindings(&report)),
            Err(TpmReferenceEvidenceError::InvalidCampaignVerification)
        );
    }

    #[test]
    fn campaign_verification_lag_is_bounded() {
        let report = report();
        let policy = policy(&report);
        let mut verification = campaign_verification(&report);
        verification.verified_at_ms = 11_001;
        assert_eq!(
            tpm_reference_candidates(&policy, &report, &verification, &bindings(&report)),
            Err(TpmReferenceEvidenceError::InvalidCampaignVerification)
        );
    }

    #[test]
    fn facet_substitution_cannot_create_candidates() {
        let report = report();
        let policy = policy(&report);
        let verification = campaign_verification(&report);
        let mut bindings = bindings(&report);
        bindings.possession.evidence_digest = report.replay_record_digest.clone();
        assert_eq!(
            tpm_reference_candidates(&policy, &report, &verification, &bindings),
            Err(TpmReferenceEvidenceError::BindingMismatch)
        );
    }

    #[test]
    fn campaign_reference_is_content_bound_inside_candidate() {
        let report = report();
        let policy = policy(&report);
        let verification = campaign_verification(&report);
        let candidates = tpm_reference_candidates(
            &policy,
            &report,
            &verification,
            &bindings(&report),
        )
        .unwrap();
        let mut candidate = candidates[0].clone();
        candidate.evidence_ref = "artifact:different-campaign-location".into();
        assert!(!candidate.validate());
    }

    #[test]
    fn evidence_admission_policy_is_content_bound_inside_candidate() {
        let report = report();
        let policy = policy(&report);
        let verification = campaign_verification(&report);
        let candidates = tpm_reference_candidates(
            &policy,
            &report,
            &verification,
            &bindings(&report),
        )
        .unwrap();
        let mut candidate = candidates[0].clone();
        candidate.evidence_policy_digest = d("other-evidence-policy");
        assert!(!candidate.validate());
    }

    #[test]
    fn campaign_verifier_identity_is_content_bound_inside_candidate() {
        let report = report();
        let policy = policy(&report);
        let verification = campaign_verification(&report);
        let candidates = tpm_reference_candidates(
            &policy,
            &report,
            &verification,
            &bindings(&report),
        )
        .unwrap();
        let mut candidate = candidates[0].clone();
        candidate.campaign_verifier_ref = "verifier:different-campaign-reviewer".into();
        assert!(!candidate.validate());
    }

    #[test]
    fn same_campaign_verifier_cannot_review_obligation() {
        let report = report();
        let policy = policy(&report);
        let verification = campaign_verification(&report);
        let candidates = tpm_reference_candidates(
            &policy,
            &report,
            &verification,
            &bindings(&report),
        )
        .unwrap();
        assert_eq!(
            candidates[0].verify(&IndependentVerification {
                receipt_id: "receipt:not-independent".into(),
                verifier_ref: verification.verifier_ref.clone(),
                verified_at_ms: 11_000,
            }),
            Err(TpmReferenceEvidenceError::VerificationNotIndependent)
        );
    }

    #[test]
    fn verified_candidate_still_does_not_make_open_case_ready() {
        let report = report();
        let policy = policy(&report);
        let verification = campaign_verification(&report);
        let candidates = tpm_reference_candidates(
            &policy,
            &report,
            &verification,
            &bindings(&report),
        )
        .unwrap();
        let receipt = candidates[0]
            .verify(&IndependentVerification {
                receipt_id: "receipt:tpm:1".into(),
                verifier_ref: "verifier:independent-safety".into(),
                verified_at_ms: 11_000,
            })
            .unwrap();
        let safety_case = SafetyCase::from_template(
            "domain-awareness-node",
            SafetyCaseTemplate::DomainAwareness,
        );
        assert!(!safety_case.is_strictly_ready(&[receipt]));
    }

    #[test]
    fn obligation_verification_cannot_predate_campaign_verification() {
        let report = report();
        let policy = policy(&report);
        let verification = campaign_verification(&report);
        let candidates = tpm_reference_candidates(
            &policy,
            &report,
            &verification,
            &bindings(&report),
        )
        .unwrap();
        assert_eq!(
            candidates[4].verify(&IndependentVerification {
                receipt_id: "receipt:early".into(),
                verifier_ref: "verifier:independent-safety".into(),
                verified_at_ms: 10_499,
            }),
            Err(TpmReferenceEvidenceError::InvalidVerification)
        );
    }
}
