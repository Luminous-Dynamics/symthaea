// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-stage common-cause diversity for TPM/reference safety evidence.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_domain_awareness_evidence::IndependentVerification;
use symthaea_evidence_verifier_diversity::VerifierFaultDomainProfile;
use symthaea_formal_safety::SafetyEvidenceReceipt;
use symthaea_tpm_reference_evidence::TpmReferenceCandidateEvidence;

const POLICY_DIGEST_SCHEMA: &[u8] = b"symthaea-tpm-reference-cross-stage-diversity-policy-v1\0";
const PROFILE_DIGEST_SCHEMA: &[u8] = b"symthaea-verifier-fault-domain-profile-v1\0";
const REPORT_DIGEST_SCHEMA: &[u8] = b"symthaea-tpm-reference-cross-stage-diversity-report-v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrossStageVerifierDiversityPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub obligation_key: String,
    pub require_distinct_organization_domain: bool,
    pub require_distinct_review_process_domain: bool,
    pub require_distinct_toolchain_domain: bool,
    pub require_distinct_evidence_source_domain: bool,
    pub max_obligation_review_lag_ms: u64,
    pub evidence_refs: Vec<String>,
}

impl CrossStageVerifierDiversityPolicy {
    pub fn validate(&self) -> bool {
        canonical_text(&self.schema_version)
            && canonical_text(&self.policy_id)
            && valid_digest(&self.obligation_key)
            && (self.require_distinct_organization_domain
                || self.require_distinct_review_process_domain
                || self.require_distinct_toolchain_domain
                || self.require_distinct_evidence_source_domain)
            && self.max_obligation_review_lag_ms > 0
            && nonempty_refs(&self.evidence_refs)
    }

    pub fn policy_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(POLICY_DIGEST_SCHEMA);
        for value in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.obligation_key.as_str(),
        ] {
            push_field(&mut hasher, value);
        }
        for value in [
            self.require_distinct_organization_domain,
            self.require_distinct_review_process_domain,
            self.require_distinct_toolchain_domain,
            self.require_distinct_evidence_source_domain,
        ] {
            push_field(&mut hasher, if value { "1" } else { "0" });
        }
        push_field(
            &mut hasher,
            &self.max_obligation_review_lag_ms.to_string(),
        );
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        format!("blake3:{}", hasher.finalize().to_hex())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrossStageVerifierDiversityStatus {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrossStageVerifierDiversityIssue {
    InvalidPolicy,
    InvalidCandidate,
    InvalidVerification,
    ObligationMismatch,
    InvalidCampaignVerifierProfile,
    InvalidObligationVerifierProfile,
    CampaignVerifierProfileMismatch,
    ObligationVerifierProfileMismatch,
    SameVerifierIdentity,
    VerificationPredatesCampaign,
    VerificationTooLate,
    SharedOrganizationDomain,
    SharedReviewProcessDomain,
    SharedToolchainDomain,
    SharedEvidenceSourceDomain,
}

impl CrossStageVerifierDiversityIssue {
    fn code(&self) -> &'static str {
        match self {
            Self::InvalidPolicy => "invalid-policy",
            Self::InvalidCandidate => "invalid-candidate",
            Self::InvalidVerification => "invalid-verification",
            Self::ObligationMismatch => "obligation-mismatch",
            Self::InvalidCampaignVerifierProfile => "invalid-campaign-profile",
            Self::InvalidObligationVerifierProfile => "invalid-obligation-profile",
            Self::CampaignVerifierProfileMismatch => "campaign-profile-mismatch",
            Self::ObligationVerifierProfileMismatch => "obligation-profile-mismatch",
            Self::SameVerifierIdentity => "same-verifier",
            Self::VerificationPredatesCampaign => "verification-predates-campaign",
            Self::VerificationTooLate => "verification-too-late",
            Self::SharedOrganizationDomain => "shared-organization",
            Self::SharedReviewProcessDomain => "shared-review-process",
            Self::SharedToolchainDomain => "shared-toolchain",
            Self::SharedEvidenceSourceDomain => "shared-evidence-source",
        }
    }

    fn is_invalid(&self) -> bool {
        matches!(
            self,
            Self::InvalidPolicy
                | Self::InvalidCandidate
                | Self::InvalidVerification
                | Self::ObligationMismatch
                | Self::InvalidCampaignVerifierProfile
                | Self::InvalidObligationVerifierProfile
                | Self::CampaignVerifierProfileMismatch
                | Self::ObligationVerifierProfileMismatch
                | Self::VerificationPredatesCampaign
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrossStageVerifierDiversityReport {
    pub policy_id: String,
    pub policy_digest: String,
    pub candidate_digest: String,
    pub obligation_key: String,
    pub campaign_verifier_ref: String,
    pub campaign_profile_digest: String,
    pub obligation_verifier_ref: String,
    pub obligation_profile_digest: String,
    pub obligation_verification_receipt_id: String,
    pub campaign_verified_at_ms: u64,
    pub obligation_verified_at_ms: u64,
    pub status: CrossStageVerifierDiversityStatus,
    pub issues: Vec<CrossStageVerifierDiversityIssue>,
}

impl CrossStageVerifierDiversityReport {
    pub fn report_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(REPORT_DIGEST_SCHEMA);
        for value in [
            self.policy_id.as_str(),
            self.policy_digest.as_str(),
            self.candidate_digest.as_str(),
            self.obligation_key.as_str(),
            self.campaign_verifier_ref.as_str(),
            self.campaign_profile_digest.as_str(),
            self.obligation_verifier_ref.as_str(),
            self.obligation_profile_digest.as_str(),
            self.obligation_verification_receipt_id.as_str(),
        ] {
            push_field(&mut hasher, value);
        }
        push_field(&mut hasher, &self.campaign_verified_at_ms.to_string());
        push_field(&mut hasher, &self.obligation_verified_at_ms.to_string());
        push_field(
            &mut hasher,
            match self.status {
                CrossStageVerifierDiversityStatus::Invalid => "invalid",
                CrossStageVerifierDiversityStatus::Blocked => "blocked",
                CrossStageVerifierDiversityStatus::Qualified => "qualified",
            },
        );
        for issue in &self.issues {
            push_field(&mut hasher, issue.code());
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrossStageQualifiedEvidence {
    pub report: CrossStageVerifierDiversityReport,
    pub receipt: SafetyEvidenceReceipt,
}

impl CrossStageQualifiedEvidence {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CrossStageVerifierDiversityError {
    NotQualified(CrossStageVerifierDiversityReport),
    BaseVerificationFailed,
}

pub fn assess_cross_stage_verifier_diversity(
    candidate: &TpmReferenceCandidateEvidence,
    verification: &IndependentVerification,
    campaign_profile: &VerifierFaultDomainProfile,
    obligation_profile: &VerifierFaultDomainProfile,
    policy: &CrossStageVerifierDiversityPolicy,
) -> CrossStageVerifierDiversityReport {
    let mut issues = Vec::new();

    if !policy.validate() {
        issues.push(CrossStageVerifierDiversityIssue::InvalidPolicy);
    }
    if !candidate.validate() {
        issues.push(CrossStageVerifierDiversityIssue::InvalidCandidate);
    }
    if !verification.validate() {
        issues.push(CrossStageVerifierDiversityIssue::InvalidVerification);
    }
    if policy.obligation_key != candidate.obligation_key() {
        issues.push(CrossStageVerifierDiversityIssue::ObligationMismatch);
    }
    if !campaign_profile.validate() {
        issues.push(CrossStageVerifierDiversityIssue::InvalidCampaignVerifierProfile);
    }
    if !obligation_profile.validate() {
        issues.push(CrossStageVerifierDiversityIssue::InvalidObligationVerifierProfile);
    }
    if campaign_profile.verifier_ref != candidate.campaign_verifier_ref {
        issues.push(CrossStageVerifierDiversityIssue::CampaignVerifierProfileMismatch);
    }
    if obligation_profile.verifier_ref != verification.verifier_ref {
        issues.push(CrossStageVerifierDiversityIssue::ObligationVerifierProfileMismatch);
    }
    if verification.verifier_ref == candidate.campaign_verifier_ref {
        issues.push(CrossStageVerifierDiversityIssue::SameVerifierIdentity);
    }

    if verification.verified_at_ms < candidate.campaign_verified_at_ms {
        issues.push(CrossStageVerifierDiversityIssue::VerificationPredatesCampaign);
    } else if verification.verified_at_ms - candidate.campaign_verified_at_ms
        > policy.max_obligation_review_lag_ms
    {
        issues.push(CrossStageVerifierDiversityIssue::VerificationTooLate);
    }

    if policy.require_distinct_organization_domain
        && campaign_profile.organization_domain == obligation_profile.organization_domain
    {
        issues.push(CrossStageVerifierDiversityIssue::SharedOrganizationDomain);
    }
    if policy.require_distinct_review_process_domain
        && campaign_profile.review_process_domain == obligation_profile.review_process_domain
    {
        issues.push(CrossStageVerifierDiversityIssue::SharedReviewProcessDomain);
    }
    if policy.require_distinct_toolchain_domain
        && campaign_profile.toolchain_domain == obligation_profile.toolchain_domain
    {
        issues.push(CrossStageVerifierDiversityIssue::SharedToolchainDomain);
    }
    if policy.require_distinct_evidence_source_domain
        && campaign_profile.evidence_source_domain == obligation_profile.evidence_source_domain
    {
        issues.push(CrossStageVerifierDiversityIssue::SharedEvidenceSourceDomain);
    }

    let status = if issues.iter().any(CrossStageVerifierDiversityIssue::is_invalid) {
        CrossStageVerifierDiversityStatus::Invalid
    } else if issues.is_empty() {
        CrossStageVerifierDiversityStatus::Qualified
    } else {
        CrossStageVerifierDiversityStatus::Blocked
    };

    CrossStageVerifierDiversityReport {
        policy_id: policy.policy_id.clone(),
        policy_digest: policy.policy_digest(),
        candidate_digest: candidate.evidence_digest.clone(),
        obligation_key: candidate.obligation_key(),
        campaign_verifier_ref: candidate.campaign_verifier_ref.clone(),
        campaign_profile_digest: verifier_profile_digest(campaign_profile),
        obligation_verifier_ref: verification.verifier_ref.clone(),
        obligation_profile_digest: verifier_profile_digest(obligation_profile),
        obligation_verification_receipt_id: verification.receipt_id.clone(),
        campaign_verified_at_ms: candidate.campaign_verified_at_ms,
        obligation_verified_at_ms: verification.verified_at_ms,
        status,
        issues,
    }
}

pub fn verify_with_cross_stage_diversity(
    candidate: &TpmReferenceCandidateEvidence,
    verification: &IndependentVerification,
    campaign_profile: &VerifierFaultDomainProfile,
    obligation_profile: &VerifierFaultDomainProfile,
    policy: &CrossStageVerifierDiversityPolicy,
) -> Result<CrossStageQualifiedEvidence, CrossStageVerifierDiversityError> {
    let report = assess_cross_stage_verifier_diversity(
        candidate,
        verification,
        campaign_profile,
        obligation_profile,
        policy,
    );
    if report.status != CrossStageVerifierDiversityStatus::Qualified {
        return Err(CrossStageVerifierDiversityError::NotQualified(report));
    }

    let mut receipt = candidate
        .verify(verification)
        .map_err(|_| CrossStageVerifierDiversityError::BaseVerificationFailed)?;
    receipt.evidence_digest = report.report_digest();

    Ok(CrossStageQualifiedEvidence { report, receipt })
}

pub fn verifier_profile_digest(profile: &VerifierFaultDomainProfile) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PROFILE_DIGEST_SCHEMA);
    for value in [
        profile.verifier_ref.as_str(),
        profile.organization_domain.as_str(),
        profile.review_process_domain.as_str(),
        profile.toolchain_domain.as_str(),
        profile.evidence_source_domain.as_str(),
    ] {
        push_field(&mut hasher, value);
    }
    push_sorted_refs(&mut hasher, &profile.evidence_refs);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn canonical_text(value: &str) -> bool {
    !value.is_empty() && value.trim() == value
}

fn valid_digest(value: &str) -> bool {
    value.strip_prefix("blake3:").is_some_and(|digest| {
        digest.len() == 64
            && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
            && !digest.bytes().any(|byte| byte.is_ascii_uppercase())
    })
}

fn nonempty_refs(values: &[String]) -> bool {
    !values.is_empty() && values.iter().all(|value| canonical_text(value))
}

fn push_sorted_refs(hasher: &mut blake3::Hasher, refs: &[String]) {
    let mut refs = refs.to_vec();
    refs.sort();
    for reference in refs {
        push_field(hasher, &reference);
    }
}

fn push_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_formal_safety::{SafetyCase, SafetyCaseTemplate};
    use symthaea_tpm_reference_evidence::{
        TpmReferenceCampaignVerification, TpmReferenceEvidenceBindings,
        TpmReferenceEvidencePolicy, tpm_reference_candidates,
    };
    use symthaea_assurance_tpm_reference_crucible::TpmReferenceChainReport;
    use symthaea_domain_awareness_evidence::ArtifactBinding;

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

    fn evidence_policy(report: &TpmReferenceChainReport) -> TpmReferenceEvidencePolicy {
        TpmReferenceEvidencePolicy {
            schema_version: "1".into(),
            policy_id: "policy:evidence".into(),
            expected_chain_policy_digest: report.chain_policy_digest.clone(),
            expected_lower_policy_bundle_digest: report.lower_policy_bundle_digest.clone(),
            expected_campaign_verifier_ref: "verifier:campaign".into(),
            expected_campaign_verification_tool_digest: d("tool:campaign"),
            max_campaign_verification_lag_ms: 1_000,
            evidence_refs: vec!["review:evidence-policy".into()],
        }
    }

    fn campaign_verification(
        report: &TpmReferenceChainReport,
    ) -> TpmReferenceCampaignVerification {
        TpmReferenceCampaignVerification {
            schema_version: "1".into(),
            verification_id: "verification:campaign".into(),
            report_digest: report.report_digest(),
            chain_policy_digest: report.chain_policy_digest.clone(),
            lower_policy_bundle_digest: report.lower_policy_bundle_digest.clone(),
            verifier_ref: "verifier:campaign".into(),
            verification_tool_ref: "tool:campaign".into(),
            verification_tool_digest: d("tool:campaign"),
            verified_at_ms: 10_500,
            chain_recomputed: true,
            chain_policy_digest_checked: true,
            policy_bundle_pin_checked: true,
            evidence_refs: vec!["audit:campaign".into()],
        }
    }

    fn binding(reference: &str, digest: String) -> ArtifactBinding {
        ArtifactBinding {
            evidence_ref: reference.into(),
            evidence_digest: digest,
        }
    }

    fn bindings(report: &TpmReferenceChainReport) -> TpmReferenceEvidenceBindings {
        TpmReferenceEvidenceBindings {
            campaign: binding("artifact:campaign", report.report_digest()),
            platform: binding("artifact:platform", report.platform_qualification_digest.clone()),
            before_counter: binding("artifact:before", report.before_observation_digest.clone()),
            after_counter: binding("artifact:after", report.after_observation_digest.clone()),
            possession: binding("artifact:possession", report.possession_digest.clone()),
            replay: binding("artifact:replay", report.replay_record_digest.clone()),
            reference: binding("artifact:reference", report.reference_report_digest.clone()),
            current_manifest: binding("artifact:manifest", report.current_manifest_digest.clone()),
            current_manifest_signature: binding(
                "artifact:signature",
                report.current_signature_receipt_digest.clone(),
            ),
            lineage: binding("artifact:lineage", report.lineage_digest.clone()),
        }
    }

    fn candidate() -> TpmReferenceCandidateEvidence {
        let report = report();
        let candidates = tpm_reference_candidates(
            &evidence_policy(&report),
            &report,
            &campaign_verification(&report),
            &bindings(&report),
        )
        .unwrap();
        candidates[0].clone()
    }

    fn verification(verifier: &str, at_ms: u64) -> IndependentVerification {
        IndependentVerification {
            receipt_id: format!("receipt:{verifier}"),
            verifier_ref: verifier.into(),
            verified_at_ms: at_ms,
        }
    }

    fn profile(
        verifier: &str,
        org: &str,
        process: &str,
        tool: &str,
        source: &str,
    ) -> VerifierFaultDomainProfile {
        VerifierFaultDomainProfile {
            verifier_ref: verifier.into(),
            organization_domain: org.into(),
            review_process_domain: process.into(),
            toolchain_domain: tool.into(),
            evidence_source_domain: source.into(),
            evidence_refs: vec![format!("profile:{verifier}")],
        }
    }

    fn policy(candidate: &TpmReferenceCandidateEvidence) -> CrossStageVerifierDiversityPolicy {
        CrossStageVerifierDiversityPolicy {
            schema_version: "1".into(),
            policy_id: "policy:cross-stage:1".into(),
            obligation_key: candidate.obligation_key(),
            require_distinct_organization_domain: true,
            require_distinct_review_process_domain: true,
            require_distinct_toolchain_domain: true,
            require_distinct_evidence_source_domain: true,
            max_obligation_review_lag_ms: 2_000,
            evidence_refs: vec!["review:cross-stage-diversity".into()],
        }
    }

    #[test]
    fn fully_diverse_stages_emit_stronger_receipt() {
        let candidate = candidate();
        let verification = verification("verifier:obligation", 11_000);
        let campaign = profile(
            "verifier:campaign",
            "org:campaign",
            "process:campaign",
            "tool:campaign",
            "source:campaign",
        );
        let obligation = profile(
            "verifier:obligation",
            "org:obligation",
            "process:obligation",
            "tool:obligation",
            "source:obligation",
        );
        let qualified = verify_with_cross_stage_diversity(
            &candidate,
            &verification,
            &campaign,
            &obligation,
            &policy(&candidate),
        )
        .unwrap();
        assert_eq!(qualified.report.status, CrossStageVerifierDiversityStatus::Qualified);
        assert_eq!(qualified.receipt.evidence_digest, qualified.report.report_digest());
        assert!(!qualified.grants_physical_authority());
        assert!(!qualified.report.grants_physical_authority());
    }

    #[test]
    fn different_ids_in_same_organization_are_blocked() {
        let candidate = candidate();
        let verification = verification("verifier:obligation", 11_000);
        let campaign = profile(
            "verifier:campaign",
            "org:shared",
            "process:a",
            "tool:a",
            "source:a",
        );
        let obligation = profile(
            "verifier:obligation",
            "org:shared",
            "process:b",
            "tool:b",
            "source:b",
        );
        let report = assess_cross_stage_verifier_diversity(
            &candidate,
            &verification,
            &campaign,
            &obligation,
            &policy(&candidate),
        );
        assert_eq!(report.status, CrossStageVerifierDiversityStatus::Blocked);
        assert!(report
            .issues
            .contains(&CrossStageVerifierDiversityIssue::SharedOrganizationDomain));
    }

    #[test]
    fn each_required_common_cause_domain_blocks_independently() {
        let candidate = candidate();
        let verification = verification("verifier:obligation", 11_000);
        let campaign = profile(
            "verifier:campaign",
            "org:a",
            "process:a",
            "tool:a",
            "source:a",
        );

        for (obligation, expected) in [
            (
                profile("verifier:obligation", "org:b", "process:a", "tool:b", "source:b"),
                CrossStageVerifierDiversityIssue::SharedReviewProcessDomain,
            ),
            (
                profile("verifier:obligation", "org:b", "process:b", "tool:a", "source:b"),
                CrossStageVerifierDiversityIssue::SharedToolchainDomain,
            ),
            (
                profile("verifier:obligation", "org:b", "process:b", "tool:b", "source:a"),
                CrossStageVerifierDiversityIssue::SharedEvidenceSourceDomain,
            ),
        ] {
            let report = assess_cross_stage_verifier_diversity(
                &candidate,
                &verification,
                &campaign,
                &obligation,
                &policy(&candidate),
            );
            assert_eq!(report.status, CrossStageVerifierDiversityStatus::Blocked);
            assert!(report.issues.contains(&expected));
        }
    }

    #[test]
    fn missing_or_mismatched_profiles_fail_closed() {
        let candidate = candidate();
        let verification = verification("verifier:obligation", 11_000);
        let campaign = profile(
            "verifier:not-campaign",
            "org:a",
            "process:a",
            "tool:a",
            "source:a",
        );
        let obligation = profile(
            "verifier:obligation",
            "org:b",
            "process:b",
            "tool:b",
            "source:b",
        );
        let report = assess_cross_stage_verifier_diversity(
            &candidate,
            &verification,
            &campaign,
            &obligation,
            &policy(&candidate),
        );
        assert_eq!(report.status, CrossStageVerifierDiversityStatus::Invalid);
        assert!(report
            .issues
            .contains(&CrossStageVerifierDiversityIssue::CampaignVerifierProfileMismatch));
    }

    #[test]
    fn stale_obligation_review_is_blocked() {
        let candidate = candidate();
        let verification = verification("verifier:obligation", 12_501);
        let campaign = profile(
            "verifier:campaign",
            "org:a",
            "process:a",
            "tool:a",
            "source:a",
        );
        let obligation = profile(
            "verifier:obligation",
            "org:b",
            "process:b",
            "tool:b",
            "source:b",
        );
        let report = assess_cross_stage_verifier_diversity(
            &candidate,
            &verification,
            &campaign,
            &obligation,
            &policy(&candidate),
        );
        assert_eq!(report.status, CrossStageVerifierDiversityStatus::Blocked);
        assert!(report
            .issues
            .contains(&CrossStageVerifierDiversityIssue::VerificationTooLate));
    }

    #[test]
    fn policy_and_profile_changes_change_qualification_digest() {
        let candidate = candidate();
        let verification = verification("verifier:obligation", 11_000);
        let campaign = profile(
            "verifier:campaign",
            "org:a",
            "process:a",
            "tool:a",
            "source:a",
        );
        let obligation_a = profile(
            "verifier:obligation",
            "org:b",
            "process:b",
            "tool:b",
            "source:b",
        );
        let obligation_b = profile(
            "verifier:obligation",
            "org:c",
            "process:b",
            "tool:b",
            "source:b",
        );
        let policy_a = policy(&candidate);
        let mut policy_b = policy_a.clone();
        policy_b.max_obligation_review_lag_ms = 3_000;

        let report_a = assess_cross_stage_verifier_diversity(
            &candidate,
            &verification,
            &campaign,
            &obligation_a,
            &policy_a,
        );
        let report_b = assess_cross_stage_verifier_diversity(
            &candidate,
            &verification,
            &campaign,
            &obligation_b,
            &policy_b,
        );
        assert_eq!(report_a.status, CrossStageVerifierDiversityStatus::Qualified);
        assert_eq!(report_b.status, CrossStageVerifierDiversityStatus::Qualified);
        assert_ne!(report_a.report_digest(), report_b.report_digest());
    }

    #[test]
    fn stronger_receipt_still_does_not_make_open_case_ready() {
        let candidate = candidate();
        let verification = verification("verifier:obligation", 11_000);
        let campaign = profile(
            "verifier:campaign",
            "org:a",
            "process:a",
            "tool:a",
            "source:a",
        );
        let obligation = profile(
            "verifier:obligation",
            "org:b",
            "process:b",
            "tool:b",
            "source:b",
        );
        let qualified = verify_with_cross_stage_diversity(
            &candidate,
            &verification,
            &campaign,
            &obligation,
            &policy(&candidate),
        )
        .unwrap();
        let case = SafetyCase::from_template(
            "domain-awareness-node",
            SafetyCaseTemplate::DomainAwareness,
        );
        assert!(!case.is_strictly_ready(&[qualified.receipt]));
    }
}
