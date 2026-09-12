// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deep readiness scoped to one exact reviewed assurance-policy manifest.

#![deny(unsafe_code)]

use std::collections::BTreeSet;

use symthaea_assurance_policy_manifest::{
    AssurancePolicyManifest, ManifestSignatureVerificationReceipt, PolicyScopedSafetyReceipt,
};
use symthaea_evidence_atomic_coverage::{AtomicCoveragePolicy, FacetEvidenceBinding};
use symthaea_evidence_deep_readiness::{
    DeepEvidenceReadinessReport, assess_deep_evidence_readiness,
};
use symthaea_evidence_deployment_scope::{
    DeploymentEvidenceContext, DeploymentScopedSafetyReceipt,
};
use symthaea_evidence_lifecycle::EvidenceLifecycleEvent;
use symthaea_evidence_quarantine::{
    EvidenceQuarantineDirective, EvidenceQuarantineResolution,
};
use symthaea_evidence_time_assurance::TrustedEvidenceTime;
use symthaea_evidence_verifier_diversity::{
    VerifierDiversityPolicy, VerifierFaultDomainProfile,
};
use symthaea_formal_safety::{SafetyCase, StrictSafetyCaseStatus};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyManifestReadinessIssue {
    InvalidManifest,
    SafetyContractMismatch,
    DeploymentContextMismatch,
    InvalidManifestSignatureVerification,
    SignatureVerifiedAfterTrustedInterval {
        verified_at_ms: u64,
        earliest_trusted_ms: u64,
    },
    InvalidPolicyScopedReceipt(String),
    DuplicateReceiptId(String),
    HistoricalManifestReceipt(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyManifestReadinessReport {
    pub manifest_digest: String,
    pub manifest_revision: u64,
    pub status: StrictSafetyCaseStatus,
    pub matched_receipt_count: usize,
    pub excluded_receipt_count: usize,
    pub issues: Vec<PolicyManifestReadinessIssue>,
    pub deep_report: DeepEvidenceReadinessReport,
}

impl PolicyManifestReadinessReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

/// Evaluate deep readiness using only receipts bound to the exact current
/// assurance-policy manifest.
///
/// Receipts from an older manifest are historical evidence and are excluded,
/// not treated as malformed. Structural problems with the current manifest,
/// safety/deployment scope, signature verification, or receipt identities fail
/// the assessment as `Invalid`.
pub fn assess_policy_manifest_readiness(
    safety_case: &SafetyCase,
    context: &DeploymentEvidenceContext,
    manifest: &AssurancePolicyManifest,
    signature_receipt: &ManifestSignatureVerificationReceipt,
    policy_receipts: &[PolicyScopedSafetyReceipt],
    lifecycle_events: &[EvidenceLifecycleEvent],
    quarantine_directives: &[EvidenceQuarantineDirective],
    quarantine_resolutions: &[EvidenceQuarantineResolution],
    trusted_time: &TrustedEvidenceTime,
    verifier_profiles: &[VerifierFaultDomainProfile],
    verifier_policy: &VerifierDiversityPolicy,
    atomic_policy: &AtomicCoveragePolicy,
    facet_bindings: &[FacetEvidenceBinding],
) -> PolicyManifestReadinessReport {
    let mut issues = Vec::new();

    if !manifest.validate_complete() {
        issues.push(PolicyManifestReadinessIssue::InvalidManifest);
    }
    if manifest.safety_contract_digest != safety_case.contract_digest() {
        issues.push(PolicyManifestReadinessIssue::SafetyContractMismatch);
    }
    if !context_matches_manifest(context, manifest) {
        issues.push(PolicyManifestReadinessIssue::DeploymentContextMismatch);
    }
    if !signature_receipt.validate_for(manifest) {
        issues.push(PolicyManifestReadinessIssue::InvalidManifestSignatureVerification);
    } else if signature_receipt.verified_at_ms > trusted_time.earliest_ms() {
        issues.push(PolicyManifestReadinessIssue::SignatureVerifiedAfterTrustedInterval {
            verified_at_ms: signature_receipt.verified_at_ms,
            earliest_trusted_ms: trusted_time.earliest_ms(),
        });
    }

    let mut seen_ids = BTreeSet::new();
    let mut matched = Vec::<DeploymentScopedSafetyReceipt>::new();
    for receipt in policy_receipts {
        let id = receipt
            .deployment_receipt
            .scoped_receipt
            .receipt
            .receipt_id
            .clone();
        if !receipt.validate() {
            issues.push(PolicyManifestReadinessIssue::InvalidPolicyScopedReceipt(id));
            continue;
        }
        if !seen_ids.insert(id.clone()) {
            issues.push(PolicyManifestReadinessIssue::DuplicateReceiptId(id));
            continue;
        }
        if receipt.matches_manifest(manifest) {
            matched.push(receipt.deployment_receipt.clone());
        } else {
            issues.push(PolicyManifestReadinessIssue::HistoricalManifestReceipt(id));
        }
    }

    let deep_report = assess_deep_evidence_readiness(
        safety_case,
        context,
        &matched,
        lifecycle_events,
        quarantine_directives,
        quarantine_resolutions,
        trusted_time,
        verifier_profiles,
        verifier_policy,
        atomic_policy,
        facet_bindings,
    );

    let structurally_invalid = issues.iter().any(|issue| {
        !matches!(issue, PolicyManifestReadinessIssue::HistoricalManifestReceipt(_))
    });
    let status = if structurally_invalid {
        StrictSafetyCaseStatus::Invalid
    } else {
        deep_report.status
    };

    PolicyManifestReadinessReport {
        manifest_digest: manifest.manifest_digest(),
        manifest_revision: manifest.revision,
        status,
        matched_receipt_count: matched.len(),
        excluded_receipt_count: policy_receipts.len().saturating_sub(matched.len()),
        issues,
        deep_report,
    }
}

fn context_matches_manifest(
    context: &DeploymentEvidenceContext,
    manifest: &AssurancePolicyManifest,
) -> bool {
    context.validate()
        && context.deployment_id == manifest.deployment_id
        && context.configuration_digest == manifest.configuration_digest
        && context.model_manifest_digest == manifest.model_manifest_digest
        && context.calibration_manifest_digest == manifest.calibration_manifest_digest
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_assurance_policy_manifest::{
        AssurancePolicyBinding, AssurancePolicyKind, bind_receipt_to_assurance_manifest,
    };
    use symthaea_evidence_atomic_coverage::AtomicEvidenceFacet;
    use symthaea_evidence_deployment_scope::bind_receipt_to_deployment;
    use symthaea_evidence_lifecycle::ScopedSafetyEvidenceReceipt;
    use symthaea_evidence_time_assurance::{
        EvidenceTimeGuard, EvidenceTimePolicy, EvidenceTimeSample,
    };
    use symthaea_evidence_verifier_diversity::VerifierDiversityRequirement;
    use symthaea_formal_safety::{EvidenceKind, ProofObligation, SafetyEvidenceReceipt};

    fn case() -> SafetyCase {
        let mut case = SafetyCase::new("policy-readiness-fixture");
        case.add_obligation(
            ProofObligation::new("composite claim", EvidenceKind::Test)
                .discharge("qualification:reviewed"),
        );
        case
    }

    fn context() -> DeploymentEvidenceContext {
        DeploymentEvidenceContext {
            schema_version: "1".into(),
            deployment_id: "node-1".into(),
            configuration_digest: "blake3:config-v1".into(),
            model_manifest_digest: Some("blake3:model-v1".into()),
            calibration_manifest_digest: Some("blake3:cal-v1".into()),
            evidence_refs: vec!["deployment:node-1".into()],
        }
    }

    fn binding(kind: AssurancePolicyKind) -> AssurancePolicyBinding {
        AssurancePolicyBinding {
            kind,
            policy_id: format!("policy:{}", kind.code()),
            schema_version: "1".into(),
            content_digest: format!("blake3:{}-v1", kind.code()),
            evidence_refs: vec![format!("review:{}", kind.code())],
        }
    }

    fn manifest(case: &SafetyCase) -> AssurancePolicyManifest {
        AssurancePolicyManifest {
            schema_version: "1".into(),
            manifest_id: "assurance:node-1".into(),
            revision: 1,
            safety_contract_digest: case.contract_digest(),
            deployment_id: "node-1".into(),
            configuration_digest: "blake3:config-v1".into(),
            model_manifest_digest: Some("blake3:model-v1".into()),
            calibration_manifest_digest: Some("blake3:cal-v1".into()),
            predecessor_manifest_digest: None,
            change_ref: None,
            policies: AssurancePolicyKind::REQUIRED
                .into_iter()
                .map(binding)
                .collect(),
            evidence_refs: vec!["review:manifest-v1".into()],
        }
    }

    fn signature(
        manifest: &AssurancePolicyManifest,
        verified_at_ms: u64,
    ) -> ManifestSignatureVerificationReceipt {
        ManifestSignatureVerificationReceipt {
            receipt_id: format!("sig:{}", manifest.revision),
            manifest_digest: manifest.manifest_digest(),
            signer_ref: "signer:safety-board".into(),
            key_ref: "key:safety-board:v1".into(),
            signature_algorithm: "ed25519".into(),
            signature_ref: format!("signature:manifest:{}", manifest.revision),
            verified_by_ref: "verifier:signature".into(),
            verification_ref: format!("verification:manifest:{}", manifest.revision),
            verified_at_ms,
            evidence_refs: vec!["audit:signature".into()],
        }
    }

    fn trusted_time() -> TrustedEvidenceTime {
        let policy = EvidenceTimePolicy {
            schema_version: "1".into(),
            policy_id: "time-v1".into(),
            expected_clock_source: "ptp-a".into(),
            expected_clock_domain: "domain-a".into(),
            expected_epoch_ref: "epoch-a".into(),
            maximum_clock_uncertainty_ms: 10,
            maximum_wall_clock_backstep_ms: 5,
        };
        let mut guard = EvidenceTimeGuard::new(policy).unwrap();
        guard
            .observe(EvidenceTimeSample {
                sample_id: "t1".into(),
                wall_time_ms: 1_000,
                monotonic_time_ms: 5_000,
                clock_source: "ptp-a".into(),
                clock_domain: "domain-a".into(),
                epoch_ref: "epoch-a".into(),
                clock_uncertainty_ms: 10,
                evidence_refs: vec!["clock:t1".into()],
            })
            .trusted_time
            .unwrap()
    }

    fn deployment_receipt(
        case: &SafetyCase,
        id: &str,
        verifier: &str,
    ) -> DeploymentScopedSafetyReceipt {
        let obligation = &case.obligations[0];
        let scoped = ScopedSafetyEvidenceReceipt {
            receipt: SafetyEvidenceReceipt {
                receipt_id: id.into(),
                obligation_key: obligation.stable_key(),
                evidence_kind: obligation.expected_evidence,
                evidence_ref: format!("artifact:{id}"),
                evidence_digest: format!("blake3:{id}"),
                verifier_ref: verifier.into(),
                verified_at_ms: 100,
            },
            contract_digest: case.contract_digest(),
            valid_from_ms: 100,
            valid_until_ms: 5_000,
            applicability_refs: vec!["deployment:node-1".into()],
        };
        bind_receipt_to_deployment(scoped, &context(), format!("scope:{id}")).unwrap()
    }

    fn policy_receipts(
        case: &SafetyCase,
        manifest: &AssurancePolicyManifest,
    ) -> Vec<PolicyScopedSafetyReceipt> {
        let sig = signature(manifest, 500);
        [
            deployment_receipt(case, "r1", "verifier:a"),
            deployment_receipt(case, "r2", "verifier:b"),
        ]
        .into_iter()
        .map(|receipt| {
            let id = receipt.scoped_receipt.receipt.receipt_id.clone();
            bind_receipt_to_assurance_manifest(
                receipt,
                manifest,
                &sig,
                format!("policy-scope:{id}"),
            )
            .unwrap()
        })
        .collect()
    }

    fn profile(verifier: &str, suffix: &str) -> VerifierFaultDomainProfile {
        VerifierFaultDomainProfile {
            verifier_ref: verifier.into(),
            organization_domain: format!("org:{suffix}"),
            review_process_domain: format!("process:{suffix}"),
            toolchain_domain: format!("tool:{suffix}"),
            evidence_source_domain: format!("source:{suffix}"),
            evidence_refs: vec![format!("profile:{verifier}")],
        }
    }

    fn diversity_policy(case: &SafetyCase) -> VerifierDiversityPolicy {
        VerifierDiversityPolicy {
            schema_version: "1".into(),
            policy_id: "diversity-v1".into(),
            requirements: vec![VerifierDiversityRequirement {
                obligation_key: case.obligations[0].stable_key(),
                minimum_distinct_verifiers: 2,
                minimum_organization_domains: 2,
                minimum_review_process_domains: 2,
                minimum_toolchain_domains: 2,
                minimum_evidence_source_domains: 2,
                evidence_refs: vec!["review:diversity".into()],
            }],
            evidence_refs: vec!["policy:diversity".into()],
        }
    }

    fn atomic_policy(case: &SafetyCase) -> AtomicCoveragePolicy {
        AtomicCoveragePolicy {
            schema_version: "1".into(),
            policy_id: "atomic-v1".into(),
            facets: vec![AtomicEvidenceFacet {
                facet_id: "facet:a".into(),
                obligation_key: case.obligations[0].stable_key(),
                controlled_claim: "facet a".into(),
                minimum_distinct_receipts: 2,
                minimum_distinct_evidence_objects: 2,
                evidence_refs: vec!["review:facet-a".into()],
            }],
            evidence_refs: vec!["policy:atomic".into()],
        }
    }

    fn facet_bindings() -> Vec<FacetEvidenceBinding> {
        vec![
            FacetEvidenceBinding {
                binding_id: "b1".into(),
                receipt_id: "r1".into(),
                facet_id: "facet:a".into(),
                facet_evidence_ref: "result:r1".into(),
                facet_evidence_digest: "blake3:facet-r1".into(),
                rationale_ref: "rationale:b1".into(),
            },
            FacetEvidenceBinding {
                binding_id: "b2".into(),
                receipt_id: "r2".into(),
                facet_id: "facet:a".into(),
                facet_evidence_ref: "result:r2".into(),
                facet_evidence_digest: "blake3:facet-r2".into(),
                rationale_ref: "rationale:b2".into(),
            },
        ]
    }

    #[test]
    fn exact_current_manifest_can_retain_deep_readiness() {
        let case = case();
        let manifest = manifest(&case);
        let report = assess_policy_manifest_readiness(
            &case,
            &context(),
            &manifest,
            &signature(&manifest, 500),
            &policy_receipts(&case, &manifest),
            &[],
            &[],
            &[],
            &trusted_time(),
            &[profile("verifier:a", "a"), profile("verifier:b", "b")],
            &diversity_policy(&case),
            &atomic_policy(&case),
            &facet_bindings(),
        );
        assert_eq!(report.status, StrictSafetyCaseStatus::Ready);
        assert_eq!(report.matched_receipt_count, 2);
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn policy_revision_excludes_old_receipts_and_blocks_readiness() {
        let case = case();
        let old = manifest(&case);
        let old_receipts = policy_receipts(&case, &old);
        let mut current = old.clone();
        current.revision = 2;
        current.predecessor_manifest_digest = Some(old.manifest_digest());
        current.change_ref = Some("change:raise-diversity-policy".into());
        current.policies[5].content_digest = "blake3:verifier-diversity-v2".into();
        assert!(current.validate_complete());

        let report = assess_policy_manifest_readiness(
            &case,
            &context(),
            &current,
            &signature(&current, 500),
            &old_receipts,
            &[],
            &[],
            &[],
            &trusted_time(),
            &[profile("verifier:a", "a"), profile("verifier:b", "b")],
            &diversity_policy(&case),
            &atomic_policy(&case),
            &facet_bindings(),
        );
        assert_eq!(report.status, StrictSafetyCaseStatus::Blocked);
        assert_eq!(report.matched_receipt_count, 0);
        assert_eq!(report.excluded_receipt_count, 2);
        assert!(report.issues.iter().all(|issue| matches!(
            issue,
            PolicyManifestReadinessIssue::HistoricalManifestReceipt(_)
        )));
    }

    #[test]
    fn late_signature_verification_is_invalid() {
        let case = case();
        let manifest = manifest(&case);
        let report = assess_policy_manifest_readiness(
            &case,
            &context(),
            &manifest,
            &signature(&manifest, 1_000),
            &policy_receipts(&case, &manifest),
            &[],
            &[],
            &[],
            &trusted_time(),
            &[profile("verifier:a", "a"), profile("verifier:b", "b")],
            &diversity_policy(&case),
            &atomic_policy(&case),
            &facet_bindings(),
        );
        assert_eq!(report.status, StrictSafetyCaseStatus::Invalid);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            PolicyManifestReadinessIssue::SignatureVerifiedAfterTrustedInterval { .. }
        )));
    }
}
