// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Signed-policy-lineage gate over policy-scoped deep evidence readiness.

#![deny(unsafe_code)]

use symthaea_assurance_policy_lineage::{
    AssurancePolicyLineageReport, AssurancePolicyLineageStatus,
    SignedAssurancePolicyRevision, assess_signed_policy_lineage,
};
use symthaea_assurance_policy_manifest::{
    AssurancePolicyManifest, ManifestSignatureVerificationReceipt, PolicyScopedSafetyReceipt,
};
use symthaea_evidence_atomic_coverage::{AtomicCoveragePolicy, FacetEvidenceBinding};
use symthaea_evidence_deployment_scope::DeploymentEvidenceContext;
use symthaea_evidence_lifecycle::EvidenceLifecycleEvent;
use symthaea_evidence_policy_readiness::{
    PolicyManifestReadinessReport, assess_policy_manifest_readiness,
};
use symthaea_evidence_quarantine::{
    EvidenceQuarantineDirective, EvidenceQuarantineResolution,
};
use symthaea_evidence_time_assurance::TrustedEvidenceTime;
use symthaea_evidence_verifier_diversity::{
    VerifierDiversityPolicy, VerifierFaultDomainProfile,
};
use symthaea_formal_safety::{SafetyCase, StrictSafetyCaseStatus};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyLineageReadinessIssue {
    InvalidLineage,
    CurrentManifestAbsentFromLineage {
        manifest_digest: String,
    },
    CurrentManifestIsNotLineageTip {
        current_revision: u64,
        tip_revision: Option<u64>,
    },
    CurrentSignatureVerificationDoesNotMatchLineageTip {
        supplied_receipt_id: String,
        lineage_receipt_id: Option<String>,
    },
    PolicyReceiptSignatureVerificationMismatch {
        receipt_id: String,
        expected_signature_receipt_id: String,
        observed_signature_receipt_id: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyLineageReadinessReport {
    pub status: StrictSafetyCaseStatus,
    pub current_manifest_digest: String,
    pub current_manifest_revision: u64,
    pub lineage_report: AssurancePolicyLineageReport,
    pub policy_report: PolicyManifestReadinessReport,
    pub issues: Vec<PolicyLineageReadinessIssue>,
}

impl PolicyLineageReadinessReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

/// Recompute policy-scoped deep readiness and require the current manifest to be
/// the exact signed tip of a valid assurance-policy lineage.
///
/// A valid but older signed manifest is a rollback/stale-policy condition and
/// therefore blocks readiness. Invalid lineages, manifests absent from the
/// lineage, signature-verification-record splicing, or current-manifest receipts
/// bound to a different signature-verification receipt fail closed as invalid.
#[allow(clippy::too_many_arguments)]
pub fn assess_policy_lineage_readiness(
    safety_case: &SafetyCase,
    context: &DeploymentEvidenceContext,
    current_manifest: &AssurancePolicyManifest,
    current_signature_receipt: &ManifestSignatureVerificationReceipt,
    signed_lineage: &[SignedAssurancePolicyRevision],
    policy_receipts: &[PolicyScopedSafetyReceipt],
    lifecycle_events: &[EvidenceLifecycleEvent],
    quarantine_directives: &[EvidenceQuarantineDirective],
    quarantine_resolutions: &[EvidenceQuarantineResolution],
    trusted_time: &TrustedEvidenceTime,
    verifier_profiles: &[VerifierFaultDomainProfile],
    verifier_policy: &VerifierDiversityPolicy,
    atomic_policy: &AtomicCoveragePolicy,
    facet_bindings: &[FacetEvidenceBinding],
) -> PolicyLineageReadinessReport {
    let current_manifest_digest = current_manifest.manifest_digest();
    let lineage_report = assess_signed_policy_lineage(signed_lineage);
    let policy_report = assess_policy_manifest_readiness(
        safety_case,
        context,
        current_manifest,
        current_signature_receipt,
        policy_receipts,
        lifecycle_events,
        quarantine_directives,
        quarantine_resolutions,
        trusted_time,
        verifier_profiles,
        verifier_policy,
        atomic_policy,
        facet_bindings,
    );

    let mut issues = Vec::new();
    if lineage_report.status != AssurancePolicyLineageStatus::Valid {
        issues.push(PolicyLineageReadinessIssue::InvalidLineage);
    }

    let current_revision = signed_lineage.iter().find(|revision| {
        revision.manifest.manifest_digest() == current_manifest_digest
    });
    if current_revision.is_none() {
        issues.push(PolicyLineageReadinessIssue::CurrentManifestAbsentFromLineage {
            manifest_digest: current_manifest_digest.clone(),
        });
    }

    let is_tip = lineage_report.contains_tip(current_manifest);
    if lineage_report.status == AssurancePolicyLineageStatus::Valid
        && current_revision.is_some()
        && !is_tip
    {
        issues.push(PolicyLineageReadinessIssue::CurrentManifestIsNotLineageTip {
            current_revision: current_manifest.revision,
            tip_revision: lineage_report.tip_revision,
        });
    }

    let lineage_signature = current_revision.map(|revision| &revision.signature_receipt);
    if lineage_signature != Some(current_signature_receipt) {
        issues.push(
            PolicyLineageReadinessIssue::CurrentSignatureVerificationDoesNotMatchLineageTip {
                supplied_receipt_id: current_signature_receipt.receipt_id.clone(),
                lineage_receipt_id: lineage_signature.map(|receipt| receipt.receipt_id.clone()),
            },
        );
    }

    for receipt in policy_receipts {
        if receipt.assurance_manifest_digest != current_manifest_digest {
            continue;
        }
        if receipt.manifest_signature_receipt_id != current_signature_receipt.receipt_id {
            issues.push(
                PolicyLineageReadinessIssue::PolicyReceiptSignatureVerificationMismatch {
                    receipt_id: receipt
                        .deployment_receipt
                        .scoped_receipt
                        .receipt
                        .receipt_id
                        .clone(),
                    expected_signature_receipt_id: current_signature_receipt.receipt_id.clone(),
                    observed_signature_receipt_id: receipt.manifest_signature_receipt_id.clone(),
                },
            );
        }
    }

    let invalid = issues.iter().any(|issue| {
        !matches!(
            issue,
            PolicyLineageReadinessIssue::CurrentManifestIsNotLineageTip { .. }
        )
    });
    let stale_or_rollback = issues.iter().any(|issue| {
        matches!(
            issue,
            PolicyLineageReadinessIssue::CurrentManifestIsNotLineageTip { .. }
        )
    });

    let status = if invalid {
        StrictSafetyCaseStatus::Invalid
    } else if stale_or_rollback {
        StrictSafetyCaseStatus::Blocked
    } else {
        policy_report.status
    };

    PolicyLineageReadinessReport {
        status,
        current_manifest_digest,
        current_manifest_revision: current_manifest.revision,
        lineage_report,
        policy_report,
        issues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_assurance_policy_manifest::{
        AssurancePolicyBinding, AssurancePolicyKind, bind_receipt_to_assurance_manifest,
    };
    use symthaea_evidence_atomic_coverage::AtomicEvidenceFacet;
    use symthaea_evidence_deployment_scope::{
        DeploymentScopedSafetyReceipt, bind_receipt_to_deployment,
    };
    use symthaea_evidence_lifecycle::ScopedSafetyEvidenceReceipt;
    use symthaea_evidence_time_assurance::{
        EvidenceTimeGuard, EvidenceTimePolicy, EvidenceTimeSample,
    };
    use symthaea_evidence_verifier_diversity::VerifierDiversityRequirement;
    use symthaea_formal_safety::{EvidenceKind, ProofObligation, SafetyEvidenceReceipt};

    fn case() -> SafetyCase {
        let mut case = SafetyCase::new("lineage-readiness-fixture");
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

    fn binding(kind: AssurancePolicyKind, revision: u64) -> AssurancePolicyBinding {
        AssurancePolicyBinding {
            kind,
            policy_id: format!("policy:{}", kind.code()),
            schema_version: "1".into(),
            content_digest: format!("blake3:{}-r{revision}", kind.code()),
            evidence_refs: vec![format!("review:{}:r{revision}", kind.code())],
        }
    }

    fn manifest(
        case: &SafetyCase,
        revision: u64,
        predecessor: Option<String>,
    ) -> AssurancePolicyManifest {
        AssurancePolicyManifest {
            schema_version: "1".into(),
            manifest_id: "assurance:node-1".into(),
            revision,
            safety_contract_digest: case.contract_digest(),
            deployment_id: "node-1".into(),
            configuration_digest: "blake3:config-v1".into(),
            model_manifest_digest: Some("blake3:model-v1".into()),
            calibration_manifest_digest: Some("blake3:cal-v1".into()),
            predecessor_manifest_digest: predecessor,
            change_ref: (revision > 1).then(|| format!("change:review-r{revision}")),
            policies: AssurancePolicyKind::REQUIRED
                .into_iter()
                .map(|kind| binding(kind, revision))
                .collect(),
            evidence_refs: vec![format!("review:manifest-r{revision}")],
        }
    }

    fn signature(
        manifest: &AssurancePolicyManifest,
        receipt_id: impl Into<String>,
    ) -> ManifestSignatureVerificationReceipt {
        ManifestSignatureVerificationReceipt {
            receipt_id: receipt_id.into(),
            manifest_digest: manifest.manifest_digest(),
            signer_ref: "signer:safety-board".into(),
            key_ref: "key:safety-board:v1".into(),
            signature_algorithm: "ed25519".into(),
            signature_ref: format!("signature:r{}", manifest.revision),
            verified_by_ref: "verifier:signature".into(),
            verification_ref: format!("verification:r{}", manifest.revision),
            verified_at_ms: 500 + manifest.revision,
            evidence_refs: vec![format!("audit:signature-r{}", manifest.revision)],
        }
    }

    fn signed(
        manifest: AssurancePolicyManifest,
        signature: ManifestSignatureVerificationReceipt,
    ) -> SignedAssurancePolicyRevision {
        SignedAssurancePolicyRevision {
            manifest,
            signature_receipt: signature,
        }
    }

    fn lineage(case: &SafetyCase) -> Vec<SignedAssurancePolicyRevision> {
        let r1 = manifest(case, 1, None);
        let s1 = signature(&r1, "sig:r1");
        let r2 = manifest(case, 2, Some(r1.manifest_digest()));
        let s2 = signature(&r2, "sig:r2");
        vec![signed(r1, s1), signed(r2, s2)]
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
        signature_receipt: &ManifestSignatureVerificationReceipt,
    ) -> Vec<PolicyScopedSafetyReceipt> {
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
                signature_receipt,
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

    fn assess(
        case: &SafetyCase,
        manifest: &AssurancePolicyManifest,
        signature_receipt: &ManifestSignatureVerificationReceipt,
        lineage: &[SignedAssurancePolicyRevision],
        receipts: &[PolicyScopedSafetyReceipt],
    ) -> PolicyLineageReadinessReport {
        assess_policy_lineage_readiness(
            case,
            &context(),
            manifest,
            signature_receipt,
            lineage,
            receipts,
            &[],
            &[],
            &[],
            &trusted_time(),
            &[profile("verifier:a", "a"), profile("verifier:b", "b")],
            &diversity_policy(case),
            &atomic_policy(case),
            &facet_bindings(),
        )
    }

    #[test]
    fn exact_signed_lineage_tip_can_retain_readiness() {
        let case = case();
        let lineage = lineage(&case);
        let current = &lineage[1];
        let receipts = policy_receipts(&case, &current.manifest, &current.signature_receipt);
        let report = assess(
            &case,
            &current.manifest,
            &current.signature_receipt,
            &lineage,
            &receipts,
        );
        assert_eq!(report.status, StrictSafetyCaseStatus::Ready);
        assert!(report.issues.is_empty());
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn rollback_to_older_signed_manifest_is_blocked_even_if_lower_readiness_is_ready() {
        let case = case();
        let lineage = lineage(&case);
        let old = &lineage[0];
        let receipts = policy_receipts(&case, &old.manifest, &old.signature_receipt);
        let report = assess(
            &case,
            &old.manifest,
            &old.signature_receipt,
            &lineage,
            &receipts,
        );
        assert_eq!(report.policy_report.status, StrictSafetyCaseStatus::Ready);
        assert_eq!(report.status, StrictSafetyCaseStatus::Blocked);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            PolicyLineageReadinessIssue::CurrentManifestIsNotLineageTip {
                current_revision: 1,
                tip_revision: Some(2)
            }
        )));
    }

    #[test]
    fn manifest_absent_from_lineage_is_invalid() {
        let case = case();
        let lineage = lineage(&case);
        let r1 = &lineage[0].manifest;
        let mut alternate = manifest(&case, 2, Some(r1.manifest_digest()));
        alternate.policies[0].content_digest = "blake3:alternate-policy".into();
        let alternate_signature = signature(&alternate, "sig:alternate");
        let receipts = policy_receipts(&case, &alternate, &alternate_signature);
        let report = assess(
            &case,
            &alternate,
            &alternate_signature,
            &lineage,
            &receipts,
        );
        assert_eq!(report.policy_report.status, StrictSafetyCaseStatus::Ready);
        assert_eq!(report.status, StrictSafetyCaseStatus::Invalid);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            PolicyLineageReadinessIssue::CurrentManifestAbsentFromLineage { .. }
        )));
    }

    #[test]
    fn alternate_valid_signature_record_cannot_be_spliced_into_lineage_tip() {
        let case = case();
        let lineage = lineage(&case);
        let current = &lineage[1];
        let alternate_signature = signature(&current.manifest, "sig:alternate");
        let receipts = policy_receipts(&case, &current.manifest, &alternate_signature);
        let report = assess(
            &case,
            &current.manifest,
            &alternate_signature,
            &lineage,
            &receipts,
        );
        assert_eq!(report.policy_report.status, StrictSafetyCaseStatus::Ready);
        assert_eq!(report.status, StrictSafetyCaseStatus::Invalid);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            PolicyLineageReadinessIssue::CurrentSignatureVerificationDoesNotMatchLineageTip { .. }
        )));
    }

    #[test]
    fn current_policy_receipt_must_name_tip_signature_record() {
        let case = case();
        let lineage = lineage(&case);
        let current = &lineage[1];
        let mut receipts = policy_receipts(&case, &current.manifest, &current.signature_receipt);
        receipts[0].manifest_signature_receipt_id = "sig:other".into();
        let report = assess(
            &case,
            &current.manifest,
            &current.signature_receipt,
            &lineage,
            &receipts,
        );
        assert_eq!(report.status, StrictSafetyCaseStatus::Invalid);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            PolicyLineageReadinessIssue::PolicyReceiptSignatureVerificationMismatch { receipt_id, .. }
                if receipt_id == "r1"
        )));
    }

    #[test]
    fn invalid_signed_lineage_invalidates_readiness() {
        let case = case();
        let mut lineage = lineage(&case);
        lineage[1].manifest.predecessor_manifest_digest = Some("blake3:forged".into());
        lineage[1].signature_receipt = signature(&lineage[1].manifest, "sig:r2-forged");
        let current = lineage[1].clone();
        let receipts = policy_receipts(&case, &current.manifest, &current.signature_receipt);
        let report = assess(
            &case,
            &current.manifest,
            &current.signature_receipt,
            &lineage,
            &receipts,
        );
        assert_eq!(report.policy_report.status, StrictSafetyCaseStatus::Ready);
        assert_eq!(report.status, StrictSafetyCaseStatus::Invalid);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            PolicyLineageReadinessIssue::InvalidLineage
        )));
    }
}
