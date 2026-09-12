// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Externally anchored signer-governed policy readiness.

#![deny(unsafe_code)]

use symthaea_assurance_policy_lineage::SignedAssurancePolicyRevision;
use symthaea_assurance_policy_lineage_anchor::{
    PolicyLineageAnchor, PolicyLineageAnchorReport, PolicyLineageAnchorStatus,
    assess_policy_lineage_anchor_chain,
};
use symthaea_assurance_policy_manifest::{
    AssurancePolicyManifest, ManifestSignatureVerificationReceipt, PolicyScopedSafetyReceipt,
};
use symthaea_assurance_signing_authority::SigningAuthorityGovernance;
use symthaea_evidence_atomic_coverage::{AtomicCoveragePolicy, FacetEvidenceBinding};
use symthaea_evidence_deployment_scope::DeploymentEvidenceContext;
use symthaea_evidence_governed_policy_readiness::{
    GovernedPolicyReadinessReport, assess_governed_policy_readiness,
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
pub enum AnchoredGovernedReadinessIssue {
    InvalidAnchorChain,
    InvalidExpectedAnchorDigest,
    AnchorDigestMismatch {
        expected: String,
        observed: String,
    },
    AnchorIdentityMismatch,
    SignedLineageBehindAnchor {
        signed_tip_revision: Option<u64>,
        anchored_tip_revision: u64,
    },
    SignedLineageAheadOfAnchor {
        signed_tip_revision: u64,
        anchored_tip_revision: u64,
    },
    SameRevisionManifestDigestMismatch {
        revision: u64,
        signed_tip_digest: Option<String>,
        anchored_tip_digest: String,
    },
    AnchorRecordedAfterTrustedInterval {
        recorded_at_ms: u64,
        earliest_trusted_ms: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnchoredGovernedReadinessReport {
    pub status: StrictSafetyCaseStatus,
    pub expected_anchor_digest: String,
    pub observed_anchor_digest: String,
    pub governed_readiness: GovernedPolicyReadinessReport,
    pub anchor_report: PolicyLineageAnchorReport,
    pub issues: Vec<AnchoredGovernedReadinessIssue>,
}

impl AnchoredGovernedReadinessReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

/// Recompute signer-governed policy readiness and require the signed policy tip
/// to agree exactly with an externally provisioned monotonic lineage anchor.
#[allow(clippy::too_many_arguments)]
pub fn assess_anchored_governed_readiness(
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
    signing_governance: &SigningAuthorityGovernance,
    expected_signing_governance_digest: &str,
    anchor_chain: &[PolicyLineageAnchor],
    expected_anchor_digest: &str,
) -> AnchoredGovernedReadinessReport {
    let governed_readiness = assess_governed_policy_readiness(
        safety_case,
        context,
        current_manifest,
        current_signature_receipt,
        signed_lineage,
        policy_receipts,
        lifecycle_events,
        quarantine_directives,
        quarantine_resolutions,
        trusted_time,
        verifier_profiles,
        verifier_policy,
        atomic_policy,
        facet_bindings,
        signing_governance,
        expected_signing_governance_digest,
    );
    let anchor_report = assess_policy_lineage_anchor_chain(anchor_chain);
    let expected_anchor_digest = expected_anchor_digest.trim().to_string();
    let observed_anchor_digest = anchor_report.tip_anchor_digest.clone().unwrap_or_default();
    let mut issues = Vec::new();

    if anchor_report.status != PolicyLineageAnchorStatus::Valid {
        issues.push(AnchoredGovernedReadinessIssue::InvalidAnchorChain);
    }
    if !valid_digest(&expected_anchor_digest) {
        issues.push(AnchoredGovernedReadinessIssue::InvalidExpectedAnchorDigest);
    } else if observed_anchor_digest != expected_anchor_digest {
        issues.push(AnchoredGovernedReadinessIssue::AnchorDigestMismatch {
            expected: expected_anchor_digest.clone(),
            observed: observed_anchor_digest.clone(),
        });
    }

    let lineage_report = &governed_readiness.lineage_readiness.lineage_report;
    let anchor_tip = anchor_chain
        .iter()
        .max_by_key(|anchor| anchor.anchor_revision);

    if let Some(anchor) = anchor_tip {
        if lineage_report.manifest_id.as_deref() != Some(anchor.manifest_id.as_str())
            || lineage_report.deployment_id.as_deref() != Some(anchor.deployment_id.as_str())
        {
            issues.push(AnchoredGovernedReadinessIssue::AnchorIdentityMismatch);
        }

        match lineage_report.tip_revision {
            Some(signed_tip) if signed_tip < anchor.tip_revision => {
                issues.push(AnchoredGovernedReadinessIssue::SignedLineageBehindAnchor {
                    signed_tip_revision: Some(signed_tip),
                    anchored_tip_revision: anchor.tip_revision,
                });
            }
            Some(signed_tip) if signed_tip > anchor.tip_revision => {
                issues.push(AnchoredGovernedReadinessIssue::SignedLineageAheadOfAnchor {
                    signed_tip_revision: signed_tip,
                    anchored_tip_revision: anchor.tip_revision,
                });
            }
            Some(signed_tip) => {
                if lineage_report.tip_manifest_digest.as_deref()
                    != Some(anchor.tip_manifest_digest.as_str())
                {
                    issues.push(
                        AnchoredGovernedReadinessIssue::SameRevisionManifestDigestMismatch {
                            revision: signed_tip,
                            signed_tip_digest: lineage_report.tip_manifest_digest.clone(),
                            anchored_tip_digest: anchor.tip_manifest_digest.clone(),
                        },
                    );
                }
            }
            None => {
                issues.push(AnchoredGovernedReadinessIssue::SignedLineageBehindAnchor {
                    signed_tip_revision: None,
                    anchored_tip_revision: anchor.tip_revision,
                });
            }
        }

        if anchor.recorded_at_ms > trusted_time.earliest_ms() {
            issues.push(
                AnchoredGovernedReadinessIssue::AnchorRecordedAfterTrustedInterval {
                    recorded_at_ms: anchor.recorded_at_ms,
                    earliest_trusted_ms: trusted_time.earliest_ms(),
                },
            );
        }
    }

    let invalid = issues.iter().any(|issue| {
        matches!(
            issue,
            AnchoredGovernedReadinessIssue::InvalidAnchorChain
                | AnchoredGovernedReadinessIssue::InvalidExpectedAnchorDigest
                | AnchoredGovernedReadinessIssue::AnchorDigestMismatch { .. }
                | AnchoredGovernedReadinessIssue::AnchorIdentityMismatch
                | AnchoredGovernedReadinessIssue::SameRevisionManifestDigestMismatch { .. }
        )
    });
    let blocked = issues.iter().any(|issue| {
        matches!(
            issue,
            AnchoredGovernedReadinessIssue::SignedLineageBehindAnchor { .. }
                | AnchoredGovernedReadinessIssue::SignedLineageAheadOfAnchor { .. }
                | AnchoredGovernedReadinessIssue::AnchorRecordedAfterTrustedInterval { .. }
        )
    });

    let status = if invalid {
        StrictSafetyCaseStatus::Invalid
    } else if blocked {
        StrictSafetyCaseStatus::Blocked
    } else {
        governed_readiness.status
    };

    AnchoredGovernedReadinessReport {
        status,
        expected_anchor_digest,
        observed_anchor_digest,
        governed_readiness,
        anchor_report,
        issues,
    }
}

fn valid_digest(value: &str) -> bool {
    value
        .split_once(':')
        .is_some_and(|(algorithm, digest)| !algorithm.is_empty() && !digest.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_assurance_policy_lineage::SignedAssurancePolicyRevision;
    use symthaea_assurance_policy_manifest::{
        AssurancePolicyBinding, AssurancePolicyKind, bind_receipt_to_assurance_manifest,
    };
    use symthaea_assurance_signing_authority::{
        ManifestSigningAuthorityPolicy, SigningAuthorityGovernance,
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
        let mut case = SafetyCase::new("anchored-readiness-fixture");
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
            change_ref: (revision > 1).then(|| format!("change:r{revision}")),
            policies: AssurancePolicyKind::REQUIRED
                .into_iter()
                .map(|kind| binding(kind, revision))
                .collect(),
            evidence_refs: vec![format!("review:manifest-r{revision}")],
        }
    }

    fn signature(manifest: &AssurancePolicyManifest) -> ManifestSignatureVerificationReceipt {
        ManifestSignatureVerificationReceipt {
            receipt_id: format!("sig:r{}", manifest.revision),
            manifest_digest: manifest.manifest_digest(),
            signer_ref: "signer:a".into(),
            key_ref: "key:a1".into(),
            signature_algorithm: "ed25519".into(),
            signature_ref: format!("signature:r{}", manifest.revision),
            verified_by_ref: "verifier:independent".into(),
            verification_ref: format!("verification:r{}", manifest.revision),
            verified_at_ms: 500 + manifest.revision,
            evidence_refs: vec![format!("audit:r{}", manifest.revision)],
        }
    }

    fn lineage(case: &SafetyCase, revisions: u64) -> Vec<SignedAssurancePolicyRevision> {
        let mut result = Vec::new();
        let mut predecessor = None;
        for revision in 1..=revisions {
            let current = manifest(case, revision, predecessor.clone());
            predecessor = Some(current.manifest_digest());
            result.push(SignedAssurancePolicyRevision {
                signature_receipt: signature(&current),
                manifest: current,
            });
        }
        result
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

    fn signing_governance() -> SigningAuthorityGovernance {
        SigningAuthorityGovernance {
            policy: ManifestSigningAuthorityPolicy {
                schema_version: "1".into(),
                policy_id: "signing-authority-v1".into(),
                manifest_id: "assurance:node-1".into(),
                deployment_id: "node-1".into(),
                initial_signer_ref: "signer:a".into(),
                initial_key_ref: "key:a1".into(),
                allowed_signature_algorithms: vec!["ed25519".into()],
                trust_root_ref: "trust-root:safety-board".into(),
                evidence_refs: vec!["review:signing-authority".into()],
            },
            transitions: vec![],
        }
    }

    fn anchor_for(
        lineage: &[SignedAssurancePolicyRevision],
        recorded_at_ms: u64,
    ) -> PolicyLineageAnchor {
        let tip = lineage.last().unwrap();
        PolicyLineageAnchor {
            schema_version: "1".into(),
            anchor_id: "anchor:node-1".into(),
            anchor_revision: 1,
            manifest_id: tip.manifest.manifest_id.clone(),
            deployment_id: tip.manifest.deployment_id.clone(),
            tip_revision: tip.manifest.revision,
            tip_manifest_digest: tip.manifest.manifest_digest(),
            predecessor_anchor_digest: None,
            recorded_at_ms,
            trust_store_ref: "trust-store:tpm-nv:index-1".into(),
            independent_verification_ref: "verification:anchor:1".into(),
            evidence_refs: vec!["audit:anchor:1".into()],
        }
    }

    fn assess(
        case: &SafetyCase,
        lineage: &[SignedAssurancePolicyRevision],
        anchor_chain: &[PolicyLineageAnchor],
        expected_anchor_digest: &str,
    ) -> AnchoredGovernedReadinessReport {
        let current = lineage.last().unwrap();
        let receipts = policy_receipts(case, &current.manifest, &current.signature_receipt);
        let signing = signing_governance();
        let signing_digest = signing.governance_digest();
        assess_anchored_governed_readiness(
            case,
            &context(),
            &current.manifest,
            &current.signature_receipt,
            lineage,
            &receipts,
            &[],
            &[],
            &[],
            &trusted_time(),
            &[profile("verifier:a", "a"), profile("verifier:b", "b")],
            &diversity_policy(case),
            &atomic_policy(case),
            &facet_bindings(),
            &signing,
            &signing_digest,
            anchor_chain,
            expected_anchor_digest,
        )
    }

    #[test]
    fn exact_external_anchor_can_retain_readiness() {
        let case = case();
        let lineage = lineage(&case, 2);
        let anchor = anchor_for(&lineage, 900);
        let digest = anchor.anchor_digest();
        let report = assess(&case, &lineage, &[anchor], &digest);
        assert_eq!(report.status, StrictSafetyCaseStatus::Ready);
        assert!(report.issues.is_empty());
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn truncated_signed_lineage_behind_anchor_is_blocked() {
        let case = case();
        let full_lineage = lineage(&case, 2);
        let anchor = anchor_for(&full_lineage, 900);
        let digest = anchor.anchor_digest();
        let truncated = vec![full_lineage[0].clone()];
        let report = assess(&case, &truncated, &[anchor], &digest);
        assert_eq!(report.governed_readiness.status, StrictSafetyCaseStatus::Ready);
        assert_eq!(report.status, StrictSafetyCaseStatus::Blocked);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            AnchoredGovernedReadinessIssue::SignedLineageBehindAnchor {
                signed_tip_revision: Some(1),
                anchored_tip_revision: 2
            }
        )));
    }

    #[test]
    fn new_policy_revision_ahead_of_anchor_is_blocked_until_checkpointed() {
        let case = case();
        let old_lineage = lineage(&case, 2);
        let old_anchor = anchor_for(&old_lineage, 900);
        let digest = old_anchor.anchor_digest();
        let new_lineage = lineage(&case, 3);
        let report = assess(&case, &new_lineage, &[old_anchor], &digest);
        assert_eq!(report.governed_readiness.status, StrictSafetyCaseStatus::Ready);
        assert_eq!(report.status, StrictSafetyCaseStatus::Blocked);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            AnchoredGovernedReadinessIssue::SignedLineageAheadOfAnchor {
                signed_tip_revision: 3,
                anchored_tip_revision: 2
            }
        )));
    }

    #[test]
    fn older_anchor_substitution_fails_provisioned_anchor_digest() {
        let case = case();
        let lineage = lineage(&case, 2);
        let current_anchor = anchor_for(&lineage, 900);
        let expected = current_anchor.anchor_digest();
        let old_lineage = vec![lineage[0].clone()];
        let old_anchor = anchor_for(&old_lineage, 800);
        let report = assess(&case, &old_lineage, &[old_anchor], &expected);
        assert_eq!(report.status, StrictSafetyCaseStatus::Invalid);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            AnchoredGovernedReadinessIssue::AnchorDigestMismatch { .. }
        )));
    }

    #[test]
    fn anchor_recorded_after_trusted_interval_cannot_retroactively_enable_readiness() {
        let case = case();
        let lineage = lineage(&case, 2);
        let anchor = anchor_for(&lineage, 1_100);
        let digest = anchor.anchor_digest();
        let report = assess(&case, &lineage, &[anchor], &digest);
        assert_eq!(report.status, StrictSafetyCaseStatus::Blocked);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            AnchoredGovernedReadinessIssue::AnchorRecordedAfterTrustedInterval { .. }
        )));
    }
}
