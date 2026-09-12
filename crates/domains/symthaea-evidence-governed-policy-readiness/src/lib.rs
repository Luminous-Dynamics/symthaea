// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Top-level policy readiness gated by an independently provisioned signer authority.

#![deny(unsafe_code)]

use symthaea_assurance_policy_lineage::SignedAssurancePolicyRevision;
use symthaea_assurance_policy_manifest::{
    AssurancePolicyManifest, ManifestSignatureVerificationReceipt, PolicyScopedSafetyReceipt,
};
use symthaea_assurance_signing_authority::{
    SigningAuthorityGovernance, SigningAuthorityReport, SigningAuthorityStatus,
    assess_signing_authority,
};
use symthaea_evidence_atomic_coverage::{AtomicCoveragePolicy, FacetEvidenceBinding};
use symthaea_evidence_deployment_scope::DeploymentEvidenceContext;
use symthaea_evidence_lifecycle::EvidenceLifecycleEvent;
use symthaea_evidence_lineage_readiness::{
    PolicyLineageReadinessReport, assess_policy_lineage_readiness,
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
pub enum GovernedPolicyReadinessIssue {
    InvalidExpectedGovernanceDigest,
    SigningGovernanceDigestMismatch {
        expected: String,
        observed: String,
    },
    SigningAuthorityInvalid,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GovernedPolicyReadinessReport {
    pub status: StrictSafetyCaseStatus,
    pub expected_signing_governance_digest: String,
    pub observed_signing_governance_digest: String,
    pub lineage_readiness: PolicyLineageReadinessReport,
    pub signing_authority: SigningAuthorityReport,
    pub issues: Vec<GovernedPolicyReadinessIssue>,
}

impl GovernedPolicyReadinessReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

/// Recompute lineage-gated deep readiness and signer/key authorization, then
/// require the signer-governance rules themselves to match an externally
/// provisioned trust-root digest.
///
/// A caller cannot preserve readiness by supplying a different, weaker signing
/// authority policy: the governance digest must equal the provisioned digest.
#[allow(clippy::too_many_arguments)]
pub fn assess_governed_policy_readiness(
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
) -> GovernedPolicyReadinessReport {
    let lineage_readiness = assess_policy_lineage_readiness(
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
    );
    let signing_authority = assess_signing_authority(signed_lineage, signing_governance);
    let observed = signing_governance.governance_digest();
    let expected = expected_signing_governance_digest.trim().to_string();
    let mut issues = Vec::new();

    if !valid_digest(&expected) {
        issues.push(GovernedPolicyReadinessIssue::InvalidExpectedGovernanceDigest);
    } else if expected != observed {
        issues.push(GovernedPolicyReadinessIssue::SigningGovernanceDigestMismatch {
            expected: expected.clone(),
            observed: observed.clone(),
        });
    }

    if signing_authority.status != SigningAuthorityStatus::Valid {
        issues.push(GovernedPolicyReadinessIssue::SigningAuthorityInvalid);
    }

    let status = if !issues.is_empty() {
        StrictSafetyCaseStatus::Invalid
    } else {
        lineage_readiness.status
    };

    GovernedPolicyReadinessReport {
        status,
        expected_signing_governance_digest: expected,
        observed_signing_governance_digest: observed,
        lineage_readiness,
        signing_authority,
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
    use symthaea_assurance_policy_manifest::{
        AssurancePolicyBinding, AssurancePolicyKind, bind_receipt_to_assurance_manifest,
    };
    use symthaea_assurance_signing_authority::{
        ManifestSigningAuthorityPolicy, SigningAuthorityTransition,
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
        let mut case = SafetyCase::new("governed-readiness-fixture");
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

    fn signature(
        manifest: &AssurancePolicyManifest,
        signer: &str,
        key: &str,
    ) -> ManifestSignatureVerificationReceipt {
        ManifestSignatureVerificationReceipt {
            receipt_id: format!("sig:r{}:{signer}:{key}", manifest.revision),
            manifest_digest: manifest.manifest_digest(),
            signer_ref: signer.into(),
            key_ref: key.into(),
            signature_algorithm: "ed25519".into(),
            signature_ref: format!("signature:r{}", manifest.revision),
            verified_by_ref: "verifier:independent".into(),
            verification_ref: format!("verification:r{}", manifest.revision),
            verified_at_ms: 500 + manifest.revision,
            evidence_refs: vec![format!("audit:r{}", manifest.revision)],
        }
    }

    fn lineage(
        case: &SafetyCase,
        second_signer: &str,
        second_key: &str,
    ) -> Vec<SignedAssurancePolicyRevision> {
        let r1 = manifest(case, 1, None);
        let r2 = manifest(case, 2, Some(r1.manifest_digest()));
        vec![
            SignedAssurancePolicyRevision {
                signature_receipt: signature(&r1, "signer:a", "key:a1"),
                manifest: r1,
            },
            SignedAssurancePolicyRevision {
                signature_receipt: signature(&r2, second_signer, second_key),
                manifest: r2,
            },
        ]
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

    fn governance(transitions: Vec<SigningAuthorityTransition>) -> SigningAuthorityGovernance {
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
            transitions,
        }
    }

    fn transition_to_b() -> SigningAuthorityTransition {
        SigningAuthorityTransition {
            transition_id: "rotation:r2".into(),
            effective_revision: 2,
            from_signer_ref: "signer:a".into(),
            from_key_ref: "key:a1".into(),
            to_signer_ref: "signer:b".into(),
            to_key_ref: "key:b1".into(),
            authorization_ref: "authorization:r2".into(),
            independent_verification_ref: "verification:rotation:r2".into(),
            evidence_refs: vec!["audit:rotation:r2".into()],
        }
    }

    fn assess(
        case: &SafetyCase,
        lineage: &[SignedAssurancePolicyRevision],
        governance: &SigningAuthorityGovernance,
        expected_digest: &str,
    ) -> GovernedPolicyReadinessReport {
        let current = lineage.last().unwrap();
        let receipts = policy_receipts(case, &current.manifest, &current.signature_receipt);
        assess_governed_policy_readiness(
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
            governance,
            expected_digest,
        )
    }

    #[test]
    fn authorized_signer_lineage_can_retain_readiness() {
        let case = case();
        let lineage = lineage(&case, "signer:a", "key:a1");
        let governance = governance(vec![]);
        let digest = governance.governance_digest();
        let report = assess(&case, &lineage, &governance, &digest);
        assert_eq!(report.status, StrictSafetyCaseStatus::Ready);
        assert!(report.issues.is_empty());
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn unreviewed_signer_replacement_invalidates_readiness() {
        let case = case();
        let lineage = lineage(&case, "signer:b", "key:b1");
        let governance = governance(vec![]);
        let digest = governance.governance_digest();
        let report = assess(&case, &lineage, &governance, &digest);
        assert_eq!(report.lineage_readiness.status, StrictSafetyCaseStatus::Ready);
        assert_eq!(report.status, StrictSafetyCaseStatus::Invalid);
        assert!(report.issues.contains(&GovernedPolicyReadinessIssue::SigningAuthorityInvalid));
    }

    #[test]
    fn explicitly_reviewed_signer_rotation_can_retain_readiness() {
        let case = case();
        let lineage = lineage(&case, "signer:b", "key:b1");
        let governance = governance(vec![transition_to_b()]);
        let digest = governance.governance_digest();
        let report = assess(&case, &lineage, &governance, &digest);
        assert_eq!(report.status, StrictSafetyCaseStatus::Ready);
        assert_eq!(report.signing_authority.status, SigningAuthorityStatus::Valid);
    }

    #[test]
    fn weaker_substituted_governance_cannot_replace_provisioned_trust_root() {
        let case = case();
        let lineage = lineage(&case, "signer:b", "key:b1");
        let provisioned = governance(vec![]);
        let substituted = SigningAuthorityGovernance {
            policy: ManifestSigningAuthorityPolicy {
                initial_signer_ref: "signer:b".into(),
                initial_key_ref: "key:b1".into(),
                ..provisioned.policy.clone()
            },
            transitions: vec![],
        };
        let report = assess(
            &case,
            &lineage,
            &substituted,
            &provisioned.governance_digest(),
        );
        assert_eq!(report.signing_authority.status, SigningAuthorityStatus::Valid);
        assert_eq!(report.status, StrictSafetyCaseStatus::Invalid);
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            GovernedPolicyReadinessIssue::SigningGovernanceDigestMismatch { .. }
        )));
    }

    #[test]
    fn malformed_provisioned_digest_is_invalid() {
        let case = case();
        let lineage = lineage(&case, "signer:a", "key:a1");
        let governance = governance(vec![]);
        let report = assess(&case, &lineage, &governance, "not-a-digest");
        assert_eq!(report.status, StrictSafetyCaseStatus::Invalid);
        assert!(report
            .issues
            .contains(&GovernedPolicyReadinessIssue::InvalidExpectedGovernanceDigest));
    }
}
