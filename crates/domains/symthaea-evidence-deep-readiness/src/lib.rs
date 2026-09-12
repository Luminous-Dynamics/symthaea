// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Composite deep-readiness assurance over the currently active evidence set.

#![deny(unsafe_code)]

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use symthaea_evidence_atomic_coverage::{
    AtomicCoveragePolicy, AtomicCoverageReport, FacetEvidenceBinding, assess_atomic_coverage,
};
use symthaea_evidence_deployment_scope::{
    DeploymentEvidenceContext, DeploymentScopedSafetyReceipt,
};
use symthaea_evidence_lifecycle::EvidenceLifecycleEvent;
use symthaea_evidence_quarantine::{
    EvidenceQuarantineDirective, EvidenceQuarantineReport, EvidenceQuarantineResolution,
};
use symthaea_evidence_time_assurance::{
    TimeRobustSafetyReport, TrustedEvidenceTime, assess_with_trusted_time,
};
use symthaea_evidence_verifier_diversity::{
    VerifierDiversityPolicy, VerifierDiversityReport, VerifierFaultDomainProfile,
    assess_verifier_diversity,
};
use symthaea_formal_safety::{SafetyCase, SafetyEvidenceReceipt, StrictSafetyCaseStatus};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeepEvidenceReadinessReport {
    pub status: StrictSafetyCaseStatus,
    pub earliest_active_receipt_count: usize,
    pub latest_active_receipt_count: usize,
    pub time_report: TimeRobustSafetyReport,
    pub earliest_verifier_diversity: VerifierDiversityReport,
    pub latest_verifier_diversity: VerifierDiversityReport,
    pub earliest_atomic_coverage: AtomicCoverageReport,
    pub latest_atomic_coverage: AtomicCoverageReport,
}

impl DeepEvidenceReadinessReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

/// Compose trusted-time/lifecycle readiness with verifier diversity and atomic
/// evidence coverage over the receipts that are actually active at each endpoint.
///
/// Historical, mismatched, quarantined, and lifecycle-excluded receipts are not
/// passed into the deeper gates.
pub fn assess_deep_evidence_readiness(
    safety_case: &SafetyCase,
    context: &DeploymentEvidenceContext,
    receipts: &[DeploymentScopedSafetyReceipt],
    lifecycle_events: &[EvidenceLifecycleEvent],
    quarantine_directives: &[EvidenceQuarantineDirective],
    quarantine_resolutions: &[EvidenceQuarantineResolution],
    trusted_time: &TrustedEvidenceTime,
    verifier_profiles: &[VerifierFaultDomainProfile],
    verifier_policy: &VerifierDiversityPolicy,
    atomic_policy: &AtomicCoveragePolicy,
    facet_bindings: &[FacetEvidenceBinding],
) -> DeepEvidenceReadinessReport {
    let time_report = assess_with_trusted_time(
        safety_case,
        context,
        receipts,
        lifecycle_events,
        quarantine_directives,
        quarantine_resolutions,
        trusted_time,
    );

    let earliest_receipts = active_receipts_for_endpoint(
        context,
        receipts,
        &time_report.earliest_assessment,
    );
    let latest_receipts = active_receipts_for_endpoint(
        context,
        receipts,
        &time_report.latest_assessment,
    );

    let earliest_ids = earliest_receipts
        .iter()
        .map(|receipt| receipt.receipt_id.as_str())
        .collect::<BTreeSet<_>>();
    let latest_ids = latest_receipts
        .iter()
        .map(|receipt| receipt.receipt_id.as_str())
        .collect::<BTreeSet<_>>();

    let earliest_bindings = facet_bindings
        .iter()
        .filter(|binding| earliest_ids.contains(binding.receipt_id.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let latest_bindings = facet_bindings
        .iter()
        .filter(|binding| latest_ids.contains(binding.receipt_id.as_str()))
        .cloned()
        .collect::<Vec<_>>();

    let earliest_verifier_diversity = assess_verifier_diversity(
        safety_case,
        &earliest_receipts,
        verifier_profiles,
        verifier_policy,
    );
    let latest_verifier_diversity = assess_verifier_diversity(
        safety_case,
        &latest_receipts,
        verifier_profiles,
        verifier_policy,
    );
    let earliest_atomic_coverage = assess_atomic_coverage(
        safety_case,
        &earliest_receipts,
        atomic_policy,
        &earliest_bindings,
    );
    let latest_atomic_coverage = assess_atomic_coverage(
        safety_case,
        &latest_receipts,
        atomic_policy,
        &latest_bindings,
    );

    let statuses = [
        time_report.status,
        earliest_verifier_diversity.status,
        latest_verifier_diversity.status,
        earliest_atomic_coverage.status,
        latest_atomic_coverage.status,
    ];
    let status = if statuses
        .iter()
        .any(|status| *status == StrictSafetyCaseStatus::Invalid)
    {
        StrictSafetyCaseStatus::Invalid
    } else if statuses
        .iter()
        .all(|status| *status == StrictSafetyCaseStatus::Ready)
    {
        StrictSafetyCaseStatus::Ready
    } else {
        StrictSafetyCaseStatus::Blocked
    };

    DeepEvidenceReadinessReport {
        status,
        earliest_active_receipt_count: earliest_receipts.len(),
        latest_active_receipt_count: latest_receipts.len(),
        time_report,
        earliest_verifier_diversity,
        latest_verifier_diversity,
        earliest_atomic_coverage,
        latest_atomic_coverage,
    }
}

fn active_receipts_for_endpoint(
    context: &DeploymentEvidenceContext,
    receipts: &[DeploymentScopedSafetyReceipt],
    report: &EvidenceQuarantineReport,
) -> Vec<SafetyEvidenceReceipt> {
    let quarantined = report
        .quarantined_receipts
        .iter()
        .map(|item| item.receipt_id.as_str())
        .collect::<BTreeSet<_>>();
    let lifecycle_excluded = report
        .deployment_report
        .lifecycle_report
        .excluded_receipts
        .iter()
        .map(|item| item.receipt_id.as_str())
        .collect::<BTreeSet<_>>();

    receipts
        .iter()
        .filter(|receipt| receipt.matches(context))
        .filter(|receipt| {
            let id = receipt.scoped_receipt.receipt.receipt_id.as_str();
            !quarantined.contains(id) && !lifecycle_excluded.contains(id)
        })
        .map(|receipt| receipt.scoped_receipt.receipt.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_evidence_atomic_coverage::AtomicEvidenceFacet;
    use symthaea_evidence_deployment_scope::bind_receipt_to_deployment;
    use symthaea_evidence_lifecycle::ScopedSafetyEvidenceReceipt;
    use symthaea_evidence_time_assurance::{
        EvidenceTimeGuard, EvidenceTimePolicy, EvidenceTimeSample,
    };
    use symthaea_evidence_verifier_diversity::VerifierDiversityRequirement;
    use symthaea_formal_safety::{EvidenceKind, ProofObligation};

    fn case() -> SafetyCase {
        let mut case = SafetyCase::new("deep-readiness-fixture");
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

    fn receipt(
        case: &SafetyCase,
        id: &str,
        verifier: &str,
        valid_until_ms: u64,
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
            valid_until_ms,
            applicability_refs: vec!["deployment:node-1".into()],
        };
        bind_receipt_to_deployment(scoped, &context(), format!("scope:{id}")).unwrap()
    }

    fn trusted_time(uncertainty_ms: u64) -> TrustedEvidenceTime {
        let policy = EvidenceTimePolicy {
            schema_version: "1".into(),
            policy_id: "time-v1".into(),
            expected_clock_source: "ptp-a".into(),
            expected_clock_domain: "domain-a".into(),
            expected_epoch_ref: "epoch-a".into(),
            maximum_clock_uncertainty_ms: 100,
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
                clock_uncertainty_ms: uncertainty_ms,
                evidence_refs: vec!["clock:t1".into()],
            })
            .trusted_time
            .expect("trusted fixture time")
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

    fn diversity_policy(case: &SafetyCase, minimum: usize) -> VerifierDiversityPolicy {
        VerifierDiversityPolicy {
            schema_version: "1".into(),
            policy_id: format!("diversity-{minimum}"),
            requirements: vec![VerifierDiversityRequirement {
                obligation_key: case.obligations[0].stable_key(),
                minimum_distinct_verifiers: minimum,
                minimum_organization_domains: minimum,
                minimum_review_process_domains: minimum,
                minimum_toolchain_domains: minimum,
                minimum_evidence_source_domains: minimum,
                evidence_refs: vec!["review:diversity".into()],
            }],
            evidence_refs: vec!["policy:diversity".into()],
        }
    }

    fn atomic_policy(case: &SafetyCase, minimum: usize) -> AtomicCoveragePolicy {
        AtomicCoveragePolicy {
            schema_version: "1".into(),
            policy_id: format!("atomic-{minimum}"),
            facets: vec![AtomicEvidenceFacet {
                facet_id: "facet:a".into(),
                obligation_key: case.obligations[0].stable_key(),
                controlled_claim: "facet a".into(),
                minimum_distinct_receipts: minimum,
                minimum_distinct_evidence_objects: minimum,
                evidence_refs: vec!["review:facet-a".into()],
            }],
            evidence_refs: vec!["policy:atomic".into()],
        }
    }

    fn binding(id: &str, receipt_id: &str, digest: &str) -> FacetEvidenceBinding {
        FacetEvidenceBinding {
            binding_id: id.into(),
            receipt_id: receipt_id.into(),
            facet_id: "facet:a".into(),
            facet_evidence_ref: format!("result:{id}"),
            facet_evidence_digest: digest.into(),
            rationale_ref: format!("rationale:{id}"),
        }
    }

    #[test]
    fn all_deep_gates_ready_across_interval() {
        let case = case();
        let receipts = vec![
            receipt(&case, "r1", "verifier:a", 2_000),
            receipt(&case, "r2", "verifier:b", 2_000),
        ];
        let report = assess_deep_evidence_readiness(
            &case,
            &context(),
            &receipts,
            &[],
            &[],
            &[],
            &trusted_time(10),
            &[profile("verifier:a", "a"), profile("verifier:b", "b")],
            &diversity_policy(&case, 2),
            &atomic_policy(&case, 2),
            &[
                binding("b1", "r1", "blake3:facet-a-r1"),
                binding("b2", "r2", "blake3:facet-a-r2"),
            ],
        );
        assert_eq!(report.status, StrictSafetyCaseStatus::Ready);
        assert_eq!(report.earliest_active_receipt_count, 2);
        assert_eq!(report.latest_active_receipt_count, 2);
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn receipt_expiring_inside_interval_cannot_count_for_verifier_diversity() {
        let case = case();
        let receipts = vec![
            receipt(&case, "r1", "verifier:a", 2_000),
            receipt(&case, "r2", "verifier:b", 1_005),
        ];
        let report = assess_deep_evidence_readiness(
            &case,
            &context(),
            &receipts,
            &[],
            &[],
            &[],
            &trusted_time(10),
            &[profile("verifier:a", "a"), profile("verifier:b", "b")],
            &diversity_policy(&case, 2),
            &atomic_policy(&case, 1),
            &[binding("b1", "r1", "blake3:facet-a-r1")],
        );
        assert_eq!(report.time_report.status, StrictSafetyCaseStatus::Ready);
        assert_eq!(report.earliest_active_receipt_count, 2);
        assert_eq!(report.latest_active_receipt_count, 1);
        assert_eq!(report.latest_verifier_diversity.status, StrictSafetyCaseStatus::Blocked);
        assert_eq!(report.status, StrictSafetyCaseStatus::Blocked);
    }

    #[test]
    fn historical_expired_receipt_cannot_count_for_atomic_replication() {
        let case = case();
        let receipts = vec![
            receipt(&case, "r1", "verifier:a", 2_000),
            receipt(&case, "r2", "verifier:b", 1_005),
        ];
        let report = assess_deep_evidence_readiness(
            &case,
            &context(),
            &receipts,
            &[],
            &[],
            &[],
            &trusted_time(10),
            &[profile("verifier:a", "a"), profile("verifier:b", "b")],
            &diversity_policy(&case, 1),
            &atomic_policy(&case, 2),
            &[
                binding("b1", "r1", "blake3:facet-a-r1"),
                binding("b2", "r2", "blake3:facet-a-r2"),
            ],
        );
        assert_eq!(report.time_report.status, StrictSafetyCaseStatus::Ready);
        assert_eq!(report.latest_atomic_coverage.status, StrictSafetyCaseStatus::Blocked);
        assert_eq!(report.status, StrictSafetyCaseStatus::Blocked);
    }
}
