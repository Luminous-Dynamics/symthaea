// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Adversarial qualification crucible for evidence-depth assurance.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_evidence_atomic_coverage::{
    AtomicCoveragePolicy, AtomicEvidenceFacet, FacetEvidenceBinding, assess_atomic_coverage,
};
use symthaea_evidence_verifier_diversity::{
    VerifierDiversityPolicy, VerifierDiversityRequirement, VerifierFaultDomainProfile,
    assess_verifier_diversity,
};
use symthaea_formal_safety::{
    EvidenceKind, ProofObligation, SafetyCase, SafetyEvidenceReceipt, StrictSafetyCaseStatus,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceDepthCrucibleStatus {
    Pass,
    Fail,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceDepthScenarioResult {
    pub scenario_id: String,
    pub expected: StrictSafetyCaseStatus,
    pub observed: StrictSafetyCaseStatus,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceDepthCrucibleReport {
    pub status: EvidenceDepthCrucibleStatus,
    pub scenarios: Vec<EvidenceDepthScenarioResult>,
}

impl EvidenceDepthCrucibleReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

pub fn run_evidence_depth_crucible() -> EvidenceDepthCrucibleReport {
    let case = fixture_case();
    let r1 = receipt(&case, "r1", "verifier:a", "blake3:bundle-r1");
    let r2 = receipt(&case, "r2", "verifier:b", "blake3:bundle-r2");
    let r2_same_verifier = receipt(&case, "r2-same", "verifier:a", "blake3:bundle-r2-same");

    let profile_a = profile("verifier:a", "org:a", "process:a", "tool:a", "source:a");
    let profile_b = profile("verifier:b", "org:b", "process:b", "tool:b", "source:b");
    let profile_b_same_org = profile(
        "verifier:b",
        "org:a",
        "process:b",
        "tool:b",
        "source:b",
    );

    let diversity_policy = diversity_policy(&case, 2);
    let mut scenarios = Vec::new();

    push(
        &mut scenarios,
        "verifier_diverse_positive",
        StrictSafetyCaseStatus::Ready,
        assess_verifier_diversity(
            &case,
            &[r1.clone(), r2.clone()],
            &[profile_a.clone(), profile_b.clone()],
            &diversity_policy,
        )
        .status,
    );

    push(
        &mut scenarios,
        "duplicate_receipts_same_verifier_block",
        StrictSafetyCaseStatus::Blocked,
        assess_verifier_diversity(
            &case,
            &[r1.clone(), r2_same_verifier],
            &[profile_a.clone()],
            &diversity_policy,
        )
        .status,
    );

    push(
        &mut scenarios,
        "distinct_verifier_ids_same_org_block",
        StrictSafetyCaseStatus::Blocked,
        assess_verifier_diversity(
            &case,
            &[r1.clone(), r2.clone()],
            &[profile_a.clone(), profile_b_same_org],
            &diversity_policy,
        )
        .status,
    );

    push(
        &mut scenarios,
        "missing_verifier_profile_blocks",
        StrictSafetyCaseStatus::Blocked,
        assess_verifier_diversity(
            &case,
            &[r1.clone(), r2.clone()],
            &[profile_a.clone()],
            &diversity_policy,
        )
        .status,
    );

    push(
        &mut scenarios,
        "duplicate_verifier_profile_invalid",
        StrictSafetyCaseStatus::Invalid,
        assess_verifier_diversity(
            &case,
            &[r1.clone(), r2.clone()],
            &[profile_a.clone(), profile_a.clone(), profile_b.clone()],
            &diversity_policy,
        )
        .status,
    );

    let atomic_policy = atomic_policy(&case);
    let explicit_bindings = vec![
        binding("b1", "r1", "facet:a", "blake3:facet-a"),
        binding("b2", "r1", "facet:b", "blake3:facet-b"),
    ];
    push(
        &mut scenarios,
        "atomic_explicit_positive",
        StrictSafetyCaseStatus::Ready,
        assess_atomic_coverage(&case, &[r1.clone()], &atomic_policy, &explicit_bindings).status,
    );

    push(
        &mut scenarios,
        "parent_receipt_has_no_implicit_facet_coverage",
        StrictSafetyCaseStatus::Blocked,
        assess_atomic_coverage(&case, &[r1.clone()], &atomic_policy, &[]).status,
    );

    push(
        &mut scenarios,
        "facet_coverage_does_not_leak_to_sibling",
        StrictSafetyCaseStatus::Blocked,
        assess_atomic_coverage(
            &case,
            &[r1.clone()],
            &atomic_policy,
            &[binding("b1", "r1", "facet:a", "blake3:facet-a")],
        )
        .status,
    );

    let replicated_policy = replicated_atomic_policy(&case);
    let repeated_object_bindings = vec![
        binding("b1", "r1", "facet:replicated", "blake3:same-result"),
        binding("b2", "r2", "facet:replicated", "blake3:same-result"),
    ];
    push(
        &mut scenarios,
        "same_subartifact_cannot_fake_distinct_evidence_objects",
        StrictSafetyCaseStatus::Blocked,
        assess_atomic_coverage(
            &case,
            &[r1.clone(), r2.clone()],
            &replicated_policy,
            &repeated_object_bindings,
        )
        .status,
    );

    let distinct_object_bindings = vec![
        binding("b1", "r1", "facet:replicated", "blake3:result-a"),
        binding("b2", "r2", "facet:replicated", "blake3:result-b"),
    ];
    push(
        &mut scenarios,
        "distinct_subartifacts_can_satisfy_replication_policy",
        StrictSafetyCaseStatus::Ready,
        assess_atomic_coverage(
            &case,
            &[r1, r2],
            &replicated_policy,
            &distinct_object_bindings,
        )
        .status,
    );

    let status = if scenarios.iter().all(|scenario| scenario.passed) {
        EvidenceDepthCrucibleStatus::Pass
    } else {
        EvidenceDepthCrucibleStatus::Fail
    };

    EvidenceDepthCrucibleReport { status, scenarios }
}

fn push(
    scenarios: &mut Vec<EvidenceDepthScenarioResult>,
    scenario_id: &str,
    expected: StrictSafetyCaseStatus,
    observed: StrictSafetyCaseStatus,
) {
    scenarios.push(EvidenceDepthScenarioResult {
        scenario_id: scenario_id.into(),
        expected,
        observed,
        passed: observed == expected,
    });
}

fn fixture_case() -> SafetyCase {
    let mut case = SafetyCase::new("evidence-depth-fixture");
    case.add_obligation(
        ProofObligation::new("composite qualification claim", EvidenceKind::Test)
            .discharge("qualification:reviewed"),
    );
    case
}

fn receipt(
    case: &SafetyCase,
    id: &str,
    verifier: &str,
    digest: &str,
) -> SafetyEvidenceReceipt {
    let obligation = &case.obligations[0];
    SafetyEvidenceReceipt {
        receipt_id: id.into(),
        obligation_key: obligation.stable_key(),
        evidence_kind: obligation.expected_evidence,
        evidence_ref: format!("artifact:{id}"),
        evidence_digest: digest.into(),
        verifier_ref: verifier.into(),
        verified_at_ms: 100,
    }
}

fn profile(
    verifier: &str,
    organization: &str,
    process: &str,
    toolchain: &str,
    source: &str,
) -> VerifierFaultDomainProfile {
    VerifierFaultDomainProfile {
        verifier_ref: verifier.into(),
        organization_domain: organization.into(),
        review_process_domain: process.into(),
        toolchain_domain: toolchain.into(),
        evidence_source_domain: source.into(),
        evidence_refs: vec![format!("profile:{verifier}")],
    }
}

fn diversity_policy(case: &SafetyCase, minimum: usize) -> VerifierDiversityPolicy {
    VerifierDiversityPolicy {
        schema_version: "1".into(),
        policy_id: "depth-diversity-v1".into(),
        requirements: vec![VerifierDiversityRequirement {
            obligation_key: case.obligations[0].stable_key(),
            minimum_distinct_verifiers: minimum,
            minimum_organization_domains: minimum,
            minimum_review_process_domains: minimum,
            minimum_toolchain_domains: minimum,
            minimum_evidence_source_domains: minimum,
            evidence_refs: vec!["review:depth-diversity".into()],
        }],
        evidence_refs: vec!["policy:depth-diversity-v1".into()],
    }
}

fn atomic_policy(case: &SafetyCase) -> AtomicCoveragePolicy {
    AtomicCoveragePolicy {
        schema_version: "1".into(),
        policy_id: "depth-atomic-v1".into(),
        facets: vec![
            facet(case, "facet:a", 1, 1),
            facet(case, "facet:b", 1, 1),
        ],
        evidence_refs: vec!["policy:depth-atomic-v1".into()],
    }
}

fn replicated_atomic_policy(case: &SafetyCase) -> AtomicCoveragePolicy {
    AtomicCoveragePolicy {
        schema_version: "1".into(),
        policy_id: "depth-atomic-replicated-v1".into(),
        facets: vec![facet(case, "facet:replicated", 2, 2)],
        evidence_refs: vec!["policy:depth-atomic-replicated-v1".into()],
    }
}

fn facet(
    case: &SafetyCase,
    id: &str,
    min_receipts: usize,
    min_objects: usize,
) -> AtomicEvidenceFacet {
    AtomicEvidenceFacet {
        facet_id: id.into(),
        obligation_key: case.obligations[0].stable_key(),
        controlled_claim: format!("controlled facet {id}"),
        minimum_distinct_receipts: min_receipts,
        minimum_distinct_evidence_objects: min_objects,
        evidence_refs: vec![format!("facet-review:{id}")],
    }
}

fn binding(
    id: &str,
    receipt_id: &str,
    facet_id: &str,
    digest: &str,
) -> FacetEvidenceBinding {
    FacetEvidenceBinding {
        binding_id: id.into(),
        receipt_id: receipt_id.into(),
        facet_id: facet_id.into(),
        facet_evidence_ref: format!("result:{id}"),
        facet_evidence_digest: digest.into(),
        rationale_ref: format!("rationale:{id}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evidence_depth_crucible_passes_all_reviewed_controls() {
        let report = run_evidence_depth_crucible();
        assert_eq!(report.status, EvidenceDepthCrucibleStatus::Pass);
        assert!(report.scenarios.len() >= 10);
        assert!(report.scenarios.iter().all(|scenario| scenario.passed));
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn required_negative_controls_are_present() {
        let report = run_evidence_depth_crucible();
        let ids = report
            .scenarios
            .iter()
            .map(|scenario| scenario.scenario_id.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        for required in [
            "duplicate_receipts_same_verifier_block",
            "distinct_verifier_ids_same_org_block",
            "parent_receipt_has_no_implicit_facet_coverage",
            "facet_coverage_does_not_leak_to_sibling",
            "same_subartifact_cannot_fake_distinct_evidence_objects",
        ] {
            assert!(ids.contains(required), "missing scenario {required}");
        }
    }
}
