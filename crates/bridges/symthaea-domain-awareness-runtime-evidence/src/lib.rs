// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Candidate-evidence bindings from runtime assurance qualifications into
//! canonical DomainAwareness obligations.
//!
//! Candidate evidence is intentionally weaker than a verified receipt. This
//! crate never mutates obligation status and never grants physical authority.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use symthaea_domain_awareness_evidence::{ArtifactBinding, CandidateEvidence};
use symthaea_evidence_dependency::{
    EvidenceDependencyEdge, EvidenceDependencyNode, EvidenceDependencyNodeKind,
    EvidenceDependencyStatus, assess_evidence_dependency_graph,
};
use symthaea_evidence_runtime_crucible::{
    EvidenceRuntimeCrucibleReport, EvidenceRuntimeCrucibleStatus,
};
use symthaea_formal_safety::DomainAwarenessObligation;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeEvidenceBindingError {
    InvalidArtifactBinding,
    RuntimeCrucibleDidNotPass,
    DuplicateScenarioId(String),
    MissingRequiredScenario(String),
    RequiredScenarioDidNotPass(String),
    DependencyQualificationDidNotPass,
}

fn candidate(
    obligation: DomainAwarenessObligation,
    binding: &ArtifactBinding,
    observed_at_ms: u64,
    source_id: &str,
    rationale: impl Into<String>,
) -> Result<CandidateEvidence, RuntimeEvidenceBindingError> {
    if !binding.validate() {
        return Err(RuntimeEvidenceBindingError::InvalidArtifactBinding);
    }
    let candidate = CandidateEvidence {
        candidate_id: format!("{}:{}:{}", obligation.code(), source_id, observed_at_ms),
        obligation,
        evidence_ref: binding.evidence_ref.clone(),
        evidence_digest: binding.evidence_digest.clone(),
        observed_at_ms,
        rationale: rationale.into(),
    };
    if !candidate.validate() {
        return Err(RuntimeEvidenceBindingError::InvalidArtifactBinding);
    }
    Ok(candidate)
}

fn scenario_index(
    report: &EvidenceRuntimeCrucibleReport,
) -> Result<BTreeMap<&str, bool>, RuntimeEvidenceBindingError> {
    if report.status != EvidenceRuntimeCrucibleStatus::Pass
        || report.scenarios.iter().any(|scenario| !scenario.passed)
    {
        return Err(RuntimeEvidenceBindingError::RuntimeCrucibleDidNotPass);
    }

    let mut by_id = BTreeMap::new();
    for scenario in &report.scenarios {
        if by_id.insert(scenario.scenario_id.as_str(), scenario.passed).is_some() {
            return Err(RuntimeEvidenceBindingError::DuplicateScenarioId(
                scenario.scenario_id.clone(),
            ));
        }
    }
    Ok(by_id)
}

fn require_scenarios(
    by_id: &BTreeMap<&str, bool>,
    required: &[&str],
) -> Result<(), RuntimeEvidenceBindingError> {
    for scenario_id in required {
        let Some(passed) = by_id.get(scenario_id).copied() else {
            return Err(RuntimeEvidenceBindingError::MissingRequiredScenario(
                (*scenario_id).to_string(),
            ));
        };
        if !passed {
            return Err(RuntimeEvidenceBindingError::RequiredScenarioDidNotPass(
                (*scenario_id).to_string(),
            ));
        }
    }
    Ok(())
}

/// Convert a passing runtime-evidence crucible artifact into candidate evidence
/// for DA-024..DA-027.
///
/// The artifact binding should identify the complete qualification bundle,
/// including the crate's edge-case integration tests. These candidates remain
/// subject to independent verification and explicit safety-case discharge.
pub fn runtime_crucible_candidates(
    report: &EvidenceRuntimeCrucibleReport,
    binding: &ArtifactBinding,
    observed_at_ms: u64,
) -> Result<Vec<CandidateEvidence>, RuntimeEvidenceBindingError> {
    let by_id = scenario_index(report)?;

    require_scenarios(
        &by_id,
        &[
            "baseline-ready",
            "expired-evidence-blocks",
            "revocation-blocks",
            "unresolved-contradiction-blocks",
        ],
    )?;
    let applicability = candidate(
        DomainAwarenessObligation::CurrentEvidenceApplicabilityRequired,
        binding,
        observed_at_ms,
        "runtime-evidence-crucible",
        "passing runtime qualification demonstrates current applicability gates for baseline, expiry, revocation, and unresolved contradiction; independent verification must also confirm the bound edge-case test bundle",
    )?;

    require_scenarios(
        &by_id,
        &[
            "configuration-drift-blocks",
            "model-manifest-drift-blocks",
            "calibration-manifest-drift-blocks",
            "replacement-evidence-restores-after-config-drift",
        ],
    )?;
    let drift = candidate(
        DomainAwarenessObligation::ConfigurationDriftInvalidatesEvidence,
        binding,
        observed_at_ms,
        "runtime-evidence-crucible",
        "passing runtime qualification demonstrates configuration/model/calibration drift removes readiness and separately qualified replacement evidence restores it",
    )?;

    require_scenarios(
        &by_id,
        &[
            "verifier-quarantine-blocks",
            "post-remediation-replacement-restores-quarantined-evidence",
            "unresolved-contradiction-blocks",
            "reviewed-contradiction-plus-replacement-restores-readiness",
        ],
    )?;
    let integrity = candidate(
        DomainAwarenessObligation::EvidenceIntegrityHoldsPropagate,
        binding,
        observed_at_ms,
        "runtime-evidence-crucible",
        "passing runtime qualification demonstrates verifier-wide quarantine and contradiction holds block readiness until reviewed remediation/replacement; independent verification must confirm all quarantine-selector edge tests in the bound artifact",
    )?;

    require_scenarios(
        &by_id,
        &[
            "trusted-clock-positive-control",
            "clock-uncertainty-straddling-expiry-blocks",
            "wall-clock-rollback-latches-untrusted",
        ],
    )?;
    let time = candidate(
        DomainAwarenessObligation::TrustedTimeGatesReadiness,
        binding,
        observed_at_ms,
        "runtime-evidence-crucible",
        "passing runtime qualification demonstrates trusted-time positive control, uncertainty-window expiry blocking, and rollback latching; independent verification must confirm time-replay edge tests in the bound artifact",
    )?;

    Ok(vec![applicability, drift, integrity, time])
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DependencyQualificationStatus {
    Pass,
    Fail,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DependencyQualificationScenario {
    pub scenario_id: String,
    pub expected: EvidenceDependencyStatus,
    pub observed: EvidenceDependencyStatus,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DependencyQualificationReport {
    pub schema_version: String,
    pub status: DependencyQualificationStatus,
    pub scenarios: Vec<DependencyQualificationScenario>,
}

impl DependencyQualificationReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

fn dependency_node(id: &str, kind: EvidenceDependencyNodeKind) -> EvidenceDependencyNode {
    EvidenceDependencyNode {
        node_id: id.into(),
        kind,
        evidence_ref: format!("qualification:{id}"),
    }
}

fn dependency_edge(dependent: &str, prerequisite: &str) -> EvidenceDependencyEdge {
    EvidenceDependencyEdge {
        dependent: dependent.into(),
        prerequisite: prerequisite.into(),
        rationale_ref: format!("qualification:{dependent}:{prerequisite}"),
    }
}

fn dependency_scenario(
    id: &str,
    expected: EvidenceDependencyStatus,
    nodes: Vec<EvidenceDependencyNode>,
    edges: Vec<EvidenceDependencyEdge>,
) -> DependencyQualificationScenario {
    let observed = assess_evidence_dependency_graph(&nodes, &edges).status;
    DependencyQualificationScenario {
        scenario_id: id.into(),
        expected,
        observed,
        passed: observed == expected,
    }
}

/// Exercise both positive and negative controls for the evidence dependency DAG.
pub fn run_dependency_qualification() -> DependencyQualificationReport {
    let contract_a = "blake3:contract-a";
    let contract_b = "blake3:contract-b";
    let scenarios = vec![
        dependency_scenario(
            "acyclic-chain-valid",
            EvidenceDependencyStatus::Valid,
            vec![
                dependency_node("raw", EvidenceDependencyNodeKind::RawObservation),
                dependency_node("verify", EvidenceDependencyNodeKind::Verification),
                dependency_node(
                    "receipt-a",
                    EvidenceDependencyNodeKind::Receipt {
                        contract_digest: contract_a.into(),
                    },
                ),
            ],
            vec![
                dependency_edge("verify", "raw"),
                dependency_edge("receipt-a", "verify"),
            ],
        ),
        dependency_scenario(
            "cycle-rejected",
            EvidenceDependencyStatus::Invalid,
            vec![
                dependency_node("a", EvidenceDependencyNodeKind::TestArtifact),
                dependency_node("b", EvidenceDependencyNodeKind::Verification),
            ],
            vec![dependency_edge("a", "b"), dependency_edge("b", "a")],
        ),
        dependency_scenario(
            "same-contract-readiness-backdependency-rejected",
            EvidenceDependencyStatus::Invalid,
            vec![
                dependency_node(
                    "ready-a",
                    EvidenceDependencyNodeKind::ReadinessDecision {
                        contract_digest: contract_a.into(),
                    },
                ),
                dependency_node("artifact", EvidenceDependencyNodeKind::TestArtifact),
                dependency_node(
                    "receipt-a",
                    EvidenceDependencyNodeKind::Receipt {
                        contract_digest: contract_a.into(),
                    },
                ),
            ],
            vec![
                dependency_edge("artifact", "ready-a"),
                dependency_edge("receipt-a", "artifact"),
            ],
        ),
        dependency_scenario(
            "cross-contract-readiness-remains-explicit",
            EvidenceDependencyStatus::Valid,
            vec![
                dependency_node(
                    "ready-b",
                    EvidenceDependencyNodeKind::ReadinessDecision {
                        contract_digest: contract_b.into(),
                    },
                ),
                dependency_node(
                    "receipt-a",
                    EvidenceDependencyNodeKind::Receipt {
                        contract_digest: contract_a.into(),
                    },
                ),
            ],
            vec![dependency_edge("receipt-a", "ready-b")],
        ),
    ];
    let status = if scenarios.iter().all(|scenario| scenario.passed) {
        DependencyQualificationStatus::Pass
    } else {
        DependencyQualificationStatus::Fail
    };
    DependencyQualificationReport {
        schema_version: "1".into(),
        status,
        scenarios,
    }
}

pub fn dependency_qualification_candidate(
    report: &DependencyQualificationReport,
    binding: &ArtifactBinding,
    observed_at_ms: u64,
) -> Result<CandidateEvidence, RuntimeEvidenceBindingError> {
    if report.status != DependencyQualificationStatus::Pass
        || report.scenarios.is_empty()
        || report.scenarios.iter().any(|scenario| !scenario.passed)
    {
        return Err(RuntimeEvidenceBindingError::DependencyQualificationDidNotPass);
    }
    let required = BTreeSet::from([
        "acyclic-chain-valid",
        "cycle-rejected",
        "same-contract-readiness-backdependency-rejected",
        "cross-contract-readiness-remains-explicit",
    ]);
    let observed = report
        .scenarios
        .iter()
        .map(|scenario| scenario.scenario_id.as_str())
        .collect::<BTreeSet<_>>();
    if !required.is_subset(&observed) {
        return Err(RuntimeEvidenceBindingError::DependencyQualificationDidNotPass);
    }

    candidate(
        DomainAwarenessObligation::EvidenceDependenciesAreAcyclicAndNonCircular,
        binding,
        observed_at_ms,
        "evidence-dependency-qualification",
        "positive and negative dependency controls demonstrate acyclic evidence is accepted while ordinary cycles and same-contract readiness back-dependencies are rejected",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_evidence_runtime_crucible::run_evidence_runtime_crucible;

    fn binding() -> ArtifactBinding {
        ArtifactBinding {
            evidence_ref: "artifact:runtime-assurance-bundle".into(),
            evidence_digest: "blake3:runtime-assurance-bundle".into(),
        }
    }

    #[test]
    fn passing_runtime_crucible_maps_to_da_024_through_da_027() {
        let report = run_evidence_runtime_crucible();
        let candidates = runtime_crucible_candidates(&report, &binding(), 2_000).unwrap();
        let obligations = candidates
            .iter()
            .map(|candidate| candidate.obligation)
            .collect::<BTreeSet<_>>();
        assert_eq!(candidates.len(), 4);
        assert!(obligations.contains(&DomainAwarenessObligation::CurrentEvidenceApplicabilityRequired));
        assert!(obligations.contains(&DomainAwarenessObligation::ConfigurationDriftInvalidatesEvidence));
        assert!(obligations.contains(&DomainAwarenessObligation::EvidenceIntegrityHoldsPropagate));
        assert!(obligations.contains(&DomainAwarenessObligation::TrustedTimeGatesReadiness));
        assert!(candidates.iter().all(|candidate| !candidate.grants_physical_authority()));
    }

    #[test]
    fn missing_required_runtime_scenario_fails_closed() {
        let mut report = run_evidence_runtime_crucible();
        report.scenarios.retain(|scenario| scenario.scenario_id != "configuration-drift-blocks");
        let result = runtime_crucible_candidates(&report, &binding(), 2_000);
        assert!(matches!(
            result,
            Err(RuntimeEvidenceBindingError::MissingRequiredScenario(id))
                if id == "configuration-drift-blocks"
        ));
    }

    #[test]
    fn dependency_qualification_has_positive_and_negative_controls() {
        let report = run_dependency_qualification();
        assert_eq!(report.status, DependencyQualificationStatus::Pass);
        assert!(report.scenarios.iter().all(|scenario| scenario.passed));
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn passing_dependency_qualification_maps_only_to_da_028() {
        let report = run_dependency_qualification();
        let candidate = dependency_qualification_candidate(&report, &binding(), 2_000).unwrap();
        assert_eq!(
            candidate.obligation,
            DomainAwarenessObligation::EvidenceDependenciesAreAcyclicAndNonCircular
        );
        assert!(!candidate.grants_physical_authority());
    }

    #[test]
    fn failed_dependency_qualification_cannot_create_candidate() {
        let mut report = run_dependency_qualification();
        report.status = DependencyQualificationStatus::Fail;
        let result = dependency_qualification_candidate(&report, &binding(), 2_000);
        assert_eq!(
            result,
            Err(RuntimeEvidenceBindingError::DependencyQualificationDidNotPass)
        );
    }
}
