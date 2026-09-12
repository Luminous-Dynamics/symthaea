// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! ETK-3A regression: simulation execution may inform cognition but must not
//! directly mutate proof-obligation authority state.

use symthaea_engineering::{
    EngineeringConcept, EngineeringManager, EngineeringRequirement, RequirementCriticality,
};
use symthaea_engineering::formal_safety::{EvidenceKind, ObligationStatus};
use symthaea_engineering::sim_bridge::{
    EngineeringDomain, SimulationBackend, SimulationError, SimulationRequest, SimulationResult,
    SolverKind,
};

#[derive(Debug)]
struct ConvergedFiniteElementBackend;

impl SimulationBackend for ConvergedFiniteElementBackend {
    fn name(&self) -> &'static str {
        "etk-3a-converged-fixture"
    }

    fn supported_solvers(&self) -> &[SolverKind] {
        &[SolverKind::FiniteElement]
    }

    fn run(&self, request: &SimulationRequest) -> Result<SimulationResult, SimulationError> {
        Ok(
            SimulationResult::converged(&request.id, 0.95)
                .with_metric("max_stress_mpa", 181.2, "MPa"),
        )
    }
}

#[test]
fn converged_simulation_updates_cognition_without_mutating_obligation_authority() {
    let mut manager = EngineeringManager::new();
    manager.registry.register(ConvergedFiniteElementBackend);

    let mut concept = EngineeringConcept::new(
        "etk-3a-bracket",
        "ETK-3A bracket",
        EngineeringDomain::Civil,
    );
    concept.add_requirement(EngineeringRequirement::new(
        "REQ-STRESS",
        EngineeringDomain::Civil,
        "stress remains below allowable under service load",
        RequirementCriticality::Blocking,
        EvidenceKind::Simulation,
    ));
    concept.add_requirement(EngineeringRequirement::new(
        "REQ-DEFLECTION",
        EngineeringDomain::Civil,
        "deflection remains below serviceability limit",
        RequirementCriticality::Blocking,
        EvidenceKind::Simulation,
    ));

    // Preserve a non-default pre-existing authority state too. The old blanket
    // loop would overwrite both Open and InProgress simulation obligations.
    concept.safety_case.obligations[1].status = ObligationStatus::InProgress;

    concept.simulation_requests.push(SimulationRequest::new(
        "sim-static-G17-LC9",
        EngineeringDomain::Civil,
        SolverKind::FiniteElement,
        "observe bracket service stress",
    ));

    assert_eq!(concept.safety_case.obligations.len(), 2);
    let before = concept
        .safety_case
        .obligations
        .iter()
        .map(|obligation| (obligation.status, obligation.evidence_refs.clone()))
        .collect::<Vec<_>>();
    assert_eq!(before[0].0, ObligationStatus::Open);
    assert_eq!(before[1].0, ObligationStatus::InProgress);
    assert!(manager.last_sensation.is_none());

    manager.evaluate_concept(&mut concept);

    // The converged result still reaches cognition through MetricEncoder.
    assert!(
        manager.last_sensation.is_some(),
        "removing authority mutation must not disable simulation sensation"
    );

    // ETK-3A authority theorem: an observation is not a discharge capability,
    // and evaluate_concept preserves every pre-existing obligation state/ref set.
    let after = concept
        .safety_case
        .obligations
        .iter()
        .map(|obligation| (obligation.status, obligation.evidence_refs.clone()))
        .collect::<Vec<_>>();
    assert_eq!(after, before);
    assert_eq!(after[0].0, ObligationStatus::Open);
    assert_eq!(after[1].0, ObligationStatus::InProgress);
    assert!(!concept.safety_case.is_discharged());
}

#[test]
fn facade_source_contains_no_evaluate_concept_auto_discharge_pattern() {
    let source = include_str!("../src/lib.rs");
    let evaluate_start = source
        .find("pub fn evaluate_concept")
        .expect("engineering facade must retain evaluate_concept");
    let remainder = &source[evaluate_start..];
    let evaluate_end = remainder
        .find("pub fn predict_intervention")
        .expect("expected next facade method after evaluate_concept");
    let body = &remainder[..evaluate_end];

    assert!(body.contains("self.last_sensation = Some(sensation);"));
    assert!(!body.contains("ObligationStatus::Discharged"));
    assert!(!body.contains("obligation.status ="));
    assert!(!body.contains("evidence_refs.push"));
}
