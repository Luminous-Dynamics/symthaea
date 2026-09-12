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
fn converged_simulation_updates_cognition_without_discharging_obligations() {
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
    concept.simulation_requests.push(SimulationRequest::new(
        "sim-static-G17-LC9",
        EngineeringDomain::Civil,
        SolverKind::FiniteElement,
        "observe bracket service stress",
    ));

    assert_eq!(concept.safety_case.obligations.len(), 1);
    assert_eq!(
        concept.safety_case.obligations[0].status,
        ObligationStatus::Open
    );
    assert!(concept.safety_case.obligations[0].evidence_refs.is_empty());
    assert!(manager.last_sensation.is_none());

    manager.evaluate_concept(&mut concept);

    // The converged result still reaches cognition through MetricEncoder.
    assert!(
        manager.last_sensation.is_some(),
        "removing authority mutation must not disable simulation sensation"
    );

    // ETK-3A authority theorem: an observation is not a discharge capability.
    let obligation = &concept.safety_case.obligations[0];
    assert_eq!(obligation.status, ObligationStatus::Open);
    assert!(obligation.evidence_refs.is_empty());
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
