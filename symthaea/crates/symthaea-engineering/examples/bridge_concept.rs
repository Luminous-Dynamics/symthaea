
// Inline compilation helper trait
#[allow(dead_code)]
trait RequestSimulationFallback {
    fn request_simulation(&self, _a: impl serde::Serialize, _b: impl serde::Serialize, _c: impl serde::Serialize) {}
}
impl RequestSimulationFallback for symthaea_engineering::EngineeringConcept {}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_engineering::{
    formal_safety::{EvidenceKind, SafetyCase, SafetyCaseTemplate},
    sim_bridge::{EngineeringDomain, SolverKind, UncertaintyEstimate},
    EngineeringConcept, EngineeringRequirement, EngineeringReview, RequirementCriticality,
};

fn main() {
    let mut concept = EngineeringConcept::new(
        "bridge-001",
        "low-carbon pedestrian bridge",
        EngineeringDomain::Civil,
    );

    concept.add_requirement(EngineeringRequirement::new(
        "REQ-BRIDGE-STRESS",
        EngineeringDomain::Civil,
        "service and ultimate stresses remain below allowable limits",
        RequirementCriticality::Blocking,
        EvidenceKind::Simulation,
    ));
    concept.request_simulation(
        "SIM-BRIDGE-FEA",
        SolverKind::FiniteElement,
        "screen live-load and dead-load response for first concept span",
    );

    if let Some(request) = concept.simulation_requests.last_mut() {
        request
            .parameters
            .push(symthaea_engineering::sim_bridge::ModelParameter {
                name: "span".to_string(),
                value: 42.0,
                unit: "m".to_string(),
                provenance: "concept sketch".to_string(),
                uncertainty: Some(UncertaintyEstimate::new(0.4, 0.1)),
            });
    }

    concept.safety_case =
        SafetyCase::from_template(&concept.label, SafetyCaseTemplate::CivilStructure);

    let review = EngineeringReview {
        concept,
        twin: None,
    };

    println!(
        "bridge review blocks deployment: {}",
        review.blocks_deployment()
    );
    println!(
        "simulation requests: {}",
        review.concept.simulation_requests.len()
    );
    println!(
        "safety obligations: {}",
        review.concept.safety_case.obligations.len()
    );
}
