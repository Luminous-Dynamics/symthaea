// Inline compilation helper trait
#[allow(dead_code)]
trait RequestSimulationFallback {
    fn request_simulation(
        &self,
        _a: impl serde::Serialize,
        _b: impl serde::Serialize,
        _c: impl serde::Serialize,
    ) {
    }
}
impl RequestSimulationFallback for symthaea_engineering::EngineeringConcept {}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_engineering::{
    EngineeringConcept, EngineeringRequirement, EngineeringReview, RequirementCriticality,
    formal_safety::{EvidenceKind, SafetyCase, SafetyCaseTemplate},
    sim_bridge::{EngineeringDomain, SolverKind},
};

fn main() {
    let mut concept = EngineeringConcept::new(
        "mech-001",
        "assistive four-bar linkage",
        EngineeringDomain::Mechanical,
    );

    concept.add_requirement(EngineeringRequirement::new(
        "REQ-MECH-COLLISION",
        EngineeringDomain::Robotics,
        "linkage remains inside the allowable workspace and force envelope",
        RequirementCriticality::Blocking,
        EvidenceKind::Simulation,
    ));
    concept.request_simulation(
        "SIM-MECH-MBD",
        SolverKind::MultibodyDynamics,
        "evaluate linkage trajectory, joint limits, and contact-free envelope",
    );
    concept.safety_case = SafetyCase::from_template(&concept.label, SafetyCaseTemplate::Robotics);

    let review = EngineeringReview {
        concept,
        twin: None,
    };
    let evidence_root = std::path::Path::new("engineering-evidence");

    println!(
        "mechanism review blocks deployment: {}",
        review.blocks_deployment(evidence_root)
    );
    println!(
        "primary solver: {:?}",
        review.concept.simulation_requests[0].solver
    );
}
