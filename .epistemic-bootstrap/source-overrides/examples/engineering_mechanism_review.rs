// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mechanical-system engineering review example.
//!
//! Run:
//!   cargo run --example engineering_mechanism_review --features engineering-foundations

use symthaea::engineering::digital_twin::{AssetClass, TelemetryPoint, TwinState};
use symthaea::engineering::formal_safety::EvidenceKind;
use symthaea::engineering::sim_bridge::{EngineeringDomain, SolverKind};
use symthaea::engineering::{
    EngineeringConcept, EngineeringRequirement, EngineeringReview, RequirementCriticality,
};

fn main() {
    let mut concept = EngineeringConcept::new(
        "actuator-linkage-001",
        "inspection robot four-bar linkage",
        EngineeringDomain::Mechanical,
    );
    concept.add_requirement(EngineeringRequirement::new(
        "REQ-CLEARANCE",
        EngineeringDomain::Mechanical,
        "linkage maintains positive clearance through commanded travel",
        RequirementCriticality::High,
        EvidenceKind::Simulation,
    ));
    concept.add_requirement(EngineeringRequirement::new(
        "REQ-TORQUE",
        EngineeringDomain::Mechanical,
        "peak motor torque remains below rated continuous torque",
        RequirementCriticality::Blocking,
        EvidenceKind::Test,
    ));
    concept.request_simulation(
        "SIM-KINEMATICS-001",
        SolverKind::MultibodyDynamics,
        "screen linkage clearance and motor torque envelope",
    );

    let mut twin = TwinState::new(
        "linkage-testbed-001",
        AssetClass::MechanicalSystem,
        "bench actuator linkage",
    );
    for observed_torque in [2.1, 2.2, 2.9] {
        twin.ingest_with_uncertainty(
            TelemetryPoint::now("motor_torque", observed_torque, "N*m"),
            2.0,
            0.25,
            0.08,
        );
    }

    let review = EngineeringReview {
        concept,
        twin: Some(twin),
    };
    let evidence_root = std::path::Path::new("engineering-evidence");

    let twin = review.twin.as_ref().expect("example attaches a twin");
    println!("subject: {}", review.concept.label);
    println!(
        "blocks_deployment: {}",
        review.blocks_deployment(evidence_root)
    );
    println!("free_energy: {:.3}", twin.free_energy);
    println!("expected_free_energy: {:.3}", twin.expected_free_energy());
    println!(
        "epistemic_uncertainty: {:.3} aleatoric_uncertainty: {:.3}",
        twin.epistemic_uncertainty, twin.aleatoric_uncertainty
    );
    println!(
        "safety_obligations: {}",
        review.concept.safety_case.obligations.len()
    );
}
