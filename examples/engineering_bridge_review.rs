// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Civil-structure engineering review example.
//!
//! Run:
//!   cargo run --example engineering_bridge_review --features engineering-foundations

use symthaea::engineering::digital_twin::{AssetClass, TelemetryPoint, TwinState};
use symthaea::engineering::formal_safety::EvidenceKind;
use symthaea::engineering::sim_bridge::{EngineeringDomain, SolverKind};
use symthaea::engineering::{
    EngineeringConcept, EngineeringRequirement, EngineeringReview, RequirementCriticality,
};

fn main() {
    let mut concept = EngineeringConcept::new(
        "bridge-foot-001",
        "low-carbon footbridge span",
        EngineeringDomain::Civil,
    );
    concept.add_requirement(EngineeringRequirement::new(
        "REQ-STRESS-SERVICE",
        EngineeringDomain::Civil,
        "service stress remains below allowable limit under pedestrian live load",
        RequirementCriticality::Blocking,
        EvidenceKind::Simulation,
    ));
    concept.request_simulation(
        "SIM-LIVE-LOAD-001",
        SolverKind::FiniteElement,
        "screen service stress and midspan displacement",
    );

    let mut twin = TwinState::new(
        "bridge-span-001",
        AssetClass::CivilStructure,
        "north approach footbridge span",
    );
    twin.ingest_with_uncertainty(
        TelemetryPoint::now("midspan_strain", 128.0, "microstrain"),
        100.0,
        15.0,
        0.12,
    );

    let review = EngineeringReview {
        concept,
        twin: Some(twin),
    };

    let twin = review.twin.as_ref().expect("example attaches a twin");
    println!("subject: {}", review.concept.label);
    println!("blocks_deployment: {}", review.blocks_deployment());
    println!("free_energy: {:.3}", twin.free_energy);
    println!("expected_free_energy: {:.3}", twin.expected_free_energy());
    println!(
        "residuals: {} latest_health={:.3}",
        twin.prediction_residuals.len(),
        twin.health
    );
    println!(
        "simulation_requests: {}",
        review.concept.simulation_requests.len()
    );
}
