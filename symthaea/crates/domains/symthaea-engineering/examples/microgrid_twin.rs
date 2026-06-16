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
    EngineeringConcept, EngineeringReview,
    digital_twin::{AssetClass, TelemetryPoint, TwinState},
    formal_safety::{SafetyCase, SafetyCaseTemplate},
    sim_bridge::{EngineeringDomain, SolverKind},
};

fn main() {
    let mut twin = TwinState::new(
        "microgrid-alpha",
        AssetClass::ElectricalSystem,
        "campus microgrid",
    );

    twin.ingest_with_uncertainty(
        TelemetryPoint::now("frequency", 49.4, "Hz"),
        50.0,
        0.2,
        0.15,
    );
    twin.ingest_with_uncertainty(
        TelemetryPoint::now("battery_soc", 0.18, "ratio"),
        0.35,
        0.1,
        0.1,
    );

    let mut concept = EngineeringConcept::new(
        "grid-001",
        "islandable microgrid dispatch",
        EngineeringDomain::Electrical,
    );
    concept.request_simulation(
        "SIM-GRID-TRANSIENT",
        SolverKind::Circuit,
        "check transient response and protection behavior under islanding",
    );
    concept.safety_case =
        SafetyCase::from_template(&concept.label, SafetyCaseTemplate::ElectricalSystem);

    let review = EngineeringReview {
        concept,
        twin: Some(twin),
    };

    let twin = review.twin.as_ref().unwrap();
    println!("microgrid free energy: {:.3}", twin.free_energy);
    println!(
        "microgrid expected free energy: {:.3}",
        twin.expected_free_energy()
    );
    println!(
        "microgrid needs intervention: {}",
        twin.needs_intervention(1.0)
    );
    println!("review blocks deployment: {}", review.blocks_deployment());
}
