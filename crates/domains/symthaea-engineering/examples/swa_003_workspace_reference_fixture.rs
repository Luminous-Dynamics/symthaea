// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! SWA-003 deterministic building -> workspace evidence fixture.
//!
//! This is deliberately an adapter/reference fixture, not a new workspace
//! domain engine. It composes the existing BuildingTwin and TwinState
//! primitives and keeps physical, economic, resilience, agency, and
//! reversibility dimensions separate.
//!
//! The fixture stops before authorization or actuation. A future Mycelix
//! adapter can project the evidence into a durable decision receipt without
//! changing the evidence semantics.

use serde::Serialize;
use symthaea_digital_twin::{AssetClass, TwinState};
use symthaea_fabrication_kernel::building::{BuildingOutput, BuildingReading, BuildingTwin};

const BASELINE: BuildingReading = BuildingReading {
    thermal_load: 0.40,
    structural_stress: 0.10,
    occupancy: 0.60,
    comfort: 0.85,
    energy_consumption: 0.35,
};

const ADAPTED: BuildingReading = BuildingReading {
    thermal_load: 0.32,
    structural_stress: 0.10,
    occupancy: 0.60,
    comfort: 0.91,
    energy_consumption: 0.29,
};

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
struct WorkspaceDimensions {
    comfort: f64,
    energy: f64,
    resilience: f64,
    lifecycle_cost: f64,
    maintenance_burden: f64,
    reversibility: f64,
    privacy: f64,
    agency: f64,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
struct WorkspaceIntervention {
    id: &'static str,
    description: &'static str,
    commitment_months: u32,
    fixed_cost: f64,
    variable_cost_per_month: f64,
    reconfiguration_cost: f64,
    exit_cost: f64,
    alternative_use_value: f64,
    downside_liquidity_exposure: f64,
    predicted: WorkspaceDimensions,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
struct WorkspaceObservation {
    zone_id: &'static str,
    reading: BuildingReading,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
struct PredictionError {
    intervention_id: &'static str,
    predicted_comfort: f64,
    observed_comfort: f64,
    comfort_error: f64,
    predicted_energy: f64,
    observed_energy: f64,
    energy_error: f64,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
struct ConstraintCandidate {
    id: &'static str,
    mechanism: &'static str,
    constraint: &'static str,
    lineage_intervention: &'static str,
    adopted: bool,
}

#[derive(Debug, Serialize)]
struct FixtureEvidence {
    fixture_id: &'static str,
    building: &'static str,
    observation: WorkspaceObservation,
    twin_health: f64,
    twin_epistemic_uncertainty: f64,
    twin_aleatoric_uncertainty: f64,
    building_free_energy: f64,
    building_action: String,
    intervention: WorkspaceIntervention,
    outcome: WorkspaceObservation,
    prediction_error: PredictionError,
    constraint_candidate: ConstraintCandidate,
    authorization_required: bool,
    actuation_performed: bool,
}

fn run_building_twin(reading: &BuildingReading) -> BuildingOutput {
    let mut twin = BuildingTwin::new();
    twin.set_reference(&BASELINE);
    twin.step(reading, 3_600.0)
}

fn fixture() -> FixtureEvidence {
    let observation = WorkspaceObservation {
        zone_id: "building-001/zone-a",
        reading: BASELINE,
    };

    let outcome = WorkspaceObservation {
        zone_id: "building-001/zone-a",
        reading: ADAPTED,
    };

    let mut state = TwinState::new(
        "building-001",
        AssetClass::CivilStructure,
        "deterministic workspace fixture",
    );

    // Deliberately avoid wall-clock timestamps in this fixture. The physical
    // evidence is fixed so replay is byte-stable after canonical serialization.
    state.health = 0.97;
    state.epistemic_uncertainty = 0.20;
    state.aleatoric_uncertainty = 0.05;

    let baseline_output = run_building_twin(&BASELINE);
    let adapted_output = run_building_twin(&ADAPTED);

    let intervention = WorkspaceIntervention {
        id: "workspace-intervention-modular-hvac",
        description: "Zone-level HVAC setpoint/configuration change with reversible controls",
        commitment_months: 3,
        fixed_cost: 1200.0,
        variable_cost_per_month: 180.0,
        reconfiguration_cost: 250.0,
        exit_cost: 100.0,
        alternative_use_value: 900.0,
        downside_liquidity_exposure: 0.12,
        predicted: WorkspaceDimensions {
            comfort: 0.90,
            energy: 0.30,
            resilience: 0.86,
            lifecycle_cost: 0.68,
            maintenance_burden: 0.18,
            reversibility: 0.91,
            privacy: 1.0,
            agency: 1.0,
        },
    };

    let error = PredictionError {
        intervention_id: intervention.id,
        predicted_comfort: intervention.predicted.comfort,
        observed_comfort: outcome.reading.comfort,
        comfort_error: outcome.reading.comfort - intervention.predicted.comfort,
        predicted_energy: intervention.predicted.energy,
        observed_energy: outcome.reading.energy_consumption,
        energy_error: outcome.reading.energy_consumption - intervention.predicted.energy,
    };

    let constraint_candidate = ConstraintCandidate {
        id: "SWA-003-C-001",
        mechanism: "Zone-level environmental control can improve comfort and energy without requiring a long-lived fixed commitment.",
        constraint: "For comparable workspace outcomes, evaluate reversible/modular interventions before commitments with materially higher exit or reconfiguration cost.",
        lineage_intervention: intervention.id,
        adopted: false,
    };

    // Keep the counterfactual dimensions explicit. The BuildingTwin result is
    // evidence about physical prediction behavior; it is not an authorization.
    assert!(baseline_output.free_energy.is_finite());
    assert!(adapted_output.free_energy.is_finite());
    assert!(intervention.predicted.privacy >= 0.0);
    assert!(intervention.predicted.agency >= 0.0);
    assert!(intervention.reversibility_cost_consistent());
    assert!(error.comfort_error.abs() < 0.02);
    assert!(error.energy_error.abs() < 0.02);
    assert!(!constraint_candidate.adopted);

    FixtureEvidence {
        fixture_id: "SWA-003-BUILDING-001",
        building: "building-001",
        observation,
        twin_health: state.health,
        twin_epistemic_uncertainty: state.epistemic_uncertainty,
        twin_aleatoric_uncertainty: state.aleatoric_uncertainty,
        building_free_energy: adapted_output.free_energy,
        building_action: format!("{:?}", adapted_output.recommended_action),
        intervention,
        outcome,
        prediction_error: error,
        constraint_candidate,
        authorization_required: true,
        actuation_performed: false,
    }
}

impl WorkspaceIntervention {
    fn reversibility_cost_consistent(&self) -> bool {
        self.reconfiguration_cost >= 0.0
            && self.exit_cost >= 0.0
            && self.alternative_use_value >= 0.0
            && self.downside_liquidity_exposure.is_finite()
            && (0.0..=1.0).contains(&self.downside_liquidity_exposure)
    }
}

fn main() {
    let first = fixture();
    let second = fixture();

    // Replay invariant: identical deterministic inputs produce identical
    // evidence, independent of who holds authority to authorize an action.
    assert_eq!(
        serde_json::to_string(&first).expect("fixture serializes"),
        serde_json::to_string(&second).expect("fixture serializes")
    );

    // Prediction is not observation; recommendation is not authorization.
    assert_ne!(
        first.intervention.predicted.comfort,
        first.outcome.reading.comfort
    );
    assert!(first.authorization_required);
    assert!(!first.actuation_performed);

    // Independent objectives remain inspectable rather than being collapsed
    // into a single opaque workspace score.
    assert!(first.intervention.predicted.resilience.is_finite());
    assert!(first.intervention.predicted.lifecycle_cost.is_finite());
    assert!(first.intervention.predicted.reversibility.is_finite());

    println!(
        "{}",
        serde_json::to_string_pretty(&first).expect("fixture serializes")
    );
}
