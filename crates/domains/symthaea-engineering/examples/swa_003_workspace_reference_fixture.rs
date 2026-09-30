// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! SWA-003 deterministic building -> workspace evidence fixture.
//!
//! This is deliberately an adapter/reference fixture, not a new workspace
//! domain engine. It composes existing BuildingTwin and TwinState primitives
//! and keeps physical, economic, resilience, agency, privacy, and
//! reversibility dimensions separate.
//!
//! The fixture stops before physical authorization/actuation. The
//! ReceiptProjection is only the narrow boundary a future Mycelix adapter can
//! project into durable provenance; it does not grant authority.

use serde::Serialize;
use symthaea_digital_twin::{AssetClass, TwinState};
use symthaea_fabrication_kernel::building::{BuildingOutput, BuildingReading, BuildingTwin};

const BASELINE_A: BuildingReading = BuildingReading {
    thermal_load: 0.40,
    structural_stress: 0.10,
    occupancy: 0.60,
    comfort: 0.85,
    energy_consumption: 0.35,
};

const BASELINE_B: BuildingReading = BuildingReading {
    thermal_load: 0.55,
    structural_stress: 0.12,
    occupancy: 0.35,
    comfort: 0.78,
    energy_consumption: 0.44,
};

const ADAPTED_A: BuildingReading = BuildingReading {
    thermal_load: 0.32,
    structural_stress: 0.10,
    occupancy: 0.60,
    comfort: 0.91,
    energy_consumption: 0.29,
};

const ADAPTED_B: BuildingReading = BuildingReading {
    thermal_load: 0.45,
    structural_stress: 0.12,
    occupancy: 0.35,
    comfort: 0.84,
    energy_consumption: 0.37,
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
    zone_id: &'static str,
    predicted_comfort: f64,
    observed_comfort: f64,
    comfort_error: f64,
    predicted_energy: f64,
    observed_energy: f64,
    energy_error: f64,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
struct ContradictoryEvidence {
    subject: &'static str,
    evidence_a: &'static str,
    value_a: f64,
    evidence_b: &'static str,
    value_b: f64,
    resolved: bool,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
struct ConstraintCandidate {
    id: &'static str,
    mechanism: &'static str,
    constraint: &'static str,
    lineage_intervention: &'static str,
    adopted: bool,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
struct ReceiptProjection {
    receipt_family: &'static str,
    source_prediction_remains_prediction: bool,
    authority_granted: bool,
    contestable: bool,
}

#[derive(Debug, Serialize)]
struct FixtureEvidence {
    fixture_id: &'static str,
    building: &'static str,
    zones: [WorkspaceObservation; 2],
    aggregate_occupancy: f64,
    infrastructure_dependencies: [&'static str; 2],
    twin_health: f64,
    twin_epistemic_uncertainty: f64,
    twin_aleatoric_uncertainty: f64,
    building_free_energy: f64,
    building_action: String,
    interventions: [WorkspaceIntervention; 2],
    outcomes: [WorkspaceObservation; 2],
    prediction_errors: [PredictionError; 2],
    contradictory_evidence: ContradictoryEvidence,
    constraint_candidate: ConstraintCandidate,
    receipt_projection: ReceiptProjection,
    authorization_required: bool,
    actuation_performed: bool,
}

fn run_building_twin(reference: &BuildingReading, reading: &BuildingReading) -> BuildingOutput {
    let mut twin = BuildingTwin::new();
    twin.set_reference(reference);
    twin.step(reading, 3_600.0)
}

fn fixture() -> FixtureEvidence {
    let zones = [
        WorkspaceObservation {
            zone_id: "building-001/zone-a",
            reading: BASELINE_A,
        },
        WorkspaceObservation {
            zone_id: "building-001/zone-b",
            reading: BASELINE_B,
        },
    ];

    let outcomes = [
        WorkspaceObservation {
            zone_id: "building-001/zone-a",
            reading: ADAPTED_A,
        },
        WorkspaceObservation {
            zone_id: "building-001/zone-b",
            reading: ADAPTED_B,
        },
    ];

    let mut state = TwinState::new(
        "building-001",
        AssetClass::CivilStructure,
        "deterministic workspace fixture",
    );

    // Deliberately avoid wall-clock timestamps. The fixture's physical inputs
    // are fixed so replay can be byte-stable after canonical serialization.
    state.health = 0.97;
    state.epistemic_uncertainty = 0.20;
    state.aleatoric_uncertainty = 0.05;

    let baseline_output = run_building_twin(&BASELINE_A, &BASELINE_A);
    let adapted_output = run_building_twin(&BASELINE_A, &ADAPTED_A);

    let interventions = [
        WorkspaceIntervention {
            id: "workspace-intervention-modular-hvac",
            description: "Zone-level HVAC configuration with reversible controls",
            commitment_months: 3,
            fixed_cost: 1_200.0,
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
        },
        WorkspaceIntervention {
            id: "workspace-intervention-fixed-fitout",
            description: "Long-lived fixed HVAC fit-out with lower configurability",
            commitment_months: 120,
            fixed_cost: 48_000.0,
            variable_cost_per_month: 110.0,
            reconfiguration_cost: 18_000.0,
            exit_cost: 25_000.0,
            alternative_use_value: 8_000.0,
            downside_liquidity_exposure: 0.72,
            predicted: WorkspaceDimensions {
                comfort: 0.93,
                energy: 0.27,
                resilience: 0.80,
                lifecycle_cost: 0.41,
                maintenance_burden: 0.44,
                reversibility: 0.24,
                privacy: 1.0,
                agency: 0.86,
            },
        },
    ];

    let prediction_errors = [
        PredictionError {
            intervention_id: interventions[0].id,
            zone_id: outcomes[0].zone_id,
            predicted_comfort: interventions[0].predicted.comfort,
            observed_comfort: outcomes[0].reading.comfort,
            comfort_error: outcomes[0].reading.comfort - interventions[0].predicted.comfort,
            predicted_energy: interventions[0].predicted.energy,
            observed_energy: outcomes[0].reading.energy_consumption,
            energy_error: outcomes[0].reading.energy_consumption
                - interventions[0].predicted.energy,
        },
        PredictionError {
            intervention_id: interventions[0].id,
            zone_id: outcomes[1].zone_id,
            predicted_comfort: interventions[0].predicted.comfort,
            observed_comfort: outcomes[1].reading.comfort,
            comfort_error: outcomes[1].reading.comfort - interventions[0].predicted.comfort,
            predicted_energy: interventions[0].predicted.energy,
            observed_energy: outcomes[1].reading.energy_consumption,
            energy_error: outcomes[1].reading.energy_consumption
                - interventions[0].predicted.energy,
        },
    ];

    let contradictory_evidence = ContradictoryEvidence {
        subject: "zone-b comfort after adaptation",
        evidence_a: "fixture-outcome-telemetry",
        value_a: outcomes[1].reading.comfort,
        evidence_b: "independent-review-fixture",
        value_b: 0.81,
        resolved: false,
    };

    let constraint_candidate = ConstraintCandidate {
        id: "SWA-003-C-001",
        mechanism: "Reversible zone-level controls can improve comfort and energy without a long-lived fixed commitment.",
        constraint: "For comparable workspace outcomes, evaluate reversible/modular interventions before commitments with materially higher exit or reconfiguration cost.",
        lineage_intervention: interventions[0].id,
        adopted: false,
    };

    let receipt_projection = ReceiptProjection {
        receipt_family: "WorkspaceDecisionReceiptV1",
        source_prediction_remains_prediction: true,
        authority_granted: false,
        contestable: true,
    };

    // The BuildingTwin result is evidence about physical prediction behavior,
    // not an authorization. Both intervention paths remain explicit.
    assert!(baseline_output.free_energy.is_finite());
    assert!(adapted_output.free_energy.is_finite());
    assert!(interventions.iter().all(WorkspaceIntervention::reversibility_cost_consistent));
    assert!(prediction_errors.iter().all(|e| e.comfort_error.is_finite()));
    assert!(prediction_errors.iter().all(|e| e.energy_error.is_finite()));
    assert!(!contradictory_evidence.resolved);
    assert!(!constraint_candidate.adopted);
    assert!(!receipt_projection.authority_granted);
    assert!(receipt_projection.source_prediction_remains_prediction);

    FixtureEvidence {
        fixture_id: "SWA-003-BUILDING-001",
        building: "building-001",
        zones,
        aggregate_occupancy: (BASELINE_A.occupancy + BASELINE_B.occupancy) / 2.0,
        infrastructure_dependencies: ["grid-feed-001", "building-network-001"],
        twin_health: state.health,
        twin_epistemic_uncertainty: state.epistemic_uncertainty,
        twin_aleatoric_uncertainty: state.aleatoric_uncertainty,
        building_free_energy: adapted_output.free_energy,
        building_action: format!("{:?}", adapted_output.recommended_action),
        interventions,
        outcomes,
        prediction_errors,
        contradictory_evidence,
        constraint_candidate,
        receipt_projection,
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
        first.interventions[0].predicted.comfort,
        first.outcomes[0].reading.comfort
    );
    assert!(first.authorization_required);
    assert!(!first.actuation_performed);

    // Contradiction is retained rather than silently resolved by attestation
    // count or by choosing one evidence source.
    assert!(!first.contradictory_evidence.resolved);

    // Independent objectives remain inspectable rather than being collapsed
    // into a single opaque workspace score.
    assert!(first.interventions[0].predicted.resilience.is_finite());
    assert!(first.interventions[0].predicted.lifecycle_cost.is_finite());
    assert!(first.interventions[0].predicted.reversibility.is_finite());
    assert!(first.interventions[1].reversibility < first.interventions[0].reversibility);

    println!(
        "{}",
        serde_json::to_string_pretty(&first).expect("fixture serializes")
    );
}
