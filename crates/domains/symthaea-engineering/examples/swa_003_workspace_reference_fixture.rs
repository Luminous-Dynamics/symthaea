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
enum EvidenceKind {
    ScenarioAssumption,
    ModelDerivedPrediction,
    ObservedOutcome,
    DerivedResidual,
    UnresolvedEvidence,
    UnknownUnmodeled,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
struct EvidenceProvenance {
    kind: EvidenceKind,
    source: &'static str,
    lineage: &'static str,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
struct DimensionValue {
    value: Option<f64>,
    provenance: EvidenceProvenance,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
struct WorkspaceDimensions {
    comfort: DimensionValue,
    energy: DimensionValue,
    resilience: DimensionValue,
    lifecycle_cost: DimensionValue,
    maintenance_burden: DimensionValue,
    reversibility: DimensionValue,
    privacy: DimensionValue,
    agency: DimensionValue,
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
    prediction_kind: EvidenceKind,
    predicted_comfort: f64,
    observed_comfort: f64,
    comfort_error: f64,
    predicted_energy: f64,
    observed_energy: f64,
    energy_error: f64,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
struct CounterfactualEvaluation {
    intervention_id: &'static str,
    comfort: DimensionValue,
    energy: DimensionValue,
    resilience: DimensionValue,
    lifecycle_cost: DimensionValue,
    maintenance_burden: DimensionValue,
    reversibility: DimensionValue,
    comparable: bool,
    exclusion_reason: &'static str,
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
    counterfactual_evaluations: [CounterfactualEvaluation; 2],
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

    let scenario = EvidenceProvenance {
        kind: EvidenceKind::ScenarioAssumption,
        source: "SWA-003 authored fixture scenario",
        lineage: "fixture-input",
    };
    let unmodeled = EvidenceProvenance {
        kind: EvidenceKind::UnknownUnmodeled,
        source: "no intervention-specific physical model",
        lineage: "counterfactual-support-gap",
    };

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
                comfort: DimensionValue { value: Some(0.90), provenance: scenario },
                energy: DimensionValue { value: Some(0.30), provenance: scenario },
                resilience: DimensionValue { value: Some(0.86), provenance: scenario },
                lifecycle_cost: DimensionValue { value: Some(0.68), provenance: scenario },
                maintenance_burden: DimensionValue { value: Some(0.18), provenance: scenario },
                reversibility: DimensionValue { value: Some(0.91), provenance: scenario },
                privacy: DimensionValue { value: Some(1.0), provenance: scenario },
                agency: DimensionValue { value: Some(1.0), provenance: scenario },
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
                comfort: DimensionValue { value: Some(0.93), provenance: scenario },
                energy: DimensionValue { value: Some(0.27), provenance: scenario },
                resilience: DimensionValue { value: Some(0.80), provenance: scenario },
                lifecycle_cost: DimensionValue { value: Some(0.41), provenance: scenario },
                maintenance_burden: DimensionValue { value: Some(0.44), provenance: scenario },
                reversibility: DimensionValue { value: Some(0.24), provenance: scenario },
                privacy: DimensionValue { value: Some(1.0), provenance: scenario },
                agency: DimensionValue { value: Some(0.86), provenance: scenario },
            },
        },
    ];

    let prediction_errors = [
        PredictionError {
            intervention_id: interventions[0].id,
            zone_id: outcomes[0].zone_id,
            prediction_kind: interventions[0].predicted.comfort.provenance.kind,
            predicted_comfort: interventions[0].predicted.comfort.value.expect("scenario value"),
            observed_comfort: outcomes[0].reading.comfort,
            comfort_error: outcomes[0].reading.comfort
                - interventions[0].predicted.comfort.value.expect("scenario value"),
            predicted_energy: interventions[0].predicted.energy.value.expect("scenario value"),
            observed_energy: outcomes[0].reading.energy_consumption,
            energy_error: outcomes[0].reading.energy_consumption
                - interventions[0].predicted.energy.value.expect("scenario value"),
        },
        PredictionError {
            intervention_id: interventions[0].id,
            zone_id: outcomes[1].zone_id,
            prediction_kind: interventions[0].predicted.comfort.provenance.kind,
            predicted_comfort: interventions[0].predicted.comfort.value.expect("scenario value"),
            observed_comfort: outcomes[1].reading.comfort,
            comfort_error: outcomes[1].reading.comfort
                - interventions[0].predicted.comfort.value.expect("scenario value"),
            predicted_energy: interventions[0].predicted.energy.value.expect("scenario value"),
            observed_energy: outcomes[1].reading.energy_consumption,
            energy_error: outcomes[1].reading.energy_consumption
                - interventions[0].predicted.energy.value.expect("scenario value"),
        },
    ];

    let counterfactual_evaluations = interventions.map(|intervention| CounterfactualEvaluation {
        intervention_id: intervention.id,
        comfort: DimensionValue { value: None, provenance: unmodeled },
        energy: DimensionValue { value: None, provenance: unmodeled },
        resilience: DimensionValue { value: None, provenance: unmodeled },
        lifecycle_cost: DimensionValue { value: None, provenance: unmodeled },
        maintenance_burden: DimensionValue { value: None, provenance: unmodeled },
        reversibility: DimensionValue { value: None, provenance: unmodeled },
        comparable: false,
        exclusion_reason: "BuildingTwin currently supplies building behavior, not an intervention-specific workspace counterfactual for these dimensions",
    });

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

    // BuildingTwin output is evidence about physical twin behavior, not an
    // intervention counterfactual and never an authorization. Authored
    // intervention values remain scenario assumptions until an intervention-
    // specific model actually produces them.
    assert!(baseline_output.free_energy.is_finite());
    assert!(adapted_output.free_energy.is_finite());
    assert!(interventions.iter().all(WorkspaceIntervention::reversibility_cost_consistent));
    assert!(prediction_errors.iter().all(|e| e.comfort_error.is_finite()));
    assert!(prediction_errors.iter().all(|e| e.energy_error.is_finite()));
    assert!(interventions.iter().all(|i| {
        i.predicted.comfort.provenance.kind == EvidenceKind::ScenarioAssumption
            && i.predicted.energy.provenance.kind == EvidenceKind::ScenarioAssumption
    }));
    assert!(counterfactual_evaluations.iter().all(|e| {
        !e.comparable
            && e.comfort.value.is_none()
            && e.energy.value.is_none()
            && e.resilience.value.is_none()
    }));
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
        counterfactual_evaluations,
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
        first.interventions[0].predicted.comfort.value,
        Some(first.outcomes[0].reading.comfort)
    );
    assert!(first.authorization_required);
    assert!(!first.actuation_performed);

    // Contradiction is retained rather than silently resolved by attestation
    // count or by choosing one evidence source.
    assert!(!first.contradictory_evidence.resolved);

    // Independent objectives remain inspectable rather than being collapsed
    // into a single opaque workspace score.
    assert!(first.interventions[0].predicted.resilience.value.is_some());
    assert!(first.interventions[0].predicted.lifecycle_cost.value.is_some());
    assert!(first.interventions[0].predicted.reversibility.value.is_some());
    assert!(
        first.interventions[1].predicted.reversibility.value
            < first.interventions[0].predicted.reversibility.value
    );

    println!(
        "{}",
        serde_json::to_string_pretty(&first).expect("fixture serializes")
    );
}
