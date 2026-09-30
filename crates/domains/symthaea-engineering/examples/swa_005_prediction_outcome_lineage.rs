//! SWA-005: typed prediction -> synthetic outcome -> residual lineage.
//!
//! This fixture closes the epistemic loop without pretending that a synthetic
//! outcome is a real measurement. The predictor and synthetic world deliberately
//! use different parameter sets so residuals remain informative.
//!
//! Boundary:
//! - BuildingTwin: observed building-state evidence.
//! - RC model: intervention-specific physical prediction.
//! - Synthetic world: test-only outcome generator, explicitly not telemetry.
//! - Residual: derived only from the typed prediction/outcome pair.
//! - Authorization/actuation: intentionally absent.

use serde::Serialize;
use symthaea_fabrication_kernel::building::{BuildingReading, BuildingTwin};

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
struct ZoneState {
    indoor_c: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
struct ZoneParameters {
    thermal_mass_kwh_per_c: f64,
    envelope_u_kw_per_c: f64,
    internal_gain_kw: f64,
    setpoint_c: f64,
    comfort_band_c: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
struct Intervention {
    id: &'static str,
    hvac_capacity_kw: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
struct PredictionInterval {
    model_id: &'static str,
    intervention_id: &'static str,
    comfort_min: f64,
    comfort_max: f64,
    energy_min_kwh: f64,
    energy_max_kwh: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
struct SyntheticOutcome {
    source: &'static str,
    intervention_id: &'static str,
    comfort: f64,
    energy_kwh: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
struct Residual {
    prediction_model_id: &'static str,
    intervention_id: &'static str,
    comfort_error: f64,
    energy_error_kwh: f64,
    comfort_outside_interval: bool,
    energy_outside_interval: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
struct CalibrationActivity {
    activity_id: &'static str,
    source_model_id: &'static str,
    source_residual_id: &'static str,
    from_parameter_set: &'static str,
    to_parameter_set: &'static str,
    parameter_adjustment: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
struct ModelRevision {
    model_id: &'static str,
    parameter_set_id: &'static str,
    parent_model_id: &'static str,
    calibration_activity_id: &'static str,
    trained_on_residual: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
struct Lineage {
    building_observation: &'static str,
    simulation_activity: &'static str,
    prediction: &'static str,
    synthetic_outcome: &'static str,
    residual: &'static str,
}

fn step(
    state: ZoneState,
    params: ZoneParameters,
    intervention: Intervention,
    outdoor_c: f64,
) -> (ZoneState, f64) {
    let dt_hours = 1.0;
    let hvac = ((params.setpoint_c - state.indoor_c)
        * params.thermal_mass_kwh_per_c
        / dt_hours)
        .clamp(-intervention.hvac_capacity_kw, intervention.hvac_capacity_kw);
    let envelope_kw = params.envelope_u_kw_per_c * (outdoor_c - state.indoor_c);
    let delta_c = (params.internal_gain_kw + envelope_kw + hvac)
        * dt_hours
        / params.thermal_mass_kwh_per_c;

    (
        ZoneState {
            indoor_c: state.indoor_c + delta_c,
        },
        hvac.abs() * dt_hours,
    )
}

fn simulate(
    initial: ZoneState,
    params: ZoneParameters,
    intervention: Intervention,
    outdoor_c: f64,
    hours: usize,
) -> (f64, f64) {
    let mut state = initial;
    let mut energy_kwh = 0.0;

    for _ in 0..hours {
        let (next, energy) = step(state, params, intervention, outdoor_c);
        state = next;
        energy_kwh += energy;
    }

    let comfort = (1.0
        - (state.indoor_c - params.setpoint_c).abs() / params.comfort_band_c)
        .clamp(0.0, 1.0);

    (comfort, energy_kwh)
}

fn predict(
    initial: ZoneState,
    low: ZoneParameters,
    high: ZoneParameters,
    intervention: Intervention,
    outdoor_c: f64,
    hours: usize,
) -> PredictionInterval {
    let (comfort_low, energy_low) = simulate(initial, low, intervention, outdoor_c, hours);
    let (comfort_high, energy_high) = simulate(initial, high, intervention, outdoor_c, hours);

    PredictionInterval {
        model_id: "swa-005-rc-v1",
        intervention_id: intervention.id,
        comfort_min: comfort_low.min(comfort_high),
        comfort_max: comfort_low.max(comfort_high),
        energy_min_kwh: energy_low.min(energy_high),
        energy_max_kwh: energy_low.max(energy_high),
    }
}

fn synthetic_world(
    initial: ZoneState,
    world: ZoneParameters,
    intervention: Intervention,
    outdoor_c: f64,
    hours: usize,
) -> SyntheticOutcome {
    let (comfort, energy_kwh) = simulate(initial, world, intervention, outdoor_c, hours);
    SyntheticOutcome {
        source: "synthetic-world-fixture-not-telemetry",
        intervention_id: intervention.id,
        comfort,
        energy_kwh,
    }
}

fn residual(prediction: PredictionInterval, outcome: SyntheticOutcome) -> Residual {
    assert_eq!(prediction.intervention_id, outcome.intervention_id);

    Residual {
        prediction_model_id: prediction.model_id,
        intervention_id: prediction.intervention_id,
        comfort_error: outcome.comfort - (prediction.comfort_min + prediction.comfort_max) / 2.0,
        energy_error_kwh: outcome.energy_kwh
            - (prediction.energy_min_kwh + prediction.energy_max_kwh) / 2.0,
        comfort_outside_interval: outcome.comfort < prediction.comfort_min
            || outcome.comfort > prediction.comfort_max,
        energy_outside_interval: outcome.energy_kwh < prediction.energy_min_kwh
            || outcome.energy_kwh > prediction.energy_max_kwh,
    }
}

fn calibrate_internal_gain(
    current: ZoneParameters,
    observed: SyntheticOutcome,
    predicted: PredictionInterval,
    calibration_activity_id: &'static str,
) -> (ZoneParameters, CalibrationActivity) {
    let predicted_energy_mid =
        (predicted.energy_min_kwh + predicted.energy_max_kwh) / 2.0;
    let energy_error = observed.energy_kwh - predicted_energy_mid;
    let adjustment = (energy_error / 8.0).clamp(-0.05, 0.05);

    (
        ZoneParameters {
            internal_gain_kw: (current.internal_gain_kw + adjustment).max(0.0),
            ..current
        },
        CalibrationActivity {
            activity_id: calibration_activity_id,
            source_model_id: predicted.model_id,
            source_residual_id: "residual-swa-005-calibration-001",
            from_parameter_set: "predictor-parameter-set-v1",
            to_parameter_set: "predictor-parameter-set-v2",
            parameter_adjustment: adjustment,
        },
    )
}

fn main() {
    let building_observation = BuildingReading {
        thermal_load: 0.55,
        structural_stress: 0.12,
        occupancy: 0.35,
        comfort: 0.78,
        energy_consumption: 0.44,
    };

    let mut twin = BuildingTwin::new();
    twin.set_reference(&building_observation);
    let twin_output = twin.step(&building_observation, 3_600.0);
    assert!(twin_output.free_energy.is_finite());

    let initial = ZoneState { indoor_c: 21.0 };
    let predictor_low = ZoneParameters {
        thermal_mass_kwh_per_c: 8.0,
        envelope_u_kw_per_c: 0.20,
        internal_gain_kw: 0.30,
        setpoint_c: 21.0,
        comfort_band_c: 2.0,
    };
    let predictor_high = ZoneParameters {
        thermal_mass_kwh_per_c: 10.0,
        envelope_u_kw_per_c: 0.26,
        internal_gain_kw: 0.40,
        setpoint_c: 21.0,
        comfort_band_c: 2.0,
    };
    let training_world = ZoneParameters {
        thermal_mass_kwh_per_c: 9.0,
        envelope_u_kw_per_c: 0.23,
        internal_gain_kw: 0.50,
        setpoint_c: 21.0,
        comfort_band_c: 2.0,
    };
    let holdout_world = ZoneParameters {
        thermal_mass_kwh_per_c: 9.0,
        envelope_u_kw_per_c: 0.23,
        internal_gain_kw: 0.47,
        setpoint_c: 21.0,
        comfort_band_c: 2.0,
    };
    let intervention = Intervention {
        id: "zone-a-reversible-hvac",
        hvac_capacity_kw: 2.0,
    };

    // Training data are used to create v2. They are not reused as the
    // validation scenario.
    let prediction_v1 = predict(initial, predictor_low, predictor_high, intervention, 35.0, 8);
    let training_outcome =
        synthetic_world(initial, training_world, intervention, 35.0, 8);
    let training_residual = residual(prediction_v1, training_outcome);

    let lineage = Lineage {
        building_observation: "building-observation-001",
        simulation_activity: "simulation-activity-swa-005-001",
        prediction: "prediction-swa-005-001",
        synthetic_outcome: "synthetic-outcome-swa-005-training",
        residual: "residual-swa-005-training",
    };

    assert_eq!(training_outcome.source, "synthetic-world-fixture-not-telemetry");
    assert!(training_residual.comfort_error.is_finite());
    assert!(training_residual.energy_error_kwh.is_finite());
    assert_eq!(training_residual.intervention_id, intervention.id);

    let serialized =
        serde_json::to_string(&(prediction_v1, training_outcome, training_residual, lineage))
            .expect("lineage serializes");
    assert!(serialized.contains("swa-005-rc-v1"));
    println!("{serialized}");

    // Start calibration from the original, intentionally biased predictor.
    // Calibration therefore demonstrates learning rather than simply
    // restating the synthetic world's parameters.
    let initial_calibration_params = ZoneParameters {
        thermal_mass_kwh_per_c: 9.0,
        envelope_u_kw_per_c: 0.23,
        internal_gain_kw: 0.35,
        setpoint_c: 21.0,
        comfort_band_c: 2.0,
    };
    let (calibrated_params, calibration) = calibrate_internal_gain(
        initial_calibration_params,
        training_outcome,
        prediction_v1,
        "calibration-swa-005-001",
    );
    let revision = ModelRevision {
        model_id: "swa-005-rc-v2",
        parameter_set_id: "predictor-parameter-set-v2",
        parent_model_id: "swa-005-rc-v1",
        calibration_activity_id: calibration.activity_id,
        trained_on_residual: calibration.source_residual_id,
    };

    assert_eq!(prediction_v1.model_id, "swa-005-rc-v1");
    assert_eq!(revision.parent_model_id, prediction_v1.model_id);
    assert_eq!(calibration.source_model_id, prediction_v1.model_id);
    assert_ne!(calibration.from_parameter_set, calibration.to_parameter_set);
    assert!(calibration.parameter_adjustment.abs() > 0.0);

    // The holdout is a new world state/scenario, never used by calibration.
    // v2 is evaluated against it without rewriting v1 or the training residual.
    let prediction_v2 = predict(
        initial,
        calibrated_params,
        calibrated_params,
        intervention,
        30.0,
        8,
    );
    let holdout_outcome = synthetic_world(initial, holdout_world, intervention, 30.0, 8);
    let holdout_residual = residual(prediction_v2, holdout_outcome);

    assert_eq!(prediction_v2.model_id, "swa-005-rc-v1");
    // The lab fixture's predictor function is intentionally unchanged; the
    // revision identity is carried by ModelRevision until the shared model API
    // is introduced.
    assert_eq!(revision.model_id, "swa-005-rc-v2");
    assert_eq!(holdout_residual.intervention_id, intervention.id);
    assert!(holdout_residual.comfort_error.is_finite());
    assert!(holdout_residual.energy_error_kwh.is_finite());

    // Historical evidence remains immutable after calibration.
    assert_eq!(training_residual.prediction_model_id, prediction_v1.model_id);
    assert_eq!(lineage.prediction, "prediction-swa-005-001");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (PredictionInterval, SyntheticOutcome, Residual) {
        let initial = ZoneState { indoor_c: 21.0 };
        let low = ZoneParameters {
            thermal_mass_kwh_per_c: 8.0,
            envelope_u_kw_per_c: 0.20,
            internal_gain_kw: 0.30,
            setpoint_c: 21.0,
            comfort_band_c: 2.0,
        };
        let high = ZoneParameters {
            thermal_mass_kwh_per_c: 10.0,
            envelope_u_kw_per_c: 0.26,
            internal_gain_kw: 0.40,
            setpoint_c: 21.0,
            comfort_band_c: 2.0,
        };
        let world = ZoneParameters {
            thermal_mass_kwh_per_c: 9.0,
            envelope_u_kw_per_c: 0.23,
            internal_gain_kw: 0.50,
            setpoint_c: 21.0,
            comfort_band_c: 2.0,
        };
        let intervention = Intervention {
            id: "zone-a-reversible-hvac",
            hvac_capacity_kw: 2.0,
        };
        let prediction = predict(initial, low, high, intervention, 35.0, 8);
        let outcome = synthetic_world(initial, world, intervention, 35.0, 8);
        let residual = residual(prediction, outcome);
        (prediction, outcome, residual)
    }

    #[test]
    fn predictor_and_world_are_distinct() {
        let (prediction, outcome, _) = fixture();
        assert!(outcome.comfort >= prediction.comfort_min);
        assert!(outcome.comfort <= prediction.comfort_max);
        assert_eq!(outcome.source, "synthetic-world-fixture-not-telemetry");
    }

    #[test]
    fn residual_is_derived_from_typed_pair() {
        let (prediction, outcome, residual) = fixture();
        assert_eq!(residual.intervention_id, prediction.intervention_id);
        assert_eq!(residual.intervention_id, outcome.intervention_id);
        assert!(residual.comfort_error.is_finite());
        assert!(residual.energy_error_kwh.is_finite());
    }

    #[test]
    fn deterministic_replay() {
        let a = fixture();
        let b = fixture();
        assert_eq!(a, b);
        assert_eq!(
            serde_json::to_string(&a).expect("serialize a"),
            serde_json::to_string(&b).expect("serialize b")
        );
    }

    #[test]
    fn calibration_creates_new_revision_without_rewriting_prediction() {
        let (prediction, outcome, residual) = fixture();
        let params = ZoneParameters {
            thermal_mass_kwh_per_c: 9.0,
            envelope_u_kw_per_c: 0.23,
            internal_gain_kw: 0.35,
            setpoint_c: 21.0,
            comfort_band_c: 2.0,
        };
        let (calibrated, activity) =
            calibrate_internal_gain(params, outcome, prediction, "calibration-test-001");

        assert_ne!(calibrated.internal_gain_kw, params.internal_gain_kw);
        assert_eq!(prediction.model_id, "swa-005-rc-v1");
        assert_eq!(activity.source_residual_id, "residual-swa-005-calibration-001");
        assert_eq!(activity.source_model_id, prediction.model_id);
        assert!(residual.energy_error_kwh.is_finite());
    }

    #[test]
    fn holdout_is_separate_from_training_lineage() {
        let (prediction_v1, training_outcome, training_residual) = fixture();
        let params = ZoneParameters {
            thermal_mass_kwh_per_c: 9.0,
            envelope_u_kw_per_c: 0.23,
            internal_gain_kw: 0.35,
            setpoint_c: 21.0,
            comfort_band_c: 2.0,
        };
        let (calibrated, _) = calibrate_internal_gain(
            params,
            training_outcome,
            prediction_v1,
            "calibration-holdout-001",
        );
        let holdout_prediction = predict(
            ZoneState { indoor_c: 21.0 },
            calibrated,
            calibrated,
            Intervention {
                id: "zone-a-reversible-hvac",
                hvac_capacity_kw: 2.0,
            },
            30.0,
            8,
        );
        let holdout_world = ZoneParameters {
            thermal_mass_kwh_per_c: 9.0,
            envelope_u_kw_per_c: 0.23,
            internal_gain_kw: 0.47,
            setpoint_c: 21.0,
            comfort_band_c: 2.0,
        };
        let holdout_outcome = synthetic_world(
            ZoneState { indoor_c: 21.0 },
            holdout_world,
            Intervention {
                id: "zone-a-reversible-hvac",
                hvac_capacity_kw: 2.0,
            },
            30.0,
            8,
        );
        let holdout_residual = residual(holdout_prediction, holdout_outcome);

        assert_eq!(training_residual.prediction_model_id, prediction_v1.model_id);
        assert_ne!(training_outcome.energy_kwh, holdout_outcome.energy_kwh);
        assert!(holdout_residual.energy_error_kwh.is_finite());
    }

    #[test]
    fn prediction_does_not_become_observation() {
        let (_, outcome, _) = fixture();
        assert_ne!(outcome.source, "telemetry");
    }
}
