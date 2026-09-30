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
        internal_gain_kw: 0.35,
        setpoint_c: 21.0,
        comfort_band_c: 2.0,
    };
    let predictor_high = ZoneParameters {
        thermal_mass_kwh_per_c: 10.0,
        envelope_u_kw_per_c: 0.26,
        internal_gain_kw: 0.50,
        setpoint_c: 21.0,
        comfort_band_c: 2.0,
    };
    let world = ZoneParameters {
        thermal_mass_kwh_per_c: 9.0,
        envelope_u_kw_per_c: 0.23,
        internal_gain_kw: 0.42,
        setpoint_c: 21.0,
        comfort_band_c: 2.0,
    };
    let intervention = Intervention {
        id: "zone-a-reversible-hvac",
        hvac_capacity_kw: 2.0,
    };

    let prediction = predict(initial, predictor_low, predictor_high, intervention, 35.0, 8);
    let outcome = synthetic_world(initial, world, intervention, 35.0, 8);
    let residual = residual(prediction, outcome);

    let lineage = Lineage {
        building_observation: "building-observation-001",
        simulation_activity: "simulation-activity-swa-005-001",
        prediction: "prediction-swa-005-001",
        synthetic_outcome: "synthetic-outcome-swa-005-001",
        residual: "residual-swa-005-001",
    };

    // The synthetic outcome is deliberately not labeled as an observation.
    assert_ne!(outcome.source, "telemetry");
    assert!(residual.comfort_error.is_finite());
    assert!(residual.energy_error_kwh.is_finite());
    assert_eq!(residual.intervention_id, intervention.id);

    // The physical model emits evidence only. No authorization or actuation
    // state exists in this example.
    let serialized = serde_json::to_string(&(prediction, outcome, residual, lineage))
        .expect("lineage serializes");
    assert!(serialized.contains("swa-005-rc-v1"));

    println!("{serialized}");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (PredictionInterval, SyntheticOutcome, Residual) {
        let initial = ZoneState { indoor_c: 21.0 };
        let low = ZoneParameters {
            thermal_mass_kwh_per_c: 8.0,
            envelope_u_kw_per_c: 0.20,
            internal_gain_kw: 0.35,
            setpoint_c: 21.0,
            comfort_band_c: 2.0,
        };
        let high = ZoneParameters {
            thermal_mass_kwh_per_c: 10.0,
            envelope_u_kw_per_c: 0.26,
            internal_gain_kw: 0.50,
            setpoint_c: 21.0,
            comfort_band_c: 2.0,
        };
        let world = ZoneParameters {
            thermal_mass_kwh_per_c: 9.0,
            envelope_u_kw_per_c: 0.23,
            internal_gain_kw: 0.42,
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
    fn prediction_does_not_become_observation() {
        let (_, outcome, _) = fixture();
        assert_ne!(outcome.source, "telemetry");
    }
}
