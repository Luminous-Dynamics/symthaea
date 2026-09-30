//! SWA-004: intervention-specific thermal counterfactual.
//!
//! This deliberately complements, rather than replaces, BuildingTwin.
//! BuildingTwin supplies building-state evidence. This small deterministic
//! resistance/capacitance model supplies the intervention-specific physics
//! needed before an intervention may be called a model-derived prediction.

#[derive(Debug, Clone, Copy, PartialEq)]
struct ZoneState {
    indoor_c: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ZoneParameters {
    thermal_mass_kwh_per_c: f64,
    envelope_u_kw_per_c: f64,
    internal_gain_kw: f64,
    setpoint_c: f64,
    comfort_band_c: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Intervention {
    hvac_capacity_kw: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct PredictionInterval {
    comfort_min: f64,
    comfort_max: f64,
    energy_min: f64,
    energy_max: f64,
}

fn step(
    state: ZoneState,
    params: ZoneParameters,
    intervention: Intervention,
    outdoor_c: f64,
    dt_hours: f64,
) -> ZoneState {
    assert!(dt_hours > 0.0 && dt_hours.is_finite());
    assert!(params.thermal_mass_kwh_per_c.is_finite() && params.thermal_mass_kwh_per_c > 0.0);
    assert!(params.envelope_u_kw_per_c.is_finite() && params.envelope_u_kw_per_c >= 0.0);
    assert!(params.internal_gain_kw.is_finite());
    assert!(params.setpoint_c.is_finite() && params.comfort_band_c.is_finite() && params.comfort_band_c > 0.0);
    assert!(intervention.hvac_capacity_kw.is_finite() && intervention.hvac_capacity_kw >= 0.0);
    assert!(state.indoor_c.is_finite() && outdoor_c.is_finite());
    let hvac = ((params.setpoint_c - state.indoor_c)
        * params.thermal_mass_kwh_per_c
        / dt_hours)
        .clamp(-intervention.hvac_capacity_kw, intervention.hvac_capacity_kw);
    let envelope_kw = params.envelope_u_kw_per_c * (outdoor_c - state.indoor_c);
    let delta_c = (params.internal_gain_kw + envelope_kw + hvac)
        * dt_hours
        / params.thermal_mass_kwh_per_c;
    ZoneState {
        indoor_c: state.indoor_c + delta_c,
    }
}

fn comfort(temp_c: f64, setpoint_c: f64, band_c: f64) -> f64 {
    assert!(temp_c.is_finite() && setpoint_c.is_finite());
    assert!(band_c.is_finite() && band_c > 0.0);
    (1.0 - (temp_c - setpoint_c).abs() / band_c).clamp(0.0, 1.0)
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
        let before = state;
        state = step(state, params, intervention, outdoor_c, 1.0);
        let hvac_kw = ((params.setpoint_c - before.indoor_c)
            * params.thermal_mass_kwh_per_c)
            .clamp(-intervention.hvac_capacity_kw, intervention.hvac_capacity_kw);
        energy_kwh += hvac_kw.abs();
    }
    (
        comfort(state.indoor_c, params.setpoint_c, params.comfort_band_c),
        energy_kwh,
    )
}

fn counterfactual_interval(
    initial: ZoneState,
    low: ZoneParameters,
    high: ZoneParameters,
    intervention: Intervention,
    outdoor_c: f64,
    hours: usize,
) -> PredictionInterval {
    let (comfort_low, energy_low) =
        simulate(initial, low, intervention, outdoor_c, hours);
    let (comfort_high, energy_high) =
        simulate(initial, high, intervention, outdoor_c, hours);
    PredictionInterval {
        comfort_min: comfort_low.min(comfort_high),
        comfort_max: comfort_low.max(comfort_high),
        energy_min: energy_low.min(energy_high),
        energy_max: energy_low.max(energy_high),
    }
}

fn main() {
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

    let reversible = Intervention { hvac_capacity_kw: 2.0 };
    let committed = Intervention { hvac_capacity_kw: 4.0 };

    let reversible_prediction =
        counterfactual_interval(initial, low, high, reversible, 35.0, 8);
    let committed_prediction =
        counterfactual_interval(initial, low, high, committed, 35.0, 8);

    println!("reversible={reversible_prediction:?}");
    println!("committed={committed_prediction:?}");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parameters() -> (ZoneParameters, ZoneParameters) {
        (
            ZoneParameters {
                thermal_mass_kwh_per_c: 8.0,
                envelope_u_kw_per_c: 0.20,
                internal_gain_kw: 0.35,
                hvac_max_kw: 2.0,
                setpoint_c: 21.0,
                comfort_band_c: 2.0,
            },
            ZoneParameters {
                thermal_mass_kwh_per_c: 10.0,
                envelope_u_kw_per_c: 0.26,
                internal_gain_kw: 0.50,
                hvac_max_kw: 2.0,
                setpoint_c: 21.0,
                comfort_band_c: 2.0,
            },
        )
    }

    #[test]
    fn intervention_changes_model_output() {
        let (low, high) = parameters();
        let initial = ZoneState { indoor_c: 21.0 };
        let low_capacity = counterfactual_interval(
            initial,
            low,
            high,
            Intervention { hvac_capacity_kw: 2.0 },
            35.0,
            8,
        );
        let high_capacity = counterfactual_interval(
            initial,
            low,
            high,
            Intervention { hvac_capacity_kw: 4.0 },
            35.0,
            8,
        );

        assert!(high_capacity.comfort_min >= low_capacity.comfort_min);
        assert!(high_capacity.energy_max >= low_capacity.energy_max);
    }

    #[test]
    fn uncertainty_is_preserved() {
        let (low, high) = parameters();
        let result = counterfactual_interval(
            ZoneState { indoor_c: 21.0 },
            low,
            high,
            Intervention { hvac_capacity_kw: 2.0 },
            35.0,
            8,
        );

        assert!(result.comfort_min <= result.comfort_max);
        assert!(result.energy_min <= result.energy_max);
        assert!(result.comfort_min.is_finite());
        assert!(result.energy_max.is_finite());
    }

    #[test]
    fn deterministic_replay() {
        let (low, high) = parameters();
        let a = counterfactual_interval(
            ZoneState { indoor_c: 21.0 },
            low,
            high,
            Intervention { hvac_capacity_kw: 2.0 },
            35.0,
            8,
        );
        let b = counterfactual_interval(
            ZoneState { indoor_c: 21.0 },
            low,
            high,
            Intervention { hvac_capacity_kw: 2.0 },
            35.0,
            8,
        );
        assert_eq!(a, b);
    }

    #[test]
    fn model_is_not_authorization() {
        let (low, high) = parameters();
        let prediction = counterfactual_interval(
            ZoneState { indoor_c: 21.0 },
            low,
            high,
            Intervention { hvac_capacity_kw: 4.0 },
            35.0,
            8,
        );

        // The model produces evidence only. There is intentionally no
        // authorization or actuation field in this model.
        assert!(prediction.comfort_min >= 0.0);
        assert!(prediction.comfort_max <= 1.0);
    }
}
