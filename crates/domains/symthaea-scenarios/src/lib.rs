// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-scenarios
//!
//! Worked **cross-domain compositions** — proof that Symthaea's standalone
//! domain crates combine into coherent models, not just isolated libraries.
//! Each scenario draws on several crates at once.
//!
//! - [`epidemic_cost`] composes **epidemiology** (SIR) + **economics** (NPV of
//!   lost output) + **geodesy** (inter-city distance).
//! - [`controlled_epidemic`] composes **control theory** (a PID intervention)
//!   + **epidemiology** — and demonstrates the controller flattens the curve.
//!
//! Pure `std` transitively (all deps are pure-std domain crates).

use symthaea_control_theory::Pid;
use symthaea_economics::finance::npv;
use symthaea_epidemiology::Sir;
use symthaea_epidemiology::sir::State;
use symthaea_geodesy::haversine_distance;

/// Result of the epidemic-cost composition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpidemicCost {
    pub r0: f64,
    pub peak_infected: f64,
    pub final_size: f64,
    pub distance_km: f64,
    /// Present value of lost economic output over the recovery period.
    pub npv_loss: f64,
}

/// Model the cost of an epidemic between two cities: run the SIR dynamics, value
/// the lost workforce as discounted GDP, and note the inter-city distance.
#[allow(clippy::too_many_arguments)]
pub fn epidemic_cost(
    beta: f64,
    gamma: f64,
    population: f64,
    gdp_per_capita_per_year: f64,
    discount_rate: f64,
    recovery_years: usize,
    city_a: (f64, f64),
    city_b: (f64, f64),
) -> EpidemicCost {
    let sir = Sir { beta, gamma };
    let start = State {
        s: 0.999,
        i: 0.001,
        r: 0.0,
    };
    let (_, peak) = sir.simulate(start, 0.1, 4000);

    // Lost output ≈ a fraction of annual GDP equal to the peak workforce out,
    // paid each recovery year, discounted to present value.
    let annual_loss = gdp_per_capita_per_year * population * peak;
    let mut flows = vec![0.0]; // t=0
    flows.extend(std::iter::repeat_n(annual_loss, recovery_years));
    let npv_loss = npv(discount_rate, &flows);

    EpidemicCost {
        r0: sir.basic_reproduction_number(),
        peak_infected: peak,
        final_size: sir.final_size(),
        distance_km: haversine_distance(city_a.0, city_a.1, city_b.0, city_b.1),
        npv_loss,
    }
}

/// Compose a PID controller with SIR dynamics: each step the controller sees the
/// current infected fraction, and when it exceeds `target` it applies an
/// intervention that reduces the effective transmission rate. Returns
/// `(peak_without_control, peak_with_control)`.
pub fn controlled_epidemic(beta: f64, gamma: f64, target: f64) -> (f64, f64) {
    let start = State {
        s: 0.999,
        i: 0.001,
        r: 0.0,
    };
    let (dt, steps) = (0.1, 4000);

    let (_, uncontrolled_peak) = Sir { beta, gamma }.simulate(start, dt, steps);

    let mut pid = Pid::new(12.0, 0.0, 0.0); // proportional control
    let mut state = start;
    let mut peak = state.i;
    for _ in 0..steps {
        let error = state.i - target; // positive when over the cap
        let intervention = pid.update(error, dt).clamp(0.0, 0.95);
        let effective = Sir {
            beta: beta * (1.0 - intervention),
            gamma,
        };
        state = effective.step(state, dt);
        if state.i > peak {
            peak = state.i;
        }
    }
    (uncontrolled_peak, peak)
}

#[cfg(test)]
mod composition_tests {
    use super::*;

    #[test]
    fn epidemic_cost_is_coherent() {
        // London and Paris, a moderate epidemic.
        let c = epidemic_cost(
            0.3,
            0.1,
            1_000_000.0,
            50_000.0,
            0.05,
            5,
            (51.5074, -0.1278),
            (48.8566, 2.3522),
        );
        assert!((c.r0 - 3.0).abs() < 1e-9);
        assert!(c.peak_infected > 0.0 && c.peak_infected < 1.0);
        assert!(c.final_size > 0.9); // R0=3 → ~94% eventually infected
        assert!((c.distance_km - 343.5).abs() < 2.0); // geodesy agrees
        assert!(c.npv_loss > 0.0);
    }

    #[test]
    fn worse_epidemic_costs_more() {
        let mild = epidemic_cost(0.2, 0.1, 1e6, 5e4, 0.05, 5, (0.0, 0.0), (0.0, 1.0));
        let severe = epidemic_cost(0.5, 0.1, 1e6, 5e4, 0.05, 5, (0.0, 0.0), (0.0, 1.0));
        assert!(severe.peak_infected > mild.peak_infected);
        assert!(severe.npv_loss > mild.npv_loss);
    }

    #[test]
    fn control_flattens_the_curve() {
        // The PID intervention should produce a lower peak than no control.
        let (uncontrolled, controlled) = controlled_epidemic(0.4, 0.1, 0.03);
        assert!(
            controlled < uncontrolled,
            "control should reduce the peak: {controlled} vs {uncontrolled}"
        );
    }
}
