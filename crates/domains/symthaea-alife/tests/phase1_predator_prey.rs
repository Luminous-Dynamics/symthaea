// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phase 1b ground-truth test, per `ALIFE_PLAN_2026-07-08.md` §1b.
//!
//! The claim, checked qualitatively as the plan explicitly allows ("agent-based emergence won't
//! reproduce the ODE exactly — say so explicitly rather than overclaiming precision"):
//!
//! 1. Both populations actually oscillate (this isn't a flat equilibrium or a
//!    predator/prey extinction), and
//! 2. Predator peaks lag prey peaks — cross-correlating the two time series should find its
//!    strongest match at a *positive* lag (predators respond to, not lead, prey), which is the
//!    structural signature Lotka-Volterra predicts and a coincident- or negative-lag relationship
//!    would falsify.
//!
//! No claim is made about matching Lotka-Volterra's exact period or amplitude.

use symthaea_alife::{OrganismConfig, PopulationConfig, PredatorPreyConfig, PredatorPreySim};

/// See `phase1_logistic_growth.rs`'s module docs: `OrganismConfig::default()`'s
/// `forage_efficiency: 0.15` leaves populations on a knife-edge given the real (~50%, not
/// ~100%) equilibrium forage rate, found via a traced diagnostic run. Both species use the same
/// bumped efficiency Phase 1a settled on, for the same reason.
fn sustainable_organism_cfg() -> OrganismConfig {
    OrganismConfig {
        forage_efficiency: 0.6,
        ..OrganismConfig::default()
    }
}

fn scenario_config() -> PredatorPreyConfig {
    PredatorPreyConfig {
        prey_cfg: PopulationConfig {
            death_energy_threshold: 0.05,
            reproduction_energy_threshold: 0.8,
            reproduction_energy_cost: 0.4,
            organism_cfg: sustainable_organism_cfg(),
            ..Default::default()
        },
        predator_cfg: PopulationConfig {
            death_energy_threshold: 0.05,
            reproduction_energy_threshold: 0.8,
            reproduction_energy_cost: 0.4,
            organism_cfg: sustainable_organism_cfg(),
            ..Default::default()
        },
        // Generous headroom above what predation pressure will remove, so prey has room to
        // recover between predator-driven crashes instead of being permanently suppressed.
        plant_resource_total: 3.0,
        predation_scale: 0.05,
        predation_efficiency: 0.05,
    }
}

#[test]
fn predator_and_prey_populations_oscillate_with_predator_lagging_prey() {
    let mut sim = PredatorPreySim::new(scenario_config(), 10, 3, 11);

    const TICKS: usize = 8000;
    let mut prey_series = Vec::with_capacity(TICKS);
    let mut predator_series = Vec::with_capacity(TICKS);
    for _ in 0..TICKS {
        let step = sim.step();
        prey_series.push(step.prey_count as f64);
        predator_series.push(step.predator_count as f64);
    }

    // Drop an initial transient so early-run settling doesn't dominate the oscillation check.
    const BURN_IN: usize = 1000;
    let prey = &prey_series[BURN_IN..];
    let predator = &predator_series[BURN_IN..];

    let prey_cv = coefficient_of_variation(prey);
    let predator_cv = coefficient_of_variation(predator);
    // Prey's threshold is lower than predator's -- measured, not assumed. A traced run gave
    // prey_cv=0.109, predator_cv=0.198: prey is buffered by a large, steady plant pool (its own
    // standalone carrying capacity is far above what predation pressure removes here), while
    // predator's entire income depends on a fluctuating prey supply, so its relative variance is
    // structurally larger. Both thresholds sit with real margin below what was actually observed.
    assert!(
        prey_cv > 0.08,
        "prey population should show real variation, not settle flat: cv={prey_cv:.3}"
    );
    assert!(
        predator_cv > 0.15,
        "predator population should show real variation, not settle flat: cv={predator_cv:.3}"
    );

    // Cross-correlate: for each candidate lag L, correlate prey[t] against predator[t+L] over
    // the overlapping range. Lotka-Volterra predicts the best match at a positive L (predator
    // follows prey), not at L<=0.
    const MAX_LAG: i64 = 400;
    let mut best_lag = 0i64;
    let mut best_corr = f64::NEG_INFINITY;
    for lag in -MAX_LAG..=MAX_LAG {
        let corr = lagged_correlation(prey, predator, lag);
        if let Some(c) = corr
            && c > best_corr
        {
            best_corr = c;
            best_lag = lag;
        }
    }

    assert!(
        best_lag > 0,
        "predator should lag prey (positive best-correlation lag), got best_lag={best_lag} \
         (corr={best_corr:.3}) -- predators are leading or exactly coincident with prey, which \
         Lotka-Volterra-style coupling does not predict"
    );
    assert!(
        best_corr > 0.3,
        "the lagged relationship should be a real, non-trivial correlation, not noise: \
         best_lag={best_lag}, best_corr={best_corr:.3}"
    );
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn coefficient_of_variation(xs: &[f64]) -> f64 {
    let m = mean(xs);
    if m <= 0.0 {
        return 0.0;
    }
    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64;
    var.sqrt() / m
}

/// Pearson correlation between `a[t]` and `b[t + lag]` over the overlapping range. `None` if the
/// overlap is too small or either series has zero variance in that window.
fn lagged_correlation(a: &[f64], b: &[f64], lag: i64) -> Option<f64> {
    let n = a.len() as i64;
    let (start, end) = if lag >= 0 { (0, n - lag) } else { (-lag, n) };
    if end - start < 100 {
        return None;
    }
    let a_slice: Vec<f64> = (start..end).map(|t| a[t as usize]).collect();
    let b_slice: Vec<f64> = (start..end).map(|t| b[(t + lag) as usize]).collect();

    let a_mean = mean(&a_slice);
    let b_mean = mean(&b_slice);
    let mut cov = 0.0;
    let mut a_var = 0.0;
    let mut b_var = 0.0;
    for i in 0..a_slice.len() {
        let da = a_slice[i] - a_mean;
        let db = b_slice[i] - b_mean;
        cov += da * db;
        a_var += da * da;
        b_var += db * db;
    }
    if a_var <= 0.0 || b_var <= 0.0 {
        return None;
    }
    Some(cov / (a_var.sqrt() * b_var.sqrt()))
}
