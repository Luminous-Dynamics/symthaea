// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Reproducible discovery run for scalar waves in different physical
//! spatial dimensions. The module's Hamiltonian oracle is not fed into the
//! search. Passing the conservation gates is only candidate evidence; the
//! truth-correlation column is a heuristic and is not a proof of equivalence.
//!
//! Usage:
//! cargo run -p symthaea-physics-bridge --example higher_dimensional_wave_discovery -- 2
//! The optional integer is the maximum spatial dimension (1..=4).

use std::error::Error;
use std::env;

use symthaea_core::hdc::conjecture_engine::{
    RegressorConfig, discover_invariants_autonomous, is_informatively_conserved,
    lie_derivative_variance,
};
use symthaea_physics_bridge::{
    hypercubic_wave_energy, hypercubic_wave_rhs, hypercubic_wave_trajectory,
};

const CONSERVATION_TOLERANCE: f64 = 1e-6;
const ENERGY_ALIGNMENT_CORRELATION: f64 = 0.995;

fn initial_state(spatial_dimensions: usize, phase: f64, scale: f64) -> Vec<f64> {
    let sites = 1usize << spatial_dimensions;
    let mut state = vec![0.0; 2 * sites];
    for i in 0..sites {
        state[i] = scale * (((i + 1) as f64 * 0.43) + phase).sin();
        state[sites + i] = scale * 0.4 * (((i + 1) as f64 * 0.31) + phase * 0.7).cos();
    }
    state
}

fn correlation_with_hamiltonian(
    formula: &symthaea_core::hdc::conjecture_engine::Expr,
    trajectory: &[Vec<f64>],
    variable_names: &[&str],
) -> Option<f64> {
    let mut candidate_values = Vec::with_capacity(trajectory.len());
    let mut energy_values = Vec::with_capacity(trajectory.len());
    for state in trajectory {
        let assignments: Vec<(&str, f64)> = variable_names
            .iter()
            .copied()
            .zip(state.iter().copied())
            .collect();
        let candidate_value = formula.eval(&assignments);
        let energy = hypercubic_wave_energy(state);
        if !candidate_value.is_finite() || !energy.is_finite() {
            return None;
        }
        candidate_values.push(candidate_value);
        energy_values.push(energy);
    }
    pearson_correlation(&candidate_values, &energy_values)
}

fn pearson_correlation(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 2 {
        return None;
    }
    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().sum::<f64>() / right.len() as f64;
    let mut covariance = 0.0;
    let mut left_variance = 0.0;
    let mut right_variance = 0.0;
    for (x, y) in left.iter().zip(right) {
        let dx = x - left_mean;
        let dy = y - right_mean;
        covariance += dx * dy;
        left_variance += dx * dx;
        right_variance += dy * dy;
    }
    let denominator = (left_variance * right_variance).sqrt();
    (denominator.is_finite() && denominator > f64::MIN_POSITIVE)
        .then_some(covariance / denominator)
}

fn main() -> Result<(), Box<dyn Error>> {
    let max_dimension = env::args()
        .nth(1)
        .map(|arg| arg.parse::<usize>())
        .transpose()?
        .unwrap_or(2);
    if !(1..=4).contains(&max_dimension) {
        return Err("maximum spatial dimension must be in 1..=4".into());
    }

    println!("Higher-dimensional scalar-wave invariant discovery");
    println!("Gate: train and holdout Lie-derivative variance < {CONSERVATION_TOLERANCE}");
    println!("Energy alignment: |Pearson r| >= {ENERGY_ALIGNMENT_CORRELATION} on both trajectories");
    println!("These are screening gates, not a proof of formula identity or new physics.\n");

    for spatial_dimensions in 1..=max_dimension {
        let sites = 1usize << spatial_dimensions;
        let train_initial = initial_state(spatial_dimensions, 0.1, 0.7);
        let holdout_initial = initial_state(spatial_dimensions, 1.3, 1.4);
        let variable_storage: Vec<String> = (0..sites)
            .map(|i| format!("u{i}"))
            .chain((0..sites).map(|i| format!("v{i}")))
            .collect();
        let variable_names: Vec<&str> = variable_storage.iter().map(String::as_str).collect();

        let config = RegressorConfig {
            population_size: 120,
            generations: 50,
            max_depth: 4,
            max_complexity: 48,
            seed: 0xC0FFEE + spatial_dimensions as u64,
            ..RegressorConfig::for_autonomous_discovery()
        };

        let candidates = discover_invariants_autonomous(
            hypercubic_wave_rhs,
            &train_initial,
            &variable_names,
            None,
            &config,
            8.0,
            0.01,
        );
        let train_trajectory = hypercubic_wave_trajectory(&train_initial, 800, 0.01);
        let holdout_trajectory = hypercubic_wave_trajectory(&holdout_initial, 800, 0.01);

        println!(
            "d={spatial_dimensions} spatial dimensions; {} lattice sites; {} ODE state variables; {} candidates",
            sites,
            2 * sites,
            candidates.len()
        );
        let mut screened = 0usize;
        for candidate in candidates.iter().take(10) {
            let train_variance = lie_derivative_variance(
                &candidate.formula,
                hypercubic_wave_rhs,
                &train_trajectory,
                &variable_names,
            );
            let holdout_variance = lie_derivative_variance(
                &candidate.formula,
                hypercubic_wave_rhs,
                &holdout_trajectory,
                &variable_names,
            );
            let informative = is_informatively_conserved(
                &candidate.formula,
                hypercubic_wave_rhs,
                &train_trajectory,
                &variable_names,
                CONSERVATION_TOLERANCE,
            );
            let train_correlation = correlation_with_hamiltonian(
                &candidate.formula,
                &train_trajectory,
                &variable_names,
            );
            let holdout_correlation = correlation_with_hamiltonian(
                &candidate.formula,
                &holdout_trajectory,
                &variable_names,
            );
            let train_ok = train_variance.is_finite()
                && train_variance < CONSERVATION_TOLERANCE;
            let holdout_ok = holdout_variance.is_finite()
                && holdout_variance < CONSERVATION_TOLERANCE;
            let aligned = train_correlation
                .is_some_and(|r| r.abs() >= ENERGY_ALIGNMENT_CORRELATION)
                && holdout_correlation
                    .is_some_and(|r| r.abs() >= ENERGY_ALIGNMENT_CORRELATION);
            let accepted = train_ok && holdout_ok && informative && aligned;
            if accepted {
                screened += 1;
            }
            println!(
                "  {} {} | train={:.3e} holdout={:.3e} informative={} r_train={} r_holdout={} | {}",
                if accepted { "[SCREENED]" } else { "[candidate]" },
                candidate.formula_str,
                train_variance,
                holdout_variance,
                informative,
                format_correlation(train_correlation),
                format_correlation(holdout_correlation),
                if accepted { "passes screening only" } else { "not accepted by full gate" },
            );
        }
        println!(
            "  screened candidates: {screened}; acceptance requires independent-initial-condition conservation and energy alignment, not just low training variance.\n"
        );
    }
    Ok(())
}

fn format_correlation(value: Option<f64>) -> String {
    value
        .map(|correlation| format!("{correlation:.4}"))
        .unwrap_or_else(|| "n/a".to_string())
}
