// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic synthetic parameter-recovery sweep.
//!
//! Emits one CSV row per independent synthetic campaign. These are design
//! checks, not calibrated power, coverage, or empirical-performance claims.

use spectra_uvsd_fit::{fit, CategoryCount, CategoryTable, FitOptions};
use std::f64::consts::TAU;

const MU_GRID: [f64; 3] = [0.5, 1.0, 1.5];
const SIGMA_GRID: [f64; 4] = [0.70, 1.0, 1.30, 1.60];
const CUTS: [f64; 5] = [-1.5, -0.75, 0.0, 0.75, 1.5];
const SEEDS: u64 = 8;
const TRIALS_PER_CLASS: usize = 4_000;

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn uniform_open(&mut self) -> f64 {
        let mantissa = self.next_u64() >> 11;
        (mantissa as f64 + 0.5) / 9_007_199_254_740_992.0
    }

    fn standard_normal(&mut self) -> f64 {
        let radius = (-2.0 * self.uniform_open().ln()).sqrt();
        let angle = TAU * self.uniform_open();
        radius * angle.cos()
    }
}

fn category_for(value: f64) -> usize {
    for (index, threshold) in CUTS.iter().enumerate() {
        if value <= *threshold {
            return index;
        }
    }
    CUTS.len()
}

fn simulate_table(mu: f64, sigma: f64, seed: u64) -> CategoryTable {
    let mut rng = SplitMix64(seed);
    let mut rows: Vec<CategoryCount> = (0..=CUTS.len())
        .map(|index| CategoryCount {
            order: index as i16,
            signal: 0,
            noise: 0,
        })
        .collect();

    for _ in 0..TRIALS_PER_CLASS {
        let noise_evidence = rng.standard_normal();
        let signal_evidence = mu + sigma * rng.standard_normal();
        rows[category_for(noise_evidence)].noise += 1;
        rows[category_for(signal_evidence)].signal += 1;
    }
    CategoryTable::new(rows).expect("both classes and ordered categories are present")
}

fn main() {
    println!(
        "seed,mu_true,sigma_true,status,mu_hat,sigma_hat,d_a_hat,nll,gradient_norm,iterations,objective_spread,parameter_spread,diagonal_curvature"
    );

    for (mu_index, &mu) in MU_GRID.iter().enumerate() {
        for (sigma_index, &sigma) in SIGMA_GRID.iter().enumerate() {
            for seed in 0..SEEDS {
                let cell_index = mu_index as u64 * SIGMA_GRID.len() as u64 + sigma_index as u64;
                let campaign_seed = 0x5350_4543_5452_4100u64
                    .wrapping_add(cell_index * SEEDS)
                    .wrapping_add(seed);
                let table = simulate_table(mu, sigma, campaign_seed);
                let result = fit(&table, FitOptions::default());
                let (mu_hat, sigma_hat, d_a_hat) = match result.estimate {
                    Some(estimate) => (
                        estimate.mu.to_string(),
                        estimate.sigma.to_string(),
                        estimate.d_a.to_string(),
                    ),
                    None => (String::new(), String::new(), String::new()),
                };
                let nll = result
                    .diagnostics
                    .negative_log_likelihood
                    .map(|value| value.to_string())
                    .unwrap_or_default();
                let gradient = result
                    .diagnostics
                    .gradient_norm
                    .map(|value| value.to_string())
                    .unwrap_or_default();
                let objective_spread = result
                    .diagnostics
                    .solution_objective_spread
                    .map(|value| value.to_string())
                    .unwrap_or_default();
                let parameter_spread = result
                    .diagnostics
                    .solution_parameter_spread
                    .as_ref()
                    .map(|values| values.iter().map(f64::to_string).collect::<Vec<_>>().join(";"))
                    .map(|value| format!(r#""{value}""#))
                    .unwrap_or_default();
                let curvature = result
                    .diagnostics
                    .diagonal_curvature
                    .as_ref()
                    .map(|values| values.iter().map(f64::to_string).collect::<Vec<_>>().join(";"))
                    .map(|value| format!(r#""{value}""#))
                    .unwrap_or_default();
                println!(
                    "{campaign_seed},{mu},{sigma},{:?},{mu_hat},{sigma_hat},{d_a_hat},{nll},{gradient},{},{objective_spread},{parameter_spread},{curvature}",
                    result.status,
                    result.diagnostics.iterations
                );
            }
        }
    }
}
