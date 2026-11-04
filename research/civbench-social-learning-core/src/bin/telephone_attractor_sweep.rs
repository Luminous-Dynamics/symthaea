// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Synthetic multi-generation content-drift and attractor sensitivity sweep.
//!
//! Channel parameters are illustrative fixtures, not estimates of any named
//! LLM or the ICLR 2025 paper's measurements.

use civbench_social_learning_core::iterated_transmission::{
    population_metrics, simulate_parallel_chains, CultureSignal, TransmissionChannelConfig,
    ATTRIBUTE_NAMES,
};

const REPLICATES: u64 = 128;
const OPEN_ENDEDNESS: [f64; 4] = [0.0, 0.25, 0.5, 1.0];
const CHAIN_LENGTHS: [usize; 4] = [4, 8, 16, 32];

#[derive(Default)]
struct OnlineStats {
    n: u64,
    sum: f64,
    sum_squares: f64,
}

impl OnlineStats {
    fn push(&mut self, value: f64) {
        self.n += 1;
        self.sum += value;
        self.sum_squares += value * value;
    }

    fn mean(&self) -> Option<f64> {
        (self.n > 0).then(|| self.sum / self.n as f64)
    }

    fn ci95_half_width(&self) -> Option<f64> {
        if self.n < 2 {
            return None;
        }
        let n = self.n as f64;
        let variance = ((self.sum_squares - self.sum * self.sum / n) / (n - 1.0)).max(0.0);
        Some(1.96 * variance.sqrt() / n.sqrt())
    }
}

fn fmt(value: Option<f64>) -> String {
    value.map(|number| format!("{number:.6}")).unwrap_or_default()
}

fn initial_signals() -> [CultureSignal; 4] {
    [
        CultureSignal::new([0.1, 0.9, 0.2, 0.2]).expect("valid synthetic signal"),
        CultureSignal::new([0.9, 0.1, 0.8, 0.8]).expect("valid synthetic signal"),
        CultureSignal::new([0.8, 0.8, 0.2, 0.4]).expect("valid synthetic signal"),
        CultureSignal::new([0.2, 0.3, 0.9, 0.9]).expect("valid synthetic signal"),
    ]
}

fn main() {
    println!(
        "open_endedness,chain_length,replicates,attribute,mean_signed_drift,mean_absolute_drift,mean_absolute_drift_ci95,mean_final_attractor_distance,attractor_convergence_index"
    );

    for open_endedness in OPEN_ENDEDNESS {
        for chain_length in CHAIN_LENGTHS {
            let config = TransmissionChannelConfig {
                // Illustrative synthetic targets/biases/noise. These must not
                // be read as empirical measurements of LLM attributes.
                attractor: [0.2, 0.8, 0.3, 0.5],
                attribute_bias: [0.8, 0.4, 0.6, 0.2],
                noise_std: [0.02; 4],
                open_endedness,
            };
            let mut signed_drift: Vec<OnlineStats> = (0..4).map(|_| OnlineStats::default()).collect();
            let mut absolute_drift: Vec<OnlineStats> = (0..4).map(|_| OnlineStats::default()).collect();
            let mut final_attractor_distance: Vec<OnlineStats> = (0..4).map(|_| OnlineStats::default()).collect();
            let mut convergence = OnlineStats::default();

            for replicate in 0..REPLICATES {
                let chains = simulate_parallel_chains(
                    &initial_signals(),
                    config,
                    chain_length,
                    0xC17_2025_u64.wrapping_add(replicate * 101),
                ).expect("predeclared synthetic channel configuration must validate");
                let metrics = population_metrics(&chains, config)
                    .expect("parallel chain metrics must be defined");

                for attribute in 0..4 {
                    signed_drift[attribute].push(metrics.mean_signed_drift[attribute]);
                    absolute_drift[attribute].push(metrics.mean_absolute_drift[attribute]);
                    final_attractor_distance[attribute]
                        .push(metrics.mean_final_attractor_distance[attribute]);
                }
                if let Some(value) = metrics.attractor_convergence_index {
                    convergence.push(value);
                }
            }

            for attribute in 0..4 {
                println!(
                    "{:.6},{},{},{},{},{},{},{},{}",
                    open_endedness,
                    chain_length,
                    REPLICATES,
                    ATTRIBUTE_NAMES[attribute],
                    fmt(signed_drift[attribute].mean()),
                    fmt(absolute_drift[attribute].mean()),
                    fmt(absolute_drift[attribute].ci95_half_width()),
                    fmt(final_attractor_distance[attribute].mean()),
                    fmt(convergence.mean()),
                );
            }
        }
    }
}
