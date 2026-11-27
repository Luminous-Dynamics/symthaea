// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Branching-population cultural transmission sweep.
//!
//! The output is a synthetic lineage-drift sensitivity study, not a real LLM
//! telephone-game experiment.

use civbench_social_learning_core::iterated_transmission::{
    branching_generation_metrics, simulate_branching_population, CultureSignal,
    TransmissionChannelConfig,
};

const REPLICATES: u64 = 128;
const OPEN_ENDEDNESS: [f64; 4] = [0.0, 0.25, 0.5, 1.0];
const GENERATION_COUNTS: [usize; 4] = [4, 8, 16, 32];

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

fn initial_population() -> [CultureSignal; 4] {
    [
        CultureSignal::new([0.1, 0.9, 0.2, 0.2]).expect("valid synthetic signal"),
        CultureSignal::new([0.9, 0.1, 0.8, 0.8]).expect("valid synthetic signal"),
        CultureSignal::new([0.8, 0.8, 0.2, 0.4]).expect("valid synthetic signal"),
        CultureSignal::new([0.2, 0.3, 0.9, 0.9]).expect("valid synthetic signal"),
    ]
}

fn main() {
    println!(
        "open_endedness,generations,replicates,generation,mean_unique_lineages,mean_unique_lineages_ci95,mean_lineage_hhi,mean_effective_diversity,mean_extinct_root_lineages,mean_pairwise_content_distance,mean_distance_from_root_signal"
    );

    for open_endedness in OPEN_ENDEDNESS {
        for generation_count in GENERATION_COUNTS {
            let config = TransmissionChannelConfig {
                // Illustrative synthetic controls, not measured model parameters.
                attractor: [0.2, 0.8, 0.3, 0.5],
                attribute_bias: [0.8, 0.4, 0.6, 0.2],
                noise_std: [0.02; 4],
                open_endedness,
            };
            let rows = generation_count + 1;
            let mut unique: Vec<OnlineStats> = (0..rows).map(|_| OnlineStats::default()).collect();
            let mut hhi: Vec<OnlineStats> = (0..rows).map(|_| OnlineStats::default()).collect();
            let mut effective_diversity: Vec<OnlineStats> = (0..rows).map(|_| OnlineStats::default()).collect();
            let mut extinct: Vec<OnlineStats> = (0..rows).map(|_| OnlineStats::default()).collect();
            let mut pairwise: Vec<OnlineStats> = (0..rows).map(|_| OnlineStats::default()).collect();
            let mut root_distance: Vec<OnlineStats> = (0..rows).map(|_| OnlineStats::default()).collect();

            for replicate in 0..REPLICATES {
                let run = simulate_branching_population(
                    &initial_population(),
                    config,
                    generation_count,
                    0xB12A_2025_u64.wrapping_add(replicate * 313),
                ).expect("predeclared branching channel configuration must validate");
                let metrics = branching_generation_metrics(&run)
                    .expect("branching population metrics must be defined");
                for metric in metrics {
                    let generation = metric.generation;
                    unique[generation].push(metric.unique_lineages as f64);
                    hhi[generation].push(metric.lineage_hhi);
                    effective_diversity[generation].push(metric.effective_lineage_diversity);
                    extinct[generation].push(metric.extinct_root_lineages as f64);
                    pairwise[generation].push(metric.mean_pairwise_content_distance);
                    root_distance[generation].push(metric.mean_distance_from_root_signal);
                }
            }

            for generation in 0..rows {
                println!(
                    "{:.6},{},{},{},{},{},{},{},{},{},{}",
                    open_endedness,
                    generation_count,
                    REPLICATES,
                    generation,
                    fmt(unique[generation].mean()),
                    fmt(unique[generation].ci95_half_width()),
                    fmt(hhi[generation].mean()),
                    fmt(effective_diversity[generation].mean()),
                    fmt(extinct[generation].mean()),
                    fmt(pairwise[generation].mean()),
                    fmt(root_distance[generation].mean()),
                );
            }
        }
    }
}
