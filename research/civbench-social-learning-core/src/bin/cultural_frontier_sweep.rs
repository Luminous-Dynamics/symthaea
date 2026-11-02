// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Three-axis cultural frontier experiment.
//!
//! Axes: optimal-strategy discovery difficulty, social acquisition probability
//! (learnability), and demonstrable selective advantage. All conditions share
//! the same 0..REPLICATES seed set. This is a synthetic mechanism study, not
//! a reproduction of the published human experiment or its exact ABM.

use civbench_social_learning_core::cultural_evolution::{
    simulate_transmission, TeacherSelectionRule, TransmissionConfig,
};

const REPLICATES: u64 = 256;
const GENERATIONS: usize = 5;
const DISCOVERY_PROBABILITIES: [f64; 3] = [0.0001, 0.001, 0.01];
const ACQUISITION_PROBABILITIES: [f64; 3] = [0.25, 0.60, 0.95];
const SELECTIVE_ADVANTAGES: [f64; 3] = [0.0, 0.25, 0.50];
const MYOPIC_REWARD: f64 = 0.5;

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
    value.map(|v| format!("{v:.6}")).unwrap_or_default()
}

fn main() {
    println!(
        "optimal_discovery_probability_per_trial,social_acquisition_probability,selective_advantage,optimal_demonstration_reward,generation,replicates,individual_optimal_discovery_rate,integrated_optimal_prevalence,integrated_optimal_prevalence_ci95,social_only_optimal_acquisition_rate,social_only_optimal_integration_rate,top_teacher_selection_rate,persistent_across_post_seed_generations_probability"
    );

    for discovery_probability in DISCOVERY_PROBABILITIES {
        for acquisition_probability in ACQUISITION_PROBABILITIES {
            for selective_advantage in SELECTIVE_ADVANTAGES {
                let config = TransmissionConfig {
                    optimal_discovery_probability_per_trial: discovery_probability,
                    social_observation_probability: 0.9,
                    social_strategy_acquisition_probability: acquisition_probability,
                    demonstration_corruption_probability: 0.05,
                    strategy_integration_probability: 0.9,
                    teacher_selection: TeacherSelectionRule::PerformanceSoftmax {
                        temperature: 0.25,
                    },
                    myopic_demonstration_reward: MYOPIC_REWARD,
                    optimal_demonstration_reward: MYOPIC_REWARD + selective_advantage,
                    ..TransmissionConfig::default()
                };

                let mut discovery = Vec::new();
                let mut prevalence = Vec::new();
                let mut social_only_acquisition = Vec::new();
                let mut social_only_integration = Vec::new();
                let mut top_teacher = Vec::new();
                for _ in 0..GENERATIONS {
                    discovery.push(OnlineStats::default());
                    prevalence.push(OnlineStats::default());
                    social_only_acquisition.push(OnlineStats::default());
                    social_only_integration.push(OnlineStats::default());
                    top_teacher.push(OnlineStats::default());
                }
                let mut persistent = OnlineStats::default();

                for seed in 0..REPLICATES {
                    let run = simulate_transmission(config, seed)
                        .expect("predeclared frontier configuration must validate");
                    persistent.push(
                        if run.optimal_present_in_every_post_seed_generation() {
                            1.0
                        } else {
                            0.0
                        },
                    );

                    for metrics in run.metrics() {
                        let generation = metrics.generation;
                        let denominator = metrics.total_agents as f64;
                        discovery[generation].push(
                            metrics.individually_discovered_optimal_count as f64 / denominator,
                        );
                        prevalence[generation].push(metrics.optimal_prevalence);
                        social_only_acquisition[generation].push(
                            metrics.social_only_optimal_acquisitions as f64 / denominator,
                        );
                        social_only_integration[generation].push(
                            metrics.social_only_optimal_integrations as f64 / denominator,
                        );
                        if let Some(value) = metrics.top_teacher_selection_rate {
                            top_teacher[generation].push(value);
                        }
                    }
                }

                for generation in 0..GENERATIONS {
                    println!(
                        "{:.6},{:.6},{:.6},{:.6},{},{},{},{},{},{},{},{},{}",
                        discovery_probability,
                        acquisition_probability,
                        selective_advantage,
                        MYOPIC_REWARD + selective_advantage,
                        generation,
                        REPLICATES,
                        fmt(discovery[generation].mean()),
                        fmt(prevalence[generation].mean()),
                        fmt(prevalence[generation].ci95_half_width()),
                        fmt(social_only_acquisition[generation].mean()),
                        fmt(social_only_integration[generation].mean()),
                        fmt(top_teacher[generation].mean()),
                        fmt(persistent.mean()),
                    );
                }
            }
        }
    }
}
