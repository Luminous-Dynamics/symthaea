// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Multi-seed sensitivity sweep for the simplified cultural-transmission model.
//!
//! The output is a mechanism sensitivity experiment, not a reproduction of the
//! Reward Network III paper or its published ABM.

use civbench_social_learning_core::cultural_evolution::{
    simulate_transmission, TeacherSelectionRule, TransmissionConfig,
};

const REPLICATES: u64 = 256;

#[derive(Clone, Copy)]
struct Condition {
    name: &'static str,
    teacher_selection: TeacherSelectionRule,
    observation_probability: f64,
    corruption_probability: f64,
}

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

    fn sample_sd(&self) -> Option<f64> {
        if self.n < 2 {
            return None;
        }
        let n = self.n as f64;
        let centered = self.sum_squares - self.sum * self.sum / n;
        Some((centered.max(0.0) / (n - 1.0)).sqrt())
    }

    fn ci95_half_width(&self) -> Option<f64> {
        self.sample_sd()
            .map(|sd| 1.96 * sd / (self.n as f64).sqrt())
    }
}

fn fmt_optional(value: Option<f64>) -> String {
    value.map(|v| format!("{v:.6}")).unwrap_or_default()
}

fn main() {
    let conditions = [
        Condition {
            name: "no_social",
            teacher_selection: TeacherSelectionRule::Uniform,
            observation_probability: 0.0,
            corruption_probability: 0.0,
        },
        Condition {
            name: "uniform_clean",
            teacher_selection: TeacherSelectionRule::Uniform,
            observation_probability: 0.9,
            corruption_probability: 0.0,
        },
        Condition {
            name: "uniform_5pct_corruption",
            teacher_selection: TeacherSelectionRule::Uniform,
            observation_probability: 0.9,
            corruption_probability: 0.05,
        },
        Condition {
            name: "uniform_20pct_corruption",
            teacher_selection: TeacherSelectionRule::Uniform,
            observation_probability: 0.9,
            corruption_probability: 0.20,
        },
        Condition {
            name: "performance_clean",
            teacher_selection: TeacherSelectionRule::PerformanceSoftmax { temperature: 0.25 },
            observation_probability: 0.9,
            corruption_probability: 0.0,
        },
        Condition {
            name: "performance_5pct_corruption",
            teacher_selection: TeacherSelectionRule::PerformanceSoftmax { temperature: 0.25 },
            observation_probability: 0.9,
            corruption_probability: 0.05,
        },
        Condition {
            name: "performance_20pct_corruption",
            teacher_selection: TeacherSelectionRule::PerformanceSoftmax { temperature: 0.25 },
            observation_probability: 0.9,
            corruption_probability: 0.20,
        },
    ];

    println!(
        "condition,generation,replicates,individual_optimal_discovery_rate,integrated_optimal_prevalence,integrated_optimal_prevalence_ci95,observation_rate,acquisition_rate_given_observation,optimal_acquisition_rate_given_observation,integration_rate_given_acquisition,optimal_integration_rate_given_observation,top_teacher_selection_rate"
    );

    for condition in conditions {
        let mut individual_optimal_discovery_rate = Vec::new();
        let mut optimal_prevalence = Vec::new();
        let mut observation_rate = Vec::new();
        let mut acquisition_rate = Vec::new();
        let mut optimal_acquisition_rate = Vec::new();
        let mut integration_rate = Vec::new();
        let mut optimal_integration_rate = Vec::new();
        let mut top_teacher_rate = Vec::new();

        for _ in 0..5 {
            individual_optimal_discovery_rate.push(OnlineStats::default());
            optimal_prevalence.push(OnlineStats::default());
            observation_rate.push(OnlineStats::default());
            acquisition_rate.push(OnlineStats::default());
            optimal_acquisition_rate.push(OnlineStats::default());
            integration_rate.push(OnlineStats::default());
            optimal_integration_rate.push(OnlineStats::default());
            top_teacher_rate.push(OnlineStats::default());
        }

        for seed in 0..REPLICATES {
            let config = TransmissionConfig {
                teacher_selection: condition.teacher_selection,
                social_observation_probability: condition.observation_probability,
                demonstration_corruption_probability: condition.corruption_probability,
                ..TransmissionConfig::default()
            };
            let run = simulate_transmission(config, seed)
                .expect("predeclared sweep configuration must validate");

            for metrics in run.metrics() {
                let generation = metrics.generation;
                individual_optimal_discovery_rate[generation].push(
                    metrics.individually_discovered_optimal_count as f64
                        / metrics.total_agents as f64,
                );
                optimal_prevalence[generation].push(metrics.optimal_prevalence);
                if let Some(value) = metrics.demonstration_observation_rate {
                    observation_rate[generation].push(value);
                }
                if let Some(value) = metrics.social_acquisition_rate_given_observation {
                    acquisition_rate[generation].push(value);
                }
                if let Some(value) = metrics.optimal_acquisition_rate_given_observation {
                    optimal_acquisition_rate[generation].push(value);
                }
                if let Some(value) = metrics.integration_rate_given_acquisition {
                    integration_rate[generation].push(value);
                }
                if let Some(value) = metrics.optimal_integration_rate_given_observation {
                    optimal_integration_rate[generation].push(value);
                }
                if let Some(value) = metrics.top_teacher_selection_rate {
                    top_teacher_rate[generation].push(value);
                }
            }
        }

        for generation in 0..5 {
            let optimal = &optimal_prevalence[generation];
            println!(
                "{},{},{},{},{},{},{},{},{},{},{},{}",
                condition.name,
                generation,
                REPLICATES,
                fmt_optional(individual_optimal_discovery_rate[generation].mean()),
                fmt_optional(optimal.mean()),
                fmt_optional(optimal.ci95_half_width()),
                fmt_optional(observation_rate[generation].mean()),
                fmt_optional(acquisition_rate[generation].mean()),
                fmt_optional(optimal_acquisition_rate[generation].mean()),
                fmt_optional(integration_rate[generation].mean()),
                fmt_optional(optimal_integration_rate[generation].mean()),
                fmt_optional(top_teacher_rate[generation].mean()),
            );
        }
    }
}
