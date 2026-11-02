// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Parameter sensitivity sweep for the paper-shaped 20-generation ABM.
//!
//! This uses 50 fixed-seed replications per condition, matching the reported
//! replication count. The parameter grid is a CivBench sensitivity design,
//! not a byte-for-byte reproduction of the authors' archived parameter grid.

use civbench_social_learning_core::paper_abm::{
    simulate, Config, MachinePlacement, PAPER_ABM_REPLICATIONS,
};

#[derive(Clone)]
struct Condition {
    name: String,
    machine_count: usize,
    machine_placement: MachinePlacement,
    machine_capacity: u32,
    machine_bias: f64,
    optimal_discovery_rate: f64,
    transmission_probability: f64,
    teacher_temperature: f64,
    reward_noise_epsilon: f64,
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

    fn ci95_half_width(&self) -> Option<f64> {
        if self.n < 2 {
            return None;
        }
        let n = self.n as f64;
        let centered = (self.sum_squares - self.sum * self.sum / n).max(0.0);
        let sd = (centered / (n - 1.0)).sqrt();
        Some(1.96 * sd / n.sqrt())
    }
}

#[derive(Default)]
struct GenerationStats {
    optimal_prevalence: OnlineStats,
    myopic_prevalence: OnlineStats,
    null_prevalence: OnlineStats,
    machine_originated_optimal_prevalence: OnlineStats,
    mean_demonstration_reward: OnlineStats,
    non_null_social_transmission_rate: OnlineStats,
}

fn fmt(value: Option<f64>) -> String {
    value.map(|number| format!("{number:.6}")).unwrap_or_default()
}

fn build_conditions() -> Vec<Condition> {
    let mut conditions = vec![Condition {
        name: "no_machine_control".to_owned(),
        machine_count: 0,
        machine_placement: MachinePlacement::ClusteredPrefix,
        machine_capacity: 1_000,
        machine_bias: 0.5,
        optimal_discovery_rate: 0.001,
        transmission_probability: 0.9,
        teacher_temperature: 0.25,
        reward_noise_epsilon: 0.1,
    }];

    // The article explicitly varies discovery difficulty, machine capacity,
    // machine optimality bias, learnability, selectivity temperature and noise.
    // Placement treatments here are a CivBench sensitivity extension.
    let placements = [
        ("clustered", MachinePlacement::ClusteredPrefix),
        ("random", MachinePlacement::UniformRandom),
    ];
    let capacities = [1_000_u32, 10_000, 100_000];
    let biases = [0.01_f64, 0.25, 0.5];
    let optimal_discovery_rates = [0.0001_f64, 0.001, 0.01];
    let transmission = [0.3_f64, 0.6, 0.9];
    let temperatures = [0.1_f64, 0.5];
    let noise = [0.1_f64, 0.5];

    for (placement_name, placement) in placements {
        for capacity in capacities {
            for bias in biases {
                for optimal_discovery_rate in optimal_discovery_rates {
                    for lambda in transmission {
                        for temperature in temperatures {
                            for epsilon in noise {
                                conditions.push(Condition {
                                    name: format!(
                                        "machine_{placement_name}_k{capacity}_q{:.2}_dopt{:.4}_lambda{:.1}_tau{:.1}_eps{:.1}",
                                        bias, optimal_discovery_rate, lambda, temperature, epsilon
                                    ).replace('.', "p"),
                                    machine_count: 5,
                                    machine_placement: placement,
                                    machine_capacity: capacity,
                                    machine_bias: bias,
                                    optimal_discovery_rate,
                                    transmission_probability: lambda,
                                    teacher_temperature: temperature,
                                    reward_noise_epsilon: epsilon,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    conditions
}

fn main() {
    let conditions = build_conditions();
    println!(
        "condition,replications,generation,machine_placement,machine_capacity,machine_bias,d_opt,lambda,tau,epsilon,optimal_prevalence_mean,optimal_prevalence_ci95,myopic_prevalence_mean,null_prevalence_mean,machine_originated_optimal_prevalence,mean_demonstration_reward,non_null_social_transmission_rate"
    );

    for (condition_index, condition) in conditions.iter().enumerate() {
        let mut generation_stats: Vec<GenerationStats> =
            (0..20).map(|_| GenerationStats::default()).collect();

        for replication in 0..PAPER_ABM_REPLICATIONS {
            let config = Config {
                machine_count_generation_zero: condition.machine_count,
                machine_placement: condition.machine_placement,
                machine_exploration_capacity: condition.machine_capacity,
                machine_optimality_bias: condition.machine_bias,
                optimal_discovery_rate: condition.optimal_discovery_rate,
                social_transmission_probability: condition.transmission_probability,
                teacher_selection_temperature: condition.teacher_temperature,
                reward_noise_epsilon: condition.reward_noise_epsilon,
                ..Config::default()
            };

            let seed = 0xC1_7B_2026_u64
                .wrapping_add((condition_index as u64) << 32)
                .wrapping_add(replication as u64);
            let run = simulate(config, seed)
                .expect("predeclared ABM configuration must validate");

            for metrics in run.metrics() {
                let index = metrics.generation;
                let denominator = metrics.population_size as f64;
                let stats = &mut generation_stats[index];

                stats.optimal_prevalence.push(metrics.optimal_prevalence);
                stats.myopic_prevalence.push(metrics.myopic_count as f64 / denominator);
                stats.null_prevalence.push(metrics.null_count as f64 / denominator);
                stats.machine_originated_optimal_prevalence.push(
                    metrics.machine_originated_optimal_count as f64 / denominator
                );
                stats.mean_demonstration_reward.push(metrics.mean_demonstration_reward);
                stats.non_null_social_transmission_rate.push(
                    metrics.successful_non_null_transmissions as f64 / denominator
                );
            }
        }

        for (generation, stats) in generation_stats.iter().enumerate() {
            println!(
                "{},{},{},{},{},{:.3},{:.4},{:.3},{:.3},{:.3},{},{},{},{},{},{},{}",
                condition.name,
                PAPER_ABM_REPLICATIONS,
                generation,
                match condition.machine_placement {
                    MachinePlacement::ClusteredPrefix => "clustered",
                    MachinePlacement::UniformRandom => "random",
                },
                condition.machine_capacity,
                condition.machine_bias,
                condition.optimal_discovery_rate,
                condition.transmission_probability,
                condition.teacher_temperature,
                condition.reward_noise_epsilon,
                fmt(stats.optimal_prevalence.mean()),
                fmt(stats.optimal_prevalence.ci95_half_width()),
                fmt(stats.myopic_prevalence.mean()),
                fmt(stats.null_prevalence.mean()),
                fmt(stats.machine_originated_optimal_prevalence.mean()),
                fmt(stats.mean_demonstration_reward.mean()),
                fmt(stats.non_null_social_transmission_rate.mean()),
            );
        }
    }
}
