// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! 50-replication sensitivity suite for CivBench's compact strategy-adoption
//! reference model.
//!
//! This is an exploratory ablation model, not the archived Reward Network III
//! notebook ABM. Use reward_network_abm_reference_sweep for the notebook-aligned
//! semantic port and the archived parameter families.

use civbench_social_learning_core::paper_abm::{
    simulate, Config, MachineAvailability, MachinePlacement, PAPER_ABM_REPLICATIONS,
};

#[derive(Clone, Copy)]
enum AvailabilityLabel {
    None,
    SeedGenerationOnly,
    AllGenerations,
}

impl AvailabilityLabel {
    fn config(self) -> (usize, MachineAvailability, &'static str) {
        match self {
            Self::None => (0, MachineAvailability::SeedGenerationOnly, "none"),
            Self::SeedGenerationOnly => (5, MachineAvailability::SeedGenerationOnly, "seed_only"),
            Self::AllGenerations => (5, MachineAvailability::AllGenerations, "all_generations"),
        }
    }
}

#[derive(Clone)]
struct Condition {
    family: &'static str,
    name: String,
    population_size: usize,
    generations: usize,
    availability: AvailabilityLabel,
    placement: MachinePlacement,
    machine_capacity: u32,
    machine_bias: f64,
    optimal_discovery_rate: f64,
    teacher_candidate_count: usize,
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
        let sample_sd = (centered / (n - 1.0)).sqrt();
        Some(1.96 * sample_sd / n.sqrt())
    }
}

#[derive(Default)]
struct GenerationStats {
    optimal_prevalence: OnlineStats,
    human_optimal_prevalence: OnlineStats,
    machine_optimal_prevalence: OnlineStats,
    machine_originated_optimal_prevalence: OnlineStats,
    human_machine_originated_optimal_prevalence: OnlineStats,
    myopic_prevalence: OnlineStats,
    null_prevalence: OnlineStats,
    mean_demonstration_reward: OnlineStats,
    social_transmission_event_rate: OnlineStats,
    successful_non_null_transmission_rate: OnlineStats,
}

fn fmt(value: Option<f64>) -> String {
    value.map(|number| format!("{number:.6}")).unwrap_or_default()
}

fn availability_name(value: AvailabilityLabel) -> &'static str {
    value.config().2
}

fn build_conditions() -> Vec<Condition> {
    let mut conditions = Vec::new();

    // A. Discovery frontier. The supplement reports machine K_m in
    // {10, 1,000, 100,000}, q_m in {0.01, 0.5}, human k_h=10/q_h=0.01,
    // and manipulates d_opt over a log-scale range. Four representative
    // orders of magnitude are used here; full grid import remains a gate.
    for machine_capacity in [10_u32, 1_000, 100_000] {
        for machine_bias in [0.01_f64, 0.5] {
            for optimal_discovery_rate in [1e-6_f64, 1e-4, 1e-2, 1.0] {
                conditions.push(Condition {
                    family: "discovery_frontier",
                    name: format!(
                        "discovery_k{machine_capacity}_q{machine_bias:.2}_dopt{optimal_discovery_rate:.6}"
                    ),
                    population_size: 20,
                    generations: 20,
                    availability: AvailabilityLabel::SeedGenerationOnly,
                    placement: MachinePlacement::ClusteredPrefix,
                    machine_capacity,
                    machine_bias,
                    optimal_discovery_rate,
                    teacher_candidate_count: 5,
                    transmission_probability: 0.9,
                    teacher_temperature: 0.2,
                    reward_noise_epsilon: 0.2,
                });
            }
        }
    }

    // B. Transmission/selectivity/noise frontier. The supplement reports
    // tau and epsilon in {0.1, 0.2, 0.4}; lambda=0.9 is the high-learning
    // condition discussed in the paper, while 0.1 and 0.5 are declared
    // CivBench sensitivity contrasts. Availability treatments follow S10.
    for availability in [
        AvailabilityLabel::None,
        AvailabilityLabel::SeedGenerationOnly,
        AvailabilityLabel::AllGenerations,
    ] {
        for transmission_probability in [0.1_f64, 0.5, 0.9] {
            for teacher_temperature in [0.1_f64, 0.2, 0.4] {
                for reward_noise_epsilon in [0.1_f64, 0.2, 0.4] {
                    conditions.push(Condition {
                        family: "transmission_frontier",
                        name: format!(
                            "transmission_{}_lambda{transmission_probability:.1}_tau{teacher_temperature:.1}_eps{reward_noise_epsilon:.1}",
                            availability_name(availability)
                        ),
                        population_size: 20,
                        generations: 20,
                        availability,
                        placement: MachinePlacement::ClusteredPrefix,
                        machine_capacity: 100_000,
                        machine_bias: 0.5,
                        optimal_discovery_rate: 1e-4,
                        teacher_candidate_count: 5,
                        transmission_probability,
                        teacher_temperature,
                        reward_noise_epsilon,
                    });
                }
            }
        }
    }

    // C. Robustness to machine prevalence and population size. The paper's
    // supplement varies N_m=1..5 and N_total=20..100. We test both clustered
    // and uniformly randomized machine positions, holding other parameters
    // fixed. This is a sensitivity design, not a claim that placement labels
    // exactly reproduce the authors' archived placement generator.
    for population_size in [20_usize, 40, 60, 80, 100] {
        for machine_count in 1_usize..=5 {
            for (placement_name, placement) in [
                ("clustered", MachinePlacement::ClusteredPrefix),
                ("random", MachinePlacement::UniformRandom),
            ] {
                conditions.push(Condition {
                    family: "population_prevalence",
                    name: format!(
                        "population_n{population_size}_machines{machine_count}_{placement_name}"
                    ),
                    population_size,
                    generations: 20,
                    availability: AvailabilityLabel::SeedGenerationOnly,
                    placement,
                    machine_capacity: 100_000,
                    machine_bias: 0.5,
                    optimal_discovery_rate: 1e-4,
                    teacher_candidate_count: 5,
                    transmission_probability: 0.9,
                    teacher_temperature: 0.2,
                    reward_noise_epsilon: 0.2,
                });
            }
        }
    }

    // D. Teacher candidate-set-size robustness. The supplement says each
    // learner samples k teachers but does not report k in the extracted
    // description. This explicitly exposes the unknown instead of assuming
    // the human experiment's five-candidate interface applies to the ABM.
    for teacher_candidate_count in [1_usize, 5, 10, 20] {
        for teacher_temperature in [0.1_f64, 0.2, 0.4] {
            conditions.push(Condition {
                family: "teacher_candidate_size",
                name: format!("teacher_k{teacher_candidate_count}_tau{teacher_temperature:.1}"),
                population_size: 20,
                generations: 20,
                availability: AvailabilityLabel::SeedGenerationOnly,
                placement: MachinePlacement::ClusteredPrefix,
                machine_capacity: 100_000,
                machine_bias: 0.5,
                optimal_discovery_rate: 1e-4,
                teacher_candidate_count,
                transmission_probability: 0.9,
                teacher_temperature,
                reward_noise_epsilon: 0.2,
            });
        }
    }

    conditions
}

fn main() {
    let conditions = build_conditions();
    println!(
        "family,condition,replications,generation,population_size,machine_count,machine_availability,machine_placement,machine_capacity,machine_bias,d_opt,teacher_candidate_count,lambda,tau,epsilon,optimal_prevalence,human_optimal_prevalence,machine_optimal_prevalence,machine_originated_optimal_prevalence,human_machine_originated_optimal_prevalence,optimal_prevalence_ci95,human_optimal_prevalence_ci95,myopic_prevalence,null_prevalence,mean_demonstration_reward,social_transmission_event_rate,successful_non_null_transmission_rate"
    );

    for (condition_index, condition) in conditions.iter().enumerate() {
        let (machine_count, machine_availability, availability) =
            condition.availability.config();
        let mut generation_stats: Vec<GenerationStats> =
            (0..condition.generations).map(|_| GenerationStats::default()).collect();

        for replication in 0..PAPER_ABM_REPLICATIONS {
            let config = Config {
                generations: condition.generations,
                population_size: condition.population_size,
                machine_count_generation_zero: machine_count,
                machine_placement: condition.placement,
                machine_availability,
                machine_exploration_capacity: condition.machine_capacity,
                machine_optimality_bias: condition.machine_bias,
                optimal_discovery_rate: condition.optimal_discovery_rate,
                teacher_candidate_count: condition.teacher_candidate_count,
                social_transmission_probability: condition.transmission_probability,
                teacher_selection_temperature: condition.teacher_temperature,
                reward_noise_epsilon: condition.reward_noise_epsilon,
                ..Config::default()
            };

            let seed = 0xC1_7B_2026_u64
                .wrapping_add((condition_index as u64) << 32)
                .wrapping_add(replication as u64);
            let run = simulate(config, seed)
                .expect("predeclared paper ABM condition must validate");

            for metrics in run.metrics() {
                let index = metrics.generation;
                let denominator = metrics.population_size as f64;
                let stats = &mut generation_stats[index];

                stats.optimal_prevalence.push(metrics.optimal_prevalence);
                if let Some(value) = metrics.human_optimal_prevalence {
                    stats.human_optimal_prevalence.push(value);
                }
                if let Some(value) = metrics.machine_optimal_prevalence {
                    stats.machine_optimal_prevalence.push(value);
                }
                stats.machine_originated_optimal_prevalence.push(
                    metrics.machine_originated_optimal_count as f64 / denominator
                );
                if metrics.human_count > 0 {
                    stats.human_machine_originated_optimal_prevalence.push(
                        metrics.human_machine_originated_optimal_count as f64
                            / metrics.human_count as f64
                    );
                }
                stats.myopic_prevalence.push(metrics.myopic_count as f64 / denominator);
                stats.null_prevalence.push(metrics.null_count as f64 / denominator);
                stats.mean_demonstration_reward.push(metrics.mean_demonstration_reward);
                stats.social_transmission_event_rate.push(
                    metrics.social_transmissions as f64 / denominator
                );
                stats.successful_non_null_transmission_rate.push(
                    metrics.successful_non_null_transmissions as f64 / denominator
                );
            }
        }

        for (generation, stats) in generation_stats.iter().enumerate() {
            println!(
                "{},{},{},{},{},{},{},{},{},{:.3},{:.6},{},{:.3},{:.3},{:.3},{},{},{},{},{},{},{},{},{},{},{},{}",
                condition.family,
                condition.name,
                PAPER_ABM_REPLICATIONS,
                generation,
                condition.population_size,
                machine_count,
                availability,
                match condition.placement {
                    MachinePlacement::ClusteredPrefix => "clustered",
                    MachinePlacement::UniformRandom => "uniform_random",
                },
                condition.machine_capacity,
                condition.machine_bias,
                condition.optimal_discovery_rate,
                condition.teacher_candidate_count,
                condition.transmission_probability,
                condition.teacher_temperature,
                condition.reward_noise_epsilon,
                fmt(stats.optimal_prevalence.mean()),
                fmt(stats.human_optimal_prevalence.mean()),
                fmt(stats.machine_optimal_prevalence.mean()),
                fmt(stats.machine_originated_optimal_prevalence.mean()),
                fmt(stats.human_machine_originated_optimal_prevalence.mean()),
                fmt(stats.optimal_prevalence.ci95_half_width()),
                fmt(stats.human_optimal_prevalence.ci95_half_width()),
                fmt(stats.myopic_prevalence.mean()),
                fmt(stats.null_prevalence.mean()),
                fmt(stats.mean_demonstration_reward.mean()),
                fmt(stats.social_transmission_event_rate.mean()),
                fmt(stats.successful_non_null_transmission_rate.mean()),
            );
        }
    }
}
