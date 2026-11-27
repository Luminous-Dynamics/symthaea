// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Source-aligned parameter sweep for the exploratory Reward Network III ABM.
//!
//! Profiles mirror the archived abm/abm.ipynb cells:
//! - human_only_discovery (cell 4);
//! - machine_seed_discoverability (cell 7);
//! - teacher_temperature (cell 9);
//! - demonstration_reward_noise (cell 11);
//! - population_size_machine_count_robustness (cell 13).
//!
//! Default profile is a small smoke/sensitivity run. Pass --profile all to
//! enumerate the archived parameter grids; no results are computed at build
//! time and no notebook outputs are treated as observed outcomes.

use civbench_social_learning_core::reward_network_abm_reference::{
    simulate, Config, GenerationMetrics, PopulationCondition, SocialFailureFallback,
};

#[derive(Debug, Clone)]
struct Case {
    profile: &'static str,
    condition: PopulationCondition,
    generations: usize,
    population_size: usize,
    machine_count: usize,
    human_exploration_trials: u32,
    machine_exploration_trials: u32,
    human_optimality_bias: f64,
    machine_optimality_bias: f64,
    optimal_discovery_probability: f64,
    myopic_discovery_probability: f64,
    teacher_candidate_count: usize,
    teacher_temperature: f64,
    social_learning_probability: f64,
    demonstration_count: u32,
    demonstration_noise_epsilon: f64,
    fallback: SocialFailureFallback,
}

impl Case {
    fn config(&self) -> Config {
        Config {
            generations: self.generations,
            population_size: self.population_size,
            machine_count: self.machine_count,
            condition: self.condition,
            human_exploration_trials: self.human_exploration_trials,
            machine_exploration_trials: self.machine_exploration_trials,
            human_optimality_bias: self.human_optimality_bias,
            machine_optimality_bias: self.machine_optimality_bias,
            optimal_discovery_probability: self.optimal_discovery_probability,
            myopic_discovery_probability: self.myopic_discovery_probability,
            teacher_candidate_count: self.teacher_candidate_count,
            teacher_selection_temperature: self.teacher_temperature,
            social_learning_probability: self.social_learning_probability,
            social_failure_fallback: self.fallback,
            demonstration_count: self.demonstration_count,
            random_reward_per_demonstration: 0.0,
            myopic_reward_per_demonstration: 0.5,
            optimal_reward_per_demonstration: 1.0,
            demonstration_noise_epsilon: self.demonstration_noise_epsilon,
        }
    }

    fn condition_name(&self) -> &'static str {
        match self.condition {
            PopulationCondition::Human => "human",
            PopulationCondition::HumanMachineSeed => "human_machine_seed",
            PopulationCondition::HumanMachineAll => "human_machine_all",
        }
    }
}

#[derive(Debug, Default)]
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

#[derive(Debug, Default)]
struct GenerationStats {
    human_optimal: OnlineStats,
    machine_optimal: OnlineStats,
    human_myopic: OnlineStats,
    human_random: OnlineStats,
    individual_optimal: OnlineStats,
    social_optimal: OnlineStats,
    social_only_optimal: OnlineStats,
    machine_lineage_optimal: OnlineStats,
    mean_reward: OnlineStats,
    social_success: OnlineStats,
}

impl GenerationStats {
    fn push_defined(stats: &mut OnlineStats, value: Option<f64>) {
        if let Some(value) = value {
            stats.push(value);
        }
    }

    fn observe(&mut self, metrics: GenerationMetrics) {
        Self::push_defined(&mut self.human_optimal, metrics.human_optimal_prevalence);
        Self::push_defined(&mut self.machine_optimal, metrics.machine_optimal_prevalence);
        Self::push_defined(&mut self.human_myopic, metrics.human_myopic_prevalence);
        Self::push_defined(&mut self.human_random, metrics.human_random_prevalence);
        Self::push_defined(
            &mut self.individual_optimal,
            metrics.human_individual_optimal_discovery_rate,
        );
        Self::push_defined(&mut self.social_optimal, metrics.human_social_optimal_acquisition_rate);
        Self::push_defined(
            &mut self.social_only_optimal,
            metrics.human_social_only_optimal_integration_rate,
        );
        Self::push_defined(
            &mut self.machine_lineage_optimal,
            metrics.human_machine_lineage_optimal_prevalence,
        );
        self.mean_reward.push(metrics.mean_demonstration_reward);
        Self::push_defined(&mut self.social_success, metrics.social_learning_rate_observed);
    }
}

fn logspace10(start: f64, end: f64, count: usize) -> Vec<f64> {
    (0..count)
        .map(|index| {
            let fraction = index as f64 / (count.saturating_sub(1).max(1)) as f64;
            10.0_f64.powf(start + (end - start) * fraction)
        })
        .collect()
}

fn logspace_integer10(start: f64, end: f64, count: usize) -> Vec<u32> {
    logspace10(start, end, count)
        .into_iter()
        .map(|value| value as u32)
        .collect()
}

fn logspace2(start: f64, end: f64, count: usize) -> Vec<f64> {
    (0..count)
        .map(|index| {
            let fraction = index as f64 / (count.saturating_sub(1).max(1)) as f64;
            2.0_f64.powf(start + (end - start) * fraction)
        })
        .collect()
}

fn linspace(start: f64, end: f64, count: usize) -> Vec<f64> {
    (0..count)
        .map(|index| {
            let fraction = index as f64 / (count.saturating_sub(1).max(1)) as f64;
            start + (end - start) * fraction
        })
        .collect()
}

fn base_case(profile: &'static str) -> Case {
    Case {
        profile,
        condition: PopulationCondition::HumanMachineSeed,
        generations: 20,
        population_size: 20,
        machine_count: 5,
        human_exploration_trials: 10,
        machine_exploration_trials: 10_000,
        human_optimality_bias: 0.01,
        machine_optimality_bias: 0.5,
        optimal_discovery_probability: 1e-3,
        myopic_discovery_probability: 0.5,
        teacher_candidate_count: 10,
        teacher_temperature: 0.1,
        social_learning_probability: 0.25,
        demonstration_count: 1,
        demonstration_noise_epsilon: 0.1,
        fallback: SocialFailureFallback::ArchivedMyopicZero,
    }
}

fn quick_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    let mut human = base_case("quick_human_only");
    human.condition = PopulationCondition::Human;
    human.machine_count = 0;
    human.generations = 5;
    human.population_size = 20;
    human.human_exploration_trials = 100;
    human.optimal_discovery_probability = 1e-3;
    human.demonstration_count = 10;
    human.social_learning_probability = 0.9;
    cases.push(human);

    let mut seed = base_case("quick_machine_seeded");
    seed.generations = 5;
    seed.social_learning_probability = 0.25;
    cases.push(seed);

    let mut all = base_case("quick_machine_all_generations");
    all.condition = PopulationCondition::HumanMachineAll;
    all.generations = 5;
    cases.push(all);

    let mut preserve = base_case("quick_preserve_individual_ablation");
    preserve.generations = 5;
    preserve.fallback = SocialFailureFallback::PreserveIndividual;
    cases.push(preserve);
    cases
}

fn human_only_discovery_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    let k_human_values = logspace_integer10(1.0, 4.0, 10);
    let q_values = logspace2(-8.0, -1.0, 8);
    let optimal_values = logspace10(0.0, -6.0, 10);

    for k_human in k_human_values {
        for q_human in &q_values {
            for d_opt in &optimal_values {
                let mut case = base_case("human_only_discovery_cell4");
                case.condition = PopulationCondition::Human;
                case.machine_count = 0;
                case.human_exploration_trials = k_human;
                case.human_optimality_bias = *q_human;
                case.optimal_discovery_probability = *d_opt;
                case.social_learning_probability = 0.9;
                case.demonstration_count = 10;
                cases.push(case);
            }
        }
    }
    cases
}

fn machine_seed_discoverability_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for k_machine in [10_u32, 1_000, 100_000] {
        for q_machine in [0.01_f64, 0.5] {
            for d_opt in logspace10(0.0, -6.0, 10) {
                for lambda in linspace(0.0, 0.5, 10) {
                    let mut case = base_case("machine_seed_discoverability_cell7");
                    case.machine_exploration_trials = k_machine;
                    case.machine_optimality_bias = q_machine;
                    case.optimal_discovery_probability = d_opt;
                    case.social_learning_probability = lambda;
                    case.demonstration_count = 1;
                    cases.push(case);
                }
            }
        }
    }
    cases
}

fn temperature_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for condition in [
        PopulationCondition::Human,
        PopulationCondition::HumanMachineSeed,
        PopulationCondition::HumanMachineAll,
    ] {
        for d_opt in logspace10(0.0, -6.0, 10) {
            for lambda in linspace(0.0, 0.5, 10) {
                for temperature in [0.1_f64, 0.2, 0.4] {
                    let mut case = base_case("teacher_temperature_cell9");
                    case.condition = condition;
                    case.optimal_discovery_probability = d_opt;
                    case.social_learning_probability = lambda;
                    case.teacher_temperature = temperature;
                    cases.push(case);
                }
            }
        }
    }
    cases
}

fn reward_noise_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for condition in [
        PopulationCondition::Human,
        PopulationCondition::HumanMachineSeed,
        PopulationCondition::HumanMachineAll,
    ] {
        for d_opt in logspace10(0.0, -6.0, 10) {
            for lambda in linspace(0.0, 0.5, 10) {
                for epsilon in [0.1_f64, 0.2, 0.4] {
                    let mut case = base_case("demonstration_noise_cell11");
                    case.condition = condition;
                    case.optimal_discovery_probability = d_opt;
                    case.social_learning_probability = lambda;
                    case.demonstration_noise_epsilon = epsilon;
                    cases.push(case);
                }
            }
        }
    }
    cases
}

fn population_robustness_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for (population_size, machine_count) in [(20_usize, 5_usize), (20, 1), (100, 5), (100, 1)] {
        for condition in [
            PopulationCondition::Human,
            PopulationCondition::HumanMachineSeed,
            PopulationCondition::HumanMachineAll,
        ] {
            for d_opt in logspace10(0.0, -6.0, 10) {
                for lambda in linspace(0.0, 0.5, 10) {
                    let mut case = base_case("population_robustness_cell13");
                    case.population_size = population_size;
                    case.machine_count = machine_count;
                    case.condition = condition;
                    case.optimal_discovery_probability = d_opt;
                    case.social_learning_probability = lambda;
                    cases.push(case);
                }
            }
        }
    }
    cases
}

fn cases_for_profile(profile: &str) -> Vec<Case> {
    match profile {
        "quick" => quick_cases(),
        "human_only_discovery" => human_only_discovery_cases(),
        "machine_seed_discoverability" => machine_seed_discoverability_cases(),
        "teacher_temperature" => temperature_cases(),
        "demonstration_noise" => reward_noise_cases(),
        "population_robustness" => population_robustness_cases(),
        "all" => {
            let mut cases = Vec::new();
            cases.extend(human_only_discovery_cases());
            cases.extend(machine_seed_discoverability_cases());
            cases.extend(temperature_cases());
            cases.extend(reward_noise_cases());
            cases.extend(population_robustness_cases());
            cases
        }
        other => panic!("unknown profile {other}; use quick, human_only_discovery, machine_seed_discoverability, teacher_temperature, demonstration_noise, population_robustness, or all"),
    }
}

fn parse_args() -> (String, u64, Option<usize>) {
    let mut profile = "quick".to_owned();
    let mut replications = None;
    let mut seed = 0x524E_3341_424D_u64;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        if argument == "--profile" {
            profile = args.next().expect("--profile requires a value");
        } else if argument == "--replications" {
            replications = Some(args.next().expect("--replications requires a value").parse::<usize>().expect("replications must be a usize"));
        } else if argument == "--seed" {
            seed = args.next().expect("--seed requires a value").parse().expect("seed must be an integer");
        } else {
            panic!("unknown argument {argument}");
        }
    }
    (profile, seed, replications)
}

fn fmt(value: Option<f64>) -> String {
    value.map(|v| format!("{v:.6}")).unwrap_or_default()
}

fn emit_case(case_index: usize, case: &Case, replications: usize, base_seed: u64) {
    let mut generations: Vec<GenerationStats> =
        (0..case.generations).map(|_| GenerationStats::default()).collect();

    for replication in 0..replications {
        let seed = base_seed
            .wrapping_add((case_index as u64) << 32)
            .wrapping_add(replication as u64);
        let run = simulate(case.config(), seed).expect("predeclared ABM case must validate");
        for metrics in run.metrics() {
            generations[metrics.generation].observe(metrics);
        }
    }

    for (generation, stats) in generations.iter().enumerate() {
        let fields = [
            case.profile.to_owned(),
            case_index.to_string(),
            replications.to_string(),
            generation.to_string(),
            case.condition_name().to_owned(),
            case.generations.to_string(),
            case.population_size.to_string(),
            case.machine_count.to_string(),
            case.human_exploration_trials.to_string(),
            case.machine_exploration_trials.to_string(),
            format!("{:.8}", case.human_optimality_bias),
            format!("{:.8}", case.machine_optimality_bias),
            format!("{:.8}", case.optimal_discovery_probability),
            format!("{:.8}", case.myopic_discovery_probability),
            case.teacher_candidate_count.to_string(),
            format!("{:.6}", case.teacher_temperature),
            format!("{:.6}", case.social_learning_probability),
            case.demonstration_count.to_string(),
            format!("{:.6}", case.demonstration_noise_epsilon),
            fmt(stats.human_optimal.mean()),
            fmt(stats.human_optimal.ci95_half_width()),
            fmt(stats.machine_optimal.mean()),
            fmt(stats.machine_optimal.ci95_half_width()),
            fmt(stats.individual_optimal.mean()),
            fmt(stats.social_optimal.mean()),
            fmt(stats.social_only_optimal.mean()),
            fmt(stats.machine_lineage_optimal.mean()),
            fmt(stats.human_myopic.mean()),
            fmt(stats.human_random.mean()),
            fmt(stats.mean_reward.mean()),
            fmt(stats.social_success.mean()),
        ];
        println!("{}", fields.join(","));
    }
}

fn main() {
    let (profile, seed, replications_override) = parse_args();
    let cases = cases_for_profile(&profile);
    let replications = replications_override.unwrap_or(if profile == "quick" { 10 } else { 50 });

    println!(
        "profile,case_index,replications,generation,condition,generations,population_size,machine_count,k_human,k_machine,q_human,q_machine,d_opt,d_myopic,n_teacher,tau,lambda,k_demo,epsilon,human_optimal_mean,human_optimal_ci95,machine_optimal_mean,machine_optimal_ci95,human_individual_optimal_rate,human_social_optimal_acquisition_rate,human_social_only_optimal_integration_rate,human_machine_lineage_optimal_prevalence,human_myopic_prevalence,human_random_prevalence,mean_demonstration_reward,social_learning_success_rate"
    );
    for (index, case) in cases.iter().enumerate() {
        emit_case(index, case, replications, seed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notebook_parameter_grids_have_expected_cardinalities() {
        assert_eq!(human_only_discovery_cases().len(), 10 * 8 * 10);
        assert_eq!(machine_seed_discoverability_cases().len(), 3 * 2 * 10 * 10);
        assert_eq!(temperature_cases().len(), 3 * 10 * 10 * 3);
        assert_eq!(reward_noise_cases().len(), 3 * 10 * 10 * 3);
        assert_eq!(population_robustness_cases().len(), 4 * 3 * 10 * 10);
    }

    #[test]
    fn all_profile_contains_all_five_notebook_parameter_families() {
        assert_eq!(
            cases_for_profile("all").len(),
            (10 * 8 * 10) + (3 * 2 * 10 * 10) + (3 * 10 * 10 * 3)
                + (3 * 10 * 10 * 3) + (4 * 3 * 10 * 10)
        );
    }

    #[test]
    fn quick_profile_is_small_and_contains_archived_and_ablation_cases() {
        let cases = quick_cases();
        assert_eq!(cases.len(), 4);
        assert_eq!(cases[0].demonstration_count, 10);
        assert_eq!(cases[1].fallback, SocialFailureFallback::ArchivedMyopicZero);
        assert_eq!(cases[3].fallback, SocialFailureFallback::PreserveIndividual);
    }

    #[test]
    fn archived_parameter_ranges_match_notebook_cells() {
        let q = logspace2(-8.0, -1.0, 8);
        assert!((q[0] - 2.0_f64.powf(-8.0)).abs() < 1e-12);
        assert!((q[7] - 0.5).abs() < 1e-12);
        let d = logspace10(0.0, -6.0, 10);
        assert!((d[0] - 1.0).abs() < 1e-12);
        assert!((d[9] - 1e-6).abs() < 1e-12);
    }
}
