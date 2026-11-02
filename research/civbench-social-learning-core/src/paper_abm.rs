// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#![forbid(unsafe_code)]

//! Simplified strategy-adoption reference model for CivBench sensitivity tests.
//!
//! This model is deliberately **not** the Reward Network III paper's archived
//! exploratory ABM. It uses a compact strategy abstraction and source-informed
//! parameters to test broad discovery/transmission mechanisms. The archived
//! notebook semantic port lives in reward_network_abm_reference.rs and must be
//! used for paper-aligned experiments.
//!
//! Keeping this module separate is intentional: this simpler model has useful
//! counterfactual controls (e.g. explicit acquisition/integration/corruption)
//! that are not part of the archived notebook implementation. Never compare
//! its outputs as though they were author ABM reproduction results.

use core::fmt;

pub const PAPER_ABM_REPLICATIONS: usize = 50;
pub const PAPER_ABM_GENERATIONS: usize = 20;
pub const PAPER_ABM_POPULATION_SIZE: usize = 20;
pub const PAPER_ABM_MACHINE_COUNT: usize = 5;
pub const PAPER_ABM_HUMAN_EXPLORATION_CAPACITY: u32 = 10;
pub const PAPER_ABM_HUMAN_OPTIMALITY_BIAS: f64 = 0.01;
pub const PAPER_ABM_MYOPIC_DISCOVERY_RATE: f64 = 0.5;
pub const PAPER_ABM_DEMONSTRATIONS: u32 = 10;
pub const PAPER_ABM_OPTIMAL_REWARD: f64 = 1.0;
pub const PAPER_ABM_MYOPIC_REWARD: f64 = 0.5;
pub const PAPER_ABM_REWARD_NOISE_EPSILON: f64 = 0.1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    None,
    Myopic,
    Optimal,
}

impl Strategy {
    fn rank(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Myopic => 1,
            Self::Optimal => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentKind {
    Human,
    Machine,
}

/// Initial-generation machine placement treatment. The paper varies placement
/// in robustness analyses; the exact archived schemes remain to be imported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MachinePlacement {
    ClusteredPrefix,
    UniformRandom,
}

/// Whether machine agents are available only in generation zero or in every
/// generation. The paper's supplementary robustness analysis compares both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MachineAvailability {
    SeedGenerationOnly,
    AllGenerations,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Agent {
    pub kind: AgentKind,
    pub individual_strategy: Strategy,
    pub social_strategy: Option<Strategy>,
    pub integrated_strategy: Strategy,
    pub teacher_index: Option<usize>,
    pub social_transmission_event: bool,
    pub demonstration_reward: f64,
    pub machine_originated_strategy: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Generation {
    pub index: usize,
    pub agents: Vec<Agent>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerationMetrics {
    pub generation: usize,
    pub population_size: usize,
    pub human_count: usize,
    pub machine_count: usize,
    pub optimal_count: usize,
    pub myopic_count: usize,
    pub null_count: usize,
    pub optimal_prevalence: f64,
    /// Undefined when a treatment has no human agents in a generation.
    pub human_optimal_prevalence: Option<f64>,
    /// Undefined when a treatment has no machine agents in a generation.
    pub machine_optimal_prevalence: Option<f64>,
    pub mean_demonstration_reward: f64,
    pub social_transmissions: usize,
    pub successful_non_null_transmissions: usize,
    pub optimal_transmissions: usize,
    pub machine_originated_optimal_count: usize,
    /// Machine-originated optimal strategies among human agents only.
    pub human_machine_originated_optimal_count: usize,
}

impl Generation {
    pub fn metrics(&self) -> GenerationMetrics {
        let human_count = self.agents.iter()
            .filter(|agent| agent.kind == AgentKind::Human).count();
        let machine_count = self.agents.iter()
            .filter(|agent| agent.kind == AgentKind::Machine).count();
        let human_optimal_count = self.agents.iter()
            .filter(|agent| agent.kind == AgentKind::Human
                && agent.integrated_strategy == Strategy::Optimal).count();
        let machine_optimal_count = self.agents.iter()
            .filter(|agent| agent.kind == AgentKind::Machine
                && agent.integrated_strategy == Strategy::Optimal).count();
        let optimal_count = self.agents.iter()
            .filter(|agent| agent.integrated_strategy == Strategy::Optimal)
            .count();
        let myopic_count = self.agents.iter()
            .filter(|agent| agent.integrated_strategy == Strategy::Myopic)
            .count();
        let null_count = self.agents.iter()
            .filter(|agent| agent.integrated_strategy == Strategy::None)
            .count();
        let mean_demonstration_reward = self.agents.iter()
            .map(|agent| agent.demonstration_reward)
            .sum::<f64>() / self.agents.len() as f64;
        let social_transmissions = self.agents.iter()
            .filter(|agent| agent.social_transmission_event)
            .count();
        let successful_non_null_transmissions = self.agents.iter()
            .filter(|agent| agent.social_transmission_event
                && agent.social_strategy.is_some())
            .count();
        let optimal_transmissions = self.agents.iter()
            .filter(|agent| agent.social_strategy == Some(Strategy::Optimal))
            .count();
        let machine_originated_optimal_count = self.agents.iter()
            .filter(|agent| agent.integrated_strategy == Strategy::Optimal
                && agent.machine_originated_strategy)
            .count();
        let human_machine_originated_optimal_count = self.agents.iter()
            .filter(|agent| agent.kind == AgentKind::Human
                && agent.integrated_strategy == Strategy::Optimal
                && agent.machine_originated_strategy)
            .count();

        GenerationMetrics {
            generation: self.index,
            population_size: self.agents.len(),
            human_count,
            machine_count,
            optimal_count,
            myopic_count,
            null_count,
            optimal_prevalence: optimal_count as f64 / self.agents.len() as f64,
            human_optimal_prevalence: if human_count == 0 {
                None
            } else {
                Some(human_optimal_count as f64 / human_count as f64)
            },
            machine_optimal_prevalence: if machine_count == 0 {
                None
            } else {
                Some(machine_optimal_count as f64 / machine_count as f64)
            },
            mean_demonstration_reward,
            social_transmissions,
            successful_non_null_transmissions,
            optimal_transmissions,
            machine_originated_optimal_count,
            human_machine_originated_optimal_count,
        }
    }
}

/// Configurable controls for the paper-described ABM.
///
/// The article reports core values for population size, machine count,
/// learning capacity/bias, demonstration count and reward noise. The exact
/// complete parameter grid and all implementation details remain to be
/// confirmed from the Zenodo archive/supplement; do not treat convenience
/// defaults as an exact reproduction of every published experiment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    pub generations: usize,
    pub population_size: usize,
    /// Machine count in generation zero; for AllGenerations this is the
    /// machine count in every generation.
    pub machine_count_generation_zero: usize,
    /// Position-selection rule for machine agents within a generation.
    pub machine_placement: MachinePlacement,
    /// Temporal availability treatment for machine agents.
    pub machine_availability: MachineAvailability,

    /// Number of individual-learning trials allocated to human agents.
    pub human_exploration_capacity: u32,
    /// Number of individual-learning trials allocated to machine agents.
    pub machine_exploration_capacity: u32,

    /// Probability mass allocated to trying the optimal rather than myopic
    /// strategy during individual exploration.
    pub human_optimality_bias: f64,
    pub machine_optimality_bias: f64,

    /// Per-trial strategy discovery rates. The paper fixes myopic discovery
    /// at 0.5 and manipulates optimal discoverability.
    pub optimal_discovery_rate: f64,
    pub myopic_discovery_rate: f64,

    /// Number of demonstrator candidates sampled from the prior generation.
    /// The article's main-text ABM description does not specify this count;
    /// V0 defaults to five as a sensitivity baseline, not a reproduction claim.
    pub teacher_candidate_count: usize,
    /// Softmax choice temperature over candidate demonstration rewards.
    pub teacher_selection_temperature: f64,
    /// Probability of acquiring the selected teacher's strategy.
    pub social_transmission_probability: f64,

    pub demonstration_count: u32,
    pub optimal_reward_per_demonstration: f64,
    pub myopic_reward_per_demonstration: f64,
    /// Per-demonstration noise parameter; total variance scales with K_demo.
    pub reward_noise_epsilon: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            generations: PAPER_ABM_GENERATIONS,
            population_size: PAPER_ABM_POPULATION_SIZE,
            machine_count_generation_zero: PAPER_ABM_MACHINE_COUNT,
            machine_placement: MachinePlacement::ClusteredPrefix,
            machine_availability: MachineAvailability::SeedGenerationOnly,
            human_exploration_capacity: PAPER_ABM_HUMAN_EXPLORATION_CAPACITY,
            machine_exploration_capacity: 1_000,
            human_optimality_bias: PAPER_ABM_HUMAN_OPTIMALITY_BIAS,
            machine_optimality_bias: 0.5,
            // V0 sensitivity reference only; the paper manipulates d_opt.
            optimal_discovery_rate: 0.001,
            myopic_discovery_rate: PAPER_ABM_MYOPIC_DISCOVERY_RATE,
            teacher_candidate_count: 5,
            teacher_selection_temperature: 0.25,
            social_transmission_probability: 0.9,
            demonstration_count: PAPER_ABM_DEMONSTRATIONS,
            optimal_reward_per_demonstration: PAPER_ABM_OPTIMAL_REWARD,
            myopic_reward_per_demonstration: PAPER_ABM_MYOPIC_REWARD,
            reward_noise_epsilon: PAPER_ABM_REWARD_NOISE_EPSILON,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    ZeroGenerations,
    ZeroPopulation,
    TooManyMachines { machines: usize, population: usize },
    ZeroExplorationCapacity,
    InvalidProbability { name: &'static str },
    InvalidBias { name: &'static str },
    InvalidTeacherCount,
    TooManyTeacherCandidates { candidates: usize, population: usize },
    InvalidTemperature,
    ZeroDemonstrations,
    InvalidReward,
    InvalidNoise,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroGenerations => write!(f, "generations must be positive"),
            Self::ZeroPopulation => write!(f, "population size must be positive"),
            Self::TooManyMachines { machines, population } =>
                write!(f, "{machines} machines exceed population size {population}"),
            Self::ZeroExplorationCapacity => write!(f, "exploration capacities must be positive"),
            Self::InvalidProbability { name } => write!(f, "{name} must be finite and in [0, 1]"),
            Self::InvalidBias { name } => write!(f, "{name} must be finite and in [0, 1]"),
            Self::InvalidTeacherCount => write!(f, "teacher candidate count must be positive"),
            Self::TooManyTeacherCandidates { candidates, population } =>
                write!(f, "{candidates} candidates exceed prior population size {population}"),
            Self::InvalidTemperature => write!(f, "teacher selection temperature must be finite and positive"),
            Self::ZeroDemonstrations => write!(f, "demonstration count must be positive"),
            Self::InvalidReward => write!(f, "demonstration rewards must be finite and non-negative"),
            Self::InvalidNoise => write!(f, "reward noise epsilon must be finite and non-negative"),
        }
    }
}
impl std::error::Error for ConfigError {}

impl Config {
    pub fn validate(self) -> Result<(), ConfigError> {
        if self.generations == 0 { return Err(ConfigError::ZeroGenerations); }
        if self.population_size == 0 { return Err(ConfigError::ZeroPopulation); }
        if self.machine_count_generation_zero > self.population_size {
            return Err(ConfigError::TooManyMachines {
                machines: self.machine_count_generation_zero,
                population: self.population_size,
            });
        }
        if self.human_exploration_capacity == 0 || self.machine_exploration_capacity == 0 {
            return Err(ConfigError::ZeroExplorationCapacity);
        }
        for (name, value) in [
            ("optimal discovery rate", self.optimal_discovery_rate),
            ("myopic discovery rate", self.myopic_discovery_rate),
            ("social transmission probability", self.social_transmission_probability),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(ConfigError::InvalidProbability { name });
            }
        }
        for (name, value) in [
            ("human optimality bias", self.human_optimality_bias),
            ("machine optimality bias", self.machine_optimality_bias),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(ConfigError::InvalidBias { name });
            }
        }
        if self.generations > 1 {
            if self.teacher_candidate_count == 0 {
                return Err(ConfigError::InvalidTeacherCount);
            }
            if self.teacher_candidate_count > self.population_size {
                return Err(ConfigError::TooManyTeacherCandidates {
                    candidates: self.teacher_candidate_count,
                    population: self.population_size,
                });
            }
        }
        if !self.teacher_selection_temperature.is_finite()
            || self.teacher_selection_temperature <= 0.0 {
            return Err(ConfigError::InvalidTemperature);
        }
        if self.demonstration_count == 0 {
            return Err(ConfigError::ZeroDemonstrations);
        }
        if !self.optimal_reward_per_demonstration.is_finite()
            || self.optimal_reward_per_demonstration < 0.0
            || !self.myopic_reward_per_demonstration.is_finite()
            || self.myopic_reward_per_demonstration < 0.0 {
            return Err(ConfigError::InvalidReward);
        }
        if !self.reward_noise_epsilon.is_finite() || self.reward_noise_epsilon < 0.0 {
            return Err(ConfigError::InvalidNoise);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Run {
    pub generations: Vec<Generation>,
}

impl Run {
    pub fn metrics(&self) -> Vec<GenerationMetrics> {
        self.generations.iter().map(Generation::metrics).collect()
    }

    pub fn final_optimal_prevalence(&self) -> Option<f64> {
        self.generations.last().map(|generation| generation.metrics().optimal_prevalence)
    }

    pub fn final_generation_contains_machine_originated_optimal(&self) -> bool {
        self.generations.last().map(|generation| {
            generation.agents.iter().any(|agent| {
                agent.integrated_strategy == Strategy::Optimal
                    && agent.machine_originated_strategy
                    && agent.kind == AgentKind::Human
            })
        }).unwrap_or(false)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self { Self { state: seed } }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
    }

    fn bernoulli(&mut self, probability: f64) -> bool {
        self.uniform() < probability
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(f64::MIN_POSITIVE);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
    }

    fn index(&mut self, upper: usize) -> usize {
        (self.next_u64() as usize) % upper
    }
}

fn discover_strategy(
    capacity: u32,
    optimality_bias: f64,
    config: Config,
    rng: &mut Rng,
) -> Strategy {
    // Exact terminal-outcome probabilities for K independent trials:
    // each trial attempts optimal with probability q and discovers it with
    // rate d_opt; otherwise it attempts myopic and discovers it with rate
    // d_myo. Optimal takes priority if both strategies are discovered.
    //
    // This closed form is distribution-equivalent to explicitly simulating
    // all K independent categorical trials, but avoids O(K) RNG work for the
    // machine exploration capacities (up to 100,000 in the paper).
    let k = f64::from(capacity);
    let q = optimality_bias;
    let p_optimal_per_trial = q * config.optimal_discovery_rate;
    let p_myopic_per_trial = (1.0 - q) * config.myopic_discovery_rate;

    let p_no_optimal = (1.0 - p_optimal_per_trial).powf(k);
    let p_no_discovery = (1.0 - p_optimal_per_trial - p_myopic_per_trial).powf(k);

    let p_optimal = (1.0 - p_no_optimal).clamp(0.0, 1.0);
    let p_myopic_only = (p_no_optimal - p_no_discovery).clamp(0.0, 1.0);

    let draw = rng.uniform();
    if draw < p_optimal {
        Strategy::Optimal
    } else if draw < p_optimal + p_myopic_only {
        Strategy::Myopic
    } else {
        Strategy::None
    }
}

fn demonstration_reward(strategy: Strategy, config: Config, rng: &mut Rng) -> f64 {
    let base = match strategy {
        Strategy::None => 0.0,
        Strategy::Myopic => f64::from(config.demonstration_count) * config.myopic_reward_per_demonstration,
        Strategy::Optimal => f64::from(config.demonstration_count) * config.optimal_reward_per_demonstration,
    };
    let noise_std = config.reward_noise_epsilon * f64::from(config.demonstration_count).sqrt();
    base + noise_std * rng.normal()
}

fn sample_candidates(population: &[Agent], count: usize, rng: &mut Rng) -> Vec<usize> {
    let mut pool: Vec<usize> = (0..population.len()).collect();
    for i in 0..count {
        let selected = i + rng.index(pool.len() - i);
        pool.swap(i, selected);
    }
    pool.truncate(count);
    pool
}

fn select_teacher(
    population: &[Agent],
    candidates: &[usize],
    temperature: f64,
    rng: &mut Rng,
) -> usize {
    let maximum = candidates.iter()
        .map(|&index| population[index].demonstration_reward)
        .fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = candidates.iter()
        .map(|&index| ((population[index].demonstration_reward - maximum) / temperature).exp())
        .collect();
    let sum: f64 = weights.iter().sum();

    if !sum.is_finite() || sum <= 0.0 {
        return candidates[0];
    }

    let mut draw = rng.uniform() * sum;
    for (&candidate, weight) in candidates.iter().zip(weights) {
        if draw < weight { return candidate; }
        draw -= weight;
    }
    *candidates.last().expect("validated candidate set is nonempty")
}

fn integrate(individual: Strategy, social: Option<Strategy>) -> Strategy {
    match social {
        Some(Strategy::Optimal) => Strategy::Optimal,
        Some(Strategy::Myopic) if individual != Strategy::Optimal => Strategy::Myopic,
        _ => individual,
    }
}

fn create_machine_mask(
    population_size: usize,
    machine_count: usize,
    placement: MachinePlacement,
    rng: &mut Rng,
) -> Vec<bool> {
    let mut mask = vec![false; population_size];
    match placement {
        MachinePlacement::ClusteredPrefix => {
            mask.iter_mut().take(machine_count).for_each(|slot| *slot = true);
        }
        MachinePlacement::UniformRandom => {
            let mut indices: Vec<usize> = (0..population_size).collect();
            for index in 0..machine_count {
                let selected = index + rng.index(indices.len() - index);
                indices.swap(index, selected);
            }
            for index in indices.into_iter().take(machine_count) {
                mask[index] = true;
            }
        }
    }
    mask
}

/// Run one generational ABM replication with a deterministic seed.
pub fn simulate(config: Config, seed: u64) -> Result<Run, ConfigError> {
    config.validate()?;
    let mut rng = Rng::new(seed);
    let mut generations = Vec::with_capacity(config.generations);

    let machine_mask = create_machine_mask(
        config.population_size,
        config.machine_count_generation_zero,
        config.machine_placement,
        &mut rng,
    );

    let mut seed_population = Vec::with_capacity(config.population_size);
    for index in 0..config.population_size {
        let kind = if machine_mask[index] {
            AgentKind::Machine
        } else {
            AgentKind::Human
        };
        let (capacity, bias) = match kind {
            AgentKind::Machine => (config.machine_exploration_capacity, config.machine_optimality_bias),
            AgentKind::Human => (config.human_exploration_capacity, config.human_optimality_bias),
        };
        let individual_strategy = discover_strategy(capacity, bias, config, &mut rng);
        let score = demonstration_reward(individual_strategy, config, &mut rng);
        seed_population.push(Agent {
            kind,
            individual_strategy,
            social_strategy: None,
            integrated_strategy: individual_strategy,
            teacher_index: None,
            social_transmission_event: false,
            demonstration_reward: score,
            machine_originated_strategy: kind == AgentKind::Machine
                && individual_strategy == Strategy::Optimal,
        });
    }
    generations.push(Generation { index: 0, agents: seed_population });

    for generation_index in 1..config.generations {
        let previous = &generations[generation_index - 1].agents;
        let mut agents = Vec::with_capacity(config.population_size);
        let generation_machine_mask = match config.machine_availability {
            MachineAvailability::SeedGenerationOnly => vec![false; config.population_size],
            MachineAvailability::AllGenerations => create_machine_mask(
                config.population_size,
                config.machine_count_generation_zero,
                config.machine_placement,
                &mut rng,
            ),
        };

        for agent_index in 0..config.population_size {
            let kind = if generation_machine_mask[agent_index] {
                AgentKind::Machine
            } else {
                AgentKind::Human
            };
            let (capacity, bias) = match kind {
                AgentKind::Machine => (
                    config.machine_exploration_capacity,
                    config.machine_optimality_bias,
                ),
                AgentKind::Human => (
                    config.human_exploration_capacity,
                    config.human_optimality_bias,
                ),
            };
            let individual_strategy = discover_strategy(
                capacity,
                bias,
                config,
                &mut rng,
            );
            let candidates = sample_candidates(previous, config.teacher_candidate_count, &mut rng);
            let teacher_index = select_teacher(
                previous,
                &candidates,
                config.teacher_selection_temperature,
                &mut rng,
            );
            let transmission_succeeded = rng.bernoulli(config.social_transmission_probability);
            let teacher_strategy = previous[teacher_index].integrated_strategy;
            let social_strategy = if transmission_succeeded && teacher_strategy != Strategy::None {
                Some(teacher_strategy)
            } else {
                None
            };
            let integrated_strategy = integrate(individual_strategy, social_strategy);
            let score = demonstration_reward(integrated_strategy, config, &mut rng);
            let machine_originated_strategy = integrated_strategy == Strategy::Optimal
                && ((kind == AgentKind::Machine
                    && individual_strategy == Strategy::Optimal)
                    || (social_strategy == Some(Strategy::Optimal)
                        && previous[teacher_index].machine_originated_strategy));

            agents.push(Agent {
                kind,
                individual_strategy,
                social_strategy,
                integrated_strategy,
                teacher_index: Some(teacher_index),
                social_transmission_event: transmission_succeeded,
                demonstration_reward: score,
                machine_originated_strategy,
            });
        }

        generations.push(Generation { index: generation_index, agents });
    }

    Ok(Run { generations })
}

/// Run the paper-reported number of independent simulation replications.
///
/// Each replication uses a deterministic seed offset. This is an experiment
/// harness, not a substitute for source-matching the authors' random streams.
pub fn simulate_replications(
    config: Config,
    replications: usize,
    base_seed: u64,
) -> Result<Vec<Run>, ConfigError> {
    config.validate()?;
    if replications == 0 {
        return Ok(Vec::new());
    }
    (0..replications)
        .map(|replication| simulate(config, base_seed.wrapping_add(replication as u64)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_reported_abm_population_shape() {
        let config = Config::default();
        assert_eq!(config.generations, 20);
        assert_eq!(config.population_size, 20);
        assert_eq!(config.machine_count_generation_zero, 5);
        assert_eq!(config.human_exploration_capacity, 10);
        assert_eq!(config.human_optimality_bias, 0.01);
        assert_eq!(config.machine_optimality_bias, 0.5);
        assert_eq!(config.myopic_discovery_rate, 0.5);
        assert_eq!(config.demonstration_count, 10);
    }

    #[test]
    fn clustered_machine_placement_uses_declared_prefix() {
        let config = Config {
            machine_placement: MachinePlacement::ClusteredPrefix,
            ..Config::default()
        };
        let run = simulate(config, 7).unwrap();
        assert!(run.generations[0].agents[..PAPER_ABM_MACHINE_COUNT]
            .iter().all(|agent| agent.kind == AgentKind::Machine));
        assert!(run.generations[0].agents[PAPER_ABM_MACHINE_COUNT..]
            .iter().all(|agent| agent.kind == AgentKind::Human));
    }

    #[test]
    fn uniform_machine_placement_is_seeded_and_keeps_exact_count() {
        let config = Config {
            machine_placement: MachinePlacement::UniformRandom,
            ..Config::default()
        };
        let first = simulate(config, 123).unwrap();
        let second = simulate(config, 123).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            first.generations[0].agents.iter()
                .filter(|agent| agent.kind == AgentKind::Machine).count(),
            PAPER_ABM_MACHINE_COUNT
        );
    }

    #[test]
    fn closed_form_discovery_matches_extreme_per_trial_cases() {
        let mut config = Config::default();
        config.optimal_discovery_rate = 1.0;
        config.myopic_discovery_rate = 1.0;
        let mut optimal_rng = Rng::new(44);
        let mut myopic_rng = Rng::new(44);
        assert_eq!(
            discover_strategy(100, 1.0, config, &mut optimal_rng),
            Strategy::Optimal
        );
        assert_eq!(
            discover_strategy(100, 0.0, config, &mut myopic_rng),
            Strategy::Myopic
        );
    }

    #[test]
    fn no_discovery_when_both_per_trial_rates_are_zero() {
        let mut config = Config::default();
        config.optimal_discovery_rate = 0.0;
        config.myopic_discovery_rate = 0.0;
        let mut rng = Rng::new(1);
        assert_eq!(discover_strategy(100_000, 0.5, config, &mut rng), Strategy::None);
    }

    #[test]
    fn optimality_bias_changes_trial_selection_not_strategy_priority() {
        let mut config = Config::default();
        config.human_exploration_capacity = 100;
        config.optimal_discovery_rate = 1.0;
        config.myopic_discovery_rate = 1.0;
        let mut high_bias_rng = Rng::new(44);
        let mut low_bias_rng = Rng::new(44);
        assert_eq!(
            discover_strategy(100, 1.0, config, &mut high_bias_rng),
            Strategy::Optimal
        );
        assert_eq!(
            discover_strategy(100, 0.0, config, &mut low_bias_rng),
            Strategy::Myopic
        );
    }

    #[test]
    fn fixed_seed_produces_identical_run() {
        let config = Config::default();
        assert_eq!(simulate(config, 42).unwrap(), simulate(config, 42).unwrap());
    }

    #[test]
    fn machine_seed_positions_are_followed_by_human_generations() {
        let run = simulate(Config::default(), 1).unwrap();
        assert_eq!(run.generations[0].agents.iter().filter(|agent| agent.kind == AgentKind::Machine).count(), 5);
        assert!(run.generations.iter().skip(1).all(|generation| {
            generation.agents.iter().all(|agent| agent.kind == AgentKind::Human)
        }));
    }

    #[test]
    fn optimal_strategy_dominates_myopic_when_both_are_available() {
        assert_eq!(integrate(Strategy::Optimal, Some(Strategy::Myopic)), Strategy::Optimal);
        assert_eq!(integrate(Strategy::Myopic, Some(Strategy::Optimal)), Strategy::Optimal);
    }

    #[test]
    fn prevalence_metrics_separate_humans_and_machines() {
        let generation = Generation {
            index: 1,
            agents: vec![
                Agent {
                    kind: AgentKind::Machine,
                    individual_strategy: Strategy::Optimal,
                    social_strategy: None,
                    integrated_strategy: Strategy::Optimal,
                    teacher_index: None,
                    social_transmission_event: false,
                    demonstration_reward: 10.0,
                    machine_originated_strategy: true,
                },
                Agent {
                    kind: AgentKind::Human,
                    individual_strategy: Strategy::Myopic,
                    social_strategy: None,
                    integrated_strategy: Strategy::Myopic,
                    teacher_index: None,
                    social_transmission_event: false,
                    demonstration_reward: 5.0,
                    machine_originated_strategy: false,
                },
                Agent {
                    kind: AgentKind::Human,
                    individual_strategy: Strategy::Optimal,
                    social_strategy: None,
                    integrated_strategy: Strategy::Optimal,
                    teacher_index: None,
                    social_transmission_event: false,
                    demonstration_reward: 10.0,
                    machine_originated_strategy: false,
                },
            ],
        };
        let metrics = generation.metrics();
        assert_eq!(metrics.optimal_prevalence, 2.0 / 3.0);
        assert_eq!(metrics.human_optimal_prevalence, Some(0.5));
        assert_eq!(metrics.machine_optimal_prevalence, Some(1.0));
        assert_eq!(metrics.human_machine_originated_optimal_count, 0);
    }

    #[test]
    fn machine_discoveries_in_later_generations_are_machine_originated() {
        let config = Config {
            generations: 3,
            machine_count_generation_zero: 5,
            machine_availability: MachineAvailability::AllGenerations,
            machine_exploration_capacity: 100,
            machine_optimality_bias: 1.0,
            human_exploration_capacity: 1,
            human_optimality_bias: 0.0,
            optimal_discovery_rate: 1.0,
            myopic_discovery_rate: 0.0,
            social_transmission_probability: 0.0,
            ..Config::default()
        };
        let run = simulate(config, 17).unwrap();
        for generation in run.generations.iter().skip(1) {
            assert!(generation.agents.iter()
                .filter(|agent| agent.kind == AgentKind::Machine)
                .all(|agent| agent.integrated_strategy == Strategy::Optimal
                    && agent.machine_originated_strategy));
            assert_eq!(generation.metrics().machine_optimal_prevalence, Some(1.0));
        }
    }

    #[test]
    fn machine_availability_can_be_first_generation_only_or_every_generation() {
        let seed_only = Config {
            machine_availability: MachineAvailability::SeedGenerationOnly,
            ..Config::default()
        };
        let seed_run = simulate(seed_only, 10).unwrap();
        assert_eq!(
            seed_run.generations[0].agents.iter()
                .filter(|agent| agent.kind == AgentKind::Machine).count(),
            PAPER_ABM_MACHINE_COUNT
        );
        assert!(seed_run.generations.iter().skip(1).all(|generation| {
            generation.agents.iter().all(|agent| agent.kind == AgentKind::Human)
        }));

        let all_generations = Config {
            machine_availability: MachineAvailability::AllGenerations,
            ..Config::default()
        };
        let all_run = simulate(all_generations, 10).unwrap();
        assert!(all_run.generations.iter().all(|generation| {
            generation.agents.iter()
                .filter(|agent| agent.kind == AgentKind::Machine).count()
                == PAPER_ABM_MACHINE_COUNT
        }));
    }

    #[test]
    fn zero_machine_count_remains_a_human_only_control() {
        let config = Config {
            machine_count_generation_zero: 0,
            machine_availability: MachineAvailability::AllGenerations,
            ..Config::default()
        };
        let run = simulate(config, 11).unwrap();
        assert!(run.generations.iter().all(|generation| {
            generation.agents.iter().all(|agent| agent.kind == AgentKind::Human)
        }));
    }

    #[test]
    fn no_social_transmission_keeps_machine_discovery_in_seed_generation_only() {
        let config = Config {
            social_transmission_probability: 0.0,
            human_optimality_bias: 0.0,
            machine_optimality_bias: 1.0,
            optimal_discovery_rate: 1.0,
            myopic_discovery_rate: 0.0,
            ..Config::default()
        };
        let run = simulate(config, 9).unwrap();
        assert!(run.generations[0].agents[..5].iter().all(|agent| agent.integrated_strategy == Strategy::Optimal));
        assert!(run.generations[1..].iter().all(|generation| {
            generation.agents.iter().all(|agent| agent.social_strategy.is_none())
        }));
    }

    #[test]
    fn perfect_transmission_preserves_optimal_strategy_after_machine_seed() {
        let config = Config {
            generations: 6,
            population_size: 20,
            machine_count_generation_zero: 5,
            human_exploration_capacity: 10,
            machine_exploration_capacity: 1_000,
            human_optimality_bias: 0.0,
            machine_optimality_bias: 1.0,
            optimal_discovery_rate: 1.0,
            myopic_discovery_rate: 0.0,
            teacher_candidate_count: 20,
            teacher_selection_temperature: 0.01,
            social_transmission_probability: 1.0,
            reward_noise_epsilon: 0.0,
            ..Config::default()
        };
        let run = simulate(config, 7).unwrap();
        assert!(run.generations[0].agents[..5].iter().all(|agent| agent.integrated_strategy == Strategy::Optimal));
        assert!(run.generations.iter().skip(1).all(|generation| {
            generation.agents.iter().all(|agent| agent.integrated_strategy == Strategy::Optimal)
        }));
    }

    #[test]
    fn machine_origin_lineage_requires_actual_social_inheritance() {
        let config = Config {
            generations: 4,
            population_size: 20,
            machine_count_generation_zero: 5,
            human_exploration_capacity: 10,
            machine_exploration_capacity: 1_000,
            human_optimality_bias: 0.0,
            machine_optimality_bias: 1.0,
            optimal_discovery_rate: 1.0,
            myopic_discovery_rate: 0.0,
            teacher_candidate_count: 20,
            teacher_selection_temperature: 0.01,
            social_transmission_probability: 1.0,
            reward_noise_epsilon: 0.0,
            ..Config::default()
        };
        let run = simulate(config, 101).unwrap();
        assert!(run.generations[1..].iter().all(|generation| {
            generation.agents.iter().all(|agent| {
                agent.integrated_strategy == Strategy::Optimal
                    && agent.machine_originated_strategy
            })
        }));
        assert!(run.final_generation_contains_machine_originated_optimal());
    }

    #[test]
    fn independent_optimal_discovery_is_not_mislabeled_machine_originated() {
        let config = Config {
            generations: 2,
            population_size: 20,
            machine_count_generation_zero: 0,
            human_exploration_capacity: 10,
            machine_exploration_capacity: 1_000,
            human_optimality_bias: 1.0,
            machine_optimality_bias: 0.5,
            optimal_discovery_rate: 1.0,
            myopic_discovery_rate: 0.0,
            teacher_candidate_count: 20,
            social_transmission_probability: 0.0,
            reward_noise_epsilon: 0.0,
            ..Config::default()
        };
        let run = simulate(config, 202).unwrap();
        assert!(run.generations[0].agents.iter().all(|agent| {
            agent.integrated_strategy == Strategy::Optimal
                && !agent.machine_originated_strategy
        }));
        assert!(run.generations[1].agents.iter().all(|agent| {
            agent.integrated_strategy == Strategy::Optimal
                && !agent.machine_originated_strategy
        }));
    }

    #[test]
    fn invalid_parameters_fail_closed() {
        let config = Config {
            social_transmission_probability: 1.5,
            ..Config::default()
        };
        assert_eq!(
            simulate(config, 1),
            Err(ConfigError::InvalidProbability { name: "social transmission probability" })
        );
    }
}
