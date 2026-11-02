// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#![forbid(unsafe_code)]

//! Small, parameterized generational cultural-transmission reference model.
//!
//! It captures individual discovery, noisy demonstration performance,
//! softmax teacher selection, probabilistic social acquisition and strategy
//! integration. It is inspired by the mechanism sequence in Brinkmann et al.
//! (2026), but is not a reproduction of their published ABM parameters or
//! code. Those require the exact supplementary information and Zenodo archive.

use core::fmt;

/// Strategy repertoire used by the reference model.
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

    fn demonstration_reward(self, demonstrations: u32) -> f64 {
        let reward_per_demo = match self {
            Self::None => 0.0,
            Self::Myopic => 0.5,
            Self::Optimal => 1.0,
        };
        f64::from(demonstrations) * reward_per_demo
    }
}

/// Role is only used for the initial machine-seeded generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentKind {
    Human,
    Machine,
}

/// Complete agent record retained for benchmark auditing.
#[derive(Debug, Clone, PartialEq)]
pub struct TransmissionAgent {
    pub kind: AgentKind,
    pub individual_strategy: Strategy,
    pub social_strategy: Option<Strategy>,
    pub integrated_strategy: Strategy,
    /// Index of the selected teacher in the previous generation.
    pub teacher_index: Option<usize>,
    pub social_acquisition_succeeded: bool,
    /// Demonstration score used by subsequent-generation teacher selection.
    pub demonstration_score: f64,
}

/// A generation is a fixed-size population in the reference experiment.
#[derive(Debug, Clone, PartialEq)]
pub struct PopulationGeneration {
    pub index: usize,
    pub agents: Vec<TransmissionAgent>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerationMetrics {
    pub generation: usize,
    pub total_agents: usize,
    pub optimal_count: usize,
    pub myopic_count: usize,
    pub uninformed_count: usize,
    pub optimal_prevalence: f64,
    pub social_acquisitions: usize,
    pub social_acquisition_rate: Option<f64>,
}

impl PopulationGeneration {
    pub fn metrics(&self) -> GenerationMetrics {
        let optimal_count = self.agents.iter()
            .filter(|agent| agent.integrated_strategy == Strategy::Optimal).count();
        let myopic_count = self.agents.iter()
            .filter(|agent| agent.integrated_strategy == Strategy::Myopic).count();
        let uninformed_count = self.agents.iter()
            .filter(|agent| agent.integrated_strategy == Strategy::None).count();
        let social_acquisitions = self.agents.iter()
            .filter(|agent| agent.social_acquisition_succeeded).count();
        let total_agents = self.agents.len();
        GenerationMetrics {
            generation: self.index,
            total_agents,
            optimal_count,
            myopic_count,
            uninformed_count,
            optimal_prevalence: optimal_count as f64 / total_agents as f64,
            social_acquisitions,
            social_acquisition_rate: if self.index == 0 {
                None
            } else {
                Some(social_acquisitions as f64 / total_agents as f64)
            },
        }
    }
}

/// Parameters for a transparent generational transmission experiment.
///
/// Discovery probabilities are per allocated exploration trial. Human and
/// machine agents may have different trial budgets and exploration biases.
/// Social acquisition is independent of teacher selection and is measured
/// separately from the learner's integrated final strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransmissionConfig {
    pub generations: usize,
    pub population_size: usize,
    pub machine_count_generation_zero: usize,
    pub human_learning_trials: u32,
    pub machine_learning_trials: u32,
    /// Fraction of individual-learning effort allocated to optimal discovery.
    pub human_optimal_exploration_bias: f64,
    pub machine_optimal_exploration_bias: f64,
    pub optimal_discovery_probability_per_trial: f64,
    pub myopic_discovery_probability_per_trial: f64,
    /// Probability that an attempted social transmission is acquired.
    pub social_acquisition_probability: f64,
    /// Number of previous-generation teacher candidates sampled per learner.
    pub teacher_candidate_count: usize,
    /// Softmax temperature over demonstration scores; lower is more selective.
    pub teacher_selection_temperature: f64,
    pub demonstration_count: u32,
    /// Per-demonstration noise scale; aggregate score noise is epsilon*sqrt(K).
    pub reward_noise_epsilon: f64,
}

impl Default for TransmissionConfig {
    fn default() -> Self {
        Self {
            generations: 5,
            population_size: 8,
            machine_count_generation_zero: 3,
            human_learning_trials: 6,
            machine_learning_trials: 1_000,
            human_optimal_exploration_bias: 0.1,
            machine_optimal_exploration_bias: 0.9,
            optimal_discovery_probability_per_trial: 0.001,
            myopic_discovery_probability_per_trial: 0.5,
            social_acquisition_probability: 0.8,
            teacher_candidate_count: 5,
            teacher_selection_temperature: 0.25,
            demonstration_count: 10,
            reward_noise_epsilon: 0.1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransmissionError {
    ZeroGenerations,
    ZeroPopulation,
    TooManyMachines { machines: usize, population: usize },
    ZeroLearningTrials,
    ZeroTeacherCandidates,
    TooManyTeacherCandidates { candidates: usize, population: usize },
    ZeroDemonstrations,
    InvalidProbability { name: &'static str },
    InvalidBias { name: &'static str },
    InvalidTemperature,
    InvalidNoiseScale,
}

impl fmt::Display for TransmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroGenerations => write!(f, "generations must be positive"),
            Self::ZeroPopulation => write!(f, "population size must be positive"),
            Self::TooManyMachines { machines, population } => write!(f, "{machines} machines exceed population size {population}"),
            Self::ZeroLearningTrials => write!(f, "individual-learning trial budgets must be positive"),
            Self::ZeroTeacherCandidates => write!(f, "teacher candidate count must be positive when generations > 1"),
            Self::TooManyTeacherCandidates { candidates, population } => write!(f, "{candidates} teacher candidates exceed prior generation size {population}"),
            Self::ZeroDemonstrations => write!(f, "demonstration count must be positive"),
            Self::InvalidProbability { name } => write!(f, "{name} must be finite and within [0, 1]"),
            Self::InvalidBias { name } => write!(f, "{name} must be finite and within [0, 1]"),
            Self::InvalidTemperature => write!(f, "teacher-selection temperature must be finite and positive"),
            Self::InvalidNoiseScale => write!(f, "reward noise epsilon must be finite and non-negative"),
        }
    }
}
impl std::error::Error for TransmissionError {}

impl TransmissionConfig {
    pub fn validate(self) -> Result<(), TransmissionError> {
        if self.generations == 0 { return Err(TransmissionError::ZeroGenerations); }
        if self.population_size == 0 { return Err(TransmissionError::ZeroPopulation); }
        if self.machine_count_generation_zero > self.population_size {
            return Err(TransmissionError::TooManyMachines {
                machines: self.machine_count_generation_zero,
                population: self.population_size,
            });
        }
        if self.human_learning_trials == 0 || self.machine_learning_trials == 0 {
            return Err(TransmissionError::ZeroLearningTrials);
        }
        if self.generations > 1 {
            if self.teacher_candidate_count == 0 { return Err(TransmissionError::ZeroTeacherCandidates); }
            if self.teacher_candidate_count > self.population_size {
                return Err(TransmissionError::TooManyTeacherCandidates {
                    candidates: self.teacher_candidate_count,
                    population: self.population_size,
                });
            }
        }
        if self.demonstration_count == 0 { return Err(TransmissionError::ZeroDemonstrations); }
        for (name, value) in [
            ("optimal discovery probability", self.optimal_discovery_probability_per_trial),
            ("myopic discovery probability", self.myopic_discovery_probability_per_trial),
            ("social acquisition probability", self.social_acquisition_probability),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(TransmissionError::InvalidProbability { name });
            }
        }
        for (name, value) in [
            ("human optimal exploration bias", self.human_optimal_exploration_bias),
            ("machine optimal exploration bias", self.machine_optimal_exploration_bias),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(TransmissionError::InvalidBias { name });
            }
        }
        if !self.teacher_selection_temperature.is_finite() || self.teacher_selection_temperature <= 0.0 {
            return Err(TransmissionError::InvalidTemperature);
        }
        if !self.reward_noise_epsilon.is_finite() || self.reward_noise_epsilon < 0.0 {
            return Err(TransmissionError::InvalidNoiseScale);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransmissionRun {
    pub generations: Vec<PopulationGeneration>,
}

impl TransmissionRun {
    pub fn metrics(&self) -> Vec<GenerationMetrics> {
        self.generations.iter().map(PopulationGeneration::metrics).collect()
    }

    /// Human generations 1..end all contain at least one optimal strategy.
    /// This is a model-side persistence descriptor, not the paper's raw-data
    /// classification implementation.
    pub fn optimal_present_in_every_post_seed_generation(&self) -> bool {
        self.generations.iter().skip(1).all(|generation| {
            generation.agents.iter().any(|agent| agent.integrated_strategy == Strategy::Optimal)
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
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
    trials: u32,
    optimal_bias: f64,
    config: TransmissionConfig,
    rng: &mut DeterministicRng,
) -> Strategy {
    let optimal_trials = f64::from(trials) * optimal_bias;
    let myopic_trials = f64::from(trials) * (1.0 - optimal_bias);
    let optimal_probability =
        1.0 - (1.0 - config.optimal_discovery_probability_per_trial).powf(optimal_trials);
    let myopic_probability =
        1.0 - (1.0 - config.myopic_discovery_probability_per_trial).powf(myopic_trials);

    let found_optimal = rng.bernoulli(optimal_probability);
    let found_myopic = rng.bernoulli(myopic_probability);
    if found_optimal {
        Strategy::Optimal
    } else if found_myopic {
        Strategy::Myopic
    } else {
        Strategy::None
    }
}

fn demonstration_score(
    strategy: Strategy,
    config: TransmissionConfig,
    rng: &mut DeterministicRng,
) -> f64 {
    let base = strategy.demonstration_reward(config.demonstration_count);
    let noise_std = config.reward_noise_epsilon * f64::from(config.demonstration_count).sqrt();
    base + noise_std * rng.normal()
}

fn sample_teacher_candidates(
    previous: &[TransmissionAgent],
    candidate_count: usize,
    rng: &mut DeterministicRng,
) -> Vec<usize> {
    let mut pool: Vec<usize> = (0..previous.len()).collect();
    for i in 0..candidate_count {
        let selected = i + rng.index(pool.len() - i);
        pool.swap(i, selected);
    }
    pool.truncate(candidate_count);
    pool
}

fn select_teacher(
    previous: &[TransmissionAgent],
    candidates: &[usize],
    temperature: f64,
    rng: &mut DeterministicRng,
) -> usize {
    let maximum = candidates.iter()
        .map(|&index| previous[index].demonstration_score)
        .fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = candidates.iter()
        .map(|&index| ((previous[index].demonstration_score - maximum) / temperature).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return candidates[0];
    }
    let mut draw = rng.uniform() * total;
    for (&candidate, weight) in candidates.iter().zip(weights) {
        if draw < weight { return candidate; }
        draw -= weight;
    }
    *candidates.last().expect("teacher candidate list is nonempty")
}

fn integrate_strategy(individual: Strategy, social: Option<Strategy>) -> Strategy {
    match social {
        Some(strategy) if strategy.rank() > individual.rank() => strategy,
        _ => individual,
    }
}

/// Run one parameterized generational experiment.
///
/// The implementation is a transparent reference model inspired by the
/// article's described mechanism sequence. It does not reproduce the exact
/// published ABM until its supplementary parameterization, implementation,
/// and archived outputs are ingested.
pub fn simulate_transmission(
    config: TransmissionConfig,
    seed: u64,
) -> Result<TransmissionRun, TransmissionError> {
    config.validate()?;
    let mut rng = DeterministicRng::new(seed);
    let mut generations = Vec::with_capacity(config.generations);

    // Seed generation: machines have a larger search budget and a separate
    // exploration bias; remaining slots are human individual learners.
    let mut initial_agents = Vec::with_capacity(config.population_size);
    for index in 0..config.population_size {
        let kind = if index < config.machine_count_generation_zero {
            AgentKind::Machine
        } else {
            AgentKind::Human
        };
        let (trials, bias) = match kind {
            AgentKind::Machine => (config.machine_learning_trials, config.machine_optimal_exploration_bias),
            AgentKind::Human => (config.human_learning_trials, config.human_optimal_exploration_bias),
        };
        let individual_strategy = discover_strategy(trials, bias, config, &mut rng);
        let score = demonstration_score(individual_strategy, config, &mut rng);
        initial_agents.push(TransmissionAgent {
            kind,
            individual_strategy,
            social_strategy: None,
            integrated_strategy: individual_strategy,
            teacher_index: None,
            social_acquisition_succeeded: false,
            demonstration_score: score,
        });
    }
    generations.push(PopulationGeneration { index: 0, agents: initial_agents });

    for generation_index in 1..config.generations {
        let previous = &generations[generation_index - 1].agents;
        let mut agents = Vec::with_capacity(config.population_size);
        for _ in 0..config.population_size {
            let individual_strategy = discover_strategy(
                config.human_learning_trials,
                config.human_optimal_exploration_bias,
                config,
                &mut rng,
            );
            let candidates = sample_teacher_candidates(previous, config.teacher_candidate_count, &mut rng);
            let teacher_index = select_teacher(
                previous,
                &candidates,
                config.teacher_selection_temperature,
                &mut rng,
            );
            let acquired = rng.bernoulli(config.social_acquisition_probability);
            let social_strategy = if acquired {
                Some(previous[teacher_index].integrated_strategy)
            } else {
                None
            };
            let integrated_strategy = integrate_strategy(individual_strategy, social_strategy);
            let score = demonstration_score(integrated_strategy, config, &mut rng);
            agents.push(TransmissionAgent {
                kind: AgentKind::Human,
                individual_strategy,
                social_strategy,
                integrated_strategy,
                teacher_index: Some(teacher_index),
                social_acquisition_succeeded: acquired,
                demonstration_score: score,
            });
        }
        generations.push(PopulationGeneration { index: generation_index, agents });
    }

    Ok(TransmissionRun { generations })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn high_transmission_config() -> TransmissionConfig {
        TransmissionConfig {
            generations: 4,
            population_size: 8,
            machine_count_generation_zero: 1,
            human_learning_trials: 1,
            machine_learning_trials: 1,
            human_optimal_exploration_bias: 0.0,
            machine_optimal_exploration_bias: 1.0,
            optimal_discovery_probability_per_trial: 1.0,
            myopic_discovery_probability_per_trial: 0.0,
            social_acquisition_probability: 1.0,
            teacher_candidate_count: 8,
            teacher_selection_temperature: 0.01,
            demonstration_count: 10,
            reward_noise_epsilon: 0.0,
        }
    }

    #[test]
    fn optimal_strategy_dominates_myopic_when_both_are_acquired() {
        assert_eq!(integrate_strategy(Strategy::Myopic, Some(Strategy::Optimal)), Strategy::Optimal);
        assert_eq!(integrate_strategy(Strategy::Optimal, Some(Strategy::Myopic)), Strategy::Optimal);
    }

    #[test]
    fn high_performance_machine_seed_is_acquired_and_retained() {
        let run = simulate_transmission(high_transmission_config(), 7).unwrap();
        assert_eq!(run.generations[0].agents[0].kind, AgentKind::Machine);
        assert_eq!(run.generations[0].agents[0].integrated_strategy, Strategy::Optimal);
        for generation in run.generations.iter().skip(1) {
            assert!(generation.agents.iter().all(|agent| agent.integrated_strategy == Strategy::Optimal));
            assert_eq!(generation.metrics().optimal_prevalence, 1.0);
        }
        assert!(run.optimal_present_in_every_post_seed_generation());
    }

    #[test]
    fn no_social_acquisition_and_no_individual_discovery_produces_no_adoption() {
        let config = TransmissionConfig {
            generations: 3,
            population_size: 6,
            machine_count_generation_zero: 1,
            human_learning_trials: 1,
            machine_learning_trials: 1,
            human_optimal_exploration_bias: 1.0,
            machine_optimal_exploration_bias: 1.0,
            optimal_discovery_probability_per_trial: 0.0,
            myopic_discovery_probability_per_trial: 0.0,
            social_acquisition_probability: 0.0,
            teacher_candidate_count: 3,
            teacher_selection_temperature: 0.5,
            demonstration_count: 10,
            reward_noise_epsilon: 0.0,
        };
        let run = simulate_transmission(config, 11).unwrap();
        assert!(run.generations.iter().all(|generation| {
            generation.agents.iter().all(|agent| agent.integrated_strategy == Strategy::None)
        }));
    }

    #[test]
    fn fixed_seed_reproduces_the_entire_transmission_run() {
        let config = TransmissionConfig::default();
        assert_eq!(simulate_transmission(config, 2026).unwrap(), simulate_transmission(config, 2026).unwrap());
    }

    #[test]
    fn invalid_probability_and_temperature_are_rejected() {
        let mut config = TransmissionConfig::default();
        config.social_acquisition_probability = 1.5;
        assert_eq!(
            simulate_transmission(config, 1),
            Err(TransmissionError::InvalidProbability { name: "social acquisition probability" })
        );

        let mut config = TransmissionConfig::default();
        config.teacher_selection_temperature = 0.0;
        assert_eq!(simulate_transmission(config, 1), Err(TransmissionError::InvalidTemperature));
    }
}
