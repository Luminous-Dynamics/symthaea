// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#![forbid(unsafe_code)]

//! Parameterized generational cultural-transmission reference model.
//!
//! This model separates individual discovery, demonstrator selection,
//! demonstration observation, actual strategy acquisition, and integration.
//! It is inspired by the mechanism sequence in Brinkmann et al. (2026), but
//! does not reproduce their published ABM parameters or code.

use core::fmt;

/// Small strategy repertoire for the reference task.
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

    fn demonstration_reward(
        self,
        demonstrations: u32,
        myopic_reward: f64,
        optimal_reward: f64,
    ) -> f64 {
        let reward_per_demo = match self {
            Self::None => 0.0,
            Self::Myopic => myopic_reward,
            Self::Optimal => optimal_reward,
        };
        f64::from(demonstrations) * reward_per_demo
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentKind {
    Human,
    Machine,
}

/// Selection mechanisms are explicit experimental treatments.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TeacherSelectionRule {
    Uniform,
    PerformanceSoftmax { temperature: f64 },
}

/// Complete agent record retained for benchmark auditing.
#[derive(Debug, Clone, PartialEq)]
pub struct TransmissionAgent {
    pub kind: AgentKind,
    /// Strategy found through this agent's own individual-learning trials.
    pub individual_strategy: Strategy,
    /// Acquired from the selected teacher, if the demonstration was observed
    /// and the teacher possessed a non-empty strategy.
    pub social_strategy: Option<Strategy>,
    /// Final strategy after comparing individual and social strategies.
    pub integrated_strategy: Strategy,
    pub teacher_index: Option<usize>,
    /// What the selected teacher had available to demonstrate; None means
    /// the teacher had no strategy, not that the learner acquired nothing.
    pub teacher_strategy: Option<Strategy>,
    /// True if the learner observed/received the selected demonstration.
    pub demonstration_observed: bool,
    /// True if a non-empty teacher strategy was successfully acquired.
    /// A successful acquisition may still be corrupted to Strategy::None.
    pub strategy_acquisition_succeeded: bool,
    /// True if the acquired strategy was integrated into the learner's
    /// active repertoire rather than merely retained as an unintegrated trace.
    pub strategy_integrated: bool,
    /// Whether selected teacher had the highest demo score among candidates.
    pub selected_top_teacher: Option<bool>,
    /// Score offered to future demonstrators.
    pub demonstration_score: f64,
}

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
    pub individually_discovered_optimal_count: usize,
    pub demonstration_observations: usize,
    pub social_knowledge_acquisitions: usize,
    pub social_optimal_acquisitions: usize,
    /// Optimal strategies acquired by learners who did not discover them individually.
    pub social_only_optimal_acquisitions: usize,
    pub social_integrations: usize,
    /// Optimal strategies newly integrated by learners who did not discover them individually.
    pub social_only_optimal_integrations: usize,
    pub social_optimal_integrations: usize,
    pub selected_top_teacher_count: usize,
    pub demonstration_observation_rate: Option<f64>,
    pub social_acquisition_rate_given_observation: Option<f64>,
    pub optimal_acquisition_rate_given_observation: Option<f64>,
    pub integration_rate_given_acquisition: Option<f64>,
    pub optimal_integration_rate_given_observation: Option<f64>,
    pub top_teacher_selection_rate: Option<f64>,
}

impl PopulationGeneration {
    pub fn metrics(&self) -> GenerationMetrics {
        let optimal_count = self.agents.iter()
            .filter(|agent| agent.integrated_strategy == Strategy::Optimal).count();
        let myopic_count = self.agents.iter()
            .filter(|agent| agent.integrated_strategy == Strategy::Myopic).count();
        let uninformed_count = self.agents.iter()
            .filter(|agent| agent.integrated_strategy == Strategy::None).count();
        let individually_discovered_optimal_count = self.agents.iter()
            .filter(|agent| agent.individual_strategy == Strategy::Optimal).count();
        let demonstration_observations = self.agents.iter()
            .filter(|agent| agent.demonstration_observed).count();
        let social_knowledge_acquisitions = self.agents.iter()
            .filter(|agent| agent.strategy_acquisition_succeeded).count();
        let social_optimal_acquisitions = self.agents.iter()
            .filter(|agent| agent.strategy_acquisition_succeeded
                && agent.social_strategy == Some(Strategy::Optimal)).count();
        let social_only_optimal_acquisitions = self.agents.iter()
            .filter(|agent| agent.strategy_acquisition_succeeded
                && agent.individual_strategy != Strategy::Optimal
                && agent.social_strategy == Some(Strategy::Optimal)).count();
        let social_integrations = self.agents.iter()
            .filter(|agent| agent.strategy_integrated).count();
        let social_optimal_integrations = self.agents.iter()
            .filter(|agent| agent.strategy_integrated
                && agent.social_strategy == Some(Strategy::Optimal)).count();
        let social_only_optimal_integrations = self.agents.iter()
            .filter(|agent| agent.strategy_integrated
                && agent.individual_strategy != Strategy::Optimal
                && agent.social_strategy == Some(Strategy::Optimal)
                && agent.integrated_strategy == Strategy::Optimal).count();
        let selected_top_teacher_count = self.agents.iter()
            .filter(|agent| agent.selected_top_teacher == Some(true)).count();
        let teacher_selection_trials = self.agents.iter()
            .filter(|agent| agent.selected_top_teacher.is_some()).count();
        let total_agents = self.agents.len();

        GenerationMetrics {
            generation: self.index,
            total_agents,
            optimal_count,
            myopic_count,
            uninformed_count,
            optimal_prevalence: optimal_count as f64 / total_agents as f64,
            individually_discovered_optimal_count,
            demonstration_observations,
            social_knowledge_acquisitions,
            social_optimal_acquisitions,
            social_only_optimal_acquisitions,
            social_integrations,
            social_optimal_integrations,
            social_only_optimal_integrations,
            selected_top_teacher_count,
            demonstration_observation_rate: if self.index == 0 {
                None
            } else {
                Some(demonstration_observations as f64 / total_agents as f64)
            },
            social_acquisition_rate_given_observation: ratio(
                social_knowledge_acquisitions,
                demonstration_observations,
            ),
            optimal_acquisition_rate_given_observation: ratio(
                social_optimal_acquisitions,
                demonstration_observations,
            ),
            integration_rate_given_acquisition: ratio(
                social_integrations,
                social_knowledge_acquisitions,
            ),
            optimal_integration_rate_given_observation: ratio(
                social_optimal_integrations,
                demonstration_observations,
            ),
            top_teacher_selection_rate: ratio(
                selected_top_teacher_count,
                teacher_selection_trials,
            ),
        }
    }
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

/// Parameters for a transparent generational transmission experiment.
///
/// The seed generation and later generations have separate human-learning
/// budgets so individual discovery opportunity is not implicitly conflated
/// across generations. These are experimental controls, not assumed universal
/// properties of human learning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransmissionConfig {
    pub generations: usize,
    pub population_size: usize,
    pub machine_count_generation_zero: usize,
    pub human_learning_trials_generation_zero: u32,
    pub human_learning_trials_later_generations: u32,
    pub machine_learning_trials: u32,
    pub human_optimal_exploration_bias: f64,
    pub machine_optimal_exploration_bias: f64,
    pub optimal_discovery_probability_per_trial: f64,
    pub myopic_discovery_probability_per_trial: f64,
    /// Probability that a learner observes/receives its selected demonstration.
    pub social_observation_probability: f64,
    /// Probability of acquiring a strategy conditional on observing it.
    pub social_strategy_acquisition_probability: f64,
    /// Probability that the received strategy is corrupted/downgraded.
    pub demonstration_corruption_probability: f64,
    /// Probability that an acquired strategy is integrated into active use.
    pub strategy_integration_probability: f64,
    pub teacher_candidate_count: usize,
    pub teacher_selection: TeacherSelectionRule,
    /// Expected per-demonstration reward signal for a myopic strategy.
    pub myopic_demonstration_reward: f64,
    /// Expected per-demonstration reward signal for an optimal strategy.
    /// The difference from the myopic reward is the demonstrable selective advantage.
    pub optimal_demonstration_reward: f64,
    pub demonstration_count: u32,
    /// Standard deviation per demonstration is epsilon; aggregate noise is
    /// epsilon * sqrt(demonstration_count).
    pub reward_noise_epsilon: f64,
}

impl Default for TransmissionConfig {
    fn default() -> Self {
        Self {
            generations: 5,
            population_size: 8,
            machine_count_generation_zero: 3,
            human_learning_trials_generation_zero: 6,
            human_learning_trials_later_generations: 2,
            machine_learning_trials: 1_000,
            human_optimal_exploration_bias: 0.1,
            machine_optimal_exploration_bias: 0.9,
            optimal_discovery_probability_per_trial: 0.001,
            myopic_discovery_probability_per_trial: 0.5,
            social_observation_probability: 0.8,
            social_strategy_acquisition_probability: 0.8,
            demonstration_corruption_probability: 0.05,
            strategy_integration_probability: 0.8,
            teacher_candidate_count: 5,
            teacher_selection: TeacherSelectionRule::PerformanceSoftmax {
                temperature: 0.25,
            },
            myopic_demonstration_reward: 0.5,
            optimal_demonstration_reward: 1.0,
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
    InvalidRewardSignal { name: &'static str },
}

impl fmt::Display for TransmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroGenerations => write!(f, "generations must be positive"),
            Self::ZeroPopulation => write!(f, "population size must be positive"),
            Self::TooManyMachines { machines, population } => {
                write!(f, "{machines} machines exceed population size {population}")
            }
            Self::ZeroLearningTrials => write!(f, "individual-learning trial budgets must be positive"),
            Self::ZeroTeacherCandidates => write!(f, "teacher candidate count must be positive when generations > 1"),
            Self::TooManyTeacherCandidates { candidates, population } => {
                write!(f, "{candidates} teacher candidates exceed prior generation size {population}")
            }
            Self::ZeroDemonstrations => write!(f, "demonstration count must be positive"),
            Self::InvalidProbability { name } => write!(f, "{name} must be finite and within [0, 1]"),
            Self::InvalidBias { name } => write!(f, "{name} must be finite and within [0, 1]"),
            Self::InvalidTemperature => write!(f, "teacher-selection temperature must be finite and positive"),
            Self::InvalidNoiseScale => write!(f, "reward noise epsilon must be finite and non-negative"),
            Self::InvalidRewardSignal { name } => write!(f, "{name} must be finite and non-negative"),
        }
    }
}
impl std::error::Error for TransmissionError {}

impl TransmissionConfig {
    pub fn validate(self) -> Result<(), TransmissionError> {
        if self.generations == 0 {
            return Err(TransmissionError::ZeroGenerations);
        }
        if self.population_size == 0 {
            return Err(TransmissionError::ZeroPopulation);
        }
        if self.machine_count_generation_zero > self.population_size {
            return Err(TransmissionError::TooManyMachines {
                machines: self.machine_count_generation_zero,
                population: self.population_size,
            });
        }
        if self.human_learning_trials_generation_zero == 0
            || self.human_learning_trials_later_generations == 0
            || self.machine_learning_trials == 0
        {
            return Err(TransmissionError::ZeroLearningTrials);
        }
        if self.generations > 1 {
            if self.teacher_candidate_count == 0 {
                return Err(TransmissionError::ZeroTeacherCandidates);
            }
            if self.teacher_candidate_count > self.population_size {
                return Err(TransmissionError::TooManyTeacherCandidates {
                    candidates: self.teacher_candidate_count,
                    population: self.population_size,
                });
            }
        }
        if self.demonstration_count == 0 {
            return Err(TransmissionError::ZeroDemonstrations);
        }
        for (name, value) in [
            ("optimal discovery probability", self.optimal_discovery_probability_per_trial),
            ("myopic discovery probability", self.myopic_discovery_probability_per_trial),
            ("social observation probability", self.social_observation_probability),
            ("social strategy acquisition probability", self.social_strategy_acquisition_probability),
            ("demonstration corruption probability", self.demonstration_corruption_probability),
            ("strategy integration probability", self.strategy_integration_probability),
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
        if let TeacherSelectionRule::PerformanceSoftmax { temperature } = self.teacher_selection {
            if !temperature.is_finite() || temperature <= 0.0 {
                return Err(TransmissionError::InvalidTemperature);
            }
        }
        if !self.reward_noise_epsilon.is_finite() || self.reward_noise_epsilon < 0.0 {
            return Err(TransmissionError::InvalidNoiseScale);
        }
        for (name, value) in [
            ("myopic demonstration reward", self.myopic_demonstration_reward),
            ("optimal demonstration reward", self.optimal_demonstration_reward),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(TransmissionError::InvalidRewardSignal { name });
            }
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

    /// At least one optimal strategy appears in every post-seed generation.
    /// This model-side descriptor is not the paper's raw-data classification.
    pub fn optimal_present_in_every_post_seed_generation(&self) -> bool {
        if self.generations.len() < 2 {
            return false;
        }
        self.generations.iter().skip(1).all(|generation| {
            generation.agents.iter()
                .any(|agent| agent.integrated_strategy == Strategy::Optimal)
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

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
    let optimal_probability = 1.0
        - (1.0 - config.optimal_discovery_probability_per_trial).powf(optimal_trials);
    let myopic_probability = 1.0
        - (1.0 - config.myopic_discovery_probability_per_trial).powf(myopic_trials);

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
    let base = strategy.demonstration_reward(
        config.demonstration_count,
        config.myopic_demonstration_reward,
        config.optimal_demonstration_reward,
    );
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
    rule: TeacherSelectionRule,
    rng: &mut DeterministicRng,
) -> (usize, bool) {
    let maximum = candidates.iter()
        .map(|&index| previous[index].demonstration_score)
        .fold(f64::NEG_INFINITY, f64::max);
    let selected = match rule {
        TeacherSelectionRule::Uniform => candidates[rng.index(candidates.len())],
        TeacherSelectionRule::PerformanceSoftmax { temperature } => {
            let weights: Vec<f64> = candidates.iter()
                .map(|&index| {
                    ((previous[index].demonstration_score - maximum) / temperature).exp()
                })
                .collect();
            let total: f64 = weights.iter().sum();
            if !total.is_finite() || total <= 0.0 {
                candidates[0]
            } else {
                let mut draw = rng.uniform() * total;
                let mut selected = *candidates.last().expect("candidate list is nonempty");
                for (&candidate, weight) in candidates.iter().zip(weights) {
                    if draw < weight {
                        selected = candidate;
                        break;
                    }
                    draw -= weight;
                }
                selected
            }
        }
    };
    let selected_score = previous[selected].demonstration_score;
    (selected, selected_score >= maximum)
}

fn integrate_strategy(
    individual: Strategy,
    social: Option<Strategy>,
    integration_succeeded: bool,
) -> Strategy {
    if !integration_succeeded {
        return individual;
    }
    match social {
        Some(strategy) if strategy.rank() > individual.rank() => strategy,
        _ => individual,
    }
}

/// One-step lossy transmission model. A corrupted optimal strategy becomes
/// myopic; a corrupted myopic strategy loses its executable rule entirely.
fn corrupt_strategy(strategy: Strategy) -> Strategy {
    match strategy {
        Strategy::Optimal => Strategy::Myopic,
        Strategy::Myopic | Strategy::None => Strategy::None,
    }
}

/// Run one parameterized generational experiment.
///
/// The model is deliberately simplified and auditable. It is not a reproduction
/// of the article's ABM until the supplementary parameterization, implementation
/// and archived outputs are ingested and independently compared.
pub fn simulate_transmission(
    config: TransmissionConfig,
    seed: u64,
) -> Result<TransmissionRun, TransmissionError> {
    config.validate()?;
    let mut rng = DeterministicRng::new(seed);
    let mut generations = Vec::with_capacity(config.generations);

    let mut initial_agents = Vec::with_capacity(config.population_size);
    for index in 0..config.population_size {
        let kind = if index < config.machine_count_generation_zero {
            AgentKind::Machine
        } else {
            AgentKind::Human
        };
        let (trials, bias) = match kind {
            AgentKind::Machine => (
                config.machine_learning_trials,
                config.machine_optimal_exploration_bias,
            ),
            AgentKind::Human => (
                config.human_learning_trials_generation_zero,
                config.human_optimal_exploration_bias,
            ),
        };
        let individual_strategy = discover_strategy(trials, bias, config, &mut rng);
        let score = demonstration_score(individual_strategy, config, &mut rng);
        initial_agents.push(TransmissionAgent {
            kind,
            individual_strategy,
            social_strategy: None,
            integrated_strategy: individual_strategy,
            teacher_index: None,
            teacher_strategy: None,
            demonstration_observed: false,
            strategy_acquisition_succeeded: false,
            strategy_integrated: false,
            selected_top_teacher: None,
            demonstration_score: score,
        });
    }
    generations.push(PopulationGeneration {
        index: 0,
        agents: initial_agents,
    });

    for generation_index in 1..config.generations {
        let previous = &generations[generation_index - 1].agents;
        let mut agents = Vec::with_capacity(config.population_size);
        for _ in 0..config.population_size {
            let individual_strategy = discover_strategy(
                config.human_learning_trials_later_generations,
                config.human_optimal_exploration_bias,
                config,
                &mut rng,
            );
            let candidates = sample_teacher_candidates(
                previous,
                config.teacher_candidate_count,
                &mut rng,
            );
            let (teacher_index, selected_top_teacher) = select_teacher(
                previous,
                &candidates,
                config.teacher_selection,
                &mut rng,
            );
            let teacher_strategy = previous[teacher_index].integrated_strategy;
            let demonstration_observed = rng.bernoulli(config.social_observation_probability);
            let strategy_acquisition_succeeded = demonstration_observed
                && teacher_strategy != Strategy::None
                && rng.bernoulli(config.social_strategy_acquisition_probability);
            let social_strategy = if strategy_acquisition_succeeded {
                let received = if rng.bernoulli(config.demonstration_corruption_probability) {
                    corrupt_strategy(teacher_strategy)
                } else {
                    teacher_strategy
                };
                Some(received)
            } else {
                None
            };
            let integration_gate_passed = strategy_acquisition_succeeded
                && rng.bernoulli(config.strategy_integration_probability);
            let integrated_strategy = integrate_strategy(
                individual_strategy,
                social_strategy,
                integration_gate_passed,
            );
            // Count integration only when the received strategy changes the
            // active repertoire, not when it duplicates or loses to prior skill.
            let strategy_integrated = integration_gate_passed
                && social_strategy
                    .map(|strategy| strategy.rank() > individual_strategy.rank())
                    .unwrap_or(false);
            let score = demonstration_score(integrated_strategy, config, &mut rng);
            agents.push(TransmissionAgent {
                kind: AgentKind::Human,
                individual_strategy,
                social_strategy,
                integrated_strategy,
                teacher_index: Some(teacher_index),
                teacher_strategy: Some(teacher_strategy),
                demonstration_observed,
                strategy_acquisition_succeeded,
                strategy_integrated,
                selected_top_teacher: Some(selected_top_teacher),
                demonstration_score: score,
            });
        }
        generations.push(PopulationGeneration {
            index: generation_index,
            agents,
        });
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
            human_learning_trials_generation_zero: 1,
            human_learning_trials_later_generations: 1,
            machine_learning_trials: 1,
            human_optimal_exploration_bias: 0.0,
            machine_optimal_exploration_bias: 1.0,
            optimal_discovery_probability_per_trial: 1.0,
            myopic_discovery_probability_per_trial: 0.0,
            social_observation_probability: 1.0,
            social_strategy_acquisition_probability: 1.0,
            demonstration_corruption_probability: 0.0,
            strategy_integration_probability: 1.0,
            teacher_candidate_count: 8,
            teacher_selection: TeacherSelectionRule::PerformanceSoftmax {
                temperature: 0.01,
            },
            myopic_demonstration_reward: 0.5,
            optimal_demonstration_reward: 1.0,
            demonstration_count: 10,
            reward_noise_epsilon: 0.0,
        }
    }

    #[test]
    fn optimal_strategy_dominates_myopic_when_both_are_acquired() {
        assert_eq!(
            integrate_strategy(Strategy::Myopic, Some(Strategy::Optimal), true),
            Strategy::Optimal
        );
        assert_eq!(
            integrate_strategy(Strategy::Optimal, Some(Strategy::Myopic), true),
            Strategy::Optimal
        );
    }

    #[test]
    fn high_performance_machine_seed_is_acquired_and_retained() {
        let run = simulate_transmission(high_transmission_config(), 7).unwrap();
        assert_eq!(run.generations[0].agents[0].kind, AgentKind::Machine);
        assert_eq!(
            run.generations[0].agents[0].integrated_strategy,
            Strategy::Optimal
        );
        for generation in run.generations.iter().skip(1) {
            assert!(generation.agents.iter().all(|agent| {
                agent.integrated_strategy == Strategy::Optimal
                    && agent.social_strategy == Some(Strategy::Optimal)
                    && agent.demonstration_observed
                    && agent.strategy_acquisition_succeeded
                    && agent.strategy_integrated
            }));
            assert_eq!(generation.metrics().optimal_prevalence, 1.0);
            assert_eq!(generation.metrics().social_acquisition_rate_given_observation, Some(1.0));
        }
        assert!(run.optimal_present_in_every_post_seed_generation());
    }

    #[test]
    fn observed_empty_teacher_is_not_counted_as_knowledge_acquisition() {
        let generation = PopulationGeneration {
            index: 1,
            agents: vec![TransmissionAgent {
                kind: AgentKind::Human,
                individual_strategy: Strategy::None,
                social_strategy: None,
                integrated_strategy: Strategy::None,
                teacher_index: Some(0),
                teacher_strategy: Some(Strategy::None),
                demonstration_observed: true,
                strategy_acquisition_succeeded: false,
                strategy_integrated: false,
                selected_top_teacher: Some(true),
                demonstration_score: 0.0,
            }],
        };
        let metrics = generation.metrics();
        assert_eq!(metrics.demonstration_observations, 1);
        assert_eq!(metrics.social_knowledge_acquisitions, 0);
        assert_eq!(metrics.social_acquisition_rate_given_observation, Some(0.0));
        assert_eq!(metrics.top_teacher_selection_rate, Some(1.0));
    }

    #[test]
    fn no_social_acquisition_and_no_individual_discovery_produces_no_adoption() {
        let config = TransmissionConfig {
            generations: 3,
            population_size: 6,
            machine_count_generation_zero: 1,
            human_learning_trials_generation_zero: 1,
            human_learning_trials_later_generations: 1,
            machine_learning_trials: 1,
            human_optimal_exploration_bias: 1.0,
            machine_optimal_exploration_bias: 1.0,
            optimal_discovery_probability_per_trial: 0.0,
            myopic_discovery_probability_per_trial: 0.0,
            social_observation_probability: 0.0,
            social_strategy_acquisition_probability: 1.0,
            demonstration_corruption_probability: 0.0,
            strategy_integration_probability: 1.0,
            teacher_candidate_count: 3,
            teacher_selection: TeacherSelectionRule::Uniform,
            myopic_demonstration_reward: 0.5,
            optimal_demonstration_reward: 1.0,
            demonstration_count: 10,
            reward_noise_epsilon: 0.0,
        };
        let run = simulate_transmission(config, 11).unwrap();
        assert!(run.generations.iter().all(|generation| {
            generation.agents.iter().all(|agent| {
                agent.integrated_strategy == Strategy::None
                    && agent.social_strategy.is_none()
            })
        }));
        assert_eq!(
            run.generations[1].metrics().social_acquisition_rate_given_observation,
            None
        );
    }

    #[test]
    fn acquisition_and_integration_are_separate_stages() {
        assert_eq!(
            integrate_strategy(Strategy::None, Some(Strategy::Optimal), false),
            Strategy::None
        );
        assert_eq!(
            integrate_strategy(Strategy::None, Some(Strategy::Optimal), true),
            Strategy::Optimal
        );
    }

    #[test]
    fn corrupted_demo_is_recorded_separately_from_teacher_strategy() {
        let config = TransmissionConfig {
            generations: 2,
            demonstration_corruption_probability: 1.0,
            ..high_transmission_config()
        };
        let run = simulate_transmission(config, 7).unwrap();
        for agent in &run.generations[1].agents {
            assert_eq!(agent.teacher_strategy, Some(Strategy::Optimal));
            assert_eq!(agent.social_strategy, Some(Strategy::Myopic));
            assert_eq!(agent.integrated_strategy, Strategy::Myopic);
            assert!(agent.strategy_acquisition_succeeded);
            assert!(agent.strategy_integrated);
        }
    }

    #[test]
    fn acquired_strategy_can_remain_unintegrated() {
        let config = TransmissionConfig {
            generations: 2,
            strategy_integration_probability: 0.0,
            ..high_transmission_config()
        };
        let run = simulate_transmission(config, 7).unwrap();
        for agent in &run.generations[1].agents {
            assert_eq!(agent.social_strategy, Some(Strategy::Optimal));
            assert!(agent.strategy_acquisition_succeeded);
            assert!(!agent.strategy_integrated);
            assert_eq!(agent.integrated_strategy, Strategy::None);
        }
    }

    #[test]
    fn corruption_downgrades_transmitted_strategy_without_rewriting_teacher_record() {
        assert_eq!(corrupt_strategy(Strategy::Optimal), Strategy::Myopic);
        assert_eq!(corrupt_strategy(Strategy::Myopic), Strategy::None);
        assert_eq!(corrupt_strategy(Strategy::None), Strategy::None);
    }

    #[test]
    fn redundant_social_optimal_acquisition_is_not_counted_as_integration() {
        let generation = PopulationGeneration {
            index: 1,
            agents: vec![TransmissionAgent {
                kind: AgentKind::Human,
                individual_strategy: Strategy::Optimal,
                social_strategy: Some(Strategy::Optimal),
                integrated_strategy: Strategy::Optimal,
                teacher_index: Some(0),
                teacher_strategy: Some(Strategy::Optimal),
                demonstration_observed: true,
                strategy_acquisition_succeeded: true,
                strategy_integrated: false,
                selected_top_teacher: Some(true),
                demonstration_score: 0.0,
            }],
        };
        let metrics = generation.metrics();
        assert_eq!(metrics.social_optimal_acquisitions, 1);
        assert_eq!(metrics.social_only_optimal_acquisitions, 0);
        assert_eq!(metrics.social_integrations, 0);
        assert_eq!(metrics.social_only_optimal_integrations, 0);
        assert_eq!(metrics.integration_rate_given_acquisition, Some(0.0));
    }

    #[test]
    fn social_only_optimal_acquisition_and_integration_are_reported() {
        let generation = PopulationGeneration {
            index: 1,
            agents: vec![TransmissionAgent {
                kind: AgentKind::Human,
                individual_strategy: Strategy::Myopic,
                social_strategy: Some(Strategy::Optimal),
                integrated_strategy: Strategy::Optimal,
                teacher_index: Some(0),
                teacher_strategy: Some(Strategy::Optimal),
                demonstration_observed: true,
                strategy_acquisition_succeeded: true,
                strategy_integrated: true,
                selected_top_teacher: Some(true),
                demonstration_score: 0.0,
            }],
        };
        let metrics = generation.metrics();
        assert_eq!(metrics.social_only_optimal_acquisitions, 1);
        assert_eq!(metrics.social_only_optimal_integrations, 1);
        assert_eq!(metrics.social_optimal_integrations, 1);
    }

    #[test]
    fn observed_demonstration_can_fail_acquisition() {
        let generation = PopulationGeneration {
            index: 1,
            agents: vec![TransmissionAgent {
                kind: AgentKind::Human,
                individual_strategy: Strategy::None,
                social_strategy: None,
                integrated_strategy: Strategy::None,
                teacher_index: Some(0),
                teacher_strategy: Some(Strategy::Optimal),
                demonstration_observed: true,
                strategy_acquisition_succeeded: false,
                strategy_integrated: false,
                selected_top_teacher: Some(true),
                demonstration_score: 0.0,
            }],
        };
        let metrics = generation.metrics();
        assert_eq!(metrics.demonstration_observations, 1);
        assert_eq!(metrics.social_knowledge_acquisitions, 0);
        assert_eq!(metrics.social_acquisition_rate_given_observation, Some(0.0));
    }

    #[test]
    fn performance_softmax_selects_best_teacher_at_low_temperature() {
        let previous = vec![
            TransmissionAgent {
                kind: AgentKind::Human,
                individual_strategy: Strategy::Myopic,
                social_strategy: None,
                integrated_strategy: Strategy::Myopic,
                teacher_index: None,
                teacher_strategy: None,
                demonstration_observed: false,
                strategy_acquisition_succeeded: false,
                strategy_integrated: false,
                selected_top_teacher: None,
                demonstration_score: 0.0,
            },
            TransmissionAgent {
                kind: AgentKind::Machine,
                individual_strategy: Strategy::Optimal,
                social_strategy: None,
                integrated_strategy: Strategy::Optimal,
                teacher_index: None,
                teacher_strategy: None,
                demonstration_observed: false,
                strategy_acquisition_succeeded: false,
                strategy_integrated: false,
                selected_top_teacher: None,
                demonstration_score: 10.0,
            },
        ];
        let mut rng = DeterministicRng::new(9);
        let (selected, is_top) = select_teacher(
            &previous,
            &[0, 1],
            TeacherSelectionRule::PerformanceSoftmax { temperature: 0.001 },
            &mut rng,
        );
        assert_eq!(selected, 1);
        assert!(is_top);
    }

    #[test]
    fn selective_advantage_is_an_explicit_demonstration_reward_gap() {
        let config = TransmissionConfig {
            myopic_demonstration_reward: 0.5,
            optimal_demonstration_reward: 0.75,
            reward_noise_epsilon: 0.0,
            ..TransmissionConfig::default()
        };
        let mut rng = DeterministicRng::new(99);
        assert_eq!(
            demonstration_score(Strategy::Myopic, config, &mut rng),
            5.0
        );
        assert_eq!(
            demonstration_score(Strategy::Optimal, config, &mut rng),
            7.5
        );
    }

    #[test]
    fn negative_demonstration_reward_signal_is_rejected() {
        let config = TransmissionConfig {
            optimal_demonstration_reward: -0.1,
            ..TransmissionConfig::default()
        };
        assert_eq!(
            simulate_transmission(config, 3),
            Err(TransmissionError::InvalidRewardSignal {
                name: "optimal demonstration reward",
            })
        );
    }

    #[test]
    fn fixed_seed_reproduces_the_entire_transmission_run() {
        let config = TransmissionConfig::default();
        assert_eq!(
            simulate_transmission(config, 2026).unwrap(),
            simulate_transmission(config, 2026).unwrap()
        );
    }

    #[test]
    fn one_generation_is_not_misreported_as_post_seed_retention() {
        let config = TransmissionConfig {
            generations: 1,
            ..high_transmission_config()
        };
        let run = simulate_transmission(config, 7).unwrap();
        assert!(!run.optimal_present_in_every_post_seed_generation());
    }

    #[test]
    fn invalid_probability_and_temperature_are_rejected() {
        let mut config = TransmissionConfig::default();
        config.social_observation_probability = 1.5;
        assert_eq!(
            simulate_transmission(config, 1),
            Err(TransmissionError::InvalidProbability {
                name: "social observation probability",
            })
        );

        let mut config = TransmissionConfig::default();
        config.teacher_selection =
            TeacherSelectionRule::PerformanceSoftmax { temperature: 0.0 };
        assert_eq!(
            simulate_transmission(config, 1),
            Err(TransmissionError::InvalidTemperature)
        );
    }
}
