// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#![forbid(unsafe_code)]

//! Source-informed Rust reference for the exploratory Reward Network III ABM.
//!
//! Primary source: the Zenodo v2.0 archive's abm/abm.ipynb notebook.
//! Notebook source SHA-256:
//! 49040b229968a5bbfa736d23b2d6fb687fadc9ae404a2225211fba636cba6999
//!
//! This is a semantic port with an explicit deterministic RNG, not a
//! bit-for-bit NumPy random-stream replay. It follows the notebook's state
//! transitions:
//! - sample optimal exploration attempts from Binomial(K_ind, q_opt);
//! - sample myopic attempts from the remainder;
//! - discover optimal, else myopic, else random using the notebook's
//!   cumulative probability construction;
//! - sample n_teacher distinct candidates and select by reward softmax;
//! - transmit the selected teacher's strategy with probability lambda;
//! - use strategy 0 (myopic) after failed social learning, as the notebook does;
//! - integrate by max(individual_strategy, social_strategy);
//! - score demonstrations as K_demo * R_strategy plus Normal(0,
//!   epsilon * sqrt(K_demo)) noise.
//!
//! This notebook is the ABM source of truth for this port. The separate
//! abm/model.py helper in the same archive has different semantics and an
//! apparent function-call arity mismatch; do not silently substitute it.

use core::fmt;

/// Strategy state values in the notebook: random=-1, myopic=0, optimal=1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i8)]
pub enum StrategyState {
    Random = -1,
    Myopic = 0,
    Optimal = 1,
}

impl StrategyState {
    pub const fn as_i8(self) -> i8 {
        self as i8
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PopulationCondition {
    Human,
    HumanMachineSeed,
    HumanMachineAll,
}

/// The archived notebook maps failed social acquisition to Myopic (0).
/// PreserveIndividual is an explicit counterfactual ablation, not the
/// archived reference behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SocialFailureFallback {
    ArchivedMyopicZero,
    PreserveIndividual,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    pub generations: usize,
    pub population_size: usize,
    pub machine_count: usize,
    pub condition: PopulationCondition,
    pub human_exploration_trials: u32,
    pub machine_exploration_trials: u32,
    /// Probability each exploration attempt is allocated to the optimal strategy.
    pub human_optimality_bias: f64,
    pub machine_optimality_bias: f64,
    /// Per-attempt probability of discovering an optimal strategy.
    pub optimal_discovery_probability: f64,
    /// Per-attempt probability of discovering a myopic strategy.
    pub myopic_discovery_probability: f64,
    /// Candidate teachers sampled without replacement for each learner.
    pub teacher_candidate_count: usize,
    /// Softmax temperature over previous-generation demonstration rewards.
    pub teacher_selection_temperature: f64,
    /// Probability of learning the selected teacher's strategy.
    pub social_learning_probability: f64,
    pub social_failure_fallback: SocialFailureFallback,
    pub demonstration_count: u32,
    pub random_reward_per_demonstration: f64,
    pub myopic_reward_per_demonstration: f64,
    pub optimal_reward_per_demonstration: f64,
    /// Actual demonstration noise SD is epsilon * sqrt(demonstration_count).
    pub demonstration_noise_epsilon: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            generations: 20,
            population_size: 20,
            machine_count: 5,
            condition: PopulationCondition::HumanMachineSeed,
            human_exploration_trials: 10,
            machine_exploration_trials: 10_000,
            human_optimality_bias: 0.01,
            machine_optimality_bias: 0.5,
            optimal_discovery_probability: 1e-3,
            myopic_discovery_probability: 0.5,
            teacher_candidate_count: 10,
            teacher_selection_temperature: 0.1,
            social_learning_probability: 0.25,
            social_failure_fallback: SocialFailureFallback::ArchivedMyopicZero,
            demonstration_count: 1,
            random_reward_per_demonstration: 0.0,
            myopic_reward_per_demonstration: 0.5,
            optimal_reward_per_demonstration: 1.0,
            demonstration_noise_epsilon: 0.1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    ZeroGenerations,
    ZeroPopulation,
    TooManyMachines { machines: usize, population: usize },
    ZeroExplorationTrials,
    ZeroTeacherCandidates,
    TooManyTeacherCandidates { candidates: usize, population: usize },
    ZeroDemonstrations,
    InvalidProbability { name: &'static str },
    InvalidBias { name: &'static str },
    InvalidTemperature,
    InvalidReward,
    InvalidNoise,
    NonFiniteDemonstrationReward,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroGenerations => write!(f, "generations must be positive"),
            Self::ZeroPopulation => write!(f, "population_size must be positive"),
            Self::TooManyMachines { machines, population } => {
                write!(f, "{machines} machine agents exceed population size {population}")
            }
            Self::ZeroExplorationTrials => write!(f, "exploration trial counts must be positive"),
            Self::ZeroTeacherCandidates => write!(f, "teacher candidate count must be positive"),
            Self::TooManyTeacherCandidates { candidates, population } => {
                write!(f, "{candidates} teacher candidates exceed population size {population}")
            }
            Self::ZeroDemonstrations => write!(f, "demonstration_count must be positive"),
            Self::InvalidProbability { name } => write!(f, "{name} must be finite and in [0, 1]"),
            Self::InvalidBias { name } => write!(f, "{name} must be finite and in [0, 1]"),
            Self::InvalidTemperature => write!(f, "teacher selection temperature must be finite and positive"),
            Self::InvalidReward => write!(f, "demonstration rewards must be finite and non-negative"),
            Self::InvalidNoise => write!(f, "demonstration noise epsilon must be finite and non-negative"),
            Self::NonFiniteDemonstrationReward => write!(f, "generated demonstration reward is non-finite"),
        }
    }
}
impl std::error::Error for ConfigError {}

impl Config {
    pub fn validate(self) -> Result<(), ConfigError> {
        if self.generations == 0 {
            return Err(ConfigError::ZeroGenerations);
        }
        if self.population_size == 0 {
            return Err(ConfigError::ZeroPopulation);
        }
        if self.machine_count > self.population_size {
            return Err(ConfigError::TooManyMachines {
                machines: self.machine_count,
                population: self.population_size,
            });
        }
        if self.human_exploration_trials == 0 || self.machine_exploration_trials == 0 {
            return Err(ConfigError::ZeroExplorationTrials);
        }
        if self.generations > 1 {
            if self.teacher_candidate_count == 0 {
                return Err(ConfigError::ZeroTeacherCandidates);
            }
            if self.teacher_candidate_count > self.population_size {
                return Err(ConfigError::TooManyTeacherCandidates {
                    candidates: self.teacher_candidate_count,
                    population: self.population_size,
                });
            }
        }
        if self.demonstration_count == 0 {
            return Err(ConfigError::ZeroDemonstrations);
        }
        for (name, value) in [
            ("optimal_discovery_probability", self.optimal_discovery_probability),
            ("myopic_discovery_probability", self.myopic_discovery_probability),
            ("social_learning_probability", self.social_learning_probability),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(ConfigError::InvalidProbability { name });
            }
        }
        for (name, value) in [
            ("human_optimality_bias", self.human_optimality_bias),
            ("machine_optimality_bias", self.machine_optimality_bias),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(ConfigError::InvalidBias { name });
            }
        }
        if !self.teacher_selection_temperature.is_finite()
            || self.teacher_selection_temperature <= 0.0
        {
            return Err(ConfigError::InvalidTemperature);
        }
        for value in [
            self.random_reward_per_demonstration,
            self.myopic_reward_per_demonstration,
            self.optimal_reward_per_demonstration,
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(ConfigError::InvalidReward);
            }
        }
        if !self.demonstration_noise_epsilon.is_finite()
            || self.demonstration_noise_epsilon < 0.0
        {
            return Err(ConfigError::InvalidNoise);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimalOrigin {
    HumanIndividualDiscovery,
    MachineIndividualDiscovery,
    HumanSocialLineage,
    MachineSocialLineage,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AgentState {
    pub is_machine: bool,
    pub individual_strategy: StrategyState,
    pub social_strategy: StrategyState,
    pub integrated_strategy: StrategyState,
    pub teacher_index: Option<usize>,
    pub social_learning_succeeded: bool,
    pub demonstration_reward: f64,
    pub optimal_origin: Option<OptimalOrigin>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Generation {
    pub index: usize,
    pub agents: Vec<AgentState>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerationMetrics {
    pub generation: usize,
    pub population_size: usize,
    pub human_count: usize,
    pub machine_count: usize,
    pub human_optimal_prevalence: Option<f64>,
    pub machine_optimal_prevalence: Option<f64>,
    pub human_myopic_prevalence: Option<f64>,
    pub human_random_prevalence: Option<f64>,
    pub human_individual_optimal_discovery_rate: Option<f64>,
    pub human_social_optimal_acquisition_rate: Option<f64>,
    pub human_social_only_optimal_integration_rate: Option<f64>,
    pub human_machine_lineage_optimal_prevalence: Option<f64>,
    pub mean_demonstration_reward: f64,
    pub social_learning_rate_observed: Option<f64>,
}

impl Generation {
    pub fn metrics(&self) -> GenerationMetrics {
        let human: Vec<&AgentState> = self.agents.iter().filter(|a| !a.is_machine).collect();
        let machine: Vec<&AgentState> = self.agents.iter().filter(|a| a.is_machine).collect();
        let human_count = human.len();
        let machine_count = machine.len();
        let rate = |numerator: usize, denominator: usize| {
            (denominator > 0).then(|| numerator as f64 / denominator as f64)
        };

        let human_optimal = human.iter()
            .filter(|a| a.integrated_strategy == StrategyState::Optimal).count();
        let human_myopic = human.iter()
            .filter(|a| a.integrated_strategy == StrategyState::Myopic).count();
        let human_random = human.iter()
            .filter(|a| a.integrated_strategy == StrategyState::Random).count();
        let machine_optimal = machine.iter()
            .filter(|a| a.integrated_strategy == StrategyState::Optimal).count();
        let human_individual = human.iter()
            .filter(|a| a.individual_strategy == StrategyState::Optimal).count();
        let human_social_optimal = human.iter()
            .filter(|a| a.social_learning_succeeded
                && a.social_strategy == StrategyState::Optimal).count();
        let human_social_only_integrated = human.iter()
            .filter(|a| a.integrated_strategy == StrategyState::Optimal
                && a.individual_strategy != StrategyState::Optimal
                && a.social_learning_succeeded
                && a.social_strategy == StrategyState::Optimal).count();
        let human_machine_lineage = human.iter()
            .filter(|a| a.integrated_strategy == StrategyState::Optimal
                && matches!(a.optimal_origin,
                    Some(OptimalOrigin::MachineIndividualDiscovery)
                    | Some(OptimalOrigin::MachineSocialLineage))).count();
        let mean_reward = if self.agents.is_empty() {
            0.0
        } else {
            self.agents.iter().map(|a| a.demonstration_reward).sum::<f64>()
                / self.agents.len() as f64
        };
        let social_successes = human.iter()
            .filter(|a| a.social_learning_succeeded).count();

        let social_opportunity_exists = self.index > 0;
        GenerationMetrics {
            generation: self.index,
            population_size: self.agents.len(),
            human_count,
            machine_count,
            human_optimal_prevalence: rate(human_optimal, human_count),
            machine_optimal_prevalence: rate(machine_optimal, machine_count),
            human_myopic_prevalence: rate(human_myopic, human_count),
            human_random_prevalence: rate(human_random, human_count),
            human_individual_optimal_discovery_rate: rate(human_individual, human_count),
            human_social_optimal_acquisition_rate: social_opportunity_exists
                .then(|| rate(human_social_optimal, human_count))
                .flatten(),
            human_social_only_optimal_integration_rate: social_opportunity_exists
                .then(|| rate(human_social_only_integrated, human_count))
                .flatten(),
            human_machine_lineage_optimal_prevalence: rate(human_machine_lineage, human_count),
            mean_demonstration_reward: mean_reward,
            social_learning_rate_observed: social_opportunity_exists
                .then(|| rate(social_successes, human_count))
                .flatten(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Run {
    pub seed: u64,
    pub generations: Vec<Generation>,
}

impl Run {
    pub fn metrics(&self) -> Vec<GenerationMetrics> {
        self.generations.iter().map(Generation::metrics).collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DeterministicRng {
    state: u64,
    has_spare_normal: bool,
    spare_normal_bits: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self { state: seed, has_spare_normal: false, spare_normal_bits: 0 }
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

    fn bernoulli(&mut self, p: f64) -> bool {
        self.uniform() < p
    }

    fn normal(&mut self) -> f64 {
        if self.has_spare_normal {
            self.has_spare_normal = false;
            return f64::from_bits(self.spare_normal_bits);
        }
        let u1 = self.uniform().max(f64::MIN_POSITIVE);
        let u2 = self.uniform();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = core::f64::consts::TAU * u2;
        let z0 = radius * angle.cos();
        let z1 = radius * angle.sin();
        self.spare_normal_bits = z1.to_bits();
        self.has_spare_normal = true;
        z0
    }

    fn index(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0);
        (self.next_u64() as usize) % upper
    }

}

/// Exact marginal terminal-strategy probabilities after integrating out the
/// notebook's latent Binomial(k_optimal ~ Binomial(K_ind, q_opt)) allocation.
/// This preserves the per-agent categorical outcome distribution without
/// spending O(K_ind) random draws per agent.
///
/// P(optimal) = 1 - (1 - q*d_opt)^K
/// P(random)  = [q(1-d_opt) + (1-q)(1-d_myo)]^K
/// P(myopic)  = (1 - q*d_opt)^K - P(random)
fn terminal_strategy_probabilities(
    trials: u32,
    optimality_bias: f64,
    d_optimal: f64,
    d_myopic: f64,
) -> [f64; 3] {
    let k = f64::from(trials);
    let p_no_optimal = (1.0 - optimality_bias * d_optimal).powf(k);
    let p_random_base = optimality_bias * (1.0 - d_optimal)
        + (1.0 - optimality_bias) * (1.0 - d_myopic);
    let p_random = p_random_base.powf(k);
    let p_optimal = (1.0 - p_no_optimal).clamp(0.0, 1.0);
    let p_myopic = (p_no_optimal - p_random).clamp(0.0, 1.0 - p_optimal);
    let p_random = (1.0 - p_optimal - p_myopic).clamp(0.0, 1.0);
    [p_random, p_myopic, p_optimal]
}

fn discover_individually(
    config: Config,
    is_machine: bool,
    rng: &mut DeterministicRng,
) -> StrategyState {
    let (trials, bias) = if is_machine {
        (config.machine_exploration_trials, config.machine_optimality_bias)
    } else {
        (config.human_exploration_trials, config.human_optimality_bias)
    };
    let [p_random, p_myopic, _p_optimal] = terminal_strategy_probabilities(
        trials,
        bias,
        config.optimal_discovery_probability,
        config.myopic_discovery_probability,
    );
    let draw = rng.uniform();
    if draw < p_random {
        StrategyState::Random
    } else if draw < p_random + p_myopic {
        StrategyState::Myopic
    } else {
        StrategyState::Optimal
    }
}

fn is_machine_agent(config: Config, generation: usize, agent_index: usize) -> bool {
    agent_index < config.machine_count
        && match config.condition {
            PopulationCondition::Human => false,
            PopulationCondition::HumanMachineSeed => generation == 0,
            PopulationCondition::HumanMachineAll => true,
        }
}

fn sample_teacher_candidates(
    population_size: usize,
    candidate_count: usize,
    rng: &mut DeterministicRng,
) -> Vec<usize> {
    let mut pool: Vec<usize> = (0..population_size).collect();
    for i in 0..candidate_count {
        let selected = i + rng.index(population_size - i);
        pool.swap(i, selected);
    }
    pool.truncate(candidate_count);
    pool
}

fn choose_teacher(
    previous: &[AgentState],
    candidates: &[usize],
    temperature: f64,
    rng: &mut DeterministicRng,
) -> usize {
    let maximum = candidates.iter()
        .map(|&index| previous[index].demonstration_reward)
        .fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = candidates.iter()
        .map(|&index| ((previous[index].demonstration_reward - maximum) / temperature).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    let mut draw = rng.uniform() * total;
    for (&candidate, weight) in candidates.iter().zip(weights) {
        if draw < weight { return candidate; }
        draw -= weight;
    }
    *candidates.last().expect("candidate list validated non-empty")
}

fn social_strategy(
    teacher_strategy: StrategyState,
    success_probability: f64,
    fallback: SocialFailureFallback,
    rng: &mut DeterministicRng,
) -> (StrategyState, bool) {
    if rng.bernoulli(success_probability) {
        (teacher_strategy, true)
    } else {
        let fallback_strategy = match fallback {
            SocialFailureFallback::ArchivedMyopicZero => StrategyState::Myopic,
            SocialFailureFallback::PreserveIndividual => StrategyState::Random,
        };
        (fallback_strategy, false)
    }
}

fn reward_demonstration(strategy: StrategyState, config: Config, rng: &mut DeterministicRng) -> f64 {
    let per_demo = match strategy {
        StrategyState::Random => config.random_reward_per_demonstration,
        StrategyState::Myopic => config.myopic_reward_per_demonstration,
        StrategyState::Optimal => config.optimal_reward_per_demonstration,
    };
    let base = f64::from(config.demonstration_count) * per_demo;
    let noise_sd = config.demonstration_noise_epsilon * f64::from(config.demonstration_count).sqrt();
    base + noise_sd * rng.normal()
}

fn optimal_origin(
    is_machine: bool,
    individual: StrategyState,
    social: StrategyState,
    social_learned: bool,
    teacher_origin: Option<OptimalOrigin>,
    integrated: StrategyState,
) -> Option<OptimalOrigin> {
    if integrated != StrategyState::Optimal { return None; }
    // A tie is attributed to individual discovery because the notebook's
    // max() integration does not preserve causal attribution.
    if individual == StrategyState::Optimal {
        return Some(if is_machine {
            OptimalOrigin::MachineIndividualDiscovery
        } else {
            OptimalOrigin::HumanIndividualDiscovery
        });
    }
    if social_learned && social == StrategyState::Optimal {
        return match teacher_origin {
            Some(OptimalOrigin::MachineIndividualDiscovery)
            | Some(OptimalOrigin::MachineSocialLineage) => Some(OptimalOrigin::MachineSocialLineage),
            _ => Some(OptimalOrigin::HumanSocialLineage),
        };
    }
    None
}

/// Run one source-informed ABM condition with a deterministic seed.
///
/// The random stream is not NumPy-compatible, but the transition distributions
/// and integration semantics are ported from the archived notebook.
pub fn simulate(config: Config, seed: u64) -> Result<Run, ConfigError> {
    config.validate()?;
    let mut rng = DeterministicRng::new(seed);
    let mut generations: Vec<Generation> = Vec::with_capacity(config.generations);

    for generation_index in 0..config.generations {
        let previous = if generation_index == 0 { None } else {
            Some(&generations[generation_index - 1].agents)
        };
        let mut agents = Vec::with_capacity(config.population_size);

        for agent_index in 0..config.population_size {
            let machine = is_machine_agent(config, generation_index, agent_index);
            let individual = discover_individually(config, machine, &mut rng);

            let (social, learned, teacher_index, teacher_origin) = if let Some(prev) = previous {
                let candidates = sample_teacher_candidates(
                    config.population_size,
                    config.teacher_candidate_count,
                    &mut rng,
                );
                let teacher_index = choose_teacher(
                    prev,
                    &candidates,
                    config.teacher_selection_temperature,
                    &mut rng,
                );
                let teacher = &prev[teacher_index];
                let (social, learned) = social_strategy(
                    teacher.integrated_strategy,
                    config.social_learning_probability,
                    config.social_failure_fallback,
                    &mut rng,
                );
                (social, learned, Some(teacher_index), teacher.optimal_origin)
            } else {
                (StrategyState::Random, false, None, None)
            };

            let integrated = if generation_index == 0 {
                individual
            } else {
                individual.max(social)
            };
            let origin = optimal_origin(
                machine, individual, social, learned, teacher_origin, integrated,
            );
            let demonstration_reward = reward_demonstration(integrated, config, &mut rng);
            if !demonstration_reward.is_finite() {
                return Err(ConfigError::NonFiniteDemonstrationReward);
            }

            agents.push(AgentState {
                is_machine: machine,
                individual_strategy: individual,
                social_strategy: social,
                integrated_strategy: integrated,
                teacher_index,
                social_learning_succeeded: learned,
                demonstration_reward,
                optimal_origin: origin,
            });
        }
        generations.push(Generation { index: generation_index, agents });
    }
    Ok(Run { seed, generations })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn forced_config() -> Config {
        Config {
            generations: 3,
            population_size: 8,
            machine_count: 2,
            condition: PopulationCondition::HumanMachineSeed,
            human_exploration_trials: 1,
            machine_exploration_trials: 1,
            human_optimality_bias: 0.0,
            machine_optimality_bias: 1.0,
            optimal_discovery_probability: 1.0,
            myopic_discovery_probability: 1.0,
            teacher_candidate_count: 8,
            teacher_selection_temperature: 0.001,
            social_learning_probability: 1.0,
            demonstration_count: 1,
            demonstration_noise_epsilon: 0.0,
            ..Config::default()
        }
    }

    #[test]
    fn defaults_match_archived_robustness_notebook_settings() {
        let c = Config::default();
        assert_eq!(c.generations, 20);
        assert_eq!(c.population_size, 20);
        assert_eq!(c.machine_count, 5);
        assert_eq!(c.human_exploration_trials, 10);
        assert_eq!(c.machine_exploration_trials, 10_000);
        assert_eq!(c.human_optimality_bias, 0.01);
        assert_eq!(c.machine_optimality_bias, 0.5);
        assert_eq!(c.myopic_discovery_probability, 0.5);
        assert_eq!(c.teacher_candidate_count, 10);
        assert_eq!(c.teacher_selection_temperature, 0.1);
        assert_eq!(c.demonstration_count, 1);
        assert_eq!(c.demonstration_noise_epsilon, 0.1);
    }

    #[test]
    fn closed_form_matches_explicit_binomial_marginalization() {
        fn binomial_probability(n: u32, k: u32, q: f64) -> f64 {
            let mut choose = 1.0;
            for j in 0..k {
                choose *= f64::from(n - j) / f64::from(j + 1);
            }
            choose * q.powf(f64::from(k)) * (1.0 - q).powf(f64::from(n - k))
        }

        for n in 1..=12 {
            for q in [0.0, 0.01, 0.2, 0.5, 0.99, 1.0] {
                for d_opt in [0.0_f64, 0.001, 0.1, 0.9, 1.0] {
                    for d_myo in [0.0_f64, 0.1, 0.5, 1.0] {
                        let mut explicit = [0.0_f64; 3];
                        for k_opt in 0..=n {
                            let pk = binomial_probability(n, k_opt, q);
                            let p_opt_cond = 1.0 - (1.0 - d_opt).powf(f64::from(k_opt));
                            let p_myo_cond = 1.0 - (1.0 - d_myo).powf(f64::from(n - k_opt));
                            explicit[2] += pk * p_opt_cond;
                            explicit[1] += pk * (1.0 - p_opt_cond) * p_myo_cond;
                            explicit[0] += pk * (1.0 - p_opt_cond) * (1.0 - p_myo_cond);
                        }
                        let exact = terminal_strategy_probabilities(n, q, d_opt, d_myo);
                        for index in 0..3 {
                            assert!(
                                (explicit[index] - exact[index]).abs() < 1e-10,
                                "n={n} q={q} d_opt={d_opt} d_myo={d_myo} category={index}: explicit={} exact={}",
                                explicit[index], exact[index]
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn individual_discovery_uses_budget_bias_and_cumulative_probability() {
        let mut rng = DeterministicRng::new(1);
        let c = Config {
            human_exploration_trials: 1,
            human_optimality_bias: 1.0,
            optimal_discovery_probability: 1.0,
            myopic_discovery_probability: 0.0,
            ..Config::default()
        };
        assert_eq!(discover_individually(c, false, &mut rng), StrategyState::Optimal);
        let c = Config {
            human_exploration_trials: 1,
            human_optimality_bias: 0.0,
            optimal_discovery_probability: 0.0,
            myopic_discovery_probability: 1.0,
            ..Config::default()
        };
        assert_eq!(discover_individually(c, false, &mut rng), StrategyState::Myopic);
        let c = Config {
            human_exploration_trials: 1,
            human_optimality_bias: 1.0,
            optimal_discovery_probability: 0.0,
            myopic_discovery_probability: 0.0,
            ..Config::default()
        };
        assert_eq!(discover_individually(c, false, &mut rng), StrategyState::Random);
    }

    #[test]
    fn archived_failed_social_learning_falls_back_to_myopic_zero() {
        let mut rng = DeterministicRng::new(9);
        let (social, learned) = social_strategy(
            StrategyState::Optimal, 0.0,
            SocialFailureFallback::ArchivedMyopicZero, &mut rng,
        );
        assert_eq!(social, StrategyState::Myopic);
        assert!(!learned);
    }

    #[test]
    fn preserve_individual_failure_fallback_is_separate_ablation() {
        let mut rng = DeterministicRng::new(9);
        let (social, learned) = social_strategy(
            StrategyState::Optimal, 0.0,
            SocialFailureFallback::PreserveIndividual, &mut rng,
        );
        assert_eq!(social, StrategyState::Random);
        assert!(!learned);
    }

    #[test]
    fn optimal_can_be_acquired_from_machine_seed_without_individual_discovery() {
        let run = simulate(forced_config(), 5).unwrap();
        assert!(run.generations[0].agents[..2].iter().all(|a| {
            a.is_machine && a.integrated_strategy == StrategyState::Optimal
        }));
        assert!(run.generations[1].agents.iter().all(|a| {
            !a.is_machine
                && a.individual_strategy == StrategyState::Myopic
                && a.social_strategy == StrategyState::Optimal
                && a.integrated_strategy == StrategyState::Optimal
                && a.social_learning_succeeded
                && a.optimal_origin == Some(OptimalOrigin::MachineSocialLineage)
        }));
    }

    #[test]
    fn teacher_candidates_are_sampled_without_replacement() {
        let mut rng = DeterministicRng::new(7);
        let candidates = sample_teacher_candidates(20, 10, &mut rng);
        let mut unique = candidates.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), 10);
    }

    #[test]
    fn softmax_selects_high_reward_teacher_at_low_temperature() {
        let prev = vec![
            AgentState {
                is_machine: false,
                individual_strategy: StrategyState::Myopic,
                social_strategy: StrategyState::Random,
                integrated_strategy: StrategyState::Myopic,
                teacher_index: None,
                social_learning_succeeded: false,
                demonstration_reward: 0.0,
                optimal_origin: None,
            },
            AgentState {
                is_machine: true,
                individual_strategy: StrategyState::Optimal,
                social_strategy: StrategyState::Random,
                integrated_strategy: StrategyState::Optimal,
                teacher_index: None,
                social_learning_succeeded: false,
                demonstration_reward: 10.0,
                optimal_origin: None,
            },
        ];
        let mut rng = DeterministicRng::new(11);
        assert_eq!(choose_teacher(&prev, &[0, 1], 1e-6, &mut rng), 1);
    }

    #[test]
    fn condition_switches_machine_presence_by_generation() {
        let c = Config {
            machine_count: 2,
            population_size: 8,
            condition: PopulationCondition::HumanMachineSeed,
            ..Config::default()
        };
        assert!(is_machine_agent(c, 0, 0));
        assert!(!is_machine_agent(c, 1, 0));
        let all = Config { condition: PopulationCondition::HumanMachineAll, ..c };
        assert!(is_machine_agent(all, 19, 1));
        let human = Config { condition: PopulationCondition::Human, ..c };
        assert!(!is_machine_agent(human, 0, 0));
    }

    #[test]
    fn fixed_seed_reproduces_the_semantic_port() {
        let c = Config::default();
        assert_eq!(simulate(c, 2026).unwrap(), simulate(c, 2026).unwrap());
    }

    #[test]
    fn random_strategy_has_zero_base_reward_before_noise() {
        let mut rng = DeterministicRng::new(4);
        let c = Config { demonstration_noise_epsilon: 0.0, ..Config::default() };
        assert_eq!(reward_demonstration(StrategyState::Random, c, &mut rng), 0.0);
        assert_eq!(reward_demonstration(StrategyState::Myopic, c, &mut rng), 0.5);
        assert_eq!(reward_demonstration(StrategyState::Optimal, c, &mut rng), 1.0);
    }

    #[test]
    fn absent_human_or_machine_subpopulations_are_undefined_not_zero() {
        let human_only = Config {
            generations: 1,
            population_size: 4,
            machine_count: 0,
            condition: PopulationCondition::Human,
            ..Config::default()
        };
        let run = simulate(human_only, 10).unwrap();
        let metrics = run.generations[0].metrics();
        assert!(metrics.human_optimal_prevalence.is_some());
        assert_eq!(metrics.machine_optimal_prevalence, None);
        assert_eq!(metrics.social_learning_rate_observed, None);
        assert_eq!(metrics.human_social_optimal_acquisition_rate, None);

        let all_machine = Config {
            generations: 1,
            population_size: 4,
            machine_count: 4,
            condition: PopulationCondition::HumanMachineAll,
            ..Config::default()
        };
        let run = simulate(all_machine, 11).unwrap();
        let metrics = run.generations[0].metrics();
        assert_eq!(metrics.human_optimal_prevalence, None);
        assert!(metrics.machine_optimal_prevalence.is_some());
    }

    #[test]
    fn social_metrics_are_undefined_in_seed_generation_but_defined_later() {
        let run = simulate(Config::default(), 12).unwrap();
        let seed = run.generations[0].metrics();
        let later = run.generations[1].metrics();
        assert_eq!(seed.human_social_optimal_acquisition_rate, None);
        assert_eq!(seed.human_social_only_optimal_integration_rate, None);
        assert_eq!(seed.social_learning_rate_observed, None);
        assert!(later.human_social_optimal_acquisition_rate.is_some());
        assert!(later.human_social_only_optimal_integration_rate.is_some());
        assert!(later.social_learning_rate_observed.is_some());
    }

    #[test]
    fn optimal_prevalence_metrics_separate_humans_and_machines() {
        let c = Config {
            generations: 1,
            population_size: 4,
            machine_count: 2,
            condition: PopulationCondition::HumanMachineSeed,
            human_exploration_trials: 1,
            machine_exploration_trials: 1,
            human_optimality_bias: 0.0,
            machine_optimality_bias: 1.0,
            optimal_discovery_probability: 1.0,
            myopic_discovery_probability: 0.0,
            demonstration_noise_epsilon: 0.0,
            ..Config::default()
        };
        let m = simulate(c, 8).unwrap().metrics()[0];
        assert_eq!(m.machine_optimal_prevalence, Some(1.0));
        assert_eq!(m.human_optimal_prevalence, Some(0.0));
        assert_eq!(m.human_individual_optimal_discovery_rate, Some(0.0));
    }

    #[test]
    fn non_finite_generated_demonstration_reward_fails_closed() {
        let config = Config {
            demonstration_count: 4,
            random_reward_per_demonstration: 0.0,
            myopic_reward_per_demonstration: 0.0,
            optimal_reward_per_demonstration: 0.0,
            demonstration_noise_epsilon: f64::MAX,
            ..Config::default()
        };
        assert_eq!(
            simulate(config, 55),
            Err(ConfigError::NonFiniteDemonstrationReward)
        );
    }

    #[test]
    fn invalid_parameters_fail_closed() {
        let c = Config { social_learning_probability: 1.2, ..Config::default() };
        assert_eq!(
            simulate(c, 1),
            Err(ConfigError::InvalidProbability { name: "social_learning_probability" })
        );
    }
}
