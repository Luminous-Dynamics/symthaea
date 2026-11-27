// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#![forbid(unsafe_code)]

//! Rust reference port of the author C implementation for:
//! Czaplicka, Baumann & Rahwan (2025), "Mutual benefits of social learning
//! and algorithmic mediation for cumulative culture".
//!
//! Pinned source:
//! https://github.com/aga-cz/Mutual-benefits-of-SL-and-RS
//! modelQinn.c blob: 8555ec4ad8bd7eec0605bcda28565fbfe2645957
//! script_runQinn.sh blob: e94d5999cc5a7c60ff072186d50febb6d87326a5
//!
//! This is a semantic port with deterministic SplitMix64 randomness, not a
//! bit-for-bit replay of the C program (which seeds rand() from wall-clock
//! time). It preserves the C model's sequential update order, ER graph,
//! death/birth, finite effort budget, innovation reward law, highest-payoff
//! neighbour choice, and global best-payoff-per-level recommendation.
//!
//! The source allocates X "branches" but does not use the branch array in any
//! update rule. V0 records X as metadata and does not invent branch dynamics.

use core::fmt;

/// Author C runner's default/sweep settings.
pub const AUTHOR_POPULATION_SIZE: usize = 100;
pub const AUTHOR_BRANCH_COUNT: usize = 100;
pub const AUTHOR_EFFORT_BUDGET: i64 = 1_000;
pub const AUTHOR_SOCIAL_COST: i64 = 5;
pub const AUTHOR_INNOVATION_COST: i64 = 10;
pub const AUTHOR_RECOMMENDATION_COST: i64 = 1;
pub const AUTHOR_REPLICATIONS: usize = 200;
pub const AUTHOR_STEPS_PER_AGENT: usize = 20;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RecommendationPolicy {
    /// Exact C reference semantics: recommend the highest payoff innovation
    /// discovered so far at the recipient's next complexity level.
    GlobalBestByLevel,
    /// CivBench extension: score candidates by payoff plus an exposure-aware
    /// upper-confidence bonus. This intentionally trades some immediate
    /// exploitation for less concentrated lineage transmission.
    ExposureAwareUcb { exploration_weight: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    pub population_size: usize,
    /// Erdos-Renyi edge probability p_ER.
    pub er_probability: f64,
    /// Retained for source parity. The pinned C source allocates but does not
    /// use this parameter in the update rules.
    pub declared_branch_count: usize,
    pub initial_effort_budget: i64,
    pub social_learning_cost: i64,
    pub innovation_cost: i64,
    pub recommendation_cost: i64,
    pub recommendation_probability: f64,
    pub steps_per_agent: usize,
    pub recommendation_policy: RecommendationPolicy,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            population_size: AUTHOR_POPULATION_SIZE,
            er_probability: 0.1,
            declared_branch_count: AUTHOR_BRANCH_COUNT,
            initial_effort_budget: AUTHOR_EFFORT_BUDGET,
            social_learning_cost: AUTHOR_SOCIAL_COST,
            innovation_cost: AUTHOR_INNOVATION_COST,
            recommendation_cost: AUTHOR_RECOMMENDATION_COST,
            recommendation_probability: 0.7,
            steps_per_agent: AUTHOR_STEPS_PER_AGENT,
            recommendation_policy: RecommendationPolicy::GlobalBestByLevel,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    ZeroPopulation,
    InvalidProbability { name: &'static str },
    ZeroBranchCount,
    NonPositiveBudgetOrCost { name: &'static str },
    ZeroStepsPerAgent,
    InvalidExplorationWeight,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroPopulation => write!(f, "population size must be positive"),
            Self::InvalidProbability { name } => write!(f, "{name} must be finite and in [0, 1]"),
            Self::ZeroBranchCount => write!(f, "declared branch count must be positive"),
            Self::NonPositiveBudgetOrCost { name } => write!(f, "{name} must be positive"),
            Self::ZeroStepsPerAgent => write!(f, "steps per agent must be positive"),
            Self::InvalidExplorationWeight => write!(f, "exposure-aware exploration weight must be finite and non-negative"),
        }
    }
}
impl std::error::Error for ConfigError {}

impl Config {
    pub fn validate(self) -> Result<(), ConfigError> {
        if self.population_size == 0 {
            return Err(ConfigError::ZeroPopulation);
        }
        for (name, value) in [
            ("Erdos-Renyi probability", self.er_probability),
            ("recommendation probability", self.recommendation_probability),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(ConfigError::InvalidProbability { name });
            }
        }
        if self.declared_branch_count == 0 {
            return Err(ConfigError::ZeroBranchCount);
        }
        for (name, value) in [
            ("initial effort budget", self.initial_effort_budget),
            ("social-learning cost", self.social_learning_cost),
            ("innovation cost", self.innovation_cost),
            ("recommendation cost", self.recommendation_cost),
        ] {
            if value <= 0 {
                return Err(ConfigError::NonPositiveBudgetOrCost { name });
            }
        }
        if self.initial_effort_budget < self.innovation_cost {
            return Err(ConfigError::NonPositiveBudgetOrCost {
                name: "initial effort budget (must cover one initial innovation)",
            });
        }
        if self.steps_per_agent == 0 {
            return Err(ConfigError::ZeroStepsPerAgent);
        }
        if let RecommendationPolicy::ExposureAwareUcb { exploration_weight } =
            self.recommendation_policy
        {
            if !exploration_weight.is_finite() || exploration_weight < 0.0 {
                return Err(ConfigError::InvalidExplorationWeight);
            }
        }
        Ok(())
    }

    pub fn time_steps(self) -> usize {
        self.population_size.saturating_mul(self.steps_per_agent)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErNetwork {
    adjacency: Vec<Vec<usize>>,
    edge_count: usize,
}

impl ErNetwork {
    fn generate(population_size: usize, probability: f64, rng: &mut DeterministicRng) -> Self {
        let mut adjacency = vec![Vec::new(); population_size];
        let mut edge_count = 0;
        for left in 0..population_size {
            for right in 0..left {
                if rng.uniform() <= probability {
                    adjacency[left].push(right);
                    adjacency[right].push(left);
                    edge_count += 1;
                }
            }
        }
        for neighbours in &mut adjacency {
            neighbours.sort_unstable();
        }
        Self { adjacency, edge_count }
    }

    pub fn population_size(&self) -> usize {
        self.adjacency.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    pub fn neighbours(&self, agent: usize) -> Option<&[usize]> {
        self.adjacency.get(agent).map(Vec::as_slice)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CulturalTrait {
    pub id: usize,
    pub level: usize,
    pub payoff: u32,
    pub origin_agent: usize,
    pub parent_trait: Option<usize>,
    pub root_lineage: usize,
    pub adoption_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentState {
    pub effort_budget: i64,
    pub level: usize,
    pub cumulative_payoff: u64,
    /// Trait ID by complexity level. None means no trait at that level.
    pub traits_by_level: Vec<Option<usize>>,
}

impl AgentState {
    fn naive(initial_effort_budget: i64, max_levels: usize) -> Self {
        Self {
            effort_budget: initial_effort_budget,
            level: 0,
            cumulative_payoff: 0,
            traits_by_level: vec![None; max_levels],
        }
    }

    fn die_and_be_reborn(&mut self, initial_effort_budget: i64) {
        self.effort_budget = initial_effort_budget;
        self.level = 0;
        self.cumulative_payoff = 0;
        self.traits_by_level.fill(None);
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepMetrics {
    pub step: usize,
    pub mean_payoff: f64,
    pub mean_level: f64,
    pub mean_remaining_budget: f64,
    /// Reproduces the source's order-dependent sum of prefix maxima. It is not
    /// the same as the true maximum payoff in the population.
    pub legacy_prefix_max_payoff_mean: f64,
    /// Reproduces the source's order-dependent sum of prefix level maxima.
    /// The C source hard-codes this to one per agent at step zero.
    pub legacy_prefix_max_level_mean: f64,
    /// Reproduces the source's max-achievable-payoff accumulation: sum, over
    /// each agent's acquired levels, of the global best innovation payoff.
    pub legacy_max_achievable_payoff_mean: f64,
    pub true_max_payoff: u64,
    pub true_max_level: usize,
    pub cumulative_unique_innovations: u64,
    pub cumulative_peer_copies: u64,
    pub cumulative_recommendations: u64,
    pub active_unique_trait_count: usize,
    pub active_lineage_count: usize,
    pub active_lineage_hhi: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Run {
    pub seed: u64,
    pub network: ErNetwork,
    pub traits: Vec<CulturalTrait>,
    pub final_agents: Vec<AgentState>,
    pub steps: Vec<StepMetrics>,
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

    /// Uniform in [0, 1), analogous to C frand().
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0
    }

    fn index(&mut self, upper: usize) -> usize {
        (self.next_u64() as usize) % upper
    }
}

/// Exact payoff transform used by modelQinn.c:
/// round(2 * X^2), with X ~ Exponential(rate=1).
pub fn payoff_from_uniform(uniform: f64) -> u32 {
    let bounded = uniform.clamp(0.0, 1.0 - f64::EPSILON);
    let exponential = -(1.0 - bounded).ln();
    (2.0 * exponential * exponential).round() as u32
}

fn sample_payoff(rng: &mut DeterministicRng) -> u32 {
    payoff_from_uniform(rng.uniform())
}

fn choose_recommendation_trait(
    level: usize,
    traits_by_level: &[Vec<usize>],
    traits: &[CulturalTrait],
    policy: RecommendationPolicy,
) -> Option<usize> {
    let candidates = traits_by_level.get(level)?;
    if candidates.is_empty() {
        return None;
    }

    match policy {
        RecommendationPolicy::GlobalBestByLevel => {
            // C updates IP[level] only on a strictly larger innovation payoff,
            // so earliest insertion wins ties.
            let mut best = candidates[0];
            for &candidate in candidates.iter().skip(1) {
                if traits[candidate].payoff > traits[best].payoff {
                    best = candidate;
                }
            }
            (traits[best].payoff > 0).then_some(best)
        }
        RecommendationPolicy::ExposureAwareUcb { exploration_weight } => {
            let total_exposures: u64 = candidates
                .iter()
                .map(|&candidate| traits[candidate].adoption_count)
                .sum();
            let log_total = ((total_exposures + 2) as f64).ln();
            let mut best = candidates[0];
            let mut best_score = f64::NEG_INFINITY;
            for &candidate in candidates {
                let record = &traits[candidate];
                let bonus = exploration_weight
                    * (log_total / (record.adoption_count + 1) as f64).sqrt();
                let score = f64::from(record.payoff) + bonus;
                if score > best_score {
                    best_score = score;
                    best = candidate;
                }
            }
            (traits[best].payoff > 0).then_some(best)
        }
    }
}

fn create_innovation(
    agent_id: usize,
    agent: &mut AgentState,
    traits: &mut Vec<CulturalTrait>,
    traits_by_level: &mut Vec<Vec<usize>>,
    innovation_cost: i64,
    rng: &mut DeterministicRng,
) -> bool {
    if agent.effort_budget < innovation_cost {
        return false;
    }

    let target_level = agent.level + 1;
    agent.effort_budget -= innovation_cost;
    let payoff = sample_payoff(rng);
    if payoff == 0 {
        return false;
    }

    if traits_by_level.len() <= target_level {
        traits_by_level.resize_with(target_level + 1, Vec::new);
    }
    let parent_trait = if agent.level > 0 {
        agent.traits_by_level.get(agent.level).copied().flatten()
    } else {
        None
    };
    let id = traits.len();
    let root_lineage = parent_trait
        .map(|parent| traits[parent].root_lineage)
        .unwrap_or(id);
    traits.push(CulturalTrait {
        id,
        level: target_level,
        payoff,
        origin_agent: agent_id,
        parent_trait,
        root_lineage,
        adoption_count: 0,
    });
    traits_by_level[target_level].push(id);

    if agent.traits_by_level.len() <= target_level {
        agent.traits_by_level.resize(target_level + 1, None);
    }
    agent.traits_by_level[target_level] = Some(id);
    agent.level = target_level;
    agent.cumulative_payoff += u64::from(payoff);
    true
}

fn copy_trait(
    agent: &mut AgentState,
    trait_id: usize,
    traits: &mut [CulturalTrait],
    cost: i64,
) {
    let trait_record = &traits[trait_id];
    let target_level = trait_record.level;
    let payoff = trait_record.payoff;

    agent.effort_budget -= cost;
    if agent.traits_by_level.len() <= target_level {
        agent.traits_by_level.resize(target_level + 1, None);
    }
    agent.traits_by_level[target_level] = Some(trait_id);
    agent.level = target_level;
    agent.cumulative_payoff += u64::from(payoff);
    traits[trait_id].adoption_count += 1;
}

fn choose_peer_trait(
    agent_id: usize,
    target_level: usize,
    network: &ErNetwork,
    agents: &[AgentState],
) -> Option<usize> {
    let neighbours = network.neighbours(agent_id)?;
    let mut best_payoff: i128 = -1;
    let mut best_agent: Option<usize> = None;

    // The C code uses >=, so the highest-index neighbour wins equal-payoff ties.
    for &candidate in neighbours {
        let payoff = agents[candidate].cumulative_payoff as i128;
        if payoff >= best_payoff {
            best_payoff = payoff;
            best_agent = Some(candidate);
        }
    }

    let teacher = best_agent?;
    if agents[teacher].level < target_level {
        return None;
    }
    agents[teacher]
        .traits_by_level
        .get(target_level)
        .copied()
        .flatten()
}

fn step_metrics(
    step: usize,
    agents: &[AgentState],
    traits: &[CulturalTrait],
    traits_by_level: &[Vec<usize>],
    cumulative_unique_innovations: u64,
    cumulative_peer_copies: u64,
    cumulative_recommendations: u64,
    legacy_prefix_max_payoff_sum: u64,
    legacy_prefix_max_level_sum: u64,
    initial_step: bool,
) -> StepMetrics {
    let n = agents.len() as f64;
    let total_payoff: u64 = agents.iter().map(|agent| agent.cumulative_payoff).sum();
    let total_level: usize = agents.iter().map(|agent| agent.level).sum();
    let total_budget: i64 = agents.iter().map(|agent| agent.effort_budget).sum();
    let true_max_payoff = agents.iter().map(|agent| agent.cumulative_payoff).max().unwrap_or(0);
    let true_max_level = agents.iter().map(|agent| agent.level).max().unwrap_or(0);

    let mut max_achievable_sum = 0u64;
    for agent in agents {
        for level in 1..=agent.level {
            if let Some(&best_trait) = traits_by_level.get(level).and_then(|items| {
                items.iter().max_by_key(|&&trait_id| traits[trait_id].payoff)
            }) {
                max_achievable_sum += u64::from(traits[best_trait].payoff);
            }
        }
    }

    let mut active_traits = std::collections::BTreeSet::new();
    let mut lineage_counts = std::collections::BTreeMap::<usize, u64>::new();
    for agent in agents {
        if let Some(Some(trait_id)) = agent.traits_by_level.get(agent.level) {
            active_traits.insert(*trait_id);
            let root = traits[*trait_id].root_lineage;
            *lineage_counts.entry(root).or_default() += 1;
        }
    }
    let active_lineage_count = lineage_counts.len();
    let active_lineage_hhi = if agents.is_empty() {
        0.0
    } else {
        lineage_counts.values()
            .map(|&count| {
                let share = count as f64 / n;
                share * share
            })
            .sum()
    };

    // Step zero in the C source increments max_lvl[0] once per user, independent
    // of actual level. Preserve that output as a separately named legacy field.
    let legacy_level_mean = if initial_step {
        1.0
    } else {
        legacy_prefix_max_level_sum as f64 / n
    };

    StepMetrics {
        step,
        mean_payoff: total_payoff as f64 / n,
        mean_level: total_level as f64 / n,
        mean_remaining_budget: total_budget as f64 / n,
        legacy_prefix_max_payoff_mean: legacy_prefix_max_payoff_sum as f64 / n,
        legacy_prefix_max_level_mean: legacy_level_mean,
        legacy_max_achievable_payoff_mean: max_achievable_sum as f64 / n,
        true_max_payoff,
        true_max_level,
        cumulative_unique_innovations,
        cumulative_peer_copies,
        cumulative_recommendations,
        active_unique_trait_count: active_traits.len(),
        active_lineage_count,
        active_lineage_hhi,
    }
}

/// Run one author-model realization with deterministic randomness.
pub fn simulate(config: Config, seed: u64) -> Result<Run, ConfigError> {
    config.validate()?;
    let mut rng = DeterministicRng::new(seed);
    let network = ErNetwork::generate(
        config.population_size,
        config.er_probability,
        &mut rng,
    );
    let time_steps = config.time_steps();
    let max_levels = time_steps.saturating_add(2);
    let mut agents: Vec<AgentState> = (0..config.population_size)
        .map(|_| AgentState::naive(config.initial_effort_budget, max_levels))
        .collect();
    let mut traits = Vec::new();
    let mut traits_by_level = vec![Vec::<usize>::new(); max_levels];
    let mut cumulative_unique_innovations = 0u64;
    let mut cumulative_peer_copies = 0u64;
    let mut cumulative_recommendations = 0u64;
    let mut steps = Vec::with_capacity(time_steps);

    // The C source first gives every agent one sequential innovation attempt.
    let mut prefix_max_payoff = 0u64;
    let mut legacy_prefix_max_payoff_sum = 0u64;
    let mut legacy_initial_max_achievable_sum = 0u64;
    for agent_id in 0..config.population_size {
        create_innovation(
            agent_id,
            &mut agents[agent_id],
            &mut traits,
            &mut traits_by_level,
            config.innovation_cost,
            &mut rng,
        );
        if agents[agent_id].cumulative_payoff > prefix_max_payoff {
            prefix_max_payoff = agents[agent_id].cumulative_payoff;
        }
        legacy_prefix_max_payoff_sum += prefix_max_payoff;
        // The C source updates max_pos_pay[0] inside this same sequential
        // seeding loop. Later discoveries therefore do not retroactively
        // change an earlier agent's contribution to the step-zero metric.
        for level in 1..=agents[agent_id].level {
            if let Some(best_trait_id) = traits_by_level.get(level).and_then(|items| {
                items.iter().max_by_key(|&&trait_id| traits[trait_id].payoff)
            }) {
                legacy_initial_max_achievable_sum +=
                    u64::from(traits[*best_trait_id].payoff);
            }
        }
        cumulative_unique_innovations = traits.len() as u64;
    }
    let legacy_initial_prefix_level_sum = config.population_size as u64;
    let mut initial_metrics = step_metrics(
        0,
        &agents,
        &traits,
        &traits_by_level,
        cumulative_unique_innovations,
        cumulative_peer_copies,
        cumulative_recommendations,
        legacy_prefix_max_payoff_sum,
        legacy_initial_prefix_level_sum,
        true,
    );
    initial_metrics.legacy_max_achievable_payoff_mean =
        legacy_initial_max_achievable_sum as f64 / config.population_size as f64;
    steps.push(initial_metrics);

    // The author implementation updates agents sequentially within each time
    // step. Later agents can observe innovations and copies made earlier in
    // the same step; do not parallelize this loop in the reference port.
    for step in 1..time_steps {
        let mut prefix_max_payoff = 0u64;
        let mut prefix_max_level = 0usize;
        let mut legacy_prefix_max_payoff_sum = 0u64;
        let mut legacy_prefix_max_level_sum = 0u64;

        for agent_id in 0..config.population_size {
            // Death-birth replacement with probability 1/N. Global best-known
            // innovations survive the death of individual agents.
            if rng.uniform() <= 1.0 / config.population_size as f64 {
                agents[agent_id].die_and_be_reborn(config.initial_effort_budget);
            }

            let channel_draw = rng.uniform();
            let mut acquired = false;
            let mut acquired_by_recommendation = false;

            if channel_draw < config.recommendation_probability
                || config.recommendation_probability == 1.0
            {
                let target_level = agents[agent_id].level + 1;
                if agents[agent_id].effort_budget >= config.recommendation_cost {
                    if let Some(trait_id) = choose_recommendation_trait(
                        target_level,
                        &traits_by_level,
                        &traits,
                        config.recommendation_policy,
                    ) {
                        copy_trait(
                            &mut agents[agent_id],
                            trait_id,
                            &mut traits,
                            config.recommendation_cost,
                        );
                        cumulative_recommendations += 1;
                        acquired = true;
                        acquired_by_recommendation = true;
                    }
                }
            } else if channel_draw > config.recommendation_probability
                || config.recommendation_probability == 0.0
            {
                let target_level = agents[agent_id].level + 1;
                if agents[agent_id].effort_budget >= config.social_learning_cost {
                    if let Some(trait_id) = choose_peer_trait(
                        agent_id,
                        target_level,
                        &network,
                        &agents,
                    ) {
                        copy_trait(
                            &mut agents[agent_id],
                            trait_id,
                            &mut traits,
                            config.social_learning_cost,
                        );
                        cumulative_peer_copies += 1;
                        acquired = true;
                    }
                }
            }

            if !acquired {
                let old_trait_count = traits.len();
                create_innovation(
                    agent_id,
                    &mut agents[agent_id],
                    &mut traits,
                    &mut traits_by_level,
                    config.innovation_cost,
                    &mut rng,
                );
                if traits.len() > old_trait_count {
                    cumulative_unique_innovations += 1;
                }
            }

            if agents[agent_id].cumulative_payoff > prefix_max_payoff {
                prefix_max_payoff = agents[agent_id].cumulative_payoff;
            }
            if agents[agent_id].level > prefix_max_level {
                prefix_max_level = agents[agent_id].level;
            }
            legacy_prefix_max_payoff_sum += prefix_max_payoff;
            legacy_prefix_max_level_sum += prefix_max_level as u64;
            let _ = acquired_by_recommendation;
        }

        steps.push(step_metrics(
            step,
            &agents,
            &traits,
            &traits_by_level,
            cumulative_unique_innovations,
            cumulative_peer_copies,
            cumulative_recommendations,
            legacy_prefix_max_payoff_sum,
            legacy_prefix_max_level_sum,
            false,
        ));
    }

    Ok(Run {
        seed,
        network,
        traits,
        final_agents: agents,
        steps,
    })
}

/// Run Q author-style independent realizations with a reproducible seed schedule.
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
        .map(|index| simulate(config, base_seed.wrapping_add(index as u64)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_parameters_match_authors_runner() {
        let config = Config::default();
        assert_eq!(config.population_size, AUTHOR_POPULATION_SIZE);
        assert_eq!(config.declared_branch_count, AUTHOR_BRANCH_COUNT);
        assert_eq!(config.initial_effort_budget, AUTHOR_EFFORT_BUDGET);
        assert_eq!(config.social_learning_cost, AUTHOR_SOCIAL_COST);
        assert_eq!(config.innovation_cost, AUTHOR_INNOVATION_COST);
        assert_eq!(config.recommendation_cost, AUTHOR_RECOMMENDATION_COST);
        assert_eq!(config.steps_per_agent, 20);
    }

    #[test]
    fn reward_transform_matches_c_rounding_contract() {
        assert_eq!(payoff_from_uniform(0.0), 0);
        assert_eq!(payoff_from_uniform(0.5), 1);
        assert_eq!(payoff_from_uniform(0.75), 4);
    }

    #[test]
    fn er_graph_has_no_self_loops_and_is_symmetric() {
        let mut rng = DeterministicRng::new(4);
        let graph = ErNetwork::generate(20, 0.25, &mut rng);
        for node in 0..20 {
            assert!(!graph.neighbours(node).unwrap().contains(&node));
            for &neighbour in graph.neighbours(node).unwrap() {
                assert!(graph.neighbours(neighbour).unwrap().contains(&node));
            }
        }
    }

    #[test]
    fn zero_density_graph_has_no_peer_edges_and_complete_graph_is_complete() {
        let mut zero_rng = DeterministicRng::new(1);
        let zero = ErNetwork::generate(8, 0.0, &mut zero_rng);
        assert_eq!(zero.edge_count(), 0);

        let mut full_rng = DeterministicRng::new(1);
        let full = ErNetwork::generate(8, 1.0, &mut full_rng);
        assert_eq!(full.edge_count(), 8 * 7 / 2);
    }

    #[test]
    fn fixed_seed_is_reproducible() {
        let config = Config {
            population_size: 12,
            steps_per_agent: 2,
            ..Config::default()
        };
        assert_eq!(simulate(config, 99).unwrap(), simulate(config, 99).unwrap());
    }

    #[test]
    fn recommendation_and_peer_channels_are_exclusive() {
        let config_recommend = Config {
            population_size: 10,
            steps_per_agent: 2,
            recommendation_probability: 1.0,
            ..Config::default()
        };
        let recommended = simulate(config_recommend, 9).unwrap();
        assert!(recommended.steps.iter().all(|step| step.cumulative_peer_copies == 0));

        let config_peer = Config {
            population_size: 10,
            steps_per_agent: 2,
            recommendation_probability: 0.0,
            ..Config::default()
        };
        let peer = simulate(config_peer, 9).unwrap();
        assert!(peer.steps.iter().all(|step| step.cumulative_recommendations == 0));
    }

    #[test]
    fn death_birth_erases_agent_memory_but_not_global_recommendation_archive() {
        let config = Config {
            population_size: 1,
            er_probability: 0.0,
            initial_effort_budget: 100,
            social_learning_cost: 5,
            innovation_cost: 10,
            recommendation_cost: 1,
            recommendation_probability: 1.0,
            steps_per_agent: 3,
            ..Config::default()
        };
        // Seed 42 yields a positive first innovation payoff.
        let run = simulate(config, 42).unwrap();
        assert!(run.traits.iter().any(|record| record.level == 1));
        assert_eq!(run.steps[1].mean_level, 1.0);
        assert_eq!(run.steps[2].mean_level, 1.0);
        assert!(run.steps[1].cumulative_recommendations >= 1);
    }

    #[test]
    fn exposure_aware_recommender_can_recover_a_rare_candidate() {
        let traits = vec![
            CulturalTrait {
                id: 0,
                level: 1,
                payoff: 6,
                origin_agent: 0,
                parent_trait: None,
                root_lineage: 0,
                adoption_count: 100,
            },
            CulturalTrait {
                id: 1,
                level: 1,
                payoff: 5,
                origin_agent: 1,
                parent_trait: None,
                root_lineage: 1,
                adoption_count: 0,
            },
        ];
        let pool = vec![vec![], vec![0, 1]];
        assert_eq!(
            choose_recommendation_trait(1, &pool, &traits, RecommendationPolicy::GlobalBestByLevel),
            Some(0)
        );
        assert_eq!(
            choose_recommendation_trait(
                1,
                &pool,
                &traits,
                RecommendationPolicy::ExposureAwareUcb { exploration_weight: 10.0 },
            ),
            Some(1)
        );
    }

    #[test]
    fn lineage_and_adoption_counters_reconcile_with_run_events() {
        let config = Config {
            population_size: 16,
            steps_per_agent: 4,
            er_probability: 0.35,
            recommendation_probability: 0.45,
            ..Config::default()
        };
        let run = simulate(config, 2025).unwrap();
        let final_step = run.steps.last().unwrap();

        assert_eq!(
            run.traits.len() as u64,
            final_step.cumulative_unique_innovations
        );

        let summed_adoptions: u64 = run.traits.iter()
            .map(|trait_record| trait_record.adoption_count)
            .sum();
        assert_eq!(
            summed_adoptions,
            final_step.cumulative_peer_copies + final_step.cumulative_recommendations
        );

        for (index, trait_record) in run.traits.iter().enumerate() {
            assert_eq!(trait_record.id, index);
            assert!(trait_record.payoff > 0);
            assert!(trait_record.level > 0);
            match trait_record.parent_trait {
                Some(parent_id) => {
                    assert!(parent_id < trait_record.id);
                    let parent = &run.traits[parent_id];
                    assert_eq!(parent.level + 1, trait_record.level);
                    assert_eq!(parent.root_lineage, trait_record.root_lineage);
                }
                None => assert_eq!(trait_record.root_lineage, trait_record.id),
            }
        }

        for agent in &run.final_agents {
            assert!(agent.effort_budget >= 0);
            assert!(agent.level < agent.traits_by_level.len());
            if agent.level > 0 {
                assert!(agent.traits_by_level[agent.level].is_some());
            }
        }
    }

    #[test]
    fn declared_branch_count_is_inert_in_the_pinned_upstream_update_rules() {
        let common = Config {
            population_size: 12,
            steps_per_agent: 3,
            er_probability: 0.25,
            recommendation_probability: 0.6,
            ..Config::default()
        };
        let one_branch = simulate(
            Config { declared_branch_count: 1, ..common },
            81,
        ).unwrap();
        let many_branches = simulate(
            Config { declared_branch_count: 1_000, ..common },
            81,
        ).unwrap();
        assert_eq!(one_branch, many_branches);
    }

    #[test]
    fn step_count_matches_population_times_steps_per_agent() {
        let config = Config {
            population_size: 9,
            steps_per_agent: 7,
            ..Config::default()
        };
        let run = simulate(config, 2).unwrap();
        assert_eq!(run.steps.len(), 9 * 7);
    }

    #[test]
    fn active_lineage_hhi_stays_bounded_over_a_complete_run() {
        let config = Config {
            population_size: 8,
            steps_per_agent: 1,
            innovation_cost: 10_000,
            initial_effort_budget: 10_000,
            social_learning_cost: 20_000,
            recommendation_cost: 20_000,
            ..Config::default()
        };
        // Agents can still make one initial innovation attempt. The assertion
        // below focuses on the metric's numeric domain, not a forced zero case.
        let run = simulate(config, 5).unwrap();
        for step in &run.steps {
            assert!((0.0..=1.0).contains(&step.active_lineage_hhi));
        }
    }

    #[test]
    fn zero_lineage_hhi_when_population_has_no_traits() {
        let agents = vec![
            AgentState::naive(100, 2),
            AgentState::naive(100, 2),
            AgentState::naive(100, 2),
        ];
        let metrics = step_metrics(
            0,
            &agents,
            &[],
            &[Vec::new(), Vec::new()],
            0,
            0,
            0,
            0,
            0,
            true,
        );
        assert_eq!(metrics.active_unique_trait_count, 0);
        assert_eq!(metrics.active_lineage_count, 0);
        assert_eq!(metrics.active_lineage_hhi, 0.0);
    }

    #[test]
    fn invalid_parameters_fail_closed() {
        let config = Config {
            recommendation_probability: 1.1,
            ..Config::default()
        };
        assert_eq!(
            simulate(config, 1),
            Err(ConfigError::InvalidProbability {
                name: "recommendation probability",
            })
        );
    }
}
