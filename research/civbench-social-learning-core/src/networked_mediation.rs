// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#![forbid(unsafe_code)]

//! Networked cultural transmission with optional algorithmic mediation.
//!
//! This is a synthetic mechanism benchmark inspired by work on the interaction
//! of social networks and recommender-mediated cultural accumulation. It is not
//! a fit to the cited paper's parameters or human participant data.

use core::fmt;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RecommenderObjective {
    /// Rank candidates solely by demonstrated capability quality.
    PerformanceOnly,
    /// Add a bounded bonus for underrepresented capability lineages.
    PerformanceDiversity { novelty_weight: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NetworkedMediationConfig {
    pub population_size: usize,
    pub generations: usize,
    /// Probability of adding each non-ring undirected shortcut edge.
    pub shortcut_probability: f64,
    /// Fraction of learners served by a global recommender rather than local peers.
    /// 0.0 = peer-only; 1.0 = recommender-only.
    pub mediation_fraction: f64,
    pub recommender_candidate_count: usize,
    /// Temperature for selecting among high-scoring recommender candidates.
    pub recommender_temperature: f64,
    pub recommender_objective: RecommenderObjective,
    pub initial_seed_agents: usize,
    pub initial_seed_quality: u16,
    pub individual_innovation_probability: f64,
    pub innovation_gain: u16,
    pub observation_probability: f64,
    pub acquisition_probability: f64,
    pub corruption_probability: f64,
    pub corruption_loss: u16,
    pub demonstration_noise_std: f64,
}

impl Default for NetworkedMediationConfig {
    fn default() -> Self {
        Self {
            population_size: 32,
            generations: 8,
            shortcut_probability: 0.05,
            mediation_fraction: 0.5,
            recommender_candidate_count: 4,
            recommender_temperature: 0.25,
            recommender_objective: RecommenderObjective::PerformanceOnly,
            initial_seed_agents: 1,
            initial_seed_quality: 10,
            individual_innovation_probability: 0.01,
            innovation_gain: 1,
            observation_probability: 0.9,
            acquisition_probability: 0.8,
            corruption_probability: 0.05,
            corruption_loss: 2,
            demonstration_noise_std: 0.25,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediationError {
    PopulationTooSmall,
    ZeroGenerations,
    TooManySeedAgents { seeds: usize, population: usize },
    ZeroSeedQuality,
    ZeroCandidateCount,
    CandidateCountTooLarge { candidates: usize, population: usize },
    InvalidProbability { name: &'static str },
    InvalidTemperature,
    InvalidNoveltyWeight,
    InvalidNoise,
    ZeroInnovationGain,
}

impl fmt::Display for MediationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PopulationTooSmall => write!(f, "population_size must be at least 3"),
            Self::ZeroGenerations => write!(f, "generations must be positive"),
            Self::TooManySeedAgents { seeds, population } => {
                write!(f, "{seeds} seed agents exceed population size {population}")
            }
            Self::ZeroSeedQuality => write!(f, "initial_seed_quality must be positive when seed agents are configured"),
            Self::ZeroCandidateCount => write!(f, "recommender_candidate_count must be positive"),
            Self::CandidateCountTooLarge { candidates, population } => {
                write!(f, "{candidates} recommender candidates exceed population size {population}")
            }
            Self::InvalidProbability { name } => write!(f, "{name} must be finite and in [0, 1]"),
            Self::InvalidTemperature => write!(f, "recommender temperature must be finite and positive"),
            Self::InvalidNoveltyWeight => write!(f, "recommender novelty weight must be finite and in [0, 1]"),
            Self::InvalidNoise => write!(f, "demonstration noise must be finite and non-negative"),
            Self::ZeroInnovationGain => write!(f, "innovation_gain must be positive"),
        }
    }
}
impl std::error::Error for MediationError {}

impl NetworkedMediationConfig {
    pub fn validate(self) -> Result<(), MediationError> {
        if self.population_size < 3 {
            return Err(MediationError::PopulationTooSmall);
        }
        if self.generations == 0 {
            return Err(MediationError::ZeroGenerations);
        }
        if self.initial_seed_agents > self.population_size {
            return Err(MediationError::TooManySeedAgents {
                seeds: self.initial_seed_agents,
                population: self.population_size,
            });
        }
        if self.initial_seed_agents > 0 && self.initial_seed_quality == 0 {
            return Err(MediationError::ZeroSeedQuality);
        }
        if self.recommender_candidate_count == 0 {
            return Err(MediationError::ZeroCandidateCount);
        }
        if self.recommender_candidate_count > self.population_size {
            return Err(MediationError::CandidateCountTooLarge {
                candidates: self.recommender_candidate_count,
                population: self.population_size,
            });
        }
        for (name, value) in [
            ("shortcut_probability", self.shortcut_probability),
            ("mediation_fraction", self.mediation_fraction),
            ("individual_innovation_probability", self.individual_innovation_probability),
            ("observation_probability", self.observation_probability),
            ("acquisition_probability", self.acquisition_probability),
            ("corruption_probability", self.corruption_probability),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(MediationError::InvalidProbability { name });
            }
        }
        if !self.recommender_temperature.is_finite() || self.recommender_temperature <= 0.0 {
            return Err(MediationError::InvalidTemperature);
        }
        if let RecommenderObjective::PerformanceDiversity { novelty_weight } = self.recommender_objective {
            if !novelty_weight.is_finite() || !(0.0..=1.0).contains(&novelty_weight) {
                return Err(MediationError::InvalidNoveltyWeight);
            }
        }
        if !self.demonstration_noise_std.is_finite() || self.demonstration_noise_std < 0.0 {
            return Err(MediationError::InvalidNoise);
        }
        if self.innovation_gain == 0 {
            return Err(MediationError::ZeroInnovationGain);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SocialGraph {
    neighbors: Vec<Vec<usize>>,
    undirected_edges: usize,
}

impl SocialGraph {
    pub fn ring_with_shortcuts(
        population_size: usize,
        shortcut_probability: f64,
        seed: u64,
    ) -> Result<Self, MediationError> {
        if population_size < 3 {
            return Err(MediationError::PopulationTooSmall);
        }
        if !shortcut_probability.is_finite() || !(0.0..=1.0).contains(&shortcut_probability) {
            return Err(MediationError::InvalidProbability {
                name: "shortcut_probability",
            });
        }

        let mut rng = DeterministicRng::new(seed);
        let mut neighbors = vec![Vec::new(); population_size];

        // A ring is the connected social-learning substrate.
        for node in 0..population_size {
            let left = (node + population_size - 1) % population_size;
            let right = (node + 1) % population_size;
            neighbors[node].push(left);
            neighbors[node].push(right);
        }

        // Add undirected long-range shortcuts to vary network density.
        for a in 0..population_size {
            for b in (a + 1)..population_size {
                let already_connected = neighbors[a].contains(&b);
                if !already_connected && rng.bernoulli(shortcut_probability) {
                    neighbors[a].push(b);
                    neighbors[b].push(a);
                }
            }
        }

        for list in &mut neighbors {
            list.sort_unstable();
            list.dedup();
        }
        let degree_sum: usize = neighbors.iter().map(Vec::len).sum();
        let undirected_edges = degree_sum / 2;
        Ok(Self {
            neighbors,
            undirected_edges,
        })
    }

    pub fn population_size(&self) -> usize {
        self.neighbors.len()
    }

    pub fn neighbors(&self, node: usize) -> Option<&[usize]> {
        self.neighbors.get(node).map(Vec::as_slice)
    }

    pub fn edge_count(&self) -> usize {
        self.undirected_edges
    }

    pub fn density(&self) -> f64 {
        let n = self.population_size();
        if n < 2 {
            return 0.0;
        }
        self.undirected_edges as f64 / ((n * (n - 1) / 2) as f64)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnowledgeSource {
    None,
    InitialMachine,
    IndividualInnovation,
    PeerSocial,
    AlgorithmicMediation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetworkedAgent {
    pub quality: u16,
    /// Stable family identifier. Zero means no non-zero skill is held.
    pub lineage_id: u64,
    pub source: KnowledgeSource,
    pub individual_innovated: bool,
    pub demonstration_observed: bool,
    pub strategy_acquisition_succeeded: bool,
    pub acquisition_source: Option<KnowledgeSource>,
    /// Selected teacher, even when acquired skill did not improve active quality.
    pub parent_index: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NetworkedGeneration {
    pub index: usize,
    pub agents: Vec<NetworkedAgent>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NetworkedGenerationMetrics {
    pub generation: usize,
    pub mean_quality: f64,
    pub max_quality: u16,
    pub seed_lineage_prevalence: f64,
    pub knowledge_coverage: f64,
    pub unique_nonzero_lineages: usize,
    pub lineage_hhi: f64,
    pub effective_lineage_diversity: f64,
    pub individual_innovations: usize,
    pub peer_adoptions: usize,
    pub mediated_adoptions: usize,
    pub peer_acquisitions: usize,
    pub mediated_acquisitions: usize,
    pub observations: usize,
    pub successful_acquisitions: usize,
}

impl NetworkedGeneration {
    pub fn metrics(&self) -> NetworkedGenerationMetrics {
        let n = self.agents.len();
        let total_quality: u64 = self.agents.iter().map(|agent| u64::from(agent.quality)).sum();
        let max_quality = self.agents.iter().map(|agent| agent.quality).max().unwrap_or(0);
        let seed_count = self.agents.iter().filter(|agent| agent.lineage_id == 1).count();
        let individual_innovations = self.agents.iter()
            .filter(|agent| agent.individual_innovated).count();
        let peer_adoptions = self.agents.iter()
            .filter(|agent| agent.source == KnowledgeSource::PeerSocial).count();
        let mediated_adoptions = self.agents.iter()
            .filter(|agent| agent.source == KnowledgeSource::AlgorithmicMediation).count();
        let peer_acquisitions = self.agents.iter()
            .filter(|agent| agent.acquisition_source == Some(KnowledgeSource::PeerSocial)).count();
        let mediated_acquisitions = self.agents.iter()
            .filter(|agent| agent.acquisition_source == Some(KnowledgeSource::AlgorithmicMediation)).count();
        let observations = self.agents.iter()
            .filter(|agent| agent.demonstration_observed).count();
        let successful_acquisitions = self.agents.iter()
            .filter(|agent| agent.strategy_acquisition_succeeded).count();

        let mut lineage_counts = BTreeMap::<u64, usize>::new();
        for agent in &self.agents {
            if agent.quality > 0 && agent.lineage_id != 0 {
                *lineage_counts.entry(agent.lineage_id).or_default() += 1;
            }
        }
        let unique_nonzero_lineages = lineage_counts.len();
        let knowledge_count = self.agents.iter().filter(|agent| agent.quality > 0).count();
        let lineage_hhi = if knowledge_count == 0 {
            0.0
        } else {
            lineage_counts.values()
                .map(|count| {
                    let share = *count as f64 / knowledge_count as f64;
                    share * share
                })
                .sum()
        };

        NetworkedGenerationMetrics {
            generation: self.index,
            mean_quality: if n == 0 { 0.0 } else { total_quality as f64 / n as f64 },
            max_quality,
            seed_lineage_prevalence: if n == 0 { 0.0 } else { seed_count as f64 / n as f64 },
            knowledge_coverage: if n == 0 { 0.0 } else { knowledge_count as f64 / n as f64 },
            unique_nonzero_lineages,
            lineage_hhi,
            effective_lineage_diversity: if lineage_hhi > 0.0 { 1.0 / lineage_hhi } else { 0.0 },
            individual_innovations,
            peer_adoptions,
            mediated_adoptions,
            peer_acquisitions,
            mediated_acquisitions,
            observations,
            successful_acquisitions,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NetworkedMediationRun {
    pub graph: SocialGraph,
    pub generations: Vec<NetworkedGeneration>,
}

impl NetworkedMediationRun {
    pub fn metrics(&self) -> Vec<NetworkedGenerationMetrics> {
        self.generations.iter().map(NetworkedGeneration::metrics).collect()
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

}

/// Rank global recommender candidates using normalized performance and,
/// optionally, a bounded rarity bonus for underrepresented lineages.
/// Unknown/empty knowledge receives no novelty bonus.
fn rank_recommender_candidates(
    candidates: &[usize],
    previous: &[NetworkedAgent],
    objective: RecommenderObjective,
    noise_std: f64,
    rng: &mut DeterministicRng,
) -> Vec<(usize, f64)> {
    let mut lineage_counts = BTreeMap::<u64, usize>::new();
    let mut knowledgeable = 0usize;
    let max_quality = candidates.iter()
        .map(|&index| previous[index].quality)
        .max()
        .unwrap_or(0);

    for &index in candidates {
        let agent = &previous[index];
        if agent.quality > 0 && agent.lineage_id != 0 {
            knowledgeable += 1;
            *lineage_counts.entry(agent.lineage_id).or_default() += 1;
        }
    }

    let mut scored: Vec<(usize, f64)> = candidates.iter().copied().map(|index| {
        let agent = &previous[index];
        let performance = if max_quality > 0 {
            f64::from(agent.quality) / f64::from(max_quality)
        } else {
            0.0
        };
        let rarity = if agent.quality > 0 && knowledgeable > 0 {
            let count = *lineage_counts.get(&agent.lineage_id).unwrap_or(&0);
            1.0 - count as f64 / knowledgeable as f64
        } else {
            0.0
        };
        let novelty = match objective {
            RecommenderObjective::PerformanceOnly => 0.0,
            RecommenderObjective::PerformanceDiversity { novelty_weight } => {
                novelty_weight * rarity
            }
        };
        (index, performance + novelty + noise_std * rng.normal())
    }).collect();

    scored.sort_by(|left, right| {
        right.1.total_cmp(&left.1).then_with(|| left.0.cmp(&right.0))
    });
    scored
}

fn choose_softmax_candidate(
    scored: &[(usize, f64)],
    temperature: f64,
    rng: &mut DeterministicRng,
) -> Option<usize> {
    let maximum = scored.iter().map(|(_, score)| *score).fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = scored.iter()
        .map(|(_, score)| ((*score - maximum) / temperature).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return scored.first().map(|(index, _)| *index);
    }
    let mut draw = rng.uniform() * total;
    for ((index, _), weight) in scored.iter().zip(weights) {
        if draw < weight {
            return Some(*index);
        }
        draw -= weight;
    }
    scored.last().map(|(index, _)| *index)
}

/// Run a fixed social graph over successive generations. The recommender sees
/// all prior-generation agents and exposes only its top-k demonstration set;
/// peer-only learners see the previous-generation states of their graph
/// neighbors. The same graph is retained across generations.
pub fn simulate_networked_mediation(
    config: NetworkedMediationConfig,
    seed: u64,
) -> Result<NetworkedMediationRun, MediationError> {
    config.validate()?;
    let mut rng = DeterministicRng::new(seed);
    let graph = SocialGraph::ring_with_shortcuts(
        config.population_size,
        config.shortcut_probability,
        seed ^ 0xA5A5_5A5A_1234_5678,
    )?;

    let mut lineage_counter = 2u64;
    let mut initial = vec![
        NetworkedAgent {
            quality: 0,
            lineage_id: 0,
            source: KnowledgeSource::None,
            individual_innovated: false,
            demonstration_observed: false,
            strategy_acquisition_succeeded: false,
            acquisition_source: None,
            parent_index: None,
        };
        config.population_size
    ];
    for index in 0..config.initial_seed_agents {
        initial[index] = NetworkedAgent {
            quality: config.initial_seed_quality,
            lineage_id: 1,
            source: KnowledgeSource::InitialMachine,
            individual_innovated: false,
            demonstration_observed: false,
            strategy_acquisition_succeeded: false,
            acquisition_source: None,
            parent_index: None,
        };
    }

    let mut generations = Vec::with_capacity(config.generations);
    generations.push(NetworkedGeneration {
        index: 0,
        agents: initial,
    });

    for generation_index in 1..config.generations {
        let previous = &generations[generation_index - 1].agents;
        let global_candidates: Vec<usize> = (0..previous.len()).collect();
        let mut current = Vec::with_capacity(config.population_size);

        for learner_index in 0..config.population_size {
            // Individual innovation is independent of social access.
            let innovates = rng.bernoulli(config.individual_innovation_probability);
            let mut quality = if innovates { config.innovation_gain } else { 0 };
            let mut lineage_id = if innovates {
                let id = lineage_counter;
                lineage_counter = lineage_counter.wrapping_add(1).max(2);
                id
            } else {
                0
            };
            let mut source = if innovates {
                KnowledgeSource::IndividualInnovation
            } else {
                KnowledgeSource::None
            };
            let individual_innovated = innovates;
            let mut demonstration_observed = false;
            let mut strategy_acquisition_succeeded = false;
            let mut acquisition_source = None;
            let mut parent_index = None;

            let uses_recommender = rng.bernoulli(config.mediation_fraction);
            let (candidate_pool, selected_source) = if uses_recommender {
                (global_candidates.as_slice(), KnowledgeSource::AlgorithmicMediation)
            } else {
                (
                    graph.neighbors(learner_index).unwrap_or(&[]),
                    KnowledgeSource::PeerSocial,
                )
            };

            if !candidate_pool.is_empty() {
                demonstration_observed = rng.bernoulli(config.observation_probability);
            }
            if demonstration_observed {
                let mut scored: Vec<(usize, f64)> = if uses_recommender {
                    rank_recommender_candidates(
                        candidate_pool,
                        previous,
                        config.recommender_objective,
                        config.demonstration_noise_std,
                        &mut rng,
                    )
                } else {
                    candidate_pool.iter().copied()
                        .map(|candidate| (
                            candidate,
                            f64::from(previous[candidate].quality)
                                + config.demonstration_noise_std * rng.normal(),
                        ))
                        .collect()
                };
                if uses_recommender {
                    scored.truncate(config.recommender_candidate_count);
                } else {
                    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                }

                let teacher_index = if uses_recommender {
                    choose_softmax_candidate(&scored, config.recommender_temperature, &mut rng)
                } else {
                    scored.first().map(|(index, _)| *index)
                };
                if let Some(teacher_index) = teacher_index {
                    if previous[teacher_index].quality > 0
                        && rng.bernoulli(config.acquisition_probability)
                    {
                        strategy_acquisition_succeeded = true;
                        acquisition_source = Some(selected_source);
                        parent_index = Some(teacher_index);
                        let transmitted_quality = if rng.bernoulli(config.corruption_probability) {
                            previous[teacher_index].quality.saturating_sub(config.corruption_loss)
                        } else {
                            previous[teacher_index].quality
                        };
                        if transmitted_quality > quality {
                            quality = transmitted_quality;
                            lineage_id = previous[teacher_index].lineage_id;
                            source = selected_source;
                        }
                    }
                }
            }

            current.push(NetworkedAgent {
                quality,
                lineage_id,
                source,
                individual_innovated,
                demonstration_observed,
                strategy_acquisition_succeeded,
                acquisition_source,
                parent_index,
            });
        }

        generations.push(NetworkedGeneration {
            index: generation_index,
            agents: current,
        });
    }

    Ok(NetworkedMediationRun { graph, generations })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_is_connected_and_shortcuts_increase_density() {
        let ring = SocialGraph::ring_with_shortcuts(16, 0.0, 7).unwrap();
        let dense = SocialGraph::ring_with_shortcuts(16, 1.0, 7).unwrap();
        assert_eq!(ring.edge_count(), 16);
        assert_eq!(dense.edge_count(), 16 * 15 / 2);
        assert!(ring.neighbors(0).unwrap().contains(&1));
        assert!(ring.neighbors(0).unwrap().contains(&15));
        assert!(dense.density() > ring.density());
        for node in 0..16 {
            assert!(ring.neighbors(node).unwrap().len() >= 2);
        }
    }

    fn spreading_config(mediation_fraction: f64) -> NetworkedMediationConfig {
        NetworkedMediationConfig {
            population_size: 8,
            generations: 2,
            shortcut_probability: 0.0,
            mediation_fraction,
            recommender_candidate_count: 1,
            recommender_temperature: 0.1,
            recommender_objective: RecommenderObjective::PerformanceOnly,
            initial_seed_agents: 1,
            initial_seed_quality: 10,
            individual_innovation_probability: 0.0,
            innovation_gain: 1,
            observation_probability: 1.0,
            acquisition_probability: 1.0,
            corruption_probability: 0.0,
            corruption_loss: 1,
            demonstration_noise_std: 0.0,
        }
    }

    #[test]
    fn global_recommender_spreads_a_rare_skill_faster_than_ring_peers() {
        let peer = simulate_networked_mediation(spreading_config(0.0), 3).unwrap();
        let mediated = simulate_networked_mediation(spreading_config(1.0), 3).unwrap();
        let peer_seed_count = peer.generations[1].agents.iter()
            .filter(|agent| agent.lineage_id == 1).count();
        let mediated_seed_count = mediated.generations[1].agents.iter()
            .filter(|agent| agent.lineage_id == 1).count();

        assert_eq!(peer_seed_count, 2);
        assert_eq!(mediated_seed_count, 8);
    }

    #[test]
    fn peer_only_and_mediated_only_report_distinct_adoption_sources() {
        let peer = simulate_networked_mediation(spreading_config(0.0), 19).unwrap();
        let mediated = simulate_networked_mediation(spreading_config(1.0), 19).unwrap();
        let peer_metrics = peer.generations[1].metrics();
        let mediated_metrics = mediated.generations[1].metrics();

        assert_eq!(peer_metrics.peer_adoptions, 2);
        assert_eq!(peer_metrics.mediated_adoptions, 0);
        assert_eq!(mediated_metrics.peer_adoptions, 0);
        assert_eq!(mediated_metrics.mediated_adoptions, 8);
    }

    #[test]
    fn corruption_reduces_transmitted_skill_quality() {
        let clean = simulate_networked_mediation(spreading_config(1.0), 9).unwrap();
        let corrupted = simulate_networked_mediation(
            NetworkedMediationConfig {
                corruption_probability: 1.0,
                corruption_loss: 4,
                ..spreading_config(1.0)
            },
            9,
        ).unwrap();
        assert_eq!(clean.generations[1].metrics().mean_quality, 10.0);
        assert_eq!(corrupted.generations[1].metrics().mean_quality, 6.0);
    }

    #[test]
    fn observation_is_not_counted_as_acquisition() {
        let config = NetworkedMediationConfig {
            acquisition_probability: 0.0,
            ..spreading_config(1.0)
        };
        let run = simulate_networked_mediation(config, 31).unwrap();
        let metrics = run.generations[1].metrics();
        assert_eq!(metrics.observations, 8);
        assert_eq!(metrics.successful_acquisitions, 0);
        assert_eq!(metrics.mediated_acquisitions, 0);
        assert_eq!(metrics.mediated_adoptions, 0);
    }

    #[test]
    fn acquisition_is_not_counted_as_adoption_when_individual_skill_is_better() {
        let config = NetworkedMediationConfig {
            individual_innovation_probability: 1.0,
            innovation_gain: 20,
            ..spreading_config(1.0)
        };
        let run = simulate_networked_mediation(config, 37).unwrap();
        let metrics = run.generations[1].metrics();
        assert_eq!(metrics.successful_acquisitions, 8);
        assert_eq!(metrics.mediated_acquisitions, 8);
        assert_eq!(metrics.mediated_adoptions, 0);
        assert_eq!(metrics.individual_innovations, 8);
        assert_eq!(metrics.mean_quality, 20.0);
    }

    #[test]
    fn fixed_seed_is_reproducible() {
        let config = NetworkedMediationConfig::default();
        assert_eq!(
            simulate_networked_mediation(config, 2026).unwrap(),
            simulate_networked_mediation(config, 2026).unwrap()
        );
    }

    #[test]
    fn performance_diversity_ranking_can_resurface_a_rare_lineage() {
        let previous = vec![
            NetworkedAgent { quality: 10, lineage_id: 1, source: KnowledgeSource::InitialMachine,
                individual_innovated: false, demonstration_observed: false,
                strategy_acquisition_succeeded: false, acquisition_source: None, parent_index: None },
            NetworkedAgent { quality: 10, lineage_id: 1, source: KnowledgeSource::PeerSocial,
                individual_innovated: false, demonstration_observed: false,
                strategy_acquisition_succeeded: false, acquisition_source: None, parent_index: None },
            NetworkedAgent { quality: 10, lineage_id: 1, source: KnowledgeSource::PeerSocial,
                individual_innovated: false, demonstration_observed: false,
                strategy_acquisition_succeeded: false, acquisition_source: None, parent_index: None },
            NetworkedAgent { quality: 9, lineage_id: 2, source: KnowledgeSource::IndividualInnovation,
                individual_innovated: true, demonstration_observed: false,
                strategy_acquisition_succeeded: false, acquisition_source: None, parent_index: None },
        ];
        let candidates = [0, 1, 2, 3];
        let mut performance_rng = DeterministicRng::new(1);
        let mut diversity_rng = DeterministicRng::new(1);

        let performance = rank_recommender_candidates(
            &candidates, &previous, RecommenderObjective::PerformanceOnly, 0.0, &mut performance_rng,
        );
        let diversity = rank_recommender_candidates(
            &candidates, &previous,
            RecommenderObjective::PerformanceDiversity { novelty_weight: 1.0 },
            0.0, &mut diversity_rng,
        );

        assert_eq!(performance[0].0, 0);
        assert_eq!(diversity[0].0, 3);
    }

    #[test]
    fn diversity_recommender_never_invents_knowledge_for_empty_lineage() {
        let previous = vec![
            NetworkedAgent { quality: 0, lineage_id: 0, source: KnowledgeSource::None,
                individual_innovated: false, demonstration_observed: false,
                strategy_acquisition_succeeded: false, acquisition_source: None, parent_index: None },
            NetworkedAgent { quality: 4, lineage_id: 1, source: KnowledgeSource::InitialMachine,
                individual_innovated: false, demonstration_observed: false,
                strategy_acquisition_succeeded: false, acquisition_source: None, parent_index: None },
        ];
        let candidates = [0, 1];
        let mut rng = DeterministicRng::new(2);
        let ranked = rank_recommender_candidates(
            &candidates,
            &previous,
            RecommenderObjective::PerformanceDiversity { novelty_weight: 1.0 },
            0.0,
            &mut rng,
        );
        assert_eq!(ranked[0].0, 1);
    }

    #[test]
    fn invalid_topology_and_mediation_parameters_are_rejected() {
        assert_eq!(
            SocialGraph::ring_with_shortcuts(2, 0.0, 1),
            Err(MediationError::PopulationTooSmall)
        );
        let config = NetworkedMediationConfig {
            mediation_fraction: 1.1,
            ..NetworkedMediationConfig::default()
        };
        assert_eq!(
            simulate_networked_mediation(config, 1),
            Err(MediationError::InvalidProbability { name: "mediation_fraction" })
        );
        let invalid_novelty = NetworkedMediationConfig {
            recommender_objective: RecommenderObjective::PerformanceDiversity { novelty_weight: 1.1 },
            ..NetworkedMediationConfig::default()
        };
        assert_eq!(
            simulate_networked_mediation(invalid_novelty, 2),
            Err(MediationError::InvalidNoveltyWeight)
        );
    }
}
