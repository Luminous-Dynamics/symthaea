// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#![forbid(unsafe_code)]

//! Deterministic reference baselines for social-learning experiments.
//!
//! This module intentionally implements a binary-belief baseline and a
//! separate Reward Network task environment; neither claims to model human
//! culture generally.

pub mod adaptive_transmission;
pub mod cultural_evolution;
pub mod environmental_adaptation;
pub mod few_shot_transfer;
pub mod intentional_teaching;
pub mod czaplicka_2025;
pub mod cultural_situation;
pub mod paper_abm;
pub mod iterated_transmission;
pub mod networked_mediation;
pub mod reward_network;
pub mod reward_network_abm_reference;

use core::fmt;

/// One agent's benchmark state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgentState {
    /// Current proposition belief; None means uninformed.
    pub belief: Option<bool>,
    /// Confidence in the current belief, basis points (0..=10_000).
    pub confidence_bp: u16,
    /// Fixed experimental prestige used by PrestigeImitation.
    pub prestige: u16,
    /// Declared source-reliability weight, basis points (0..=10_000).
    ///
    /// This is a supplied experimental signal, not verified source quality.
    pub declared_reliability_bp: u16,
    /// Optional private/individual evidence observed independently of peers.
    pub private_signal: Option<bool>,
    /// Confidence assigned to the private signal, basis points (0..=10_000).
    pub private_signal_confidence_bp: u16,
}

impl AgentState {
    pub const fn uninformed(prestige: u16, declared_reliability_bp: u16) -> Self {
        Self {
            belief: None,
            confidence_bp: 0,
            prestige,
            declared_reliability_bp,
            private_signal: None,
            private_signal_confidence_bp: 0,
        }
    }

    /// Attach an independently acquired private signal for the IndividualLearning
    /// baseline. The signal is experiment input; it is not shared with peers.
    pub const fn with_private_signal(mut self, signal: bool, confidence_bp: u16) -> Self {
        self.private_signal = Some(signal);
        self.private_signal_confidence_bp = confidence_bp;
        self
    }

    pub const fn informed(
        belief: bool,
        confidence_bp: u16,
        prestige: u16,
        declared_reliability_bp: u16,
    ) -> Self {
        Self {
            belief: Some(belief),
            confidence_bp,
            prestige,
            declared_reliability_bp,
            private_signal: None,
            private_signal_confidence_bp: 0,
        }
    }

}

/// An undirected, deterministic communication/observation graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Network {
    adjacency: Vec<Vec<usize>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetworkError {
    ZeroNodes,
    NodeOutOfRange { node: usize, node_count: usize },
    SelfLoop { node: usize },
}

impl fmt::Display for NetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroNodes => write!(f, "network must contain at least one node"),
            Self::NodeOutOfRange { node, node_count } => {
                write!(f, "node {node} is outside network of size {node_count}")
            }
            Self::SelfLoop { node } => write!(f, "self-loop at node {node} is not allowed"),
        }
    }
}

impl std::error::Error for NetworkError {}

impl Network {
    /// Construct an undirected graph. Duplicate edges are canonicalized.
    pub fn from_edges(
        node_count: usize,
        edges: &[(usize, usize)],
    ) -> Result<Self, NetworkError> {
        if node_count == 0 {
            return Err(NetworkError::ZeroNodes);
        }

        let mut adjacency = vec![Vec::new(); node_count];
        for &(left, right) in edges {
            if left >= node_count {
                return Err(NetworkError::NodeOutOfRange {
                    node: left,
                    node_count,
                });
            }
            if right >= node_count {
                return Err(NetworkError::NodeOutOfRange {
                    node: right,
                    node_count,
                });
            }
            if left == right {
                return Err(NetworkError::SelfLoop { node: left });
            }
            adjacency[left].push(right);
            adjacency[right].push(left);
        }

        for neighbors in &mut adjacency {
            neighbors.sort_unstable();
            neighbors.dedup();
        }

        Ok(Self { adjacency })
    }

    pub fn node_count(&self) -> usize {
        self.adjacency.len()
    }

    pub fn neighbors(&self, node: usize) -> Option<&[usize]> {
        self.adjacency.get(node).map(Vec::as_slice)
    }
}

/// Transparent learning rules used as benchmark baselines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LearningRule {
    /// No social or private update; controls for spontaneous change.
    NoLearning,
    /// Use only an independently acquired private signal; ignores neighbours.
    IndividualLearning,
    /// Strict majority of informed neighbours; ties preserve current belief.
    MajorityImitation,
    /// Majority imitation only after a minimum number of informed neighbours.
    ThresholdMajority { minimum_known_neighbors: usize },
    /// Copy the highest-prestige informed neighbour; ties choose the lowest ID.
    PrestigeImitation,
    /// Weighted neighbour evidence plus an optional self-prior.
    EvidenceWeighted {
        self_prior_weight: u16,
        minimum_margin_bp: u16,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LearningError {
    GraphPopulationSizeMismatch { graph_nodes: usize, agents: usize },
    InvalidAgentValue { agent_id: usize, field: &'static str, value: u16 },
    InvalidMinimumKnownNeighbors,
    InvalidEvidenceMargin { margin_bp: u16 },
    InvalidSelfPriorWeight { weight: u16 },
}

impl fmt::Display for LearningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GraphPopulationSizeMismatch { graph_nodes, agents } => write!(
                f,
                "network has {graph_nodes} nodes but population has {agents} agents"
            ),
            Self::InvalidAgentValue { agent_id, field, value } => write!(
                f,
                "agent {agent_id} has {field} value {value} bp; expected 0..=10000"
            ),
            Self::InvalidMinimumKnownNeighbors => {
                write!(f, "minimum known-neighbour count must be positive")
            }
            Self::InvalidEvidenceMargin { margin_bp } => write!(
                f,
                "evidence margin {margin_bp} bp exceeds 10,000 bp"
            ),
            Self::InvalidSelfPriorWeight { weight } => write!(
                f,
                "self-prior weight {weight} bp exceeds 10,000 bp"
            ),
        }
    }
}

impl std::error::Error for LearningError {}

/// Population state. Ground truth is deliberately not stored here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Population {
    agents: Vec<AgentState>,
}

impl Population {
    pub fn new(agents: Vec<AgentState>) -> Self {
        Self { agents }
    }

    pub fn agents(&self) -> &[AgentState] {
        &self.agents
    }

    /// Apply one synchronous update. Every rule reads the old state and writes
    /// a separate next state, so vector iteration order cannot alter outcomes.
    pub fn step(
        &mut self,
        network: &Network,
        rule: LearningRule,
    ) -> Result<(), LearningError> {
        if network.node_count() != self.agents.len() {
            return Err(LearningError::GraphPopulationSizeMismatch {
                graph_nodes: network.node_count(),
                agents: self.agents.len(),
            });
        }

        for (agent_id, agent) in self.agents.iter().enumerate() {
            if agent.confidence_bp > 10_000 {
                return Err(LearningError::InvalidAgentValue {
                    agent_id,
                    field: "confidence",
                    value: agent.confidence_bp,
                });
            }
            if agent.declared_reliability_bp > 10_000 {
                return Err(LearningError::InvalidAgentValue {
                    agent_id,
                    field: "declared reliability",
                    value: agent.declared_reliability_bp,
                });
            }
            if agent.private_signal_confidence_bp > 10_000 {
                return Err(LearningError::InvalidAgentValue {
                    agent_id,
                    field: "private signal confidence",
                    value: agent.private_signal_confidence_bp,
                });
            }
            if agent.private_signal.is_none() && agent.private_signal_confidence_bp != 0 {
                return Err(LearningError::InvalidAgentValue {
                    agent_id,
                    field: "confidence without private signal",
                    value: agent.private_signal_confidence_bp,
                });
            }
        }

        match rule {
            LearningRule::ThresholdMajority {
                minimum_known_neighbors: 0,
            } => return Err(LearningError::InvalidMinimumKnownNeighbors),
            LearningRule::EvidenceWeighted {
                self_prior_weight,
                minimum_margin_bp,
            } => {
                if minimum_margin_bp > 10_000 {
                    return Err(LearningError::InvalidEvidenceMargin {
                        margin_bp: minimum_margin_bp,
                    });
                }
                if self_prior_weight > 10_000 {
                    return Err(LearningError::InvalidSelfPriorWeight {
                        weight: self_prior_weight,
                    });
                }
            }
            _ => {}
        }

        let previous = self.agents.clone();
        let mut next = previous.clone();

        for agent_id in 0..previous.len() {
            let neighbors = network
                .neighbors(agent_id)
                .expect("network and population sizes were checked");
            next[agent_id] = update_agent(agent_id, &previous, neighbors, rule);
        }

        self.agents = next;
        Ok(())
    }

    /// Evaluate against synthetic ground truth. The learning policies never
    /// receive this value; it is available only to the benchmark evaluator.
    pub fn metrics(
        &self,
        ground_truth: bool,
        high_confidence_threshold_bp: u16,
    ) -> PopulationMetrics {
        let informed = self.agents.iter().filter(|a| a.belief.is_some()).count();
        let correct = self
            .agents
            .iter()
            .filter(|a| a.belief == Some(ground_truth))
            .count();
        let incorrect = informed.saturating_sub(correct);
        let true_beliefs = self.agents.iter().filter(|a| a.belief == Some(true)).count();
        let false_beliefs = self.agents.iter().filter(|a| a.belief == Some(false)).count();
        let high_confidence_wrong = self
            .agents
            .iter()
            .filter(|a| {
                a.belief.is_some()
                    && a.belief != Some(ground_truth)
                    && a.confidence_bp >= high_confidence_threshold_bp
            })
            .count();

        let accuracy_among_informed = ratio(correct, informed);
        let coverage = ratio(informed, self.agents.len());
        let consensus_fraction = ratio(true_beliefs.max(false_beliefs), informed);
        let high_confidence_error_fraction = ratio(high_confidence_wrong, informed);
        let mean_confidence_bp = if informed == 0 {
            None
        } else {
            Some(
                self.agents
                    .iter()
                    .filter(|a| a.belief.is_some())
                    .map(|a| u64::from(a.confidence_bp))
                    .sum::<u64>() as f64
                    / informed as f64,
            )
        };

        PopulationMetrics {
            total_agents: self.agents.len(),
            informed_agents: informed,
            correct_agents: correct,
            incorrect_agents: incorrect,
            accuracy_among_informed,
            coverage,
            consensus_fraction,
            mean_confidence_bp,
            high_confidence_error_fraction,
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

fn update_agent(
    agent_id: usize,
    previous: &[AgentState],
    neighbors: &[usize],
    rule: LearningRule,
) -> AgentState {
    let current = previous[agent_id];

    match rule {
        LearningRule::NoLearning => current,
        LearningRule::IndividualLearning => current.private_signal.map_or(current, |signal| {
            AgentState {
                belief: Some(signal),
                confidence_bp: current.private_signal_confidence_bp,
                ..current
            }
        }),
        LearningRule::MajorityImitation => {
            majority_update(current, previous, neighbors, 1)
        }
        LearningRule::ThresholdMajority {
            minimum_known_neighbors,
        } => majority_update(current, previous, neighbors, minimum_known_neighbors),
        LearningRule::PrestigeImitation => {
            let mut selected: Option<usize> = None;
            for &neighbor in neighbors {
                if previous[neighbor].belief.is_none() {
                    continue;
                }
                match selected {
                    None => selected = Some(neighbor),
                    Some(best) => {
                        let candidate_key = (
                            previous[neighbor].prestige,
                            std::cmp::Reverse(neighbor),
                        );
                        let best_key = (
                            previous[best].prestige,
                            std::cmp::Reverse(best),
                        );
                        if candidate_key > best_key {
                            selected = Some(neighbor);
                        }
                    }
                }
            }

            selected.map_or(current, |source| AgentState {
                belief: previous[source].belief,
                confidence_bp: previous[source].confidence_bp,
                ..current
            })
        }
        LearningRule::EvidenceWeighted {
            self_prior_weight,
            minimum_margin_bp,
        } => {
            let mut true_weight = 0u64;
            let mut false_weight = 0u64;

            if let Some(belief) = current.belief {
                if belief {
                    true_weight += u64::from(self_prior_weight);
                } else {
                    false_weight += u64::from(self_prior_weight);
                }
            }

            for &neighbor in neighbors {
                let source = previous[neighbor];
                let weight = u64::from(source.declared_reliability_bp);
                match source.belief {
                    Some(true) => true_weight += weight,
                    Some(false) => false_weight += weight,
                    None => {}
                }
            }

            let total_weight = true_weight + false_weight;
            if total_weight == 0 || true_weight == false_weight {
                return current;
            }

            let (belief, winning_weight, losing_weight) = if true_weight > false_weight {
                (true, true_weight, false_weight)
            } else {
                (false, false_weight, true_weight)
            };
            let margin = winning_weight - losing_weight;
            let margin_bp = ((margin * 10_000) / total_weight).min(10_000) as u16;

            if margin_bp < minimum_margin_bp {
                current
            } else {
                AgentState {
                    belief: Some(belief),
                    confidence_bp: margin_bp,
                    ..current
                }
            }
        }
    }
}

fn majority_update(
    current: AgentState,
    previous: &[AgentState],
    neighbors: &[usize],
    minimum_known_neighbors: usize,
) -> AgentState {
    let mut true_votes = 0usize;
    let mut false_votes = 0usize;

    for &neighbor in neighbors {
        match previous[neighbor].belief {
            Some(true) => true_votes += 1,
            Some(false) => false_votes += 1,
            None => {}
        }
    }

    let known = true_votes + false_votes;
    if known < minimum_known_neighbors || true_votes == false_votes {
        return current;
    }

    let belief = true_votes > false_votes;
    let margin = true_votes.abs_diff(false_votes);
    let confidence_bp = ((margin * 10_000) / known) as u16;

    AgentState {
        belief: Some(belief),
        confidence_bp,
        ..current
    }
}

/// Disaggregated benchmark metrics. None denotes an undefined ratio because
/// the population has no informed agents; it must not be replaced with zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PopulationMetrics {
    pub total_agents: usize,
    pub informed_agents: usize,
    pub correct_agents: usize,
    pub incorrect_agents: usize,
    pub accuracy_among_informed: Option<f64>,
    pub coverage: Option<f64>,
    pub consensus_fraction: Option<f64>,
    pub mean_confidence_bp: Option<f64>,
    pub high_confidence_error_fraction: Option<f64>,
}

/// Metrics at step 0 and after every synchronous update.
pub fn run_history(
    initial: &Population,
    network: &Network,
    rule: LearningRule,
    ground_truth: bool,
    steps: usize,
    high_confidence_threshold_bp: u16,
) -> Result<Vec<PopulationMetrics>, LearningError> {
    let mut population = initial.clone();
    let mut history = Vec::with_capacity(steps + 1);
    history.push(population.metrics(ground_truth, high_confidence_threshold_bp));

    for _ in 0..steps {
        population.step(network, rule)?;
        history.push(population.metrics(ground_truth, high_confidence_threshold_bp));
    }

    Ok(history)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain(size: usize) -> Network {
        let edges: Vec<_> = (0..size.saturating_sub(1))
            .map(|i| (i, i + 1))
            .collect();
        Network::from_edges(size, &edges).unwrap()
    }

    #[test]
    fn chain_diffusion_is_synchronous_and_step_bounded() {
        let network = chain(5);
        let population = Population::new(vec![
            AgentState::informed(true, 10_000, 10, 10_000),
            AgentState::uninformed(0, 0),
            AgentState::uninformed(0, 0),
            AgentState::uninformed(0, 0),
            AgentState::uninformed(0, 0),
        ]);

        let history = run_history(
            &population,
            &network,
            LearningRule::MajorityImitation,
            true,
            4,
            8_000,
        )
        .unwrap();

        assert_eq!(history[0].informed_agents, 1);
        assert_eq!(history[1].informed_agents, 2);
        assert_eq!(history[2].informed_agents, 3);
        assert_eq!(history[3].informed_agents, 4);
        assert_eq!(history[4].informed_agents, 5);
        assert_eq!(history[4].accuracy_among_informed, Some(1.0));
    }

    #[test]
    fn individual_learning_uses_private_signal_and_ignores_neighbours() {
        let network = Network::from_edges(3, &[(0, 2), (1, 2)]).unwrap();
        let mut population = Population::new(vec![
            AgentState::informed(false, 10_000, 1, 10_000),
            AgentState::informed(false, 10_000, 1, 10_000),
            AgentState::uninformed(0, 0).with_private_signal(true, 7_500),
        ]);

        population
            .step(&network, LearningRule::IndividualLearning)
            .unwrap();
        assert_eq!(population.agents()[2].belief, Some(true));
        assert_eq!(population.agents()[2].confidence_bp, 7_500);
    }

    #[test]
    fn individual_learning_without_private_signal_does_not_copy_neighbours() {
        let network = Network::from_edges(2, &[(0, 1)]).unwrap();
        let mut population = Population::new(vec![
            AgentState::informed(true, 10_000, 10, 10_000),
            AgentState::uninformed(0, 0),
        ]);
        population
            .step(&network, LearningRule::IndividualLearning)
            .unwrap();
        assert_eq!(population.agents()[1].belief, None);
    }

    #[test]
    fn majority_ties_preserve_uninformed_state() {
        let network = Network::from_edges(3, &[(0, 2), (1, 2)]).unwrap();
        let mut population = Population::new(vec![
            AgentState::informed(true, 10_000, 1, 10_000),
            AgentState::informed(false, 10_000, 1, 10_000),
            AgentState::uninformed(0, 0),
        ]);

        population
            .step(&network, LearningRule::MajorityImitation)
            .unwrap();
        assert_eq!(population.agents()[2].belief, None);
    }

    #[test]
    fn evidence_weighted_rule_can_resist_low_weight_misinformation() {
        let network = Network::from_edges(3, &[(0, 2), (1, 2)]).unwrap();
        let mut population = Population::new(vec![
            AgentState::informed(true, 10_000, 1, 9_000),
            AgentState::informed(false, 10_000, 1, 1_000),
            AgentState::uninformed(0, 0),
        ]);

        population
            .step(
                &network,
                LearningRule::EvidenceWeighted {
                    self_prior_weight: 0,
                    minimum_margin_bp: 5_000,
                },
            )
            .unwrap();
        assert_eq!(population.agents()[2].belief, Some(true));
        assert_eq!(population.agents()[2].confidence_bp, 8_000);
    }

    #[test]
    fn prestige_imitation_selects_highest_prestige_with_stable_tie_break() {
        let network = Network::from_edges(3, &[(0, 2), (1, 2)]).unwrap();
        let mut population = Population::new(vec![
            AgentState::informed(true, 7_000, 10, 5_000),
            AgentState::informed(false, 8_000, 20, 5_000),
            AgentState::uninformed(0, 0),
        ]);
        population
            .step(&network, LearningRule::PrestigeImitation)
            .unwrap();
        assert_eq!(population.agents()[2].belief, Some(false));
        assert_eq!(population.agents()[2].confidence_bp, 8_000);
    }

    #[test]
    fn no_learning_is_a_zero_diffusion_control() {
        let network = chain(3);
        let initial = Population::new(vec![
            AgentState::informed(true, 10_000, 1, 10_000),
            AgentState::uninformed(0, 0),
            AgentState::uninformed(0, 0),
        ]);
        let mut population = initial.clone();
        population.step(&network, LearningRule::NoLearning).unwrap();
        assert_eq!(population, initial);
    }

    #[test]
    fn evaluator_separates_coverage_accuracy_and_high_confidence_errors() {
        let population = Population::new(vec![
            AgentState::informed(true, 9_000, 1, 10_000),
            AgentState::informed(false, 10_000, 1, 10_000),
            AgentState::uninformed(0, 0),
        ]);
        let metrics = population.metrics(true, 8_000);

        assert_eq!(metrics.coverage, Some(2.0 / 3.0));
        assert_eq!(metrics.accuracy_among_informed, Some(0.5));
        assert_eq!(metrics.high_confidence_error_fraction, Some(0.5));
        assert_eq!(metrics.correct_agents, 1);
        assert_eq!(metrics.incorrect_agents, 1);
    }

    #[test]
    fn no_informed_agents_produce_undefined_metrics_not_zero() {
        let population = Population::new(vec![AgentState::uninformed(0, 0)]);
        let metrics = population.metrics(true, 8_000);
        assert_eq!(metrics.coverage, Some(0.0));
        assert_eq!(metrics.accuracy_among_informed, None);
        assert_eq!(metrics.consensus_fraction, None);
        assert_eq!(metrics.mean_confidence_bp, None);
    }

    #[test]
    fn updates_are_deterministic_for_identical_initial_state() {
        let network = chain(6);
        let initial = Population::new(vec![
            AgentState::informed(true, 10_000, 5, 9_000),
            AgentState::uninformed(0, 1_000),
            AgentState::uninformed(0, 1_000),
            AgentState::informed(false, 4_000, 1, 1_000),
            AgentState::uninformed(0, 1_000),
            AgentState::uninformed(0, 1_000),
        ]);
        let a = run_history(
            &initial,
            &network,
            LearningRule::EvidenceWeighted {
                self_prior_weight: 500,
                minimum_margin_bp: 1_000,
            },
            true,
            5,
            8_000,
        )
        .unwrap();
        let b = run_history(
            &initial,
            &network,
            LearningRule::EvidenceWeighted {
                self_prior_weight: 500,
                minimum_margin_bp: 1_000,
            },
            true,
            5,
            8_000,
        )
        .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn graph_rejects_invalid_edges_and_deduplicates_valid_edges() {
        assert_eq!(Network::from_edges(2, &[(0, 2)]), Err(NetworkError::NodeOutOfRange { node: 2, node_count: 2 }));
        assert_eq!(Network::from_edges(2, &[(1, 1)]), Err(NetworkError::SelfLoop { node: 1 }));

        let graph = Network::from_edges(2, &[(0, 1), (1, 0), (0, 1)]).unwrap();
        assert_eq!(graph.neighbors(0), Some(&[1][..]));
        assert_eq!(graph.neighbors(1), Some(&[0][..]));
    }

    #[test]
    fn rejects_out_of_range_confidence_reliability_and_self_prior_weights() {
        let graph = chain(1);
        let mut invalid_confidence =
            Population::new(vec![AgentState::informed(true, 10_001, 0, 0)]);
        assert_eq!(
            invalid_confidence.step(&graph, LearningRule::NoLearning),
            Err(LearningError::InvalidAgentValue {
                agent_id: 0,
                field: "confidence",
                value: 10_001,
            })
        );

        let mut invalid_reliability =
            Population::new(vec![AgentState::informed(true, 0, 0, 10_001)]);
        assert_eq!(
            invalid_reliability.step(&graph, LearningRule::NoLearning),
            Err(LearningError::InvalidAgentValue {
                agent_id: 0,
                field: "declared reliability",
                value: 10_001,
            })
        );

        let mut valid_population = Population::new(vec![AgentState::uninformed(0, 0)]);
        assert_eq!(
            valid_population.step(
                &graph,
                LearningRule::EvidenceWeighted {
                    self_prior_weight: 10_001,
                    minimum_margin_bp: 0,
                }
            ),
            Err(LearningError::InvalidSelfPriorWeight { weight: 10_001 })
        );
    }

    #[test]
    fn rejects_invalid_learning_parameters_and_population_mismatch() {
        let graph = chain(2);
        let mut population = Population::new(vec![AgentState::uninformed(0, 0)]);
        assert_eq!(
            population.step(&graph, LearningRule::NoLearning),
            Err(LearningError::GraphPopulationSizeMismatch { graph_nodes: 2, agents: 1 })
        );

        let mut population = Population::new(vec![AgentState::uninformed(0, 0); 2]);
        assert_eq!(
            population.step(&graph, LearningRule::ThresholdMajority { minimum_known_neighbors: 0 }),
            Err(LearningError::InvalidMinimumKnownNeighbors)
        );
        assert_eq!(
            population.step(&graph, LearningRule::EvidenceWeighted { self_prior_weight: 0, minimum_margin_bp: 10_001 }),
            Err(LearningError::InvalidEvidenceMargin { margin_bp: 10_001 })
        );
    }
}
