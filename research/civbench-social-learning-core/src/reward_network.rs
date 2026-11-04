// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#![forbid(unsafe_code)]

//! Reward Network task mechanics and transparent reference policies.
//! V0 models directed transitions with stable edge-index tie-breaking; verify
//! those semantics against the original archive before experimental replay.
//! Tests use a synthetic graph, not an original experimental network.

use core::fmt;
use std::collections::HashSet;

pub const PUBLISHED_REWARD_VALUES: [i16; 5] = [-50, 0, 100, 200, 400];
/// Task structure explicitly specified by the 2026 paper.
pub const PUBLISHED_NODE_COUNT: usize = 12;
pub const PUBLISHED_EDGE_COUNT: usize = 30;
pub const PUBLISHED_MIN_NODE_LEVEL: u8 = 0;
pub const PUBLISHED_MAX_NODE_LEVEL: u8 = 3;
pub const PUBLISHED_TERMINAL_LEVEL: u8 = 3;
pub const PUBLISHED_REQUIRED_NEGATIVE_EDGES: usize = 3;
/// The 2026 paper specifies ten moves through a reward network.
pub const PUBLISHED_MOVE_BUDGET: usize = 10;

/// Older public GitHub repository snapshot observations; not asserted to be
/// the final Zenodo v2.0 schema.
pub const LEGACY_GITHUB_NODE_COUNT: usize = 10;
pub const LEGACY_GITHUB_EDGE_COUNT: usize = 20;
pub const LEGACY_GITHUB_MOVE_BUDGET: usize = 8;
pub const LEGACY_GITHUB_OBSERVED_MAX_NODE_LEVEL: u8 = 4;
pub const LEGACY_GITHUB_VALIDATOR_MAX_NODE_LEVEL: u8 = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RewardEdge {
    pub from: usize,
    pub to: usize,
    pub reward: i16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewardNetwork {
    node_levels: Vec<u8>,
    edges: Vec<RewardEdge>,
    outgoing: Vec<Vec<usize>>,
    start_node: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RewardNetworkError {
    EmptyNetwork,
    StartOutOfRange { start: usize, nodes: usize },
    StartNotLevelZero { start: usize, level: u8 },
    EndpointOutOfRange { edge: usize, node: usize, nodes: usize },
    SelfLoop { edge: usize, node: usize },
    DuplicateEdge { from: usize, to: usize },
    UnsupportedReward { edge: usize, reward: i16 },
    ShapeMismatch { nodes: usize, edges: usize, expected_nodes: usize, expected_edges: usize },
    LevelOutOfRange { node: usize, level: u8, max_level: u8 },
    TerminalLevelUnreachable { start: usize, terminal_level: u8 },
    TooFewLossesOnPath { start: usize, reached: usize, losses: usize, required_losses: usize },
    MaxRewardOutsideTerminalLevel { edge: usize, source_level: u8, terminal_level: u8 },
}

impl fmt::Display for RewardNetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyNetwork => write!(f, "network must contain at least one node"),
            Self::StartOutOfRange { start, nodes } => write!(f, "start {start} outside {nodes} nodes"),
            Self::StartNotLevelZero { start, level } => write!(f, "start node {start} has level {level}, expected 0"),
            Self::EndpointOutOfRange { edge, node, nodes } => write!(f, "edge {edge} endpoint {node} outside {nodes} nodes"),
            Self::SelfLoop { edge, node } => write!(f, "edge {edge} is a self-loop at {node}"),
            Self::DuplicateEdge { from, to } => write!(f, "duplicate directed edge {from}->{to}"),
            Self::UnsupportedReward { edge, reward } => write!(f, "edge {edge} has unsupported reward {reward}"),
            Self::ShapeMismatch { nodes, edges, expected_nodes, expected_edges } => write!(f, "expected {expected_nodes} nodes/{expected_edges} edges, got {nodes}/{edges}"),
            Self::LevelOutOfRange { node, level, max_level } => write!(f, "node {node} has level {level}, above declared maximum {max_level}"),
            Self::TerminalLevelUnreachable { start, terminal_level } => write!(f, "terminal level {terminal_level} is unreachable from level-0 node {start}"),
            Self::TooFewLossesOnPath { start, reached, losses, required_losses } => write!(f, "terminal node {reached} is reachable from level-0 node {start} with {losses} negative edges; require at least {required_losses}"),
            Self::MaxRewardOutsideTerminalLevel { edge, source_level, terminal_level } => write!(f, "edge {edge} offers +400 from level {source_level}, expected terminal level {terminal_level}"),
        }
    }
}
impl std::error::Error for RewardNetworkError {}

impl RewardNetwork {
    pub fn new(node_levels: Vec<u8>, edges: Vec<RewardEdge>, start_node: usize) -> Result<Self, RewardNetworkError> {
        if node_levels.is_empty() { return Err(RewardNetworkError::EmptyNetwork); }
        if start_node >= node_levels.len() {
            return Err(RewardNetworkError::StartOutOfRange { start: start_node, nodes: node_levels.len() });
        }
        // Level bounds are source-profile specific. Validate separately so
        // archived task snapshots with different level ranges remain importable.
        if node_levels[start_node] != 0 {
            return Err(RewardNetworkError::StartNotLevelZero { start: start_node, level: node_levels[start_node] });
        }

        let mut seen = HashSet::with_capacity(edges.len());
        let mut outgoing = vec![Vec::new(); node_levels.len()];
        for (index, edge) in edges.iter().enumerate() {
            for endpoint in [edge.from, edge.to] {
                if endpoint >= node_levels.len() {
                    return Err(RewardNetworkError::EndpointOutOfRange { edge: index, node: endpoint, nodes: node_levels.len() });
                }
            }
            if edge.from == edge.to { return Err(RewardNetworkError::SelfLoop { edge: index, node: edge.from }); }
            if !PUBLISHED_REWARD_VALUES.contains(&edge.reward) {
                return Err(RewardNetworkError::UnsupportedReward { edge: index, reward: edge.reward });
            }
            if !seen.insert((edge.from, edge.to)) {
                return Err(RewardNetworkError::DuplicateEdge { from: edge.from, to: edge.to });
            }
            outgoing[edge.from].push(index);
        }
        for list in &mut outgoing { list.sort_unstable(); }
        Ok(Self { node_levels, edges, outgoing, start_node })
    }

    /// Validate a source-specific shape without baking one repository's
    /// topology into the generic task engine.
    pub fn validate_shape(&self, expected_nodes: usize, expected_edges: usize) -> Result<(), RewardNetworkError> {
        if self.node_levels.len() != expected_nodes || self.edges.len() != expected_edges {
            return Err(RewardNetworkError::ShapeMismatch {
                nodes: self.node_levels.len(), edges: self.edges.len(), expected_nodes, expected_edges,
            });
        }
        Ok(())
    }

    /// Validate this network against the task-level structure explicitly
    /// described in the 2026 paper: 12 nodes, 30 edges, levels 0..=3,
    /// +400 edges sourced only at level 3, and at least three negative edges
    /// on every directed path from a level-0 node to level 3.
    ///
    /// Passing these structural checks does not prove that a network came
    /// from the paper's experimental set or Zenodo v2.0 archive.
    pub fn validate_published_profile(&self) -> Result<(), RewardNetworkError> {
        self.validate_shape(PUBLISHED_NODE_COUNT, PUBLISHED_EDGE_COUNT)?;
        self.validate_level_range(PUBLISHED_MAX_NODE_LEVEL)?;
        self.validate_terminal_strategy(
            PUBLISHED_TERMINAL_LEVEL,
            PUBLISHED_REQUIRED_NEGATIVE_EDGES,
        )
    }

    /// Validate a source-specific node-level bound.
    pub fn validate_level_range(&self, max_level: u8) -> Result<(), RewardNetworkError> {
        if let Some((node, level)) = self.node_levels.iter().copied().enumerate()
            .find(|(_, level)| *level > max_level)
        {
            return Err(RewardNetworkError::LevelOutOfRange { node, level, max_level });
        }
        Ok(())
    }

    /// Validate a generic "losses before maximum reward" contract. This is a
    /// structural check, not proof of origin in the 2026 paper's experimental pool.
    pub fn validate_terminal_strategy(
        &self,
        terminal_level: u8,
        required_losses: usize,
    ) -> Result<(), RewardNetworkError> {
        for (index, edge) in self.edges.iter().enumerate() {
            if edge.reward == 400 && self.node_levels[edge.from] != terminal_level {
                return Err(RewardNetworkError::MaxRewardOutsideTerminalLevel {
                    edge: index,
                    source_level: self.node_levels[edge.from],
                    terminal_level,
                });
            }
        }

        // State = (node, negative-edge count capped at required_losses).
        // Check every existing directed path from every level-0 node. A
        // disconnected non-start level-0 node is allowed; the designated
        // task start must have at least one route to the terminal level.
        let mut start_reaches_terminal = false;
        for (start, &level) in self.node_levels.iter().enumerate() {
            if level != 0 {
                continue;
            }
            let mut visited = vec![
                vec![false; required_losses.saturating_add(1)];
                self.node_levels.len()
            ];
            let mut queue = std::collections::VecDeque::new();
            queue.push_back((start, 0usize));
            visited[start][0] = true;

            while let Some((node, loss_count)) = queue.pop_front() {
                if self.node_levels[node] == terminal_level {
                    if loss_count < required_losses {
                        return Err(RewardNetworkError::TooFewLossesOnPath {
                            start,
                            reached: node,
                            losses: loss_count,
                            required_losses,
                        });
                    }
                    if start == self.start_node {
                        start_reaches_terminal = true;
                    }
                    continue;
                }

                for &edge_index in &self.outgoing[node] {
                    let edge = self.edges[edge_index];
                    let next_losses = (
                        loss_count + if edge.reward < 0 { 1 } else { 0 }
                    ).min(required_losses);
                    if !visited[edge.to][next_losses] {
                        visited[edge.to][next_losses] = true;
                        queue.push_back((edge.to, next_losses));
                    }
                }
            }
        }

        if !start_reaches_terminal {
            return Err(RewardNetworkError::TerminalLevelUnreachable {
                start: self.start_node,
                terminal_level,
            });
        }
        Ok(())
    }

    pub fn node_count(&self) -> usize { self.node_levels.len() }
    pub fn edges(&self) -> &[RewardEdge] { &self.edges }
    pub fn outgoing_edge_indices(&self, node: usize) -> Option<&[usize]> { self.outgoing.get(node).map(Vec::as_slice) }
    pub fn start_node(&self) -> usize { self.start_node }
    /// Evaluator-only hidden level. Do not expose this to a participant policy.
    pub fn evaluator_node_level(&self, node: usize) -> Option<u8> { self.node_levels.get(node).copied() }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RewardPolicy {
    Random { seed: u64 },
    ImmediateGreedy,
    /// Prefer an available -50 edge, otherwise choose the best immediate reward.
    LossSeeking,
    /// Prefer -50 edges until the declared loss count is reached, then be greedy.
    LossThenGreedy { losses_before_switch: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoveRecord {
    pub step: usize,
    pub from: usize,
    pub to: usize,
    pub reward: i16,
    pub cumulative_reward: i32,
    pub cumulative_losses: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RewardEpisode {
    pub moves: Vec<MoveRecord>,
    pub total_reward: i32,
    pub losses: usize,
    pub terminal_node: usize,
}

fn best_edge(network: &RewardNetwork, candidates: &[usize]) -> Option<usize> {
    candidates.iter().copied().max_by(|&a, &b| {
        network.edges[a].reward.cmp(&network.edges[b].reward).then_with(|| b.cmp(&a))
    })
}
fn loss_edge(network: &RewardNetwork, candidates: &[usize]) -> Option<usize> {
    candidates.iter().copied().filter(|&i| network.edges[i].reward == -50).min()
}
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub fn play_episode(network: &RewardNetwork, policy: RewardPolicy, move_budget: usize) -> RewardEpisode {
    let mut current = network.start_node;
    let mut total = 0i32;
    let mut losses = 0usize;
    let mut moves = Vec::with_capacity(move_budget);
    let mut rng = match policy { RewardPolicy::Random { seed } => seed, _ => 0 };

    for step in 0..move_budget {
        let candidates = &network.outgoing[current];
        if candidates.is_empty() { break; }
        let selected = match policy {
            RewardPolicy::Random { .. } => candidates[(splitmix64(&mut rng) as usize) % candidates.len()],
            RewardPolicy::ImmediateGreedy => best_edge(network, candidates).expect("nonempty"),
            RewardPolicy::LossSeeking => loss_edge(network, candidates).or_else(|| best_edge(network, candidates)).expect("nonempty"),
            RewardPolicy::LossThenGreedy { losses_before_switch } => {
                if losses < losses_before_switch {
                    loss_edge(network, candidates).or_else(|| best_edge(network, candidates)).expect("nonempty")
                } else { best_edge(network, candidates).expect("nonempty") }
            }
        };
        let edge = network.edges[selected];
        total += i32::from(edge.reward);
        if edge.reward < 0 { losses += 1; }
        moves.push(MoveRecord {
            step, from: edge.from, to: edge.to, reward: edge.reward,
            cumulative_reward: total, cumulative_losses: losses,
        });
        current = edge.to;
    }
    RewardEpisode { moves, total_reward: total, losses, terminal_node: current }
}

/// Full-information finite-horizon upper bound; not an admissible participant policy.
pub fn full_information_oracle(network: &RewardNetwork, move_budget: usize) -> i32 {
    fn best(network: &RewardNetwork, node: usize, remaining: usize, memo: &mut [Vec<Option<i32>>]) -> i32 {
        if remaining == 0 { return 0; }
        if let Some(value) = memo[node][remaining] { return value; }
        if network.outgoing[node].is_empty() {
            memo[node][remaining] = Some(0);
            return 0;
        }
        let mut value = i32::MIN;
        for &index in &network.outgoing[node] {
            let edge = network.edges[index];
            value = value.max(i32::from(edge.reward) + best(network, edge.to, remaining - 1, memo));
        }
        memo[node][remaining] = Some(value);
        value
    }
    let mut memo = vec![vec![None; move_budget + 1]; network.node_count()];
    best(network, network.start_node, move_budget, &mut memo)
}

/// One zero-shot policy evaluation on a named source profile.
#[derive(Debug, Clone, PartialEq)]
pub struct TransferProfileResult {
    pub profile_id: String,
    pub policy_reward: i32,
    pub oracle_reward: i32,
    pub regret: i32,
    /// Policy reward divided by the full-information oracle reward.
    /// None when oracle reward is zero or negative.
    pub oracle_fraction: Option<f64>,
    pub invalid_moves: usize,
}

/// Evaluate the same frozen policy on multiple graph profiles without
/// refitting it. This is a zero-shot rule-transfer harness; it is not a
/// claim about learned Symthaea representations or human skill transfer.
pub fn evaluate_zero_shot_transfer(
    profiles: &[(&str, &RewardNetwork, usize)],
    policy: RewardPolicy,
) -> Vec<TransferProfileResult> {
    profiles.iter().map(|(profile_id, network, move_budget)| {
        let episode = play_episode(network, policy, *move_budget);
        let oracle_reward = full_information_oracle(network, *move_budget);
        let invalid_moves = episode.moves.iter().filter(|movement| {
            network.outgoing_edge_indices(movement.from)
                .map(|indices| !indices.iter().any(|&index| {
                    let edge = network.edges[index];
                    edge.to == movement.to && edge.reward == movement.reward
                }))
                .unwrap_or(true)
        }).count();
        TransferProfileResult {
            profile_id: (*profile_id).to_string(),
            policy_reward: episode.total_reward,
            oracle_reward,
            regret: oracle_reward - episode.total_reward,
            oracle_fraction: if oracle_reward > 0 {
                Some(episode.total_reward as f64 / oracle_reward as f64)
            } else {
                None
            },
            invalid_moves,
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic only: demonstrates a three-loss route on a 12-node/30-edge
    /// graph. Not an original article or repository task network.
    fn synthetic_network() -> RewardNetwork {
        let mut edges = Vec::new();
        for from in 0..3 { for to in 0..3 { if from != to { edges.push(RewardEdge { from, to, reward: 200 }); } } }
        for from in 3..6 { for to in 3..6 { if from != to { edges.push(RewardEdge { from, to, reward: 100 }); } } }
        for from in 6..9 { for to in 6..9 { if from != to { edges.push(RewardEdge { from, to, reward: 0 }); } } }
        edges.extend([
            RewardEdge { from: 9, to: 10, reward: 400 },
            RewardEdge { from: 10, to: 11, reward: 400 },
            RewardEdge { from: 11, to: 9, reward: 400 },
        ]);
        for offset in 0..3 {
            edges.push(RewardEdge { from: offset, to: offset + 3, reward: -50 });
            edges.push(RewardEdge { from: offset + 3, to: offset + 6, reward: -50 });
            edges.push(RewardEdge { from: offset + 6, to: offset + 9, reward: -50 });
        }
        assert_eq!(edges.len(), 30);
        RewardNetwork::new(vec![0,0,0,1,1,1,2,2,2,3,3,3], edges, 0).unwrap()
    }

    /// Synthetic compatibility profile for the older public GitHub snapshot:
    /// 10 nodes, 20 directed edges, levels 0..=4 and an 8-transition trial.
    /// It is not copied from the historical 841-network file.
    fn synthetic_legacy_profile_network() -> RewardNetwork {
        let edges = vec![
            RewardEdge { from: 0, to: 1, reward: 100 },
            RewardEdge { from: 1, to: 0, reward: 100 },
            RewardEdge { from: 2, to: 3, reward: 100 },
            RewardEdge { from: 3, to: 2, reward: 100 },
            RewardEdge { from: 4, to: 5, reward: 0 },
            RewardEdge { from: 5, to: 4, reward: 0 },
            RewardEdge { from: 6, to: 7, reward: 100 },
            RewardEdge { from: 7, to: 6, reward: 100 },
            RewardEdge { from: 8, to: 9, reward: 400 },
            RewardEdge { from: 9, to: 8, reward: 400 },
            RewardEdge { from: 0, to: 2, reward: -50 },
            RewardEdge { from: 1, to: 3, reward: -50 },
            RewardEdge { from: 0, to: 3, reward: -50 },
            RewardEdge { from: 2, to: 4, reward: -50 },
            RewardEdge { from: 3, to: 5, reward: -50 },
            RewardEdge { from: 2, to: 5, reward: -50 },
            RewardEdge { from: 4, to: 6, reward: -50 },
            RewardEdge { from: 5, to: 7, reward: -50 },
            RewardEdge { from: 6, to: 8, reward: 200 },
            RewardEdge { from: 7, to: 9, reward: 200 },
        ];
        assert_eq!(edges.len(), LEGACY_GITHUB_EDGE_COUNT);
        RewardNetwork::new(
            vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            edges,
            0,
        ).unwrap()
    }

    #[test]
    fn synthetic_fixture_obeys_its_declared_test_profile() {
        let n = synthetic_network();
        assert!(n.validate_published_profile().is_ok());
        assert_eq!(n.evaluator_node_level(9), Some(3));
    }
    #[test]
    fn immediate_greedy_scores_2000_on_synthetic_fixture() {
        let e = play_episode(&synthetic_network(), RewardPolicy::ImmediateGreedy, 10);
        assert_eq!(e.total_reward, 2000);
        assert_eq!(e.losses, 0);
    }
    #[test]
    fn loss_seeking_scores_2650_on_synthetic_fixture() {
        let e = play_episode(&synthetic_network(), RewardPolicy::LossSeeking, 10);
        assert_eq!(e.total_reward, 2650);
        assert_eq!(e.losses, 3);
        assert!(e.moves.iter().take(3).all(|m| m.reward == -50));
        assert!(e.moves.iter().skip(3).all(|m| m.reward == 400));
    }
    #[test]
    fn loss_then_greedy_scores_2650_when_switching_after_three_losses() {
        let e = play_episode(&synthetic_network(), RewardPolicy::LossThenGreedy { losses_before_switch: 3 }, 10);
        assert_eq!(e.total_reward, 2650);
        assert_eq!(e.losses, 3);
    }
    #[test]
    fn full_information_oracle_scores_2650_on_synthetic_fixture() {
        assert_eq!(full_information_oracle(&synthetic_network(), 10), 2650);
    }
    #[test]
    fn seeded_random_policy_is_reproducible() {
        let n = synthetic_network();
        assert_eq!(play_episode(&n, RewardPolicy::Random { seed: 42 }, 10), play_episode(&n, RewardPolicy::Random { seed: 42 }, 10));
    }
    #[test]
    fn published_profile_rejects_wrong_shape_and_level_range() {
        let n = synthetic_network();
        assert_eq!(
            n.validate_shape(PUBLISHED_NODE_COUNT, PUBLISHED_EDGE_COUNT),
            Ok(())
        );

        let wrong_shape = RewardNetwork::new(
            vec![0; 1],
            vec![],
            0,
        ).unwrap();
        assert_eq!(
            wrong_shape.validate_published_profile(),
            Err(RewardNetworkError::ShapeMismatch {
                nodes: 1,
                edges: 0,
                expected_nodes: PUBLISHED_NODE_COUNT,
                expected_edges: PUBLISHED_EDGE_COUNT,
            })
        );

        let above_range = RewardNetwork::new(
            vec![0, 4],
            vec![],
            0,
        ).unwrap();
        assert_eq!(
            above_range.validate_level_range(PUBLISHED_MAX_NODE_LEVEL),
            Err(RewardNetworkError::LevelOutOfRange {
                node: 1,
                level: 4,
                max_level: PUBLISHED_MAX_NODE_LEVEL,
            })
        );
    }

    #[test]
    fn disconnected_non_start_level_zero_node_does_not_invalidate_task() {
        let network = RewardNetwork::new(
            vec![0, 0, 1, 2, 3],
            vec![
                RewardEdge { from: 0, to: 2, reward: -50 },
                RewardEdge { from: 2, to: 3, reward: -50 },
                RewardEdge { from: 3, to: 4, reward: -50 },
            ],
            0,
        ).unwrap();
        assert!(network.validate_terminal_strategy(3, 3).is_ok());
    }

    #[test]
    fn designated_start_must_reach_terminal_level() {
        let network = RewardNetwork::new(
            vec![0, 0, 3],
            vec![
                RewardEdge { from: 1, to: 2, reward: -50 },
            ],
            0,
        ).unwrap();
        assert_eq!(
            network.validate_terminal_strategy(3, 1),
            Err(RewardNetworkError::TerminalLevelUnreachable {
                start: 0,
                terminal_level: 3,
            })
        );
    }

    #[test]
    fn task_constraint_rejects_shortcut_to_terminal_level() {
        let n = synthetic_network();
        let mut edges = n.edges().to_vec();
        edges.push(RewardEdge { from: 0, to: 9, reward: 200 });
        edges.retain(|edge| !(edge.from == 0 && edge.to == 3));
        // Keep 30 edges while replacing one valid -50 transition with a shortcut.
        let n = RewardNetwork::new(
            (0..12).map(|i| (i / 3) as u8).collect(),
            edges,
            0,
        ).unwrap();
        assert!(matches!(
            n.validate_terminal_strategy(3, 3),
            Err(RewardNetworkError::TooFewLossesOnPath {
                start: 0, reached: 9, losses: 0, required_losses: 3,
            })
        ));
    }

    #[test]
    fn legacy_github_profile_fixture_is_valid_but_remains_synthetic() {
        let network = synthetic_legacy_profile_network();
        assert!(network.validate_shape(LEGACY_GITHUB_NODE_COUNT, LEGACY_GITHUB_EDGE_COUNT).is_ok());
        assert!(network.validate_level_range(LEGACY_GITHUB_OBSERVED_MAX_NODE_LEVEL).is_ok());
        assert!(network.validate_terminal_strategy(4, 3).is_ok());
    }

    #[test]
    fn legacy_profile_fixture_reproduces_its_own_eight_transition_score_envelope() {
        let network = synthetic_legacy_profile_network();
        let loss = play_episode(&network, RewardPolicy::LossSeeking, LEGACY_GITHUB_MOVE_BUDGET);
        let myopic = play_episode(&network, RewardPolicy::ImmediateGreedy, LEGACY_GITHUB_MOVE_BUDGET);
        assert_eq!(loss.total_reward, 1_650);
        assert_eq!(loss.losses, 3);
        assert_eq!(myopic.total_reward, 800);
        assert_eq!(myopic.losses, 0);
    }

    #[test]
    fn legacy_profile_full_information_oracle_matches_loss_path() {
        assert_eq!(
            full_information_oracle(&synthetic_legacy_profile_network(), LEGACY_GITHUB_MOVE_BUDGET),
            1_650
        );
    }

    #[test]
    fn loss_seeking_rule_transfers_zero_shot_across_different_synthetic_profiles() {
        let paper_incentive = synthetic_network();
        let legacy_profile = synthetic_legacy_profile_network();
        let profiles = [
            ("synthetic-ten-move-incentive", &paper_incentive, PUBLISHED_MOVE_BUDGET),
            ("synthetic-legacy-eight-transition", &legacy_profile, LEGACY_GITHUB_MOVE_BUDGET),
        ];
        let results = evaluate_zero_shot_transfer(&profiles, RewardPolicy::LossSeeking);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].policy_reward, 2_650);
        assert_eq!(results[0].oracle_reward, 2_650);
        assert_eq!(results[0].regret, 0);
        assert_eq!(results[1].policy_reward, 1_650);
        assert_eq!(results[1].oracle_reward, 1_650);
        assert_eq!(results[1].regret, 0);
        assert!(results.iter().all(|result| result.invalid_moves == 0));
    }

    #[test]
    fn transfer_harness_exposes_myopic_regret_on_both_profiles() {
        let paper_incentive = synthetic_network();
        let legacy_profile = synthetic_legacy_profile_network();
        let profiles = [
            ("synthetic-ten-move-incentive", &paper_incentive, PUBLISHED_MOVE_BUDGET),
            ("synthetic-legacy-eight-transition", &legacy_profile, LEGACY_GITHUB_MOVE_BUDGET),
        ];
        let results = evaluate_zero_shot_transfer(&profiles, RewardPolicy::ImmediateGreedy);
        assert_eq!(results[0].regret, 650);
        assert_eq!(results[1].regret, 850);
        assert!(results.iter().all(|result| result.invalid_moves == 0));
    }

    #[test]
    fn malformed_network_is_rejected() {
        assert_eq!(RewardNetwork::new(vec![0], vec![], 1), Err(RewardNetworkError::StartOutOfRange { start: 1, nodes: 1 }));
        let above_bound = RewardNetwork::new(vec![0, 4], vec![], 0).unwrap();
        assert_eq!(
            above_bound.validate_level_range(3),
            Err(RewardNetworkError::LevelOutOfRange { node: 1, level: 4, max_level: 3 })
        );
        assert_eq!(RewardNetwork::new(vec![0], vec![RewardEdge { from: 0, to: 0, reward: 200 }], 0), Err(RewardNetworkError::SelfLoop { edge: 0, node: 0 }));
    }
}
