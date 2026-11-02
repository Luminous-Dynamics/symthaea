// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#![forbid(unsafe_code)]

//! Reward Network task mechanics and transparent reference policies.
//! Tests use a synthetic graph, not an original experimental network.

use core::fmt;
use std::collections::HashSet;

pub const PUBLISHED_REWARD_VALUES: [i16; 5] = [-50, 0, 100, 200, 400];
pub const PUBLISHED_NODE_COUNT: usize = 12;
pub const PUBLISHED_EDGE_COUNT: usize = 30;
pub const PUBLISHED_MOVE_BUDGET: usize = 10;

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
    InvalidLevel { node: usize, level: u8 },
    EndpointOutOfRange { edge: usize, node: usize, nodes: usize },
    SelfLoop { edge: usize, node: usize },
    DuplicateEdge { from: usize, to: usize },
    UnsupportedReward { edge: usize, reward: i16 },
    PublishedShapeMismatch { nodes: usize, edges: usize },
    LevelThreeUnreachable { start: usize },
    TooFewLossesOnPath { start: usize, reached: usize, losses: usize },
    MaxRewardOutsideLevelThree { edge: usize, source_level: u8 },
}

impl fmt::Display for RewardNetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyNetwork => write!(f, "network must contain at least one node"),
            Self::StartOutOfRange { start, nodes } => write!(f, "start {start} outside {nodes} nodes"),
            Self::StartNotLevelZero { start, level } => write!(f, "start node {start} has level {level}, expected 0"),
            Self::InvalidLevel { node, level } => write!(f, "node {node} has level {level}, expected 0..=3"),
            Self::EndpointOutOfRange { edge, node, nodes } => write!(f, "edge {edge} endpoint {node} outside {nodes} nodes"),
            Self::SelfLoop { edge, node } => write!(f, "edge {edge} is a self-loop at {node}"),
            Self::DuplicateEdge { from, to } => write!(f, "duplicate directed edge {from}->{to}"),
            Self::UnsupportedReward { edge, reward } => write!(f, "edge {edge} has unsupported reward {reward}"),
            Self::PublishedShapeMismatch { nodes, edges } => write!(f, "expected 12 nodes/30 edges, got {nodes}/{edges}"),
            Self::LevelThreeUnreachable { start } => write!(f, "no level-3 node is reachable from level-0 start node {start}"),
            Self::TooFewLossesOnPath { start, reached, losses } => write!(f, "level-3 node {reached} is reachable from level-0 node {start} with only {losses} negative edges"),
            Self::MaxRewardOutsideLevelThree { edge, source_level } => write!(f, "edge {edge} offers +400 from level {source_level}, expected level 3"),
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
        for (node, &level) in node_levels.iter().enumerate() {
            if level > 3 { return Err(RewardNetworkError::InvalidLevel { node, level }); }
        }
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

    pub fn validate_published_shape(&self) -> Result<(), RewardNetworkError> {
        if self.node_levels.len() != PUBLISHED_NODE_COUNT || self.edges.len() != PUBLISHED_EDGE_COUNT {
            return Err(RewardNetworkError::PublishedShapeMismatch { nodes: self.node_levels.len(), edges: self.edges.len() });
        }
        Ok(())
    }

    /// Validate the published level/reward constraints on an imported task.
    ///
    /// This is a structural check only; it does not prove that the network
    /// originated in the article's experimental pool.
    pub fn validate_published_constraints(&self) -> Result<(), RewardNetworkError> {
        self.validate_published_shape()?;
        for (index, edge) in self.edges.iter().enumerate() {
            if edge.reward == 400 && self.node_levels[edge.from] != 3 {
                return Err(RewardNetworkError::MaxRewardOutsideLevelThree {
                    edge: index,
                    source_level: self.node_levels[edge.from],
                });
            }
        }

        // State = (node, negative-edge count capped at 3). This finite state
        // search detects any path from a level-0 node to level 3 with <3 losses,
        // including paths that contain cycles.
        for (start, &level) in self.node_levels.iter().enumerate() {
            if level != 0 { continue; }
            let mut visited = vec![[false; 4]; self.node_levels.len()];
            let mut queue = std::collections::VecDeque::new();
            queue.push_back((start, 0usize));
            visited[start][0] = true;
            let mut reached_level_three = false;

            while let Some((node, loss_count)) = queue.pop_front() {
                if self.node_levels[node] == 3 {
                    if loss_count < 3 {
                        return Err(RewardNetworkError::TooFewLossesOnPath {
                            start,
                            reached: node,
                            losses: loss_count,
                        });
                    }
                    reached_level_three = true;
                    continue;
                }
                for &edge_index in &self.outgoing[node] {
                    let edge = self.edges[edge_index];
                    let next_losses = (loss_count + if edge.reward < 0 { 1 } else { 0 }).min(3);
                    if !visited[edge.to][next_losses] {
                        visited[edge.to][next_losses] = true;
                        queue.push_back((edge.to, next_losses));
                    }
                }
            }

            if !reached_level_three {
                return Err(RewardNetworkError::LevelThreeUnreachable { start });
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic only: demonstrates the published three-loss incentive and
    /// the 12-node/30-edge shape. Not an original article network.
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

    #[test]
    fn synthetic_fixture_has_published_shape() {
        let n = synthetic_network();
        assert!(n.validate_published_shape().is_ok());
        assert!(n.validate_published_constraints().is_ok());
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
    fn published_constraint_rejects_shortcut_to_level_three() {
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
            n.validate_published_constraints(),
            Err(RewardNetworkError::TooFewLossesOnPath { start: 0, reached: 9, losses: 0 })
        ));
    }

    #[test]
    fn malformed_network_is_rejected() {
        assert_eq!(RewardNetwork::new(vec![0], vec![], 1), Err(RewardNetworkError::StartOutOfRange { start: 1, nodes: 1 }));
        assert_eq!(RewardNetwork::new(vec![0,4], vec![], 0), Err(RewardNetworkError::InvalidLevel { node: 1, level: 4 }));
        assert_eq!(RewardNetwork::new(vec![0], vec![RewardEdge { from: 0, to: 0, reward: 200 }], 0), Err(RewardNetworkError::SelfLoop { edge: 0, node: 0 }));
    }
}
