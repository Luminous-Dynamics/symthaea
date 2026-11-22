// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Controlled structural-failure evaluation for junction–hypha graphs.
//!
//! This module measures whether declared demand pairs remain connected and
//! retain short geometric paths after explicit junction/segment removal.
//! It does not simulate cytoplasmic flow, conductance, biological viability,
//! repair dynamics, or time-dependent physiology. All lengths remain in the
//! source graph's organism-local coordinate unit.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, VecDeque};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::mycelial_junction_graph::{
    JunctionLengthUnit, MycelialJunctionGraphError, MycelialJunctionHyphaGraph,
};
use crate::mycelial_model::SpatialPoint3;

pub const MYCELIAL_FAILURE_PROFILE_SCHEMA_VERSION: u16 = 1;
pub const MYCELIAL_FAILURE_ATTACK_PLAN_SCHEMA_VERSION: u16 = 1;
pub const MAX_FAILURE_JUNCTIONS: usize = 256;
pub const MAX_FAILURE_SEGMENTS: usize = 512;
pub const MAX_FAILURE_DEMAND_PAIRS: usize = 256;
pub const MAX_FAILURE_DEMAND_SOURCES: usize = 32;
pub const MAX_FAILURE_SCENARIOS: usize = 768;
pub const MAX_FAILURE_ID_BYTES: usize = 128;
/// Largest total demand weight exactly representable as an IEEE-754 f64 integer.
pub const MAX_FAILURE_TOTAL_DEMAND_WEIGHT: u128 = 9_007_199_254_740_992;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialFailureAttackStrategy {
    RandomSegments,
    RandomJunctions,
    HighestDegreeJunctions,
    AdaptiveHighestDegreeJunctions,
    BetweennessJunctions,
    LowestDegreeJunctions,
    DemandPathLoadSegments,
    DemandPathLoadJunctions,
    AdaptiveDemandPathLoadSegments,
    AdaptiveDemandPathLoadJunctions,
    SpatialClusterJunctions,
}

impl MycelialFailureAttackStrategy {
    pub fn stable_id(self) -> &'static str {
        match self {
            Self::RandomSegments => "random-segments-v1",
            Self::RandomJunctions => "random-junctions-v1",
            Self::HighestDegreeJunctions => "highest-degree-junctions-v1",
            Self::AdaptiveHighestDegreeJunctions => "adaptive-highest-degree-junctions-v1",
            Self::BetweennessJunctions => "betweenness-junctions-v1",
            Self::LowestDegreeJunctions => "lowest-degree-junctions-v1",
            Self::DemandPathLoadSegments => "demand-path-load-segments-v1",
            Self::DemandPathLoadJunctions => "demand-path-load-junctions-v1",
            Self::AdaptiveDemandPathLoadSegments => "adaptive-demand-path-load-segments-v1",
            Self::AdaptiveDemandPathLoadJunctions => "adaptive-demand-path-load-junctions-v1",
            Self::SpatialClusterJunctions => "spatial-cluster-junctions-v1",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelialFailureAttackConfig {
    pub strategy: MycelialFailureAttackStrategy,
    pub seed: u64,
    /// Prefix budgets from one through this many failed entities.
    pub maximum_failure_budget: usize,
    /// Required only for SpatialClusterJunctions.
    pub spatial_seed_junction_id: Option<String>,
}

/// A deterministic ranked attack sequence and its full nested budget sweep.
/// Each scenario removes the first k IDs in ranked_entity_ids; thus the
/// budget-k failed set is a subset of budget-(k+1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelialFailureAttackPlan {
    pub schema_version: u16,
    pub source_graph_digest: String,
    pub demand_digest: String,
    pub strategy: MycelialFailureAttackStrategy,
    pub seed: u64,
    pub maximum_failure_budget: usize,
    pub spatial_seed_junction_id: Option<String>,
    /// Selected entity order, one choice per sequential intervention step.
    pub ranked_entity_ids: Vec<String>,
    /// Complete candidate ranking visible at each step, after prior removals.
    pub ranked_entity_ids_per_step: Vec<Vec<String>>,
    pub scenarios: Vec<MycelialFailureScenario>,
}

/// Synthetic or calibrated attack-delivery uncertainty. The probability is
/// the per-attempt chance that one requested entity removal is enacted. It is
/// not a biological failure probability unless independently calibrated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelialFailureDeliveryConfig {
    pub trials_per_scenario: usize,
    /// Fixed-point probability in [0, 1_000_000].
    pub removal_success_probability_micros: u32,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialFailureDeliveryTrial {
    pub scenario_id: String,
    pub trial_index: usize,
    pub realized_failed_segment_ids: Vec<String>,
    pub realized_failed_junction_ids: Vec<String>,
    pub weighted_reachability: Option<f64>,
    /// Demand-weighted directness among reachable demand pairs; None when no demand is reachable.
    pub weighted_mean_directness: Option<f64>,
    pub weighted_mean_network_path_length: Option<f64>,
    pub connected_components: usize,
    pub cycle_rank: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialFailureDeliveryScenarioSummary {
    pub scenario_id: String,
    pub attempted_entity_count: usize,
    pub trial_count: usize,
    pub mean_realized_failure_count: f64,
    pub probability_at_least_one_failure: f64,
    pub mean_weighted_reachability: Option<f64>,
    pub weighted_reachability_p05: Option<f64>,
    pub weighted_reachability_p50: Option<f64>,
    pub weighted_reachability_p95: Option<f64>,
    pub weighted_reachability_minimum: Option<f64>,
    pub weighted_reachability_maximum: Option<f64>,
    pub directness_defined_trial_count: usize,
    pub mean_weighted_directness_among_reachable: Option<f64>,
    pub mean_weighted_path_length_among_reachable: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialFailureDeliveryProfile {
    pub schema_version: u16,
    pub source_graph_digest: String,
    pub demand_digest: String,
    pub attack_plan_digest: String,
    pub config: MycelialFailureDeliveryConfig,
    pub trials: Vec<MycelialFailureDeliveryTrial>,
    pub scenarios: Vec<MycelialFailureDeliveryScenarioSummary>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialFailureAttackEvaluation {
    pub attack_plan_digest: String,
    pub resilience_profile: MycelialFailureResilienceProfile,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelialDemandPathLoad {
    pub entity_id: String,
    pub demand_weighted_path_load: u128,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelialDemandPair {
    pub id: String,
    pub source_junction_id: String,
    pub target_junction_id: String,
    /// Positive integer demand weight. Keep weights in the source domain's
    /// declared demand unit; no implicit normalization is performed.
    pub weight: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelialFailureScenario {
    pub scenario_id: String,
    /// Sorted, unique source hypha segment IDs removed by this intervention.
    pub failed_segment_ids: Vec<String>,
    /// Sorted, unique junction IDs removed by this intervention.
    pub failed_junction_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialDemandPairOutcome {
    pub demand_id: String,
    pub direct_distance_model_units: Option<f64>,
    pub network_path_length_model_units: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialFailureMetrics {
    pub scenario_id: String,
    pub failed_segment_ids: Vec<String>,
    pub failed_junction_ids: Vec<String>,
    pub demand_pair_outcomes: Vec<MycelialDemandPairOutcome>,
    pub failed_segment_count: usize,
    pub failed_junction_count: usize,
    pub remaining_junction_count: usize,
    pub remaining_segment_count: usize,
    pub connected_components: usize,
    pub cycle_rank: usize,
    pub demand_pair_count: usize,
    pub reachable_demand_pair_count: usize,
    pub unreachable_demand_pair_count: usize,
    pub total_demand_weight: u128,
    pub reachable_demand_weight: u128,
    pub unreachable_demand_weight: u128,
    pub weighted_reachability: Option<f64>,
    /// Demand-weighted Euclidean/network shortest-path ratio, over reachable
    /// demand pairs only. Unreachable weight is reported separately.
    pub weighted_mean_directness: Option<f64>,
    pub weighted_mean_network_path_length: Option<f64>,
    pub length_unit: JunctionLengthUnit,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialFailureResilienceProfile {
    pub schema_version: u16,
    pub source_graph_digest: String,
    pub demand_digest: String,
    /// Complete sorted demand-pair inventory used to derive all scenario weights.
    pub demands: Vec<MycelialDemandPair>,
    pub baseline: MycelialFailureMetrics,
    /// Sorted by scenario ID for canonical receipts.
    pub scenarios: Vec<MycelialFailureMetrics>,
    pub worst_case_weighted_reachability: Option<f64>,
    pub worst_case_weighted_directness: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MycelialFailureResilienceError {
    InvalidGraph(MycelialJunctionGraphError),
    TooManyJunctions(usize),
    TooManySegments(usize),
    TooManyDemandPairs(usize),
    TooManyDemandSources(usize),
    DemandWeightLimitExceeded(u128),
    TooManyScenarios(usize),
    InvalidDemand(String),
    DuplicateDemandId(String),
    UnknownDemandJunction(String),
    InvalidScenario(String),
    DuplicateScenarioId(String),
    InvalidAttackConfig(&'static str),
    InvalidAttackPlan(&'static str),
    UnknownSpatialSeedJunction(String),
    InvalidDeliveryConfig(&'static str),
    InvalidDeliveryProfile(&'static str),
    PathShorterThanEuclideanDistance {
        demand_id: String,
        direct_distance: f64,
        network_path_length: f64,
    },
    UnknownFailedSegment(String),
    UnknownFailedJunction(String),
    InvalidProfile(&'static str),
}

impl fmt::Display for MycelialFailureResilienceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGraph(error) => write!(f, "invalid source junction graph: {error}"),
            Self::TooManyJunctions(count) => {
                write!(f, "junction count {count} exceeds {MAX_FAILURE_JUNCTIONS}")
            }
            Self::TooManySegments(count) => {
                write!(f, "segment count {count} exceeds {MAX_FAILURE_SEGMENTS}")
            }
            Self::TooManyDemandPairs(count) => {
                write!(
                    f,
                    "demand-pair count {count} exceeds {MAX_FAILURE_DEMAND_PAIRS}"
                )
            }
            Self::TooManyDemandSources(count) => {
                write!(
                    f,
                    "distinct demand-source count {count} exceeds {MAX_FAILURE_DEMAND_SOURCES}"
                )
            }
            Self::DemandWeightLimitExceeded(total) => {
                write!(
                    f,
                    "total demand weight {total} exceeds exact f64 integer bound {MAX_FAILURE_TOTAL_DEMAND_WEIGHT}"
                )
            }
            Self::TooManyScenarios(count) => {
                write!(
                    f,
                    "failure-scenario count {count} exceeds {MAX_FAILURE_SCENARIOS}"
                )
            }
            Self::InvalidDemand(id) => write!(f, "invalid demand pair {id}"),
            Self::DuplicateDemandId(id) => write!(f, "duplicate demand-pair ID {id}"),
            Self::UnknownDemandJunction(id) => {
                write!(f, "demand pair references unknown junction {id}")
            }
            Self::InvalidScenario(id) => write!(f, "invalid failure scenario {id}"),
            Self::DuplicateScenarioId(id) => write!(f, "duplicate failure scenario ID {id}"),
            Self::InvalidAttackConfig(reason) => write!(f, "invalid failure attack configuration: {reason}"),
            Self::InvalidAttackPlan(reason) => write!(f, "invalid failure attack plan: {reason}"),
            Self::UnknownSpatialSeedJunction(id) => write!(f, "unknown spatial attack seed junction {id}"),
            Self::InvalidDeliveryConfig(reason) => write!(f, "invalid attack-delivery uncertainty config: {reason}"),
            Self::InvalidDeliveryProfile(reason) => write!(f, "invalid attack-delivery uncertainty profile: {reason}"),
            Self::PathShorterThanEuclideanDistance {
                demand_id,
                direct_distance,
                network_path_length,
            } => write!(
                f,
                "demand {demand_id} has network path length {network_path_length:.6} below Euclidean distance {direct_distance:.6}"
            ),
            Self::UnknownFailedSegment(id) => write!(f, "failure references unknown segment {id}"),
            Self::UnknownFailedJunction(id) => {
                write!(f, "failure references unknown junction {id}")
            }
            Self::InvalidProfile(reason) => {
                write!(f, "invalid failure-resilience profile: {reason}")
            }
        }
    }
}

impl Error for MycelialFailureResilienceError {}

impl From<MycelialJunctionGraphError> for MycelialFailureResilienceError {
    fn from(value: MycelialJunctionGraphError) -> Self {
        Self::InvalidGraph(value)
    }
}

#[derive(Debug, Clone)]
struct IndexedGraph {
    node_ids: Vec<String>,
    positions: Vec<SpatialPoint3>,
    edge_ids: Vec<String>,
    edge_endpoints: Vec<(usize, usize, f64)>,
    node_index: HashMap<String, usize>,
    edge_index: HashMap<String, usize>,
}

fn index_source_graph(
    graph: &MycelialJunctionHyphaGraph,
) -> Result<IndexedGraph, MycelialFailureResilienceError> {
    graph.validate()?;
    if graph.junctions.len() > MAX_FAILURE_JUNCTIONS {
        return Err(MycelialFailureResilienceError::TooManyJunctions(
            graph.junctions.len(),
        ));
    }
    if graph.segments.len() > MAX_FAILURE_SEGMENTS {
        return Err(MycelialFailureResilienceError::TooManySegments(
            graph.segments.len(),
        ));
    }

    let mut sorted_junctions: Vec<_> = graph.junctions.iter().collect();
    sorted_junctions.sort_by(|left, right| left.id.cmp(&right.id));
    let node_ids: Vec<_> = sorted_junctions
        .iter()
        .map(|junction| junction.id.clone())
        .collect();
    let positions: Vec<_> = sorted_junctions
        .iter()
        .map(|junction| junction.position)
        .collect();
    let node_index: HashMap<_, _> = node_ids
        .iter()
        .enumerate()
        .map(|(index, id)| (id.clone(), index))
        .collect();

    let mut sorted_segments: Vec<_> = graph.segments.iter().collect();
    sorted_segments.sort_by(|left, right| left.id.cmp(&right.id));
    let edge_ids: Vec<_> = sorted_segments
        .iter()
        .map(|segment| segment.id.clone())
        .collect();
    let edge_index: HashMap<_, _> = edge_ids
        .iter()
        .enumerate()
        .map(|(index, id)| (id.clone(), index))
        .collect();
    let mut edge_endpoints = Vec::with_capacity(sorted_segments.len());
    for segment in sorted_segments {
        let source = *node_index.get(&segment.start_junction_id).ok_or(
            MycelialFailureResilienceError::UnknownDemandJunction(
                segment.start_junction_id.clone(),
            ),
        )?;
        let target = *node_index.get(&segment.end_junction_id).ok_or(
            MycelialFailureResilienceError::UnknownDemandJunction(segment.end_junction_id.clone()),
        )?;
        edge_endpoints.push((source, target, segment.traversal_length_model_units()));
    }

    Ok(IndexedGraph {
        node_ids,
        positions,
        edge_ids,
        edge_endpoints,
        node_index,
        edge_index,
    })
}

fn validate_demands(
    indexed: &IndexedGraph,
    demands: &[MycelialDemandPair],
) -> Result<(), MycelialFailureResilienceError> {
    if demands.len() > MAX_FAILURE_DEMAND_PAIRS {
        return Err(MycelialFailureResilienceError::TooManyDemandPairs(
            demands.len(),
        ));
    }
    let mut demand_ids = BTreeSet::new();
    let mut sources = BTreeSet::new();
    let mut previous_id: Option<&str> = None;
    let mut total_weight = 0u128;
    for demand in demands {
        if !valid_id(&demand.id)
            || demand.weight == 0
            || demand.source_junction_id == demand.target_junction_id
            || previous_id.is_some_and(|previous| previous >= demand.id.as_str())
            || !demand_ids.insert(demand.id.as_str())
        {
            return Err(MycelialFailureResilienceError::InvalidDemand(
                demand.id.clone(),
            ));
        }
        previous_id = Some(&demand.id);
        total_weight = total_weight.saturating_add(u128::from(demand.weight));
        if !indexed.node_index.contains_key(&demand.source_junction_id) {
            return Err(MycelialFailureResilienceError::UnknownDemandJunction(
                demand.source_junction_id.clone(),
            ));
        }
        if !indexed.node_index.contains_key(&demand.target_junction_id) {
            return Err(MycelialFailureResilienceError::UnknownDemandJunction(
                demand.target_junction_id.clone(),
            ));
        }
        sources.insert(demand.source_junction_id.as_str());
    }
    if sources.len() > MAX_FAILURE_DEMAND_SOURCES {
        return Err(MycelialFailureResilienceError::TooManyDemandSources(
            sources.len(),
        ));
    }
    if total_weight > MAX_FAILURE_TOTAL_DEMAND_WEIGHT {
        return Err(MycelialFailureResilienceError::DemandWeightLimitExceeded(
            total_weight,
        ));
    }
    Ok(())
}

fn validate_scenarios(
    indexed: &IndexedGraph,
    scenarios: &[MycelialFailureScenario],
) -> Result<(), MycelialFailureResilienceError> {
    if scenarios.len() > MAX_FAILURE_SCENARIOS {
        return Err(MycelialFailureResilienceError::TooManyScenarios(
            scenarios.len(),
        ));
    }
    let mut scenario_ids = BTreeSet::new();
    for scenario in scenarios {
        if !valid_id(&scenario.scenario_id)
            || scenario.scenario_id == "baseline"
            || (scenario.failed_segment_ids.is_empty() && scenario.failed_junction_ids.is_empty())
            || !scenario_ids.insert(scenario.scenario_id.as_str())
            || scenario
                .failed_segment_ids
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || scenario
                .failed_junction_ids
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(MycelialFailureResilienceError::InvalidScenario(
                scenario.scenario_id.clone(),
            ));
        }
        for edge_id in &scenario.failed_segment_ids {
            if !indexed.edge_index.contains_key(edge_id) {
                return Err(MycelialFailureResilienceError::UnknownFailedSegment(
                    edge_id.clone(),
                ));
            }
        }
        for node_id in &scenario.failed_junction_ids {
            if !indexed.node_index.contains_key(node_id) {
                return Err(MycelialFailureResilienceError::UnknownFailedJunction(
                    node_id.clone(),
                ));
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct QueueEntry {
    node: usize,
    distance: f64,
}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node && self.distance.to_bits() == other.distance.to_bits()
    }
}
impl Eq for QueueEntry {}
impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .distance
            .total_cmp(&self.distance)
            .then_with(|| other.node.cmp(&self.node))
    }
}

fn dijkstra(adjacency: &[Vec<(usize, f64)>], source: usize) -> Vec<f64> {
    let mut distances = vec![f64::INFINITY; adjacency.len()];
    let mut queue = BinaryHeap::new();
    distances[source] = 0.0;
    queue.push(QueueEntry {
        node: source,
        distance: 0.0,
    });
    while let Some(current) = queue.pop() {
        if current.distance > distances[current.node] {
            continue;
        }
        for &(target, weight) in &adjacency[current.node] {
            let candidate = current.distance + weight;
            if candidate < distances[target] {
                distances[target] = candidate;
                queue.push(QueueEntry {
                    node: target,
                    distance: candidate,
                });
            }
        }
    }
    distances
}

fn component_count(adjacency: &[Vec<(usize, f64)>], active_nodes: &[bool]) -> usize {
    let mut visited = vec![false; adjacency.len()];
    let mut components = 0usize;
    for start in 0..adjacency.len() {
        if !active_nodes[start] || visited[start] {
            continue;
        }
        components += 1;
        visited[start] = true;
        let mut queue = VecDeque::from([start]);
        while let Some(current) = queue.pop_front() {
            for &(target, _) in &adjacency[current] {
                if active_nodes[target] && !visited[target] {
                    visited[target] = true;
                    queue.push_back(target);
                }
            }
        }
    }
    components
}

fn evaluate_one(
    indexed: &IndexedGraph,
    demands: &[MycelialDemandPair],
    scenario_id: &str,
    failed_segment_ids: &[String],
    failed_junction_ids: &[String],
    length_unit: JunctionLengthUnit,
) -> Result<MycelialFailureMetrics, MycelialFailureResilienceError> {
    let failed_edges: BTreeSet<_> = failed_segment_ids.iter().map(String::as_str).collect();
    let failed_nodes: BTreeSet<_> = failed_junction_ids.iter().map(String::as_str).collect();
    let active_nodes: Vec<_> = indexed
        .node_ids
        .iter()
        .map(|id| !failed_nodes.contains(id.as_str()))
        .collect();

    let mut adjacency = vec![Vec::new(); indexed.node_ids.len()];
    let mut remaining_segment_count = 0usize;
    for (edge_index, &(source, target, weight)) in indexed.edge_endpoints.iter().enumerate() {
        if failed_edges.contains(indexed.edge_ids[edge_index].as_str())
            || !active_nodes[source]
            || !active_nodes[target]
        {
            continue;
        }
        adjacency[source].push((target, weight));
        adjacency[target].push((source, weight));
        remaining_segment_count += 1;
    }
    for neighbours in &mut adjacency {
        neighbours.sort_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.total_cmp(&right.1))
        });
    }

    let remaining_junction_count = active_nodes.iter().filter(|active| **active).count();
    let connected_components = component_count(&adjacency, &active_nodes);
    let cycle_rank = remaining_segment_count
        .saturating_add(connected_components)
        .saturating_sub(remaining_junction_count);

    let mut distances_by_source = HashMap::<usize, Vec<f64>>::new();
    let mut outcomes = Vec::with_capacity(demands.len());
    let mut total_demand_weight = 0u128;
    let mut reachable_demand_weight = 0u128;
    let mut reachable_demand_pair_count = 0usize;
    let mut weighted_directness_sum = 0.0f64;
    let mut weighted_path_length_sum = 0.0f64;
    for demand in demands {
        let weight = u128::from(demand.weight);
        total_demand_weight += weight;
        let source = indexed.node_index[&demand.source_junction_id];
        let target = indexed.node_index[&demand.target_junction_id];
        if !active_nodes[source] || !active_nodes[target] {
            outcomes.push(MycelialDemandPairOutcome {
                demand_id: demand.id.clone(),
                direct_distance_model_units: None,
                network_path_length_model_units: None,
            });
            continue;
        }
        let distances = distances_by_source
            .entry(source)
            .or_insert_with(|| dijkstra(&adjacency, source));
        let path_length = distances[target];
        if !path_length.is_finite() {
            outcomes.push(MycelialDemandPairOutcome {
                demand_id: demand.id.clone(),
                direct_distance_model_units: None,
                network_path_length_model_units: None,
            });
            continue;
        }
        let direct_distance = f64::from(
            indexed.positions[source]
                .sub(indexed.positions[target])
                .length(),
        );
        if path_length <= 0.0 || direct_distance > path_length + 1.0e-5 {
            return Err(
                MycelialFailureResilienceError::PathShorterThanEuclideanDistance {
                    demand_id: demand.id.clone(),
                    direct_distance,
                    network_path_length: path_length,
                },
            );
        }
        reachable_demand_pair_count += 1;
        reachable_demand_weight += weight;
        weighted_directness_sum += (direct_distance / path_length) * weight as f64;
        weighted_path_length_sum += path_length * weight as f64;
        outcomes.push(MycelialDemandPairOutcome {
            demand_id: demand.id.clone(),
            direct_distance_model_units: Some(direct_distance),
            network_path_length_model_units: Some(path_length),
        });
    }
    let unreachable_demand_pair_count = demands.len().saturating_sub(reachable_demand_pair_count);
    let unreachable_demand_weight = total_demand_weight.saturating_sub(reachable_demand_weight);
    let weighted_reachability = (total_demand_weight > 0)
        .then_some(reachable_demand_weight as f64 / total_demand_weight as f64);
    let weighted_mean_directness = (reachable_demand_weight > 0)
        .then_some(weighted_directness_sum / reachable_demand_weight as f64);
    let weighted_mean_network_path_length = (reachable_demand_weight > 0)
        .then_some(weighted_path_length_sum / reachable_demand_weight as f64);

    Ok(MycelialFailureMetrics {
        scenario_id: scenario_id.to_string(),
        failed_segment_ids: failed_segment_ids.to_vec(),
        failed_junction_ids: failed_junction_ids.to_vec(),
        demand_pair_outcomes: outcomes,
        failed_segment_count: failed_segment_ids.len(),
        failed_junction_count: failed_junction_ids.len(),
        remaining_junction_count,
        remaining_segment_count,
        connected_components,
        cycle_rank,
        demand_pair_count: demands.len(),
        reachable_demand_pair_count,
        unreachable_demand_pair_count,
        total_demand_weight,
        reachable_demand_weight,
        unreachable_demand_weight,
        weighted_reachability,
        weighted_mean_directness,
        weighted_mean_network_path_length,
        length_unit,
    })
}

/// Evaluate baseline demand functionality and explicit controlled failures.
///
/// Demand pairs must be sorted by unique ID. Each scenario's failed IDs must
/// also be sorted and unique. Failure evaluation changes only a derived graph
/// view; it does not modify the source graph, accepted evidence, or morphology.
pub fn build_mycelial_failure_resilience_profile(
    graph: &MycelialJunctionHyphaGraph,
    demands: &[MycelialDemandPair],
    scenarios: &[MycelialFailureScenario],
) -> Result<MycelialFailureResilienceProfile, MycelialFailureResilienceError> {
    let indexed = index_source_graph(graph)?;
    validate_demands(&indexed, demands)?;
    validate_scenarios(&indexed, scenarios)?;
    let source_graph_digest = graph.canonical_digest_hex()?;
    let demand_digest = canonical_demand_digest(demands);
    let baseline = evaluate_one(&indexed, demands, "baseline", &[], &[], graph.length_unit)?;
    let mut ordered_scenarios: Vec<_> = scenarios.iter().collect();
    ordered_scenarios.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
    let metrics = ordered_scenarios
        .into_iter()
        .map(|scenario| {
            evaluate_one(
                &indexed,
                demands,
                &scenario.scenario_id,
                &scenario.failed_segment_ids,
                &scenario.failed_junction_ids,
                graph.length_unit,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let worst_case_weighted_reachability = metrics
        .iter()
        .filter_map(|metric| metric.weighted_reachability)
        .min_by(f64::total_cmp);
    let worst_case_weighted_directness = metrics
        .iter()
        .filter_map(|metric| metric.weighted_mean_directness)
        .min_by(f64::total_cmp);
    let profile = MycelialFailureResilienceProfile {
        schema_version: MYCELIAL_FAILURE_PROFILE_SCHEMA_VERSION,
        source_graph_digest,
        demand_digest,
        demands: demands.to_vec(),
        baseline,
        scenarios: metrics,
        worst_case_weighted_reachability,
        worst_case_weighted_directness,
    };
    profile.validate()?;
    Ok(profile)
}

/// Generate one intervention per hypha segment and one per junction. This is
/// an exhaustive single-failure structural stress sweep within the module's
/// bounds; it does not assume random or independent failures.
pub fn build_single_failure_resilience_profile(
    graph: &MycelialJunctionHyphaGraph,
    demands: &[MycelialDemandPair],
) -> Result<MycelialFailureResilienceProfile, MycelialFailureResilienceError> {
    let indexed = index_source_graph(graph)?;
    let mut scenarios = Vec::with_capacity(indexed.edge_ids.len() + indexed.node_ids.len());
    for edge_id in &indexed.edge_ids {
        scenarios.push(MycelialFailureScenario {
            scenario_id: format!("edge:{edge_id}"),
            failed_segment_ids: vec![edge_id.clone()],
            failed_junction_ids: Vec::new(),
        });
    }
    for node_id in &indexed.node_ids {
        scenarios.push(MycelialFailureScenario {
            scenario_id: format!("junction:{node_id}"),
            failed_segment_ids: Vec::new(),
            failed_junction_ids: vec![node_id.clone()],
        });
    }
    scenarios.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
    build_mycelial_failure_resilience_profile(graph, demands, &scenarios)
}

fn attack_strategy_targets_segments(strategy: MycelialFailureAttackStrategy) -> bool {
    matches!(
        strategy,
        MycelialFailureAttackStrategy::RandomSegments
            | MycelialFailureAttackStrategy::DemandPathLoadSegments
            | MycelialFailureAttackStrategy::AdaptiveDemandPathLoadSegments
    )
}

fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn failure_entity_tie_priority(seed: u64, entity_kind: u8, entity_id: &str) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-failure-attack-tie-priority-v1");
    hasher.update(&seed.to_le_bytes());
    hasher.update(&[entity_kind]);
    hash_string(&mut hasher, entity_id);
    let digest = hasher.finalize();
    u64::from_le_bytes(digest.as_bytes()[..8].try_into().expect("digest prefix is eight bytes"))
}

fn failure_entity_order(
    seed: u64,
    entity_kind: u8,
    left: &str,
    right: &str,
) -> Ordering {
    failure_entity_tie_priority(seed, entity_kind, left)
        .cmp(&failure_entity_tie_priority(seed, entity_kind, right))
        .then_with(|| left.cmp(right))
}

fn deterministic_shuffle<T>(values: &mut [T], seed: u64) {
    let mut state = seed;
    for index in (1..values.len()).rev() {
        let selected = (splitmix64_next(&mut state) % (index as u64 + 1)) as usize;
        values.swap(index, selected);
    }
}

fn demand_weighted_path_load(
    indexed: &IndexedGraph,
    demands: &[MycelialDemandPair],
    failed_segment_ids: &BTreeSet<String>,
    failed_junction_ids: &BTreeSet<String>,
) -> (Vec<u128>, Vec<u128>) {
    let mut adjacency = vec![Vec::<(usize, f64, usize)>::new(); indexed.node_ids.len()];
    let active_nodes: Vec<_> = indexed
        .node_ids
        .iter()
        .map(|id| !failed_junction_ids.contains(id))
        .collect();
    for (edge_index, &(source, target, weight)) in indexed.edge_endpoints.iter().enumerate() {
        if failed_segment_ids.contains(&indexed.edge_ids[edge_index])
            || !active_nodes[source]
            || !active_nodes[target]
        {
            continue;
        }
        adjacency[source].push((target, weight, edge_index));
        adjacency[target].push((source, weight, edge_index));
    }
    for neighbours in &mut adjacency {
        neighbours.sort_by(|left, right| {
            left.0.cmp(&right.0).then_with(|| left.2.cmp(&right.2))
        });
    }

    let mut segment_load = vec![0u128; indexed.edge_ids.len()];
    let mut junction_load = vec![0u128; indexed.node_ids.len()];
    let mut cache = HashMap::<usize, (Vec<f64>, Vec<Option<usize>>, Vec<Option<usize>>)>::new();

    for demand in demands {
        let source = indexed.node_index[&demand.source_junction_id];
        let target = indexed.node_index[&demand.target_junction_id];
        if !active_nodes[source] || !active_nodes[target] {
            continue;
        }
        let (distances, predecessor_node, predecessor_edge) = cache.entry(source).or_insert_with(|| {
            let mut distances = vec![f64::INFINITY; indexed.node_ids.len()];
            let mut predecessor_node = vec![None; indexed.node_ids.len()];
            let mut predecessor_edge = vec![None; indexed.node_ids.len()];
            let mut queue = BinaryHeap::new();
            distances[source] = 0.0;
            queue.push(QueueEntry {
                node: source,
                distance: 0.0,
            });
            while let Some(current) = queue.pop() {
                if current.distance > distances[current.node] {
                    continue;
                }
                for &(next, weight, edge_index) in &adjacency[current.node] {
                    let candidate = current.distance + weight;
                    if candidate < distances[next] {
                        distances[next] = candidate;
                        predecessor_node[next] = Some(current.node);
                        predecessor_edge[next] = Some(edge_index);
                        queue.push(QueueEntry {
                            node: next,
                            distance: candidate,
                        });
                    }
                }
            }
            (distances, predecessor_node, predecessor_edge)
        });
        // Unreachable demand pairs contribute no path load.
        if !distances[target].is_finite() {
            continue;
        }

        let mut cursor = target;
        let mut reversed_nodes = vec![target];
        let mut steps = 0usize;
        while cursor != source {
            let (Some(previous), Some(edge_index)) = (
                predecessor_node[cursor],
                predecessor_edge[cursor],
            ) else {
                // A positive-weight reachable path must have a complete predecessor chain.
                reversed_nodes.clear();
                break;
            };
            segment_load[edge_index] =
                segment_load[edge_index].saturating_add(u128::from(demand.weight));
            cursor = previous;
            reversed_nodes.push(cursor);
            steps += 1;
            if steps > indexed.node_ids.len() {
                reversed_nodes.clear();
                break;
            }
        }
        if reversed_nodes.is_empty() {
            continue;
        }
        // Demand-path junction load counts internal transit junctions, not
        // the originating or terminating demand endpoints.
        for node_index in reversed_nodes.iter().skip(1).take(reversed_nodes.len().saturating_sub(2)) {
            junction_load[*node_index] =
                junction_load[*node_index].saturating_add(u128::from(demand.weight));
        }
    }
    (segment_load, junction_load)
}

/// Brandes unweighted undirected node betweenness. Shortest-path
/// multiplicities are accumulated in the log domain to avoid integer overflow
/// on graphs with many equal-length routes. Scores are divided by two for the
/// undirected graph convention; this does not alter ranking.
fn unweighted_static_betweenness(indexed: &IndexedGraph) -> Vec<f64> {
    let node_count = indexed.node_ids.len();
    let mut adjacency = vec![Vec::<usize>::new(); node_count];
    for &(source, target, _) in &indexed.edge_endpoints {
        adjacency[source].push(target);
        adjacency[target].push(source);
    }
    for neighbours in &mut adjacency {
        neighbours.sort_unstable();
        neighbours.dedup();
    }

    let mut centrality = vec![0.0f64; node_count];
    for source in 0..node_count {
        let mut predecessors = vec![Vec::<usize>::new(); node_count];
        let mut distance = vec![usize::MAX; node_count];
        let mut log_sigma = vec![f64::NEG_INFINITY; node_count];
        let mut queue = VecDeque::new();
        let mut stack = Vec::with_capacity(node_count);
        distance[source] = 0;
        log_sigma[source] = 0.0;
        queue.push_back(source);

        while let Some(current) = queue.pop_front() {
            stack.push(current);
            for &next in &adjacency[current] {
                if distance[next] == usize::MAX {
                    distance[next] = distance[current].saturating_add(1);
                    queue.push_back(next);
                }
                if distance[next] == distance[current].saturating_add(1) {
                    predecessors[next].push(current);
                    let existing = log_sigma[next];
                    let incoming = log_sigma[current];
                    if existing.is_finite() {
                        let maximum = existing.max(incoming);
                        let minimum = existing.min(incoming);
                        log_sigma[next] =
                            maximum + (1.0 + (minimum - maximum).exp()).ln();
                    } else {
                        log_sigma[next] = incoming;
                    }
                }
            }
        }

        let mut dependency = vec![0.0f64; node_count];
        while let Some(node) = stack.pop() {
            for &predecessor in &predecessors[node] {
                let path_fraction = (log_sigma[predecessor] - log_sigma[node]).exp();
                dependency[predecessor] += path_fraction * (1.0 + dependency[node]);
            }
            if node != source {
                centrality[node] += dependency[node];
            }
        }
    }
    for value in &mut centrality {
        *value *= 0.5;
    }
    centrality
}

fn rank_failure_entities(
    indexed: &IndexedGraph,
    demands: &[MycelialDemandPair],
    config: &MycelialFailureAttackConfig,
) -> Result<Vec<String>, MycelialFailureResilienceError> {
    let targets_segments = attack_strategy_targets_segments(config.strategy);
    let mut ranked = if targets_segments {
        indexed.edge_ids.clone()
    } else {
        indexed.node_ids.clone()
    };

    match config.strategy {
        MycelialFailureAttackStrategy::RandomSegments
        | MycelialFailureAttackStrategy::RandomJunctions => {
            deterministic_shuffle(&mut ranked, config.seed);
        }
        MycelialFailureAttackStrategy::HighestDegreeJunctions => {
            let mut degree = vec![0usize; indexed.node_ids.len()];
            for &(source, target, _) in &indexed.edge_endpoints {
                degree[source] = degree[source].saturating_add(1);
                degree[target] = degree[target].saturating_add(1);
            }
            ranked.sort_by(|left, right| {
                let left_index = indexed.node_index[left];
                let right_index = indexed.node_index[right];
                degree[right_index]
                    .cmp(&degree[left_index])
                    .then_with(|| failure_entity_order(config.seed, 1, left, right))
            });
        }
        MycelialFailureAttackStrategy::AdaptiveHighestDegreeJunctions => {
            let mut degree = vec![0usize; indexed.node_ids.len()];
            for &(source, target, _) in &indexed.edge_endpoints {
                degree[source] = degree[source].saturating_add(1);
                degree[target] = degree[target].saturating_add(1);
            }
            ranked.sort_by(|left, right| {
                let left_index = indexed.node_index[left];
                let right_index = indexed.node_index[right];
                degree[right_index]
                    .cmp(&degree[left_index])
                    .then_with(|| failure_entity_order(config.seed, 1, left, right))
            });
        }
        MycelialFailureAttackStrategy::BetweennessJunctions => {
            let centrality = unweighted_static_betweenness(indexed);
            ranked.sort_by(|left, right| {
                let left_score = centrality[indexed.node_index[left]];
                let right_score = centrality[indexed.node_index[right]];
                right_score.total_cmp(&left_score).then_with(|| {
                    failure_entity_order(config.seed, 1, left, right)
                })
            });
        }
        MycelialFailureAttackStrategy::LowestDegreeJunctions => {
            let mut degree = vec![0usize; indexed.node_ids.len()];
            for &(source, target, _) in &indexed.edge_endpoints {
                degree[source] = degree[source].saturating_add(1);
                degree[target] = degree[target].saturating_add(1);
            }
            ranked.sort_by(|left, right| {
                let left_index = indexed.node_index[left];
                let right_index = indexed.node_index[right];
                degree[left_index]
                    .cmp(&degree[right_index])
                    .then_with(|| failure_entity_order(config.seed, 1, left, right))
            });
        }
        MycelialFailureAttackStrategy::DemandPathLoadSegments
        | MycelialFailureAttackStrategy::DemandPathLoadJunctions
        | MycelialFailureAttackStrategy::AdaptiveDemandPathLoadSegments
        | MycelialFailureAttackStrategy::AdaptiveDemandPathLoadJunctions => {
            let (segment_load, junction_load) =
                demand_weighted_path_load(indexed, demands, &BTreeSet::new(), &BTreeSet::new());
            ranked.sort_by(|left, right| {
                let left_load = if targets_segments {
                    segment_load[indexed.edge_index[left]]
                } else {
                    junction_load[indexed.node_index[left]]
                };
                let right_load = if targets_segments {
                    segment_load[indexed.edge_index[right]]
                } else {
                    junction_load[indexed.node_index[right]]
                };
                right_load.cmp(&left_load).then_with(|| {
                    failure_entity_order(
                        config.seed,
                        if targets_segments { 0 } else { 1 },
                        left,
                        right,
                    )
                })
            });
        }
        MycelialFailureAttackStrategy::SpatialClusterJunctions => {
            let seed_id = config.spatial_seed_junction_id.as_deref().ok_or(
                MycelialFailureResilienceError::InvalidAttackConfig(
                    "spatial-cluster strategy requires a seed junction ID",
                ),
            )?;
            let seed_index = indexed
                .node_index
                .get(seed_id)
                .copied()
                .ok_or_else(|| {
                    MycelialFailureResilienceError::UnknownSpatialSeedJunction(
                        seed_id.to_string(),
                    )
                })?;
            let seed_position = indexed.positions[seed_index];
            ranked.sort_by(|left, right| {
                let left_distance = indexed.positions[indexed.node_index[left]]
                    .sub(seed_position)
                    .length();
                let right_distance = indexed.positions[indexed.node_index[right]]
                    .sub(seed_position)
                    .length();
                left_distance.total_cmp(&right_distance).then_with(|| {
                    failure_entity_order(config.seed, 1, left, right)
                })
            });
        }
    }
    Ok(ranked)
}

fn adaptive_highest_degree_ranking(
    indexed: &IndexedGraph,
    failed_junction_ids: &BTreeSet<String>,
    seed: u64,
) -> Vec<String> {
    let mut degree = vec![0usize; indexed.node_ids.len()];
    let active_nodes: Vec<_> = indexed
        .node_ids
        .iter()
        .map(|id| !failed_junction_ids.contains(id))
        .collect();
    for &(source, target, _) in &indexed.edge_endpoints {
        if active_nodes[source] && active_nodes[target] {
            degree[source] = degree[source].saturating_add(1);
            degree[target] = degree[target].saturating_add(1);
        }
    }
    let mut ranked: Vec<_> = indexed
        .node_ids
        .iter()
        .filter(|id| !failed_junction_ids.contains(*id))
        .cloned()
        .collect();
    ranked.sort_by(|left, right| {
        let left_index = indexed.node_index[left];
        let right_index = indexed.node_index[right];
        degree[right_index]
            .cmp(&degree[left_index])
            .then_with(|| failure_entity_order(seed, 1, left, right))
    });
    ranked
}

fn adaptive_demand_path_load_ranking(
    indexed: &IndexedGraph,
    demands: &[MycelialDemandPair],
    strategy: MycelialFailureAttackStrategy,
    failed_segment_ids: &BTreeSet<String>,
    failed_junction_ids: &BTreeSet<String>,
    seed: u64,
) -> Vec<String> {
    let targets_segments = attack_strategy_targets_segments(strategy);
    let (segment_load, junction_load) = demand_weighted_path_load(
        indexed,
        demands,
        failed_segment_ids,
        failed_junction_ids,
    );
    let mut ranked: Vec<_> = if targets_segments {
        indexed
            .edge_ids
            .iter()
            .filter(|id| !failed_segment_ids.contains(*id))
            .cloned()
            .collect()
    } else {
        indexed
            .node_ids
            .iter()
            .filter(|id| !failed_junction_ids.contains(*id))
            .cloned()
            .collect()
    };
    ranked.sort_by(|left, right| {
        let left_load = if targets_segments {
            segment_load[indexed.edge_index[left]]
        } else {
            junction_load[indexed.node_index[left]]
        };
        let right_load = if targets_segments {
            segment_load[indexed.edge_index[right]]
        } else {
            junction_load[indexed.node_index[right]]
        };
        right_load.cmp(&left_load).then_with(|| {
            failure_entity_order(
                seed,
                if targets_segments { 0 } else { 1 },
                left,
                right,
            )
        })
    });
    ranked
}

fn build_attack_rankings(
    indexed: &IndexedGraph,
    demands: &[MycelialDemandPair],
    config: &MycelialFailureAttackConfig,
) -> Result<(Vec<String>, Vec<Vec<String>>), MycelialFailureResilienceError> {
    let base_ranking = rank_failure_entities(indexed, demands, config)?;
    if config.maximum_failure_budget > base_ranking.len() {
        return Err(MycelialFailureResilienceError::InvalidAttackConfig(
            "maximum failure budget exceeds the number of attackable entities",
        ));
    }
    let mut failed_junctions = BTreeSet::new();
    let mut failed_segments = BTreeSet::new();
    let mut selected = Vec::with_capacity(config.maximum_failure_budget);
    let mut rankings_per_step = Vec::with_capacity(config.maximum_failure_budget);
    for _ in 0..config.maximum_failure_budget {
        let ranking = match config.strategy {
            MycelialFailureAttackStrategy::AdaptiveHighestDegreeJunctions => {
                adaptive_highest_degree_ranking(indexed, &failed_junctions, config.seed)
            }
            MycelialFailureAttackStrategy::AdaptiveDemandPathLoadSegments
            | MycelialFailureAttackStrategy::AdaptiveDemandPathLoadJunctions => {
                adaptive_demand_path_load_ranking(
                    indexed,
                    demands,
                    config.strategy,
                    &failed_segments,
                    &failed_junctions,
                    config.seed,
                )
            }
            _ => base_ranking
                .iter()
                .filter(|id| !selected.iter().any(|chosen| chosen == *id))
                .cloned()
                .collect(),
        };
        let Some(next_id) = ranking.first().cloned() else {
            return Err(MycelialFailureResilienceError::InvalidAttackConfig(
                "attack ranking became empty before the requested budget",
            ));
        };
        rankings_per_step.push(ranking);
        selected.push(next_id.clone());
        if attack_strategy_targets_segments(config.strategy) {
            failed_segments.insert(next_id);
        } else {
            failed_junctions.insert(next_id);
        }
    }
    Ok((selected, rankings_per_step))
}

pub fn build_mycelial_failure_attack_plan(
    graph: &MycelialJunctionHyphaGraph,
    demands: &[MycelialDemandPair],
    config: MycelialFailureAttackConfig,
) -> Result<MycelialFailureAttackPlan, MycelialFailureResilienceError> {
    let indexed = index_source_graph(graph)?;
    validate_demands(&indexed, demands)?;
    if config.maximum_failure_budget == 0
        || config.maximum_failure_budget > MAX_FAILURE_SCENARIOS
    {
        return Err(MycelialFailureResilienceError::InvalidAttackConfig(
            "maximum failure budget must be within the scenario limit",
        ));
    }
    match (
        config.strategy,
        config.spatial_seed_junction_id.as_deref(),
    ) {
        (MycelialFailureAttackStrategy::SpatialClusterJunctions, Some(id))
            if valid_id(id) => {}
        (MycelialFailureAttackStrategy::SpatialClusterJunctions, _) => {
            return Err(MycelialFailureResilienceError::InvalidAttackConfig(
                "spatial-cluster strategy requires a valid seed junction ID",
            ));
        }
        (_, None) => {}
        (_, Some(_)) => {
            return Err(MycelialFailureResilienceError::InvalidAttackConfig(
                "spatial seed junction is only valid for spatial-cluster strategy",
            ));
        }
    }

    let (ranked_entity_ids, ranked_entity_ids_per_step) =
        build_attack_rankings(&indexed, demands, &config)?;
    let strategy_id = config.strategy.stable_id();
    let seed_id = config.seed;
    let mut scenarios = Vec::with_capacity(config.maximum_failure_budget);
    for budget in 1..=config.maximum_failure_budget {
        let mut failed_ids = ranked_entity_ids[..budget].to_vec();
        failed_ids.sort_unstable();
        let scenario_id = format!(
            "attack:{strategy_id}:seed:{seed_id:016x}:budget:{budget:04}"
        );
        scenarios.push(MycelialFailureScenario {
            scenario_id,
            failed_segment_ids: if attack_strategy_targets_segments(config.strategy) {
                failed_ids
            } else {
                Vec::new()
            },
            failed_junction_ids: if attack_strategy_targets_segments(config.strategy) {
                Vec::new()
            } else {
                failed_ids
            },
        });
    }

    let plan = MycelialFailureAttackPlan {
        schema_version: MYCELIAL_FAILURE_ATTACK_PLAN_SCHEMA_VERSION,
        source_graph_digest: graph.canonical_digest_hex()?,
        demand_digest: canonical_demand_digest(demands),
        strategy: config.strategy,
        seed: config.seed,
        maximum_failure_budget: config.maximum_failure_budget,
        spatial_seed_junction_id: config.spatial_seed_junction_id,
        ranked_entity_ids,
        ranked_entity_ids_per_step,
        scenarios,
    };
    plan.validate_against(graph, demands)?;
    Ok(plan)
}

impl MycelialFailureAttackPlan {
    pub fn validate_against(
        &self,
        graph: &MycelialJunctionHyphaGraph,
        demands: &[MycelialDemandPair],
    ) -> Result<(), MycelialFailureResilienceError> {
        let config = MycelialFailureAttackConfig {
            strategy: self.strategy,
            seed: self.seed,
            maximum_failure_budget: self.maximum_failure_budget,
            spatial_seed_junction_id: self.spatial_seed_junction_id.clone(),
        };
        match (
            config.strategy,
            config.spatial_seed_junction_id.as_deref(),
        ) {
            (MycelialFailureAttackStrategy::SpatialClusterJunctions, Some(id))
                if valid_id(id) => {}
            (MycelialFailureAttackStrategy::SpatialClusterJunctions, _) => {
                return Err(MycelialFailureResilienceError::InvalidAttackPlan(
                    "spatial-cluster plan lacks a valid seed junction",
                ));
            }
            (_, None) => {}
            (_, Some(_)) => {
                return Err(MycelialFailureResilienceError::InvalidAttackPlan(
                    "non-spatial attack plan contains a spatial seed",
                ));
            }
        }
        if self.schema_version != MYCELIAL_FAILURE_ATTACK_PLAN_SCHEMA_VERSION
            || self.source_graph_digest != graph.canonical_digest_hex()?
            || self.demand_digest != canonical_demand_digest(demands)
        {
            return Err(MycelialFailureResilienceError::InvalidAttackPlan(
                "attack plan source graph, demand inventory or schema mismatch",
            ));
        }
        let indexed = index_source_graph(graph)?;
        validate_demands(&indexed, demands)?;
        let (expected_rank, expected_rankings_per_step) =
            build_attack_rankings(&indexed, demands, &config)?;
        if self.ranked_entity_ids != expected_rank
            || self.ranked_entity_ids_per_step != expected_rankings_per_step
            || self.maximum_failure_budget == 0
            || self.maximum_failure_budget > expected_rank.len()
            || self.scenarios.len() != self.maximum_failure_budget
        {
            return Err(MycelialFailureResilienceError::InvalidAttackPlan(
                "attack ranking or budget inventory does not match its policy",
            ));
        }
        let target_is_segment = attack_strategy_targets_segments(self.strategy);
        for (index, scenario) in self.scenarios.iter().enumerate() {
            let budget = index + 1;
            let mut expected_ids = expected_rank[..budget].to_vec();
            expected_ids.sort_unstable();
            let expected_id = format!(
                "attack:{}:seed:{:016x}:budget:{budget:04}",
                self.strategy.stable_id(),
                self.seed
            );
            if scenario.scenario_id != expected_id
                || scenario.failed_segment_ids
                    != if target_is_segment { expected_ids.clone() } else { Vec::new() }
                || scenario.failed_junction_ids
                    != if target_is_segment { Vec::new() } else { expected_ids }
            {
                return Err(MycelialFailureResilienceError::InvalidAttackPlan(
                    "attack scenario is not the declared nested ranked prefix",
                ));
            }
        }
        Ok(())
    }

    pub fn canonical_digest_hex(
        &self,
    ) -> Result<String, MycelialFailureResilienceError> {
        if self.schema_version != MYCELIAL_FAILURE_ATTACK_PLAN_SCHEMA_VERSION
            || self.maximum_failure_budget == 0
            || self.maximum_failure_budget > MAX_FAILURE_SCENARIOS
            || self.scenarios.len() != self.maximum_failure_budget
            || self.ranked_entity_ids.len() < self.maximum_failure_budget
            || self.ranked_entity_ids_per_step.len() != self.maximum_failure_budget
            || self.ranked_entity_ids_per_step.iter().any(|ranking| ranking.is_empty())
            || self
                .ranked_entity_ids_per_step
                .iter()
                .zip(&self.ranked_entity_ids)
                .any(|(ranking, selected)| {
                    ranking.first() != Some(selected)
                        || ranking.iter().any(|id| !valid_id(id))
                        || ranking.iter().collect::<BTreeSet<_>>().len() != ranking.len()
                })
            || self.ranked_entity_ids_per_step.iter().enumerate().any(|(step, ranking)| {
                let already_selected: BTreeSet<_> =
                    self.ranked_entity_ids[..step].iter().map(String::as_str).collect();
                ranking.iter().any(|id| already_selected.contains(id.as_str()))
            })
            || !valid_digest(&self.source_graph_digest)
            || !valid_digest(&self.demand_digest)
            || self.ranked_entity_ids.iter().any(|id| !valid_id(id))
            || self.spatial_seed_junction_id.as_deref().is_some_and(|id| !valid_id(id))
            || self.ranked_entity_ids.iter().collect::<BTreeSet<_>>().len()
                != self.ranked_entity_ids.len()
        {
            return Err(MycelialFailureResilienceError::InvalidAttackPlan(
                "attack plan fields are invalid",
            ));
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-mycelial-failure-attack-plan-v1");
        hasher.update(&self.schema_version.to_le_bytes());
        hasher.update(self.source_graph_digest.as_bytes());
        hasher.update(self.demand_digest.as_bytes());
        hash_string(&mut hasher, self.strategy.stable_id());
        hasher.update(&self.seed.to_le_bytes());
        hasher.update(&(self.maximum_failure_budget as u64).to_le_bytes());
        match &self.spatial_seed_junction_id {
            Some(id) => {
                hasher.update(&[1]);
                hash_string(&mut hasher, id);
            }
            None => hasher.update(&[0]),
        }
        hasher.update(&(self.ranked_entity_ids.len() as u64).to_le_bytes());
        for id in &self.ranked_entity_ids {
            hash_string(&mut hasher, id);
        }
        hasher.update(&(self.ranked_entity_ids_per_step.len() as u64).to_le_bytes());
        for ranking in &self.ranked_entity_ids_per_step {
            hasher.update(&(ranking.len() as u64).to_le_bytes());
            for id in ranking {
                hash_string(&mut hasher, id);
            }
        }
        hasher.update(&(self.scenarios.len() as u64).to_le_bytes());
        for scenario in &self.scenarios {
            hash_string(&mut hasher, &scenario.scenario_id);
            hasher.update(&(scenario.failed_segment_ids.len() as u64).to_le_bytes());
            for id in &scenario.failed_segment_ids {
                hash_string(&mut hasher, id);
            }
            hasher.update(&(scenario.failed_junction_ids.len() as u64).to_le_bytes());
            for id in &scenario.failed_junction_ids {
                hash_string(&mut hasher, id);
            }
        }
        Ok(hex_digest(hasher.finalize().as_bytes()))
    }
}

impl MycelialFailureAttackEvaluation {
    pub fn canonical_digest_hex(
        &self,
    ) -> Result<String, MycelialFailureResilienceError> {
        if !valid_digest(&self.attack_plan_digest) {
            return Err(MycelialFailureResilienceError::InvalidProfile(
                "attack evaluation has an invalid plan digest",
            ));
        }
        let profile_digest = self.resilience_profile.canonical_digest_hex()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-mycelial-failure-attack-evaluation-v1");
        hasher.update(self.attack_plan_digest.as_bytes());
        hasher.update(profile_digest.as_bytes());
        Ok(hex_digest(hasher.finalize().as_bytes()))
    }
}

pub fn evaluate_mycelial_failure_attack_plan(
    graph: &MycelialJunctionHyphaGraph,
    demands: &[MycelialDemandPair],
    plan: &MycelialFailureAttackPlan,
) -> Result<MycelialFailureAttackEvaluation, MycelialFailureResilienceError> {
    plan.validate_against(graph, demands)?;
    let attack_plan_digest = plan.canonical_digest_hex()?;
    let resilience_profile =
        build_mycelial_failure_resilience_profile(graph, demands, &plan.scenarios)?;
    Ok(MycelialFailureAttackEvaluation {
        attack_plan_digest,
        resilience_profile,
    })
}

fn validate_delivery_config(
    config: MycelialFailureDeliveryConfig,
    scenario_count: usize,
) -> Result<(), MycelialFailureResilienceError> {
    if config.trials_per_scenario == 0
        || config.trials_per_scenario > MAX_FAILURE_DELIVERY_TRIALS_PER_SCENARIO
        || config.removal_success_probability_micros > 1_000_000
        || scenario_count.saturating_mul(config.trials_per_scenario)
            > MAX_FAILURE_DELIVERY_EVALUATIONS
    {
        return Err(MycelialFailureResilienceError::InvalidDeliveryConfig(
            "trial count, delivery probability or total evaluation budget is out of bounds",
        ));
    }
    Ok(())
}

fn delivery_entity_draw(
    seed: u64,
    trial_index: usize,
    entity_kind: u8,
    entity_id: &str,
) -> u64 {
    // Counter-based entity-keyed draw: the same source entity receives the
    // same outcome in a given trial across strategies and nested budgets.
    // This supplies common random numbers for paired attack-policy comparison.
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-failure-delivery-entity-draw-v2");
    hasher.update(&seed.to_le_bytes());
    hasher.update(&(trial_index as u64).to_le_bytes());
    hasher.update(&[entity_kind]);
    hash_string(&mut hasher, entity_id);
    let digest = hasher.finalize();
    u64::from_le_bytes(digest.as_bytes()[..8].try_into().expect("digest prefix is eight bytes"))
        % 1_000_000
}

fn realize_attack_delivery(
    scenario: &MycelialFailureScenario,
    config: MycelialFailureDeliveryConfig,
    trial_index: usize,
) -> (Vec<String>, Vec<String>) {
    let mut failed_segments = Vec::new();
    let mut failed_junctions = Vec::new();
    for id in &scenario.failed_segment_ids {
        if delivery_entity_draw(config.seed, trial_index, 0, id)
            < u64::from(config.removal_success_probability_micros)
        {
            failed_segments.push(id.clone());
        }
    }
    for id in &scenario.failed_junction_ids {
        if delivery_entity_draw(config.seed, trial_index, 1, id)
            < u64::from(config.removal_success_probability_micros)
        {
            failed_junctions.push(id.clone());
        }
    }
    (failed_segments, failed_junctions)
}

fn summarize_delivery_trials(
    scenario_id: &str,
    attempted_entity_count: usize,
    trials: &[MycelialFailureDeliveryTrial],
) -> Result<MycelialFailureDeliveryScenarioSummary, MycelialFailureResilienceError> {
    if trials.is_empty()
        || trials.iter().any(|trial| trial.scenario_id != scenario_id)
    {
        return Err(MycelialFailureResilienceError::InvalidDeliveryProfile(
            "scenario summary has no matching trial inventory",
        ));
    }
    let mean_realized_failure_count = trials
        .iter()
        .map(|trial| {
            (trial.realized_failed_segment_ids.len()
                + trial.realized_failed_junction_ids.len()) as f64
        })
        .sum::<f64>()
        / trials.len() as f64;
    let probability_at_least_one_failure = trials
        .iter()
        .filter(|trial| {
            !trial.realized_failed_segment_ids.is_empty()
                || !trial.realized_failed_junction_ids.is_empty()
        })
        .count() as f64
        / trials.len() as f64;

    let mut reachability: Vec<_> = trials
        .iter()
        .filter_map(|trial| trial.weighted_reachability)
        .collect();
    reachability.sort_by(f64::total_cmp);
    let mean_weighted_reachability = mean_optional(&reachability);
    let weighted_reachability_p05 = quantile_sorted(&reachability, 0.05);
    let weighted_reachability_p50 = quantile_sorted(&reachability, 0.50);
    let weighted_reachability_p95 = quantile_sorted(&reachability, 0.95);
    let weighted_reachability_minimum = reachability.first().copied();
    let weighted_reachability_maximum = reachability.last().copied();

    let mut directness: Vec<_> = trials
        .iter()
        .filter_map(|trial| trial.weighted_mean_directness)
        .collect();
    directness.sort_by(f64::total_cmp);
    let mut path_lengths: Vec<_> = trials
        .iter()
        .filter_map(|trial| trial.weighted_mean_network_path_length)
        .collect();
    path_lengths.sort_by(f64::total_cmp);

    Ok(MycelialFailureDeliveryScenarioSummary {
        scenario_id: scenario_id.to_string(),
        attempted_entity_count,
        trial_count: trials.len(),
        mean_realized_failure_count,
        probability_at_least_one_failure,
        mean_weighted_reachability,
        weighted_reachability_p05,
        weighted_reachability_p50,
        weighted_reachability_p95,
        weighted_reachability_minimum,
        weighted_reachability_maximum,
        directness_defined_trial_count: directness.len(),
        mean_weighted_directness_among_reachable: mean_optional(&directness),
        mean_weighted_path_length_among_reachable: mean_optional(&path_lengths),
    })
}

fn mean_optional(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then_some(values.iter().sum::<f64>() / values.len() as f64)
}

fn quantile_sorted(values: &[f64], quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let index = ((values.len() - 1) as f64 * quantile).ceil() as usize;
    values.get(index).copied()
}

pub fn build_mycelial_failure_delivery_profile(
    graph: &MycelialJunctionHyphaGraph,
    demands: &[MycelialDemandPair],
    plan: &MycelialFailureAttackPlan,
    config: MycelialFailureDeliveryConfig,
) -> Result<MycelialFailureDeliveryProfile, MycelialFailureResilienceError> {
    plan.validate_against(graph, demands)?;
    validate_delivery_config(config, plan.scenarios.len())?;
    let indexed = index_source_graph(graph)?;
    validate_demands(&indexed, demands)?;
    let source_graph_digest = graph.canonical_digest_hex()?;
    let demand_digest = canonical_demand_digest(demands);
    let attack_plan_digest = plan.canonical_digest_hex()?;
    let mut trials = Vec::with_capacity(plan.scenarios.len() * config.trials_per_scenario);
    let mut summaries = Vec::with_capacity(plan.scenarios.len());

    for scenario in &plan.scenarios {
        let start = trials.len();
        for trial_index in 0..config.trials_per_scenario {
            let (failed_segments, failed_junctions) =
                realize_attack_delivery(scenario, config, trial_index);
            let metric = evaluate_one(
                &indexed,
                demands,
                &scenario.scenario_id,
                &failed_segments,
                &failed_junctions,
                graph.length_unit,
            )?;
            trials.push(MycelialFailureDeliveryTrial {
                scenario_id: scenario.scenario_id.clone(),
                trial_index,
                realized_failed_segment_ids: failed_segments,
                realized_failed_junction_ids: failed_junctions,
                weighted_reachability: metric.weighted_reachability,
                weighted_mean_directness: metric.weighted_mean_directness,
                weighted_mean_network_path_length: metric.weighted_mean_network_path_length,
                connected_components: metric.connected_components,
                cycle_rank: metric.cycle_rank,
            });
        }
        let scenario_trials = &trials[start..];
        summaries.push(summarize_delivery_trials(
            &scenario.scenario_id,
            scenario.failed_segment_ids.len() + scenario.failed_junction_ids.len(),
            scenario_trials,
        )?);
    }

    let profile = MycelialFailureDeliveryProfile {
        schema_version: MYCELIAL_FAILURE_DELIVERY_PROFILE_SCHEMA_VERSION,
        source_graph_digest,
        demand_digest,
        attack_plan_digest,
        config,
        trials,
        scenarios: summaries,
    };
    profile.validate_against(graph, demands, plan)?;
    Ok(profile)
}

impl MycelialFailureDeliveryProfile {
    pub fn validate(&self) -> Result<(), MycelialFailureResilienceError> {
        if self.schema_version != MYCELIAL_FAILURE_DELIVERY_PROFILE_SCHEMA_VERSION
            || !valid_digest(&self.source_graph_digest)
            || !valid_digest(&self.demand_digest)
            || !valid_digest(&self.attack_plan_digest)
            || self.scenarios.is_empty()
            || self.scenarios.len() > MAX_FAILURE_SCENARIOS
            || self.scenarios.windows(2).any(|pair| {
                pair[0].scenario_id >= pair[1].scenario_id
            })
        {
            return Err(MycelialFailureResilienceError::InvalidDeliveryProfile(
                "invalid schema, digest or scenario summary order",
            ));
        }
        validate_delivery_config(self.config, self.scenarios.len())?;
        if self.trials.len()
            != self.scenarios.len().saturating_mul(self.config.trials_per_scenario)
            || self.trials.windows(2).any(|pair| {
                pair[0].scenario_id > pair[1].scenario_id
                    || (pair[0].scenario_id == pair[1].scenario_id
                        && pair[0].trial_index >= pair[1].trial_index)
            })
        {
            return Err(MycelialFailureResilienceError::InvalidDeliveryProfile(
                "trial inventory is incomplete, duplicated or unsorted",
            ));
        }
        let mut cursor = 0usize;
        for summary in &self.scenarios {
            if !valid_id(&summary.scenario_id)
                || summary.attempted_entity_count == 0
                || summary.trial_count != self.config.trials_per_scenario
            {
                return Err(MycelialFailureResilienceError::InvalidDeliveryProfile(
                    "scenario summary metadata is invalid",
                ));
            }
            let start = cursor;
            while cursor < self.trials.len()
                && self.trials[cursor].scenario_id == summary.scenario_id
            {
                let trial = &self.trials[cursor];
                if trial.trial_index != cursor - start
                    || trial.realized_failed_segment_ids.len()
                        + trial.realized_failed_junction_ids.len()
                        > summary.attempted_entity_count
                    || trial.realized_failed_segment_ids.iter().any(|id| !valid_id(id))
                    || trial.realized_failed_junction_ids.iter().any(|id| !valid_id(id))
                    || trial.realized_failed_segment_ids.windows(2).any(|pair| pair[0] >= pair[1])
                    || trial.realized_failed_junction_ids.windows(2).any(|pair| pair[0] >= pair[1])
                    || trial.weighted_reachability.is_some_and(|value| {
                        !value.is_finite() || !(0.0..=1.0).contains(&value)
                    })
                    || trial.weighted_mean_directness.is_some_and(|value| {
                        !value.is_finite() || !(0.0..=1.0 + 1.0e-5).contains(&value)
                    })
                    || trial.weighted_mean_network_path_length.is_some_and(|value| {
                        !value.is_finite() || value <= 0.0
                    })
                    || trial.weighted_mean_directness.is_some()
                        != trial.weighted_mean_network_path_length.is_some()
                    || trial.connected_components > MAX_FAILURE_JUNCTIONS
                    || trial.cycle_rank > MAX_FAILURE_SEGMENTS
                {
                    return Err(MycelialFailureResilienceError::InvalidDeliveryProfile(
                        "trial identity, failed-entity inventory or metric value is invalid",
                    ));
                }
                cursor += 1;
            }
            if cursor - start != self.config.trials_per_scenario {
                return Err(MycelialFailureResilienceError::InvalidDeliveryProfile(
                    "scenario summary does not have the configured number of trials",
                ));
            }
            let expected = summarize_delivery_trials(
                &summary.scenario_id,
                summary.attempted_entity_count,
                &self.trials[start..cursor],
            )?;
            if &expected != summary {
                return Err(MycelialFailureResilienceError::InvalidDeliveryProfile(
                    "scenario distribution summary does not match trial-level receipts",
                ));
            }
        }
        if cursor != self.trials.len() {
            return Err(MycelialFailureResilienceError::InvalidDeliveryProfile(
                "trial inventory includes an undeclared scenario",
            ));
        }
        Ok(())
    }

    pub fn validate_against(
        &self,
        graph: &MycelialJunctionHyphaGraph,
        demands: &[MycelialDemandPair],
        plan: &MycelialFailureAttackPlan,
    ) -> Result<(), MycelialFailureResilienceError> {
        self.validate()?;
        plan.validate_against(graph, demands)?;
        validate_delivery_config(self.config, plan.scenarios.len())?;
        if self.schema_version != MYCELIAL_FAILURE_DELIVERY_PROFILE_SCHEMA_VERSION
            || self.source_graph_digest != graph.canonical_digest_hex()?
            || self.demand_digest != canonical_demand_digest(demands)
            || self.attack_plan_digest != plan.canonical_digest_hex()?
            || self.scenarios.len() != plan.scenarios.len()
            || self.trials.len()
                != plan
                    .scenarios
                    .len()
                    .saturating_mul(self.config.trials_per_scenario)
        {
            return Err(MycelialFailureResilienceError::InvalidDeliveryProfile(
                "source graph, demand, plan, scenario or trial inventory mismatch",
            ));
        }
        let indexed = index_source_graph(graph)?;
        validate_demands(&indexed, demands)?;
        let mut cursor = 0usize;
        for (scenario_index, scenario) in plan.scenarios.iter().enumerate() {
            let start = cursor;
            for trial_index in 0..self.config.trials_per_scenario {
                let Some(trial) = self.trials.get(cursor) else {
                    return Err(MycelialFailureResilienceError::InvalidDeliveryProfile(
                        "trial inventory ended before all configured scenario trials",
                    ));
                };
                let (expected_segments, expected_junctions) =
                    realize_attack_delivery(scenario, self.config, trial_index);
                if trial.scenario_id != scenario.scenario_id
                    || trial.trial_index != trial_index
                    || trial.realized_failed_segment_ids != expected_segments
                    || trial.realized_failed_junction_ids != expected_junctions
                    || trial.realized_failed_segment_ids.windows(2).any(|pair| pair[0] >= pair[1])
                    || trial.realized_failed_junction_ids.windows(2).any(|pair| pair[0] >= pair[1])
                    || trial.weighted_reachability.is_some_and(|value| {
                        !value.is_finite() || !(0.0..=1.0).contains(&value)
                    })
                    || trial.weighted_mean_directness.is_some_and(|value| {
                        !value.is_finite() || !(0.0..=1.0 + 1.0e-5).contains(&value)
                    })
                    || trial.weighted_mean_network_path_length.is_some_and(|value| {
                        !value.is_finite() || value <= 0.0
                    })
                    || trial.weighted_mean_directness.is_some()
                        != trial.weighted_mean_network_path_length.is_some()
                {
                    return Err(MycelialFailureResilienceError::InvalidDeliveryProfile(
                        "trial delivery realization or metric fields are inconsistent",
                    ));
                }
                let recomputed = evaluate_one(
                    &indexed,
                    demands,
                    &scenario.scenario_id,
                    &expected_segments,
                    &expected_junctions,
                    graph.length_unit,
                )?;
                if trial.weighted_reachability != recomputed.weighted_reachability
                    || trial.weighted_mean_directness != recomputed.weighted_mean_directness
                    || trial.weighted_mean_network_path_length
                        != recomputed.weighted_mean_network_path_length
                    || trial.connected_components != recomputed.connected_components
                    || trial.cycle_rank != recomputed.cycle_rank
                {
                    return Err(MycelialFailureResilienceError::InvalidDeliveryProfile(
                        "trial metrics do not match recomputation from realized failures",
                    ));
                }
                cursor += 1;
            }
            let attempted_entity_count =
                scenario.failed_segment_ids.len() + scenario.failed_junction_ids.len();
            let expected_summary = summarize_delivery_trials(
                &scenario.scenario_id,
                attempted_entity_count,
                &self.trials[start..cursor],
            )?;
            if self.scenarios.get(scenario_index) != Some(&expected_summary) {
                return Err(MycelialFailureResilienceError::InvalidDeliveryProfile(
                    "scenario summary does not match its complete trial inventory",
                ));
            }
        }
        // Entity-keyed common random numbers make successful removals nested
        // across the plan's budget prefixes for each trial.
        for trial_index in 0..self.config.trials_per_scenario {
            let mut previous_segments = BTreeSet::new();
            let mut previous_junctions = BTreeSet::new();
            for scenario_index in 0..plan.scenarios.len() {
                let offset = scenario_index
                    .saturating_mul(self.config.trials_per_scenario)
                    .saturating_add(trial_index);
                let trial = self.trials.get(offset).ok_or(
                    MycelialFailureResilienceError::InvalidDeliveryProfile(
                        "nested common-random-number trial is missing",
                    ),
                )?;
                let current_segments: BTreeSet<_> =
                    trial.realized_failed_segment_ids.iter().cloned().collect();
                let current_junctions: BTreeSet<_> =
                    trial.realized_failed_junction_ids.iter().cloned().collect();
                if !previous_segments.is_subset(&current_segments)
                    || !previous_junctions.is_subset(&current_junctions)
                {
                    return Err(MycelialFailureResilienceError::InvalidDeliveryProfile(
                        "realized removals are not nested across attack budgets",
                    ));
                }
                previous_segments = current_segments;
                previous_junctions = current_junctions;
            }
        }
        Ok(())
    }

    pub fn canonical_digest_hex(
        &self,
    ) -> Result<String, MycelialFailureResilienceError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-mycelial-failure-delivery-profile-v1");
        hasher.update(&self.schema_version.to_le_bytes());
        hasher.update(self.source_graph_digest.as_bytes());
        hasher.update(self.demand_digest.as_bytes());
        hasher.update(self.attack_plan_digest.as_bytes());
        hasher.update(&(self.config.trials_per_scenario as u64).to_le_bytes());
        hasher.update(&self.config.removal_success_probability_micros.to_le_bytes());
        hasher.update(&self.config.seed.to_le_bytes());
        hasher.update(&(self.trials.len() as u64).to_le_bytes());
        for trial in &self.trials {
            hash_string(&mut hasher, &trial.scenario_id);
            hasher.update(&(trial.trial_index as u64).to_le_bytes());
            hasher.update(&(trial.realized_failed_segment_ids.len() as u64).to_le_bytes());
            for id in &trial.realized_failed_segment_ids {
                hash_string(&mut hasher, id);
            }
            hasher.update(&(trial.realized_failed_junction_ids.len() as u64).to_le_bytes());
            for id in &trial.realized_failed_junction_ids {
                hash_string(&mut hasher, id);
            }
            hash_option(&mut hasher, trial.weighted_reachability);
            hash_option(&mut hasher, trial.weighted_mean_directness);
            hash_option(&mut hasher, trial.weighted_mean_network_path_length);
            hasher.update(&(trial.connected_components as u64).to_le_bytes());
            hasher.update(&(trial.cycle_rank as u64).to_le_bytes());
        }
        for scenario in &self.scenarios {
            hash_string(&mut hasher, &scenario.scenario_id);
            hasher.update(&(scenario.attempted_entity_count as u64).to_le_bytes());
            hasher.update(&(scenario.trial_count as u64).to_le_bytes());
            for value in [
                scenario.mean_realized_failure_count,
                scenario.probability_at_least_one_failure,
            ] {
                hasher.update(&value.to_bits().to_le_bytes());
            }
            for value in [
                scenario.mean_weighted_reachability,
                scenario.weighted_reachability_p05,
                scenario.weighted_reachability_p50,
                scenario.weighted_reachability_p95,
                scenario.weighted_reachability_minimum,
                scenario.weighted_reachability_maximum,
                scenario.mean_weighted_directness_among_reachable,
                scenario.mean_weighted_path_length_among_reachable,
            ] {
                hash_option(&mut hasher, value);
            }
            hasher.update(&(scenario.directness_defined_trial_count as u64).to_le_bytes());
        }
        Ok(hex_digest(hasher.finalize().as_bytes()))
    }
}

impl MycelialFailureResilienceProfile {
    pub fn validate(&self) -> Result<(), MycelialFailureResilienceError> {
        if self.schema_version != MYCELIAL_FAILURE_PROFILE_SCHEMA_VERSION
            || !valid_digest(&self.source_graph_digest)
            || !valid_digest(&self.demand_digest)
            || self.demands.len() > MAX_FAILURE_DEMAND_PAIRS
            || self
                .demands
                .iter()
                .map(|demand| u128::from(demand.weight))
                .sum::<u128>()
                > MAX_FAILURE_TOTAL_DEMAND_WEIGHT
            || self.demands.windows(2).any(|pair| pair[0].id >= pair[1].id)
            || self.demands.iter().any(|demand| {
                !valid_id(&demand.id)
                    || !valid_id(&demand.source_junction_id)
                    || !valid_id(&demand.target_junction_id)
                    || demand.source_junction_id == demand.target_junction_id
                    || demand.weight == 0
            })
            || self.demand_digest != canonical_demand_digest(&self.demands)
            || self.baseline.scenario_id != "baseline"
            || !self.baseline.failed_segment_ids.is_empty()
            || !self.baseline.failed_junction_ids.is_empty()
            || self.baseline.failed_segment_count != 0
            || self.baseline.failed_junction_count != 0
            || self.scenarios.len() > MAX_FAILURE_SCENARIOS
            || self
                .scenarios
                .windows(2)
                .any(|pair| pair[0].scenario_id >= pair[1].scenario_id)
            || self.scenarios.iter().any(|metric| {
                metric.failed_segment_ids.is_empty() && metric.failed_junction_ids.is_empty()
            })
        {
            return Err(MycelialFailureResilienceError::InvalidProfile(
                "invalid version, digest, demand inventory, baseline or scenario order",
            ));
        }

        let mut source_ids = BTreeSet::new();
        for demand in &self.demands {
            source_ids.insert(demand.source_junction_id.as_str());
        }
        if source_ids.len() > MAX_FAILURE_DEMAND_SOURCES {
            return Err(MycelialFailureResilienceError::InvalidProfile(
                "demand source count exceeds profile limit",
            ));
        }

        for metric in std::iter::once(&self.baseline).chain(self.scenarios.iter()) {
            if !valid_id(&metric.scenario_id)
                || metric.demand_pair_count != self.demands.len()
                || metric.demand_pair_outcomes.len() != self.demands.len()
                || metric
                    .demand_pair_outcomes
                    .iter()
                    .zip(&self.demands)
                    .any(|(outcome, demand)| outcome.demand_id != demand.id)
                || metric.failed_segment_ids.iter().any(|id| !valid_id(id))
                || metric.failed_junction_ids.iter().any(|id| !valid_id(id))
                || metric
                    .failed_segment_ids
                    .iter()
                    .any(|id| self.baseline.failed_segment_ids.contains(id))
                || metric
                    .failed_junction_ids
                    .iter()
                    .any(|id| self.baseline.failed_junction_ids.contains(id))
                || metric.remaining_junction_count > self.baseline.remaining_junction_count
                || metric.remaining_segment_count > self.baseline.remaining_segment_count
                || metric
                    .failed_segment_ids
                    .windows(2)
                    .any(|pair| pair[0] >= pair[1])
                || metric
                    .failed_junction_ids
                    .windows(2)
                    .any(|pair| pair[0] >= pair[1])
                || metric.failed_segment_count != metric.failed_segment_ids.len()
                || metric.failed_junction_count != metric.failed_junction_ids.len()
                || metric.reachable_demand_pair_count + metric.unreachable_demand_pair_count
                    != metric.demand_pair_count
                || metric.reachable_demand_weight + metric.unreachable_demand_weight
                    != metric.total_demand_weight
                || metric.total_demand_weight
                    != self
                        .demands
                        .iter()
                        .map(|demand| u128::from(demand.weight))
                        .sum::<u128>()
                || metric.remaining_junction_count > MAX_FAILURE_JUNCTIONS
                || metric.remaining_segment_count > MAX_FAILURE_SEGMENTS
                || metric.cycle_rank
                    != metric
                        .remaining_segment_count
                        .saturating_add(metric.connected_components)
                        .saturating_sub(metric.remaining_junction_count)
                || metric.length_unit != self.baseline.length_unit
            {
                return Err(MycelialFailureResilienceError::InvalidProfile(
                    "scenario identity, bounds or demand totals are inconsistent",
                ));
            }

            let mut reachable_count = 0usize;
            let mut reachable_weight = 0u128;
            let mut weighted_directness_sum = 0.0f64;
            let mut weighted_path_length_sum = 0.0f64;
            for (outcome, demand) in metric.demand_pair_outcomes.iter().zip(&self.demands) {
                match (
                    outcome.direct_distance_model_units,
                    outcome.network_path_length_model_units,
                ) {
                    (Some(direct_distance), Some(path_length)) => {
                        if !direct_distance.is_finite()
                            || direct_distance < 0.0
                            || !path_length.is_finite()
                            || path_length <= 0.0
                            || direct_distance > path_length + 1.0e-5
                        {
                            return Err(MycelialFailureResilienceError::InvalidProfile(
                                "reachable demand outcome has invalid geometric path values",
                            ));
                        }
                        reachable_count += 1;
                        reachable_weight += u128::from(demand.weight);
                        weighted_directness_sum +=
                            (direct_distance / path_length) * f64::from(demand.weight);
                        weighted_path_length_sum += path_length * f64::from(demand.weight);
                    }
                    (None, None) => {}
                    _ => {
                        return Err(MycelialFailureResilienceError::InvalidProfile(
                            "demand outcome must report both path values or neither",
                        ));
                    }
                }
            }
            let unreachable_count = metric.demand_pair_count - reachable_count;
            let unreachable_weight = metric.total_demand_weight - reachable_weight;
            let expected_reachability = (metric.total_demand_weight > 0)
                .then_some(reachable_weight as f64 / metric.total_demand_weight as f64);
            let expected_directness =
                (reachable_weight > 0).then_some(weighted_directness_sum / reachable_weight as f64);
            let expected_path_length = (reachable_weight > 0)
                .then_some(weighted_path_length_sum / reachable_weight as f64);
            if metric.reachable_demand_pair_count != reachable_count
                || metric.unreachable_demand_pair_count != unreachable_count
                || metric.reachable_demand_weight != reachable_weight
                || metric.unreachable_demand_weight != unreachable_weight
                || metric.weighted_reachability != expected_reachability
                || metric.weighted_mean_directness != expected_directness
                || metric.weighted_mean_network_path_length != expected_path_length
            {
                return Err(MycelialFailureResilienceError::InvalidProfile(
                    "scenario aggregates disagree with per-demand outcome receipts",
                ));
            }
        }

        let expected_worst_reachability = self
            .scenarios
            .iter()
            .filter_map(|metric| metric.weighted_reachability)
            .min_by(f64::total_cmp);
        let expected_worst_directness = self
            .scenarios
            .iter()
            .filter_map(|metric| metric.weighted_mean_directness)
            .min_by(f64::total_cmp);
        if self.worst_case_weighted_reachability != expected_worst_reachability
            || self.worst_case_weighted_directness != expected_worst_directness
        {
            return Err(MycelialFailureResilienceError::InvalidProfile(
                "worst-case summary does not match scenario metrics",
            ));
        }
        Ok(())
    }

    pub fn canonical_digest_hex(&self) -> Result<String, MycelialFailureResilienceError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-mycelial-failure-resilience-v1");
        hasher.update(&self.schema_version.to_le_bytes());
        hasher.update(self.source_graph_digest.as_bytes());
        hasher.update(self.demand_digest.as_bytes());
        hash_metrics(&mut hasher, &self.baseline);
        for metric in &self.scenarios {
            hash_metrics(&mut hasher, metric);
        }
        hash_option(&mut hasher, self.worst_case_weighted_reachability);
        hash_option(&mut hasher, self.worst_case_weighted_directness);
        Ok(hex_digest(hasher.finalize().as_bytes()))
    }
}

fn canonical_demand_digest(demands: &[MycelialDemandPair]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-mycelial-demand-pairs-v1");
    hasher.update(&(demands.len() as u64).to_le_bytes());
    for demand in demands {
        hash_string(&mut hasher, &demand.id);
        hash_string(&mut hasher, &demand.source_junction_id);
        hash_string(&mut hasher, &demand.target_junction_id);
        hasher.update(&demand.weight.to_le_bytes());
    }
    hex_digest(hasher.finalize().as_bytes())
}

fn hash_metrics(hasher: &mut blake3::Hasher, metric: &MycelialFailureMetrics) {
    hash_string(hasher, &metric.scenario_id);
    hasher.update(&(metric.failed_segment_ids.len() as u64).to_le_bytes());
    for id in &metric.failed_segment_ids {
        hash_string(hasher, id);
    }
    hasher.update(&(metric.failed_junction_ids.len() as u64).to_le_bytes());
    for id in &metric.failed_junction_ids {
        hash_string(hasher, id);
    }
    hasher.update(&(metric.demand_pair_outcomes.len() as u64).to_le_bytes());
    for outcome in &metric.demand_pair_outcomes {
        hash_string(hasher, &outcome.demand_id);
        hash_option(hasher, outcome.direct_distance_model_units);
        hash_option(hasher, outcome.network_path_length_model_units);
    }
    for value in [
        metric.failed_segment_count,
        metric.failed_junction_count,
        metric.remaining_junction_count,
        metric.remaining_segment_count,
        metric.connected_components,
        metric.cycle_rank,
        metric.demand_pair_count,
        metric.reachable_demand_pair_count,
        metric.unreachable_demand_pair_count,
    ] {
        hasher.update(&(value as u64).to_le_bytes());
    }
    for value in [
        metric.total_demand_weight,
        metric.reachable_demand_weight,
        metric.unreachable_demand_weight,
    ] {
        hasher.update(&value.to_le_bytes());
    }
    hash_option(hasher, metric.weighted_reachability);
    hash_option(hasher, metric.weighted_mean_directness);
    hash_option(hasher, metric.weighted_mean_network_path_length);
    hasher.update(&[match metric.length_unit {
        JunctionLengthUnit::OrganismLocalModelUnit => 0,
    }]);
}

fn hash_option(hasher: &mut blake3::Hasher, value: Option<f64>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            hasher.update(&value.to_bits().to_le_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn valid_id(value: &str) -> bool {
    !value.is_empty() && value.len() <= MAX_FAILURE_ID_BYTES && !value.chars().any(char::is_control)
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn hash_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hex_digest(bytes: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes {
        use fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mycelial_junction_graph::{
        JunctionKind, MYCELIAL_JUNCTION_GRAPH_SCHEMA_VERSION, MycelialHyphaSegment,
        MycelialJunction, MycelialJunctionHyphaGraph, PARAMETER_SCALE,
    };

    fn triangle_graph() -> MycelialJunctionHyphaGraph {
        let mut a = MycelialJunction {
            id: "A".to_string(),
            kind: JunctionKind::Branch,
            position: SpatialPoint3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            incident_filament_ids: vec![1, 3],
            terminal_filament_ids: Vec::new(),
            anastomosis_ids: Vec::new(),
        };
        let mut b = MycelialJunction {
            id: "B".to_string(),
            kind: JunctionKind::Tip,
            position: SpatialPoint3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            incident_filament_ids: vec![1, 2],
            terminal_filament_ids: vec![1],
            anastomosis_ids: Vec::new(),
        };
        let mut c = MycelialJunction {
            id: "C".to_string(),
            kind: JunctionKind::Anastomosis,
            position: SpatialPoint3 {
                x: 0.5,
                y: 1.0,
                z: 0.0,
            },
            incident_filament_ids: vec![2, 3],
            terminal_filament_ids: vec![2],
            anastomosis_ids: vec![7],
        };
        // Keep the fixture identity lists canonical for the graph validator.
        a.incident_filament_ids.sort_unstable();
        b.incident_filament_ids.sort_unstable();
        c.incident_filament_ids.sort_unstable();
        let segment =
            |id: &str, filament_id, start: &str, end: &str, length: f64| MycelialHyphaSegment {
                id: id.to_string(),
                filament_id,
                t_start_micros: 0,
                t_end_micros: PARAMETER_SCALE,
                start_junction_id: start.to_string(),
                end_junction_id: end.to_string(),
                arc_length_model_units: length,
                start_junction_offset_model_units: 0.0,
                end_junction_offset_model_units: 0.0,
                radius_start_model_units: 1.0,
                radius_end_model_units: 1.0,
            };
        MycelialJunctionHyphaGraph {
            schema_version: MYCELIAL_JUNCTION_GRAPH_SCHEMA_VERSION,
            source_model_version: 1,
            length_unit: JunctionLengthUnit::OrganismLocalModelUnit,
            source_model_digest: "a".repeat(64),
            source_filament_ids: vec![1, 2, 3],
            maximum_junction_gap_model_units: 0.0,
            junctions: vec![a, b, c],
            segments: vec![
                segment("edge-ab", 1, "A", "B", 1.0),
                segment("edge-bc", 2, "B", "C", 1.2),
                segment("edge-ca", 3, "C", "A", 1.2),
            ],
        }
    }

    fn adaptive_attack_graph() -> MycelialJunctionHyphaGraph {
        let junction = |id: &str,
                        kind: JunctionKind,
                        x: f32,
                        y: f32,
                        incident_filament_ids: Vec<u32>,
                        terminal_filament_ids: Vec<u32>| MycelialJunction {
            id: id.to_string(),
            kind,
            position: SpatialPoint3 { x, y, z: 0.0 },
            incident_filament_ids,
            terminal_filament_ids,
            anastomosis_ids: Vec::new(),
        };
        let segment =
            |id: &str, filament_id, start: &str, end: &str, length: f64| MycelialHyphaSegment {
                id: id.to_string(),
                filament_id,
                t_start_micros: 0,
                t_end_micros: PARAMETER_SCALE,
                start_junction_id: start.to_string(),
                end_junction_id: end.to_string(),
                arc_length_model_units: length,
                start_junction_offset_model_units: 0.0,
                end_junction_offset_model_units: 0.0,
                radius_start_model_units: 1.0,
                radius_end_model_units: 1.0,
            };
        MycelialJunctionHyphaGraph {
            schema_version: MYCELIAL_JUNCTION_GRAPH_SCHEMA_VERSION,
            source_model_version: 1,
            length_unit: JunctionLengthUnit::OrganismLocalModelUnit,
            source_model_digest: "c".repeat(64),
            source_filament_ids: vec![1, 2, 3, 4, 5, 6],
            maximum_junction_gap_model_units: 0.0,
            junctions: vec![
                junction("A", JunctionKind::Branch, 0.0, 0.0, vec![1, 2, 3], vec![]),
                junction("B", JunctionKind::Branch, 1.0, 0.0, vec![1, 4], vec![]),
                junction("C", JunctionKind::Branch, 0.0, 1.0, vec![2, 5], vec![]),
                junction("D", JunctionKind::Branch, 0.0, -1.0, vec![3, 6], vec![]),
                junction("E", JunctionKind::Branch, 0.0, 2.0, vec![5, 6], vec![]),
                junction("F", JunctionKind::Tip, 2.0, 0.0, vec![4], vec![4]),
            ],
            segments: vec![
                segment("edge-ab", 1, "A", "B", 1.0),
                segment("edge-ac", 2, "A", "C", 1.0),
                segment("edge-ad", 3, "A", "D", 1.0),
                segment("edge-bf", 4, "B", "F", 1.0),
                segment("edge-ce", 5, "C", "E", 1.0),
                segment("edge-de", 6, "D", "E", 3.0),
            ],
        }
    }

    fn adaptive_path_load_graph() -> MycelialJunctionHyphaGraph {
        let junction = |id: &str, incident_filament_ids: Vec<u32>| MycelialJunction {
            id: id.to_string(),
            kind: JunctionKind::Branch,
            position: match id {
                "A" => SpatialPoint3 { x: 0.0, y: 0.0, z: 0.0 },
                "B" => SpatialPoint3 { x: 1.0, y: 0.0, z: 0.0 },
                "C" => SpatialPoint3 { x: 1.0, y: 1.0, z: 0.0 },
                "D" => SpatialPoint3 { x: 0.0, y: 1.0, z: 0.0 },
                _ => SpatialPoint3::ZERO,
            },
            incident_filament_ids,
            terminal_filament_ids: Vec::new(),
            anastomosis_ids: Vec::new(),
        };
        let segment =
            |id: &str, filament_id, start: &str, end: &str, length: f64| MycelialHyphaSegment {
                id: id.to_string(),
                filament_id,
                t_start_micros: 0,
                t_end_micros: PARAMETER_SCALE,
                start_junction_id: start.to_string(),
                end_junction_id: end.to_string(),
                arc_length_model_units: length,
                start_junction_offset_model_units: 0.0,
                end_junction_offset_model_units: 0.0,
                radius_start_model_units: 1.0,
                radius_end_model_units: 1.0,
            };
        let mut graph = MycelialJunctionHyphaGraph {
            schema_version: MYCELIAL_JUNCTION_GRAPH_SCHEMA_VERSION,
            source_model_version: 1,
            length_unit: JunctionLengthUnit::OrganismLocalModelUnit,
            source_model_digest: "e".repeat(64),
            source_filament_ids: vec![1, 2, 3, 4, 5],
            maximum_junction_gap_model_units: 0.0,
            junctions: vec![
                junction("A", vec![1, 3, 4]),
                junction("B", vec![1, 2]),
                junction("C", vec![2, 3, 5]),
                junction("D", vec![4, 5]),
            ],
            segments: vec![
                segment("edge-ab", 1, "A", "B", 1.0),
                segment("edge-ac", 3, "A", "C", 10.0),
                segment("edge-ad", 4, "A", "D", 1.0),
                segment("edge-bc", 2, "B", "C", 1.0),
                segment("edge-cd", 5, "C", "D", 1.0),
            ],
        };
        graph.segments.sort_by(|left, right| left.id.cmp(&right.id));
        graph
    }

    fn adaptive_path_load_demands() -> Vec<MycelialDemandPair> {
        vec![
            MycelialDemandPair {
                id: "demand-ab".to_string(),
                source_junction_id: "A".to_string(),
                target_junction_id: "B".to_string(),
                weight: 1,
            },
            MycelialDemandPair {
                id: "demand-ac".to_string(),
                source_junction_id: "A".to_string(),
                target_junction_id: "C".to_string(),
                weight: 10,
            },
        ]
    }

    fn demands() -> Vec<MycelialDemandPair> {
        vec![
            MycelialDemandPair {
                id: "demand-ab".to_string(),
                source_junction_id: "A".to_string(),
                target_junction_id: "B".to_string(),
                weight: 1,
            },
            MycelialDemandPair {
                id: "demand-ac".to_string(),
                source_junction_id: "A".to_string(),
                target_junction_id: "C".to_string(),
                weight: 2,
            },
            MycelialDemandPair {
                id: "demand-bc".to_string(),
                source_junction_id: "B".to_string(),
                target_junction_id: "C".to_string(),
                weight: 1,
            },
        ]
    }

    #[test]
    fn triangle_edge_failure_preserves_demand_reachability() {
        let graph = triangle_graph();
        let scenario = MycelialFailureScenario {
            scenario_id: "remove-edge-ab".to_string(),
            failed_segment_ids: vec!["edge-ab".to_string()],
            failed_junction_ids: Vec::new(),
        };
        let profile =
            build_mycelial_failure_resilience_profile(&graph, &demands(), &[scenario]).unwrap();
        assert_eq!(profile.baseline.weighted_reachability, Some(1.0));
        assert_eq!(profile.scenarios[0].weighted_reachability, Some(1.0));
        assert_eq!(profile.scenarios[0].connected_components, 1);
        assert_eq!(profile.scenarios[0].cycle_rank, 0);
        assert!(
            profile.scenarios[0]
                .weighted_mean_network_path_length
                .unwrap()
                > profile.baseline.weighted_mean_network_path_length.unwrap()
        );
        assert_ne!(
            profile.canonical_digest_hex().unwrap(),
            graph.canonical_digest_hex().unwrap()
        );
    }

    #[test]
    fn anastomosis_cycle_preserves_demand_after_edge_failure_better_than_tree() {
        let ring = triangle_graph();
        let mut tree = triangle_graph();
        tree.source_model_digest = "b".repeat(64);
        tree.source_filament_ids = vec![1, 2];
        tree.segments.retain(|segment| segment.id != "edge-ca");
        tree.junctions[0].kind = JunctionKind::Tip;
        tree.junctions[0].incident_filament_ids = vec![1];
        tree.junctions[0].terminal_filament_ids = vec![1];
        tree.junctions[1].kind = JunctionKind::Branch;
        tree.junctions[1].terminal_filament_ids.clear();
        tree.junctions[2].kind = JunctionKind::Tip;
        tree.junctions[2].incident_filament_ids = vec![2];
        tree.junctions[2].terminal_filament_ids = vec![2];
        tree.junctions[2].anastomosis_ids.clear();

        let scenario = MycelialFailureScenario {
            scenario_id: "remove-edge-ab".to_string(),
            failed_segment_ids: vec!["edge-ab".to_string()],
            failed_junction_ids: Vec::new(),
        };
        let ring_profile =
            build_mycelial_failure_resilience_profile(&ring, &demands(), &[scenario.clone()])
                .unwrap();
        let tree_profile =
            build_mycelial_failure_resilience_profile(&tree, &demands(), &[scenario]).unwrap();
        assert_eq!(ring_profile.scenarios[0].weighted_reachability, Some(1.0));
        assert_eq!(tree_profile.scenarios[0].weighted_reachability, Some(0.25));
        assert_eq!(ring_profile.scenarios[0].connected_components, 1);
        assert_eq!(tree_profile.scenarios[0].connected_components, 2);
    }

    #[test]
    fn junction_failure_exposes_loss_of_functional_demand() {
        let graph = triangle_graph();
        let scenario = MycelialFailureScenario {
            scenario_id: "remove-junction-a".to_string(),
            failed_segment_ids: Vec::new(),
            failed_junction_ids: vec!["A".to_string()],
        };
        let profile =
            build_mycelial_failure_resilience_profile(&graph, &demands(), &[scenario]).unwrap();
        assert_eq!(profile.scenarios[0].weighted_reachability, Some(0.25));
        assert_eq!(profile.scenarios[0].unreachable_demand_pair_count, 2);
        assert_eq!(profile.scenarios[0].connected_components, 1);
    }

    fn bfs_reachable(
        graph: &MycelialJunctionHyphaGraph,
        demand: &MycelialDemandPair,
        failed_segments: &[String],
        failed_junctions: &[String],
    ) -> bool {
        let failed_segments: BTreeSet<_> = failed_segments.iter().map(String::as_str).collect();
        let failed_junctions: BTreeSet<_> = failed_junctions.iter().map(String::as_str).collect();
        if failed_junctions.contains(demand.source_junction_id.as_str())
            || failed_junctions.contains(demand.target_junction_id.as_str())
        {
            return false;
        }
        let mut adjacency = BTreeMap::<&str, Vec<&str>>::new();
        for junction in &graph.junctions {
            adjacency.insert(junction.id.as_str(), Vec::new());
        }
        for segment in &graph.segments {
            if failed_segments.contains(segment.id.as_str())
                || failed_junctions.contains(segment.start_junction_id.as_str())
                || failed_junctions.contains(segment.end_junction_id.as_str())
            {
                continue;
            }
            adjacency
                .get_mut(segment.start_junction_id.as_str())
                .unwrap()
                .push(segment.end_junction_id.as_str());
            adjacency
                .get_mut(segment.end_junction_id.as_str())
                .unwrap()
                .push(segment.start_junction_id.as_str());
        }
        let mut visited = BTreeSet::from([demand.source_junction_id.as_str()]);
        let mut queue = VecDeque::from([demand.source_junction_id.as_str()]);
        while let Some(current) = queue.pop_front() {
            if current == demand.target_junction_id.as_str() {
                return true;
            }
            for neighbour in &adjacency[current] {
                if !failed_junctions.contains(*neighbour) && visited.insert(*neighbour) {
                    queue.push_back(*neighbour);
                }
            }
        }
        false
    }

    #[test]
    fn single_failure_sweep_matches_independent_bfs_reachability_oracle() {
        let graph = triangle_graph();
        let demands = demands();
        let profile = build_single_failure_resilience_profile(&graph, &demands).unwrap();
        for metric in std::iter::once(&profile.baseline).chain(profile.scenarios.iter()) {
            for (demand, outcome) in demands.iter().zip(&metric.demand_pair_outcomes) {
                let expected = bfs_reachable(
                    &graph,
                    demand,
                    &metric.failed_segment_ids,
                    &metric.failed_junction_ids,
                );
                let measured = outcome.network_path_length_model_units.is_some();
                assert_eq!(
                    measured, expected,
                    "scenario={} demand={}",
                    metric.scenario_id, demand.id
                );
            }
        }
    }

    #[test]
    fn single_failure_sweep_is_deterministic_and_covers_every_edge_and_junction() {
        let graph = triangle_graph();
        let first = build_single_failure_resilience_profile(&graph, &demands()).unwrap();
        let second = build_single_failure_resilience_profile(&graph, &demands()).unwrap();
        assert_eq!(
            first.canonical_digest_hex().unwrap(),
            second.canonical_digest_hex().unwrap()
        );
        assert_eq!(first.scenarios.len(), 6);
        assert!(
            first
                .scenarios
                .iter()
                .any(|scenario| scenario.scenario_id == "edge:edge-ab")
        );
        assert!(
            first
                .scenarios
                .iter()
                .any(|scenario| scenario.scenario_id == "junction:A")
        );
    }

    #[test]
    fn profile_validator_recomputes_weighted_reachability_from_pair_outcomes() {
        let graph = triangle_graph();
        let scenario = MycelialFailureScenario {
            scenario_id: "remove-edge-ab".to_string(),
            failed_segment_ids: vec!["edge-ab".to_string()],
            failed_junction_ids: Vec::new(),
        };
        let mut profile =
            build_mycelial_failure_resilience_profile(&graph, &demands(), &[scenario]).unwrap();
        profile.scenarios[0].reachable_demand_weight -= 1;
        assert!(matches!(
            profile.validate(),
            Err(MycelialFailureResilienceError::InvalidProfile(_))
        ));
    }

    #[test]
    fn profile_validator_binds_exact_failed_identity_lists() {
        let graph = triangle_graph();
        let scenario = MycelialFailureScenario {
            scenario_id: "remove-edge-ab".to_string(),
            failed_segment_ids: vec!["edge-ab".to_string()],
            failed_junction_ids: Vec::new(),
        };
        let mut profile =
            build_mycelial_failure_resilience_profile(&graph, &demands(), &[scenario]).unwrap();
        profile.scenarios[0].failed_segment_ids.clear();
        assert!(matches!(
            profile.validate(),
            Err(MycelialFailureResilienceError::InvalidProfile(_))
        ));
    }

    #[test]
    fn attack_plans_are_deterministic_nested_and_bound_to_source_evidence() {
        let graph = triangle_graph();
        let demand_pairs = demands();
        let config = MycelialFailureAttackConfig {
            strategy: MycelialFailureAttackStrategy::RandomSegments,
            seed: 0x5eed,
            maximum_failure_budget: 2,
            spatial_seed_junction_id: None,
        };
        let first = build_mycelial_failure_attack_plan(
            &graph,
            &demand_pairs,
            config.clone(),
        )
        .unwrap();
        let second =
            build_mycelial_failure_attack_plan(&graph, &demand_pairs, config).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.scenarios.len(), 2);
        assert_eq!(first.ranked_entity_ids.len(), 3);
        let budget_one: BTreeSet<_> =
            first.scenarios[0].failed_segment_ids.iter().collect();
        let budget_two: BTreeSet<_> =
            first.scenarios[1].failed_segment_ids.iter().collect();
        assert!(budget_one.is_subset(&budget_two));
        first.validate_against(&graph, &demand_pairs).unwrap();

        let evaluation =
            evaluate_mycelial_failure_attack_plan(&graph, &demand_pairs, &first).unwrap();
        assert_eq!(evaluation.resilience_profile.scenarios.len(), 2);
        assert_eq!(
            evaluation.attack_plan_digest,
            first.canonical_digest_hex().unwrap()
        );
        assert_eq!(
            evaluation.canonical_digest_hex().unwrap(),
            evaluation.canonical_digest_hex().unwrap()
        );
    }

    #[test]
    fn targeted_and_spatial_attack_rankings_match_their_declared_policies() {
        let graph = triangle_graph();
        let demand_pairs = demands();

        let degree = build_mycelial_failure_attack_plan(
            &graph,
            &demand_pairs,
            MycelialFailureAttackConfig {
                strategy: MycelialFailureAttackStrategy::HighestDegreeJunctions,
                seed: 0,
                maximum_failure_budget: 1,
                spatial_seed_junction_id: None,
            },
        )
        .unwrap();
        assert_eq!(degree.ranked_entity_ids[0], "A");

        let load = build_mycelial_failure_attack_plan(
            &graph,
            &demand_pairs,
            MycelialFailureAttackConfig {
                strategy: MycelialFailureAttackStrategy::DemandPathLoadSegments,
                seed: 0,
                maximum_failure_budget: 1,
                spatial_seed_junction_id: None,
            },
        )
        .unwrap();
        assert_eq!(load.ranked_entity_ids[0], "edge-ca");

        let spatial = build_mycelial_failure_attack_plan(
            &graph,
            &demand_pairs,
            MycelialFailureAttackConfig {
                strategy: MycelialFailureAttackStrategy::SpatialClusterJunctions,
                seed: 0,
                maximum_failure_budget: 2,
                spatial_seed_junction_id: Some("A".to_string()),
            },
        )
        .unwrap();
        assert_eq!(spatial.ranked_entity_ids[0], "A");
        assert!(["B", "C"].contains(&spatial.ranked_entity_ids[1].as_str()));
        assert_eq!(spatial.scenarios[1].failed_junction_ids[0], "A");
        assert!(["B", "C"].contains(&spatial.scenarios[1].failed_junction_ids[1].as_str()));
    }

    #[test]
    fn adaptive_demand_path_load_recomputes_on_the_residual_graph() {
        let graph = adaptive_path_load_graph();
        graph.validate().unwrap();
        let demand_pairs = adaptive_path_load_demands();

        let static_plan = build_mycelial_failure_attack_plan(
            &graph,
            &demand_pairs,
            MycelialFailureAttackConfig {
                strategy: MycelialFailureAttackStrategy::DemandPathLoadSegments,
                seed: 0,
                maximum_failure_budget: 2,
                spatial_seed_junction_id: None,
            },
        )
        .unwrap();
        let adaptive_plan = build_mycelial_failure_attack_plan(
            &graph,
            &demand_pairs,
            MycelialFailureAttackConfig {
                strategy: MycelialFailureAttackStrategy::AdaptiveDemandPathLoadSegments,
                seed: 0,
                maximum_failure_budget: 2,
                spatial_seed_junction_id: None,
            },
        )
        .unwrap();
        assert_eq!(&static_plan.ranked_entity_ids[..2], &["edge-ab", "edge-bc"]);
        assert_eq!(adaptive_plan.ranked_entity_ids[0], "edge-ab");
        assert!(
            ["edge-ad", "edge-cd"].contains(&adaptive_plan.ranked_entity_ids[1].as_str())
        );
        assert_eq!(
            adaptive_plan.ranked_entity_ids_per_step[1][0],
            adaptive_plan.ranked_entity_ids[1]
        );
        assert!(
            adaptive_plan.ranked_entity_ids_per_step[1]
                .iter()
                .all(|id| id != "edge-ab")
        );
        adaptive_plan.validate_against(&graph, &demand_pairs).unwrap();

        let static_junction_plan = build_mycelial_failure_attack_plan(
            &graph,
            &demand_pairs,
            MycelialFailureAttackConfig {
                strategy: MycelialFailureAttackStrategy::DemandPathLoadJunctions,
                seed: 0,
                maximum_failure_budget: 2,
                spatial_seed_junction_id: None,
            },
        )
        .unwrap();
        let adaptive_junction_plan = build_mycelial_failure_attack_plan(
            &graph,
            &demand_pairs,
            MycelialFailureAttackConfig {
                strategy: MycelialFailureAttackStrategy::AdaptiveDemandPathLoadJunctions,
                seed: 0,
                maximum_failure_budget: 2,
                spatial_seed_junction_id: None,
            },
        )
        .unwrap();
        assert_eq!(static_junction_plan.ranked_entity_ids[0], "B");
        assert_eq!(adaptive_junction_plan.ranked_entity_ids[0], "B");
        assert_eq!(adaptive_junction_plan.ranked_entity_ids[1], "D");
        assert_eq!(
            adaptive_junction_plan.ranked_entity_ids_per_step[1][0],
            "D"
        );
        adaptive_junction_plan
            .validate_against(&graph, &demand_pairs)
            .unwrap();
    }

    #[test]
    fn adaptive_degree_recomputes_ranking_after_each_prior_removal() {
        let graph = adaptive_attack_graph();
        graph.validate().unwrap();
        let demand_pairs = demands();
        let static_plan = build_mycelial_failure_attack_plan(
            &graph,
            &demand_pairs,
            MycelialFailureAttackConfig {
                strategy: MycelialFailureAttackStrategy::HighestDegreeJunctions,
                seed: 0,
                maximum_failure_budget: 2,
                spatial_seed_junction_id: None,
            },
        )
        .unwrap();
        let adaptive_plan = build_mycelial_failure_attack_plan(
            &graph,
            &demand_pairs,
            MycelialFailureAttackConfig {
                strategy: MycelialFailureAttackStrategy::AdaptiveHighestDegreeJunctions,
                seed: 0,
                maximum_failure_budget: 2,
                spatial_seed_junction_id: None,
            },
        )
        .unwrap();
        assert_eq!(static_plan.ranked_entity_ids[0], "A");
        assert!(["B", "C", "D", "E"].contains(&static_plan.ranked_entity_ids[1].as_str()));
        assert_eq!(adaptive_plan.ranked_entity_ids[0], "A");
        assert_eq!(adaptive_plan.ranked_entity_ids[1], "E");
        assert_eq!(adaptive_plan.ranked_entity_ids_per_step[1][0], "E");
        assert!(
            adaptive_plan.ranked_entity_ids_per_step[1]
                .iter()
                .all(|id| id != "A")
        );
        adaptive_plan.validate_against(&graph, &demand_pairs).unwrap();
    }

    #[test]
    fn lowest_degree_attack_prioritizes_peripheral_junctions() {
        let mut graph = triangle_graph();
        graph.source_model_digest = "d".repeat(64);
        graph.source_filament_ids.push(4);
        graph.junctions[0].incident_filament_ids.push(4);
        graph.junctions[0].incident_filament_ids.sort_unstable();
        graph.junctions.push(MycelialJunction {
            id: "D".to_string(),
            kind: JunctionKind::Tip,
            position: SpatialPoint3 {
                x: 0.0,
                y: 0.0,
                z: 2.0,
            },
            incident_filament_ids: vec![4],
            terminal_filament_ids: vec![4],
            anastomosis_ids: Vec::new(),
        });
        graph.segments.push(MycelialHyphaSegment {
            id: "edge-ad".to_string(),
            filament_id: 4,
            t_start_micros: 0,
            t_end_micros: PARAMETER_SCALE,
            start_junction_id: "A".to_string(),
            end_junction_id: "D".to_string(),
            arc_length_model_units: 2.0,
            start_junction_offset_model_units: 0.0,
            end_junction_offset_model_units: 0.0,
            radius_start_model_units: 1.0,
            radius_end_model_units: 1.0,
        });
        graph.segments.sort_by(|left, right| left.id.cmp(&right.id));
        graph.validate().unwrap();

        let plan = build_mycelial_failure_attack_plan(
            &graph,
            &demands(),
            MycelialFailureAttackConfig {
                strategy: MycelialFailureAttackStrategy::LowestDegreeJunctions,
                seed: 0,
                maximum_failure_budget: 1,
                spatial_seed_junction_id: None,
            },
        )
        .unwrap();
        assert_eq!(plan.ranked_entity_ids[0], "D");
    }

    #[test]
    fn attack_plan_mutation_and_missing_spatial_seed_fail_closed() {
        let graph = triangle_graph();
        let demand_pairs = demands();
        let mut plan = build_mycelial_failure_attack_plan(
            &graph,
            &demand_pairs,
            MycelialFailureAttackConfig {
                strategy: MycelialFailureAttackStrategy::HighestDegreeJunctions,
                seed: 7,
                maximum_failure_budget: 2,
                spatial_seed_junction_id: None,
            },
        )
        .unwrap();
        plan.ranked_entity_ids.reverse();
        assert!(matches!(
            plan.validate_against(&graph, &demand_pairs),
            Err(MycelialFailureResilienceError::InvalidAttackPlan(_))
        ));
        assert!(matches!(
            build_mycelial_failure_attack_plan(
                &graph,
                &demand_pairs,
                MycelialFailureAttackConfig {
                    strategy: MycelialFailureAttackStrategy::SpatialClusterJunctions,
                    seed: 7,
                    maximum_failure_budget: 1,
                    spatial_seed_junction_id: None,
                }
            ),
            Err(MycelialFailureResilienceError::InvalidAttackConfig(_))
        ));
    }

    #[test]
    fn attack_delivery_zero_probability_reproduces_intact_baseline() {
        let graph = triangle_graph();
        let demand_pairs = demands();
        let plan = build_mycelial_failure_attack_plan(
            &graph,
            &demand_pairs,
            MycelialFailureAttackConfig {
                strategy: MycelialFailureAttackStrategy::HighestDegreeJunctions,
                seed: 17,
                maximum_failure_budget: 2,
                spatial_seed_junction_id: None,
            },
        )
        .unwrap();
        let profile = build_mycelial_failure_delivery_profile(
            &graph,
            &demand_pairs,
            &plan,
            MycelialFailureDeliveryConfig {
                trials_per_scenario: 8,
                removal_success_probability_micros: 0,
                seed: 0xdead_beef,
            },
        )
        .unwrap();
        assert_eq!(profile.trials.len(), 16);
        assert!(
            profile.trials.iter().all(|trial| {
                trial.realized_failed_segment_ids.is_empty()
                    && trial.realized_failed_junction_ids.is_empty()
                    && trial.weighted_reachability == Some(1.0)
            })
        );
        assert!(
            profile
                .scenarios
                .iter()
                .all(|summary| summary.mean_realized_failure_count == 0.0
                    && summary.probability_at_least_one_failure == 0.0
                    && summary.mean_weighted_reachability == Some(1.0))
        );
        profile.validate_against(&graph, &demand_pairs, &plan).unwrap();
    }

    #[test]
    fn attack_delivery_full_success_matches_deterministic_removal_results() {
        let graph = triangle_graph();
        let demand_pairs = demands();
        let plan = build_mycelial_failure_attack_plan(
            &graph,
            &demand_pairs,
            MycelialFailureAttackConfig {
                strategy: MycelialFailureAttackStrategy::DemandPathLoadSegments,
                seed: 91,
                maximum_failure_budget: 2,
                spatial_seed_junction_id: None,
            },
        )
        .unwrap();
        let deterministic =
            build_mycelial_failure_resilience_profile(&graph, &demand_pairs, &plan.scenarios)
                .unwrap();
        let delivery = build_mycelial_failure_delivery_profile(
            &graph,
            &demand_pairs,
            &plan,
            MycelialFailureDeliveryConfig {
                trials_per_scenario: 4,
                removal_success_probability_micros: 1_000_000,
                seed: 123,
            },
        )
        .unwrap();
        for (scenario, metric) in plan.scenarios.iter().zip(&deterministic.scenarios) {
            let trials: Vec<_> = delivery
                .trials
                .iter()
                .filter(|trial| trial.scenario_id == scenario.scenario_id)
                .collect();
            assert_eq!(trials.len(), 4);
            assert!(
                trials.iter().all(|trial| {
                    trial.realized_failed_segment_ids == scenario.failed_segment_ids
                        && trial.realized_failed_junction_ids == scenario.failed_junction_ids
                        && trial.weighted_reachability == metric.weighted_reachability
                        && trial.weighted_mean_directness == metric.weighted_mean_directness
                        && trial.weighted_mean_network_path_length
                            == metric.weighted_mean_network_path_length
                        && trial.connected_components == metric.connected_components
                        && trial.cycle_rank == metric.cycle_rank
                })
            );
        }
        assert!(
            delivery
                .scenarios
                .iter()
                .all(|summary| summary.mean_realized_failure_count
                    == summary.attempted_entity_count as f64
                    && summary.probability_at_least_one_failure == 1.0)
        );
    }

    #[test]
    fn attack_delivery_uses_common_random_numbers_across_strategies_and_nested_budgets() {
        let graph = triangle_graph();
        let demand_pairs = demands();
        let random_plan = build_mycelial_failure_attack_plan(
            &graph,
            &demand_pairs,
            MycelialFailureAttackConfig {
                strategy: MycelialFailureAttackStrategy::RandomSegments,
                seed: 11,
                maximum_failure_budget: 3,
                spatial_seed_junction_id: None,
            },
        )
        .unwrap();
        let demand_plan = build_mycelial_failure_attack_plan(
            &graph,
            &demand_pairs,
            MycelialFailureAttackConfig {
                strategy: MycelialFailureAttackStrategy::DemandPathLoadSegments,
                seed: 97,
                maximum_failure_budget: 3,
                spatial_seed_junction_id: None,
            },
        )
        .unwrap();
        let config = MycelialFailureDeliveryConfig {
            trials_per_scenario: 24,
            removal_success_probability_micros: 400_000,
            seed: 0x1a2b_3c4d,
        };
        let random_profile =
            build_mycelial_failure_delivery_profile(&graph, &demand_pairs, &random_plan, config)
                .unwrap();
        let demand_profile =
            build_mycelial_failure_delivery_profile(&graph, &demand_pairs, &demand_plan, config)
                .unwrap();

        let random_final_start = (random_plan.scenarios.len() - 1) * config.trials_per_scenario;
        let demand_final_start = (demand_plan.scenarios.len() - 1) * config.trials_per_scenario;
        for trial_index in 0..config.trials_per_scenario {
            assert_eq!(
                random_profile.trials[random_final_start + trial_index]
                    .realized_failed_segment_ids,
                demand_profile.trials[demand_final_start + trial_index]
                    .realized_failed_segment_ids
            );
        }

        for trial_index in 0..config.trials_per_scenario {
            let mut previous = BTreeSet::new();
            for scenario_index in 0..random_plan.scenarios.len() {
                let trial =
                    &random_profile.trials[scenario_index * config.trials_per_scenario + trial_index];
                let current: BTreeSet<_> =
                    trial.realized_failed_segment_ids.iter().cloned().collect();
                assert!(previous.is_subset(&current));
                previous = current;
            }
        }
        random_profile
            .validate_against(&graph, &demand_pairs, &random_plan)
            .unwrap();
        demand_profile
            .validate_against(&graph, &demand_pairs, &demand_plan)
            .unwrap();
    }

    #[test]
    fn attack_delivery_trials_are_seed_replayable_and_detached_mutations_reject() {
        let graph = triangle_graph();
        let demand_pairs = demands();
        let plan = build_mycelial_failure_attack_plan(
            &graph,
            &demand_pairs,
            MycelialFailureAttackConfig {
                strategy: MycelialFailureAttackStrategy::RandomSegments,
                seed: 0x1234_5678,
                maximum_failure_budget: 2,
                spatial_seed_junction_id: None,
            },
        )
        .unwrap();
        let config = MycelialFailureDeliveryConfig {
            trials_per_scenario: 32,
            removal_success_probability_micros: 500_000,
            seed: 0xfeed_1234,
        };
        let first =
            build_mycelial_failure_delivery_profile(&graph, &demand_pairs, &plan, config)
                .unwrap();
        let second =
            build_mycelial_failure_delivery_profile(&graph, &demand_pairs, &plan, config)
                .unwrap();
        assert_eq!(first, second);
        assert_eq!(
            first.canonical_digest_hex().unwrap(),
            second.canonical_digest_hex().unwrap()
        );
        let mut tampered = first.clone();
        tampered.trials[0].realized_failed_segment_ids.clear();
        assert!(tampered.canonical_digest_hex().is_err());
        assert!(
            tampered
                .validate_against(&graph, &demand_pairs, &plan)
                .is_err()
        );

        let mut metric_tampered = first;
        metric_tampered.trials[0].connected_components =
            metric_tampered.trials[0].connected_components.saturating_add(1);
        assert!(
            metric_tampered
                .validate_against(&graph, &demand_pairs, &plan)
                .is_err()
        );
    }

    #[test]
    fn invalid_or_ambiguous_demand_scenarios_fail_closed() {
        let graph = triangle_graph();
        let mut bad_demands = demands();
        bad_demands.reverse();
        assert!(matches!(
            build_mycelial_failure_resilience_profile(&graph, &bad_demands, &[]),
            Err(MycelialFailureResilienceError::InvalidDemand(_))
        ));
        let mut duplicate_edge_scenario = MycelialFailureScenario {
            scenario_id: "bad".to_string(),
            failed_segment_ids: vec!["edge-ab".to_string(), "edge-ab".to_string()],
            failed_junction_ids: Vec::new(),
        };
        assert!(matches!(
            build_mycelial_failure_resilience_profile(
                &graph,
                &demands(),
                &[duplicate_edge_scenario.clone()]
            ),
            Err(MycelialFailureResilienceError::InvalidScenario(_))
        ));
        duplicate_edge_scenario.failed_segment_ids = vec!["missing-edge".to_string()];
        assert!(matches!(
            build_mycelial_failure_resilience_profile(
                &graph,
                &demands(),
                &[duplicate_edge_scenario]
            ),
            Err(MycelialFailureResilienceError::UnknownFailedSegment(_))
        ));
    }
}
