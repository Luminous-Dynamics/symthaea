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
    JunctionLengthUnit, MycelialJunctionHyphaGraph, MycelialJunctionGraphError,
};
use crate::mycelial_model::SpatialPoint3;

pub const MYCELIAL_FAILURE_PROFILE_SCHEMA_VERSION: u16 = 1;
pub const MAX_FAILURE_JUNCTIONS: usize = 256;
pub const MAX_FAILURE_SEGMENTS: usize = 512;
pub const MAX_FAILURE_DEMAND_PAIRS: usize = 256;
pub const MAX_FAILURE_DEMAND_SOURCES: usize = 32;
pub const MAX_FAILURE_SCENARIOS: usize = 768;
pub const MAX_FAILURE_ID_BYTES: usize = 128;

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
pub struct MycelialFailureMetrics {
    pub scenario_id: String,
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
    TooManyScenarios(usize),
    InvalidDemand(String),
    DuplicateDemandId(String),
    UnknownDemandJunction(String),
    InvalidScenario(String),
    DuplicateScenarioId(String),
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
                write!(f, "demand-pair count {count} exceeds {MAX_FAILURE_DEMAND_PAIRS}")
            }
            Self::TooManyDemandSources(count) => {
                write!(f, "distinct demand-source count {count} exceeds {MAX_FAILURE_DEMAND_SOURCES}")
            }
            Self::TooManyScenarios(count) => {
                write!(f, "failure-scenario count {count} exceeds {MAX_FAILURE_SCENARIOS}")
            }
            Self::InvalidDemand(id) => write!(f, "invalid demand pair {id}"),
            Self::DuplicateDemandId(id) => write!(f, "duplicate demand-pair ID {id}"),
            Self::UnknownDemandJunction(id) => {
                write!(f, "demand pair references unknown junction {id}")
            }
            Self::InvalidScenario(id) => write!(f, "invalid failure scenario {id}"),
            Self::DuplicateScenarioId(id) => write!(f, "duplicate failure scenario ID {id}"),
            Self::UnknownFailedSegment(id) => write!(f, "failure references unknown segment {id}"),
            Self::UnknownFailedJunction(id) => write!(f, "failure references unknown junction {id}"),
            Self::InvalidProfile(reason) => write!(f, "invalid failure-resilience profile: {reason}"),
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
        let source = *node_index
            .get(&segment.start_junction_id)
            .ok_or(MycelialFailureResilienceError::UnknownDemandJunction(
                segment.start_junction_id.clone(),
            ))?;
        let target = *node_index
            .get(&segment.end_junction_id)
            .ok_or(MycelialFailureResilienceError::UnknownDemandJunction(
                segment.end_junction_id.clone(),
            ))?;
        edge_endpoints.push((
            source,
            target,
            segment.traversal_length_model_units(),
        ));
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
            || !scenario_ids.insert(scenario.scenario_id.as_str())
            || scenario.failed_segment_ids.windows(2).any(|pair| pair[0] >= pair[1])
            || scenario.failed_junction_ids.windows(2).any(|pair| pair[0] >= pair[1])
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
) -> MycelialFailureMetrics {
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
            left.0.cmp(&right.0).then_with(|| left.1.total_cmp(&right.1))
        });
    }

    let remaining_junction_count = active_nodes.iter().filter(|active| **active).count();
    let connected_components = component_count(&adjacency, &active_nodes);
    let cycle_rank = remaining_segment_count
        .saturating_add(connected_components)
        .saturating_sub(remaining_junction_count);

    let mut distances_by_source = HashMap::<usize, Vec<f64>>::new();
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
            continue;
        }
        let distances = distances_by_source
            .entry(source)
            .or_insert_with(|| dijkstra(&adjacency, source));
        let path_length = distances[target];
        if !path_length.is_finite() {
            continue;
        }
        let direct_distance = indexed.positions[source]
            .sub(indexed.positions[target])
            .length() as f64;
        if !path_length.is_finite() || path_length <= 0.0
            || direct_distance as f64 > path_length + 1.0e-5
        {
            // A graph path shorter than Euclidean endpoint distance is an invalid
            // source graph or a metric-contract violation. The enclosing
            // profile builder validates source geometry; use no contribution.
            continue;
        }
        reachable_demand_pair_count += 1;
        reachable_demand_weight += weight;
        weighted_directness_sum += (direct_distance / path_length) * weight as f64;
        weighted_path_length_sum += path_length * weight as f64;
    }
    let unreachable_demand_pair_count = demands.len().saturating_sub(reachable_demand_pair_count);
    let unreachable_demand_weight = total_demand_weight.saturating_sub(reachable_demand_weight);
    let weighted_reachability = (total_demand_weight > 0)
        .then_some(reachable_demand_weight as f64 / total_demand_weight as f64);
    let weighted_mean_directness = (reachable_demand_weight > 0)
        .then_some(weighted_directness_sum / reachable_demand_weight as f64);
    let weighted_mean_network_path_length = (reachable_demand_weight > 0)
        .then_some(weighted_path_length_sum / reachable_demand_weight as f64);

    MycelialFailureMetrics {
        scenario_id: scenario_id.to_string(),
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
    }
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
    let baseline = evaluate_one(
        &indexed,
        demands,
        "baseline",
        &[],
        &[],
        graph.length_unit,
    );
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
        .collect::<Vec<_>>();
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

impl MycelialFailureResilienceProfile {
    pub fn validate(&self) -> Result<(), MycelialFailureResilienceError> {
        if self.schema_version != MYCELIAL_FAILURE_PROFILE_SCHEMA_VERSION
            || !valid_digest(&self.source_graph_digest)
            || !valid_digest(&self.demand_digest)
            || self.baseline.scenario_id != "baseline"
            || self.baseline.failed_segment_count != 0
            || self.baseline.failed_junction_count != 0
            || self.scenarios.windows(2).any(|pair| {
                pair[0].scenario_id >= pair[1].scenario_id
            })
        {
            return Err(MycelialFailureResilienceError::InvalidProfile(
                "invalid version, digest, baseline or scenario order",
            ));
        }
        for metric in std::iter::once(&self.baseline).chain(self.scenarios.iter()) {
            if !valid_id(&metric.scenario_id)
                || metric.reachable_demand_pair_count + metric.unreachable_demand_pair_count
                    != metric.demand_pair_count
                || metric.reachable_demand_weight + metric.unreachable_demand_weight
                    != metric.total_demand_weight
                || metric.remaining_junction_count > MAX_FAILURE_JUNCTIONS
                || metric.remaining_segment_count > MAX_FAILURE_SEGMENTS
                || metric.cycle_rank
                    != metric.remaining_segment_count
                        .saturating_add(metric.connected_components)
                        .saturating_sub(metric.remaining_junction_count)
                || metric.weighted_reachability.is_some_and(|value| {
                    !value.is_finite() || !(0.0..=1.0).contains(&value)
                })
                || metric.weighted_mean_directness.is_some_and(|value| {
                    !value.is_finite() || !(0.0..=1.0 + 1.0e-8).contains(&value)
                })
                || metric.weighted_mean_network_path_length.is_some_and(|value| {
                    !value.is_finite() || value <= 0.0
                })
                || (metric.total_demand_weight == 0) != metric.weighted_reachability.is_none()
                || (metric.reachable_demand_weight == 0)
                    != (metric.weighted_mean_directness.is_none()
                        && metric.weighted_mean_network_path_length.is_none())
                || (metric.reachable_demand_weight > 0
                    && (metric.weighted_mean_directness.is_none()
                        || metric.weighted_mean_network_path_length.is_none()))
            {
                return Err(MycelialFailureResilienceError::InvalidProfile(
                    "scenario metric totals, bounds or optional values are inconsistent",
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
    !value.is_empty()
        && value.len() <= MAX_FAILURE_ID_BYTES
        && !value.chars().any(char::is_control)
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
        JunctionKind, MycelialHyphaSegment, MycelialJunction, MycelialJunctionHyphaGraph,
        MYCELIAL_JUNCTION_GRAPH_SCHEMA_VERSION, PARAMETER_SCALE,
    };

    fn triangle_graph() -> MycelialJunctionHyphaGraph {
        let mut a = MycelialJunction {
            id: "A".to_string(),
            kind: JunctionKind::Branch,
            position: SpatialPoint3 { x: 0.0, y: 0.0, z: 0.0 },
            incident_filament_ids: vec![1, 3],
            terminal_filament_ids: Vec::new(),
            anastomosis_ids: Vec::new(),
        };
        let mut b = MycelialJunction {
            id: "B".to_string(),
            kind: JunctionKind::Tip,
            position: SpatialPoint3 { x: 1.0, y: 0.0, z: 0.0 },
            incident_filament_ids: vec![1, 2],
            terminal_filament_ids: vec![1],
            anastomosis_ids: Vec::new(),
        };
        let mut c = MycelialJunction {
            id: "C".to_string(),
            kind: JunctionKind::Anastomosis,
            position: SpatialPoint3 { x: 0.5, y: 1.0, z: 0.0 },
            incident_filament_ids: vec![2, 3],
            terminal_filament_ids: vec![2],
            anastomosis_ids: vec![7],
        };
        // Keep the fixture identity lists canonical for the graph validator.
        a.incident_filament_ids.sort_unstable();
        b.incident_filament_ids.sort_unstable();
        c.incident_filament_ids.sort_unstable();
        let segment = |id: &str, filament_id, start: &str, end: &str, length: f64| MycelialHyphaSegment {
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
        assert!(profile.scenarios[0].weighted_mean_network_path_length.unwrap()
            > profile.baseline.weighted_mean_network_path_length.unwrap());
        assert_ne!(
            profile.canonical_digest_hex().unwrap(),
            graph.canonical_digest_hex().unwrap()
        );
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

    #[test]
    fn single_failure_sweep_is_deterministic_and_covers_every_edge_and_junction() {
        let graph = triangle_graph();
        let first = build_single_failure_resilience_profile(&graph, &demands()).unwrap();
        let second = build_single_failure_resilience_profile(&graph, &demands()).unwrap();
        assert_eq!(first.canonical_digest_hex().unwrap(), second.canonical_digest_hex().unwrap());
        assert_eq!(first.scenarios.len(), 6);
        assert!(first.scenarios.iter().any(|scenario| scenario.scenario_id == "edge:edge-ab"));
        assert!(first.scenarios.iter().any(|scenario| scenario.scenario_id == "junction:A"));
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
