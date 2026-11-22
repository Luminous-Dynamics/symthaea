// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Matched reference networks for the renderer-neutral junction–hypha graph.
//!
//! This module currently implements the weighted minimum-spanning forest of
//! the observed network. It is not a Euclidean MST: it selects a spanning
//! forest from the same observed hypha-segment edge set, weighted by curve
//! traversal length. A Euclidean MST and Delaunay graph require separate,
//! explicitly dimensioned reference constructions.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::mycelial_junction_graph::{
    GeometricEfficiencyReport, JunctionLengthUnit, MycelialJunctionHyphaGraph,
};

pub const MAX_REFERENCE_NETWORK_JUNCTIONS: usize = 2_048;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialMinimumSpanningForestReport {
    pub source_graph_digest: String,
    pub length_unit: JunctionLengthUnit,
    /// Selected hypha segment IDs, in deterministic Kruskal selection order.
    pub selected_segment_ids: Vec<String>,
    pub selected_segment_count: usize,
    pub connected_components: usize,
    /// Cost proxy: selected source-curve arc lengths, excluding weld offsets.
    pub hypha_arc_cost_model_units_microunits: u64,
    /// Path-weight sum: arc lengths plus the separately reported weld offsets.
    pub traversal_cost_model_units_microunits: u64,
    pub global_geometric_efficiency: GeometricEfficiencyReport,
    pub root_directed_geometric_efficiency: GeometricEfficiencyReport,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MycelialReferenceNetworkError {
    InvalidSourceGraph(String),
    TooManyJunctions(usize),
    UnknownRootJunction(String),
    InvalidReferenceGraph(&'static str),
}

impl fmt::Display for MycelialReferenceNetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSourceGraph(message) => {
                write!(f, "invalid source junction–hypha graph: {message}")
            }
            Self::TooManyJunctions(count) => {
                write!(f, "reference calculation is bounded to {MAX_REFERENCE_NETWORK_JUNCTIONS} junctions, got {count}")
            }
            Self::UnknownRootJunction(id) => {
                write!(f, "unknown declared root junction {id}")
            }
            Self::InvalidReferenceGraph(reason) => {
                write!(f, "invalid minimum-spanning-forest reference: {reason}")
            }
        }
    }
}

impl Error for MycelialReferenceNetworkError {}

/// Construct a weighted minimum-spanning forest from the observed hypha graph.
///
/// This is the same conceptual reference used in the AMF travelling-wave
/// analysis: it retains only observed network edges and minimizes their
/// measured segment-length weights. It does not invent spatial links between
/// disconnected components. Root labels are caller-declared; the generated
/// renderer model does not supply biological plant-root identities.
pub fn build_observed_network_minimum_spanning_forest(
    graph: &MycelialJunctionHyphaGraph,
    root_junction_ids: &[String],
) -> Result<MycelialMinimumSpanningForestReport, MycelialReferenceNetworkError> {
    graph
        .validate()
        .map_err(|error| MycelialReferenceNetworkError::InvalidSourceGraph(error.to_string()))?;
    if graph.junctions.len() > MAX_REFERENCE_NETWORK_JUNCTIONS {
        return Err(MycelialReferenceNetworkError::TooManyJunctions(
            graph.junctions.len(),
        ));
    }

    let source_graph_digest = graph
        .canonical_digest_hex()
        .map_err(|error| MycelialReferenceNetworkError::InvalidSourceGraph(error.to_string()))?;
    let index_by_id: HashMap<_, _> = graph
        .junctions
        .iter()
        .enumerate()
        .map(|(index, junction)| (junction.id.as_str(), index))
        .collect();

    let mut edges = Vec::with_capacity(graph.segments.len());
    for segment in &graph.segments {
        let source = index_by_id
            .get(segment.start_junction_id.as_str())
            .copied()
            .ok_or(MycelialReferenceNetworkError::InvalidReferenceGraph(
                "segment source junction is missing",
            ))?;
        let target = index_by_id
            .get(segment.end_junction_id.as_str())
            .copied()
            .ok_or(MycelialReferenceNetworkError::InvalidReferenceGraph(
                "segment target junction is missing",
            ))?;
        let weight = segment.traversal_length_model_units();
        if !weight.is_finite() || weight <= 0.0 || source == target {
            return Err(MycelialReferenceNetworkError::InvalidReferenceGraph(
                "segment has an invalid endpoint or traversal weight",
            ));
        }
        edges.push((
            weight,
            segment.id.as_str(),
            source,
            target,
            segment.arc_length_model_units,
        ));
    }
    edges.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(right.1))
            .then_with(|| left.2.cmp(&right.2))
            .then_with(|| left.3.cmp(&right.3))
    });

    let mut disjoint = DisjointSet::new(graph.junctions.len());
    let mut adjacency = vec![Vec::<(usize, f64)>::new(); graph.junctions.len()];
    let mut selected_segment_ids = Vec::new();
    let mut total_arc_length = 0.0f64;
    let mut total_traversal_length = 0.0f64;
    for (weight, segment_id, source, target, arc_length) in edges {
        if disjoint.union(source, target) {
            adjacency[source].push((target, weight));
            adjacency[target].push((source, weight));
            selected_segment_ids.push(segment_id.to_string());
            total_arc_length += arc_length;
            total_traversal_length += weight;
        }
    }
    for neighbours in &mut adjacency {
        neighbours.sort_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.total_cmp(&right.1))
        });
    }

    let components = disjoint.component_count();
    let selected_segment_count = selected_segment_ids.len();
    if selected_segment_count
        != graph
            .junctions
            .len()
            .saturating_sub(components)
    {
        return Err(MycelialReferenceNetworkError::InvalidReferenceGraph(
            "selected edge count does not describe a spanning forest",
        ));
    }

    let global_geometric_efficiency = global_efficiency(graph, &adjacency)?;
    let root_directed_geometric_efficiency =
        root_directed_efficiency(graph, &adjacency, root_junction_ids)?;

    Ok(MycelialMinimumSpanningForestReport {
        source_graph_digest,
        length_unit: graph.length_unit,
        selected_segment_ids,
        selected_segment_count,
        connected_components: components,
        hypha_arc_cost_model_units_microunits: quantize_nonnegative(total_arc_length),
        traversal_cost_model_units_microunits: quantize_nonnegative(total_traversal_length),
        global_geometric_efficiency,
        root_directed_geometric_efficiency,
    })
}

fn global_efficiency(
    graph: &MycelialJunctionHyphaGraph,
    adjacency: &[Vec<(usize, f64)>],
) -> Result<GeometricEfficiencyReport, MycelialReferenceNetworkError> {
    let count = graph.junctions.len();
    let eligible_pairs = count.saturating_mul(count.saturating_sub(1)) / 2;
    let mut sum_directness = 0.0f64;
    let mut sum_path_length = 0.0f64;
    let mut reachable_pairs = 0usize;
    for source in 0..count {
        let distances = shortest_path_distances(adjacency, source);
        for target in (source + 1)..count {
            let path_length = distances[target];
            if !path_length.is_finite() {
                continue;
            }
            let direct = f64::from(
                graph.junctions[source]
                    .position
                    .sub(graph.junctions[target].position)
                    .length(),
            );
            if path_length <= 0.0 || direct > path_length + 1.0e-4 {
                return Err(MycelialReferenceNetworkError::InvalidReferenceGraph(
                    "Euclidean distance exceeds minimum-spanning-forest path length",
                ));
            }
            sum_directness += direct / path_length;
            sum_path_length += path_length;
            reachable_pairs += 1;
        }
    }
    Ok(GeometricEfficiencyReport {
        mean_directness: (reachable_pairs > 0)
            .then_some(sum_directness / reachable_pairs as f64),
        eligible_pairs,
        reachable_pairs,
        unreachable_pairs: eligible_pairs - reachable_pairs,
        mean_network_path_length_model_units: (reachable_pairs > 0)
            .then_some(sum_path_length / reachable_pairs as f64),
        length_unit: graph.length_unit,
    })
}

fn root_directed_efficiency(
    graph: &MycelialJunctionHyphaGraph,
    adjacency: &[Vec<(usize, f64)>],
    root_junction_ids: &[String],
) -> Result<GeometricEfficiencyReport, MycelialReferenceNetworkError> {
    let index_by_id: HashMap<_, _> = graph
        .junctions
        .iter()
        .enumerate()
        .map(|(index, junction)| (junction.id.as_str(), index))
        .collect();
    let mut roots = BTreeSet::new();
    for root_id in root_junction_ids {
        let root = index_by_id
            .get(root_id.as_str())
            .copied()
            .ok_or_else(|| MycelialReferenceNetworkError::UnknownRootJunction(
                root_id.clone(),
            ))?;
        roots.insert(root);
    }
    let terminals: Vec<_> = graph
        .junctions
        .iter()
        .enumerate()
        .filter(|(_, junction)| !junction.terminal_filament_ids.is_empty())
        .map(|(index, _)| index)
        .collect();

    let mut sum_directness = 0.0f64;
    let mut sum_path_length = 0.0f64;
    let mut reachable_pairs = 0usize;
    for terminal in &terminals {
        let distances = shortest_path_distances(adjacency, *terminal);
        let nearest = roots
            .iter()
            .filter_map(|root| {
                let distance = distances[*root];
                distance.is_finite().then_some((*root, distance))
            })
            .min_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
        if let Some((root, path_length)) = nearest {
            let direct = f64::from(
                graph.junctions[*terminal]
                    .position
                    .sub(graph.junctions[root].position)
                    .length(),
            );
            if path_length <= 0.0 || direct > path_length + 1.0e-4 {
                return Err(MycelialReferenceNetworkError::InvalidReferenceGraph(
                    "direct tip-to-root distance exceeds minimum-spanning-forest path length",
                ));
            }
            sum_directness += direct / path_length;
            sum_path_length += path_length;
            reachable_pairs += 1;
        }
    }
    let eligible_pairs = terminals.len();
    Ok(GeometricEfficiencyReport {
        mean_directness: (reachable_pairs > 0)
            .then_some(sum_directness / reachable_pairs as f64),
        eligible_pairs,
        reachable_pairs,
        unreachable_pairs: eligible_pairs - reachable_pairs,
        mean_network_path_length_model_units: (reachable_pairs > 0)
            .then_some(sum_path_length / reachable_pairs as f64),
        length_unit: graph.length_unit,
    })
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

fn shortest_path_distances(adjacency: &[Vec<(usize, f64)>], source: usize) -> Vec<f64> {
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

struct DisjointSet {
    parent: Vec<usize>,
    components: usize,
}

impl DisjointSet {
    fn new(count: usize) -> Self {
        Self {
            parent: (0..count).collect(),
            components: count,
        }
    }

    fn find(&mut self, value: usize) -> usize {
        let mut root = value;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        let mut cursor = value;
        while self.parent[cursor] != cursor {
            let next = self.parent[cursor];
            self.parent[cursor] = root;
            cursor = next;
        }
        root
    }

    fn union(&mut self, left: usize, right: usize) -> bool {
        let left_root = self.find(left);
        let right_root = self.find(right);
        if left_root == right_root {
            return false;
        }
        let (lower, higher) = if left_root < right_root {
            (left_root, right_root)
        } else {
            (right_root, left_root)
        };
        self.parent[higher] = lower;
        self.components = self.components.saturating_sub(1);
        true
    }

    fn component_count(&self) -> usize {
        self.components
    }
}

fn quantize_nonnegative(value: f64) -> u64 {
    if value.is_finite() && value >= 0.0 {
        (value * 1_000_000.0).round().min(u64::MAX as f64) as u64
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mycelial_junction_graph::{
        MycelialJunctionGraphConfig, build_mycelial_junction_hypha_graph,
    };
    use crate::mycelial_model::{MycelialFilament, MycelialModel, OrbitalCell, PresenceMode, SpatialPoint3};
    use crate::mycelial_network::{AnastomosisKind, MycelialAnastomosis};

    fn filament(
        id: u32,
        parent_id: Option<u32>,
        terminal: bool,
        start: SpatialPoint3,
        control_a: SpatialPoint3,
        control_b: SpatialPoint3,
        end: SpatialPoint3,
    ) -> MycelialFilament {
        MycelialFilament {
            id,
            parent_id,
            generation: u8::from(parent_id.is_some()),
            terminal,
            start,
            control_a,
            control_b,
            end,
            radius_start: 1.0,
            radius_end: 0.5,
        }
    }

    fn fixture_model() -> MycelialModel {
        let branch = SpatialPoint3 {
            x: 0.0,
            y: 10.0,
            z: 0.0,
        };
        MycelialModel {
            schema_version: crate::MYCELIAL_MODEL_VERSION,
            mode: PresenceMode::Ambient,
            activity: None,
            speech_output: None,
            viseme: None,
            gaze: None,
            uncertainty: None,
            filaments: vec![
                filament(
                    1,
                    None,
                    false,
                    SpatialPoint3::ZERO,
                    SpatialPoint3 { x: 0.0, y: 3.0, z: 0.0 },
                    SpatialPoint3 { x: 0.0, y: 7.0, z: 0.0 },
                    branch,
                ),
                filament(
                    2,
                    Some(1),
                    true,
                    branch,
                    SpatialPoint3 { x: 0.0, y: 11.5, z: 0.0 },
                    SpatialPoint3 { x: 0.0, y: 13.5, z: 0.0 },
                    SpatialPoint3 { x: 0.0, y: 15.0, z: 0.0 },
                ),
                filament(
                    3,
                    Some(1),
                    true,
                    branch,
                    SpatialPoint3 { x: 0.0, y: 12.0, z: 0.0 },
                    SpatialPoint3 { x: 0.0, y: 18.0, z: 0.0 },
                    SpatialPoint3 { x: 0.0, y: 20.0, z: 0.0 },
                ),
            ],
            orbital_cells: Vec::<OrbitalCell>::new(),
        }
    }

    #[test]
    fn forest_of_a_tree_matches_observed_graph_cost_and_paths() {
        let graph = build_mycelial_junction_hypha_graph(
            &fixture_model(),
            &[],
            MycelialJunctionGraphConfig::default(),
        )
        .unwrap();
        let roots = vec!["origin:1".to_string()];
        let report =
            build_observed_network_minimum_spanning_forest(&graph, &roots).unwrap();
        let metrics = graph.metrics().unwrap();
        assert_eq!(report.selected_segment_count, 3);
        assert_eq!(report.connected_components, 1);
        assert_eq!(
            report.hypha_arc_cost_model_units_microunits,
            metrics.total_hypha_length_model_units_microunits
        );
        assert_eq!(
            report.global_geometric_efficiency.mean_directness,
            graph.global_geometric_efficiency().unwrap().mean_directness
        );
        assert_eq!(
            report.root_directed_geometric_efficiency.mean_directness,
            graph
                .root_directed_geometric_efficiency(&roots)
                .unwrap()
                .mean_directness
        );
    }

    #[test]
    fn weighted_forest_removes_cycle_edges_and_never_increases_observed_cost() {
        let model = fixture_model();
        let fusion = MycelialAnastomosis {
            id: 17,
            source_filament_id: 2,
            source_t: 1.0,
            target_filament_id: 3,
            target_t: 0.5,
            kind: AnastomosisKind::TipToSide,
            junction_position: SpatialPoint3 { x: 0.0, y: 15.0, z: 0.0 },
        };
        let graph = build_mycelial_junction_hypha_graph(
            &model,
            &[fusion],
            MycelialJunctionGraphConfig::default(),
        )
        .unwrap();
        assert_eq!(graph.metrics().unwrap().cycle_rank, 1);
        let report = build_observed_network_minimum_spanning_forest(
            &graph,
            &["origin:1".to_string()],
        )
        .unwrap();
        assert_eq!(
            report.selected_segment_count,
            graph.junctions.len() - graph.metrics().unwrap().connected_components
        );
        assert_eq!(report.connected_components, 1);
        assert!(
            report.hypha_arc_cost_model_units_microunits
                <= graph.metrics().unwrap().total_hypha_length_model_units_microunits
        );
    }

    #[test]
    fn disconnected_components_remain_disconnected_in_the_spanning_forest() {
        let model = MycelialModel {
            schema_version: crate::MYCELIAL_MODEL_VERSION,
            mode: PresenceMode::Ambient,
            activity: None,
            speech_output: None,
            viseme: None,
            gaze: None,
            uncertainty: None,
            filaments: vec![
                filament(
                    4,
                    None,
                    true,
                    SpatialPoint3::ZERO,
                    SpatialPoint3 { x: 0.0, y: 1.0, z: 0.0 },
                    SpatialPoint3 { x: 0.0, y: 2.0, z: 0.0 },
                    SpatialPoint3 { x: 0.0, y: 3.0, z: 0.0 },
                ),
                filament(
                    5,
                    None,
                    true,
                    SpatialPoint3 { x: 10.0, y: 0.0, z: 0.0 },
                    SpatialPoint3 { x: 10.0, y: 1.0, z: 0.0 },
                    SpatialPoint3 { x: 10.0, y: 2.0, z: 0.0 },
                    SpatialPoint3 { x: 10.0, y: 3.0, z: 0.0 },
                ),
            ],
            orbital_cells: Vec::<OrbitalCell>::new(),
        };
        let graph = build_mycelial_junction_hypha_graph(
            &model,
            &[],
            MycelialJunctionGraphConfig::default(),
        )
        .unwrap();
        let report = build_observed_network_minimum_spanning_forest(
            &graph,
            &["origin:4".to_string(), "origin:5".to_string()],
        )
        .unwrap();
        assert_eq!(report.connected_components, 2);
        assert_eq!(report.selected_segment_count, 2);
        assert_eq!(
            report.global_geometric_efficiency.unreachable_pairs,
            graph
                .global_geometric_efficiency()
                .unwrap()
                .unreachable_pairs
        );
    }
}
