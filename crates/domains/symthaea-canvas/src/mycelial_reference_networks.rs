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
use std::collections::{BTreeSet, BinaryHeap, HashMap};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::mycelial_junction_graph::{
    GeometricEfficiencyReport, JunctionLengthUnit, MycelialJunctionHyphaGraph,
};

pub const MAX_REFERENCE_NETWORK_JUNCTIONS: usize = 2_048;
pub const MYCELIAL_EUCLIDEAN_MST_SCHEMA_VERSION: u16 = 1;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpatialReferenceDimensionality {
    TwoD,
    ThreeD,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialEuclideanMstEdge {
    pub source_junction_id: String,
    pub target_junction_id: String,
    pub euclidean_length_model_units: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialReferenceEfficiencyReport {
    pub eligible_pairs: usize,
    pub reachable_pairs: usize,
    pub unreachable_pairs: usize,
    /// Connected pairs whose direct and network distances are both zero.
    /// They are excluded from directness rather than dividing 0/0.
    pub coincident_zero_path_pairs: usize,
    pub scored_pairs: usize,
    pub mean_directness: Option<f64>,
    pub mean_network_path_length_model_units: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialEuclideanMstReport {
    pub schema_version: u16,
    pub source_graph_digest: String,
    pub dimensionality: SpatialReferenceDimensionality,
    pub length_unit: JunctionLengthUnit,
    /// Complete sorted junction identity set from the source graph.
    pub junction_ids: Vec<String>,
    /// The root identities explicitly supplied by the caller.
    pub declared_root_junction_ids: Vec<String>,
    /// Source graph terminals used for the root-directed efficiency denominator.
    pub terminal_junction_ids: Vec<String>,
    pub selected_edges: Vec<MycelialEuclideanMstEdge>,
    pub selected_edge_count: usize,
    pub connected_components: usize,
    pub total_euclidean_cost_model_units_microunits: u64,
    pub global_geometric_efficiency: SpatialReferenceEfficiencyReport,
    pub root_directed_geometric_efficiency: SpatialReferenceEfficiencyReport,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MycelialReferenceNetworkError {
    InvalidSourceGraph(String),
    TooManyJunctions(usize),
    UnknownRootJunction(String),
    NonPlanarJunction { junction_id: String, z: f32 },
    InvalidReferenceGraph(&'static str),
}

impl fmt::Display for MycelialReferenceNetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSourceGraph(message) => {
                write!(f, "invalid source junction–hypha graph: {message}")
            }
            Self::TooManyJunctions(count) => {
                write!(
                    f,
                    "reference calculation is bounded to {MAX_REFERENCE_NETWORK_JUNCTIONS} junctions, got {count}"
                )
            }
            Self::UnknownRootJunction(id) => {
                write!(f, "unknown declared root junction {id}")
            }
            Self::NonPlanarJunction { junction_id, z } => {
                write!(f, "junction {junction_id} has z={z}, but the Euclidean reference is explicitly 2D")
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
    if selected_segment_count != graph.junctions.len().saturating_sub(components) {
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

/// Build a Euclidean minimum-spanning tree over all junction positions.
///
/// Unlike the observed-network minimum-spanning forest, this reference creates
/// straight-line spatial links that need not correspond to any observed hypha.
/// It is therefore a geometric lower-cost reference, not a physically accepted
/// network. The caller must explicitly choose 2D or 3D distance. A 2D projection
/// fails if any input junction has non-zero z rather than silently flattening
/// a volumetric graph.
pub fn build_euclidean_minimum_spanning_tree(
    graph: &MycelialJunctionHyphaGraph,
    root_junction_ids: &[String],
    dimensionality: SpatialReferenceDimensionality,
) -> Result<MycelialEuclideanMstReport, MycelialReferenceNetworkError> {
    graph
        .validate()
        .map_err(|error| MycelialReferenceNetworkError::InvalidSourceGraph(error.to_string()))?;
    let count = graph.junctions.len();
    if count > MAX_REFERENCE_NETWORK_JUNCTIONS {
        return Err(MycelialReferenceNetworkError::TooManyJunctions(count));
    }
    if dimensionality == SpatialReferenceDimensionality::TwoD {
        for junction in &graph.junctions {
            if junction.position.z.abs() > 1.0e-6 {
                return Err(MycelialReferenceNetworkError::NonPlanarJunction {
                    junction_id: junction.id.clone(),
                    z: junction.position.z,
                });
            }
        }
    }

    let source_graph_digest = graph
        .canonical_digest_hex()
        .map_err(|error| MycelialReferenceNetworkError::InvalidSourceGraph(error.to_string()))?;
    let mut junctions: Vec<_> = graph.junctions.iter().collect();
    junctions.sort_by(|left, right| left.id.cmp(&right.id));

    let mut selected_edges = Vec::with_capacity(count.saturating_sub(1));
    let mut adjacency = vec![Vec::<(usize, f64)>::new(); count];
    let mut total_cost = 0.0f64;
    if count > 0 {
        let mut in_tree = vec![false; count];
        let mut best_distance = vec![f64::INFINITY; count];
        let mut best_parent = vec![None::<usize>; count];
        best_distance[0] = 0.0;

        for _ in 0..count {
            let next = (0..count)
                .filter(|index| !in_tree[*index])
                .min_by(|left, right| {
                    best_distance[*left]
                        .total_cmp(&best_distance[*right])
                        .then_with(|| junctions[*left].id.cmp(&junctions[*right].id))
                })
                .ok_or(MycelialReferenceNetworkError::InvalidReferenceGraph(
                    "Euclidean Prim traversal lost an unvisited junction",
                ))?;
            if !best_distance[next].is_finite() {
                return Err(MycelialReferenceNetworkError::InvalidReferenceGraph(
                    "complete Euclidean candidate graph became disconnected",
                ));
            }
            in_tree[next] = true;
            if let Some(parent) = best_parent[next] {
                let distance = best_distance[next];
                let (source, target) = if junctions[parent].id <= junctions[next].id {
                    (parent, next)
                } else {
                    (next, parent)
                };
                selected_edges.push(MycelialEuclideanMstEdge {
                    source_junction_id: junctions[source].id.clone(),
                    target_junction_id: junctions[target].id.clone(),
                    euclidean_length_model_units: distance,
                });
                adjacency[parent].push((next, distance));
                adjacency[next].push((parent, distance));
                total_cost += distance;
            }
            for candidate in 0..count {
                if in_tree[candidate] {
                    continue;
                }
                let distance = spatial_distance(
                    junctions[next].position,
                    junctions[candidate].position,
                    dimensionality,
                );
                let replace = distance < best_distance[candidate]
                    || (distance == best_distance[candidate]
                        && best_parent[candidate].is_none_or(|current_parent| {
                            junctions[next].id < junctions[current_parent].id
                        }));
                if replace {
                    best_distance[candidate] = distance;
                    best_parent[candidate] = Some(next);
                }
            }
        }
    }
    selected_edges.sort_by(|left, right| {
        left.source_junction_id
            .cmp(&right.source_junction_id)
            .then_with(|| left.target_junction_id.cmp(&right.target_junction_id))
            .then_with(|| {
                left.euclidean_length_model_units
                    .total_cmp(&right.euclidean_length_model_units)
            })
    });
    for neighbours in &mut adjacency {
        neighbours.sort_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.total_cmp(&right.1))
        });
    }

    let connected_components = if count > 0 { 1 } else { 0 };
    if selected_edges.len() != count.saturating_sub(connected_components) {
        return Err(MycelialReferenceNetworkError::InvalidReferenceGraph(
            "Euclidean MST edge count is not |V|-1",
        ));
    }
    let global_geometric_efficiency =
        euclidean_global_efficiency(&junctions, &adjacency, dimensionality)?;
    let root_directed_geometric_efficiency =
        euclidean_root_efficiency(&junctions, &adjacency, root_junction_ids, dimensionality)?;

    let mut declared_root_junction_ids = root_junction_ids.to_vec();
    declared_root_junction_ids.sort_unstable();
    declared_root_junction_ids.dedup();
    let mut terminal_junction_ids: Vec<_> = junctions
        .iter()
        .filter(|junction| !junction.terminal_filament_ids.is_empty())
        .map(|junction| junction.id.clone())
        .collect();
    terminal_junction_ids.sort_unstable();
    let report = MycelialEuclideanMstReport {
        schema_version: MYCELIAL_EUCLIDEAN_MST_SCHEMA_VERSION,
        source_graph_digest,
        dimensionality,
        length_unit: graph.length_unit,
        junction_ids: junctions.iter().map(|junction| junction.id.clone()).collect(),
        declared_root_junction_ids,
        terminal_junction_ids,
        selected_edge_count: selected_edges.len(),
        selected_edges,
        connected_components,
        total_euclidean_cost_model_units_microunits: quantize_nonnegative(total_cost),
        global_geometric_efficiency,
        root_directed_geometric_efficiency,
    };
    report
        .validate()
        .map_err(|_| MycelialReferenceNetworkError::InvalidReferenceGraph(
            "Euclidean MST receipt failed its internal validation",
        ))?;
    Ok(report)
}

impl MycelialEuclideanMstReport {
    pub fn validate(&self) -> Result<(), MycelialReferenceNetworkError> {
        let valid_id = |value: &str| {
            !value.is_empty()
                && value.len() <= 128
                && !value.chars().any(char::is_control)
        };
        let valid_digest = self.source_graph_digest.len() == 64
            && self
                .source_graph_digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
        let junction_set: BTreeSet<_> = self.junction_ids.iter().map(String::as_str).collect();
        let roots_valid = self.declared_root_junction_ids.windows(2).all(|pair| {
            pair[0] < pair[1]
        }) && self
            .declared_root_junction_ids
            .iter()
            .all(|id| junction_set.contains(id.as_str()));
        let terminals_valid = self.terminal_junction_ids.windows(2).all(|pair| {
            pair[0] < pair[1]
        }) && self
            .terminal_junction_ids
            .iter()
            .all(|id| junction_set.contains(id.as_str()));
        let mut edge_set = BTreeSet::new();
        let edges_valid = self.selected_edges.iter().all(|edge| {
            valid_id(&edge.source_junction_id)
                && valid_id(&edge.target_junction_id)
                && edge.source_junction_id < edge.target_junction_id
                && junction_set.contains(edge.source_junction_id.as_str())
                && junction_set.contains(edge.target_junction_id.as_str())
                && edge.euclidean_length_model_units.is_finite()
                && edge.euclidean_length_model_units >= 0.0
                && edge_set.insert((
                    edge.source_junction_id.as_str(),
                    edge.target_junction_id.as_str(),
                ))
        }) && self.selected_edges.windows(2).all(|pair| {
            (
                pair[0].source_junction_id.as_str(),
                pair[0].target_junction_id.as_str(),
            ) <= (
                pair[1].source_junction_id.as_str(),
                pair[1].target_junction_id.as_str(),
            )
        });
        let expected_components = if self.junction_ids.is_empty() { 0 } else { 1 };
        let expected_edges = self.junction_ids.len().saturating_sub(expected_components);
        let expected_cost = quantize_nonnegative(
            self.selected_edges
                .iter()
                .map(|edge| edge.euclidean_length_model_units)
                .sum(),
        );
        let global = &self.global_geometric_efficiency;
        let expected_global_pairs = self
            .junction_ids
            .len()
            .saturating_mul(self.junction_ids.len().saturating_sub(1))
            / 2;
        let root = &self.root_directed_geometric_efficiency;
        let global_valid = global.eligible_pairs == expected_global_pairs
            && global.reachable_pairs == expected_global_pairs
            && global.unreachable_pairs == 0
            && global.scored_pairs + global.coincident_zero_path_pairs
                == global.reachable_pairs
            && valid_efficiency_values(global);
        let expected_root_reachable = if self.declared_root_junction_ids.is_empty() {
            0
        } else {
            self.terminal_junction_ids.len()
        };
        let root_valid = root.eligible_pairs == self.terminal_junction_ids.len()
            && root.reachable_pairs == expected_root_reachable
            && root.unreachable_pairs == root.eligible_pairs.saturating_sub(root.reachable_pairs)
            && root.scored_pairs + root.coincident_zero_path_pairs == root.reachable_pairs
            && valid_efficiency_values(root);
        if self.schema_version != MYCELIAL_EUCLIDEAN_MST_SCHEMA_VERSION
            || !valid_digest
            || self.junction_ids.len() > MAX_REFERENCE_NETWORK_JUNCTIONS
            || self.junction_ids.windows(2).any(|pair| pair[0] >= pair[1])
            || self.junction_ids.iter().any(|id| !valid_id(id))
            || !roots_valid
            || !terminals_valid
            || !edges_valid
            || self.selected_edge_count != self.selected_edges.len()
            || self.selected_edge_count != expected_edges
            || self.connected_components != expected_components
            || self.total_euclidean_cost_model_units_microunits != expected_cost
            || !global_valid
            || !root_valid
        {
            return Err(MycelialReferenceNetworkError::InvalidReferenceGraph(
                "Euclidean MST receipt is internally inconsistent",
            ));
        }
        Ok(())
    }

    pub fn canonical_digest_hex(
        &self,
    ) -> Result<String, MycelialReferenceNetworkError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-euclidean-mst-reference-v1");
        hasher.update(&self.schema_version.to_le_bytes());
        hasher.update(self.source_graph_digest.as_bytes());
        hasher.update(&[match self.dimensionality {
            SpatialReferenceDimensionality::TwoD => 0,
            SpatialReferenceDimensionality::ThreeD => 1,
        }]);
        hasher.update(&[match self.length_unit {
            JunctionLengthUnit::OrganismLocalModelUnit => 0,
        }]);
        hash_string_list(&mut hasher, &self.junction_ids);
        hash_string_list(&mut hasher, &self.declared_root_junction_ids);
        hash_string_list(&mut hasher, &self.terminal_junction_ids);
        for edge in &self.selected_edges {
            hash_str(&mut hasher, &edge.source_junction_id);
            hash_str(&mut hasher, &edge.target_junction_id);
            hasher.update(&edge.euclidean_length_model_units.to_bits().to_le_bytes());
        }
        hasher.update(&(self.selected_edge_count as u64).to_le_bytes());
        hasher.update(&(self.connected_components as u64).to_le_bytes());
        hasher.update(&self.total_euclidean_cost_model_units_microunits.to_le_bytes());
        hash_efficiency(&mut hasher, &self.global_geometric_efficiency);
        hash_efficiency(&mut hasher, &self.root_directed_geometric_efficiency);
        Ok(hex_digest(hasher.finalize().as_bytes()))
    }
}

fn valid_efficiency_values(report: &SpatialReferenceEfficiencyReport) -> bool {
    let mean_values_valid = report.mean_directness.is_none_or(|value| {
        value.is_finite() && (0.0..=1.0 + 1.0e-9).contains(&value)
    }) && report
        .mean_network_path_length_model_units
        .is_none_or(|value| value.is_finite() && value > 0.0);
    let scored_values_present = if report.scored_pairs == 0 {
        report.mean_directness.is_none()
            && report.mean_network_path_length_model_units.is_none()
    } else {
        report.mean_directness.is_some()
            && report.mean_network_path_length_model_units.is_some()
    };
    report.reachable_pairs <= report.eligible_pairs
        && report.unreachable_pairs
            == report.eligible_pairs.saturating_sub(report.reachable_pairs)
        && report.scored_pairs + report.coincident_zero_path_pairs
            == report.reachable_pairs
        && mean_values_valid
        && scored_values_present
}

fn hash_string_list(hasher: &mut blake3::Hasher, values: &[String]) {
    hasher.update(&(values.len() as u64).to_le_bytes());
    for value in values {
        hash_str(hasher, value);
    }
}

fn hash_str(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hash_efficiency(hasher: &mut blake3::Hasher, report: &SpatialReferenceEfficiencyReport) {
    for value in [
        report.eligible_pairs,
        report.reachable_pairs,
        report.unreachable_pairs,
        report.coincident_zero_path_pairs,
        report.scored_pairs,
    ] {
        hasher.update(&(value as u64).to_le_bytes());
    }
    hash_optional_f64(hasher, report.mean_directness);
    hash_optional_f64(hasher, report.mean_network_path_length_model_units);
}

fn hash_optional_f64(hasher: &mut blake3::Hasher, value: Option<f64>) {
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

fn hex_digest(bytes: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes {
        use fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn spatial_distance(
    left: crate::mycelial_model::SpatialPoint3,
    right: crate::mycelial_model::SpatialPoint3,
    dimensionality: SpatialReferenceDimensionality,
) -> f64 {
    let dx = f64::from(left.x) - f64::from(right.x);
    let dy = f64::from(left.y) - f64::from(right.y);
    let mut squared = dx * dx + dy * dy;
    if dimensionality == SpatialReferenceDimensionality::ThreeD {
        let dz = f64::from(left.z) - f64::from(right.z);
        squared += dz * dz;
    }
    squared.sqrt()
}

fn euclidean_global_efficiency(
    junctions: &[&crate::mycelial_junction_graph::MycelialJunction],
    adjacency: &[Vec<(usize, f64)>],
    dimensionality: SpatialReferenceDimensionality,
) -> Result<SpatialReferenceEfficiencyReport, MycelialReferenceNetworkError> {
    let count = junctions.len();
    let eligible_pairs = count.saturating_mul(count.saturating_sub(1)) / 2;
    let mut sum_directness = 0.0f64;
    let mut sum_path_length = 0.0f64;
    let mut reachable_pairs = 0usize;
    let mut coincident_zero_path_pairs = 0usize;
    let mut scored_pairs = 0usize;
    for source in 0..count {
        let distances = shortest_path_distances(adjacency, source);
        for target in (source + 1)..count {
            let path_length = distances[target];
            if !path_length.is_finite() {
                continue;
            }
            reachable_pairs += 1;
            let direct = spatial_distance(
                junctions[source].position,
                junctions[target].position,
                dimensionality,
            );
            if path_length <= 1.0e-12 && direct <= 1.0e-12 {
                coincident_zero_path_pairs += 1;
                continue;
            }
            if path_length <= 0.0 || direct > path_length + 1.0e-6 {
                return Err(MycelialReferenceNetworkError::InvalidReferenceGraph(
                    "Euclidean distance exceeds Euclidean-MST path length",
                ));
            }
            sum_directness += direct / path_length;
            sum_path_length += path_length;
            scored_pairs += 1;
        }
    }
    Ok(SpatialReferenceEfficiencyReport {
        eligible_pairs,
        reachable_pairs,
        unreachable_pairs: eligible_pairs.saturating_sub(reachable_pairs),
        coincident_zero_path_pairs,
        scored_pairs,
        mean_directness: (scored_pairs > 0).then_some(sum_directness / scored_pairs as f64),
        mean_network_path_length_model_units: (scored_pairs > 0)
            .then_some(sum_path_length / scored_pairs as f64),
    })
}

fn euclidean_root_efficiency(
    junctions: &[&crate::mycelial_junction_graph::MycelialJunction],
    adjacency: &[Vec<(usize, f64)>],
    root_junction_ids: &[String],
    dimensionality: SpatialReferenceDimensionality,
) -> Result<SpatialReferenceEfficiencyReport, MycelialReferenceNetworkError> {
    let index_by_id: HashMap<_, _> = junctions
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
    let terminals: Vec<_> = junctions
        .iter()
        .enumerate()
        .filter(|(_, junction)| !junction.terminal_filament_ids.is_empty())
        .map(|(index, _)| index)
        .collect();
    let mut sum_directness = 0.0f64;
    let mut sum_path_length = 0.0f64;
    let mut reachable_pairs = 0usize;
    let mut coincident_zero_path_pairs = 0usize;
    let mut scored_pairs = 0usize;
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
                    .then_with(|| junctions[left.0].id.cmp(&junctions[right.0].id))
            });
        if let Some((root, path_length)) = nearest {
            reachable_pairs += 1;
            let direct = spatial_distance(
                junctions[*terminal].position,
                junctions[root].position,
                dimensionality,
            );
            if path_length <= 1.0e-12 && direct <= 1.0e-12 {
                coincident_zero_path_pairs += 1;
                continue;
            }
            if path_length <= 0.0 || direct > path_length + 1.0e-6 {
                return Err(MycelialReferenceNetworkError::InvalidReferenceGraph(
                    "direct tip-to-root distance exceeds Euclidean-MST path length",
                ));
            }
            sum_directness += direct / path_length;
            sum_path_length += path_length;
            scored_pairs += 1;
        }
    }
    let eligible_pairs = terminals.len();
    Ok(SpatialReferenceEfficiencyReport {
        eligible_pairs,
        reachable_pairs,
        unreachable_pairs: eligible_pairs.saturating_sub(reachable_pairs),
        coincident_zero_path_pairs,
        scored_pairs,
        mean_directness: (scored_pairs > 0).then_some(sum_directness / scored_pairs as f64),
        mean_network_path_length_model_units: (scored_pairs > 0)
            .then_some(sum_path_length / scored_pairs as f64),
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
        mean_directness: (reachable_pairs > 0).then_some(sum_directness / reachable_pairs as f64),
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
            .ok_or_else(|| MycelialReferenceNetworkError::UnknownRootJunction(root_id.clone()))?;
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
        mean_directness: (reachable_pairs > 0).then_some(sum_directness / reachable_pairs as f64),
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
    use crate::mycelial_model::{
        MycelialFilament, MycelialModel, OrbitalCell, PresenceMode, SpatialPoint3,
    };
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
                    SpatialPoint3 {
                        x: 0.0,
                        y: 3.0,
                        z: 0.0,
                    },
                    SpatialPoint3 {
                        x: 0.0,
                        y: 7.0,
                        z: 0.0,
                    },
                    branch,
                ),
                filament(
                    2,
                    Some(1),
                    true,
                    branch,
                    SpatialPoint3 {
                        x: 0.0,
                        y: 11.5,
                        z: 0.0,
                    },
                    SpatialPoint3 {
                        x: 0.0,
                        y: 13.5,
                        z: 0.0,
                    },
                    SpatialPoint3 {
                        x: 0.0,
                        y: 15.0,
                        z: 0.0,
                    },
                ),
                filament(
                    3,
                    Some(1),
                    true,
                    branch,
                    SpatialPoint3 {
                        x: 0.0,
                        y: 12.0,
                        z: 0.0,
                    },
                    SpatialPoint3 {
                        x: 0.0,
                        y: 18.0,
                        z: 0.0,
                    },
                    SpatialPoint3 {
                        x: 0.0,
                        y: 20.0,
                        z: 0.0,
                    },
                ),
            ],
            orbital_cells: Vec::<OrbitalCell>::new(),
        }
    }

    fn brute_force_minimum_forest_traversal_cost(graph: &MycelialJunctionHyphaGraph) -> f64 {
        let components = graph.metrics().unwrap().connected_components;
        let required_edges = graph.junctions.len().saturating_sub(components);
        let edge_count = graph.segments.len();
        assert!(edge_count < usize::BITS as usize);
        let index_by_id: HashMap<_, _> = graph
            .junctions
            .iter()
            .enumerate()
            .map(|(index, junction)| (junction.id.as_str(), index))
            .collect();
        let mut best = f64::INFINITY;
        for mask in 0usize..(1usize << edge_count) {
            if mask.count_ones() as usize != required_edges {
                continue;
            }
            let mut parent: Vec<_> = (0..graph.junctions.len()).collect();
            let mut component_count = graph.junctions.len();
            let mut cost = 0.0;
            let mut is_forest = true;
            for (edge_index, segment) in graph.segments.iter().enumerate() {
                if mask & (1usize << edge_index) == 0 {
                    continue;
                }
                let mut source = index_by_id[segment.start_junction_id.as_str()];
                while parent[source] != source {
                    source = parent[source];
                }
                let mut target = index_by_id[segment.end_junction_id.as_str()];
                while parent[target] != target {
                    target = parent[target];
                }
                if source == target {
                    is_forest = false;
                    break;
                }
                let (lower, higher) = if source < target {
                    (source, target)
                } else {
                    (target, source)
                };
                parent[higher] = lower;
                component_count -= 1;
                cost += segment.traversal_length_model_units();
            }
            if is_forest && component_count == components {
                best = best.min(cost);
            }
        }
        best
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
        let report = build_observed_network_minimum_spanning_forest(&graph, &roots).unwrap();
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
            junction_position: SpatialPoint3 {
                x: 0.0,
                y: 15.0,
                z: 0.0,
            },
        };
        let graph = build_mycelial_junction_hypha_graph(
            &model,
            &[fusion],
            MycelialJunctionGraphConfig::default(),
        )
        .unwrap();
        assert_eq!(graph.metrics().unwrap().cycle_rank, 1);
        let report =
            build_observed_network_minimum_spanning_forest(&graph, &["origin:1".to_string()])
                .unwrap();
        assert_eq!(
            report.selected_segment_count,
            graph.junctions.len() - graph.metrics().unwrap().connected_components
        );
        assert_eq!(report.connected_components, 1);
        assert!(
            report.hypha_arc_cost_model_units_microunits
                <= graph
                    .metrics()
                    .unwrap()
                    .total_hypha_length_model_units_microunits
        );
    }

    #[test]
    fn kruskal_cost_matches_exhaustive_small_graph_oracle() {
        let model = fixture_model();
        let fusion = MycelialAnastomosis {
            id: 17,
            source_filament_id: 2,
            source_t: 1.0,
            target_filament_id: 3,
            target_t: 0.5,
            kind: AnastomosisKind::TipToSide,
            junction_position: SpatialPoint3 {
                x: 0.0,
                y: 15.0,
                z: 0.0,
            },
        };
        let graph = build_mycelial_junction_hypha_graph(
            &model,
            &[fusion],
            MycelialJunctionGraphConfig::default(),
        )
        .unwrap();
        let report =
            build_observed_network_minimum_spanning_forest(&graph, &["origin:1".to_string()])
                .unwrap();
        let independent_cost = brute_force_minimum_forest_traversal_cost(&graph);
        let report_cost = report.traversal_cost_model_units_microunits as f64 / 1_000_000.0;
        assert!(
            (report_cost - independent_cost).abs() <= 1.0e-6,
            "Kruskal cost {report_cost} differs from exhaustive optimum {independent_cost}"
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
                    SpatialPoint3 {
                        x: 0.0,
                        y: 1.0,
                        z: 0.0,
                    },
                    SpatialPoint3 {
                        x: 0.0,
                        y: 2.0,
                        z: 0.0,
                    },
                    SpatialPoint3 {
                        x: 0.0,
                        y: 3.0,
                        z: 0.0,
                    },
                ),
                filament(
                    5,
                    None,
                    true,
                    SpatialPoint3 {
                        x: 10.0,
                        y: 0.0,
                        z: 0.0,
                    },
                    SpatialPoint3 {
                        x: 10.0,
                        y: 1.0,
                        z: 0.0,
                    },
                    SpatialPoint3 {
                        x: 10.0,
                        y: 2.0,
                        z: 0.0,
                    },
                    SpatialPoint3 {
                        x: 10.0,
                        y: 3.0,
                        z: 0.0,
                    },
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
    fn brute_force_euclidean_tree_cost(
        graph: &MycelialJunctionHyphaGraph,
        dimensionality: SpatialReferenceDimensionality,
    ) -> f64 {
        let mut junctions: Vec<_> = graph.junctions.iter().collect();
        junctions.sort_by(|left, right| left.id.cmp(&right.id));
        let count = junctions.len();
        if count <= 1 {
            return 0.0;
        }
        let mut edges = Vec::new();
        for source in 0..count {
            for target in (source + 1)..count {
                edges.push((
                    source,
                    target,
                    spatial_distance(
                        junctions[source].position,
                        junctions[target].position,
                        dimensionality,
                    ),
                ));
            }
        }
        assert!(edges.len() < usize::BITS as usize);
        let mut best = f64::INFINITY;
        for mask in 0usize..(1usize << edges.len()) {
            if mask.count_ones() as usize != count - 1 {
                continue;
            }
            let mut parent: Vec<_> = (0..count).collect();
            let mut components = count;
            let mut cost = 0.0;
            let mut is_tree = true;
            for (edge_index, (source, target, weight)) in edges.iter().enumerate() {
                if mask & (1usize << edge_index) == 0 {
                    continue;
                }
                let mut left = *source;
                while parent[left] != left {
                    left = parent[left];
                }
                let mut right = *target;
                while parent[right] != right {
                    right = parent[right];
                }
                if left == right {
                    is_tree = false;
                    break;
                }
                let (lower, higher) = if left < right {
                    (left, right)
                } else {
                    (right, left)
                };
                parent[higher] = lower;
                components -= 1;
                cost += weight;
            }
            if is_tree && components == 1 {
                best = best.min(cost);
            }
        }
        best
    }

    #[test]
    fn euclidean_mst_matches_exhaustive_point_set_oracle() {
        let graph = build_mycelial_junction_hypha_graph(
            &fixture_model(),
            &[],
            MycelialJunctionGraphConfig::default(),
        )
        .unwrap();
        let report = build_euclidean_minimum_spanning_tree(
            &graph,
            &["origin:1".to_string()],
            SpatialReferenceDimensionality::ThreeD,
        )
        .unwrap();
        let expected = brute_force_euclidean_tree_cost(
            &graph,
            SpatialReferenceDimensionality::ThreeD,
        );
        let actual = report.total_euclidean_cost_model_units_microunits as f64 / 1_000_000.0;
        assert_eq!(report.selected_edge_count, graph.junctions.len() - 1);
        assert_eq!(report.connected_components, 1);
        assert!((actual - expected).abs() <= 1.0e-6);
    }

    #[test]
    fn euclidean_mst_is_permutation_invariant_and_uses_explicit_2d_metric() {
        let graph = build_mycelial_junction_hypha_graph(
            &fixture_model(),
            &[],
            MycelialJunctionGraphConfig::default(),
        )
        .unwrap();
        let roots = vec!["origin:1".to_string()];
        let first = build_euclidean_minimum_spanning_tree(
            &graph,
            &roots,
            SpatialReferenceDimensionality::TwoD,
        )
        .unwrap();
        let mut reordered = graph.clone();
        reordered.junctions.reverse();
        reordered.segments.reverse();
        let second = build_euclidean_minimum_spanning_tree(
            &reordered,
            &roots,
            SpatialReferenceDimensionality::TwoD,
        )
        .unwrap();
        assert_eq!(first.selected_edges, second.selected_edges);
        assert_eq!(
            first.total_euclidean_cost_model_units_microunits,
            second.total_euclidean_cost_model_units_microunits
        );
        assert_eq!(
            first.global_geometric_efficiency.mean_directness,
            second.global_geometric_efficiency.mean_directness
        );
        assert_eq!(
            first.root_directed_geometric_efficiency.mean_directness,
            second.root_directed_geometric_efficiency.mean_directness
        );
    }

    #[test]
    fn euclidean_mst_rejects_implicit_flattening_of_nonplanar_graphs() {
        let model = MycelialModel {
            schema_version: crate::MYCELIAL_MODEL_VERSION,
            mode: PresenceMode::Ambient,
            activity: None,
            speech_output: None,
            viseme: None,
            gaze: None,
            uncertainty: None,
            filaments: vec![filament(
                99,
                None,
                true,
                SpatialPoint3 { x: 0.0, y: 0.0, z: 0.5 },
                SpatialPoint3 { x: 1.0, y: 1.0, z: 0.5 },
                SpatialPoint3 { x: 2.0, y: 2.0, z: 0.5 },
                SpatialPoint3 { x: 3.0, y: 3.0, z: 0.5 },
            )],
            orbital_cells: Vec::new(),
        };
        let graph = build_mycelial_junction_hypha_graph(
            &model,
            &[],
            MycelialJunctionGraphConfig::default(),
        )
        .unwrap();
        assert!(matches!(
            build_euclidean_minimum_spanning_tree(
                &graph,
                &[],
                SpatialReferenceDimensionality::TwoD
            ),
            Err(MycelialReferenceNetworkError::NonPlanarJunction { .. })
        ));
    }

    #[test]
    fn euclidean_mst_reports_coincident_zero_path_pairs_separately() {
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
                    SpatialPoint3::ZERO,
                    SpatialPoint3 { x: 0.0, y: 1.0, z: 0.0 },
                    SpatialPoint3 { x: 0.0, y: 2.0, z: 0.0 },
                    SpatialPoint3 { x: 0.0, y: 3.0, z: 0.0 },
                ),
            ],
            orbital_cells: Vec::new(),
        };
        let graph = build_mycelial_junction_hypha_graph(
            &model,
            &[],
            MycelialJunctionGraphConfig::default(),
        )
        .unwrap();
        let report = build_euclidean_minimum_spanning_tree(
            &graph,
            &[],
            SpatialReferenceDimensionality::ThreeD,
        )
        .unwrap();
        assert_eq!(report.selected_edge_count, graph.junctions.len() - 1);
        assert!(report.global_geometric_efficiency.coincident_zero_path_pairs > 0);
        assert_eq!(
            report.global_geometric_efficiency.reachable_pairs,
            report.global_geometric_efficiency.eligible_pairs
        );
        assert_eq!(
            report.global_geometric_efficiency.scored_pairs
                + report.global_geometric_efficiency.coincident_zero_path_pairs,
            report.global_geometric_efficiency.reachable_pairs
        );
    }

}
