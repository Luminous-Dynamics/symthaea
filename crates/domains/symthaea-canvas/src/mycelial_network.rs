// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Graph-level network view for branching and hyphal anastomosis experiments.
//!
//! MycelialModel v1 is an immutable branching-tree geometry baseline. This
//! module derives a separate network graph and accepts explicit fusion records.
//! It never edits the source model or CognitiveTopology. A geometric crossing
//! is not a connection unless a validated anastomosis record is supplied.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::mycelial_model::{MYCELIAL_MODEL_VERSION, MycelialModel, SpatialPoint3};

pub const MYCELIAL_NETWORK_SCHEMA_VERSION: u16 = 1;
pub const MAX_MYCELIAL_ANASTOMOSES: usize = 512;
pub const DEFAULT_MAX_FUSION_DISTANCE: f32 = 5.0;
const TIP_PARAMETER_EPSILON: f32 = 1.0e-4;
const INTERIOR_PARAMETER_MARGIN: f32 = 0.05;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnastomosisKind {
    TipToTip,
    TipToSide,
}

/// Explicit geometric fusion candidate. source_t must identify a source tip;
/// target_t distinguishes target-tip from target-side contact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialAnastomosis {
    pub id: u32,
    pub source_filament_id: u32,
    pub source_t: f32,
    pub target_filament_id: u32,
    pub target_t: f32,
    pub kind: AnastomosisKind,
    pub junction_position: SpatialPoint3,
}

/// A proximity-based geometric suggestion. It is not an accepted connection
/// until a caller explicitly supplies its fusion record to the graph builder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialAnastomosisCandidate {
    pub fusion: MycelialAnastomosis,
    pub contact_distance: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialConnectionKind {
    ParentBranch,
    Anastomosis,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelialNetworkConnection {
    pub source_filament_id: u32,
    pub target_filament_id: u32,
    pub kind: MycelialConnectionKind,
    /// Present only for anastomosis connections.
    pub anastomosis_id: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MycelialNetworkConfig {
    pub max_fusion_distance: f32,
}

impl Default for MycelialNetworkConfig {
    fn default() -> Self {
        Self {
            max_fusion_distance: DEFAULT_MAX_FUSION_DISTANCE,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelialNetworkMetrics {
    pub filament_count: usize,
    pub parent_branch_count: usize,
    pub anastomosis_count: usize,
    pub connection_count: usize,
    pub connected_components: usize,
    pub cycle_rank: usize,
    pub maximum_degree: usize,
    /// Number of filament vertices whose removal increases component count.
    pub articulation_filament_count: usize,
    /// Worst connected-component count after removing any single filament.
    pub max_components_after_single_filament_failure: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialNetworkGraph {
    pub schema_version: u16,
    pub source_model_version: u16,
    pub filament_ids: Vec<u32>,
    pub connections: Vec<MycelialNetworkConnection>,
    pub anastomoses: Vec<MycelialAnastomosis>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MycelialNetworkError {
    UnsupportedModelVersion(u16),
    InvalidConfig,
    FilamentCountExceeded(usize),
    AnastomosisCountExceeded(usize),
    DuplicateFilamentId(u32),
    MissingParent {
        filament_id: u32,
        parent_id: u32,
    },
    DuplicateAnastomosisId(u32),
    MissingSourceFilament(u32),
    MissingTargetFilament(u32),
    SelfFusion(u32),
    SourceIsNotTerminalTip(u32),
    TargetIsNotTerminalTip(u32),
    InvalidFusionParameter(u32),
    InvalidJunctionPosition(u32),
    FusionTooFar {
        anastomosis_id: u32,
        distance: f32,
        limit: f32,
    },
}

impl fmt::Display for MycelialNetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedModelVersion(version) => {
                write!(f, "unsupported source MycelialModel version {version}")
            }
            Self::InvalidConfig => {
                write!(f, "maximum fusion distance must be finite and in (0, 100]")
            }
            Self::FilamentCountExceeded(count) => {
                write!(f, "filament count {count} exceeds network limit")
            }
            Self::AnastomosisCountExceeded(count) => {
                write!(
                    f,
                    "anastomosis count {count} exceeds limit {MAX_MYCELIAL_ANASTOMOSES}"
                )
            }
            Self::DuplicateFilamentId(id) => write!(f, "duplicate filament ID {id}"),
            Self::MissingParent {
                filament_id,
                parent_id,
            } => {
                write!(
                    f,
                    "filament {filament_id} references missing parent {parent_id}"
                )
            }
            Self::DuplicateAnastomosisId(id) => {
                write!(f, "duplicate anastomosis ID {id}")
            }
            Self::MissingSourceFilament(id) => {
                write!(f, "anastomosis source filament {id} does not exist")
            }
            Self::MissingTargetFilament(id) => {
                write!(f, "anastomosis target filament {id} does not exist")
            }
            Self::SelfFusion(id) => write!(f, "anastomosis {id} connects a filament to itself"),
            Self::SourceIsNotTerminalTip(id) => {
                write!(f, "anastomosis {id} source is not a terminal hyphal tip")
            }
            Self::TargetIsNotTerminalTip(id) => {
                write!(f, "anastomosis {id} target is not a terminal hyphal tip")
            }
            Self::InvalidFusionParameter(id) => {
                write!(
                    f,
                    "anastomosis {id} has invalid parameter values for its fusion kind"
                )
            }
            Self::InvalidJunctionPosition(id) => {
                write!(f, "anastomosis {id} has a non-finite junction position")
            }
            Self::FusionTooFar {
                anastomosis_id,
                distance,
                limit,
            } => {
                write!(
                    f,
                    "anastomosis {anastomosis_id} contact distance {distance:.3} exceeds limit {limit:.3}"
                )
            }
        }
    }
}

impl Error for MycelialNetworkError {}

/// Detect possible tip-to-tip and tip-to-side contacts with a deterministic,
/// bounded closest-point search. Candidates never alter the model or graph;
/// the caller must explicitly approve them before graph construction.
pub fn detect_anastomosis_candidates(
    model: &MycelialModel,
    config: MycelialNetworkConfig,
) -> Result<Vec<MycelialAnastomosisCandidate>, MycelialNetworkError> {
    // Reuse the canonical model/topology checks before running geometric search.
    build_mycelial_network_graph(model, &[], config)?;

    let mut filaments: Vec<_> = model.filaments.iter().collect();
    filaments.sort_by_key(|filament| filament.id);
    let mut candidates = Vec::new();

    for source in &filaments {
        if !source.terminal {
            continue;
        }
        let source_tip = source.end;

        for target in &filaments {
            if source.id == target.id
                || source.parent_id == Some(target.id)
                || target.parent_id == Some(source.id)
            {
                continue;
            }

            if target.terminal && source.id < target.id {
                let tip_distance = source_tip.sub(target.end).length();
                if tip_distance.is_finite() && tip_distance <= config.max_fusion_distance {
                    let junction = source_tip.add(target.end).scale(0.5);
                    candidates.push(MycelialAnastomosisCandidate {
                        fusion: MycelialAnastomosis {
                            id: 0,
                            source_filament_id: source.id,
                            source_t: 1.0,
                            target_filament_id: target.id,
                            target_t: 1.0,
                            kind: AnastomosisKind::TipToTip,
                            junction_position: junction,
                        },
                        contact_distance: tip_distance,
                    });
                }
            }

            if !bounds_may_contact(source_tip, target, config.max_fusion_distance) {
                continue;
            }
            let (target_t, target_point, distance) = nearest_bezier_parameter(target, source_tip);
            if !distance.is_finite() || distance > config.max_fusion_distance {
                continue;
            }
            if !(INTERIOR_PARAMETER_MARGIN..=(1.0 - INTERIOR_PARAMETER_MARGIN)).contains(&target_t)
            {
                continue;
            }

            let junction = source_tip.add(target_point).scale(0.5);
            candidates.push(MycelialAnastomosisCandidate {
                fusion: MycelialAnastomosis {
                    id: 0,
                    source_filament_id: source.id,
                    source_t: 1.0,
                    target_filament_id: target.id,
                    target_t,
                    kind: AnastomosisKind::TipToSide,
                    junction_position: junction,
                },
                contact_distance: distance,
            });
        }
    }

    candidates.sort_by(|left, right| {
        (
            left.fusion.source_filament_id,
            left.fusion.target_filament_id,
            left.fusion.kind,
        )
            .cmp(&(
                right.fusion.source_filament_id,
                right.fusion.target_filament_id,
                right.fusion.kind,
            ))
            .then_with(|| left.fusion.target_t.total_cmp(&right.fusion.target_t))
    });
    candidates.dedup_by(|left, right| {
        left.fusion.source_filament_id == right.fusion.source_filament_id
            && left.fusion.target_filament_id == right.fusion.target_filament_id
            && left.fusion.kind == right.fusion.kind
            && (left.fusion.target_t - right.fusion.target_t).abs() < 1.0e-4
    });
    if candidates.len() > MAX_MYCELIAL_ANASTOMOSES {
        return Err(MycelialNetworkError::AnastomosisCountExceeded(
            candidates.len(),
        ));
    }
    for (index, candidate) in candidates.iter_mut().enumerate() {
        candidate.fusion.id = index as u32;
    }
    Ok(candidates)
}

fn bounds_may_contact(
    point: SpatialPoint3,
    filament: &crate::mycelial_model::MycelialFilament,
    maximum_distance: f32,
) -> bool {
    let xs = [
        filament.start.x,
        filament.control_a.x,
        filament.control_b.x,
        filament.end.x,
    ];
    let ys = [
        filament.start.y,
        filament.control_a.y,
        filament.control_b.y,
        filament.end.y,
    ];
    let zs = [
        filament.start.z,
        filament.control_a.z,
        filament.control_b.z,
        filament.end.z,
    ];
    let distance_to_interval = |value: f32, low: f32, high: f32| {
        if value < low {
            low - value
        } else if value > high {
            value - high
        } else {
            0.0
        }
    };
    let dx = distance_to_interval(
        point.x,
        xs.into_iter().fold(f32::INFINITY, f32::min) - maximum_distance,
        xs.into_iter().fold(f32::NEG_INFINITY, f32::max) + maximum_distance,
    );
    let dy = distance_to_interval(
        point.y,
        ys.into_iter().fold(f32::INFINITY, f32::min) - maximum_distance,
        ys.into_iter().fold(f32::NEG_INFINITY, f32::max) + maximum_distance,
    );
    let dz = distance_to_interval(
        point.z,
        zs.into_iter().fold(f32::INFINITY, f32::min) - maximum_distance,
        zs.into_iter().fold(f32::NEG_INFINITY, f32::max) + maximum_distance,
    );
    dx * dx + dy * dy + dz * dz <= maximum_distance * maximum_distance
}

fn nearest_bezier_parameter(
    filament: &crate::mycelial_model::MycelialFilament,
    query: SpatialPoint3,
) -> (f32, SpatialPoint3, f32) {
    let low = INTERIOR_PARAMETER_MARGIN;
    let high = 1.0 - INTERIOR_PARAMETER_MARGIN;
    let mut best_t = low;
    let mut best_point = bezier_point(filament, low);
    let mut best_distance = best_point.sub(query).length();

    // Fixed coarse scan avoids unpredictable convergence and brackets the
    // deterministic local refinement used for each fixture/device.
    const COARSE_STEPS: usize = 48;
    for index in 1..=COARSE_STEPS {
        let t = low + (high - low) * index as f32 / COARSE_STEPS as f32;
        let point = bezier_point(filament, t);
        let distance = point.sub(query).length();
        if distance < best_distance {
            best_t = t;
            best_point = point;
            best_distance = distance;
        }
    }

    let mut step = (high - low) / COARSE_STEPS as f32;
    for _ in 0..20 {
        for candidate_t in [(best_t - step).max(low), (best_t + step).min(high)] {
            let point = bezier_point(filament, candidate_t);
            let distance = point.sub(query).length();
            if distance < best_distance {
                best_t = candidate_t;
                best_point = point;
                best_distance = distance;
            }
        }
        step *= 0.5;
    }
    (best_t, best_point, best_distance)
}

/// Derive a graph from the immutable filament tree plus explicit anastomoses.
/// Each filament becomes one graph vertex; parent relations are tree edges,
/// while accepted fusion records add network edges that may create cycles.
pub fn build_mycelial_network_graph(
    model: &MycelialModel,
    anastomoses: &[MycelialAnastomosis],
    config: MycelialNetworkConfig,
) -> Result<MycelialNetworkGraph, MycelialNetworkError> {
    if model.schema_version != MYCELIAL_MODEL_VERSION {
        return Err(MycelialNetworkError::UnsupportedModelVersion(
            model.schema_version,
        ));
    }
    if !config.max_fusion_distance.is_finite()
        || !(0.0..=100.0).contains(&config.max_fusion_distance)
        || config.max_fusion_distance == 0.0
    {
        return Err(MycelialNetworkError::InvalidConfig);
    }
    if model.filaments.len() > crate::mycelial_mesh::MAX_MYCELIAL_FILAMENTS {
        return Err(MycelialNetworkError::FilamentCountExceeded(
            model.filaments.len(),
        ));
    }
    if anastomoses.len() > MAX_MYCELIAL_ANASTOMOSES {
        return Err(MycelialNetworkError::AnastomosisCountExceeded(
            anastomoses.len(),
        ));
    }

    let mut filaments = BTreeMap::new();
    for filament in &model.filaments {
        if filaments.insert(filament.id, filament).is_some() {
            return Err(MycelialNetworkError::DuplicateFilamentId(filament.id));
        }
    }

    let mut connections =
        Vec::with_capacity(model.filaments.len().saturating_sub(1) + anastomoses.len());
    for filament in &model.filaments {
        if let Some(parent_id) = filament.parent_id {
            if !filaments.contains_key(&parent_id) {
                return Err(MycelialNetworkError::MissingParent {
                    filament_id: filament.id,
                    parent_id,
                });
            }
            connections.push(MycelialNetworkConnection {
                source_filament_id: parent_id,
                target_filament_id: filament.id,
                kind: MycelialConnectionKind::ParentBranch,
                anastomosis_id: None,
            });
        }
    }

    let mut seen_anastomosis_ids = BTreeSet::new();
    let mut accepted_anastomoses = Vec::with_capacity(anastomoses.len());
    for fusion in anastomoses {
        if !seen_anastomosis_ids.insert(fusion.id) {
            return Err(MycelialNetworkError::DuplicateAnastomosisId(fusion.id));
        }
        if fusion.source_filament_id == fusion.target_filament_id {
            return Err(MycelialNetworkError::SelfFusion(fusion.id));
        }
        let source = filaments.get(&fusion.source_filament_id).ok_or(
            MycelialNetworkError::MissingSourceFilament(fusion.source_filament_id),
        )?;
        let target = filaments.get(&fusion.target_filament_id).ok_or(
            MycelialNetworkError::MissingTargetFilament(fusion.target_filament_id),
        )?;
        if !source.terminal {
            return Err(MycelialNetworkError::SourceIsNotTerminalTip(fusion.id));
        }

        if !fusion.source_t.is_finite()
            || !fusion.target_t.is_finite()
            || !(0.0..=1.0).contains(&fusion.source_t)
            || !(0.0..=1.0).contains(&fusion.target_t)
            || (fusion.source_t - 1.0).abs() > TIP_PARAMETER_EPSILON
        {
            return Err(MycelialNetworkError::InvalidFusionParameter(fusion.id));
        }
        match fusion.kind {
            AnastomosisKind::TipToTip => {
                if !target.terminal {
                    return Err(MycelialNetworkError::TargetIsNotTerminalTip(fusion.id));
                }
                if (fusion.target_t - 1.0).abs() > TIP_PARAMETER_EPSILON {
                    return Err(MycelialNetworkError::InvalidFusionParameter(fusion.id));
                }
            }
            AnastomosisKind::TipToSide => {
                if fusion.target_t <= INTERIOR_PARAMETER_MARGIN
                    || fusion.target_t >= 1.0 - INTERIOR_PARAMETER_MARGIN
                {
                    return Err(MycelialNetworkError::InvalidFusionParameter(fusion.id));
                }
            }
        }
        if !fusion.junction_position.is_finite() {
            return Err(MycelialNetworkError::InvalidJunctionPosition(fusion.id));
        }

        let source_point = bezier_point(source, fusion.source_t);
        let target_point = bezier_point(target, fusion.target_t);
        let source_distance = source_point.sub(fusion.junction_position).length();
        let target_distance = target_point.sub(fusion.junction_position).length();
        let contact_distance = source_distance.max(target_distance);
        if !contact_distance.is_finite() || contact_distance > config.max_fusion_distance {
            return Err(MycelialNetworkError::FusionTooFar {
                anastomosis_id: fusion.id,
                distance: contact_distance,
                limit: config.max_fusion_distance,
            });
        }

        connections.push(MycelialNetworkConnection {
            source_filament_id: fusion.source_filament_id,
            target_filament_id: fusion.target_filament_id,
            kind: MycelialConnectionKind::Anastomosis,
            anastomosis_id: Some(fusion.id),
        });
        accepted_anastomoses.push(fusion.clone());
    }

    connections.sort_by_key(|connection| {
        (
            connection.source_filament_id,
            connection.target_filament_id,
            connection.kind,
            connection.anastomosis_id,
        )
    });
    accepted_anastomoses.sort_by_key(|fusion| fusion.id);
    let mut filament_ids: Vec<_> = filaments.keys().copied().collect();
    filament_ids.sort_unstable();

    Ok(MycelialNetworkGraph {
        schema_version: MYCELIAL_NETWORK_SCHEMA_VERSION,
        source_model_version: model.schema_version,
        filament_ids,
        connections,
        anastomoses: accepted_anastomoses,
    })
}

fn bezier_point(filament: &crate::mycelial_model::MycelialFilament, t: f32) -> SpatialPoint3 {
    let u = 1.0 - t;
    filament
        .start
        .scale(u * u * u)
        .add(filament.control_a.scale(3.0 * u * u * t))
        .add(filament.control_b.scale(3.0 * u * t * t))
        .add(filament.end.scale(t * t * t))
}

impl MycelialNetworkGraph {
    pub fn metrics(&self) -> MycelialNetworkMetrics {
        let mut parent_branch_count = 0;
        let mut anastomosis_count = 0;
        let mut degree = HashMap::<u32, usize>::with_capacity(self.filament_ids.len());
        let mut parent = HashMap::<u32, u32>::with_capacity(self.filament_ids.len());
        for id in &self.filament_ids {
            degree.insert(*id, 0);
            parent.insert(*id, *id);
        }

        for connection in &self.connections {
            if let Some(value) = degree.get_mut(&connection.source_filament_id) {
                *value += 1;
            }
            if let Some(value) = degree.get_mut(&connection.target_filament_id) {
                *value += 1;
            }
            match connection.kind {
                MycelialConnectionKind::ParentBranch => parent_branch_count += 1,
                MycelialConnectionKind::Anastomosis => anastomosis_count += 1,
            }
            union_components(
                &mut parent,
                connection.source_filament_id,
                connection.target_filament_id,
            );
        }

        let mut component_roots = BTreeSet::new();
        for id in &self.filament_ids {
            component_roots.insert(find_component(&mut parent, *id));
        }
        let connected_components = component_roots.len();
        let vertex_count = self.filament_ids.len();
        let connection_count = self.connections.len();
        let cycle_rank = connection_count
            .saturating_add(connected_components)
            .saturating_sub(vertex_count);
        let maximum_degree = degree.values().copied().max().unwrap_or(0);
        let (articulation_filament_count, max_components_after_single_filament_failure) =
            single_filament_failure_metrics(&self.filament_ids, &self.connections);

        MycelialNetworkMetrics {
            filament_count: vertex_count,
            parent_branch_count,
            anastomosis_count,
            connection_count,
            connected_components,
            cycle_rank,
            maximum_degree,
            articulation_filament_count,
            max_components_after_single_filament_failure,
        }
    }

    /// Canonical topology digest; geometry coordinates and camera state are not
    /// included. Fusion IDs, filament endpoints and connection kinds are bound.
    pub fn canonical_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-mycelial-network-v1");
        hasher.update(&self.schema_version.to_le_bytes());
        hasher.update(&self.source_model_version.to_le_bytes());
        hasher.update(&(self.filament_ids.len() as u64).to_le_bytes());
        for id in &self.filament_ids {
            hasher.update(&id.to_le_bytes());
        }
        hasher.update(&(self.connections.len() as u64).to_le_bytes());
        for connection in &self.connections {
            hasher.update(&connection.source_filament_id.to_le_bytes());
            hasher.update(&connection.target_filament_id.to_le_bytes());
            match connection.kind {
                MycelialConnectionKind::ParentBranch => hasher.update(&[0]),
                MycelialConnectionKind::Anastomosis => hasher.update(&[1]),
            };
            match connection.anastomosis_id {
                Some(id) => {
                    hasher.update(&[1]);
                    hasher.update(&id.to_le_bytes());
                }
                None => {
                    hasher.update(&[0]);
                }
            }
        }
        let mut output = String::with_capacity(64);
        for byte in hasher.finalize().as_bytes() {
            use fmt::Write as _;
            write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
        }
        output
    }
}

/// Count single-filament articulation points and worst-case component
/// count after deleting one vertex. This is a graph-theoretic stress indicator,
/// not a claim about biological recovery or organism viability.
fn single_filament_failure_metrics(
    filament_ids: &[u32],
    connections: &[MycelialNetworkConnection],
) -> (usize, usize) {
    if filament_ids.is_empty() {
        return (0, 0);
    }
    let mut adjacency = HashMap::<u32, Vec<u32>>::with_capacity(filament_ids.len());
    for id in filament_ids {
        adjacency.insert(*id, Vec::new());
    }
    for connection in connections {
        if let Some(neighbours) = adjacency.get_mut(&connection.source_filament_id) {
            neighbours.push(connection.target_filament_id);
        }
        if let Some(neighbours) = adjacency.get_mut(&connection.target_filament_id) {
            neighbours.push(connection.source_filament_id);
        }
    }

    let baseline_components = count_components_without(&adjacency, None);
    let mut articulation_count = 0;
    let mut worst_components = 0;
    for removed in filament_ids {
        let components = count_components_without(&adjacency, Some(*removed));
        worst_components = worst_components.max(components);
        if components > baseline_components {
            articulation_count += 1;
        }
    }
    (articulation_count, worst_components)
}

fn count_components_without(adjacency: &HashMap<u32, Vec<u32>>, excluded: Option<u32>) -> usize {
    let mut visited = BTreeSet::new();
    let mut components = 0;
    for start in adjacency.keys().copied() {
        if Some(start) == excluded || visited.contains(&start) {
            continue;
        }
        components += 1;
        let mut stack = vec![start];
        visited.insert(start);
        while let Some(current) = stack.pop() {
            if let Some(neighbours) = adjacency.get(&current) {
                for neighbour in neighbours {
                    if Some(*neighbour) != excluded && visited.insert(*neighbour) {
                        stack.push(*neighbour);
                    }
                }
            }
        }
    }
    components
}

fn find_component(parent: &mut HashMap<u32, u32>, id: u32) -> u32 {
    let current = parent.get(&id).copied().unwrap_or(id);
    if current == id {
        id
    } else {
        let root = find_component(parent, current);
        parent.insert(id, root);
        root
    }
}

fn union_components(parent: &mut HashMap<u32, u32>, left: u32, right: u32) {
    let left_root = find_component(parent, left);
    let right_root = find_component(parent, right);
    if left_root != right_root {
        let (lower, higher) = if left_root < right_root {
            (left_root, right_root)
        } else {
            (right_root, left_root)
        };
        parent.insert(higher, lower);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CognitiveTopology;
    use crate::mycelial_model::{
        MycelialFilament, OrbitalCell, PresenceMode, build_mycelial_model,
    };

    fn filament(
        id: u32,
        parent_id: Option<u32>,
        start: SpatialPoint3,
        control_a: SpatialPoint3,
        control_b: SpatialPoint3,
        end: SpatialPoint3,
    ) -> MycelialFilament {
        MycelialFilament {
            id,
            parent_id,
            generation: u8::from(parent_id.is_some()),
            terminal: true,
            start,
            control_a,
            control_b,
            end,
            radius_start: 1.0,
            radius_end: 0.5,
        }
    }

    fn fork_model() -> MycelialModel {
        let root_end = SpatialPoint3 {
            x: 0.0,
            y: 3.0,
            z: 0.0,
        };
        MycelialModel {
            schema_version: MYCELIAL_MODEL_VERSION,
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
                    root_end,
                ),
                filament(
                    2,
                    Some(1),
                    root_end,
                    SpatialPoint3 {
                        x: 1.0,
                        y: 4.0,
                        z: 0.0,
                    },
                    SpatialPoint3 {
                        x: 2.0,
                        y: 5.0,
                        z: 0.0,
                    },
                    SpatialPoint3 {
                        x: 3.0,
                        y: 6.0,
                        z: 0.0,
                    },
                ),
                filament(
                    3,
                    Some(1),
                    root_end,
                    SpatialPoint3 {
                        x: 2.0,
                        y: 5.0,
                        z: 0.0,
                    },
                    SpatialPoint3 {
                        x: 4.0,
                        y: 7.0,
                        z: 0.0,
                    },
                    SpatialPoint3 {
                        x: 6.0,
                        y: 9.0,
                        z: 0.0,
                    },
                ),
            ],
            orbital_cells: Vec::<OrbitalCell>::new(),
        }
    }

    #[test]
    fn deleting_the_only_filament_leaves_zero_components() {
        let graph = MycelialNetworkGraph {
            schema_version: MYCELIAL_NETWORK_SCHEMA_VERSION,
            source_model_version: MYCELIAL_MODEL_VERSION,
            filament_ids: vec![7],
            connections: Vec::new(),
            anastomoses: Vec::new(),
        };
        let metrics = graph.metrics();
        assert_eq!(metrics.filament_count, 1);
        assert_eq!(metrics.connected_components, 1);
        assert_eq!(metrics.cycle_rank, 0);
        assert_eq!(metrics.articulation_filament_count, 0);
        assert_eq!(metrics.max_components_after_single_filament_failure, 0);
    }

    #[test]
    fn binary_branching_tree_has_zero_cycle_rank() {
        let model = build_mycelial_model(
            &crate::embodiment::EmbodimentSnapshot::default(),
            PresenceMode::Ambient,
        );
        let graph =
            build_mycelial_network_graph(&model, &[], MycelialNetworkConfig::default()).unwrap();
        let metrics = graph.metrics();
        assert_eq!(metrics.filament_count, 120);
        assert_eq!(metrics.parent_branch_count, 112);
        assert_eq!(metrics.anastomosis_count, 0);
        assert_eq!(metrics.connected_components, 8);
        assert_eq!(metrics.cycle_rank, 0);
        assert_eq!(metrics.connected_components, 8);
        assert_eq!(metrics.max_components_after_single_filament_failure, 10);
    }

    #[test]
    fn tip_to_side_anastomosis_creates_one_cycle_without_editing_model() {
        let model = fork_model();
        let original = model.clone();
        let fusion = MycelialAnastomosis {
            id: 7,
            source_filament_id: 2,
            source_t: 1.0,
            target_filament_id: 3,
            target_t: 0.5,
            kind: AnastomosisKind::TipToSide,
            junction_position: SpatialPoint3 {
                x: 3.0,
                y: 6.0,
                z: 0.0,
            },
        };
        let graph =
            build_mycelial_network_graph(&model, &[fusion], MycelialNetworkConfig::default())
                .unwrap();
        let metrics = graph.metrics();
        assert_eq!(metrics.filament_count, 3);
        assert_eq!(metrics.parent_branch_count, 2);
        assert_eq!(metrics.anastomosis_count, 1);
        assert_eq!(metrics.connected_components, 1);
        assert_eq!(metrics.cycle_rank, 1);
        assert_eq!(metrics.articulation_filament_count, 0);
        assert_eq!(metrics.max_components_after_single_filament_failure, 1);
        assert_eq!(model, original);
    }

    #[test]
    fn anastomosis_detection_is_deterministic_and_only_proposes_contacts() {
        let model = fork_model();
        let original = model.clone();
        let first =
            detect_anastomosis_candidates(&model, MycelialNetworkConfig::default()).unwrap();
        let second =
            detect_anastomosis_candidates(&model, MycelialNetworkConfig::default()).unwrap();
        assert_eq!(first, second);
        assert!(first.iter().any(|candidate| {
            candidate.fusion.source_filament_id == 2
                && candidate.fusion.target_filament_id == 3
                && candidate.fusion.kind == AnastomosisKind::TipToSide
                && (candidate.fusion.target_t - 0.5).abs() < 0.02
                && candidate.contact_distance < 0.05
        }));
        assert_eq!(model, original);

        // Detection itself is proposal-only: the tree still has no cycles.
        let tree =
            build_mycelial_network_graph(&model, &[], MycelialNetworkConfig::default()).unwrap();
        assert_eq!(tree.metrics().cycle_rank, 0);
    }

    #[test]
    fn accepted_candidate_is_the_only_operation_that_adds_network_connectivity() {
        let model = fork_model();
        let candidate = detect_anastomosis_candidates(&model, MycelialNetworkConfig::default())
            .unwrap()
            .into_iter()
            .find(|candidate| {
                candidate.fusion.source_filament_id == 2
                    && candidate.fusion.target_filament_id == 3
                    && candidate.fusion.kind == AnastomosisKind::TipToSide
            })
            .expect("known geometric contact");
        let graph = build_mycelial_network_graph(
            &model,
            &[candidate.fusion],
            MycelialNetworkConfig::default(),
        )
        .unwrap();
        assert_eq!(graph.metrics().cycle_rank, 1);
    }

    #[test]
    fn a_geometric_near_miss_is_not_accepted_as_fusion() {
        let model = fork_model();
        let fusion = MycelialAnastomosis {
            id: 8,
            source_filament_id: 2,
            source_t: 1.0,
            target_filament_id: 3,
            target_t: 0.5,
            kind: AnastomosisKind::TipToSide,
            junction_position: SpatialPoint3 {
                x: 50.0,
                y: 50.0,
                z: 0.0,
            },
        };
        assert!(matches!(
            build_mycelial_network_graph(&model, &[fusion], MycelialNetworkConfig::default(),),
            Err(MycelialNetworkError::FusionTooFar { .. })
        ));
    }

    #[test]
    fn invalid_fusion_kind_and_self_contact_fail_closed() {
        let model = fork_model();
        let bad_parameter = MycelialAnastomosis {
            id: 9,
            source_filament_id: 2,
            source_t: 1.0,
            target_filament_id: 3,
            target_t: 0.5,
            kind: AnastomosisKind::TipToTip,
            junction_position: SpatialPoint3 {
                x: 3.0,
                y: 6.0,
                z: 0.0,
            },
        };
        assert!(matches!(
            build_mycelial_network_graph(
                &model,
                &[bad_parameter],
                MycelialNetworkConfig::default()
            ),
            Err(MycelialNetworkError::InvalidFusionParameter(9))
        ));

        let self_contact = MycelialAnastomosis {
            id: 10,
            source_filament_id: 2,
            source_t: 1.0,
            target_filament_id: 2,
            target_t: 0.5,
            kind: AnastomosisKind::TipToSide,
            junction_position: SpatialPoint3 {
                x: 3.0,
                y: 6.0,
                z: 0.0,
            },
        };
        assert!(matches!(
            build_mycelial_network_graph(&model, &[self_contact], MycelialNetworkConfig::default()),
            Err(MycelialNetworkError::SelfFusion(10))
        ));
    }

    #[test]
    fn morphology_fusion_does_not_mutate_cognitive_topology_digest() {
        let topology: CognitiveTopology =
            serde_json::from_str(include_str!("../tests/fixtures/cognitive_topology_v1.json"))
                .unwrap();
        let before = topology.canonical_digest_hex().unwrap();
        let model = fork_model();
        let fusion = MycelialAnastomosis {
            id: 11,
            source_filament_id: 2,
            source_t: 1.0,
            target_filament_id: 3,
            target_t: 0.5,
            kind: AnastomosisKind::TipToSide,
            junction_position: SpatialPoint3 {
                x: 3.0,
                y: 6.0,
                z: 0.0,
            },
        };
        let _graph =
            build_mycelial_network_graph(&model, &[fusion], MycelialNetworkConfig::default())
                .unwrap();
        assert_eq!(topology.canonical_digest_hex().unwrap(), before);
    }
}
