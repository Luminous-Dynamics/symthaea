// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Contact-tolerance filtration for uncertain geometric anastomosis candidates.
//!
//! This is a monotone weighted-graph filtration, not a temporal zigzag
//! persistence implementation. Candidate contacts are sorted by their measured
//! geometric separation. Increasing the tolerance admits more candidate links.
//! No candidate is accepted into MycelialNetworkGraph or MycelialModel.
//!
//! Since this implementation only adds graph edges and does not add 2-simplices,
//! H1 classes have no finite death in this graph filtration. Cycle births are
//! therefore explicitly right-censored at the configured maximum tolerance.
//! They must not be described as full finite-lifetime persistence intervals.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::mycelial_model::{MycelialModel, MycelialFilament, SpatialPoint3};
use crate::mycelial_network::{
    AnastomosisKind, MycelialAnastomosis, MycelialAnastomosisCandidate,
    MycelialNetworkConfig, build_mycelial_network_graph,
};

pub const MYCELIAL_CONTACT_FILTRATION_SCHEMA_VERSION: u16 = 1;
pub const DEFAULT_MAX_CONTACT_TOLERANCE_MODEL_UNITS: f32 = 5.0;
pub const MAX_CONTACT_FILTRATION_CANDIDATES: usize = 512;
const CONTACT_DISTANCE_ABSOLUTE_EPSILON: f32 = 1.0e-3;
const CONTACT_DISTANCE_RELATIVE_EPSILON: f32 = 1.0e-3;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MycelialContactFiltrationConfig {
    /// Upper censoring threshold, in renderer-local model units.
    pub maximum_contact_tolerance_model_units: f32,
}

impl Default for MycelialContactFiltrationConfig {
    fn default() -> Self {
        Self {
            maximum_contact_tolerance_model_units: DEFAULT_MAX_CONTACT_TOLERANCE_MODEL_UNITS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelialContactCandidateObservation {
    pub fusion_candidate_id: u32,
    pub source_filament_id: u32,
    pub target_filament_id: u32,
    pub kind: AnastomosisKind,
    pub source_parameter_micros: u32,
    pub target_parameter_micros: u32,
    pub contact_distance_microunits: u64,
    pub admitted_within_maximum_tolerance: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelialContactFiltrationSnapshot {
    /// Current candidate admission threshold. This is not a physical distance.
    pub contact_tolerance_microunits: u64,
    pub admitted_candidate_count: usize,
    pub connected_components: usize,
    pub cycle_rank: usize,
    pub component_merges_at_threshold: usize,
    pub cycle_births_at_threshold: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelialContactComponentMerge {
    pub fusion_candidate_id: u32,
    pub contact_distance_microunits: u64,
    pub source_filament_id: u32,
    pub target_filament_id: u32,
    pub connected_components_after_merge: usize,
}

/// One edge in the previously existing path that a newly admitted candidate
/// closes into a fundamental cycle. None denotes a parent-child branch edge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelialCyclePathStep {
    pub from_filament_id: u32,
    pub to_filament_id: u32,
    pub prior_fusion_candidate_id: Option<u32>,
}

/// A fundamental cycle is born when this candidate edge joins two vertices
/// already connected by the preceding filtration graph. Its class is right
/// censored at the maximum configured contact tolerance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelialContactCycleBirth {
    pub fusion_candidate_id: u32,
    pub contact_distance_microunits: u64,
    pub right_censor_tolerance_microunits: u64,
    pub source_filament_id: u32,
    pub target_filament_id: u32,
    /// Verifiable path in the pre-existing graph; the candidate edge closes it.
    pub witness_path: Vec<MycelialCyclePathStep>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelialContactFiltrationMetrics {
    pub baseline_filament_count: usize,
    pub baseline_parent_edge_count: usize,
    pub baseline_connected_components: usize,
    pub candidate_count_total: usize,
    pub candidate_count_in_filtration: usize,
    pub candidate_count_above_maximum: usize,
    pub component_merge_count: usize,
    pub cycle_birth_count: usize,
    pub final_connected_components: usize,
    pub final_cycle_rank: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialContactFiltration {
    pub schema_version: u16,
    pub length_unit: String,
    pub source_geometry_digest: String,
    pub maximum_contact_tolerance_model_units: f32,
    pub metrics: MycelialContactFiltrationMetrics,
    /// Complete sorted input inventory, including candidates above the sweep cap.
    pub candidate_observations: Vec<MycelialContactCandidateObservation>,
    /// One snapshot after each distinct admitted contact distance.
    pub snapshots: Vec<MycelialContactFiltrationSnapshot>,
    pub component_merges: Vec<MycelialContactComponentMerge>,
    pub cycle_births: Vec<MycelialContactCycleBirth>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MycelialContactFiltrationError {
    InvalidConfig,
    TooManyCandidates(usize),
    DuplicateCandidateId(u32),
    InvalidCandidateDistance(u32),
    UnknownFilament(u32),
    InvalidCandidateGeometry(u32),
    CandidateDistanceMismatch {
        fusion_candidate_id: u32,
        supplied: f32,
        measured: f32,
    },
    InvalidBaseline(String),
    InvalidFiltration(&'static str),
}

impl fmt::Display for MycelialContactFiltrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => write!(f, "invalid maximum contact tolerance"),
            Self::TooManyCandidates(count) => {
                write!(f, "candidate count {count} exceeds the filtration limit")
            }
            Self::DuplicateCandidateId(id) => {
                write!(f, "duplicate fusion candidate ID {id}")
            }
            Self::InvalidCandidateDistance(id) => {
                write!(f, "fusion candidate {id} has invalid contact distance")
            }
            Self::UnknownFilament(id) => {
                write!(f, "fusion candidate references missing filament {id}")
            }
            Self::InvalidCandidateGeometry(id) => {
                write!(f, "fusion candidate {id} has invalid contact geometry")
            }
            Self::CandidateDistanceMismatch {
                fusion_candidate_id,
                supplied,
                measured,
            } => write!(
                f,
                "fusion candidate {fusion_candidate_id} distance mismatch: supplied {supplied:.6}, measured {measured:.6}"
            ),
            Self::InvalidBaseline(message) => {
                write!(f, "invalid baseline mycelial graph: {message}")
            }
            Self::InvalidFiltration(reason) => {
                write!(f, "invalid contact-tolerance filtration: {reason}")
            }
        }
    }
}

impl Error for MycelialContactFiltrationError {}

/// Construct a contact-tolerance filtration from **candidate-only** fusion data.
///
/// The baseline contains parent/child connections only. Candidate contacts are
/// inserted in ascending contact-distance order. At each distinct threshold we
/// record component merges and independent cycle births. All contact distances
/// are recomputed from the supplied source geometry; inconsistent candidate
/// measurements fail closed.
///
/// This function never accepts a fusion, mutates source geometry, or returns an
/// accepted network graph. The output is an uncertainty-sensitivity analysis.
pub fn build_mycelial_contact_filtration(
    model: &MycelialModel,
    candidates: &[MycelialAnastomosisCandidate],
    config: MycelialContactFiltrationConfig,
) -> Result<MycelialContactFiltration, MycelialContactFiltrationError> {
    if !config.maximum_contact_tolerance_model_units.is_finite()
        || !(0.0..=100.0).contains(&config.maximum_contact_tolerance_model_units)
        || config.maximum_contact_tolerance_model_units == 0.0
    {
        return Err(MycelialContactFiltrationError::InvalidConfig);
    }
    if candidates.len() > MAX_CONTACT_FILTRATION_CANDIDATES {
        return Err(MycelialContactFiltrationError::TooManyCandidates(
            candidates.len(),
        ));
    }

    let baseline = crate::mycelial_junction_graph::build_mycelial_junction_hypha_graph(
        model,
        &[],
        crate::mycelial_junction_graph::MycelialJunctionGraphConfig::default(),
    )
    .map_err(|error| MycelialContactFiltrationError::InvalidBaseline(error.to_string()))?;

    let mut filaments = BTreeMap::<u32, &MycelialFilament>::new();
    for filament in &model.filaments {
        filaments.insert(filament.id, filament);
    }

    let mut seen_ids = BTreeSet::new();
    let mut seen_contacts = BTreeSet::new();
    let mut validated_fusions = Vec::with_capacity(candidates.len());
    let mut measured_candidates = Vec::with_capacity(candidates.len());
    let mut candidate_observations = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let fusion = &candidate.fusion;
        if !seen_ids.insert(fusion.id) {
            return Err(MycelialContactFiltrationError::DuplicateCandidateId(
                fusion.id,
            ));
        }
        if !candidate.contact_distance.is_finite()
            || candidate.contact_distance < 0.0
            || candidate.contact_distance > 100.0
        {
            return Err(MycelialContactFiltrationError::InvalidCandidateDistance(
                fusion.id,
            ));
        }
        let source = filaments
            .get(&fusion.source_filament_id)
            .copied()
            .ok_or(MycelialContactFiltrationError::UnknownFilament(
                fusion.source_filament_id,
            ))?;
        let target = filaments
            .get(&fusion.target_filament_id)
            .copied()
            .ok_or(MycelialContactFiltrationError::UnknownFilament(
                fusion.target_filament_id,
            ))?;
        let source_point = bezier_point(source, fusion.source_t);
        let target_point = bezier_point(target, fusion.target_t);
        if !source_point.is_finite() || !target_point.is_finite() {
            return Err(MycelialContactFiltrationError::InvalidCandidateGeometry(
                fusion.id,
            ));
        }
        let measured_distance = point_distance(source_point, target_point);
        let expected_anchor = source_point.add(target_point).scale(0.5);
        let anchor_error = point_distance(expected_anchor, fusion.junction_position);
        let allowed_error = CONTACT_DISTANCE_ABSOLUTE_EPSILON
            .max(measured_distance * CONTACT_DISTANCE_RELATIVE_EPSILON);
        let parameter_micros = |value: f32| {
            (f64::from(value.clamp(0.0, 1.0)) * 1_000_000.0).round() as u32
        };
        let contact_key = (
            fusion.source_filament_id,
            parameter_micros(fusion.source_t),
            fusion.target_filament_id,
            parameter_micros(fusion.target_t),
            fusion.kind,
        );
        if !seen_contacts.insert(contact_key)
            || !anchor_error.is_finite()
            || anchor_error > CONTACT_DISTANCE_ABSOLUTE_EPSILON
        {
            return Err(MycelialContactFiltrationError::InvalidCandidateGeometry(
                fusion.id,
            ));
        }
        if !measured_distance.is_finite()
            || (candidate.contact_distance - measured_distance).abs() > allowed_error
        {
            return Err(MycelialContactFiltrationError::CandidateDistanceMismatch {
                fusion_candidate_id: fusion.id,
                supplied: candidate.contact_distance,
                measured: measured_distance,
            });
        }
        validated_fusions.push(fusion.clone());
        let admitted = quantize(measured_distance)
            <= quantize(config.maximum_contact_tolerance_model_units);
        candidate_observations.push(MycelialContactCandidateObservation {
            fusion_candidate_id: fusion.id,
            source_filament_id: fusion.source_filament_id,
            target_filament_id: fusion.target_filament_id,
            kind: fusion.kind,
            source_parameter_micros: parameter_micros(fusion.source_t),
            target_parameter_micros: parameter_micros(fusion.target_t),
            contact_distance_microunits: quantize(measured_distance),
            admitted_within_maximum_tolerance: admitted,
        });
        measured_candidates.push((candidate.clone(), measured_distance));
    }

    // Reuse the existing candidate contact contract without changing or
    // persisting its output as accepted topology.
    build_mycelial_network_graph(
        model,
        &validated_fusions,
        MycelialNetworkConfig {
            max_fusion_distance: 100.0,
        },
    )
    .map_err(|error| MycelialContactFiltrationError::InvalidBaseline(error.to_string()))?;

    candidate_observations.sort_by_key(|value| value.fusion_candidate_id);
    measured_candidates.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.fusion.source_filament_id.cmp(&right.0.fusion.source_filament_id))
            .then_with(|| left.0.fusion.target_filament_id.cmp(&right.0.fusion.target_filament_id))
            .then_with(|| left.0.fusion.kind.cmp(&right.0.fusion.kind))
            .then_with(|| left.0.fusion.target_t.total_cmp(&right.0.fusion.target_t))
            .then_with(|| left.0.fusion.id.cmp(&right.0.fusion.id))
    });

    let mut disjoint = DisjointSet::new(baseline.filament_ids.iter().copied());
    let mut adjacency = BTreeMap::<u32, Vec<(u32, Option<u32>)>>::new();
    for filament_id in &baseline.filament_ids {
        adjacency.insert(*filament_id, Vec::new());
    }
    for connection in &baseline.connections {
        add_adjacency(
            &mut adjacency,
            connection.source_filament_id,
            connection.target_filament_id,
            None,
        );
        let _ = disjoint.union(
            connection.source_filament_id,
            connection.target_filament_id,
        );
    }
    let baseline_components = disjoint.component_count();
    let baseline_parent_edges = baseline.connections.len();

    let maximum_tolerance_microunits = quantize(config.maximum_contact_tolerance_model_units);
    let mut snapshots = Vec::new();
    let mut component_merges = Vec::new();
    let mut cycle_births = Vec::new();
    let mut candidate_count_in_filtration = 0usize;
    let mut cursor = 0usize;

    while cursor < measured_candidates.len() {
        let (_, first_distance) = &measured_candidates[cursor];
        let threshold_microunits = quantize(*first_distance);
        if threshold_microunits > maximum_tolerance_microunits {
            break;
        }
        let mut merges_at_threshold = 0usize;
        let mut cycles_at_threshold = 0usize;

        while cursor < measured_candidates.len()
            && quantize(measured_candidates[cursor].1) == threshold_microunits
        {
            let (candidate, measured_distance) = &measured_candidates[cursor];
            let source_id = candidate.fusion.source_filament_id;
            let target_id = candidate.fusion.target_filament_id;
            candidate_count_in_filtration += 1;

            if disjoint.union(source_id, target_id) {
                merges_at_threshold += 1;
                component_merges.push(MycelialContactComponentMerge {
                    fusion_candidate_id: candidate.fusion.id,
                    contact_distance_microunits: threshold_microunits,
                    source_filament_id: source_id,
                    target_filament_id: target_id,
                    connected_components_after_merge: disjoint.component_count(),
                });
            } else {
                let witness_path = find_path(&adjacency, source_id, target_id)
                    .ok_or(MycelialContactFiltrationError::InvalidFiltration(
                        "union-find reports connected vertices but no witness path exists",
                    ))?;
                cycles_at_threshold += 1;
                cycle_births.push(MycelialContactCycleBirth {
                    fusion_candidate_id: candidate.fusion.id,
                    contact_distance_microunits: threshold_microunits,
                    right_censor_tolerance_microunits: maximum_tolerance_microunits,
                    source_filament_id: source_id,
                    target_filament_id: target_id,
                    witness_path,
                });
            }
            add_adjacency(
                &mut adjacency,
                source_id,
                target_id,
                Some(candidate.fusion.id),
            );
            cursor += 1;
            let _ = measured_distance;
        }

        let admitted_candidate_count = candidate_count_in_filtration;
        let connected_components = disjoint.component_count();
        let cycle_rank = admitted_candidate_count
            .saturating_sub(baseline_components.saturating_sub(connected_components));
        snapshots.push(MycelialContactFiltrationSnapshot {
            contact_tolerance_microunits: threshold_microunits,
            admitted_candidate_count,
            connected_components,
            cycle_rank,
            component_merges_at_threshold: merges_at_threshold,
            cycle_births_at_threshold: cycles_at_threshold,
        });
    }

    let candidate_count_above_maximum = measured_candidates
        .len()
        .saturating_sub(candidate_count_in_filtration);
    let final_components = disjoint.component_count();
    let final_cycle_rank = candidate_count_in_filtration
        .saturating_sub(baseline_components.saturating_sub(final_components));

    let profile = MycelialContactFiltration {
        schema_version: MYCELIAL_CONTACT_FILTRATION_SCHEMA_VERSION,
        length_unit: "organism_local_model_unit".to_string(),
        source_geometry_digest: baseline.source_geometry_digest,
        maximum_contact_tolerance_model_units: config.maximum_contact_tolerance_model_units,
        metrics: MycelialContactFiltrationMetrics {
            baseline_filament_count: baseline.filament_ids.len(),
            baseline_parent_edge_count: baseline_parent_edges,
            baseline_connected_components: baseline_components,
            candidate_count_total: candidates.len(),
            candidate_count_in_filtration,
            candidate_count_above_maximum,
            component_merge_count: component_merges.len(),
            cycle_birth_count: cycle_births.len(),
            final_connected_components: final_components,
            final_cycle_rank,
        },
        candidate_observations,
        snapshots,
        component_merges,
        cycle_births,
    };
    profile.validate()?;
    Ok(profile)
}

impl MycelialContactFiltration {
    pub fn validate(&self) -> Result<(), MycelialContactFiltrationError> {
        if self.schema_version != MYCELIAL_CONTACT_FILTRATION_SCHEMA_VERSION
            || self.length_unit != "organism_local_model_unit"
            || self.source_geometry_digest.len() != 64
            || !self
                .source_geometry_digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            || !self.maximum_contact_tolerance_model_units.is_finite()
            || !(0.0..=100.0).contains(&self.maximum_contact_tolerance_model_units)
            || self.maximum_contact_tolerance_model_units == 0.0
        {
            return Err(MycelialContactFiltrationError::InvalidFiltration(
                "invalid schema, length unit, source digest or maximum tolerance",
            ));
        }
        if self.candidate_observations.len() != self.metrics.candidate_count_total
            || self.candidate_observations.windows(2).any(|pair| {
                pair[0].fusion_candidate_id >= pair[1].fusion_candidate_id
            })
        {
            return Err(MycelialContactFiltrationError::InvalidFiltration(
                "candidate observation inventory is not complete, unique and sorted",
            ));
        }
        let observed_admitted = self
            .candidate_observations
            .iter()
            .filter(|candidate| candidate.admitted_within_maximum_tolerance)
            .count();
        if observed_admitted != self.metrics.candidate_count_in_filtration
            || self
                .candidate_observations
                .iter()
                .any(|candidate| {
                    candidate.admitted_within_maximum_tolerance
                        != (candidate.contact_distance_microunits
                            <= quantize(self.maximum_contact_tolerance_model_units))
                })
        {
            return Err(MycelialContactFiltrationError::InvalidFiltration(
                "candidate admission flags do not match their threshold distances",
            ));
        }
        if self.metrics.candidate_count_total
            != self.metrics.candidate_count_in_filtration
                + self.metrics.candidate_count_above_maximum
            || self.metrics.component_merge_count + self.metrics.cycle_birth_count
                != self.metrics.candidate_count_in_filtration
            || self.metrics.final_connected_components
                != self.metrics.baseline_connected_components
                    .saturating_sub(self.metrics.component_merge_count)
            || self.metrics.final_cycle_rank != self.metrics.cycle_birth_count
        {
            return Err(MycelialContactFiltrationError::InvalidFiltration(
                "candidate, component and cycle totals disagree",
            ));
        }
        let observation_by_id: BTreeMap<_, _> = self
            .candidate_observations
            .iter()
            .map(|candidate| (candidate.fusion_candidate_id, candidate))
            .collect();
        let mut event_ids = BTreeSet::new();
        for event in &self.component_merges {
            let candidate = observation_by_id
                .get(&event.fusion_candidate_id)
                .ok_or(MycelialContactFiltrationError::InvalidFiltration(
                    "component merge has no candidate observation",
                ))?;
            if !candidate.admitted_within_maximum_tolerance
                || candidate.source_filament_id != event.source_filament_id
                || candidate.target_filament_id != event.target_filament_id
                || candidate.contact_distance_microunits != event.contact_distance_microunits
                || !event_ids.insert(event.fusion_candidate_id)
            {
                return Err(MycelialContactFiltrationError::InvalidFiltration(
                    "component merge does not match a unique admitted candidate",
                ));
            }
        }
        for event in &self.cycle_births {
            let candidate = observation_by_id
                .get(&event.fusion_candidate_id)
                .ok_or(MycelialContactFiltrationError::InvalidFiltration(
                    "cycle birth has no candidate observation",
                ))?;
            if !candidate.admitted_within_maximum_tolerance
                || candidate.source_filament_id != event.source_filament_id
                || candidate.target_filament_id != event.target_filament_id
                || candidate.contact_distance_microunits != event.contact_distance_microunits
                || !event_ids.insert(event.fusion_candidate_id)
                || event.witness_path.first().map(|step| step.from_filament_id)
                    != Some(event.source_filament_id)
                || event.witness_path.last().map(|step| step.to_filament_id)
                    != Some(event.target_filament_id)
                || event.witness_path.windows(2).any(|pair| {
                    pair[0].to_filament_id != pair[1].from_filament_id
                })
            {
                return Err(MycelialContactFiltrationError::InvalidFiltration(
                    "cycle birth does not match a unique candidate or has an invalid witness path",
                ));
            }
        }
        if event_ids.len() != self.metrics.candidate_count_in_filtration {
            return Err(MycelialContactFiltrationError::InvalidFiltration(
                "not every admitted candidate has exactly one topological event",
            ));
        }
        let mut previous_tolerance = None;
        let mut previous_candidates = 0usize;
        let mut previous_components = self.metrics.baseline_connected_components;
        let mut previous_cycles = 0usize;
        for snapshot in &self.snapshots {
            if previous_tolerance.is_some_and(|value| {
                snapshot.contact_tolerance_microunits <= value
            })
                || snapshot.admitted_candidate_count <= previous_candidates
                || snapshot.connected_components > previous_components
                || snapshot.cycle_rank < previous_cycles
                || snapshot.admitted_candidate_count
                    != self.metrics.candidate_count_in_filtration.min(
                        snapshot.admitted_candidate_count,
                    )
            {
                return Err(MycelialContactFiltrationError::InvalidFiltration(
                    "filtration snapshots are not monotone",
                ));
            }
            let merge_total: usize = self
                .component_merges
                .iter()
                .filter(|event| event.contact_distance_microunits <= snapshot.contact_tolerance_microunits)
                .count();
            let cycle_total: usize = self
                .cycle_births
                .iter()
                .filter(|event| event.contact_distance_microunits <= snapshot.contact_tolerance_microunits)
                .count();
            if snapshot.connected_components
                != self.metrics.baseline_connected_components.saturating_sub(merge_total)
                || snapshot.cycle_rank != cycle_total
                || snapshot.admitted_candidate_count != merge_total + cycle_total
                || snapshot.component_merges_at_threshold
                    != self
                        .component_merges
                        .iter()
                        .filter(|event| event.contact_distance_microunits == snapshot.contact_tolerance_microunits)
                        .count()
                || snapshot.cycle_births_at_threshold
                    != self
                        .cycle_births
                        .iter()
                        .filter(|event| event.contact_distance_microunits == snapshot.contact_tolerance_microunits)
                        .count()
            {
                return Err(MycelialContactFiltrationError::InvalidFiltration(
                    "snapshot totals do not match admitted component/cycle events",
                ));
            }
            previous_tolerance = Some(snapshot.contact_tolerance_microunits);
            previous_candidates = snapshot.admitted_candidate_count;
            previous_components = snapshot.connected_components;
            previous_cycles = snapshot.cycle_rank;
        }
        for event in &self.cycle_births {
            if event.contact_distance_microunits > event.right_censor_tolerance_microunits
                || event.right_censor_tolerance_microunits
                    != quantize(self.maximum_contact_tolerance_model_units)
                || event.witness_path.is_empty()
            {
                return Err(MycelialContactFiltrationError::InvalidFiltration(
                    "cycle birth has invalid censoring or lacks a witness path",
                ));
            }
        }
        Ok(())
    }

    pub fn canonical_digest_hex(&self) -> Result<String, MycelialContactFiltrationError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-mycelial-contact-filtration-v1");
        hasher.update(&self.schema_version.to_le_bytes());
        hasher.update(self.length_unit.as_bytes());
        hasher.update(self.source_geometry_digest.as_bytes());
        hasher.update(&self.maximum_contact_tolerance_model_units.to_bits().to_le_bytes());
        for value in [
            self.metrics.baseline_filament_count,
            self.metrics.baseline_parent_edge_count,
            self.metrics.baseline_connected_components,
            self.metrics.candidate_count_total,
            self.metrics.candidate_count_in_filtration,
            self.metrics.candidate_count_above_maximum,
            self.metrics.component_merge_count,
            self.metrics.cycle_birth_count,
            self.metrics.final_connected_components,
            self.metrics.final_cycle_rank,
        ] {
            hasher.update(&(value as u64).to_le_bytes());
        }
        for candidate in &self.candidate_observations {
            hasher.update(&candidate.fusion_candidate_id.to_le_bytes());
            hasher.update(&candidate.source_filament_id.to_le_bytes());
            hasher.update(&candidate.target_filament_id.to_le_bytes());
            hasher.update(&[match candidate.kind {
                AnastomosisKind::TipToTip => 0,
                AnastomosisKind::TipToSide => 1,
            }]);
            hasher.update(&candidate.source_parameter_micros.to_le_bytes());
            hasher.update(&candidate.target_parameter_micros.to_le_bytes());
            hasher.update(&candidate.contact_distance_microunits.to_le_bytes());
            hasher.update(&[u8::from(candidate.admitted_within_maximum_tolerance)]);
        }
        for snapshot in &self.snapshots {
            hasher.update(&snapshot.contact_tolerance_microunits.to_le_bytes());
            hasher.update(&(snapshot.admitted_candidate_count as u64).to_le_bytes());
            hasher.update(&(snapshot.connected_components as u64).to_le_bytes());
            hasher.update(&(snapshot.cycle_rank as u64).to_le_bytes());
            hasher.update(&(snapshot.component_merges_at_threshold as u64).to_le_bytes());
            hasher.update(&(snapshot.cycle_births_at_threshold as u64).to_le_bytes());
        }
        for event in &self.component_merges {
            hasher.update(&event.fusion_candidate_id.to_le_bytes());
            hasher.update(&event.contact_distance_microunits.to_le_bytes());
            hasher.update(&event.source_filament_id.to_le_bytes());
            hasher.update(&event.target_filament_id.to_le_bytes());
            hasher.update(&(event.connected_components_after_merge as u64).to_le_bytes());
        }
        for event in &self.cycle_births {
            hasher.update(&event.fusion_candidate_id.to_le_bytes());
            hasher.update(&event.contact_distance_microunits.to_le_bytes());
            hasher.update(&event.right_censor_tolerance_microunits.to_le_bytes());
            hasher.update(&event.source_filament_id.to_le_bytes());
            hasher.update(&event.target_filament_id.to_le_bytes());
            hasher.update(&(event.witness_path.len() as u64).to_le_bytes());
            for step in &event.witness_path {
                hasher.update(&step.from_filament_id.to_le_bytes());
                hasher.update(&step.to_filament_id.to_le_bytes());
                match step.prior_fusion_candidate_id {
                    Some(id) => {
                        hasher.update(&[1]);
                        hasher.update(&id.to_le_bytes());
                    }
                    None => hasher.update(&[0]),
                }
            }
        }
        Ok(hex_digest(hasher.finalize().as_bytes()))
    }
}

struct DisjointSet {
    parent: BTreeMap<u32, u32>,
    components: usize,
}

impl DisjointSet {
    fn new(values: impl IntoIterator<Item = u32>) -> Self {
        let parent: BTreeMap<_, _> = values.into_iter().map(|id| (id, id)).collect();
        let components = parent.len();
        Self { parent, components }
    }

    fn find(&mut self, id: u32) -> Option<u32> {
        let current = *self.parent.get(&id)?;
        if current == id {
            Some(id)
        } else {
            let root = self.find(current)?;
            self.parent.insert(id, root);
            Some(root)
        }
    }

    fn union(&mut self, left: u32, right: u32) -> bool {
        let Some(left_root) = self.find(left) else {
            return false;
        };
        let Some(right_root) = self.find(right) else {
            return false;
        };
        if left_root == right_root {
            return false;
        }
        let (lower, higher) = if left_root < right_root {
            (left_root, right_root)
        } else {
            (right_root, left_root)
        };
        self.parent.insert(higher, lower);
        self.components = self.components.saturating_sub(1);
        true
    }

    fn component_count(&self) -> usize {
        self.components
    }
}

fn add_adjacency(
    adjacency: &mut BTreeMap<u32, Vec<(u32, Option<u32>)>>,
    source: u32,
    target: u32,
    candidate_id: Option<u32>,
) {
    adjacency.entry(source).or_default().push((target, candidate_id));
    adjacency.entry(target).or_default().push((source, candidate_id));
    for edges in adjacency.values_mut() {
        edges.sort_unstable();
    }
}

fn find_path(
    adjacency: &BTreeMap<u32, Vec<(u32, Option<u32>)>>,
    source: u32,
    target: u32,
) -> Option<Vec<MycelialCyclePathStep>> {
    let mut queue = VecDeque::from([source]);
    let mut predecessor = BTreeMap::<u32, (u32, Option<u32>)>::new();
    let mut visited = BTreeSet::from([source]);
    while let Some(current) = queue.pop_front() {
        if current == target {
            break;
        }
        for (next, candidate_id) in adjacency.get(&current)? {
            if visited.insert(*next) {
                predecessor.insert(*next, (current, *candidate_id));
                queue.push_back(*next);
            }
        }
    }
    if !visited.contains(&target) {
        return None;
    }
    let mut reversed = Vec::new();
    let mut cursor = target;
    while cursor != source {
        let (previous, candidate_id) = *predecessor.get(&cursor)?;
        reversed.push(MycelialCyclePathStep {
            from_filament_id: previous,
            to_filament_id: cursor,
            prior_fusion_candidate_id: candidate_id,
        });
        cursor = previous;
    }
    reversed.reverse();
    Some(reversed)
}

fn bezier_point(filament: &MycelialFilament, parameter: f32) -> SpatialPoint3 {
    let t = f64::from(parameter);
    let u = 1.0 - t;
    let coordinate = |p0: f32, p1: f32, p2: f32, p3: f32| {
        (u.powi(3) * f64::from(p0)
            + 3.0 * u.powi(2) * t * f64::from(p1)
            + 3.0 * u * t.powi(2) * f64::from(p2)
            + t.powi(3) * f64::from(p3)) as f32
    };
    SpatialPoint3 {
        x: coordinate(
            filament.start.x,
            filament.control_a.x,
            filament.control_b.x,
            filament.end.x,
        ),
        y: coordinate(
            filament.start.y,
            filament.control_a.y,
            filament.control_b.y,
            filament.end.y,
        ),
        z: coordinate(
            filament.start.z,
            filament.control_a.z,
            filament.control_b.z,
            filament.end.z,
        ),
    }
}

fn point_distance(left: SpatialPoint3, right: SpatialPoint3) -> f32 {
    left.sub(right).length()
}

fn quantize(value: f32) -> u64 {
    (f64::from(value) * 1_000_000.0).round() as u64
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
    use crate::mycelial_model::{PresenceMode};
    use crate::mycelial_network::AnastomosisKind;

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

    fn fixture_model(source_tip_offset: f32) -> MycelialModel {
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
                        x: source_tip_offset * 0.25,
                        y: 11.5,
                        z: 0.0,
                    },
                    SpatialPoint3 {
                        x: source_tip_offset * 0.75,
                        y: 13.5,
                        z: 0.0,
                    },
                    SpatialPoint3 {
                        x: source_tip_offset,
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
            orbital_cells: Vec::new(),
        }
    }

    fn tip_to_side_candidate(model: &MycelialModel, id: u32, distance: f32) -> MycelialAnastomosisCandidate {
        let target = model.filaments.iter().find(|value| value.id == 3).unwrap();
        let target_point = bezier_point(target, 0.5);
        let source = model.filaments.iter().find(|value| value.id == 2).unwrap();
        let source_point = source.end;
        MycelialAnastomosisCandidate {
            fusion: MycelialAnastomosis {
                id,
                source_filament_id: 2,
                source_t: 1.0,
                target_filament_id: 3,
                target_t: 0.5,
                kind: AnastomosisKind::TipToSide,
                junction_position: source_point.add(target_point).scale(0.5),
            },
            contact_distance: distance,
        }
    }

    #[test]
    fn contact_filtration_records_a_cycle_birth_and_a_verifiable_witness() {
        let model = fixture_model(0.25);
        let candidate = tip_to_side_candidate(&model, 17, 0.25);
        let profile = build_mycelial_contact_filtration(
            &model,
            &[candidate],
            MycelialContactFiltrationConfig {
                maximum_contact_tolerance_model_units: 1.0,
            },
        )
        .unwrap();
        assert_eq!(profile.metrics.baseline_connected_components, 1);
        assert_eq!(profile.metrics.candidate_count_in_filtration, 1);
        assert_eq!(profile.metrics.component_merge_count, 0);
        assert_eq!(profile.metrics.cycle_birth_count, 1);
        assert_eq!(profile.metrics.final_cycle_rank, 1);
        assert_eq!(profile.snapshots.len(), 1);
        assert_eq!(profile.snapshots[0].contact_tolerance_microunits, 250_000);
        assert_eq!(
            profile.cycle_births[0].right_censor_tolerance_microunits,
            1_000_000
        );
        assert_eq!(profile.cycle_births[0].witness_path.len(), 2);
        assert!(profile.cycle_births[0].witness_path.iter().all(|step| {
            step.prior_fusion_candidate_id.is_none()
        }));
        profile.validate().unwrap();
        assert_eq!(
            profile.canonical_digest_hex().unwrap(),
            profile.canonical_digest_hex().unwrap()
        );
    }

    #[test]
    fn disconnected_roots_merge_before_any_cycle_is_born() {
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
                    SpatialPoint3 { x: 1.0, y: 0.0, z: 0.0 },
                    SpatialPoint3 { x: 1.0, y: 1.0, z: 0.0 },
                    SpatialPoint3 { x: 1.0, y: 2.0, z: 0.0 },
                    SpatialPoint3 { x: 1.0, y: 3.0, z: 0.0 },
                ),
            ],
            orbital_cells: Vec::new(),
        };
        let source = &model.filaments[0];
        let target = &model.filaments[1];
        let distance = point_distance(source.end, target.end);
        let candidate = MycelialAnastomosisCandidate {
            fusion: MycelialAnastomosis {
                id: 9,
                source_filament_id: 4,
                source_t: 1.0,
                target_filament_id: 5,
                target_t: 1.0,
                kind: AnastomosisKind::TipToTip,
                junction_position: source.end.add(target.end).scale(0.5),
            },
            contact_distance: distance,
        };
        let profile = build_mycelial_contact_filtration(
            &model,
            &[candidate],
            MycelialContactFiltrationConfig {
                maximum_contact_tolerance_model_units: 5.0,
            },
        )
        .unwrap();
        assert_eq!(profile.metrics.baseline_connected_components, 2);
        assert_eq!(profile.metrics.component_merge_count, 1);
        assert_eq!(profile.metrics.cycle_birth_count, 0);
        assert_eq!(profile.metrics.final_connected_components, 1);
        assert_eq!(profile.metrics.final_cycle_rank, 0);
    }

    #[test]
    fn candidate_above_maximum_tolerance_is_reported_but_not_admitted() {
        let model = fixture_model(0.25);
        let candidate = tip_to_side_candidate(&model, 17, 0.25);
        let profile = build_mycelial_contact_filtration(
            &model,
            &[candidate],
            MycelialContactFiltrationConfig {
                maximum_contact_tolerance_model_units: 0.1,
            },
        )
        .unwrap();
        assert_eq!(profile.metrics.candidate_count_total, 1);
        assert_eq!(profile.metrics.candidate_count_in_filtration, 0);
        assert_eq!(profile.metrics.candidate_count_above_maximum, 1);
        assert_eq!(profile.metrics.final_cycle_rank, 0);
        assert!(profile.snapshots.is_empty());
    }

    #[test]
    fn candidate_distance_is_recomputed_and_mismatch_fails_closed() {
        let model = fixture_model(0.25);
        let candidate = tip_to_side_candidate(&model, 17, 0.5);
        assert!(matches!(
            build_mycelial_contact_filtration(
                &model,
                &[candidate],
                MycelialContactFiltrationConfig::default(),
            ),
            Err(MycelialContactFiltrationError::CandidateDistanceMismatch { .. })
        ));
    }

    #[test]
    fn candidate_analysis_does_not_change_source_geometry() {
        let model = fixture_model(0.25);
        let before = crate::mycelial_junction_graph::build_mycelial_junction_hypha_graph(
            &model,
            &[],
            crate::mycelial_junction_graph::MycelialJunctionGraphConfig::default(),
        )
        .unwrap()
        .source_geometry_digest;
        let candidate = tip_to_side_candidate(&model, 17, 0.25);
        let _profile = build_mycelial_contact_filtration(
            &model,
            &[candidate],
            MycelialContactFiltrationConfig::default(),
        )
        .unwrap();
        let after = crate::mycelial_junction_graph::build_mycelial_junction_hypha_graph(
            &model,
            &[],
            crate::mycelial_junction_graph::MycelialJunctionGraphConfig::default(),
        )
        .unwrap()
        .source_geometry_digest;
        assert_eq!(before, after);
    }
}
