// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Ground-truth graph reconstruction scoring for annotated mycelial skeletons.
//!
//! This module scores graph reconstruction after an explicit one-to-one
//! junction identity matching policy has aligned predicted and annotated
//! junctions. It keeps object detection, junction classification, edge
//! topology, coordinate localization, edge-length error and graph topology
//! as separate metrics. It does not assume that renderer-local coordinates
//! are biological measurements.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

pub const MYCELIAL_RECONSTRUCTION_SCHEMA_VERSION: u16 = 1;
pub const MYCELIAL_RECONSTRUCTION_SCORE_SCHEMA_VERSION: u16 = 1;
pub const MAX_RECONSTRUCTION_JUNCTIONS: usize = 8_192;
pub const MAX_RECONSTRUCTION_SEGMENTS: usize = 16_384;
pub const MAX_AUTOMATIC_MATCH_JUNCTIONS_PER_GRAPH: usize = 128;
pub const MAX_RECONSTRUCTION_ID_BYTES: usize = 128;
const MAX_RECONSTRUCTION_COORDINATE: f64 = 1.0e9;
const MAX_SEGMENT_LENGTH: f64 = 1.0e12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReconstructionDimensionality {
    TwoD,
    ThreeD,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReconstructionLengthUnit {
    Pixel,
    Micrometer,
    Millimeter,
    OrganismLocalModelUnit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnnotatedJunctionKind {
    Origin,
    Branch,
    Tip,
    Anastomosis,
    ProjectedCrossing,
    Uncertain,
}

const ANNOTATED_JUNCTION_KINDS: [AnnotatedJunctionKind; 6] = [
    AnnotatedJunctionKind::Origin,
    AnnotatedJunctionKind::Branch,
    AnnotatedJunctionKind::Tip,
    AnnotatedJunctionKind::Anastomosis,
    AnnotatedJunctionKind::ProjectedCrossing,
    AnnotatedJunctionKind::Uncertain,
];

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReconstructionPoint3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl ReconstructionPoint3 {
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnnotatedJunction {
    pub id: String,
    pub kind: AnnotatedJunctionKind,
    pub position: ReconstructionPoint3,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnnotatedHyphaSegment {
    pub id: String,
    pub start_junction_id: String,
    pub end_junction_id: String,
    pub length: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnnotatedMycelialSkeleton {
    pub schema_version: u16,
    pub frame_id: String,
    /// Coordinate axes/origin used by all junction positions in this frame.
    pub coordinate_frame_id: String,
    /// Digest of the immutable calibration manifest (pixel scale, voxel spacing,
    /// coordinate transform, distortion correction and uncertainty model).
    pub calibration_digest: String,
    /// Digest of the acquired image, volume or immutable source observation.
    pub source_observation_digest: String,
    pub dimensionality: ReconstructionDimensionality,
    pub length_unit: ReconstructionLengthUnit,
    pub producer_id: String,
    pub producer_version: String,
    pub junctions: Vec<AnnotatedJunction>,
    pub segments: Vec<AnnotatedHyphaSegment>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconstructionCounts {
    pub true_positive: usize,
    pub false_positive: usize,
    pub false_negative: usize,
}

impl ReconstructionCounts {
    pub fn precision(&self) -> Option<f64> {
        let denominator = self.true_positive + self.false_positive;
        (denominator > 0).then_some(self.true_positive as f64 / denominator as f64)
    }

    pub fn recall(&self) -> Option<f64> {
        let denominator = self.true_positive + self.false_negative;
        (denominator > 0).then_some(self.true_positive as f64 / denominator as f64)
    }

    pub fn f1(&self) -> Option<f64> {
        let denominator = 2 * self.true_positive + self.false_positive + self.false_negative;
        (denominator > 0).then_some(2.0 * self.true_positive as f64 / denominator as f64)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JunctionKindScore {
    pub kind: AnnotatedJunctionKind,
    pub counts: ReconstructionCounts,
    pub f1: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconstructionTopologySummary {
    pub junction_count: usize,
    pub segment_count: usize,
    pub connected_components: usize,
    pub cycle_rank: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconstructionTopologyDelta {
    pub junction_count: i64,
    pub segment_count: i64,
    pub connected_components: i64,
    pub cycle_rank: i64,
}

/// One relationship in the graph-component attribution table. Component
/// identity is the lexicographically smallest stable junction ID in that
/// connected component. An absent side denotes a wholly unmatched component.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconstructionComponentAttribution {
    pub predicted_component_id: Option<String>,
    pub ground_truth_component_id: Option<String>,
    pub matched_junction_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconstructionComponentErrorSummary {
    pub matched_junction_count: usize,
    pub matched_prediction_component_count: usize,
    pub matched_ground_truth_component_count: usize,
    /// Additional ground-truth components incorrectly joined per predicted component.
    pub false_merge_count: usize,
    /// Additional predicted components that fragment each ground-truth component.
    pub false_split_count: usize,
    pub unmatched_prediction_component_count: usize,
    pub unmatched_ground_truth_component_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReconstructionLocalizationError {
    pub matched_junction_count: usize,
    pub mean_error: Option<f64>,
    pub median_error: Option<f64>,
    pub p95_error: Option<f64>,
    pub maximum_error: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReconstructionLengthError {
    pub matched_segment_count: usize,
    pub mean_absolute_error: Option<f64>,
    pub mean_absolute_relative_error: Option<f64>,
    pub p95_absolute_error: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JunctionIdentityMatch {
    pub predicted_junction_id: String,
    pub ground_truth_junction_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReconstructionEvaluationConfig {
    /// Maximum allowed Euclidean separation for a supplied identity match,
    /// expressed in the shared coordinate unit.
    pub maximum_junction_match_distance: f64,
    /// Lower bound used for relative segment-length error denominators.
    pub length_relative_epsilon: f64,
}

impl Default for ReconstructionEvaluationConfig {
    fn default() -> Self {
        Self {
            maximum_junction_match_distance: 2.0,
            length_relative_epsilon: 1.0e-12,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialReconstructionScore {
    pub schema_version: u16,
    pub frame_id: String,
    pub coordinate_frame_id: String,
    pub calibration_digest: String,
    pub source_observation_digest: String,
    pub dimensionality: ReconstructionDimensionality,
    pub length_unit: ReconstructionLengthUnit,
    pub matching_policy_id: String,
    pub maximum_junction_match_distance: f64,
    pub length_relative_epsilon: f64,
    pub ground_truth_digest: String,
    pub prediction_digest: String,
    pub identity_matches: Vec<JunctionIdentityMatch>,
    pub junction_detection: ReconstructionCounts,
    pub junction_kind_scores: Vec<JunctionKindScore>,
    pub junction_kind_macro_f1: Option<f64>,
    pub false_anastomosis_junction_count: usize,
    pub edge_topology: ReconstructionCounts,
    pub unmatched_prediction_edge_count: usize,
    pub ground_truth_topology: ReconstructionTopologySummary,
    pub prediction_topology: ReconstructionTopologySummary,
    pub topology_delta: ReconstructionTopologyDelta,
    pub component_attributions: Vec<ReconstructionComponentAttribution>,
    pub component_error_summary: ReconstructionComponentErrorSummary,
    pub localization_error: ReconstructionLocalizationError,
    pub segment_length_error: ReconstructionLengthError,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MycelialReconstructionError {
    UnsupportedSchemaVersion(u16),
    InvalidGraph(&'static str),
    TooManyJunctions(usize),
    TooManySegments(usize),
    DuplicateJunctionId(String),
    DuplicateSegmentId(String),
    MissingSegmentEndpoint {
        segment_id: String,
        junction_id: String,
    },
    InvalidSegment(String),
    FrameMismatch,
    CoordinateFrameMismatch,
    CalibrationMismatch,
    SourceObservationMismatch,
    DimensionalityMismatch,
    LengthUnitMismatch,
    InvalidConfig,
    InvalidMatchingPolicy,
    AutomaticMatchingLimitExceeded {
        prediction_count: usize,
        ground_truth_count: usize,
        limit_per_graph: usize,
    },
    DuplicatePredictionMatch(String),
    DuplicateGroundTruthMatch(String),
    UnknownPredictionJunction(String),
    UnknownGroundTruthJunction(String),
    JunctionMatchOutOfTolerance {
        predicted_id: String,
        ground_truth_id: String,
        distance: f64,
        limit: f64,
    },
}

impl fmt::Display for MycelialReconstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion(version) => {
                write!(f, "unsupported annotated skeleton schema version {version}")
            }
            Self::InvalidGraph(reason) => write!(f, "invalid annotated skeleton: {reason}"),
            Self::TooManyJunctions(count) => {
                write!(
                    f,
                    "junction count {count} exceeds {MAX_RECONSTRUCTION_JUNCTIONS}"
                )
            }
            Self::TooManySegments(count) => {
                write!(
                    f,
                    "segment count {count} exceeds {MAX_RECONSTRUCTION_SEGMENTS}"
                )
            }
            Self::DuplicateJunctionId(id) => write!(f, "duplicate junction ID {id}"),
            Self::DuplicateSegmentId(id) => write!(f, "duplicate segment ID {id}"),
            Self::MissingSegmentEndpoint {
                segment_id,
                junction_id,
            } => {
                write!(
                    f,
                    "segment {segment_id} references missing junction {junction_id}"
                )
            }
            Self::InvalidSegment(id) => write!(f, "segment {id} has invalid geometry or length"),
            Self::FrameMismatch => write!(f, "ground truth and prediction frame IDs differ"),
            Self::CoordinateFrameMismatch => {
                write!(f, "ground truth and prediction coordinate-frame IDs differ")
            }
            Self::CalibrationMismatch => {
                write!(
                    f,
                    "ground truth and prediction calibration manifests differ"
                )
            }
            Self::SourceObservationMismatch => {
                write!(
                    f,
                    "ground truth and prediction source-observation digests differ"
                )
            }
            Self::DimensionalityMismatch => {
                write!(f, "ground truth and prediction dimensionalities differ")
            }
            Self::LengthUnitMismatch => {
                write!(f, "ground truth and prediction length units differ")
            }
            Self::InvalidConfig => write!(f, "invalid reconstruction evaluation configuration"),
            Self::InvalidMatchingPolicy => {
                write!(f, "matching policy ID must be non-empty and bounded")
            }
            Self::AutomaticMatchingLimitExceeded {
                prediction_count,
                ground_truth_count,
                limit_per_graph,
            } => write!(
                f,
                "automatic matcher is bounded to {limit_per_graph} junctions per graph; prediction={prediction_count}, ground_truth={ground_truth_count}"
            ),
            Self::DuplicatePredictionMatch(id) => {
                write!(f, "prediction junction {id} is matched more than once")
            }
            Self::DuplicateGroundTruthMatch(id) => {
                write!(f, "ground-truth junction {id} is matched more than once")
            }
            Self::UnknownPredictionJunction(id) => {
                write!(
                    f,
                    "matching policy references unknown prediction junction {id}"
                )
            }
            Self::UnknownGroundTruthJunction(id) => {
                write!(
                    f,
                    "matching policy references unknown ground-truth junction {id}"
                )
            }
            Self::JunctionMatchOutOfTolerance {
                predicted_id,
                ground_truth_id,
                distance,
                limit,
            } => write!(
                f,
                "junction match {predicted_id}->{ground_truth_id} distance {distance:.6} exceeds limit {limit:.6}"
            ),
        }
    }
}

impl Error for MycelialReconstructionError {}

/// Validate an annotated skeleton before it is scored or serialized.
impl AnnotatedMycelialSkeleton {
    pub fn validate(&self) -> Result<(), MycelialReconstructionError> {
        if self.schema_version != MYCELIAL_RECONSTRUCTION_SCHEMA_VERSION {
            return Err(MycelialReconstructionError::UnsupportedSchemaVersion(
                self.schema_version,
            ));
        }
        if !valid_id(&self.frame_id)
            || !valid_id(&self.coordinate_frame_id)
            || !valid_digest(&self.calibration_digest)
            || self.producer_id.is_empty()
            || self.producer_id.len() > MAX_RECONSTRUCTION_ID_BYTES
            || self.producer_id.chars().any(char::is_control)
            || self.producer_version.is_empty()
            || self.producer_version.len() > MAX_RECONSTRUCTION_ID_BYTES
            || self.producer_version.chars().any(char::is_control)
            || !valid_digest(&self.source_observation_digest)
        {
            return Err(MycelialReconstructionError::InvalidGraph(
                "invalid frame, source digest or producer metadata",
            ));
        }
        if self.junctions.len() > MAX_RECONSTRUCTION_JUNCTIONS {
            return Err(MycelialReconstructionError::TooManyJunctions(
                self.junctions.len(),
            ));
        }
        if self.segments.len() > MAX_RECONSTRUCTION_SEGMENTS {
            return Err(MycelialReconstructionError::TooManySegments(
                self.segments.len(),
            ));
        }
        let mut junctions = BTreeSet::new();
        for junction in &self.junctions {
            let position = junction.position;
            if !valid_id(&junction.id)
                || !junctions.insert(junction.id.as_str())
                || !position.is_finite()
                || position.x.abs() > MAX_RECONSTRUCTION_COORDINATE
                || position.y.abs() > MAX_RECONSTRUCTION_COORDINATE
                || position.z.abs() > MAX_RECONSTRUCTION_COORDINATE
                || (self.dimensionality == ReconstructionDimensionality::TwoD
                    && position.z.abs() > 1.0e-9)
            {
                return Err(MycelialReconstructionError::InvalidGraph(
                    "duplicate/invalid junction ID or out-of-range coordinate",
                ));
            }
        }
        let mut segment_ids = BTreeSet::new();
        for segment in &self.segments {
            if !valid_id(&segment.id) || !segment_ids.insert(segment.id.as_str()) {
                return Err(MycelialReconstructionError::DuplicateSegmentId(
                    segment.id.clone(),
                ));
            }
            if segment.start_junction_id == segment.end_junction_id
                || !segment.length.is_finite()
                || segment.length <= 0.0
                || segment.length > MAX_SEGMENT_LENGTH
            {
                return Err(MycelialReconstructionError::InvalidSegment(
                    segment.id.clone(),
                ));
            }
            for junction_id in [&segment.start_junction_id, &segment.end_junction_id] {
                if !junctions.contains(junction_id.as_str()) {
                    return Err(MycelialReconstructionError::MissingSegmentEndpoint {
                        segment_id: segment.id.clone(),
                        junction_id: junction_id.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    pub fn canonical_digest_hex(&self) -> Result<String, MycelialReconstructionError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-annotated-mycelial-skeleton-v1");
        hasher.update(&self.schema_version.to_le_bytes());
        hash_string(&mut hasher, &self.frame_id);
        hash_string(&mut hasher, &self.coordinate_frame_id);
        hasher.update(self.calibration_digest.as_bytes());
        hasher.update(self.source_observation_digest.as_bytes());
        hasher.update(&[match self.dimensionality {
            ReconstructionDimensionality::TwoD => 0,
            ReconstructionDimensionality::ThreeD => 1,
        }]);
        hasher.update(&[match self.length_unit {
            ReconstructionLengthUnit::Pixel => 0,
            ReconstructionLengthUnit::Micrometer => 1,
            ReconstructionLengthUnit::Millimeter => 2,
            ReconstructionLengthUnit::OrganismLocalModelUnit => 3,
        }]);
        hash_string(&mut hasher, &self.producer_id);
        hash_string(&mut hasher, &self.producer_version);
        let mut junctions: Vec<_> = self.junctions.iter().collect();
        junctions.sort_by(|left, right| left.id.cmp(&right.id));
        hasher.update(&(junctions.len() as u64).to_le_bytes());
        for junction in junctions {
            hash_string(&mut hasher, &junction.id);
            hasher.update(&[junction_kind_index(junction.kind) as u8]);
            for coordinate in [
                junction.position.x,
                junction.position.y,
                junction.position.z,
            ] {
                hasher.update(&coordinate.to_bits().to_le_bytes());
            }
        }
        let mut segments: Vec<_> = self.segments.iter().collect();
        segments.sort_by(|left, right| left.id.cmp(&right.id));
        hasher.update(&(segments.len() as u64).to_le_bytes());
        for segment in segments {
            hash_string(&mut hasher, &segment.id);
            hash_string(&mut hasher, &segment.start_junction_id);
            hash_string(&mut hasher, &segment.end_junction_id);
            hasher.update(&segment.length.to_bits().to_le_bytes());
        }
        Ok(hex_digest(hasher.finalize().as_bytes()))
    }
}

/// Compute a deterministic one-to-one junction correspondence using a
/// bounded Hungarian assignment.
///
/// The assignment objective is lexicographic: maximize the number of matches
/// within `maximum_distance`, then minimize their total Euclidean localization
/// error. Junction IDs are sorted before assignment so ties resolve reproducibly.
/// This function only proposes correspondences; the caller still names the
/// matching policy when constructing the evaluation receipt.
///
/// Complexity is O((n_prediction + n_truth)^3), bounded to 128 junctions per
/// graph. Larger reconstructions must use a separately qualified sparse or
/// spatially partitioned matcher and report its policy.
pub fn match_junctions_hungarian(
    ground_truth: &AnnotatedMycelialSkeleton,
    prediction: &AnnotatedMycelialSkeleton,
    maximum_distance: f64,
) -> Result<Vec<JunctionIdentityMatch>, MycelialReconstructionError> {
    ground_truth.validate()?;
    prediction.validate()?;
    if ground_truth.frame_id != prediction.frame_id {
        return Err(MycelialReconstructionError::FrameMismatch);
    }
    if ground_truth.coordinate_frame_id != prediction.coordinate_frame_id {
        return Err(MycelialReconstructionError::CoordinateFrameMismatch);
    }
    if ground_truth.calibration_digest != prediction.calibration_digest {
        return Err(MycelialReconstructionError::CalibrationMismatch);
    }
    if ground_truth.source_observation_digest != prediction.source_observation_digest {
        return Err(MycelialReconstructionError::SourceObservationMismatch);
    }
    if ground_truth.dimensionality != prediction.dimensionality {
        return Err(MycelialReconstructionError::DimensionalityMismatch);
    }
    if ground_truth.length_unit != prediction.length_unit {
        return Err(MycelialReconstructionError::LengthUnitMismatch);
    }
    if !maximum_distance.is_finite()
        || maximum_distance <= 0.0
        || maximum_distance > MAX_RECONSTRUCTION_COORDINATE
    {
        return Err(MycelialReconstructionError::InvalidConfig);
    }
    if prediction.junctions.len() > MAX_AUTOMATIC_MATCH_JUNCTIONS_PER_GRAPH
        || ground_truth.junctions.len() > MAX_AUTOMATIC_MATCH_JUNCTIONS_PER_GRAPH
    {
        return Err(
            MycelialReconstructionError::AutomaticMatchingLimitExceeded {
                prediction_count: prediction.junctions.len(),
                ground_truth_count: ground_truth.junctions.len(),
                limit_per_graph: MAX_AUTOMATIC_MATCH_JUNCTIONS_PER_GRAPH,
            },
        );
    }

    let mut prediction_nodes: Vec<_> = prediction.junctions.iter().collect();
    let mut ground_truth_nodes: Vec<_> = ground_truth.junctions.iter().collect();
    prediction_nodes.sort_by(|left, right| left.id.cmp(&right.id));
    ground_truth_nodes.sort_by(|left, right| left.id.cmp(&right.id));
    let prediction_count = prediction_nodes.len();
    let truth_count = ground_truth_nodes.len();
    let matrix_size = prediction_count + truth_count;
    if matrix_size == 0 {
        return Ok(Vec::new());
    }

    // Normalize admissible distances by the maximum threshold. This keeps
    // the assignment objective stable for tiny physical units and large
    // coordinate scales without changing the minimum-distance ordering.
    // An extra match reduces two unmatched penalties. 2*(min+1) dominates
    // the largest possible aggregate change in admissible match costs, so
    // cardinality is strictly primary and total distance is secondary.
    let unmatched_cost = prediction_count.min(truth_count) as f64 + 1.0;
    // A forbidden real edge must cost more than assigning every real endpoint
    // to its own unmatched dummy. Therefore it can never appear in an optimum.
    let forbidden_cost = (matrix_size as f64 + 1.0) * unmatched_cost;
    let mut costs = vec![vec![0.0f64; matrix_size]; matrix_size];

    for prediction_index in 0..prediction_count {
        for truth_index in 0..truth_count {
            let distance = point_distance(
                prediction_nodes[prediction_index].position,
                ground_truth_nodes[truth_index].position,
                ground_truth.dimensionality,
            );
            costs[prediction_index][truth_index] = if distance <= maximum_distance {
                distance / maximum_distance
            } else {
                forbidden_cost
            };
        }
        for dummy_column in truth_count..matrix_size {
            costs[prediction_index][dummy_column] = unmatched_cost;
        }
    }
    for dummy_row in prediction_count..matrix_size {
        for truth_column in 0..truth_count {
            costs[dummy_row][truth_column] = unmatched_cost;
        }
        for dummy_column in truth_count..matrix_size {
            costs[dummy_row][dummy_column] = 0.0;
        }
    }

    let assignment = hungarian_min_assignment(&costs);
    let mut matches = Vec::new();
    for (prediction_index, &column) in assignment.iter().take(prediction_count).enumerate() {
        if column >= truth_count {
            continue;
        }
        let normalized_distance = costs[prediction_index][column];
        if normalized_distance <= 1.0 {
            matches.push(JunctionIdentityMatch {
                predicted_junction_id: prediction_nodes[prediction_index].id.clone(),
                ground_truth_junction_id: ground_truth_nodes[column].id.clone(),
            });
        }
    }
    matches.sort_by(|left, right| {
        left.predicted_junction_id
            .cmp(&right.predicted_junction_id)
            .then_with(|| {
                left.ground_truth_junction_id
                    .cmp(&right.ground_truth_junction_id)
            })
    });
    Ok(matches)
}

fn hungarian_min_assignment(costs: &[Vec<f64>]) -> Vec<usize> {
    let size = costs.len();
    if size == 0 {
        return Vec::new();
    }
    let mut row_potential = vec![0.0f64; size + 1];
    let mut column_potential = vec![0.0f64; size + 1];
    let mut matched_row_for_column = vec![0usize; size + 1];
    let mut predecessor_column = vec![0usize; size + 1];

    for row in 1..=size {
        matched_row_for_column[0] = row;
        let mut current_column = 0usize;
        let mut minimum_reduced_cost = vec![f64::INFINITY; size + 1];
        let mut used_columns = vec![false; size + 1];
        loop {
            used_columns[current_column] = true;
            let current_row = matched_row_for_column[current_column];
            let mut delta = f64::INFINITY;
            let mut next_column = 0usize;
            for column in 1..=size {
                if used_columns[column] {
                    continue;
                }
                let reduced_cost = costs[current_row - 1][column - 1]
                    - row_potential[current_row]
                    - column_potential[column];
                if reduced_cost < minimum_reduced_cost[column] {
                    minimum_reduced_cost[column] = reduced_cost;
                    predecessor_column[column] = current_column;
                }
                if minimum_reduced_cost[column] < delta {
                    delta = minimum_reduced_cost[column];
                    next_column = column;
                }
            }
            for column in 0..=size {
                if used_columns[column] {
                    row_potential[matched_row_for_column[column]] += delta;
                    column_potential[column] -= delta;
                } else {
                    minimum_reduced_cost[column] -= delta;
                }
            }
            current_column = next_column;
            if matched_row_for_column[current_column] == 0 {
                break;
            }
        }
        loop {
            let previous_column = predecessor_column[current_column];
            matched_row_for_column[current_column] = matched_row_for_column[previous_column];
            current_column = previous_column;
            if current_column == 0 {
                break;
            }
        }
    }

    let mut assignment = vec![usize::MAX; size];
    for column in 1..=size {
        if matched_row_for_column[column] != 0 {
            assignment[matched_row_for_column[column] - 1] = column - 1;
        }
    }
    assignment
}

/// Evaluate prediction fidelity after an explicit one-to-one junction matching
/// procedure has aligned predicted junction IDs with ground-truth junction IDs.
///
/// The evaluator never invents identity matches. The matching-policy ID and
/// the complete mapping are retained in the score receipt. A caller may use
/// annotation-ID matching, a geometric assignment, or a tracked identity map,
/// but must name and freeze that policy for fair comparisons.
pub fn evaluate_mycelial_reconstruction(
    ground_truth: &AnnotatedMycelialSkeleton,
    prediction: &AnnotatedMycelialSkeleton,
    matching_policy_id: &str,
    matches: &[JunctionIdentityMatch],
    config: ReconstructionEvaluationConfig,
) -> Result<MycelialReconstructionScore, MycelialReconstructionError> {
    ground_truth.validate()?;
    prediction.validate()?;
    if ground_truth.frame_id != prediction.frame_id {
        return Err(MycelialReconstructionError::FrameMismatch);
    }
    if ground_truth.coordinate_frame_id != prediction.coordinate_frame_id {
        return Err(MycelialReconstructionError::CoordinateFrameMismatch);
    }
    if ground_truth.calibration_digest != prediction.calibration_digest {
        return Err(MycelialReconstructionError::CalibrationMismatch);
    }
    if ground_truth.source_observation_digest != prediction.source_observation_digest {
        return Err(MycelialReconstructionError::SourceObservationMismatch);
    }
    if ground_truth.dimensionality != prediction.dimensionality {
        return Err(MycelialReconstructionError::DimensionalityMismatch);
    }
    if ground_truth.length_unit != prediction.length_unit {
        return Err(MycelialReconstructionError::LengthUnitMismatch);
    }
    if !valid_id(matching_policy_id) {
        return Err(MycelialReconstructionError::InvalidMatchingPolicy);
    }
    if !config.maximum_junction_match_distance.is_finite()
        || !(0.0..=MAX_RECONSTRUCTION_COORDINATE).contains(&config.maximum_junction_match_distance)
        || config.maximum_junction_match_distance == 0.0
        || !config.length_relative_epsilon.is_finite()
        || !(1.0e-15..=1.0e-3).contains(&config.length_relative_epsilon)
    {
        return Err(MycelialReconstructionError::InvalidConfig);
    }

    let ground_truth_nodes: BTreeMap<_, _> = ground_truth
        .junctions
        .iter()
        .map(|junction| (junction.id.as_str(), junction))
        .collect();
    let prediction_nodes: BTreeMap<_, _> = prediction
        .junctions
        .iter()
        .map(|junction| (junction.id.as_str(), junction))
        .collect();
    let mut prediction_to_truth = BTreeMap::new();
    let mut matched_truth = BTreeSet::new();
    let mut normalized_matches = matches.to_vec();
    normalized_matches.sort_by(|left, right| {
        left.predicted_junction_id
            .cmp(&right.predicted_junction_id)
            .then_with(|| {
                left.ground_truth_junction_id
                    .cmp(&right.ground_truth_junction_id)
            })
    });
    let mut localization_errors = Vec::with_capacity(matches.len());

    for matched in &normalized_matches {
        if prediction_to_truth
            .insert(
                matched.predicted_junction_id.as_str(),
                matched.ground_truth_junction_id.as_str(),
            )
            .is_some()
        {
            return Err(MycelialReconstructionError::DuplicatePredictionMatch(
                matched.predicted_junction_id.clone(),
            ));
        }
        if !matched_truth.insert(matched.ground_truth_junction_id.as_str()) {
            return Err(MycelialReconstructionError::DuplicateGroundTruthMatch(
                matched.ground_truth_junction_id.clone(),
            ));
        }
        let predicted = prediction_nodes
            .get(matched.predicted_junction_id.as_str())
            .copied()
            .ok_or_else(|| {
                MycelialReconstructionError::UnknownPredictionJunction(
                    matched.predicted_junction_id.clone(),
                )
            })?;
        let truth = ground_truth_nodes
            .get(matched.ground_truth_junction_id.as_str())
            .copied()
            .ok_or_else(|| {
                MycelialReconstructionError::UnknownGroundTruthJunction(
                    matched.ground_truth_junction_id.clone(),
                )
            })?;
        let distance = point_distance(
            predicted.position,
            truth.position,
            ground_truth.dimensionality,
        );
        if !distance.is_finite() || distance > config.maximum_junction_match_distance {
            return Err(MycelialReconstructionError::JunctionMatchOutOfTolerance {
                predicted_id: matched.predicted_junction_id.clone(),
                ground_truth_id: matched.ground_truth_junction_id.clone(),
                distance,
                limit: config.maximum_junction_match_distance,
            });
        }
        localization_errors.push(distance);
    }

    let junction_detection = ReconstructionCounts {
        true_positive: normalized_matches.len(),
        false_positive: prediction
            .junctions
            .len()
            .saturating_sub(normalized_matches.len()),
        false_negative: ground_truth
            .junctions
            .len()
            .saturating_sub(normalized_matches.len()),
    };

    let mut junction_kind_scores = Vec::with_capacity(ANNOTATED_JUNCTION_KINDS.len());
    let mut macro_f1_values = Vec::new();
    for kind in ANNOTATED_JUNCTION_KINDS {
        let true_positive = normalized_matches
            .iter()
            .filter(|matched| {
                prediction_nodes[matched.predicted_junction_id.as_str()].kind == kind
                    && ground_truth_nodes[matched.ground_truth_junction_id.as_str()].kind == kind
            })
            .count();
        let predicted_count = prediction
            .junctions
            .iter()
            .filter(|junction| junction.kind == kind)
            .count();
        let ground_truth_count = ground_truth
            .junctions
            .iter()
            .filter(|junction| junction.kind == kind)
            .count();
        let counts = ReconstructionCounts {
            true_positive,
            false_positive: predicted_count.saturating_sub(true_positive),
            false_negative: ground_truth_count.saturating_sub(true_positive),
        };
        let f1 = counts.f1();
        if let Some(value) = f1 {
            macro_f1_values.push(value);
        }
        junction_kind_scores.push(JunctionKindScore { kind, counts, f1 });
    }

    let (edge_topology, unmatched_prediction_edge_count, length_error) = evaluate_edges(
        ground_truth,
        prediction,
        &prediction_to_truth,
        config.length_relative_epsilon,
    );
    let ground_truth_topology = topology_summary(ground_truth);
    let prediction_topology = topology_summary(prediction);
    let topology_delta = ReconstructionTopologyDelta {
        junction_count: signed_difference(
            prediction_topology.junction_count,
            ground_truth_topology.junction_count,
        ),
        segment_count: signed_difference(
            prediction_topology.segment_count,
            ground_truth_topology.segment_count,
        ),
        connected_components: signed_difference(
            prediction_topology.connected_components,
            ground_truth_topology.connected_components,
        ),
        cycle_rank: signed_difference(
            prediction_topology.cycle_rank,
            ground_truth_topology.cycle_rank,
        ),
    };
    let (component_attributions, component_error_summary) = component_attribution(
        ground_truth,
        prediction,
        &normalized_matches,
    );

    let localization_error = summarize_errors(localization_errors);
    let junction_kind_macro_f1 = if macro_f1_values.is_empty() {
        None
    } else {
        Some(macro_f1_values.iter().sum::<f64>() / macro_f1_values.len() as f64)
    };
    let false_anastomosis_junction_count = junction_kind_scores
        .iter()
        .find(|score| score.kind == AnnotatedJunctionKind::Anastomosis)
        .map(|score| score.counts.false_positive)
        .unwrap_or(0);

    let score = MycelialReconstructionScore {
        schema_version: MYCELIAL_RECONSTRUCTION_SCORE_SCHEMA_VERSION,
        frame_id: ground_truth.frame_id.clone(),
        coordinate_frame_id: ground_truth.coordinate_frame_id.clone(),
        calibration_digest: ground_truth.calibration_digest.clone(),
        source_observation_digest: ground_truth.source_observation_digest.clone(),
        dimensionality: ground_truth.dimensionality,
        length_unit: ground_truth.length_unit,
        matching_policy_id: matching_policy_id.to_string(),
        maximum_junction_match_distance: config.maximum_junction_match_distance,
        length_relative_epsilon: config.length_relative_epsilon,
        ground_truth_digest: ground_truth.canonical_digest_hex()?,
        prediction_digest: prediction.canonical_digest_hex()?,
        identity_matches: normalized_matches,
        junction_detection,
        junction_kind_scores,
        junction_kind_macro_f1,
        false_anastomosis_junction_count,
        edge_topology,
        unmatched_prediction_edge_count,
        ground_truth_topology,
        prediction_topology,
        topology_delta,
        component_attributions,
        component_error_summary,
        localization_error,
        segment_length_error: length_error,
    };
    score.validate()?;
    Ok(score)
}

impl MycelialReconstructionScore {
    pub fn validate(&self) -> Result<(), MycelialReconstructionError> {
        let mut predicted_matches = BTreeSet::new();
        let mut truth_matches = BTreeSet::new();
        if self.identity_matches.iter().any(|matched| {
            !valid_id(&matched.predicted_junction_id)
                || !valid_id(&matched.ground_truth_junction_id)
                || !predicted_matches.insert(matched.predicted_junction_id.as_str())
                || !truth_matches.insert(matched.ground_truth_junction_id.as_str())
        }) {
            return Err(MycelialReconstructionError::InvalidGraph(
                "identity-match receipt is invalid or not one-to-one",
            ));
        }
        let class_true_positive: usize = self
            .junction_kind_scores
            .iter()
            .map(|score| score.counts.true_positive)
            .sum();
        let class_false_positive: usize = self
            .junction_kind_scores
            .iter()
            .map(|score| score.counts.false_positive)
            .sum();
        let class_false_negative: usize = self
            .junction_kind_scores
            .iter()
            .map(|score| score.counts.false_negative)
            .sum();
        let expected_macro_f1_values: Vec<_> = self
            .junction_kind_scores
            .iter()
            .filter_map(|score| score.counts.f1())
            .collect();
        let expected_macro_f1 = if expected_macro_f1_values.is_empty() {
            None
        } else {
            Some(
                expected_macro_f1_values.iter().sum::<f64>()
                    / expected_macro_f1_values.len() as f64,
            )
        };
        let expected_topology_delta = ReconstructionTopologyDelta {
            junction_count: signed_difference(
                self.prediction_topology.junction_count,
                self.ground_truth_topology.junction_count,
            ),
            segment_count: signed_difference(
                self.prediction_topology.segment_count,
                self.ground_truth_topology.segment_count,
            ),
            connected_components: signed_difference(
                self.prediction_topology.connected_components,
                self.ground_truth_topology.connected_components,
            ),
            cycle_rank: signed_difference(
                self.prediction_topology.cycle_rank,
                self.ground_truth_topology.cycle_rank,
            ),
        };
        let localization_values_valid = [
            self.localization_error.mean_error,
            self.localization_error.median_error,
            self.localization_error.p95_error,
            self.localization_error.maximum_error,
        ]
        .into_iter()
        .flatten()
        .all(|value| value.is_finite() && value >= 0.0);
        let length_values_valid = [
            self.segment_length_error.mean_absolute_error,
            self.segment_length_error.mean_absolute_relative_error,
            self.segment_length_error.p95_absolute_error,
        ]
        .into_iter()
        .flatten()
        .all(|value| value.is_finite() && value >= 0.0);
        let localization_is_empty = self.localization_error.matched_junction_count == 0;
        let length_is_empty = self.segment_length_error.matched_segment_count == 0;
        if self.schema_version != MYCELIAL_RECONSTRUCTION_SCORE_SCHEMA_VERSION
            || !valid_id(&self.frame_id)
            || !valid_id(&self.coordinate_frame_id)
            || !valid_digest(&self.calibration_digest)
            || !valid_digest(&self.source_observation_digest)
            || !valid_id(&self.matching_policy_id)
            || !self.maximum_junction_match_distance.is_finite()
            || self.maximum_junction_match_distance <= 0.0
            || self.maximum_junction_match_distance > MAX_RECONSTRUCTION_COORDINATE
            || !self.length_relative_epsilon.is_finite()
            || !(1.0e-15..=1.0e-3).contains(&self.length_relative_epsilon)
            || !valid_digest(&self.ground_truth_digest)
            || !valid_digest(&self.prediction_digest)
            || self
                .identity_matches
                .windows(2)
                .any(|pair| pair[0].predicted_junction_id >= pair[1].predicted_junction_id)
            || self.junction_kind_scores.len() != ANNOTATED_JUNCTION_KINDS.len()
            || self
                .junction_kind_scores
                .iter()
                .zip(ANNOTATED_JUNCTION_KINDS)
                .any(|(score, expected)| score.kind != expected || score.f1 != score.counts.f1())
            || self.junction_kind_macro_f1 != expected_macro_f1
            || class_true_positive + class_false_positive != self.prediction_topology.junction_count
            || class_true_positive + class_false_negative
                != self.ground_truth_topology.junction_count
            || self.false_anastomosis_junction_count
                != self
                    .junction_kind_scores
                    .iter()
                    .find(|score| score.kind == AnnotatedJunctionKind::Anastomosis)
                    .map(|score| score.counts.false_positive)
                    .unwrap_or(0)
            || self.junction_detection.true_positive != self.identity_matches.len()
            || self.junction_detection.false_positive + self.junction_detection.true_positive
                != self.prediction_topology.junction_count
            || self.junction_detection.false_negative + self.junction_detection.true_positive
                != self.ground_truth_topology.junction_count
            || self.localization_error.matched_junction_count != self.identity_matches.len()
            || (localization_is_empty != self.localization_error.mean_error.is_none()
                || localization_is_empty != self.localization_error.median_error.is_none()
                || localization_is_empty != self.localization_error.p95_error.is_none()
                || localization_is_empty != self.localization_error.maximum_error.is_none())
            || !localization_values_valid
            || self.edge_topology.true_positive + self.edge_topology.false_positive
                != self.prediction_topology.segment_count
            || self.edge_topology.true_positive + self.edge_topology.false_negative
                != self.ground_truth_topology.segment_count
            || self.unmatched_prediction_edge_count > self.edge_topology.false_positive
            || self.segment_length_error.matched_segment_count != self.edge_topology.true_positive
            || (length_is_empty != self.segment_length_error.mean_absolute_error.is_none()
                || length_is_empty
                    != self
                        .segment_length_error
                        .mean_absolute_relative_error
                        .is_none()
                || length_is_empty != self.segment_length_error.p95_absolute_error.is_none())
            || !length_values_valid
            || self.topology_delta != expected_topology_delta
            || summarize_component_attributions(&self.component_attributions)
                .as_ref() != Some(&self.component_error_summary)
            || self.component_error_summary.matched_junction_count
                != self.identity_matches.len()
            || self.component_error_summary.matched_prediction_component_count
                + self.component_error_summary.unmatched_prediction_component_count
                != self.prediction_topology.connected_components
            || self.component_error_summary.matched_ground_truth_component_count
                + self.component_error_summary.unmatched_ground_truth_component_count
                != self.ground_truth_topology.connected_components
        {
            return Err(MycelialReconstructionError::InvalidGraph(
                "reconstruction score receipt is internally inconsistent",
            ));
        }
        Ok(())
    }

    pub fn canonical_digest_hex(&self) -> Result<String, MycelialReconstructionError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-mycelial-reconstruction-score-v1");
        hasher.update(&self.schema_version.to_le_bytes());
        hash_string(&mut hasher, &self.frame_id);
        hash_string(&mut hasher, &self.coordinate_frame_id);
        hasher.update(self.calibration_digest.as_bytes());
        hasher.update(self.source_observation_digest.as_bytes());
        hasher.update(&[match self.dimensionality {
            ReconstructionDimensionality::TwoD => 0,
            ReconstructionDimensionality::ThreeD => 1,
        }]);
        hasher.update(&[match self.length_unit {
            ReconstructionLengthUnit::Pixel => 0,
            ReconstructionLengthUnit::Micrometer => 1,
            ReconstructionLengthUnit::Millimeter => 2,
            ReconstructionLengthUnit::OrganismLocalModelUnit => 3,
        }]);
        hash_string(&mut hasher, &self.matching_policy_id);
        hasher.update(&self.maximum_junction_match_distance.to_bits().to_le_bytes());
        hasher.update(&self.length_relative_epsilon.to_bits().to_le_bytes());
        hasher.update(self.ground_truth_digest.as_bytes());
        hasher.update(self.prediction_digest.as_bytes());
        for matched in &self.identity_matches {
            hash_string(&mut hasher, &matched.predicted_junction_id);
            hash_string(&mut hasher, &matched.ground_truth_junction_id);
        }
        hash_counts(&mut hasher, &self.junction_detection);
        for score in &self.junction_kind_scores {
            hasher.update(&[junction_kind_index(score.kind) as u8]);
            hash_counts(&mut hasher, &score.counts);
        }
        hash_option_f64(&mut hasher, self.junction_kind_macro_f1);
        hasher.update(&(self.false_anastomosis_junction_count as u64).to_le_bytes());
        hash_counts(&mut hasher, &self.edge_topology);
        hasher.update(&(self.unmatched_prediction_edge_count as u64).to_le_bytes());
        hash_topology(&mut hasher, &self.ground_truth_topology);
        hash_topology(&mut hasher, &self.prediction_topology);
        for value in [
            self.component_error_summary.matched_junction_count,
            self.component_error_summary.matched_prediction_component_count,
            self.component_error_summary.matched_ground_truth_component_count,
            self.component_error_summary.false_merge_count,
            self.component_error_summary.false_split_count,
            self.component_error_summary.unmatched_prediction_component_count,
            self.component_error_summary.unmatched_ground_truth_component_count,
        ] {
            hasher.update(&(value as u64).to_le_bytes());
        }
        for attribution in &self.component_attributions {
            hash_optional_string(&mut hasher, attribution.predicted_component_id.as_deref());
            hash_optional_string(&mut hasher, attribution.ground_truth_component_id.as_deref());
            hasher.update(&(attribution.matched_junction_count as u64).to_le_bytes());
        }
        hash_error_summary(&mut hasher, &self.localization_error);
        hash_option_f64(&mut hasher, self.segment_length_error.mean_absolute_error);
        hash_option_f64(
            &mut hasher,
            self.segment_length_error.mean_absolute_relative_error,
        );
        hash_option_f64(&mut hasher, self.segment_length_error.p95_absolute_error);
        hasher.update(&(self.segment_length_error.matched_segment_count as u64).to_le_bytes());
        Ok(hex_digest(hasher.finalize().as_bytes()))
    }
}

fn evaluate_edges(
    ground_truth: &AnnotatedMycelialSkeleton,
    prediction: &AnnotatedMycelialSkeleton,
    prediction_to_truth: &BTreeMap<&str, &str>,
    relative_epsilon: f64,
) -> (ReconstructionCounts, usize, ReconstructionLengthError) {
    let mut truth_edges = BTreeMap::<(String, String), Vec<f64>>::new();
    for segment in &ground_truth.segments {
        let key = canonical_edge(&segment.start_junction_id, &segment.end_junction_id);
        truth_edges.entry(key).or_default().push(segment.length);
    }
    let mut prediction_edges = BTreeMap::<(String, String), Vec<f64>>::new();
    let mut unmatched_prediction_edge_count = 0usize;
    for segment in &prediction.segments {
        let Some(start) = prediction_to_truth.get(segment.start_junction_id.as_str()) else {
            unmatched_prediction_edge_count += 1;
            continue;
        };
        let Some(end) = prediction_to_truth.get(segment.end_junction_id.as_str()) else {
            unmatched_prediction_edge_count += 1;
            continue;
        };
        if start == end {
            unmatched_prediction_edge_count += 1;
            continue;
        }
        let key = canonical_edge(start, end);
        prediction_edges
            .entry(key)
            .or_default()
            .push(segment.length);
    }

    for values in truth_edges.values_mut() {
        values.sort_by(f64::total_cmp);
    }
    for values in prediction_edges.values_mut() {
        values.sort_by(f64::total_cmp);
    }

    let mut true_positive = 0usize;
    let mut false_positive = unmatched_prediction_edge_count;
    let mut false_negative = 0usize;
    let mut absolute_errors = Vec::new();
    let mut relative_errors = Vec::new();

    let all_keys: BTreeSet<_> = truth_edges
        .keys()
        .chain(prediction_edges.keys())
        .cloned()
        .collect();
    for key in all_keys {
        let truth_lengths = truth_edges.get(&key).map(Vec::as_slice).unwrap_or_default();
        let predicted_lengths = prediction_edges
            .get(&key)
            .map(Vec::as_slice)
            .unwrap_or_default();
        let matched = truth_lengths.len().min(predicted_lengths.len());
        true_positive += matched;
        false_positive += predicted_lengths.len().saturating_sub(matched);
        false_negative += truth_lengths.len().saturating_sub(matched);
        for (truth_length, predicted_length) in truth_lengths
            .iter()
            .zip(predicted_lengths.iter())
            .take(matched)
        {
            let absolute = (truth_length - predicted_length).abs();
            absolute_errors.push(absolute);
            relative_errors.push(absolute / truth_length.abs().max(relative_epsilon));
        }
    }
    let edge_counts = ReconstructionCounts {
        true_positive,
        false_positive,
        false_negative,
    };
    (
        edge_counts,
        unmatched_prediction_edge_count,
        ReconstructionLengthError {
            matched_segment_count: absolute_errors.len(),
            mean_absolute_error: mean(&absolute_errors),
            mean_absolute_relative_error: mean(&relative_errors),
            p95_absolute_error: percentile(&absolute_errors, 0.95),
        },
    )
}

fn topology_summary(graph: &AnnotatedMycelialSkeleton) -> ReconstructionTopologySummary {
    let mut adjacency = BTreeMap::<&str, Vec<&str>>::new();
    for junction in &graph.junctions {
        adjacency.insert(junction.id.as_str(), Vec::new());
    }
    for segment in &graph.segments {
        if let Some(neighbours) = adjacency.get_mut(segment.start_junction_id.as_str()) {
            neighbours.push(segment.end_junction_id.as_str());
        }
        if let Some(neighbours) = adjacency.get_mut(segment.end_junction_id.as_str()) {
            neighbours.push(segment.start_junction_id.as_str());
        }
    }
    let mut visited = BTreeSet::new();
    let mut components = 0usize;
    for start in adjacency.keys().copied() {
        if !visited.insert(start) {
            continue;
        }
        components += 1;
        let mut queue = VecDeque::from([start]);
        while let Some(current) = queue.pop_front() {
            if let Some(neighbours) = adjacency.get(current) {
                for neighbour in neighbours {
                    if visited.insert(*neighbour) {
                        queue.push_back(*neighbour);
                    }
                }
            }
        }
    }
    let cycle_rank = graph
        .segments
        .len()
        .saturating_add(components)
        .saturating_sub(graph.junctions.len());
    ReconstructionTopologySummary {
        junction_count: graph.junctions.len(),
        segment_count: graph.segments.len(),
        connected_components: components,
        cycle_rank,
    }
}

fn component_membership(
    graph: &AnnotatedMycelialSkeleton,
) -> BTreeMap<String, String> {
    let mut adjacency = BTreeMap::<String, Vec<String>>::new();
    for junction in &graph.junctions {
        adjacency.insert(junction.id.clone(), Vec::new());
    }
    for segment in &graph.segments {
        adjacency
            .entry(segment.start_junction_id.clone())
            .or_default()
            .push(segment.end_junction_id.clone());
        adjacency
            .entry(segment.end_junction_id.clone())
            .or_default()
            .push(segment.start_junction_id.clone());
    }
    for neighbours in adjacency.values_mut() {
        neighbours.sort();
        neighbours.dedup();
    }
    let mut membership = BTreeMap::new();
    let mut visited = BTreeSet::new();
    for start in adjacency.keys() {
        if !visited.insert(start.clone()) {
            continue;
        }
        let mut queue = VecDeque::from([start.clone()]);
        let mut members = Vec::new();
        while let Some(current) = queue.pop_front() {
            members.push(current.clone());
            if let Some(neighbours) = adjacency.get(&current) {
                for neighbour in neighbours {
                    if visited.insert(neighbour.clone()) {
                        queue.push_back(neighbour.clone());
                    }
                }
            }
        }
        let representative = members.iter().min().cloned().unwrap_or_else(|| start.clone());
        for member in members {
            membership.insert(member, representative.clone());
        }
    }
    membership
}

fn component_attribution(
    ground_truth: &AnnotatedMycelialSkeleton,
    prediction: &AnnotatedMycelialSkeleton,
    matches: &[JunctionIdentityMatch],
) -> (
    Vec<ReconstructionComponentAttribution>,
    ReconstructionComponentErrorSummary,
) {
    let prediction_membership = component_membership(prediction);
    let truth_membership = component_membership(ground_truth);
    let mut matched_pairs = BTreeMap::<(String, String), usize>::new();
    for matched in matches {
        if let (Some(predicted_component), Some(truth_component)) = (
            prediction_membership.get(&matched.predicted_junction_id),
            truth_membership.get(&matched.ground_truth_junction_id),
        ) {
            *matched_pairs
                .entry((predicted_component.clone(), truth_component.clone()))
                .or_default() += 1;
        }
    }

    let all_prediction_components: BTreeSet<_> =
        prediction_membership.values().cloned().collect();
    let all_truth_components: BTreeSet<_> = truth_membership.values().cloned().collect();
    let mut truth_by_prediction = BTreeMap::<String, BTreeSet<String>>::new();
    let mut prediction_by_truth = BTreeMap::<String, BTreeSet<String>>::new();
    let mut attributions = Vec::new();

    for ((predicted_component, truth_component), matched_junction_count) in matched_pairs {
        truth_by_prediction
            .entry(predicted_component.clone())
            .or_default()
            .insert(truth_component.clone());
        prediction_by_truth
            .entry(truth_component.clone())
            .or_default()
            .insert(predicted_component.clone());
        attributions.push(ReconstructionComponentAttribution {
            predicted_component_id: Some(predicted_component),
            ground_truth_component_id: Some(truth_component),
            matched_junction_count,
        });
    }
    for component_id in &all_prediction_components {
        if !truth_by_prediction.contains_key(component_id) {
            attributions.push(ReconstructionComponentAttribution {
                predicted_component_id: Some(component_id.clone()),
                ground_truth_component_id: None,
                matched_junction_count: 0,
            });
        }
    }
    for component_id in &all_truth_components {
        if !prediction_by_truth.contains_key(component_id) {
            attributions.push(ReconstructionComponentAttribution {
                predicted_component_id: None,
                ground_truth_component_id: Some(component_id.clone()),
                matched_junction_count: 0,
            });
        }
    }
    attributions.sort_by(|left, right| {
        left.predicted_component_id
            .cmp(&right.predicted_component_id)
            .then_with(|| {
                left.ground_truth_component_id
                    .cmp(&right.ground_truth_component_id)
            })
    });

    let matched_prediction_component_count = truth_by_prediction.len();
    let matched_ground_truth_component_count = prediction_by_truth.len();
    let false_merge_count = truth_by_prediction
        .values()
        .map(|components| components.len().saturating_sub(1))
        .sum();
    let false_split_count = prediction_by_truth
        .values()
        .map(|components| components.len().saturating_sub(1))
        .sum();
    let unmatched_prediction_component_count = all_prediction_components
        .len()
        .saturating_sub(matched_prediction_component_count);
    let unmatched_ground_truth_component_count = all_truth_components
        .len()
        .saturating_sub(matched_ground_truth_component_count);
    (
        attributions,
        ReconstructionComponentErrorSummary {
            matched_junction_count: matches.len(),
            matched_prediction_component_count,
            matched_ground_truth_component_count,
            false_merge_count,
            false_split_count,
            unmatched_prediction_component_count,
            unmatched_ground_truth_component_count,
        },
    )
}

fn summarize_component_attributions(
    rows: &[ReconstructionComponentAttribution],
) -> Option<ReconstructionComponentErrorSummary> {
    let mut truth_by_prediction = BTreeMap::<String, BTreeSet<String>>::new();
    let mut prediction_by_truth = BTreeMap::<String, BTreeSet<String>>::new();
    let mut unmatched_predictions = BTreeSet::new();
    let mut unmatched_truth = BTreeSet::new();
    let mut matched_junction_count = 0usize;
    let mut relation_pairs = BTreeSet::new();

    for row in rows {
        match (
            row.predicted_component_id.as_deref(),
            row.ground_truth_component_id.as_deref(),
        ) {
            (Some(predicted), Some(truth)) => {
                if !valid_id(predicted)
                    || !valid_id(truth)
                    || row.matched_junction_count == 0
                    || !relation_pairs.insert((predicted.to_string(), truth.to_string()))
                {
                    return None;
                }
                truth_by_prediction
                    .entry(predicted.to_string())
                    .or_default()
                    .insert(truth.to_string());
                prediction_by_truth
                    .entry(truth.to_string())
                    .or_default()
                    .insert(predicted.to_string());
                matched_junction_count =
                    matched_junction_count.checked_add(row.matched_junction_count)?;
            }
            (Some(predicted), None) => {
                if !valid_id(predicted)
                    || row.matched_junction_count != 0
                    || !unmatched_predictions.insert(predicted.to_string())
                {
                    return None;
                }
            }
            (None, Some(truth)) => {
                if !valid_id(truth)
                    || row.matched_junction_count != 0
                    || !unmatched_truth.insert(truth.to_string())
                {
                    return None;
                }
            }
            (None, None) => return None,
        }
    }
    if rows.windows(2).any(|pair| {
        (
            pair[0].predicted_component_id.as_deref(),
            pair[0].ground_truth_component_id.as_deref(),
        ) >= (
            pair[1].predicted_component_id.as_deref(),
            pair[1].ground_truth_component_id.as_deref(),
        )
    }) || unmatched_predictions
        .iter()
        .any(|id| truth_by_prediction.contains_key(id))
        || unmatched_truth
            .iter()
            .any(|id| prediction_by_truth.contains_key(id))
    {
        return None;
    }
    Some(ReconstructionComponentErrorSummary {
        matched_junction_count,
        matched_prediction_component_count: truth_by_prediction.len(),
        matched_ground_truth_component_count: prediction_by_truth.len(),
        false_merge_count: truth_by_prediction
            .values()
            .map(|components| components.len().saturating_sub(1))
            .sum(),
        false_split_count: prediction_by_truth
            .values()
            .map(|components| components.len().saturating_sub(1))
            .sum(),
        unmatched_prediction_component_count: unmatched_predictions.len(),
        unmatched_ground_truth_component_count: unmatched_truth.len(),
    })
}

fn point_distance(
    left: ReconstructionPoint3,
    right: ReconstructionPoint3,
    dimensionality: ReconstructionDimensionality,
) -> f64 {
    // hypot is internally scaled, avoiding underflow/overflow in squared
    // coordinate differences, even at very small physical unit scales.
    let planar = (left.x - right.x).hypot(left.y - right.y);
    if dimensionality == ReconstructionDimensionality::ThreeD {
        planar.hypot(left.z - right.z)
    } else {
        planar
    }
}

fn summarize_errors(mut errors: Vec<f64>) -> ReconstructionLocalizationError {
    errors.sort_by(f64::total_cmp);
    ReconstructionLocalizationError {
        matched_junction_count: errors.len(),
        mean_error: mean(&errors),
        median_error: percentile(&errors, 0.5),
        p95_error: percentile(&errors, 0.95),
        maximum_error: errors.last().copied(),
    }
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then_some(values.iter().sum::<f64>() / values.len() as f64)
}

fn percentile(sorted_values: &[f64], percentile: f64) -> Option<f64> {
    if sorted_values.is_empty() {
        return None;
    }
    let rank = ((sorted_values.len() - 1) as f64 * percentile).ceil() as usize;
    sorted_values.get(rank).copied()
}

fn signed_difference(predicted: usize, truth: usize) -> i64 {
    predicted as i64 - truth as i64
}

fn canonical_edge(left: &str, right: &str) -> (String, String) {
    if left <= right {
        (left.to_string(), right.to_string())
    } else {
        (right.to_string(), left.to_string())
    }
}

fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_RECONSTRUCTION_ID_BYTES
        && !value.chars().any(char::is_control)
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn junction_kind_index(kind: AnnotatedJunctionKind) -> usize {
    match kind {
        AnnotatedJunctionKind::Origin => 0,
        AnnotatedJunctionKind::Branch => 1,
        AnnotatedJunctionKind::Tip => 2,
        AnnotatedJunctionKind::Anastomosis => 3,
        AnnotatedJunctionKind::ProjectedCrossing => 4,
        AnnotatedJunctionKind::Uncertain => 5,
    }
}

fn hash_optional_string(hasher: &mut blake3::Hasher, value: Option<&str>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            hash_string(hasher, value);
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn hash_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hash_counts(hasher: &mut blake3::Hasher, counts: &ReconstructionCounts) {
    hasher.update(&(counts.true_positive as u64).to_le_bytes());
    hasher.update(&(counts.false_positive as u64).to_le_bytes());
    hasher.update(&(counts.false_negative as u64).to_le_bytes());
}

fn hash_topology(hasher: &mut blake3::Hasher, summary: &ReconstructionTopologySummary) {
    for value in [
        summary.junction_count,
        summary.segment_count,
        summary.connected_components,
        summary.cycle_rank,
    ] {
        hasher.update(&(value as u64).to_le_bytes());
    }
}

fn hash_error_summary(hasher: &mut blake3::Hasher, summary: &ReconstructionLocalizationError) {
    hasher.update(&(summary.matched_junction_count as u64).to_le_bytes());
    hash_option_f64(hasher, summary.mean_error);
    hash_option_f64(hasher, summary.median_error);
    hash_option_f64(hasher, summary.p95_error);
    hash_option_f64(hasher, summary.maximum_error);
}

fn hash_option_f64(hasher: &mut blake3::Hasher, value: Option<f64>) {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn source_digest() -> String {
        "a".repeat(64)
    }

    fn graph(
        producer: &str,
        nodes: Vec<AnnotatedJunction>,
        edges: Vec<AnnotatedHyphaSegment>,
    ) -> AnnotatedMycelialSkeleton {
        AnnotatedMycelialSkeleton {
            schema_version: MYCELIAL_RECONSTRUCTION_SCHEMA_VERSION,
            frame_id: "frame:001".to_string(),
            coordinate_frame_id: "image-frame:001".to_string(),
            calibration_digest: "c".repeat(64),
            source_observation_digest: source_digest(),
            dimensionality: ReconstructionDimensionality::ThreeD,
            length_unit: ReconstructionLengthUnit::Micrometer,
            producer_id: producer.to_string(),
            producer_version: "v1".to_string(),
            junctions: nodes,
            segments: edges,
        }
    }

    fn node(id: &str, kind: AnnotatedJunctionKind, x: f64, y: f64, z: f64) -> AnnotatedJunction {
        AnnotatedJunction {
            id: id.to_string(),
            kind,
            position: ReconstructionPoint3 { x, y, z },
        }
    }

    fn edge(id: &str, start: &str, end: &str, length: f64) -> AnnotatedHyphaSegment {
        AnnotatedHyphaSegment {
            id: id.to_string(),
            start_junction_id: start.to_string(),
            end_junction_id: end.to_string(),
            length,
        }
    }

    #[test]
    fn hungarian_matching_preserves_cardinality_when_greedy_would_lose_a_match() {
        let truth = graph(
            "human",
            vec![
                node("A", AnnotatedJunctionKind::Branch, 0.0, 0.0, 0.0),
                node("B", AnnotatedJunctionKind::Branch, 2.0, 0.0, 0.0),
            ],
            Vec::new(),
        );
        let prediction = graph(
            "model",
            vec![
                node("p1", AnnotatedJunctionKind::Branch, 0.9, 0.0, 0.0),
                node("p2", AnnotatedJunctionKind::Branch, -0.9, 0.0, 0.0),
            ],
            Vec::new(),
        );
        // A greedy nearest-neighbour pass taking p1->A first leaves p2
        // unmatched. A global assignment must choose p1->B and p2->A.
        let matches = match_junctions_hungarian(&truth, &prediction, 2.0).unwrap();
        assert_eq!(matches.len(), 2);
        assert_eq!(
            matches,
            vec![
                JunctionIdentityMatch {
                    predicted_junction_id: "p1".to_string(),
                    ground_truth_junction_id: "B".to_string(),
                },
                JunctionIdentityMatch {
                    predicted_junction_id: "p2".to_string(),
                    ground_truth_junction_id: "A".to_string(),
                },
            ]
        );
    }

    #[test]
    fn hungarian_matching_uses_cardinality_before_total_distance() {
        let truth = graph(
            "human",
            vec![
                node("A", AnnotatedJunctionKind::Branch, 0.0, 0.0, 0.0),
                node("B", AnnotatedJunctionKind::Branch, 10.0, 0.0, 0.0),
            ],
            Vec::new(),
        );
        let prediction = graph(
            "model",
            vec![
                node("p1", AnnotatedJunctionKind::Branch, 0.0, 0.0, 0.0),
                node("p2", AnnotatedJunctionKind::Branch, -10.0, 0.0, 0.0),
            ],
            Vec::new(),
        );
        // The one-match solution p1->A has zero localization error. The only
        // two-match solution costs 20 units (p1->B, p2->A). Cardinality must
        // win before total distance, otherwise a small dummy penalty picks 1.
        let matches = match_junctions_hungarian(&truth, &prediction, 10.0).unwrap();
        assert_eq!(matches.len(), 2);
        assert_eq!(
            matches,
            vec![
                JunctionIdentityMatch {
                    predicted_junction_id: "p1".to_string(),
                    ground_truth_junction_id: "B".to_string(),
                },
                JunctionIdentityMatch {
                    predicted_junction_id: "p2".to_string(),
                    ground_truth_junction_id: "A".to_string(),
                },
            ]
        );
    }

    #[test]
    fn hungarian_matching_is_stable_for_subnormal_coordinate_units() {
        let truth = graph(
            "human",
            vec![node("A", AnnotatedJunctionKind::Tip, 5.0e-321, 0.0, 0.0)],
            Vec::new(),
        );
        let prediction = graph(
            "model",
            vec![node("pA", AnnotatedJunctionKind::Tip, 0.0, 0.0, 0.0)],
            Vec::new(),
        );
        let matches = match_junctions_hungarian(&truth, &prediction, 1.0e-320).unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].predicted_junction_id, "pA");
        assert_eq!(matches[0].ground_truth_junction_id, "A");
    }

    #[test]
    fn hungarian_matching_is_order_independent_and_respects_distance_cutoff() {
        let truth = graph(
            "human",
            vec![
                node("A", AnnotatedJunctionKind::Tip, 0.0, 0.0, 0.0),
                node("B", AnnotatedJunctionKind::Tip, 4.0, 0.0, 0.0),
            ],
            Vec::new(),
        );
        let prediction = graph(
            "model",
            vec![
                node("pA", AnnotatedJunctionKind::Tip, 0.1, 0.0, 0.0),
                node("pB", AnnotatedJunctionKind::Tip, 4.1, 0.0, 0.0),
            ],
            Vec::new(),
        );
        let first = match_junctions_hungarian(&truth, &prediction, 0.5).unwrap();
        let mut reordered_truth = truth.clone();
        reordered_truth.junctions.reverse();
        let mut reordered_prediction = prediction.clone();
        reordered_prediction.junctions.reverse();
        let second =
            match_junctions_hungarian(&reordered_truth, &reordered_prediction, 0.5).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 2);

        let distant = graph(
            "model",
            vec![node("far", AnnotatedJunctionKind::Tip, 10.0, 0.0, 0.0)],
            Vec::new(),
        );
        assert!(
            match_junctions_hungarian(&truth, &distant, 0.5)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn hungarian_matching_fails_closed_above_declared_size_limit() {
        let truth_nodes: Vec<_> = (0..=MAX_AUTOMATIC_MATCH_JUNCTIONS_PER_GRAPH)
            .map(|index| {
                node(
                    &format!("g{index:03}"),
                    AnnotatedJunctionKind::Tip,
                    index as f64,
                    0.0,
                    0.0,
                )
            })
            .collect();
        let prediction_nodes = truth_nodes
            .iter()
            .map(|value| {
                node(
                    &format!("p{}", value.id),
                    value.kind,
                    value.position.x,
                    value.position.y,
                    value.position.z,
                )
            })
            .collect();
        let truth = graph("human", truth_nodes, Vec::new());
        let prediction = graph("model", prediction_nodes, Vec::new());
        assert!(matches!(
            match_junctions_hungarian(&truth, &prediction, 1.0),
            Err(MycelialReconstructionError::AutomaticMatchingLimitExceeded { .. })
        ));
    }

    #[test]
    fn perfect_reconstruction_scores_all_graph_layers_as_exact() {
        let truth = graph(
            "human_annotation",
            vec![
                node("A", AnnotatedJunctionKind::Branch, 0.0, 0.0, 0.0),
                node("B", AnnotatedJunctionKind::Tip, 1.0, 0.0, 0.0),
                node("C", AnnotatedJunctionKind::ProjectedCrossing, 0.0, 1.0, 0.0),
            ],
            vec![edge("e1", "A", "B", 1.0), edge("e2", "A", "C", 1.0)],
        );
        let prediction = graph(
            "symthaea-candidate",
            vec![
                node("pA", AnnotatedJunctionKind::Branch, 0.01, 0.0, 0.0),
                node("pB", AnnotatedJunctionKind::Tip, 1.01, 0.0, 0.0),
                node("pC", AnnotatedJunctionKind::Tip, 0.0, 1.01, 0.0),
            ],
            vec![edge("s1", "pA", "pB", 1.0), edge("s2", "pA", "pC", 1.0)],
        );
        let matches = vec![
            JunctionIdentityMatch {
                predicted_junction_id: "pA".to_string(),
                ground_truth_junction_id: "A".to_string(),
            },
            JunctionIdentityMatch {
                predicted_junction_id: "pB".to_string(),
                ground_truth_junction_id: "B".to_string(),
            },
            JunctionIdentityMatch {
                predicted_junction_id: "pC".to_string(),
                ground_truth_junction_id: "C".to_string(),
            },
        ];
        let score = evaluate_mycelial_reconstruction(
            &truth,
            &prediction,
            "synthetic-explicit-match-v1",
            &matches,
            ReconstructionEvaluationConfig {
                maximum_junction_match_distance: 0.05,
                ..ReconstructionEvaluationConfig::default()
            },
        )
        .unwrap();
        assert_eq!(score.junction_detection.f1(), Some(1.0));
        assert_eq!(score.edge_topology.f1(), Some(1.0));
        assert_eq!(score.junction_kind_macro_f1, Some(1.0));
        assert_eq!(score.false_anastomosis_junction_count, 0);
        assert_eq!(score.topology_delta.cycle_rank, 0);
        assert!(score.localization_error.mean_error.unwrap() <= 0.01);
        assert_eq!(
            score.canonical_digest_hex().unwrap(),
            score.canonical_digest_hex().unwrap()
        );
    }

    #[test]
    fn false_crossing_fusion_is_counted_as_edge_and_anastomosis_false_positive() {
        let truth = graph(
            "human_annotation",
            vec![
                node("A", AnnotatedJunctionKind::Branch, 0.0, 0.0, 0.0),
                node("B", AnnotatedJunctionKind::Tip, 1.0, 0.0, 0.0),
                node("C", AnnotatedJunctionKind::Tip, 0.0, 1.0, 0.0),
            ],
            vec![edge("e1", "A", "B", 1.0), edge("e2", "A", "C", 1.0)],
        );
        let prediction = graph(
            "symthaea-candidate",
            vec![
                node("pA", AnnotatedJunctionKind::Branch, 0.0, 0.0, 0.0),
                node("pB", AnnotatedJunctionKind::Tip, 1.0, 0.0, 0.0),
                node("pC", AnnotatedJunctionKind::Anastomosis, 0.0, 1.0, 0.0),
            ],
            vec![
                edge("s1", "pA", "pB", 1.0),
                edge("s2", "pA", "pC", 1.0),
                edge("false-fusion", "pB", "pC", 1.414),
            ],
        );
        let matches = vec![
            JunctionIdentityMatch {
                predicted_junction_id: "pA".to_string(),
                ground_truth_junction_id: "A".to_string(),
            },
            JunctionIdentityMatch {
                predicted_junction_id: "pB".to_string(),
                ground_truth_junction_id: "B".to_string(),
            },
            JunctionIdentityMatch {
                predicted_junction_id: "pC".to_string(),
                ground_truth_junction_id: "C".to_string(),
            },
        ];
        let score = evaluate_mycelial_reconstruction(
            &truth,
            &prediction,
            "explicit-match-v1",
            &matches,
            ReconstructionEvaluationConfig::default(),
        )
        .unwrap();
        assert_eq!(score.edge_topology.true_positive, 2);
        assert_eq!(score.edge_topology.false_positive, 1);
        assert_eq!(score.edge_topology.false_negative, 0);
        assert_eq!(score.false_anastomosis_junction_count, 1);
        assert_eq!(score.topology_delta.cycle_rank, 1);
    }

    #[test]
    fn component_attribution_counts_a_false_merge_even_when_all_nodes_match() {
        let truth = graph(
            "human",
            vec![
                node("A", AnnotatedJunctionKind::Origin, 0.0, 0.0, 0.0),
                node("B", AnnotatedJunctionKind::Tip, 1.0, 0.0, 0.0),
                node("C", AnnotatedJunctionKind::Origin, 10.0, 0.0, 0.0),
                node("D", AnnotatedJunctionKind::Tip, 11.0, 0.0, 0.0),
            ],
            vec![edge("gt-ab", "A", "B", 1.0), edge("gt-cd", "C", "D", 1.0)],
        );
        let prediction = graph(
            "model",
            vec![
                node("pA", AnnotatedJunctionKind::Origin, 0.0, 0.0, 0.0),
                node("pB", AnnotatedJunctionKind::Tip, 1.0, 0.0, 0.0),
                node("pC", AnnotatedJunctionKind::Origin, 10.0, 0.0, 0.0),
                node("pD", AnnotatedJunctionKind::Tip, 11.0, 0.0, 0.0),
            ],
            vec![
                edge("p-ab", "pA", "pB", 1.0),
                edge("false-merge", "pB", "pC", 9.0),
                edge("p-cd", "pC", "pD", 1.0),
            ],
        );
        let matches = vec![
            JunctionIdentityMatch {
                predicted_junction_id: "pA".to_string(),
                ground_truth_junction_id: "A".to_string(),
            },
            JunctionIdentityMatch {
                predicted_junction_id: "pB".to_string(),
                ground_truth_junction_id: "B".to_string(),
            },
            JunctionIdentityMatch {
                predicted_junction_id: "pC".to_string(),
                ground_truth_junction_id: "C".to_string(),
            },
            JunctionIdentityMatch {
                predicted_junction_id: "pD".to_string(),
                ground_truth_junction_id: "D".to_string(),
            },
        ];
        let score = evaluate_mycelial_reconstruction(
            &truth,
            &prediction,
            "explicit-match-v1",
            &matches,
            ReconstructionEvaluationConfig::default(),
        )
        .unwrap();
        assert_eq!(score.ground_truth_topology.connected_components, 2);
        assert_eq!(score.prediction_topology.connected_components, 1);
        assert_eq!(score.component_error_summary.false_merge_count, 1);
        assert_eq!(score.component_error_summary.false_split_count, 0);
        assert_eq!(score.component_error_summary.matched_junction_count, 4);
        assert_eq!(score.edge_topology.false_positive, 1);
        score.validate().unwrap();
    }

    #[test]
    fn component_attribution_counts_a_false_split_into_disconnected_prediction_parts() {
        let truth = graph(
            "human",
            vec![
                node("A", AnnotatedJunctionKind::Branch, 0.0, 0.0, 0.0),
                node("B", AnnotatedJunctionKind::Branch, 1.0, 0.0, 0.0),
                node("C", AnnotatedJunctionKind::Tip, 2.0, 0.0, 0.0),
            ],
            vec![edge("gt-ab", "A", "B", 1.0), edge("gt-bc", "B", "C", 1.0)],
        );
        let prediction = graph(
            "model",
            vec![
                node("pA", AnnotatedJunctionKind::Branch, 0.0, 0.0, 0.0),
                node("pB", AnnotatedJunctionKind::Branch, 1.0, 0.0, 0.0),
                node("pC", AnnotatedJunctionKind::Tip, 2.0, 0.0, 0.0),
            ],
            vec![edge("p-ab", "pA", "pB", 1.0)],
        );
        let matches = vec![
            JunctionIdentityMatch {
                predicted_junction_id: "pA".to_string(),
                ground_truth_junction_id: "A".to_string(),
            },
            JunctionIdentityMatch {
                predicted_junction_id: "pB".to_string(),
                ground_truth_junction_id: "B".to_string(),
            },
            JunctionIdentityMatch {
                predicted_junction_id: "pC".to_string(),
                ground_truth_junction_id: "C".to_string(),
            },
        ];
        let score = evaluate_mycelial_reconstruction(
            &truth,
            &prediction,
            "explicit-match-v1",
            &matches,
            ReconstructionEvaluationConfig::default(),
        )
        .unwrap();
        assert_eq!(score.ground_truth_topology.connected_components, 1);
        assert_eq!(score.prediction_topology.connected_components, 2);
        assert_eq!(score.component_error_summary.false_merge_count, 0);
        assert_eq!(score.component_error_summary.false_split_count, 1);
        assert_eq!(score.edge_topology.false_negative, 1);
    }

    #[test]
    fn component_attribution_separates_unmatched_components_from_splits_and_merges() {
        let truth = graph(
            "human",
            vec![
                node("A", AnnotatedJunctionKind::Tip, 0.0, 0.0, 0.0),
                node("B", AnnotatedJunctionKind::Tip, 1.0, 0.0, 0.0),
            ],
            Vec::new(),
        );
        let prediction = graph(
            "model",
            vec![
                node("pA", AnnotatedJunctionKind::Tip, 0.0, 0.0, 0.0),
                node("pB", AnnotatedJunctionKind::Tip, 1.0, 0.0, 0.0),
                node("pC", AnnotatedJunctionKind::Tip, 2.0, 0.0, 0.0),
            ],
            Vec::new(),
        );
        let matches = vec![
            JunctionIdentityMatch {
                predicted_junction_id: "pA".to_string(),
                ground_truth_junction_id: "A".to_string(),
            },
            JunctionIdentityMatch {
                predicted_junction_id: "pB".to_string(),
                ground_truth_junction_id: "B".to_string(),
            },
        ];
        let score = evaluate_mycelial_reconstruction(
            &truth,
            &prediction,
            "explicit-match-v1",
            &matches,
            ReconstructionEvaluationConfig::default(),
        )
        .unwrap();
        assert_eq!(
            score.component_error_summary.unmatched_prediction_component_count,
            1
        );
        assert_eq!(
            score.component_error_summary.unmatched_ground_truth_component_count,
            0
        );
        assert_eq!(score.component_error_summary.false_merge_count, 0);
        assert_eq!(score.component_error_summary.false_split_count, 0);
        let digest = score.canonical_digest_hex().unwrap();
        let mut tampered = score;
        tampered.component_error_summary.false_merge_count = 1;
        assert!(tampered.validate().is_err());
        assert_ne!(tampered.canonical_digest_hex(), Ok(digest));
    }

    #[test]
    fn mismatched_junction_class_is_separate_from_detection_success() {
        let truth = graph(
            "human_annotation",
            vec![node("A", AnnotatedJunctionKind::Tip, 0.0, 0.0, 0.0)],
            Vec::new(),
        );
        let prediction = graph(
            "candidate",
            vec![node(
                "pA",
                AnnotatedJunctionKind::Anastomosis,
                0.0,
                0.0,
                0.0,
            )],
            Vec::new(),
        );
        let score = evaluate_mycelial_reconstruction(
            &truth,
            &prediction,
            "explicit-match-v1",
            &[JunctionIdentityMatch {
                predicted_junction_id: "pA".to_string(),
                ground_truth_junction_id: "A".to_string(),
            }],
            ReconstructionEvaluationConfig::default(),
        )
        .unwrap();
        assert_eq!(score.junction_detection.f1(), Some(1.0));
        assert_eq!(score.false_anastomosis_junction_count, 1);
        assert_eq!(
            score
                .junction_kind_scores
                .iter()
                .find(|metric| metric.kind == AnnotatedJunctionKind::Tip)
                .unwrap()
                .counts
                .false_negative,
            1
        );
    }

    #[test]
    fn unmatched_nodes_and_edges_are_false_positives_or_false_negatives() {
        let truth = graph(
            "human",
            vec![node("A", AnnotatedJunctionKind::Tip, 0.0, 0.0, 0.0)],
            Vec::new(),
        );
        let prediction = graph(
            "model",
            vec![
                node("pA", AnnotatedJunctionKind::Tip, 0.0, 0.0, 0.0),
                node("pB", AnnotatedJunctionKind::Tip, 2.0, 0.0, 0.0),
            ],
            Vec::new(),
        );
        let score = evaluate_mycelial_reconstruction(
            &truth,
            &prediction,
            "explicit-match-v1",
            &[JunctionIdentityMatch {
                predicted_junction_id: "pA".to_string(),
                ground_truth_junction_id: "A".to_string(),
            }],
            ReconstructionEvaluationConfig::default(),
        )
        .unwrap();
        assert_eq!(score.junction_detection.true_positive, 1);
        assert_eq!(score.junction_detection.false_positive, 1);
        assert_eq!(score.junction_detection.false_negative, 0);
    }

    #[test]
    fn invalid_duplicate_match_and_mismatched_source_frame_fail_closed() {
        let truth = graph(
            "human",
            vec![
                node("A", AnnotatedJunctionKind::Tip, 0.0, 0.0, 0.0),
                node("B", AnnotatedJunctionKind::Tip, 1.0, 0.0, 0.0),
            ],
            Vec::new(),
        );
        let prediction = graph(
            "model",
            vec![
                node("pA", AnnotatedJunctionKind::Tip, 0.0, 0.0, 0.0),
                node("pB", AnnotatedJunctionKind::Tip, 1.0, 0.0, 0.0),
            ],
            Vec::new(),
        );
        let duplicate = vec![
            JunctionIdentityMatch {
                predicted_junction_id: "pA".to_string(),
                ground_truth_junction_id: "A".to_string(),
            },
            JunctionIdentityMatch {
                predicted_junction_id: "pB".to_string(),
                ground_truth_junction_id: "A".to_string(),
            },
        ];
        assert!(matches!(
            evaluate_mycelial_reconstruction(
                &truth,
                &prediction,
                "explicit-match-v1",
                &duplicate,
                ReconstructionEvaluationConfig::default()
            ),
            Err(MycelialReconstructionError::DuplicateGroundTruthMatch(_))
        ));

        let mut wrong_frame = prediction;
        wrong_frame.frame_id = "frame:002".to_string();
        assert!(matches!(
            evaluate_mycelial_reconstruction(
                &truth,
                &wrong_frame,
                "explicit-match-v1",
                &[],
                ReconstructionEvaluationConfig::default()
            ),
            Err(MycelialReconstructionError::FrameMismatch)
        ));
    }

    #[test]
    fn graph_order_permutations_preserve_canonical_digests() {
        let truth = graph(
            "human_annotation",
            vec![
                node("A", AnnotatedJunctionKind::Branch, 0.0, 0.0, 0.0),
                node("B", AnnotatedJunctionKind::Tip, 1.0, 0.0, 0.0),
                node("C", AnnotatedJunctionKind::ProjectedCrossing, 0.0, 1.0, 0.0),
            ],
            vec![edge("e1", "A", "B", 1.0), edge("e2", "A", "C", 1.0)],
        );
        let mut reordered = truth.clone();
        reordered.junctions.reverse();
        reordered.segments.reverse();
        assert_eq!(
            truth.canonical_digest_hex().unwrap(),
            reordered.canonical_digest_hex().unwrap()
        );
    }

    #[test]
    fn score_serialization_roundtrip_preserves_digest_and_validation() {
        let truth = graph(
            "human_annotation",
            vec![
                node("A", AnnotatedJunctionKind::Branch, 0.0, 0.0, 0.0),
                node("B", AnnotatedJunctionKind::Tip, 1.0, 0.0, 0.0),
            ],
            vec![edge("e1", "A", "B", 1.0)],
        );
        let prediction = graph(
            "model",
            vec![
                node("pA", AnnotatedJunctionKind::Branch, 0.01, 0.0, 0.0),
                node("pB", AnnotatedJunctionKind::Tip, 1.01, 0.0, 0.0),
            ],
            vec![edge("s1", "pA", "pB", 1.0)],
        );
        let score = evaluate_mycelial_reconstruction(
            &truth,
            &prediction,
            "explicit-match-v1",
            &[
                JunctionIdentityMatch {
                    predicted_junction_id: "pA".to_string(),
                    ground_truth_junction_id: "A".to_string(),
                },
                JunctionIdentityMatch {
                    predicted_junction_id: "pB".to_string(),
                    ground_truth_junction_id: "B".to_string(),
                },
            ],
            ReconstructionEvaluationConfig {
                maximum_junction_match_distance: 0.05,
                ..ReconstructionEvaluationConfig::default()
            },
        )
        .unwrap();
        let digest = score.canonical_digest_hex().unwrap();
        let bytes = serde_json::to_vec(&score).unwrap();
        let restored: MycelialReconstructionScore = serde_json::from_slice(&bytes).unwrap();
        restored.validate().unwrap();
        assert_eq!(restored.canonical_digest_hex().unwrap(), digest);
    }

    #[test]
    fn two_dimensional_annotations_reject_nonzero_depth() {
        let mut planar = graph(
            "human",
            vec![node("A", AnnotatedJunctionKind::Tip, 0.0, 0.0, 0.0)],
            Vec::new(),
        );
        planar.dimensionality = ReconstructionDimensionality::TwoD;
        planar.junctions[0].position.z = 0.01;
        assert!(matches!(
            planar.validate(),
            Err(MycelialReconstructionError::InvalidGraph(_))
        ));
    }

    #[test]
    fn coordinate_frame_and_calibration_must_match_before_scoring() {
        let truth = graph(
            "human",
            vec![node("A", AnnotatedJunctionKind::Tip, 0.0, 0.0, 0.0)],
            Vec::new(),
        );
        let mut wrong_coordinate_frame = graph(
            "model",
            vec![node("pA", AnnotatedJunctionKind::Tip, 0.0, 0.0, 0.0)],
            Vec::new(),
        );
        wrong_coordinate_frame.coordinate_frame_id = "camera-frame:rotated".to_string();
        assert_eq!(
            evaluate_mycelial_reconstruction(
                &truth,
                &wrong_coordinate_frame,
                "explicit-match-v1",
                &[],
                ReconstructionEvaluationConfig::default()
            ),
            Err(MycelialReconstructionError::CoordinateFrameMismatch)
        );

        let mut wrong_calibration = graph(
            "model",
            vec![node("pA", AnnotatedJunctionKind::Tip, 0.0, 0.0, 0.0)],
            Vec::new(),
        );
        wrong_calibration.calibration_digest = "d".repeat(64);
        assert_eq!(
            evaluate_mycelial_reconstruction(
                &truth,
                &wrong_calibration,
                "explicit-match-v1",
                &[],
                ReconstructionEvaluationConfig::default()
            ),
            Err(MycelialReconstructionError::CalibrationMismatch)
        );
    }

    #[test]
    fn calibration_digest_and_coordinate_frame_are_bound_to_graph_digest() {
        let original = graph(
            "human",
            vec![node("A", AnnotatedJunctionKind::Tip, 0.0, 0.0, 0.0)],
            Vec::new(),
        );
        let original_digest = original.canonical_digest_hex().unwrap();

        let mut changed_calibration = original.clone();
        changed_calibration.calibration_digest = "d".repeat(64);
        assert_ne!(
            original_digest,
            changed_calibration.canonical_digest_hex().unwrap()
        );

        let mut changed_frame = original.clone();
        changed_frame.coordinate_frame_id = "another-frame".to_string();
        assert_ne!(
            original_digest,
            changed_frame.canonical_digest_hex().unwrap()
        );
    }

    #[test]
    fn unit_and_dimensionality_are_part_of_the_benchmark_contract() {
        let truth = graph("human", vec![], vec![]);
        let mut prediction = graph("model", vec![], vec![]);
        prediction.length_unit = ReconstructionLengthUnit::Pixel;
        assert_eq!(
            evaluate_mycelial_reconstruction(
                &truth,
                &prediction,
                "explicit-match-v1",
                &[],
                ReconstructionEvaluationConfig::default()
            ),
            Err(MycelialReconstructionError::LengthUnitMismatch)
        );
    }
}
