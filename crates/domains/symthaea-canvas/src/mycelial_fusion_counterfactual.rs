// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Read-only, candidate-specific anastomosis counterfactuals.
//!
//! The rejected and accepted-for-analysis views share the same frozen annotated
//! source graph. For tip-to-side contacts, both views contain the same
//! length-conserving target-segment split; only the candidate fusion edge differs.
//! The source skeleton is never mutated and an analysis-only edge is never an
//! authorization to accept a biological fusion.
//!
//! The input skeleton contains scalar segment lengths, not measured centerlines.
//! Side-contact parameters are therefore normalized arc-length fractions, not
//! Bezier parameters. A measured bridge length and explicit length unit are
//! required; no straight-line distance is silently treated as hyphal material.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::mycelial_reconstruction_metrics::{
    AnnotatedJunctionKind, AnnotatedMycelialSkeleton, ReconstructionDimensionality,
    ReconstructionLengthUnit, ReconstructionPoint3,
};

pub const MYCELIAL_FUSION_COUNTERFACTUAL_SCHEMA_VERSION: u16 = 1;
pub const MAX_MYCELIAL_FUSION_COUNTERFACTUAL_CANDIDATES: usize = 512;
pub const MAX_MYCELIAL_FUSION_COUNTERFACTUAL_JUNCTIONS: usize = 2_048;
pub const MAX_MYCELIAL_FUSION_COUNTERFACTUAL_SEGMENTS: usize = 4_096;
pub const MAX_MYCELIAL_FUSION_DEMAND_PAIRS: usize = 256;
pub const MAX_MYCELIAL_FUSION_DISRUPTION_SCENARIOS: usize = 768;
pub const MAX_MYCELIAL_FUSION_ID_BYTES: usize = 128;
pub const MAX_MYCELIAL_FUSION_TOTAL_DEMAND_WEIGHT: u128 = 9_007_199_254_740_992;
pub const MAX_MYCELIAL_FUSION_DEMAND_EVALUATIONS: usize = 65_536;
pub const MAX_MYCELIAL_FUSION_GRAPH_VIEW_EVALUATIONS: usize = 8_192;
pub const MIN_SIDE_CONTACT_FRACTION_MICROS: u32 = 50_000;
pub const MAX_SIDE_CONTACT_FRACTION_MICROS: u32 = 950_000;
const PARAMETER_SCALE: f64 = 1_000_000.0;
const MAX_CANDIDATE_BRIDGE_LENGTH: f64 = 1.0e9;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "target_kind", rename_all = "snake_case")]
pub enum MycelialFusionTarget {
    TipJunction {
        junction_id: String,
    },
    HyphaSide {
        segment_id: String,
        /// Normalized fraction of measured segment arc length, not curve parameter t.
        arc_length_fraction_micros: u32,
    },
}

/// A proposal plus the probability provenance needed to audit its review priority.
/// Probabilities describe the classifier output; they never authorize a fusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialFusionBridgeLengthBasis {
    /// Bridge length comes from a measurement with declared acquisition provenance.
    Measured,
    /// Bridge length is an explicit model assumption, not a biological observation.
    DeclaredAssumption,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialFusionCandidate {
    pub candidate_id: String,
    pub source_tip_junction_id: String,
    pub target: MycelialFusionTarget,
    pub bridge_length: f64,
    pub bridge_length_unit: ReconstructionLengthUnit,
    pub bridge_length_basis: MycelialFusionBridgeLengthBasis,
    /// Digest of the measurement record or declared bridge-length assumption.
    pub bridge_length_evidence_digest: String,
    pub fusion_probability: f64,
    pub calibration_policy_id: String,
    pub probability_evidence_digest: String,
    /// Candidates in one group compete for a shared physical contact claim.
    pub conflict_group_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelialFusionDemandPair {
    pub demand_id: String,
    /// Caller-declared root/source; this is never inferred from renderer origins.
    pub source_junction_id: String,
    /// Caller-declared demand/target junction.
    pub target_junction_id: String,
    pub weight: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelialFusionDisruptionScenario {
    pub scenario_id: String,
    /// IDs refer to original source segments. A tip-to-side split maps the
    /// original failure to both child intervals in both counterfactual views.
    pub failed_segment_ids: Vec<String>,
    pub failed_junction_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialFusionTopologyEffect {
    ComponentMerge,
    CycleClosing,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialFusionDemandOutcome {
    pub demand_id: String,
    pub source_junction_id: String,
    pub target_junction_id: String,
    pub weight: u64,
    pub reachable: bool,
    pub shortest_path_length: Option<f64>,
    pub direct_distance: Option<f64>,
    /// Direct distance / network path length; geometric directness is not flow.
    pub geometric_directness: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialFusionViewMetrics {
    pub junction_count: usize,
    pub hypha_segment_count: usize,
    pub connected_components: usize,
    pub cycle_rank: usize,
    pub total_hypha_length: f64,
    pub reachable_demand_weight: u64,
    pub total_demand_weight: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialFusionDemandDelta {
    pub demand_id: String,
    pub rejected: MycelialFusionDemandOutcome,
    pub accepted_for_analysis: MycelialFusionDemandOutcome,
    /// accepted minus rejected; None when either view is unreachable.
    pub shortest_path_length_delta: Option<f64>,
    pub reachability_changed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialFusionDisruptionDelta {
    pub scenario_id: String,
    pub failed_segment_ids: Vec<String>,
    pub failed_junction_ids: Vec<String>,
    pub rejected: MycelialFusionViewMetrics,
    pub accepted_for_analysis: MycelialFusionViewMetrics,
    pub reachable_demand_weight_delta: i128,
    pub connected_component_delta: i64,
    pub cycle_rank_delta: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialFusionCandidateImpact {
    pub candidate_id: String,
    pub candidate_digest: String,
    pub source_tip_junction_id: String,
    pub target: MycelialFusionTarget,
    pub bridge_length: f64,
    pub bridge_length_unit: ReconstructionLengthUnit,
    pub bridge_length_evidence_digest: String,
    pub calibration_policy_id: String,
    pub probability_evidence_digest: String,
    pub fusion_probability: f64,
    pub bridge_length_basis: MycelialFusionBridgeLengthBasis,
    pub contact_anchor_interpolation_used: bool,
    pub conflict_group_id: Option<String>,
    /// All other candidates that share this candidate's conflict group.
    pub conflicting_candidate_ids: Vec<String>,
    pub topology_effect: MycelialFusionTopologyEffect,
    pub rejected_view_digest: String,
    pub accepted_analysis_view_digest: String,
    pub rejected: MycelialFusionViewMetrics,
    pub accepted_for_analysis: MycelialFusionViewMetrics,
    /// Signed accepted-minus-rejected deltas.
    pub connected_component_delta: i64,
    pub cycle_rank_delta: i64,
    /// Existing path in the rejected view when the candidate closes a cycle.
    pub cycle_witness_segment_ids: Vec<String>,
    pub demand_deltas: Vec<MycelialFusionDemandDelta>,
    pub disruption_deltas: Vec<MycelialFusionDisruptionDelta>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialFusionCounterfactualReport {
    pub schema_version: u16,
    pub source_skeleton_schema_version: u16,
    pub frame_id: String,
    pub coordinate_frame_id: String,
    pub calibration_digest: String,
    pub dimensionality: ReconstructionDimensionality,
    pub length_unit: ReconstructionLengthUnit,
    pub source_observation_digest: String,
    pub source_graph_digest: String,
    pub candidate_inventory_digest: String,
    pub analysis_only: bool,
    pub source_graph_unchanged: bool,
    pub candidates: Vec<MycelialFusionCandidateImpact>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MycelialFusionCounterfactualError {
    InvalidSourceGraph(String),
    TooManyCandidates(usize),
    TooManyDemandPairs(usize),
    TooManyDisruptionScenarios(usize),
    DemandEvaluationBudgetExceeded(usize),
    GraphViewEvaluationBudgetExceeded(usize),
    ResourceLimitExceeded,
    InvalidCandidate(String),
    DuplicateCandidateId(String),
    DuplicateDemandId(String),
    DuplicateScenarioId(String),
    UnknownSourceTip(String),
    SourceIsNotTip(String),
    UnknownTargetTip(String),
    TargetIsNotTip(String),
    UnknownTargetSegment(String),
    InvalidSideContactFraction(u32),
    LengthUnitMismatch,
    DuplicateExistingConnection(String),
    CounterfactualTopologyMismatch,
    UnknownDemandEndpoint(String),
    InvalidDemand(String),
    InvalidDisruption(String),
    ReportMismatch,
}

impl fmt::Display for MycelialFusionCounterfactualError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSourceGraph(reason) => write!(f, "invalid source skeleton: {reason}"),
            Self::TooManyCandidates(count) => write!(
                f,
                "fusion candidate count {count} exceeds {MAX_MYCELIAL_FUSION_COUNTERFACTUAL_CANDIDATES}"
            ),
            Self::TooManyDemandPairs(count) => write!(
                f,
                "demand pair count {count} exceeds {MAX_MYCELIAL_FUSION_DEMAND_PAIRS}"
            ),
            Self::TooManyDisruptionScenarios(count) => write!(
                f,
                "disruption scenario count {count} exceeds {MAX_MYCELIAL_FUSION_DISRUPTION_SCENARIOS}"
            ),
            Self::DemandEvaluationBudgetExceeded(count) => write!(
                f,
                "fusion counterfactual demand evaluations {count} exceed {MAX_MYCELIAL_FUSION_DEMAND_EVALUATIONS}"
            ),
            Self::GraphViewEvaluationBudgetExceeded(count) => write!(
                f,
                "fusion counterfactual graph-view evaluations {count} exceed {MAX_MYCELIAL_FUSION_GRAPH_VIEW_EVALUATIONS}"
            ),
            Self::ResourceLimitExceeded => {
                write!(f, "counterfactual graph exceeds resource limits")
            }
            Self::InvalidCandidate(id) => write!(f, "fusion candidate {id} has invalid metadata"),
            Self::DuplicateCandidateId(id) => write!(f, "duplicate fusion candidate ID {id}"),
            Self::DuplicateDemandId(id) => write!(f, "duplicate demand ID {id}"),
            Self::DuplicateScenarioId(id) => write!(f, "duplicate disruption scenario ID {id}"),
            Self::UnknownSourceTip(id) => write!(f, "unknown source tip junction {id}"),
            Self::SourceIsNotTip(id) => write!(f, "source junction {id} is not annotated as a tip"),
            Self::UnknownTargetTip(id) => write!(f, "unknown target tip junction {id}"),
            Self::TargetIsNotTip(id) => write!(f, "target junction {id} is not annotated as a tip"),
            Self::UnknownTargetSegment(id) => write!(f, "unknown target hypha segment {id}"),
            Self::InvalidSideContactFraction(value) => {
                write!(
                    f,
                    "side-contact arc-length fraction {value} is outside the interior margin"
                )
            }
            Self::LengthUnitMismatch => {
                write!(f, "candidate bridge length unit differs from source graph")
            }
            Self::DuplicateExistingConnection(id) => {
                write!(f, "candidate {id} duplicates an existing source connection")
            }
            Self::CounterfactualTopologyMismatch => {
                write!(
                    f,
                    "single-edge counterfactual did not produce the expected beta-0/beta-1 delta"
                )
            }
            Self::UnknownDemandEndpoint(id) => write!(f, "unknown declared demand endpoint {id}"),
            Self::InvalidDemand(id) => write!(f, "invalid declared demand pair {id}"),
            Self::InvalidDisruption(id) => write!(f, "invalid frozen disruption scenario {id}"),
            Self::ReportMismatch => write!(
                f,
                "counterfactual report does not recompute from source evidence"
            ),
        }
    }
}

impl Error for MycelialFusionCounterfactualError {}

#[derive(Debug, Clone)]
struct ViewNode {
    position: ReconstructionPoint3,
}

#[derive(Debug, Clone)]
struct ViewSegment {
    id: String,
    /// Source segment identity for failure-set mapping; None for candidate link.
    original_segment_id: Option<String>,
    start_junction_id: String,
    end_junction_id: String,
    length: f64,
}

#[derive(Debug, Clone)]
struct CounterfactualView {
    nodes: BTreeMap<String, ViewNode>,
    segments: Vec<ViewSegment>,
}

fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_MYCELIAL_FUSION_ID_BYTES
        && !value.chars().any(char::is_control)
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn point_distance(left: ReconstructionPoint3, right: ReconstructionPoint3) -> f64 {
    let dx = left.x - right.x;
    let dy = left.y - right.y;
    let dz = left.z - right.z;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn interpolate(
    left: ReconstructionPoint3,
    right: ReconstructionPoint3,
    fraction: f64,
) -> ReconstructionPoint3 {
    ReconstructionPoint3 {
        x: left.x + (right.x - left.x) * fraction,
        y: left.y + (right.y - left.y) * fraction,
        z: left.z + (right.z - left.z) * fraction,
    }
}

fn hash_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn length_unit_tag(unit: ReconstructionLengthUnit) -> u8 {
    match unit {
        ReconstructionLengthUnit::Pixel => 0,
        ReconstructionLengthUnit::Micrometer => 1,
        ReconstructionLengthUnit::Millimeter => 2,
        ReconstructionLengthUnit::OrganismLocalModelUnit => 3,
    }
}

fn candidate_digest(candidate: &MycelialFusionCandidate) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-mycelial-fusion-candidate-v1");
    hash_string(&mut hasher, &candidate.candidate_id);
    hash_string(&mut hasher, &candidate.source_tip_junction_id);
    match &candidate.target {
        MycelialFusionTarget::TipJunction { junction_id } => {
            hasher.update(&[0]);
            hash_string(&mut hasher, junction_id);
        }
        MycelialFusionTarget::HyphaSide {
            segment_id,
            arc_length_fraction_micros,
        } => {
            hasher.update(&[1]);
            hash_string(&mut hasher, segment_id);
            hasher.update(&arc_length_fraction_micros.to_le_bytes());
        }
    }
    hasher.update(&candidate.bridge_length.to_bits().to_le_bytes());
    hasher.update(&[length_unit_tag(candidate.bridge_length_unit)]);
    hasher.update(&[match candidate.bridge_length_basis {
        MycelialFusionBridgeLengthBasis::Measured => 0,
        MycelialFusionBridgeLengthBasis::DeclaredAssumption => 1,
    }]);
    hash_string(&mut hasher, &candidate.bridge_length_evidence_digest);
    hasher.update(&candidate.fusion_probability.to_bits().to_le_bytes());
    hash_string(&mut hasher, &candidate.calibration_policy_id);
    hash_string(&mut hasher, &candidate.probability_evidence_digest);
    match &candidate.conflict_group_id {
        Some(id) => {
            hasher.update(&[1]);
            hash_string(&mut hasher, id);
        }
        None => {
            hasher.update(&[0]);
        }
    }
    hex_digest(hasher.finalize().as_bytes())
}

fn side_contact_anchor_digest(segment_id: &str, arc_length_fraction_micros: u32) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-mycelial-fusion-side-anchor-v1");
    hash_string(&mut hasher, segment_id);
    hasher.update(&arc_length_fraction_micros.to_le_bytes());
    hex_digest(hasher.finalize().as_bytes())
}

fn candidate_edge_identity_digest(candidate_id: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-mycelial-fusion-edge-identity-v1");
    hash_string(&mut hasher, candidate_id);
    hex_digest(hasher.finalize().as_bytes())
}

fn candidate_inventory_digest(candidates: &[MycelialFusionCandidate]) -> String {
    let mut ordered: Vec<_> = candidates.iter().collect();
    ordered.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-mycelial-fusion-candidate-inventory-v1");
    hasher.update(&(ordered.len() as u64).to_le_bytes());
    for candidate in ordered {
        hasher.update(candidate_digest(candidate).as_bytes());
    }
    hex_digest(hasher.finalize().as_bytes())
}

fn hex_digest(bytes: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes {
        use fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn graph_from_skeleton(
    skeleton: &AnnotatedMycelialSkeleton,
) -> Result<CounterfactualView, MycelialFusionCounterfactualError> {
    skeleton.validate().map_err(|error| {
        MycelialFusionCounterfactualError::InvalidSourceGraph(error.to_string())
    })?;
    if skeleton.junctions.len() > MAX_MYCELIAL_FUSION_COUNTERFACTUAL_JUNCTIONS
        || skeleton.segments.len() > MAX_MYCELIAL_FUSION_COUNTERFACTUAL_SEGMENTS
    {
        return Err(MycelialFusionCounterfactualError::ResourceLimitExceeded);
    }
    let nodes = skeleton
        .junctions
        .iter()
        .map(|junction| {
            (
                junction.id.clone(),
                ViewNode {
                    position: junction.position,
                },
            )
        })
        .collect();
    let mut segments: Vec<_> = skeleton
        .segments
        .iter()
        .map(|segment| ViewSegment {
            id: segment.id.clone(),
            original_segment_id: Some(segment.id.clone()),
            start_junction_id: segment.start_junction_id.clone(),
            end_junction_id: segment.end_junction_id.clone(),
            length: segment.length,
        })
        .collect();
    segments.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(CounterfactualView { nodes, segments })
}

fn validate_candidate(
    skeleton: &AnnotatedMycelialSkeleton,
    view: &CounterfactualView,
    candidate: &MycelialFusionCandidate,
) -> Result<(), MycelialFusionCounterfactualError> {
    if !valid_id(&candidate.candidate_id)
        || !valid_id(&candidate.source_tip_junction_id)
        || !candidate.bridge_length.is_finite()
        || candidate.bridge_length <= 0.0
        || candidate.bridge_length > MAX_CANDIDATE_BRIDGE_LENGTH
        || candidate.bridge_length_unit != skeleton.length_unit
        || !valid_digest(&candidate.bridge_length_evidence_digest)
        || !candidate.fusion_probability.is_finite()
        || !(0.0..=1.0).contains(&candidate.fusion_probability)
        || !valid_id(&candidate.calibration_policy_id)
        || !valid_digest(&candidate.probability_evidence_digest)
        || candidate
            .conflict_group_id
            .as_ref()
            .is_some_and(|id| !valid_id(id))
    {
        if candidate.bridge_length_unit != skeleton.length_unit {
            return Err(MycelialFusionCounterfactualError::LengthUnitMismatch);
        }
        return Err(MycelialFusionCounterfactualError::InvalidCandidate(
            candidate.candidate_id.clone(),
        ));
    }
    let source = skeleton
        .junctions
        .iter()
        .find(|junction| junction.id == candidate.source_tip_junction_id)
        .ok_or_else(|| {
            MycelialFusionCounterfactualError::UnknownSourceTip(
                candidate.source_tip_junction_id.clone(),
            )
        })?;
    if source.kind != AnnotatedJunctionKind::Tip {
        return Err(MycelialFusionCounterfactualError::SourceIsNotTip(
            source.id.clone(),
        ));
    }
    match &candidate.target {
        MycelialFusionTarget::TipJunction { junction_id } => {
            let target = skeleton
                .junctions
                .iter()
                .find(|junction| junction.id == *junction_id)
                .ok_or_else(|| {
                    MycelialFusionCounterfactualError::UnknownTargetTip(junction_id.clone())
                })?;
            if target.kind != AnnotatedJunctionKind::Tip || target.id == source.id {
                return Err(MycelialFusionCounterfactualError::TargetIsNotTip(
                    target.id.clone(),
                ));
            }
            if view.segments.iter().any(|segment| {
                (segment.start_junction_id == source.id && segment.end_junction_id == target.id)
                    || (segment.start_junction_id == target.id
                        && segment.end_junction_id == source.id)
            }) {
                return Err(
                    MycelialFusionCounterfactualError::DuplicateExistingConnection(
                        candidate.candidate_id.clone(),
                    ),
                );
            }
        }
        MycelialFusionTarget::HyphaSide {
            segment_id,
            arc_length_fraction_micros,
        } => {
            if *arc_length_fraction_micros <= MIN_SIDE_CONTACT_FRACTION_MICROS
                || *arc_length_fraction_micros >= MAX_SIDE_CONTACT_FRACTION_MICROS
            {
                return Err(
                    MycelialFusionCounterfactualError::InvalidSideContactFraction(
                        *arc_length_fraction_micros,
                    ),
                );
            }
            let target_segment = view
                .segments
                .iter()
                .find(|segment| segment.id == *segment_id)
                .ok_or_else(|| {
                    MycelialFusionCounterfactualError::UnknownTargetSegment(segment_id.clone())
                })?;
            if candidate.source_tip_junction_id == target_segment.start_junction_id
                || candidate.source_tip_junction_id == target_segment.end_junction_id
            {
                return Err(MycelialFusionCounterfactualError::InvalidCandidate(
                    candidate.candidate_id.clone(),
                ));
            }
        }
    }
    Ok(())
}

/// Split the target segment in the rejected view and return the split junction.
/// The same split is then reused in the accepted view, so the only edge toggle
/// is the candidate fusion link.
fn split_target_segment(
    view: &mut CounterfactualView,
    candidate: &MycelialFusionCandidate,
) -> Result<String, MycelialFusionCounterfactualError> {
    let MycelialFusionTarget::HyphaSide {
        segment_id,
        arc_length_fraction_micros,
    } = &candidate.target
    else {
        return Err(MycelialFusionCounterfactualError::InvalidCandidate(
            candidate.candidate_id.clone(),
        ));
    };
    let index = view
        .segments
        .iter()
        .position(|segment| segment.id == *segment_id)
        .ok_or_else(|| {
            MycelialFusionCounterfactualError::UnknownTargetSegment(segment_id.clone())
        })?;
    let original = view.segments.remove(index);
    let start_position = view
        .nodes
        .get(&original.start_junction_id)
        .ok_or_else(|| MycelialFusionCounterfactualError::UnknownTargetSegment(segment_id.clone()))?
        .position;
    let end_position = view
        .nodes
        .get(&original.end_junction_id)
        .ok_or_else(|| MycelialFusionCounterfactualError::UnknownTargetSegment(segment_id.clone()))?
        .position;
    let fraction = f64::from(*arc_length_fraction_micros) / PARAMETER_SCALE;
    let split_identity = side_contact_anchor_digest(segment_id, *arc_length_fraction_micros);
    let split_junction_id = format!("fusion-split:{split_identity}");
    if !valid_id(&split_junction_id) || view.nodes.contains_key(&split_junction_id) {
        return Err(MycelialFusionCounterfactualError::InvalidCandidate(
            candidate.candidate_id.clone(),
        ));
    }
    view.nodes.insert(
        split_junction_id.clone(),
        ViewNode {
            // The source contract has no centerline points. This is an explicit
            // linear display anchor, not a measured point on the hypha.
            position: interpolate(start_position, end_position, fraction),
        },
    );
    let left_length = original.length * fraction;
    let right_length = original.length - left_length;
    if !left_length.is_finite()
        || !right_length.is_finite()
        || left_length <= 0.0
        || right_length <= 0.0
        || (left_length + right_length - original.length).abs()
            > original.length.abs() * 1.0e-12 + 1.0e-12
    {
        return Err(MycelialFusionCounterfactualError::InvalidCandidate(
            candidate.candidate_id.clone(),
        ));
    }
    view.segments.push(ViewSegment {
        id: format!("{}::split:{}:left", original.id, candidate.candidate_id),
        original_segment_id: Some(original.id.clone()),
        start_junction_id: original.start_junction_id,
        end_junction_id: split_junction_id.clone(),
        length: left_length,
    });
    view.segments.push(ViewSegment {
        id: format!("{}::split:{}:right", original.id, candidate.candidate_id),
        original_segment_id: Some(original.id),
        start_junction_id: split_junction_id.clone(),
        end_junction_id: original.end_junction_id,
        length: right_length,
    });
    view.segments.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(split_junction_id)
}

fn graph_view_digest(view: &CounterfactualView) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-mycelial-counterfactual-view-v1");
    hasher.update(&(view.nodes.len() as u64).to_le_bytes());
    for (id, node) in &view.nodes {
        hash_string(&mut hasher, id);
        hasher.update(&node.position.x.to_bits().to_le_bytes());
        hasher.update(&node.position.y.to_bits().to_le_bytes());
        hasher.update(&node.position.z.to_bits().to_le_bytes());
    }
    let mut segments = view.segments.clone();
    segments.sort_by(|left, right| left.id.cmp(&right.id));
    hasher.update(&(segments.len() as u64).to_le_bytes());
    for segment in &segments {
        hash_string(&mut hasher, &segment.id);
        hash_string(&mut hasher, &segment.start_junction_id);
        hash_string(&mut hasher, &segment.end_junction_id);
        hasher.update(&segment.length.to_bits().to_le_bytes());
        match &segment.original_segment_id {
            Some(id) => {
                hasher.update(&[1]);
                hash_string(&mut hasher, id);
            }
            None => {
                hasher.update(&[0]);
            }
        }
    }
    hex_digest(hasher.finalize().as_bytes())
}

fn active_graph<'a>(
    view: &'a CounterfactualView,
    failed_segment_ids: &BTreeSet<String>,
    failed_junction_ids: &BTreeSet<String>,
) -> (BTreeSet<&'a str>, Vec<&'a ViewSegment>) {
    let nodes: BTreeSet<_> = view
        .nodes
        .keys()
        .filter(|id| !failed_junction_ids.contains(*id))
        .map(String::as_str)
        .collect();
    let segments = view
        .segments
        .iter()
        .filter(|segment| {
            !failed_segment_ids.contains(&segment.id)
                && segment
                    .original_segment_id
                    .as_ref()
                    .is_none_or(|id| !failed_segment_ids.contains(id))
                && nodes.contains(segment.start_junction_id.as_str())
                && nodes.contains(segment.end_junction_id.as_str())
        })
        .collect();
    (nodes, segments)
}

fn component_count(nodes: &BTreeSet<&str>, segments: &[&ViewSegment]) -> usize {
    let mut adjacency = BTreeMap::<&str, Vec<&str>>::new();
    for node in nodes {
        adjacency.insert(*node, Vec::new());
    }
    for segment in segments {
        if let Some(neighbours) = adjacency.get_mut(segment.start_junction_id.as_str()) {
            neighbours.push(segment.end_junction_id.as_str());
        }
        if let Some(neighbours) = adjacency.get_mut(segment.end_junction_id.as_str()) {
            neighbours.push(segment.start_junction_id.as_str());
        }
    }
    for neighbours in adjacency.values_mut() {
        neighbours.sort_unstable();
    }
    let mut visited = BTreeSet::new();
    let mut components = 0usize;
    for start in nodes {
        if !visited.insert(*start) {
            continue;
        }
        components += 1;
        let mut pending = VecDeque::from([*start]);
        while let Some(current) = pending.pop_front() {
            if let Some(neighbours) = adjacency.get(current) {
                for neighbour in neighbours {
                    if visited.insert(*neighbour) {
                        pending.push_back(neighbour);
                    }
                }
            }
        }
    }
    components
}

fn shortest_path_length(
    nodes: &BTreeSet<&str>,
    segments: &[&ViewSegment],
    source: &str,
    target: &str,
) -> Option<f64> {
    if !nodes.contains(source) || !nodes.contains(target) {
        return None;
    }
    if source == target {
        return Some(0.0);
    }
    let mut adjacency = BTreeMap::<&str, Vec<(&str, f64)>>::new();
    for node in nodes {
        adjacency.insert(*node, Vec::new());
    }
    for segment in segments {
        adjacency
            .get_mut(segment.start_junction_id.as_str())?
            .push((segment.end_junction_id.as_str(), segment.length));
        adjacency
            .get_mut(segment.end_junction_id.as_str())?
            .push((segment.start_junction_id.as_str(), segment.length));
    }
    let mut distances = BTreeMap::<&str, f64>::new();
    let mut visited = BTreeSet::<&str>::new();
    distances.insert(source, 0.0);
    loop {
        let next = distances
            .iter()
            .filter(|(id, _)| !visited.contains(**id))
            .min_by(|left, right| left.1.total_cmp(right.1).then_with(|| left.0.cmp(right.0)))
            .map(|(id, distance)| (*id, *distance));
        let (current, distance) = next?;
        if current == target {
            return Some(distance);
        }
        visited.insert(current);
        if let Some(neighbours) = adjacency.get(current) {
            for &(neighbour, weight) in neighbours {
                if visited.contains(neighbour) {
                    continue;
                }
                let candidate = distance + weight;
                let replace = distances
                    .get(neighbour)
                    .is_none_or(|previous| candidate < *previous);
                if replace {
                    distances.insert(neighbour, candidate);
                }
            }
        }
    }
}

fn path_witness_segment_ids(
    nodes: &BTreeSet<&str>,
    segments: &[&ViewSegment],
    source: &str,
    target: &str,
) -> Vec<String> {
    if !nodes.contains(source) || !nodes.contains(target) || source == target {
        return Vec::new();
    }
    let mut adjacency = BTreeMap::<&str, Vec<(&str, &str)>>::new();
    for node in nodes {
        adjacency.insert(*node, Vec::new());
    }
    for segment in segments {
        if let Some(neighbours) = adjacency.get_mut(segment.start_junction_id.as_str()) {
            neighbours.push((segment.end_junction_id.as_str(), segment.id.as_str()));
        }
        if let Some(neighbours) = adjacency.get_mut(segment.end_junction_id.as_str()) {
            neighbours.push((segment.start_junction_id.as_str(), segment.id.as_str()));
        }
    }
    for neighbours in adjacency.values_mut() {
        neighbours.sort_unstable();
    }
    let mut previous = BTreeMap::<&str, (&str, &str)>::new();
    let mut visited = BTreeSet::from([source]);
    let mut pending = VecDeque::from([source]);
    while let Some(current) = pending.pop_front() {
        if current == target {
            break;
        }
        if let Some(neighbours) = adjacency.get(current) {
            for &(neighbour, segment_id) in neighbours {
                if visited.insert(neighbour) {
                    previous.insert(neighbour, (current, segment_id));
                    pending.push_back(neighbour);
                }
            }
        }
    }
    if !visited.contains(target) {
        return Vec::new();
    }
    let mut path = Vec::new();
    let mut current = target;
    while current != source {
        let Some((prior, segment_id)) = previous.get(current).copied() else {
            return Vec::new();
        };
        path.push(segment_id.to_string());
        current = prior;
    }
    path.reverse();
    path
}

fn demand_outcome(
    view: &CounterfactualView,
    nodes: &BTreeSet<&str>,
    segments: &[&ViewSegment],
    demand: &MycelialFusionDemandPair,
) -> MycelialFusionDemandOutcome {
    let path_length = shortest_path_length(
        nodes,
        segments,
        &demand.source_junction_id,
        &demand.target_junction_id,
    );
    let direct_distance = if path_length.is_some() {
        view.nodes
            .get(&demand.source_junction_id)
            .zip(view.nodes.get(&demand.target_junction_id))
            .map(|(source, target)| point_distance(source.position, target.position))
    } else {
        None
    };
    let geometric_directness = direct_distance
        .zip(path_length)
        .and_then(|(direct, path)| (path > 0.0).then_some(direct / path));
    MycelialFusionDemandOutcome {
        demand_id: demand.demand_id.clone(),
        source_junction_id: demand.source_junction_id.clone(),
        target_junction_id: demand.target_junction_id.clone(),
        weight: demand.weight,
        reachable: path_length.is_some(),
        shortest_path_length: path_length,
        direct_distance,
        geometric_directness,
    }
}

fn view_metrics(
    view: &CounterfactualView,
    demands: &[MycelialFusionDemandPair],
    failed_segment_ids: &BTreeSet<String>,
    failed_junction_ids: &BTreeSet<String>,
) -> (MycelialFusionViewMetrics, Vec<MycelialFusionDemandOutcome>) {
    let (nodes, segments) = active_graph(view, failed_segment_ids, failed_junction_ids);
    let components = component_count(&nodes, &segments);
    let cycle_rank = segments
        .len()
        .saturating_add(components)
        .saturating_sub(nodes.len());
    let total_hypha_length = segments.iter().map(|segment| segment.length).sum();
    let demand_outcomes: Vec<_> = demands
        .iter()
        .map(|demand| demand_outcome(view, &nodes, &segments, demand))
        .collect();
    let reachable_demand_weight = demand_outcomes
        .iter()
        .filter(|outcome| outcome.reachable)
        .map(|outcome| u128::from(outcome.weight))
        .sum::<u128>()
        .min(u128::from(u64::MAX)) as u64;
    let total_demand_weight = demands
        .iter()
        .map(|demand| u128::from(demand.weight))
        .sum::<u128>()
        .min(u128::from(u64::MAX)) as u64;
    (
        MycelialFusionViewMetrics {
            junction_count: nodes.len(),
            hypha_segment_count: segments.len(),
            connected_components: components,
            cycle_rank,
            total_hypha_length,
            reachable_demand_weight,
            total_demand_weight,
        },
        demand_outcomes,
    )
}

fn signed_difference(after: usize, before: usize) -> i64 {
    after as i64 - before as i64
}

fn build_candidate_impact(
    skeleton: &AnnotatedMycelialSkeleton,
    source_view: &CounterfactualView,
    candidate: &MycelialFusionCandidate,
    conflicting_candidate_ids: Vec<String>,
    demands: &[MycelialFusionDemandPair],
    disruptions: &[MycelialFusionDisruptionScenario],
) -> Result<MycelialFusionCandidateImpact, MycelialFusionCounterfactualError> {
    let mut rejected_view = source_view.clone();
    let target_node_id = match &candidate.target {
        MycelialFusionTarget::TipJunction { junction_id } => junction_id.clone(),
        MycelialFusionTarget::HyphaSide { .. } => {
            split_target_segment(&mut rejected_view, candidate)?
        }
    };
    let source_id = candidate.source_tip_junction_id.as_str();
    let rejected_digest = graph_view_digest(&rejected_view);
    let (rejected_metrics, rejected_demand_outcomes) =
        view_metrics(&rejected_view, demands, &BTreeSet::new(), &BTreeSet::new());
    let (rejected_active_nodes, rejected_active_segments) =
        active_graph(&rejected_view, &BTreeSet::new(), &BTreeSet::new());
    let witness = path_witness_segment_ids(
        &rejected_active_nodes,
        &rejected_active_segments,
        source_id,
        &target_node_id,
    );
    let topology_effect = if witness.is_empty() {
        MycelialFusionTopologyEffect::ComponentMerge
    } else {
        MycelialFusionTopologyEffect::CycleClosing
    };

    let mut accepted_view = rejected_view.clone();
    let fusion_edge_id = format!(
        "fusion-edge:{}",
        candidate_edge_identity_digest(&candidate.candidate_id)
    );
    if accepted_view
        .segments
        .iter()
        .any(|segment| segment.id == fusion_edge_id)
    {
        return Err(
            MycelialFusionCounterfactualError::DuplicateExistingConnection(
                candidate.candidate_id.clone(),
            ),
        );
    }
    accepted_view.segments.push(ViewSegment {
        id: fusion_edge_id,
        original_segment_id: None,
        start_junction_id: candidate.source_tip_junction_id.clone(),
        end_junction_id: target_node_id,
        length: candidate.bridge_length,
    });
    accepted_view
        .segments
        .sort_by(|left, right| left.id.cmp(&right.id));
    let accepted_digest = graph_view_digest(&accepted_view);
    let (accepted_metrics, accepted_demand_outcomes) =
        view_metrics(&accepted_view, demands, &BTreeSet::new(), &BTreeSet::new());
    let topology_delta_is_valid = match topology_effect {
        MycelialFusionTopologyEffect::ComponentMerge => {
            rejected_metrics.connected_components
                == accepted_metrics.connected_components.saturating_add(1)
                && rejected_metrics.cycle_rank == accepted_metrics.cycle_rank
        }
        MycelialFusionTopologyEffect::CycleClosing => {
            rejected_metrics.connected_components == accepted_metrics.connected_components
                && accepted_metrics.cycle_rank == rejected_metrics.cycle_rank.saturating_add(1)
        }
    };
    if !topology_delta_is_valid {
        return Err(MycelialFusionCounterfactualError::CounterfactualTopologyMismatch);
    }
    let demand_deltas = rejected_demand_outcomes
        .into_iter()
        .zip(accepted_demand_outcomes)
        .map(
            |(rejected, accepted_for_analysis)| MycelialFusionDemandDelta {
                shortest_path_length_delta: rejected
                    .shortest_path_length
                    .zip(accepted_for_analysis.shortest_path_length)
                    .map(|(before, after)| after - before),
                reachability_changed: rejected.reachable != accepted_for_analysis.reachable,
                demand_id: rejected.demand_id.clone(),
                rejected,
                accepted_for_analysis,
            },
        )
        .collect();

    let mut disruption_deltas = Vec::with_capacity(disruptions.len());
    for scenario in disruptions {
        let failed_segments: BTreeSet<_> = scenario.failed_segment_ids.iter().cloned().collect();
        let failed_junctions: BTreeSet<_> = scenario.failed_junction_ids.iter().cloned().collect();
        let (before, _) =
            view_metrics(&rejected_view, demands, &failed_segments, &failed_junctions);
        let (after, _) = view_metrics(&accepted_view, demands, &failed_segments, &failed_junctions);
        disruption_deltas.push(MycelialFusionDisruptionDelta {
            scenario_id: scenario.scenario_id.clone(),
            failed_segment_ids: scenario.failed_segment_ids.clone(),
            failed_junction_ids: scenario.failed_junction_ids.clone(),
            reachable_demand_weight_delta: i128::from(after.reachable_demand_weight)
                - i128::from(before.reachable_demand_weight),
            connected_component_delta: signed_difference(
                after.connected_components,
                before.connected_components,
            ),
            cycle_rank_delta: signed_difference(after.cycle_rank, before.cycle_rank),
            rejected: before,
            accepted_for_analysis: after,
        });
    }

    let _ = skeleton; // Source identity is bound by the enclosing report.
    Ok(MycelialFusionCandidateImpact {
        candidate_id: candidate.candidate_id.clone(),
        candidate_digest: candidate_digest(candidate),
        source_tip_junction_id: candidate.source_tip_junction_id.clone(),
        target: candidate.target.clone(),
        bridge_length: candidate.bridge_length,
        bridge_length_unit: candidate.bridge_length_unit,
        bridge_length_evidence_digest: candidate.bridge_length_evidence_digest.clone(),
        calibration_policy_id: candidate.calibration_policy_id.clone(),
        probability_evidence_digest: candidate.probability_evidence_digest.clone(),
        fusion_probability: candidate.fusion_probability,
        bridge_length_basis: candidate.bridge_length_basis,
        contact_anchor_interpolation_used: matches!(
            &candidate.target,
            MycelialFusionTarget::HyphaSide { .. }
        ),
        conflict_group_id: candidate.conflict_group_id.clone(),
        conflicting_candidate_ids,
        topology_effect,
        rejected_view_digest: rejected_digest,
        accepted_analysis_view_digest: accepted_digest,
        connected_component_delta: signed_difference(
            accepted_metrics.connected_components,
            rejected_metrics.connected_components,
        ),
        cycle_rank_delta: signed_difference(
            accepted_metrics.cycle_rank,
            rejected_metrics.cycle_rank,
        ),
        cycle_witness_segment_ids: if topology_effect == MycelialFusionTopologyEffect::CycleClosing
        {
            witness
        } else {
            Vec::new()
        },
        rejected: rejected_metrics,
        accepted_for_analysis: accepted_metrics,
        demand_deltas,
        disruption_deltas,
    })
}

/// Return true when two proposal identities cannot both be treated as
/// independent contact events under the pairwise-fusion contract. Explicit
/// caller conflict groups take precedence, but common source tips, shared
/// tip endpoints, and identical tip-to-side contacts are also detected.
fn candidates_conflict(left: &MycelialFusionCandidate, right: &MycelialFusionCandidate) -> bool {
    if left.conflict_group_id.is_some() && left.conflict_group_id == right.conflict_group_id
    {
        return true;
    }
    if left.source_tip_junction_id == right.source_tip_junction_id {
        return true;
    }
    match (&left.target, &right.target) {
        (
            MycelialFusionTarget::TipJunction {
                junction_id: left_target,
            },
            MycelialFusionTarget::TipJunction {
                junction_id: right_target,
            },
        ) => {
            let left_endpoints = [left.source_tip_junction_id.as_str(), left_target.as_str()];
            let right_endpoints = [right.source_tip_junction_id.as_str(), right_target.as_str()];
            left_endpoints.iter().any(|id| right_endpoints.contains(id))
        }
        (
            MycelialFusionTarget::HyphaSide {
                segment_id: left_segment,
                arc_length_fraction_micros: left_fraction,
            },
            MycelialFusionTarget::HyphaSide {
                segment_id: right_segment,
                arc_length_fraction_micros: right_fraction,
            },
        ) => left_segment == right_segment && left_fraction == right_fraction,
        _ => false,
    }
}

fn validate_inputs(
    skeleton: &AnnotatedMycelialSkeleton,
    candidates: &[MycelialFusionCandidate],
    demands: &[MycelialFusionDemandPair],
    disruptions: &[MycelialFusionDisruptionScenario],
) -> Result<(), MycelialFusionCounterfactualError> {
    if candidates.len() > MAX_MYCELIAL_FUSION_COUNTERFACTUAL_CANDIDATES {
        return Err(MycelialFusionCounterfactualError::TooManyCandidates(
            candidates.len(),
        ));
    }
    if demands.len() > MAX_MYCELIAL_FUSION_DEMAND_PAIRS {
        return Err(MycelialFusionCounterfactualError::TooManyDemandPairs(
            demands.len(),
        ));
    }
    if disruptions.len() > MAX_MYCELIAL_FUSION_DISRUPTION_SCENARIOS {
        return Err(
            MycelialFusionCounterfactualError::TooManyDisruptionScenarios(disruptions.len()),
        );
    }
    let graph_view_evaluations = candidates
        .len()
        .saturating_mul(disruptions.len().saturating_add(1))
        .saturating_mul(2);
    if graph_view_evaluations > MAX_MYCELIAL_FUSION_GRAPH_VIEW_EVALUATIONS {
        return Err(
            MycelialFusionCounterfactualError::GraphViewEvaluationBudgetExceeded(
                graph_view_evaluations,
            ),
        );
    }
    let demand_evaluations = graph_view_evaluations.saturating_mul(demands.len());
    if demand_evaluations > MAX_MYCELIAL_FUSION_DEMAND_EVALUATIONS {
        return Err(
            MycelialFusionCounterfactualError::DemandEvaluationBudgetExceeded(demand_evaluations),
        );
    }
    let mut candidate_ids = BTreeSet::new();
    for candidate in candidates {
        if !candidate_ids.insert(candidate.candidate_id.as_str()) {
            return Err(MycelialFusionCounterfactualError::DuplicateCandidateId(
                candidate.candidate_id.clone(),
            ));
        }
    }
    let mut demand_ids = BTreeSet::new();
    let mut total_demand_weight = 0u128;
    for demand in demands {
        if !valid_id(&demand.demand_id)
            || !demand_ids.insert(demand.demand_id.as_str())
            || demand.weight == 0
            || demand.source_junction_id == demand.target_junction_id
        {
            return Err(MycelialFusionCounterfactualError::InvalidDemand(
                demand.demand_id.clone(),
            ));
        }
        total_demand_weight = total_demand_weight.saturating_add(u128::from(demand.weight));
        if skeleton
            .junctions
            .iter()
            .all(|junction| junction.id != demand.source_junction_id)
        {
            return Err(MycelialFusionCounterfactualError::UnknownDemandEndpoint(
                demand.source_junction_id.clone(),
            ));
        }
        if skeleton
            .junctions
            .iter()
            .all(|junction| junction.id != demand.target_junction_id)
        {
            return Err(MycelialFusionCounterfactualError::UnknownDemandEndpoint(
                demand.target_junction_id.clone(),
            ));
        }
    }
    if total_demand_weight > MAX_MYCELIAL_FUSION_TOTAL_DEMAND_WEIGHT {
        return Err(MycelialFusionCounterfactualError::ResourceLimitExceeded);
    }
    let source_segments: BTreeSet<_> = skeleton
        .segments
        .iter()
        .map(|segment| segment.id.as_str())
        .collect();
    let source_junctions: BTreeSet<_> = skeleton
        .junctions
        .iter()
        .map(|junction| junction.id.as_str())
        .collect();
    let mut scenario_ids = BTreeSet::new();
    for scenario in disruptions {
        if !valid_id(&scenario.scenario_id) || !scenario_ids.insert(scenario.scenario_id.as_str()) {
            return Err(MycelialFusionCounterfactualError::DuplicateScenarioId(
                scenario.scenario_id.clone(),
            ));
        }
        if scenario
            .failed_segment_ids
            .iter()
            .any(|id| !source_segments.contains(id.as_str()))
            || scenario
                .failed_junction_ids
                .iter()
                .any(|id| !source_junctions.contains(id.as_str()))
            || scenario
                .failed_segment_ids
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || scenario
                .failed_junction_ids
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(MycelialFusionCounterfactualError::InvalidDisruption(
                scenario.scenario_id.clone(),
            ));
        }
    }
    Ok(())
}

/// Evaluate each candidate independently against the exact same frozen source.
/// Candidates in a conflict group are never combined into one accepted graph.
pub fn evaluate_mycelial_fusion_counterfactuals(
    skeleton: &AnnotatedMycelialSkeleton,
    candidates: &[MycelialFusionCandidate],
    demands: &[MycelialFusionDemandPair],
    disruptions: &[MycelialFusionDisruptionScenario],
) -> Result<MycelialFusionCounterfactualReport, MycelialFusionCounterfactualError> {
    skeleton.validate().map_err(|error| {
        MycelialFusionCounterfactualError::InvalidSourceGraph(error.to_string())
    })?;
    validate_inputs(skeleton, candidates, demands, disruptions)?;
    let source_graph_digest = skeleton.canonical_digest_hex().map_err(|error| {
        MycelialFusionCounterfactualError::InvalidSourceGraph(error.to_string())
    })?;
    let source_view = graph_from_skeleton(skeleton)?;
    let mut ordered: Vec<_> = candidates.iter().collect();
    ordered.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
    let mut impacts = Vec::with_capacity(ordered.len());
    for candidate in &ordered {
        validate_candidate(skeleton, &source_view, candidate)?;
        let mut conflicting_candidate_ids: Vec<_> = ordered
            .iter()
            .filter(|other| {
                other.candidate_id != candidate.candidate_id
                    && candidates_conflict(candidate, other)
            })
            .map(|other| other.candidate_id.clone())
            .collect();
        conflicting_candidate_ids.sort_unstable();
        impacts.push(build_candidate_impact(
            skeleton,
            &source_view,
            candidate,
            conflicting_candidate_ids,
            demands,
            disruptions,
        )?);
    }
    Ok(MycelialFusionCounterfactualReport {
        schema_version: MYCELIAL_FUSION_COUNTERFACTUAL_SCHEMA_VERSION,
        source_skeleton_schema_version: skeleton.schema_version,
        frame_id: skeleton.frame_id.clone(),
        coordinate_frame_id: skeleton.coordinate_frame_id.clone(),
        calibration_digest: skeleton.calibration_digest.clone(),
        dimensionality: skeleton.dimensionality,
        length_unit: skeleton.length_unit,
        source_observation_digest: skeleton.source_observation_digest.clone(),
        source_graph_digest,
        candidate_inventory_digest: candidate_inventory_digest(candidates),
        analysis_only: true,
        source_graph_unchanged: true,
        candidates: impacts,
    })
}

impl MycelialFusionCounterfactualReport {
    /// Independent recomputation from the original skeleton and candidate inventory.
    pub fn validate_against(
        &self,
        skeleton: &AnnotatedMycelialSkeleton,
        candidates: &[MycelialFusionCandidate],
        demands: &[MycelialFusionDemandPair],
        disruptions: &[MycelialFusionDisruptionScenario],
    ) -> Result<(), MycelialFusionCounterfactualError> {
        let expected =
            evaluate_mycelial_fusion_counterfactuals(skeleton, candidates, demands, disruptions)?;
        if self != &expected {
            return Err(MycelialFusionCounterfactualError::ReportMismatch);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mycelial_reconstruction_metrics::{
        AnnotatedHyphaSegment, AnnotatedJunction, ReconstructionDimensionality,
    };

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn junction(id: &str, kind: AnnotatedJunctionKind, x: f64, y: f64) -> AnnotatedJunction {
        AnnotatedJunction {
            id: id.to_string(),
            kind,
            position: ReconstructionPoint3 { x, y, z: 0.0 },
        }
    }

    fn segment(id: &str, start: &str, end: &str, length: f64) -> AnnotatedHyphaSegment {
        AnnotatedHyphaSegment {
            id: id.to_string(),
            start_junction_id: start.to_string(),
            end_junction_id: end.to_string(),
            length,
        }
    }

    /// Independent test oracle: Floyd-Warshall for paths plus reachability
    /// components. It intentionally does not call the production graph helpers.
    fn floyd_warshall_oracle(
        node_count: usize,
        edges: &[(usize, usize, f64)],
    ) -> (usize, usize, Vec<Vec<f64>>) {
        let mut distances = vec![vec![f64::INFINITY; node_count]; node_count];
        for (index, row) in distances.iter_mut().enumerate() {
            row[index] = 0.0;
        }
        for &(source, target, length) in edges {
            distances[source][target] = distances[source][target].min(length);
            distances[target][source] = distances[target][source].min(length);
        }
        for pivot in 0..node_count {
            for source in 0..node_count {
                for target in 0..node_count {
                    let through_pivot = distances[source][pivot] + distances[pivot][target];
                    if through_pivot < distances[source][target] {
                        distances[source][target] = through_pivot;
                    }
                }
            }
        }
        let mut visited = vec![false; node_count];
        let mut components = 0usize;
        for start in 0..node_count {
            if visited[start] {
                continue;
            }
            components += 1;
            visited[start] = true;
            let mut pending = vec![start];
            while let Some(current) = pending.pop() {
                for next in 0..node_count {
                    if distances[current][next].is_finite() && !visited[next] {
                        visited[next] = true;
                        pending.push(next);
                    }
                }
            }
        }
        let cycle_rank = edges
            .len()
            .saturating_add(components)
            .saturating_sub(node_count);
        (components, cycle_rank, distances)
    }

    fn skeleton(
        junctions: Vec<AnnotatedJunction>,
        segments: Vec<AnnotatedHyphaSegment>,
    ) -> AnnotatedMycelialSkeleton {
        AnnotatedMycelialSkeleton {
            schema_version:
                crate::mycelial_reconstruction_metrics::MYCELIAL_RECONSTRUCTION_SCHEMA_VERSION,
            frame_id: "frame:001".to_string(),
            coordinate_frame_id: "stage:001".to_string(),
            calibration_digest: digest('b'),
            source_observation_digest: digest('a'),
            dimensionality: ReconstructionDimensionality::TwoD,
            length_unit: ReconstructionLengthUnit::Micrometer,
            producer_id: "annotation-fixture".to_string(),
            producer_version: "v1".to_string(),
            junctions,
            segments,
        }
    }

    fn candidate_tip_to_tip(
        id: &str,
        source: &str,
        target: &str,
        conflict: Option<&str>,
    ) -> MycelialFusionCandidate {
        MycelialFusionCandidate {
            candidate_id: id.to_string(),
            source_tip_junction_id: source.to_string(),
            target: MycelialFusionTarget::TipJunction {
                junction_id: target.to_string(),
            },
            bridge_length: 1.0,
            bridge_length_unit: ReconstructionLengthUnit::Micrometer,
            bridge_length_basis: MycelialFusionBridgeLengthBasis::DeclaredAssumption,
            bridge_length_evidence_digest: digest('h'),
            fusion_probability: 0.65,
            calibration_policy_id: "calibration-v1".to_string(),
            probability_evidence_digest: digest('c'),
            conflict_group_id: conflict.map(str::to_string),
        }
    }

    #[test]
    fn tip_to_tip_toggle_reports_cycle_and_shorter_declared_demand_path_without_mutation() {
        let source = skeleton(
            vec![
                junction("A", AnnotatedJunctionKind::Origin, 0.0, 0.0),
                junction("B", AnnotatedJunctionKind::Tip, 1.0, 0.0),
                junction("C", AnnotatedJunctionKind::Tip, 0.0, 1.0),
            ],
            vec![segment("ab", "A", "B", 1.0), segment("ac", "A", "C", 1.0)],
        );
        let source_digest = source.canonical_digest_hex().unwrap();
        let candidates = vec![candidate_tip_to_tip("fusion-1", "B", "C", None)];
        let demands = vec![MycelialFusionDemandPair {
            demand_id: "root-to-demand".to_string(),
            source_junction_id: "B".to_string(),
            target_junction_id: "C".to_string(),
            weight: 10,
        }];
        let report =
            evaluate_mycelial_fusion_counterfactuals(&source, &candidates, &demands, &[]).unwrap();
        let impact = &report.candidates[0];
        assert_eq!(
            impact.topology_effect,
            MycelialFusionTopologyEffect::CycleClosing
        );
        assert_eq!(impact.rejected.cycle_rank, 0);
        assert_eq!(impact.accepted_for_analysis.cycle_rank, 1);
        assert_eq!(impact.cycle_rank_delta, 1);
        assert_eq!(impact.connected_component_delta, 0);
        assert!(!impact.contact_anchor_interpolation_used);
        assert_eq!(impact.cycle_witness_segment_ids.len(), 2);
        assert_eq!(
            impact.demand_deltas[0].shortest_path_length_delta,
            Some(-1.0)
        );
        assert!(report.analysis_only);
        assert!(report.source_graph_unchanged);
        assert_eq!(source.canonical_digest_hex().unwrap(), source_digest);
        report
            .validate_against(&source, &candidates, &demands, &[])
            .unwrap();

        let (oracle_before_components, oracle_before_cycles, before_paths) =
            floyd_warshall_oracle(3, &[(0, 1, 1.0), (0, 2, 1.0)]);
        let (oracle_after_components, oracle_after_cycles, after_paths) =
            floyd_warshall_oracle(3, &[(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)]);
        assert_eq!(
            impact.rejected.connected_components,
            oracle_before_components
        );
        assert_eq!(impact.rejected.cycle_rank, oracle_before_cycles);
        assert_eq!(
            impact.accepted_for_analysis.connected_components,
            oracle_after_components
        );
        assert_eq!(impact.accepted_for_analysis.cycle_rank, oracle_after_cycles);
        assert_eq!(before_paths[1][2], 2.0);
        assert_eq!(after_paths[1][2], 1.0);
        assert_eq!(
            impact.demand_deltas[0].rejected.shortest_path_length,
            Some(before_paths[1][2])
        );
        assert_eq!(
            impact.demand_deltas[0]
                .accepted_for_analysis
                .shortest_path_length,
            Some(after_paths[1][2])
        );
    }

    #[test]
    fn tip_to_side_counterfactual_splits_length_once_and_merges_components() {
        let source = skeleton(
            vec![
                junction("A", AnnotatedJunctionKind::Origin, 0.0, 0.0),
                junction("C", AnnotatedJunctionKind::Tip, 0.0, 10.0),
                junction("D", AnnotatedJunctionKind::Origin, 30.0, 0.0),
                junction("E", AnnotatedJunctionKind::Tip, 40.0, 0.0),
            ],
            vec![
                segment("target-side", "A", "C", 10.0),
                segment("root-tip", "D", "E", 10.0),
            ],
        );
        let candidate = MycelialFusionCandidate {
            candidate_id: "side-fusion".to_string(),
            source_tip_junction_id: "E".to_string(),
            target: MycelialFusionTarget::HyphaSide {
                segment_id: "target-side".to_string(),
                arc_length_fraction_micros: 400_000,
            },
            bridge_length: 2.0,
            bridge_length_unit: ReconstructionLengthUnit::Micrometer,
            bridge_length_basis: MycelialFusionBridgeLengthBasis::DeclaredAssumption,
            bridge_length_evidence_digest: digest('i'),
            fusion_probability: 0.72,
            calibration_policy_id: "calibration-v1".to_string(),
            probability_evidence_digest: digest('d'),
            conflict_group_id: Some("tip-E-contact".to_string()),
        };
        let report =
            evaluate_mycelial_fusion_counterfactuals(&source, &[candidate], &[], &[]).unwrap();
        let impact = &report.candidates[0];
        assert_eq!(
            impact.topology_effect,
            MycelialFusionTopologyEffect::ComponentMerge
        );
        assert_eq!(impact.rejected.connected_components, 2);
        assert_eq!(impact.accepted_for_analysis.connected_components, 1);
        assert_eq!(impact.connected_component_delta, -1);
        assert_eq!(impact.cycle_rank_delta, 0);
        assert!(impact.contact_anchor_interpolation_used);
        assert_eq!(
            impact.bridge_length_basis,
            MycelialFusionBridgeLengthBasis::DeclaredAssumption
        );
        assert!((impact.rejected.total_hypha_length - 20.0).abs() < 1.0e-12);
        assert!((impact.accepted_for_analysis.total_hypha_length - 22.0).abs() < 1.0e-12);
        let (oracle_before_components, oracle_before_cycles, _) =
            floyd_warshall_oracle(5, &[(0, 4, 4.0), (4, 1, 6.0), (2, 3, 10.0)]);
        let (oracle_after_components, oracle_after_cycles, _) =
            floyd_warshall_oracle(5, &[(0, 4, 4.0), (4, 1, 6.0), (2, 3, 10.0), (3, 4, 2.0)]);
        assert_eq!(
            impact.rejected.connected_components,
            oracle_before_components
        );
        assert_eq!(impact.rejected.cycle_rank, oracle_before_cycles);
        assert_eq!(
            impact.accepted_for_analysis.connected_components,
            oracle_after_components
        );
        assert_eq!(impact.accepted_for_analysis.cycle_rank, oracle_after_cycles);
        assert_eq!(
            impact.rejected.hypha_segment_count + 1,
            impact.accepted_for_analysis.hypha_segment_count
        );
        assert_eq!(impact.conflicting_candidate_ids, Vec::<String>::new());
    }

    #[test]
    fn conflicting_candidates_are_evaluated_individually_not_as_a_joint_acceptance() {
        let source = skeleton(
            vec![
                junction("A", AnnotatedJunctionKind::Origin, 0.0, 0.0),
                junction("B", AnnotatedJunctionKind::Tip, 1.0, 0.0),
                junction("C", AnnotatedJunctionKind::Tip, 0.0, 1.0),
                junction("D", AnnotatedJunctionKind::Tip, -1.0, 0.0),
            ],
            vec![
                segment("ab", "A", "B", 1.0),
                segment("ac", "A", "C", 1.0),
                segment("ad", "A", "D", 1.0),
            ],
        );
        let candidates = vec![
            candidate_tip_to_tip("one", "B", "C", Some("same-contact")),
            candidate_tip_to_tip("two", "B", "D", Some("same-contact")),
        ];
        let report =
            evaluate_mycelial_fusion_counterfactuals(&source, &candidates, &[], &[]).unwrap();
        assert_eq!(report.candidates.len(), 2);
        assert_eq!(
            report.candidates[0].conflicting_candidate_ids,
            vec!["two".to_string()]
        );
        assert_eq!(
            report.candidates[1].conflicting_candidate_ids,
            vec!["one".to_string()]
        );
        assert!(
            report
                .candidates
                .iter()
                .all(|impact| impact.accepted_for_analysis.cycle_rank == 1)
        );
    }

    #[test]
    fn candidates_sharing_a_source_tip_are_conflicts_without_a_group_id() {
        let source = skeleton(
            vec![
                junction("A", AnnotatedJunctionKind::Origin, 0.0, 0.0),
                junction("B", AnnotatedJunctionKind::Tip, 1.0, 0.0),
                junction("C", AnnotatedJunctionKind::Tip, 0.0, 1.0),
                junction("D", AnnotatedJunctionKind::Tip, -1.0, 0.0),
            ],
            vec![
                segment("ab", "A", "B", 1.0),
                segment("ac", "A", "C", 1.0),
                segment("ad", "A", "D", 1.0),
            ],
        );
        let candidates = vec![
            candidate_tip_to_tip("one", "B", "C", None),
            candidate_tip_to_tip("two", "B", "D", None),
        ];
        let report =
            evaluate_mycelial_fusion_counterfactuals(&source, &candidates, &[], &[]).unwrap();
        assert_eq!(
            report.candidates[0].conflicting_candidate_ids,
            vec!["two".to_string()]
        );
        assert_eq!(
            report.candidates[1].conflicting_candidate_ids,
            vec!["one".to_string()]
        );
    }

    #[test]
    fn candidates_claiming_the_same_side_contact_are_conflicts() {
        let source = skeleton(
            vec![
                junction("A", AnnotatedJunctionKind::Origin, 0.0, 0.0),
                junction("B", AnnotatedJunctionKind::Tip, 10.0, 0.0),
                junction("C", AnnotatedJunctionKind::Tip, 0.0, 10.0),
                junction("D", AnnotatedJunctionKind::Tip, 10.0, 10.0),
                junction("E", AnnotatedJunctionKind::Origin, 0.0, 20.0),
                junction("F", AnnotatedJunctionKind::Origin, 10.0, 20.0),
            ],
            vec![
                segment("shared-side", "A", "B", 10.0),
                segment("source-c", "E", "C", 10.0),
                segment("source-d", "F", "D", 10.0),
            ],
        );
        let side_candidate = |id: &str, source_tip_junction_id: &str| MycelialFusionCandidate {
            candidate_id: id.to_string(),
            source_tip_junction_id: source_tip_junction_id.to_string(),
            target: MycelialFusionTarget::HyphaSide {
                segment_id: "shared-side".to_string(),
                arc_length_fraction_micros: 500_000,
            },
            bridge_length: 1.0,
            bridge_length_unit: ReconstructionLengthUnit::Micrometer,
            bridge_length_basis: MycelialFusionBridgeLengthBasis::DeclaredAssumption,
            bridge_length_evidence_digest: digest('k'),
            fusion_probability: 0.7,
            calibration_policy_id: "calibration-v1".to_string(),
            probability_evidence_digest: digest('g'),
            conflict_group_id: None,
        };
        let candidates = vec![
            side_candidate("side-one", "C"),
            side_candidate("side-two", "D"),
        ];
        let report =
            evaluate_mycelial_fusion_counterfactuals(&source, &candidates, &[], &[]).unwrap();
        assert_eq!(
            report.candidates[0].conflicting_candidate_ids,
            vec!["side-two".to_string()]
        );
        assert_eq!(
            report.candidates[1].conflicting_candidate_ids,
            vec!["side-one".to_string()]
        );
        assert_eq!(
            report.candidates[0].rejected_view_digest,
            report.candidates[1].rejected_view_digest
        );
        assert_ne!(
            report.candidates[0].accepted_analysis_view_digest,
            report.candidates[1].accepted_analysis_view_digest
        );
    }

    #[test]
    fn frozen_failure_of_split_source_segment_removes_both_intervals() {
        let source = skeleton(
            vec![
                junction("A", AnnotatedJunctionKind::Origin, 0.0, 0.0),
                junction("B", AnnotatedJunctionKind::Tip, 10.0, 0.0),
                junction("C", AnnotatedJunctionKind::Origin, 30.0, 0.0),
                junction("D", AnnotatedJunctionKind::Tip, 40.0, 0.0),
            ],
            vec![
                segment("target-side", "A", "B", 10.0),
                segment("source-tip", "C", "D", 10.0),
            ],
        );
        let candidate = MycelialFusionCandidate {
            candidate_id: "side".to_string(),
            source_tip_junction_id: "D".to_string(),
            target: MycelialFusionTarget::HyphaSide {
                segment_id: "target-side".to_string(),
                arc_length_fraction_micros: 500_000,
            },
            bridge_length: 1.0,
            bridge_length_unit: ReconstructionLengthUnit::Micrometer,
            bridge_length_basis: MycelialFusionBridgeLengthBasis::DeclaredAssumption,
            bridge_length_evidence_digest: digest('j'),
            fusion_probability: 0.8,
            calibration_policy_id: "calibration-v1".to_string(),
            probability_evidence_digest: digest('e'),
            conflict_group_id: None,
        };
        let demands = vec![MycelialFusionDemandPair {
            demand_id: "A-to-D".to_string(),
            source_junction_id: "A".to_string(),
            target_junction_id: "D".to_string(),
            weight: 1,
        }];
        let disruptions = vec![MycelialFusionDisruptionScenario {
            scenario_id: "remove-target-original".to_string(),
            failed_segment_ids: vec!["target-side".to_string()],
            failed_junction_ids: Vec::new(),
        }];
        let report =
            evaluate_mycelial_fusion_counterfactuals(&source, &[candidate], &demands, &disruptions)
                .unwrap();
        let impact = &report.candidates[0];
        assert_eq!(
            impact.disruption_deltas[0].rejected.reachable_demand_weight,
            0
        );
        assert_eq!(
            impact.disruption_deltas[0]
                .accepted_for_analysis
                .reachable_demand_weight,
            0
        );
    }
}
