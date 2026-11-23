// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Source-bound, tool-neutral graph sidecars for mycelial reconstruction interchange.
//!
//! OME-Zarr remains the image/volume container. This sidecar references that
//! source and carries a typed graph plus provenance and external object IDs.
//! It does not claim to extend the OME-Zarr core schema or to be a finalized
//! OME-Zarr graph annotation standard.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::mycelial_reconstruction_metrics::{
    AnnotatedJunctionKind, AnnotatedMycelialSkeleton, MycelialReconstructionError,
    ReconstructionDimensionality, ReconstructionLengthUnit,
};

pub const MYCELIAL_GRAPH_SIDECAR_SCHEMA_VERSION: u16 = 1;
pub const MAX_MYCELIAL_GRAPH_SIDECAR_ID_BYTES: usize = 128;
pub const MAX_MYCELIAL_GRAPH_EXTERNAL_MAPPINGS: usize = 1_000_000;
pub const MAX_MYCELIAL_GRAPH_SIDECAR_URI_BYTES: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialGraphImageContainerFormat {
    OmeZarrNgff,
    N5,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelialGraphImageReference {
    /// URI to the immutable image container (e.g. https, s3 or file URI).
    pub dataset_uri: String,
    /// Relative group/dataset path inside the container; empty means the root group.
    pub dataset_path: String,
    pub container_format: MycelialGraphImageContainerFormat,
    pub dataset_id: String,
    pub source_observation_digest: String,
    pub frame_id: String,
    pub coordinate_frame_id: String,
    pub calibration_digest: String,
    /// OME-NGFF version, populated for OME-Zarr sources.
    pub ome_ngff_version: Option<String>,
    /// Name of the NGFF coordinate system to which graph coordinates are bound.
    pub ome_ngff_coordinate_system_name: Option<String>,
    /// Relative NGFF metadata path identifying the transform-chain declaration.
    pub ome_ngff_coordinate_transform_path: Option<String>,
    /// Digest of the exact NGFF transform chain used by this graph.
    pub ome_ngff_coordinate_transform_digest: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelialGraphSidecarProducer {
    /// Stable adapter / application identifier (e.g. fiji-snt, imaris, arivis).
    pub tool_id: String,
    pub tool_version: String,
    pub adapter_id: String,
    pub adapter_version: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialGraphObjectKind {
    Junction,
    Segment,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelialExternalObjectMapping {
    pub target_system_id: String,
    pub target_system_version: String,
    pub target_dataset_id: String,
    pub external_object_id: String,
    pub internal_object_kind: MycelialGraphObjectKind,
    pub internal_object_id: String,
    /// Digest of the import/export receipt or user-confirmed identity mapping.
    pub mapping_evidence_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialGraphSidecar {
    pub schema_version: u16,
    pub sidecar_id: String,
    pub source: MycelialGraphImageReference,
    pub producer: MycelialGraphSidecarProducer,
    pub graph: AnnotatedMycelialSkeleton,
    /// Sorted by target system, dataset and external object ID.
    pub external_object_mappings: Vec<MycelialExternalObjectMapping>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MycelialGraphSidecarError {
    InvalidSchemaVersion(u16),
    InvalidMetadata,
    InvalidSourceReference,
    SourceGraphBindingMismatch,
    InvalidGraph(String),
    TooManyExternalMappings(usize),
    InvalidExternalMapping(String),
    DuplicateOrUnorderedExternalMapping(String),
}

impl fmt::Display for MycelialGraphSidecarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSchemaVersion(version) => {
                write!(
                    f,
                    "unsupported mycelial graph sidecar schema version {version}"
                )
            }
            Self::InvalidMetadata => write!(f, "invalid mycelial graph sidecar metadata"),
            Self::InvalidSourceReference => {
                write!(
                    f,
                    "invalid image-container URI, group path or source reference"
                )
            }
            Self::SourceGraphBindingMismatch => write!(
                f,
                "image source digest, frame, coordinate frame or calibration does not match the graph"
            ),
            Self::InvalidGraph(reason) => write!(f, "invalid graph in sidecar: {reason}"),
            Self::TooManyExternalMappings(count) => write!(
                f,
                "external object mapping count {count} exceeds {MAX_MYCELIAL_GRAPH_EXTERNAL_MAPPINGS}"
            ),
            Self::InvalidExternalMapping(id) => {
                write!(f, "invalid external object mapping {id}")
            }
            Self::DuplicateOrUnorderedExternalMapping(id) => {
                write!(
                    f,
                    "external object mappings are duplicated or unordered at {id}"
                )
            }
        }
    }
}

impl Error for MycelialGraphSidecarError {}

fn valid_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value.len() <= MAX_MYCELIAL_GRAPH_SIDECAR_ID_BYTES
        && value.bytes().all(|byte| !byte.is_ascii_control())
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn valid_uri(value: &str) -> bool {
    if value.is_empty()
        || value.len() > MAX_MYCELIAL_GRAPH_SIDECAR_URI_BYTES
        || value
            .bytes()
            .any(|byte| byte.is_ascii_whitespace() || byte.is_ascii_control())
    {
        return false;
    }
    ["https://", "http://", "s3://", "gs://", "file://"]
        .iter()
        .any(|scheme| value.starts_with(scheme))
}

fn valid_dataset_path(value: &str) -> bool {
    !value.starts_with('/')
        && !value.contains('\\')
        && value
            .split('/')
            .all(|part| part.is_empty() || (part != "." && part != ".."))
}

fn hash_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hash_graph_content(hasher: &mut blake3::Hasher, graph: &AnnotatedMycelialSkeleton) {
    hasher.update(&graph.schema_version.to_le_bytes());
    hash_string(hasher, &graph.frame_id);
    hash_string(hasher, &graph.coordinate_frame_id);
    hash_string(hasher, &graph.calibration_digest);
    hash_string(hasher, &graph.source_observation_digest);
    hasher.update(&[match graph.dimensionality {
        ReconstructionDimensionality::TwoD => 0,
        ReconstructionDimensionality::ThreeD => 1,
    }]);
    hasher.update(&[match graph.length_unit {
        ReconstructionLengthUnit::Pixel => 0,
        ReconstructionLengthUnit::Micrometer => 1,
        ReconstructionLengthUnit::Millimeter => 2,
        ReconstructionLengthUnit::OrganismLocalModelUnit => 3,
    }]);
    hash_string(hasher, &graph.producer_id);
    hash_string(hasher, &graph.producer_version);
    hasher.update(&(graph.junctions.len() as u64).to_le_bytes());
    for junction in &graph.junctions {
        hash_string(hasher, &junction.id);
        hasher.update(&[match junction.kind {
            AnnotatedJunctionKind::Origin => 0,
            AnnotatedJunctionKind::Branch => 1,
            AnnotatedJunctionKind::Tip => 2,
            AnnotatedJunctionKind::Anastomosis => 3,
            AnnotatedJunctionKind::ProjectedCrossing => 4,
            AnnotatedJunctionKind::Uncertain => 5,
        }]);
        hasher.update(&junction.position.x.to_bits().to_le_bytes());
        hasher.update(&junction.position.y.to_bits().to_le_bytes());
        hasher.update(&junction.position.z.to_bits().to_le_bytes());
    }
    hasher.update(&(graph.segments.len() as u64).to_le_bytes());
    for segment in &graph.segments {
        hash_string(hasher, &segment.id);
        hash_string(hasher, &segment.start_junction_id);
        hash_string(hasher, &segment.end_junction_id);
        hasher.update(&segment.length.to_bits().to_le_bytes());
    }
}

fn mapping_order_key(mapping: &MycelialExternalObjectMapping) -> (&str, &str, &str) {
    (
        mapping.target_system_id.as_str(),
        mapping.target_dataset_id.as_str(),
        mapping.external_object_id.as_str(),
    )
}

impl MycelialGraphSidecar {
    pub fn validate(&self) -> Result<(), MycelialGraphSidecarError> {
        if self.schema_version != MYCELIAL_GRAPH_SIDECAR_SCHEMA_VERSION {
            return Err(MycelialGraphSidecarError::InvalidSchemaVersion(
                self.schema_version,
            ));
        }
        if !valid_id(&self.sidecar_id)
            || !valid_id(&self.source.dataset_id)
            || !valid_id(&self.producer.tool_id)
            || !valid_id(&self.producer.tool_version)
            || !valid_id(&self.producer.adapter_id)
            || !valid_id(&self.producer.adapter_version)
        {
            return Err(MycelialGraphSidecarError::InvalidMetadata);
        }
        if !valid_uri(&self.source.dataset_uri)
            || !valid_dataset_path(&self.source.dataset_path)
            || !valid_digest(&self.source.source_observation_digest)
            || !valid_id(&self.source.frame_id)
            || !valid_id(&self.source.coordinate_frame_id)
            || !valid_digest(&self.source.calibration_digest)
        {
            return Err(MycelialGraphSidecarError::InvalidSourceReference);
        }
        match self.source.container_format {
            MycelialGraphImageContainerFormat::OmeZarrNgff => {
                let valid_ngff_coordinates = self
                    .source
                    .ome_ngff_version
                    .as_deref()
                    .is_some_and(valid_id)
                    && self
                        .source
                        .ome_ngff_coordinate_transform_digest
                        .as_deref()
                        .is_some_and(valid_digest);
                let has_system = self.source.ome_ngff_coordinate_system_name.is_some();
                let has_path = self.source.ome_ngff_coordinate_transform_path.is_some();
                let valid_named_system = if self.source.ome_ngff_version.as_deref() == Some("0.6") {
                    self.source
                        .ome_ngff_coordinate_system_name
                        .as_deref()
                        .is_some_and(valid_id)
                        && self
                            .source
                            .ome_ngff_coordinate_transform_path
                            .as_deref()
                            .is_some_and(valid_dataset_path)
                } else {
                    has_system == has_path
                        && self
                            .source
                            .ome_ngff_coordinate_system_name
                            .as_deref()
                            .is_none_or(valid_id)
                        && self
                            .source
                            .ome_ngff_coordinate_transform_path
                            .as_deref()
                            .is_none_or(valid_dataset_path)
                };
                if !valid_ngff_coordinates || !valid_named_system {
                    return Err(MycelialGraphSidecarError::InvalidSourceReference);
                }
            }
            MycelialGraphImageContainerFormat::N5 | MycelialGraphImageContainerFormat::Other => {
                if self.source.ome_ngff_version.is_some()
                    || self.source.ome_ngff_coordinate_system_name.is_some()
                    || self.source.ome_ngff_coordinate_transform_path.is_some()
                    || self.source.ome_ngff_coordinate_transform_digest.is_some()
                {
                    return Err(MycelialGraphSidecarError::InvalidSourceReference);
                }
            }
        }
        self.graph
            .validate()
            .map_err(|error: MycelialReconstructionError| {
                MycelialGraphSidecarError::InvalidGraph(error.to_string())
            })?;
        if self.source.source_observation_digest != self.graph.source_observation_digest
            || self.source.frame_id != self.graph.frame_id
            || self.source.coordinate_frame_id != self.graph.coordinate_frame_id
            || self.source.calibration_digest != self.graph.calibration_digest
        {
            return Err(MycelialGraphSidecarError::SourceGraphBindingMismatch);
        }
        if self.external_object_mappings.len() > MAX_MYCELIAL_GRAPH_EXTERNAL_MAPPINGS {
            return Err(MycelialGraphSidecarError::TooManyExternalMappings(
                self.external_object_mappings.len(),
            ));
        }

        let junction_ids: BTreeSet<_> = self
            .graph
            .junctions
            .iter()
            .map(|item| item.id.as_str())
            .collect();
        let segment_ids: BTreeSet<_> = self
            .graph
            .segments
            .iter()
            .map(|item| item.id.as_str())
            .collect();
        let mut external_keys = BTreeSet::new();
        let mut previous_key: Option<(&str, &str, &str)> = None;
        for mapping in &self.external_object_mappings {
            let key = mapping_order_key(mapping);
            if !valid_id(mapping.target_system_id.as_str())
                || !valid_id(mapping.target_system_version.as_str())
                || !valid_id(mapping.target_dataset_id.as_str())
                || !valid_id(mapping.external_object_id.as_str())
                || !valid_id(mapping.internal_object_id.as_str())
                || !valid_digest(mapping.mapping_evidence_digest.as_str())
            {
                return Err(MycelialGraphSidecarError::InvalidExternalMapping(
                    mapping.external_object_id.clone(),
                ));
            }
            if previous_key.is_some_and(|previous| previous >= key)
                || !external_keys.insert((
                    mapping.target_system_id.as_str(),
                    mapping.target_dataset_id.as_str(),
                    mapping.external_object_id.as_str(),
                ))
            {
                return Err(
                    MycelialGraphSidecarError::DuplicateOrUnorderedExternalMapping(
                        mapping.external_object_id.clone(),
                    ),
                );
            }
            previous_key = Some(key);
            let internal_exists = match mapping.internal_object_kind {
                MycelialGraphObjectKind::Junction => {
                    junction_ids.contains(mapping.internal_object_id.as_str())
                }
                MycelialGraphObjectKind::Segment => {
                    segment_ids.contains(mapping.internal_object_id.as_str())
                }
            };
            if !internal_exists {
                return Err(MycelialGraphSidecarError::InvalidExternalMapping(
                    mapping.external_object_id.clone(),
                ));
            }
        }
        Ok(())
    }

    /// Content-addresses exact sidecar content. Validity is checked separately by `validate`.
    pub fn canonical_digest_hex(&self) -> Result<String, MycelialGraphSidecarError> {
        if self.external_object_mappings.len() > MAX_MYCELIAL_GRAPH_EXTERNAL_MAPPINGS {
            return Err(MycelialGraphSidecarError::TooManyExternalMappings(
                self.external_object_mappings.len(),
            ));
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-mycelial-graph-sidecar-v1");
        hasher.update(&self.schema_version.to_le_bytes());
        hash_string(&mut hasher, &self.sidecar_id);
        hash_string(&mut hasher, &self.source.dataset_uri);
        hash_string(&mut hasher, &self.source.dataset_path);
        hasher.update(&[match self.source.container_format {
            MycelialGraphImageContainerFormat::OmeZarrNgff => 0,
            MycelialGraphImageContainerFormat::N5 => 1,
            MycelialGraphImageContainerFormat::Other => 2,
        }]);
        hash_string(&mut hasher, &self.source.dataset_id);
        hash_string(&mut hasher, &self.source.source_observation_digest);
        hash_string(&mut hasher, &self.source.frame_id);
        hash_string(&mut hasher, &self.source.coordinate_frame_id);
        hash_string(&mut hasher, &self.source.calibration_digest);
        match &self.source.ome_ngff_version {
            Some(value) => {
                hasher.update(&[1]);
                hash_string(&mut hasher, value);
            }
            None => {
                hasher.update(&[0]);
            }
        }
        match &self.source.ome_ngff_coordinate_system_name {
            Some(value) => {
                hasher.update(&[1]);
                hash_string(&mut hasher, value);
            }
            None => {
                hasher.update(&[0]);
            }
        }
        match &self.source.ome_ngff_coordinate_transform_path {
            Some(value) => {
                hasher.update(&[1]);
                hash_string(&mut hasher, value);
            }
            None => {
                hasher.update(&[0]);
            }
        }
        match &self.source.ome_ngff_coordinate_transform_digest {
            Some(value) => {
                hasher.update(&[1]);
                hash_string(&mut hasher, value);
            }
            None => {
                hasher.update(&[0]);
            }
        }
        hash_string(&mut hasher, &self.producer.tool_id);
        hash_string(&mut hasher, &self.producer.tool_version);
        hash_string(&mut hasher, &self.producer.adapter_id);
        hash_string(&mut hasher, &self.producer.adapter_version);
        hash_graph_content(&mut hasher, &self.graph);
        hasher.update(&(self.external_object_mappings.len() as u64).to_le_bytes());
        for mapping in &self.external_object_mappings {
            hash_string(&mut hasher, &mapping.target_system_id);
            hash_string(&mut hasher, &mapping.target_system_version);
            hash_string(&mut hasher, &mapping.target_dataset_id);
            hash_string(&mut hasher, &mapping.external_object_id);
            hasher.update(&[match mapping.internal_object_kind {
                MycelialGraphObjectKind::Junction => 0,
                MycelialGraphObjectKind::Segment => 1,
            }]);
            hash_string(&mut hasher, &mapping.internal_object_id);
            hash_string(&mut hasher, &mapping.mapping_evidence_digest);
        }
        let digest = hasher.finalize();
        let mut output = String::with_capacity(64);
        use fmt::Write as _;
        for byte in digest.as_bytes() {
            let _ = write!(&mut output, "{byte:02x}");
        }
        Ok(output)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelialGraphInteroperabilityAssessment {
    pub graph_digest: String,
    pub junction_count: usize,
    pub segment_count: usize,
    pub connected_component_count: usize,
    pub cycle_rank: usize,
    pub branch_junction_count: usize,
    /// A forest can represent graph topology in tree-oriented formats, but only
    /// if the required geometric primitives are also available.
    pub forest_topology_compatible_with_swc: bool,
    /// False for the current endpoint-only segment schema: no centerline samples
    /// or per-segment radii are available to encode a truthful SWC polyline.
    pub swc_geometry_export_ready: bool,
    pub swc_blocked_reasons: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelialGraphSidecarRoundTripReport {
    pub source_sidecar_digest: String,
    pub imported_sidecar_digest: String,
    pub graph_digest_match: bool,
    pub source_observation_digest_match: bool,
    pub coordinate_frame_match: bool,
    pub calibration_digest_match: bool,
    pub junction_count_match: bool,
    pub segment_count_match: bool,
    pub cycle_rank_match: bool,
    pub lossless_graph_round_trip: bool,
}

fn graph_topology(graph: &AnnotatedMycelialSkeleton) -> (usize, usize, Vec<usize>) {
    let mut index_by_id = BTreeMap::new();
    for (index, junction) in graph.junctions.iter().enumerate() {
        index_by_id.insert(junction.id.as_str(), index);
    }
    let mut parent: Vec<usize> = (0..graph.junctions.len()).collect();
    let mut rank = vec![0usize; graph.junctions.len()];
    let mut degree = vec![0usize; graph.junctions.len()];

    fn find(parent: &mut [usize], mut node: usize) -> usize {
        while parent[node] != node {
            let grandparent = parent[parent[node]];
            parent[node] = grandparent;
            node = grandparent;
        }
        node
    }
    for segment in &graph.segments {
        let Some(&left) = index_by_id.get(segment.start_junction_id.as_str()) else {
            continue;
        };
        let Some(&right) = index_by_id.get(segment.end_junction_id.as_str()) else {
            continue;
        };
        degree[left] = degree[left].saturating_add(1);
        degree[right] = degree[right].saturating_add(1);
        let left_root = find(&mut parent, left);
        let right_root = find(&mut parent, right);
        if left_root != right_root {
            if rank[left_root] < rank[right_root] {
                parent[left_root] = right_root;
            } else if rank[left_root] > rank[right_root] {
                parent[right_root] = left_root;
            } else {
                parent[right_root] = left_root;
                rank[left_root] = rank[left_root].saturating_add(1);
            }
        }
    }
    let mut component_roots = BTreeSet::new();
    for index in 0..graph.junctions.len() {
        component_roots.insert(find(&mut parent, index));
    }
    let components = component_roots.len();
    let cycle_rank = graph
        .segments
        .len()
        .saturating_add(components)
        .saturating_sub(graph.junctions.len());
    (components, cycle_rank, degree)
}

pub fn assess_mycelial_graph_interoperability(
    sidecar: &MycelialGraphSidecar,
) -> Result<MycelialGraphInteroperabilityAssessment, MycelialGraphSidecarError> {
    sidecar.validate()?;
    let graph_digest = sidecar
        .graph
        .canonical_digest_hex()
        .map_err(|error| MycelialGraphSidecarError::InvalidGraph(error.to_string()))?;
    let (connected_component_count, cycle_rank, degree) = graph_topology(&sidecar.graph);
    let branch_junction_count = degree.iter().filter(|degree| **degree >= 3).count();
    let forest_topology_compatible_with_swc = cycle_rank == 0;
    // The current graph stores endpoints and segment length, not centerline
    // samples or radii. Never synthesize these missing physical measurements.
    let swc_geometry_export_ready = false;
    let mut swc_blocked_reasons = Vec::new();
    if !forest_topology_compatible_with_swc {
        swc_blocked_reasons.push("cyclic_graph_cannot_be_losslessly_encoded_as_swc".to_string());
    }
    swc_blocked_reasons.push("centerline_samples_are_not_present_in_source_schema".to_string());
    swc_blocked_reasons.push("segment_radii_are_not_present_in_source_schema".to_string());

    Ok(MycelialGraphInteroperabilityAssessment {
        graph_digest,
        junction_count: sidecar.graph.junctions.len(),
        segment_count: sidecar.graph.segments.len(),
        connected_component_count,
        cycle_rank,
        branch_junction_count,
        forest_topology_compatible_with_swc,
        swc_geometry_export_ready,
        swc_blocked_reasons,
    })
}

pub fn evaluate_mycelial_graph_sidecar_round_trip(
    source: &MycelialGraphSidecar,
    imported: &MycelialGraphSidecar,
) -> Result<MycelialGraphSidecarRoundTripReport, MycelialGraphSidecarError> {
    source.validate()?;
    imported.validate()?;
    let source_graph_digest = source
        .graph
        .canonical_digest_hex()
        .map_err(|error| MycelialGraphSidecarError::InvalidGraph(error.to_string()))?;
    let imported_graph_digest = imported
        .graph
        .canonical_digest_hex()
        .map_err(|error| MycelialGraphSidecarError::InvalidGraph(error.to_string()))?;
    let graph_digest_match = source_graph_digest == imported_graph_digest;
    let source_observation_digest_match =
        source.source.source_observation_digest == imported.source.source_observation_digest;
    let coordinate_frame_match =
        source.graph.coordinate_frame_id == imported.graph.coordinate_frame_id;
    let calibration_digest_match =
        source.graph.calibration_digest == imported.graph.calibration_digest;
    let junction_count_match = source.graph.junctions.len() == imported.graph.junctions.len();
    let segment_count_match = source.graph.segments.len() == imported.graph.segments.len();
    let source_assessment = assess_mycelial_graph_interoperability(source)?;
    let imported_assessment = assess_mycelial_graph_interoperability(imported)?;
    let cycle_rank_match = source_assessment.cycle_rank == imported_assessment.cycle_rank;
    let lossless_graph_round_trip = graph_digest_match
        && source_observation_digest_match
        && coordinate_frame_match
        && calibration_digest_match
        && junction_count_match
        && segment_count_match
        && cycle_rank_match;

    Ok(MycelialGraphSidecarRoundTripReport {
        source_sidecar_digest: source.canonical_digest_hex()?,
        imported_sidecar_digest: imported.canonical_digest_hex()?,
        graph_digest_match,
        source_observation_digest_match,
        coordinate_frame_match,
        calibration_digest_match,
        junction_count_match,
        segment_count_match,
        cycle_rank_match,
        lossless_graph_round_trip,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mycelial_reconstruction_metrics::{
        AnnotatedHyphaSegment, AnnotatedJunction, AnnotatedJunctionKind,
        MYCELIAL_RECONSTRUCTION_SCHEMA_VERSION, ReconstructionPoint3,
    };

    fn digest(character: char) -> String {
        std::iter::repeat_n(character, 64).collect()
    }

    fn skeleton() -> AnnotatedMycelialSkeleton {
        AnnotatedMycelialSkeleton {
            schema_version: MYCELIAL_RECONSTRUCTION_SCHEMA_VERSION,
            frame_id: "frame-1".to_string(),
            coordinate_frame_id: "physical-stage-1".to_string(),
            calibration_digest: digest('b'),
            source_observation_digest: digest('a'),
            dimensionality: ReconstructionDimensionality::ThreeD,
            length_unit: ReconstructionLengthUnit::Micrometer,
            producer_id: "human-annotation".to_string(),
            producer_version: "v1".to_string(),
            junctions: vec![
                AnnotatedJunction {
                    id: "A".to_string(),
                    kind: AnnotatedJunctionKind::Origin,
                    position: ReconstructionPoint3 {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                },
                AnnotatedJunction {
                    id: "B".to_string(),
                    kind: AnnotatedJunctionKind::Branch,
                    position: ReconstructionPoint3 {
                        x: 10.0,
                        y: 0.0,
                        z: 0.0,
                    },
                },
                AnnotatedJunction {
                    id: "C".to_string(),
                    kind: AnnotatedJunctionKind::Tip,
                    position: ReconstructionPoint3 {
                        x: 20.0,
                        y: 0.0,
                        z: 0.0,
                    },
                },
                AnnotatedJunction {
                    id: "D".to_string(),
                    kind: AnnotatedJunctionKind::Tip,
                    position: ReconstructionPoint3 {
                        x: 10.0,
                        y: 10.0,
                        z: 0.0,
                    },
                },
            ],
            segments: vec![
                AnnotatedHyphaSegment {
                    id: "ab".to_string(),
                    start_junction_id: "A".to_string(),
                    end_junction_id: "B".to_string(),
                    length: 10.0,
                },
                AnnotatedHyphaSegment {
                    id: "bc".to_string(),
                    start_junction_id: "B".to_string(),
                    end_junction_id: "C".to_string(),
                    length: 10.0,
                },
                AnnotatedHyphaSegment {
                    id: "bd".to_string(),
                    start_junction_id: "B".to_string(),
                    end_junction_id: "D".to_string(),
                    length: 10.0,
                },
                AnnotatedHyphaSegment {
                    id: "cd".to_string(),
                    start_junction_id: "C".to_string(),
                    end_junction_id: "D".to_string(),
                    length: 14.14,
                },
            ],
        }
    }

    fn sidecar() -> MycelialGraphSidecar {
        let graph = skeleton();
        MycelialGraphSidecar {
            schema_version: MYCELIAL_GRAPH_SIDECAR_SCHEMA_VERSION,
            sidecar_id: "sidecar-1".to_string(),
            source: MycelialGraphImageReference {
                dataset_uri: "https://example.org/mycelium.ome.zarr".to_string(),
                dataset_path: "0".to_string(),
                container_format: MycelialGraphImageContainerFormat::OmeZarrNgff,
                dataset_id: "sample-1".to_string(),
                source_observation_digest: graph.source_observation_digest.clone(),
                frame_id: graph.frame_id.clone(),
                coordinate_frame_id: graph.coordinate_frame_id.clone(),
                calibration_digest: graph.calibration_digest.clone(),
                ome_ngff_version: Some("0.6".to_string()),
                ome_ngff_coordinate_system_name: Some("physical".to_string()),
                ome_ngff_coordinate_transform_path: Some("scene/physical-transform".to_string()),
                ome_ngff_coordinate_transform_digest: Some(digest('e')),
            },
            producer: MycelialGraphSidecarProducer {
                tool_id: "snt".to_string(),
                tool_version: "5.0".to_string(),
                adapter_id: "symthaea-snt-bridge".to_string(),
                adapter_version: "1".to_string(),
            },
            graph,
            external_object_mappings: vec![
                MycelialExternalObjectMapping {
                    target_system_id: "snt".to_string(),
                    target_system_version: "5.0".to_string(),
                    target_dataset_id: "dataset-1".to_string(),
                    external_object_id: "node-1".to_string(),
                    internal_object_kind: MycelialGraphObjectKind::Junction,
                    internal_object_id: "A".to_string(),
                    mapping_evidence_digest: digest('c'),
                },
                MycelialExternalObjectMapping {
                    target_system_id: "snt".to_string(),
                    target_system_version: "5.0".to_string(),
                    target_dataset_id: "dataset-1".to_string(),
                    external_object_id: "path-1".to_string(),
                    internal_object_kind: MycelialGraphObjectKind::Segment,
                    internal_object_id: "ab".to_string(),
                    mapping_evidence_digest: digest('d'),
                },
            ],
        }
    }

    #[test]
    fn source_image_graph_frame_and_calibration_are_bound() {
        let value = sidecar();
        value.validate().unwrap();
        let mut detached = value;
        detached.source.source_observation_digest = digest('e');
        assert_eq!(
            detached.validate().unwrap_err(),
            MycelialGraphSidecarError::SourceGraphBindingMismatch
        );
    }

    #[test]
    fn ngff_06_named_coordinate_system_and_transform_digest_are_mandatory() {
        let mut value = sidecar();
        value.source.ome_ngff_coordinate_transform_digest = None;
        assert_eq!(
            value.validate().unwrap_err(),
            MycelialGraphSidecarError::InvalidSourceReference
        );
        let mut no_name = sidecar();
        no_name.source.ome_ngff_coordinate_system_name = None;
        assert_eq!(
            no_name.validate().unwrap_err(),
            MycelialGraphSidecarError::InvalidSourceReference
        );
    }

    #[test]
    fn external_mappings_must_reference_real_graph_objects() {
        let mut value = sidecar();
        value.external_object_mappings[1].internal_object_id = "missing".to_string();
        assert!(matches!(
            value.validate(),
            Err(MycelialGraphSidecarError::InvalidExternalMapping(_))
        ));
    }

    #[test]
    fn external_mapping_order_and_ids_are_unique() {
        let mut value = sidecar();
        value.external_object_mappings[1].external_object_id = "node-1".to_string();
        assert!(matches!(
            value.validate(),
            Err(MycelialGraphSidecarError::DuplicateOrUnorderedExternalMapping(_))
        ));
    }

    #[test]
    fn content_digest_changes_on_mapping_or_graph_mutation() {
        let value = sidecar();
        let original = value.canonical_digest_hex().unwrap();
        let mut changed_mapping = value.clone();
        changed_mapping.external_object_mappings[0].mapping_evidence_digest = digest('f');
        assert_ne!(changed_mapping.canonical_digest_hex().unwrap(), original);
        let mut changed_graph = value;
        changed_graph.graph.segments[0].length += 0.25;
        assert_ne!(changed_graph.canonical_digest_hex().unwrap(), original);
    }

    #[test]
    fn cyclic_graph_is_not_swc_topology_compatible() {
        let value = sidecar();
        let report = assess_mycelial_graph_interoperability(&value).unwrap();
        assert_eq!(report.cycle_rank, 1);
        assert!(!report.forest_topology_compatible_with_swc);
        assert!(!report.swc_geometry_export_ready);
        assert!(
            report
                .swc_blocked_reasons
                .iter()
                .any(|reason| reason == "cyclic_graph_cannot_be_losslessly_encoded_as_swc")
        );
    }

    #[test]
    fn a_tree_can_be_topology_compatible_but_still_lack_swc_geometry() {
        let mut value = sidecar();
        value.graph.segments.pop();
        let report = assess_mycelial_graph_interoperability(&value).unwrap();
        assert_eq!(report.cycle_rank, 0);
        assert!(report.forest_topology_compatible_with_swc);
        assert!(!report.swc_geometry_export_ready);
        assert!(
            report
                .swc_blocked_reasons
                .iter()
                .any(|reason| reason == "centerline_samples_are_not_present_in_source_schema")
        );
    }

    #[test]
    fn round_trip_report_detects_exact_graph_and_source_changes() {
        let original = sidecar();
        let mut imported = original.clone();
        imported.sidecar_id = "round-tripped-sidecar".to_string();
        imported.producer.tool_id = "arivis".to_string();
        imported.producer.tool_version = "4.6".to_string();
        imported.producer.adapter_id = "arivis-import-export".to_string();
        let exact = evaluate_mycelial_graph_sidecar_round_trip(&original, &imported).unwrap();
        assert!(exact.graph_digest_match);
        assert!(exact.lossless_graph_round_trip);
        imported.graph.segments[0].length += 0.5;
        let changed = evaluate_mycelial_graph_sidecar_round_trip(&original, &imported).unwrap();
        assert!(!changed.graph_digest_match);
        assert!(!changed.lossless_graph_round_trip);
    }

    #[test]
    fn dataset_path_rejects_parent_traversal() {
        let mut value = sidecar();
        value.source.dataset_path = "0/../1".to_string();
        assert_eq!(
            value.validate().unwrap_err(),
            MycelialGraphSidecarError::InvalidSourceReference
        );
    }
}
