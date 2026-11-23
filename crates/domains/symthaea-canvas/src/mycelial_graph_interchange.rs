// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Loss-aware interchange sidecars for cyclic mycelial graphs.
//!
//! The canonical sidecar preserves the complete annotated graph and its source
//! provenance. SWC is treated as a constrained projection: it can encode a
//! rooted forest but cannot preserve anastomosis semantics, scalar hyphal arc
//! lengths, measured centerlines or the full graph provenance. Cyclic inputs
//! therefore fail closed rather than silently dropping fusion edges.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::mycelial_reconstruction_metrics::{
    AnnotatedJunctionKind, AnnotatedMycelialSkeleton, ReconstructionLengthUnit,
    ReconstructionPoint3,
};

pub const MYCELIAL_GRAPH_INTERCHANGE_SCHEMA_VERSION: u16 = 1;
pub const MYCELIAL_SWC_PROJECTION_SCHEMA_VERSION: u16 = 1;
pub const MAX_MYCELIAL_GRAPH_FOREIGN_BINDINGS: usize = 65_536;
pub const MAX_MYCELIAL_INTERCHANGE_ID_BYTES: usize = 128;
pub const MAX_MYCELIAL_SOURCE_URI_BYTES: usize = 2_048;
pub const MAX_MYCELIAL_SWC_NODES: usize = 65_536;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialSourceFormat {
    OmeZarr,
    Imaris,
    ZeissArivis,
    FijiSnt,
    GenericGraph,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialRightsStatus {
    VerifiedOpen,
    Restricted,
    PermissionRequired,
    Unclear,
    NotApplicable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelialSourceDatasetReference {
    /// URI or relative store path. Parent-directory traversal is not allowed.
    pub source_uri: String,
    pub source_format: MycelialSourceFormat,
    pub dataset_id: String,
    pub acquisition_session_id: String,
    /// Digest of the immutable source image / volume container or dataset manifest.
    pub source_content_digest: String,
    /// Digest of source axes and multiscale metadata (e.g. OME-Zarr axes).
    pub axes_metadata_digest: String,
    /// Digest of the spatial transform chain connecting source and graph coordinates.
    pub spatial_transform_digest: String,
    /// Rights state is explicit; a paper licence does not imply every data file is open.
    pub rights_status: MycelialRightsStatus,
    /// Digest of the file-level rights and access evidence record.
    pub rights_evidence_digest: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialForeignEntityKind {
    Junction,
    Segment,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelialForeignIdentityBinding {
    pub tool_id: String,
    pub entity_kind: MycelialForeignEntityKind,
    /// Local ID in the canonical annotated skeleton.
    pub local_entity_id: String,
    /// Stable object / trace ID in the external application or file.
    pub foreign_object_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialGraphInterchangeSidecar {
    pub schema_version: u16,
    pub source: MycelialSourceDatasetReference,
    /// Canonical graph snapshot; source observations are referenced, not copied.
    pub graph: AnnotatedMycelialSkeleton,
    pub graph_digest: String,
    pub connected_components: usize,
    pub cycle_rank: usize,
    /// Sorted by (tool_id, entity_kind, local_entity_id, foreign_object_id).
    pub foreign_identity_bindings: Vec<MycelialForeignIdentityBinding>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialSwcLoss {
    ScalarSegmentArcLength,
    MeasuredHyphaCenterline,
    AnastomosisAndFusionSemantics,
    JunctionConfidenceAndUncertainty,
    TemporalEvidence,
    NativeSourceAndCalibrationProvenance,
    OriginalSegmentIdentity,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialSwcJunctionRadius {
    pub junction_id: String,
    /// Must be measured / explicitly supplied; the exporter never invents radii.
    pub radius: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialSwcNode {
    pub id: u32,
    /// SWC type 5 = fork point, 6 = end point, 0 = undefined/custom.
    pub node_type: i32,
    pub position: ReconstructionPoint3,
    pub radius: f64,
    /// -1 denotes a component root.
    pub parent_id: i64,
    /// Retained in the audit model; not encoded in the SWC row itself.
    pub source_junction_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialSwcProjectionAudit {
    pub schema_version: u16,
    pub source_graph_digest: String,
    pub projection_digest: String,
    pub coordinate_frame_id: String,
    pub length_unit: ReconstructionLengthUnit,
    pub component_count: usize,
    pub node_count: usize,
    pub edge_count: usize,
    pub source_cycle_rank: usize,
    /// True for this endpoint-only projection even when the source is a tree.
    pub lossy: bool,
    pub geometry_semantics: String,
    pub discarded_semantics: Vec<MycelialSwcLoss>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialSwcProjection {
    pub nodes: Vec<MycelialSwcNode>,
    pub audit: MycelialSwcProjectionAudit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MycelialGraphInterchangeError {
    InvalidSchemaVersion(u16),
    InvalidSourceMetadata,
    InvalidGraph(String),
    SourceGraphDigestMismatch,
    TopologySummaryMismatch,
    TooManyForeignBindings(usize),
    InvalidForeignBinding(String),
    DuplicateOrUnorderedForeignBinding(String),
    UnknownLocalEntity(String),
    TooManySwcNodes(usize),
    CyclicGraphNotRepresentable(usize),
    InvalidSwcRoots,
    InvalidSwcRadii,
    InvalidSwcRadius(String),
    ReportMismatch,
}

impl fmt::Display for MycelialGraphInterchangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSchemaVersion(version) => {
                write!(f, "unsupported mycelial interchange schema version {version}")
            }
            Self::InvalidSourceMetadata => {
                write!(f, "invalid source dataset reference or rights evidence")
            }
            Self::InvalidGraph(reason) => write!(f, "invalid graph in interchange sidecar: {reason}"),
            Self::SourceGraphDigestMismatch => {
                write!(f, "sidecar graph digest does not match the embedded graph")
            }
            Self::TopologySummaryMismatch => {
                write!(f, "sidecar topology summary does not match the embedded graph")
            }
            Self::TooManyForeignBindings(count) => {
                write!(f, "foreign binding count {count} exceeds {MAX_MYCELIAL_GRAPH_FOREIGN_BINDINGS}")
            }
            Self::InvalidForeignBinding(id) => {
                write!(f, "invalid foreign identity binding {id}")
            }
            Self::DuplicateOrUnorderedForeignBinding(id) => {
                write!(f, "foreign identity bindings are duplicated or unordered at {id}")
            }
            Self::UnknownLocalEntity(id) => {
                write!(f, "foreign identity binding references unknown local entity {id}")
            }
            Self::TooManySwcNodes(count) => {
                write!(f, "SWC node count {count} exceeds {MAX_MYCELIAL_SWC_NODES}")
            }
            Self::CyclicGraphNotRepresentable(rank) => {
                write!(f, "SWC cannot represent source graph with cycle rank {rank}")
            }
            Self::InvalidSwcRoots => {
                write!(f, "SWC projection requires exactly one explicit root per connected component")
            }
            Self::InvalidSwcRadii => {
                write!(f, "SWC projection requires one canonical measured radius for every junction")
            }
            Self::InvalidSwcRadius(id) => write!(f, "invalid SWC radius for junction {id}"),
            Self::ReportMismatch => {
                write!(f, "SWC projection does not match independent recomputation")
            }
        }
    }
}

impl Error for MycelialGraphInterchangeError {}

fn valid_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value.len() <= MAX_MYCELIAL_INTERCHANGE_ID_BYTES
        && value.bytes().all(|byte| !byte.is_ascii_control())
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn valid_source_uri(value: &str) -> bool {
    if value.trim().is_empty()
        || value.len() > MAX_MYCELIAL_SOURCE_URI_BYTES
        || value.bytes().any(|byte| byte.is_ascii_control() || byte.is_ascii_whitespace())
    {
        return false;
    }
    !value
        .split(['/', '\\'])
        .any(|component| component == "..")
}

fn hash_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hex_digest(bytes: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    use fmt::Write as _;
    for byte in bytes {
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

#[derive(Debug, Clone)]
struct GraphTopology {
    components: usize,
    cycle_rank: usize,
    adjacency: BTreeMap<String, Vec<(String, String)>>,
}

fn graph_topology(
    graph: &AnnotatedMycelialSkeleton,
) -> Result<GraphTopology, MycelialGraphInterchangeError> {
    let mut parent: BTreeMap<String, String> = graph
        .junctions
        .iter()
        .map(|junction| (junction.id.clone(), junction.id.clone()))
        .collect();

    fn find(parent: &mut BTreeMap<String, String>, id: &str) -> String {
        let current = parent
            .get(id)
            .expect("validated junction ID")
            .clone();
        if current == id {
            return current;
        }
        let root = find(parent, &current);
        parent.insert(id.to_string(), root.clone());
        root
    }

    let mut adjacency: BTreeMap<String, Vec<(String, String)>> = graph
        .junctions
        .iter()
        .map(|junction| (junction.id.clone(), Vec::new()))
        .collect();

    for segment in &graph.segments {
        let left = find(&mut parent, &segment.start_junction_id);
        let right = find(&mut parent, &segment.end_junction_id);
        if left != right {
            parent.insert(left, right);
        }
        adjacency
            .get_mut(&segment.start_junction_id)
            .expect("validated segment source")
            .push((segment.end_junction_id.clone(), segment.id.clone()));
        adjacency
            .get_mut(&segment.end_junction_id)
            .expect("validated segment target")
            .push((segment.start_junction_id.clone(), segment.id.clone()));
    }

    for neighbours in adjacency.values_mut() {
        neighbours.sort();
    }
    let mut roots = BTreeSet::new();
    let node_count = parent.len();
    for id in parent.keys().cloned().collect::<Vec<_>>() {
        roots.insert(find(&mut parent, &id));
    }
    let components = roots.len();
    let cycle_rank = graph
        .segments
        .len()
        .checked_add(components)
        .and_then(|value| value.checked_sub(node_count))
        .ok_or_else(|| {
            MycelialGraphInterchangeError::InvalidGraph("invalid graph topology arithmetic".to_string())
        })?;
    Ok(GraphTopology {
        components,
        cycle_rank,
        adjacency,
    })
}

impl MycelialGraphInterchangeSidecar {
    pub fn new(
        source: MycelialSourceDatasetReference,
        graph: AnnotatedMycelialSkeleton,
        foreign_identity_bindings: Vec<MycelialForeignIdentityBinding>,
    ) -> Result<Self, MycelialGraphInterchangeError> {
        graph.validate().map_err(|error| {
            MycelialGraphInterchangeError::InvalidGraph(error.to_string())
        })?;
        let graph_digest = graph.canonical_digest_hex().map_err(|error| {
            MycelialGraphInterchangeError::InvalidGraph(error.to_string())
        })?;
        let topology = graph_topology(&graph)?;
        let sidecar = Self {
            schema_version: MYCELIAL_GRAPH_INTERCHANGE_SCHEMA_VERSION,
            source,
            graph,
            graph_digest,
            connected_components: topology.components,
            cycle_rank: topology.cycle_rank,
            foreign_identity_bindings,
        };
        sidecar.validate()?;
        Ok(sidecar)
    }

    pub fn validate(&self) -> Result<(), MycelialGraphInterchangeError> {
        if self.schema_version != MYCELIAL_GRAPH_INTERCHANGE_SCHEMA_VERSION {
            return Err(MycelialGraphInterchangeError::InvalidSchemaVersion(
                self.schema_version,
            ));
        }
        if !valid_source_uri(&self.source.source_uri)
            || !valid_id(&self.source.dataset_id)
            || !valid_id(&self.source.acquisition_session_id)
            || !valid_digest(&self.source.source_content_digest)
            || !valid_digest(&self.source.axes_metadata_digest)
            || !valid_digest(&self.source.spatial_transform_digest)
            || !valid_digest(&self.source.rights_evidence_digest)
        {
            return Err(MycelialGraphInterchangeError::InvalidSourceMetadata);
        }
        self.graph.validate().map_err(|error| {
            MycelialGraphInterchangeError::InvalidGraph(error.to_string())
        })?;
        let actual_graph_digest = self.graph.canonical_digest_hex().map_err(|error| {
            MycelialGraphInterchangeError::InvalidGraph(error.to_string())
        })?;
        if self.graph_digest != actual_graph_digest
            || self.graph.source_observation_digest != self.source.source_content_digest
        {
            return Err(MycelialGraphInterchangeError::SourceGraphDigestMismatch);
        }
        let topology = graph_topology(&self.graph)?;
        if self.connected_components != topology.components || self.cycle_rank != topology.cycle_rank {
            return Err(MycelialGraphInterchangeError::TopologySummaryMismatch);
        }
        if self.foreign_identity_bindings.len() > MAX_MYCELIAL_GRAPH_FOREIGN_BINDINGS {
            return Err(MycelialGraphInterchangeError::TooManyForeignBindings(
                self.foreign_identity_bindings.len(),
            ));
        }

        let junction_ids: BTreeSet<_> = self.graph.junctions.iter().map(|item| item.id.as_str()).collect();
        let segment_ids: BTreeSet<_> = self.graph.segments.iter().map(|item| item.id.as_str()).collect();
        let mut previous_key: Option<(String, u8, String, String)> = None;
        for binding in &self.foreign_identity_bindings {
            let key = (
                binding.tool_id.clone(),
                match binding.entity_kind {
                    MycelialForeignEntityKind::Junction => 0,
                    MycelialForeignEntityKind::Segment => 1,
                },
                binding.local_entity_id.clone(),
                binding.foreign_object_id.clone(),
            );
            if !valid_id(&binding.tool_id)
                || !valid_id(&binding.local_entity_id)
                || !valid_id(&binding.foreign_object_id)
            {
                return Err(MycelialGraphInterchangeError::InvalidForeignBinding(
                    binding.foreign_object_id.clone(),
                ));
            }
            if previous_key.as_ref().is_some_and(|previous| previous >= &key) {
                return Err(
                    MycelialGraphInterchangeError::DuplicateOrUnorderedForeignBinding(
                        binding.foreign_object_id.clone(),
                    ),
                );
            }
            previous_key = Some(key);
            let known = match binding.entity_kind {
                MycelialForeignEntityKind::Junction => junction_ids.contains(binding.local_entity_id.as_str()),
                MycelialForeignEntityKind::Segment => segment_ids.contains(binding.local_entity_id.as_str()),
            };
            if !known {
                return Err(MycelialGraphInterchangeError::UnknownLocalEntity(
                    binding.local_entity_id.clone(),
                ));
            }
        }
        Ok(())
    }

    pub fn canonical_digest_hex(&self) -> Result<String, MycelialGraphInterchangeError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-mycelial-graph-interchange-sidecar-v1");
        hasher.update(&self.schema_version.to_le_bytes());
        hash_string(&mut hasher, &self.source.source_uri);
        hasher.update(&[match self.source.source_format {
            MycelialSourceFormat::OmeZarr => 0,
            MycelialSourceFormat::Imaris => 1,
            MycelialSourceFormat::ZeissArivis => 2,
            MycelialSourceFormat::FijiSnt => 3,
            MycelialSourceFormat::GenericGraph => 4,
        }]);
        hash_string(&mut hasher, &self.source.dataset_id);
        hash_string(&mut hasher, &self.source.acquisition_session_id);
        hash_string(&mut hasher, &self.source.source_content_digest);
        hash_string(&mut hasher, &self.source.axes_metadata_digest);
        hash_string(&mut hasher, &self.source.spatial_transform_digest);
        hasher.update(&[match self.source.rights_status {
            MycelialRightsStatus::VerifiedOpen => 0,
            MycelialRightsStatus::Restricted => 1,
            MycelialRightsStatus::PermissionRequired => 2,
            MycelialRightsStatus::Unclear => 3,
            MycelialRightsStatus::NotApplicable => 4,
        }]);
        hash_string(&mut hasher, &self.source.rights_evidence_digest);
        hash_string(&mut hasher, &self.graph_digest);
        hasher.update(&(self.connected_components as u64).to_le_bytes());
        hasher.update(&(self.cycle_rank as u64).to_le_bytes());
        hasher.update(&(self.foreign_identity_bindings.len() as u64).to_le_bytes());
        for binding in &self.foreign_identity_bindings {
            hash_string(&mut hasher, &binding.tool_id);
            hasher.update(&[match binding.entity_kind {
                MycelialForeignEntityKind::Junction => 0,
                MycelialForeignEntityKind::Segment => 1,
            }]);
            hash_string(&mut hasher, &binding.local_entity_id);
            hash_string(&mut hasher, &binding.foreign_object_id);
        }
        Ok(hex_digest(hasher.finalize().as_bytes()))
    }
}

fn swc_node_type(kind: AnnotatedJunctionKind) -> i32 {
    match kind {
        AnnotatedJunctionKind::Branch => 5,
        AnnotatedJunctionKind::Tip => 6,
        // Origin, anastomosis, crossing and uncertain are custom meanings;
        // type 0 avoids falsely relabelling them as a soma or neurite subtype.
        AnnotatedJunctionKind::Origin
        | AnnotatedJunctionKind::Anastomosis
        | AnnotatedJunctionKind::ProjectedCrossing
        | AnnotatedJunctionKind::Uncertain => 0,
    }
}

fn swc_projection_digest(nodes: &[MycelialSwcNode]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-mycelial-swc-endpoint-projection-v1");
    hasher.update(&(nodes.len() as u64).to_le_bytes());
    for node in nodes {
        hasher.update(&node.id.to_le_bytes());
        hasher.update(&node.node_type.to_le_bytes());
        hasher.update(&node.position.x.to_bits().to_le_bytes());
        hasher.update(&node.position.y.to_bits().to_le_bytes());
        hasher.update(&node.position.z.to_bits().to_le_bytes());
        hasher.update(&node.radius.to_bits().to_le_bytes());
        hasher.update(&node.parent_id.to_le_bytes());
        hash_string(&mut hasher, &node.source_junction_id);
    }
    hex_digest(hasher.finalize().as_bytes())
}

pub fn project_mycelial_skeleton_to_swc(
    graph: &AnnotatedMycelialSkeleton,
    root_junction_ids: &[String],
    radii: &[MycelialSwcJunctionRadius],
) -> Result<MycelialSwcProjection, MycelialGraphInterchangeError> {
    graph.validate().map_err(|error| {
        MycelialGraphInterchangeError::InvalidGraph(error.to_string())
    })?;
    if graph.junctions.len() > MAX_MYCELIAL_SWC_NODES {
        return Err(MycelialGraphInterchangeError::TooManySwcNodes(
            graph.junctions.len(),
        ));
    }
    let topology = graph_topology(graph)?;
    if topology.cycle_rank != 0 {
        return Err(MycelialGraphInterchangeError::CyclicGraphNotRepresentable(
            topology.cycle_rank,
        ));
    }

    let mut sorted_roots = root_junction_ids.to_vec();
    sorted_roots.sort();
    if sorted_roots.windows(2).any(|pair| pair[0] == pair[1])
        || sorted_roots.len() != topology.components
    {
        return Err(MycelialGraphInterchangeError::InvalidSwcRoots);
    }

    let mut radius_by_id = BTreeMap::new();
    let mut previous_radius_id: Option<&str> = None;
    for radius in radii {
        if !valid_id(&radius.junction_id)
            || previous_radius_id.is_some_and(|previous| previous >= radius.junction_id.as_str())
        {
            return Err(MycelialGraphInterchangeError::InvalidSwcRadii);
        }
        if !radius.radius.is_finite() || radius.radius <= 0.0 {
            return Err(MycelialGraphInterchangeError::InvalidSwcRadius(
                radius.junction_id.clone(),
            ));
        }
        previous_radius_id = Some(&radius.junction_id);
        radius_by_id.insert(radius.junction_id.as_str(), radius.radius);
    }
    if radius_by_id.len() != graph.junctions.len()
        || graph
            .junctions
            .iter()
            .any(|junction| !radius_by_id.contains_key(junction.id.as_str()))
    {
        return Err(MycelialGraphInterchangeError::InvalidSwcRadii);
    }

    let junctions: BTreeMap<_, _> = graph
        .junctions
        .iter()
        .map(|junction| (junction.id.as_str(), junction))
        .collect();
    let mut visited = BTreeSet::new();
    let mut traversal = Vec::<(String, Option<String>)>::new();

    for root_id in sorted_roots {
        if !junctions.contains_key(root_id.as_str()) {
            return Err(MycelialGraphInterchangeError::InvalidSwcRoots);
        }
        if visited.contains(root_id.as_str()) {
            return Err(MycelialGraphInterchangeError::InvalidSwcRoots);
        }
        let mut queue = VecDeque::new();
        queue.push_back((root_id.clone(), None));
        visited.insert(root_id.clone());
        while let Some((current_id, parent_id)) = queue.pop_front() {
            traversal.push((current_id.clone(), parent_id));
            for (neighbour_id, _segment_id) in topology
                .adjacency
                .get(&current_id)
                .expect("topology contains every junction")
            {
                if visited.insert(neighbour_id.clone()) {
                    queue.push_back((neighbour_id.clone(), Some(current_id.clone())));
                }
            }
        }
    }

    if visited.len() != graph.junctions.len() {
        return Err(MycelialGraphInterchangeError::InvalidSwcRoots);
    }
    let numeric_ids: BTreeMap<_, _> = traversal
        .iter()
        .enumerate()
        .map(|(index, (junction_id, _))| (junction_id.as_str(), (index + 1) as u32))
        .collect();
    let nodes: Vec<_> = traversal
        .iter()
        .enumerate()
        .map(|(index, (junction_id, parent_id))| {
            let junction = junctions[junction_id.as_str()];
            MycelialSwcNode {
                id: (index + 1) as u32,
                node_type: swc_node_type(junction.kind),
                position: junction.position,
                radius: radius_by_id[junction_id.as_str()],
                parent_id: parent_id
                    .as_ref()
                    .map(|parent| numeric_ids[parent.as_str()] as i64)
                    .unwrap_or(-1),
                source_junction_id: junction_id.clone(),
            }
        })
        .collect();

    let source_graph_digest = graph.canonical_digest_hex().map_err(|error| {
        MycelialGraphInterchangeError::InvalidGraph(error.to_string())
    })?;
    let projection_digest = swc_projection_digest(&nodes);
    let audit = MycelialSwcProjectionAudit {
        schema_version: MYCELIAL_SWC_PROJECTION_SCHEMA_VERSION,
        source_graph_digest,
        projection_digest,
        coordinate_frame_id: graph.coordinate_frame_id.clone(),
        length_unit: graph.length_unit,
        component_count: topology.components,
        node_count: nodes.len(),
        edge_count: graph.segments.len(),
        source_cycle_rank: topology.cycle_rank,
        lossy: true,
        geometry_semantics: "junction coordinates connected by endpoint-only straight segments; not measured hyphal centerlines".to_string(),
        discarded_semantics: vec![
            MycelialSwcLoss::ScalarSegmentArcLength,
            MycelialSwcLoss::MeasuredHyphaCenterline,
            MycelialSwcLoss::AnastomosisAndFusionSemantics,
            MycelialSwcLoss::JunctionConfidenceAndUncertainty,
            MycelialSwcLoss::TemporalEvidence,
            MycelialSwcLoss::NativeSourceAndCalibrationProvenance,
            MycelialSwcLoss::OriginalSegmentIdentity,
        ],
    };
    Ok(MycelialSwcProjection { nodes, audit })
}

impl MycelialSwcProjection {
    pub fn validate_against(
        &self,
        graph: &AnnotatedMycelialSkeleton,
        root_junction_ids: &[String],
        radii: &[MycelialSwcJunctionRadius],
    ) -> Result<(), MycelialGraphInterchangeError> {
        let expected =
            project_mycelial_skeleton_to_swc(graph, root_junction_ids, radii)?;
        if self != &expected {
            return Err(MycelialGraphInterchangeError::ReportMismatch);
        }
        Ok(())
    }

    /// Produce deterministic SWC text with an explicit audit header.
    pub fn to_swc_text(&self) -> String {
        let mut output = String::new();
        use fmt::Write as _;
        let _ = writeln!(output, "# Symthaea loss-aware SWC projection v{}", self.audit.schema_version);
        let _ = writeln!(output, "# source_graph_digest {}", self.audit.source_graph_digest);
        let _ = writeln!(output, "# projection_digest {}", self.audit.projection_digest);
        let _ = writeln!(output, "# coordinate_frame_id {}", self.audit.coordinate_frame_id);
        let _ = writeln!(output, "# length_unit {:?}", self.audit.length_unit);
        let _ = writeln!(output, "# geometry_semantics {}", self.audit.geometry_semantics);
        let _ = writeln!(output, "# lossy_projection true");
        for node in &self.nodes {
            let _ = writeln!(
                output,
                "{} {} {:.9} {:.9} {:.9} {:.9} {}",
                node.id,
                node.node_type,
                node.position.x,
                node.position.y,
                node.position.z,
                node.radius,
                node.parent_id
            );
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mycelial_reconstruction_metrics::{
        AnnotatedHyphaSegment, AnnotatedJunction, MYCELIAL_RECONSTRUCTION_SCHEMA_VERSION,
    };

    fn digest(character: char) -> String {
        std::iter::repeat_n(character, 64).collect()
    }

    fn graph_with_cycle() -> AnnotatedMycelialSkeleton {
        AnnotatedMycelialSkeleton {
            schema_version: MYCELIAL_RECONSTRUCTION_SCHEMA_VERSION,
            frame_id: "frame-interchange".to_string(),
            coordinate_frame_id: "stage-frame-1".to_string(),
            calibration_digest: digest('b'),
            source_observation_digest: digest('a'),
            dimensionality: crate::mycelial_reconstruction_metrics::ReconstructionDimensionality::ThreeD,
            length_unit: ReconstructionLengthUnit::Micrometer,
            producer_id: "manual-annotation".to_string(),
            producer_version: "v1".to_string(),
            junctions: vec![
                AnnotatedJunction {
                    id: "A".to_string(),
                    kind: AnnotatedJunctionKind::Origin,
                    position: ReconstructionPoint3 { x: 0.0, y: 0.0, z: 0.0 },
                },
                AnnotatedJunction {
                    id: "B".to_string(),
                    kind: AnnotatedJunctionKind::Tip,
                    position: ReconstructionPoint3 { x: 1.0, y: 0.0, z: 0.0 },
                },
                AnnotatedJunction {
                    id: "C".to_string(),
                    kind: AnnotatedJunctionKind::Anastomosis,
                    position: ReconstructionPoint3 { x: 0.0, y: 1.0, z: 0.0 },
                },
            ],
            segments: vec![
                AnnotatedHyphaSegment {
                    id: "ab".to_string(),
                    start_junction_id: "A".to_string(),
                    end_junction_id: "B".to_string(),
                    length: 1.0,
                },
                AnnotatedHyphaSegment {
                    id: "bc".to_string(),
                    start_junction_id: "B".to_string(),
                    end_junction_id: "C".to_string(),
                    length: 1.0,
                },
                AnnotatedHyphaSegment {
                    id: "ca".to_string(),
                    start_junction_id: "C".to_string(),
                    end_junction_id: "A".to_string(),
                    length: 1.0,
                },
            ],
        }
    }

    fn source_ref() -> MycelialSourceDatasetReference {
        MycelialSourceDatasetReference {
            source_uri: "https://example.org/experiment.ome.zarr".to_string(),
            source_format: MycelialSourceFormat::OmeZarr,
            dataset_id: "dataset-1".to_string(),
            acquisition_session_id: "session-1".to_string(),
            source_content_digest: digest('a'),
            axes_metadata_digest: digest('c'),
            spatial_transform_digest: digest('d'),
            rights_status: MycelialRightsStatus::VerifiedOpen,
            rights_evidence_digest: digest('e'),
        }
    }

    fn tree_graph() -> AnnotatedMycelialSkeleton {
        let mut graph = graph_with_cycle();
        graph.segments.pop();
        graph
    }

    fn radii(graph: &AnnotatedMycelialSkeleton) -> Vec<MycelialSwcJunctionRadius> {
        graph
            .junctions
            .iter()
            .map(|junction| MycelialSwcJunctionRadius {
                junction_id: junction.id.clone(),
                radius: 0.5,
            })
            .collect()
    }

    #[test]
    fn sidecar_binds_graph_source_topology_and_rights_provenance() {
        let graph = graph_with_cycle();
        let sidecar =
            MycelialGraphInterchangeSidecar::new(source_ref(), graph, Vec::new()).unwrap();
        assert_eq!(sidecar.connected_components, 1);
        assert_eq!(sidecar.cycle_rank, 1);
        assert_eq!(
            sidecar.canonical_digest_hex().unwrap(),
            sidecar.canonical_digest_hex().unwrap()
        );
        let mut tampered = sidecar.clone();
        tampered.graph_digest = digest('f');
        assert_eq!(
            tampered.validate().unwrap_err(),
            MycelialGraphInterchangeError::SourceGraphDigestMismatch
        );
    }

    #[test]
    fn source_uri_parent_traversal_is_rejected() {
        let graph = graph_with_cycle();
        let mut source = source_ref();
        source.source_uri = "https://example.org/data/../private.ome.zarr".to_string();
        assert_eq!(
            MycelialGraphInterchangeSidecar::new(source, graph, Vec::new()).unwrap_err(),
            MycelialGraphInterchangeError::InvalidSourceMetadata
        );
    }

    #[test]
    fn foreign_tool_ids_are_bound_to_existing_local_graph_entities() {
        let graph = tree_graph();
        let sidecar = MycelialGraphInterchangeSidecar::new(
            source_ref(),
            graph,
            vec![
                MycelialForeignIdentityBinding {
                    tool_id: "fiji-snt".to_string(),
                    entity_kind: MycelialForeignEntityKind::Junction,
                    local_entity_id: "A".to_string(),
                    foreign_object_id: "snt-node-001".to_string(),
                },
                MycelialForeignIdentityBinding {
                    tool_id: "imaris".to_string(),
                    entity_kind: MycelialForeignEntityKind::Segment,
                    local_entity_id: "ab".to_string(),
                    foreign_object_id: "filament-42".to_string(),
                },
            ],
        )
        .unwrap();
        sidecar.validate().unwrap();

        let mut tampered = sidecar;
        tampered.foreign_identity_bindings[1].local_entity_id = "missing-edge".to_string();
        assert!(matches!(
            tampered.validate(),
            Err(MycelialGraphInterchangeError::UnknownLocalEntity(id))
                if id == "missing-edge"
        ));
    }

    #[test]
    fn swc_projection_is_deterministic_and_declares_all_losses() {
        let graph = tree_graph();
        let roots = vec!["A".to_string()];
        let radius_values = radii(&graph);
        let projection =
            project_mycelial_skeleton_to_swc(&graph, &roots, &radius_values).unwrap();
        assert_eq!(projection.nodes.len(), 3);
        assert_eq!(projection.audit.edge_count, 2);
        assert_eq!(projection.audit.source_cycle_rank, 0);
        assert!(projection.audit.lossy);
        assert!(
            projection
                .audit
                .discarded_semantics
                .contains(&MycelialSwcLoss::AnastomosisAndFusionSemantics)
        );
        assert_eq!(
            projection.audit.projection_digest,
            project_mycelial_skeleton_to_swc(&graph, &roots, &radius_values)
                .unwrap()
                .audit
                .projection_digest
        );
        projection.validate_against(&graph, &roots, &radius_values).unwrap();
        let text = projection.to_swc_text();
        assert!(text.contains("# lossy_projection true"));
        assert!(text.contains(&projection.audit.source_graph_digest));
    }

    #[test]
    fn cyclic_graph_is_rejected_by_swc_projection() {
        let graph = graph_with_cycle();
        assert_eq!(
            project_mycelial_skeleton_to_swc(
                &graph,
                &["A".to_string()],
                &radii(&graph)
            )
            .unwrap_err(),
            MycelialGraphInterchangeError::CyclicGraphNotRepresentable(1)
        );
    }

    #[test]
    fn swc_projection_requires_one_explicit_root_per_component() {
        let graph = graph_with_cycle();
        let mut tree = graph.clone();
        tree.segments.pop();
        assert_eq!(
            project_mycelial_skeleton_to_swc(&tree, &[], &radii(&tree)).unwrap_err(),
            MycelialGraphInterchangeError::InvalidSwcRoots
        );
    }

    #[test]
    fn swc_projection_requires_measured_radius_for_every_junction() {
        let graph = tree_graph();
        let mut measured_radii = radii(&graph);
        measured_radii.pop();
        assert_eq!(
            project_mycelial_skeleton_to_swc(&graph, &["A".to_string()], &measured_radii)
                .unwrap_err(),
            MycelialGraphInterchangeError::InvalidSwcRadii
        );
    }

    #[test]
    fn invalid_radius_is_not_replaced_with_an_invented_value() {
        let graph = tree_graph();
        let mut measured_radii = radii(&graph);
        measured_radii[0].radius = 0.0;
        assert!(matches!(
            project_mycelial_skeleton_to_swc(&graph, &["A".to_string()], &measured_radii),
            Err(MycelialGraphInterchangeError::InvalidSwcRadius(_))
        ));
    }
}
