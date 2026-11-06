//! Model-independent recoverability audit for a frozen benchmark search surface.
//!
//! The evaluator classifies whether an exact required evidence artifact is
//! reachable from a frozen start state through declared route operations and
//! within a fixed budget. It does not execute a retriever, query the network,
//! use an LLM judge, or claim open-world search completeness.

use crate::sha256_hex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

pub const RECOVERABILITY_MANIFEST_SCHEMA: &str =
    "symthaea.science_bench_refresh_recoverability_manifest.v1";
pub const RECOVERABILITY_REPORT_SCHEMA: &str =
    "symthaea.science_bench_refresh_recoverability_report.v1";

const HASH_PREFIX: &[u8] = b"symthaea.science_bench_refresh.recoverability.v1\0";
const MAX_NODES: usize = 10_000;
const MAX_EDGES: usize = 50_000;
const MAX_EVIDENCE_UNITS: usize = 50_000;
const MAX_REFERENCE_BYTES: usize = 256;

/// Completeness of the finite route inventory, not merely whether the corpus
/// snapshot has a stable identifier. Only CompleteFrozenInventory can support
/// a negative reachability claim; PartialObserved can support positive route
/// witnesses, while Unknown supports neither.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchSurfaceStateV1 {
    CompleteFrozenInventory,
    PartialObserved,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactAvailabilityV1 {
    PresentPinned,
    Unavailable,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteOperationV1 {
    InitialRetrieval,
    QueryReformulation,
    PostReformulationRetrieval,
    OpenDocument,
    FollowLink,
    OpenTableOfContents,
    Pagination,
    InDocumentFind,
}

impl RouteOperationV1 {
    fn is_navigation(self) -> bool {
        matches!(
            self,
            Self::OpenDocument
                | Self::FollowLink
                | Self::OpenTableOfContents
                | Self::Pagination
                | Self::InDocumentFind
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoverabilityClassV1 {
    DirectlyReachable,
    ReformulationRecoverable,
    NavigationRecoverable,
    UnreachableWithinBudget,
    NotEvaluable,
}

impl RecoverabilityClassV1 {
    pub fn rank(self) -> u8 {
        match self {
            Self::DirectlyReachable => 0,
            Self::ReformulationRecoverable => 1,
            Self::NavigationRecoverable => 2,
            Self::UnreachableWithinBudget => 3,
            Self::NotEvaluable => 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoverabilityNodeV1 {
    pub node_ref: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoverabilityEdgeV1 {
    pub edge_ref: String,
    pub from_node_ref: String,
    pub to_node_ref: String,
    pub operation: RouteOperationV1,
    pub cost: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequiredEvidenceUnitV1 {
    pub evidence_unit_id: String,
    pub target_node_ref: String,
    pub artifact_sha256: String,
    pub source_family_ref: String,
    pub artifact_availability: ArtifactAvailabilityV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoverabilityManifestFieldsV1 {
    pub schema: String,
    pub manifest_id: String,
    pub generation_id: String,
    pub corpus_snapshot_sha256: String,
    pub search_surface_sha256: String,
    pub route_policy_sha256: String,
    pub surface_state: SearchSurfaceStateV1,
    pub start_node_ref: String,
    pub max_route_cost: u32,
    pub budget_unit: String,
    pub nodes: Vec<RecoverabilityNodeV1>,
    pub edges: Vec<RecoverabilityEdgeV1>,
    pub evidence_units: Vec<RequiredEvidenceUnitV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoverabilityManifestV1 {
    pub schema: String,
    pub manifest_id: String,
    pub generation_id: String,
    pub corpus_snapshot_sha256: String,
    pub search_surface_sha256: String,
    pub route_policy_sha256: String,
    pub surface_state: SearchSurfaceStateV1,
    pub start_node_ref: String,
    pub max_route_cost: u32,
    pub budget_unit: String,
    pub nodes: Vec<RecoverabilityNodeV1>,
    pub edges: Vec<RecoverabilityEdgeV1>,
    pub evidence_units: Vec<RequiredEvidenceUnitV1>,
    pub manifest_commitment_sha256: String,
}

impl From<&RecoverabilityManifestV1> for RecoverabilityManifestFieldsV1 {
    fn from(value: &RecoverabilityManifestV1) -> Self {
        Self {
            schema: value.schema.clone(),
            manifest_id: value.manifest_id.clone(),
            generation_id: value.generation_id.clone(),
            corpus_snapshot_sha256: value.corpus_snapshot_sha256.clone(),
            search_surface_sha256: value.search_surface_sha256.clone(),
            route_policy_sha256: value.route_policy_sha256.clone(),
            surface_state: value.surface_state,
            start_node_ref: value.start_node_ref.clone(),
            max_route_cost: value.max_route_cost,
            budget_unit: value.budget_unit.clone(),
            nodes: value.nodes.clone(),
            edges: value.edges.clone(),
            evidence_units: value.evidence_units.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoverabilityRecordV1 {
    pub evidence_unit_id: String,
    pub target_node_ref: String,
    pub source_family_ref: String,
    pub recoverability: RecoverabilityClassV1,
    pub shortest_route_cost: Option<u32>,
    pub route_edge_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RecoverabilityAuditReportV1 {
    pub schema: &'static str,
    pub manifest_id: String,
    pub manifest_sha256: String,
    pub manifest_commitment_sha256: String,
    pub manifest_commitment_matches: bool,
    pub valid: bool,
    pub model_free: bool,
    pub benchmark_relative_only: bool,
    pub external_anchor_required: bool,
    pub evidence_unit_count: usize,
    pub distinct_declared_source_family_ref_count: usize,
    pub class_counts: BTreeMap<String, usize>,
    pub records: Vec<RecoverabilityRecordV1>,
    pub nonclaims: Vec<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SearchState {
    node_ref: String,
    used_initial_retrieval: bool,
    used_reformulation: bool,
    used_post_reformulation_retrieval: bool,
    used_navigation: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RouteWitness {
    cost: u32,
    edge_refs: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
enum RouteMode {
    Direct,
    Reformulation,
    Navigation,
}

fn is_nonzero_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        && value.bytes().any(|byte| byte != b'0')
}

fn require_ref(value: &str, path: &str, errors: &mut Vec<String>) {
    if value.trim().is_empty() || value.len() > MAX_REFERENCE_BYTES {
        errors.push(format!("{path}: reference must be non-empty and at most {MAX_REFERENCE_BYTES} bytes"));
    }
}

fn validate_fields(fields: &RecoverabilityManifestFieldsV1) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    if fields.schema != RECOVERABILITY_MANIFEST_SCHEMA {
        errors.push(format!(
            "schema: expected {RECOVERABILITY_MANIFEST_SCHEMA}, got {}",
            fields.schema
        ));
    }
    require_ref(&fields.manifest_id, "manifest_id", &mut errors);
    require_ref(&fields.generation_id, "generation_id", &mut errors);
    require_ref(&fields.start_node_ref, "start_node_ref", &mut errors);
    require_ref(&fields.budget_unit, "budget_unit", &mut errors);

    for (path, digest) in [
        ("corpus_snapshot_sha256", &fields.corpus_snapshot_sha256),
        ("search_surface_sha256", &fields.search_surface_sha256),
        ("route_policy_sha256", &fields.route_policy_sha256),
    ] {
        if !is_nonzero_sha256(digest) {
            errors.push(format!("{path}: expected non-zero lowercase SHA-256"));
        }
    }

    if fields.max_route_cost == 0 {
        errors.push("max_route_cost: must be greater than zero".to_string());
    }
    if fields.nodes.is_empty() || fields.nodes.len() > MAX_NODES {
        errors.push(format!("nodes: count must be between 1 and {MAX_NODES}"));
    }
    if fields.edges.len() > MAX_EDGES {
        errors.push(format!("edges: count must not exceed {MAX_EDGES}"));
    }
    if fields.evidence_units.is_empty() || fields.evidence_units.len() > MAX_EVIDENCE_UNITS {
        errors.push(format!(
            "evidence_units: count must be between 1 and {MAX_EVIDENCE_UNITS}"
        ));
    }

    let mut node_refs = BTreeSet::new();
    for (index, node) in fields.nodes.iter().enumerate() {
        require_ref(&node.node_ref, &format!("nodes[{index}].node_ref"), &mut errors);
        if !node_refs.insert(node.node_ref.as_str()) {
            errors.push(format!("nodes[{index}].node_ref: duplicate node reference"));
        }
    }
    if !node_refs.contains(fields.start_node_ref.as_str()) {
        errors.push("start_node_ref: start node does not exist".to_string());
    }

    let mut edge_refs = BTreeSet::new();
    for (index, edge) in fields.edges.iter().enumerate() {
        require_ref(&edge.edge_ref, &format!("edges[{index}].edge_ref"), &mut errors);
        if !edge_refs.insert(edge.edge_ref.as_str()) {
            errors.push(format!("edges[{index}].edge_ref: duplicate edge reference"));
        }
        if !node_refs.contains(edge.from_node_ref.as_str()) {
            errors.push(format!("edges[{index}].from_node_ref: unknown node"));
        }
        if !node_refs.contains(edge.to_node_ref.as_str()) {
            errors.push(format!("edges[{index}].to_node_ref: unknown node"));
        }
        if edge.cost == 0 {
            errors.push(format!("edges[{index}].cost: must be greater than zero"));
        }
    }

    let mut evidence_ids = BTreeSet::new();
    for (index, unit) in fields.evidence_units.iter().enumerate() {
        require_ref(
            &unit.evidence_unit_id,
            &format!("evidence_units[{index}].evidence_unit_id"),
            &mut errors,
        );
        require_ref(
            &unit.target_node_ref,
            &format!("evidence_units[{index}].target_node_ref"),
            &mut errors,
        );
        require_ref(
            &unit.source_family_ref,
            &format!("evidence_units[{index}].source_family_ref"),
            &mut errors,
        );
        if !evidence_ids.insert(unit.evidence_unit_id.as_str()) {
            errors.push(format!("evidence_units[{index}].evidence_unit_id: duplicate ID"));
        }
        if !node_refs.contains(unit.target_node_ref.as_str()) {
            errors.push(format!("evidence_units[{index}].target_node_ref: unknown node"));
        }
        if !is_nonzero_sha256(&unit.artifact_sha256) {
            errors.push(format!(
                "evidence_units[{index}].artifact_sha256: expected non-zero lowercase SHA-256"
            ));
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn framed_commitment(domain: &[u8], payload: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(HASH_PREFIX);
    hasher.update((domain.len() as u64).to_be_bytes());
    hasher.update(domain);
    hasher.update((payload.len() as u64).to_be_bytes());
    hasher.update(payload);
    let digest = hasher.finalize();
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

pub fn compute_recoverability_manifest_commitment(
    fields_bytes: &[u8],
) -> Result<String, Vec<String>> {
    let fields: RecoverabilityManifestFieldsV1 = serde_json::from_slice(fields_bytes)
        .map_err(|error| vec![format!("manifest fields JSON: {error}")])?;
    validate_fields(&fields)?;
    let payload = serde_json::to_vec(&fields)
        .map_err(|error| vec![format!("manifest fields serialization: {error}")])?;
    Ok(framed_commitment(b"manifest-fields", &payload))
}

fn reconstruct_route(
    start: &SearchState,
    target: &SearchState,
    predecessors: &BTreeMap<SearchState, (SearchState, String)>,
) -> Option<Vec<String>> {
    let mut edge_refs = Vec::new();
    let mut current = target.clone();
    while &current != start {
        let (previous, edge_ref) = predecessors.get(&current)?;
        edge_refs.push(edge_ref.clone());
        current = previous.clone();
    }
    edge_refs.reverse();
    Some(edge_refs)
}

fn find_route(
    fields: &RecoverabilityManifestFieldsV1,
    adjacency: &BTreeMap<String, Vec<&RecoverabilityEdgeV1>>,
    target_node_ref: &str,
    mode: RouteMode,
) -> Option<RouteWitness> {
    let start = SearchState {
        node_ref: fields.start_node_ref.clone(),
        used_initial_retrieval: false,
        used_reformulation: false,
        used_post_reformulation_retrieval: false,
        used_navigation: false,
    };
    // Dijkstra expands equal-cost states in stable structural order. The first
    // equal-cost predecessor wins; adjacency lists are sorted by edge_ref.
    // Keeping predecessor links avoids cloning every full path at each edge.
    let mut best_costs: BTreeMap<SearchState, u32> = BTreeMap::new();
    let mut predecessors: BTreeMap<SearchState, (SearchState, String)> = BTreeMap::new();
    let mut queue: BinaryHeap<Reverse<(u32, SearchState)>> = BinaryHeap::new();
    best_costs.insert(start.clone(), 0);
    queue.push(Reverse((0, start.clone())));

    while let Some(Reverse((cost, state))) = queue.pop() {
        if best_costs.get(&state) != Some(&cost) {
            continue;
        }

        let reached = state.node_ref == target_node_ref;
        let qualifies = match mode {
            RouteMode::Direct => state.used_initial_retrieval
                && !state.used_reformulation
                && !state.used_navigation,
            RouteMode::Reformulation => {
                state.used_initial_retrieval
                    && state.used_reformulation
                    && state.used_post_reformulation_retrieval
                    && !state.used_navigation
            }
            RouteMode::Navigation => state.used_navigation,
        };
        if reached && qualifies {
            return Some(RouteWitness {
                cost,
                edge_refs: reconstruct_route(&start, &state, &predecessors)?,
            });
        }

        let Some(edges) = adjacency.get(&state.node_ref) else {
            continue;
        };
        for edge in edges {
            let allowed = match mode {
                RouteMode::Direct => edge.operation == RouteOperationV1::InitialRetrieval,
                RouteMode::Reformulation => matches!(
                    edge.operation,
                    RouteOperationV1::InitialRetrieval
                        | RouteOperationV1::QueryReformulation
                        | RouteOperationV1::PostReformulationRetrieval
                ),
                RouteMode::Navigation => true,
            };
            if !allowed {
                continue;
            }
            let Some(next_cost) = cost.checked_add(edge.cost) else {
                continue;
            };
            if next_cost > fields.max_route_cost {
                continue;
            }

            let next_state = SearchState {
                node_ref: edge.to_node_ref.clone(),
                used_initial_retrieval: state.used_initial_retrieval
                    || edge.operation == RouteOperationV1::InitialRetrieval,
                used_reformulation: state.used_reformulation
                    || (state.used_initial_retrieval
                        && edge.operation == RouteOperationV1::QueryReformulation),
                used_post_reformulation_retrieval: state.used_post_reformulation_retrieval
                    || (state.used_reformulation
                        && edge.operation == RouteOperationV1::PostReformulationRetrieval),
                used_navigation: state.used_navigation || edge.operation.is_navigation(),
            };

            let improve = best_costs
                .get(&next_state)
                .is_none_or(|existing| next_cost < *existing);
            if improve {
                best_costs.insert(next_state.clone(), next_cost);
                predecessors.insert(next_state.clone(), (state.clone(), edge.edge_ref.clone()));
                queue.push(Reverse((next_cost, next_state)));
            }
        }
    }
    None
}

fn classify_unit(
    fields: &RecoverabilityManifestFieldsV1,
    adjacency: &BTreeMap<String, Vec<&RecoverabilityEdgeV1>>,
    unit: &RequiredEvidenceUnitV1,
) -> RecoverabilityRecordV1 {
    let artifact_unavailable =
        unit.artifact_availability != ArtifactAvailabilityV1::PresentPinned;
    let surface_unknown = fields.surface_state == SearchSurfaceStateV1::Unknown;

    // A concrete pinned route is positive evidence even when the route inventory
    // is partial. Absence of a route is informative only for a frozen, complete
    // inventory of the declared finite search surface.
    let result = if artifact_unavailable || surface_unknown {
        (RecoverabilityClassV1::NotEvaluable, None)
    } else if let Some(route) = find_route(
        fields,
        adjacency,
        &unit.target_node_ref,
        RouteMode::Direct,
    ) {
        (RecoverabilityClassV1::DirectlyReachable, Some(route))
    } else if let Some(route) = find_route(
        fields,
        adjacency,
        &unit.target_node_ref,
        RouteMode::Reformulation,
    ) {
        (
            RecoverabilityClassV1::ReformulationRecoverable,
            Some(route),
        )
    } else if let Some(route) = find_route(
        fields,
        adjacency,
        &unit.target_node_ref,
        RouteMode::Navigation,
    ) {
        (RecoverabilityClassV1::NavigationRecoverable, Some(route))
    } else if fields.surface_state == SearchSurfaceStateV1::CompleteFrozenInventory {
        (RecoverabilityClassV1::UnreachableWithinBudget, None)
    } else {
        (RecoverabilityClassV1::NotEvaluable, None)
    };

    RecoverabilityRecordV1 {
        evidence_unit_id: unit.evidence_unit_id.clone(),
        target_node_ref: unit.target_node_ref.clone(),
        source_family_ref: unit.source_family_ref.clone(),
        recoverability: result.0,
        shortest_route_cost: result.1.as_ref().map(|route| route.cost),
        route_edge_refs: result
            .1
            .map(|route| route.edge_refs)
            .unwrap_or_default(),
    }
}

pub fn verify_recoverability_manifest(
    manifest_bytes: &[u8],
) -> Result<RecoverabilityAuditReportV1, Vec<String>> {
    let manifest: RecoverabilityManifestV1 = serde_json::from_slice(manifest_bytes)
        .map_err(|error| vec![format!("recoverability manifest JSON: {error}")])?;
    let fields = RecoverabilityManifestFieldsV1::from(&manifest);
    validate_fields(&fields)?;

    let fields_bytes = serde_json::to_vec(&fields)
        .map_err(|error| vec![format!("manifest fields serialization: {error}")])?;
    let expected = framed_commitment(b"manifest-fields", &fields_bytes);
    let commitment_matches = expected == manifest.manifest_commitment_sha256;

    let mut adjacency: BTreeMap<String, Vec<&RecoverabilityEdgeV1>> = BTreeMap::new();
    for edge in &manifest.edges {
        adjacency
            .entry(edge.from_node_ref.clone())
            .or_default()
            .push(edge);
    }
    for edges in adjacency.values_mut() {
        edges.sort_by(|left, right| left.edge_ref.cmp(&right.edge_ref));
    }

    let records = if commitment_matches {
        manifest
            .evidence_units
            .iter()
            .map(|unit| classify_unit(&fields, &adjacency, unit))
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    let families: BTreeSet<_> = manifest
        .evidence_units
        .iter()
        .map(|unit| unit.source_family_ref.as_str())
        .collect();
    let mut class_counts = BTreeMap::new();
    for record in &records {
        *class_counts
            .entry(format!("{:?}", record.recoverability))
            .or_insert(0usize) += 1;
    }

    Ok(RecoverabilityAuditReportV1 {
        schema: RECOVERABILITY_REPORT_SCHEMA,
        manifest_id: manifest.manifest_id,
        manifest_sha256: sha256_hex(manifest_bytes),
        manifest_commitment_sha256: manifest.manifest_commitment_sha256,
        manifest_commitment_matches: commitment_matches,
        valid: commitment_matches && records.len() == fields.evidence_units.len(),
        model_free: true,
        benchmark_relative_only: true,
        external_anchor_required: true,
        evidence_unit_count: fields.evidence_units.len(),
        distinct_declared_source_family_ref_count: families.len(),
        class_counts,
        records,
        nonclaims: vec![
            "recoverability is relative only to the exact frozen corpus, search surface, route policy, and budget",
            "a route witness does not establish that a model actually followed the route",
            "unreachable within budget is not world-level or open-web unavailability",
            "pinned artifact identity does not establish source authenticity or factual truth",
            "the manifest commitment requires independent pre-run anchoring",
            "this evaluator does not prove the absence of unlogged tools or network channels",
            "source-family references are opaque declarations; their count is not independent provenance verification",
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(character: char) -> String {
        std::iter::repeat_n(character, 64).collect()
    }

    fn base_fields() -> RecoverabilityManifestFieldsV1 {
        RecoverabilityManifestFieldsV1 {
            schema: RECOVERABILITY_MANIFEST_SCHEMA.to_string(),
            manifest_id: "recoverability:synthetic-v1".to_string(),
            generation_id: "generation:synthetic-v1".to_string(),
            corpus_snapshot_sha256: digest('a'),
            search_surface_sha256: digest('b'),
            route_policy_sha256: digest('c'),
            surface_state: SearchSurfaceStateV1::CompleteFrozenInventory,
            start_node_ref: "state:start".to_string(),
            max_route_cost: 8,
            budget_unit: "synthetic_action_credit".to_string(),
            nodes: vec![
                RecoverabilityNodeV1 { node_ref: "state:start".into() },
                RecoverabilityNodeV1 { node_ref: "evidence:direct".into() },
                RecoverabilityNodeV1 { node_ref: "state:rewrite".into() },
                RecoverabilityNodeV1 { node_ref: "evidence:rewrite".into() },
                RecoverabilityNodeV1 { node_ref: "document:known".into() },
                RecoverabilityNodeV1 { node_ref: "page:second".into() },
                RecoverabilityNodeV1 { node_ref: "evidence:navigation".into() },
                RecoverabilityNodeV1 { node_ref: "evidence:unreachable".into() },
                RecoverabilityNodeV1 { node_ref: "evidence:unknown".into() },
            ],
            edges: vec![
                RecoverabilityEdgeV1 {
                    edge_ref: "route:direct".into(),
                    from_node_ref: "state:start".into(),
                    to_node_ref: "evidence:direct".into(),
                    operation: RouteOperationV1::InitialRetrieval,
                    cost: 1,
                },
                RecoverabilityEdgeV1 {
                    edge_ref: "route:rewrite-query".into(),
                    from_node_ref: "state:start".into(),
                    to_node_ref: "state:rewrite".into(),
                    operation: RouteOperationV1::QueryReformulation,
                    cost: 1,
                },
                RecoverabilityEdgeV1 {
                    edge_ref: "route:rewrite-search".into(),
                    from_node_ref: "state:rewrite".into(),
                    to_node_ref: "evidence:rewrite".into(),
                    operation: RouteOperationV1::PostReformulationRetrieval,
                    cost: 1,
                },
                RecoverabilityEdgeV1 {
                    edge_ref: "route:open-document".into(),
                    from_node_ref: "state:start".into(),
                    to_node_ref: "document:known".into(),
                    operation: RouteOperationV1::OpenDocument,
                    cost: 1,
                },
                RecoverabilityEdgeV1 {
                    edge_ref: "route:paginate".into(),
                    from_node_ref: "document:known".into(),
                    to_node_ref: "page:second".into(),
                    operation: RouteOperationV1::Pagination,
                    cost: 1,
                },
                RecoverabilityEdgeV1 {
                    edge_ref: "route:find".into(),
                    from_node_ref: "page:second".into(),
                    to_node_ref: "evidence:navigation".into(),
                    operation: RouteOperationV1::InDocumentFind,
                    cost: 1,
                },
            ],
            evidence_units: vec![
                RequiredEvidenceUnitV1 {
                    evidence_unit_id: "unit:direct".into(),
                    target_node_ref: "evidence:direct".into(),
                    artifact_sha256: digest('1'),
                    source_family_ref: "family:one".into(),
                    artifact_availability: ArtifactAvailabilityV1::PresentPinned,
                },
                RequiredEvidenceUnitV1 {
                    evidence_unit_id: "unit:rewrite".into(),
                    target_node_ref: "evidence:rewrite".into(),
                    artifact_sha256: digest('2'),
                    source_family_ref: "family:two".into(),
                    artifact_availability: ArtifactAvailabilityV1::PresentPinned,
                },
                RequiredEvidenceUnitV1 {
                    evidence_unit_id: "unit:navigation".into(),
                    target_node_ref: "evidence:navigation".into(),
                    artifact_sha256: digest('3'),
                    source_family_ref: "family:one".into(),
                    artifact_availability: ArtifactAvailabilityV1::PresentPinned,
                },
                RequiredEvidenceUnitV1 {
                    evidence_unit_id: "unit:unreachable".into(),
                    target_node_ref: "evidence:unreachable".into(),
                    artifact_sha256: digest('4'),
                    source_family_ref: "family:three".into(),
                    artifact_availability: ArtifactAvailabilityV1::PresentPinned,
                },
                RequiredEvidenceUnitV1 {
                    evidence_unit_id: "unit:unknown".into(),
                    target_node_ref: "evidence:unknown".into(),
                    artifact_sha256: digest('5'),
                    source_family_ref: "family:four".into(),
                    artifact_availability: ArtifactAvailabilityV1::Unknown,
                },
            ],
        }
    }

    fn manifest_bytes(fields: RecoverabilityManifestFieldsV1) -> Vec<u8> {
        let fields_bytes = serde_json::to_vec(&fields).unwrap();
        let commitment = compute_recoverability_manifest_commitment(&fields_bytes).unwrap();
        let manifest = RecoverabilityManifestV1 {
            schema: fields.schema,
            manifest_id: fields.manifest_id,
            generation_id: fields.generation_id,
            corpus_snapshot_sha256: fields.corpus_snapshot_sha256,
            search_surface_sha256: fields.search_surface_sha256,
            route_policy_sha256: fields.route_policy_sha256,
            surface_state: fields.surface_state,
            start_node_ref: fields.start_node_ref,
            max_route_cost: fields.max_route_cost,
            budget_unit: fields.budget_unit,
            nodes: fields.nodes,
            edges: fields.edges,
            evidence_units: fields.evidence_units,
            manifest_commitment_sha256: commitment,
        };
        serde_json::to_vec(&manifest).unwrap()
    }

    fn report(fields: RecoverabilityManifestFieldsV1) -> RecoverabilityAuditReportV1 {
        verify_recoverability_manifest(&manifest_bytes(fields)).unwrap()
    }

    #[test]
    fn classifies_direct_reformulation_navigation_unreachable_and_unknown() {
        let result = report(base_fields());
        assert!(result.valid);
        assert_eq!(result.evidence_unit_count, 5);
        assert_eq!(result.distinct_declared_source_family_ref_count, 4);
        let actual: BTreeMap<_, _> = result
            .records
            .iter()
            .map(|record| (record.evidence_unit_id.as_str(), record.recoverability))
            .collect();
        assert_eq!(actual["unit:direct"], RecoverabilityClassV1::DirectlyReachable);
        assert_eq!(
            actual["unit:rewrite"],
            RecoverabilityClassV1::ReformulationRecoverable
        );
        assert_eq!(
            actual["unit:navigation"],
            RecoverabilityClassV1::NavigationRecoverable
        );
        assert_eq!(
            actual["unit:unreachable"],
            RecoverabilityClassV1::UnreachableWithinBudget
        );
        assert_eq!(actual["unit:unknown"], RecoverabilityClassV1::NotEvaluable);
    }

    #[test]
    fn partial_surface_preserves_positive_routes_but_never_claims_unreachable() {
        let mut fields = base_fields();
        fields.surface_state = SearchSurfaceStateV1::PartialObserved;
        let result = report(fields);
        assert!(result.valid);
        let actual: BTreeMap<_, _> = result
            .records
            .iter()
            .map(|record| (record.evidence_unit_id.as_str(), record.recoverability))
            .collect();
        assert_eq!(actual["unit:direct"], RecoverabilityClassV1::DirectlyReachable);
        assert_eq!(
            actual["unit:rewrite"],
            RecoverabilityClassV1::ReformulationRecoverable
        );
        assert_eq!(
            actual["unit:navigation"],
            RecoverabilityClassV1::NavigationRecoverable
        );
        assert_eq!(
            actual["unit:unreachable"],
            RecoverabilityClassV1::NotEvaluable
        );
        assert_eq!(actual["unit:unknown"], RecoverabilityClassV1::NotEvaluable);
        assert!(result.records
            .iter()
            .find(|record| record.evidence_unit_id == "unit:unreachable")
            .unwrap()
            .route_edge_refs
            .is_empty());
    }

    #[test]
    fn unknown_search_surface_makes_even_observed_route_untrusted() {
        let mut fields = base_fields();
        fields.surface_state = SearchSurfaceStateV1::Unknown;
        let result = report(fields);
        assert!(result.valid);
        assert!(result.records.iter().all(|record| {
            record.recoverability == RecoverabilityClassV1::NotEvaluable
                && record.route_edge_refs.is_empty()
        }));
    }

    #[test]
    fn reducing_budget_cannot_improve_recoverability() {
        let mut fields = base_fields();
        fields.max_route_cost = 3;
        let before = report(fields.clone());
        let before_class = before.records.iter().find(|r| r.evidence_unit_id == "unit:navigation").unwrap().recoverability;
        fields.max_route_cost = 2;
        let after = report(fields);
        let after_class = after.records.iter().find(|r| r.evidence_unit_id == "unit:navigation").unwrap().recoverability;
        assert!(after_class.rank() >= before_class.rank());
    }

    #[test]
    fn adding_a_direct_route_improves_but_never_erases_unit_identity() {
        let mut fields = base_fields();
        let before = report(fields.clone());
        assert_eq!(
            before.records.iter().find(|r| r.evidence_unit_id == "unit:unreachable").unwrap().recoverability,
            RecoverabilityClassV1::UnreachableWithinBudget
        );
        fields.edges.push(RecoverabilityEdgeV1 {
            edge_ref: "route:new-direct".into(),
            from_node_ref: "state:start".into(),
            to_node_ref: "evidence:unreachable".into(),
            operation: RouteOperationV1::InitialRetrieval,
            cost: 1,
        });
        let after = report(fields);
        assert_eq!(
            after.records.iter().find(|r| r.evidence_unit_id == "unit:unreachable").unwrap().recoverability,
            RecoverabilityClassV1::DirectlyReachable
        );
    }

    #[test]
    fn reformulation_before_first_retrieval_does_not_qualify() {
        let mut fields = base_fields();
        fields.edges.retain(|edge| {
            edge.edge_ref != "route:rewrite-query" && edge.edge_ref != "route:rewrite-search"
        });
        fields.nodes.push(RecoverabilityNodeV1 {
            node_ref: "state:retrieved-too-late".into(),
        });
        fields.edges.push(RecoverabilityEdgeV1 {
            edge_ref: "route:rewrite-first".into(),
            from_node_ref: "state:start".into(),
            to_node_ref: "state:retrieved-too-late".into(),
            operation: RouteOperationV1::QueryReformulation,
            cost: 1,
        });
        fields.edges.push(RecoverabilityEdgeV1 {
            edge_ref: "route:initial-after-rewrite".into(),
            from_node_ref: "state:retrieved-too-late".into(),
            to_node_ref: "evidence:rewrite".into(),
            operation: RouteOperationV1::InitialRetrieval,
            cost: 1,
        });
        fields.edges.push(RecoverabilityEdgeV1 {
            edge_ref: "route:post-after-rewrite".into(),
            from_node_ref: "evidence:rewrite".into(),
            to_node_ref: "evidence:rewrite".into(),
            operation: RouteOperationV1::PostReformulationRetrieval,
            cost: 1,
        });
        let result = report(fields);
        let row = result
            .records
            .iter()
            .find(|record| record.evidence_unit_id == "unit:rewrite")
            .unwrap();
        assert_eq!(
            row.recoverability,
            RecoverabilityClassV1::UnreachableWithinBudget
        );
    }

    #[test]
    fn post_reformulation_retrieval_before_reformulation_does_not_qualify() {
        let mut fields = base_fields();
        fields.edges.retain(|edge| edge.edge_ref != "route:rewrite-query");
        fields.edges.retain(|edge| edge.edge_ref != "route:rewrite-search");
        fields.nodes.push(RecoverabilityNodeV1 {
            node_ref: "state:pre-retrieval".into(),
        });
        fields.edges.push(RecoverabilityEdgeV1 {
            edge_ref: "route:post-before-rewrite".into(),
            from_node_ref: "state:start".into(),
            to_node_ref: "state:pre-retrieval".into(),
            operation: RouteOperationV1::PostReformulationRetrieval,
            cost: 1,
        });
        fields.edges.push(RecoverabilityEdgeV1 {
            edge_ref: "route:rewrite-after-search".into(),
            from_node_ref: "state:pre-retrieval".into(),
            to_node_ref: "evidence:rewrite".into(),
            operation: RouteOperationV1::QueryReformulation,
            cost: 1,
        });
        let result = report(fields);
        let row = result
            .records
            .iter()
            .find(|record| record.evidence_unit_id == "unit:rewrite")
            .unwrap();
        assert_eq!(
            row.recoverability,
            RecoverabilityClassV1::UnreachableWithinBudget
        );
        assert!(row.route_edge_refs.is_empty());
    }

    #[test]
    fn removing_an_edge_cannot_improve_recoverability() {
        let mut fields = base_fields();
        let before = report(fields.clone());
        fields.edges.retain(|edge| edge.edge_ref != "route:open-document");
        let after = report(fields);
        let before_class = before.records.iter().find(|r| r.evidence_unit_id == "unit:navigation").unwrap().recoverability;
        let after_class = after.records.iter().find(|r| r.evidence_unit_id == "unit:navigation").unwrap().recoverability;
        assert!(after_class.rank() >= before_class.rank());
    }

    #[test]
    fn duplicate_mirrors_do_not_inflate_source_family_count() {
        let mut fields = base_fields();
        fields.evidence_units.push(RequiredEvidenceUnitV1 {
            evidence_unit_id: "unit:mirror".into(),
            target_node_ref: "evidence:direct".into(),
            artifact_sha256: digest('6'),
            source_family_ref: "family:one".into(),
            artifact_availability: ArtifactAvailabilityV1::PresentPinned,
        });
        let result = report(fields);
        assert_eq!(result.evidence_unit_count, 6);
        assert_eq!(result.distinct_declared_source_family_ref_count, 4);
    }

    #[test]
    fn model_dependent_fields_are_rejected() {
        let mut value: serde_json::Value =
            serde_json::from_slice(&manifest_bytes(base_fields())).unwrap();
        value["model_judgment"] = serde_json::json!("navigation_recoverable");
        assert!(serde_json::from_value::<RecoverabilityManifestV1>(value).is_err());
    }

    #[test]
    fn changed_manifest_bytes_fail_commitment_closed() {
        let bytes = manifest_bytes(base_fields());
        let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        value["max_route_cost"] = serde_json::json!(9);
        let report = verify_recoverability_manifest(&serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(!report.valid);
        assert!(!report.manifest_commitment_matches);
        assert!(report.records.is_empty());
    }

    #[test]
    fn malformed_references_and_zero_cost_edges_reject() {
        let mut fields = base_fields();
        fields.edges[0].cost = 0;
        assert!(validate_fields(&fields).is_err());
        fields.edges[0].cost = 1;
        fields.edges[0].from_node_ref = "node:missing".into();
        assert!(validate_fields(&fields).is_err());
    }

    #[test]
    fn shortest_route_is_deterministic_when_costs_tie() {
        let mut fields = base_fields();
        fields.edges.push(RecoverabilityEdgeV1 {
            edge_ref: "route:direct-a".into(),
            from_node_ref: "state:start".into(),
            to_node_ref: "evidence:unreachable".into(),
            operation: RouteOperationV1::InitialRetrieval,
            cost: 1,
        });
        fields.edges.push(RecoverabilityEdgeV1 {
            edge_ref: "route:direct-b".into(),
            from_node_ref: "state:start".into(),
            to_node_ref: "evidence:unreachable".into(),
            operation: RouteOperationV1::InitialRetrieval,
            cost: 1,
        });
        let first = report(fields.clone());
        let second = report(fields);
        let route1 = first.records.iter().find(|r| r.evidence_unit_id == "unit:unreachable").unwrap();
        let route2 = second.records.iter().find(|r| r.evidence_unit_id == "unit:unreachable").unwrap();
        assert_eq!(route1.route_edge_refs, route2.route_edge_refs);
    }
}
