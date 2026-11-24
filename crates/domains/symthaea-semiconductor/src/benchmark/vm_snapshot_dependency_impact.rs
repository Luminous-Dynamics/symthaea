// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit VM model-derivation graph and requalification impact closure.
//!
//! This module propagates K1 snapshot impact only across declared inherited
//! dependencies. It does not infer lineage from names, files, commits or model
//! similarity; it does not retrain, roll back, deploy, or control equipment.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use super::label_consumption_guard::LabelConsumptionEligibilityV1;
use super::vm_snapshot_evidence::{
    VmSnapshotCurrentDispositionV1, VmSnapshotPurposeV1, VmSnapshotRequalificationReportV1,
};

pub const VM_DEPENDENCY_GRAPH_SCHEMA_V1: &str = "luminous.semi-bench.vm-dependency-graph.v1";
pub const VM_DEPENDENCY_EDGE_SCHEMA_V1: &str = "luminous.semi-bench.vm-dependency-edge.v1";
pub const VM_IMPACT_CLOSURE_INPUT_SCHEMA_V1: &str =
    "luminous.semi-bench.vm-impact-closure-input.v1";
pub const VM_IMPACT_CLOSURE_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.vm-impact-closure-report.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VmModelDerivationKindV1 {
    FineTuneFromParent,
    WarmStartFromParent,
    EnsembleOfParents,
    DistilledFromParent,
    CalibrationHeadFromParent,
    IndependentRetrain,
    EvaluationOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VmDependencyDimensionV1 {
    ModelWeights,
    FeatureSchema,
    CalibrationState,
    EvaluationTarget,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VmImpactClosureStatusV1 {
    Complete,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VmSnapshotImpactStateV1 {
    Unaffected,
    DirectRebuildRequired,
    DirectIndeterminate,
    InheritedRebuildRequired,
    InheritedIndeterminate,
    HistoricalOnly,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VmModelSnapshotNodeV1 {
    pub snapshot_id: String,
    pub artifact_sha256: String,
    pub manifest_sha256: String,
    pub purpose: VmSnapshotPurposeV1,
    pub derivation_kind: VmModelDerivationKindV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VmSnapshotDependencyEdgeV1 {
    pub schema: String,
    pub edge_id: String,
    pub parent_snapshot_id: String,
    pub child_snapshot_id: String,
    pub parent_artifact_sha256: String,
    pub inherited_dimensions: Vec<VmDependencyDimensionV1>,
    pub derivation_evidence_sha256: String,
    pub edge_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VmSnapshotDependencyGraphV1 {
    pub schema: String,
    pub graph_id: String,
    pub inventory_complete: bool,
    pub ordering_complete: bool,
    pub nodes: Vec<VmModelSnapshotNodeV1>,
    pub edges: Vec<VmSnapshotDependencyEdgeV1>,
    pub graph_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VmImpactClosureInputV1 {
    pub schema: String,
    pub input_id: String,
    pub graph: VmSnapshotDependencyGraphV1,
    pub snapshot_requalification_reports: Vec<VmSnapshotRequalificationReportV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VmSnapshotImpactEntryV1 {
    pub snapshot_id: String,
    pub state: VmSnapshotImpactStateV1,
    pub directly_impacted: bool,
    pub direct_label_status: Option<VmSnapshotCurrentDispositionV1>,
    pub inherited_from_snapshot_ids: Vec<String>,
    pub direct_affected_label_ids: Vec<String>,
    pub inherited_affected_label_ids: Vec<String>,
    pub rebuild_candidate: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VmImpactClosureReportV1 {
    pub schema: String,
    pub input_id: String,
    pub graph_id: String,
    pub graph_sha256: String,
    pub status: VmImpactClosureStatusV1,
    pub reason_codes: Vec<String>,
    pub direct_impact_snapshot_ids: Vec<String>,
    pub transitive_impact_snapshot_ids: Vec<String>,
    pub rebuild_order_snapshot_ids: Vec<String>,
    pub unresolved_snapshot_ids: Vec<String>,
    pub entries: Vec<VmSnapshotImpactEntryV1>,
    pub source_authenticity_proven: bool,
    pub retraining_executed: bool,
    pub deployment_rollback_executed: bool,
    pub equipment_control_authority: bool,
    pub report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct VmImpactClosureErrorV1(pub String);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum ImpactRiskV1 {
    None,
    Historical,
    Unknown,
    Blocked,
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_text(value: &str, field: &str) -> Result<(), VmImpactClosureErrorV1> {
    if value.trim().is_empty() {
        return Err(VmImpactClosureErrorV1(format!(
            "{field}: expected non-empty text"
        )));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, VmImpactClosureErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        VmImpactClosureErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(sha256_hex(&bytes))
}

fn edge_digest(edge: &VmSnapshotDependencyEdgeV1) -> Result<String, VmImpactClosureErrorV1> {
    let mut canonical = edge.clone();
    canonical.edge_sha256.clear();
    canonical.inherited_dimensions.sort();
    json_sha256(&canonical)
}

fn graph_digest(graph: &VmSnapshotDependencyGraphV1) -> Result<String, VmImpactClosureErrorV1> {
    let mut canonical = graph.clone();
    canonical.graph_sha256.clear();
    canonical
        .nodes
        .sort_by(|a, b| a.snapshot_id.cmp(&b.snapshot_id));
    canonical.edges.sort_by(|a, b| a.edge_id.cmp(&b.edge_id));
    for edge in &mut canonical.edges {
        edge.inherited_dimensions.sort();
    }
    json_sha256(&canonical)
}

fn requalification_digest(
    report: &VmSnapshotRequalificationReportV1,
) -> Result<String, VmImpactClosureErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

/// Seals a dependency edge after all parent/child identities and inherited
/// dependency dimensions have been declared. A digest is content identity,
/// not proof that the build system actually consumed the declared parent.
pub fn seal_vm_snapshot_dependency_edge_v1(
    edge: &VmSnapshotDependencyEdgeV1,
) -> Result<String, VmImpactClosureErrorV1> {
    edge_digest(edge)
}

/// Seals a graph in canonical node-ID and edge-ID order.
pub fn seal_vm_snapshot_dependency_graph_v1(
    graph: &VmSnapshotDependencyGraphV1,
) -> Result<String, VmImpactClosureErrorV1> {
    graph_digest(graph)
}

fn derivation_requires_parent(kind: VmModelDerivationKindV1) -> bool {
    !matches!(kind, VmModelDerivationKindV1::IndependentRetrain)
}

fn validate_node(node: &VmModelSnapshotNodeV1) -> Result<(), VmImpactClosureErrorV1> {
    require_text(&node.snapshot_id, "node.snapshot_id")?;
    if !valid_sha256(&node.artifact_sha256) || !valid_sha256(&node.manifest_sha256) {
        return Err(VmImpactClosureErrorV1(format!(
            "node {} has malformed artifact or manifest digest",
            node.snapshot_id
        )));
    }
    if (node.derivation_kind == VmModelDerivationKindV1::EvaluationOnly)
        != (node.purpose == VmSnapshotPurposeV1::Evaluation)
    {
        return Err(VmImpactClosureErrorV1(format!(
            "node {} has incompatible evaluation purpose and derivation kind",
            node.snapshot_id
        )));
    }
    Ok(())
}

fn validate_edge(edge: &VmSnapshotDependencyEdgeV1) -> Result<(), VmImpactClosureErrorV1> {
    if edge.schema != VM_DEPENDENCY_EDGE_SCHEMA_V1 {
        return Err(VmImpactClosureErrorV1(
            "dependency edge schema is unsupported".into(),
        ));
    }
    require_text(&edge.edge_id, "edge.edge_id")?;
    require_text(&edge.parent_snapshot_id, "edge.parent_snapshot_id")?;
    require_text(&edge.child_snapshot_id, "edge.child_snapshot_id")?;
    if edge.parent_snapshot_id == edge.child_snapshot_id {
        return Err(VmImpactClosureErrorV1(
            "self-dependency edge is not permitted".into(),
        ));
    }
    if !valid_sha256(&edge.parent_artifact_sha256)
        || !valid_sha256(&edge.derivation_evidence_sha256)
        || !valid_sha256(&edge.edge_sha256)
    {
        return Err(VmImpactClosureErrorV1(format!(
            "edge {} contains a malformed digest",
            edge.edge_id
        )));
    }
    if edge.inherited_dimensions.is_empty() {
        return Err(VmImpactClosureErrorV1(format!(
            "edge {} must declare at least one dependency dimension",
            edge.edge_id
        )));
    }
    let unique: BTreeSet<VmDependencyDimensionV1> =
        edge.inherited_dimensions.iter().copied().collect();
    if unique.len() != edge.inherited_dimensions.len() {
        return Err(VmImpactClosureErrorV1(format!(
            "edge {} repeats an inherited dependency dimension",
            edge.edge_id
        )));
    }
    if edge_digest(edge)? != edge.edge_sha256 {
        return Err(VmImpactClosureErrorV1(format!(
            "edge {} content does not match its digest",
            edge.edge_id
        )));
    }
    Ok(())
}

/// Returns a deterministic Kahn topological order, breaking ties by snapshot ID.
fn topological_order(
    nodes: &BTreeMap<String, VmModelSnapshotNodeV1>,
    edges: &[VmSnapshotDependencyEdgeV1],
) -> Result<Vec<String>, VmImpactClosureErrorV1> {
    let mut indegree: BTreeMap<String, usize> = nodes.keys().map(|id| (id.clone(), 0)).collect();
    let mut outgoing: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for edge in edges {
        *indegree
            .get_mut(&edge.child_snapshot_id)
            .expect("edge endpoints were validated") += 1;
        outgoing
            .entry(edge.parent_snapshot_id.clone())
            .or_default()
            .push(edge.child_snapshot_id.clone());
    }
    for children in outgoing.values_mut() {
        children.sort();
    }

    let mut ready: BTreeSet<String> = indegree
        .iter()
        .filter(|(_, degree)| **degree == 0)
        .map(|(id, _)| id.clone())
        .collect();
    let mut ordered = Vec::with_capacity(nodes.len());

    while let Some(next) = ready.pop_first() {
        ordered.push(next.clone());
        for child in outgoing.get(&next).into_iter().flatten() {
            let degree = indegree.get_mut(child).expect("child node exists");
            *degree -= 1;
            if *degree == 0 {
                ready.insert(child.clone());
            }
        }
    }

    if ordered.len() != nodes.len() {
        return Err(VmImpactClosureErrorV1(
            "dependency graph contains a directed cycle".into(),
        ));
    }
    Ok(ordered)
}

fn validate_graph(
    graph: &VmSnapshotDependencyGraphV1,
) -> Result<(BTreeMap<String, VmModelSnapshotNodeV1>, Vec<String>), VmImpactClosureErrorV1> {
    if graph.schema != VM_DEPENDENCY_GRAPH_SCHEMA_V1 {
        return Err(VmImpactClosureErrorV1(
            "dependency graph schema is unsupported".into(),
        ));
    }
    require_text(&graph.graph_id, "graph.graph_id")?;
    if !valid_sha256(&graph.graph_sha256) || graph_digest(graph)? != graph.graph_sha256 {
        return Err(VmImpactClosureErrorV1(
            "dependency graph digest is missing or does not match its content".into(),
        ));
    }

    let mut nodes = BTreeMap::new();
    for node in &graph.nodes {
        validate_node(node)?;
        if nodes
            .insert(node.snapshot_id.clone(), node.clone())
            .is_some()
        {
            return Err(VmImpactClosureErrorV1(format!(
                "duplicate snapshot node {}",
                node.snapshot_id
            )));
        }
    }
    let mut edge_ids = BTreeSet::new();
    let mut endpoint_pairs = BTreeSet::new();
    let mut incoming: BTreeMap<String, usize> = nodes.keys().map(|id| (id.clone(), 0)).collect();

    for edge in &graph.edges {
        validate_edge(edge)?;
        if !edge_ids.insert(edge.edge_id.as_str()) {
            return Err(VmImpactClosureErrorV1(
                "duplicate dependency edge ID".into(),
            ));
        }
        if !endpoint_pairs.insert((
            edge.parent_snapshot_id.as_str(),
            edge.child_snapshot_id.as_str(),
        )) {
            return Err(VmImpactClosureErrorV1(
                "duplicate parent-child edge pair".into(),
            ));
        }
        let parent = nodes.get(&edge.parent_snapshot_id);
        let child = nodes.get(&edge.child_snapshot_id);
        let (Some(parent), Some(child)) = (parent, child) else {
            return Err(VmImpactClosureErrorV1(format!(
                "edge {} has a dangling parent or child endpoint",
                edge.edge_id
            )));
        };
        if edge.parent_artifact_sha256 != parent.artifact_sha256 {
            return Err(VmImpactClosureErrorV1(format!(
                "edge {} parent artifact digest does not match the declared node",
                edge.edge_id
            )));
        }
        let inherits_model_state = edge.inherited_dimensions.iter().any(|dimension| {
            matches!(
                dimension,
                VmDependencyDimensionV1::ModelWeights
                    | VmDependencyDimensionV1::FeatureSchema
                    | VmDependencyDimensionV1::CalibrationState
            )
        });
        let requires_dimension = match child.derivation_kind {
            VmModelDerivationKindV1::FineTuneFromParent
            | VmModelDerivationKindV1::WarmStartFromParent
            | VmModelDerivationKindV1::EnsembleOfParents
            | VmModelDerivationKindV1::DistilledFromParent => {
                Some(VmDependencyDimensionV1::ModelWeights)
            }
            VmModelDerivationKindV1::CalibrationHeadFromParent => {
                Some(VmDependencyDimensionV1::FeatureSchema)
            }
            VmModelDerivationKindV1::IndependentRetrain => None,
            VmModelDerivationKindV1::EvaluationOnly => {
                Some(VmDependencyDimensionV1::EvaluationTarget)
            }
        };
        if let Some(required) = requires_dimension
            && !edge.inherited_dimensions.contains(&required)
        {
            return Err(VmImpactClosureErrorV1(format!(
                "edge {} does not declare the required {:?} dependency for child {}",
                edge.edge_id, required, child.snapshot_id
            )));
        }
        if child.derivation_kind == VmModelDerivationKindV1::IndependentRetrain
            && edge
                .inherited_dimensions
                .contains(&VmDependencyDimensionV1::ModelWeights)
        {
            return Err(VmImpactClosureErrorV1(format!(
                "independent retrain node {} cannot inherit parent model weights",
                child.snapshot_id
            )));
        }
        if child.derivation_kind == VmModelDerivationKindV1::EvaluationOnly && inherits_model_state
        {
            return Err(VmImpactClosureErrorV1(format!(
                "evaluation-only node {} cannot declare model-state inheritance",
                child.snapshot_id
            )));
        }
        *incoming
            .get_mut(&child.snapshot_id)
            .expect("child node exists") += 1;
    }

    for node in nodes.values() {
        let degree = incoming[&node.snapshot_id];
        if derivation_requires_parent(node.derivation_kind) && degree == 0 {
            return Err(VmImpactClosureErrorV1(format!(
                "derived node {} has no declared parent",
                node.snapshot_id
            )));
        }
    }
    let ordered = topological_order(&nodes, &graph.edges)?;
    Ok((nodes, ordered))
}

fn validate_requalification_report(
    report: &VmSnapshotRequalificationReportV1,
    node: &VmModelSnapshotNodeV1,
) -> Result<(), VmImpactClosureErrorV1> {
    if report.consumed_label_count != report.label_statuses.len() {
        return Err(VmImpactClosureErrorV1(format!(
            "snapshot {} consumed-label count does not match its row-level K1 statuses",
            report.snapshot_id
        )));
    }

    let mut label_ids = BTreeSet::new();
    let mut eligible = 0_usize;
    let mut unknown = 0_usize;
    let mut blocked = 0_usize;
    for label in &report.label_statuses {
        require_text(&label.label_id, "requalification.label_id")?;
        if !label_ids.insert(label.label_id.as_str()) {
            return Err(VmImpactClosureErrorV1(format!(
                "snapshot {} repeats K1 label status {}",
                report.snapshot_id, label.label_id
            )));
        }
        match label.k1_eligibility {
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy => eligible += 1,
            LabelConsumptionEligibilityV1::Unknown => unknown += 1,
            LabelConsumptionEligibilityV1::Blocked => blocked += 1,
        }
        if !valid_sha256(&label.k0_projection_sha256) || !valid_sha256(&label.k1_report_sha256) {
            return Err(VmImpactClosureErrorV1(format!(
                "snapshot {} has malformed per-label K0/K1 receipt digest",
                report.snapshot_id
            )));
        }
    }

    if eligible != report.current_eligible_count
        || unknown != report.current_unknown_count
        || blocked != report.current_blocked_count
    {
        return Err(VmImpactClosureErrorV1(format!(
            "snapshot {} aggregate K1 counts disagree with row-level statuses",
            report.snapshot_id
        )));
    }

    let disposition_consistent = match report.current_disposition {
        VmSnapshotCurrentDispositionV1::CurrentEligible => {
            eligible == report.consumed_label_count && unknown == 0 && blocked == 0
        }
        VmSnapshotCurrentDispositionV1::Indeterminate => unknown > 0 && blocked == 0,
        VmSnapshotCurrentDispositionV1::RebuildRequired => {
            node.purpose == VmSnapshotPurposeV1::Training && blocked > 0
        }
        VmSnapshotCurrentDispositionV1::HistoricalOnly => {
            node.purpose == VmSnapshotPurposeV1::Evaluation && blocked > 0
        }
    };
    if !disposition_consistent {
        return Err(VmImpactClosureErrorV1(format!(
            "snapshot {} K1 disposition conflicts with its purpose or label statuses",
            report.snapshot_id
        )));
    }

    Ok(())
}

fn direct_risk(disposition: VmSnapshotCurrentDispositionV1) -> ImpactRiskV1 {
    match disposition {
        VmSnapshotCurrentDispositionV1::CurrentEligible => ImpactRiskV1::None,
        VmSnapshotCurrentDispositionV1::RebuildRequired => ImpactRiskV1::Blocked,
        VmSnapshotCurrentDispositionV1::Indeterminate => ImpactRiskV1::Unknown,
        VmSnapshotCurrentDispositionV1::HistoricalOnly => ImpactRiskV1::Historical,
    }
}

fn inherited_risk(
    parent_risk: ImpactRiskV1,
    dimensions: &[VmDependencyDimensionV1],
) -> ImpactRiskV1 {
    let mut propagated = ImpactRiskV1::None;
    if dimensions.iter().any(|dimension| {
        matches!(
            dimension,
            VmDependencyDimensionV1::ModelWeights | VmDependencyDimensionV1::CalibrationState
        )
    }) {
        propagated = propagated.max(parent_risk);
    }
    if dimensions.contains(&VmDependencyDimensionV1::EvaluationTarget) {
        let evaluation_risk = match parent_risk {
            ImpactRiskV1::Blocked | ImpactRiskV1::Historical => ImpactRiskV1::Historical,
            other => other,
        };
        propagated = propagated.max(evaluation_risk);
    }
    propagated
}

fn impact_state(
    direct_risk: ImpactRiskV1,
    inherited_risk: ImpactRiskV1,
) -> VmSnapshotImpactStateV1 {
    let overall = direct_risk.max(inherited_risk);
    match overall {
        ImpactRiskV1::None => VmSnapshotImpactStateV1::Unaffected,
        ImpactRiskV1::Blocked if direct_risk == ImpactRiskV1::Blocked => {
            VmSnapshotImpactStateV1::DirectRebuildRequired
        }
        ImpactRiskV1::Blocked => VmSnapshotImpactStateV1::InheritedRebuildRequired,
        ImpactRiskV1::Unknown if direct_risk == ImpactRiskV1::Unknown => {
            VmSnapshotImpactStateV1::DirectIndeterminate
        }
        ImpactRiskV1::Unknown => VmSnapshotImpactStateV1::InheritedIndeterminate,
        ImpactRiskV1::Historical => VmSnapshotImpactStateV1::HistoricalOnly,
    }
}

fn report_digest(report: &VmImpactClosureReportV1) -> Result<String, VmImpactClosureErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn unknown_report(
    input: &VmImpactClosureInputV1,
    reason: &str,
) -> Result<VmImpactClosureReportV1, VmImpactClosureErrorV1> {
    let mut unresolved: Vec<String> = input
        .graph
        .nodes
        .iter()
        .map(|node| node.snapshot_id.clone())
        .collect();
    unresolved.sort();
    let mut report = VmImpactClosureReportV1 {
        schema: VM_IMPACT_CLOSURE_REPORT_SCHEMA_V1.into(),
        input_id: input.input_id.clone(),
        graph_id: input.graph.graph_id.clone(),
        graph_sha256: input.graph.graph_sha256.clone(),
        status: VmImpactClosureStatusV1::Unknown,
        reason_codes: vec![reason.into()],
        direct_impact_snapshot_ids: Vec::new(),
        transitive_impact_snapshot_ids: Vec::new(),
        rebuild_order_snapshot_ids: Vec::new(),
        unresolved_snapshot_ids: unresolved,
        entries: Vec::new(),
        source_authenticity_proven: false,
        retraining_executed: false,
        deployment_rollback_executed: false,
        equipment_control_authority: false,
        report_sha256: String::new(),
    };
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}

/// Computes direct and transitive snapshot impact using only declared edges.
///
/// Incomplete graph inventories or non-total dependency ordering return UNKNOWN
/// and list the entire visible node set as unresolved. Complete graphs are
/// validated for cycles, dangling endpoints, digest mismatches and invalid
/// derivation semantics before any "unaffected" result can be emitted.
pub fn compute_vm_snapshot_impact_closure_v1(
    input: &VmImpactClosureInputV1,
) -> Result<VmImpactClosureReportV1, VmImpactClosureErrorV1> {
    if input.schema != VM_IMPACT_CLOSURE_INPUT_SCHEMA_V1 {
        return Err(VmImpactClosureErrorV1(
            "impact input schema is unsupported".into(),
        ));
    }
    require_text(&input.input_id, "input.input_id")?;
    if !input.graph.inventory_complete {
        return unknown_report(input, "dependency_inventory_incomplete");
    }
    if !input.graph.ordering_complete {
        return unknown_report(input, "dependency_ordering_incomplete");
    }

    let (nodes, topological) = validate_graph(&input.graph)?;
    let mut requalification_by_snapshot = BTreeMap::new();
    for report in &input.snapshot_requalification_reports {
        require_text(&report.snapshot_id, "requalification.snapshot_id")?;
        if requalification_digest(report)? != report.report_sha256 {
            return Err(VmImpactClosureErrorV1(format!(
                "snapshot {} requalification receipt digest mismatch",
                report.snapshot_id
            )));
        }
        if report.historical_manifest_rewritten
            || report.training_snapshot_mutated
            || report.model_retrained
            || report.equipment_control_authority
        {
            return Err(VmImpactClosureErrorV1(format!(
                "snapshot {} report exceeds the read-only impact contract",
                report.snapshot_id
            )));
        }
        if requalification_by_snapshot
            .insert(report.snapshot_id.as_str(), report)
            .is_some()
        {
            return Err(VmImpactClosureErrorV1(
                "duplicate snapshot requalification report".into(),
            ));
        }
    }

    let missing_reports: Vec<String> = nodes
        .keys()
        .filter(|snapshot_id| !requalification_by_snapshot.contains_key(snapshot_id.as_str()))
        .cloned()
        .collect();
    if !missing_reports.is_empty() {
        let mut report = unknown_report(input, "snapshot_requalification_coverage_incomplete")?;
        report.unresolved_snapshot_ids = missing_reports;
        report.report_sha256 = report_digest(&report)?;
        return Ok(report);
    }

    for (snapshot_id, report) in &requalification_by_snapshot {
        let node = nodes.get(*snapshot_id).expect("report node is present");
        if report.historical_manifest_sha256 != node.manifest_sha256 {
            return Err(VmImpactClosureErrorV1(format!(
                "snapshot {} requalification report references another manifest",
                snapshot_id
            )));
        }
        validate_requalification_report(report, node)?;
    }

    let mut direct_risks: BTreeMap<String, ImpactRiskV1> = BTreeMap::new();
    let mut inherited_risks: BTreeMap<String, ImpactRiskV1> = BTreeMap::new();
    let mut direct_statuses: BTreeMap<String, VmSnapshotCurrentDispositionV1> = BTreeMap::new();
    let mut direct_label_ids: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut inherited_label_ids: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut inherited_from: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

    for snapshot_id in &topological {
        let report = requalification_by_snapshot
            .get(snapshot_id.as_str())
            .expect("coverage is complete");
        direct_risks.insert(snapshot_id.clone(), direct_risk(report.current_disposition));
        inherited_risks.insert(snapshot_id.clone(), ImpactRiskV1::None);
        direct_statuses.insert(snapshot_id.clone(), report.current_disposition);
        let affected_labels: BTreeSet<String> = report
            .label_statuses
            .iter()
            .filter(|label| {
                label.k1_eligibility != LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy
            })
            .map(|label| label.label_id.clone())
            .collect();
        direct_label_ids.insert(snapshot_id.clone(), affected_labels);
        inherited_label_ids.insert(snapshot_id.clone(), BTreeSet::new());
    }

    for parent_id in &topological {
        let parent_risk = direct_risks[parent_id].max(inherited_risks[parent_id]);
        if parent_risk == ImpactRiskV1::None {
            continue;
        }
        for edge in input
            .graph
            .edges
            .iter()
            .filter(|edge| edge.parent_snapshot_id == *parent_id)
        {
            let propagated = inherited_risk(parent_risk, &edge.inherited_dimensions);
            if propagated == ImpactRiskV1::None {
                continue;
            }
            let current = inherited_risks[&edge.child_snapshot_id];
            if propagated > current {
                inherited_risks.insert(edge.child_snapshot_id.clone(), propagated);
            }
            inherited_from
                .entry(edge.child_snapshot_id.clone())
                .or_default()
                .insert(parent_id.clone());
            let mut propagated_label_ids = direct_label_ids[parent_id].clone();
            propagated_label_ids.extend(inherited_label_ids[parent_id].iter().cloned());
            inherited_label_ids
                .entry(edge.child_snapshot_id.clone())
                .or_default()
                .extend(propagated_label_ids);
        }
    }

    let mut entries = Vec::with_capacity(nodes.len());
    let mut direct_impact_snapshot_ids = Vec::new();
    let mut transitive_impact_snapshot_ids = Vec::new();
    let mut rebuild_candidates = BTreeSet::new();

    for snapshot_id in &topological {
        let direct_risk = direct_risks[snapshot_id];
        let inherited_risk = inherited_risks[snapshot_id];
        let disposition = direct_statuses[snapshot_id];
        let state = impact_state(direct_risk, inherited_risk);
        let inherited: Vec<String> = inherited_from
            .get(snapshot_id)
            .map(|values| values.iter().cloned().collect())
            .unwrap_or_default();
        if direct_risk != ImpactRiskV1::None {
            direct_impact_snapshot_ids.push(snapshot_id.clone());
        }
        if inherited_risk != ImpactRiskV1::None && direct_risk == ImpactRiskV1::None {
            transitive_impact_snapshot_ids.push(snapshot_id.clone());
        }
        let node = &nodes[snapshot_id];
        let rebuild_candidate = node.purpose == VmSnapshotPurposeV1::Training
            && matches!(
                state,
                VmSnapshotImpactStateV1::DirectRebuildRequired
                    | VmSnapshotImpactStateV1::InheritedRebuildRequired
            );
        if rebuild_candidate {
            rebuild_candidates.insert(snapshot_id.clone());
        }
        let direct_affected_label_ids: Vec<String> =
            direct_label_ids[snapshot_id].iter().cloned().collect();
        let inherited_affected_label_ids: Vec<String> =
            inherited_label_ids[snapshot_id].iter().cloned().collect();
        entries.push(VmSnapshotImpactEntryV1 {
            snapshot_id: snapshot_id.clone(),
            state,
            directly_impacted: direct_risk != ImpactRiskV1::None,
            direct_label_status: if direct_risk != ImpactRiskV1::None {
                Some(disposition)
            } else {
                None
            },
            inherited_from_snapshot_ids: inherited,
            direct_affected_label_ids,
            inherited_affected_label_ids,
            rebuild_candidate,
        });
    }

    // The topological order is already deterministic; filtering it preserves
    // valid parent-before-child rebuild order.
    let rebuild_order_snapshot_ids = topological
        .into_iter()
        .filter(|snapshot_id| rebuild_candidates.contains(snapshot_id))
        .collect();
    let mut report = VmImpactClosureReportV1 {
        schema: VM_IMPACT_CLOSURE_REPORT_SCHEMA_V1.into(),
        input_id: input.input_id.clone(),
        graph_id: input.graph.graph_id.clone(),
        graph_sha256: input.graph.graph_sha256.clone(),
        status: VmImpactClosureStatusV1::Complete,
        reason_codes: Vec::new(),
        direct_impact_snapshot_ids,
        transitive_impact_snapshot_ids,
        rebuild_order_snapshot_ids,
        unresolved_snapshot_ids: Vec::new(),
        entries,
        source_authenticity_proven: false,
        retraining_executed: false,
        deployment_rollback_executed: false,
        equipment_control_authority: false,
        report_sha256: String::new(),
    };
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::vm_snapshot_evidence::{
        VmSnapshotLabelCurrentStatusV1, VmSnapshotRequalificationReportV1,
    };

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn node(
        id: &str,
        kind: VmModelDerivationKindV1,
        purpose: VmSnapshotPurposeV1,
        artifact: char,
        manifest: char,
    ) -> VmModelSnapshotNodeV1 {
        VmModelSnapshotNodeV1 {
            snapshot_id: id.into(),
            artifact_sha256: digest(artifact),
            manifest_sha256: digest(manifest),
            purpose,
            derivation_kind: kind,
        }
    }

    fn edge(
        id: &str,
        parent: &VmModelSnapshotNodeV1,
        child: &VmModelSnapshotNodeV1,
        dimensions: Vec<VmDependencyDimensionV1>,
    ) -> VmSnapshotDependencyEdgeV1 {
        let mut value = VmSnapshotDependencyEdgeV1 {
            schema: VM_DEPENDENCY_EDGE_SCHEMA_V1.into(),
            edge_id: id.into(),
            parent_snapshot_id: parent.snapshot_id.clone(),
            child_snapshot_id: child.snapshot_id.clone(),
            parent_artifact_sha256: parent.artifact_sha256.clone(),
            inherited_dimensions: dimensions,
            derivation_evidence_sha256: digest('e'),
            edge_sha256: String::new(),
        };
        value.edge_sha256 = seal_vm_snapshot_dependency_edge_v1(&value).unwrap();
        value
    }

    fn graph(
        nodes: Vec<VmModelSnapshotNodeV1>,
        edges: Vec<VmSnapshotDependencyEdgeV1>,
        inventory_complete: bool,
        ordering_complete: bool,
    ) -> VmSnapshotDependencyGraphV1 {
        let mut value = VmSnapshotDependencyGraphV1 {
            schema: VM_DEPENDENCY_GRAPH_SCHEMA_V1.into(),
            graph_id: "vm-model-graph-001".into(),
            inventory_complete,
            ordering_complete,
            nodes,
            edges,
            graph_sha256: String::new(),
        };
        value.graph_sha256 = seal_vm_snapshot_dependency_graph_v1(&value).unwrap();
        value
    }

    fn requal(
        snapshot_id: &str,
        manifest: &str,
        disposition: VmSnapshotCurrentDispositionV1,
    ) -> VmSnapshotRequalificationReportV1 {
        let mut report = VmSnapshotRequalificationReportV1 {
            schema: "luminous.semi-bench.vm-snapshot-requalification.v1".into(),
            snapshot_id: snapshot_id.into(),
            historical_manifest_sha256: manifest.into(),
            requalified_at_k1_unix_seconds: 1_880_000_000,
            current_disposition: disposition,
            consumed_label_count: 1,
            current_eligible_count: if disposition == VmSnapshotCurrentDispositionV1::CurrentEligible { 1 } else { 0 },
            current_unknown_count: if disposition == VmSnapshotCurrentDispositionV1::Indeterminate { 1 } else { 0 },
            current_blocked_count: if matches!(
                disposition,
                VmSnapshotCurrentDispositionV1::RebuildRequired
                    | VmSnapshotCurrentDispositionV1::HistoricalOnly
            ) { 1 } else { 0 },
            label_statuses: vec![VmSnapshotLabelCurrentStatusV1 {
                label_id: format!("label-{snapshot_id}"),
                k0_projection_sha256: digest('k'),
                k1_eligibility: match disposition {
                    VmSnapshotCurrentDispositionV1::CurrentEligible => {
                        super::super::label_consumption_guard::LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy
                    }
                    VmSnapshotCurrentDispositionV1::Indeterminate => {
                        super::super::label_consumption_guard::LabelConsumptionEligibilityV1::Unknown
                    }
                    VmSnapshotCurrentDispositionV1::RebuildRequired
                    | VmSnapshotCurrentDispositionV1::HistoricalOnly => {
                        super::super::label_consumption_guard::LabelConsumptionEligibilityV1::Blocked
                    }
                },
                k1_report_sha256: digest('r'),
                reasons: Vec::new(),
            }],
            historical_manifest_rewritten: false,
            training_snapshot_mutated: false,
            model_retrained: false,
            equipment_control_authority: false,
            report_sha256: String::new(),
        };
        report.report_sha256 = requalification_digest(&report).unwrap();
        report
    }

    fn input(
        graph: VmSnapshotDependencyGraphV1,
        reports: Vec<VmSnapshotRequalificationReportV1>,
    ) -> VmImpactClosureInputV1 {
        VmImpactClosureInputV1 {
            schema: VM_IMPACT_CLOSURE_INPUT_SCHEMA_V1.into(),
            input_id: "vm-impact-input-001".into(),
            graph,
            snapshot_requalification_reports: reports,
        }
    }

    #[test]
    fn clean_single_root_is_unaffected() {
        let root = node(
            "root",
            VmModelDerivationKindV1::IndependentRetrain,
            VmSnapshotPurposeV1::Training,
            'a',
            'b',
        );
        let g = graph(vec![root.clone()], Vec::new(), true, true);
        let report = compute_vm_snapshot_impact_closure_v1(&input(
            g,
            vec![requal(
                "root",
                &root.manifest_sha256,
                VmSnapshotCurrentDispositionV1::CurrentEligible,
            )],
        ))
        .unwrap();
        assert_eq!(report.status, VmImpactClosureStatusV1::Complete);
        assert_eq!(report.entries[0].state, VmSnapshotImpactStateV1::Unaffected);
        assert!(report.rebuild_order_snapshot_ids.is_empty());
    }

    #[test]
    fn blocked_base_propagates_through_fine_tune_chain() {
        let root = node(
            "root",
            VmModelDerivationKindV1::IndependentRetrain,
            VmSnapshotPurposeV1::Training,
            'a',
            'b',
        );
        let child = node(
            "child",
            VmModelDerivationKindV1::FineTuneFromParent,
            VmSnapshotPurposeV1::Training,
            'c',
            'd',
        );
        let grandchild = node(
            "grandchild",
            VmModelDerivationKindV1::WarmStartFromParent,
            VmSnapshotPurposeV1::Training,
            'e',
            'f',
        );
        let edges = vec![
            edge(
                "e1",
                &root,
                &child,
                vec![VmDependencyDimensionV1::ModelWeights],
            ),
            edge(
                "e2",
                &child,
                &grandchild,
                vec![VmDependencyDimensionV1::ModelWeights],
            ),
        ];
        let g = graph(
            vec![grandchild.clone(), child.clone(), root.clone()],
            edges,
            true,
            true,
        );
        let reports = vec![
            requal(
                "root",
                &root.manifest_sha256,
                VmSnapshotCurrentDispositionV1::RebuildRequired,
            ),
            requal(
                "child",
                &child.manifest_sha256,
                VmSnapshotCurrentDispositionV1::CurrentEligible,
            ),
            requal(
                "grandchild",
                &grandchild.manifest_sha256,
                VmSnapshotCurrentDispositionV1::CurrentEligible,
            ),
        ];
        let report = compute_vm_snapshot_impact_closure_v1(&input(g, reports)).unwrap();
        assert_eq!(report.direct_impact_snapshot_ids, vec!["root".to_string()]);
        assert_eq!(
            report.transitive_impact_snapshot_ids,
            vec!["child".to_string(), "grandchild".to_string()]
        );
        assert_eq!(
            report.rebuild_order_snapshot_ids,
            vec![
                "root".to_string(),
                "child".to_string(),
                "grandchild".to_string()
            ]
        );
    }

    #[test]
    fn impacted_descendants_retain_exact_source_label_ids() {
        let root = node(
            "root",
            VmModelDerivationKindV1::IndependentRetrain,
            VmSnapshotPurposeV1::Training,
            'a',
            'b',
        );
        let child = node(
            "child",
            VmModelDerivationKindV1::FineTuneFromParent,
            VmSnapshotPurposeV1::Training,
            'c',
            'd',
        );
        let g = graph(
            vec![root.clone(), child.clone()],
            vec![edge(
                "inherits-root",
                &root,
                &child,
                vec![VmDependencyDimensionV1::ModelWeights],
            )],
            true,
            true,
        );
        let report = compute_vm_snapshot_impact_closure_v1(&input(
            g,
            vec![
                requal(
                    "root",
                    &root.manifest_sha256,
                    VmSnapshotCurrentDispositionV1::RebuildRequired,
                ),
                requal(
                    "child",
                    &child.manifest_sha256,
                    VmSnapshotCurrentDispositionV1::CurrentEligible,
                ),
            ],
        ))
        .unwrap();
        let root_entry = report
            .entries
            .iter()
            .find(|entry| entry.snapshot_id == "root")
            .unwrap();
        let child_entry = report
            .entries
            .iter()
            .find(|entry| entry.snapshot_id == "child")
            .unwrap();
        assert_eq!(root_entry.direct_affected_label_ids, vec!["label-root"]);
        assert_eq!(child_entry.inherited_affected_label_ids, vec!["label-root"]);
        assert!(child_entry.direct_affected_label_ids.is_empty());
    }

    #[test]
    fn independent_retrain_does_not_inherit_parent_model_state() {
        let root = node(
            "root",
            VmModelDerivationKindV1::IndependentRetrain,
            VmSnapshotPurposeV1::Training,
            'a',
            'b',
        );
        let child = node(
            "child",
            VmModelDerivationKindV1::IndependentRetrain,
            VmSnapshotPurposeV1::Training,
            'c',
            'd',
        );
        let g = graph(
            vec![root.clone(), child.clone()],
            vec![edge(
                "baseline-reference",
                &root,
                &child,
                vec![VmDependencyDimensionV1::FeatureSchema],
            )],
            true,
            true,
        );
        let report = compute_vm_snapshot_impact_closure_v1(&input(
            g,
            vec![
                requal(
                    "root",
                    &root.manifest_sha256,
                    VmSnapshotCurrentDispositionV1::RebuildRequired,
                ),
                requal(
                    "child",
                    &child.manifest_sha256,
                    VmSnapshotCurrentDispositionV1::CurrentEligible,
                ),
            ],
        ))
        .unwrap();
        assert_eq!(report.transitive_impact_snapshot_ids, Vec::<String>::new());
        assert_eq!(report.entries[1].state, VmSnapshotImpactStateV1::Unaffected);
    }

    #[test]
    fn evaluation_child_becomes_historical_only_not_rebuild_candidate() {
        let root = node(
            "root",
            VmModelDerivationKindV1::IndependentRetrain,
            VmSnapshotPurposeV1::Training,
            'a',
            'b',
        );
        let eval = node(
            "eval",
            VmModelDerivationKindV1::EvaluationOnly,
            VmSnapshotPurposeV1::Evaluation,
            'c',
            'd',
        );
        let g = graph(
            vec![root.clone(), eval.clone()],
            vec![edge(
                "evaluate-root",
                &root,
                &eval,
                vec![VmDependencyDimensionV1::EvaluationTarget],
            )],
            true,
            true,
        );
        let report = compute_vm_snapshot_impact_closure_v1(&input(
            g,
            vec![
                requal(
                    "root",
                    &root.manifest_sha256,
                    VmSnapshotCurrentDispositionV1::RebuildRequired,
                ),
                requal(
                    "eval",
                    &eval.manifest_sha256,
                    VmSnapshotCurrentDispositionV1::CurrentEligible,
                ),
            ],
        ))
        .unwrap();
        let entry = report
            .entries
            .iter()
            .find(|entry| entry.snapshot_id == "eval")
            .unwrap();
        assert_eq!(entry.state, VmSnapshotImpactStateV1::HistoricalOnly);
        assert!(!entry.rebuild_candidate);
    }

    #[test]
    fn ensemble_is_impacted_when_any_parent_model_is_blocked() {
        let left = node(
            "left",
            VmModelDerivationKindV1::IndependentRetrain,
            VmSnapshotPurposeV1::Training,
            'a',
            'b',
        );
        let right = node(
            "right",
            VmModelDerivationKindV1::IndependentRetrain,
            VmSnapshotPurposeV1::Training,
            'c',
            'd',
        );
        let ensemble = node(
            "ensemble",
            VmModelDerivationKindV1::EnsembleOfParents,
            VmSnapshotPurposeV1::Training,
            'e',
            'f',
        );
        let g = graph(
            vec![left.clone(), right.clone(), ensemble.clone()],
            vec![
                edge(
                    "left-parent",
                    &left,
                    &ensemble,
                    vec![VmDependencyDimensionV1::ModelWeights],
                ),
                edge(
                    "right-parent",
                    &right,
                    &ensemble,
                    vec![VmDependencyDimensionV1::ModelWeights],
                ),
            ],
            true,
            true,
        );
        let report = compute_vm_snapshot_impact_closure_v1(&input(
            g,
            vec![
                requal(
                    "left",
                    &left.manifest_sha256,
                    VmSnapshotCurrentDispositionV1::RebuildRequired,
                ),
                requal(
                    "right",
                    &right.manifest_sha256,
                    VmSnapshotCurrentDispositionV1::CurrentEligible,
                ),
                requal(
                    "ensemble",
                    &ensemble.manifest_sha256,
                    VmSnapshotCurrentDispositionV1::CurrentEligible,
                ),
            ],
        ))
        .unwrap();
        let entry = report
            .entries
            .iter()
            .find(|entry| entry.snapshot_id == "ensemble")
            .unwrap();
        assert_eq!(
            entry.state,
            VmSnapshotImpactStateV1::InheritedRebuildRequired
        );
        assert_eq!(entry.inherited_from_snapshot_ids, vec!["left".to_string()]);
        assert!(entry.rebuild_candidate);
    }

    #[test]
    fn distilled_student_inherits_teacher_model_state_dependency() {
        let teacher = node(
            "teacher",
            VmModelDerivationKindV1::IndependentRetrain,
            VmSnapshotPurposeV1::Training,
            'a',
            'b',
        );
        let student = node(
            "student",
            VmModelDerivationKindV1::DistilledFromParent,
            VmSnapshotPurposeV1::Training,
            'c',
            'd',
        );
        let g = graph(
            vec![teacher.clone(), student.clone()],
            vec![edge(
                "teacher-targets",
                &teacher,
                &student,
                vec![VmDependencyDimensionV1::ModelWeights],
            )],
            true,
            true,
        );
        let report = compute_vm_snapshot_impact_closure_v1(&input(
            g,
            vec![
                requal(
                    "teacher",
                    &teacher.manifest_sha256,
                    VmSnapshotCurrentDispositionV1::RebuildRequired,
                ),
                requal(
                    "student",
                    &student.manifest_sha256,
                    VmSnapshotCurrentDispositionV1::CurrentEligible,
                ),
            ],
        ))
        .unwrap();
        assert_eq!(
            report.transitive_impact_snapshot_ids,
            vec!["student".to_string()]
        );
    }

    #[test]
    fn blocked_parent_dominates_unknown_parent_without_losing_source_nodes() {
        let blocked = node(
            "blocked",
            VmModelDerivationKindV1::IndependentRetrain,
            VmSnapshotPurposeV1::Training,
            'a',
            'b',
        );
        let unknown = node(
            "unknown",
            VmModelDerivationKindV1::IndependentRetrain,
            VmSnapshotPurposeV1::Training,
            'c',
            'd',
        );
        let child = node(
            "child",
            VmModelDerivationKindV1::EnsembleOfParents,
            VmSnapshotPurposeV1::Training,
            'e',
            'f',
        );
        let g = graph(
            vec![blocked.clone(), unknown.clone(), child.clone()],
            vec![
                edge(
                    "blocked-parent",
                    &blocked,
                    &child,
                    vec![VmDependencyDimensionV1::ModelWeights],
                ),
                edge(
                    "unknown-parent",
                    &unknown,
                    &child,
                    vec![VmDependencyDimensionV1::ModelWeights],
                ),
            ],
            true,
            true,
        );
        let report = compute_vm_snapshot_impact_closure_v1(&input(
            g,
            vec![
                requal(
                    "blocked",
                    &blocked.manifest_sha256,
                    VmSnapshotCurrentDispositionV1::RebuildRequired,
                ),
                requal(
                    "unknown",
                    &unknown.manifest_sha256,
                    VmSnapshotCurrentDispositionV1::Indeterminate,
                ),
                requal(
                    "child",
                    &child.manifest_sha256,
                    VmSnapshotCurrentDispositionV1::CurrentEligible,
                ),
            ],
        ))
        .unwrap();
        let entry = report
            .entries
            .iter()
            .find(|entry| entry.snapshot_id == "child")
            .unwrap();
        assert_eq!(
            entry.state,
            VmSnapshotImpactStateV1::InheritedRebuildRequired
        );
        assert_eq!(
            entry.inherited_from_snapshot_ids,
            vec!["blocked".to_string(), "unknown".to_string()]
        );
    }

    #[test]
    fn unknown_parent_propagates_unknown_to_declared_dependants() {
        let root = node(
            "root",
            VmModelDerivationKindV1::IndependentRetrain,
            VmSnapshotPurposeV1::Training,
            'a',
            'b',
        );
        let child = node(
            "child",
            VmModelDerivationKindV1::FineTuneFromParent,
            VmSnapshotPurposeV1::Training,
            'c',
            'd',
        );
        let g = graph(
            vec![root.clone(), child.clone()],
            vec![edge(
                "e1",
                &root,
                &child,
                vec![VmDependencyDimensionV1::ModelWeights],
            )],
            true,
            true,
        );
        let report = compute_vm_snapshot_impact_closure_v1(&input(
            g,
            vec![
                requal(
                    "root",
                    &root.manifest_sha256,
                    VmSnapshotCurrentDispositionV1::Indeterminate,
                ),
                requal(
                    "child",
                    &child.manifest_sha256,
                    VmSnapshotCurrentDispositionV1::CurrentEligible,
                ),
            ],
        ))
        .unwrap();
        assert_eq!(
            report.entries[1].state,
            VmSnapshotImpactStateV1::InheritedIndeterminate
        );
    }

    #[test]
    fn incomplete_inventory_returns_unknown_not_an_empty_impact_set() {
        let root = node(
            "root",
            VmModelDerivationKindV1::IndependentRetrain,
            VmSnapshotPurposeV1::Training,
            'a',
            'b',
        );
        let g = graph(vec![root.clone()], Vec::new(), false, true);
        let report = compute_vm_snapshot_impact_closure_v1(&input(g, Vec::new())).unwrap();
        assert_eq!(report.status, VmImpactClosureStatusV1::Unknown);
        assert_eq!(report.unresolved_snapshot_ids, vec!["root".to_string()]);
    }

    #[test]
    fn non_total_graph_ordering_returns_unknown() {
        let root = node(
            "root",
            VmModelDerivationKindV1::IndependentRetrain,
            VmSnapshotPurposeV1::Training,
            'a',
            'b',
        );
        let g = graph(vec![root.clone()], Vec::new(), true, false);
        let report = compute_vm_snapshot_impact_closure_v1(&input(g, Vec::new())).unwrap();
        assert_eq!(report.status, VmImpactClosureStatusV1::Unknown);
    }

    #[test]
    fn cycle_is_rejected() {
        let a = node(
            "a",
            VmModelDerivationKindV1::FineTuneFromParent,
            VmSnapshotPurposeV1::Training,
            'a',
            'b',
        );
        let b = node(
            "b",
            VmModelDerivationKindV1::FineTuneFromParent,
            VmSnapshotPurposeV1::Training,
            'c',
            'd',
        );
        let g = graph(
            vec![a.clone(), b.clone()],
            vec![
                edge("ab", &a, &b, vec![VmDependencyDimensionV1::ModelWeights]),
                edge("ba", &b, &a, vec![VmDependencyDimensionV1::ModelWeights]),
            ],
            true,
            true,
        );
        assert!(validate_graph(&g).is_err());
    }

    #[test]
    fn parent_artifact_digest_mismatch_is_rejected() {
        let root = node(
            "root",
            VmModelDerivationKindV1::IndependentRetrain,
            VmSnapshotPurposeV1::Training,
            'a',
            'b',
        );
        let child = node(
            "child",
            VmModelDerivationKindV1::FineTuneFromParent,
            VmSnapshotPurposeV1::Training,
            'c',
            'd',
        );
        let mut bad = edge(
            "e1",
            &root,
            &child,
            vec![VmDependencyDimensionV1::ModelWeights],
        );
        bad.parent_artifact_sha256 = digest('z');
        bad.edge_sha256 = seal_vm_snapshot_dependency_edge_v1(&bad).unwrap();
        let g = graph(vec![root, child], vec![bad], true, true);
        assert!(validate_graph(&g).is_err());
    }

    #[test]
    fn independent_retrain_cannot_declare_model_weight_inheritance() {
        let root = node(
            "root",
            VmModelDerivationKindV1::IndependentRetrain,
            VmSnapshotPurposeV1::Training,
            'a',
            'b',
        );
        let child = node(
            "child",
            VmModelDerivationKindV1::IndependentRetrain,
            VmSnapshotPurposeV1::Training,
            'c',
            'd',
        );
        let g = graph(
            vec![root.clone(), child.clone()],
            vec![edge(
                "e1",
                &root,
                &child,
                vec![VmDependencyDimensionV1::ModelWeights],
            )],
            true,
            true,
        );
        assert!(validate_graph(&g).is_err());
    }

    #[test]
    fn inherited_dimension_order_does_not_change_edge_or_graph_digest() {
        let root = node(
            "root",
            VmModelDerivationKindV1::IndependentRetrain,
            VmSnapshotPurposeV1::Training,
            'a',
            'b',
        );
        let child = node(
            "child",
            VmModelDerivationKindV1::FineTuneFromParent,
            VmSnapshotPurposeV1::Training,
            'c',
            'd',
        );
        let mut forward = edge(
            "e1",
            &root,
            &child,
            vec![
                VmDependencyDimensionV1::ModelWeights,
                VmDependencyDimensionV1::CalibrationState,
            ],
        );
        let forward_digest = forward.edge_sha256.clone();
        forward.inherited_dimensions.reverse();
        assert_eq!(
            forward_digest,
            seal_vm_snapshot_dependency_edge_v1(&forward).unwrap()
        );

        let first = graph(
            vec![root.clone(), child.clone()],
            vec![edge(
                "e1",
                &root,
                &child,
                vec![
                    VmDependencyDimensionV1::ModelWeights,
                    VmDependencyDimensionV1::CalibrationState,
                ],
            )],
            true,
            true,
        );
        let second = graph(
            vec![root.clone(), child.clone()],
            vec![edge(
                "e1",
                &root,
                &child,
                vec![
                    VmDependencyDimensionV1::CalibrationState,
                    VmDependencyDimensionV1::ModelWeights,
                ],
            )],
            true,
            true,
        );
        assert_eq!(first.graph_sha256, second.graph_sha256);
    }

    #[test]
    fn edge_order_and_node_order_do_not_change_graph_digest() {
        let root = node(
            "root",
            VmModelDerivationKindV1::IndependentRetrain,
            VmSnapshotPurposeV1::Training,
            'a',
            'b',
        );
        let child = node(
            "child",
            VmModelDerivationKindV1::FineTuneFromParent,
            VmSnapshotPurposeV1::Training,
            'c',
            'd',
        );
        let edge = edge(
            "e1",
            &root,
            &child,
            vec![VmDependencyDimensionV1::ModelWeights],
        );
        let mut first = graph(
            vec![root.clone(), child.clone()],
            vec![edge.clone()],
            true,
            true,
        );
        let first_digest = first.graph_sha256.clone();
        first.nodes.reverse();
        first.edges.reverse();
        first.graph_sha256 = seal_vm_snapshot_dependency_graph_v1(&first).unwrap();
        assert_eq!(first_digest, first.graph_sha256);
    }

    #[test]
    fn missing_requalification_report_returns_unknown() {
        let root = node(
            "root",
            VmModelDerivationKindV1::IndependentRetrain,
            VmSnapshotPurposeV1::Training,
            'a',
            'b',
        );
        let g = graph(vec![root.clone()], Vec::new(), true, true);
        let report = compute_vm_snapshot_impact_closure_v1(&input(g, Vec::new())).unwrap();
        assert_eq!(report.status, VmImpactClosureStatusV1::Unknown);
        assert_eq!(
            report.reason_codes,
            vec!["snapshot_requalification_coverage_incomplete"]
        );
    }

    #[test]
    fn inconsistent_row_level_and_aggregate_requalification_counts_are_rejected() {
        let root = node(
            "root",
            VmModelDerivationKindV1::IndependentRetrain,
            VmSnapshotPurposeV1::Training,
            'a',
            'b',
        );
        let g = graph(vec![root.clone()], Vec::new(), true, true);
        let mut report = requal(
            "root",
            &root.manifest_sha256,
            VmSnapshotCurrentDispositionV1::CurrentEligible,
        );
        report.current_eligible_count = 0;
        report.report_sha256 = requalification_digest(&report).unwrap();
        assert!(compute_vm_snapshot_impact_closure_v1(&input(g, vec![report])).is_err());
    }

    #[test]
    fn tampered_snapshot_requalification_receipt_is_rejected() {
        let root = node(
            "root",
            VmModelDerivationKindV1::IndependentRetrain,
            VmSnapshotPurposeV1::Training,
            'a',
            'b',
        );
        let g = graph(vec![root.clone()], Vec::new(), true, true);
        let mut report = requal(
            "root",
            &root.manifest_sha256,
            VmSnapshotCurrentDispositionV1::CurrentEligible,
        );
        report.current_eligible_count = 0;
        assert!(compute_vm_snapshot_impact_closure_v1(&input(g, vec![report])).is_err());
    }
}
