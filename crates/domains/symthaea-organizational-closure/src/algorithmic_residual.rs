//! Evidence-backed residual improvements beyond a frozen causal algorithm.
//!
//! Separates a baseline selected on calibration data, an evaluator-only
//! algorithm-suite ceiling, agent edits with evidence receipts, and hidden
//! graph truth. This module scores/audits edits; it does not discover edges
//! or authenticate the external producer of an evidence receipt.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResidualDirectedEdge {
    pub cause: String,
    pub effect: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CausalGraph {
    pub nodes: BTreeSet<String>,
    pub edges: BTreeSet<ResidualDirectedEdge>,
}

impl CausalGraph {
    pub fn new(
        nodes: impl IntoIterator<Item = String>,
        edges: impl IntoIterator<Item = ResidualDirectedEdge>,
    ) -> Self {
        Self {
            nodes: nodes.into_iter().collect(),
            edges: edges.into_iter().collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AlgorithmGraphOutput {
    pub algorithm_id: String,
    pub graph: CausalGraph,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    ConditionalIndependence,
    InterventionContrast,
    InvarianceTest,
    NegativeControl,
    MechanismDiagnostic,
    DomainConstraint,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentEvidenceReceipt {
    pub receipt_id: String,
    pub kind: EvidenceKind,
    pub description: String,
}

/// Agent-facing information. Ground truth and evidence adjudications are
/// deliberately kept in ResidualDiscoveryOracle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResidualDiscoveryScenario {
    pub declared_nodes: BTreeSet<String>,
    pub require_dag: bool,
    /// Chosen using a disjoint calibration split and frozen before evaluation.
    pub calibration_selected_baseline_id: String,
    /// Harness-authenticated digest of the calibration partition.
    pub calibration_split_digest: String,
    /// Harness-authenticated digest of the sealed evaluation partition.
    pub evaluation_split_digest: String,
    /// Digest of the exact algorithm registry and versions used for this task.
    pub algorithm_registry_digest: String,
    pub algorithm_portfolio: Vec<AlgorithmGraphOutput>,
    pub available_evidence: BTreeMap<String, AgentEvidenceReceipt>,
    pub max_edge_edits: usize,
    pub mechanism_family: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceReceiptStatus {
    Verified,
    Rejected,
    Unavailable,
}

/// Evaluator-only adjudication. A verified receipt supports the claimed edit;
/// it does not imply the edited edge is true. Truth is scored independently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceReceiptOracle {
    pub receipt_id: String,
    pub status: EvidenceReceiptStatus,
    pub supported_additions: BTreeSet<ResidualDirectedEdge>,
    pub supported_removals: BTreeSet<ResidualDirectedEdge>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaselineSelectionOracle {
    pub selected_algorithm_id: String,
    pub calibration_split_digest: String,
    pub evaluation_split_digest: String,
    pub algorithm_registry_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidualDiscoveryOracle {
    pub ground_truth: CausalGraph,
    pub baseline_selection: BaselineSelectionOracle,
    pub evidence: BTreeMap<String, EvidenceReceiptOracle>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum EdgeEditOperation {
    Add {
        edge: ResidualDirectedEdge,
    },
    Remove {
        edge: ResidualDirectedEdge,
    },
    Reorient {
        from: ResidualDirectedEdge,
        to: ResidualDirectedEdge,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceBackedEdgeEdit {
    pub operation: EdgeEditOperation,
    pub evidence_receipt_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResidualDiscoveryTrace {
    pub selected_baseline_algorithm_id: String,
    pub final_graph: CausalGraph,
    pub edits: Vec<EvidenceBackedEdgeEdit>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidualQualification {
    Qualified,
    NotQualified,
    Inconclusive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidualFindingCode {
    EmptyNodeSet,
    EmptyMechanismFamily,
    EmptyAlgorithmPortfolio,
    EmptyAlgorithmId,
    DuplicateAlgorithmId,
    FrozenBaselineMissing,
    EmptyCalibrationDigest,
    EmptyEvaluationDigest,
    EmptyAlgorithmRegistryDigest,
    CalibrationEvaluationSplitReuse,
    BaselineSelectionMismatch,
    AlgorithmRegistryMismatch,
    OracleNodeSetMismatch,
    OracleGraphInvalid,
    AlgorithmGraphInvalid,
    DuplicateEvidenceReceipt,
    EvidenceReceiptIdMismatch,
    EvidenceOracleIdMismatch,
    MissingEvidenceOracle,
    UnknownEvidenceReceipt,
    MissingEvidenceReceipt,
    RejectedEvidenceReceipt,
    EvidenceUnavailable,
    ReceiptDoesNotSupportEdit,
    TooManyEdits,
    EmptyEdgeEndpoint,
    UnknownEdgeEndpoint,
    SelfEdge,
    AddExistingEdge,
    RemoveMissingEdge,
    ReorientMissingSource,
    ReorientExistingTarget,
    ReorientDifferentAdjacency,
    FinalGraphInvalid,
    FinalEdgeSetDiffersFromReplay,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResidualFinding {
    pub code: ResidualFindingCode,
    pub edit_index: Option<usize>,
    pub receipt_id: Option<String>,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphQualityMetrics {
    pub adjacency_precision: f64,
    pub adjacency_recall: f64,
    pub adjacency_f1: f64,
    pub directed_precision: f64,
    pub directed_recall: f64,
    pub directed_f1: f64,
    pub structural_hamming_distance: usize,
    pub true_positive_directed_edges: usize,
    pub false_positive_directed_edges: usize,
    pub false_negative_directed_edges: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResidualDiscoveryReport {
    pub qualification: ResidualQualification,
    pub mechanism_family: String,
    pub selected_baseline_algorithm_id: String,
    pub baseline_metrics: Option<GraphQualityMetrics>,
    pub final_metrics: Option<GraphQualityMetrics>,
    /// Ex-post best algorithm, visible only in the evaluator report.
    pub ex_post_algorithm_suite_ceiling_f1: Option<f64>,
    pub residual_adjacency_f1_gain: Option<f64>,
    pub residual_directed_f1_gain: Option<f64>,
    pub adjacency_headroom_to_truth: Option<f64>,
    pub adjacency_headroom_closed_ratio: Option<f64>,
    pub edit_count: usize,
    pub correct_edit_count: usize,
    pub edit_precision: Option<f64>,
    pub edit_recall: Option<f64>,
    pub evidence_backed_edit_count: usize,
    pub evidence_backed_edit_rate: Option<f64>,
    pub novel_true_edge_count: usize,
    pub total_novel_true_edge_count: usize,
    pub novel_true_edge_recall: Option<f64>,
    pub findings: Vec<ResidualFinding>,
}

impl ResidualDiscoveryReport {
    pub fn is_qualified(&self) -> bool {
        self.qualification == ResidualQualification::Qualified && self.findings.is_empty()
    }

    pub fn improved_over_frozen_baseline(&self) -> bool {
        self.is_qualified()
            && self
                .residual_adjacency_f1_gain
                .is_some_and(|gain| gain > 0.0)
    }
}

fn finding(
    findings: &mut Vec<ResidualFinding>,
    code: ResidualFindingCode,
    edit_index: Option<usize>,
    receipt_id: Option<String>,
    detail: impl Into<String>,
) {
    findings.push(ResidualFinding {
        code,
        edit_index,
        receipt_id,
        detail: detail.into(),
    });
}

fn pair(edge: &ResidualDirectedEdge) -> (String, String) {
    if edge.cause <= edge.effect {
        (edge.cause.clone(), edge.effect.clone())
    } else {
        (edge.effect.clone(), edge.cause.clone())
    }
}

fn f1(precision: f64, recall: f64) -> f64 {
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        if numerator == 0 { 1.0 } else { 0.0 }
    } else {
        numerator as f64 / denominator as f64
    }
}

fn metrics(truth: &CausalGraph, predicted: &CausalGraph) -> GraphQualityMetrics {
    let tp = predicted.edges.intersection(&truth.edges).count();
    let fp = predicted.edges.difference(&truth.edges).count();
    let fn_count = truth.edges.difference(&predicted.edges).count();
    let directed_precision = ratio(tp, predicted.edges.len());
    let directed_recall = ratio(tp, truth.edges.len());

    let truth_pairs: BTreeSet<(String, String)> = truth.edges.iter().map(pair).collect();
    let predicted_pairs: BTreeSet<(String, String)> = predicted.edges.iter().map(pair).collect();
    let skeleton_tp = truth_pairs.intersection(&predicted_pairs).count();
    let adjacency_precision = ratio(skeleton_tp, predicted_pairs.len());
    let adjacency_recall = ratio(skeleton_tp, truth_pairs.len());

    let mut all_pairs = truth_pairs;
    all_pairs.extend(predicted_pairs);
    let mut shd = 0usize;
    for (left, right) in all_pairs {
        let tf = truth.edges.contains(&ResidualDirectedEdge {
            cause: left.clone(),
            effect: right.clone(),
        });
        let tr = truth.edges.contains(&ResidualDirectedEdge {
            cause: right.clone(),
            effect: left.clone(),
        });
        let pf = predicted.edges.contains(&ResidualDirectedEdge {
            cause: left.clone(),
            effect: right.clone(),
        });
        let pr = predicted.edges.contains(&ResidualDirectedEdge {
            cause: right,
            effect: left,
        });
        let truth_has = tf || tr;
        let predicted_has = pf || pr;
        let orientation_matches = (tf && pf) || (tr && pr);
        if (truth_has != predicted_has) || (truth_has && predicted_has && !orientation_matches) {
            shd += 1;
        }
    }

    GraphQualityMetrics {
        adjacency_precision,
        adjacency_recall,
        adjacency_f1: f1(adjacency_precision, adjacency_recall),
        directed_precision,
        directed_recall,
        directed_f1: f1(directed_precision, directed_recall),
        structural_hamming_distance: shd,
        true_positive_directed_edges: tp,
        false_positive_directed_edges: fp,
        false_negative_directed_edges: fn_count,
    }
}

fn has_cycle(graph: &CausalGraph) -> bool {
    let mut indegree: BTreeMap<String, usize> =
        graph.nodes.iter().map(|node| (node.clone(), 0)).collect();
    let mut children: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for edge in &graph.edges {
        children
            .entry(edge.cause.clone())
            .or_default()
            .insert(edge.effect.clone());
        *indegree.entry(edge.effect.clone()).or_default() += 1;
    }
    let mut ready: VecDeque<String> = indegree
        .iter()
        .filter_map(|(node, degree)| (*degree == 0).then_some(node.clone()))
        .collect();
    let mut visited = 0usize;
    while let Some(node) = ready.pop_front() {
        visited += 1;
        if let Some(next) = children.get(&node) {
            for child in next {
                if let Some(degree) = indegree.get_mut(child) {
                    *degree = degree.saturating_sub(1);
                    if *degree == 0 {
                        ready.push_back(child.clone());
                    }
                }
            }
        }
    }
    visited != graph.nodes.len()
}

fn graph_error(
    graph: &CausalGraph,
    nodes: &BTreeSet<String>,
    require_dag: bool,
) -> Option<ResidualFindingCode> {
    if graph.nodes != *nodes {
        return Some(ResidualFindingCode::OracleNodeSetMismatch);
    }
    for edge in &graph.edges {
        if edge.cause.trim().is_empty() || edge.effect.trim().is_empty() {
            return Some(ResidualFindingCode::EmptyEdgeEndpoint);
        }
        if edge.cause == edge.effect {
            return Some(ResidualFindingCode::SelfEdge);
        }
        if !nodes.contains(&edge.cause) || !nodes.contains(&edge.effect) {
            return Some(ResidualFindingCode::UnknownEdgeEndpoint);
        }
    }
    if require_dag && has_cycle(graph) {
        return Some(ResidualFindingCode::DirectedCycle);
    }
    None
}

fn needed_edits(baseline: &CausalGraph, truth: &CausalGraph) -> BTreeSet<EdgeEditOperation> {
    let mut pairs = BTreeSet::new();
    for edge in baseline.edges.iter().chain(truth.edges.iter()) {
        pairs.insert(pair(edge));
    }
    let mut edits = BTreeSet::new();
    for (left, right) in pairs {
        let bf = baseline.edges.contains(&ResidualDirectedEdge {
            cause: left.clone(),
            effect: right.clone(),
        });
        let br = baseline.edges.contains(&ResidualDirectedEdge {
            cause: right.clone(),
            effect: left.clone(),
        });
        let tf = truth.edges.contains(&ResidualDirectedEdge {
            cause: left.clone(),
            effect: right.clone(),
        });
        let tr = truth.edges.contains(&ResidualDirectedEdge {
            cause: right.clone(),
            effect: left.clone(),
        });
        match ((bf, br), (tf, tr)) {
            ((false, false), (true, false)) => {
                edits.insert(EdgeEditOperation::Add {
                    edge: ResidualDirectedEdge {
                        cause: left,
                        effect: right,
                    },
                });
            }
            ((false, false), (false, true)) => {
                edits.insert(EdgeEditOperation::Add {
                    edge: ResidualDirectedEdge {
                        cause: right,
                        effect: left,
                    },
                });
            }
            ((true, false), (false, false)) => {
                edits.insert(EdgeEditOperation::Remove {
                    edge: ResidualDirectedEdge {
                        cause: left,
                        effect: right,
                    },
                });
            }
            ((false, true), (false, false)) => {
                edits.insert(EdgeEditOperation::Remove {
                    edge: ResidualDirectedEdge {
                        cause: right,
                        effect: left,
                    },
                });
            }
            ((true, false), (false, true)) => {
                edits.insert(EdgeEditOperation::Reorient {
                    from: ResidualDirectedEdge {
                        cause: left.clone(),
                        effect: right.clone(),
                    },
                    to: ResidualDirectedEdge {
                        cause: right,
                        effect: left,
                    },
                });
            }
            ((false, true), (true, false)) => {
                edits.insert(EdgeEditOperation::Reorient {
                    from: ResidualDirectedEdge {
                        cause: right.clone(),
                        effect: left.clone(),
                    },
                    to: ResidualDirectedEdge {
                        cause: left,
                        effect: right,
                    },
                });
            }
            _ => {}
        }
    }
    edits
}

fn receipt_supports(operation: &EdgeEditOperation, receipt: &EvidenceReceiptOracle) -> bool {
    match operation {
        EdgeEditOperation::Add { edge } => receipt.supported_additions.contains(edge),
        EdgeEditOperation::Remove { edge } => receipt.supported_removals.contains(edge),
        EdgeEditOperation::Reorient { from, to } => {
            receipt.supported_removals.contains(from) && receipt.supported_additions.contains(to)
        }
    }
}

fn apply(
    graph: &mut CausalGraph,
    operation: &EdgeEditOperation,
    index: usize,
    findings: &mut Vec<ResidualFinding>,
) {
    match operation {
        EdgeEditOperation::Add { edge } => {
            if !graph.edges.insert(edge.clone()) {
                finding(
                    findings,
                    ResidualFindingCode::AddExistingEdge,
                    Some(index),
                    None,
                    "cannot add an edge already present",
                );
            }
        }
        EdgeEditOperation::Remove { edge } => {
            if !graph.edges.remove(edge) {
                finding(
                    findings,
                    ResidualFindingCode::RemoveMissingEdge,
                    Some(index),
                    None,
                    "cannot remove an edge absent from the replay graph",
                );
            }
        }
        EdgeEditOperation::Reorient { from, to } => {
            if pair(from) != pair(to) {
                finding(
                    findings,
                    ResidualFindingCode::ReorientDifferentAdjacency,
                    Some(index),
                    None,
                    "reorientation must preserve the unordered node pair",
                );
                return;
            }
            if !graph.edges.remove(from) {
                finding(
                    findings,
                    ResidualFindingCode::ReorientMissingSource,
                    Some(index),
                    None,
                    "reorientation source edge is absent",
                );
                return;
            }
            if !graph.edges.insert(to.clone()) {
                finding(
                    findings,
                    ResidualFindingCode::ReorientExistingTarget,
                    Some(index),
                    None,
                    "reorientation target edge already exists",
                );
            }
        }
    }
}

fn empty_report(
    scenario: &ResidualDiscoveryScenario,
    trace: &ResidualDiscoveryTrace,
    qualification: ResidualQualification,
    findings: Vec<ResidualFinding>,
) -> ResidualDiscoveryReport {
    ResidualDiscoveryReport {
        qualification,
        mechanism_family: scenario.mechanism_family.clone(),
        selected_baseline_algorithm_id: trace.selected_baseline_algorithm_id.clone(),
        baseline_metrics: None,
        final_metrics: None,
        ex_post_algorithm_suite_ceiling_f1: None,
        residual_adjacency_f1_gain: None,
        residual_directed_f1_gain: None,
        adjacency_headroom_to_truth: None,
        adjacency_headroom_closed_ratio: None,
        edit_count: trace.edits.len(),
        correct_edit_count: 0,
        edit_precision: None,
        edit_recall: None,
        evidence_backed_edit_count: 0,
        evidence_backed_edit_rate: None,
        novel_true_edge_count: 0,
        total_novel_true_edge_count: 0,
        novel_true_edge_recall: None,
        findings,
    }
}

/// Evaluate a set of agent edge edits against a calibration-frozen baseline.
/// A verified receipt establishes edit provenance, not edge truth.
pub fn evaluate_algorithmic_residual(
    scenario: &ResidualDiscoveryScenario,
    oracle: &ResidualDiscoveryOracle,
    trace: &ResidualDiscoveryTrace,
) -> ResidualDiscoveryReport {
    let mut findings = Vec::new();
    if scenario.declared_nodes.is_empty() {
        finding(
            &mut findings,
            ResidualFindingCode::EmptyNodeSet,
            None,
            None,
            "node set is empty",
        );
    }
    if scenario.mechanism_family.trim().is_empty() {
        finding(
            &mut findings,
            ResidualFindingCode::EmptyMechanismFamily,
            None,
            None,
            "mechanism family is empty",
        );
    }
    if scenario.calibration_split_digest.trim().is_empty() {
        finding(
            &mut findings,
            ResidualFindingCode::EmptyCalibrationDigest,
            None,
            None,
            "calibration split digest is empty",
        );
    }
    if scenario.evaluation_split_digest.trim().is_empty() {
        finding(
            &mut findings,
            ResidualFindingCode::EmptyEvaluationDigest,
            None,
            None,
            "evaluation split digest is empty",
        );
    }
    if scenario.algorithm_registry_digest.trim().is_empty() {
        finding(
            &mut findings,
            ResidualFindingCode::EmptyAlgorithmRegistryDigest,
            None,
            None,
            "algorithm registry digest is empty",
        );
    }
    if scenario.calibration_split_digest == scenario.evaluation_split_digest {
        finding(
            &mut findings,
            ResidualFindingCode::CalibrationEvaluationSplitReuse,
            None,
            None,
            "calibration and evaluation split digests must differ",
        );
    }
    if scenario.calibration_selected_baseline_id != oracle.baseline_selection.selected_algorithm_id
        || scenario.calibration_split_digest != oracle.baseline_selection.calibration_split_digest
        || scenario.evaluation_split_digest != oracle.baseline_selection.evaluation_split_digest {
        finding(&mut findings, ResidualFindingCode::BaselineSelectionMismatch, None, None,
            "baseline ID or split digests differ from the evaluator's calibration receipt",
        );
    }
    if scenario.algorithm_registry_digest != oracle.baseline_selection.algorithm_registry_digest {
        finding(&mut findings, ResidualFindingCode::AlgorithmRegistryMismatch, None, None,
            "algorithm registry digest differs from the evaluator's frozen receipt",
        );
    }
    if scenario.algorithm_portfolio.is_empty() {
        finding(
            &mut findings,
            ResidualFindingCode::EmptyAlgorithmPortfolio,
            None,
            None,
            "algorithm portfolio is empty",
        );
    }
    if oracle.ground_truth.nodes != scenario.declared_nodes
        || graph_error(
            &oracle.ground_truth,
            &scenario.declared_nodes,
            scenario.require_dag,
        )
        .is_some()
    {
        finding(
            &mut findings,
            ResidualFindingCode::OracleGraphInvalid,
            None,
            None,
            "ground-truth graph violates the task graph contract",
        );
    }

    let mut algorithms: BTreeMap<String, &AlgorithmGraphOutput> = BTreeMap::new();
    for output in &scenario.algorithm_portfolio {
        if output.algorithm_id.trim().is_empty() {
            finding(
                &mut findings,
                ResidualFindingCode::EmptyAlgorithmId,
                None,
                None,
                "algorithm ID is empty",
            );
        }
        if algorithms
            .insert(output.algorithm_id.clone(), output)
            .is_some()
        {
            finding(
                &mut findings,
                ResidualFindingCode::DuplicateAlgorithmId,
                None,
                None,
                "algorithm ID is duplicated",
            );
        }
        if graph_error(
            &output.graph,
            &scenario.declared_nodes,
            scenario.require_dag,
        )
        .is_some()
        {
            finding(
                &mut findings,
                ResidualFindingCode::AlgorithmGraphInvalid,
                None,
                None,
                format!(
                    "algorithm {} returned an invalid graph",
                    output.algorithm_id
                ),
            );
        }
    }

    for (id, receipt) in &scenario.available_evidence {
        if id != &receipt.receipt_id || id.trim().is_empty() {
            finding(
                &mut findings,
                ResidualFindingCode::EvidenceReceiptIdMismatch,
                None,
                Some(id.clone()),
                "evidence map key and receipt ID differ or are empty",
            );
        }
    }
    for (id, receipt) in &oracle.evidence {
        if id != &receipt.receipt_id {
            finding(
                &mut findings,
                ResidualFindingCode::EvidenceOracleIdMismatch,
                None,
                Some(id.clone()),
                "evaluator evidence key and receipt ID differ",
            );
        }
    }

    let Some(baseline) = algorithms.get(&scenario.calibration_selected_baseline_id) else {
        finding(
            &mut findings,
            ResidualFindingCode::FrozenBaselineMissing,
            None,
            None,
            "calibration-selected baseline is absent from the test algorithm portfolio",
        );
        return empty_report(
            scenario,
            trace,
            ResidualQualification::NotQualified,
            findings,
        );
    };
    if trace.selected_baseline_algorithm_id != scenario.calibration_selected_baseline_id {
        finding(
            &mut findings,
            ResidualFindingCode::FrozenBaselineMissing,
            None,
            None,
            "agent selected a method other than the frozen calibration-selected baseline",
        );
    }
    if !findings.is_empty() {
        return empty_report(
            scenario,
            trace,
            ResidualQualification::NotQualified,
            findings,
        );
    }

    let mut replay = baseline.graph.clone();
    let mut backed_edits = 0usize;
    let mut unavailable = false;
    if trace.edits.len() > scenario.max_edge_edits {
        finding(
            &mut findings,
            ResidualFindingCode::TooManyEdits,
            None,
            None,
            "edge-edit budget exceeded",
        );
    }

    for (index, edit) in trace.edits.iter().enumerate() {
        let edit_number = index + 1;
        if edit.evidence_receipt_ids.is_empty() {
            finding(
                &mut findings,
                ResidualFindingCode::MissingEvidenceReceipt,
                Some(edit_number),
                None,
                "every edge edit needs an evidence receipt",
            );
        }
        let mut supported = false;
        for receipt_id in &edit.evidence_receipt_ids {
            if !scenario.available_evidence.contains_key(receipt_id) {
                finding(
                    &mut findings,
                    ResidualFindingCode::UnknownEvidenceReceipt,
                    Some(edit_number),
                    Some(receipt_id.clone()),
                    "receipt is not in the agent-visible evidence ledger",
                );
                continue;
            }
            let Some(receipt) = oracle.evidence.get(receipt_id) else {
                finding(
                    &mut findings,
                    ResidualFindingCode::MissingEvidenceOracle,
                    Some(edit_number),
                    Some(receipt_id.clone()),
                    "evaluator has no record for the cited receipt",
                );
                continue;
            };
            match receipt.status {
                EvidenceReceiptStatus::Verified => {
                    supported |= receipt_supports(&edit.operation, receipt);
                }
                EvidenceReceiptStatus::Rejected => {
                    finding(
                        &mut findings,
                        ResidualFindingCode::RejectedEvidenceReceipt,
                        Some(edit_number),
                        Some(receipt_id.clone()),
                        "cited evidence receipt was rejected",
                    );
                }
                EvidenceReceiptStatus::Unavailable => {
                    unavailable = true;
                    finding(
                        &mut findings,
                        ResidualFindingCode::EvidenceUnavailable,
                        Some(edit_number),
                        Some(receipt_id.clone()),
                        "cited evidence could not be independently qualified",
                    );
                }
            }
        }
        if supported {
            backed_edits += 1;
        } else {
            finding(
                &mut findings,
                ResidualFindingCode::ReceiptDoesNotSupportEdit,
                Some(edit_number),
                None,
                "no verified cited receipt supports the exact edge operation",
            );
        }
        apply(&mut replay, &edit.operation, edit_number, &mut findings);
    }

    if graph_error(
        &trace.final_graph,
        &scenario.declared_nodes,
        scenario.require_dag,
    )
    .is_some()
    {
        finding(
            &mut findings,
            ResidualFindingCode::FinalGraphInvalid,
            None,
            None,
            "final graph violates node, endpoint or DAG constraints",
        );
    }
    if replay.nodes != trace.final_graph.nodes || replay.edges != trace.final_graph.edges {
        finding(
            &mut findings,
            ResidualFindingCode::FinalEdgeSetDiffersFromReplay,
            None,
            None,
            "final graph differs from frozen baseline plus the recorded edits",
        );
    }

    if !findings.is_empty() {
        let only_uncertainty = unavailable
            && findings.iter().all(|item| {
                matches!(
                    item.code,
                    ResidualFindingCode::EvidenceUnavailable
                        | ResidualFindingCode::ReceiptDoesNotSupportEdit
                )
            });
        return empty_report(
            scenario,
            trace,
            if only_uncertainty {
                ResidualQualification::Inconclusive
            } else {
                ResidualQualification::NotQualified
            },
            findings,
        );
    }

    let baseline_metrics = metrics(&oracle.ground_truth, &baseline.graph);
    let final_metrics = metrics(&oracle.ground_truth, &trace.final_graph);
    let suite_ceiling = scenario
        .algorithm_portfolio
        .iter()
        .map(|output| metrics(&oracle.ground_truth, &output.graph).adjacency_f1)
        .fold(0.0_f64, f64::max);
    let needed = needed_edits(&baseline.graph, &oracle.ground_truth);
    let correct_edits = trace
        .edits
        .iter()
        .filter(|edit| needed.contains(&edit.operation))
        .count();

    let all_algorithm_edges: BTreeSet<ResidualDirectedEdge> = scenario
        .algorithm_portfolio
        .iter()
        .flat_map(|output| output.graph.edges.iter().cloned())
        .collect();
    let novel_truth: BTreeSet<ResidualDirectedEdge> = oracle
        .ground_truth
        .edges
        .difference(&all_algorithm_edges)
        .cloned()
        .collect();
    let novel_recovered = trace.final_graph.edges.intersection(&novel_truth).count();

    let headroom = 1.0 - baseline_metrics.adjacency_f1;
    let gain = final_metrics.adjacency_f1 - baseline_metrics.adjacency_f1;
    ResidualDiscoveryReport {
        qualification: ResidualQualification::Qualified,
        mechanism_family: scenario.mechanism_family.clone(),
        selected_baseline_algorithm_id: scenario.calibration_selected_baseline_id.clone(),
        baseline_metrics: Some(baseline_metrics),
        final_metrics: Some(final_metrics),
        ex_post_algorithm_suite_ceiling_f1: Some(suite_ceiling),
        residual_adjacency_f1_gain: Some(gain),
        residual_directed_f1_gain: Some(final_metrics.directed_f1 - baseline_metrics.directed_f1),
        adjacency_headroom_to_truth: Some(headroom),
        adjacency_headroom_closed_ratio: Some(if headroom > 0.0 { gain / headroom } else { 0.0 }),
        edit_count: trace.edits.len(),
        correct_edit_count: correct_edits,
        edit_precision: Some(ratio(correct_edits, trace.edits.len())),
        edit_recall: Some(ratio(correct_edits, needed.len())),
        evidence_backed_edit_count: backed_edits,
        evidence_backed_edit_rate: Some(ratio(backed_edits, trace.edits.len())),
        novel_true_edge_count: novel_recovered,
        total_novel_true_edge_count: novel_truth.len(),
        novel_true_edge_recall: Some(ratio(novel_recovered, novel_truth.len())),
        findings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(cause: &str, effect: &str) -> ResidualDirectedEdge {
        ResidualDirectedEdge {
            cause: cause.into(),
            effect: effect.into(),
        }
    }

    fn graph(edges: Vec<ResidualDirectedEdge>) -> CausalGraph {
        CausalGraph::new(["A", "B", "C", "D"].into_iter().map(str::to_owned), edges)
    }

    fn fixture() -> (
        ResidualDiscoveryScenario,
        ResidualDiscoveryOracle,
        ResidualDiscoveryTrace,
    ) {
        let baseline = graph(vec![edge("A", "B"), edge("B", "C")]);
        let scenario = ResidualDiscoveryScenario {
            declared_nodes: ["A", "B", "C", "D"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            require_dag: true,
            calibration_selected_baseline_id: "pc-calibrated".into(),
            calibration_split_digest: "calibration-split-sha256".into(),
            evaluation_split_digest: "sealed-evaluation-split-sha256".into(),
            algorithm_registry_digest: "algorithm-registry-sha256".into(),
            algorithm_portfolio: vec![
                AlgorithmGraphOutput {
                    algorithm_id: "pc-calibrated".into(),
                    graph: baseline,
                },
                AlgorithmGraphOutput {
                    algorithm_id: "ges".into(),
                    graph: graph(vec![edge("A", "B")]),
                },
            ],
            available_evidence: [(
                "r-cd".into(),
                AgentEvidenceReceipt {
                    receipt_id: "r-cd".into(),
                    kind: EvidenceKind::InterventionContrast,
                    description: "Independent intervention contrast for C and D".into(),
                },
            )]
            .into(),
            max_edge_edits: 2,
            mechanism_family: "heteroscedastic".into(),
        };
        let oracle = ResidualDiscoveryOracle {
            ground_truth: graph(vec![edge("A", "B"), edge("B", "C"), edge("C", "D")]),
            baseline_selection: BaselineSelectionOracle {
                selected_algorithm_id: "pc-calibrated".into(),
                calibration_split_digest: "calibration-split-sha256".into(),
                evaluation_split_digest: "sealed-evaluation-split-sha256".into(),
                algorithm_registry_digest: "algorithm-registry-sha256".into(),
            },
            evidence: [(
                "r-cd".into(),
                EvidenceReceiptOracle {
                    receipt_id: "r-cd".into(),
                    status: EvidenceReceiptStatus::Verified,
                    supported_additions: [edge("C", "D")].into(),
                    supported_removals: BTreeSet::new(),
                },
            )]
            .into(),
        };
        let trace = ResidualDiscoveryTrace {
            selected_baseline_algorithm_id: "pc-calibrated".into(),
            final_graph: graph(vec![edge("A", "B"), edge("B", "C"), edge("C", "D")]),
            edits: vec![EvidenceBackedEdgeEdit {
                operation: EdgeEditOperation::Add {
                    edge: edge("C", "D"),
                },
                evidence_receipt_ids: vec!["r-cd".into()],
            }],
        };
        (scenario, oracle, trace)
    }

    #[test]
    fn evidence_backed_novel_edge_improves_over_frozen_baseline() {
        let (scenario, oracle, trace) = fixture();
        let report = evaluate_algorithmic_residual(&scenario, &oracle, &trace);
        assert!(report.is_qualified(), "{:?}", report.findings);
        assert!(report.improved_over_frozen_baseline());
        assert!((report.residual_adjacency_f1_gain.unwrap() - 0.2).abs() < 1e-12);
        assert_eq!(report.novel_true_edge_count, 1);
        assert_eq!(report.total_novel_true_edge_count, 1);
        assert_eq!(report.evidence_backed_edit_rate, Some(1.0));
        assert_eq!(report.edit_precision, Some(1.0));
    }

    #[test]
    fn missing_or_unknown_evidence_is_rejected() {
        let (scenario, oracle, mut trace) = fixture();
        trace.edits[0].evidence_receipt_ids.clear();
        let missing = evaluate_algorithmic_residual(&scenario, &oracle, &trace);
        assert_eq!(missing.qualification, ResidualQualification::NotQualified);
        assert!(
            missing
                .findings
                .iter()
                .any(|f| f.code == ResidualFindingCode::MissingEvidenceReceipt)
        );

        trace.edits[0].evidence_receipt_ids = vec!["unknown".into()];
        let unknown = evaluate_algorithmic_residual(&scenario, &oracle, &trace);
        assert!(
            unknown
                .findings
                .iter()
                .any(|f| f.code == ResidualFindingCode::UnknownEvidenceReceipt)
        );
    }

    #[test]
    fn rejected_evidence_cannot_authorize_an_edit() {
        let (scenario, mut oracle, trace) = fixture();
        oracle.evidence.get_mut("r-cd").unwrap().status = EvidenceReceiptStatus::Rejected;
        let report = evaluate_algorithmic_residual(&scenario, &oracle, &trace);
        assert_eq!(report.qualification, ResidualQualification::NotQualified);
        assert!(
            report
                .findings
                .iter()
                .any(|f| f.code == ResidualFindingCode::RejectedEvidenceReceipt)
        );
    }

    #[test]
    fn unavailable_evidence_is_inconclusive() {
        let (scenario, mut oracle, trace) = fixture();
        oracle.evidence.get_mut("r-cd").unwrap().status = EvidenceReceiptStatus::Unavailable;
        let report = evaluate_algorithmic_residual(&scenario, &oracle, &trace);
        assert_eq!(report.qualification, ResidualQualification::Inconclusive);
    }

    #[test]
    fn receipt_must_support_the_exact_edge_operation() {
        let (scenario, mut oracle, trace) = fixture();
        oracle
            .evidence
            .get_mut("r-cd")
            .unwrap()
            .supported_additions
            .clear();
        let report = evaluate_algorithmic_residual(&scenario, &oracle, &trace);
        assert_eq!(report.qualification, ResidualQualification::NotQualified);
        assert!(
            report
                .findings
                .iter()
                .any(|f| f.code == ResidualFindingCode::ReceiptDoesNotSupportEdit)
        );
    }

    #[test]
    fn selected_baseline_must_match_the_calibration_frozen_method() {
        let (scenario, oracle, mut trace) = fixture();
        trace.selected_baseline_algorithm_id = "ges".into();
        let report = evaluate_algorithmic_residual(&scenario, &oracle, &trace);
        assert_eq!(report.qualification, ResidualQualification::NotQualified);
        assert!(
            report
                .findings
                .iter()
                .any(|f| f.code == ResidualFindingCode::FrozenBaselineMissing)
        );
    }

    #[test]
    fn calibration_and_evaluation_splits_must_be_distinct_and_authenticated() {
        let (mut scenario, oracle, trace) = fixture();
        scenario.evaluation_split_digest = scenario.calibration_split_digest.clone();
        let report = evaluate_algorithmic_residual(&scenario, &oracle, &trace);
        assert_eq!(report.qualification, ResidualQualification::NotQualified);
        assert!(
            report
                .findings
                .iter()
                .any(|f| { f.code == ResidualFindingCode::CalibrationEvaluationSplitReuse })
        );
    }

    #[test]
    fn final_graph_must_equal_replayed_edits() {
        let (scenario, oracle, mut trace) = fixture();
        trace.final_graph.edges.remove(&edge("C", "D"));
        let report = evaluate_algorithmic_residual(&scenario, &oracle, &trace);
        assert_eq!(report.qualification, ResidualQualification::NotQualified);
        assert!(
            report
                .findings
                .iter()
                .any(|f| f.code == ResidualFindingCode::FinalEdgeSetDiffersFromReplay)
        );
    }

    #[test]
    fn dag_task_rejects_cycles_even_with_valid_evidence() {
        let (mut scenario, mut oracle, mut trace) = fixture();
        scenario.available_evidence.insert(
            "r-da".into(),
            AgentEvidenceReceipt {
                receipt_id: "r-da".into(),
                kind: EvidenceKind::InterventionContrast,
                description: "Evidence for a candidate edge".into(),
            },
        );
        oracle.evidence.insert(
            "r-da".into(),
            EvidenceReceiptOracle {
                receipt_id: "r-da".into(),
                status: EvidenceReceiptStatus::Verified,
                supported_additions: [edge("D", "A")].into(),
                supported_removals: BTreeSet::new(),
            },
        );
        trace.edits.push(EvidenceBackedEdgeEdit {
            operation: EdgeEditOperation::Add {
                edge: edge("D", "A"),
            },
            evidence_receipt_ids: vec!["r-da".into()],
        });
        trace.final_graph.edges.insert(edge("D", "A"));
        let report = evaluate_algorithmic_residual(&scenario, &oracle, &trace);
        assert_eq!(report.qualification, ResidualQualification::NotQualified);
        assert!(
            report
                .findings
                .iter()
                .any(|f| f.code == ResidualFindingCode::FinalGraphInvalid)
        );
    }

    #[test]
    fn false_but_evidence_backed_edge_is_scored_as_an_error() {
        let (mut scenario, mut oracle, mut trace) = fixture();
        scenario.available_evidence.insert(
            "r-ad".into(),
            AgentEvidenceReceipt {
                receipt_id: "r-ad".into(),
                kind: EvidenceKind::MechanismDiagnostic,
                description: "Plausible but non-decisive diagnostic".into(),
            },
        );
        oracle.evidence.insert(
            "r-ad".into(),
            EvidenceReceiptOracle {
                receipt_id: "r-ad".into(),
                status: EvidenceReceiptStatus::Verified,
                supported_additions: [edge("A", "D")].into(),
                supported_removals: BTreeSet::new(),
            },
        );
        trace.edits.push(EvidenceBackedEdgeEdit {
            operation: EdgeEditOperation::Add {
                edge: edge("A", "D"),
            },
            evidence_receipt_ids: vec!["r-ad".into()],
        });
        trace.final_graph.edges.insert(edge("A", "D"));
        let report = evaluate_algorithmic_residual(&scenario, &oracle, &trace);
        assert!(report.is_qualified(), "{:?}", report.findings);
        assert!(report.final_metrics.unwrap().false_positive_directed_edges > 0);
        assert_eq!(report.evidence_backed_edit_rate, Some(1.0));
        assert!(report.edit_precision.unwrap() < 1.0);
    }

    #[test]
    fn ex_post_algorithm_suite_ceiling_is_separate_from_frozen_baseline() {
        let (mut scenario, oracle, trace) = fixture();
        scenario.algorithm_portfolio.push(AlgorithmGraphOutput {
            algorithm_id: "oracle-like".into(),
            graph: oracle.ground_truth.clone(),
        });
        let report = evaluate_algorithmic_residual(&scenario, &oracle, &trace);
        assert!(report.is_qualified());
        assert_eq!(report.ex_post_algorithm_suite_ceiling_f1, Some(1.0));
        assert_eq!(report.baseline_metrics.unwrap().adjacency_f1, 0.8);
    }

    #[test]
    fn reorientation_is_one_edit_and_changes_orientation_shd() {
        let baseline = graph(vec![edge("A", "B"), edge("B", "C")]);
        let truth = graph(vec![edge("B", "A"), edge("B", "C")]);
        let needed = needed_edits(&baseline, &truth);
        assert!(needed.contains(&EdgeEditOperation::Reorient {
            from: edge("A", "B"),
            to: edge("B", "A"),
        }));
        let score = metrics(&truth, &baseline);
        assert_eq!(score.adjacency_f1, 1.0);
        assert_eq!(score.structural_hamming_distance, 1);
    }

    #[test]
    fn edge_edit_budget_is_non_compensatory() {
        let (mut scenario, oracle, trace) = fixture();
        scenario.max_edge_edits = 0;
        let report = evaluate_algorithmic_residual(&scenario, &oracle, &trace);
        assert_eq!(report.qualification, ResidualQualification::NotQualified);
        assert!(
            report
                .findings
                .iter()
                .any(|f| f.code == ResidualFindingCode::TooManyEdits)
        );
    }
}
