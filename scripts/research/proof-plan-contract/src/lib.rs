//! Structural validator for the versioned ProofPlanV1 research contract.
//!
//! This validates JSON shape, graph references, evidence references, action
//! coverage, and status/evidence consistency. It does not validate mathematics,
//! run Lean, or certify the theorem.

use serde::Deserialize;
use std::collections::{HashMap, HashSet};

#[cfg(test)]
const AND_COMM_FIXTURE: &str =
    include_str!("../../../../docs/research/PROOF_PLAN_AND_COMM_V1.json");
#[cfg(test)]
const MUTATION_FIXTURES: &str =
    include_str!("../../../../docs/research/PROOF_PLAN_MUTATION_FIXTURES_V1.json");

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProofPlan {
    pub schema_version: String,
    pub plan_id: String,
    pub status: PlanStatus,
    pub theorem: TheoremIdentity,
    pub environment: EnvironmentIdentity,
    pub root_goal_id: String,
    pub nodes: Vec<PlanNode>,
    pub edges: Vec<PlanEdge>,
    pub evidence: Vec<EvidenceRecord>,
    pub alternatives: Vec<PlanAlternative>,
    pub validation: ValidationState,
    pub reviewers: Vec<Reviewer>,
    pub limitations: Vec<String>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlanStatus {
    SourcePinnedProvisional,
    InProgress,
    Reviewed,
    Accepted,
    Rejected,
    Inconclusive,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TheoremIdentity {
    pub declaration: String,
    pub statement: String,
    pub identity_status: IdentityStatus,
    pub source: SourceSpan,
    pub elaborated_type_sha256: Option<String>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IdentityStatus {
    SourcePinnedProvisional,
    ElaboratedTypeBound,
    KernelChecked,
    Unresolved,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvironmentIdentity {
    pub status: EnvironmentStatus,
    pub lean_version: Option<String>,
    pub mathlib_revision: Option<String>,
    pub lake_manifest_sha256: Option<String>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EnvironmentStatus {
    SourceOnly,
    Built,
    KernelChecked,
    Unresolved,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceSpan {
    pub repository: String,
    pub revision: String,
    pub path: String,
    pub blob_sha: String,
    pub line_start: u64,
    pub line_end: u64,
    pub url: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanNode {
    pub id: String,
    pub kind: NodeKind,
    pub label: String,
    pub statement: Option<String>,
    pub epistemic_class: EpistemicClass,
    pub evidence_refs: Vec<String>,
    pub action_spec: Option<ActionSpec>,
    pub analogy_boundary: Option<String>,
    pub unresolved_reason: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    GoalState,
    PlanAction,
    Justification,
    SideCondition,
    Subgoal,
    WitnessOrConstruction,
    Invariant,
    TerminalClaim,
    UnresolvedStep,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EpistemicClass {
    FormalFact,
    MathematicalInterpretation,
    Analogy,
    Unknown,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionSpec {
    pub operation: ActionKind,
    pub input_goal_ids: Vec<String>,
    pub output_subgoal_ids: Vec<String>,
    pub consumed_side_condition_ids: Vec<String>,
    pub justification_node_ids: Vec<String>,
    pub rationale: String,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionKind {
    SplitIff,
    Introduce,
    Specialize,
    ApplyLemma,
    Rewrite,
    Cases,
    Induction,
    ConstructWitness,
    Contradiction,
    Bound,
    Normalize,
    Transport,
    Reduce,
    Compose,
    Other,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanEdge {
    pub source_id: String,
    pub target_id: String,
    pub relation: EdgeRelation,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EdgeRelation {
    Transforms,
    JustifiedBy,
    Requires,
    CreatesSubgoal,
    Preserves,
    Discharges,
    Constructs,
    AlternativeTo,
    EvidencedBy,
    UnknownBecause,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceRecord {
    pub id: String,
    pub kind: EvidenceKind,
    pub authority: EvidenceAuthority,
    pub status: EvidenceStatus,
    pub description: String,
    pub source: Option<SourceSpan>,
    pub sha256: Option<String>,
    pub checker: Option<String>,
    pub checker_version: Option<String>,
    pub verdict: Option<CheckerVerdict>,
}

#[derive(Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    SourceSpan,
    DeclarationMetadata,
    ElaboratedType,
    ElaboratedValue,
    LeanKernelCheck,
    IndependentChecker,
    FiniteWitness,
    HumanReview,
    ExternalGraph,
    Other,
}

#[derive(Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceAuthority {
    RepositorySourceInspection,
    LeanEnvironmentExtractor,
    LeanKernel,
    IndependentChecker,
    FiniteReferenceOracle,
    MathematicalReviewer,
    ExternalDataset,
    Other,
}

#[derive(Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceStatus {
    Observed,
    NotRun,
    Passed,
    Rejected,
    Inconclusive,
}

#[derive(Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckerVerdict {
    Accepted,
    Rejected,
    Error,
    Timeout,
    Inconclusive,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanAlternative {
    pub id: String,
    pub label: String,
    pub root_goal_id: String,
    pub node_ids: Vec<String>,
    pub status: AlternativeStatus,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlternativeStatus {
    Primary,
    Candidate,
    Rejected,
    Unknown,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidationState {
    pub graph_structure_status: GraphStatus,
    pub lean_kernel_status: KernelStatus,
    pub expert_review_status: ReviewStatus,
    pub human_comprehension_status: HumanStatus,
    pub validator_version: Option<String>,
}

#[derive(Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphStatus {
    NotRun,
    Passed,
    Failed,
    Inconclusive,
}

#[derive(Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelStatus {
    NotRun,
    Accepted,
    Rejected,
    Error,
    Timeout,
    Inconclusive,
}

#[derive(Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewStatus {
    NotReviewed,
    Accepted,
    Rejected,
    Disputed,
    Inconclusive,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HumanStatus {
    NotRun,
    InProgress,
    Complete,
    Inconclusive,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reviewer {
    pub reviewer_id: String,
    pub role: String,
    pub disposition: ReviewerDisposition,
    pub rationale: Option<String>,
}

#[derive(Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewerDisposition {
    NotReviewed,
    Accepted,
    Rejected,
    Disputed,
    Inconclusive,
}

pub fn parse_plan(input: &str) -> Result<ProofPlan, String> {
    serde_json::from_str(input).map_err(|err| format!("invalid ProofPlanV1 JSON: {err}"))
}

pub fn validate_plan(plan: &ProofPlan) -> Vec<String> {
    let mut errors = Vec::new();

    if plan.schema_version != "1.0.0" {
        errors.push(format!(
            "unsupported schema_version: {}",
            plan.schema_version
        ));
    }
    if plan.plan_id.trim().is_empty() {
        errors.push("plan_id must not be empty".to_owned());
    }
    if plan.theorem.declaration.trim().is_empty() || plan.theorem.statement.trim().is_empty() {
        errors.push("theorem declaration and statement must not be empty".to_owned());
    }
    if plan.limitations.is_empty() || plan.limitations.iter().any(|s| s.trim().is_empty()) {
        errors.push("limitations must contain at least one nonempty entry".to_owned());
    }
    if plan.theorem.source.line_start == 0
        || plan.theorem.source.line_end < plan.theorem.source.line_start
    {
        errors.push("theorem source line range is invalid".to_owned());
    }
    if plan.theorem.identity_status == IdentityStatus::KernelChecked
        && (plan.theorem.elaborated_type_sha256.is_none()
            || plan.validation.lean_kernel_status != KernelStatus::Accepted)
    {
        errors.push(
            "kernel_checked theorem identity requires a type digest and accepted kernel status"
                .to_owned(),
        );
    }
    if plan.environment.status == EnvironmentStatus::KernelChecked
        && plan.validation.lean_kernel_status != KernelStatus::Accepted
    {
        errors.push("kernel_checked environment requires accepted kernel status".to_owned());
    }

    let mut node_kinds = HashMap::<String, NodeKind>::new();
    for node in &plan.nodes {
        if node.id.trim().is_empty() {
            errors.push("node ID must not be empty".to_owned());
            continue;
        }
        if node_kinds.insert(node.id.clone(), node.kind).is_some() {
            errors.push(format!("duplicate node ID: {}", node.id));
        }
    }

    let mut evidence_ids = HashSet::<String>::new();
    let mut evidence_by_id = HashMap::<String, &EvidenceRecord>::new();
    for evidence in &plan.evidence {
        if evidence.id.trim().is_empty() {
            errors.push("evidence ID must not be empty".to_owned());
            continue;
        }
        if !evidence_ids.insert(evidence.id.clone()) {
            errors.push(format!("duplicate evidence ID: {}", evidence.id));
        }
        evidence_by_id.insert(evidence.id.clone(), evidence);
        if let Some(source) = &evidence.source {
            if source.line_start == 0 || source.line_end < source.line_start {
                errors.push(format!(
                    "evidence {} has an invalid source range",
                    evidence.id
                ));
            }
        }
        if evidence.kind == EvidenceKind::LeanKernelCheck
            && evidence.status == EvidenceStatus::Passed
            && evidence.verdict != Some(CheckerVerdict::Accepted)
        {
            errors.push(format!(
                "passed Lean kernel evidence {} must have accepted verdict",
                evidence.id
            ));
        }
    }

    let check_evidence_refs = |owner: &str, refs: &[String], errors: &mut Vec<String>| {
        for id in refs {
            if !evidence_ids.contains(id) {
                errors.push(format!("{owner} references missing evidence ID {id}"));
            }
        }
    };

    for node in &plan.nodes {
        check_evidence_refs(
            &format!("node {}", node.id),
            &node.evidence_refs,
            &mut errors,
        );
        match node.epistemic_class {
            EpistemicClass::FormalFact | EpistemicClass::MathematicalInterpretation => {
                if node.evidence_refs.is_empty() {
                    errors.push(format!(
                        "{} requires evidence for its epistemic class",
                        node.id
                    ));
                }
            }
            EpistemicClass::Analogy => {
                if node
                    .analogy_boundary
                    .as_deref()
                    .unwrap_or("")
                    .trim()
                    .is_empty()
                {
                    errors.push(format!(
                        "analogy node {} requires analogy_boundary",
                        node.id
                    ));
                }
            }
            EpistemicClass::Unknown => {
                if node
                    .unresolved_reason
                    .as_deref()
                    .unwrap_or("")
                    .trim()
                    .is_empty()
                {
                    errors.push(format!(
                        "unknown node {} requires unresolved_reason",
                        node.id
                    ));
                }
            }
        }

        if node.kind == NodeKind::PlanAction {
            if let Some(action) = &node.action_spec {
                if action.input_goal_ids.is_empty() {
                    errors.push(format!(
                        "action {} requires at least one input goal",
                        node.id
                    ));
                }
                if action.rationale.trim().is_empty() {
                    errors.push(format!("action {} requires a rationale", node.id));
                }
                if action.evidence_refs.is_empty() {
                    errors.push(format!("action {} requires evidence_refs", node.id));
                }
                check_evidence_refs(
                    &format!("action {}", node.id),
                    &action.evidence_refs,
                    &mut errors,
                );
            } else {
                errors.push(format!(
                    "plan_action node {} is missing action_spec",
                    node.id
                ));
            }
        } else if node.action_spec.is_some() {
            errors.push(format!(
                "non-action node {} must not have action_spec",
                node.id
            ));
        }
    }

    let edge_exists = |source: &str, target: &str, relation: EdgeRelation| {
        plan.edges.iter().any(|edge| {
            edge.source_id == source && edge.target_id == target && edge.relation == relation
        })
    };

    for edge in &plan.edges {
        if !node_kinds.contains_key(&edge.source_id) {
            errors.push(format!("edge has missing source node {}", edge.source_id));
        }
        if !node_kinds.contains_key(&edge.target_id) {
            errors.push(format!("edge has missing target node {}", edge.target_id));
        }
        check_evidence_refs(
            &format!("edge {} -> {}", edge.source_id, edge.target_id),
            &edge.evidence_refs,
            &mut errors,
        );
        if edge.evidence_refs.is_empty() {
            errors.push(format!(
                "edge {} -> {} requires evidence_refs",
                edge.source_id, edge.target_id
            ));
        }
    }

    match node_kinds.get(&plan.root_goal_id) {
        Some(NodeKind::GoalState) => {}
        Some(_) => errors.push(format!(
            "root_goal_id {} is not a goal_state node",
            plan.root_goal_id
        )),
        None => errors.push(format!("root_goal_id {} does not exist", plan.root_goal_id)),
    }

    for node in &plan.nodes {
        let Some(action) = &node.action_spec else {
            continue;
        };

        for goal_id in &action.input_goal_ids {
            match node_kinds.get(goal_id) {
                Some(NodeKind::GoalState | NodeKind::Subgoal) => {}
                Some(_) => errors.push(format!(
                    "action {} input {} is not a goal/subgoal",
                    node.id, goal_id
                )),
                None => errors.push(format!(
                    "action {} input goal {} does not exist",
                    node.id, goal_id
                )),
            }
            if !edge_exists(&node.id, goal_id, EdgeRelation::Discharges)
                && !edge_exists(&node.id, goal_id, EdgeRelation::Transforms)
            {
                errors.push(format!(
                    "action {} input {} lacks a discharges/transforms edge",
                    node.id, goal_id
                ));
            }
        }

        for subgoal_id in &action.output_subgoal_ids {
            if node_kinds.get(subgoal_id) != Some(&NodeKind::Subgoal) {
                errors.push(format!(
                    "action {} output {} is not a subgoal",
                    node.id, subgoal_id
                ));
            }
            if !edge_exists(&node.id, subgoal_id, EdgeRelation::CreatesSubgoal) {
                errors.push(format!(
                    "action {} output {} lacks creates_subgoal edge",
                    node.id, subgoal_id
                ));
            }
        }

        for condition_id in &action.consumed_side_condition_ids {
            if node_kinds.get(condition_id) != Some(&NodeKind::SideCondition) {
                errors.push(format!(
                    "action {} consumed condition {} is not a side_condition",
                    node.id, condition_id
                ));
            }
            if !edge_exists(&node.id, condition_id, EdgeRelation::Requires) {
                errors.push(format!(
                    "action {} condition {} lacks requires edge",
                    node.id, condition_id
                ));
            }
        }

        for justification_id in &action.justification_node_ids {
            if node_kinds.get(justification_id) != Some(&NodeKind::Justification) {
                errors.push(format!(
                    "action {} justification {} is not a justification node",
                    node.id, justification_id
                ));
            }
            if !edge_exists(&node.id, justification_id, EdgeRelation::JustifiedBy) {
                errors.push(format!(
                    "action {} justification {} lacks justified_by edge",
                    node.id, justification_id
                ));
            }
        }
    }

    let mut alternative_ids = HashSet::<String>::new();
    for alternative in &plan.alternatives {
        if alternative.id.trim().is_empty() {
            errors.push("alternative ID must not be empty".to_owned());
        }
        if !alternative_ids.insert(alternative.id.clone()) {
            errors.push(format!("duplicate alternative ID: {}", alternative.id));
        }
        if node_kinds.get(&alternative.root_goal_id) != Some(&NodeKind::GoalState) {
            errors.push(format!(
                "alternative {} root_goal_id is missing or not a goal_state",
                alternative.id
            ));
        }
        if !alternative.node_ids.contains(&alternative.root_goal_id) {
            errors.push(format!(
                "alternative {} node_ids must include its root goal",
                alternative.id
            ));
        }
        for node_id in &alternative.node_ids {
            if !node_kinds.contains_key(node_id) {
                errors.push(format!(
                    "alternative {} references missing node {}",
                    alternative.id, node_id
                ));
            }
        }
    }

    if plan.validation.lean_kernel_status == KernelStatus::Accepted {
        let has_kernel_receipt = evidence_by_id.values().any(|evidence| {
            evidence.kind == EvidenceKind::LeanKernelCheck
                && evidence.status == EvidenceStatus::Passed
                && evidence.verdict == Some(CheckerVerdict::Accepted)
        });
        if !has_kernel_receipt {
            errors.push("lean_kernel_status=accepted requires passed lean_kernel_check evidence with accepted verdict".to_owned());
        }
    }

    let mut reviewer_ids = HashSet::<String>::new();
    for reviewer in &plan.reviewers {
        if reviewer.reviewer_id.trim().is_empty() {
            errors.push("reviewer_id must not be empty".to_owned());
        }
        if !reviewer_ids.insert(reviewer.reviewer_id.clone()) {
            errors.push(format!("duplicate reviewer ID: {}", reviewer.reviewer_id));
        }
    }

    if plan.validation.expert_review_status == ReviewStatus::Accepted {
        let accepted_reviewers = plan
            .reviewers
            .iter()
            .filter(|reviewer| reviewer.disposition == ReviewerDisposition::Accepted)
            .count();
        let distinct_reviewers: HashSet<&str> = plan
            .reviewers
            .iter()
            .filter(|reviewer| reviewer.disposition == ReviewerDisposition::Accepted)
            .map(|reviewer| reviewer.reviewer_id.as_str())
            .collect();
        if accepted_reviewers < 2 || distinct_reviewers.len() < 2 {
            errors.push(
                "expert_review_status=accepted requires two distinct accepted reviewers".to_owned(),
            );
        }
    }

    if plan.status == PlanStatus::Accepted
        && (plan.validation.graph_structure_status != GraphStatus::Passed
            || plan.validation.lean_kernel_status != KernelStatus::Accepted
            || plan.validation.expert_review_status != ReviewStatus::Accepted)
    {
        errors.push("status=accepted requires passed graph validation, accepted Lean kernel, and accepted expert review".to_owned());
    }

    errors
}

pub fn validate_json(input: &str) -> Result<ProofPlan, Vec<String>> {
    let plan = parse_plan(input).map_err(|err| vec![err])?;
    let errors = validate_plan(&plan);
    if errors.is_empty() {
        Ok(plan)
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn fixture_value() -> Value {
        serde_json::from_str(AND_COMM_FIXTURE).expect("valid fixture JSON")
    }

    fn validate_value(value: &Value) -> Vec<String> {
        let input = serde_json::to_string(value).expect("serialize mutated fixture");
        match validate_json(&input) {
            Ok(_) => Vec::new(),
            Err(errors) => errors,
        }
    }

    #[test]
    fn provisional_and_comm_plan_is_structurally_valid() {
        let plan = validate_json(AND_COMM_FIXTURE).expect("valid provisional plan");
        assert_eq!(plan.plan_id, "PROOFPLAN-AND-COMM-001");
        assert_eq!(plan.validation.lean_kernel_status, KernelStatus::NotRun);
        assert_eq!(
            plan.validation.expert_review_status,
            ReviewStatus::NotReviewed
        );
    }

    #[test]
    fn duplicate_node_ids_are_rejected() {
        let mut value = fixture_value();
        let duplicate = value["nodes"][0].clone();
        value["nodes"].as_array_mut().unwrap().push(duplicate);
        assert!(
            validate_value(&value)
                .iter()
                .any(|e| e.contains("duplicate node ID"))
        );
    }

    #[test]
    fn dangling_edge_endpoints_are_rejected() {
        let mut value = fixture_value();
        value["edges"][0]["target_id"] = Value::String("MISSING_GOAL".to_owned());
        assert!(
            validate_value(&value)
                .iter()
                .any(|e| e.contains("missing target node"))
        );
    }

    #[test]
    fn missing_formal_fact_evidence_is_rejected() {
        let mut value = fixture_value();
        value["nodes"][0]["evidence_refs"] = Value::Array(Vec::new());
        assert!(
            validate_value(&value)
                .iter()
                .any(|e| e.contains("requires evidence"))
        );
    }

    #[test]
    fn dangling_evidence_references_are_rejected() {
        let mut value = fixture_value();
        value["nodes"][0]["evidence_refs"] = serde_json::json!(["EVIDENCE_MISSING"]);
        assert!(
            validate_value(&value)
                .iter()
                .any(|e| e.contains("references missing evidence ID"))
        );
    }

    #[test]
    fn analogy_without_a_boundary_is_rejected() {
        let mut value = fixture_value();
        let node = value["nodes"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|node| node["id"] == "J_IFF")
            .unwrap();
        node["epistemic_class"] = Value::String("analogy".to_owned());
        assert!(
            validate_value(&value)
                .iter()
                .any(|e| e.contains("requires analogy_boundary"))
        );
    }

    #[test]
    fn action_output_without_subgoal_edge_is_rejected() {
        let mut value = fixture_value();
        let edges = value["edges"].as_array_mut().unwrap();
        edges.retain(|edge| {
            !(edge["source_id"] == "A_SPLIT_IFF"
                && edge["target_id"] == "G_FORWARD"
                && edge["relation"] == "creates_subgoal")
        });
        assert!(
            validate_value(&value)
                .iter()
                .any(|e| e.contains("lacks creates_subgoal edge"))
        );
    }

    #[test]
    fn missing_justification_edge_is_rejected() {
        let mut value = fixture_value();
        let edges = value["edges"].as_array_mut().unwrap();
        edges.retain(|edge| {
            !(edge["source_id"] == "A_FORWARD"
                && edge["target_id"] == "J_AND_SYMM"
                && edge["relation"] == "justified_by")
        });
        assert!(
            validate_value(&value)
                .iter()
                .any(|e| e.contains("lacks justified_by edge"))
        );
    }

    #[test]
    fn consumed_side_condition_without_requires_edge_is_rejected() {
        let mut value = fixture_value();
        value["nodes"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({
                "id":"S_POSITIVE","kind":"side_condition","label":"positivity",
                "statement":"0 < x","epistemic_class":"mathematical_interpretation",
                "evidence_refs":["E_SOURCE"]
            }));
        let action = value["nodes"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|node| node["id"] == "A_FORWARD")
            .unwrap();
        action["action_spec"]["consumed_side_condition_ids"] = serde_json::json!(["S_POSITIVE"]);
        assert!(
            validate_value(&value)
                .iter()
                .any(|e| e.contains("lacks requires edge"))
        );
    }

    #[test]
    fn accepted_kernel_status_requires_kernel_receipt() {
        let mut value = fixture_value();
        value["validation"]["lean_kernel_status"] = Value::String("accepted".to_owned());
        assert!(
            validate_value(&value)
                .iter()
                .any(|e| e.contains("requires passed lean_kernel_check evidence"))
        );
    }

    #[test]
    fn accepted_plan_requires_kernel_and_independent_review_gates() {
        let mut value = fixture_value();
        value["status"] = Value::String("accepted".to_owned());
        assert!(
            validate_value(&value)
                .iter()
                .any(|e| e.contains("status=accepted requires"))
        );
    }

    #[test]
    fn conceptual_cycles_are_not_rejected_by_structure_validator() {
        let mut value = fixture_value();
        value["edges"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({
                "source_id":"J_IFF","target_id":"J_AND_SYMM","relation":"alternative_to",
                "evidence_refs":["E_SOURCE"]
            }));
        value["edges"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({
                "source_id":"J_AND_SYMM","target_id":"J_IFF","relation":"alternative_to",
                "evidence_refs":["E_SOURCE"]
            }));
        assert!(validate_value(&value).is_empty());
    }

    #[test]
    fn mutation_fixtures_keep_route_change_separate_from_refutation() {
        let fixtures: Value =
            serde_json::from_str(MUTATION_FIXTURES).expect("mutation fixture JSON");
        let cases = fixtures["cases"].as_array().expect("cases array");
        let route_only = cases
            .iter()
            .find(|case| case["id"] == "MUT-ROUTE-ONLY")
            .expect("route-only control");
        assert_eq!(route_only["expected_diagnostic"], "proof_route_changed");
        assert_eq!(route_only["mathematical_disposition"], "not_refuted");
        assert_eq!(route_only["oracle_status"], "not_run");

        let invalid_warrant = cases
            .iter()
            .find(|case| case["id"] == "MUT-AND-LEFT")
            .expect("invalid-warrant control");
        assert_eq!(invalid_warrant["mathematical_disposition"], "unknown");
        assert_eq!(invalid_warrant["oracle_status"], "not_run");
    }

    #[test]
    fn local_global_context_control_is_not_a_theorem_refutation() {
        let fixtures: Value =
            serde_json::from_str(MUTATION_FIXTURES).expect("mutation fixture JSON");
        let cases = fixtures["cases"].as_array().expect("cases array");
        let context_case = cases
            .iter()
            .find(|case| case["id"] == "MUT-LOCAL-GLOBAL-CONTEXT")
            .expect("local/global consistency control");
        assert_eq!(
            context_case["expected_diagnostic"],
            "local_global_inconsistency"
        );
        assert_eq!(context_case["mathematical_disposition"], "unknown");
        assert_eq!(context_case["oracle_status"], "not_run");
        assert!(
            context_case["claim_ceiling"]
                .as_str()
                .unwrap()
                .contains("does not establish")
        );
    }

    #[test]
    fn unknown_json_fields_are_rejected() {
        let mut value = fixture_value();
        value["unexpected_status"] = Value::Bool(true);
        let input = serde_json::to_string(&value).unwrap();
        assert!(parse_plan(&input).unwrap_err().contains("unknown field"));
    }
}

pub mod local_global;
