use crate::{canonical_token, hash_field, ManufacturingContractError};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use symthaea_manufacturing_process::ProcessDefinitionId;

const PLAN_DOMAIN: &str = "symthaea-manufacturing-contracts::process-plan-v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlanNodeKindV1 {
    ProcessStep {
        process_id: ProcessDefinitionId,
        recipe_or_commitment_ref: Option<String>,
        capability_requirement_ref: String,
        input_state_refs: Vec<String>,
        output_state_refs: Vec<String>,
    },
    Inspection {
        inspection_profile_ref: String,
        state_ref: String,
    },
    HoldPoint {
        required_evidence_profile_ref: String,
    },
    AssemblyJoin {
        output_state_ref: String,
    },
    ExternalProvider {
        process_id: ProcessDefinitionId,
        capability_requirement_ref: String,
        expected_output_state_ref: String,
    },
    MaterialHandling {
        input_state_ref: String,
        output_state_ref: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessPlanNodeV1 {
    pub node_id: String,
    pub kind: PlanNodeKindV1,
    pub display_label: Option<String>,
    pub ui_x: Option<i32>,
    pub ui_y: Option<i32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PlanEdgeKindV1 {
    Normal,
    ParallelBranch,
    JoinDependency,
    ConditionalAccept,
    ConditionalReject,
    Rework,
    HoldRelease,
    ExternalTransfer,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessPlanEdgeV1 {
    pub from: String,
    pub to: String,
    pub kind: PlanEdgeKindV1,
    pub condition_ref: Option<String>,
    pub max_iterations: Option<u32>,
    pub disposition_ref: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessPlanV1 {
    pub semantic_version: String,
    pub nodes: Vec<ProcessPlanNodeV1>,
    pub edges: Vec<ProcessPlanEdgeV1>,
    pub display_label: Option<String>,
}

impl ProcessPlanV1 {
    pub fn validate(&self) -> Result<(), ManufacturingContractError> {
        canonical_token("plan.semantic_version", &self.semantic_version)?;
        if self.nodes.is_empty() {
            return Err(ManufacturingContractError::Invalid(
                "process plan requires at least one node",
            ));
        }

        let mut nodes_by_id: BTreeMap<&str, &ProcessPlanNodeV1> = BTreeMap::new();
        for node in &self.nodes {
            canonical_token("plan.node_id", &node.node_id)?;
            if nodes_by_id.insert(node.node_id.as_str(), node).is_some() {
                return Err(ManufacturingContractError::DuplicateReference {
                    field: "plan.nodes",
                    value: node.node_id.clone(),
                });
            }
            validate_node(node)?;
        }

        let mut edge_keys = BTreeSet::new();
        for edge in &self.edges {
            canonical_token("plan.edge.from", &edge.from)?;
            canonical_token("plan.edge.to", &edge.to)?;
            if !nodes_by_id.contains_key(edge.from.as_str())
                || !nodes_by_id.contains_key(edge.to.as_str())
            {
                return Err(ManufacturingContractError::Invalid(
                    "process plan contains dangling edge",
                ));
            }
            if edge.from == edge.to {
                return Err(ManufacturingContractError::Invalid(
                    "self edge is not admitted",
                ));
            }
            validate_edge(edge)?;
            let key = (
                edge.from.clone(),
                edge.to.clone(),
                edge.kind,
                edge.condition_ref.clone(),
                edge.max_iterations,
                edge.disposition_ref.clone(),
            );
            if !edge_keys.insert(key) {
                return Err(ManufacturingContractError::Invalid(
                    "duplicate process-plan edge",
                ));
            }
        }

        validate_mainline_dag(self, &nodes_by_id)?;
        validate_join_dependencies(self, &nodes_by_id)?;
        validate_reachability(self, &nodes_by_id)?;
        Ok(())
    }

    pub fn plan_id(&self) -> Result<String, ManufacturingContractError> {
        self.validate()?;
        let mut nodes = self.nodes.clone();
        nodes.sort_by(|a, b| a.node_id.cmp(&b.node_id));
        for node in &mut nodes {
            node.display_label = None;
            node.ui_x = None;
            node.ui_y = None;
            canonicalize_node_semantics(node);
        }

        let mut edges = self.edges.clone();
        edges.sort_by(|a, b| {
            (
                a.from.as_str(),
                a.to.as_str(),
                a.kind,
                a.condition_ref.as_deref(),
                a.max_iterations,
                a.disposition_ref.as_deref(),
            )
                .cmp(&(
                    b.from.as_str(),
                    b.to.as_str(),
                    b.kind,
                    b.condition_ref.as_deref(),
                    b.max_iterations,
                    b.disposition_ref.as_deref(),
                ))
        });

        let mut hasher = blake3::Hasher::new();
        hash_field(&mut hasher, PLAN_DOMAIN);
        hash_field(&mut hasher, &self.semantic_version);
        hash_field(
            &mut hasher,
            &serde_json::to_string(&nodes).map_err(|_| {
                ManufacturingContractError::Invalid("failed to serialize canonical plan nodes")
            })?,
        );
        hash_field(
            &mut hasher,
            &serde_json::to_string(&edges).map_err(|_| {
                ManufacturingContractError::Invalid("failed to serialize canonical plan edges")
            })?,
        );
        Ok(hasher.finalize().to_hex().to_string())
    }
}

fn validate_node(node: &ProcessPlanNodeV1) -> Result<(), ManufacturingContractError> {
    match &node.kind {
        PlanNodeKindV1::ProcessStep {
            process_id,
            recipe_or_commitment_ref,
            capability_requirement_ref,
            input_state_refs,
            output_state_refs,
        } => {
            canonical_token("plan.process_step.process_id", &process_id.0)?;
            canonical_token(
                "plan.process_step.capability_requirement_ref",
                capability_requirement_ref,
            )?;
            if let Some(recipe) = recipe_or_commitment_ref {
                canonical_token("plan.process_step.recipe_ref", recipe)?;
            }
            validate_ref_set(
                "plan.process_step.input_state_refs",
                input_state_refs,
                "process step requires input state",
            )?;
            validate_ref_set(
                "plan.process_step.output_state_refs",
                output_state_refs,
                "process step requires output state",
            )?;
        }
        PlanNodeKindV1::Inspection {
            inspection_profile_ref,
            state_ref,
        } => {
            canonical_token("plan.inspection.profile_ref", inspection_profile_ref)?;
            canonical_token("plan.inspection.state_ref", state_ref)?;
        }
        PlanNodeKindV1::HoldPoint {
            required_evidence_profile_ref,
        } => canonical_token(
            "plan.hold.required_evidence_profile_ref",
            required_evidence_profile_ref,
        )?,
        PlanNodeKindV1::AssemblyJoin { output_state_ref } => {
            canonical_token("plan.join.output_state_ref", output_state_ref)?
        }
        PlanNodeKindV1::ExternalProvider {
            process_id,
            capability_requirement_ref,
            expected_output_state_ref,
        } => {
            canonical_token("plan.external.process_id", &process_id.0)?;
            canonical_token(
                "plan.external.capability_requirement_ref",
                capability_requirement_ref,
            )?;
            canonical_token(
                "plan.external.expected_output_state_ref",
                expected_output_state_ref,
            )?;
        }
        PlanNodeKindV1::MaterialHandling {
            input_state_ref,
            output_state_ref,
        } => {
            canonical_token("plan.handling.input_state_ref", input_state_ref)?;
            canonical_token("plan.handling.output_state_ref", output_state_ref)?;
        }
    }
    Ok(())
}

fn validate_edge(edge: &ProcessPlanEdgeV1) -> Result<(), ManufacturingContractError> {
    if let Some(condition) = &edge.condition_ref {
        canonical_token("plan.edge.condition_ref", condition)?;
    }
    if let Some(disposition) = &edge.disposition_ref {
        canonical_token("plan.edge.disposition_ref", disposition)?;
    }
    match edge.kind {
        PlanEdgeKindV1::Rework => {
            if edge.condition_ref.is_none() {
                return Err(ManufacturingContractError::Invalid(
                    "rework edge requires explicit condition ref",
                ));
            }
            if edge.max_iterations.unwrap_or(0) == 0 {
                return Err(ManufacturingContractError::Invalid(
                    "rework edge requires positive bounded max_iterations",
                ));
            }
            if edge.disposition_ref.is_none() {
                return Err(ManufacturingContractError::Invalid(
                    "rework edge requires exhaustion disposition ref",
                ));
            }
        }
        PlanEdgeKindV1::ConditionalAccept | PlanEdgeKindV1::ConditionalReject => {
            if edge.condition_ref.is_none() {
                return Err(ManufacturingContractError::Invalid(
                    "conditional edge requires condition ref",
                ));
            }
            if edge.max_iterations.is_some() || edge.disposition_ref.is_some() {
                return Err(ManufacturingContractError::Invalid(
                    "conditional edge cannot carry rework bounds/disposition",
                ));
            }
        }
        PlanEdgeKindV1::HoldRelease => {
            if edge.condition_ref.is_none() {
                return Err(ManufacturingContractError::Invalid(
                    "hold release requires explicit evidence/approval condition ref",
                ));
            }
            if edge.max_iterations.is_some() || edge.disposition_ref.is_some() {
                return Err(ManufacturingContractError::Invalid(
                    "hold release cannot carry rework bounds/disposition",
                ));
            }
        }
        _ => {
            if edge.max_iterations.is_some() || edge.disposition_ref.is_some() {
                return Err(ManufacturingContractError::Invalid(
                    "non-rework edge cannot carry rework bounds/disposition",
                ));
            }
        }
    }
    Ok(())
}

fn validate_mainline_dag(
    plan: &ProcessPlanV1,
    nodes_by_id: &BTreeMap<&str, &ProcessPlanNodeV1>,
) -> Result<(), ManufacturingContractError> {
    let mut indegree: BTreeMap<&str, usize> =
        nodes_by_id.keys().map(|id| (*id, 0_usize)).collect();
    let mut outgoing: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for edge in plan.edges.iter().filter(|edge| edge.kind != PlanEdgeKindV1::Rework) {
        *indegree.get_mut(edge.to.as_str()).expect("validated target") += 1;
        outgoing
            .entry(edge.from.as_str())
            .or_default()
            .push(edge.to.as_str());
    }
    let starts: Vec<&str> = indegree
        .iter()
        .filter_map(|(id, degree)| (*degree == 0).then_some(*id))
        .collect();
    if starts.len() != 1 {
        return Err(ManufacturingContractError::Invalid(
            "v1 process plan requires exactly one mainline start node",
        ));
    }

    let mut queue = VecDeque::from(starts);
    let mut visited = 0_usize;
    while let Some(node) = queue.pop_front() {
        visited += 1;
        if let Some(nexts) = outgoing.get(node) {
            for next in nexts {
                let degree = indegree.get_mut(next).expect("validated target");
                *degree -= 1;
                if *degree == 0 {
                    queue.push_back(next);
                }
            }
        }
    }
    if visited != nodes_by_id.len() {
        return Err(ManufacturingContractError::Invalid(
            "ordinary process-plan flow must be acyclic",
        ));
    }

    let finals = nodes_by_id
        .keys()
        .filter(|id| {
            !plan.edges.iter().any(|edge| {
                edge.kind != PlanEdgeKindV1::Rework && edge.from.as_str() == **id
            })
        })
        .count();
    if finals != 1 {
        return Err(ManufacturingContractError::Invalid(
            "v1 process plan requires exactly one mainline final node",
        ));
    }
    Ok(())
}

fn validate_join_dependencies(
    plan: &ProcessPlanV1,
    nodes_by_id: &BTreeMap<&str, &ProcessPlanNodeV1>,
) -> Result<(), ManufacturingContractError> {
    for (id, node) in nodes_by_id {
        if matches!(&node.kind, PlanNodeKindV1::AssemblyJoin { .. }) {
            let incoming = plan
                .edges
                .iter()
                .filter(|edge| {
                    edge.to.as_str() == *id
                        && matches!(
                            edge.kind,
                            PlanEdgeKindV1::JoinDependency | PlanEdgeKindV1::ParallelBranch
                        )
                })
                .count();
            if incoming < 2 {
                return Err(ManufacturingContractError::Invalid(
                    "assembly join requires at least two explicit incoming dependencies",
                ));
            }
        }
    }
    Ok(())
}

fn validate_reachability(
    plan: &ProcessPlanV1,
    nodes_by_id: &BTreeMap<&str, &ProcessPlanNodeV1>,
) -> Result<(), ManufacturingContractError> {
    let mut incoming: BTreeMap<&str, usize> =
        nodes_by_id.keys().map(|id| (*id, 0_usize)).collect();
    for edge in plan.edges.iter().filter(|edge| edge.kind != PlanEdgeKindV1::Rework) {
        *incoming.get_mut(edge.to.as_str()).expect("validated target") += 1;
    }
    let start = incoming
        .iter()
        .find_map(|(id, degree)| (*degree == 0).then_some(*id))
        .ok_or(ManufacturingContractError::Invalid("missing mainline start"))?;
    let mut seen = BTreeSet::new();
    let mut queue = VecDeque::from([start]);
    while let Some(id) = queue.pop_front() {
        if !seen.insert(id) {
            continue;
        }
        for edge in plan.edges.iter().filter(|edge| {
            edge.kind != PlanEdgeKindV1::Rework && edge.from.as_str() == id
        }) {
            queue.push_back(edge.to.as_str());
        }
    }
    if seen.len() != nodes_by_id.len() {
        return Err(ManufacturingContractError::Invalid(
            "required process-plan node is unreachable from start",
        ));
    }
    Ok(())
}

fn validate_ref_set(
    field: &'static str,
    values: &[String],
    empty_error: &'static str,
) -> Result<(), ManufacturingContractError> {
    if values.is_empty() {
        return Err(ManufacturingContractError::Invalid(empty_error));
    }
    let mut seen = BTreeSet::new();
    for value in values {
        canonical_token(field, value)?;
        if !seen.insert(value.clone()) {
            return Err(ManufacturingContractError::DuplicateReference {
                field,
                value: value.clone(),
            });
        }
    }
    Ok(())
}

fn canonicalize_node_semantics(node: &mut ProcessPlanNodeV1) {
    if let PlanNodeKindV1::ProcessStep {
        input_state_refs,
        output_state_refs,
        ..
    } = &mut node.kind
    {
        input_state_refs.sort();
        output_state_refs.sort();
    }
}
