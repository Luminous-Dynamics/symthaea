use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use symthaea_manufacturing_process::ProcessDefinitionId;

const PLAN_DOMAIN: &str = "symthaea-manufacturing-process::process-plan-v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum PlanNodeKindV1 {
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
struct ProcessPlanNodeV1 {
    node_id: String,
    kind: PlanNodeKindV1,
    display_label: Option<String>,
    ui_x: Option<i32>,
    ui_y: Option<i32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
enum PlanEdgeKindV1 {
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
struct ProcessPlanEdgeV1 {
    from: String,
    to: String,
    kind: PlanEdgeKindV1,
    condition_ref: Option<String>,
    max_iterations: Option<u32>,
    disposition_ref: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct ProcessPlanV1 {
    semantic_version: String,
    nodes: Vec<ProcessPlanNodeV1>,
    edges: Vec<ProcessPlanEdgeV1>,
    display_label: Option<String>,
}

impl ProcessPlanV1 {
    fn validate(&self) -> Result<(), &'static str> {
        canonical(&self.semantic_version)?;
        if self.nodes.is_empty() {
            return Err("process plan requires at least one node");
        }

        let mut nodes_by_id = BTreeMap::new();
        for node in &self.nodes {
            canonical(&node.node_id)?;
            if nodes_by_id.insert(node.node_id.as_str(), node).is_some() {
                return Err("duplicate process-plan node id");
            }
            validate_node(node)?;
        }

        let mut edge_keys = BTreeSet::new();
        for edge in &self.edges {
            canonical(&edge.from)?;
            canonical(&edge.to)?;
            if !nodes_by_id.contains_key(edge.from.as_str())
                || !nodes_by_id.contains_key(edge.to.as_str())
            {
                return Err("dangling process-plan edge");
            }
            if edge.from == edge.to {
                return Err("self edge is not admitted");
            }
            validate_edge(edge)?;
            let key = (
                edge.from.as_str(),
                edge.to.as_str(),
                edge.kind,
                edge.condition_ref.as_deref(),
                edge.max_iterations,
                edge.disposition_ref.as_deref(),
            );
            if !edge_keys.insert(key) {
                return Err("duplicate process-plan edge");
            }
        }

        self.validate_mainline_dag(&nodes_by_id)?;
        self.validate_join_dependencies(&nodes_by_id)?;
        self.validate_reachability(&nodes_by_id)?;
        Ok(())
    }

    fn validate_mainline_dag(
        &self,
        nodes_by_id: &BTreeMap<&str, &ProcessPlanNodeV1>,
    ) -> Result<(), &'static str> {
        let mut indegree: BTreeMap<&str, usize> =
            nodes_by_id.keys().map(|id| (*id, 0_usize)).collect();
        let mut outgoing: BTreeMap<&str, Vec<&str>> = BTreeMap::new();

        for edge in self
            .edges
            .iter()
            .filter(|edge| edge.kind != PlanEdgeKindV1::Rework)
        {
            *indegree.get_mut(edge.to.as_str()).unwrap() += 1;
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
            return Err("v1 process plan requires exactly one mainline start node");
        }

        let mut queue = VecDeque::from(starts);
        let mut visited = 0_usize;
        while let Some(node) = queue.pop_front() {
            visited += 1;
            if let Some(nexts) = outgoing.get(node) {
                for next in nexts {
                    let degree = indegree.get_mut(next).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(next);
                    }
                }
            }
        }
        if visited != nodes_by_id.len() {
            return Err("ordinary process-plan flow must be acyclic");
        }

        let finals: Vec<&str> = nodes_by_id
            .keys()
            .copied()
            .filter(|id| {
                !self.edges.iter().any(|edge| {
                    edge.kind != PlanEdgeKindV1::Rework && edge.from.as_str() == *id
                })
            })
            .collect();
        if finals.len() != 1 {
            return Err("v1 process plan requires exactly one mainline final node");
        }
        Ok(())
    }

    fn validate_join_dependencies(
        &self,
        nodes_by_id: &BTreeMap<&str, &ProcessPlanNodeV1>,
    ) -> Result<(), &'static str> {
        for (id, node) in nodes_by_id {
            if matches!(&node.kind, PlanNodeKindV1::AssemblyJoin { .. }) {
                let incoming = self
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
                    return Err("assembly join requires at least two explicit incoming dependencies");
                }
            }
        }
        Ok(())
    }

    fn validate_reachability(
        &self,
        nodes_by_id: &BTreeMap<&str, &ProcessPlanNodeV1>,
    ) -> Result<(), &'static str> {
        let mut incoming = BTreeMap::new();
        for id in nodes_by_id.keys() {
            incoming.insert(*id, 0_usize);
        }
        for edge in self
            .edges
            .iter()
            .filter(|edge| edge.kind != PlanEdgeKindV1::Rework)
        {
            *incoming.get_mut(edge.to.as_str()).unwrap() += 1;
        }
        let start = *incoming
            .iter()
            .find_map(|(id, degree)| (*degree == 0).then_some(id))
            .ok_or("missing mainline start")?;

        let mut seen = BTreeSet::new();
        let mut queue = VecDeque::from([start]);
        while let Some(id) = queue.pop_front() {
            if !seen.insert(id) {
                continue;
            }
            for edge in self.edges.iter().filter(|edge| {
                edge.kind != PlanEdgeKindV1::Rework && edge.from.as_str() == id
            }) {
                queue.push_back(edge.to.as_str());
            }
        }
        if seen.len() != nodes_by_id.len() {
            return Err("required process-plan node is unreachable from start");
        }
        Ok(())
    }

    fn plan_id(&self) -> Result<String, &'static str> {
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
        hash_field(&mut hasher, &serde_json::to_string(&nodes).unwrap());
        hash_field(&mut hasher, &serde_json::to_string(&edges).unwrap());
        Ok(hasher.finalize().to_hex().to_string())
    }
}

fn validate_node(node: &ProcessPlanNodeV1) -> Result<(), &'static str> {
    match &node.kind {
        PlanNodeKindV1::ProcessStep {
            process_id,
            recipe_or_commitment_ref,
            capability_requirement_ref,
            input_state_refs,
            output_state_refs,
        } => {
            canonical(&process_id.0)?;
            canonical(capability_requirement_ref)?;
            if let Some(recipe) = recipe_or_commitment_ref {
                canonical(recipe)?;
            }
            validate_ref_set(input_state_refs, "process step requires input state")?;
            validate_ref_set(output_state_refs, "process step requires output state")?;
        }
        PlanNodeKindV1::Inspection {
            inspection_profile_ref,
            state_ref,
        } => {
            canonical(inspection_profile_ref)?;
            canonical(state_ref)?;
        }
        PlanNodeKindV1::HoldPoint {
            required_evidence_profile_ref,
        } => canonical(required_evidence_profile_ref)?,
        PlanNodeKindV1::AssemblyJoin { output_state_ref } => canonical(output_state_ref)?,
        PlanNodeKindV1::ExternalProvider {
            process_id,
            capability_requirement_ref,
            expected_output_state_ref,
        } => {
            canonical(&process_id.0)?;
            canonical(capability_requirement_ref)?;
            canonical(expected_output_state_ref)?;
        }
        PlanNodeKindV1::MaterialHandling {
            input_state_ref,
            output_state_ref,
        } => {
            canonical(input_state_ref)?;
            canonical(output_state_ref)?;
        }
    }
    Ok(())
}

fn validate_edge(edge: &ProcessPlanEdgeV1) -> Result<(), &'static str> {
    if let Some(condition) = &edge.condition_ref {
        canonical(condition)?;
    }
    if let Some(disposition) = &edge.disposition_ref {
        canonical(disposition)?;
    }
    match edge.kind {
        PlanEdgeKindV1::Rework => {
            if edge.condition_ref.is_none() {
                return Err("rework edge requires explicit condition ref");
            }
            if edge.max_iterations.unwrap_or(0) == 0 {
                return Err("rework edge requires positive bounded max_iterations");
            }
            if edge.disposition_ref.is_none() {
                return Err("rework edge requires exhaustion disposition ref");
            }
        }
        PlanEdgeKindV1::ConditionalAccept | PlanEdgeKindV1::ConditionalReject => {
            if edge.condition_ref.is_none() {
                return Err("conditional edge requires condition ref");
            }
            if edge.max_iterations.is_some() || edge.disposition_ref.is_some() {
                return Err("conditional edge cannot carry rework bounds/disposition");
            }
        }
        PlanEdgeKindV1::HoldRelease => {
            if edge.condition_ref.is_none() {
                return Err("hold release requires explicit evidence/approval condition ref");
            }
            if edge.max_iterations.is_some() || edge.disposition_ref.is_some() {
                return Err("hold release cannot carry rework bounds/disposition");
            }
        }
        _ => {
            if edge.max_iterations.is_some() || edge.disposition_ref.is_some() {
                return Err("non-rework edge cannot carry rework bounds/disposition");
            }
        }
    }
    Ok(())
}

fn validate_ref_set(values: &[String], empty_error: &'static str) -> Result<(), &'static str> {
    if values.is_empty() {
        return Err(empty_error);
    }
    let mut seen = BTreeSet::new();
    for value in values {
        canonical(value)?;
        if !seen.insert(value) {
            return Err("duplicate semantic reference");
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

fn canonical(value: &str) -> Result<(), &'static str> {
    if value.is_empty() || value.trim() != value || value.chars().any(char::is_control) {
        return Err("non-canonical reference");
    }
    Ok(())
}

fn hash_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn process_node(id: &str, input: &str, output: &str) -> ProcessPlanNodeV1 {
    ProcessPlanNodeV1 {
        node_id: id.into(),
        kind: PlanNodeKindV1::ProcessStep {
            process_id: ProcessDefinitionId(format!("process:{id}")),
            recipe_or_commitment_ref: Some(format!("mfg:recipe:{id}")),
            capability_requirement_ref: format!("mfg:capability:{id}"),
            input_state_refs: vec![input.into()],
            output_state_refs: vec![output.into()],
        },
        display_label: Some(id.into()),
        ui_x: Some(0),
        ui_y: Some(0),
    }
}

fn edge(from: &str, to: &str, kind: PlanEdgeKindV1) -> ProcessPlanEdgeV1 {
    ProcessPlanEdgeV1 {
        from: from.into(),
        to: to.into(),
        kind,
        condition_ref: None,
        max_iterations: None,
        disposition_ref: None,
    }
}

fn linear_fixture() -> ProcessPlanV1 {
    ProcessPlanV1 {
        semantic_version: "1".into(),
        nodes: vec![
            process_node("rough-machine", "state:stock", "state:rough"),
            ProcessPlanNodeV1 {
                node_id: "inspect".into(),
                kind: PlanNodeKindV1::Inspection {
                    inspection_profile_ref: "qif:inspection:dimensional-v1".into(),
                    state_ref: "state:rough".into(),
                },
                display_label: Some("Dimensional inspection".into()),
                ui_x: Some(10),
                ui_y: Some(20),
            },
            process_node("finish-machine", "state:rough-accepted", "state:finished"),
        ],
        edges: vec![
            edge("rough-machine", "inspect", PlanEdgeKindV1::Normal),
            ProcessPlanEdgeV1 {
                from: "inspect".into(),
                to: "finish-machine".into(),
                kind: PlanEdgeKindV1::ConditionalAccept,
                condition_ref: Some("quality:condition:inspection-pass".into()),
                max_iterations: None,
                disposition_ref: None,
            },
        ],
        display_label: Some("Reference routing".into()),
    }
}

#[test]
fn linear_plan_validates() {
    assert!(linear_fixture().validate().is_ok());
}

#[test]
fn parallel_branches_with_explicit_join_validate() {
    let mut plan = ProcessPlanV1 {
        semantic_version: "1".into(),
        nodes: vec![
            process_node("prepare", "state:raw", "state:prepared"),
            process_node("housing", "state:prepared", "state:housing"),
            process_node("electronics", "state:prepared", "state:electronics"),
            ProcessPlanNodeV1 {
                node_id: "join".into(),
                kind: PlanNodeKindV1::AssemblyJoin {
                    output_state_ref: "state:assembled".into(),
                },
                display_label: None,
                ui_x: None,
                ui_y: None,
            },
            process_node("final-test", "state:assembled", "state:accepted"),
        ],
        edges: vec![
            edge("prepare", "housing", PlanEdgeKindV1::ParallelBranch),
            edge("prepare", "electronics", PlanEdgeKindV1::ParallelBranch),
            edge("housing", "join", PlanEdgeKindV1::JoinDependency),
            edge("electronics", "join", PlanEdgeKindV1::JoinDependency),
            edge("join", "final-test", PlanEdgeKindV1::Normal),
        ],
        display_label: None,
    };
    assert!(plan.validate().is_ok());

    plan.edges
        .retain(|e| !(e.from == "electronics" && e.to == "join"));
    assert!(plan.validate().is_err());
}

#[test]
fn dangling_edge_rejects() {
    let mut plan = linear_fixture();
    plan.edges
        .push(edge("inspect", "missing", PlanEdgeKindV1::Normal));
    assert!(plan.validate().is_err());
}

#[test]
fn ordinary_cycle_rejects() {
    let mut plan = linear_fixture();
    plan.edges.push(edge(
        "finish-machine",
        "rough-machine",
        PlanEdgeKindV1::Normal,
    ));
    assert!(plan.validate().is_err());
}

#[test]
fn bounded_explicit_rework_validates_but_unbounded_rework_rejects() {
    let mut plan = linear_fixture();
    plan.edges.push(ProcessPlanEdgeV1 {
        from: "inspect".into(),
        to: "rough-machine".into(),
        kind: PlanEdgeKindV1::Rework,
        condition_ref: Some("quality:condition:inspection-reject".into()),
        max_iterations: Some(2),
        disposition_ref: Some("quality:disposition:hold-after-rework-exhaustion".into()),
    });
    assert!(plan.validate().is_ok());

    plan.edges.last_mut().unwrap().max_iterations = None;
    assert!(plan.validate().is_err());
}

#[test]
fn hold_release_requires_explicit_evidence_condition() {
    let mut plan = linear_fixture();
    plan.nodes.insert(
        2,
        ProcessPlanNodeV1 {
            node_id: "hold".into(),
            kind: PlanNodeKindV1::HoldPoint {
                required_evidence_profile_ref: "quality:hold:release-profile-v1".into(),
            },
            display_label: None,
            ui_x: None,
            ui_y: None,
        },
    );
    plan.edges
        .retain(|edge| !(edge.from == "inspect" && edge.to == "finish-machine"));
    plan.edges.push(ProcessPlanEdgeV1 {
        from: "inspect".into(),
        to: "hold".into(),
        kind: PlanEdgeKindV1::ConditionalAccept,
        condition_ref: Some("quality:condition:inspection-pass".into()),
        max_iterations: None,
        disposition_ref: None,
    });
    plan.edges.push(ProcessPlanEdgeV1 {
        from: "hold".into(),
        to: "finish-machine".into(),
        kind: PlanEdgeKindV1::HoldRelease,
        condition_ref: Some("quality:evidence:release-approved".into()),
        max_iterations: None,
        disposition_ref: None,
    });
    assert!(plan.validate().is_ok());

    plan.edges.last_mut().unwrap().condition_ref = None;
    assert!(plan.validate().is_err());
}

#[test]
fn ui_metadata_does_not_change_plan_identity() {
    let a = linear_fixture();
    let mut b = a.clone();
    b.display_label = Some("Renamed route".into());
    for (index, node) in b.nodes.iter_mut().enumerate() {
        node.display_label = Some(format!("renamed-{index}"));
        node.ui_x = Some(index as i32 * 100);
        node.ui_y = Some(index as i32 * -50);
    }
    assert_eq!(a.plan_id().unwrap(), b.plan_id().unwrap());
}

#[test]
fn semantic_changes_change_plan_identity() {
    let a = linear_fixture();
    let mut b = a.clone();
    if let PlanNodeKindV1::ProcessStep { process_id, .. } = &mut b.nodes[0].kind {
        *process_id = ProcessDefinitionId("process:alternate-rough-machine".into());
    }
    assert_ne!(a.plan_id().unwrap(), b.plan_id().unwrap());
}

#[test]
fn serde_round_trip_preserves_graph_identity() {
    let plan = linear_fixture();
    let encoded = serde_json::to_string(&plan).unwrap();
    let decoded: ProcessPlanV1 = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, plan);
    assert_eq!(decoded.plan_id().unwrap(), plan.plan_id().unwrap());
}
