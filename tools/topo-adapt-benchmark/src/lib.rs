#![forbid(unsafe_code)]

use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub const FIXTURE_JSON: &str =
    include_str!("../fixtures/topo_adapt_fixture_v0_1.json");

#[derive(Clone, Debug, Deserialize)]
pub struct Fixture {
    pub schema_version: String,
    pub graph_semantics: GraphSemantics,
    pub cases: Vec<Episode>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct GraphSemantics {
    pub directed: bool,
    pub edit_budget: usize,
    pub candidate_edges_per_episode: usize,
    pub accounting: Accounting,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Accounting {
    pub accepted_edge_addition_cost_units: u64,
    pub message_budget_bytes: u64,
    pub controller_update_budget: u64,
    pub maximum_active_edges: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Node {
    pub id: u16,
    pub role: String,
    pub supply: Option<u64>,
    pub demand: Option<u64>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Edge {
    pub id: Option<String>,
    pub from: u16,
    pub to: u16,
    pub capacity: u64,
    pub latency: u64,
    pub edit_cost: Option<u64>,
    pub maintenance_cost: u64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Episode {
    pub id: String,
    pub nodes: Vec<Node>,
    pub active_edges: Vec<Edge>,
    pub candidate_edges: Vec<Edge>,
    pub demand_schedule: Vec<DemandSchedule>,
    pub interventions: Vec<Intervention>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DemandSchedule {
    pub time: u64,
    pub sinks: Vec<SinkDemand>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SinkDemand {
    pub node: u16,
    pub demand: u64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Intervention {
    pub time: u64,
    #[serde(rename = "type")]
    pub kind: String,
    pub from: Option<u16>,
    pub to: Option<u16>,
    pub capacity_after: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct FlowResult {
    pub max_flow: u64,
    pub min_cut: u64,
    pub requested_flow: u64,
}

impl Fixture {
    pub fn parse(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| format!("fixture JSON parse failed: {e}"))
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != "topo_adapt_fixture_v0_1" {
            return Err(format!("unsupported fixture schema: {}", self.schema_version));
        }
        if !self.graph_semantics.directed {
            return Err("v0.1 reference evaluator requires directed edges".to_string());
        }
        if self.graph_semantics.edit_budget != 1 {
            return Err("v0.1 fixtures require an exact one-edge edit budget".to_string());
        }
        if self.cases.is_empty() {
            return Err("fixture must contain at least one episode".to_string());
        }

        for case in &self.cases {
            validate_episode(case, &self.graph_semantics)?;
        }
        Ok(())
    }
}

fn validate_episode(case: &Episode, semantics: &GraphSemantics) -> Result<(), String> {
    if !(12..=20).contains(&case.nodes.len()) {
        return Err(format!(
            "{}: node count must be between 12 and 20",
            case.id
        ));
    }

    let mut node_ids = BTreeSet::new();
    for node in &case.nodes {
        if !node_ids.insert(node.id) {
            return Err(format!("{}: duplicate node ID {}", case.id, node.id));
        }
    }

    let sources: Vec<&Node> = case.nodes.iter().filter(|n| n.role == "source").collect();
    if sources.len() != 1 {
        return Err(format!("{}: expected exactly one source node", case.id));
    }
    if sources[0].supply.unwrap_or(0) == 0 {
        return Err(format!("{}: source supply must be positive", case.id));
    }

    let sinks: BTreeSet<u16> = case
        .nodes
        .iter()
        .filter(|n| n.role == "sink")
        .map(|n| n.id)
        .collect();

    let mut active_pairs = BTreeSet::new();
    for edge in &case.active_edges {
        if edge.from == edge.to {
            return Err(format!("{}: self-loop in active edge", case.id));
        }
        if !node_ids.contains(&edge.from) || !node_ids.contains(&edge.to) {
            return Err(format!("{}: active edge has unknown endpoint", case.id));
        }
        if edge.capacity == 0 {
            return Err(format!("{}: zero-capacity edge must be an intervention result, not an initial edge", case.id));
        }
        if !active_pairs.insert((edge.from, edge.to)) {
            return Err(format!("{}: duplicate active directed edge", case.id));
        }
    }

    if case.candidate_edges.len() != semantics.candidate_edges_per_episode {
        return Err(format!(
            "{}: expected {} candidate edges, received {}",
            case.id,
            semantics.candidate_edges_per_episode,
            case.candidate_edges.len()
        ));
    }

    let mut candidate_ids = BTreeSet::new();
    let mut candidate_pairs = BTreeSet::new();
    for edge in &case.candidate_edges {
        let id = edge
            .id
            .as_deref()
            .ok_or_else(|| format!("{}: candidate edge is missing an ID", case.id))?;
        if !candidate_ids.insert(id.to_string()) {
            return Err(format!("{}: duplicate candidate edge ID {id}", case.id));
        }
        if edge.from == edge.to {
            return Err(format!("{}: candidate self-loop", case.id));
        }
        if !node_ids.contains(&edge.from) || !node_ids.contains(&edge.to) {
            return Err(format!("{}: candidate edge has unknown endpoint", case.id));
        }
        if edge.capacity == 0 {
            return Err(format!("{}: candidate edge capacity must be positive", case.id));
        }
        if active_pairs.contains(&(edge.from, edge.to))
            || !candidate_pairs.insert((edge.from, edge.to))
        {
            return Err(format!("{}: candidate duplicates an active or candidate edge", case.id));
        }
    }

    if case.active_edges.len() + semantics.edit_budget > semantics.accounting.maximum_active_edges {
        return Err(format!("{}: active-edge budget is exceeded", case.id));
    }

    if case.demand_schedule.is_empty() {
        return Err(format!("{}: demand schedule is empty", case.id));
    }
    let mut last_time = None;
    for step in &case.demand_schedule {
        if last_time.is_some_and(|t| step.time <= t) {
            return Err(format!("{}: demand schedule times are not strictly increasing", case.id));
        }
        last_time = Some(step.time);
        let mut used_sinks = BTreeSet::new();
        for sink in &step.sinks {
            if !sinks.contains(&sink.node) {
                return Err(format!("{}: demand references a non-sink node {}", case.id, sink.node));
            }
            if !used_sinks.insert(sink.node) {
                return Err(format!("{}: duplicate sink demand at time {}", case.id, step.time));
            }
        }
    }

    for intervention in &case.interventions {
        match intervention.kind.as_str() {
            "edge_capacity_failure" => {
                let from = intervention
                    .from
                    .ok_or_else(|| format!("{}: failure missing source endpoint", case.id))?;
                let to = intervention
                    .to
                    .ok_or_else(|| format!("{}: failure missing target endpoint", case.id))?;
                let after = intervention
                    .capacity_after
                    .ok_or_else(|| format!("{}: failure missing post-intervention capacity", case.id))?;
                let edge = case
                    .active_edges
                    .iter()
                    .find(|e| e.from == from && e.to == to)
                    .ok_or_else(|| format!("{}: failure references an unknown active edge", case.id))?;
                if after > edge.capacity {
                    return Err(format!("{}: failure cannot increase edge capacity", case.id));
                }
            }
            other => return Err(format!("{}: unsupported intervention kind {other}", case.id)),
        }
    }

    Ok(())
}

pub fn evaluate_episode(
    case: &Episode,
    semantics: &GraphSemantics,
    time: u64,
    accepted_candidate_ids: &[&str],
) -> Result<FlowResult, String> {
    validate_episode(case, semantics)?;
    if accepted_candidate_ids.len() > semantics.edit_budget {
        return Err(format!("{}: graph edit budget exceeded", case.id));
    }

    let mut accepted = BTreeSet::new();
    for id in accepted_candidate_ids {
        if !accepted.insert(*id) {
            return Err(format!("{}: duplicate accepted candidate ID {id}", case.id));
        }
        if !case.candidate_edges.iter().any(|e| e.id.as_deref() == Some(*id)) {
            return Err(format!("{}: unknown candidate edge ID {id}", case.id));
        }
    }

    let demand_step = case
        .demand_schedule
        .iter()
        .filter(|step| step.time <= time)
        .max_by_key(|step| step.time)
        .ok_or_else(|| format!("{}: no demand schedule at or before time {time}", case.id))?;

    let mut active_edges = case.active_edges.clone();
    for intervention in case.interventions.iter().filter(|i| i.time <= time) {
        match intervention.kind.as_str() {
            "edge_capacity_failure" => {
                let from = intervention
                    .from
                    .ok_or_else(|| format!("{}: intervention source endpoint missing", case.id))?;
                let to = intervention
                    .to
                    .ok_or_else(|| format!("{}: intervention target endpoint missing", case.id))?;
                let capacity_after = intervention
                    .capacity_after
                    .ok_or_else(|| format!("{}: intervention capacity missing", case.id))?;
                let edge = active_edges
                    .iter_mut()
                    .find(|e| e.from == from && e.to == to)
                    .ok_or_else(|| format!("{}: intervention edge disappeared", case.id))?;
                edge.capacity = capacity_after;
            }
            other => return Err(format!("{}: unsupported intervention kind {other}", case.id)),
        }
    }

    for id in &accepted {
        let candidate = case
            .candidate_edges
            .iter()
            .find(|e| e.id.as_deref() == Some(*id))
            .expect("candidate ID validated");
        active_edges.push(candidate.clone());
    }

    if active_edges.len() > semantics.accounting.maximum_active_edges {
        return Err(format!("{}: maximum active-edge count exceeded", case.id));
    }

    solve_flow(case, &active_edges, &demand_step.sinks)
}

fn solve_flow(
    case: &Episode,
    edges: &[Edge],
    sink_demands: &[SinkDemand],
) -> Result<FlowResult, String> {
    let mut ids: Vec<u16> = case.nodes.iter().map(|n| n.id).collect();
    ids.sort_unstable();
    let index: BTreeMap<u16, usize> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect();

    let source_node = case
        .nodes
        .iter()
        .find(|n| n.role == "source")
        .ok_or_else(|| format!("{}: no source node", case.id))?;
    let source_index = *index
        .get(&source_node.id)
        .ok_or_else(|| format!("{}: source index missing", case.id))?;
    let super_source = ids.len();
    let super_sink = ids.len() + 1;
    let matrix_size = ids.len() + 2;

    let mut capacities = vec![vec![0_u64; matrix_size]; matrix_size];

    for edge in edges {
        let from = *index
            .get(&edge.from)
            .ok_or_else(|| format!("{}: edge source missing", case.id))?;
        let to = *index
            .get(&edge.to)
            .ok_or_else(|| format!("{}: edge target missing", case.id))?;
        capacities[from][to] = capacities[from][to]
            .checked_add(edge.capacity)
            .ok_or_else(|| format!("{}: capacity overflow", case.id))?;
    }

    capacities[super_source][source_index] = source_node.supply.unwrap_or(0);
    let mut requested_flow = 0_u64;
    for sink in sink_demands {
        let to = *index
            .get(&sink.node)
            .ok_or_else(|| format!("{}: sink index missing", case.id))?;
        capacities[to][super_sink] = capacities[to][super_sink]
            .checked_add(sink.demand)
            .ok_or_else(|| format!("{}: sink demand overflow", case.id))?;
        requested_flow = requested_flow
            .checked_add(sink.demand)
            .ok_or_else(|| format!("{}: requested-flow overflow", case.id))?;
    }

    let mut residual = capacities.clone();
    let mut max_flow = 0_u64;

    loop {
        let mut parent: Vec<Option<usize>> = vec![None; matrix_size];
        parent[super_source] = Some(super_source);
        let mut queue = VecDeque::new();
        queue.push_back(super_source);

        while let Some(from) = queue.pop_front() {
            for to in 0..matrix_size {
                if parent[to].is_none() && residual[from][to] > 0 {
                    parent[to] = Some(from);
                    queue.push_back(to);
                }
            }
            if parent[super_sink].is_some() {
                break;
            }
        }

        if parent[super_sink].is_none() {
            break;
        }

        let mut augmentation = u64::MAX;
        let mut cursor = super_sink;
        while cursor != super_source {
            let from = parent[cursor].expect("augmenting path parent exists");
            augmentation = augmentation.min(residual[from][cursor]);
            cursor = from;
        }
        if augmentation == 0 || augmentation == u64::MAX {
            return Err(format!("{}: invalid augmenting path capacity", case.id));
        }

        cursor = super_sink;
        while cursor != super_source {
            let from = parent[cursor].expect("augmenting path parent exists");
            residual[from][cursor] -= augmentation;
            residual[cursor][from] = residual[cursor][from]
                .checked_add(augmentation)
                .ok_or_else(|| format!("{}: residual capacity overflow", case.id))?;
            cursor = from;
        }
        max_flow = max_flow
            .checked_add(augmentation)
            .ok_or_else(|| format!("{}: maximum-flow overflow", case.id))?;
    }

    let node_count = ids.len();
    if node_count > 20 {
        return Err(format!("{}: exhaustive min-cut oracle is capped at 20 nodes", case.id));
    }
    let subset_count = 1_usize
        .checked_shl(node_count as u32)
        .ok_or_else(|| format!("{}: min-cut subset count overflow", case.id))?;
    let mut min_cut = u64::MAX;

    for mask in 0..subset_count {
        let in_source_side = |vertex: usize| -> bool {
            if vertex == super_source {
                true
            } else if vertex == super_sink {
                false
            } else {
                (mask & (1_usize << vertex)) != 0
            }
        };

        let mut cut_capacity = 0_u64;
        for from in 0..matrix_size {
            if !in_source_side(from) {
                continue;
            }
            for to in 0..matrix_size {
                if !in_source_side(to) {
                    cut_capacity = cut_capacity
                        .checked_add(capacities[from][to])
                        .ok_or_else(|| format!("{}: cut capacity overflow", case.id))?;
                }
            }
        }
        min_cut = min_cut.min(cut_capacity);
    }

    if max_flow != min_cut {
        return Err(format!(
            "{}: max-flow/min-cut mismatch: flow={max_flow}, cut={min_cut}",
            case.id
        ));
    }

    Ok(FlowResult {
        max_flow,
        min_cut,
        requested_flow,
    })
}
