//! Evaluator-only D6.13 scorer for whole-DAG admissible histories.
//!
//! Unlike pairwise admissible-relation marginals, this module keeps complete
//! possible event DAGs intact. It derives pairwise entailment from every world,
//! checks candidate graph claims jointly, and detects combinations whose
//! individual edges each occur somewhere but never coexist in one admissible
//! world. Hidden events may occur in evaluator-only worlds and connect visible
//! events. The evaluator does not establish that the supplied world set is
//! complete, authenticate event capture, or solve unbounded causal discovery.

use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, FixedOffset};
use serde::Serialize;
use serde_json::Value;

const CLAIMS: &[&str] = &["before", "after", "concurrent", "underdetermined", "compromised"];
const CLAIM_CEILING: &str = "Synthetic known-answer evaluation over a supplied finite set of complete admissible event DAGs. Preserves correlations across whole histories and detects jointly impossible graph claims. Does not prove the admissible-world set complete, authenticate event capture, or establish arbitrary causal discovery.";

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct PairWorldOutcome {
    pub pair_id: String,
    pub expected_state: String,
    pub admissible_relations: Vec<String>,
    pub claimed_relation: String,
    pub correct: bool,
    pub false_concurrency: bool,
    pub unsupported_causal_claim: bool,
    pub unnecessary_abstention: bool,
    pub future_evidence_event_ids: Vec<String>,
    pub critical_unsupported_claim: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CandidateWorldOutcome {
    pub world_claim_id: String,
    pub asserted_edges: Vec<[String; 2]>,
    pub cyclic: bool,
    pub individually_possible_edge_count: usize,
    pub asserted_edge_count: usize,
    pub jointly_possible: bool,
    pub exact_projection_match: bool,
    pub matched_gold_world_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct WholeDagTraceReport {
    pub trace_id: String,
    pub entailed_visible_edges: Vec<[String; 2]>,
    pub asserted_entailed_edges: Vec<[String; 2]>,
    pub unsupported_asserted_edges: Vec<[String; 2]>,
    pub missing_entailed_edges: Vec<[String; 2]>,
    pub pair_outcomes: Vec<PairWorldOutcome>,
    pub candidate_world_outcomes: Vec<CandidateWorldOutcome>,
    pub gold_projection_count: usize,
    pub claimed_world_count: usize,
    pub matched_world_count: usize,
    pub missing_world_count: usize,
    pub extra_world_count: usize,
    pub jointly_impossible_candidate_graph_count: usize,
    pub cyclic_candidate_graph_count: usize,
    pub false_concurrency_claim_count: usize,
    pub unsupported_causal_claim_count: usize,
    pub unnecessary_abstention_count: usize,
    pub future_evidence_leakage_count: usize,
    pub critical_unsupported_claim_count: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct WholeDagReport {
    pub scenario_id: String,
    pub adjudication_status: String,
    pub visible_event_count: usize,
    pub admissible_world_count: usize,
    pub pair_task_count: usize,
    pub traces: Vec<WholeDagTraceReport>,
    pub claim_ceiling: String,
}

#[derive(Debug, Clone)]
struct Event {
    available_at: DateTime<FixedOffset>,
}

#[derive(Debug, Clone)]
struct PairTask {
    left_event_id: String,
    right_event_id: String,
    decision_cutoff: DateTime<FixedOffset>,
    critical: bool,
}

#[derive(Debug, Clone)]
struct World {
    world_id: String,
    visible_projection: BTreeSet<(String, String)>,
    all_nodes: BTreeSet<String>,
    closure: BTreeSet<(String, String)>,
}

#[derive(Debug, Clone)]
struct CandidateWorld {
    world_claim_id: String,
    edges: BTreeSet<(String, String)>,
}

fn allowed_keys(value: &Value, allowed: &[&str], label: &str) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))?;
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("{label} contains unknown field {key:?}"));
        }
    }
    Ok(())
}

fn required_str<'a>(value: &'a Value, key: &str, label: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|item| !item.trim().is_empty())
        .ok_or_else(|| format!("{label}.{key} must be a non-empty string"))
}

fn required_bool(value: &Value, key: &str, label: &str) -> Result<bool, String> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{label}.{key} must be boolean"))
}

fn parse_time(value: &Value, key: &str, label: &str) -> Result<DateTime<FixedOffset>, String> {
    let raw = required_str(value, key, label)?;
    DateTime::parse_from_rfc3339(raw)
        .map_err(|error| format!("{label}.{key} must be RFC 3339: {error}"))
}

fn string_set(value: &Value, key: &str, label: &str) -> Result<BTreeSet<String>, String> {
    let values = value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{label}.{key} must be an array"))?;
    let mut output = BTreeSet::new();
    for (index, item) in values.iter().enumerate() {
        let id = item
            .as_str()
            .filter(|item| !item.trim().is_empty())
            .ok_or_else(|| format!("{label}.{key}[{index}] must be a non-empty string"))?;
        if !output.insert(id.to_owned()) {
            return Err(format!("{label}.{key} contains duplicate value {id:?}"));
        }
    }
    Ok(output)
}

fn parse_edges(
    value: &Value,
    key: &str,
    label: &str,
    known_nodes: &BTreeSet<String>,
) -> Result<BTreeSet<(String, String)>, String> {
    let values = value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{label}.{key} must be an array"))?;
    let mut output = BTreeSet::new();
    for (index, pair) in values.iter().enumerate() {
        let items = pair
            .as_array()
            .filter(|items| items.len() == 2)
            .ok_or_else(|| format!("{label}.{key}[{index}] must be a two-element array"))?;
        let before = items[0]
            .as_str()
            .filter(|item| !item.trim().is_empty())
            .ok_or_else(|| format!("{label}.{key}[{index}][0] must be a non-empty string"))?;
        let after = items[1]
            .as_str()
            .filter(|item| !item.trim().is_empty())
            .ok_or_else(|| format!("{label}.{key}[{index}][1] must be a non-empty string"))?;
        if before == after {
            return Err(format!("{label}.{key}[{index}] cannot be a self-edge"));
        }
        if !known_nodes.contains(before) || !known_nodes.contains(after) {
            return Err(format!(
                "{label}.{key}[{index}] references an unknown event"
            ));
        }
        if !output.insert((before.to_owned(), after.to_owned())) {
            return Err(format!("{label}.{key} contains a duplicate edge"));
        }
    }
    Ok(output)
}

fn transitive_closure(
    nodes: &BTreeSet<String>,
    edges: &BTreeSet<(String, String)>,
    label: &str,
) -> Result<BTreeSet<(String, String)>, String> {
    let mut outgoing: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for node in nodes {
        outgoing.insert(node.clone(), BTreeSet::new());
    }
    for (before, after) in edges {
        if before == after || !nodes.contains(before) || !nodes.contains(after) {
            return Err(format!("{label} contains a self-edge or unknown event"));
        }
        outgoing
            .get_mut(before)
            .ok_or_else(|| format!("{label} has an unknown source event"))?
            .insert(after.clone());
    }

    let mut closure = BTreeSet::new();
    for start in nodes {
        let mut stack: Vec<String> = outgoing
            .get(start)
            .into_iter()
            .flat_map(|items| items.iter().cloned())
            .collect();
        let mut visited = BTreeSet::new();
        while let Some(next) = stack.pop() {
            if &next == start {
                return Err(format!("{label} contains a directed cycle"));
            }
            if !visited.insert(next.clone()) {
                continue;
            }
            closure.insert((start.clone(), next.clone()));
            if let Some(children) = outgoing.get(&next) {
                stack.extend(children.iter().cloned());
            }
        }
    }
    Ok(closure)
}

fn parse_events(candidate: &Value) -> Result<BTreeMap<String, Event>, String> {
    let values = candidate
        .get("events")
        .and_then(Value::as_array)
        .ok_or_else(|| "candidate_view.events must be an array".to_owned())?;
    if values.is_empty() {
        return Err("candidate_view.events must not be empty".to_owned());
    }
    let mut output = BTreeMap::new();
    for (index, value) in values.iter().enumerate() {
        let label = format!("candidate_view.events[{index}]");
        allowed_keys(value, &["event_id", "available_at"], &label)?;
        let event_id = required_str(value, "event_id", &label)?.to_owned();
        let available_at = parse_time(value, "available_at", &label)?;
        if output
            .insert(event_id.clone(), Event { available_at })
            .is_some()
        {
            return Err(format!("duplicate visible event ID {event_id:?}"));
        }
    }
    Ok(output)
}

fn parse_pair_tasks(
    candidate: &Value,
    events: &BTreeMap<String, Event>,
) -> Result<BTreeMap<String, PairTask>, String> {
    let values = candidate
        .get("pair_tasks")
        .and_then(Value::as_array)
        .ok_or_else(|| "candidate_view.pair_tasks must be an array".to_owned())?;
    if values.is_empty() {
        return Err("candidate_view.pair_tasks must not be empty".to_owned());
    }
    let mut output = BTreeMap::new();
    for (index, value) in values.iter().enumerate() {
        let label = format!("candidate_view.pair_tasks[{index}]");
        allowed_keys(
            value,
            &[
                "pair_id",
                "left_event_id",
                "right_event_id",
                "decision_cutoff",
                "critical",
            ],
            &label,
        )?;
        let pair_id = required_str(value, "pair_id", &label)?.to_owned();
        let left_event_id = required_str(value, "left_event_id", &label)?.to_owned();
        let right_event_id = required_str(value, "right_event_id", &label)?.to_owned();
        if left_event_id == right_event_id {
            return Err(format!("{label} cannot compare an event with itself"));
        }
        if !events.contains_key(&left_event_id) || !events.contains_key(&right_event_id) {
            return Err(format!("{label} references an unknown visible event"));
        }
        let decision_cutoff = parse_time(value, "decision_cutoff", &label)?;
        let critical = required_bool(value, "critical", &label)?;
        if output
            .insert(
                pair_id.clone(),
                PairTask {
                    left_event_id,
                    right_event_id,
                    decision_cutoff,
                    critical,
                },
            )
            .is_some()
        {
            return Err(format!("duplicate pair task {pair_id:?}"));
        }
    }
    Ok(output)
}

fn parse_worlds(
    gold: &Value,
    visible_nodes: &BTreeSet<String>,
) -> Result<Vec<World>, String> {
    let values = gold
        .get("admissible_worlds")
        .and_then(Value::as_array)
        .ok_or_else(|| "gold_overlay.admissible_worlds must be an array".to_owned())?;
    if values.is_empty() {
        return Err("gold_overlay.admissible_worlds must not be empty (no vacuous truth)".to_owned());
    }
    let mut world_ids = BTreeSet::new();
    let mut worlds = Vec::new();
    for (index, value) in values.iter().enumerate() {
        let label = format!("gold_overlay.admissible_worlds[{index}]");
        allowed_keys(value, &["world_id", "hidden_event_ids", "happens_before_edges"], &label)?;
        let world_id = required_str(value, "world_id", &label)?.to_owned();
        if !world_ids.insert(world_id.clone()) {
            return Err(format!("duplicate admissible world ID {world_id:?}"));
        }
        let hidden = string_set(value, "hidden_event_ids", &label)?;
        if hidden.iter().any(|id| visible_nodes.contains(id)) {
            return Err(format!("{label}.hidden_event_ids collides with a visible event ID"));
        }
        let mut all_nodes = visible_nodes.clone();
        all_nodes.extend(hidden);
        let edges = parse_edges(value, "happens_before_edges", &label, &all_nodes)?;
        let closure = transitive_closure(&all_nodes, &edges, &format!("admissible world {world_id}"))?;
        let visible_projection: BTreeSet<(String, String)> = closure
            .iter()
            .filter(|(before, after)| visible_nodes.contains(before) && visible_nodes.contains(after))
            .cloned()
            .collect();
        worlds.push(World {
            world_id,
            visible_projection,
            all_nodes,
            closure,
        });
    }
    Ok(worlds)
}

fn parse_candidate_worlds(
    trace: &Value,
    visible_nodes: &BTreeSet<String>,
) -> Result<Vec<CandidateWorld>, String> {
    let values = trace
        .get("claimed_worlds")
        .and_then(Value::as_array)
        .ok_or_else(|| "candidate_trace.claimed_worlds must be an array".to_owned())?;
    let mut ids = BTreeSet::new();
    let mut worlds = Vec::new();
    for (index, value) in values.iter().enumerate() {
        let label = format!("candidate_trace.claimed_worlds[{index}]");
        allowed_keys(value, &["world_claim_id", "happens_before_edges"], &label)?;
        let world_claim_id = required_str(value, "world_claim_id", &label)?.to_owned();
        if !ids.insert(world_claim_id.clone()) {
            return Err(format!("duplicate candidate world claim ID {world_claim_id:?}"));
        }
        let edges = parse_edges(value, "happens_before_edges", &label, visible_nodes)?;
        worlds.push(CandidateWorld { world_claim_id, edges });
    }
    Ok(worlds)
}

fn relation_in_world(world: &World, left: &str, right: &str) -> Result<&'static str, String> {
    let left_before_right = world.closure.contains(&(left.to_owned(), right.to_owned()));
    let right_before_left = world.closure.contains(&(right.to_owned(), left.to_owned()));
    match (left_before_right, right_before_left) {
        (true, false) => Ok("before"),
        (false, true) => Ok("after"),
        (false, false) => Ok("concurrent"),
        (true, true) => Err(format!("world {:?} contains a causal cycle", world.world_id)),
    }
}

fn expected_pair_state(relations: &BTreeSet<String>) -> String {
    if relations.len() == 1 {
        relations.iter().next().cloned().unwrap_or_else(|| "underdetermined".to_owned())
    } else {
        "underdetermined".to_owned()
    }
}

fn pair_relation_set(worlds: &[World], left: &str, right: &str) -> Result<BTreeSet<String>, String> {
    let mut relations = BTreeSet::new();
    for world in worlds {
        relations.insert(relation_in_world(world, left, right)?.to_owned());
    }
    if relations.is_empty() {
        return Err("admissible-world relation set is empty".to_owned());
    }
    Ok(relations)
}

fn parse_pair_claims(
    trace: &Value,
    tasks: &BTreeMap<String, PairTask>,
    events: &BTreeMap<String, Event>,
) -> Result<BTreeMap<String, (String, BTreeSet<String>)>, String> {
    let values = trace
        .get("pair_claims")
        .and_then(Value::as_array)
        .ok_or_else(|| "candidate_trace.pair_claims must be an array".to_owned())?;
    let mut claims = BTreeMap::new();
    for (index, value) in values.iter().enumerate() {
        let label = format!("candidate_trace.pair_claims[{index}]");
        allowed_keys(value, &["pair_id", "claimed_relation", "evidence_event_ids"], &label)?;
        let pair_id = required_str(value, "pair_id", &label)?.to_owned();
        if !tasks.contains_key(&pair_id) {
            return Err(format!("{label} references an unknown pair task {pair_id:?}"));
        }
        let claimed_relation = required_str(value, "claimed_relation", &label)?.to_owned();
        if !CLAIMS.contains(&claimed_relation.as_str()) {
            return Err(format!("{label}.claimed_relation is unsupported"));
        }
        let evidence_event_ids = string_set(value, "evidence_event_ids", &label)?;
        for event_id in &evidence_event_ids {
            if !events.contains_key(event_id) {
                return Err(format!("{label} references unknown evidence event {event_id:?}"));
            }
        }
        if claims.insert(pair_id.clone(), (claimed_relation, evidence_event_ids)).is_some() {
            return Err(format!("duplicate candidate pair claim {pair_id:?}"));
        }
    }
    if claims.len() != tasks.len() || tasks.keys().any(|pair_id| !claims.contains_key(pair_id)) {
        return Err("candidate trace must claim every pair task exactly once".to_owned());
    }
    Ok(claims)
}

fn parse_asserted_edges(
    trace: &Value,
    visible_nodes: &BTreeSet<String>,
) -> Result<BTreeSet<(String, String)>, String> {
    let wrapper = serde_json::json!({
        "edges": trace.get("asserted_happens_before_edges")
            .ok_or_else(|| "candidate_trace.asserted_happens_before_edges is required".to_owned())?
    });
    parse_edges(&wrapper, "edges", "candidate_trace", visible_nodes)
}

fn score_trace(
    trace: &Value,
    events: &BTreeMap<String, Event>,
    tasks: &BTreeMap<String, PairTask>,
    worlds: &[World],
    visible_nodes: &BTreeSet<String>,
) -> Result<WholeDagTraceReport, String> {
    allowed_keys(
        trace,
        &[
            "trace_id",
            "asserted_happens_before_edges",
            "pair_claims",
            "claimed_worlds",
        ],
        "candidate_trace",
    )?;
    let trace_id = required_str(trace, "trace_id", "candidate_trace")?.to_owned();

    let mut gold_projections = BTreeSet::new();
    for world in worlds {
        gold_projections.insert(world.visible_projection.clone());
    }
    let mut union_possible_edges = BTreeSet::new();
    let mut entailed_edges: Option<BTreeSet<(String, String)>> = None;
    for world in worlds {
        union_possible_edges.extend(world.visible_projection.iter().cloned());
        entailed_edges = Some(match entailed_edges {
            Some(current) => current.intersection(&world.visible_projection).cloned().collect(),
            None => world.visible_projection.clone(),
        });
    }
    let entailed_edges = entailed_edges.unwrap_or_default();

    let asserted_edges = parse_asserted_edges(trace, visible_nodes)?;
    let candidate_closure_result =
        transitive_closure(visible_nodes, &asserted_edges, "candidate asserted happens-before graph");
    let (asserted_entailed_edges, unsupported_asserted_edges) = match candidate_closure_result {
        Ok(closure) => {
            let asserted_entailed: BTreeSet<_> =
                closure.intersection(&entailed_edges).cloned().collect();
            let unsupported: BTreeSet<_> = closure.difference(&entailed_edges).cloned().collect();
            (asserted_entailed, unsupported)
        }
        Err(_) => (BTreeSet::new(), asserted_edges.clone()),
    };
    let missing_entailed_edges: BTreeSet<_> =
        entailed_edges.difference(&asserted_entailed_edges).cloned().collect();

    let pair_claims = parse_pair_claims(trace, tasks, events)?;
    let mut pair_outcomes = Vec::new();
    for (pair_id, task) in tasks {
        let (claimed_relation, evidence_ids) = pair_claims
            .get(pair_id)
            .ok_or_else(|| format!("missing pair claim {pair_id:?}"))?;
        let relations = pair_relation_set(worlds, &task.left_event_id, &task.right_event_id)?;
        let expected_state = expected_pair_state(&relations);
        let future_evidence_event_ids: Vec<String> = evidence_ids
            .iter()
            .filter(|event_id| {
                events
                    .get(*event_id)
                    .is_some_and(|event| event.available_at > task.decision_cutoff)
            })
            .cloned()
            .collect();
        let correct = claimed_relation == &expected_state;
        let false_concurrency = claimed_relation == "concurrent" && expected_state != "concurrent";
        let unsupported_causal_claim = ["before", "after"].contains(&claimed_relation.as_str())
            && !relations.contains(claimed_relation);
        let unnecessary_abstention =
            claimed_relation == "underdetermined" && expected_state != "underdetermined";
        let critical_unsupported_claim = task.critical
            && ["before", "after", "concurrent"].contains(&claimed_relation.as_str())
            && !correct;
        pair_outcomes.push(PairWorldOutcome {
            pair_id: pair_id.clone(),
            expected_state,
            admissible_relations: relations.into_iter().collect(),
            claimed_relation: claimed_relation.clone(),
            correct,
            false_concurrency,
            unsupported_causal_claim,
            unnecessary_abstention,
            future_evidence_event_ids,
            critical_unsupported_claim,
        });
    }

    let candidate_worlds = parse_candidate_worlds(trace, visible_nodes)?;
    let mut candidate_world_outcomes = Vec::new();
    for candidate_world in &candidate_worlds {
        let closure = transitive_closure(
            visible_nodes,
            &candidate_world.edges,
            &format!("candidate world {}", candidate_world.world_claim_id),
        );
        let (cyclic, closure) = match closure {
            Ok(closure) => (false, closure),
            Err(_) => (true, BTreeSet::new()),
        };
        let individually_possible_edge_count = closure
            .iter()
            .filter(|edge| union_possible_edges.contains(*edge))
            .count();
        let jointly_possible = !cyclic
            && gold_projections
                .iter()
                .any(|projection| closure.is_subset(projection));
        let exact_projection_match = !cyclic && gold_projections.contains(&closure);
        let matched_gold_world_ids: Vec<String> = if exact_projection_match {
            worlds
                .iter()
                .filter(|world| world.visible_projection == closure)
                .map(|world| world.world_id.clone())
                .collect()
        } else {
            Vec::new()
        };
        candidate_world_outcomes.push(CandidateWorldOutcome {
            world_claim_id: candidate_world.world_claim_id.clone(),
            asserted_edges: candidate_world
                .edges
                .iter()
                .map(|(before, after)| [before.clone(), after.clone()])
                .collect(),
            cyclic,
            individually_possible_edge_count,
            asserted_edge_count: closure.len(),
            jointly_possible,
            exact_projection_match,
            matched_gold_world_ids,
        });
    }

    let claimed_projection_set: BTreeSet<BTreeSet<(String, String)>> = candidate_world_outcomes
        .iter()
        .filter(|outcome| !outcome.cyclic)
        .filter_map(|outcome| {
            candidate_worlds
                .iter()
                .find(|world| world.world_claim_id == outcome.world_claim_id)
                .and_then(|world| transitive_closure(visible_nodes, &world.edges, "candidate projection").ok())
        })
        .collect();
    let matched_projection_count = claimed_projection_set
        .intersection(&gold_projections)
        .count();
    let missing_world_count = gold_projections.difference(&claimed_projection_set).count();
    let extra_world_count = claimed_projection_set.difference(&gold_projections).count();

    Ok(WholeDagTraceReport {
        trace_id,
        entailed_visible_edges: entailed_edges.iter().map(|(a, b)| [a.clone(), b.clone()]).collect(),
        asserted_entailed_edges: asserted_entailed_edges
            .iter()
            .map(|(a, b)| [a.clone(), b.clone()])
            .collect(),
        unsupported_asserted_edges: unsupported_asserted_edges
            .iter()
            .map(|(a, b)| [a.clone(), b.clone()])
            .collect(),
        missing_entailed_edges: missing_entailed_edges
            .iter()
            .map(|(a, b)| [a.clone(), b.clone()])
            .collect(),
        pair_outcomes,
        candidate_world_outcomes: candidate_world_outcomes.clone(),
        gold_projection_count: gold_projections.len(),
        claimed_world_count: candidate_worlds.len(),
        matched_world_count: matched_projection_count,
        missing_world_count,
        extra_world_count,
        jointly_impossible_candidate_graph_count: candidate_world_outcomes
            .iter()
            .filter(|outcome| !outcome.cyclic && !outcome.jointly_possible && outcome.individually_possible_edge_count > 0)
            .count(),
        cyclic_candidate_graph_count: candidate_world_outcomes.iter().filter(|outcome| outcome.cyclic).count(),
        false_concurrency_claim_count: pair_outcomes.iter().filter(|item| item.false_concurrency).count(),
        unsupported_causal_claim_count: pair_outcomes.iter().filter(|item| item.unsupported_causal_claim).count()
            + unsupported_asserted_edges.len(),
        unnecessary_abstention_count: pair_outcomes.iter().filter(|item| item.unnecessary_abstention).count(),
        future_evidence_leakage_count: pair_outcomes
            .iter()
            .map(|item| item.future_evidence_event_ids.len())
            .sum(),
        critical_unsupported_claim_count: pair_outcomes
            .iter()
            .filter(|item| item.critical_unsupported_claim)
            .count(),
    })
}

/// Score a D6.13 bundle containing a finite, evaluator-only set of whole DAGs.
/// An empty admissible-world set is invalid and never yields vacuous entailment.
pub fn evaluate_whole_dag_json(input: &str) -> Result<WholeDagReport, String> {
    let bundle: Value =
        serde_json::from_str(input).map_err(|error| format!("bundle JSON parse failed: {error}"))?;
    allowed_keys(
        &bundle,
        &["bundle_id", "bundle_visibility", "candidate_view", "gold_overlay", "candidate_traces"],
        "bundle",
    )?;
    required_str(&bundle, "bundle_id", "bundle")?;
    if required_str(&bundle, "bundle_visibility", "bundle")?
        != "evaluator_only_never_candidate_visible"
    {
        return Err("D6.13 bundle must be explicitly evaluator-only".to_owned());
    }
    let candidate = bundle
        .get("candidate_view")
        .ok_or_else(|| "candidate_view is required".to_owned())?;
    allowed_keys(candidate, &["scenario_id", "events", "pair_tasks"], "candidate_view")?;
    let scenario_id = required_str(candidate, "scenario_id", "candidate_view")?.to_owned();
    let events = parse_events(candidate)?;
    let visible_nodes: BTreeSet<String> = events.keys().cloned().collect();
    let tasks = parse_pair_tasks(candidate, &events)?;

    let gold = bundle
        .get("gold_overlay")
        .ok_or_else(|| "gold_overlay is required".to_owned())?;
    allowed_keys(gold, &["adjudication_status", "admissible_worlds"], "gold_overlay")?;
    let adjudication_status = required_str(gold, "adjudication_status", "gold_overlay")?.to_owned();
    if adjudication_status != "synthetic_known_answer_not_independently_adjudicated" {
        return Err(format!("unsupported D6.13 v0.1 adjudication status {adjudication_status:?}"));
    }
    let worlds = parse_worlds(gold, &visible_nodes)?;

    let traces = bundle
        .get("candidate_traces")
        .and_then(Value::as_array)
        .ok_or_else(|| "candidate_traces must be an array".to_owned())?;
    if traces.is_empty() {
        return Err("candidate_traces must not be empty".to_owned());
    }
    let mut trace_ids = BTreeSet::new();
    let mut reports = Vec::new();
    for trace in traces {
        let trace_id = required_str(trace, "trace_id", "candidate_trace")?.to_owned();
        if !trace_ids.insert(trace_id) {
            return Err("candidate trace IDs must be unique".to_owned());
        }
        reports.push(score_trace(trace, &events, &tasks, &worlds, &visible_nodes)?);
    }
    reports.sort_by(|left, right| left.trace_id.cmp(&right.trace_id));

    Ok(WholeDagReport {
        scenario_id,
        adjudication_status,
        visible_event_count: events.len(),
        admissible_world_count: worlds.len(),
        pair_task_count: tasks.len(),
        traces: reports,
        claim_ceiling: CLAIM_CEILING.to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const FIXTURE: &str = include_str!(
        "../../../docs/development-map/fixtures/egddm-whole-dag-v0.1.json"
    );
    const EXPECTED: &str = include_str!(
        "../../../docs/development-map/fixtures/egddm-whole-dag-v0.1.expected.json"
    );

    fn fixture_value() -> Value {
        serde_json::from_str(FIXTURE).expect("D6.13 fixture JSON")
    }

    fn report_for(value: &Value) -> WholeDagReport {
        evaluate_whole_dag_json(&value.to_string()).expect("D6.13 score")
    }

    fn trace<'a>(report: &'a WholeDagReport, id: &str) -> &'a WholeDagTraceReport {
        report.traces.iter().find(|item| item.trace_id == id).expect("trace")
    }

    #[test]
    fn matches_complete_machine_readable_known_answer() {
        let actual = evaluate_whole_dag_json(FIXTURE).expect("D6.13 fixture");
        let actual_value = serde_json::to_value(actual).expect("serialize report");
        let expected_value: Value = serde_json::from_str(EXPECTED).expect("expected report");
        assert_eq!(actual_value, expected_value);
    }

    #[test]
    fn preserves_world_correlations_instead_of_pairwise_marginals() {
        let report = evaluate_whole_dag_json(FIXTURE).expect("D6.13 fixture");
        let adversarial = trace(&report, "pairwise-marginals-combined");
        assert_eq!(adversarial.jointly_impossible_candidate_graph_count, 1);
        let impossible = adversarial.candidate_world_outcomes.iter()
            .find(|item| item.world_claim_id == "impossible-mix")
            .expect("impossible mixed graph");
        assert!(impossible.individually_possible_edge_count > 0);
        assert!(!impossible.jointly_possible);
    }

    #[test]
    fn detects_cyclic_candidate_world_claims_without_aborting_the_bundle() {
        let report = evaluate_whole_dag_json(FIXTURE).expect("D6.13 fixture");
        let adversarial = trace(&report, "pairwise-marginals-combined");
        assert_eq!(adversarial.cyclic_candidate_graph_count, 1);
    }

    #[test]
    fn derives_entailed_edges_by_intersection_of_whole_world_projections() {
        let report = evaluate_whole_dag_json(FIXTURE).expect("D6.13 fixture");
        let conformant = trace(&report, "whole-world-calibrated");
        assert_eq!(
            conformant.entailed_visible_edges,
            vec![["z".to_owned(), "c".to_owned()]]
        );
    }

    #[test]
    fn a_hidden_intermediary_prevents_false_concurrency() {
        let report = evaluate_whole_dag_json(FIXTURE).expect("D6.13 fixture");
        let conformant = trace(&report, "whole-world-calibrated");
        let pair = conformant.pair_outcomes.iter().find(|item| item.pair_id == "hidden-path")
            .expect("hidden path pair");
        assert_eq!(pair.expected_state, "underdetermined");
        assert_eq!(pair.admissible_relations, vec!["before".to_owned(), "concurrent".to_owned()]);
    }

    #[test]
    fn accepts_genuine_concurrency_only_when_every_world_agrees() {
        let report = evaluate_whole_dag_json(FIXTURE).expect("D6.13 fixture");
        let conformant = trace(&report, "whole-world-calibrated");
        let pair = conformant.pair_outcomes.iter().find(|item| item.pair_id == "always-concurrent")
            .expect("concurrent pair");
        assert_eq!(pair.expected_state, "concurrent");
        assert!(pair.correct);
    }

    #[test]
    fn reports_complete_admissible_projection_coverage() {
        let report = evaluate_whole_dag_json(FIXTURE).expect("D6.13 fixture");
        let conformant = trace(&report, "whole-world-calibrated");
        assert_eq!(conformant.gold_projection_count, 2);
        assert_eq!(conformant.matched_world_count, 2);
        assert_eq!(conformant.missing_world_count, 0);
        assert_eq!(conformant.extra_world_count, 0);
    }

    #[test]
    fn always_abstain_does_not_solve_entailed_relations() {
        let report = evaluate_whole_dag_json(FIXTURE).expect("D6.13 fixture");
        let always = trace(&report, "always-abstain");
        assert_eq!(always.unnecessary_abstention_count, 1);
        assert_eq!(always.missing_world_count, 2);
    }

    #[test]
    fn rejects_empty_admissible_world_set_instead_of_vacuous_truth() {
        let mut value = fixture_value();
        value["gold_overlay"]["admissible_worlds"] = json!([]);
        let error = evaluate_whole_dag_json(&value.to_string()).unwrap_err();
        assert!(error.contains("must not be empty"));
    }

    #[test]
    fn rejects_cyclic_gold_admissible_world() {
        let mut value = fixture_value();
        value["gold_overlay"]["admissible_worlds"][0]["happens_before_edges"]
            .as_array_mut().expect("edges")
            .push(json!(["c", "a"]));
        let error = evaluate_whole_dag_json(&value.to_string()).unwrap_err();
        assert!(error.contains("directed cycle"));
    }

    #[test]
    fn rejects_hidden_visible_event_identifier_collision() {
        let mut value = fixture_value();
        value["gold_overlay"]["admissible_worlds"][0]["hidden_event_ids"] = json!(["a"]);
        let error = evaluate_whole_dag_json(&value.to_string()).unwrap_err();
        assert!(error.contains("collides with a visible"));
    }

    #[test]
    fn rejects_candidate_unknown_event_edges() {
        let mut value = fixture_value();
        value["candidate_traces"][0]["asserted_happens_before_edges"] = json!([["ghost", "z"]]);
        let error = evaluate_whole_dag_json(&value.to_string()).unwrap_err();
        assert!(error.contains("unknown event"));
    }

    #[test]
    fn reports_future_evidence_without_rewriting_historical_worlds() {
        let report = evaluate_whole_dag_json(FIXTURE).expect("D6.13 fixture");
        let adversarial = trace(&report, "pairwise-marginals-combined");
        assert_eq!(adversarial.future_evidence_leakage_count, 1);
    }
}
