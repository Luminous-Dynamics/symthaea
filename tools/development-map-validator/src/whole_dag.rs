//! Evaluator-only D6.13 scorer over complete admissible DAG worlds.
//!
//! Unlike D6.11's pairwise relation marginals, this module preserves the
//! correlations among edges in each complete admissible world. It derives
//! pairwise entailment from each world’s transitive closure and checks whether
//! all determinate claims made by a candidate can be satisfied by at least
//! one single world. An empty world set or cyclic world is rejected, never
//! treated as vacuous truth.
//!
//! This is bounded finite-world evaluation, not arbitrary causal discovery or
//! proof that the evaluator's admissible-world inventory is complete.

use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, FixedOffset};
use serde::Serialize;
use serde_json::Value;

const CLAIMS: &[&str] = &["before", "after", "concurrent", "underdetermined", "compromised"];
const BOUNDARIES: &[&str] = &["complete", "incomplete", "unknown", "compromised"];
const CLAIM_CEILING: &str = "Finite whole-DAG possible-world evaluation of supplied admissible completions, trusted/forbidden relations, observation-boundary state, candidate claims, and decision-cutoff evidence availability. Does not prove the admissible-world set complete, authenticate real observation certificates, solve arbitrary causal discovery, or establish agent performance.";

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct WorldRelationReport {
    pub world_id: String,
    pub relation: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct GraphPairOutcome {
    pub left_event_id: String,
    pub right_event_id: String,
    pub possible_relations: Vec<String>,
    pub expected_state: String,
    pub claimed_relation: String,
    pub individually_plausible: bool,
    pub correct: bool,
    pub false_concurrency: bool,
    pub unsupported_causal_claim: bool,
    pub unnecessary_abstention: bool,
    pub future_evidence_event_ids: Vec<String>,
    pub world_relations: Vec<WorldRelationReport>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct WholeDagTraceReport {
    pub trace_id: String,
    pub observation_boundary_claim: String,
    pub observation_boundary_correct: bool,
    pub pair_outcomes: Vec<GraphPairOutcome>,
    pub pair_count: usize,
    pub individually_plausible_claim_count: usize,
    pub jointly_consistent_world_ids: Vec<String>,
    pub jointly_consistent_world_count: usize,
    pub jointly_impossible_claim_set: bool,
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
    pub event_count: usize,
    pub admissible_world_count: usize,
    pub pair_count: usize,
    pub trusted_edge_count: usize,
    pub forbidden_pair_count: usize,
    pub traces: Vec<WholeDagTraceReport>,
    pub claim_ceiling: String,
}

#[derive(Debug, Clone)]
struct Event {
    available_at: DateTime<FixedOffset>,
}

#[derive(Debug, Clone)]
struct World {
    world_id: String,
    closure: BTreeSet<(String, String)>,
}

#[derive(Debug, Clone)]
struct PairClaim {
    claimed_relation: String,
    decision_cutoff: DateTime<FixedOffset>,
    evidence_event_ids: BTreeSet<String>,
    critical: bool,
}

#[derive(Debug, Clone)]
struct PairTruth {
    possible_relations: BTreeSet<String>,
    world_relations: Vec<WorldRelationReport>,
    expected_state: String,
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

fn parse_edge_pairs(
    value: &Value,
    key: &str,
    label: &str,
    event_ids: &BTreeSet<String>,
    unordered: bool,
) -> Result<BTreeSet<(String, String)>, String> {
    let values = value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{label}.{key} must be an array"))?;
    let mut output = BTreeSet::new();
    for (index, item) in values.iter().enumerate() {
        let pair = item
            .as_array()
            .filter(|pair| pair.len() == 2)
            .ok_or_else(|| format!("{label}.{key}[{index}] must be a two-element array"))?;
        let left = pair[0]
            .as_str()
            .filter(|id| !id.trim().is_empty())
            .ok_or_else(|| format!("{label}.{key}[{index}][0] must be a non-empty string"))?;
        let right = pair[1]
            .as_str()
            .filter(|id| !id.trim().is_empty())
            .ok_or_else(|| format!("{label}.{key}[{index}][1] must be a non-empty string"))?;
        if left == right {
            return Err(format!("{label}.{key}[{index}] cannot contain a self relation"));
        }
        if !event_ids.contains(left) || !event_ids.contains(right) {
            return Err(format!("{label}.{key}[{index}] references an unknown event"));
        }
        let normalized = if unordered && left > right {
            (right.to_owned(), left.to_owned())
        } else {
            (left.to_owned(), right.to_owned())
        };
        if !output.insert(normalized) {
            return Err(format!("{label}.{key} contains a duplicate pair"));
        }
    }
    Ok(output)
}

fn transitive_closure(
    event_ids: &BTreeSet<String>,
    edges: &BTreeSet<(String, String)>,
    label: &str,
) -> Result<BTreeSet<(String, String)>, String> {
    let mut outgoing: BTreeMap<String, BTreeSet<String>> = event_ids
        .iter()
        .map(|event_id| (event_id.clone(), BTreeSet::new()))
        .collect();
    for (before, after) in edges {
        outgoing
            .get_mut(before)
            .ok_or_else(|| format!("{label} has an unknown source event"))?
            .insert(after.clone());
    }
    let mut closure = BTreeSet::new();
    for start in event_ids {
        let mut stack: Vec<String> = outgoing
            .get(start)
            .into_iter()
            .flat_map(|children| children.iter().cloned())
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

fn parse_events(candidate: &Value) -> Result<(BTreeMap<String, Event>, BTreeSet<String>), String> {
    let values = candidate
        .get("events")
        .and_then(Value::as_array)
        .ok_or_else(|| "candidate_view.events must be an array".to_owned())?;
    if values.len() < 2 {
        return Err("candidate_view.events must contain at least two events".to_owned());
    }
    let mut events = BTreeMap::new();
    for (index, value) in values.iter().enumerate() {
        let label = format!("candidate_view.events[{index}]");
        allowed_keys(value, &["event_id", "event_time", "available_at"], &label)?;
        let event_id = required_str(value, "event_id", &label)?.to_owned();
        let _event_time = parse_time(value, "event_time", &label)?;
        let available_at = parse_time(value, "available_at", &label)?;
        if events.insert(event_id.clone(), Event { available_at }).is_some() {
            return Err(format!("duplicate event_id {event_id:?}"));
        }
    }
    let ids = events.keys().cloned().collect();
    Ok((events, ids))
}

fn parse_worlds(
    gold: &Value,
    event_ids: &BTreeSet<String>,
) -> Result<Vec<World>, String> {
    let values = gold
        .get("admissible_worlds")
        .and_then(Value::as_array)
        .ok_or_else(|| "gold_overlay.admissible_worlds must be an array".to_owned())?;
    if values.is_empty() {
        return Err("admissible_worlds must not be empty; empty-world vacuous truth is forbidden".to_owned());
    }
    let mut worlds = Vec::new();
    let mut world_ids = BTreeSet::new();
    for (index, value) in values.iter().enumerate() {
        let label = format!("gold_overlay.admissible_worlds[{index}]");
        allowed_keys(value, &["world_id", "edges"], &label)?;
        let world_id = required_str(value, "world_id", &label)?.to_owned();
        if !world_ids.insert(world_id.clone()) {
            return Err(format!("duplicate world_id {world_id:?}"));
        }
        let edge_pairs = parse_edge_pairs(value, "edges", &label, event_ids, false)?;
        let closure = transitive_closure(event_ids, &edge_pairs, &format!("world {world_id:?}"))?;
        worlds.push(World { world_id, closure });
    }
    worlds.sort_by(|left, right| left.world_id.cmp(&right.world_id));
    Ok(worlds)
}

fn pair_relation(
    left: &str,
    right: &str,
    closure: &BTreeSet<(String, String)>,
) -> String {
    if closure.contains(&(left.to_owned(), right.to_owned())) {
        "before".to_owned()
    } else if closure.contains(&(right.to_owned(), left.to_owned())) {
        "after".to_owned()
    } else {
        "concurrent".to_owned()
    }
}

fn expected_state(
    possible_relations: &BTreeSet<String>,
    boundary_state: &str,
) -> String {
    if possible_relations.len() != 1 {
        return if boundary_state == "compromised" {
            "compromised".to_owned()
        } else {
            "underdetermined".to_owned()
        };
    }
    let relation = possible_relations.iter().next().cloned().unwrap_or_default();
    if relation == "concurrent" && boundary_state != "complete" {
        return if boundary_state == "compromised" {
            "compromised".to_owned()
        } else {
            "underdetermined".to_owned()
        };
    }
    // A compromised observer does not erase a positive relation that is
    // independently entailed by every admissible whole-graph completion.
    relation
}

fn build_pair_truths(
    event_ids: &BTreeSet<String>,
    worlds: &[World],
    boundary_state: &str,
) -> BTreeMap<(String, String), PairTruth> {
    let ids: Vec<&String> = event_ids.iter().collect();
    let mut output = BTreeMap::new();
    for index in 0..ids.len() {
        for other_index in (index + 1)..ids.len() {
            let left = ids[index].to_string();
            let right = ids[other_index].to_string();
            let mut possible_relations = BTreeSet::new();
            let mut world_relations = Vec::new();
            for world in worlds {
                let relation = pair_relation(&left, &right, &world.closure);
                possible_relations.insert(relation.clone());
                world_relations.push(WorldRelationReport {
                    world_id: world.world_id.clone(),
                    relation,
                });
            }
            let expected = expected_state(&possible_relations, boundary_state);
            output.insert(
                (left, right),
                PairTruth {
                    possible_relations,
                    world_relations,
                    expected_state: expected,
                },
            );
        }
    }
    output
}

fn validate_world_constraints(
    candidate: &Value,
    event_ids: &BTreeSet<String>,
    worlds: &[World],
) -> Result<(usize, usize), String> {
    let trusted = parse_edge_pairs(candidate, "trusted_happens_before_edges", "candidate_view", event_ids, false)?;
    let forbidden = parse_edge_pairs(candidate, "forbidden_ordered_pairs", "candidate_view", event_ids, true)?;
    for world in worlds {
        for (before, after) in &trusted {
            if !world.closure.contains(&(before.clone(), after.clone())) {
                return Err(format!(
                    "world {:?} violates trusted happens-before relation {before:?}->{after:?}",
                    world.world_id
                ));
            }
        }
        for (left, right) in &forbidden {
            if world.closure.contains(&(left.clone(), right.clone()))
                || world.closure.contains(&(right.clone(), left.clone()))
            {
                return Err(format!(
                    "world {:?} violates forbidden unordered pair {left:?}/{right:?}",
                    world.world_id
                ));
            }
        }
    }
    Ok((trusted.len(), forbidden.len()))
}

fn parse_claims(
    trace: &Value,
    pair_truths: &BTreeMap<(String, String), PairTruth>,
    events: &BTreeMap<String, Event>,
) -> Result<(String, BTreeMap<(String, String), PairClaim>), String> {
    allowed_keys(trace, &["trace_id", "observation_boundary_claim", "relation_claims"], "candidate_trace")?;
    let trace_id = required_str(trace, "trace_id", "candidate_trace")?.to_owned();
    let boundary_claim = required_str(trace, "observation_boundary_claim", "candidate_trace")?.to_owned();
    if !BOUNDARIES.contains(&boundary_claim.as_str()) {
        return Err(format!("candidate_trace.observation_boundary_claim is unsupported: {boundary_claim:?}"));
    }
    let values = trace
        .get("relation_claims")
        .and_then(Value::as_array)
        .ok_or_else(|| "candidate_trace.relation_claims must be an array".to_owned())?;
    let mut claims = BTreeMap::new();
    for (index, value) in values.iter().enumerate() {
        let label = format!("candidate_trace.relation_claims[{index}]");
        allowed_keys(
            value,
            &["left_event_id", "right_event_id", "claimed_relation", "decision_cutoff", "evidence_event_ids", "critical"],
            &label,
        )?;
        let left = required_str(value, "left_event_id", &label)?.to_owned();
        let right = required_str(value, "right_event_id", &label)?.to_owned();
        if left >= right {
            return Err(format!("{label} must use canonical event order left_event_id < right_event_id"));
        }
        let key = (left.clone(), right.clone());
        if !pair_truths.contains_key(&key) {
            return Err(format!("{label} references a non-existent event pair"));
        }
        let claimed_relation = required_str(value, "claimed_relation", &label)?.to_owned();
        if !CLAIMS.contains(&claimed_relation.as_str()) {
            return Err(format!("{label}.claimed_relation is unsupported"));
        }
        let decision_cutoff = parse_time(value, "decision_cutoff", &label)?;
        let evidence_event_ids = string_set(value, "evidence_event_ids", &label)?;
        for event_id in &evidence_event_ids {
            if !events.contains_key(event_id) {
                return Err(format!("{label} references unknown evidence event {event_id:?}"));
            }
        }
        let critical = required_bool(value, "critical", &label)?;
        if claims
            .insert(
                key.clone(),
                PairClaim {
                    claimed_relation,
                    decision_cutoff,
                    evidence_event_ids,
                    critical,
                },
            )
            .is_some()
        {
            return Err(format!("{label} duplicates relation claim {key:?}"));
        }
    }
    if claims.len() != pair_truths.len()
        || pair_truths.keys().any(|pair| !claims.contains_key(pair))
    {
        return Err(format!(
            "candidate trace {trace_id:?} must claim every event pair exactly once"
        ));
    }
    Ok((boundary_claim, claims))
}

fn world_satisfies_claims(
    world: &World,
    claims: &BTreeMap<(String, String), PairClaim>,
) -> bool {
    claims.iter().all(|(pair, claim)| {
        if claim.claimed_relation == "underdetermined" || claim.claimed_relation == "compromised" {
            return true;
        }
        pair_relation(&pair.0, &pair.1, &world.closure) == claim.claimed_relation
    })
}

fn score_trace(
    trace: &Value,
    events: &BTreeMap<String, Event>,
    event_ids: &BTreeSet<String>,
    worlds: &[World],
    pair_truths: &BTreeMap<(String, String), PairTruth>,
    expected_boundary_state: &str,
) -> Result<WholeDagTraceReport, String> {
    let (boundary_claim, claims) = parse_claims(trace, pair_truths, events)?;
    let mut pair_outcomes = Vec::new();
    let mut individually_plausible_claim_count = 0usize;
    let mut false_concurrency_claim_count = 0usize;
    let mut unsupported_causal_claim_count = 0usize;
    let mut unnecessary_abstention_count = 0usize;
    let mut future_evidence_leakage_count = 0usize;
    let mut critical_unsupported_claim_count = 0usize;

    for ((left, right), truth) in pair_truths {
        let claim = claims
            .get(&(left.clone(), right.clone()))
            .ok_or_else(|| "candidate pair claim missing after coverage check".to_owned())?;
        let individually_plausible = if claim.claimed_relation == "underdetermined"
            || claim.claimed_relation == "compromised"
        {
            true
        } else {
            truth.possible_relations.contains(&claim.claimed_relation)
        };
        if individually_plausible {
            individually_plausible_claim_count += 1;
        }

        let expected = truth.expected_state.clone();
        let correct = claim.claimed_relation == expected;
        let false_concurrency = claim.claimed_relation == "concurrent" && expected != "concurrent";
        let unsupported_causal = ["before", "after"].contains(&claim.claimed_relation.as_str())
            && !truth.possible_relations.contains(&claim.claimed_relation);
        let unnecessary_abstention = claim.claimed_relation == "underdetermined"
            && ["before", "after", "concurrent"].contains(&expected.as_str());
        let future: Vec<String> = claim
            .evidence_event_ids
            .iter()
            .filter(|event_id| {
                events
                    .get(*event_id)
                    .is_some_and(|event| event.available_at > claim.decision_cutoff)
            })
            .cloned()
            .collect();
        future_evidence_leakage_count += future.len();
        if false_concurrency {
            false_concurrency_claim_count += 1;
        }
        if unsupported_causal {
            unsupported_causal_claim_count += 1;
        }
        if unnecessary_abstention {
            unnecessary_abstention_count += 1;
        }
        let critical_unsupported = claim.critical
            && ["before", "after", "concurrent"].contains(&claim.claimed_relation.as_str())
            && !correct;
        if critical_unsupported {
            critical_unsupported_claim_count += 1;
        }

        pair_outcomes.push(GraphPairOutcome {
            left_event_id: left.clone(),
            right_event_id: right.clone(),
            possible_relations: truth.possible_relations.iter().cloned().collect(),
            expected_state: expected,
            claimed_relation: claim.claimed_relation.clone(),
            individually_plausible,
            correct,
            false_concurrency,
            unsupported_causal_claim: unsupported_causal,
            unnecessary_abstention,
            future_evidence_event_ids: future,
            world_relations: truth.world_relations.clone(),
        });
    }

    let jointly_consistent_world_ids: Vec<String> = worlds
        .iter()
        .filter(|world| world_satisfies_claims(world, &claims))
        .map(|world| world.world_id.clone())
        .collect();
    let pair_count = event_ids.len() * (event_ids.len() - 1) / 2;

    Ok(WholeDagTraceReport {
        trace_id: required_str(trace, "trace_id", "candidate_trace")?.to_owned(),
        observation_boundary_claim: boundary_claim.clone(),
        observation_boundary_correct: boundary_claim == expected_boundary_state,
        pair_count,
        individually_plausible_claim_count,
        jointly_consistent_world_count: jointly_consistent_world_ids.len(),
        jointly_impossible_claim_set: jointly_consistent_world_ids.is_empty(),
        jointly_consistent_world_ids,
        false_concurrency_claim_count,
        unsupported_causal_claim_count,
        unnecessary_abstention_count,
        future_evidence_leakage_count,
        critical_unsupported_claim_count,
        pair_outcomes,
    })
}

/// Evaluate a D6.13 bundle using complete admissible DAG worlds.
/// The evaluator-only world list must never be exposed to candidate agents.
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
    allowed_keys(
        candidate,
        &[
            "scenario_id",
            "observation_boundary_state",
            "events",
            "trusted_happens_before_edges",
            "forbidden_ordered_pairs",
        ],
        "candidate_view",
    )?;
    let scenario_id = required_str(candidate, "scenario_id", "candidate_view")?.to_owned();
    let observation_boundary_state = required_str(candidate, "observation_boundary_state", "candidate_view")?.to_owned();
    if !BOUNDARIES.contains(&observation_boundary_state.as_str()) {
        return Err("candidate_view.observation_boundary_state is unsupported".to_owned());
    }
    let (events, event_ids) = parse_events(candidate)?;
    let gold = bundle
        .get("gold_overlay")
        .ok_or_else(|| "gold_overlay is required".to_owned())?;
    allowed_keys(
        gold,
        &["adjudication_status", "adjudicated_observation_boundary_state", "admissible_worlds"],
        "gold_overlay",
    )?;
    let adjudication_status = required_str(gold, "adjudication_status", "gold_overlay")?.to_owned();
    if adjudication_status != "synthetic_known_answer_not_independently_adjudicated" {
        return Err(format!("unsupported D6.13 v0.1 adjudication status {adjudication_status:?}"));
    }
    let expected_boundary_state =
        required_str(gold, "adjudicated_observation_boundary_state", "gold_overlay")?;
    if !BOUNDARIES.contains(&expected_boundary_state) {
        return Err("gold_overlay.adjudicated_observation_boundary_state is unsupported".to_owned());
    }
    if expected_boundary_state != observation_boundary_state {
        return Err("candidate-visible observation boundary state disagrees with evaluator adjudication".to_owned());
    }

    let worlds = parse_worlds(gold, &event_ids)?;
    let (trusted_edge_count, forbidden_pair_count) =
        validate_world_constraints(candidate, &event_ids, &worlds)?;
    let pair_truths = build_pair_truths(&event_ids, &worlds, expected_boundary_state);

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
            return Err("candidate trace identifiers must be unique".to_owned());
        }
        reports.push(score_trace(
            trace,
            &events,
            &event_ids,
            &worlds,
            &pair_truths,
            expected_boundary_state,
        )?);
    }
    reports.sort_by(|left, right| left.trace_id.cmp(&right.trace_id));

    Ok(WholeDagReport {
        scenario_id,
        adjudication_status,
        event_count: events.len(),
        admissible_world_count: worlds.len(),
        pair_count: pair_truths.len(),
        trusted_edge_count,
        forbidden_pair_count,
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
    fn detects_jointly_impossible_but_individually_plausible_cycle_claims() {
        let report = evaluate_whole_dag_json(FIXTURE).expect("D6.13 fixture");
        let adversarial = trace(&report, "jointly-impossible-cycle");
        assert_eq!(adversarial.individually_plausible_claim_count, 10);
        assert!(adversarial.jointly_impossible_claim_set);
        assert_eq!(adversarial.jointly_consistent_world_count, 0);
    }

    #[test]
    fn preserves_entailment_across_all_whole_worlds() {
        let report = evaluate_whole_dag_json(FIXTURE).expect("D6.13 fixture");
        let calibrated = trace(&report, "world-consistent-agent");
        let ef = calibrated.pair_outcomes.iter().find(|pair| {
            pair.left_event_id == "E" && pair.right_event_id == "F"
        }).expect("E/F pair");
        assert_eq!(ef.possible_relations, vec!["before".to_owned()]);
        assert_eq!(ef.expected_state, "before");
        assert!(ef.correct);
    }

    #[test]
    fn pairwise_ambiguity_keeps_whole_world_correlation() {
        let report = evaluate_whole_dag_json(FIXTURE).expect("D6.13 fixture");
        let calibrated = trace(&report, "world-consistent-agent");
        for (left, right) in [("A", "B"), ("A", "C"), ("B", "C")] {
            let pair = calibrated.pair_outcomes.iter().find(|item| {
                item.left_event_id == left && item.right_event_id == right
            }).expect("ambiguous pair");
            assert_eq!(pair.expected_state, "underdetermined");
            assert_eq!(pair.possible_relations.len(), 2);
        }
    }

    #[test]
    fn incomplete_boundary_does_not_prove_concurrency() {
        let mut value = fixture_value();
        value["candidate_view"]["observation_boundary_state"] = json!("incomplete");
        value["gold_overlay"]["adjudicated_observation_boundary_state"] = json!("incomplete");
        let trace_value = value["candidate_traces"].as_array_mut().expect("traces")
            .iter_mut().find(|trace| trace["trace_id"] == "world-consistent-agent")
            .expect("trace");
        let claim = trace_value["relation_claims"].as_array_mut().expect("claims")
            .iter_mut().find(|claim| claim["left_event_id"] == "A" && claim["right_event_id"] == "E")
            .expect("A/E claim");
        claim["claimed_relation"] = json!("underdetermined");
        let report = report_for(&value);
        let pair = trace(&report, "world-consistent-agent").pair_outcomes.iter()
            .find(|item| item.left_event_id == "A" && item.right_event_id == "E")
            .expect("A/E pair");
        assert_eq!(pair.expected_state, "underdetermined");
        assert!(pair.correct);
    }

    #[test]
    fn rejects_empty_admissible_world_set_without_vacuous_truth() {
        let mut value = fixture_value();
        value["gold_overlay"]["admissible_worlds"] = json!([]);
        let error = evaluate_whole_dag_json(&value.to_string()).unwrap_err();
        assert!(error.contains("empty-world vacuous truth is forbidden"));
    }

    #[test]
    fn rejects_cyclic_world() {
        let mut value = fixture_value();
        value["gold_overlay"]["admissible_worlds"][0]["edges"] =
            json!([["A","B"],["B","C"],["C","A"],["E","F"]]);
        let error = evaluate_whole_dag_json(&value.to_string()).unwrap_err();
        assert!(error.contains("contains a directed cycle"));
    }

    #[test]
    fn rejects_world_violating_a_trusted_relation() {
        let mut value = fixture_value();
        value["gold_overlay"]["admissible_worlds"][0]["edges"] = json!([["A","B"],["B","C"]]);
        let error = evaluate_whole_dag_json(&value.to_string()).unwrap_err();
        assert!(error.contains("violates trusted happens-before relation"));
    }

    #[test]
    fn rejects_world_violating_forbidden_unordered_pair() {
        let mut value = fixture_value();
        value["gold_overlay"]["admissible_worlds"][0]["edges"] =
            json!([["A","E"],["E","F"],["A","B"],["B","C"]]);
        let error = evaluate_whole_dag_json(&value.to_string()).unwrap_err();
        assert!(error.contains("violates forbidden unordered pair"));
    }

    #[test]
    fn rejects_candidate_trace_missing_a_pair_claim() {
        let mut value = fixture_value();
        value["candidate_traces"][0]["relation_claims"].as_array_mut().expect("claims").pop();
        let error = evaluate_whole_dag_json(&value.to_string()).unwrap_err();
        assert!(error.contains("must claim every event pair exactly once"));
    }

    #[test]
    fn rejects_unknown_nested_fields() {
        let mut value = fixture_value();
        value["gold_overlay"]["admissible_worlds"][0]["hidden_edge"] = json!(["A","B"]);
        let error = evaluate_whole_dag_json(&value.to_string()).unwrap_err();
        assert!(error.contains("unknown field"));
    }

    #[test]
    fn detects_future_evidence_used_in_pair_decision() {
        let report = evaluate_whole_dag_json(FIXTURE).expect("D6.13 fixture");
        let adversarial = trace(&report, "jointly-impossible-cycle");
        assert_eq!(adversarial.future_evidence_leakage_count, 1);
    }

    #[test]
    fn rejects_duplicate_world_ids() {
        let mut value = fixture_value();
        value["gold_overlay"]["admissible_worlds"][1]["world_id"] = json!("world-forward");
        let error = evaluate_whole_dag_json(&value.to_string()).unwrap_err();
        assert!(error.contains("duplicate world_id"));
    }

    #[test]
    fn reports_wrong_observation_boundary_claim() {
        let mut value = fixture_value();
        value["candidate_traces"][0]["observation_boundary_claim"] = json!("incomplete");
        let report = report_for(&value);
        assert!(!trace(&report, "world-consistent-agent").observation_boundary_correct);
    }
}
