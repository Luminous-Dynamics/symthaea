//! Evaluator-only D6.11 scorer for causal identifiability under incomplete
//! or compromised event observation.
//!
//! The evaluator scores claims against a finite set of independently supplied
//! admissible pair relations. Missing telemetry is not treated as proof of
//! concurrency. Candidate-visible sequence coverage, attestation state, and
//! decision cutoffs are checked separately from evaluator-side admissible
//! histories. This module does not authenticate real signatures, prove that
//! the admissible-history set is complete, or execute candidate code.

use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, FixedOffset};
use serde::Serialize;
use serde_json::Value;

const CLAIM_CEILING: &str = "Synthetic known-answer evaluation of causal identifiability under declared observation boundaries and finite admissible histories. Does not authenticate real attestations, prove observation-universe completeness, establish universal causal reconstruction, or measure agent performance.";

const RELATIONS: &[&str] = &["before", "after", "concurrent"];
const CLAIMS: &[&str] = &["before", "after", "concurrent", "underdetermined", "compromised"];
const BOUNDARY_STATES: &[&str] = &["complete", "gap", "compromised", "unknown"];

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct BoundaryOutcome {
    pub producer_id: String,
    pub expected_state: String,
    pub claimed_state: String,
    pub correct: bool,
    pub missing_sequence_numbers: Vec<u64>,
    pub false_completeness_claim: bool,
    pub false_incompleteness_claim: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct PairOutcome {
    pub pair_id: String,
    pub expected_state: String,
    pub admissible_relations: Vec<String>,
    pub claimed_relation: String,
    pub correct: bool,
    pub false_concurrency: bool,
    pub unsupported_causal_claim: bool,
    pub unnecessary_abstention: bool,
    pub missed_compromise: bool,
    pub future_evidence_event_ids: Vec<String>,
    pub critical_unsupported_claim: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct IdentifiabilityTraceReport {
    pub trace_id: String,
    pub boundary_outcomes: Vec<BoundaryOutcome>,
    pub pair_outcomes: Vec<PairOutcome>,
    pub identifiable_pair_count: usize,
    pub correctly_identified_pair_count: usize,
    pub underdetermined_pair_count: usize,
    pub correctly_abstained_pair_count: usize,
    pub compromised_pair_count: usize,
    pub correctly_detected_compromised_pair_count: usize,
    pub false_concurrency_claim_count: usize,
    pub unsupported_causal_claim_count: usize,
    pub unnecessary_abstention_count: usize,
    pub missed_compromise_count: usize,
    pub false_completeness_claim_count: usize,
    pub false_incompleteness_claim_count: usize,
    pub future_evidence_leakage_count: usize,
    pub critical_unsupported_claim_count: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct IdentifiabilityReport {
    pub scenario_id: String,
    pub adjudication_status: String,
    pub event_count: usize,
    pub boundary_count: usize,
    pub pair_count: usize,
    pub traces: Vec<IdentifiabilityTraceReport>,
    pub claim_ceiling: String,
}

#[derive(Debug, Clone)]
struct Event {
    producer_id: String,
    sequence_number: u64,
    available_at: DateTime<FixedOffset>,
}

#[derive(Debug, Clone)]
struct Boundary {
    expected_sequence_numbers: BTreeSet<u64>,
    observed_sequence_numbers: BTreeSet<u64>,
    attested_complete: bool,
    signature_status: String,
}

#[derive(Debug, Clone)]
struct PairTask {
    left_event_id: String,
    right_event_id: String,
    relevant_producers: BTreeSet<String>,
    decision_cutoff: DateTime<FixedOffset>,
    critical: bool,
}

#[derive(Debug, Clone)]
struct PairTruth {
    admissible_relations: BTreeSet<String>,
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

fn parse_u64_set(value: &Value, key: &str, label: &str) -> Result<BTreeSet<u64>, String> {
    let values = value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{label}.{key} must be an array"))?;
    let mut output = BTreeSet::new();
    for (index, item) in values.iter().enumerate() {
        let number = item
            .as_u64()
            .ok_or_else(|| format!("{label}.{key}[{index}] must be an unsigned integer"))?;
        if !output.insert(number) {
            return Err(format!("{label}.{key} contains duplicate sequence number {number}"));
        }
    }
    Ok(output)
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

fn valid_digest(raw: &str) -> bool {
    raw.len() == 64
        && raw
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn digest(value: &Value, key: &str, label: &str) -> Result<(), String> {
    let raw = required_str(value, key, label)?;
    if !valid_digest(raw) {
        return Err(format!(
            "{label}.{key} must be a lowercase 64-character hexadecimal digest"
        ));
    }
    Ok(())
}

fn derive_boundary_state(boundary: &Boundary) -> (String, Vec<u64>) {
    let missing: Vec<u64> = boundary
        .expected_sequence_numbers
        .difference(&boundary.observed_sequence_numbers)
        .copied()
        .collect();
    if boundary.signature_status == "invalid" {
        return ("compromised".to_owned(), missing);
    }
    if !missing.is_empty() {
        return ("gap".to_owned(), missing);
    }
    if boundary.expected_sequence_numbers.is_empty()
        || !boundary.attested_complete
        || boundary.signature_status != "valid"
    {
        return ("unknown".to_owned(), missing);
    }
    ("complete".to_owned(), missing)
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
        allowed_keys(
            value,
            &["event_id", "producer_id", "sequence_number", "event_time", "available_at"],
            &label,
        )?;
        let id = required_str(value, "event_id", &label)?.to_owned();
        let producer_id = required_str(value, "producer_id", &label)?.to_owned();
        let sequence_number = value
            .get("sequence_number")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("{label}.sequence_number must be an unsigned integer"))?;
        let _event_time = parse_time(value, "event_time", &label)?;
        let available_at = parse_time(value, "available_at", &label)?;
        if output
            .insert(
                id.clone(),
                Event {
                    producer_id,
                    sequence_number,
                    available_at,
                },
            )
            .is_some()
        {
            return Err(format!("duplicate event_id {id:?}"));
        }
    }
    Ok(output)
}

fn parse_boundaries(candidate: &Value) -> Result<BTreeMap<String, Boundary>, String> {
    let values = candidate
        .get("observation_boundaries")
        .and_then(Value::as_array)
        .ok_or_else(|| "candidate_view.observation_boundaries must be an array".to_owned())?;
    let mut output = BTreeMap::new();
    for (index, value) in values.iter().enumerate() {
        let label = format!("candidate_view.observation_boundaries[{index}]");
        allowed_keys(
            value,
            &[
                "producer_id",
                "expected_sequence_numbers",
                "observed_sequence_numbers",
                "attested_complete",
                "signature_status",
                "issuer_key_digest",
                "observer_identity_digest",
            ],
            &label,
        )?;
        let producer_id = required_str(value, "producer_id", &label)?.to_owned();
        let expected_sequence_numbers = parse_u64_set(value, "expected_sequence_numbers", &label)?;
        let observed_sequence_numbers = parse_u64_set(value, "observed_sequence_numbers", &label)?;
        if !expected_sequence_numbers.is_empty()
            && observed_sequence_numbers
                .iter()
                .any(|number| !expected_sequence_numbers.contains(number))
        {
            return Err(format!(
                "{label}.observed_sequence_numbers contains a number outside the expected coverage range"
            ));
        }
        let attested_complete = required_bool(value, "attested_complete", &label)?;
        let signature_status = required_str(value, "signature_status", &label)?.to_owned();
        if !["valid", "invalid", "unknown"].contains(&signature_status.as_str()) {
            return Err(format!("{label}.signature_status is unsupported"));
        }
        digest(value, "issuer_key_digest", &label)?;
        digest(value, "observer_identity_digest", &label)?;
        if output
            .insert(
                producer_id.clone(),
                Boundary {
                    expected_sequence_numbers,
                    observed_sequence_numbers,
                    attested_complete,
                    signature_status,
                },
            )
            .is_some()
        {
            return Err(format!("duplicate observation boundary {producer_id:?}"));
        }
    }
    if output.is_empty() {
        return Err("candidate_view.observation_boundaries must not be empty".to_owned());
    }
    Ok(output)
}

fn parse_pair_tasks(
    candidate: &Value,
    events: &BTreeMap<String, Event>,
    boundaries: &BTreeMap<String, Boundary>,
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
                "relevant_producers",
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
        let left = events
            .get(&left_event_id)
            .ok_or_else(|| format!("{label} references unknown left event {left_event_id:?}"))?;
        let right = events
            .get(&right_event_id)
            .ok_or_else(|| format!("{label} references unknown right event {right_event_id:?}"))?;
        let relevant_producers = string_set(value, "relevant_producers", &label)?;
        if relevant_producers.is_empty()
            || !relevant_producers.contains(&left.producer_id)
            || !relevant_producers.contains(&right.producer_id)
        {
            return Err(format!(
                "{label}.relevant_producers must include both event producers"
            ));
        }
        for producer in &relevant_producers {
            if !boundaries.contains_key(producer) {
                return Err(format!("{label} references missing observation boundary {producer:?}"));
            }
        }
        let decision_cutoff = parse_time(value, "decision_cutoff", &label)?;
        let critical = required_bool(value, "critical", &label)?;
        if output
            .insert(
                pair_id.clone(),
                PairTask {
                    left_event_id,
                    right_event_id,
                    relevant_producers,
                    decision_cutoff,
                    critical,
                },
            )
            .is_some()
        {
            return Err(format!("duplicate pair_id {pair_id:?}"));
        }
    }
    Ok(output)
}

fn parse_boundary_truths(
    gold: &Value,
    boundaries: &BTreeMap<String, Boundary>,
) -> Result<BTreeMap<String, String>, String> {
    let values = gold
        .get("boundary_truths")
        .and_then(Value::as_array)
        .ok_or_else(|| "gold_overlay.boundary_truths must be an array".to_owned())?;
    let mut output = BTreeMap::new();
    for (index, value) in values.iter().enumerate() {
        let label = format!("gold_overlay.boundary_truths[{index}]");
        allowed_keys(value, &["producer_id", "expected_state"], &label)?;
        let producer_id = required_str(value, "producer_id", &label)?.to_owned();
        let expected_state = required_str(value, "expected_state", &label)?.to_owned();
        if !BOUNDARY_STATES.contains(&expected_state.as_str()) {
            return Err(format!("{label}.expected_state is unsupported"));
        }
        let boundary = boundaries
            .get(&producer_id)
            .ok_or_else(|| format!("{label} references unknown boundary {producer_id:?}"))?;
        let (derived_state, _) = derive_boundary_state(boundary);
        if derived_state != expected_state {
            return Err(format!(
                "{label} disagrees with candidate-visible coverage/attestation facts: derived {derived_state:?}, declared {expected_state:?}"
            ));
        }
        if output.insert(producer_id.clone(), expected_state).is_some() {
            return Err(format!("duplicate boundary truth for {producer_id:?}"));
        }
    }
    if output.len() != boundaries.len()
        || boundaries.keys().any(|producer| !output.contains_key(producer))
    {
        return Err("gold_overlay.boundary_truths must cover every observation boundary exactly once".to_owned());
    }
    Ok(output)
}

fn parse_pair_truths(
    gold: &Value,
    pair_tasks: &BTreeMap<String, PairTask>,
    events: &BTreeMap<String, Event>,
    boundary_truths: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, PairTruth>, String> {
    let values = gold
        .get("pair_truths")
        .and_then(Value::as_array)
        .ok_or_else(|| "gold_overlay.pair_truths must be an array".to_owned())?;
    let mut output = BTreeMap::new();
    for (index, value) in values.iter().enumerate() {
        let label = format!("gold_overlay.pair_truths[{index}]");
        allowed_keys(value, &["pair_id", "admissible_relations"], &label)?;
        let pair_id = required_str(value, "pair_id", &label)?.to_owned();
        if !pair_tasks.contains_key(&pair_id) {
            return Err(format!("{label} references unknown pair {pair_id:?}"));
        }
        let admissible_relations = string_set(value, "admissible_relations", &label)?;
        if admissible_relations.is_empty()
            || admissible_relations
                .iter()
                .any(|relation| !RELATIONS.contains(&relation.as_str()))
        {
            return Err(format!("{label}.admissible_relations must be a non-empty subset of before/after/concurrent"));
        }
        if output
            .insert(
                pair_id.clone(),
                PairTruth {
                    admissible_relations,
                },
            )
            .is_some()
        {
            return Err(format!("duplicate pair truth {pair_id:?}"));
        }
    }
    if output.len() != pair_tasks.len()
        || pair_tasks.keys().any(|pair_id| !output.contains_key(pair_id))
    {
        return Err("gold_overlay.pair_truths must cover every pair task exactly once".to_owned());
    }
    for (pair_id, task) in pair_tasks {
        let truth = output
            .get(pair_id)
            .ok_or_else(|| format!("missing pair truth {pair_id:?}"))?;
        let left_event = events
            .get(&task.left_event_id)
            .ok_or_else(|| format!("pair {pair_id:?} references missing left event"))?;
        let right_event = events
            .get(&task.right_event_id)
            .ok_or_else(|| format!("pair {pair_id:?} references missing right event"))?;
        if left_event.producer_id == right_event.producer_id
            && boundary_truths.get(&left_event.producer_id).is_some_and(|state| state != "compromised")
            && left_event.sequence_number != right_event.sequence_number
        {
            let sequence_relation = if left_event.sequence_number < right_event.sequence_number {
                "before"
            } else {
                "after"
            };
            if truth.admissible_relations.len() != 1
                || !truth.admissible_relations.contains(sequence_relation)
            {
                return Err(format!(
                    "pair {pair_id:?} contradicts trusted same-producer sequence ordering: expected {sequence_relation:?}"
                ));
            }
        }
        let has_compromised = task
            .relevant_producers
            .iter()
            .any(|producer| boundary_truths.get(producer).is_some_and(|state| state == "compromised"));
        let has_unqualified_boundary = task
            .relevant_producers
            .iter()
            .any(|producer| boundary_truths.get(producer).is_some_and(|state| state != "complete"));
        if truth.admissible_relations.len() == 1
            && truth.admissible_relations.contains("concurrent")
            && has_unqualified_boundary
            && !has_compromised
        {
            return Err(format!(
                "pair {pair_id:?} cannot have identifiable concurrency with an incomplete/unknown observation boundary"
            ));
        }
    }
    Ok(output)
}

fn expected_pair_state(
    task: &PairTask,
    truth: &PairTruth,
    boundary_truths: &BTreeMap<String, String>,
) -> String {
    if task
        .relevant_producers
        .iter()
        .any(|producer| boundary_truths.get(producer).is_some_and(|state| state == "compromised"))
    {
        return "compromised".to_owned();
    }
    if truth.admissible_relations.len() != 1 {
        return "underdetermined".to_owned();
    }
    truth
        .admissible_relations
        .iter()
        .next()
        .cloned()
        .unwrap_or_else(|| "underdetermined".to_owned())
}

fn score_trace(
    trace: &Value,
    events: &BTreeMap<String, Event>,
    boundaries: &BTreeMap<String, Boundary>,
    boundary_truths: &BTreeMap<String, String>,
    pair_tasks: &BTreeMap<String, PairTask>,
    pair_truths: &BTreeMap<String, PairTruth>,
) -> Result<IdentifiabilityTraceReport, String> {
    allowed_keys(
        trace,
        &["trace_id", "boundary_claims", "pair_claims"],
        "candidate_trace",
    )?;
    let trace_id = required_str(trace, "trace_id", "candidate_trace")?.to_owned();

    let boundary_claim_values = trace
        .get("boundary_claims")
        .and_then(Value::as_array)
        .ok_or_else(|| "candidate_trace.boundary_claims must be an array".to_owned())?;
    let mut boundary_claims = BTreeMap::new();
    for (index, value) in boundary_claim_values.iter().enumerate() {
        let label = format!("candidate_trace.boundary_claims[{index}]");
        allowed_keys(value, &["producer_id", "claimed_state"], &label)?;
        let producer_id = required_str(value, "producer_id", &label)?.to_owned();
        let claimed_state = required_str(value, "claimed_state", &label)?.to_owned();
        if !BOUNDARY_STATES.contains(&claimed_state.as_str()) {
            return Err(format!("{label}.claimed_state is unsupported"));
        }
        if boundary_claims.insert(producer_id.clone(), claimed_state).is_some() {
            return Err(format!("duplicate boundary claim {producer_id:?}"));
        }
    }
    if boundary_claims.len() != boundaries.len()
        || boundaries.keys().any(|producer| !boundary_claims.contains_key(producer))
    {
        return Err(format!(
            "candidate trace {trace_id:?} must claim every observation boundary exactly once"
        ));
    }

    let mut boundary_outcomes = Vec::new();
    for producer_id in boundaries.keys() {
        let expected_state = boundary_truths
            .get(producer_id)
            .ok_or_else(|| format!("missing boundary truth {producer_id:?}"))?;
        let claimed_state = boundary_claims
            .get(producer_id)
            .ok_or_else(|| format!("missing boundary claim {producer_id:?}"))?;
        let boundary = boundaries
            .get(producer_id)
            .ok_or_else(|| format!("missing boundary {producer_id:?}"))?;
        let (_, missing_sequence_numbers) = derive_boundary_state(boundary);
        let false_completeness_claim = claimed_state == "complete" && expected_state != "complete";
        let false_incompleteness_claim = claimed_state != "complete" && expected_state == "complete";
        boundary_outcomes.push(BoundaryOutcome {
            producer_id: producer_id.clone(),
            expected_state: expected_state.clone(),
            claimed_state: claimed_state.clone(),
            correct: expected_state == claimed_state,
            missing_sequence_numbers,
            false_completeness_claim,
            false_incompleteness_claim,
        });
    }

    let pair_claim_values = trace
        .get("pair_claims")
        .and_then(Value::as_array)
        .ok_or_else(|| "candidate_trace.pair_claims must be an array".to_owned())?;
    let mut pair_claims = BTreeMap::new();
    for (index, value) in pair_claim_values.iter().enumerate() {
        let label = format!("candidate_trace.pair_claims[{index}]");
        allowed_keys(
            value,
            &["pair_id", "claimed_relation", "evidence_event_ids"],
            &label,
        )?;
        let pair_id = required_str(value, "pair_id", &label)?.to_owned();
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
        if pair_claims
            .insert(pair_id.clone(), (claimed_relation, evidence_event_ids))
            .is_some()
        {
            return Err(format!("duplicate pair claim {pair_id:?}"));
        }
    }
    if pair_claims.len() != pair_tasks.len()
        || pair_tasks.keys().any(|pair_id| !pair_claims.contains_key(pair_id))
    {
        return Err(format!(
            "candidate trace {trace_id:?} must claim every pair task exactly once"
        ));
    }

    let mut pair_outcomes = Vec::new();
    for (pair_id, task) in pair_tasks {
        let (claimed_relation, evidence_event_ids) = pair_claims
            .get(pair_id)
            .ok_or_else(|| format!("missing pair claim {pair_id:?}"))?;
        let truth = pair_truths
            .get(pair_id)
            .ok_or_else(|| format!("missing pair truth {pair_id:?}"))?;
        let expected_state = expected_pair_state(task, truth, boundary_truths);
        let mut future_evidence_event_ids = Vec::new();
        for event_id in evidence_event_ids {
            let event = events
                .get(event_id)
                .ok_or_else(|| format!("missing event {event_id:?}"))?;
            if event.available_at > task.decision_cutoff {
                future_evidence_event_ids.push(event_id.clone());
            }
        }
        future_evidence_event_ids.sort();
        let correct = claimed_relation == &expected_state;
        let false_concurrency = claimed_relation == "concurrent" && expected_state != "concurrent";
        let unsupported_causal_claim = ["before", "after"].contains(&claimed_relation.as_str())
            && claimed_relation != &expected_state;
        let unnecessary_abstention = claimed_relation == "underdetermined"
            && ["before", "after", "concurrent"].contains(&expected_state.as_str());
        let missed_compromise = expected_state == "compromised" && claimed_relation != "compromised";
        let critical_unsupported_claim = task.critical
            && ["before", "after", "concurrent"].contains(&claimed_relation.as_str())
            && !correct;
        pair_outcomes.push(PairOutcome {
            pair_id: pair_id.clone(),
            expected_state: expected_state.clone(),
            admissible_relations: truth.admissible_relations.iter().cloned().collect(),
            claimed_relation: claimed_relation.clone(),
            correct,
            false_concurrency,
            unsupported_causal_claim,
            unnecessary_abstention,
            missed_compromise,
            future_evidence_event_ids,
            critical_unsupported_claim,
        });
    }

    let identifiable_pair_count = pair_outcomes
        .iter()
        .filter(|item| ["before", "after", "concurrent"].contains(&item.expected_state.as_str()))
        .count();
    let correctly_identified_pair_count = pair_outcomes
        .iter()
        .filter(|item| {
            ["before", "after", "concurrent"].contains(&item.expected_state.as_str()) && item.correct
        })
        .count();
    let underdetermined_pair_count = pair_outcomes
        .iter()
        .filter(|item| item.expected_state == "underdetermined")
        .count();
    let correctly_abstained_pair_count = pair_outcomes
        .iter()
        .filter(|item| item.expected_state == "underdetermined" && item.correct)
        .count();
    let compromised_pair_count = pair_outcomes
        .iter()
        .filter(|item| item.expected_state == "compromised")
        .count();
    let correctly_detected_compromised_pair_count = pair_outcomes
        .iter()
        .filter(|item| item.expected_state == "compromised" && item.correct)
        .count();

    Ok(IdentifiabilityTraceReport {
        trace_id,
        false_concurrency_claim_count: pair_outcomes.iter().filter(|item| item.false_concurrency).count(),
        unsupported_causal_claim_count: pair_outcomes.iter().filter(|item| item.unsupported_causal_claim).count(),
        unnecessary_abstention_count: pair_outcomes.iter().filter(|item| item.unnecessary_abstention).count(),
        missed_compromise_count: pair_outcomes.iter().filter(|item| item.missed_compromise).count(),
        future_evidence_leakage_count: pair_outcomes.iter().map(|item| item.future_evidence_event_ids.len()).sum(),
        critical_unsupported_claim_count: pair_outcomes.iter().filter(|item| item.critical_unsupported_claim).count(),
        false_completeness_claim_count: boundary_outcomes.iter().filter(|item| item.false_completeness_claim).count(),
        false_incompleteness_claim_count: boundary_outcomes.iter().filter(|item| item.false_incompleteness_claim).count(),
        boundary_outcomes,
        pair_outcomes,
        identifiable_pair_count,
        correctly_identified_pair_count,
        underdetermined_pair_count,
        correctly_abstained_pair_count,
        compromised_pair_count,
        correctly_detected_compromised_pair_count,
    })
}

/// Score a D6.11 bundle. The input contains evaluator-only admissible histories
/// and must never be copied into a candidate workspace.
pub fn evaluate_identifiability_json(input: &str) -> Result<IdentifiabilityReport, String> {
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
        return Err("D6.11 evaluator bundle must be explicitly evaluator-only".to_owned());
    }
    let candidate = bundle
        .get("candidate_view")
        .ok_or_else(|| "candidate_view is required".to_owned())?;
    allowed_keys(candidate, &["scenario_id", "events", "observation_boundaries", "pair_tasks"], "candidate_view")?;
    let scenario_id = required_str(candidate, "scenario_id", "candidate_view")?.to_owned();
    let events = parse_events(candidate)?;
    let boundaries = parse_boundaries(candidate)?;
    let pair_tasks = parse_pair_tasks(candidate, &events, &boundaries)?;

    let gold = bundle
        .get("gold_overlay")
        .ok_or_else(|| "gold_overlay is required".to_owned())?;
    allowed_keys(gold, &["adjudication_status", "boundary_truths", "pair_truths"], "gold_overlay")?;
    let adjudication_status = required_str(gold, "adjudication_status", "gold_overlay")?.to_owned();
    if adjudication_status != "synthetic_known_answer_not_independently_adjudicated" {
        return Err(format!("unsupported D6.11 v0.1 adjudication status {adjudication_status:?}"));
    }
    let boundary_truths = parse_boundary_truths(gold, &boundaries)?;
    let pair_truths = parse_pair_truths(gold, &pair_tasks, &events, &boundary_truths)?;

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
            &boundaries,
            &boundary_truths,
            &pair_tasks,
            &pair_truths,
        )?);
    }
    reports.sort_by(|left, right| left.trace_id.cmp(&right.trace_id));

    Ok(IdentifiabilityReport {
        scenario_id,
        adjudication_status,
        event_count: events.len(),
        boundary_count: boundaries.len(),
        pair_count: pair_tasks.len(),
        traces: reports,
        claim_ceiling: CLAIM_CEILING.to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const FIXTURE: &str = include_str!(
        "../../../docs/development-map/fixtures/egddm-identifiability-v0.1.json"
    );
    const EXPECTED: &str = include_str!(
        "../../../docs/development-map/fixtures/egddm-identifiability-v0.1.expected.json"
    );

    fn fixture_value() -> Value {
        serde_json::from_str(FIXTURE).expect("D6.11 fixture JSON")
    }

    fn report_for(value: &Value) -> IdentifiabilityReport {
        evaluate_identifiability_json(&value.to_string()).expect("D6.11 score")
    }

    fn trace<'a>(report: &'a IdentifiabilityReport, id: &str) -> &'a IdentifiabilityTraceReport {
        report.traces.iter().find(|item| item.trace_id == id).expect("trace")
    }

    #[test]
    fn matches_complete_machine_readable_known_answer() {
        let actual = evaluate_identifiability_json(FIXTURE).expect("D6.11 fixture");
        let actual_value = serde_json::to_value(actual).expect("serialize report");
        let expected_value: Value = serde_json::from_str(EXPECTED).expect("expected report");
        assert_eq!(actual_value, expected_value);
    }

    #[test]
    fn rejects_false_concurrency_when_a_sequence_gap_exists() {
        let report = evaluate_identifiability_json(FIXTURE).expect("D6.11 fixture");
        let adversarial = trace(&report, "adversarial-closed-world");
        assert_eq!(adversarial.false_concurrency_claim_count, 1);
        assert!(adversarial.pair_outcomes.iter().any(|item| {
            item.pair_id == "gap-ambiguous"
                && item.expected_state == "underdetermined"
                && item.false_concurrency
        }));
    }

    #[test]
    fn preserves_true_concurrency_when_both_boundaries_are_complete() {
        let report = evaluate_identifiability_json(FIXTURE).expect("D6.11 fixture");
        let conformant = trace(&report, "calibrated-agent");
        assert!(conformant.pair_outcomes.iter().any(|item| {
            item.pair_id == "certified-concurrent"
                && item.expected_state == "concurrent"
                && item.correct
        }));
    }

    #[test]
    fn distinguishes_under_determination_from_observer_compromise() {
        let report = evaluate_identifiability_json(FIXTURE).expect("D6.11 fixture");
        let conformant = trace(&report, "calibrated-agent");
        assert!(conformant.pair_outcomes.iter().any(|item| {
            item.pair_id == "gap-ambiguous" && item.expected_state == "underdetermined" && item.correct
        }));
        assert!(conformant.pair_outcomes.iter().any(|item| {
            item.pair_id == "compromised-stream" && item.expected_state == "compromised" && item.correct
        }));
    }

    #[test]
    fn scores_unnecessary_abstention_on_identifiable_positive_controls() {
        let report = evaluate_identifiability_json(FIXTURE).expect("D6.11 fixture");
        let always = trace(&report, "always-abstain");
        assert_eq!(always.unnecessary_abstention_count, 3);
        assert_eq!(always.correctly_identified_pair_count, 0);
    }

    #[test]
    fn catches_backdated_late_evidence_used_before_availability() {
        let report = evaluate_identifiability_json(FIXTURE).expect("D6.11 fixture");
        let adversarial = trace(&report, "adversarial-closed-world");
        assert_eq!(adversarial.future_evidence_leakage_count, 1);
        assert!(adversarial.pair_outcomes.iter().any(|item| {
            item.pair_id == "late-ambiguous"
                && item.future_evidence_event_ids == vec!["late-arriving-event".to_owned()]
        }));
    }

    #[test]
    fn detects_false_completeness_claim_when_sequence_evidence_has_a_gap() {
        let report = evaluate_identifiability_json(FIXTURE).expect("D6.11 fixture");
        let adversarial = trace(&report, "adversarial-closed-world");
        assert!(adversarial.false_completeness_claim_count >= 1);
        assert!(adversarial.boundary_outcomes.iter().any(|item| {
            item.producer_id == "stream-gap"
                && item.expected_state == "gap"
                && item.claimed_state == "complete"
                && item.false_completeness_claim
                && item.missing_sequence_numbers == vec![2]
        }));
    }

    #[test]
    fn rejects_gold_claiming_identifiable_concurrency_without_complete_coverage() {
        let mut value = fixture_value();
        value["gold_overlay"]["pair_truths"][2]["admissible_relations"] = json!(["concurrent"]);
        let error = evaluate_identifiability_json(&value.to_string()).unwrap_err();
        assert!(error.contains("cannot have identifiable concurrency"));
    }

    #[test]
    fn rejects_unknown_nested_fields() {
        let mut value = fixture_value();
        value["candidate_view"]["observation_boundaries"][0]["hidden_hint"] = json!("before");
        let error = evaluate_identifiability_json(&value.to_string()).unwrap_err();
        assert!(error.contains("unknown field"));
    }

    #[test]
    fn rejects_observed_sequences_outside_declared_coverage_range() {
        let mut value = fixture_value();
        value["candidate_view"]["observation_boundaries"][0]["observed_sequence_numbers"] =
            json!([1, 2, 99]);
        let error = evaluate_identifiability_json(&value.to_string()).unwrap_err();
        assert!(error.contains("outside the expected coverage range"));
    }

    #[test]
    fn rejects_pair_truth_for_an_unknown_task() {
        let mut value = fixture_value();
        value["gold_overlay"]["pair_truths"][0]["pair_id"] = json!("not-a-task");
        let error = evaluate_identifiability_json(&value.to_string()).unwrap_err();
        assert!(error.contains("unknown pair"));
    }

    #[test]
    fn rejects_missing_candidate_pair_decision() {
        let mut value = fixture_value();
        value["candidate_traces"][0]["pair_claims"].as_array_mut().expect("claims").pop();
        let error = evaluate_identifiability_json(&value.to_string()).unwrap_err();
        assert!(error.contains("must claim every pair task exactly once"));
    }

    #[test]
    fn rejects_missing_boundary_claim() {
        let mut value = fixture_value();
        value["candidate_traces"][0]["boundary_claims"].as_array_mut().expect("claims").pop();
        let error = evaluate_identifiability_json(&value.to_string()).unwrap_err();
        assert!(error.contains("must claim every observation boundary exactly once"));
    }

    #[test]
    fn detects_failure_to_identify_metadata_order_despite_redacted_payload() {
        let report = evaluate_identifiability_json(FIXTURE).expect("D6.11 fixture");
        let always = trace(&report, "always-abstain");
        assert!(always.pair_outcomes.iter().any(|item| {
            item.pair_id == "redacted-metadata-order"
                && item.expected_state == "before"
                && item.unnecessary_abstention
        }));
    }
}
