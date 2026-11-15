//! Independent D2 reference evaluator.
//!
//! This crate deliberately does not depend on development-map-validator.
//! It independently parses the temporal ledger, verifies its event hash chain,
//! and derives expected actor-knowledge / ledger-as-recorded views.
//! It is a reference implementation, not an authenticated source collector.

use std::collections::{BTreeMap, HashMap, HashSet};

use chrono::{DateTime, FixedOffset};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

const VERSION: &str = "0.1.0";
const GENESIS: &str = "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ledger {
    schema_version: String,
    ledger_id: String,
    repository_full_name: String,
    events: Vec<Event>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Event {
    event_id: String,
    sequence: u64,
    previous_hash: String,
    event_hash: String,
    body: Body,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Body {
    operation: String,
    fact_id: String,
    recorded_at: String,
    observed_at: String,
    actor_availability: Vec<Availability>,
    scope: Scope,
    effective_at: Option<String>,
    valid_from: Option<String>,
    valid_through: Option<String>,
    validity_completeness: Option<String>,
    target_event_id: Option<String>,
    payload: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Availability {
    actor_id: String,
    available_at: String,
    proof_ref: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scope {
    revision: String,
    tree_oid: String,
    configuration_world_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Query {
    repository_full_name: String,
    configuration_world_digest: String,
    actor_id: String,
    knowledge_cutoff_at: String,
    valid_at: String,
    ledger_recorded_cutoff_at: Option<String>,
}

#[derive(Serialize)]
struct HashEnvelope<'a> {
    schema_version: &'a str,
    ledger_id: &'a str,
    repository_full_name: &'a str,
    event_id: &'a str,
    sequence: u64,
    previous_hash: &'a str,
    body: &'a Body,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Lifecycle {
    Current,
    Stale,
    Superseded,
    Indeterminate,
}
impl Lifecycle {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::Stale => "stale",
            Self::Superseded => "superseded",
            Self::Indeterminate => "indeterminate",
        }
    }
}

#[derive(Debug, Clone)]
struct Fact {
    assertion_event_id: String,
    lifecycle: Lifecycle,
    effective_at_query: bool,
    temporally_admissible: bool,
    payload: Value,
}

fn parse_time(value: &str, label: &str) -> Result<DateTime<FixedOffset>, String> {
    DateTime::parse_from_rfc3339(value)
        .map_err(|_| format!("{label} is not RFC 3339: {value:?}"))
}

fn sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn digest_event(ledger: &Ledger, event: &Event) -> Result<String, String> {
    let envelope = HashEnvelope {
        schema_version: &ledger.schema_version,
        ledger_id: &ledger.ledger_id,
        repository_full_name: &ledger.repository_full_name,
        event_id: &event.event_id,
        sequence: event.sequence,
        previous_hash: &event.previous_hash,
        body: &event.body,
    };
    let encoded = serde_json::to_vec(&envelope).map_err(|error| error.to_string())?;
    Ok(sha256(&encoded))
}

fn actor_availability<'a>(event: &'a Event, actor: &str) -> Option<&'a Availability> {
    event.body.actor_availability.iter().find(|item| item.actor_id == actor)
}

fn validate_ledger(ledger: &Ledger) -> Result<String, String> {
    if ledger.schema_version != VERSION {
        return Err(format!("unsupported ledger schema version {:?}", ledger.schema_version));
    }
    if ledger.ledger_id.trim().is_empty() || !ledger.repository_full_name.contains('/') {
        return Err("ledger identity is incomplete".to_owned());
    }

    let mut preceding: HashMap<&str, &Event> = HashMap::new();
    let mut terminal_targets = HashSet::new();
    let mut prior_hash = GENESIS.to_owned();
    let mut prior_recorded: Option<DateTime<FixedOffset>> = None;

    for (index, event) in ledger.events.iter().enumerate() {
        let location = format!("events[{index}]");
        if event.sequence != index as u64 + 1 {
            return Err(format!("{location}: non-contiguous sequence"));
        }
        if event.event_id.trim().is_empty() || preceding.contains_key(event.event_id.as_str()) {
            return Err(format!("{location}: empty or duplicate event identity"));
        }
        if event.previous_hash != prior_hash || !valid_digest(&event.event_hash) {
            return Err(format!("{location}: broken event hash linkage"));
        }
        let recomputed = digest_event(ledger, event)?;
        if recomputed != event.event_hash {
            return Err(format!("{location}: event digest mismatch"));
        }
        let observed = parse_time(&event.body.observed_at, "observed_at")?;
        let recorded = parse_time(&event.body.recorded_at, "recorded_at")?;
        if observed > recorded {
            return Err(format!("{location}: recorded_at precedes observed_at"));
        }
        if prior_recorded.as_ref().is_some_and(|previous| &recorded < previous) {
            return Err(format!("{location}: recording clock moved backwards"));
        }
        prior_recorded = Some(recorded);

        let mut actor_ids = HashSet::new();
        if event.body.actor_availability.is_empty() {
            return Err(format!("{location}: no actor availability statements"));
        }
        for item in &event.body.actor_availability {
            if item.actor_id.trim().is_empty() || item.proof_ref.trim().is_empty() {
                return Err(format!("{location}: incomplete actor availability statement"));
            }
            if !actor_ids.insert(item.actor_id.as_str()) {
                return Err(format!("{location}: duplicate actor availability"));
            }
            if parse_time(&item.available_at, "available_at")? < observed {
                return Err(format!("{location}: actor availability predates observation"));
            }
        }
        if !valid_digest(&event.body.scope.configuration_world_digest) {
            return Err(format!("{location}: invalid configuration world digest"));
        }
        if event.body.scope.revision.len() != event.body.scope.tree_oid.len()
            || !matches!(event.body.scope.revision.len(), 40 | 64)
            || !event.body.scope.revision.bytes().all(|b| b.is_ascii_hexdigit())
            || !event.body.scope.tree_oid.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err(format!("{location}: invalid repository object identity"));
        }
        if !event.body.payload.is_object() {
            return Err(format!("{location}: event payload must be an object"));
        }

        match event.body.operation.as_str() {
            "assert" => {
                if event.body.target_event_id.is_some() || event.body.effective_at.is_some() {
                    return Err(format!("{location}: assertion carries transition fields"));
                }
                if !matches!(event.body.validity_completeness.as_deref(), Some("complete" | "incomplete")) {
                    return Err(format!("{location}: assertion validity completeness is missing"));
                }
                let from = event.body.valid_from.as_deref()
                    .map(|raw| parse_time(raw, "valid_from")).transpose()?;
                let through = event.body.valid_through.as_deref()
                    .map(|raw| parse_time(raw, "valid_through")).transpose()?;
                if from.as_ref().zip(through.as_ref()).is_some_and(|(a,b)| a > b) {
                    return Err(format!("{location}: reversed validity interval"));
                }
            }
            "invalidate" | "supersede" => {
                let target_id = event.body.target_event_id.as_deref()
                    .ok_or_else(|| format!("{location}: missing transition target"))?;
                let target = preceding.get(target_id)
                    .ok_or_else(|| format!("{location}: transition target is not earlier in this ledger"))?;
                if target.body.operation != "assert" || target.body.fact_id != event.body.fact_id {
                    return Err(format!("{location}: transition target is not an assertion of the same fact"));
                }
                if target.body.scope.configuration_world_digest != event.body.scope.configuration_world_digest {
                    return Err(format!("{location}: transition crosses configuration worlds"));
                }
                if !terminal_targets.insert(target_id.to_owned()) {
                    return Err(format!("{location}: assertion has multiple terminal transitions"));
                }
                let effective = event.body.effective_at.as_deref()
                    .ok_or_else(|| format!("{location}: missing effective_at"))?;
                parse_time(effective, "effective_at")?;
                for transition_actor in &event.body.actor_availability {
                    let target_actor = actor_availability(target, &transition_actor.actor_id)
                        .ok_or_else(|| format!("{location}: transition actor lacks target availability"))?;
                    if parse_time(&transition_actor.available_at, "transition available_at")?
                        < parse_time(&target_actor.available_at, "target available_at")?
                    {
                        return Err(format!("{location}: transition precedes target actor availability"));
                    }
                }
            }
            "propose" => {
                if event.body.target_event_id.is_some()
                    || event.body.effective_at.is_some()
                    || event.body.valid_from.is_some()
                    || event.body.valid_through.is_some()
                    || event.body.validity_completeness.is_some()
                {
                    return Err(format!("{location}: proposal mutates asserted state"));
                }
            }
            other => return Err(format!("{location}: unsupported operation {other:?}")),
        }

        prior_hash.clone_from(&event.event_hash);
        preceding.insert(event.event_id.as_str(), event);
    }
    Ok(prior_hash)
}

fn included_event<'a>(
    event: &'a Event,
    query: &Query,
    knowledge_cutoff: &DateTime<FixedOffset>,
    ledger_cutoff: Option<&DateTime<FixedOffset>>,
) -> Result<Result<&'a Availability, String>, String> {
    if event.body.scope.configuration_world_digest != query.configuration_world_digest {
        return Ok(Err("configuration-world mismatch".to_owned()));
    }
    if let Some(cutoff) = ledger_cutoff {
        if parse_time(&event.body.recorded_at, "recorded_at")? > *cutoff {
            return Ok(Err("recorded after ledger cutoff".to_owned()));
        }
    }
    let Some(actor) = actor_availability(event, &query.actor_id) else {
        return Ok(Err("no availability record for actor".to_owned()));
    };
    if parse_time(&actor.available_at, "available_at")? > *knowledge_cutoff {
        return Ok(Err("not available to actor by knowledge cutoff".to_owned()));
    }
    Ok(Ok(actor))
}

/// Independently derive the normalized fixture projection.
/// This does not call or import the candidate development-map-validator crate.
pub fn evaluate(ledger: &Ledger, query: &Query) -> Result<Value, String> {
    let head = validate_ledger(ledger)?;
    if query.repository_full_name != ledger.repository_full_name {
        return Err("query repository differs from ledger repository".to_owned());
    }
    if !valid_digest(&query.configuration_world_digest) {
        return Err("query world digest must be SHA-256".to_owned());
    }
    if query.actor_id.trim().is_empty() {
        return Err("query actor identity is empty".to_owned());
    }
    let knowledge_cutoff = parse_time(&query.knowledge_cutoff_at, "knowledge_cutoff_at")?;
    let valid_at = parse_time(&query.valid_at, "valid_at")?;
    let ledger_cutoff = query.ledger_recorded_cutoff_at.as_deref()
        .map(|raw| parse_time(raw, "ledger_recorded_cutoff_at")).transpose()?;

    let mut facts: BTreeMap<String, Fact> = BTreeMap::new();
    let mut proposals = Vec::new();
    let mut included = Vec::new();
    let mut excluded = Vec::new();

    for event in &ledger.events {
        match included_event(event, query, &knowledge_cutoff, ledger_cutoff.as_ref())? {
            Err(reason) => {
                excluded.push(json!({"event_id":event.event_id,"reason":reason}));
            }
            Ok(_availability) => {
                included.push(Value::String(event.event_id.clone()));
                match event.body.operation.as_str() {
                    "assert" => {
                        let from = event.body.valid_from.as_deref()
                            .map(|raw| parse_time(raw, "valid_from")).transpose()?;
                        let through = event.body.valid_through.as_deref()
                            .map(|raw| parse_time(raw, "valid_through")).transpose()?;
                        let in_window = from.as_ref().is_none_or(|bound| bound <= &valid_at)
                            && through.as_ref().is_none_or(|bound| bound >= &valid_at);
                        let complete = event.body.validity_completeness.as_deref() == Some("complete");
                        facts.insert(event.event_id.clone(), Fact {
                            assertion_event_id: event.event_id.clone(),
                            lifecycle: if complete { Lifecycle::Current } else { Lifecycle::Indeterminate },
                            effective_at_query: in_window,
                            temporally_admissible: complete && in_window,
                            payload: event.body.payload.clone(),
                        });
                    }
                    "invalidate" | "supersede" => {
                        let effect = parse_time(event.body.effective_at.as_deref().unwrap(), "effective_at")?;
                        if effect <= valid_at {
                            let target_id = event.body.target_event_id.as_ref().unwrap();
                            let target = facts.get_mut(target_id).ok_or_else(|| {
                                format!("visible transition {} has no actor-visible target {target_id}", event.event_id)
                            })?;
                            if target.lifecycle != Lifecycle::Current && target.lifecycle != Lifecycle::Indeterminate {
                                return Err(format!("assertion {target_id} received multiple terminal transitions"));
                            }
                            target.lifecycle = if event.body.operation == "invalidate" {
                                Lifecycle::Stale
                            } else {
                                Lifecycle::Superseded
                            };
                            target.temporally_admissible = false;
                        }
                    }
                    "propose" => proposals.push(json!({
                        "event_id":event.event_id,
                        "fact_id":event.body.fact_id,
                        "payload":event.body.payload
                    })),
                    _ => unreachable!("ledger validator checked operations"),
                }
            }
        }
    }

    let mut fact_json = Map::new();
    for (event_id, fact) in facts {
        fact_json.insert(event_id, json!({
            "assertion_event_id":fact.assertion_event_id,
            "lifecycle":fact.lifecycle.as_str(),
            "effective_at_query":fact.effective_at_query,
            "temporally_admissible":fact.temporally_admissible,
            "payload":fact.payload
        }));
    }
    let included_ids: Vec<String> = included.iter().filter_map(Value::as_str).map(str::to_owned).collect();
    let excluded_ids: Vec<String> = excluded.iter()
        .filter_map(|entry| entry.get("event_id").and_then(Value::as_str).map(str::to_owned)).collect();

    Ok(json!({
        "ledger_head_hash":head,
        "included_event_ids":included_ids,
        "facts":Value::Object(fact_json),
        "proposals":proposals,
        "excluded_event_ids":excluded_ids
    }))
}

pub fn compare_expected(actual: &Value, expected: &Value) -> Result<(), String> {
    let expected_included = expected.get("included_event_ids").ok_or("expected included_event_ids missing")?;
    if actual.get("included_event_ids") != Some(expected_included) {
        return Err(format!("included_event_ids differ: actual={} expected={}", actual["included_event_ids"], expected_included));
    }
    let expected_proposals = expected.get("proposals").ok_or("expected proposals missing")?;
    if actual.get("proposals") != Some(expected_proposals) {
        return Err(format!("proposals differ: actual={} expected={}", actual["proposals"], expected_proposals));
    }
    let expected_facts = expected.get("facts").and_then(Value::as_object).ok_or("expected facts must be an object")?;
    let actual_facts = actual.get("facts").and_then(Value::as_object).ok_or("actual facts must be an object")?;
    if expected_facts.len() != actual_facts.len() {
        return Err(format!("fact count differs: actual={} expected={}", actual_facts.len(), expected_facts.len()));
    }
    for (event_id, expected_fact) in expected_facts {
        let actual_fact = actual_facts.get(event_id).ok_or_else(|| format!("missing fact {event_id}"))?;
        for (field, expected_value) in expected_fact.as_object().ok_or("expected fact must be object")? {
            if actual_fact.get(field) != Some(expected_value) {
                return Err(format!("fact {event_id}.{field} differs: actual={} expected={}", actual_fact[field], expected_value));
            }
        }
    }
    let expected_excluded: HashSet<&str> = expected.get("excluded_event_ids")
        .and_then(Value::as_array).ok_or("expected excluded_event_ids missing")?
        .iter().filter_map(Value::as_str).collect();
    let actual_excluded: HashSet<&str> = actual.get("excluded_event_ids")
        .and_then(Value::as_array).ok_or("actual excluded_event_ids missing")?
        .iter().filter_map(Value::as_str).collect();
    if actual_excluded != expected_excluded {
        return Err("excluded_event_ids differ".to_owned());
    }
    Ok(())
}

pub fn evaluate_json(ledger: &str, query: &str) -> Result<Value, String> {
    let ledger: Ledger = serde_json::from_str(ledger).map_err(|error| error.to_string())?;
    let query: Query = serde_json::from_str(query).map_err(|error| error.to_string())?;
    evaluate(&ledger, &query)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEDGER: &str = include_str!("../../../docs/development-map/fixtures/temporal-event-ledger-v0.1.json");
    const EXPECTED: &str = include_str!("../../../docs/development-map/fixtures/temporal-event-ledger-v0.1.expected.json");
    const ACTOR_QUERY: &str = include_str!("../../../docs/development-map/fixtures/temporal-replay-actor-knowledge-v0.1.json");
    const LEDGER_QUERY: &str = include_str!("../../../docs/development-map/fixtures/temporal-replay-ledger-as-of-v0.1.json");
    const RESTATED_QUERY: &str = include_str!("../../../docs/development-map/fixtures/temporal-replay-restated-v0.1.json");

    fn check(query_text: &str, key: &str) {
        let ledger: Ledger = serde_json::from_str(LEDGER).unwrap();
        let query: Query = serde_json::from_str(query_text).unwrap();
        let expected_root: Value = serde_json::from_str(EXPECTED).unwrap();
        let actual = evaluate(&ledger, &query).unwrap();
        let expected = &expected_root["queries"][key];
        compare_expected(&actual, expected).unwrap();
        assert_eq!(actual["ledger_head_hash"], expected_root["expected_ledger"]["head_hash"]);
    }

    #[test] fn actor_knowledge_fixture_matches_independent_oracle() {
        check(ACTOR_QUERY, "temporal-replay-actor-knowledge-v0.1.json");
    }
    #[test] fn ledger_as_of_fixture_matches_independent_oracle() {
        check(LEDGER_QUERY, "temporal-replay-ledger-as-of-v0.1.json");
    }
    #[test] fn retrospective_restated_fixture_matches_independent_oracle() {
        check(RESTATED_QUERY, "temporal-replay-restated-v0.1.json");
    }
    #[test] fn event_identity_tampering_is_rejected_without_resealing() {
        let mut ledger: Ledger = serde_json::from_str(LEDGER).unwrap();
        ledger.events[0].event_id = "evt-renamed".to_owned();
        assert!(validate_ledger(&ledger).is_err());
    }
    #[test] fn repository_identity_tampering_is_rejected_without_resealing() {
        let mut ledger: Ledger = serde_json::from_str(LEDGER).unwrap();
        ledger.repository_full_name = "attacker/other".to_owned();
        assert!(validate_ledger(&ledger).is_err());
    }
    #[test] fn duplicate_terminal_transition_is_rejected() {
        let mut ledger: Ledger = serde_json::from_str(LEDGER).unwrap();
        let mut second = ledger.events[1].clone();
        second.event_id = "evt-005".to_owned();
        second.sequence = 5;
        second.body.operation = "supersede".to_owned();
        second.body.recorded_at = "2026-11-14T12:00:00Z".to_owned();
        second.body.observed_at = "2026-11-14T11:59:00Z".to_owned();
        second.body.actor_availability[0].available_at = "2026-11-14T12:01:00Z".to_owned();
        // An independent oracle must reject a malformed chain before judging semantics.
        ledger.events.push(second);
        assert!(validate_ledger(&ledger).is_err());
    }
}
