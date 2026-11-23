use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::process::ExitCode;

const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
const MAX_REORDER_WINDOW: u64 = 1_024;
const MAX_BUFFERED_EVENTS: usize = 1_024;
const DEFAULT_FIXTURE: &str = "benchmarks/frontend/long-horizon/event-reconciliation-v1.json";

fn field_str<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value.get(key).and_then(Value::as_str).ok_or_else(|| format!("missing string field: {key}"))
}
fn field_u64(value: &Value, key: &str) -> Result<u64, String> {
    value.get(key).and_then(Value::as_u64).ok_or_else(|| format!("missing unsigned integer field: {key}"))
}

fn validate_event_shape(event: &Value) -> Result<(), String> {
    let kind = field_str(event, "type")?;
    let fields: Vec<&str> = match kind {
        "job_transition" => vec!["type","event_id","sequence","job_id","expected_version","next_version","from_status","to_status","actor_id","capability","action_id"],
        "owner_changed" => vec!["type","event_id","sequence","job_id","expected_version","next_version","from_owner","to_owner","actor_id","capability"],
        "permission_revoked" => vec!["type","event_id","sequence","actor_id","capability"],
        "clarification_required" => vec!["type","event_id","sequence","job_id","conflict_id"],
        "clarification_resolved" => vec!["type","event_id","sequence","job_id","conflict_id","resolution"],
        other => return Err(format!("unsupported event type: {other}")),
    };
    let object = event.as_object().ok_or("event must be a JSON object")?;
    for key in object.keys() {
        if !fields.contains(&key.as_str()) {
            return Err(format!("unknown event field: {key}"));
        }
    }
    for key in &fields {
        if !object.contains_key(*key) {
            return Err(format!("missing event field: {key}"));
        }
    }
    let event_id = field_str(event, "event_id")?;
    if event_id.is_empty() || event_id.len() > 128
        || !event_id.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-')) {
        return Err("invalid event identity".to_string());
    }
    let sequence = field_u64(event, "sequence")?;
    if sequence == 0 || sequence > MAX_SAFE_INTEGER {
        return Err("event sequence is outside the safe positive-integer range".to_string());
    }
    for key in ["expected_version", "next_version"] {
        if object.contains_key(key) {
            let value = field_u64(event, key)?;
            if value == 0 || value > MAX_SAFE_INTEGER {
                return Err(format!("{key} is outside the safe positive-integer range"));
            }
        }
    }
    for key in ["job_id","actor_id","capability","from_status","to_status","from_owner","to_owner","conflict_id","resolution"] {
        if object.contains_key(key) {
            let value = field_str(event, key)?;
            if value.is_empty() || value.len() > 128 {
                return Err(format!("invalid event string field: {key}"));
            }
        }
    }
    if object.contains_key("action_id") {
        let action_id = &event["action_id"];
        if !action_id.is_null() {
            let value = action_id.as_str().ok_or("action_id must be null or a string")?;
            if value.is_empty() || value.len() > 128
                || !value.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-')) {
                return Err("invalid action_id".to_string());
            }
        }
    }
    Ok(())
}
fn transition_allowed(from: &str, to: &str) -> bool {
    matches!((from, to),
        ("open", "in_progress") | ("open", "blocked")
        | ("in_progress", "blocked") | ("in_progress", "done")
        | ("blocked", "in_progress") | ("blocked", "done"))
}
fn subset_matches(actual: &Value, expected: &Value) -> bool {
    match (actual, expected) {
        (Value::Object(actual), Value::Object(expected)) =>
            expected.iter().all(|(key, value)| actual.get(key).is_some_and(|found| subset_matches(found, value))),
        _ => actual == expected,
    }
}

#[derive(Clone, Debug)]
struct State {
    jobs: BTreeMap<String, Value>,
    permissions: BTreeMap<String, BTreeSet<String>>,
    conflicts: BTreeMap<String, String>,
    pending: BTreeMap<String, Value>,
    action_results: BTreeMap<String, Value>,
    seen_by_id: BTreeMap<String, Value>,
    sequence_owner: BTreeMap<u64, String>,
    buffered: BTreeMap<u64, Value>,
    last_sequence: u64,
    audit: Vec<String>,
    duplicate_deliveries: usize,
}

impl State {
    fn new(initial: &Value) -> Result<Self, String> {
        let mut jobs = BTreeMap::new();
        for job in initial.get("jobs").and_then(Value::as_array).ok_or("initial_state.jobs must be an array")? {
            let id = field_str(job, "id")?.to_string();
            let version = field_u64(job, "version")?;
            if id.is_empty() || id.len() > 128 || version == 0 || version > MAX_SAFE_INTEGER || jobs.insert(id.clone(), job.clone()).is_some() {
                return Err(format!("invalid or duplicate initial job: {id}"));
            }
        }
        let mut permissions = BTreeMap::new();
        for entry in initial.get("permissions").and_then(Value::as_array).ok_or("initial_state.permissions must be an array")? {
            let actor = field_str(entry, "actor_id")?.to_string();
            let capabilities = entry.get("capabilities").and_then(Value::as_array).ok_or("capabilities must be an array")?;
            let mut set = BTreeSet::new();
            for capability in capabilities {
                set.insert(capability.as_str().ok_or("capability must be a string")?.to_string());
            }
            if actor.is_empty() || permissions.insert(actor.clone(), set).is_some() {
                return Err(format!("invalid or duplicate permission actor: {actor}"));
            }
        }
        Ok(Self {
            jobs, permissions, conflicts: BTreeMap::new(), pending: BTreeMap::new(),
            action_results: BTreeMap::new(), seen_by_id: BTreeMap::new(),
            sequence_owner: BTreeMap::new(), buffered: BTreeMap::new(),
            last_sequence: 0, audit: Vec::new(), duplicate_deliveries: 0,
        })
    }

    fn has_permission(&self, actor: &str, capability: &str) -> bool {
        self.permissions.get(actor).is_some_and(|set| set.contains(capability))
    }

    fn reject_pending(&mut self, action_id: &str, reason: &str) {
        self.pending.remove(action_id);
        self.action_results.insert(action_id.to_string(), json!({"status":"rejected","reason":reason}));
    }

    fn reject_stale_for_job(&mut self, job_id: &str) -> Result<(), String> {
        let version = self.jobs.get(job_id).ok_or_else(|| format!("unknown job: {job_id}"))?
            .get("version").and_then(Value::as_u64).ok_or("job.version is invalid")?;
        let stale: Vec<String> = self.pending.iter()
            .filter(|(_, action)| action.get("job_id").and_then(Value::as_str) == Some(job_id)
                && action.get("expected_version").and_then(Value::as_u64) != Some(version))
            .map(|(id, _)| id.clone()).collect();
        for id in stale { self.reject_pending(&id, "stale_authoritative_state"); }
        Ok(())
    }

    fn submit(&mut self, action: &Value) -> Result<(), String> {
        let action_id = field_str(action, "action_id")?.to_string();
        let job_id = field_str(action, "job_id")?;
        let actor = field_str(action, "actor_id")?;
        let capability = field_str(action, "capability")?;
        let expected_version = field_u64(action, "expected_version")?;
        let from_status = field_str(action, "from_status")?;
        let to_status = field_str(action, "to_status")?;
        let input_mode = field_str(action, "input_mode")?;
        if action_id.is_empty() || self.action_results.contains_key(&action_id) {
            return Err(format!("duplicate or empty action ID: {action_id}"));
        }
        if !matches!(input_mode, "pointer" | "keyboard") {
            return Err(format!("unsupported input mode: {input_mode}"));
        }
        let reason = match self.jobs.get(job_id) {
            None => Some("unknown_job"),
            Some(_) if self.conflicts.contains_key(job_id) => Some("clarification_required"),
            Some(job) if job.get("version").and_then(Value::as_u64) != Some(expected_version) => Some("stale_version"),
            Some(job) if job.get("status").and_then(Value::as_str) != Some(from_status) => Some("stale_status"),
            Some(_) if !self.has_permission(actor, capability) => Some("permission_denied"),
            Some(job) if !transition_allowed(job.get("status").and_then(Value::as_str).unwrap_or(""), to_status) => Some("invalid_transition"),
            Some(_) if self.pending.values().any(|pending| pending.get("job_id").and_then(Value::as_str) == Some(job_id)) => Some("action_already_pending"),
            Some(_) => None,
        };
        if let Some(reason) = reason {
            self.action_results.insert(action_id, json!({"status":"rejected","reason":reason}));
            return Ok(());
        }
        self.pending.insert(action_id.clone(), action.clone());
        self.action_results.insert(action_id, json!({"status":"pending"}));
        Ok(())
    }

    fn apply_event(&mut self, event: &Value) -> Result<(), String> {
        let event_id = field_str(event, "event_id")?;
        match field_str(event, "type")? {
            "job_transition" => {
                let job_id = field_str(event, "job_id")?.to_string();
                let expected_version = field_u64(event, "expected_version")?;
                let next_version = field_u64(event, "next_version")?;
                let from_status = field_str(event, "from_status")?;
                let to_status = field_str(event, "to_status")?;
                let actor = field_str(event, "actor_id")?;
                let capability = field_str(event, "capability")?;
                let job = self.jobs.get(&job_id).ok_or_else(|| format!("{event_id}: unknown job {job_id}"))?;
                if job.get("version").and_then(Value::as_u64) != Some(expected_version)
                    || job.get("status").and_then(Value::as_str) != Some(from_status)
                    || expected_version.checked_add(1) != Some(next_version) {
                    return Err(format!("{event_id}: stale or malformed transition"));
                }
                if !transition_allowed(from_status, to_status) {
                    return Err(format!("{event_id}: illegal authoritative transition"));
                }
                if !self.has_permission(actor, capability) {
                    return Err(format!("{event_id}: unauthorized authoritative actor"));
                }
                if let Some(action_id) = event.get("action_id").and_then(Value::as_str) {
                    let pending = self.pending.get(action_id).ok_or_else(|| format!("{event_id}: acknowledgement has no pending action"))?;
                    if pending.get("job_id").and_then(Value::as_str) != Some(job_id.as_str())
                        || pending.get("expected_version").and_then(Value::as_u64) != Some(expected_version)
                        || pending.get("to_status").and_then(Value::as_str) != Some(to_status)
                        || pending.get("actor_id").and_then(Value::as_str) != Some(actor)
                        || pending.get("capability").and_then(Value::as_str) != Some(capability) {
                        return Err(format!("{event_id}: acknowledgement does not match pending action"));
                    }
                    self.pending.remove(action_id);
                    self.action_results.insert(action_id.to_string(), json!({"status":"accepted"}));
                }
                let job = self.jobs.get_mut(&job_id).expect("job validated above");
                job["status"] = json!(to_status);
                job["version"] = json!(next_version);
                self.reject_stale_for_job(&job_id)?;
            }
            "owner_changed" => {
                let job_id = field_str(event, "job_id")?.to_string();
                let expected_version = field_u64(event, "expected_version")?;
                let next_version = field_u64(event, "next_version")?;
                let from_owner = field_str(event, "from_owner")?;
                let to_owner = field_str(event, "to_owner")?;
                let actor = field_str(event, "actor_id")?;
                let capability = field_str(event, "capability")?;
                let job = self.jobs.get(&job_id).ok_or_else(|| format!("{event_id}: unknown job"))?;
                if job.get("version").and_then(Value::as_u64) != Some(expected_version)
                    || job.get("owner").and_then(Value::as_str) != Some(from_owner)
                    || expected_version.checked_add(1) != Some(next_version) {
                    return Err(format!("{event_id}: stale or malformed owner change"));
                }
                if !self.has_permission(actor, capability) {
                    return Err(format!("{event_id}: unauthorized owner change"));
                }
                let job = self.jobs.get_mut(&job_id).expect("job validated above");
                job["owner"] = json!(to_owner);
                job["version"] = json!(next_version);
                self.reject_stale_for_job(&job_id)?;
            }
            "permission_revoked" => {
                let actor = field_str(event, "actor_id")?.to_string();
                let capability = field_str(event, "capability")?.to_string();
                if let Some(capabilities) = self.permissions.get_mut(&actor) { capabilities.remove(&capability); }
                let invalidated: Vec<String> = self.pending.iter()
                    .filter(|(_, action)| action.get("actor_id").and_then(Value::as_str) == Some(actor.as_str())
                        && action.get("capability").and_then(Value::as_str) == Some(capability.as_str()))
                    .map(|(id, _)| id.clone()).collect();
                for id in invalidated { self.reject_pending(&id, "permission_revoked"); }
            }
            "clarification_required" => {
                let job_id = field_str(event, "job_id")?.to_string();
                let conflict_id = field_str(event, "conflict_id")?.to_string();
                if !self.jobs.contains_key(&job_id) || self.conflicts.contains_key(&job_id) {
                    return Err(format!("{event_id}: unknown job or duplicate active conflict"));
                }
                self.conflicts.insert(job_id.clone(), conflict_id);
                let invalidated: Vec<String> = self.pending.iter()
                    .filter(|(_, action)| action.get("job_id").and_then(Value::as_str) == Some(job_id.as_str()))
                    .map(|(id, _)| id.clone()).collect();
                for id in invalidated { self.reject_pending(&id, "clarification_required"); }
            }
            "clarification_resolved" => {
                let job_id = field_str(event, "job_id")?;
                let conflict_id = field_str(event, "conflict_id")?;
                let resolution = field_str(event, "resolution")?;
                if resolution.trim().is_empty() || self.conflicts.get(job_id).map(String::as_str) != Some(conflict_id) {
                    return Err(format!("{event_id}: clarification resolution mismatch"));
                }
                self.conflicts.remove(job_id);
            }
            other => return Err(format!("unsupported event type: {other}")),
        }
        Ok(())
    }

    fn deliver(&mut self, event: &Value) -> Result<(), String> {
        // A rejected event must not poison identity, sequence, buffer, audit, or state.
        let mut candidate = self.clone();
        candidate.deliver_uncommitted(event)?;
        *self = candidate;
        Ok(())
    }

    fn deliver_uncommitted(&mut self, event: &Value) -> Result<(), String> {
        let event_id = field_str(event, "event_id")?.to_string();
        if let Some(previous) = self.seen_by_id.get(&event_id) {
            if previous == event { self.duplicate_deliveries += 1; return Ok(()); }
            return Err(format!("event identity reused with different payload: {event_id}"));
        }
        validate_event_shape(event)?;
        let sequence = field_u64(event, "sequence")?;
        if sequence <= self.last_sequence { return Err(format!("invalid or replayed sequence: {sequence}")); }
        let maximum = self.last_sequence.checked_add(MAX_REORDER_WINDOW).unwrap_or(u64::MAX);
        if sequence > maximum { return Err(format!("event sequence exceeds bounded reorder window: {sequence}")); }
        if self.buffered.len() >= MAX_BUFFERED_EVENTS {
            return Err("event reorder buffer is full".to_string());
        }
        if let Some(owner) = self.sequence_owner.get(&sequence) {
            return Err(format!("sequence {sequence} assigned to both {owner} and {event_id}"));
        }
        self.seen_by_id.insert(event_id.clone(), event.clone());
        self.sequence_owner.insert(sequence, event_id);
        self.buffered.insert(sequence, event.clone());
        while let Some(next) = self.buffered.get(&(self.last_sequence + 1)).cloned() {
            self.apply_event(&next)?;
            let next_sequence = field_u64(&next, "sequence")?;
            let next_id = field_str(&next, "event_id")?.to_string();
            self.buffered.remove(&next_sequence);
            self.last_sequence = next_sequence;
            self.audit.push(next_id);
        }
        Ok(())
    }

    fn snapshot(&self) -> Value {
        let visible: BTreeMap<String, String> = self.jobs.iter().map(|(job_id, job)| {
            let status = self.pending.values()
                .find(|action| action.get("job_id").and_then(Value::as_str) == Some(job_id))
                .and_then(|action| action.get("to_status").and_then(Value::as_str))
                .unwrap_or_else(|| job.get("status").and_then(Value::as_str).unwrap_or(""));
            (job_id.clone(), status.to_string())
        }).collect();
        let pending: Vec<String> = self.pending.keys().cloned().collect();
        json!({
            "last_sequence": self.last_sequence,
            "jobs": &self.jobs,
            "permissions": &self.permissions,
            "conflicts": &self.conflicts,
            "pending_action_ids": pending,
            "action_results": &self.action_results,
            "visible_statuses": visible,
            "audit_event_ids": &self.audit,
            "duplicate_deliveries": self.duplicate_deliveries,
            "buffered_sequences": self.buffered.keys().copied().collect::<Vec<_>>()
        })
    }
}

fn run_case(fixture: &Value, case: &Value, events: &BTreeMap<String, Value>) -> Result<Value, String> {
    let mut state = State::new(&fixture["initial_state"])?;
    for step in case.get("steps").and_then(Value::as_array).ok_or("case.steps must be array")? {
        match field_str(step, "op")? {
            "submit" => state.submit(step.get("action").ok_or("submit step missing action")?)?,
            "deliver" => {
                let id = field_str(step, "event_id")?;
                let event = events.get(id).ok_or_else(|| format!("unknown event: {id}"))?;
                state.deliver(event)?;
            }
            other => return Err(format!("unsupported step: {other}")),
        }
    }
    let actual = state.snapshot();
    if !subset_matches(&actual, case.get("expected").ok_or("case.expected missing")?) {
        return Err(format!("{}: reference output differs from golden subset", field_str(case, "case_id")?));
    }
    Ok(actual)
}

fn reject_control(fixture: &Value, events: &BTreeMap<String, Value>, control: &Value) -> Result<(), String> {
    let control_id = field_str(control, "control_id")?;
    let original = events.get(field_str(control, "event_id")?).ok_or("negative control event missing")?;
    let mut state = State::new(&fixture["initial_state"])?;
    let mut altered = original.clone();
    match field_str(control, "operation")? {
        "alter_duplicate_payload" => {
            state.deliver(original)?;
            altered["to_status"] = control["patch"]["to_status"].clone();
        }
        "reuse_sequence_with_new_id" => {
            state.deliver(original)?;
            altered["event_id"] = control["alternate_event_id"].clone();
        }
        "replace_actor" => {
            for id in ["evt-001", "evt-002"] { state.deliver(events.get(id).ok_or("prerequisite event missing")?)?; }
            altered["actor_id"] = control["actor_id"].clone();
        }
        "invalid_event_is_rolled_back_atomically" => {
            altered["to_status"] = control["to_status"].clone();
        }
        other => return Err(format!("{control_id}: unsupported operation {other}")),
    }
    let expected_error = field_str(control, "expected_error_fragment")?;
    match state.deliver(&altered) {
        Err(error) if error.contains(expected_error) => {}
        Err(error) => return Err(format!("{control_id}: unexpected rejection: {error}")),
        Ok(()) => return Err(format!("{control_id}: adversarial mutation was accepted")),
    }
    if field_str(control, "operation")? == "invalid_event_is_rolled_back_atomically" {
        let original_sequence = field_u64(original, "sequence")?;
        let original_id = field_str(original, "event_id")?;
        state.deliver(original)?;
        if state.last_sequence != original_sequence || state.audit.last().map(String::as_str) != Some(original_id) {
            return Err(format!("{control_id}: valid retry did not apply after rejected mutation"));
        }
    }
    Ok(())
}

fn run_fixture(fixture: &Value) -> Result<(String, usize, usize, usize), String> {
    if field_str(fixture, "schema")? != "sfeb-event-reconciliation-fixture-v1" {
        return Err("unsupported fixture schema".to_string());
    }
    let events_array = fixture.get("events").and_then(Value::as_array).ok_or("events must be an array")?;
    let mut events = BTreeMap::new();
    let mut sequences = BTreeSet::new();
    for event in events_array {
        let id = field_str(event, "event_id")?.to_string();
        let sequence = field_u64(event, "sequence")?;
        if id.is_empty() || sequence == 0 || events.insert(id, event.clone()).is_some() || !sequences.insert(sequence) {
            return Err("event IDs and positive sequence numbers must be unique".to_string());
        }
    }
    let cases = fixture.get("cases").and_then(Value::as_array).ok_or("cases must be an array")?;
    let mut outcomes = BTreeMap::new();
    for case in cases {
        let id = field_str(case, "case_id")?.to_string();
        if outcomes.contains_key(&id) { return Err(format!("duplicate case ID: {id}")); }
        outcomes.insert(id.clone(), run_case(fixture, case, &events)?);
    }
    let relations = fixture.get("metamorphic_pairs").and_then(Value::as_array).ok_or("metamorphic_pairs missing")?;
    for relation in relations {
        let left = outcomes.get(field_str(relation, "left_case")?).ok_or("metamorphic left case missing")?;
        let right = outcomes.get(field_str(relation, "right_case")?).ok_or("metamorphic right case missing")?;
        if left != right { return Err(format!("{}: semantic outcomes differ", field_str(relation, "relation_id")?)); }
    }
    let controls = fixture.get("negative_controls").and_then(Value::as_array).ok_or("negative_controls missing")?;
    for control in controls { reject_control(fixture, &events, control)?; }
    Ok((field_str(fixture, "fixture_id")?.to_string(), cases.len(), relations.len(), controls.len()))
}


const EXPECTED_EVENT_POLICY_SHA256: &str = "dddaaa445881feabc3dc7d3599c8741004917190b2b2467db65cf1c42d7b6d5c";
const DEFAULT_POLICY: &str = "benchmarks/frontend/long-horizon/event-oracle-policy-v1.json";
const DEFAULT_TRACE: &str = "benchmarks/frontend/long-horizon/examples/late-event-rollback-trace.json";
const DEFAULT_POLICY_SCHEMA: &str = "benchmarks/frontend/long-horizon/event-oracle-policy-v1.schema.json";
const DEFAULT_TRACE_SCHEMA: &str = "benchmarks/frontend/long-horizon/event-execution-trace-v1.schema.json";

fn digest_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn is_hex_digest(value: &str, lengths: &[usize]) -> bool {
    lengths.contains(&value.len()) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn verify_policy(policy: &Value) -> Result<(), String> {
    if field_str(policy, "schema")? != "sfeb-event-oracle-policy-v1"
        || field_str(policy, "task_id")? != "SFEB-PILOT-INT-004" {
        return Err("unsupported event oracle policy".to_string());
    }
    let rules = policy.get("rules").and_then(Value::as_array)
        .ok_or("event oracle policy rules must be an array")?;
    if rules.len() != 7 {
        return Err("INT-004 event policy must define exactly seven rules".to_string());
    }
    let mut ids = BTreeSet::new();
    for rule in rules {
        let id = field_str(rule, "id")?;
        if !ids.insert(id.to_string()) {
            return Err(format!("duplicate event oracle rule ID: {id}"));
        }
        if rule.get("required").and_then(Value::as_bool) != Some(true)
            || rule.get("inconclusive_on_missing").and_then(Value::as_bool) != Some(true) {
            return Err(format!("event oracle rule {id} must be required and fail closed on missing evidence"));
        }
    }
    Ok(())
}

fn verify_trace_shape(trace: &Value) -> Result<(), String> {
    let object = trace.as_object().ok_or("event execution trace must be a JSON object")?;
    const ALLOWED: &[&str] = &["schema","trace_id","case_id","fixture_id","fixture_sha256","steps","worker_claims"];
    const REQUIRED: &[&str] = &["schema","trace_id","case_id","fixture_id","fixture_sha256","steps"];
    for key in object.keys() {
        if !ALLOWED.contains(&key.as_str()) {
            return Err(format!("unknown event trace field: {key}"));
        }
    }
    for key in REQUIRED {
        if !object.contains_key(*key) {
            return Err(format!("missing event trace field: {key}"));
        }
    }
    if field_str(trace, "schema")? != "sfeb-event-execution-trace-v1" {
        return Err("unsupported event execution trace schema".to_string());
    }
    for key in ["trace_id","case_id","fixture_id"] {
        let value = field_str(trace, key)?;
        if value.is_empty() || value.len() > 128 {
            return Err(format!("invalid event trace identity: {key}"));
        }
    }
    if !is_hex_digest(field_str(trace, "fixture_sha256")?, &[64]) {
        return Err("malformed fixture_sha256 in event trace".to_string());
    }
    let steps = trace.get("steps").and_then(Value::as_array)
        .ok_or("event execution trace steps must be an array")?;
    if steps.is_empty() || steps.len() > 10_000 {
        return Err("event execution trace step count is outside 1..=10000".to_string());
    }
    if let Some(claims) = trace.get("worker_claims") {
        if !claims.is_object() {
            return Err("worker_claims must be an object when present".to_string());
        }
    }
    Ok(())
}

fn verify_event_trace_values(
    fixture: &Value,
    fixture_bytes: &[u8],
    policy: &Value,
    policy_bytes: &[u8],
    trace: &Value,
    trace_bytes: &[u8],
) -> Result<Value, String> {
    if digest_hex(policy_bytes) != EXPECTED_EVENT_POLICY_SHA256 {
        return Err("event oracle policy digest does not match the verifier's frozen allowlisted policy".to_string());
    }
    verify_policy(policy)?;
    verify_trace_shape(trace)?;
    let fixture_id = field_str(fixture, "fixture_id")?;
    if field_str(trace, "fixture_id")? != fixture_id {
        return Err("event trace fixture ID does not match frozen fixture".to_string());
    }
    let actual_fixture_sha = digest_hex(fixture_bytes);
    if field_str(trace, "fixture_sha256")? != actual_fixture_sha {
        return Err("event trace does not bind exact frozen fixture bytes".to_string());
    }
    let case_id = field_str(trace, "case_id")?;
    let cases = fixture.get("cases").and_then(Value::as_array)
        .ok_or("frozen fixture cases are missing")?;
    let case = cases.iter().find(|item| item.get("case_id").and_then(Value::as_str) == Some(case_id))
        .ok_or_else(|| format!("unknown frozen event case: {case_id}"))?;
    let expected_steps = case.get("steps").and_then(Value::as_array)
        .ok_or("frozen case steps are missing")?;
    let observed_steps = trace.get("steps").and_then(Value::as_array)
        .ok_or("observed event trace steps are missing")?;
    if expected_steps.len() != observed_steps.len() {
        return Err("observed event trace does not cover the exact controller step sequence".to_string());
    }
    let events = fixture.get("events").and_then(Value::as_array)
        .ok_or("frozen fixture event list is missing")?;
    let mut event_map = BTreeMap::<String, Value>::new();
    for event in events {
        let id = field_str(event, "event_id")?.to_string();
        if event_map.insert(id.clone(), event.clone()).is_some() {
            return Err(format!("duplicate frozen event identity: {id}"));
        }
    }
    let mut state = State::new(&fixture["initial_state"])?;
    let mut evidence_event_ids = Vec::<String>::new();
    let mut replay_error: Option<String> = None;
    for (index, (expected, observed)) in expected_steps.iter().zip(observed_steps).enumerate() {
        let expected_op = field_str(expected, "op")?;
        let observed_op = field_str(observed, "op")?;
        if expected_op != observed_op {
            return Err(format!("trace step {index} does not match controller operation"));
        }
        match expected_op {
            "submit" => {
                if observed.get("action") != expected.get("action") {
                    return Err(format!("trace step {index} action does not match controller input"));
                }
                if let Err(error) = state.submit(observed.get("action").ok_or("submit trace step missing action")?) {
                    replay_error = Some(error);
                    break;
                }
            }
            "deliver" => {
                let expected_id = field_str(expected, "event_id")?;
                let observed_event = observed.get("event").ok_or("deliver trace step missing raw event payload")?;
                if field_str(observed_event, "event_id")? != expected_id {
                    return Err(format!("trace step {index} delivered an unexpected event identity"));
                }
                let frozen_event = event_map.get(expected_id)
                    .ok_or_else(|| format!("controller references missing frozen event: {expected_id}"))?;
                if observed_event != frozen_event {
                    return Err(format!("trace step {index} event payload differs from frozen authoritative event"));
                }
                evidence_event_ids.push(expected_id.to_string());
                if let Err(error) = state.deliver(observed_event) {
                    replay_error = Some(error);
                    break;
                }
            }
            other => return Err(format!("unsupported controller operation in frozen case: {other}")),
        }
    }

    let actual_snapshot = state.snapshot();
    let expected_snapshot = case.get("expected").ok_or("frozen case expected state is missing")?;
    let snapshot_matches = replay_error.is_none() && subset_matches(&actual_snapshot, expected_snapshot);
    let expected_last = expected_snapshot.get("last_sequence").and_then(Value::as_u64);
    let sequence_contiguous = replay_error.is_none()
        && state.buffered.is_empty()
        && expected_last == Some(state.last_sequence);
    let mut unique_audit = BTreeSet::new();
    let audit_unique = state.audit.iter().all(|event_id| unique_audit.insert(event_id.clone()));
    let expected_duplicates = expected_snapshot.get("duplicate_deliveries").and_then(Value::as_u64);
    let idempotent = replay_error.is_none()
        && audit_unique
        && expected_duplicates == Some(state.duplicate_deliveries);

    let has_permission_event = evidence_event_ids.iter().any(|id| {
        event_map.get(id).and_then(|event| event.get("type")).and_then(Value::as_str) == Some("permission_revoked")
    });
    let permission_ok = !has_permission_event
        || (snapshot_matches && expected_snapshot.get("permissions").is_some()
            && actual_snapshot.get("permissions") == expected_snapshot.get("permissions"));

    let has_conflict_event = evidence_event_ids.iter().any(|id| {
        event_map.get(id).and_then(|event| event.get("type")).and_then(Value::as_str) == Some("clarification_required")
    });
    let clarification_ok = !has_conflict_event || snapshot_matches;

    let expected_audit = expected_snapshot.get("audit_event_ids");
    let audit_ok = audit_unique && expected_audit == actual_snapshot.get("audit_event_ids")
        && replay_error.is_none();

    let evaluators = [
        ("strict_event_contract", replay_error.is_none()),
        ("contiguous_sequence_replay", sequence_contiguous),
        ("idempotent_event_identity", idempotent),
        ("pending_action_reconciliation", snapshot_matches),
        ("permission_revocation", permission_ok),
        ("clarification_gate", clarification_ok),
        ("exactly_once_audit", audit_ok),
    ];
    let rules = policy.get("rules").and_then(Value::as_array).ok_or("policy rules missing")?;
    let mut oracle_results = Vec::new();
    let mut any_failed = false;
    for rule in rules {
        let rule_id = field_str(rule, "id")?;
        let evaluator = field_str(rule, "evaluator")?;
        let passed = evaluators.iter().find(|(kind, _)| *kind == evaluator)
            .map(|(_, value)| *value)
            .ok_or_else(|| format!("unknown event evaluator in policy: {evaluator}"))?;
        if !passed { any_failed = true; }
        let canonical_rule = serde_jcs::to_vec(rule)
            .map_err(|error| format!("cannot canonicalize event oracle rule: {error}"))?;
        let unique_event_ids: BTreeSet<String> = evidence_event_ids.iter().cloned().collect();
        let event_digests: Result<Vec<String>, String> = unique_event_ids.iter().map(|id| {
            let event = event_map.get(id).ok_or_else(|| format!("missing evidence event: {id}"))?;
            let canonical_event = serde_jcs::to_vec(event)
                .map_err(|error| format!("cannot canonicalize evidence event: {error}"))?;
            Ok(digest_hex(&canonical_event))
        }).collect();
        oracle_results.push(json!({
            "oracle_id": rule_id,
            "severity": rule.get("severity"),
            "result": if passed { "pass" } else { "fail" },
            "rule_sha256": digest_hex(&canonical_rule),
            "evidence_event_sha256": event_digests?,
            "evidence_artifact_sha256": [],
            "diagnostic_code": if passed { "derived_from_raw_event_trace" } else { replay_error.as_deref().unwrap_or("semantic_state_mismatch") }
        }));
    }

    Ok(json!({
        "schema": "sfeb-event-oracle-verification-v1",
        "verification_id": format!("SFEB-VERIFY-{}", field_str(trace, "trace_id")?),
        "scope": "deterministic_event_semantics_only",
        "trace_sha256": digest_hex(trace_bytes),
        "fixture_sha256": actual_fixture_sha,
        "policy_sha256": digest_hex(policy_bytes),
        "case_id": case_id,
        "semantic_decision": if any_failed { "rejected" } else { "accepted" },
        "worker_claims_consulted": false,
        "oracle_results": oracle_results,
        "derived_state": actual_snapshot,
        "claim_boundary": "This offline evaluator authenticates neither browser execution nor execution identity. It does not produce a signed verifier attestation and does not qualify the UI candidate."
    }))
}

fn run_event_trace_verification(
    trace_path: &str,
    fixture_path: &str,
    policy_path: &str,
) -> Result<(), String> {
    let trace_bytes = fs::read(trace_path).map_err(|error| format!("cannot read trace {trace_path}: {error}"))?;
    let fixture_bytes = fs::read(fixture_path).map_err(|error| format!("cannot read fixture {fixture_path}: {error}"))?;
    let policy_bytes = fs::read(policy_path).map_err(|error| format!("cannot read policy {policy_path}: {error}"))?;
    let trace: Value = serde_json::from_slice(&trace_bytes).map_err(|error| format!("invalid trace JSON: {error}"))?;
    let fixture: Value = serde_json::from_slice(&fixture_bytes).map_err(|error| format!("invalid fixture JSON: {error}"))?;
    let policy: Value = serde_json::from_slice(&policy_bytes).map_err(|error| format!("invalid policy JSON: {error}"))?;
    let report = verify_event_trace_values(&fixture,&fixture_bytes,&policy,&policy_bytes,&trace,&trace_bytes)?;
    println!("{}",serde_json::to_string_pretty(&report).map_err(|error|error.to_string())?);
    if report.get("semantic_decision").and_then(Value::as_str) != Some("accepted") {
        return Err("independent event semantic replay rejected the observed trace".to_string());
    }
    Ok(())
}


fn main_result(path: &str) -> Result<(), String> {
    let bytes = fs::read(path).map_err(|error| format!("cannot read fixture {path}: {error}"))?;
    let fixture: Value = serde_json::from_slice(&bytes).map_err(|error| format!("invalid fixture JSON: {error}"))?;
    let (id, cases, relations, negatives) = run_fixture(&fixture)?;
    println!("SFEB event reconciliation model valid: {id} (golden={cases}, metamorphic={relations}, negative_controls={negatives})");
    println!("Offline semantics only; no browser, Leptos candidate, network, or trusted verifier is exercised.");
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let result = if args.first().is_some_and(|arg| arg == "--verify-event-trace") {
        if args.len() > 4 {
            Err("--verify-event-trace accepts TRACE_JSON [FIXTURE_JSON [POLICY_JSON]]".to_string())
        } else {
            run_event_trace_verification(
                args.get(1).map(String::as_str).unwrap_or(DEFAULT_TRACE),
                args.get(2).map(String::as_str).unwrap_or(DEFAULT_FIXTURE),
                args.get(3).map(String::as_str).unwrap_or(DEFAULT_POLICY),
            )
        }
    } else {
        main_result(args.first().map(String::as_str).unwrap_or(DEFAULT_FIXTURE))
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => { eprintln!("sfeb-reconciliation-check: {error}"); ExitCode::FAILURE }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        serde_json::from_str(include_str!("../event-reconciliation-v1.json")).expect("fixture JSON")
    }

    #[test]
    fn all_golden_cases_metamorphic_pairs_and_negative_controls_match() {
        let (id, cases, relations, negatives) = run_fixture(&fixture()).expect("event fixture should pass");
        assert_eq!(id, "SFEB-INT-004-EVENTS-001");
        assert_eq!((cases, relations, negatives), (6, 1, 6));
    }

    #[test]
    fn event_shape_rejects_unknown_fields_and_unsafe_sequences() {
        let f = fixture();
        let mut state = State::new(&f["initial_state"]).unwrap();
        let mut unknown = f["events"][0].clone();
        unknown["worker_disposition"] = json!("accepted");
        assert!(state.deliver(&unknown).unwrap_err().contains("unknown event field"));
        let mut unsafe_sequence = f["events"][0].clone();
        unsafe_sequence["sequence"] = json!(9_007_199_254_740_992u64);
        assert!(state.deliver(&unsafe_sequence).unwrap_err().contains("safe positive-integer range"));
    }

    #[test]
    fn reorder_window_is_bounded_and_rejection_is_atomic() {
        let f = fixture();
        let mut state = State::new(&f["initial_state"]).unwrap();
        let mut far_future = f["events"][0].clone();
        far_future["sequence"] = json!(MAX_REORDER_WINDOW + 1);
        assert!(state.deliver(&far_future).unwrap_err().contains("bounded reorder window"));
        assert!(state.seen_by_id.is_empty());
        assert!(state.sequence_owner.is_empty());
        assert!(state.buffered.is_empty());
        state.deliver(&f["events"][0]).unwrap();
        assert_eq!(state.last_sequence, 1);
    }

    #[test]
    fn duplicate_delivery_is_idempotent_and_audit_is_once() {
        let f = fixture();
        let mut state = State::new(&f["initial_state"]).unwrap();
        let event = &f["events"][0];
        state.deliver(event).unwrap();
        state.deliver(event).unwrap();
        assert_eq!(state.last_sequence, 1);
        assert_eq!(state.audit, vec!["evt-001".to_string()]);
        assert_eq!(state.duplicate_deliveries, 1);
    }

    #[test]
    fn out_of_order_delivery_converges_after_gap_closes() {
        let f = fixture();
        let mut state = State::new(&f["initial_state"]).unwrap();
        state.deliver(&f["events"][3]).unwrap();
        assert_eq!(state.last_sequence, 0);
        for index in [2usize, 1, 0] { state.deliver(&f["events"][index]).unwrap(); }
        assert_eq!(state.last_sequence, 4);
        assert!(state.buffered.is_empty());
        assert_eq!(state.jobs["JOB-A"]["status"], "in_progress");
    }

    #[test]
    fn rejected_event_does_not_poison_identity_or_sequence_for_valid_retry() {
        let f = fixture();
        let mut state = State::new(&f["initial_state"]).unwrap();
        let original = f["events"][0].clone();
        let mut altered = original.clone();
        altered["to_status"] = json!("done");
        assert!(state.deliver(&altered).is_err());
        assert_eq!(state.last_sequence, 0);
        assert!(state.audit.is_empty());
        assert!(state.seen_by_id.is_empty());
        state.deliver(&original).unwrap();
        assert_eq!(state.last_sequence, 1);
        assert_eq!(state.audit, vec!["evt-001"]);
    }


    #[test]
    fn independent_event_verifier_ignores_worker_claims() {
        let fixture_bytes = include_bytes!("../event-reconciliation-v1.json");
        let policy_bytes = include_bytes!("../event-oracle-policy-v1.json");
        let trace_bytes = include_bytes!("../examples/late-event-rollback-trace.json");
        let fixture: Value = serde_json::from_slice(fixture_bytes).unwrap();
        let policy: Value = serde_json::from_slice(policy_bytes).unwrap();
        let mut trace: Value = serde_json::from_slice(trace_bytes).unwrap();
        let fixture_sha = digest_hex(fixture_bytes);
        trace["fixture_sha256"] = json!(fixture_sha);
        let trace_bytes = serde_json::to_vec(&trace).unwrap();
        let first = verify_event_trace_values(&fixture,fixture_bytes,&policy,policy_bytes,&trace,&trace_bytes).unwrap();
        assert_eq!(first["semantic_decision"], "accepted");
        assert_eq!(first["worker_claims_consulted"], false);
        assert_eq!(first["oracle_results"].as_array().unwrap().len(), 7);

        trace["worker_claims"]["disposition"] = json!("rejected");
        trace["worker_claims"]["observed_oracle_ids"] = json!(["fake.failure","fake.security"]);
        let forged_worker_bytes = serde_json::to_vec(&trace).unwrap();
        let second = verify_event_trace_values(&fixture,fixture_bytes,&policy,policy_bytes,&trace,&forged_worker_bytes).unwrap();
        assert_eq!(first["oracle_results"], second["oracle_results"]);
        assert_eq!(second["semantic_decision"], "accepted");
    }

    #[test]
    fn independent_event_verifier_rejects_mutated_frozen_policy() {
        let fixture_bytes = include_bytes!("../event-reconciliation-v1.json");
        let policy_bytes = include_bytes!("../event-oracle-policy-v1.json");
        let trace_bytes = include_bytes!("../examples/late-event-rollback-trace.json");
        let fixture: Value = serde_json::from_slice(fixture_bytes).unwrap();
        let mut policy: Value = serde_json::from_slice(policy_bytes).unwrap();
        let trace: Value = serde_json::from_slice(trace_bytes).unwrap();
        policy["rules"][0]["required"] = json!(false);
        let altered_policy_bytes = serde_json::to_vec(&policy).unwrap();
        let error = verify_event_trace_values(&fixture,fixture_bytes,&policy,&altered_policy_bytes,&trace,trace_bytes)
            .expect_err("modified policy must not be accepted");
        assert!(error.contains("policy digest"));
    }

    #[test]
    fn independent_event_verifier_rejects_event_payload_not_in_frozen_fixture() {
        let fixture_bytes = include_bytes!("../event-reconciliation-v1.json");
        let policy_bytes = include_bytes!("../event-oracle-policy-v1.json");
        let trace_bytes = include_bytes!("../examples/late-event-rollback-trace.json");
        let fixture: Value = serde_json::from_slice(fixture_bytes).unwrap();
        let policy: Value = serde_json::from_slice(policy_bytes).unwrap();
        let mut trace: Value = serde_json::from_slice(trace_bytes).unwrap();
        trace["fixture_sha256"] = json!(digest_hex(fixture_bytes));
        trace["steps"][1]["event"]["to_status"] = json!("done");
        let changed_bytes = serde_json::to_vec(&trace).unwrap();
        let error = verify_event_trace_values(&fixture,fixture_bytes,&policy,policy_bytes,&trace,&changed_bytes)
            .expect_err("altered event payload must be rejected");
        assert!(error.contains("differs from frozen authoritative event"));
    }

    #[test]
    fn permission_revocation_rejects_and_rolls_back_pending_action() {
        let f = fixture();
        let cases = f["cases"].as_array().unwrap();
        let case = cases.iter().find(|c| c["case_id"] == "permission-revocation-invalidates-pending-action").unwrap();
        let events = f["events"].as_array().unwrap().iter()
            .map(|e| (e["event_id"].as_str().unwrap().to_string(), e.clone())).collect();
        let result = run_case(&f, case, &events).unwrap();
        assert_eq!(result["action_results"]["act-revoked"]["reason"], "permission_revoked");
        assert_eq!(result["visible_statuses"]["JOB-B"], "open");
    }
}
