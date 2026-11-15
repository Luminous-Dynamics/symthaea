//! Deterministic actor-aware temporal replay for EGDDM.
//! Hash chaining detects mutation of the supplied ledger, but does not authenticate
//! source records, actor availability claims, Git objects, signatures, or authority.

use std::collections::{BTreeMap, HashSet};
use chrono::{DateTime, FixedOffset};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

const GENESIS_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const LEDGER_SCHEMA_VERSION: &str = "0.1.0";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventLedger {
    pub schema_version: String,
    pub ledger_id: String,
    pub repository_full_name: String,
    pub events: Vec<LedgerEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LedgerEvent {
    pub event_id: String,
    pub sequence: u64,
    pub previous_hash: String,
    pub event_hash: String,
    pub body: LedgerEventBody,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LedgerEventBody {
    pub operation: EventOperation,
    pub fact_id: String,
    pub recorded_at: String,
    pub observed_at: String,
    pub actor_availability: Vec<ActorAvailability>,
    pub scope: EventScope,
    pub effective_at: Option<String>,
    pub valid_from: Option<String>,
    pub valid_through: Option<String>,
    pub validity_completeness: Option<ValidityCompleteness>,
    pub target_event_id: Option<String>,
    pub payload: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventOperation { Assert, Invalidate, Supersede, Propose }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidityCompleteness { Complete, Incomplete }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActorAvailability {
    pub actor_id: String,
    pub available_at: String,
    pub proof_ref: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventScope {
    pub revision: String,
    pub tree_oid: String,
    pub configuration_world_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayQuery {
    pub repository_full_name: String,
    pub configuration_world_digest: String,
    pub actor_id: String,
    pub knowledge_cutoff_at: String,
    pub valid_at: String,
    /// Optional ledger/system-time bound. Without it, late-captured evidence
    /// can be considered when its actor-availability claim predates the cutoff.
    pub ledger_recorded_cutoff_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayDiagnostic {
    pub code: String,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerValidationReport {
    pub valid: bool,
    pub event_count: usize,
    pub verified_head_hash: String,
    pub diagnostics: Vec<ReplayDiagnostic>,
    pub claim_ceiling: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactLifecycle { Current, Stale, Superseded, Indeterminate }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactProjection {
    pub fact_id: String,
    pub assertion_event_id: String,
    pub source_revision: String,
    pub source_tree_oid: String,
    pub lifecycle: FactLifecycle,
    pub effective_at_query: bool,
    /// Temporal eligibility only; it does not establish authenticity or authority.
    pub temporally_admissible: bool,
    pub valid_from: Option<String>,
    pub valid_through: Option<String>,
    pub transition_event_ids: Vec<String>,
    pub payload: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalProjection {
    pub fact_id: String,
    pub event_id: String,
    pub source_revision: String,
    pub source_tree_oid: String,
    pub payload: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExcludedEvent {
    pub event_id: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayProjection {
    pub schema_version: String,
    pub ledger_id: String,
    pub query: ReplayQuery,
    pub facts: Vec<FactProjection>,
    pub proposals: Vec<ProposalProjection>,
    pub included_event_ids: Vec<String>,
    pub excluded_events: Vec<ExcludedEvent>,
    pub verified_ledger_head_hash: String,
    pub claim_ceiling: String,
}

#[derive(Serialize)]
struct HashMaterial<'a> {
    schema_version: &'a str,
    ledger_id: &'a str,
    repository_full_name: &'a str,
    event_id: &'a str,
    sequence: u64,
    previous_hash: &'a str,
    body: &'a LedgerEventBody,
}

fn diagnostic(code: &str, path: impl Into<String>, message: impl Into<String>) -> ReplayDiagnostic {
    ReplayDiagnostic { code: code.to_owned(), path: path.into(), message: message.into() }
}
fn parse_time(raw: &str) -> Result<DateTime<FixedOffset>, String> {
    DateTime::parse_from_rfc3339(raw).map_err(|_| format!("invalid RFC 3339 timestamp {raw:?}"))
}
fn valid_oid(raw: &str) -> bool {
    matches!(raw.len(), 40 | 64) && raw.bytes().all(|byte| byte.is_ascii_hexdigit())
}
fn valid_sha256(raw: &str) -> bool {
    raw.len() == 64 && raw.bytes().all(|byte| byte.is_ascii_hexdigit())
}
fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut result = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(&mut result, "{byte:02x}");
    }
    result
}
fn calculate_event_hash(
    schema_version: &str,
    ledger_id: &str,
    repository_full_name: &str,
    event: &LedgerEvent,
) -> Result<String, String> {
    let material = HashMaterial {
        schema_version,
        ledger_id,
        repository_full_name,
        event_id: &event.event_id,
        sequence: event.sequence,
        previous_hash: &event.previous_hash,
        body: &event.body,
    };
    Ok(sha256_hex(&serde_json::to_vec(&material).map_err(|error| error.to_string())?))
}
fn actor_availability<'a>(body: &'a LedgerEventBody, actor_id: &str) -> Option<&'a ActorAvailability> {
    body.actor_availability.iter().find(|item| item.actor_id == actor_id)
}

fn validate_body(
    event: &LedgerEvent,
    index: usize,
    prior_events: &BTreeMap<String, LedgerEvent>,
    diagnostics: &mut Vec<ReplayDiagnostic>,
) {
    let body = &event.body;
    let path = format!("events[{index}].body");
    if body.fact_id.trim().is_empty() {
        diagnostics.push(diagnostic("D2-FACT-001", format!("{path}.fact_id"), "fact_id must not be empty"));
    }
    if !body.payload.is_object() {
        diagnostics.push(diagnostic("D2-PAYLOAD-001", format!("{path}.payload"), "payload must be a JSON object"));
    }
    let recorded = match parse_time(&body.recorded_at) {
        Ok(value) => Some(value),
        Err(error) => { diagnostics.push(diagnostic("D2-TIME-001", format!("{path}.recorded_at"), error)); None }
    };
    let observed = match parse_time(&body.observed_at) {
        Ok(value) => Some(value),
        Err(error) => { diagnostics.push(diagnostic("D2-TIME-002", format!("{path}.observed_at"), error)); None }
    };
    if observed.as_ref().zip(recorded.as_ref()).is_some_and(|(o, r)| o > r) {
        diagnostics.push(diagnostic("D2-TIME-003", &path, "ledger recording cannot precede source observation"));
    }
    if body.actor_availability.is_empty() {
        diagnostics.push(diagnostic("D2-AVAIL-001", format!("{path}.actor_availability"), "at least one actor-availability record is required"));
    }
    let mut actor_ids = HashSet::new();
    for (i, item) in body.actor_availability.iter().enumerate() {
        let item_path = format!("{path}.actor_availability[{i}]");
        if item.actor_id.trim().is_empty() || item.proof_ref.trim().is_empty() {
            diagnostics.push(diagnostic("D2-AVAIL-002", &item_path, "actor_id and proof_ref must not be empty"));
        }
        if !actor_ids.insert(item.actor_id.clone()) {
            diagnostics.push(diagnostic("D2-AVAIL-003", format!("{item_path}.actor_id"), "actor availability must be unique per actor"));
        }
        match parse_time(&item.available_at) {
            Ok(available) if observed.as_ref().is_some_and(|o| &available < o) => diagnostics.push(diagnostic(
                "D2-AVAIL-004", format!("{item_path}.available_at"), "actor availability cannot precede source observation")),
            Ok(_) => {}
            Err(error) => diagnostics.push(diagnostic("D2-TIME-004", format!("{item_path}.available_at"), error)),
        }
    }
    if !valid_oid(&body.scope.revision) {
        diagnostics.push(diagnostic("D2-SCOPE-001", format!("{path}.scope.revision"), "revision must be a 40- or 64-character hexadecimal Git object ID"));
    }
    if !valid_oid(&body.scope.tree_oid) || body.scope.tree_oid.len() != body.scope.revision.len() {
        diagnostics.push(diagnostic("D2-SCOPE-002", format!("{path}.scope.tree_oid"), "tree_oid must use the revision's Git object-ID length"));
    }
    if !valid_sha256(&body.scope.configuration_world_digest) {
        diagnostics.push(diagnostic("D2-SCOPE-003", format!("{path}.scope.configuration_world_digest"), "configuration world must be a SHA-256 digest"));
    }

    match body.operation {
        EventOperation::Assert => {
            if body.target_event_id.is_some() || body.effective_at.is_some() {
                diagnostics.push(diagnostic("D2-EVENT-001", &path, "assert events cannot have target_event_id or effective_at"));
            }
            if body.validity_completeness.is_none() {
                diagnostics.push(diagnostic("D2-VALIDITY-001", format!("{path}.validity_completeness"), "assert events require explicit validity_completeness"));
            }
            let from = body.valid_from.as_deref().map(parse_time).transpose();
            let through = body.valid_through.as_deref().map(parse_time).transpose();
            if let Err(error) = &from { diagnostics.push(diagnostic("D2-TIME-005", format!("{path}.valid_from"), error)); }
            if let Err(error) = &through { diagnostics.push(diagnostic("D2-TIME-006", format!("{path}.valid_through"), error)); }
            if from.ok().flatten().zip(through.ok().flatten()).is_some_and(|(a, b)| a > b) {
                diagnostics.push(diagnostic("D2-VALIDITY-002", &path, "validity interval must satisfy valid_from <= valid_through"));
            }
        }
        EventOperation::Invalidate | EventOperation::Supersede => {
            if body.target_event_id.is_none() || body.effective_at.is_none() {
                diagnostics.push(diagnostic("D2-EVENT-002", &path, "invalidate/supersede require target_event_id and effective_at"));
            }
            if body.validity_completeness.is_some() || body.valid_from.is_some() || body.valid_through.is_some() {
                diagnostics.push(diagnostic("D2-EVENT-003", &path, "lifecycle transitions cannot define assertion validity bounds"));
            }
            if let Some(effective_at) = body.effective_at.as_deref() {
                if let Err(error) = parse_time(effective_at) { diagnostics.push(diagnostic("D2-TIME-007", format!("{path}.effective_at"), error)); }
            }
            if let Some(target_id) = body.target_event_id.as_deref() {
                match prior_events.get(target_id) {
                    None => diagnostics.push(diagnostic("D2-TARGET-001", format!("{path}.target_event_id"), "target must be an earlier event in the same ledger")),
                    Some(target) => {
                        if target.body.operation != EventOperation::Assert {
                            diagnostics.push(diagnostic("D2-TARGET-002", format!("{path}.target_event_id"), "target must be an assertion event"));
                        }
                        if target.body.fact_id != body.fact_id {
                            diagnostics.push(diagnostic("D2-TARGET-003", format!("{path}.target_event_id"), "target must refer to the same fact_id"));
                        }
                        if target.body.scope.configuration_world_digest != body.scope.configuration_world_digest {
                            diagnostics.push(diagnostic("D2-TARGET-004", format!("{path}.target_event_id"), "target and transition must belong to the same configuration world"));
                        }
                        if prior_events.values().any(|prior| {
                            prior.body.target_event_id.as_deref() == Some(target_id)
                                && matches!(prior.body.operation, EventOperation::Invalidate | EventOperation::Supersede)
                        }) {
                            diagnostics.push(diagnostic("D2-TARGET-007", format!("{path}.target_event_id"), "an assertion may receive only one terminal lifecycle transition"));
                        }
                        for transition_actor in &body.actor_availability {
                            match actor_availability(&target.body, &transition_actor.actor_id) {
                                Some(target_actor) => {
                                    if let (Ok(target_at), Ok(transition_at)) = (parse_time(&target_actor.available_at), parse_time(&transition_actor.available_at)) {
                                        if target_at > transition_at {
                                            diagnostics.push(diagnostic("D2-TARGET-005", format!("{path}.actor_availability"), "actor cannot receive a transition before its target assertion"));
                                        }
                                    }
                                }
                                None => diagnostics.push(diagnostic("D2-TARGET-006", format!("{path}.actor_availability"), "each transition actor must have availability evidence for the target assertion")),
                            }
                        }
                    }
                }
            }
        }
        EventOperation::Propose => {
            if body.target_event_id.is_some() || body.effective_at.is_some()
                || body.validity_completeness.is_some() || body.valid_from.is_some() || body.valid_through.is_some() {
                diagnostics.push(diagnostic("D2-EVENT-004", &path, "proposals cannot mutate an asserted fact or define assertion validity"));
            }
        }
    }
}

pub fn validate_ledger(ledger: &EventLedger) -> LedgerValidationReport {
    let mut diagnostics = Vec::new();
    if ledger.schema_version != LEDGER_SCHEMA_VERSION {
        diagnostics.push(diagnostic("D2-SCHEMA-001", "schema_version", format!("expected {LEDGER_SCHEMA_VERSION}, found {:?}", ledger.schema_version)));
    }
    if ledger.ledger_id.trim().is_empty() {
        diagnostics.push(diagnostic("D2-LEDGER-001", "ledger_id", "ledger_id must not be empty"));
    }
    if !ledger.repository_full_name.contains('/') || ledger.repository_full_name.chars().any(char::is_whitespace) {
        diagnostics.push(diagnostic("D2-LEDGER-002", "repository_full_name", "repository_full_name must be a compact owner/repository identity"));
    }

    let mut prior_events: BTreeMap<String, LedgerEvent> = BTreeMap::new();
    let mut previous_hash = GENESIS_HASH.to_owned();
    let mut previous_recorded: Option<DateTime<FixedOffset>> = None;
    for (index, event) in ledger.events.iter().enumerate() {
        let path = format!("events[{index}]");
        let expected_sequence = index as u64 + 1;
        if event.sequence != expected_sequence {
            diagnostics.push(diagnostic("D2-SEQUENCE-001", format!("{path}.sequence"), format!("expected {expected_sequence}, found {}", event.sequence)));
        }
        if event.event_id.trim().is_empty() {
            diagnostics.push(diagnostic("D2-ID-001", format!("{path}.event_id"), "event_id must not be empty"));
        }
        if prior_events.contains_key(&event.event_id) {
            diagnostics.push(diagnostic("D2-ID-002", format!("{path}.event_id"), format!("duplicate event_id {:?}", event.event_id)));
        }
        if event.previous_hash != previous_hash {
            diagnostics.push(diagnostic("D2-HASH-001", format!("{path}.previous_hash"), "previous_hash does not match the preceding event hash"));
        }
        if !valid_sha256(&event.previous_hash) || !valid_sha256(&event.event_hash) {
            diagnostics.push(diagnostic("D2-HASH-002", &path, "event hashes must be 64-character SHA-256 digests"));
        }
        match calculate_event_hash(&ledger.schema_version, &ledger.ledger_id, &ledger.repository_full_name, event) {
            Ok(expected) if expected == event.event_hash => {}
            Ok(_) => diagnostics.push(diagnostic("D2-HASH-003", format!("{path}.event_hash"), "event_hash does not match canonical event content")),
            Err(error) => diagnostics.push(diagnostic("D2-HASH-004", &path, error)),
        }
        validate_body(event, index, &prior_events, &mut diagnostics);
        if let Ok(recorded) = parse_time(&event.body.recorded_at) {
            if previous_recorded.as_ref().is_some_and(|previous| &recorded < previous) {
                diagnostics.push(diagnostic("D2-TIME-008", format!("{path}.body.recorded_at"), "recorded_at must be non-decreasing in append order"));
            }
            previous_recorded = Some(recorded);
        }
        previous_hash = event.event_hash.clone();
        prior_events.insert(event.event_id.clone(), event.clone());
    }

    LedgerValidationReport {
        valid: diagnostics.is_empty(),
        event_count: ledger.events.len(),
        verified_head_hash: previous_hash,
        diagnostics,
        claim_ceiling: "Internal hash-chain and structural validation only; no external source, actor-availability proof, Git object, signature, runtime, CI, or authority record is authenticated.".to_owned(),
    }
}

/// Append one event atomically. Existing history is validated first; an invalid
/// candidate is rolled back and never left in the in-memory ledger.
pub fn append_event(
    ledger: &mut EventLedger,
    event_id: String,
    body: LedgerEventBody,
) -> Result<String, String> {
    let before = validate_ledger(ledger);
    if !before.valid {
        return Err(format!("cannot append to an invalid ledger: {}", before.diagnostics.iter()
            .map(|item| format!("{} {}: {}", item.code, item.path, item.message))
            .collect::<Vec<_>>().join("; ")));
    }
    if event_id.trim().is_empty() || ledger.events.iter().any(|event| event.event_id == event_id) {
        return Err("event_id must be non-empty and unique within the ledger".to_owned());
    }
    let sequence = ledger.events.len() as u64 + 1;
    let previous_hash = ledger.events.last()
        .map(|event| event.event_hash.clone())
        .unwrap_or_else(|| GENESIS_HASH.to_owned());
    let mut event = LedgerEvent {
        event_id,
        sequence,
        previous_hash,
        event_hash: String::new(),
        body,
    };
    event.event_hash = calculate_event_hash(
        &ledger.schema_version,
        &ledger.ledger_id,
        &ledger.repository_full_name,
        &event,
    )?;
    let event_hash = event.event_hash.clone();
    ledger.events.push(event);
    let after = validate_ledger(ledger);
    if !after.valid {
        ledger.events.pop();
        return Err(format!("candidate event rejected: {}", after.diagnostics.iter()
            .map(|item| format!("{} {}: {}", item.code, item.path, item.message))
            .collect::<Vec<_>>().join("; ")));
    }
    Ok(event_hash)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReplayAccumulator {
    facts: BTreeMap<String, FactProjection>,
    proposals: Vec<ProposalProjection>,
    included: Vec<String>,
    excluded: Vec<ExcludedEvent>,
}
impl ReplayAccumulator {
    fn new() -> Self {
        Self { facts: BTreeMap::new(), proposals: Vec::new(), included: Vec::new(), excluded: Vec::new() }
    }
    fn append(
        &mut self,
        event: &LedgerEvent,
        query: &ReplayQuery,
        knowledge_cutoff: &DateTime<FixedOffset>,
        valid_at: &DateTime<FixedOffset>,
        ledger_cutoff: Option<&DateTime<FixedOffset>>,
    ) -> Result<(), String> {
        if event.body.scope.configuration_world_digest != query.configuration_world_digest {
            self.excluded.push(ExcludedEvent { event_id: event.event_id.clone(), reason: "configuration-world digest differs from query".to_owned() });
            return Ok(());
        }
        if let Some(cutoff) = ledger_cutoff {
            if parse_time(&event.body.recorded_at)? > *cutoff {
                self.excluded.push(ExcludedEvent { event_id: event.event_id.clone(), reason: "event was recorded after ledger_recorded_cutoff_at".to_owned() });
                return Ok(());
            }
        }
        let Some(availability) = actor_availability(&event.body, &query.actor_id) else {
            self.excluded.push(ExcludedEvent { event_id: event.event_id.clone(), reason: "no availability record for replay actor".to_owned() });
            return Ok(());
        };
        if parse_time(&availability.available_at)? > *knowledge_cutoff {
            self.excluded.push(ExcludedEvent { event_id: event.event_id.clone(), reason: "event became available after knowledge_cutoff_at".to_owned() });
            return Ok(());
        }
        self.included.push(event.event_id.clone());

        match event.body.operation {
            EventOperation::Assert => {
                let complete = event.body.validity_completeness == Some(ValidityCompleteness::Complete);
                let from = event.body.valid_from.as_deref().map(parse_time).transpose()?;
                let through = event.body.valid_through.as_deref().map(parse_time).transpose()?;
                let within_bounds = from.as_ref().is_none_or(|bound| bound <= valid_at)
                    && through.as_ref().is_none_or(|bound| bound >= valid_at);
                let fact = FactProjection {
                    fact_id: event.body.fact_id.clone(),
                    assertion_event_id: event.event_id.clone(),
                    source_revision: event.body.scope.revision.clone(),
                    source_tree_oid: event.body.scope.tree_oid.clone(),
                    lifecycle: if complete { FactLifecycle::Current } else { FactLifecycle::Indeterminate },
                    effective_at_query: within_bounds,
                    temporally_admissible: complete && within_bounds,
                    valid_from: event.body.valid_from.clone(),
                    valid_through: event.body.valid_through.clone(),
                    transition_event_ids: Vec::new(),
                    payload: event.body.payload.clone(),
                };
                self.facts.insert(event.event_id.clone(), fact);
            }
            EventOperation::Invalidate | EventOperation::Supersede => {
                let effective = parse_time(event.body.effective_at.as_deref().ok_or("missing effective_at")?)?;
                if effective <= *valid_at {
                    let target_id = event.body.target_event_id.as_deref().ok_or("missing target_event_id")?;
                    let target = self.facts.get_mut(target_id).ok_or_else(|| format!(
                        "transition {} is visible but target assertion {} is not visible", event.event_id, target_id
                    ))?;
                    match target.lifecycle {
                        FactLifecycle::Current | FactLifecycle::Indeterminate => {
                            target.lifecycle = if event.body.operation == EventOperation::Invalidate {
                                FactLifecycle::Stale
                            } else { FactLifecycle::Superseded };
                            target.temporally_admissible = false;
                            target.transition_event_ids.push(event.event_id.clone());
                        }
                        FactLifecycle::Stale | FactLifecycle::Superseded => {
                            return Err(format!("assertion {target_id} received multiple terminal transitions"));
                        }
                    }
                }
            }
            EventOperation::Propose => self.proposals.push(ProposalProjection {
                fact_id: event.body.fact_id.clone(),
                event_id: event.event_id.clone(),
                source_revision: event.body.scope.revision.clone(),
                source_tree_oid: event.body.scope.tree_oid.clone(),
                payload: event.body.payload.clone(),
            }),
        }
        Ok(())
    }
}

fn validate_query(
    ledger: &EventLedger,
    query: &ReplayQuery,
) -> Result<(DateTime<FixedOffset>, DateTime<FixedOffset>, Option<DateTime<FixedOffset>>), String> {
    if query.repository_full_name != ledger.repository_full_name {
        return Err("replay query repository does not match the event ledger".to_owned());
    }
    if query.actor_id.trim().is_empty() { return Err("actor_id must not be empty".to_owned()); }
    if !valid_sha256(&query.configuration_world_digest) { return Err("configuration_world_digest must be a SHA-256 digest".to_owned()); }
    Ok((
        parse_time(&query.knowledge_cutoff_at)?,
        parse_time(&query.valid_at)?,
        query.ledger_recorded_cutoff_at.as_deref().map(parse_time).transpose()?,
    ))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayCursor {
    schema_version: String,
    ledger_id: String,
    repository_full_name: String,
    query: ReplayQuery,
    processed_sequence: u64,
    head_hash: String,
    last_recorded_at: Option<String>,
    prior_events: BTreeMap<String, LedgerEvent>,
    state: ReplayAccumulator,
    checkpoint_hash: String,
}

#[derive(Serialize)]
struct CursorHashMaterial<'a> {
    schema_version: &'a str,
    ledger_id: &'a str,
    repository_full_name: &'a str,
    query: &'a ReplayQuery,
    processed_sequence: u64,
    head_hash: &'a str,
    last_recorded_at: &'a Option<String>,
    prior_events: &'a BTreeMap<String, LedgerEvent>,
    state: &'a ReplayAccumulator,
}

fn calculate_cursor_hash(cursor: &ReplayCursor) -> Result<String, String> {
    let material = CursorHashMaterial {
        schema_version: &cursor.schema_version,
        ledger_id: &cursor.ledger_id,
        repository_full_name: &cursor.repository_full_name,
        query: &cursor.query,
        processed_sequence: cursor.processed_sequence,
        head_hash: &cursor.head_hash,
        last_recorded_at: &cursor.last_recorded_at,
        prior_events: &cursor.prior_events,
        state: &cursor.state,
    };
    Ok(sha256_hex(&serde_json::to_vec(&material).map_err(|error| error.to_string())?))
}

fn seal_cursor(cursor: &mut ReplayCursor) -> Result<(), String> {
    cursor.checkpoint_hash = calculate_cursor_hash(cursor)?;
    Ok(())
}

fn validate_cursor_checkpoint(cursor: &ReplayCursor) -> Result<(), String> {
    if cursor.schema_version != LEDGER_SCHEMA_VERSION {
        return Err("unsupported replay cursor schema version".to_owned());
    }
    if cursor.processed_sequence != cursor.prior_events.len() as u64 {
        return Err("replay cursor sequence does not match its prior-event index".to_owned());
    }
    if cursor.checkpoint_hash != calculate_cursor_hash(cursor)? {
        return Err("replay cursor checkpoint hash mismatch".to_owned());
    }
    let expected_head = cursor.prior_events.values().max_by_key(|event| event.sequence)
        .map(|event| event.event_hash.as_str())
        .unwrap_or(GENESIS_HASH);
    if cursor.head_hash != expected_head {
        return Err("replay cursor head hash does not match its prior-event index".to_owned());
    }
    if cursor.prior_events.iter().any(|(event_id, event)| event_id != &event.event_id) {
        return Err("replay cursor event index contains a mismatched event identity".to_owned());
    }
    let expected_recorded_at = cursor.prior_events.values().max_by_key(|event| event.sequence)
        .map(|event| event.body.recorded_at.as_str());
    if cursor.last_recorded_at.as_deref() != expected_recorded_at {
        return Err("replay cursor last_recorded_at does not match its final event".to_owned());
    }
    Ok(())
}

impl ReplayCursor {
    pub fn projection(&self) -> ReplayProjection {
        ReplayProjection {
            schema_version: self.schema_version.clone(),
            ledger_id: self.ledger_id.clone(),
            query: self.query.clone(),
            facts: self.state.facts.values().cloned().collect(),
            proposals: self.state.proposals.clone(),
            included_event_ids: self.state.included.clone(),
            excluded_events: self.state.excluded.clone(),
            verified_ledger_head_hash: self.head_hash.clone(),
            claim_ceiling: "Temporal replay and internal hash-chain/checkpoint validation only. Source authenticity, actor-availability proof authenticity, Git object identity, authority, execution, CI, and downstream correctness are not independently qualified.".to_owned(),
        }
    }
}

fn empty_cursor(ledger: &EventLedger, query: &ReplayQuery) -> Result<ReplayCursor, String> {
    validate_query(ledger, query)?;
    let mut cursor = ReplayCursor {
        schema_version: LEDGER_SCHEMA_VERSION.to_owned(),
        ledger_id: ledger.ledger_id.clone(),
        repository_full_name: ledger.repository_full_name.clone(),
        query: query.clone(),
        processed_sequence: 0,
        head_hash: GENESIS_HASH.to_owned(),
        last_recorded_at: None,
        prior_events: BTreeMap::new(),
        state: ReplayAccumulator::new(),
        checkpoint_hash: String::new(),
    };
    seal_cursor(&mut cursor)?;
    Ok(cursor)
}

/// Build a validated checkpoint by replaying a known prefix from genesis.
pub fn start_replay_cursor(prefix: &EventLedger, query: &ReplayQuery) -> Result<ReplayCursor, String> {
    let validation = validate_ledger(prefix);
    if !validation.valid {
        let errors = validation.diagnostics.iter()
            .map(|item| format!("{} {}: {}", item.code, item.path, item.message))
            .collect::<Vec<_>>().join("; ");
        return Err(format!("invalid temporal ledger prefix: {errors}"));
    }
    let mut cursor = empty_cursor(prefix, query)?;
    cursor = resume_replay_cursor(cursor, &prefix.events)?;
    Ok(cursor)
}

/// Validate a saved checkpoint and apply only the new append-only suffix.
pub fn resume_replay_cursor(
    mut cursor: ReplayCursor,
    appended: &[LedgerEvent],
) -> Result<ReplayCursor, String> {
    validate_cursor_checkpoint(&cursor)?;
    let (knowledge, valid_at, ledger_cutoff) = validate_query(
        &EventLedger {
            schema_version: cursor.schema_version.clone(),
            ledger_id: cursor.ledger_id.clone(),
            repository_full_name: cursor.repository_full_name.clone(),
            events: Vec::new(),
        },
        &cursor.query,
    )?;

    for (index, event) in appended.iter().enumerate() {
        let path = format!("appended_events[{index}]");
        let expected_sequence = cursor.processed_sequence + 1;
        if event.sequence != expected_sequence {
            return Err(format!("{path}.sequence: expected {expected_sequence}, found {}", event.sequence));
        }
        if event.event_id.trim().is_empty() || cursor.prior_events.contains_key(&event.event_id) {
            return Err(format!("{path}.event_id: empty or duplicate event ID"));
        }
        if event.previous_hash != cursor.head_hash {
            return Err(format!("{path}.previous_hash: does not match the cursor head"));
        }
        if !valid_sha256(&event.previous_hash) || !valid_sha256(&event.event_hash) {
            return Err(format!("{path}: event hashes must be SHA-256 digests"));
        }
        let expected_hash = calculate_event_hash(&cursor.schema_version, &cursor.ledger_id, &cursor.repository_full_name, event)?;
        if event.event_hash != expected_hash {
            return Err(format!("{path}.event_hash: content hash mismatch"));
        }

        let mut body_diagnostics = Vec::new();
        validate_body(event, index + cursor.processed_sequence as usize, &cursor.prior_events, &mut body_diagnostics);
        if !body_diagnostics.is_empty() {
            return Err(body_diagnostics.iter()
                .map(|item| format!("{} {}: {}", item.code, item.path, item.message))
                .collect::<Vec<_>>().join("; "));
        }

        let recorded = parse_time(&event.body.recorded_at)?;
        if cursor.last_recorded_at.as_deref().map(parse_time).transpose()?.is_some_and(|last| recorded < last) {
            return Err(format!("{path}.body.recorded_at: append timestamp moved backwards"));
        }

        cursor.state.append(event, &cursor.query, &knowledge, &valid_at, ledger_cutoff.as_ref())?;
        cursor.processed_sequence = expected_sequence;
        cursor.head_hash = event.event_hash.clone();
        cursor.last_recorded_at = Some(event.body.recorded_at.clone());
        cursor.prior_events.insert(event.event_id.clone(), event.clone());
    }
    seal_cursor(&mut cursor)?;
    Ok(cursor)
}

/// Clean replay reference: start at genesis and consume the entire ledger.
pub fn replay_ledger(ledger: &EventLedger, query: &ReplayQuery) -> Result<ReplayProjection, String> {
    Ok(start_replay_cursor(ledger, query)?.projection())
}

/// Start from a validated prefix checkpoint, then consume only the supplied suffix.
pub fn replay_incremental(
    prefix: &EventLedger,
    appended: &[LedgerEvent],
    query: &ReplayQuery,
) -> Result<ReplayProjection, String> {
    let cursor = start_replay_cursor(prefix, query)?;
    let cursor = resume_replay_cursor(cursor, appended)?;
    Ok(cursor.projection())
}

pub fn serialize_replay_cursor(cursor: &ReplayCursor) -> Result<String, String> {
    validate_cursor_checkpoint(cursor)?;
    serde_json::to_string_pretty(cursor).map_err(|error| error.to_string())
}

pub fn deserialize_replay_cursor(input: &str) -> Result<ReplayCursor, String> {
    let cursor: ReplayCursor = serde_json::from_str(input).map_err(|error| error.to_string())?;
    validate_cursor_checkpoint(&cursor)?;
    Ok(cursor)
}

pub fn validate_temporal_ledger_json(input: &str) -> Result<LedgerValidationReport, String> {
    let ledger: EventLedger = serde_json::from_str(input).map_err(|error| error.to_string())?;
    Ok(validate_ledger(&ledger))
}
pub fn replay_temporal_ledger_json(ledger_json: &str, query_json: &str) -> Result<ReplayProjection, String> {
    let ledger: EventLedger = serde_json::from_str(ledger_json).map_err(|error| error.to_string())?;
    let query: ReplayQuery = serde_json::from_str(query_json).map_err(|error| error.to_string())?;
    replay_ledger(&ledger, &query)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(operation: EventOperation, fact_id: &str, observed: &str, recorded: &str, available: &str) -> LedgerEventBody {
        LedgerEventBody {
            operation, fact_id: fact_id.to_owned(), recorded_at: recorded.to_owned(), observed_at: observed.to_owned(),
            actor_availability: vec![ActorAvailability {
                actor_id: "reviewer".to_owned(), available_at: available.to_owned(), proof_ref: "receipt://synthetic/proof".to_owned(),
            }],
            scope: EventScope {
                revision: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
                tree_oid: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_owned(),
                configuration_world_digest: "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".to_owned(),
            },
            effective_at: None, valid_from: None, valid_through: None,
            validity_completeness: None, target_event_id: None, payload: serde_json::json!({"result":"pass"}),
        }
    }
    fn reseal(ledger: &mut EventLedger) {
        let schema_version = ledger.schema_version.clone();
        let ledger_id = ledger.ledger_id.clone();
        let repository_full_name = ledger.repository_full_name.clone();
        let mut previous = GENESIS_HASH.to_owned();
        for (index, event) in ledger.events.iter_mut().enumerate() {
            event.sequence = index as u64 + 1;
            event.previous_hash = previous.clone();
            event.event_hash = calculate_event_hash(&schema_version, &ledger_id, &repository_full_name, event).expect("hash");
            previous = event.event_hash.clone();
        }
    }
    fn sample_ledger() -> EventLedger {
        let mut assertion = body(EventOperation::Assert, "contract-check", "2026-11-14T08:00:00Z", "2026-11-14T08:05:00Z", "2026-11-14T08:10:00Z");
        assertion.valid_from = Some("2026-11-14T08:00:00Z".to_owned());
        assertion.validity_completeness = Some(ValidityCompleteness::Complete);

        let mut proposal = body(EventOperation::Propose, "contract-v2-plan", "2026-11-14T08:29:00Z", "2026-11-14T08:30:00Z", "2026-11-14T08:31:00Z");
        proposal.payload = serde_json::json!({"state":"proposed_only"});

        let mut invalidation = body(EventOperation::Invalidate, "contract-check", "2026-11-14T10:15:00Z", "2026-11-14T10:20:00Z", "2026-11-14T10:30:00Z");
        invalidation.scope.revision = "dddddddddddddddddddddddddddddddddddddddd".to_owned();
        invalidation.scope.tree_oid = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee".to_owned();
        invalidation.effective_at = Some("2026-11-14T09:30:00Z".to_owned());
        invalidation.target_event_id = Some("event-001".to_owned());
        invalidation.payload = serde_json::json!({"reason":"retroactive correction"});

        let mut late = body(EventOperation::Assert, "late-captured-evidence", "2026-11-14T09:00:00Z", "2026-11-14T11:30:00Z", "2026-11-14T09:05:00Z");
        late.valid_from = Some("2026-11-14T09:00:00Z".to_owned());
        late.validity_completeness = Some(ValidityCompleteness::Complete);

        let mut ledger = EventLedger {
            schema_version: LEDGER_SCHEMA_VERSION.to_owned(),
            ledger_id: "synthetic-d2-ledger".to_owned(),
            repository_full_name: "synthetic/repository".to_owned(),
            events: vec![
                LedgerEvent { event_id: "event-001".to_owned(), sequence: 0, previous_hash: String::new(), event_hash: String::new(), body: assertion },
                LedgerEvent { event_id: "event-002".to_owned(), sequence: 0, previous_hash: String::new(), event_hash: String::new(), body: proposal },
                LedgerEvent { event_id: "event-003".to_owned(), sequence: 0, previous_hash: String::new(), event_hash: String::new(), body: invalidation },
                LedgerEvent { event_id: "event-004".to_owned(), sequence: 0, previous_hash: String::new(), event_hash: String::new(), body: late },
            ],
        };
        reseal(&mut ledger);
        ledger
    }
    fn query(knowledge: &str, valid: &str, recorded: Option<&str>) -> ReplayQuery {
        ReplayQuery {
            repository_full_name: "synthetic/repository".to_owned(),
            configuration_world_digest: "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".to_owned(),
            actor_id: "reviewer".to_owned(), knowledge_cutoff_at: knowledge.to_owned(), valid_at: valid.to_owned(),
            ledger_recorded_cutoff_at: recorded.map(str::to_owned),
        }
    }

    const FIXTURE_LEDGER: &str = include_str!("../../../docs/development-map/fixtures/temporal-event-ledger-v0.1.json");
    const FIXTURE_EXPECTED: &str = include_str!("../../../docs/development-map/fixtures/temporal-event-ledger-v0.1.expected.json");

    fn assert_public_fixture(query_text: &str, key: &str) {
        let ledger: EventLedger = serde_json::from_str(FIXTURE_LEDGER).expect("fixture ledger");
        let query: ReplayQuery = serde_json::from_str(query_text).expect("fixture query");
        let expected_root: Value = serde_json::from_str(FIXTURE_EXPECTED).expect("fixture expected states");
        let actual = replay_ledger(&ledger, &query).expect("candidate replay");
        let expected = &expected_root["queries"][key];

        let actual_included: Vec<String> = actual.included_event_ids.clone();
        let expected_included: Vec<String> = serde_json::from_value(expected["included_event_ids"].clone()).unwrap();
        assert_eq!(actual_included, expected_included);
        let actual_excluded: HashSet<String> = actual.excluded_events.iter().map(|item| item.event_id.clone()).collect();
        let expected_excluded: HashSet<String> = serde_json::from_value(expected["excluded_event_ids"].clone()).unwrap();
        assert_eq!(actual_excluded, expected_excluded);

        let expected_facts = expected["facts"].as_object().unwrap();
        assert_eq!(actual.facts.len(), expected_facts.len());
        for (event_id, expected_fact) in expected_facts {
            let actual_fact = actual.facts.iter().find(|fact| fact.assertion_event_id == *event_id).expect("expected fact");
            if let Some(value) = expected_fact.get("lifecycle") {
                assert_eq!(serde_json::to_value(&actual_fact.lifecycle).unwrap(), *value);
            }
            if let Some(value) = expected_fact.get("effective_at_query") {
                assert_eq!(actual_fact.effective_at_query, value.as_bool().unwrap());
            }
            if let Some(value) = expected_fact.get("temporally_admissible") {
                assert_eq!(actual_fact.temporally_admissible, value.as_bool().unwrap());
            }
            if let Some(value) = expected_fact.get("payload") {
                assert_eq!(actual_fact.payload, *value);
            }
        }
        assert_eq!(actual.proposals.len(), expected["proposals"].as_array().unwrap().len());
        assert_eq!(actual.verified_ledger_head_hash, expected_root["expected_ledger"]["head_hash"].as_str().unwrap());
    }

    #[test] fn public_actor_knowledge_fixture_matches_candidate_replay() {
        assert_public_fixture(
            include_str!("../../../docs/development-map/fixtures/temporal-replay-actor-knowledge-v0.1.json"),
            "temporal-replay-actor-knowledge-v0.1.json",
        );
    }
    #[test] fn public_ledger_as_of_fixture_matches_candidate_replay() {
        assert_public_fixture(
            include_str!("../../../docs/development-map/fixtures/temporal-replay-ledger-as-of-v0.1.json"),
            "temporal-replay-ledger-as-of-v0.1.json",
        );
    }
    #[test] fn public_retrospective_fixture_matches_candidate_replay() {
        assert_public_fixture(
            include_str!("../../../docs/development-map/fixtures/temporal-replay-restated-v0.1.json"),
            "temporal-replay-restated-v0.1.json",
        );
    }

    #[test] fn valid_hash_chain_is_accepted() { assert!(validate_ledger(&sample_ledger()).valid); }
    #[test] fn append_event_assigns_identity_sequence_and_hash_atomically() {
        let mut ledger = sample_ledger();
        let proposal = body(EventOperation::Propose, "next-change", "2026-11-14T12:00:00Z", "2026-11-14T12:01:00Z", "2026-11-14T12:02:00Z");
        let hash = append_event(&mut ledger, "event-005".to_owned(), proposal).expect("append");
        assert_eq!(ledger.events.len(), 5);
        assert_eq!(ledger.events[4].sequence, 5);
        assert_eq!(ledger.events[4].event_hash, hash);
        assert!(validate_ledger(&ledger).valid);
    }
    #[test] fn invalid_append_is_rolled_back() {
        let mut ledger = sample_ledger();
        let mut bad = body(EventOperation::Invalidate, "unknown-fact", "2026-11-14T12:00:00Z", "2026-11-14T12:01:00Z", "2026-11-14T12:02:00Z");
        bad.effective_at = Some("2026-11-14T11:00:00Z".to_owned());
        bad.target_event_id = Some("missing-target".to_owned());
        let before = ledger.events.len();
        assert!(append_event(&mut ledger, "event-invalid".to_owned(), bad).is_err());
        assert_eq!(ledger.events.len(), before);
        assert!(validate_ledger(&ledger).valid);
    }
    #[test] fn tampered_event_payload_is_rejected() {
        let mut ledger = sample_ledger();
        ledger.events[0].body.payload = serde_json::json!({"result":"fail"});
        assert!(validate_ledger(&ledger).diagnostics.iter().any(|d| d.code == "D2-HASH-003"));
    }
    #[test] fn sequence_gaps_are_rejected() {
        let mut ledger = sample_ledger();
        ledger.events[1].sequence = 9;
        assert!(validate_ledger(&ledger).diagnostics.iter().any(|d| d.code == "D2-SEQUENCE-001"));
    }
    #[test] fn later_correction_does_not_rewrite_earlier_actor_knowledge() {
        let ledger = sample_ledger();
        let early = replay_ledger(&ledger, &query("2026-11-14T10:00:00Z", "2026-11-14T10:00:00Z", None)).expect("early");
        assert_eq!(early.facts.iter().find(|f| f.assertion_event_id == "event-001").unwrap().lifecycle, FactLifecycle::Current);
        let restated = replay_ledger(&ledger, &query("2026-11-14T11:00:00Z", "2026-11-14T10:00:00Z", None)).expect("restated");
        assert_eq!(restated.facts.iter().find(|f| f.assertion_event_id == "event-001").unwrap().lifecycle, FactLifecycle::Stale);
    }
    #[test] fn ledger_cutoff_distinguishes_late_capture_from_actor_knowledge() {
        let ledger = sample_ledger();
        let retro = replay_ledger(&ledger, &query("2026-11-14T10:00:00Z", "2026-11-14T10:00:00Z", None)).expect("retro");
        assert!(retro.facts.iter().any(|f| f.assertion_event_id == "event-004"));
        let strict = replay_ledger(&ledger, &query("2026-11-14T10:00:00Z", "2026-11-14T10:00:00Z", Some("2026-11-14T10:00:00Z"))).expect("strict");
        assert!(!strict.facts.iter().any(|f| f.assertion_event_id == "event-004"));
    }
    #[test] fn proposal_is_visible_but_never_mutates_observed_facts() {
        let ledger = sample_ledger();
        let output = replay_ledger(&ledger, &query("2026-11-14T10:00:00Z", "2026-11-14T10:00:00Z", None)).expect("replay");
        assert_eq!(output.proposals.len(), 1);
        assert!(output.facts.iter().all(|f| f.fact_id != "contract-v2-plan"));
    }
    #[test] fn recorded_cutoff_excludes_events_not_yet_in_the_ledger() {
        let ledger = sample_ledger();
        let output = replay_ledger(&ledger, &query("2026-11-14T11:00:00Z", "2026-11-14T10:00:00Z", Some("2026-11-14T10:00:00Z"))).expect("replay");
        assert_eq!(output.facts.len(), 1);
        assert_eq!(output.facts[0].lifecycle, FactLifecycle::Current);
    }
    #[test] fn exact_configuration_world_is_required() {
        let ledger = sample_ledger();
        let mut other = query("2026-11-14T12:00:00Z", "2026-11-14T12:00:00Z", None);
        other.configuration_world_digest = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff".to_owned();
        let output = replay_ledger(&ledger, &other).expect("replay");
        assert!(output.facts.is_empty());
        assert_eq!(output.excluded_events.len(), ledger.events.len());
    }
    #[test] fn validity_boundaries_are_inclusive() {
        let mut ledger = sample_ledger();
        ledger.events[0].body.valid_through = Some("2026-11-14T10:00:00Z".to_owned());
        reseal(&mut ledger);
        let output = replay_ledger(&ledger, &query("2026-11-14T10:00:00Z", "2026-11-14T10:00:00Z", None)).expect("replay");
        assert!(output.facts.iter().find(|f| f.assertion_event_id == "event-001").unwrap().effective_at_query);
    }
    #[test] fn incomplete_validity_is_indeterminate_not_eligible() {
        let mut ledger = sample_ledger();
        ledger.events[0].body.validity_completeness = Some(ValidityCompleteness::Incomplete);
        reseal(&mut ledger);
        let output = replay_ledger(&ledger, &query("2026-11-14T10:00:00Z", "2026-11-14T10:00:00Z", None)).expect("replay");
        let fact = output.facts.iter().find(|f| f.assertion_event_id == "event-001").unwrap();
        assert_eq!(fact.lifecycle, FactLifecycle::Indeterminate);
        assert!(!fact.temporally_admissible);
    }
    #[test] fn incremental_suffix_replay_matches_clean_rebuild() {
        let ledger = sample_ledger();
        let prefix = EventLedger {
            schema_version: ledger.schema_version.clone(), ledger_id: ledger.ledger_id.clone(),
            repository_full_name: ledger.repository_full_name.clone(), events: ledger.events[..2].to_vec(),
        };
        let query = query("2026-11-14T11:00:00Z", "2026-11-14T10:00:00Z", None);
        let clean = replay_ledger(&ledger, &query).expect("clean");
        let incremental = replay_incremental(&prefix, &ledger.events[2..], &query).expect("incremental");
        assert_eq!(serde_json::to_value(clean).unwrap(), serde_json::to_value(incremental).unwrap());
    }
    #[test] fn missing_transition_target_is_rejected() {
        let mut ledger = sample_ledger();
        ledger.events[2].body.target_event_id = Some("missing".to_owned());
        reseal(&mut ledger);
        assert!(validate_ledger(&ledger).diagnostics.iter().any(|d| d.code == "D2-TARGET-001"));
    }
    #[test] fn transition_cannot_precede_target_availability() {
        let mut ledger = sample_ledger();
        ledger.events[0].body.actor_availability[0].available_at = "2026-11-14T10:25:00Z".to_owned();
        ledger.events[2].body.actor_availability[0].available_at = "2026-11-14T10:20:00Z".to_owned();
        reseal(&mut ledger);
        assert!(validate_ledger(&ledger).diagnostics.iter().any(|d| d.code == "D2-TARGET-005"));
    }
    #[test] fn event_reordering_breaks_sequence_and_hash_chain() {
        let mut ledger = sample_ledger();
        ledger.events.swap(0, 1);
        assert!(!validate_ledger(&ledger).valid);
    }
    #[test] fn repository_mismatch_is_rejected() {
        let ledger = sample_ledger();
        let mut query = query("2026-11-14T12:00:00Z", "2026-11-14T12:00:00Z", None);
        query.repository_full_name = "other/repository".to_owned();
        assert!(replay_ledger(&ledger, &query).is_err());
    }
    #[test] fn serialized_checkpoint_can_resume_only_the_suffix() {
        let ledger = sample_ledger();
        let prefix = EventLedger {
            schema_version: ledger.schema_version.clone(),
            ledger_id: ledger.ledger_id.clone(),
            repository_full_name: ledger.repository_full_name.clone(),
            events: ledger.events[..2].to_vec(),
        };
        let query = query("2026-11-14T11:00:00Z", "2026-11-14T10:00:00Z", None);
        let cursor = start_replay_cursor(&prefix, &query).expect("prefix cursor");
        let saved = serialize_replay_cursor(&cursor).expect("serialize cursor");
        let restored = deserialize_replay_cursor(&saved).expect("restore cursor");
        let resumed = resume_replay_cursor(restored, &ledger.events[2..]).expect("resume suffix");
        let clean = replay_ledger(&ledger, &query).expect("clean replay");
        assert_eq!(serde_json::to_value(clean).unwrap(), serde_json::to_value(resumed.projection()).unwrap());
    }
    #[test] fn tampered_checkpoint_is_rejected() {
        let ledger = sample_ledger();
        let query = query("2026-11-14T11:00:00Z", "2026-11-14T10:00:00Z", None);
        let cursor = start_replay_cursor(&ledger, &query).expect("cursor");
        let mut saved: Value = serde_json::from_str(&serialize_replay_cursor(&cursor).unwrap()).unwrap();
        saved["state"]["facts"] = serde_json::json!({});
        assert!(deserialize_replay_cursor(&saved.to_string()).is_err());
    }
    #[test] fn duplicate_terminal_transitions_are_rejected() {
        let mut ledger = sample_ledger();
        let mut second = ledger.events[2].clone();
        second.event_id = "event-005".to_owned();
        second.body.operation = EventOperation::Supersede;
        second.body.recorded_at = "2026-11-14T10:40:00Z".to_owned();
        second.body.observed_at = "2026-11-14T10:39:00Z".to_owned();
        second.body.actor_availability[0].available_at = "2026-11-14T10:45:00Z".to_owned();
        ledger.events.push(second);
        reseal(&mut ledger);
        assert!(validate_ledger(&ledger).diagnostics.iter().any(|d| d.code == "D2-TARGET-007"));
    }
}
