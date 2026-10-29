// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Policy-pinned bitemporal evidence governance for agent memory.
//!
//! This module is deliberately separate from the BM25 worker protocol. It
//! provides typed reference semantics for source admission, versioned authority,
//! and append-only retractions. Callers must authenticate source identity before
//! inserting events; a source string plus a hash is not authentication.

use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourcePolicy {
    pub source: String,
    pub authority_rank: u32,
    pub attributes: BTreeSet<String>,
    pub can_retract: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthorityPolicy {
    pub policy_version: u64,
    pub recorded_at: String,
    pub sources: Vec<SourcePolicy>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LedgerEvent {
    Assert {
        event_id: String,
        subject: String,
        attribute: String,
        value: String,
        valid_from: String,
        valid_to: Option<String>,
        recorded_at: String,
        source: String,
    },
    Retract {
        event_id: String,
        target_event_id: String,
        recorded_at: String,
        source: String,
    },
}

impl LedgerEvent {
    pub fn event_id(&self) -> &str {
        match self {
            Self::Assert { event_id, .. } | Self::Retract { event_id, .. } => event_id,
        }
    }

    pub fn source(&self) -> &str {
        match self {
            Self::Assert { source, .. } | Self::Retract { source, .. } => source,
        }
    }

    pub fn recorded_at(&self) -> &str {
        match self {
            Self::Assert { recorded_at, .. } | Self::Retract { recorded_at, .. } => recorded_at,
        }
    }

    fn assertion_fields(&self) -> Option<(&str, &str, &str, &str, Option<&str>)> {
        match self {
            Self::Assert {
                subject,
                attribute,
                value,
                valid_from,
                valid_to,
                ..
            } => Some((
                subject,
                attribute,
                value,
                valid_from,
                valid_to.as_deref(),
            )),
            Self::Retract { .. } => None,
        }
    }

    fn retraction_target(&self) -> Option<&str> {
        match self {
            Self::Retract { target_event_id, .. } => Some(target_event_id),
            Self::Assert { .. } => None,
        }
    }
}

/// Strict JSON event parser. Event payloads do not have an authority_rank field.
pub fn parse_event_json(input: &str) -> Result<LedgerEvent, String> {
    let value: serde_json::Value =
        serde_json::from_str(input).map_err(|error| format!("invalid event JSON: {error}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "event must be a JSON object".to_string())?;
    let kind = object
        .get("kind")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "event kind must be a string".to_string())?;
    let expected: BTreeSet<&str> = match kind {
        "assert" => [
            "event_id",
            "kind",
            "subject",
            "attribute",
            "value",
            "valid_from",
            "valid_to",
            "recorded_at",
            "source",
        ]
        .into_iter()
        .collect(),
        "retract" => [
            "event_id",
            "kind",
            "target_event_id",
            "recorded_at",
            "source",
        ]
        .into_iter()
        .collect(),
        _ => return Err("event kind must be assert or retract".into()),
    };
    let actual: BTreeSet<&str> = object.keys().map(String::as_str).collect();
    if actual != expected {
        let extra: Vec<&str> = actual.difference(&expected).copied().collect();
        let missing: Vec<&str> = expected.difference(&actual).copied().collect();
        return Err(format!(
            "event schema mismatch: extra={extra:?}, missing={missing:?}"
        ));
    }
    serde_json::from_value(value).map_err(|error| format!("invalid event fields: {error}"))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LedgerQuery {
    pub query_id: String,
    pub subject: String,
    pub attribute: String,
    pub valid_at: String,
    pub known_at: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolutionState {
    Resolved,
    Conflict,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectedEvidence {
    pub event_id: String,
    pub source: String,
    pub value: String,
    pub valid_from: String,
    pub valid_to: Option<String>,
    pub recorded_at: String,
    pub authority_rank: u32,
    pub record_policy_version: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExcludedEvent {
    pub event_id: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerResolution {
    pub query_id: String,
    pub state: ResolutionState,
    pub value: Option<String>,
    pub conflicting_values: Vec<String>,
    pub policy_version: Option<u64>,
    pub eligible_event_ids: Vec<String>,
    pub selected_event_ids: Vec<String>,
    pub selected_evidence: Vec<SelectedEvidence>,
    pub applied_retractions: Vec<String>,
    pub excluded_events: Vec<ExcludedEvent>,
    pub selected_valid_from: Option<String>,
    pub selected_authority_rank: Option<u32>,
    pub policy_sha256: String,
    pub resolution_rule: String,
}

#[derive(Debug, Clone)]
struct PolicyVersion {
    version: u64,
    recorded_at: DateTime<Utc>,
    sources: BTreeMap<String, SourcePolicy>,
}

#[derive(Debug, Clone)]
struct EventRecord {
    event: LedgerEvent,
    recorded_at: DateTime<Utc>,
    valid_from: Option<NaiveDate>,
    valid_to: Option<NaiveDate>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Admission {
    admitted: bool,
    record_policy_version: Option<u64>,
    reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PolicyPinnedLedger {
    policies: Vec<PolicyVersion>,
    events: Vec<EventRecord>,
    event_index: BTreeMap<String, usize>,
    admission: Vec<Admission>,
    policy_sha256: String,
}

fn parse_time(value: &str, field: &str) -> Result<DateTime<Utc>, String> {
    DateTime::parse_from_rfc3339(value)
        .map(|time| time.with_timezone(&Utc))
        .map_err(|error| format!("{field} must be RFC3339 with UTC offset: {error}"))
}

fn parse_day(value: &str, field: &str) -> Result<NaiveDate, String> {
    NaiveDate::parse_from_str(value, "%Y-%m-%d")
        .map_err(|error| format!("{field} must be YYYY-MM-DD: {error}"))
}

fn non_empty(value: &str, field: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{field} must not be empty"));
    }
    Ok(())
}

impl PolicyPinnedLedger {
    pub fn new(
        policy_inputs: Vec<AuthorityPolicy>,
        event_inputs: Vec<LedgerEvent>,
    ) -> Result<Self, String> {
        if policy_inputs.is_empty() {
            return Err("at least one authority policy is required".into());
        }
        if event_inputs.is_empty() {
            return Err("at least one ledger event is required".into());
        }

        let mut policies = Vec::with_capacity(policy_inputs.len());
        let mut last_version = 0_u64;
        let mut last_time: Option<DateTime<Utc>> = None;
        for (index, policy) in policy_inputs.into_iter().enumerate() {
            if policy.policy_version == 0 || policy.policy_version <= last_version {
                return Err(format!("policy row {index}: versions must increase strictly from 1"));
            }
            let recorded_at = parse_time(&policy.recorded_at, "policy recorded_at")?;
            if last_time.is_some_and(|previous| recorded_at <= previous) {
                return Err(format!("policy row {index}: recorded_at must increase strictly"));
            }
            if policy.sources.is_empty() {
                return Err(format!("policy row {index}: sources must not be empty"));
            }
            let mut sources = BTreeMap::new();
            for source in policy.sources {
                non_empty(&source.source, "policy source")?;
                if source.attributes.is_empty()
                    || source.attributes.iter().any(|attribute| attribute.trim().is_empty())
                {
                    return Err(format!(
                        "policy row {index}: source {} requires non-empty attribute scopes",
                        source.source
                    ));
                }
                if sources.insert(source.source.clone(), source).is_some() {
                    return Err(format!("policy row {index}: duplicate source name"));
                }
            }
            policies.push(PolicyVersion {
                version: policy.policy_version,
                recorded_at,
                sources,
            });
            last_version = policy.policy_version;
            last_time = Some(recorded_at);
        }

        let mut events = Vec::with_capacity(event_inputs.len());
        let mut event_index = BTreeMap::new();
        for (index, event) in event_inputs.into_iter().enumerate() {
            non_empty(event.event_id(), "event_id")?;
            non_empty(event.source(), "event source")?;
            let recorded_at = parse_time(event.recorded_at(), "event recorded_at")?;
            let (valid_from, valid_to) = match event.assertion_fields() {
                Some((subject, attribute, value, from, to)) => {
                    non_empty(subject, "event subject")?;
                    non_empty(attribute, "event attribute")?;
                    non_empty(value, "event value")?;
                    let from_date = parse_day(from, "event valid_from")?;
                    let to_date = to
                        .map(|value| parse_day(value, "event valid_to"))
                        .transpose()?;
                    if to_date.is_some_and(|end| end <= from_date) {
                        return Err(format!("event {}: valid_to must be after valid_from", event.event_id()));
                    }
                    (Some(from_date), to_date)
                }
                None => (None, None),
            };
            let id = event.event_id().to_owned();
            if event_index.insert(id.clone(), index).is_some() {
                return Err(format!("duplicate event_id {id:?}"));
            }
            events.push(EventRecord {
                event,
                recorded_at,
                valid_from,
                valid_to,
            });
        }

        let mut retracted_targets = BTreeSet::new();
        for record in &events {
            let Some(target_id) = record.event.retraction_target() else {
                continue;
            };
            let target_index = event_index
                .get(target_id)
                .copied()
                .ok_or_else(|| format!("retraction {}: target is missing", record.event.event_id()))?;
            let target = &events[target_index];
            if target.event.retraction_target().is_some() {
                return Err(format!(
                    "retraction {}: target must be an assertion",
                    record.event.event_id()
                ));
            }
            if record.recorded_at < target.recorded_at {
                return Err(format!(
                    "retraction {}: cannot precede target recorded_at",
                    record.event.event_id()
                ));
            }
            if !retracted_targets.insert(target_id.to_owned()) {
                return Err(format!("multiple retractions target {target_id:?}"));
            }
        }

        let canonical_policies: Vec<(u64, String, BTreeMap<String, SourcePolicy>)> = policies
            .iter()
            .map(|policy| {
                (
                    policy.version,
                    policy.recorded_at.to_rfc3339(),
                    policy.sources.clone(),
                )
            })
            .collect();
        let policy_bytes = serde_json::to_vec(&canonical_policies)
            .map_err(|error| format!("cannot canonicalize policy: {error}"))?;
        let policy_sha256 = format!("{:x}", Sha256::digest(policy_bytes));

        let mut ledger = Self {
            policies,
            events,
            event_index,
            admission: Vec::new(),
            policy_sha256,
        };
        let admission: Vec<Admission> = (0..ledger.events.len())
            .map(|index| ledger.compute_admission(index))
            .collect();
        ledger.admission = admission;
        Ok(ledger)
    }

    fn policy_at(&self, cutoff: DateTime<Utc>) -> Option<&PolicyVersion> {
        self.policies
            .iter()
            .filter(|policy| policy.recorded_at <= cutoff)
            .last()
    }

    fn compute_admission(&self, event_index: usize) -> Admission {
        let record = &self.events[event_index];
        let Some(policy) = self.policy_at(record.recorded_at) else {
            return Admission {
                admitted: false,
                record_policy_version: None,
                reason: Some("no authority policy existed when event was recorded".into()),
            };
        };
        let Some(source) = policy.sources.get(record.event.source()) else {
            return Admission {
                admitted: false,
                record_policy_version: Some(policy.version),
                reason: Some("source was not registered when event was recorded".into()),
            };
        };
        match &record.event {
            LedgerEvent::Assert { attribute, .. } => {
                if !source.attributes.contains(attribute) {
                    return Admission {
                        admitted: false,
                        record_policy_version: Some(policy.version),
                        reason: Some("source was outside attribute scope when recorded".into()),
                    };
                }
            }
            LedgerEvent::Retract { target_event_id, .. } => {
                if !source.can_retract {
                    return Admission {
                        admitted: false,
                        record_policy_version: Some(policy.version),
                        reason: Some("source lacked retraction authority when recorded".into()),
                    };
                }
                let Some(target_index) = self.event_index.get(target_event_id) else {
                    return Admission {
                        admitted: false,
                        record_policy_version: Some(policy.version),
                        reason: Some("retraction target is missing".into()),
                    };
                };
                let target = &self.events[*target_index];
                let target_attribute = match &target.event {
                    LedgerEvent::Assert { attribute, .. } => attribute,
                    LedgerEvent::Retract { .. } => {
                        return Admission {
                            admitted: false,
                            record_policy_version: Some(policy.version),
                            reason: Some("retraction target is not an assertion".into()),
                        };
                    }
                };
                if !source.attributes.contains(target_attribute) {
                    return Admission {
                        admitted: false,
                        record_policy_version: Some(policy.version),
                        reason: Some("source lacked authority over target scope when recorded".into()),
                    };
                }
            }
        }
        Admission {
            admitted: true,
            record_policy_version: Some(policy.version),
            reason: None,
        }
    }

    pub fn policy_sha256(&self) -> &str {
        &self.policy_sha256
    }

    pub fn resolve(&self, query: &LedgerQuery) -> Result<LedgerResolution, String> {
        non_empty(&query.query_id, "query_id")?;
        non_empty(&query.subject, "query subject")?;
        non_empty(&query.attribute, "query attribute")?;
        let valid_at = parse_day(&query.valid_at, "query valid_at")?;
        let known_at = parse_time(&query.known_at, "query known_at")?;
        let policy = self.policy_at(known_at);
        let mut excluded_events = Vec::new();
        let mut applied_retractions = Vec::new();
        let mut retracted_ids = BTreeSet::new();

        let Some(policy) = policy else {
            return Ok(LedgerResolution {
                query_id: query.query_id.clone(),
                state: ResolutionState::Unknown,
                value: None,
                conflicting_values: Vec::new(),
                policy_version: None,
                eligible_event_ids: Vec::new(),
                selected_event_ids: Vec::new(),
                selected_evidence: Vec::new(),
                applied_retractions: Vec::new(),
                excluded_events: Vec::new(),
                selected_valid_from: None,
                selected_authority_rank: None,
                policy_sha256: self.policy_sha256.clone(),
                resolution_rule: "no authority policy was known by cutoff".into(),
            });
        };

        for (index, record) in self.events.iter().enumerate() {
            let LedgerEvent::Retract { target_event_id, .. } = &record.event else {
                continue;
            };
            if record.recorded_at > known_at {
                continue;
            }
            let target_index = self.event_index[target_event_id];
            let target = &self.events[target_index];
            let LedgerEvent::Assert {
                subject, attribute, ..
            } = &target.event
            else {
                continue;
            };
            if subject != &query.subject || attribute != &query.attribute {
                continue;
            }
            if !self.admission[index].admitted {
                excluded_events.push(ExcludedEvent {
                    event_id: record.event.event_id().to_owned(),
                    reason: self.admission[index]
                        .reason
                        .clone()
                        .unwrap_or_else(|| "retraction was not admitted".into()),
                });
                continue;
            }
            if !self.admission[target_index].admitted {
                excluded_events.push(ExcludedEvent {
                    event_id: record.event.event_id().to_owned(),
                    reason: "retraction target was not admitted as evidence".into(),
                });
                continue;
            }
            retracted_ids.insert(target_event_id.clone());
            applied_retractions.push(record.event.event_id().to_owned());
        }

        let mut eligible_indices = Vec::new();
        for (index, record) in self.events.iter().enumerate() {
            let LedgerEvent::Assert {
                subject,
                attribute,
                valid_from,
                valid_to,
                ..
            } = &record.event
            else {
                continue;
            };
            if record.recorded_at > known_at
                || subject != &query.subject
                || attribute != &query.attribute
            {
                continue;
            }
            if !self.admission[index].admitted {
                excluded_events.push(ExcludedEvent {
                    event_id: record.event.event_id().to_owned(),
                    reason: self.admission[index]
                        .reason
                        .clone()
                        .unwrap_or_else(|| "assertion was not admitted".into()),
                });
                continue;
            }
            let Some(current_source) = policy.sources.get(record.event.source()) else {
                excluded_events.push(ExcludedEvent {
                    event_id: record.event.event_id().to_owned(),
                    reason: "source absent from policy at query cutoff".into(),
                });
                continue;
            };
            if !current_source.attributes.contains(attribute) {
                excluded_events.push(ExcludedEvent {
                    event_id: record.event.event_id().to_owned(),
                    reason: "source outside current policy scope".into(),
                });
                continue;
            }
            if retracted_ids.contains(record.event.event_id()) {
                continue;
            }
            let from = record.valid_from.expect("validated assertion valid_from");
            if from > valid_at {
                continue;
            }
            if record.valid_to.is_some_and(|end| valid_at >= end) {
                continue;
            }
            let _ = (valid_from, valid_to);
            eligible_indices.push(index);
        }

        eligible_indices.sort_by_key(|index| self.events[*index].event.event_id().to_owned());
        let mut eligible_event_ids: Vec<String> = eligible_indices
            .iter()
            .map(|index| self.events[*index].event.event_id().to_owned())
            .collect();
        eligible_event_ids.sort();
        applied_retractions.sort();
        excluded_events.sort_by(|left, right| {
            (&left.event_id, &left.reason).cmp(&(&right.event_id, &right.reason))
        });

        if eligible_indices.is_empty() {
            return Ok(LedgerResolution {
                query_id: query.query_id.clone(),
                state: ResolutionState::Unknown,
                value: None,
                conflicting_values: Vec::new(),
                policy_version: Some(policy.version),
                eligible_event_ids,
                selected_event_ids: Vec::new(),
                selected_evidence: Vec::new(),
                applied_retractions,
                excluded_events,
                selected_valid_from: None,
                selected_authority_rank: None,
                policy_sha256: self.policy_sha256.clone(),
                resolution_rule: "no admitted evidence matched subject, attribute, valid time, and knowledge cutoff".into(),
            });
        }

        let latest_start = eligible_indices
            .iter()
            .filter_map(|index| self.events[*index].valid_from)
            .max()
            .expect("non-empty eligible assertion set");
        let rank_at_start = eligible_indices
            .iter()
            .filter(|index| self.events[**index].valid_from == Some(latest_start))
            .filter_map(|index| policy.sources.get(self.events[*index].event.source()))
            .map(|source| source.authority_rank)
            .max()
            .expect("eligible source must exist under current policy");
        let selected_indices: Vec<usize> = eligible_indices
            .iter()
            .copied()
            .filter(|index| self.events[*index].valid_from == Some(latest_start))
            .filter(|index| {
                policy
                    .sources
                    .get(self.events[*index].event.source())
                    .is_some_and(|source| source.authority_rank == rank_at_start)
            })
            .collect();

        let mut values = BTreeSet::new();
        let mut selected_evidence = Vec::new();
        for index in &selected_indices {
            let record = &self.events[*index];
            let LedgerEvent::Assert {
                event_id,
                source,
                value,
                valid_from,
                valid_to,
                recorded_at,
                ..
            } = &record.event
            else {
                continue;
            };
            values.insert(value.clone());
            selected_evidence.push(SelectedEvidence {
                event_id: event_id.clone(),
                source: source.clone(),
                value: value.clone(),
                valid_from: valid_from.clone(),
                valid_to: valid_to.clone(),
                recorded_at: recorded_at.clone(),
                authority_rank: rank_at_start,
                record_policy_version: self.admission[*index]
                    .record_policy_version
                    .unwrap_or_default(),
            });
        }
        selected_evidence.sort_by(|left, right| left.event_id.cmp(&right.event_id));
        let selected_event_ids = selected_evidence
            .iter()
            .map(|evidence| evidence.event_id.clone())
            .collect();
        let conflicting_values: Vec<String> = values.into_iter().collect();
        let (state, value, resolution_rule) = if conflicting_values.len() > 1 {
            (
                ResolutionState::Conflict,
                None,
                "latest valid start, then policy-pinned authority; equal-rank disagreement abstains",
            )
        } else {
            (
                ResolutionState::Resolved,
                conflicting_values.first().cloned(),
                "latest valid start, then policy-pinned authority",
            )
        };

        Ok(LedgerResolution {
            query_id: query.query_id.clone(),
            state,
            value,
            conflicting_values,
            policy_version: Some(policy.version),
            eligible_event_ids,
            selected_event_ids,
            selected_evidence,
            applied_retractions,
            excluded_events,
            selected_valid_from: Some(latest_start.to_string()),
            selected_authority_rank: Some(rank_at_start),
            policy_sha256: self.policy_sha256.clone(),
            resolution_rule: resolution_rule.into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(version: u64, recorded_at: &str, incident_rank: u32) -> AuthorityPolicy {
        AuthorityPolicy {
            policy_version: version,
            recorded_at: recorded_at.into(),
            sources: vec![
                SourcePolicy {
                    source: "operations".into(),
                    authority_rank: 10,
                    attributes: ["endpoint", "notes"].into_iter().map(str::to_owned).collect(),
                    can_retract: true,
                },
                SourcePolicy {
                    source: "incident-review".into(),
                    authority_rank: incident_rank,
                    attributes: ["endpoint"].into_iter().map(str::to_owned).collect(),
                    can_retract: false,
                },
                SourcePolicy {
                    source: "user-notes".into(),
                    authority_rank: 1,
                    attributes: ["notes"].into_iter().map(str::to_owned).collect(),
                    can_retract: false,
                },
            ],
        }
    }

    fn assertion(
        id: &str,
        source: &str,
        value: &str,
        valid_from: &str,
        recorded_at: &str,
        attribute: &str,
    ) -> LedgerEvent {
        LedgerEvent::Assert {
            event_id: id.into(),
            subject: "db-prod".into(),
            attribute: attribute.into(),
            value: value.into(),
            valid_from: valid_from.into(),
            valid_to: None,
            recorded_at: recorded_at.into(),
            source: source.into(),
        }
    }

    fn retraction(id: &str, target: &str, recorded_at: &str, source: &str) -> LedgerEvent {
        LedgerEvent::Retract {
            event_id: id.into(),
            target_event_id: target.into(),
            recorded_at: recorded_at.into(),
            source: source.into(),
        }
    }

    fn sample_ledger() -> PolicyPinnedLedger {
        PolicyPinnedLedger::new(
            vec![
                policy(1, "2026-01-01T00:00:00Z", 8),
                policy(2, "2026-03-15T00:00:00Z", 12),
            ],
            vec![
                assertion("E1", "operations", "A", "2026-01-01", "2026-01-01T09:00:00Z", "endpoint"),
                assertion("E2", "operations", "C", "2026-01-15", "2026-03-01T08:00:00Z", "endpoint"),
                assertion("E3", "incident-review", "D", "2026-01-15", "2026-03-02T08:00:00Z", "endpoint"),
                retraction("R1", "E3", "2026-04-01T00:00:00Z", "operations"),
            ],
        )
        .expect("valid sample ledger")
    }

    fn query(id: &str, known_at: &str) -> LedgerQuery {
        LedgerQuery {
            query_id: id.into(),
            subject: "db-prod".into(),
            attribute: "endpoint".into(),
            valid_at: "2026-01-20".into(),
            known_at: known_at.into(),
        }
    }

    #[test]
    fn event_payload_cannot_claim_authority_rank() {
        let input = r#"{"event_id":"E","kind":"assert","subject":"db-prod","attribute":"endpoint","value":"A","valid_from":"2026-01-01","valid_to":null,"recorded_at":"2026-01-01T00:00:00Z","source":"operations","authority_rank":9999}"#;
        assert!(parse_event_json(input).unwrap_err().contains("authority_rank"));
    }

    #[test]
    fn strict_event_parser_accepts_assertion_and_retraction_shapes() {
        let assertion = r#"{"event_id":"E","kind":"assert","subject":"x","attribute":"a","value":"v","valid_from":"2026-01-01","valid_to":null,"recorded_at":"2026-01-01T00:00:00Z","source":"operations"}"#;
        let retract = r#"{"event_id":"R","kind":"retract","target_event_id":"E","recorded_at":"2026-01-02T00:00:00Z","source":"operations"}"#;
        assert!(matches!(parse_event_json(assertion), Ok(LedgerEvent::Assert { .. })));
        assert!(matches!(parse_event_json(retract), Ok(LedgerEvent::Retract { .. })));
    }

    #[test]
    fn policy_rank_is_selected_at_query_cutoff() {
        let ledger = sample_ledger();
        let before = ledger.resolve(&query("before", "2026-03-10T00:00:00Z")).unwrap();
        let after = ledger.resolve(&query("after", "2026-03-20T00:00:00Z")).unwrap();
        assert_eq!(before.value.as_deref(), Some("C"));
        assert_eq!(before.policy_version, Some(1));
        assert_eq!(after.value.as_deref(), Some("D"));
        assert_eq!(after.policy_version, Some(2));
    }

    #[test]
    fn policy_change_is_invisible_before_its_record_time() {
        let ledger = sample_ledger();
        let result = ledger.resolve(&query("before-policy", "2026-03-14T23:59:59Z")).unwrap();
        assert_eq!(result.policy_version, Some(1));
        assert_eq!(result.value.as_deref(), Some("C"));
    }

    #[test]
    fn policy_change_is_visible_at_exact_record_time() {
        let ledger = sample_ledger();
        let result = ledger.resolve(&query("at-policy", "2026-03-15T00:00:00Z")).unwrap();
        assert_eq!(result.policy_version, Some(2));
        assert_eq!(result.value.as_deref(), Some("D"));
    }

    #[test]
    fn retraction_does_not_leak_before_recorded_time() {
        let ledger = sample_ledger();
        let result = ledger.resolve(&query("before-retract", "2026-03-31T23:59:59Z")).unwrap();
        assert_eq!(result.value.as_deref(), Some("D"));
        assert!(result.applied_retractions.is_empty());
    }

    #[test]
    fn authorized_retraction_applies_at_exact_record_time() {
        let ledger = sample_ledger();
        let result = ledger.resolve(&query("at-retract", "2026-04-01T00:00:00Z")).unwrap();
        assert_eq!(result.value.as_deref(), Some("C"));
        assert_eq!(result.applied_retractions, vec!["R1".to_owned()]);
    }

    #[test]
    fn unauthorized_retraction_does_not_remove_evidence() {
        let ledger = PolicyPinnedLedger::new(
            vec![policy(1, "2026-01-01T00:00:00Z", 8)],
            vec![
                assertion("E1", "operations", "A", "2026-01-01", "2026-01-01T09:00:00Z", "endpoint"),
                retraction("R1", "E1", "2026-01-02T00:00:00Z", "user-notes"),
            ],
        )
        .unwrap();
        let result = ledger.resolve(&query("unauthorized", "2026-01-03T00:00:00Z")).unwrap();
        assert_eq!(result.value.as_deref(), Some("A"));
        assert!(result.applied_retractions.is_empty());
        assert!(result.excluded_events.iter().any(|event| event.event_id == "R1"));
    }

    #[test]
    fn unknown_source_is_excluded() {
        let ledger = PolicyPinnedLedger::new(
            vec![policy(1, "2026-01-01T00:00:00Z", 8)],
            vec![
                assertion("E1", "operations", "A", "2026-01-01", "2026-01-01T09:00:00Z", "endpoint"),
                assertion("E2", "mystery", "M", "2026-01-10", "2026-01-10T09:00:00Z", "endpoint"),
            ],
        )
        .unwrap();
        let result = ledger.resolve(&query("unknown-source", "2026-01-20T00:00:00Z")).unwrap();
        assert_eq!(result.value.as_deref(), Some("A"));
        assert!(!result.eligible_event_ids.contains(&"E2".into()));
    }

    #[test]
    fn source_scope_is_enforced() {
        let ledger = PolicyPinnedLedger::new(
            vec![policy(1, "2026-01-01T00:00:00Z", 8)],
            vec![
                assertion("E1", "operations", "A", "2026-01-01", "2026-01-01T09:00:00Z", "endpoint"),
                assertion("E2", "user-notes", "spoof", "2026-01-10", "2026-01-10T09:00:00Z", "endpoint"),
            ],
        )
        .unwrap();
        let result = ledger.resolve(&query("scope", "2026-01-20T00:00:00Z")).unwrap();
        assert_eq!(result.value.as_deref(), Some("A"));
        assert!(!result.eligible_event_ids.contains(&"E2".into()));
    }

    #[test]
    fn equal_authority_disagreement_returns_conflict() {
        let ledger = PolicyPinnedLedger::new(
            vec![policy(1, "2026-01-01T00:00:00Z", 10)],
            vec![
                assertion("E1", "operations", "A", "2026-01-15", "2026-01-20T00:00:00Z", "endpoint"),
                assertion("E2", "incident-review", "B", "2026-01-15", "2026-01-21T00:00:00Z", "endpoint"),
            ],
        )
        .unwrap();
        let result = ledger.resolve(&query("conflict", "2026-01-22T00:00:00Z")).unwrap();
        assert_eq!(result.state, ResolutionState::Conflict);
        assert!(result.value.is_none());
        assert_eq!(result.conflicting_values, vec!["A".to_owned(), "B".to_owned()]);
    }

    #[test]
    fn duplicate_retraction_target_is_rejected() {
        let result = PolicyPinnedLedger::new(
            vec![policy(1, "2026-01-01T00:00:00Z", 8)],
            vec![
                assertion("E1", "operations", "A", "2026-01-01", "2026-01-01T09:00:00Z", "endpoint"),
                retraction("R1", "E1", "2026-01-02T00:00:00Z", "operations"),
                retraction("R2", "E1", "2026-01-03T00:00:00Z", "operations"),
            ],
        );
        assert!(result.unwrap_err().contains("multiple retractions"));
    }

    #[test]
    fn dangling_retraction_target_is_rejected() {
        let result = PolicyPinnedLedger::new(
            vec![policy(1, "2026-01-01T00:00:00Z", 8)],
            vec![retraction("R1", "missing", "2026-01-02T00:00:00Z", "operations")],
        );
        assert!(result.unwrap_err().contains("target is missing"));
    }

    #[test]
    fn retraction_before_target_is_rejected() {
        let result = PolicyPinnedLedger::new(
            vec![policy(1, "2026-01-01T00:00:00Z", 8)],
            vec![
                assertion("E1", "operations", "A", "2026-01-01", "2026-01-02T00:00:00Z", "endpoint"),
                retraction("R1", "E1", "2026-01-01T00:00:00Z", "operations"),
            ],
        );
        assert!(result.unwrap_err().contains("cannot precede"));
    }

    #[test]
    fn event_input_order_does_not_change_resolution() {
        let policies = vec![
            policy(1, "2026-01-01T00:00:00Z", 8),
            policy(2, "2026-03-15T00:00:00Z", 12),
        ];
        let events = vec![
            assertion("E1", "operations", "A", "2026-01-01", "2026-01-01T09:00:00Z", "endpoint"),
            assertion("E2", "operations", "C", "2026-01-15", "2026-03-01T08:00:00Z", "endpoint"),
            assertion("E3", "incident-review", "D", "2026-01-15", "2026-03-02T08:00:00Z", "endpoint"),
        ];
        let first = PolicyPinnedLedger::new(policies.clone(), events.clone()).unwrap();
        let second = PolicyPinnedLedger::new(policies, events.into_iter().rev().collect()).unwrap();
        let q = query("order", "2026-03-20T00:00:00Z");
        let a = first.resolve(&q).unwrap();
        let b = second.resolve(&q).unwrap();
        assert_eq!(a.state, b.state);
        assert_eq!(a.value, b.value);
        assert_eq!(a.selected_event_ids, b.selected_event_ids);
    }

    #[test]
    fn naive_timestamps_are_rejected() {
        let result = PolicyPinnedLedger::new(
            vec![policy(1, "2026-01-01T00:00:00", 8)],
            vec![assertion("E1", "operations", "A", "2026-01-01", "2026-01-01T09:00:00Z", "endpoint")],
        );
        assert!(result.unwrap_err().contains("RFC3339 with UTC offset"));
    }

    #[test]
    fn policy_digest_is_deterministic() {
        let left = sample_ledger();
        let right = sample_ledger();
        assert_eq!(left.policy_sha256(), right.policy_sha256());
        assert_eq!(left.policy_sha256().len(), 64);
    }
}
