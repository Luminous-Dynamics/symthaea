// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Append-only bitemporal source-event reference ledger.
//!
//! Queries distinguish valid time (when an assertion applied in the modeled
//! world) from transaction time (when the ledger had recorded it). A query at
//! (valid_at, known_at) never sees a correction recorded after known_at.
//!
//! This is a reference model, not a persistence adapter, source authenticity
//! verifier, authorization engine, or natural-language entailment system.
//! Scope is an exact partition key, not proof that a caller is authorized.
//! Conflicting active values are surfaced; arrival order never selects a winner.

use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

fn non_empty(value: &str, field: &str) -> Result<(), String> {
    if value.trim().is_empty() || value.trim() != value {
        return Err(format!("{field} must be non-empty and trimmed"));
    }
    Ok(())
}

fn valid_sha256(value: &str, field: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{field} must be 64 lowercase hexadecimal characters"));
    }
    Ok(())
}

/// Immutable source assertion with world-time interval and provenance envelope.
/// The payload digest binds external evidence but is not verified by this module.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceAssertion {
    pub subject: String,
    pub attribute: String,
    pub value: String,
    pub valid_from: NaiveDate,
    /// Exclusive end of the half-open interval [valid_from, valid_to).
    pub valid_to: Option<NaiveDate>,
    pub source_id: String,
    /// Exact scope partition. This is not an authorization decision.
    pub scope: String,
    pub payload_sha256: String,
}

impl SourceAssertion {
    fn validate(&self) -> Result<(), String> {
        for (field, value) in [
            ("subject", self.subject.as_str()),
            ("attribute", self.attribute.as_str()),
            ("value", self.value.as_str()),
            ("source_id", self.source_id.as_str()),
            ("scope", self.scope.as_str()),
        ] {
            non_empty(value, field)?;
        }
        if self.valid_to.is_some_and(|end| end <= self.valid_from) {
            return Err("valid_to must be strictly after valid_from".into());
        }
        valid_sha256(&self.payload_sha256, "payload_sha256")?;
        Ok(())
    }

    fn has_same_key(&self, other: &Self) -> bool {
        self.subject == other.subject
            && self.attribute == other.attribute
            && self.scope == other.scope
    }

    fn is_valid_at(&self, valid_at: NaiveDate) -> bool {
        self.valid_from <= valid_at && self.valid_to.is_none_or(|end| valid_at < end)
    }
}

/// Corrections are new assertions that explicitly supersede one earlier
/// revision of the same subject/attribute/scope key. No operation deletes data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SourceEventOperation {
    Assert { assertion: SourceAssertion },
    Supersede {
        target_revision_id: String,
        assertion: SourceAssertion,
    },
    Retract {
        target_revision_id: String,
        reason: String,
    },
}

impl SourceEventOperation {
    fn assertion(&self) -> Option<&SourceAssertion> {
        match self {
            Self::Assert { assertion } | Self::Supersede { assertion, .. } => Some(assertion),
            Self::Retract { .. } => None,
        }
    }

    fn target_revision_id(&self) -> Option<&str> {
        match self {
            Self::Assert { .. } => None,
            Self::Supersede { target_revision_id, .. }
            | Self::Retract { target_revision_id, .. } => Some(target_revision_id),
        }
    }
}

/// One immutable ledger event. recorded_at is transaction/knowledge time,
/// not the date at which the proposition became true in the world.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceEventRevision {
    pub revision_id: String,
    pub recorded_at: DateTime<Utc>,
    pub operation: SourceEventOperation,
}

/// A source revision visible at a particular transaction-time coordinate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceAssertionRecord {
    pub revision_id: String,
    pub recorded_at: DateTime<Utc>,
    pub assertion: SourceAssertion,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum BitemporalEvidenceStatus {
    Unknown,
    OutsideValidity,
    Supported {
        value: String,
        supporting_revision_ids: Vec<String>,
    },
    Contested {
        distinct_values: BTreeSet<String>,
        supporting_revision_ids: Vec<String>,
    },
}

/// known_assertions contains active assertions known at known_at, including
/// those outside the requested valid interval. eligible_assertions is the
/// subset whose valid-time interval contains valid_at.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BitemporalQueryResult {
    pub subject: String,
    pub attribute: String,
    pub scope: String,
    pub valid_at: NaiveDate,
    pub known_at: DateTime<Utc>,
    pub known_assertions: Vec<SourceAssertionRecord>,
    pub eligible_assertions: Vec<SourceAssertionRecord>,
    pub status: BitemporalEvidenceStatus,
}

/// The vector order is transaction commit order and is part of the state digest.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BitemporalSourceLedger {
    events: Vec<SourceEventRevision>,
}

impl BitemporalSourceLedger {
    /// Append one event as a one-event atomic transaction.
    pub fn append(&mut self, revision: SourceEventRevision) -> Result<(), String> {
        self.append_batch([revision])
    }

    /// Append a batch atomically at one transaction timestamp.
    ///
    /// All revisions in a batch share recorded_at, which must be strictly later
    /// than the previous committed batch. This prevents separately committed
    /// states from becoming indistinguishable in timestamp-only as-known
    /// queries. Any failure leaves the original ledger unchanged.
    pub fn append_batch(
        &mut self,
        revisions: impl IntoIterator<Item = SourceEventRevision>,
    ) -> Result<(), String> {
        let revisions: Vec<_> = revisions.into_iter().collect();
        if revisions.is_empty() {
            return Err("an append batch must contain at least one revision".into());
        }
        let transaction_time = revisions[0].recorded_at;
        if revisions.iter().any(|revision| revision.recorded_at != transaction_time) {
            return Err("all revisions in one atomic batch must share recorded_at".into());
        }
        if self.events.last().is_some_and(|last| transaction_time <= last.recorded_at) {
            return Err("recorded_at must be later than the previous committed batch".into());
        }

        let mut candidate = self.clone();
        for revision in revisions {
            candidate.append_unchecked(revision)?;
        }
        candidate.verify_consistency()?;
        *self = candidate;
        Ok(())
    }

    fn append_unchecked(&mut self, revision: SourceEventRevision) -> Result<(), String> {
        non_empty(&revision.revision_id, "revision_id")?;
        if self.events.iter().any(|event| event.revision_id == revision.revision_id) {
            return Err(format!("duplicate source revision ID {:?}", revision.revision_id));
        }
        if let Some(assertion) = revision.operation.assertion() {
            assertion.validate()?;
        }

        if let Some(target_id) = revision.operation.target_revision_id() {
            non_empty(target_id, "target_revision_id")?;
            let target = self.events.iter()
                .find(|event| event.revision_id == target_id)
                .ok_or_else(|| format!("unknown target revision {target_id:?}"))?;
            if target.operation.assertion().is_none() {
                return Err(format!("target revision {target_id:?} is not an assertion"));
            }
            if self.events.iter().any(|event| {
                event.operation.target_revision_id().is_some_and(|existing| existing == target_id)
            }) {
                return Err(format!("target revision {target_id:?} has already been superseded or retracted"));
            }

            match &revision.operation {
                SourceEventOperation::Supersede { assertion, .. } => {
                    let target_assertion = target.operation.assertion()
                        .expect("target assertion was checked above");
                    if !assertion.has_same_key(target_assertion) {
                        return Err("a superseding assertion must preserve subject, attribute, and scope".into());
                    }
                }
                SourceEventOperation::Retract { reason, .. } => non_empty(reason, "retraction reason")?,
                SourceEventOperation::Assert { .. } => unreachable!(),
            }
        }

        self.events.push(revision);
        Ok(())
    }

    /// Query one exact subject/attribute/scope at (valid_at, known_at).
    /// Independent active values remain contested; no last-write-wins rule exists.
    pub fn query(
        &self,
        subject: &str,
        attribute: &str,
        scope: &str,
        valid_at: NaiveDate,
        known_at: DateTime<Utc>,
    ) -> Result<BitemporalQueryResult, String> {
        for (field, value) in [("subject", subject), ("attribute", attribute), ("scope", scope)] {
            non_empty(value, field)?;
        }

        let visible_events: Vec<_> = self.events.iter()
            .filter(|event| event.recorded_at <= known_at)
            .collect();
        let invalidated_revision_ids: BTreeSet<String> = visible_events.iter()
            .filter_map(|event| event.operation.target_revision_id())
            .map(str::to_owned)
            .collect();

        let mut known_assertions: Vec<_> = visible_events.iter().filter_map(|event| {
            let assertion = event.operation.assertion()?;
            if invalidated_revision_ids.contains(&event.revision_id)
                || assertion.subject != subject
                || assertion.attribute != attribute
                || assertion.scope != scope
            {
                return None;
            }
            Some(SourceAssertionRecord {
                revision_id: event.revision_id.clone(),
                recorded_at: event.recorded_at,
                assertion: assertion.clone(),
            })
        }).collect();
        known_assertions.sort_by(|left, right| left.revision_id.cmp(&right.revision_id));

        let eligible_assertions: Vec<_> = known_assertions.iter()
            .filter(|record| record.assertion.is_valid_at(valid_at))
            .cloned()
            .collect();
        let distinct_values: BTreeSet<_> = eligible_assertions.iter()
            .map(|record| record.assertion.value.clone())
            .collect();
        let supporting_revision_ids: Vec<_> = eligible_assertions.iter()
            .map(|record| record.revision_id.clone())
            .collect();

        let status = match distinct_values.len() {
            0 if known_assertions.is_empty() => BitemporalEvidenceStatus::Unknown,
            0 => BitemporalEvidenceStatus::OutsideValidity,
            1 => BitemporalEvidenceStatus::Supported {
                value: distinct_values.iter().next().expect("one value").clone(),
                supporting_revision_ids,
            },
            _ => BitemporalEvidenceStatus::Contested {
                distinct_values,
                supporting_revision_ids,
            },
        };

        Ok(BitemporalQueryResult {
            subject: subject.into(),
            attribute: attribute.into(),
            scope: scope.into(),
            valid_at,
            known_at,
            known_assertions,
            eligible_assertions,
            status,
        })
    }

    /// SHA-256 over the complete append-only history. Transaction order is
    /// intentionally included because it is part of the ledger's semantics.
    pub fn state_sha256(&self) -> Result<String, String> {
        let encoded = serde_json::to_vec(&self.events)
            .map_err(|error| format!("failed to serialize bitemporal ledger: {error}"))?;
        Ok(format!("{:x}", Sha256::digest(encoded)))
    }

    pub fn events(&self) -> &[SourceEventRevision] {
        &self.events
    }

    /// Recheck revision uniqueness, transaction ordering, target ancestry,
    /// one-time mutation, and assertion invariants without mutating the ledger.
    pub fn verify_consistency(&self) -> Result<(), String> {
        let mut seen: BTreeMap<String, &SourceEventRevision> = BTreeMap::new();
        let mut mutated_targets: BTreeSet<String> = BTreeSet::new();
        let mut prior_transaction_time: Option<DateTime<Utc>> = None;

        for event in &self.events {
            non_empty(&event.revision_id, "revision_id")?;
            if seen.contains_key(&event.revision_id) {
                return Err(format!("duplicate source revision ID {:?}", event.revision_id));
            }
            if prior_transaction_time.is_some_and(|prior| event.recorded_at < prior) {
                return Err("ledger transaction timestamps must be nondecreasing".into());
            }
            prior_transaction_time = Some(event.recorded_at);

            if let Some(assertion) = event.operation.assertion() {
                assertion.validate()?;
            }
            if let Some(target_id) = event.operation.target_revision_id() {
                let target = seen.get(target_id)
                    .ok_or_else(|| format!("target revision {target_id:?} must precede its mutation"))?;
                if target.operation.assertion().is_none() {
                    return Err(format!("target revision {target_id:?} is not an assertion"));
                }
                if !mutated_targets.insert(target_id.to_owned()) {
                    return Err(format!("target revision {target_id:?} is mutated more than once"));
                }
                match &event.operation {
                    SourceEventOperation::Supersede { assertion, .. } => {
                        let target_assertion = target.operation.assertion()
                            .expect("target assertion was checked above");
                        if !assertion.has_same_key(target_assertion) {
                            return Err("a superseding assertion must preserve subject, attribute, and scope".into());
                        }
                    }
                    SourceEventOperation::Retract { reason, .. } => non_empty(reason, "retraction reason")?,
                    SourceEventOperation::Assert { .. } => unreachable!(),
                }
            }
            seen.insert(event.revision_id.clone(), event);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn date(value: &str) -> NaiveDate {
        NaiveDate::parse_from_str(value, "%Y-%m-%d").unwrap()
    }

    fn instant(value: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(value).unwrap().with_timezone(&Utc)
    }

    fn assertion(value: &str, valid_from: &str, valid_to: Option<&str>) -> SourceAssertion {
        SourceAssertion {
            subject: "account-17".into(),
            attribute: "plan".into(),
            value: value.into(),
            valid_from: date(valid_from),
            valid_to: valid_to.map(date),
            source_id: "source-A".into(),
            scope: "tenant-1".into(),
            payload_sha256: "a".repeat(64),
        }
    }

    fn event(revision_id: &str, recorded_at: &str, operation: SourceEventOperation) -> SourceEventRevision {
        SourceEventRevision {
            revision_id: revision_id.into(),
            recorded_at: instant(recorded_at),
            operation,
        }
    }

    fn assert_event(revision_id: &str, recorded_at: &str, value: &str) -> SourceEventRevision {
        event(revision_id, recorded_at, SourceEventOperation::Assert {
            assertion: assertion(value, "2026-01-01", None),
        })
    }

    fn query(ledger: &BitemporalSourceLedger, valid_at: &str, known_at: &str) -> BitemporalQueryResult {
        ledger.query("account-17", "plan", "tenant-1", date(valid_at), instant(known_at)).unwrap()
    }

    #[test]
    fn late_correction_does_not_leak_into_earlier_as_known_query() {
        let mut ledger = BitemporalSourceLedger::default();
        ledger.append(assert_event("r1", "2026-01-02T10:00:00Z", "basic")).unwrap();
        ledger.append(event("r2", "2026-03-01T10:00:00Z", SourceEventOperation::Supersede {
            target_revision_id: "r1".into(),
            assertion: assertion("premium", "2026-01-01", None),
        })).unwrap();

        assert_eq!(query(&ledger, "2026-01-15", "2026-02-01T00:00:00Z").status,
            BitemporalEvidenceStatus::Supported { value: "basic".into(), supporting_revision_ids: vec!["r1".into()] });
        assert_eq!(query(&ledger, "2026-01-15", "2026-03-02T00:00:00Z").status,
            BitemporalEvidenceStatus::Supported { value: "premium".into(), supporting_revision_ids: vec!["r2".into()] });
    }

    #[test]
    fn delayed_retraction_preserves_the_prior_knowledge_view() {
        let mut ledger = BitemporalSourceLedger::default();
        ledger.append(assert_event("r1", "2026-01-02T10:00:00Z", "basic")).unwrap();
        ledger.append(event("retract-r1", "2026-03-01T10:00:00Z", SourceEventOperation::Retract {
            target_revision_id: "r1".into(), reason: "source withdrawn".into(),
        })).unwrap();

        assert!(matches!(query(&ledger, "2026-01-15", "2026-02-01T00:00:00Z").status,
            BitemporalEvidenceStatus::Supported { .. }));
        assert_eq!(query(&ledger, "2026-01-15", "2026-03-02T00:00:00Z").status,
            BitemporalEvidenceStatus::Unknown);
        assert_eq!(ledger.events().len(), 2);
    }

    #[test]
    fn valid_time_uses_inclusive_start_and_exclusive_end() {
        let mut ledger = BitemporalSourceLedger::default();
        let bounded = SourceAssertion {
            valid_from: date("2026-02-01"),
            valid_to: Some(date("2026-03-01")),
            ..assertion("basic", "2026-01-01", None)
        };
        ledger.append(event("bounded", "2026-01-10T00:00:00Z",
            SourceEventOperation::Assert { assertion: bounded })).unwrap();

        assert_eq!(query(&ledger, "2026-01-31", "2026-02-01T00:00:00Z").status,
            BitemporalEvidenceStatus::OutsideValidity);
        assert!(matches!(query(&ledger, "2026-02-01", "2026-02-01T00:00:00Z").status,
            BitemporalEvidenceStatus::Supported { .. }));
        assert_eq!(query(&ledger, "2026-03-01", "2026-03-02T00:00:00Z").status,
            BitemporalEvidenceStatus::OutsideValidity);
    }

    #[test]
    fn independent_conflicting_assertions_remain_contested() {
        let mut ledger = BitemporalSourceLedger::default();
        ledger.append(assert_event("r1", "2026-01-02T10:00:00Z", "basic")).unwrap();
        let second = SourceAssertion {
            source_id: "source-B".into(),
            payload_sha256: "b".repeat(64),
            ..assertion("enterprise", "2026-01-01", None)
        };
        ledger.append(event("r2", "2026-01-03T10:00:00Z",
            SourceEventOperation::Assert { assertion: second })).unwrap();

        assert_eq!(query(&ledger, "2026-01-15", "2026-01-04T00:00:00Z").status,
            BitemporalEvidenceStatus::Contested {
                distinct_values: BTreeSet::from(["basic".into(), "enterprise".into()]),
                supporting_revision_ids: vec!["r1".into(), "r2".into()],
            });
    }

    #[test]
    fn exact_scope_partition_prevents_cross_scope_evidence_mixing() {
        let mut ledger = BitemporalSourceLedger::default();
        ledger.append(assert_event("r1", "2026-01-02T10:00:00Z", "basic")).unwrap();
        let result = ledger.query("account-17", "plan", "tenant-2", date("2026-01-15"),
            instant("2026-01-03T00:00:00Z")).unwrap();
        assert_eq!(result.status, BitemporalEvidenceStatus::Unknown);
        assert!(result.known_assertions.is_empty());
    }

    #[test]
    fn failed_batch_is_atomic_and_does_not_mutate_the_ledger() {
        let mut ledger = BitemporalSourceLedger::default();
        let before = ledger.state_sha256().unwrap();
        let batch = [
            assert_event("r1", "2026-01-02T10:00:00Z", "basic"),
            event("bad-retraction", "2026-01-02T10:00:00Z", SourceEventOperation::Retract {
                target_revision_id: "missing".into(), reason: "invalid target".into(),
            }),
        ];
        assert!(ledger.append_batch(batch).is_err());
        assert!(ledger.events().is_empty());
        assert_eq!(ledger.state_sha256().unwrap(), before);
    }

    #[test]
    fn an_assertion_can_be_mutated_at_most_once() {
        let mut ledger = BitemporalSourceLedger::default();
        ledger.append(assert_event("r1", "2026-01-02T10:00:00Z", "basic")).unwrap();
        ledger.append(event("r2", "2026-02-01T10:00:00Z", SourceEventOperation::Retract {
            target_revision_id: "r1".into(), reason: "withdrawn".into(),
        })).unwrap();
        let before = ledger.state_sha256().unwrap();

        let duplicate = event("r3", "2026-03-01T10:00:00Z", SourceEventOperation::Retract {
            target_revision_id: "r1".into(), reason: "duplicate withdrawal".into(),
        });
        assert!(ledger.append(duplicate).is_err());
        assert_eq!(ledger.state_sha256().unwrap(), before);
    }

    #[test]
    fn replacement_must_preserve_the_target_key() {
        let mut ledger = BitemporalSourceLedger::default();
        ledger.append(assert_event("r1", "2026-01-02T10:00:00Z", "basic")).unwrap();
        let replacement = SourceAssertion {
            subject: "account-18".into(),
            ..assertion("premium", "2026-01-01", None)
        };
        let correction = event("r2", "2026-02-01T10:00:00Z", SourceEventOperation::Supersede {
            target_revision_id: "r1".into(), assertion: replacement,
        });
        let before = ledger.state_sha256().unwrap();
        assert!(ledger.append(correction).is_err());
        assert_eq!(ledger.state_sha256().unwrap(), before);
    }

    #[test]
    fn one_timestamp_batch_is_visible_atomically() {
        let mut ledger = BitemporalSourceLedger::default();
        ledger.append_batch([
            assert_event("r1", "2026-01-02T10:00:00Z", "basic"),
            event("r2", "2026-01-02T10:00:00Z", SourceEventOperation::Supersede {
                target_revision_id: "r1".into(), assertion: assertion("premium", "2026-01-01", None),
            }),
        ]).unwrap();
        assert_eq!(query(&ledger, "2026-01-15", "2026-01-02T10:00:00Z").status,
            BitemporalEvidenceStatus::Supported { value: "premium".into(), supporting_revision_ids: vec!["r2".into()] });
    }

    #[test]
    fn separately_committed_batches_require_strictly_increasing_time() {
        let mut ledger = BitemporalSourceLedger::default();
        ledger.append(assert_event("r1", "2026-01-02T10:00:00Z", "basic")).unwrap();
        let before = ledger.state_sha256().unwrap();
        assert!(ledger.append(assert_event("r2", "2026-01-02T10:00:00Z", "premium")).is_err());
        assert_eq!(ledger.state_sha256().unwrap(), before);
    }

    #[test]
    fn unknown_and_outside_validity_are_distinct() {
        let ledger = BitemporalSourceLedger::default();
        assert_eq!(query(&ledger, "2026-01-15", "2026-02-01T00:00:00Z").status,
            BitemporalEvidenceStatus::Unknown);

        let mut populated = BitemporalSourceLedger::default();
        let bounded = SourceAssertion {
            valid_from: date("2026-02-01"),
            valid_to: None,
            ..assertion("basic", "2026-01-01", None)
        };
        populated.append(event("r1", "2026-01-02T10:00:00Z",
            SourceEventOperation::Assert { assertion: bounded })).unwrap();
        assert_eq!(query(&populated, "2026-01-15", "2026-02-01T00:00:00Z").status,
            BitemporalEvidenceStatus::OutsideValidity);
    }

    #[test]
    fn digest_is_replay_deterministic_and_changes_on_append() {
        fn build() -> BitemporalSourceLedger {
            let mut ledger = BitemporalSourceLedger::default();
            ledger.append(assert_event("r1", "2026-01-02T10:00:00Z", "basic")).unwrap();
            ledger
        }
        let mut left = build();
        let right = build();
        assert_eq!(left.state_sha256().unwrap(), right.state_sha256().unwrap());
        let before = left.state_sha256().unwrap();
        left.append(event("retract-r1", "2026-02-01T10:00:00Z", SourceEventOperation::Retract {
            target_revision_id: "r1".into(), reason: "withdrawn".into(),
        })).unwrap();
        assert_ne!(left.state_sha256().unwrap(), before);
        left.verify_consistency().unwrap();
    }

    #[test]
    fn empty_batches_and_invalid_digests_are_rejected_without_mutation() {
        let mut ledger = BitemporalSourceLedger::default();
        let before = ledger.state_sha256().unwrap();
        assert!(ledger.append_batch(Vec::new()).is_err());
        let invalid = SourceAssertion {
            payload_sha256: "not-a-digest".into(),
            ..assertion("basic", "2026-01-01", None)
        };
        assert!(ledger.append(event("bad", "2026-01-02T10:00:00Z",
            SourceEventOperation::Assert { assertion: invalid })).is_err());
        assert_eq!(ledger.state_sha256().unwrap(), before);
    }
}
