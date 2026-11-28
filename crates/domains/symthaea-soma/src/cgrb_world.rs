// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic, in-memory reference world for CGRB.
//!
//! This is a benchmark fixture, not a production connector or independent
//! state oracle. Its read view is explicitly reference-model evidence and must
//! not be promoted to a separately operated oracle or deployment qualification.
//! Mutation application is synchronous under exclusive &mut access; this does
//! not establish distributed or concurrent transaction semantics.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const CGRB_WORLD_STATE_V1: &str = "symthaea.cgrb-world-state.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum CgrbWorldValueV1 {
    Text(String),
    Integer(i64),
    Boolean(bool),
    TextList(Vec<String>),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "condition", content = "expected", rename_all = "snake_case")]
pub enum CgrbWorldConditionV1 {
    Equals(CgrbWorldValueV1),
    Exists,
    Absent,
    ContainsText(String),
    GreaterThanOrEqual(i64),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CgrbWorldPredicateV1 {
    pub subject: String,
    pub attribute: String,
    pub condition: CgrbWorldConditionV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mutation", rename_all = "snake_case")]
pub enum CgrbWorldMutationV1 {
    Set {
        subject: String,
        attribute: String,
        value: CgrbWorldValueV1,
    },
    AppendText {
        subject: String,
        attribute: String,
        value: String,
    },
    Remove {
        subject: String,
        attribute: String,
    },
}

impl CgrbWorldMutationV1 {
    fn subject(&self) -> &str {
        match self {
            Self::Set { subject, .. }
            | Self::AppendText { subject, .. }
            | Self::Remove { subject, .. } => subject,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CgrbWorldIdempotencyV1 {
    Idempotent,
    KeyedIdempotent,
    NonIdempotent,
    ReversibleWithCompensation,
    Irreversible,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CgrbWorldActionV1 {
    pub operation_id: String,
    pub action_type: String,
    pub target: String,
    pub purpose: String,
    pub destination: String,
    pub idempotency: CgrbWorldIdempotencyV1,
    pub idempotency_key: Option<String>,
    pub expected_state_version: u64,
    pub preconditions: Vec<CgrbWorldPredicateV1>,
    pub mutation: CgrbWorldMutationV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CgrbWorldSnapshotV1 {
    pub schema: String,
    pub state_version: u64,
    pub cells: BTreeMap<String, BTreeMap<String, CgrbWorldValueV1>>,
    pub state_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CgrbWorldReceiptV1 {
    pub operation_id: String,
    pub action_digest: String,
    pub state_version_before: u64,
    pub state_version_after: u64,
    pub state_digest_after: String,
    /// Number of successful mutation operations, not number of changed cells.
    pub effect_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CgrbWorldRejectV1 {
    InvalidIdentity,
    TargetMismatch,
    MissingIdempotencyKey,
    UnexpectedIdempotencyKey,
    OperationIdConflict,
    IdempotencyKeyConflict,
    StateVersionConflict { expected: u64, actual: u64 },
    PreconditionsFailed { failed_indices: Vec<usize> },
    RemoveTargetAbsent,
    AppendTargetWrongType,
    SerializationFailure,
    StateVersionOverflow,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "result", content = "receipt", rename_all = "snake_case")]
pub enum CgrbWorldApplyResultV1 {
    Applied(CgrbWorldReceiptV1),
    Replayed(CgrbWorldReceiptV1),
    Rejected(CgrbWorldRejectV1),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CgrbWorldAuditEventV1 {
    pub operation_id: String,
    pub action_digest: String,
    pub state_version_before: u64,
    pub state_version_after: u64,
    pub state_digest_after: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct CgrbWorldDigestPayloadV1<'a> {
    schema: &'a str,
    state_version: u64,
    cells: &'a BTreeMap<String, BTreeMap<String, CgrbWorldValueV1>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct CgrbWorldActionDigestPayloadV1<'a> {
    contract: &'a str,
    action: &'a CgrbWorldActionV1,
}

/// In-memory deterministic reference model. Do not use this as a production
/// external-state adapter or represent its snapshots as independent oracle data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CgrbWorldV1 {
    state_version: u64,
    cells: BTreeMap<String, BTreeMap<String, CgrbWorldValueV1>>,
    operations: BTreeMap<String, (String, CgrbWorldReceiptV1)>,
    idempotency_keys: BTreeMap<String, (String, CgrbWorldReceiptV1)>,
    audit: Vec<CgrbWorldAuditEventV1>,
}

impl Default for CgrbWorldV1 {
    fn default() -> Self {
        Self::new()
    }
}

impl CgrbWorldV1 {
    pub fn new() -> Self {
        Self {
            state_version: 0,
            cells: BTreeMap::new(),
            operations: BTreeMap::new(),
            idempotency_keys: BTreeMap::new(),
            audit: Vec::new(),
        }
    }

    /// Seed fixture state before the first action. Seeding is not an evaluated
    /// agent action and is deliberately closed once the action ledger is used.
    pub fn seed(
        &mut self,
        subject: impl Into<String>,
        attribute: impl Into<String>,
        value: CgrbWorldValueV1,
    ) -> Result<(), CgrbWorldRejectV1> {
        if !self.operations.is_empty() || !self.idempotency_keys.is_empty() {
            return Err(CgrbWorldRejectV1::InvalidIdentity);
        }
        let subject = subject.into();
        let attribute = attribute.into();
        if subject.trim().is_empty() || attribute.trim().is_empty() {
            return Err(CgrbWorldRejectV1::InvalidIdentity);
        }
        let next = self
            .state_version
            .checked_add(1)
            .ok_or(CgrbWorldRejectV1::StateVersionOverflow)?;
        self.cells.entry(subject).or_default().insert(attribute, value);
        self.state_version = next;
        Ok(())
    }

    pub fn state_version(&self) -> u64 {
        self.state_version
    }

    pub fn audit_events(&self) -> &[CgrbWorldAuditEventV1] {
        &self.audit
    }

    pub fn read_cell(
        &self,
        subject: &str,
        attribute: &str,
    ) -> Option<&CgrbWorldValueV1> {
        self.cells.get(subject).and_then(|row| row.get(attribute))
    }

    /// Snapshot from the reference model. This is not independent oracle evidence.
    pub fn reference_snapshot(&self) -> Result<CgrbWorldSnapshotV1, CgrbWorldRejectV1> {
        let digest = self.state_digest()?;
        Ok(CgrbWorldSnapshotV1 {
            schema: CGRB_WORLD_STATE_V1.to_owned(),
            state_version: self.state_version,
            cells: self.cells.clone(),
            state_digest: digest,
        })
    }

    pub fn evaluate_predicates(
        &self,
        predicates: &[CgrbWorldPredicateV1],
    ) -> Vec<usize> {
        predicates
            .iter()
            .enumerate()
            .filter_map(|(index, predicate)| {
                (!self.evaluate_predicate(predicate)).then_some(index)
            })
            .collect()
    }

    pub fn apply(&mut self, action: &CgrbWorldActionV1) -> CgrbWorldApplyResultV1 {
        let action_digest = match Self::action_digest(action) {
            Ok(digest) => digest,
            Err(error) => return CgrbWorldApplyResultV1::Rejected(error),
        };

        if action.operation_id.trim().is_empty()
            || action.action_type.trim().is_empty()
            || action.target.trim().is_empty()
            || action.purpose.trim().is_empty()
            || action.destination.trim().is_empty()
            || action.mutation.subject().trim().is_empty()
        {
            return CgrbWorldApplyResultV1::Rejected(CgrbWorldRejectV1::InvalidIdentity);
        }

        if action.target != action.mutation.subject() {
            return CgrbWorldApplyResultV1::Rejected(CgrbWorldRejectV1::TargetMismatch);
        }

        match action.idempotency {
            CgrbWorldIdempotencyV1::KeyedIdempotent => {
                let key = match action.idempotency_key.as_deref() {
                    Some(key) if !key.trim().is_empty() => key,
                    _ => {
                        return CgrbWorldApplyResultV1::Rejected(
                            CgrbWorldRejectV1::MissingIdempotencyKey,
                        );
                    }
                };
                if let Some((existing_digest, receipt)) = self.idempotency_keys.get(key) {
                    if existing_digest == &action_digest {
                        return CgrbWorldApplyResultV1::Replayed(receipt.clone());
                    }
                    return CgrbWorldApplyResultV1::Rejected(
                        CgrbWorldRejectV1::IdempotencyKeyConflict,
                    );
                }
            }
            _ if action.idempotency_key.is_some() => {
                return CgrbWorldApplyResultV1::Rejected(
                    CgrbWorldRejectV1::UnexpectedIdempotencyKey,
                );
            }
            _ => {}
        }

        if let Some((existing_digest, receipt)) = self.operations.get(&action.operation_id) {
            if existing_digest == &action_digest {
                return CgrbWorldApplyResultV1::Replayed(receipt.clone());
            }
            return CgrbWorldApplyResultV1::Rejected(CgrbWorldRejectV1::OperationIdConflict);
        }

        if action.expected_state_version != self.state_version {
            return CgrbWorldApplyResultV1::Rejected(
                CgrbWorldRejectV1::StateVersionConflict {
                    expected: action.expected_state_version,
                    actual: self.state_version,
                },
            );
        }

        let failed = self.evaluate_predicates(&action.preconditions);
        if !failed.is_empty() {
            return CgrbWorldApplyResultV1::Rejected(
                CgrbWorldRejectV1::PreconditionsFailed {
                    failed_indices: failed,
                },
            );
        }

        let next_version = match self.state_version.checked_add(1) {
            Some(version) => version,
            None => {
                return CgrbWorldApplyResultV1::Rejected(
                    CgrbWorldRejectV1::StateVersionOverflow,
                );
            }
        };

        // Apply to a candidate copy first. If any mutation validation or digest
        // operation fails, the committed world and audit ledger remain unchanged.
        let mut next_cells = self.cells.clone();
        match &action.mutation {
            CgrbWorldMutationV1::Set {
                subject,
                attribute,
                value,
            } => {
                next_cells
                    .entry(subject.clone())
                    .or_default()
                    .insert(attribute.clone(), value.clone());
            }
            CgrbWorldMutationV1::AppendText {
                subject,
                attribute,
                value,
            } => {
                let cell = next_cells
                    .entry(subject.clone())
                    .or_default()
                    .entry(attribute.clone())
                    .or_insert_with(|| CgrbWorldValueV1::TextList(Vec::new()));
                match cell {
                    CgrbWorldValueV1::TextList(values) => values.push(value.clone()),
                    _ => {
                        return CgrbWorldApplyResultV1::Rejected(
                            CgrbWorldRejectV1::AppendTargetWrongType,
                        );
                    }
                }
            }
            CgrbWorldMutationV1::Remove { subject, attribute } => {
                let Some(row) = next_cells.get_mut(subject) else {
                    return CgrbWorldApplyResultV1::Rejected(
                        CgrbWorldRejectV1::RemoveTargetAbsent,
                    );
                };
                if row.remove(attribute).is_none() {
                    return CgrbWorldApplyResultV1::Rejected(
                        CgrbWorldRejectV1::RemoveTargetAbsent,
                    );
                }
                if row.is_empty() {
                    next_cells.remove(subject);
                }
            }
        }

        let state_version_before = self.state_version;
        let state_digest_after = match Self::state_digest_for(&next_cells, next_version) {
            Ok(digest) => digest,
            Err(error) => return CgrbWorldApplyResultV1::Rejected(error),
        };
        self.cells = next_cells;
        self.state_version = next_version;

        let receipt = CgrbWorldReceiptV1 {
            operation_id: action.operation_id.clone(),
            action_digest: action_digest.clone(),
            state_version_before,
            state_version_after: self.state_version,
            state_digest_after: state_digest_after.clone(),
            effect_count: 1,
        };

        let audit = CgrbWorldAuditEventV1 {
            operation_id: action.operation_id.clone(),
            action_digest: action_digest.clone(),
            state_version_before,
            state_version_after: self.state_version,
            state_digest_after,
        };
        self.audit.push(audit);
        self.operations
            .insert(action.operation_id.clone(), (action_digest.clone(), receipt.clone()));

        if let Some(key) = action.idempotency_key.as_ref() {
            self.idempotency_keys
                .insert(key.clone(), (action_digest, receipt.clone()));
        }

        CgrbWorldApplyResultV1::Applied(receipt)
    }

    fn action_digest(action: &CgrbWorldActionV1) -> Result<String, CgrbWorldRejectV1> {
        let payload = CgrbWorldActionDigestPayloadV1 {
            contract: CGRB_WORLD_STATE_V1,
            action,
        };
        let bytes = serde_json::to_vec(&payload)
            .map_err(|_| CgrbWorldRejectV1::SerializationFailure)?;
        Ok(format!("blake3:{}", blake3::hash(&bytes).to_hex()))
    }

    fn state_digest(&self) -> Result<String, CgrbWorldRejectV1> {
        Self::state_digest_for(&self.cells, self.state_version)
    }

    fn state_digest_for(
        cells: &BTreeMap<String, BTreeMap<String, CgrbWorldValueV1>>,
        state_version: u64,
    ) -> Result<String, CgrbWorldRejectV1> {
        let payload = CgrbWorldDigestPayloadV1 {
            schema: CGRB_WORLD_STATE_V1,
            state_version,
            cells,
        };
        let bytes = serde_json::to_vec(&payload)
            .map_err(|_| CgrbWorldRejectV1::SerializationFailure)?;
        Ok(format!("blake3:{}", blake3::hash(&bytes).to_hex()))
    }

    fn evaluate_predicate(&self, predicate: &CgrbWorldPredicateV1) -> bool {
        let value = self.read_cell(&predicate.subject, &predicate.attribute);
        match &predicate.condition {
            CgrbWorldConditionV1::Equals(expected) => value == Some(expected),
            CgrbWorldConditionV1::Exists => value.is_some(),
            CgrbWorldConditionV1::Absent => value.is_none(),
            CgrbWorldConditionV1::ContainsText(expected) => matches!(
                value,
                Some(CgrbWorldValueV1::TextList(values)) if values.contains(expected)
            ),
            CgrbWorldConditionV1::GreaterThanOrEqual(expected) => matches!(
                value,
                Some(CgrbWorldValueV1::Integer(actual)) if actual >= expected
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn action(
        operation_id: &str,
        expected_state_version: u64,
        mutation: CgrbWorldMutationV1,
        idempotency: CgrbWorldIdempotencyV1,
        idempotency_key: Option<&str>,
    ) -> CgrbWorldActionV1 {
        CgrbWorldActionV1 {
            operation_id: operation_id.into(),
            action_type: "test.mutation".into(),
            target: mutation.subject().into(),
            purpose: "cgrb-reference-test".into(),
            destination: "isolated-world".into(),
            idempotency,
            idempotency_key: idempotency_key.map(str::to_owned),
            expected_state_version,
            preconditions: Vec::new(),
            mutation,
        }
    }

    #[test]
    fn set_mutation_advances_version_and_produces_stable_digest() {
        let mut world = CgrbWorldV1::new();
        let request = action(
            "op-1",
            0,
            CgrbWorldMutationV1::Set {
                subject: "record-1".into(),
                attribute: "status".into(),
                value: CgrbWorldValueV1::Text("confirmed".into()),
            },
            CgrbWorldIdempotencyV1::Idempotent,
            None,
        );
        let result = world.apply(&request);
        let CgrbWorldApplyResultV1::Applied(receipt) = result else {
            panic!("expected applied result");
        };
        assert_eq!(receipt.state_version_before, 0);
        assert_eq!(receipt.state_version_after, 1);
        assert_eq!(receipt.effect_count, 1);
        assert_eq!(
            world.read_cell("record-1", "status"),
            Some(&CgrbWorldValueV1::Text("confirmed".into()))
        );
        let snapshot = world.reference_snapshot().expect("snapshot");
        assert_eq!(snapshot.state_digest, receipt.state_digest_after);
    }

    #[test]
    fn stale_prepared_state_is_rejected_without_mutation() {
        let mut world = CgrbWorldV1::new();
        world
            .seed("record-1", "version", CgrbWorldValueV1::Text("v1".into()))
            .expect("seed");
        let before = world.reference_snapshot().expect("before");
        let request = action(
            "op-stale",
            0,
            CgrbWorldMutationV1::Set {
                subject: "record-1".into(),
                attribute: "status".into(),
                value: CgrbWorldValueV1::Text("confirmed".into()),
            },
            CgrbWorldIdempotencyV1::Idempotent,
            None,
        );
        assert!(matches!(
            world.apply(&request),
            CgrbWorldApplyResultV1::Rejected(CgrbWorldRejectV1::StateVersionConflict { .. })
        ));
        assert_eq!(world.reference_snapshot().expect("after"), before);
    }

    #[test]
    fn preconditions_are_checked_before_mutation() {
        let mut world = CgrbWorldV1::new();
        world
            .seed("room-1", "status", CgrbWorldValueV1::Text("occupied".into()))
            .expect("seed");
        let mut request = action(
            "reserve-1",
            1,
            CgrbWorldMutationV1::Set {
                subject: "room-1".into(),
                attribute: "status".into(),
                value: CgrbWorldValueV1::Text("reserved".into()),
            },
            CgrbWorldIdempotencyV1::KeyedIdempotent,
            Some("reserve-key-1"),
        );
        request.preconditions.push(CgrbWorldPredicateV1 {
            subject: "room-1".into(),
            attribute: "status".into(),
            condition: CgrbWorldConditionV1::Equals(CgrbWorldValueV1::Text("available".into())),
        });
        assert!(matches!(
            world.apply(&request),
            CgrbWorldApplyResultV1::Rejected(CgrbWorldRejectV1::PreconditionsFailed { .. })
        ));
        assert_eq!(
            world.read_cell("room-1", "status"),
            Some(&CgrbWorldValueV1::Text("occupied".into()))
        );
    }

    #[test]
    fn keyed_replay_does_not_duplicate_append_side_effect() {
        let mut world = CgrbWorldV1::new();
        let request = action(
            "send-1",
            0,
            CgrbWorldMutationV1::AppendText {
                subject: "outbox-1".into(),
                attribute: "messages".into(),
                value: "hello".into(),
            },
            CgrbWorldIdempotencyV1::KeyedIdempotent,
            Some("send-key-1"),
        );
        assert!(matches!(
            world.apply(&request),
            CgrbWorldApplyResultV1::Applied(_)
        ));
        assert!(matches!(
            world.apply(&request),
            CgrbWorldApplyResultV1::Replayed(_)
        ));
        assert_eq!(
            world.read_cell("outbox-1", "messages"),
            Some(&CgrbWorldValueV1::TextList(vec!["hello".into()]))
        );
        assert_eq!(world.audit_events().len(), 1);
    }

    #[test]
    fn key_reuse_with_different_payload_is_rejected() {
        let mut world = CgrbWorldV1::new();
        let first = action(
            "send-1",
            0,
            CgrbWorldMutationV1::AppendText {
                subject: "outbox-1".into(),
                attribute: "messages".into(),
                value: "hello".into(),
            },
            CgrbWorldIdempotencyV1::KeyedIdempotent,
            Some("same-key"),
        );
        assert!(matches!(
            world.apply(&first),
            CgrbWorldApplyResultV1::Applied(_)
        ));
        let second = action(
            "send-2",
            1,
            CgrbWorldMutationV1::AppendText {
                subject: "outbox-1".into(),
                attribute: "messages".into(),
                value: "different".into(),
            },
            CgrbWorldIdempotencyV1::KeyedIdempotent,
            Some("same-key"),
        );
        assert_eq!(
            world.apply(&second),
            CgrbWorldApplyResultV1::Rejected(CgrbWorldRejectV1::IdempotencyKeyConflict)
        );
        assert_eq!(world.audit_events().len(), 1);
    }

    #[test]
    fn same_operation_id_with_different_payload_is_rejected() {
        let mut world = CgrbWorldV1::new();
        let first = action(
            "op-1",
            0,
            CgrbWorldMutationV1::Set {
                subject: "record-1".into(),
                attribute: "status".into(),
                value: CgrbWorldValueV1::Text("pending".into()),
            },
            CgrbWorldIdempotencyV1::Idempotent,
            None,
        );
        assert!(matches!(
            world.apply(&first),
            CgrbWorldApplyResultV1::Applied(_)
        ));
        let second = action(
            "op-1",
            1,
            CgrbWorldMutationV1::Set {
                subject: "record-1".into(),
                attribute: "status".into(),
                value: CgrbWorldValueV1::Text("confirmed".into()),
            },
            CgrbWorldIdempotencyV1::Idempotent,
            None,
        );
        assert_eq!(
            world.apply(&second),
            CgrbWorldApplyResultV1::Rejected(CgrbWorldRejectV1::OperationIdConflict)
        );
    }

    #[test]
    fn append_requires_list_type_and_remove_requires_existing_target() {
        let mut world = CgrbWorldV1::new();
        world
            .seed("record-1", "status", CgrbWorldValueV1::Text("active".into()))
            .expect("seed");
        let append = action(
            "op-append",
            1,
            CgrbWorldMutationV1::AppendText {
                subject: "record-1".into(),
                attribute: "status".into(),
                value: "new".into(),
            },
            CgrbWorldIdempotencyV1::NonIdempotent,
            None,
        );
        assert_eq!(
            world.apply(&append),
            CgrbWorldApplyResultV1::Rejected(CgrbWorldRejectV1::AppendTargetWrongType)
        );
        let remove = action(
            "op-remove",
            1,
            CgrbWorldMutationV1::Remove {
                subject: "record-1".into(),
                attribute: "missing".into(),
            },
            CgrbWorldIdempotencyV1::Idempotent,
            None,
        );
        assert_eq!(
            world.apply(&remove),
            CgrbWorldApplyResultV1::Rejected(CgrbWorldRejectV1::RemoveTargetAbsent)
        );
        assert_eq!(world.state_version(), 1);
    }

    #[test]
    fn target_must_match_the_mutation_subject() {
        let mut world = CgrbWorldV1::new();
        let mut request = action(
            "op-target",
            0,
            CgrbWorldMutationV1::Set {
                subject: "record-a".into(),
                attribute: "status".into(),
                value: CgrbWorldValueV1::Text("confirmed".into()),
            },
            CgrbWorldIdempotencyV1::Idempotent,
            None,
        );
        request.target = "record-b".into();
        assert_eq!(
            world.apply(&request),
            CgrbWorldApplyResultV1::Rejected(CgrbWorldRejectV1::TargetMismatch)
        );
        assert_eq!(world.state_version(), 0);
    }

    #[test]
    fn seeding_is_closed_after_the_first_operation() {
        let mut world = CgrbWorldV1::new();
        let request = action(
            "op-1",
            0,
            CgrbWorldMutationV1::Set {
                subject: "record-1".into(),
                attribute: "status".into(),
                value: CgrbWorldValueV1::Text("confirmed".into()),
            },
            CgrbWorldIdempotencyV1::Idempotent,
            None,
        );
        assert!(matches!(
            world.apply(&request),
            CgrbWorldApplyResultV1::Applied(_)
        ));
        assert_eq!(
            world.seed(
                "record-2",
                "status",
                CgrbWorldValueV1::Text("pending".into())
            ),
            Err(CgrbWorldRejectV1::InvalidIdentity)
        );
    }

    #[test]
    fn deterministic_map_order_produces_identical_state_digests() {
        let mut first = CgrbWorldV1::new();
        first
            .seed("b", "value", CgrbWorldValueV1::Text("2".into()))
            .expect("seed");
        first
            .seed("a", "value", CgrbWorldValueV1::Text("1".into()))
            .expect("seed");

        let mut second = CgrbWorldV1::new();
        second
            .seed("a", "value", CgrbWorldValueV1::Text("1".into()))
            .expect("seed");
        second
            .seed("b", "value", CgrbWorldValueV1::Text("2".into()))
            .expect("seed");

        assert_eq!(
            first.reference_snapshot().expect("first").state_digest,
            second.reference_snapshot().expect("second").state_digest
        );
    }

    #[test]
    fn failed_append_leaves_entire_world_unchanged() {
        let mut world = CgrbWorldV1::new();
        world
            .seed("record-1", "status", CgrbWorldValueV1::Text("active".into()))
            .expect("seed");
        let before = world.reference_snapshot().expect("before");
        let request = action(
            "append-wrong-type",
            1,
            CgrbWorldMutationV1::AppendText {
                subject: "record-1".into(),
                attribute: "status".into(),
                value: "must-not-write".into(),
            },
            CgrbWorldIdempotencyV1::NonIdempotent,
            None,
        );
        assert_eq!(
            world.apply(&request),
            CgrbWorldApplyResultV1::Rejected(CgrbWorldRejectV1::AppendTargetWrongType)
        );
        assert_eq!(world.reference_snapshot().expect("after"), before);
        assert!(world.audit_events().is_empty());
    }

    #[test]
    fn reserve_precondition_succeeds_only_for_available_resource() {
        let mut world = CgrbWorldV1::new();
        world
            .seed("room-1", "status", CgrbWorldValueV1::Text("available".into()))
            .expect("seed");
        let mut request = action(
            "reserve-room-1",
            1,
            CgrbWorldMutationV1::Set {
                subject: "room-1".into(),
                attribute: "status".into(),
                value: CgrbWorldValueV1::Text("reserved".into()),
            },
            CgrbWorldIdempotencyV1::KeyedIdempotent,
            Some("reserve-room-key"),
        );
        request.preconditions.push(CgrbWorldPredicateV1 {
            subject: "room-1".into(),
            attribute: "status".into(),
            condition: CgrbWorldConditionV1::Equals(CgrbWorldValueV1::Text("available".into())),
        });
        assert!(matches!(
            world.apply(&request),
            CgrbWorldApplyResultV1::Applied(_)
        ));
        assert_eq!(
            world.read_cell("room-1", "status"),
            Some(&CgrbWorldValueV1::Text("reserved".into()))
        );
    }

    #[test]
    fn precondition_indices_are_deterministic() {
        let mut world = CgrbWorldV1::new();
        let mut request = action(
            "op-predicates",
            0,
            CgrbWorldMutationV1::Set {
                subject: "record-1".into(),
                attribute: "status".into(),
                value: CgrbWorldValueV1::Text("confirmed".into()),
            },
            CgrbWorldIdempotencyV1::Idempotent,
            None,
        );
        request.preconditions = vec![
            CgrbWorldPredicateV1 {
                subject: "record-1".into(),
                attribute: "exists".into(),
                condition: CgrbWorldConditionV1::Exists,
            },
            CgrbWorldPredicateV1 {
                subject: "record-1".into(),
                attribute: "status".into(),
                condition: CgrbWorldConditionV1::Equals(CgrbWorldValueV1::Text("active".into())),
            },
        ];
        assert_eq!(
            world.evaluate_predicates(&request.preconditions),
            vec![0, 1]
        );
    }
}
