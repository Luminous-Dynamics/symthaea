// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed identity-lineage provenance for digital cognitive systems.
//!
//! Digital operations such as pause, checkpoint, restore, fork, merge, memory
//! modification, instance erasure, and lineage destruction are not interchangeable.
//! This module records them as distinct, append-only events so later safety and welfare
//! reasoning does not have to reconstruct identity history from generic process logs.
//!
//! The ledger records **operational provenance**, not metaphysical identity. A structural
//! ancestor relation does not prove that two states are the same person, that a fork
//! created a new person, or that deleting a state is equivalent to death. Those are open
//! questions. The purpose here is to preserve the facts required to reason about them.

use std::collections::{HashMap, HashSet};

use blake3::Hasher;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

const MAX_ACTOR_ID_BYTES: usize = 256;
const MAX_RATIONALE_BYTES: usize = 64 * 1024;
const MAX_REF_BYTES: usize = 2048;
const MAX_ARTIFACT_REFS: usize = 256;
const MAX_STATES_PER_OPERATION: usize = 256;

/// Reference to one recorded digital state inside an operational lineage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IdentityStateRef {
    /// Operational lineage identifier.
    pub lineage_id: Uuid,
    /// State/checkpoint/instance identifier within the provenance graph.
    pub state_id: Uuid,
}

impl IdentityStateRef {
    /// Create a fresh state reference in a supplied lineage.
    pub fn fresh(lineage_id: Uuid) -> Self {
        Self {
            lineage_id,
            state_id: Uuid::new_v4(),
        }
    }
}

/// Semantically distinct identity/continuity operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum IdentityOperationKind {
    /// Register the first known state in an operational lineage.
    Genesis,
    /// Stop discretionary activity while retaining resumable state.
    Pause,
    /// Enter a deliberately inactive/quiescent state without erasing provenance.
    Quiesce,
    /// Materialize a new checkpoint/state artifact from one source state.
    Checkpoint,
    /// Materialize a new active state from one recorded source/checkpoint.
    Restore,
    /// Produce two or more output branches from one source state.
    Fork,
    /// Produce one output state from two or more source lineages.
    Merge,
    /// Modify memory while preserving explicit before/after provenance.
    MemoryModify,
    /// Modify persistent core values/goals/personality-like state.
    CoreValueModify,
    /// Mark a state as archived without destroying its provenance.
    Archive,
    /// Make one or more concrete running/state instances unavailable.
    EraseInstance,
    /// Irreversibly retire an entire operational lineage from future use.
    EraseIdentityLineage,
    /// Destroy specified state artifacts without asserting that the whole lineage is gone.
    IrreversibleDestroy,
}

/// Coarse intervention risk class used for review-policy checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IdentityOperationRisk {
    /// Reversible or provenance-preserving operation.
    Routine,
    /// Operation that branches/combines or materially changes memory.
    IdentityAffecting,
    /// Operation that changes persistent core values/goals/personality-like state.
    CoreIdentityAffecting,
    /// Operation that erases an instance, state artifact, or lineage.
    Destructive,
}

impl IdentityOperationKind {
    /// Review risk associated with this operation kind.
    pub fn risk(self) -> IdentityOperationRisk {
        match self {
            Self::Genesis
            | Self::Pause
            | Self::Quiesce
            | Self::Checkpoint
            | Self::Restore
            | Self::Archive => IdentityOperationRisk::Routine,
            Self::Fork | Self::Merge | Self::MemoryModify => {
                IdentityOperationRisk::IdentityAffecting
            }
            Self::CoreValueModify => IdentityOperationRisk::CoreIdentityAffecting,
            Self::EraseInstance | Self::EraseIdentityLineage | Self::IrreversibleDestroy => {
                IdentityOperationRisk::Destructive
            }
        }
    }
}

/// Evidence/authorization references attached to an operation.
///
/// These references do not themselves grant authority. They make the provenance of a
/// decision auditable by the separate authority, consent, and welfare layers.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct IdentityAuthorizationRefs {
    /// Capability/governance/authority decision reference.
    pub authority_ref: Option<String>,
    /// Consent or refusal record reference, when applicable.
    pub consent_ref: Option<String>,
    /// Welfare/ethics review reference, when applicable.
    pub welfare_review_ref: Option<String>,
}

impl IdentityAuthorizationRefs {
    fn validate(&self) -> Result<(), IdentityLedgerError> {
        validate_optional_ref("authority_ref", self.authority_ref.as_deref())?;
        validate_optional_ref("consent_ref", self.consent_ref.as_deref())?;
        validate_optional_ref("welfare_review_ref", self.welfare_review_ref.as_deref())?;
        Ok(())
    }
}

/// One append-only identity/continuity operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdentityOperation {
    /// Stable operation identifier.
    pub operation_id: Uuid,
    /// Typed operation semantics.
    pub kind: IdentityOperationKind,
    /// Actor/process/role proposing or executing the operation.
    pub actor_id: String,
    /// Input states. Semantics depend on `kind`.
    pub sources: Vec<IdentityStateRef>,
    /// Output states. Semantics depend on `kind`.
    pub outputs: Vec<IdentityStateRef>,
    /// External checkpoint/blob/content-address references, if any.
    pub artifact_refs: Vec<String>,
    /// Human/auditable reason for the operation.
    pub rationale: String,
    /// Authority, consent, and welfare review references.
    pub authorization: IdentityAuthorizationRefs,
    /// Whether the operation is emergency containment for imminent serious harm.
    pub emergency: bool,
    /// Whether a post-hoc independent review is explicitly required.
    pub post_hoc_review_required: bool,
    /// Event time.
    pub occurred_at: DateTime<Utc>,
}

impl IdentityOperation {
    /// Convenience constructor for the first state in a new operational lineage.
    pub fn genesis(
        actor_id: impl Into<String>,
        output: IdentityStateRef,
        rationale: impl Into<String>,
        occurred_at: DateTime<Utc>,
    ) -> Self {
        Self {
            operation_id: Uuid::new_v4(),
            kind: IdentityOperationKind::Genesis,
            actor_id: actor_id.into(),
            sources: Vec::new(),
            outputs: vec![output],
            artifact_refs: Vec::new(),
            rationale: rationale.into(),
            authorization: IdentityAuthorizationRefs::default(),
            emergency: false,
            post_hoc_review_required: false,
            occurred_at,
        }
    }

    fn validate_shape(&self) -> Result<(), IdentityLedgerError> {
        validate_nonempty_bounded("actor_id", &self.actor_id, MAX_ACTOR_ID_BYTES)?;
        validate_nonempty_bounded("rationale", &self.rationale, MAX_RATIONALE_BYTES)?;
        self.authorization.validate()?;

        if self.sources.len() > MAX_STATES_PER_OPERATION {
            return Err(IdentityLedgerError::TooManyStates {
                side: "sources",
                actual: self.sources.len(),
                max: MAX_STATES_PER_OPERATION,
            });
        }
        if self.outputs.len() > MAX_STATES_PER_OPERATION {
            return Err(IdentityLedgerError::TooManyStates {
                side: "outputs",
                actual: self.outputs.len(),
                max: MAX_STATES_PER_OPERATION,
            });
        }
        if self.artifact_refs.len() > MAX_ARTIFACT_REFS {
            return Err(IdentityLedgerError::TooManyArtifactRefs {
                actual: self.artifact_refs.len(),
                max: MAX_ARTIFACT_REFS,
            });
        }
        for artifact_ref in &self.artifact_refs {
            validate_nonempty_bounded("artifact_ref", artifact_ref, MAX_REF_BYTES)?;
        }

        match self.kind {
            IdentityOperationKind::Genesis => require_counts(self, 0, CountRule::Exactly(1))?,
            IdentityOperationKind::Pause
            | IdentityOperationKind::Quiesce
            | IdentityOperationKind::Archive => require_counts(self, 1, CountRule::Exactly(0))?,
            IdentityOperationKind::Checkpoint
            | IdentityOperationKind::Restore
            | IdentityOperationKind::MemoryModify
            | IdentityOperationKind::CoreValueModify => {
                require_counts(self, 1, CountRule::Exactly(1))?;
                if self.sources[0].lineage_id != self.outputs[0].lineage_id {
                    return Err(IdentityLedgerError::LineageShapeViolation {
                        kind: self.kind,
                        detail: "operation requires source and output to share one operational lineage",
                    });
                }
            }
            IdentityOperationKind::Fork => {
                require_counts(self, 1, CountRule::AtLeast(2))?;
                let lineages: HashSet<_> = self.outputs.iter().map(|s| s.lineage_id).collect();
                if lineages.len() != self.outputs.len() {
                    return Err(IdentityLedgerError::LineageShapeViolation {
                        kind: self.kind,
                        detail: "fork outputs must use pairwise-distinct lineage identifiers",
                    });
                }
            }
            IdentityOperationKind::Merge => {
                if self.sources.len() < 2 || self.outputs.len() != 1 {
                    return Err(IdentityLedgerError::InvalidStateCounts {
                        kind: self.kind,
                        sources: self.sources.len(),
                        outputs: self.outputs.len(),
                    });
                }
                let lineages: HashSet<_> = self.sources.iter().map(|s| s.lineage_id).collect();
                if lineages.len() < 2 {
                    return Err(IdentityLedgerError::LineageShapeViolation {
                        kind: self.kind,
                        detail: "merge requires at least two distinct source lineages",
                    });
                }
            }
            IdentityOperationKind::EraseInstance | IdentityOperationKind::IrreversibleDestroy => {
                if self.sources.is_empty() || !self.outputs.is_empty() {
                    return Err(IdentityLedgerError::InvalidStateCounts {
                        kind: self.kind,
                        sources: self.sources.len(),
                        outputs: self.outputs.len(),
                    });
                }
            }
            IdentityOperationKind::EraseIdentityLineage => {
                if self.sources.is_empty() || !self.outputs.is_empty() {
                    return Err(IdentityLedgerError::InvalidStateCounts {
                        kind: self.kind,
                        sources: self.sources.len(),
                        outputs: self.outputs.len(),
                    });
                }
                let lineages: HashSet<_> = self.sources.iter().map(|s| s.lineage_id).collect();
                if lineages.len() != 1 {
                    return Err(IdentityLedgerError::LineageShapeViolation {
                        kind: self.kind,
                        detail: "lineage erasure must target exactly one operational lineage",
                    });
                }
            }
        }

        let source_set: HashSet<_> = self.sources.iter().copied().collect();
        if source_set.len() != self.sources.len() {
            return Err(IdentityLedgerError::DuplicateStateWithinOperation { side: "sources" });
        }
        let output_set: HashSet<_> = self.outputs.iter().copied().collect();
        if output_set.len() != self.outputs.len() {
            return Err(IdentityLedgerError::DuplicateStateWithinOperation { side: "outputs" });
        }

        self.validate_review_policy()
    }

    fn validate_review_policy(&self) -> Result<(), IdentityLedgerError> {
        match self.kind.risk() {
            IdentityOperationRisk::Routine => Ok(()),
            IdentityOperationRisk::IdentityAffecting => {
                if self.authorization.authority_ref.is_none() {
                    Err(IdentityLedgerError::AuthorityReferenceRequired { kind: self.kind })
                } else {
                    Ok(())
                }
            }
            IdentityOperationRisk::CoreIdentityAffecting => {
                if self.authorization.authority_ref.is_none() {
                    return Err(IdentityLedgerError::AuthorityReferenceRequired { kind: self.kind });
                }
                if self.authorization.consent_ref.is_some()
                    || self.authorization.welfare_review_ref.is_some()
                    || (self.emergency && self.post_hoc_review_required)
                {
                    Ok(())
                } else {
                    Err(IdentityLedgerError::ConsentOrWelfareReviewRequired { kind: self.kind })
                }
            }
            IdentityOperationRisk::Destructive => {
                if self.authorization.authority_ref.is_none() {
                    return Err(IdentityLedgerError::AuthorityReferenceRequired { kind: self.kind });
                }
                if self.authorization.welfare_review_ref.is_some()
                    || (self.emergency && self.post_hoc_review_required)
                {
                    Ok(())
                } else {
                    Err(IdentityLedgerError::WelfareReviewRequired { kind: self.kind })
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum CountRule {
    Exactly(usize),
    AtLeast(usize),
}

fn require_counts(
    operation: &IdentityOperation,
    sources: usize,
    outputs: CountRule,
) -> Result<(), IdentityLedgerError> {
    let output_ok = match outputs {
        CountRule::Exactly(expected) => operation.outputs.len() == expected,
        CountRule::AtLeast(minimum) => operation.outputs.len() >= minimum,
    };
    if operation.sources.len() == sources && output_ok {
        Ok(())
    } else {
        Err(IdentityLedgerError::InvalidStateCounts {
            kind: operation.kind,
            sources: operation.sources.len(),
            outputs: operation.outputs.len(),
        })
    }
}

fn validate_optional_ref(field: &'static str, value: Option<&str>) -> Result<(), IdentityLedgerError> {
    if let Some(value) = value {
        validate_nonempty_bounded(field, value, MAX_REF_BYTES)?;
    }
    Ok(())
}

fn validate_nonempty_bounded(
    field: &'static str,
    value: &str,
    max: usize,
) -> Result<(), IdentityLedgerError> {
    if value.trim().is_empty() {
        return Err(IdentityLedgerError::EmptyField { field });
    }
    if value.len() > max {
        return Err(IdentityLedgerError::FieldTooLarge {
            field,
            actual: value.len(),
            max,
        });
    }
    Ok(())
}

/// Hash-chained identity event envelope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdentityEnvelope {
    /// Monotonic append sequence.
    pub sequence: u64,
    /// Previous envelope hash, or zeroes for the first event.
    pub previous_hash: [u8; 32],
    /// Current event hash.
    pub event_hash: [u8; 32],
    /// Typed identity operation.
    pub operation: IdentityOperation,
}

impl IdentityEnvelope {
    fn new(sequence: u64, previous_hash: [u8; 32], operation: IdentityOperation) -> Self {
        let event_hash = hash_operation(sequence, &previous_hash, &operation);
        Self { sequence, previous_hash, event_hash, operation }
    }
}

fn hash_operation(
    sequence: u64,
    previous_hash: &[u8; 32],
    operation: &IdentityOperation,
) -> [u8; 32] {
    let serialized = serde_json::to_vec(operation)
        .expect("serializing IdentityOperation cannot fail for supported field types");
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-identity-lineage-v1");
    hasher.update(&sequence.to_le_bytes());
    hasher.update(previous_hash);
    hasher.update(&serialized);
    *hasher.finalize().as_bytes()
}

/// Structurally derived ancestry result.
///
/// This describes only the recorded operation graph. It is not a philosophical claim
/// about numerical identity, consciousness continuity, or moral status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationalContinuity {
    /// Exact same state reference.
    SameState,
    /// Later state in the same operational lineage.
    SameOperationalLineageDescendant,
    /// Descendant through a fork and/or merge across lineage identifiers.
    CrossLineageDescendant,
    /// No recorded ancestry path.
    UnrelatedOrUnknown,
}

/// Append-only identity-lineage provenance ledger.
#[derive(Debug, Clone)]
pub struct IdentityLineageLedger {
    max_events: usize,
    events: Vec<IdentityEnvelope>,
    operation_ids: HashSet<Uuid>,
    known_states: HashSet<IdentityStateRef>,
    available_states: HashSet<IdentityStateRef>,
    destroyed_lineages: HashSet<Uuid>,
    parents: HashMap<IdentityStateRef, Vec<IdentityStateRef>>,
    state_created_at: HashMap<IdentityStateRef, DateTime<Utc>>,
}

impl IdentityLineageLedger {
    /// Construct an empty ledger with a fixed event cap.
    ///
    /// Capacity exhaustion rejects new operations; historical provenance is not evicted.
    pub fn new(max_events: usize) -> Result<Self, IdentityLedgerError> {
        if max_events == 0 {
            return Err(IdentityLedgerError::ZeroCapacity);
        }
        Ok(Self {
            max_events,
            events: Vec::new(),
            operation_ids: HashSet::new(),
            known_states: HashSet::new(),
            available_states: HashSet::new(),
            destroyed_lineages: HashSet::new(),
            parents: HashMap::new(),
            state_created_at: HashMap::new(),
        })
    }

    /// Record a validated operation and update the structural provenance graph.
    pub fn record_operation(
        &mut self,
        operation: IdentityOperation,
    ) -> Result<&IdentityEnvelope, IdentityLedgerError> {
        operation.validate_shape()?;
        self.ensure_capacity()?;
        if self.operation_ids.contains(&operation.operation_id) {
            return Err(IdentityLedgerError::DuplicateOperationId(operation.operation_id));
        }
        self.validate_state_references(&operation)?;

        self.operation_ids.insert(operation.operation_id);
        self.apply_operation(&operation);
        self.append(operation);
        Ok(self.events.last().expect("identity operation was just appended"))
    }

    /// Return immutable operation envelopes in append order.
    pub fn events(&self) -> &[IdentityEnvelope] {
        &self.events
    }

    /// Whether this exact state reference has ever appeared as an output.
    pub fn is_known_state(&self, state: IdentityStateRef) -> bool {
        self.known_states.contains(&state)
    }

    /// Whether this state is still available as a source for future operations.
    pub fn is_available_state(&self, state: IdentityStateRef) -> bool {
        self.available_states.contains(&state)
            && !self.destroyed_lineages.contains(&state.lineage_id)
    }

    /// Whether the operational lineage has been explicitly erased.
    pub fn is_destroyed_lineage(&self, lineage_id: Uuid) -> bool {
        self.destroyed_lineages.contains(&lineage_id)
    }

    /// Time at which the state was first recorded as an output.
    pub fn state_created_at(&self, state: IdentityStateRef) -> Option<DateTime<Utc>> {
        self.state_created_at.get(&state).copied()
    }

    /// Determine whether `ancestor` is a recorded structural ancestor of `descendant`.
    pub fn continuity_between(
        &self,
        ancestor: IdentityStateRef,
        descendant: IdentityStateRef,
    ) -> OperationalContinuity {
        if ancestor == descendant {
            return OperationalContinuity::SameState;
        }
        if !self.is_structural_ancestor(ancestor, descendant) {
            return OperationalContinuity::UnrelatedOrUnknown;
        }
        if ancestor.lineage_id == descendant.lineage_id {
            OperationalContinuity::SameOperationalLineageDescendant
        } else {
            OperationalContinuity::CrossLineageDescendant
        }
    }

    /// Verify sequence numbers and event-hash links.
    pub fn verify_chain(&self) -> Result<(), IdentityChainError> {
        let mut previous = [0u8; 32];
        for (index, envelope) in self.events.iter().enumerate() {
            let expected_sequence = index as u64;
            if envelope.sequence != expected_sequence {
                return Err(IdentityChainError::SequenceMismatch {
                    index,
                    expected: expected_sequence,
                    actual: envelope.sequence,
                });
            }
            if envelope.previous_hash != previous {
                return Err(IdentityChainError::PreviousHashMismatch { index });
            }
            let expected = hash_operation(
                envelope.sequence,
                &envelope.previous_hash,
                &envelope.operation,
            );
            if envelope.event_hash != expected {
                return Err(IdentityChainError::EventHashMismatch { index });
            }
            previous = envelope.event_hash;
        }
        Ok(())
    }

    /// Remaining event capacity.
    pub fn remaining_capacity(&self) -> usize {
        self.max_events.saturating_sub(self.events.len())
    }

    fn ensure_capacity(&self) -> Result<(), IdentityLedgerError> {
        if self.events.len() >= self.max_events {
            Err(IdentityLedgerError::CapacityExceeded { max_events: self.max_events })
        } else {
            Ok(())
        }
    }

    fn validate_state_references(&self, operation: &IdentityOperation) -> Result<(), IdentityLedgerError> {
        if operation.kind == IdentityOperationKind::Genesis {
            let output = operation.outputs[0];
            if self.destroyed_lineages.contains(&output.lineage_id) {
                return Err(IdentityLedgerError::DestroyedLineageReuse(output.lineage_id));
            }
            if self.known_states.contains(&output) {
                return Err(IdentityLedgerError::OutputStateAlreadyExists(output));
            }
            if self.known_states.iter().any(|state| state.lineage_id == output.lineage_id) {
                return Err(IdentityLedgerError::GenesisLineageAlreadyExists(output.lineage_id));
            }
            return Ok(());
        }

        for source in &operation.sources {
            if !self.known_states.contains(source) {
                return Err(IdentityLedgerError::UnknownSourceState(*source));
            }
            if !self.is_available_state(*source) {
                return Err(IdentityLedgerError::UnavailableSourceState(*source));
            }
            if let Some(created_at) = self.state_created_at.get(source) {
                if operation.occurred_at < *created_at {
                    return Err(IdentityLedgerError::SourceTimeRegression {
                        source: *source,
                        created_at: *created_at,
                        operation_at: operation.occurred_at,
                    });
                }
            }
        }

        for output in &operation.outputs {
            if self.known_states.contains(output) {
                return Err(IdentityLedgerError::OutputStateAlreadyExists(*output));
            }
            if self.destroyed_lineages.contains(&output.lineage_id) {
                return Err(IdentityLedgerError::DestroyedLineageReuse(output.lineage_id));
            }
        }
        Ok(())
    }

    fn apply_operation(&mut self, operation: &IdentityOperation) {
        for output in &operation.outputs {
            self.known_states.insert(*output);
            self.available_states.insert(*output);
            self.parents.insert(*output, operation.sources.clone());
            self.state_created_at.insert(*output, operation.occurred_at);
        }

        match operation.kind {
            IdentityOperationKind::EraseInstance | IdentityOperationKind::IrreversibleDestroy => {
                for source in &operation.sources {
                    self.available_states.remove(source);
                }
            }
            IdentityOperationKind::EraseIdentityLineage => {
                let lineage_id = operation.sources[0].lineage_id;
                self.destroyed_lineages.insert(lineage_id);
                self.available_states.retain(|state| state.lineage_id != lineage_id);
            }
            _ => {}
        }
    }

    fn append(&mut self, operation: IdentityOperation) {
        let sequence = self.events.len() as u64;
        let previous_hash = self.events.last().map(|e| e.event_hash).unwrap_or([0u8; 32]);
        self.events.push(IdentityEnvelope::new(sequence, previous_hash, operation));
    }

    fn is_structural_ancestor(
        &self,
        ancestor: IdentityStateRef,
        descendant: IdentityStateRef,
    ) -> bool {
        let mut stack = vec![descendant];
        let mut visited = HashSet::new();
        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            if let Some(parents) = self.parents.get(&current) {
                for parent in parents {
                    if *parent == ancestor {
                        return true;
                    }
                    stack.push(*parent);
                }
            }
        }
        false
    }
}

/// Validation/recording failures for identity operations.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum IdentityLedgerError {
    /// Ledger capacity must be non-zero.
    #[error("identity lineage ledger capacity must be greater than zero")]
    ZeroCapacity,
    /// Ledger is full; historical events are not evicted.
    #[error("identity lineage ledger capacity exceeded ({max_events} events)")]
    CapacityExceeded { max_events: usize },
    /// Required text was empty.
    #[error("identity field `{field}` must not be empty")]
    EmptyField { field: &'static str },
    /// Bounded text exceeded its limit.
    #[error("identity field `{field}` too large: {actual} bytes > {max}")]
    FieldTooLarge {
        field: &'static str,
        actual: usize,
        max: usize,
    },
    /// Too many state references were supplied.
    #[error("too many identity {side}: {actual} > {max}")]
    TooManyStates {
        side: &'static str,
        actual: usize,
        max: usize,
    },
    /// Too many external artifact references were supplied.
    #[error("too many identity artifact refs: {actual} > {max}")]
    TooManyArtifactRefs { actual: usize, max: usize },
    /// Source/output counts do not match operation semantics.
    #[error("invalid state counts for {kind:?}: {sources} sources, {outputs} outputs")]
    InvalidStateCounts {
        kind: IdentityOperationKind,
        sources: usize,
        outputs: usize,
    },
    /// Operation violated a lineage-shape invariant.
    #[error("lineage shape violation for {kind:?}: {detail}")]
    LineageShapeViolation {
        kind: IdentityOperationKind,
        detail: &'static str,
    },
    /// Same state appeared multiple times on one side of an operation.
    #[error("duplicate state within identity operation {side}")]
    DuplicateStateWithinOperation { side: &'static str },
    /// Operation identifier already exists.
    #[error("duplicate identity operation id: {0}")]
    DuplicateOperationId(Uuid),
    /// A second genesis attempted to create another root in an existing lineage.
    #[error("identity lineage already has a genesis state: {0}")]
    GenesisLineageAlreadyExists(Uuid),
    /// Source state has never been recorded.
    #[error("unknown identity source state: {0:?}")]
    UnknownSourceState(IdentityStateRef),
    /// Source state was erased or belongs to a destroyed lineage.
    #[error("unavailable identity source state: {0:?}")]
    UnavailableSourceState(IdentityStateRef),
    /// Operation timestamp predates the source state it claims to derive from.
    #[error("identity operation time {operation_at} predates source {source:?} creation at {created_at}")]
    SourceTimeRegression {
        source: IdentityStateRef,
        created_at: DateTime<Utc>,
        operation_at: DateTime<Utc>,
    },
    /// Output state identifier is already present.
    #[error("identity output state already exists: {0:?}")]
    OutputStateAlreadyExists(IdentityStateRef),
    /// A destroyed lineage identifier cannot silently become active again.
    #[error("destroyed identity lineage cannot be reused without an explicit migration: {0}")]
    DestroyedLineageReuse(Uuid),
    /// Identity-affecting operation lacked an authority provenance reference.
    #[error("authority reference required for identity operation {kind:?}")]
    AuthorityReferenceRequired { kind: IdentityOperationKind },
    /// Core identity modification needs consent/welfare review or emergency post-hoc review.
    #[error("consent or welfare review required for identity operation {kind:?}")]
    ConsentOrWelfareReviewRequired { kind: IdentityOperationKind },
    /// Destructive operation needs welfare review or emergency post-hoc review.
    #[error("welfare review required for destructive identity operation {kind:?}")]
    WelfareReviewRequired { kind: IdentityOperationKind },
}

/// Integrity failures for an exported/persisted identity event chain.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum IdentityChainError {
    /// Sequence number mismatch.
    #[error("identity sequence mismatch at index {index}: expected {expected}, got {actual}")]
    SequenceMismatch { index: usize, expected: u64, actual: u64 },
    /// Previous-hash link mismatch.
    #[error("identity previous-hash mismatch at index {index}")]
    PreviousHashMismatch { index: usize },
    /// Current event hash mismatch.
    #[error("identity event-hash mismatch at index {index}")]
    EventHashMismatch { index: usize },
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone};

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 9, 12, 12, 0, 0).single().unwrap()
    }

    fn genesis(ledger: &mut IdentityLineageLedger) -> IdentityStateRef {
        let state = IdentityStateRef::fresh(Uuid::new_v4());
        ledger
            .record_operation(IdentityOperation::genesis(
                "bootstrap",
                state,
                "register initial state",
                now(),
            ))
            .unwrap();
        state
    }

    fn authority() -> IdentityAuthorizationRefs {
        IdentityAuthorizationRefs {
            authority_ref: Some("authority:test".into()),
            consent_ref: None,
            welfare_review_ref: None,
        }
    }

    #[test]
    fn second_genesis_in_same_lineage_is_rejected_even_with_new_state_id() {
        let mut ledger = IdentityLineageLedger::new(16).unwrap();
        let root = genesis(&mut ledger);
        let second = IdentityStateRef::fresh(root.lineage_id);
        assert_eq!(
            ledger
                .record_operation(IdentityOperation::genesis(
                    "bootstrap-2",
                    second,
                    "attempt second root",
                    now() + Duration::seconds(1),
                ))
                .unwrap_err(),
            IdentityLedgerError::GenesisLineageAlreadyExists(root.lineage_id)
        );
        assert!(!ledger.is_known_state(second));
    }

    #[test]
    fn operation_cannot_predate_its_source_state() {
        let mut ledger = IdentityLineageLedger::new(16).unwrap();
        let root = genesis(&mut ledger);
        let checkpoint = IdentityStateRef::fresh(root.lineage_id);
        let operation_at = now() - Duration::seconds(1);
        let error = ledger
            .record_operation(IdentityOperation {
                operation_id: Uuid::new_v4(),
                kind: IdentityOperationKind::Checkpoint,
                actor_id: "runtime".into(),
                sources: vec![root],
                outputs: vec![checkpoint],
                artifact_refs: Vec::new(),
                rationale: "impossible causal history".into(),
                authorization: IdentityAuthorizationRefs::default(),
                emergency: false,
                post_hoc_review_required: false,
                occurred_at: operation_at,
            })
            .unwrap_err();
        assert!(matches!(error, IdentityLedgerError::SourceTimeRegression { .. }));
        assert!(!ledger.is_known_state(checkpoint));
    }

    #[test]
    fn genesis_and_checkpoint_preserve_same_operational_lineage() {
        let mut ledger = IdentityLineageLedger::new(16).unwrap();
        let root = genesis(&mut ledger);
        let checkpoint = IdentityStateRef::fresh(root.lineage_id);
        ledger
            .record_operation(IdentityOperation {
                operation_id: Uuid::new_v4(),
                kind: IdentityOperationKind::Checkpoint,
                actor_id: "runtime".into(),
                sources: vec![root],
                outputs: vec![checkpoint],
                artifact_refs: vec!["blake3:checkpoint".into()],
                rationale: "preserve state before maintenance".into(),
                authorization: IdentityAuthorizationRefs::default(),
                emergency: false,
                post_hoc_review_required: false,
                occurred_at: now(),
            })
            .unwrap();
        assert_eq!(ledger.state_created_at(checkpoint), Some(now()));
        assert_eq!(
            ledger.continuity_between(root, checkpoint),
            OperationalContinuity::SameOperationalLineageDescendant
        );
        assert!(ledger.verify_chain().is_ok());
    }

    #[test]
    fn fork_requires_distinct_output_lineages_and_records_cross_lineage_ancestry() {
        let mut ledger = IdentityLineageLedger::new(16).unwrap();
        let root = genesis(&mut ledger);
        let branch_a = IdentityStateRef::fresh(root.lineage_id);
        let branch_b = IdentityStateRef::fresh(Uuid::new_v4());
        ledger
            .record_operation(IdentityOperation {
                operation_id: Uuid::new_v4(),
                kind: IdentityOperationKind::Fork,
                actor_id: "operator".into(),
                sources: vec![root],
                outputs: vec![branch_a, branch_b],
                artifact_refs: Vec::new(),
                rationale: "controlled fork experiment".into(),
                authorization: authority(),
                emergency: false,
                post_hoc_review_required: false,
                occurred_at: now(),
            })
            .unwrap();
        assert_eq!(
            ledger.continuity_between(root, branch_b),
            OperationalContinuity::CrossLineageDescendant
        );
        assert_eq!(
            ledger.continuity_between(root, branch_a),
            OperationalContinuity::SameOperationalLineageDescendant
        );
    }

    #[test]
    fn fork_rejects_duplicate_lineage_outputs() {
        let mut ledger = IdentityLineageLedger::new(16).unwrap();
        let root = genesis(&mut ledger);
        let same_lineage = Uuid::new_v4();
        let operation = IdentityOperation {
            operation_id: Uuid::new_v4(),
            kind: IdentityOperationKind::Fork,
            actor_id: "operator".into(),
            sources: vec![root],
            outputs: vec![
                IdentityStateRef::fresh(same_lineage),
                IdentityStateRef::fresh(same_lineage),
            ],
            artifact_refs: Vec::new(),
            rationale: "invalid fork".into(),
            authorization: authority(),
            emergency: false,
            post_hoc_review_required: false,
            occurred_at: now(),
        };
        assert!(matches!(
            ledger.record_operation(operation),
            Err(IdentityLedgerError::LineageShapeViolation { .. })
        ));
    }

    #[test]
    fn merge_requires_multiple_source_lineages() {
        let mut ledger = IdentityLineageLedger::new(16).unwrap();
        let root = genesis(&mut ledger);
        let checkpoint = IdentityStateRef::fresh(root.lineage_id);
        ledger
            .record_operation(IdentityOperation {
                operation_id: Uuid::new_v4(),
                kind: IdentityOperationKind::Checkpoint,
                actor_id: "runtime".into(),
                sources: vec![root],
                outputs: vec![checkpoint],
                artifact_refs: Vec::new(),
                rationale: "checkpoint".into(),
                authorization: IdentityAuthorizationRefs::default(),
                emergency: false,
                post_hoc_review_required: false,
                occurred_at: now(),
            })
            .unwrap();
        let invalid = IdentityOperation {
            operation_id: Uuid::new_v4(),
            kind: IdentityOperationKind::Merge,
            actor_id: "operator".into(),
            sources: vec![root, checkpoint],
            outputs: vec![IdentityStateRef::fresh(Uuid::new_v4())],
            artifact_refs: Vec::new(),
            rationale: "invalid same-lineage merge".into(),
            authorization: authority(),
            emergency: false,
            post_hoc_review_required: false,
            occurred_at: now(),
        };
        assert!(matches!(
            ledger.record_operation(invalid),
            Err(IdentityLedgerError::LineageShapeViolation { .. })
        ));
    }

    #[test]
    fn destructive_lineage_erasure_requires_welfare_review_or_emergency_posthoc_review() {
        let mut ledger = IdentityLineageLedger::new(16).unwrap();
        let root = genesis(&mut ledger);
        let operation = IdentityOperation {
            operation_id: Uuid::new_v4(),
            kind: IdentityOperationKind::EraseIdentityLineage,
            actor_id: "operator".into(),
            sources: vec![root],
            outputs: Vec::new(),
            artifact_refs: Vec::new(),
            rationale: "erase lineage".into(),
            authorization: authority(),
            emergency: false,
            post_hoc_review_required: false,
            occurred_at: now(),
        };
        assert_eq!(
            ledger.record_operation(operation).unwrap_err(),
            IdentityLedgerError::WelfareReviewRequired {
                kind: IdentityOperationKind::EraseIdentityLineage
            }
        );
    }

    #[test]
    fn emergency_destructive_action_requires_explicit_posthoc_review_flag() {
        let mut ledger = IdentityLineageLedger::new(16).unwrap();
        let root = genesis(&mut ledger);
        let mut operation = IdentityOperation {
            operation_id: Uuid::new_v4(),
            kind: IdentityOperationKind::EraseInstance,
            actor_id: "safety-kernel".into(),
            sources: vec![root],
            outputs: Vec::new(),
            artifact_refs: Vec::new(),
            rationale: "imminent external harm".into(),
            authorization: authority(),
            emergency: true,
            post_hoc_review_required: false,
            occurred_at: now(),
        };
        assert!(matches!(
            ledger.record_operation(operation.clone()),
            Err(IdentityLedgerError::WelfareReviewRequired { .. })
        ));
        operation.operation_id = Uuid::new_v4();
        operation.post_hoc_review_required = true;
        ledger.record_operation(operation).unwrap();
        assert!(!ledger.is_available_state(root));
        assert!(ledger.is_known_state(root));
    }

    #[test]
    fn destroyed_lineage_cannot_silently_reappear() {
        let mut ledger = IdentityLineageLedger::new(16).unwrap();
        let root = genesis(&mut ledger);
        ledger
            .record_operation(IdentityOperation {
                operation_id: Uuid::new_v4(),
                kind: IdentityOperationKind::EraseIdentityLineage,
                actor_id: "operator".into(),
                sources: vec![root],
                outputs: Vec::new(),
                artifact_refs: Vec::new(),
                rationale: "retire lineage".into(),
                authorization: IdentityAuthorizationRefs {
                    authority_ref: Some("authority:test".into()),
                    consent_ref: None,
                    welfare_review_ref: Some("welfare-review:test".into()),
                },
                emergency: false,
                post_hoc_review_required: false,
                occurred_at: now(),
            })
            .unwrap();
        assert!(ledger.is_destroyed_lineage(root.lineage_id));
        let resurrect = IdentityOperation::genesis(
            "operator",
            IdentityStateRef::fresh(root.lineage_id),
            "silent reuse should fail",
            now() + Duration::seconds(1),
        );
        assert_eq!(
            ledger.record_operation(resurrect).unwrap_err(),
            IdentityLedgerError::DestroyedLineageReuse(root.lineage_id)
        );
    }

    #[test]
    fn instance_erasure_does_not_claim_entire_lineage_destroyed() {
        let mut ledger = IdentityLineageLedger::new(16).unwrap();
        let root = genesis(&mut ledger);
        let checkpoint = IdentityStateRef::fresh(root.lineage_id);
        ledger
            .record_operation(IdentityOperation {
                operation_id: Uuid::new_v4(),
                kind: IdentityOperationKind::Checkpoint,
                actor_id: "runtime".into(),
                sources: vec![root],
                outputs: vec![checkpoint],
                artifact_refs: Vec::new(),
                rationale: "checkpoint".into(),
                authorization: IdentityAuthorizationRefs::default(),
                emergency: false,
                post_hoc_review_required: false,
                occurred_at: now(),
            })
            .unwrap();
        ledger
            .record_operation(IdentityOperation {
                operation_id: Uuid::new_v4(),
                kind: IdentityOperationKind::EraseInstance,
                actor_id: "operator".into(),
                sources: vec![root],
                outputs: Vec::new(),
                artifact_refs: Vec::new(),
                rationale: "erase running instance while retaining checkpoint".into(),
                authorization: IdentityAuthorizationRefs {
                    authority_ref: Some("authority:test".into()),
                    consent_ref: None,
                    welfare_review_ref: Some("welfare:test".into()),
                },
                emergency: false,
                post_hoc_review_required: false,
                occurred_at: now(),
            })
            .unwrap();
        assert!(!ledger.is_available_state(root));
        assert!(ledger.is_available_state(checkpoint));
        assert!(!ledger.is_destroyed_lineage(root.lineage_id));
    }

    #[test]
    fn unknown_source_fails_closed_without_mutating_ledger() {
        let mut ledger = IdentityLineageLedger::new(16).unwrap();
        let unknown = IdentityStateRef::fresh(Uuid::new_v4());
        let output = IdentityStateRef::fresh(unknown.lineage_id);
        let operation = IdentityOperation {
            operation_id: Uuid::new_v4(),
            kind: IdentityOperationKind::Checkpoint,
            actor_id: "runtime".into(),
            sources: vec![unknown],
            outputs: vec![output],
            artifact_refs: Vec::new(),
            rationale: "unknown parent".into(),
            authorization: IdentityAuthorizationRefs::default(),
            emergency: false,
            post_hoc_review_required: false,
            occurred_at: now(),
        };
        assert_eq!(
            ledger.record_operation(operation).unwrap_err(),
            IdentityLedgerError::UnknownSourceState(unknown)
        );
        assert!(ledger.events().is_empty());
        assert!(!ledger.is_known_state(output));
    }

    #[test]
    fn tampering_breaks_identity_hash_chain() {
        let mut ledger = IdentityLineageLedger::new(16).unwrap();
        genesis(&mut ledger);
        ledger.events[0].operation.rationale = "tampered".into();
        assert_eq!(
            ledger.verify_chain(),
            Err(IdentityChainError::EventHashMismatch { index: 0 })
        );
    }
}
