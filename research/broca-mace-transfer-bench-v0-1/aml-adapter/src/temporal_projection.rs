// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Dependency-aware invalidation for derived temporal memory projections.
//!
//! Summaries, embeddings, preference models, plans and caches are derived
//! artifacts. A correction or rights/lifecycle transition affecting their
//! source event must invalidate them and every downstream projection. This
//! module is an in-memory reference index; it does not delete external bytes
//! or replace the source/decision authorization authorities.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use sha2::{Digest, Sha256};

use crate::streaming_session::{
    StreamingInvalidationCauseV1, StreamingSessionLedgerV1,
    StreamingSessionMutationReceiptV1,
};

const MAX_PROJECTIONS_V1: usize = 1_000_000;
const MAX_PROJECTION_ID_BYTES_V1: usize = 512;
const MAX_DEPENDENCIES_V1: usize = 100_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DerivedProjectionKindV1 {
    Summary,
    Embedding,
    PreferenceModel,
    Plan,
    FactIndex,
    AnswerCache,
    Other,
}

impl DerivedProjectionKindV1 {
    fn tag(self) -> u8 {
        match self {
            Self::Summary => 1,
            Self::Embedding => 2,
            Self::PreferenceModel => 3,
            Self::Plan => 4,
            Self::FactIndex => 5,
            Self::AnswerCache => 6,
            Self::Other => 7,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DerivedProjectionStatusV1 {
    Current,
    Stale,
    Quarantined,
    Revoked,
    ErasureRequested,
    Rebuilt,
    Erased,
}

impl DerivedProjectionStatusV1 {
    fn tag(self) -> u8 {
        match self {
            Self::Current => 1,
            Self::Stale => 2,
            Self::Quarantined => 3,
            Self::Revoked => 4,
            Self::ErasureRequested => 5,
            Self::Rebuilt => 6,
            Self::Erased => 7,
        }
    }

    fn severity(self) -> u8 {
        match self {
            Self::Current | Self::Rebuilt => 0,
            Self::Stale => 1,
            Self::Quarantined => 2,
            Self::Revoked => 3,
            Self::ErasureRequested => 4,
            Self::Erased => 5,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TemporalProjectionRejectV1 {
    InvalidScope,
    InvalidProjectionId,
    InvalidDigest,
    EmptyDependencies,
    TooManyDependencies,
    DuplicateDependency,
    UnknownSourceEvent,
    SourceEventNotActive,
    UnknownParentProjection,
    ParentProjectionNotCurrent,
    DuplicateProjectionId,
    ProjectionLimitExceeded,
    UnknownProjection,
    ProjectionNotInvalidated,
    InvalidTransitionReceipt,
    WrongSession,
    SequenceGap { expected: u64, actual: u64 },
    EventIdentityConflict,
    InvalidInvalidationCause,
    InvalidReplacement,
    ReplacementAlreadyLinked,
    AlreadyErased,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DerivedProjectionSpecV1 {
    pub projection_id: String,
    pub kind: DerivedProjectionKindV1,
    pub content_sha256: String,
    pub derivation_receipt_sha256: String,
    /// Exact message event IDs consumed by this projection; sorted and unique.
    pub source_event_ids: Vec<String>,
    /// Exact upstream projection IDs consumed by this projection; sorted and unique.
    pub parent_projection_ids: Vec<String>,
    /// Must equal the source session ledger's last committed sequence.
    pub built_at_sequence: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DerivedMemoryProjectionV1 {
    projection_id: String,
    kind: DerivedProjectionKindV1,
    content_sha256: Option<String>,
    derivation_receipt_sha256: String,
    source_event_ids: Vec<String>,
    parent_projection_ids: Vec<String>,
    built_at_sequence: u64,
    status: DerivedProjectionStatusV1,
    invalidated_by_event_id: Option<String>,
    invalidated_at_sequence: Option<u64>,
    invalidation_receipt_sha256: Option<String>,
    replacement_projection_id: Option<String>,
    replacement_receipt_sha256: Option<String>,
    erasure_receipt_sha256: Option<String>,
}

impl DerivedMemoryProjectionV1 {
    pub fn projection_id(&self) -> &str { &self.projection_id }
    pub fn kind(&self) -> DerivedProjectionKindV1 { self.kind }
    pub fn content_sha256(&self) -> Option<&str> { self.content_sha256.as_deref() }
    pub fn derivation_receipt_sha256(&self) -> &str { &self.derivation_receipt_sha256 }
    pub fn source_event_ids(&self) -> &[String] { &self.source_event_ids }
    pub fn parent_projection_ids(&self) -> &[String] { &self.parent_projection_ids }
    pub fn built_at_sequence(&self) -> u64 { self.built_at_sequence }
    pub fn status(&self) -> DerivedProjectionStatusV1 { self.status }
    pub fn invalidation_receipt_sha256(&self) -> Option<&str> {
        self.invalidation_receipt_sha256.as_deref()
    }
    pub fn replacement_projection_id(&self) -> Option<&str> {
        self.replacement_projection_id.as_deref()
    }
    pub fn erasure_receipt_sha256(&self) -> Option<&str> {
        self.erasure_receipt_sha256.as_deref()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectionInvalidationEntryV1 {
    pub projection_id: String,
    pub previous_status: DerivedProjectionStatusV1,
    pub resulting_status: DerivedProjectionStatusV1,
    pub cause: StreamingInvalidationCauseV1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectionInvalidationReceiptV1 {
    pub user_id: String,
    pub session_id: String,
    pub mutation_event_id: String,
    pub mutation_sequence: u64,
    pub source_transition_receipt_sha256: String,
    pub previous_index_sha256: String,
    pub resulting_index_sha256: String,
    pub invalidated_projection_ids: Vec<String>,
    pub entries: Vec<ProjectionInvalidationEntryV1>,
    pub receipt_sha256: String,
}

impl ProjectionInvalidationReceiptV1 {
    pub fn verify_integrity(&self) -> bool {
        self.invalidated_projection_ids == self.entries.iter()
            .map(|entry| entry.projection_id.clone()).collect::<Vec<_>>()
            && self.receipt_sha256 == projection_invalidation_digest(
                &self.user_id,
                &self.session_id,
                &self.mutation_event_id,
                self.mutation_sequence,
                &self.source_transition_receipt_sha256,
                &self.previous_index_sha256,
                &self.resulting_index_sha256,
                &self.entries,
            )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProjectionTransitionStatusV1 {
    Applied,
    IdempotentReplay,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectionTransitionResultV1 {
    pub status: ProjectionTransitionStatusV1,
    pub receipt: ProjectionInvalidationReceiptV1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProjectionRegisterStatusV1 {
    Created,
    IdempotentReplay,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DerivedProjectionIndexV1 {
    user_id: String,
    session_id: String,
    projections: BTreeMap<String, DerivedMemoryProjectionV1>,
    applied_mutations: BTreeMap<String, String>,
    transition_receipts: BTreeMap<String, ProjectionInvalidationReceiptV1>,
    last_sequence: u64,
}

impl DerivedProjectionIndexV1 {
    pub fn new(
        user_id: impl Into<String>,
        session_id: impl Into<String>,
    ) -> Result<Self, TemporalProjectionRejectV1> {
        let user_id = user_id.into();
        let session_id = session_id.into();
        if !valid_atom(&user_id, 256) || !valid_atom(&session_id, 256) {
            return Err(TemporalProjectionRejectV1::InvalidScope);
        }
        Ok(Self {
            user_id,
            session_id,
            projections: BTreeMap::new(),
            applied_mutations: BTreeMap::new(),
            transition_receipts: BTreeMap::new(),
            last_sequence: 0,
        })
    }

    pub fn user_id(&self) -> &str { &self.user_id }
    pub fn session_id(&self) -> &str { &self.session_id }
    pub fn last_sequence(&self) -> u64 { self.last_sequence }
    pub fn len(&self) -> usize { self.projections.len() }
    pub fn is_empty(&self) -> bool { self.projections.is_empty() }

    pub fn projection(&self, projection_id: &str) -> Option<&DerivedMemoryProjectionV1> {
        self.projections.get(projection_id)
    }

    pub fn projections(&self) -> impl Iterator<Item = &DerivedMemoryProjectionV1> {
        self.projections.values()
    }

    pub fn transition_receipt(&self, event_id: &str) -> Option<&ProjectionInvalidationReceiptV1> {
        self.transition_receipts.get(event_id)
    }

    pub fn current_projection_ids(&self) -> Vec<String> {
        self.projections.values()
            .filter(|item| item.status == DerivedProjectionStatusV1::Current)
            .map(|item| item.projection_id.clone())
            .collect()
    }

    pub fn stale_projection_ids(&self) -> Vec<String> {
        self.projections.values()
            .filter(|item| matches!(
                item.status,
                DerivedProjectionStatusV1::Stale
                    | DerivedProjectionStatusV1::Quarantined
                    | DerivedProjectionStatusV1::Revoked
                    | DerivedProjectionStatusV1::ErasureRequested
            ))
            .map(|item| item.projection_id.clone())
            .collect()
    }

    /// Register a new immutable derived artifact against one exact session snapshot.
    /// Rebuilds receive new projection IDs; old projections remain in the audit
    /// history with an explicit stale/rebuilt/revoked/erased status.
    pub fn register_projection(
        &mut self,
        ledger: &StreamingSessionLedgerV1,
        spec: DerivedProjectionSpecV1,
    ) -> Result<ProjectionRegisterStatusV1, TemporalProjectionRejectV1> {
        if ledger.user_id() != self.user_id || ledger.session_id() != self.session_id {
            return Err(TemporalProjectionRejectV1::WrongSession);
        }
        validate_projection_spec(&spec)?;
        if self.projections.contains_key(&spec.projection_id) {
            let existing = self.projections.get(&spec.projection_id)
                .ok_or(TemporalProjectionRejectV1::DuplicateProjectionId)?;
            if existing.kind == spec.kind
                && existing.content_sha256.as_deref() == Some(spec.content_sha256.as_str())
                && existing.derivation_receipt_sha256 == spec.derivation_receipt_sha256
                && existing.source_event_ids == spec.source_event_ids
                && existing.parent_projection_ids == spec.parent_projection_ids
                && existing.built_at_sequence == spec.built_at_sequence
            {
                return Ok(ProjectionRegisterStatusV1::IdempotentReplay);
            }
            return Err(TemporalProjectionRejectV1::DuplicateProjectionId);
        }
        if self.projections.len() >= MAX_PROJECTIONS_V1 {
            return Err(TemporalProjectionRejectV1::ProjectionLimitExceeded);
        }
        if spec.built_at_sequence != ledger.last_sequence() {
            return Err(TemporalProjectionRejectV1::SequenceGap {
                expected: ledger.last_sequence(),
                actual: spec.built_at_sequence,
            });
        }
        if self.projections.is_empty() && self.last_sequence == 0 {
            // Bootstrap an index from the exact current session snapshot.
            // Later gaps cannot be skipped because they may contain invalidations.
            self.last_sequence = ledger.last_sequence();
        } else if self.last_sequence != ledger.last_sequence() {
            return Err(TemporalProjectionRejectV1::SequenceGap {
                expected: self.last_sequence,
                actual: ledger.last_sequence(),
            });
        }

        let active_events = ledger.snapshot().messages.into_iter()
            .map(|item| item.event_id).collect::<BTreeSet<_>>();
        for event_id in &spec.source_event_ids {
            if !active_events.contains(event_id) {
                return Err(TemporalProjectionRejectV1::SourceEventNotActive);
            }
        }

        for parent_id in &spec.parent_projection_ids {
            let parent = self.projections.get(parent_id)
                .ok_or(TemporalProjectionRejectV1::UnknownParentProjection)?;
            if parent.status != DerivedProjectionStatusV1::Current {
                return Err(TemporalProjectionRejectV1::ParentProjectionNotCurrent);
            }
        }

        self.projections.insert(spec.projection_id.clone(), DerivedMemoryProjectionV1 {
            projection_id: spec.projection_id,
            kind: spec.kind,
            content_sha256: Some(spec.content_sha256),
            derivation_receipt_sha256: spec.derivation_receipt_sha256,
            source_event_ids: spec.source_event_ids,
            parent_projection_ids: spec.parent_projection_ids,
            built_at_sequence: spec.built_at_sequence,
            status: DerivedProjectionStatusV1::Current,
            invalidated_by_event_id: None,
            invalidated_at_sequence: None,
            invalidation_receipt_sha256: None,
            replacement_projection_id: None,
            replacement_receipt_sha256: None,
            erasure_receipt_sha256: None,
        });
        Ok(ProjectionRegisterStatusV1::Created)
    }

    /// Apply the invalidation effect of one verified session transition. The
    /// source receipt's invalidated_event_ids seed the affected projections;
    /// invalidation then propagates transitively through derived-on-derived
    /// dependencies. No downstream artifact is silently kept current.
    pub fn apply_transition(
        &mut self,
        transition: &StreamingSessionMutationReceiptV1,
    ) -> Result<ProjectionTransitionResultV1, TemporalProjectionRejectV1> {
        if !transition.verify_integrity() {
            return Err(TemporalProjectionRejectV1::InvalidTransitionReceipt);
        }
        if transition.user_id != self.user_id || transition.session_id != self.session_id {
            return Err(TemporalProjectionRejectV1::WrongSession);
        }
        let receipt_sha256 = transition.receipt_sha256.clone();
        if let Some(previous) = self.applied_mutations.get(&transition.event_id) {
            if previous != &receipt_sha256 {
                return Err(TemporalProjectionRejectV1::EventIdentityConflict);
            }
            let receipt = self.transition_receipts.get(&transition.event_id)
                .cloned().ok_or(TemporalProjectionRejectV1::EventIdentityConflict)?;
            return Ok(ProjectionTransitionResultV1 {
                status: ProjectionTransitionStatusV1::IdempotentReplay,
                receipt,
            });
        }
        let expected = self.last_sequence.checked_add(1)
            .ok_or(TemporalProjectionRejectV1::ProjectionLimitExceeded)?;
        if transition.sequence != expected {
            return Err(TemporalProjectionRejectV1::SequenceGap {
                expected,
                actual: transition.sequence,
            });
        }
        if !transition.invalidated_event_ids.is_empty() && transition.invalidation_cause.is_none() {
            return Err(TemporalProjectionRejectV1::InvalidInvalidationCause);
        }

        let mut staged = self.clone();
        let receipt = staged.apply_transition_inner(transition)?;
        *self = staged;
        Ok(ProjectionTransitionResultV1 {
            status: ProjectionTransitionStatusV1::Applied,
            receipt,
        })
    }

    fn apply_transition_inner(
        &mut self,
        transition: &StreamingSessionMutationReceiptV1,
    ) -> Result<ProjectionInvalidationReceiptV1, TemporalProjectionRejectV1> {
        let previous_index_sha256 = self.state_sha256();
        let cause = transition.invalidation_cause;
        let mut pending = VecDeque::<(String, StreamingInvalidationCauseV1)>::new();

        if let Some(cause) = cause {
            let invalidated = transition.invalidated_event_ids.iter().collect::<BTreeSet<_>>();
            for projection in self.projections.values() {
                if projection.source_event_ids.iter().any(|id| invalidated.contains(id)) {
                    pending.push_back((projection.projection_id.clone(), cause));
                }
            }
        }

        let mut entries = BTreeMap::<String, ProjectionInvalidationEntryV1>::new();
        while let Some((projection_id, incoming_cause)) = pending.pop_front() {
            let projection = self.projections.get_mut(&projection_id)
                .ok_or(TemporalProjectionRejectV1::UnknownProjection)?;
            let desired_status = status_for_cause(incoming_cause);
            if projection.status == DerivedProjectionStatusV1::Current
                || desired_status.severity() > projection.status.severity()
            {
                let previous_status = projection.status;
                projection.status = desired_status;
                projection.invalidated_by_event_id = Some(transition.event_id.clone());
                projection.invalidated_at_sequence = Some(transition.sequence);
                projection.invalidation_receipt_sha256 = Some(transition.receipt_sha256.clone());
                entries.insert(projection_id.clone(), ProjectionInvalidationEntryV1 {
                    projection_id: projection_id.clone(),
                    previous_status,
                    resulting_status: desired_status,
                    cause: incoming_cause,
                });

                let descendants = self.projections.values()
                    .filter(|child| child.parent_projection_ids.contains(&projection_id))
                    .map(|child| child.projection_id.clone())
                    .collect::<Vec<_>>();
                for child_id in descendants {
                    pending.push_back((child_id, incoming_cause));
                }
            }
        }

        let resulting_index_sha256 = self.state_sha256();
        let entries = entries.into_values().collect::<Vec<_>>();
        let invalidated_projection_ids = entries.iter()
            .map(|entry| entry.projection_id.clone())
            .collect::<Vec<_>>();
        let receipt_sha256 = projection_invalidation_digest(
            &self.user_id,
            &self.session_id,
            &transition.event_id,
            transition.sequence,
            &transition.receipt_sha256,
            &previous_index_sha256,
            &resulting_index_sha256,
            &entries,
        );
        let receipt = ProjectionInvalidationReceiptV1 {
            user_id: self.user_id.clone(),
            session_id: self.session_id.clone(),
            mutation_event_id: transition.event_id.clone(),
            mutation_sequence: transition.sequence,
            source_transition_receipt_sha256: transition.receipt_sha256.clone(),
            previous_index_sha256,
            resulting_index_sha256,
            invalidated_projection_ids,
            entries,
            receipt_sha256,
        };
        self.applied_mutations.insert(transition.event_id.clone(), transition.receipt_sha256.clone());
        self.transition_receipts.insert(transition.event_id.clone(), receipt.clone());
        self.last_sequence = transition.sequence;
        Ok(receipt)
    }

    /// Link an old stale/quarantined projection to a newly built current
    /// replacement. The replacement must have been registered against the
    /// latest session sequence and current dependencies.
    pub fn link_replacement(
        &mut self,
        stale_projection_id: &str,
        replacement_projection_id: &str,
        rebuild_receipt_sha256: &str,
    ) -> Result<(), TemporalProjectionRejectV1> {
        validate_digest(rebuild_receipt_sha256)?;
        let replacement = self.projections.get(replacement_projection_id)
            .ok_or(TemporalProjectionRejectV1::InvalidReplacement)?;
        if replacement.status != DerivedProjectionStatusV1::Current
            || replacement_projection_id == stale_projection_id
        {
            return Err(TemporalProjectionRejectV1::InvalidReplacement);
        }
        let stale = self.projections.get_mut(stale_projection_id)
            .ok_or(TemporalProjectionRejectV1::UnknownProjection)?;
        if !matches!(
            stale.status,
            DerivedProjectionStatusV1::Stale | DerivedProjectionStatusV1::Quarantined
        ) || stale.replacement_projection_id.is_some()
        {
            return Err(TemporalProjectionRejectV1::ReplacementAlreadyLinked);
        }
        stale.status = DerivedProjectionStatusV1::Rebuilt;
        stale.replacement_projection_id = Some(replacement_projection_id.to_owned());
        stale.replacement_receipt_sha256 = Some(rebuild_receipt_sha256.to_owned());
        Ok(())
    }

    /// Record an independent deletion acknowledgement for one projection
    /// whose source dependency received a rights-erasure transition. This
    /// updates the in-memory index only; the caller remains responsible for
    /// deleting external bytes/indexes/replicas and obtaining this receipt
    /// from the responsible storage operator.
    pub fn confirm_erasure(
        &mut self,
        projection_id: &str,
        deletion_receipt_sha256: &str,
    ) -> Result<(), TemporalProjectionRejectV1> {
        validate_digest(deletion_receipt_sha256)?;
        let projection = self.projections.get_mut(projection_id)
            .ok_or(TemporalProjectionRejectV1::UnknownProjection)?;
        if projection.status != DerivedProjectionStatusV1::ErasureRequested {
            return Err(TemporalProjectionRejectV1::AlreadyErased);
        }
        projection.content_sha256 = None;
        projection.source_event_ids.clear();
        projection.parent_projection_ids.clear();
        projection.derivation_receipt_sha256.clear();
        projection.status = DerivedProjectionStatusV1::Erased;
        projection.erasure_receipt_sha256 = Some(deletion_receipt_sha256.to_owned());
        Ok(())
    }

    fn state_sha256(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"luminous:mace:derived-projection-index:v1\0");
        hash_field(&mut h, &self.user_id);
        hash_field(&mut h, &self.session_id);
        h.update(self.last_sequence.to_be_bytes());
        h.update((self.projections.len() as u64).to_be_bytes());
        for projection in self.projections.values() {
            hash_field(&mut h, &projection.projection_id);
            h.update([projection.kind.tag(), projection.status.tag()]);
            match &projection.content_sha256 {
                Some(digest) => { h.update([1]); hash_field(&mut h, digest); }
                None => h.update([0]),
            }
            hash_field(&mut h, &projection.derivation_receipt_sha256);
            h.update(projection.built_at_sequence.to_be_bytes());
            hash_strings(&mut h, &projection.source_event_ids);
            hash_strings(&mut h, &projection.parent_projection_ids);
            hash_optional(&mut h, projection.invalidated_by_event_id.as_deref());
            hash_optional(&mut h, projection.invalidated_at_sequence.map(|v| v.to_string()).as_deref());
            hash_optional(&mut h, projection.invalidation_receipt_sha256.as_deref());
            hash_optional(&mut h, projection.replacement_projection_id.as_deref());
            hash_optional(&mut h, projection.replacement_receipt_sha256.as_deref());
            hash_optional(&mut h, projection.erasure_receipt_sha256.as_deref());
        }
        hex_digest(&h.finalize())
    }
}

fn validate_projection_spec(spec: &DerivedProjectionSpecV1) -> Result<(), TemporalProjectionRejectV1> {
    if !valid_atom(&spec.projection_id, MAX_PROJECTION_ID_BYTES_V1) {
        return Err(TemporalProjectionRejectV1::InvalidProjectionId);
    }
    validate_digest(&spec.content_sha256)?;
    validate_digest(&spec.derivation_receipt_sha256)?;
    if spec.source_event_ids.is_empty() && spec.parent_projection_ids.is_empty() {
        return Err(TemporalProjectionRejectV1::EmptyDependencies);
    }
    if spec.source_event_ids.len() > MAX_DEPENDENCIES_V1
        || spec.parent_projection_ids.len() > MAX_DEPENDENCIES_V1
    {
        return Err(TemporalProjectionRejectV1::TooManyDependencies);
    }
    if !strict_sorted_unique(&spec.source_event_ids)
        || !strict_sorted_unique(&spec.parent_projection_ids)
    {
        return Err(TemporalProjectionRejectV1::DuplicateDependency);
    }
    for id in spec.source_event_ids.iter().chain(spec.parent_projection_ids.iter()) {
        if !valid_atom(id, MAX_PROJECTION_ID_BYTES_V1) {
            return Err(TemporalProjectionRejectV1::InvalidProjectionId);
        }
    }
    Ok(())
}

fn status_for_cause(cause: StreamingInvalidationCauseV1) -> DerivedProjectionStatusV1 {
    match cause {
        StreamingInvalidationCauseV1::Correction => DerivedProjectionStatusV1::Stale,
        StreamingInvalidationCauseV1::Quarantine => DerivedProjectionStatusV1::Quarantined,
        StreamingInvalidationCauseV1::Revocation => DerivedProjectionStatusV1::Revoked,
        StreamingInvalidationCauseV1::RightsErasure => DerivedProjectionStatusV1::ErasureRequested,
    }
}

fn strict_sorted_unique(values: &[String]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn valid_atom(value: &str, maximum: usize) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && value.len() <= maximum
        && !value.chars().any(char::is_control)
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_digest(value: &str) -> Result<(), TemporalProjectionRejectV1> {
    if valid_digest(value) { Ok(()) } else { Err(TemporalProjectionRejectV1::InvalidDigest) }
}

fn hash_strings(hasher: &mut Sha256, values: &[String]) {
    hasher.update((values.len() as u64).to_be_bytes());
    for value in values { hash_field(hasher, value); }
}

fn hash_optional(hasher: &mut Sha256, value: Option<&str>) {
    match value {
        Some(value) => { hasher.update([1]); hash_field(hasher, value); }
        None => hasher.update([0]),
    }
}

fn hash_field(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn projection_invalidation_digest(
    user_id: &str,
    session_id: &str,
    mutation_event_id: &str,
    sequence: u64,
    source_transition_receipt_sha256: &str,
    previous_index_sha256: &str,
    resulting_index_sha256: &str,
    entries: &[ProjectionInvalidationEntryV1],
) -> String {
    let mut h = Sha256::new();
    h.update(b"luminous:mace:projection-invalidation-receipt:v1\0");
    hash_field(&mut h, user_id);
    hash_field(&mut h, session_id);
    hash_field(&mut h, mutation_event_id);
    h.update(sequence.to_be_bytes());
    hash_field(&mut h, source_transition_receipt_sha256);
    hash_field(&mut h, previous_index_sha256);
    hash_field(&mut h, resulting_index_sha256);
    h.update((entries.len() as u64).to_be_bytes());
    for entry in entries {
        hash_field(&mut h, &entry.projection_id);
        h.update([entry.previous_status.tag(), entry.resulting_status.tag(), entry.cause.tag()]);
    }
    hex_digest(&h.finalize())
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw_session::RawSessionMessageV1;
    use crate::streaming_session::{
        StreamingMutationKindV1, StreamingSessionMutationV1,
    };
    use crate::Role;

    fn message(turn_id: &str, timestamp_ms: i64, text: &str) -> RawSessionMessageV1 {
        RawSessionMessageV1 {
            turn_id: turn_id.to_owned(),
            role: Role::User,
            timestamp_ms,
            text: text.to_owned(),
            raw_line_sha256: "a".repeat(64),
        }
    }

    fn append(event_id: &str, sequence: u64, timestamp_ms: i64, text: &str) -> StreamingSessionMutationV1 {
        StreamingSessionMutationV1 {
            event_id: event_id.to_owned(),
            sequence,
            effective_at_ms: timestamp_ms,
            kind: StreamingMutationKindV1::Append {
                message: message(&format!("turn-{sequence}"), timestamp_ms, text),
            },
        }
    }

    fn ledger() -> StreamingSessionLedgerV1 {
        let mut ledger = StreamingSessionLedgerV1::new("user-1", "session-1").unwrap();
        ledger.apply(append("event-1", 1, 100, "source fact"), 1_000).unwrap();
        ledger
    }

    fn spec(
        id: &str,
        seq: u64,
        sources: &[&str],
        parents: &[&str],
    ) -> DerivedProjectionSpecV1 {
        DerivedProjectionSpecV1 {
            projection_id: id.to_owned(),
            kind: DerivedProjectionKindV1::Summary,
            content_sha256: "b".repeat(64),
            derivation_receipt_sha256: "c".repeat(64),
            source_event_ids: sources.iter().map(|v| (*v).to_owned()).collect(),
            parent_projection_ids: parents.iter().map(|v| (*v).to_owned()).collect(),
            built_at_sequence: seq,
        }
    }

    fn correction(sequence: u64) -> StreamingSessionMutationV1 {
        StreamingSessionMutationV1 {
            event_id: format!("event-{sequence}"),
            sequence,
            effective_at_ms: 100,
            kind: StreamingMutationKindV1::Correct {
                target_event_id: "event-1".into(),
                message: message("turn-1", 100, "corrected source fact"),
            },
        }
    }

    #[test]
    fn projection_index_must_not_skip_unprocessed_session_sequences() {
        let mut ledger = ledger();
        let mut index = DerivedProjectionIndexV1::new("user-1", "session-1").unwrap();
        // Registering the first projection bootstraps the index at sequence one.
        index.register_projection(&ledger, spec("summary-v1", 1, &["event-1"], &[])).unwrap();
        ledger.apply(append("event-2", 2, 200, "new fact"), 1_100).unwrap();
        let result = index.register_projection(&ledger, spec("summary-v2", 2, &["event-2"], &[]));
        assert_eq!(
            result,
            Err(TemporalProjectionRejectV1::SequenceGap { expected: 1, actual: 2 })
        );
    }

    #[test]
    fn projection_invalidation_receipt_is_self_committing() {
        let mut ledger = ledger();
        let mut index = DerivedProjectionIndexV1::new("user-1", "session-1").unwrap();
        index.register_projection(&ledger, spec("summary-v1", 1, &["event-1"], &[])).unwrap();
        let transition = ledger.apply(correction(2), 1_100).unwrap().receipt;
        let result = index.apply_transition(&transition).unwrap();
        assert!(result.receipt.verify_integrity());
        let mut altered = result.receipt.clone();
        altered.invalidated_projection_ids.clear();
        assert!(!altered.verify_integrity());
    }

    #[test]
    fn source_correction_invalidates_dependent_projection_and_cascades_to_plan() {
        let mut ledger = ledger();
        let mut index = DerivedProjectionIndexV1::new("user-1", "session-1").unwrap();
        assert_eq!(
            index.register_projection(&ledger, spec("summary-v1", 1, &["event-1"], &[])),
            Ok(ProjectionRegisterStatusV1::Created)
        );
        assert_eq!(
            index.register_projection(&ledger, spec("plan-v1", 1, &[], &["summary-v1"])),
            Ok(ProjectionRegisterStatusV1::Created)
        );
        let transition = ledger.apply(correction(2), 1_100).unwrap().receipt;
        let result = index.apply_transition(&transition).unwrap();
        assert_eq!(result.receipt.invalidated_projection_ids, vec!["plan-v1", "summary-v1"]);
        assert_eq!(index.projection("summary-v1").unwrap().status(), DerivedProjectionStatusV1::Stale);
        assert_eq!(index.projection("plan-v1").unwrap().status(), DerivedProjectionStatusV1::Stale);
    }

    #[test]
    fn unrelated_source_event_does_not_invalidate_projection() {
        let mut ledger = ledger();
        let mut index = DerivedProjectionIndexV1::new("user-1", "session-1").unwrap();
        index.register_projection(&ledger, spec("summary-v1", 1, &["event-1"], &[])).unwrap();
        let unrelated = ledger.apply(append("event-2", 2, 200, "unrelated"), 1_100).unwrap().receipt;
        let result = index.apply_transition(&unrelated).unwrap();
        assert!(result.receipt.invalidated_projection_ids.is_empty());
        assert_eq!(index.projection("summary-v1").unwrap().status(), DerivedProjectionStatusV1::Current);
    }

    #[test]
    fn quarantine_invalidates_projection_and_revalidation_does_not_restore_it() {
        let mut ledger = ledger();
        let mut index = DerivedProjectionIndexV1::new("user-1", "session-1").unwrap();
        index.register_projection(&ledger, spec("summary-v1", 1, &["event-1"], &[])).unwrap();
        let quarantine = StreamingSessionMutationV1 {
            event_id: "event-2".into(),
            sequence: 2,
            effective_at_ms: 200,
            kind: StreamingMutationKindV1::Quarantine {
                target_event_id: "event-1".into(),
                reason_sha256: "d".repeat(64),
            },
        };
        let receipt = ledger.apply(quarantine, 1_100).unwrap().receipt;
        index.apply_transition(&receipt).unwrap();
        let revalidate = StreamingSessionMutationV1 {
            event_id: "event-3".into(),
            sequence: 3,
            effective_at_ms: 300,
            kind: StreamingMutationKindV1::Revalidate {
                target_event_id: "event-1".into(),
                validation_receipt_sha256: "e".repeat(64),
            },
        };
        let receipt = ledger.apply(revalidate, 1_200).unwrap().receipt;
        index.apply_transition(&receipt).unwrap();
        assert_eq!(index.projection("summary-v1").unwrap().status(), DerivedProjectionStatusV1::Quarantined);
    }

    #[test]
    fn rebuilt_projection_requires_new_active_dependencies_and_receipt() {
        let mut ledger = ledger();
        let mut index = DerivedProjectionIndexV1::new("user-1", "session-1").unwrap();
        index.register_projection(&ledger, spec("summary-v1", 1, &["event-1"], &[])).unwrap();
        let transition = ledger.apply(correction(2), 1_100).unwrap().receipt;
        index.apply_transition(&transition).unwrap();
        assert_eq!(
            index.register_projection(&ledger, spec("summary-v2", 2, &["event-2"], &[])),
            Ok(ProjectionRegisterStatusV1::Created)
        );
        index.link_replacement("summary-v1", "summary-v2", &"f".repeat(64)).unwrap();
        assert_eq!(index.projection("summary-v1").unwrap().status(), DerivedProjectionStatusV1::Rebuilt);
        assert_eq!(index.projection("summary-v1").unwrap().replacement_projection_id(), Some("summary-v2"));
        assert_eq!(index.projection("summary-v2").unwrap().status(), DerivedProjectionStatusV1::Current);
    }

    #[test]
    fn source_revocation_and_erasure_propagate_to_all_descendants() {
        let mut ledger = ledger();
        let mut index = DerivedProjectionIndexV1::new("user-1", "session-1").unwrap();
        index.register_projection(&ledger, spec("summary-v1", 1, &["event-1"], &[])).unwrap();
        index.register_projection(&ledger, spec("answer-cache-v1", 1, &[], &["summary-v1"])).unwrap();
        let revoke = StreamingSessionMutationV1 {
            event_id: "event-2".into(),
            sequence: 2,
            effective_at_ms: 200,
            kind: StreamingMutationKindV1::Revoke {
                target_event_id: "event-1".into(),
                revocation_receipt_sha256: "d".repeat(64),
            },
        };
        let receipt = ledger.apply(revoke, 1_100).unwrap().receipt;
        index.apply_transition(&receipt).unwrap();
        assert_eq!(index.projection("summary-v1").unwrap().status(), DerivedProjectionStatusV1::Revoked);
        assert_eq!(index.projection("answer-cache-v1").unwrap().status(), DerivedProjectionStatusV1::Revoked);

        // New ledger fixture because revocation is terminal; erasure is tested from
        // an independently constructed active source rather than after revocation.
        let mut ledger = StreamingSessionLedgerV1::new("user-1", "session-2").unwrap();
        ledger.apply(append("source-1", 1, 100, "private source"), 1_000).unwrap();
        let mut index = DerivedProjectionIndexV1::new("user-1", "session-2").unwrap();
        index.register_projection(&ledger, spec("embedding-v1", 1, &["source-1"], &[])).unwrap();
        let erase = StreamingSessionMutationV1 {
            event_id: "erase-1".into(),
            sequence: 2,
            effective_at_ms: 200,
            kind: StreamingMutationKindV1::ErasureRequest {
                target_event_id: "source-1".into(),
                rights_receipt_sha256: "e".repeat(64),
            },
        };
        let receipt = ledger.apply(erase, 1_100).unwrap().receipt;
        index.apply_transition(&receipt).unwrap();
        assert_eq!(index.projection("embedding-v1").unwrap().status(), DerivedProjectionStatusV1::ErasureRequested);
        index.confirm_erasure("embedding-v1", &"f".repeat(64)).unwrap();
        assert_eq!(index.projection("embedding-v1").unwrap().status(), DerivedProjectionStatusV1::Erased);
        assert_eq!(index.projection("embedding-v1").unwrap().content_sha256(), None);
    }

    #[test]
    fn projection_scope_and_transition_receipt_cannot_be_crossed() {
        let ledger = ledger();
        let mut index = DerivedProjectionIndexV1::new("different-user", "session-1").unwrap();
        assert_eq!(
            index.register_projection(&ledger, spec("summary-v1", 1, &["event-1"], &[])),
            Err(TemporalProjectionRejectV1::WrongSession)
        );
    }

    #[test]
    fn invalidation_receipt_replay_is_idempotent_and_conflicting_rebind_fails() {
        let mut ledger = ledger();
        let mut index = DerivedProjectionIndexV1::new("user-1", "session-1").unwrap();
        index.register_projection(&ledger, spec("summary-v1", 1, &["event-1"], &[])).unwrap();
        let transition = ledger.apply(correction(2), 1_100).unwrap().receipt;
        let first = index.apply_transition(&transition).unwrap();
        let replay = index.apply_transition(&transition).unwrap();
        assert_eq!(replay.status, ProjectionTransitionStatusV1::IdempotentReplay);
        assert_eq!(replay.receipt, first.receipt);
        let mut altered = transition.clone();
        altered.session_id = "other".into();
        assert_eq!(index.apply_transition(&altered), Err(TemporalProjectionRejectV1::InvalidTransitionReceipt));
    }
}
