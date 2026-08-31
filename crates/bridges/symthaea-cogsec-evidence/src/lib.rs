// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shadow-mode evidence substrate for CogSec.
//!
//! This crate is deliberately outside the small `symthaea-cogsec` reference-
//! monitor TCB. It connects monitor evaluation records to a typed, bounded,
//! causal event stream and reconciles that stream against the generic
//! `symthaea-evidence-plane` mechanism counters.
//!
//! The central separation is:
//!
//! - mechanism counters answer **whether/how often a hook fired**;
//! - typed events answer **which exact transition was evaluated/observed**;
//! - neither portable representation is authority by itself.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_cogsec::{
    DecisionOutcome, Digest32, MutationReceiptRecord, ReceiptStage, ResourceId,
};
use symthaea_evidence_plane::{EvidenceCounters, RunId};
use thiserror::Error;

/// Schema version for the first typed shadow-event contract.
pub const SHADOW_EVENT_SCHEMA_V1: u16 = 1;

/// Evidence-plane counter: number of CogSec policy evaluations observed.
pub const COUNTER_MONITOR_INVOCATIONS: &str = "cogsec_monitor_invocations";
/// Evidence-plane counter: shadow evaluations that would allow.
pub const COUNTER_WOULD_ALLOW: &str = "cogsec_audit_would_allow";
/// Evidence-plane counter: shadow evaluations that would deny.
pub const COUNTER_WOULD_DENY: &str = "cogsec_audit_would_deny";
/// Evidence-plane counter: shadow evaluations that would quarantine.
pub const COUNTER_WOULD_QUARANTINE: &str = "cogsec_audit_would_quarantine";
/// Evidence-plane counter: shadow evaluations that require authorization.
pub const COUNTER_WOULD_REQUIRE_AUTHORIZATION: &str =
    "cogsec_audit_would_require_authorization";
/// Evidence-plane counter: shadow evaluations that require revalidation.
pub const COUNTER_WOULD_REQUIRE_REVALIDATION: &str =
    "cogsec_audit_would_require_revalidation";
/// Evidence-plane counter: shadow evaluations that defer.
pub const COUNTER_WOULD_DEFER: &str = "cogsec_audit_would_defer";
/// Evidence-plane counter: legacy mutations that proceeded after a non-Allow result.
pub const COUNTER_LEGACY_COMMITS_AFTER_NON_ALLOW: &str =
    "legacy_commits_after_cogsec_non_allow";
/// Evidence-plane counter: scoped P0 mutation attempts.
pub const COUNTER_P0_MUTATION_ATTEMPTS: &str = "p0_mutation_attempts";
/// Evidence-plane counter: scoped P0 attempts with a matching CogSec evaluation.
pub const COUNTER_P0_MEDIATED_ATTEMPTS: &str = "p0_mediated_attempts";
/// Evidence-plane counter: scoped P0 commits without a matching evaluation.
pub const COUNTER_P0_UNMEDIATED_COMMITS: &str = "p0_unmediated_commits";
/// Evidence-plane counter: events the bounded shadow ledger could not retain.
pub const COUNTER_EVIDENCE_EVENTS_LOST: &str = "cogsec_evidence_events_lost";

const MAX_EXACT_F64_INTEGER: u64 = 1_u64 << 53;

/// Caller-visible correlation identity. This is not transaction authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ProposalId(pub Digest32);

/// Protected-owner transaction identity. Shadow-only flows may leave this absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TransactionId(pub Digest32);

/// Owner-issued identity for one append-only security event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EventId {
    /// Ledger epoch. Restart/continuity changes must use a new epoch.
    pub ledger_epoch: u64,
    /// Strictly increasing sequence within the ledger epoch.
    pub sequence: u64,
}

impl std::fmt::Display for EventId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.ledger_epoch, self.sequence)
    }
}

/// Cognitive tick used only for correlation, never authorization validity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CognitiveTick(pub u64);

/// Cheap owner-issued freshness token for one protected resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ResourceVersion {
    /// Owner/restart epoch.
    pub owner_epoch: u64,
    /// Monotonic mutation counter within the owner epoch.
    pub counter: u64,
}

impl ResourceVersion {
    /// Whether `self` is exactly one accepted mutation after `before`.
    pub fn is_successor_of(self, before: Self) -> bool {
        self.owner_epoch == before.owner_epoch
            && before.counter.checked_add(1).is_some_and(|next| self.counter == next)
    }
}

/// Executing agent/workload identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ActorId(pub String);

/// Human/organization principal whose delegated authority is being exercised.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AuthorityPrincipalId(pub String);

/// Owner/tenant/security domain controlling the protected resource.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ResourceOwnerId(pub String);

/// Optional information-source identity. Source identity is non-authoritative by default.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SourcePrincipalId(pub String);

/// Role-explicit identity context associated with an event.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrincipalContext {
    /// Concrete executing agent/workload.
    pub actor: Option<ActorId>,
    /// Human/organization principal on whose behalf the actor runs.
    pub authority_principal: Option<AuthorityPrincipalId>,
    /// Owner/tenant of the protected resource.
    pub resource_owner: Option<ResourceOwnerId>,
    /// Provenance/source identity, if known.
    pub source: Option<SourcePrincipalId>,
}

/// Confidentiality policy for the security event itself.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum EvidenceConfidentiality {
    /// Safe for public qualification export.
    Public,
    /// Restricted to an organization/tenant boundary.
    Tenant,
    /// Private to the local user/device by default.
    LocalPrivate,
    /// Highest restriction; explicit declassification is required.
    Restricted,
}

/// Coarse ingress provenance class for shadow observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IngressClass {
    /// Existing legacy API with no trusted CogSec classification.
    LegacyUnclassified,
    /// Locally authenticated/trusted adapter.
    TrustedLocal,
    /// Peer/social/mesh ingress.
    Peer,
    /// Web/research ingestion.
    WebResearch,
    /// Tool/service output returned into cognition.
    ToolOutput,
    /// Other explicitly named source class.
    Other(String),
}

/// Stage-specific shadow event kinds for the first runtime tranche.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum ShadowEventKind {
    /// Input was received/observed.
    IngressObserved,
    /// Working-memory admission was evaluated.
    WorkingMemoryAdmissionEvaluated,
    /// Legacy working-memory admission was observed.
    WorkingMemoryAdmissionObserved,
    /// Legacy working-memory eviction was observed.
    WorkingMemoryEvictionObserved,
    /// Persistent-memory graduation was evaluated.
    GraduationEvaluated,
    /// Legacy graduation/persistence was observed.
    GraduationObserved,
    /// Active Holocell/current-thought influence was evaluated.
    WorkingStateInfluenceEvaluated,
    /// Legacy active working-state influence was observed.
    WorkingStateInfluenceObserved,
    /// Goal activation was evaluated.
    GoalActivationEvaluated,
    /// Legacy goal activation was observed.
    GoalActivationObserved,
    /// Affective mutation was evaluated.
    AffectMutationEvaluated,
    /// Legacy affect mutation was observed.
    AffectMutationObserved,
    /// Dream-memory merge was evaluated.
    DreamMergeEvaluated,
    /// Legacy dream-memory merge was observed.
    DreamMergeObserved,
    /// Required/optional evidence was lost or continuity degraded.
    EvidenceGapObserved,
}

impl ShadowEventKind {
    /// Whether this event contains a CogSec monitor evaluation record.
    pub fn is_evaluation(self) -> bool {
        matches!(
            self,
            Self::WorkingMemoryAdmissionEvaluated
                | Self::GraduationEvaluated
                | Self::WorkingStateInfluenceEvaluated
                | Self::GoalActivationEvaluated
                | Self::AffectMutationEvaluated
                | Self::DreamMergeEvaluated
        )
    }

    /// Whether this event represents a legacy state mutation that should have an evaluation pair.
    pub fn is_paired_legacy_observation(self) -> bool {
        matches!(
            self,
            Self::WorkingMemoryAdmissionObserved
                | Self::GraduationObserved
                | Self::WorkingStateInfluenceObserved
                | Self::GoalActivationObserved
                | Self::AffectMutationObserved
                | Self::DreamMergeObserved
        )
    }

    /// Whether this event represents any observed legacy resource mutation.
    pub fn mutates_resource(self) -> bool {
        self.is_paired_legacy_observation() || self == Self::WorkingMemoryEvictionObserved
    }

    /// Evaluation kind expected as the direct causal parent of a paired observation.
    pub fn expected_evaluation(self) -> Option<Self> {
        match self {
            Self::WorkingMemoryAdmissionObserved => Some(Self::WorkingMemoryAdmissionEvaluated),
            Self::GraduationObserved => Some(Self::GraduationEvaluated),
            Self::WorkingStateInfluenceObserved => Some(Self::WorkingStateInfluenceEvaluated),
            Self::GoalActivationObserved => Some(Self::GoalActivationEvaluated),
            Self::AffectMutationObserved => Some(Self::AffectMutationEvaluated),
            Self::DreamMergeObserved => Some(Self::DreamMergeEvaluated),
            _ => None,
        }
    }
}

/// Stage-specific event payload. Raw cognitive content is intentionally absent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShadowEventPayload {
    /// Ingress classification only; payload bytes remain outside the security ledger.
    Ingress {
        /// Coarse ingress class.
        ingress_class: IngressClass,
    },
    /// Exact monitor-produced evaluation record exported from an opaque receipt.
    Evaluation {
        /// Portable monitor evaluation data. This remains unauthenticated data cross-boundary.
        receipt: MutationReceiptRecord,
    },
    /// Legacy runtime mutation observation.
    MutationObserved {
        /// Whether legacy code actually applied the state change.
        applied: bool,
    },
    /// Working-memory eviction observation, referenced by an opaque commitment.
    EvictionObserved {
        /// Commitment/reference for the evicted item; never raw memory content.
        evicted_item_ref: Digest32,
    },
    /// Explicit evidence loss/degradation observation.
    EvidenceGap {
        /// Event kind that could not be retained, when known.
        lost_kind: ShadowEventKind,
        /// Number of lost events represented by this gap event.
        lost_count: u64,
    },
}

/// Event proposal before the ledger owner allocates an `EventId`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShadowEventDraft {
    /// Stage/type of event.
    pub kind: ShadowEventKind,
    /// Caller-visible proposal correlation only.
    pub proposal_id: Option<ProposalId>,
    /// Protected-owner transaction identity, absent in early pure shadow mode.
    pub transaction_id: Option<TransactionId>,
    /// Role-explicit principal context.
    pub principals: PrincipalContext,
    /// Protected resource when the event concerns one.
    pub resource: Option<ResourceId>,
    /// Resource version before the observed/evaluated transition.
    pub resource_version_before: Option<ResourceVersion>,
    /// Resource version after the observed legacy mutation.
    pub resource_version_after: Option<ResourceVersion>,
    /// Optional cryptographic state commitment before the transition.
    pub state_root_before: Option<Digest32>,
    /// Optional cryptographic state commitment after the transition.
    pub state_root_after: Option<Digest32>,
    /// Policy root used/associated with the event, if applicable.
    pub policy_root: Option<Digest32>,
    /// Policy epoch used/associated with the event, if applicable.
    pub policy_epoch: Option<u64>,
    /// Authorization epoch used/associated with the event, if applicable.
    pub authorization_epoch: Option<u64>,
    /// Revocation epoch used/associated with the event, if applicable.
    pub revocation_epoch: Option<u64>,
    /// Explicit causal predecessor set. Wall-clock/tick order is not a substitute.
    pub causal_parents: BTreeSet<EventId>,
    /// Cognitive tick for correlation only.
    pub cognitive_tick: Option<CognitiveTick>,
    /// Qualification-run grouping label only.
    pub run_id: Option<RunId>,
    /// Confidentiality of this evidence record.
    pub confidentiality: EvidenceConfidentiality,
    /// Typed stage payload.
    pub payload: ShadowEventPayload,
}

/// Owner-issued append-only shadow event record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShadowEvent {
    /// Event schema version.
    pub schema_version: u16,
    /// Owner-issued event identity.
    pub event_id: EventId,
    /// Caller-visible proposal correlation only.
    pub proposal_id: Option<ProposalId>,
    /// Protected-owner transaction identity, if one exists.
    pub transaction_id: Option<TransactionId>,
    /// Role-explicit principal context.
    pub principals: PrincipalContext,
    /// Stage/type of event.
    pub kind: ShadowEventKind,
    /// Protected resource when applicable.
    pub resource: Option<ResourceId>,
    /// Resource version before the transition.
    pub resource_version_before: Option<ResourceVersion>,
    /// Resource version after the transition.
    pub resource_version_after: Option<ResourceVersion>,
    /// Optional cryptographic state commitment before the transition.
    pub state_root_before: Option<Digest32>,
    /// Optional cryptographic state commitment after the transition.
    pub state_root_after: Option<Digest32>,
    /// Policy root used/associated with the event.
    pub policy_root: Option<Digest32>,
    /// Policy epoch used/associated with the event.
    pub policy_epoch: Option<u64>,
    /// Authorization epoch used/associated with the event.
    pub authorization_epoch: Option<u64>,
    /// Revocation epoch used/associated with the event.
    pub revocation_epoch: Option<u64>,
    /// Explicit causal predecessor set.
    pub causal_parents: BTreeSet<EventId>,
    /// Cognitive tick for correlation only.
    pub cognitive_tick: Option<CognitiveTick>,
    /// Qualification-run grouping label only.
    pub run_id: Option<RunId>,
    /// Confidentiality of this evidence record.
    pub confidentiality: EvidenceConfidentiality,
    /// Typed stage payload.
    pub payload: ShadowEventPayload,
}

impl ShadowEvent {
    fn observed_applied(&self) -> Option<bool> {
        match self.payload {
            ShadowEventPayload::MutationObserved { applied } => Some(applied),
            ShadowEventPayload::EvictionObserved { .. } => Some(true),
            _ => None,
        }
    }

    fn evaluation_outcome(&self) -> Option<DecisionOutcome> {
        match &self.payload {
            ShadowEventPayload::Evaluation { receipt } => Some(receipt.outcome),
            _ => None,
        }
    }
}

/// Qualification requirements for one shadow evidence run/scope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualificationManifest {
    /// Event kinds that must be retained for the run to claim completeness.
    pub required_kinds: BTreeSet<ShadowEventKind>,
    /// Observed event kinds counted as P0 mutation attempts for this scope.
    pub p0_observed_kinds: BTreeSet<ShadowEventKind>,
    /// Whether observed resource mutations must carry before/after `ResourceVersion`.
    pub require_resource_versions: bool,
}

impl QualificationManifest {
    /// Build a manifest from required event kinds and scoped P0 observation kinds.
    pub fn new(
        required_kinds: impl IntoIterator<Item = ShadowEventKind>,
        p0_observed_kinds: impl IntoIterator<Item = ShadowEventKind>,
    ) -> Self {
        Self {
            required_kinds: required_kinds.into_iter().collect(),
            p0_observed_kinds: p0_observed_kinds.into_iter().collect(),
            require_resource_versions: false,
        }
    }

    /// Require coherent owner-issued resource versions on observed mutations.
    pub fn with_required_resource_versions(mut self, required: bool) -> Self {
        self.require_resource_versions = required;
        self
    }

    /// Whether an event kind is required for qualification completeness.
    pub fn is_required(&self, kind: ShadowEventKind) -> bool {
        self.required_kinds.contains(&kind)
    }
}

/// Completeness state for the local shadow evidence stream.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize,
)]
pub enum EvidenceCompleteness {
    /// All required and optional evidence retained so far.
    Complete,
    /// Optional evidence was lost; diagnostic use remains possible.
    DegradedEvidence,
    /// Required evidence or structural integrity was lost; coverage claims are invalid.
    InvalidQualification,
}

/// Ledger accounting independent of the generic evidence-plane counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerStats {
    /// Number of event sequence numbers allocated, including lost/invalid attempts.
    pub assigned_sequences: u64,
    /// Number of events retained in the bounded ledger.
    pub stored_events: u64,
    /// Optional events lost to capacity pressure.
    pub optional_events_lost: u64,
    /// Required events lost to capacity pressure.
    pub required_events_lost: u64,
    /// Structurally invalid append attempts.
    pub invalid_event_attempts: u64,
}

/// Non-blocking append failure. The ledger records qualification degradation internally.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AppendError {
    /// Event sequence exhausted.
    #[error("shadow event sequence exhausted")]
    SequenceExhausted,
    /// Causal parent belongs to another ledger epoch.
    #[error("causal parent {parent} belongs to another ledger epoch")]
    ForeignParentEpoch {
        /// Invalid parent identity.
        parent: EventId,
    },
    /// Causal parent has not been retained by this ledger.
    #[error("causal parent {parent} is missing")]
    MissingParent {
        /// Missing parent identity.
        parent: EventId,
    },
    /// Causal parent sequence is not earlier than the new event.
    #[error("causal parent {parent} is not earlier than event {event_id}")]
    ParentNotEarlier {
        /// Parent identity.
        parent: EventId,
        /// New event identity.
        event_id: EventId,
    },
    /// Event kind and payload variant disagree.
    #[error("payload does not match shadow event kind {kind:?}")]
    PayloadKindMismatch {
        /// Event kind whose payload was malformed.
        kind: ShadowEventKind,
    },
    /// Qualification requires owner versions but the event omitted them.
    #[error("event {kind:?} is missing required resource versions")]
    RequiredResourceVersionMissing {
        /// Event kind missing versions.
        kind: ShadowEventKind,
    },
    /// Before/after resource versions are inconsistent with the observed mutation.
    #[error("event {kind:?} has an invalid resource-version transition")]
    InvalidResourceVersionTransition {
        /// Event kind with invalid version semantics.
        kind: ShadowEventKind,
    },
    /// Bounded ledger cannot retain the event without blocking cognition.
    #[error("shadow evidence buffer full at event {event_id}; required={required}")]
    BufferFull {
        /// Reserved event identity that was not retained.
        event_id: EventId,
        /// Whether loss invalidates qualification completeness.
        required: bool,
    },
}

/// Bounded owner-issued event ledger for local shadow instrumentation.
///
/// The ledger never blocks waiting for storage. Capacity pressure degrades or
/// invalidates the evidence claim instead of perturbing legacy cognition.
#[derive(Debug)]
pub struct ShadowEventLedger {
    ledger_epoch: u64,
    next_sequence: u64,
    capacity: usize,
    manifest: QualificationManifest,
    completeness: EvidenceCompleteness,
    events: Vec<ShadowEvent>,
    retained_ids: BTreeSet<EventId>,
    stats: LedgerStats,
}

impl ShadowEventLedger {
    /// Create one bounded local ledger. Sequence numbering begins at one.
    pub fn new(ledger_epoch: u64, capacity: usize, manifest: QualificationManifest) -> Self {
        Self {
            ledger_epoch,
            next_sequence: 1,
            capacity,
            manifest,
            completeness: EvidenceCompleteness::Complete,
            events: Vec::with_capacity(capacity.min(4096)),
            retained_ids: BTreeSet::new(),
            stats: LedgerStats::default(),
        }
    }

    /// Current evidence completeness state.
    pub fn completeness(&self) -> EvidenceCompleteness {
        self.completeness
    }

    /// Current local ledger accounting.
    pub fn stats(&self) -> LedgerStats {
        self.stats
    }

    /// Retained events in owner-issued sequence order.
    pub fn events(&self) -> &[ShadowEvent] {
        &self.events
    }

    /// Append one event without blocking. The ledger owner allocates the identity.
    pub fn try_append(&mut self, draft: ShadowEventDraft) -> Result<EventId, AppendError> {
        let sequence = self.next_sequence;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(AppendError::SequenceExhausted)?;

        let event_id = EventId {
            ledger_epoch: self.ledger_epoch,
            sequence,
        };
        self.stats.assigned_sequences = self.stats.assigned_sequences.saturating_add(1);

        if let Err(error) = self.validate_draft(event_id, &draft) {
            self.stats.invalid_event_attempts = self.stats.invalid_event_attempts.saturating_add(1);
            self.completeness = EvidenceCompleteness::InvalidQualification;
            return Err(error);
        }

        if self.events.len() >= self.capacity {
            let required = self.manifest.is_required(draft.kind);
            if required {
                self.stats.required_events_lost = self.stats.required_events_lost.saturating_add(1);
                self.completeness = EvidenceCompleteness::InvalidQualification;
            } else {
                self.stats.optional_events_lost = self.stats.optional_events_lost.saturating_add(1);
                self.completeness = self
                    .completeness
                    .max(EvidenceCompleteness::DegradedEvidence);
            }
            return Err(AppendError::BufferFull { event_id, required });
        }

        let event = ShadowEvent {
            schema_version: SHADOW_EVENT_SCHEMA_V1,
            event_id,
            proposal_id: draft.proposal_id,
            transaction_id: draft.transaction_id,
            principals: draft.principals,
            kind: draft.kind,
            resource: draft.resource,
            resource_version_before: draft.resource_version_before,
            resource_version_after: draft.resource_version_after,
            state_root_before: draft.state_root_before,
            state_root_after: draft.state_root_after,
            policy_root: draft.policy_root,
            policy_epoch: draft.policy_epoch,
            authorization_epoch: draft.authorization_epoch,
            revocation_epoch: draft.revocation_epoch,
            causal_parents: draft.causal_parents,
            cognitive_tick: draft.cognitive_tick,
            run_id: draft.run_id,
            confidentiality: draft.confidentiality,
            payload: draft.payload,
        };

        self.retained_ids.insert(event_id);
        self.events.push(event);
        self.stats.stored_events = self.stats.stored_events.saturating_add(1);
        Ok(event_id)
    }

    /// Export an ordinary serializable snapshot for reconciliation/qualification.
    ///
    /// This record is evidence data, not authenticated authority.
    pub fn snapshot(&self) -> EvidenceLedgerSnapshot {
        EvidenceLedgerSnapshot {
            schema_version: SHADOW_EVENT_SCHEMA_V1,
            ledger_epoch: self.ledger_epoch,
            last_assigned_sequence: self.next_sequence.saturating_sub(1),
            manifest: self.manifest.clone(),
            completeness: self.completeness,
            stats: self.stats,
            events: self.events.clone(),
        }
    }

    fn validate_draft(&self, event_id: EventId, draft: &ShadowEventDraft) -> Result<(), AppendError> {
        for parent in &draft.causal_parents {
            if parent.ledger_epoch != self.ledger_epoch {
                return Err(AppendError::ForeignParentEpoch { parent: *parent });
            }
            if parent.sequence >= event_id.sequence {
                return Err(AppendError::ParentNotEarlier {
                    parent: *parent,
                    event_id,
                });
            }
            if !self.retained_ids.contains(parent) {
                return Err(AppendError::MissingParent { parent: *parent });
            }
        }

        if !payload_matches_kind(draft.kind, &draft.payload) {
            return Err(AppendError::PayloadKindMismatch { kind: draft.kind });
        }

        validate_version_semantics(
            draft.kind,
            &draft.payload,
            draft.resource_version_before,
            draft.resource_version_after,
            self.manifest.require_resource_versions,
        )
    }
}

/// Serializable local ledger snapshot used for deterministic reconciliation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceLedgerSnapshot {
    /// Event schema version.
    pub schema_version: u16,
    /// Ledger epoch.
    pub ledger_epoch: u64,
    /// Highest owner-issued sequence, including events that were not retained.
    pub last_assigned_sequence: u64,
    /// Qualification scope/requirements.
    pub manifest: QualificationManifest,
    /// Evidence completeness state observed by the ledger owner.
    pub completeness: EvidenceCompleteness,
    /// Local append accounting.
    pub stats: LedgerStats,
    /// Retained event records.
    pub events: Vec<ShadowEvent>,
}

/// Event-derived mechanism counts. These remain measurement, not authority.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DerivedShadowMetrics {
    /// Number of evaluation events.
    pub monitor_invocations: u64,
    /// Number of Allow outcomes.
    pub would_allow: u64,
    /// Number of Deny outcomes.
    pub would_deny: u64,
    /// Number of Quarantine outcomes.
    pub would_quarantine: u64,
    /// Number of RequireAuthorization outcomes.
    pub would_require_authorization: u64,
    /// Number of RequireRevalidation outcomes.
    pub would_require_revalidation: u64,
    /// Number of Defer outcomes.
    pub would_defer: u64,
    /// Legacy applied mutations whose paired evaluation was not Allow.
    pub legacy_commits_after_non_allow: u64,
    /// Scoped P0 mutation attempts.
    pub p0_mutation_attempts: u64,
    /// Scoped P0 attempts with a paired evaluation.
    pub p0_mediated_attempts: u64,
    /// Scoped P0 applied mutations without a paired evaluation.
    pub p0_unmediated_commits: u64,
    /// Required + optional event losses recorded by the bounded ledger.
    pub evidence_events_lost: u64,
}

impl DerivedShadowMetrics {
    /// Export integer metrics into the generic `EvidenceCounters` measurement plane.
    pub fn to_evidence_counters(self) -> Result<EvidenceCounters, MetricExportError> {
        let mut counters = EvidenceCounters::new();
        for (name, value) in self.named_values() {
            if value > MAX_EXACT_F64_INTEGER {
                return Err(MetricExportError::CounterTooLarge { name, value });
            }
            counters.record(name, value as f64);
        }
        Ok(counters)
    }

    fn named_values(self) -> [(&'static str, u64); 12] {
        [
            (COUNTER_MONITOR_INVOCATIONS, self.monitor_invocations),
            (COUNTER_WOULD_ALLOW, self.would_allow),
            (COUNTER_WOULD_DENY, self.would_deny),
            (COUNTER_WOULD_QUARANTINE, self.would_quarantine),
            (
                COUNTER_WOULD_REQUIRE_AUTHORIZATION,
                self.would_require_authorization,
            ),
            (
                COUNTER_WOULD_REQUIRE_REVALIDATION,
                self.would_require_revalidation,
            ),
            (COUNTER_WOULD_DEFER, self.would_defer),
            (
                COUNTER_LEGACY_COMMITS_AFTER_NON_ALLOW,
                self.legacy_commits_after_non_allow,
            ),
            (COUNTER_P0_MUTATION_ATTEMPTS, self.p0_mutation_attempts),
            (COUNTER_P0_MEDIATED_ATTEMPTS, self.p0_mediated_attempts),
            (COUNTER_P0_UNMEDIATED_COMMITS, self.p0_unmediated_commits),
            (COUNTER_EVIDENCE_EVENTS_LOST, self.evidence_events_lost),
        ]
    }
}

/// Failure converting exact integer metrics to the generic `f64` measurement plane.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum MetricExportError {
    /// Value exceeds the largest integer exactly representable by `f64`.
    #[error("counter {name}={value} exceeds exact f64 integer range")]
    CounterTooLarge {
        /// Counter name.
        name: &'static str,
        /// Exact integer value.
        value: u64,
    },
}

/// One deterministic reconciliation violation.
#[derive(Debug, Clone, PartialEq)]
pub enum ReconciliationViolation {
    /// Snapshot schema differs from the implementation schema.
    SnapshotSchemaMismatch { found: u16 },
    /// Event schema differs from the snapshot/implementation schema.
    EventSchemaMismatch { event_id: EventId, found: u16 },
    /// An event belongs to another ledger epoch.
    ForeignLedgerEpoch { event_id: EventId },
    /// Two records claim the same `EventId`.
    DuplicateEventId { event_id: EventId },
    /// An owner-issued sequence is absent from the retained records.
    MissingAssignedSequence { sequence: u64 },
    /// Event kind and payload variant disagree.
    PayloadKindMismatch { event_id: EventId },
    /// Event references a missing causal parent.
    MissingParent { event_id: EventId, parent: EventId },
    /// Causal parent is not earlier than the child.
    ParentNotEarlier { event_id: EventId, parent: EventId },
    /// Evaluation record is not explicitly an evaluation-stage receipt.
    InvalidEvaluationReceiptStage { event_id: EventId },
    /// Observed mutation has no direct matching evaluation parent.
    MissingPairedEvaluation {
        event_id: EventId,
        expected: ShadowEventKind,
    },
    /// Observed mutation has more than one matching direct evaluation parent.
    AmbiguousPairedEvaluation {
        event_id: EventId,
        expected: ShadowEventKind,
    },
    /// Required event kind never appeared in the retained run.
    RequiredEventKindMissing { kind: ShadowEventKind },
    /// Required resource version was absent or before/after semantics were invalid.
    InvalidResourceVersion { event_id: EventId },
    /// Evidence owner reported degraded/incomplete qualification state.
    IncompleteEvidence { completeness: EvidenceCompleteness },
    /// Generic evidence-plane counter was non-finite or non-integral.
    InvalidMeasuredCounter {
        name: &'static str,
        measured: f64,
    },
    /// Generic evidence-plane counter disagrees with event-derived accounting.
    CounterMismatch {
        name: &'static str,
        expected: u64,
        measured: f64,
    },
}

/// Deterministic reconciliation result for one shadow evidence snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct ReconciliationReport {
    /// Metrics derived from event-level evidence.
    pub metrics: DerivedShadowMetrics,
    /// Structural/coverage/counter violations.
    pub violations: Vec<ReconciliationViolation>,
}

impl ReconciliationReport {
    /// Whether this run is eligible to claim complete scoped shadow coverage.
    pub fn qualifies_for_full_coverage(&self) -> bool {
        self.violations.is_empty() && self.metrics.p0_unmediated_commits == 0
    }
}

/// Reconcile a portable event snapshot and, optionally, generic mechanism counters.
///
/// Event ordering in the input vector is not trusted. `EventId` and explicit
/// causal parents define ledger/causal structure.
pub fn reconcile_shadow_evidence(
    snapshot: &EvidenceLedgerSnapshot,
    measured: Option<&EvidenceCounters>,
) -> ReconciliationReport {
    let mut violations = Vec::new();
    let mut metrics = DerivedShadowMetrics {
        evidence_events_lost: snapshot
            .stats
            .required_events_lost
            .saturating_add(snapshot.stats.optional_events_lost),
        ..DerivedShadowMetrics::default()
    };

    if snapshot.schema_version != SHADOW_EVENT_SCHEMA_V1 {
        violations.push(ReconciliationViolation::SnapshotSchemaMismatch {
            found: snapshot.schema_version,
        });
    }
    if snapshot.completeness != EvidenceCompleteness::Complete {
        violations.push(ReconciliationViolation::IncompleteEvidence {
            completeness: snapshot.completeness,
        });
    }

    let mut by_id: BTreeMap<EventId, &ShadowEvent> = BTreeMap::new();
    let mut kind_counts: BTreeMap<ShadowEventKind, u64> = BTreeMap::new();

    for event in &snapshot.events {
        if event.schema_version != SHADOW_EVENT_SCHEMA_V1 {
            violations.push(ReconciliationViolation::EventSchemaMismatch {
                event_id: event.event_id,
                found: event.schema_version,
            });
        }
        if event.event_id.ledger_epoch != snapshot.ledger_epoch {
            violations.push(ReconciliationViolation::ForeignLedgerEpoch {
                event_id: event.event_id,
            });
        }
        if by_id.insert(event.event_id, event).is_some() {
            violations.push(ReconciliationViolation::DuplicateEventId {
                event_id: event.event_id,
            });
        }
        *kind_counts.entry(event.kind).or_insert(0) += 1;

        if !payload_matches_kind(event.kind, &event.payload) {
            violations.push(ReconciliationViolation::PayloadKindMismatch {
                event_id: event.event_id,
            });
        }
        if let ShadowEventPayload::Evaluation { receipt } = &event.payload {
            if receipt.stage != ReceiptStage::Evaluation {
                violations.push(ReconciliationViolation::InvalidEvaluationReceiptStage {
                    event_id: event.event_id,
                });
            }
            metrics.monitor_invocations = metrics.monitor_invocations.saturating_add(1);
            match receipt.outcome {
                DecisionOutcome::Allow => metrics.would_allow = metrics.would_allow.saturating_add(1),
                DecisionOutcome::Deny => metrics.would_deny = metrics.would_deny.saturating_add(1),
                DecisionOutcome::Quarantine => {
                    metrics.would_quarantine = metrics.would_quarantine.saturating_add(1)
                }
                DecisionOutcome::RequireAuthorization => {
                    metrics.would_require_authorization =
                        metrics.would_require_authorization.saturating_add(1)
                }
                DecisionOutcome::RequireRevalidation => {
                    metrics.would_require_revalidation =
                        metrics.would_require_revalidation.saturating_add(1)
                }
                DecisionOutcome::Defer => metrics.would_defer = metrics.would_defer.saturating_add(1),
            }
        }

        if validate_version_semantics(
            event.kind,
            &event.payload,
            event.resource_version_before,
            event.resource_version_after,
            snapshot.manifest.require_resource_versions,
        )
        .is_err()
        {
            violations.push(ReconciliationViolation::InvalidResourceVersion {
                event_id: event.event_id,
            });
        }
    }

    for sequence in 1..=snapshot.last_assigned_sequence {
        let id = EventId {
            ledger_epoch: snapshot.ledger_epoch,
            sequence,
        };
        if !by_id.contains_key(&id) {
            violations.push(ReconciliationViolation::MissingAssignedSequence { sequence });
        }
    }

    for event in by_id.values().copied() {
        for parent in &event.causal_parents {
            let Some(_) = by_id.get(parent) else {
                violations.push(ReconciliationViolation::MissingParent {
                    event_id: event.event_id,
                    parent: *parent,
                });
                continue;
            };
            if parent.ledger_epoch != event.event_id.ledger_epoch
                || parent.sequence >= event.event_id.sequence
            {
                violations.push(ReconciliationViolation::ParentNotEarlier {
                    event_id: event.event_id,
                    parent: *parent,
                });
            }
        }

        let is_p0 = snapshot.manifest.p0_observed_kinds.contains(&event.kind);
        if is_p0 {
            metrics.p0_mutation_attempts = metrics.p0_mutation_attempts.saturating_add(1);
        }

        let Some(expected_evaluation) = event.kind.expected_evaluation() else {
            continue;
        };

        let matching_parents: Vec<&ShadowEvent> = event
            .causal_parents
            .iter()
            .filter_map(|parent| by_id.get(parent).copied())
            .filter(|parent| parent.kind == expected_evaluation)
            .collect();

        match matching_parents.as_slice() {
            [] => {
                violations.push(ReconciliationViolation::MissingPairedEvaluation {
                    event_id: event.event_id,
                    expected: expected_evaluation,
                });
                if is_p0 && event.observed_applied() == Some(true) {
                    metrics.p0_unmediated_commits =
                        metrics.p0_unmediated_commits.saturating_add(1);
                }
            }
            [evaluation] => {
                if is_p0 {
                    metrics.p0_mediated_attempts = metrics.p0_mediated_attempts.saturating_add(1);
                }
                if event.observed_applied() == Some(true)
                    && evaluation.evaluation_outcome() != Some(DecisionOutcome::Allow)
                {
                    metrics.legacy_commits_after_non_allow =
                        metrics.legacy_commits_after_non_allow.saturating_add(1);
                }
            }
            _ => violations.push(ReconciliationViolation::AmbiguousPairedEvaluation {
                event_id: event.event_id,
                expected: expected_evaluation,
            }),
        }
    }

    for kind in &snapshot.manifest.required_kinds {
        if kind_counts.get(kind).copied().unwrap_or(0) == 0 {
            violations.push(ReconciliationViolation::RequiredEventKindMissing { kind: *kind });
        }
    }

    if let Some(measured) = measured {
        for (name, expected) in metrics.named_values() {
            let actual = measured.get(name);
            if !actual.is_finite() || actual < 0.0 || actual.fract() != 0.0 {
                violations.push(ReconciliationViolation::InvalidMeasuredCounter {
                    name,
                    measured: actual,
                });
                continue;
            }
            if expected > MAX_EXACT_F64_INTEGER || actual != expected as f64 {
                violations.push(ReconciliationViolation::CounterMismatch {
                    name,
                    expected,
                    measured: actual,
                });
            }
        }
    }

    ReconciliationReport { metrics, violations }
}

fn payload_matches_kind(kind: ShadowEventKind, payload: &ShadowEventPayload) -> bool {
    match payload {
        ShadowEventPayload::Ingress { .. } => kind == ShadowEventKind::IngressObserved,
        ShadowEventPayload::Evaluation { receipt } => {
            kind.is_evaluation() && receipt.stage == ReceiptStage::Evaluation
        }
        ShadowEventPayload::MutationObserved { .. } => kind.is_paired_legacy_observation(),
        ShadowEventPayload::EvictionObserved { .. } => {
            kind == ShadowEventKind::WorkingMemoryEvictionObserved
        }
        ShadowEventPayload::EvidenceGap { .. } => kind == ShadowEventKind::EvidenceGapObserved,
    }
}

fn validate_version_semantics(
    kind: ShadowEventKind,
    payload: &ShadowEventPayload,
    before: Option<ResourceVersion>,
    after: Option<ResourceVersion>,
    required: bool,
) -> Result<(), AppendError> {
    if !kind.mutates_resource() {
        return Ok(());
    }

    if required && (before.is_none() || after.is_none()) {
        return Err(AppendError::RequiredResourceVersionMissing { kind });
    }

    match (before, after) {
        (None, None) => Ok(()),
        (Some(_), None) | (None, Some(_)) => {
            Err(AppendError::InvalidResourceVersionTransition { kind })
        }
        (Some(before), Some(after)) => {
            let applied = match payload {
                ShadowEventPayload::MutationObserved { applied } => *applied,
                ShadowEventPayload::EvictionObserved { .. } => true,
                _ => return Err(AppendError::PayloadKindMismatch { kind }),
            };
            let valid = if applied {
                after.is_successor_of(before)
            } else {
                after == before
            };
            if valid {
                Ok(())
            } else {
                Err(AppendError::InvalidResourceVersionTransition { kind })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_cogsec::{
        CognitiveSecurityLabel, Consequence, MutationKind, PrincipalId, ReasonCode,
    };

    fn d(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn receipt(outcome: DecisionOutcome, kind: MutationKind) -> MutationReceiptRecord {
        MutationReceiptRecord {
            stage: ReceiptStage::Evaluation,
            request_id: d(1),
            subject: PrincipalId("local-user".into()),
            kind,
            resource: ResourceId("mind/goals".into()),
            mutation_digest: d(2),
            consequence: Consequence::High,
            input_label: CognitiveSecurityLabel::default(),
            expected_resource_state_root: d(3),
            observed_resource_state_root: d(3),
            expected_policy_root: d(4),
            evaluated_policy_root: d(4),
            trusted_policy_root: d(4),
            policy_epoch: 7,
            authorization_epoch: 11,
            revocation_epoch: 13,
            sequence: 42,
            capability_id: Some(d(5)),
            outcome,
            reasons: if outcome == DecisionOutcome::Allow {
                vec![]
            } else {
                vec![ReasonCode::MissingCapability]
            },
        }
    }

    fn draft(
        kind: ShadowEventKind,
        payload: ShadowEventPayload,
        parents: impl IntoIterator<Item = EventId>,
    ) -> ShadowEventDraft {
        ShadowEventDraft {
            kind,
            proposal_id: Some(ProposalId(d(8))),
            transaction_id: None,
            principals: PrincipalContext::default(),
            resource: Some(ResourceId("mind/goals".into())),
            resource_version_before: None,
            resource_version_after: None,
            state_root_before: None,
            state_root_after: None,
            policy_root: Some(d(4)),
            policy_epoch: Some(7),
            authorization_epoch: Some(11),
            revocation_epoch: Some(13),
            causal_parents: parents.into_iter().collect(),
            cognitive_tick: Some(CognitiveTick(100)),
            run_id: Some(RunId::new("shadow-test")),
            confidentiality: EvidenceConfidentiality::LocalPrivate,
            payload,
        }
    }

    fn goal_manifest(require_versions: bool) -> QualificationManifest {
        QualificationManifest::new(
            [
                ShadowEventKind::IngressObserved,
                ShadowEventKind::GoalActivationEvaluated,
                ShadowEventKind::GoalActivationObserved,
            ],
            [ShadowEventKind::GoalActivationObserved],
        )
        .with_required_resource_versions(require_versions)
    }

    #[test]
    fn ledger_assigns_monotonic_ids_and_preserves_causal_lineage() {
        let mut ledger = ShadowEventLedger::new(9, 16, goal_manifest(false));
        let ingress = ledger
            .try_append(draft(
                ShadowEventKind::IngressObserved,
                ShadowEventPayload::Ingress {
                    ingress_class: IngressClass::LegacyUnclassified,
                },
                [],
            ))
            .unwrap();
        let evaluation = ledger
            .try_append(draft(
                ShadowEventKind::GoalActivationEvaluated,
                ShadowEventPayload::Evaluation {
                    receipt: receipt(DecisionOutcome::Allow, MutationKind::GoalActivation),
                },
                [ingress],
            ))
            .unwrap();
        let mut observed = draft(
            ShadowEventKind::GoalActivationObserved,
            ShadowEventPayload::MutationObserved { applied: true },
            [evaluation],
        );
        observed.resource_version_before = Some(ResourceVersion {
            owner_epoch: 3,
            counter: 4,
        });
        observed.resource_version_after = Some(ResourceVersion {
            owner_epoch: 3,
            counter: 5,
        });
        let mutation = ledger.try_append(observed).unwrap();

        assert_eq!(ingress, EventId { ledger_epoch: 9, sequence: 1 });
        assert_eq!(evaluation.sequence, 2);
        assert_eq!(mutation.sequence, 3);
        assert_eq!(ledger.completeness(), EvidenceCompleteness::Complete);

        let report = reconcile_shadow_evidence(&ledger.snapshot(), None);
        assert!(report.qualifies_for_full_coverage());
        assert_eq!(report.metrics.monitor_invocations, 1);
        assert_eq!(report.metrics.p0_mutation_attempts, 1);
        assert_eq!(report.metrics.p0_mediated_attempts, 1);
        assert_eq!(report.metrics.p0_unmediated_commits, 0);
    }

    #[test]
    fn required_buffer_loss_invalidates_qualification_without_blocking() {
        let mut ledger = ShadowEventLedger::new(4, 1, goal_manifest(false));
        let ingress = ledger
            .try_append(draft(
                ShadowEventKind::IngressObserved,
                ShadowEventPayload::Ingress {
                    ingress_class: IngressClass::LegacyUnclassified,
                },
                [],
            ))
            .unwrap();

        let result = ledger.try_append(draft(
            ShadowEventKind::GoalActivationEvaluated,
            ShadowEventPayload::Evaluation {
                receipt: receipt(
                    DecisionOutcome::RequireAuthorization,
                    MutationKind::GoalActivation,
                ),
            },
            [ingress],
        ));

        assert!(matches!(
            result,
            Err(AppendError::BufferFull { required: true, .. })
        ));
        assert_eq!(ledger.completeness(), EvidenceCompleteness::InvalidQualification);
        assert_eq!(ledger.stats().required_events_lost, 1);
        assert_eq!(ledger.stats().assigned_sequences, 2);
        assert_eq!(ledger.stats().stored_events, 1);
    }

    #[test]
    fn non_allow_shadow_decision_and_legacy_commit_remain_distinct() {
        let mut ledger = ShadowEventLedger::new(2, 16, goal_manifest(true));
        let ingress = ledger
            .try_append(draft(
                ShadowEventKind::IngressObserved,
                ShadowEventPayload::Ingress {
                    ingress_class: IngressClass::LegacyUnclassified,
                },
                [],
            ))
            .unwrap();
        let evaluation = ledger
            .try_append(draft(
                ShadowEventKind::GoalActivationEvaluated,
                ShadowEventPayload::Evaluation {
                    receipt: receipt(
                        DecisionOutcome::RequireAuthorization,
                        MutationKind::GoalActivation,
                    ),
                },
                [ingress],
            ))
            .unwrap();
        let mut observed = draft(
            ShadowEventKind::GoalActivationObserved,
            ShadowEventPayload::MutationObserved { applied: true },
            [evaluation],
        );
        observed.resource_version_before = Some(ResourceVersion {
            owner_epoch: 5,
            counter: 10,
        });
        observed.resource_version_after = Some(ResourceVersion {
            owner_epoch: 5,
            counter: 11,
        });
        ledger.try_append(observed).unwrap();

        let report = reconcile_shadow_evidence(&ledger.snapshot(), None);
        assert!(report.qualifies_for_full_coverage());
        assert_eq!(report.metrics.would_require_authorization, 1);
        assert_eq!(report.metrics.legacy_commits_after_non_allow, 1);
        assert_eq!(report.metrics.p0_unmediated_commits, 0);
    }

    #[test]
    fn p0_observation_without_matching_evaluation_is_rejected_by_reconciliation() {
        let manifest = QualificationManifest::new(
            [ShadowEventKind::GoalActivationObserved],
            [ShadowEventKind::GoalActivationObserved],
        )
        .with_required_resource_versions(true);
        let mut ledger = ShadowEventLedger::new(12, 8, manifest);
        let mut observed = draft(
            ShadowEventKind::GoalActivationObserved,
            ShadowEventPayload::MutationObserved { applied: true },
            [],
        );
        observed.resource_version_before = Some(ResourceVersion {
            owner_epoch: 1,
            counter: 20,
        });
        observed.resource_version_after = Some(ResourceVersion {
            owner_epoch: 1,
            counter: 21,
        });
        ledger.try_append(observed).unwrap();

        let report = reconcile_shadow_evidence(&ledger.snapshot(), None);
        assert!(!report.qualifies_for_full_coverage());
        assert_eq!(report.metrics.p0_mutation_attempts, 1);
        assert_eq!(report.metrics.p0_mediated_attempts, 0);
        assert_eq!(report.metrics.p0_unmediated_commits, 1);
        assert!(report.violations.iter().any(|violation| matches!(
            violation,
            ReconciliationViolation::MissingPairedEvaluation { .. }
        )));
    }

    #[test]
    fn mechanism_counters_must_reconcile_with_event_derived_counts() {
        let mut ledger = ShadowEventLedger::new(7, 16, goal_manifest(false));
        let ingress = ledger
            .try_append(draft(
                ShadowEventKind::IngressObserved,
                ShadowEventPayload::Ingress {
                    ingress_class: IngressClass::LegacyUnclassified,
                },
                [],
            ))
            .unwrap();
        let evaluation = ledger
            .try_append(draft(
                ShadowEventKind::GoalActivationEvaluated,
                ShadowEventPayload::Evaluation {
                    receipt: receipt(DecisionOutcome::Allow, MutationKind::GoalActivation),
                },
                [ingress],
            ))
            .unwrap();
        ledger
            .try_append(draft(
                ShadowEventKind::GoalActivationObserved,
                ShadowEventPayload::MutationObserved { applied: false },
                [evaluation],
            ))
            .unwrap();

        let snapshot = ledger.snapshot();
        let clean_metrics = reconcile_shadow_evidence(&snapshot, None).metrics;
        let mut counters = clean_metrics.to_evidence_counters().unwrap();
        counters.record(COUNTER_MONITOR_INVOCATIONS, 2.0);

        let report = reconcile_shadow_evidence(&snapshot, Some(&counters));
        assert!(!report.qualifies_for_full_coverage());
        assert!(report.violations.iter().any(|violation| matches!(
            violation,
            ReconciliationViolation::CounterMismatch {
                name: COUNTER_MONITOR_INVOCATIONS,
                expected: 1,
                measured: 2.0,
            }
        )));
    }

    #[test]
    fn public_event_serialization_contains_no_raw_cognitive_payload_field() {
        let mut ledger = ShadowEventLedger::new(1, 4, goal_manifest(false));
        ledger
            .try_append(draft(
                ShadowEventKind::IngressObserved,
                ShadowEventPayload::Ingress {
                    ingress_class: IngressClass::WebResearch,
                },
                [],
            ))
            .unwrap();
        let json = serde_json::to_string(&ledger.snapshot()).unwrap();
        assert!(!json.contains("raw_prompt"));
        assert!(!json.contains("raw_memory"));
        assert!(!json.contains("content_text"));
    }
}
