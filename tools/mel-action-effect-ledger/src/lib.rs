//! Reference reducer for live musical action-to-effect receipts.
//!
//! This crate binds decisions to downstream observations. It does not run the
//! turn-taking classifier, mutate MuseStream, own the output epoch clock,
//! execute CPAL callbacks, read Studio storage, authenticate consent, or
//! control audio hardware.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fmt;

/// Versioned protocol identity.
pub const ACTION_EFFECT_PROTOCOL_V1: &str = "mel-action-effect-ledger-v1";

/// A half-open musical sample-frame interval.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameWindow {
    /// Inclusive start frame.
    pub start: u64,
    /// Exclusive end frame.
    pub end: u64,
}

impl FrameWindow {
    /// Returns true when start precedes end.
    pub fn is_valid(self) -> bool {
        self.start < self.end
    }

    /// Returns true when frame lies inside this window.
    pub fn contains(self, frame: u64) -> bool {
        self.start <= frame && frame < self.end
    }
}

/// Explicit streaming floor-control action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TurnAction {
    /// Preserve listening space; this is an accounted no-op.
    Wait,
    /// Begin a new bounded contribution.
    Respond,
    /// Preserve the current commitment and continue it.
    Continue,
    /// Relinquish the floor and invalidate pending application output.
    Yield,
}

/// Output state owned by the note/output adapter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputState {
    /// Current output generation epoch.
    pub epoch: u64,
    /// Active musical commitment.
    pub active_commitment_id: Option<String>,
    /// Plan digest for the active commitment.
    pub active_plan_sha256: Option<String>,
    /// Position in the note stream.
    pub note_cursor: u64,
    /// Generated notes not yet returned by the note source.
    pub queued_unconsumed_notes: u64,
    /// Rendered frames pending in the application-owned queue.
    pub pending_application_frames: u64,
}

impl Default for OutputState {
    fn default() -> Self {
        Self {
            epoch: 0,
            active_commitment_id: None,
            active_plan_sha256: None,
            note_cursor: 0,
            queued_unconsumed_notes: 0,
            pending_application_frames: 0,
        }
    }
}

/// Decision committed for one exact input chunk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecisionReceipt {
    /// Stable decision identity.
    pub decision_id: String,
    /// Session identity.
    pub session_id: String,
    /// Turn identity.
    pub turn_id: String,
    /// Zero-based contiguous input chunk index.
    pub chunk_index: u32,
    /// End frame of the fully received chunk.
    pub chunk_end_frame: u64,
    /// Last input frame actually observed by the decision process.
    pub observed_through_frame: u64,
    /// Runner-monotonic time when the chunk was fully received.
    pub chunk_received_at_ns: u64,
    /// Runner-monotonic time when the decision was committed.
    pub decision_at_ns: u64,
    /// Identity of the runner monotonic clock origin.
    pub runner_clock_origin_id: String,
    /// Monotonic control sequence.
    pub control_sequence: u64,
    /// Output epoch under which the action was committed.
    pub output_epoch: u64,
    /// Selected action.
    pub action: TurnAction,
    /// Opaque reference to a trusted authority receipt. Not authenticated here.
    pub authority_receipt_ref: String,
    /// Authorization expiry in the runner clock domain.
    pub expires_at_ns: u64,
    /// Committed semantic plan digest, required for RESPOND.
    pub plan_sha256: Option<String>,
    /// Forecast/commitment identity, required for RESPOND and CONTINUE.
    pub commitment_id: Option<String>,
    /// Allowed response onset window, required for RESPOND.
    pub effect_window: Option<FrameWindow>,
}

/// Clock domains are separate and must not be subtracted without a verified
/// mapping between them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ClockDomain {
    /// Runner Instant used by decisions and queue operations.
    RunnerMonotonic,
    /// CPAL callback timestamp domain.
    CpalCallback,
    /// CPAL predicted playback timestamp domain.
    CpalPlayback,
    /// Independent audio loopback capture domain.
    IndependentLoopback,
}

/// Distinct downstream evidence boundaries for a response artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum EffectStage {
    /// Response applied to the bounded note queue.
    AppliedAtNoteQueue,
    /// Artifact enqueued in application-owned output.
    EnqueuedToApplicationOutput,
    /// CPAL callback observed/admitted output.
    CallbackObserved,
    /// Playback time predicted using a qualified clock-fit bundle.
    CpalPredictedPlayback,
    /// Independent loopback capture observed output.
    LoopbackObserved,
}

/// One receipt advancing a response through an evidence boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EffectReceipt {
    /// Stable receipt identity.
    pub receipt_id: String,
    /// Decision that authorized the response.
    pub decision_id: String,
    /// Exact response artifact digest.
    pub artifact_sha256: String,
    /// Evidence stage reached.
    pub stage: EffectStage,
    /// Clock domain for observed_at.
    pub clock_domain: ClockDomain,
    /// Origin identity within the clock domain.
    pub clock_origin_id: String,
    /// Timestamp in the declared clock domain.
    pub observed_at: u64,
    /// Output epoch attached to the artifact.
    pub output_epoch: u64,
    /// Required for predicted playback.
    pub clock_bridge_sha256: Option<String>,
    /// Required for independent loopback evidence.
    pub loopback_evidence_sha256: Option<String>,
}

/// Receipt proving CONTINUE preserved its current state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContinueReceipt {
    /// Decision identity.
    pub decision_id: String,
    /// Commitment that remains active.
    pub commitment_id: String,
    /// State immediately before operation.
    pub before: OutputState,
    /// Independently read state immediately after operation.
    pub after: OutputState,
}

/// Receipt proving YIELD invalidated exact pending application-owned work.
/// It does not prove that in-flight callbacks or hardware buffers were recalled.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct YieldReceipt {
    /// Decision identity.
    pub decision_id: String,
    /// State before invalidation.
    pub before: OutputState,
    /// State after invalidation.
    pub after: OutputState,
    /// Exact number of unconsumed notes discarded.
    pub discarded_unconsumed_notes: u64,
    /// Exact number of application-owned frames invalidated.
    pub discarded_application_frames: u64,
}

/// Receipt proving a collaborator veto was applied at the application boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VetoReceipt {
    /// Session identity.
    pub session_id: String,
    /// Turn identity.
    pub turn_id: String,
    /// Pseudonymous participant reference.
    pub participant_id: String,
    /// Result from the trusted authority boundary.
    pub authority_valid: bool,
    /// Monotonic control sequence.
    pub control_sequence: u64,
    /// State before the veto.
    pub before: OutputState,
    /// State after the epoch fence.
    pub after: OutputState,
    /// Exact unconsumed notes discarded.
    pub discarded_unconsumed_notes: u64,
    /// Exact pending application frames invalidated.
    pub discarded_application_frames: u64,
}

/// Reducer input event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EventKind {
    /// Commit one explicit action for an input chunk.
    DecisionCommitted(DecisionReceipt),
    /// Record RESPOND's exact plan/artifact application to the note queue.
    RespondApplied {
        /// Decision identity.
        decision_id: String,
        /// Exact plan digest consumed by the note queue.
        plan_sha256: String,
        /// Exact artifact digest produced by the plan.
        artifact_sha256: String,
        /// Commitment/forecast identity realized.
        commitment_id: String,
        /// Musical onset frame.
        effect_frame: u64,
        /// Runner-monotonic time when the note queue applied the response.
        applied_at_ns: u64,
        /// Runner clock origin.
        runner_clock_origin_id: String,
        /// Epoch tagged on the output.
        output_epoch: u64,
    },
    /// Record CONTINUE while preserving state.
    ContinueApplied(ContinueReceipt),
    /// Record YIELD and its exact invalidation boundary.
    YieldApplied(YieldReceipt),
    /// Record collaborator veto.
    VetoApplied(VetoReceipt),
    /// Record a downstream output observation.
    EffectObserved(EffectReceipt),
    /// Close a turn after verifying every input chunk has a decision.
    FinalizeTurn {
        /// Session identity.
        session_id: String,
        /// Turn identity.
        turn_id: String,
        /// Number of chunks in the complete turn.
        chunk_count: u32,
    },
}

/// Ordered event with idempotency identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LedgerEvent {
    /// Strictly increasing modeled event order.
    pub order: u64,
    /// Stable idempotency key.
    pub event_id: String,
    /// Event payload.
    pub kind: EventKind,
}

/// Rejected state transition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LedgerError {
    /// A required identity is empty.
    EmptyIdentity,
    /// Digest is not 64 lowercase hexadecimal characters.
    InvalidDigest,
    /// Input observation did not end at the declared chunk boundary.
    LookaheadOrObservedThroughMismatch,
    /// Runner timestamps are inconsistent.
    InvalidRunnerTime,
    /// A different runner clock origin was used.
    ClockOriginMismatch,
    /// Event order did not increase.
    OutOfOrderEvent { previous: u64, received: u64 },
    /// An idempotency key was reused with another payload.
    ConflictingDuplicateEvent { event_id: String },
    /// Decision id or input chunk was already committed.
    DuplicateDecision,
    /// Session/turn did not match the ledger scope.
    SessionTurnMismatch,
    /// Action-specific fields are missing or disallowed.
    InvalidActionPayload,
    /// Control sequence is stale.
    StaleControlSequence,
    /// Decision uses an old output epoch.
    StaleOutputEpoch { expected: u64, received: u64 },
    /// Authorization expired before application.
    AuthorizationExpired,
    /// Trusted authority boundary rejected the veto.
    VetoAuthorityRejected,
    /// Veto did not advance epoch and invalidate exact pending work.
    InvalidVetoTransition,
    /// An effect was applied to a non-RESPOND action.
    EffectForNonResponse,
    /// RESPOND receipt does not match its decision contract.
    ResponseBindingMismatch,
    /// Response onset lies outside the authorized window.
    EffectOutsideAuthorizedWindow,
    /// CONTINUE changed a state it was required to preserve.
    ContinueMutatedState,
    /// YIELD receipt did not account for invalidated work exactly.
    YieldAccountingMismatch,
    /// Event references an unknown decision.
    UnknownDecision,
    /// Decision was revoked by a later epoch transition.
    RevokedDecision,
    /// Evidence stage skipped or reversed a boundary.
    InvalidStageTransition,
    /// Clock domain did not match evidence stage.
    WrongClockDomain,
    /// A required clock bridge or independent evidence digest is absent.
    MissingEvidenceDigest,
    /// Evidence is bound to another artifact or output epoch.
    EffectEvidenceBindingMismatch,
    /// A clock regressed within its own domain and origin.
    ClockRegression,
    /// Turn close found missing or extra chunk decisions.
    IncompleteTurn { expected: u32, observed: usize },
    /// Turn close did not match the ledger session/turn.
    FinalizeMismatch,
}

impl fmt::Display for LedgerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for LedgerError {}

#[derive(Clone, Debug)]
struct DecisionState {
    receipt: DecisionReceipt,
    response_applied: bool,
    response_artifact_sha256: Option<String>,
    current_stage: Option<EffectStage>,
    revoked: bool,
}

/// Evidence counters derived only from accepted receipts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LedgerSummary {
    /// Number of committed decisions.
    pub decisions: usize,
    /// Number of explicit WAIT chunks.
    pub wait_chunks: usize,
    /// Number of RESPOND decisions.
    pub respond_decisions: usize,
    /// Number of RESPOND decisions applied at the note queue.
    pub respond_applied: usize,
    /// Number of CONTINUE decisions.
    pub continue_decisions: usize,
    /// Number of CONTINUE receipts proving state preservation.
    pub continue_preserved: usize,
    /// Number of YIELD decisions.
    pub yield_decisions: usize,
    /// Number of YIELD receipts applied.
    pub yield_applied: usize,
    /// Number of accepted veto receipts.
    pub vetoes: usize,
    /// Number of application enqueue observations.
    pub enqueued: usize,
    /// Number of CPAL callback observations.
    pub callback_observed: usize,
    /// Number of CPAL predicted playback observations.
    pub cpal_predicted: usize,
    /// Number of independent loopback observations.
    pub loopback_observed: usize,
    /// Number of decisions revoked by a later output epoch.
    pub revoked_decisions: usize,
    /// RESPOND decisions with no note-queue receipt and no revocation.
    pub unresolved_responds: usize,
}

/// Deterministic reducer scoped to one session and turn.
#[derive(Debug)]
pub struct ActionEffectLedger {
    session_id: String,
    turn_id: String,
    runner_clock_origin_id: String,
    state: OutputState,
    last_order: Option<u64>,
    last_control_sequence: u64,
    last_chunk_end_frame: Option<u64>,
    last_runner_time_ns: Option<u64>,
    events: BTreeMap<String, LedgerEvent>,
    decisions: BTreeMap<String, DecisionState>,
    chunk_decisions: BTreeMap<u32, String>,
    clock_times: BTreeMap<(ClockDomain, String), u64>,
    finalized: bool,
}

impl ActionEffectLedger {
    /// Construct a ledger using one session, turn, and runner clock origin.
    pub fn new(
        session_id: impl Into<String>,
        turn_id: impl Into<String>,
        runner_clock_origin_id: impl Into<String>,
        initial_state: OutputState,
    ) -> Result<Self, LedgerError> {
        let session_id = session_id.into();
        let turn_id = turn_id.into();
        let runner_clock_origin_id = runner_clock_origin_id.into();
        if session_id.trim().is_empty()
            || turn_id.trim().is_empty()
            || runner_clock_origin_id.trim().is_empty()
        {
            return Err(LedgerError::EmptyIdentity);
        }
        Ok(Self {
            session_id,
            turn_id,
            runner_clock_origin_id,
            state: initial_state,
            last_order: None,
            last_control_sequence: 0,
            last_chunk_end_frame: None,
            last_runner_time_ns: None,
            events: BTreeMap::new(),
            decisions: BTreeMap::new(),
            chunk_decisions: BTreeMap::new(),
            clock_times: BTreeMap::new(),
            finalized: false,
        })
    }

    /// Current output state after all accepted transitions.
    pub fn state(&self) -> &OutputState {
        &self.state
    }

    /// Apply one event. Exact replays are idempotent. Invalid transitions do
    /// not advance order or mutate state.
    pub fn apply(&mut self, event: LedgerEvent) -> Result<(), LedgerError> {
        if event.event_id.trim().is_empty() {
            return Err(LedgerError::EmptyIdentity);
        }
        if let Some(existing) = self.events.get(&event.event_id) {
            return if existing == &event {
                Ok(())
            } else {
                Err(LedgerError::ConflictingDuplicateEvent {
                    event_id: event.event_id,
                })
            };
        }
        if self.finalized {
            return Err(LedgerError::FinalizeMismatch);
        }
        if let Some(previous) = self.last_order {
            if event.order <= previous {
                return Err(LedgerError::OutOfOrderEvent {
                    previous,
                    received: event.order,
                });
            }
        }

        match &event.kind {
            EventKind::DecisionCommitted(receipt) => self.apply_decision(receipt)?,
            EventKind::RespondApplied {
                decision_id,
                plan_sha256,
                artifact_sha256,
                commitment_id,
                effect_frame,
                applied_at_ns,
                runner_clock_origin_id,
                output_epoch,
            } => self.apply_respond(
                decision_id,
                plan_sha256,
                artifact_sha256,
                commitment_id,
                *effect_frame,
                *applied_at_ns,
                runner_clock_origin_id,
                *output_epoch,
            )?,
            EventKind::ContinueApplied(receipt) => self.apply_continue(receipt)?,
            EventKind::YieldApplied(receipt) => self.apply_yield(receipt)?,
            EventKind::VetoApplied(receipt) => self.apply_veto(receipt)?,
            EventKind::EffectObserved(receipt) => self.apply_effect(receipt)?,
            EventKind::FinalizeTurn {
                session_id,
                turn_id,
                chunk_count,
            } => self.apply_finalize(session_id, turn_id, *chunk_count)?,
        }
        self.last_order = Some(event.order);
        self.events.insert(event.event_id.clone(), event);
        Ok(())
    }

    fn apply_decision(&mut self, receipt: &DecisionReceipt) -> Result<(), LedgerError> {
        if receipt.session_id != self.session_id || receipt.turn_id != self.turn_id {
            return Err(LedgerError::SessionTurnMismatch);
        }
        if receipt.decision_id.trim().is_empty()
            || receipt.authority_receipt_ref.trim().is_empty()
            || receipt.runner_clock_origin_id.trim().is_empty()
        {
            return Err(LedgerError::EmptyIdentity);
        }
        if self.decisions.contains_key(&receipt.decision_id)
            || self.chunk_decisions.contains_key(&receipt.chunk_index)
        {
            return Err(LedgerError::DuplicateDecision);
        }
        if receipt.observed_through_frame != receipt.chunk_end_frame {
            return Err(LedgerError::LookaheadOrObservedThroughMismatch);
        }
        if receipt.chunk_received_at_ns > receipt.decision_at_ns
            || receipt.decision_at_ns > receipt.expires_at_ns
        {
            return Err(LedgerError::InvalidRunnerTime);
        }
        if receipt.runner_clock_origin_id != self.runner_clock_origin_id {
            return Err(LedgerError::ClockOriginMismatch);
        }
        if receipt.control_sequence <= self.last_control_sequence {
            return Err(LedgerError::StaleControlSequence);
        }
        if receipt.output_epoch != self.state.epoch {
            return Err(LedgerError::StaleOutputEpoch {
                expected: self.state.epoch,
                received: receipt.output_epoch,
            });
        }
        if let Some(previous_end) = self.last_chunk_end_frame {
            if receipt.chunk_end_frame <= previous_end {
                return Err(LedgerError::LookaheadOrObservedThroughMismatch);
            }
        }
        if let Some(previous_time) = self.last_runner_time_ns {
            if receipt.chunk_received_at_ns < previous_time
                || receipt.decision_at_ns < previous_time
            {
                return Err(LedgerError::InvalidRunnerTime);
            }
        }
        match receipt.action {
            TurnAction::Wait | TurnAction::Yield => {
                if receipt.plan_sha256.is_some()
                    || receipt.commitment_id.is_some()
                    || receipt.effect_window.is_some()
                {
                    return Err(LedgerError::InvalidActionPayload);
                }
            }
            TurnAction::Respond => {
                let plan = receipt.plan_sha256.as_deref().ok_or(LedgerError::InvalidActionPayload)?;
                let commitment = receipt.commitment_id.as_deref().ok_or(LedgerError::InvalidActionPayload)?;
                let window = receipt.effect_window.ok_or(LedgerError::InvalidActionPayload)?;
                if !valid_sha256(plan) || commitment.trim().is_empty() || !window.is_valid() {
                    return Err(LedgerError::InvalidActionPayload);
                }
            }
            TurnAction::Continue => {
                if receipt.commitment_id.as_deref().unwrap_or("").trim().is_empty()
                    || receipt.plan_sha256.is_some()
                    || receipt.effect_window.is_some()
                    || self.state.active_commitment_id.as_deref() != receipt.commitment_id.as_deref()
                {
                    return Err(LedgerError::ContinueMutatedState);
                }
            }
        }
        self.last_control_sequence = receipt.control_sequence;
        self.last_chunk_end_frame = Some(receipt.chunk_end_frame);
        self.last_runner_time_ns = Some(receipt.decision_at_ns);
        self.clock_times.insert(
            (ClockDomain::RunnerMonotonic, self.runner_clock_origin_id.clone()),
            receipt.decision_at_ns,
        );
        self.chunk_decisions.insert(receipt.chunk_index, receipt.decision_id.clone());
        self.decisions.insert(
            receipt.decision_id.clone(),
            DecisionState {
                receipt: receipt.clone(),
                response_applied: false,
                response_artifact_sha256: None,
                current_stage: None,
                revoked: false,
            },
        );
        Ok(())
    }

    fn apply_respond(
        &mut self,
        decision_id: &str,
        plan_sha256: &str,
        artifact_sha256: &str,
        commitment_id: &str,
        effect_frame: u64,
        applied_at_ns: u64,
        runner_clock_origin_id: &str,
        output_epoch: u64,
    ) -> Result<(), LedgerError> {
        let state = self.decisions.get_mut(decision_id).ok_or(LedgerError::UnknownDecision)?;
        if state.revoked {
            return Err(LedgerError::RevokedDecision);
        }
        if state.receipt.action != TurnAction::Respond {
            return Err(LedgerError::EffectForNonResponse);
        }
        if state.response_applied {
            return Err(LedgerError::ResponseBindingMismatch);
        }
        if state.receipt.output_epoch != self.state.epoch || output_epoch != self.state.epoch {
            return Err(LedgerError::StaleOutputEpoch {
                expected: self.state.epoch,
                received: output_epoch,
            });
        }
        if runner_clock_origin_id != self.runner_clock_origin_id {
            return Err(LedgerError::ClockOriginMismatch);
        }
        if applied_at_ns < state.receipt.decision_at_ns || applied_at_ns > state.receipt.expires_at_ns {
            return Err(LedgerError::AuthorizationExpired);
        }
        let expected_plan = state.receipt.plan_sha256.as_deref().unwrap_or("");
        let expected_commitment = state.receipt.commitment_id.as_deref().unwrap_or("");
        let window = state.receipt.effect_window.ok_or(LedgerError::InvalidActionPayload)?;
        if !valid_sha256(plan_sha256) || !valid_sha256(artifact_sha256) {
            return Err(LedgerError::InvalidDigest);
        }
        if plan_sha256 != expected_plan || commitment_id != expected_commitment {
            return Err(LedgerError::ResponseBindingMismatch);
        }
        if !window.contains(effect_frame) {
            return Err(LedgerError::EffectOutsideAuthorizedWindow);
        }
        let key = (ClockDomain::RunnerMonotonic, self.runner_clock_origin_id.clone());
        if self.clock_times.get(&key).is_some_and(|previous| applied_at_ns < *previous) {
            return Err(LedgerError::ClockRegression);
        }
        self.clock_times.insert(key, applied_at_ns);
        state.response_applied = true;
        state.response_artifact_sha256 = Some(artifact_sha256.to_owned());
        state.current_stage = Some(EffectStage::AppliedAtNoteQueue);
        self.state.active_commitment_id = Some(commitment_id.to_owned());
        self.state.active_plan_sha256 = Some(plan_sha256.to_owned());
        Ok(())
    }

    fn apply_continue(&self, receipt: &ContinueReceipt) -> Result<(), LedgerError> {
        let state = self.decisions.get(&receipt.decision_id).ok_or(LedgerError::UnknownDecision)?;
        if state.revoked {
            return Err(LedgerError::RevokedDecision);
        }
        if state.receipt.action != TurnAction::Continue {
            return Err(LedgerError::EffectForNonResponse);
        }
        if state.receipt.output_epoch != self.state.epoch
            || receipt.before != self.state
            || receipt.after != receipt.before
            || receipt.commitment_id != state.receipt.commitment_id.as_deref().unwrap_or("")
            || receipt.before.active_commitment_id.as_deref() != Some(receipt.commitment_id.as_str())
        {
            return Err(LedgerError::ContinueMutatedState);
        }
        Ok(())
    }

    fn apply_yield(&mut self, receipt: &YieldReceipt) -> Result<(), LedgerError> {
        let state = self.decisions.get(&receipt.decision_id).ok_or(LedgerError::UnknownDecision)?;
        if state.revoked {
            return Err(LedgerError::RevokedDecision);
        }
        if state.receipt.action != TurnAction::Yield {
            return Err(LedgerError::EffectForNonResponse);
        }
        if state.receipt.output_epoch != self.state.epoch || receipt.before != self.state {
            return Err(LedgerError::StaleOutputEpoch {
                expected: self.state.epoch,
                received: receipt.before.epoch,
            });
        }
        validate_invalidation(
            &receipt.before,
            &receipt.after,
            receipt.discarded_unconsumed_notes,
            receipt.discarded_application_frames,
        )?;
        self.state = receipt.after.clone();
        self.revoke_old_epoch_decisions();
        Ok(())
    }

    fn apply_veto(&mut self, receipt: &VetoReceipt) -> Result<(), LedgerError> {
        if receipt.session_id != self.session_id || receipt.turn_id != self.turn_id {
            return Err(LedgerError::SessionTurnMismatch);
        }
        if receipt.participant_id.trim().is_empty() {
            return Err(LedgerError::EmptyIdentity);
        }
        if !receipt.authority_valid {
            return Err(LedgerError::VetoAuthorityRejected);
        }
        if receipt.control_sequence <= self.last_control_sequence {
            return Err(LedgerError::StaleControlSequence);
        }
        if receipt.before != self.state {
            return Err(LedgerError::StaleOutputEpoch {
                expected: self.state.epoch,
                received: receipt.before.epoch,
            });
        }
        validate_invalidation(
            &receipt.before,
            &receipt.after,
            receipt.discarded_unconsumed_notes,
            receipt.discarded_application_frames,
        )
        .map_err(|_| LedgerError::InvalidVetoTransition)?;
        self.last_control_sequence = receipt.control_sequence;
        self.state = receipt.after.clone();
        self.revoke_old_epoch_decisions();
        Ok(())
    }

    fn revoke_old_epoch_decisions(&mut self) {
        for decision in self.decisions.values_mut() {
            if decision.receipt.output_epoch < self.state.epoch {
                decision.revoked = true;
            }
        }
    }

    fn apply_effect(&mut self, receipt: &EffectReceipt) -> Result<(), LedgerError> {
        if receipt.receipt_id.trim().is_empty() || receipt.clock_origin_id.trim().is_empty() {
            return Err(LedgerError::EmptyIdentity);
        }
        let state = self.decisions.get_mut(&receipt.decision_id).ok_or(LedgerError::UnknownDecision)?;
        if state.revoked && state.current_stage != Some(EffectStage::CallbackObserved)
            && state.current_stage != Some(EffectStage::CpalPredictedPlayback)
        {
            return Err(LedgerError::RevokedDecision);
        }
        if state.receipt.action != TurnAction::Respond || !state.response_applied {
            return Err(LedgerError::EffectForNonResponse);
        }
        if receipt.output_epoch != state.receipt.output_epoch
            || receipt.artifact_sha256 != state.response_artifact_sha256.as_deref().unwrap_or("")
            || (!state.revoked && receipt.output_epoch != self.state.epoch)
        {
            return Err(LedgerError::EffectEvidenceBindingMismatch);
        }
        let expected_domain = match receipt.stage {
            EffectStage::AppliedAtNoteQueue => ClockDomain::RunnerMonotonic,
            EffectStage::EnqueuedToApplicationOutput => ClockDomain::RunnerMonotonic,
            EffectStage::CallbackObserved => ClockDomain::CpalCallback,
            EffectStage::CpalPredictedPlayback => ClockDomain::CpalPlayback,
            EffectStage::LoopbackObserved => ClockDomain::IndependentLoopback,
        };
        if receipt.clock_domain != expected_domain {
            return Err(LedgerError::WrongClockDomain);
        }
        match receipt.stage {
            EffectStage::AppliedAtNoteQueue => return Err(LedgerError::InvalidStageTransition),
            EffectStage::EnqueuedToApplicationOutput => {
                if state.current_stage != Some(EffectStage::AppliedAtNoteQueue) {
                    return Err(LedgerError::InvalidStageTransition);
                }
            }
            EffectStage::CallbackObserved => {
                if state.current_stage != Some(EffectStage::EnqueuedToApplicationOutput) {
                    return Err(LedgerError::InvalidStageTransition);
                }
            }
            EffectStage::CpalPredictedPlayback => {
                if state.current_stage != Some(EffectStage::CallbackObserved) {
                    return Err(LedgerError::InvalidStageTransition);
                }
                if receipt.clock_bridge_sha256.as_deref().is_none_or(|digest| !valid_sha256(digest)) {
                    return Err(LedgerError::MissingEvidenceDigest);
                }
            }
            EffectStage::LoopbackObserved => {
                if !matches!(
                    state.current_stage,
                    Some(EffectStage::CallbackObserved | EffectStage::CpalPredictedPlayback)
                ) {
                    return Err(LedgerError::InvalidStageTransition);
                }
                if receipt.loopback_evidence_sha256.as_deref().is_none_or(|digest| !valid_sha256(digest)) {
                    return Err(LedgerError::MissingEvidenceDigest);
                }
            }
        }
        if (receipt.stage != EffectStage::CpalPredictedPlayback && receipt.clock_bridge_sha256.is_some())
            || (receipt.stage != EffectStage::LoopbackObserved && receipt.loopback_evidence_sha256.is_some())
        {
            return Err(LedgerError::MissingEvidenceDigest);
        }
        let key = (receipt.clock_domain, receipt.clock_origin_id.clone());
        if self.clock_times.get(&key).is_some_and(|previous| receipt.observed_at < *previous) {
            return Err(LedgerError::ClockRegression);
        }
        self.clock_times.insert(key, receipt.observed_at);
        state.current_stage = Some(receipt.stage);
        Ok(())
    }

    fn apply_finalize(
        &mut self,
        session_id: &str,
        turn_id: &str,
        chunk_count: u32,
    ) -> Result<(), LedgerError> {
        if session_id != self.session_id || turn_id != self.turn_id {
            return Err(LedgerError::FinalizeMismatch);
        }
        if self.chunk_decisions.len() != chunk_count as usize
            || (0..chunk_count).any(|index| !self.chunk_decisions.contains_key(&index))
        {
            return Err(LedgerError::IncompleteTurn {
                expected: chunk_count,
                observed: self.chunk_decisions.len(),
            });
        }
        self.finalized = true;
        Ok(())
    }

    /// Summarize only transitions supported by accepted receipts.
    pub fn summary(&self) -> LedgerSummary {
        let mut summary = LedgerSummary::default();
        summary.decisions = self.decisions.len();
        for decision in self.decisions.values() {
            match decision.receipt.action {
                TurnAction::Wait => summary.wait_chunks += 1,
                TurnAction::Respond => {
                    summary.respond_decisions += 1;
                    if decision.response_applied {
                        summary.respond_applied += 1;
                    } else if !decision.revoked {
                        summary.unresolved_responds += 1;
                    }
                }
                TurnAction::Continue => summary.continue_decisions += 1,
                TurnAction::Yield => summary.yield_decisions += 1,
            }
            if decision.revoked {
                summary.revoked_decisions += 1;
            }
            match decision.current_stage {
                Some(EffectStage::EnqueuedToApplicationOutput) => summary.enqueued += 1,
                Some(EffectStage::CallbackObserved) => summary.callback_observed += 1,
                Some(EffectStage::CpalPredictedPlayback) => {
                    summary.callback_observed += 1;
                    summary.cpal_predicted += 1;
                }
                Some(EffectStage::LoopbackObserved) => summary.loopback_observed += 1,
                Some(EffectStage::AppliedAtNoteQueue) | None => {}
            }
        }
        summary.vetoes = self.events.values().filter(|event| matches!(event.kind, EventKind::VetoApplied(_))).count();
        summary.yield_applied = self.events.values().filter(|event| matches!(event.kind, EventKind::YieldApplied(_))).count();
        summary.continue_preserved = self.events.values().filter(|event| matches!(event.kind, EventKind::ContinueApplied(_))).count();
        summary.enqueued = self.decisions.values().filter(|decision| decision.current_stage.map(|stage| stage >= EffectStage::EnqueuedToApplicationOutput).unwrap_or(false)).count();
        summary.callback_observed = self.decisions.values().filter(|decision| decision.current_stage.map(|stage| stage >= EffectStage::CallbackObserved).unwrap_or(false)).count();
        summary.cpal_predicted = self.decisions.values().filter(|decision| decision.current_stage == Some(EffectStage::CpalPredictedPlayback)).count();
        summary.loopback_observed = self.decisions.values().filter(|decision| decision.current_stage == Some(EffectStage::LoopbackObserved)).count();
        summary
    }

    /// Whether a decision was revoked.
    pub fn decision_revoked(&self, decision_id: &str) -> Option<bool> {
        self.decisions.get(decision_id).map(|state| state.revoked)
    }

    /// Highest accepted output evidence stage for a decision.
    pub fn effect_stage(&self, decision_id: &str) -> Option<EffectStage> {
        self.decisions.get(decision_id).and_then(|state| state.current_stage)
    }
}

fn validate_invalidation(
    before: &OutputState,
    after: &OutputState,
    discarded_notes: u64,
    discarded_frames: u64,
) -> Result<(), LedgerError> {
    let expected_epoch = before.epoch.checked_add(1).ok_or(LedgerError::InvalidVetoTransition)?;
    let note_delta = before
        .queued_unconsumed_notes
        .checked_sub(after.queued_unconsumed_notes)
        .ok_or(LedgerError::YieldAccountingMismatch)?;
    let frame_delta = before
        .pending_application_frames
        .checked_sub(after.pending_application_frames)
        .ok_or(LedgerError::YieldAccountingMismatch)?;
    if after.epoch != expected_epoch
        || discarded_notes != note_delta
        || discarded_frames != frame_delta
        || after.queued_unconsumed_notes != 0
        || after.pending_application_frames != 0
        || after.note_cursor != before.note_cursor
        || after.active_commitment_id.is_some()
        || after.active_plan_sha256.is_some()
    {
        return Err(LedgerError::YieldAccountingMismatch);
    }
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    const RUNNER: &str = "runner-origin-1";
    const PLAN: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const ARTIFACT: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const BRIDGE: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const LOOPBACK: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

    fn initial_state() -> OutputState {
        OutputState {
            epoch: 0,
            active_commitment_id: Some("active-0".into()),
            active_plan_sha256: Some(PLAN.into()),
            note_cursor: 12,
            queued_unconsumed_notes: 5,
            pending_application_frames: 480,
        }
    }

    fn decision(action: TurnAction, chunk_index: u32, control_sequence: u64) -> DecisionReceipt {
        let (plan_sha256, commitment_id, effect_window) = match action {
            TurnAction::Wait | TurnAction::Yield => (None, None, None),
            TurnAction::Respond => (
                Some(PLAN.into()),
                Some("commit-respond".into()),
                Some(FrameWindow { start: 100, end: 200 }),
            ),
            TurnAction::Continue => (None, Some("active-0".into()), None),
        };
        DecisionReceipt {
            decision_id: format!("decision-{chunk_index}"),
            session_id: "session-1".into(),
            turn_id: "turn-1".into(),
            chunk_index,
            chunk_end_frame: (chunk_index as u64 + 1) * 256,
            observed_through_frame: (chunk_index as u64 + 1) * 256,
            chunk_received_at_ns: 1_000 + chunk_index as u64 * 100,
            decision_at_ns: 1_050 + chunk_index as u64 * 100,
            runner_clock_origin_id: RUNNER.into(),
            control_sequence,
            output_epoch: 0,
            action,
            authority_receipt_ref: "authority-ref-1".into(),
            expires_at_ns: 5_000 + chunk_index as u64 * 100,
            plan_sha256,
            commitment_id,
            effect_window,
        }
    }

    fn ledger() -> ActionEffectLedger {
        ActionEffectLedger::new("session-1", "turn-1", RUNNER, initial_state()).unwrap()
    }

    fn event(order: u64, event_id: &str, kind: EventKind) -> LedgerEvent {
        LedgerEvent {
            order,
            event_id: event_id.into(),
            kind,
        }
    }

    fn commit(ledger: &mut ActionEffectLedger, order: u64, receipt: DecisionReceipt) {
        let event_id = format!("commit-{}", receipt.decision_id);
        ledger
            .apply(event(order, &event_id, EventKind::DecisionCommitted(receipt)))
            .unwrap();
    }

    fn respond_event(decision_id: &str, order: u64) -> LedgerEvent {
        event(
            order,
            &format!("respond-{decision_id}"),
            EventKind::RespondApplied {
                decision_id: decision_id.into(),
                plan_sha256: PLAN.into(),
                artifact_sha256: ARTIFACT.into(),
                commitment_id: "commit-respond".into(),
                effect_frame: 150,
                applied_at_ns: 1_100,
                runner_clock_origin_id: RUNNER.into(),
                output_epoch: 0,
            },
        )
    }

    fn stage_event(
        decision_id: &str,
        receipt_id: &str,
        order: u64,
        stage: EffectStage,
        domain: ClockDomain,
        origin: &str,
        at: u64,
    ) -> LedgerEvent {
        event(
            order,
            &format!("effect-{receipt_id}"),
            EventKind::EffectObserved(EffectReceipt {
                receipt_id: receipt_id.into(),
                decision_id: decision_id.into(),
                artifact_sha256: ARTIFACT.into(),
                stage,
                clock_domain: domain,
                clock_origin_id: origin.into(),
                observed_at: at,
                output_epoch: 0,
                clock_bridge_sha256: (stage == EffectStage::CpalPredictedPlayback).then_some(BRIDGE.into()),
                loopback_evidence_sha256: (stage == EffectStage::LoopbackObserved).then_some(LOOPBACK.into()),
            }),
        )
    }

    fn invalidate_state(before: &OutputState) -> OutputState {
        OutputState {
            epoch: before.epoch + 1,
            active_commitment_id: None,
            active_plan_sha256: None,
            note_cursor: before.note_cursor,
            queued_unconsumed_notes: 0,
            pending_application_frames: 0,
        }
    }

    #[test]
    fn wait_is_explicit_and_has_no_output_effect() {
        let mut ledger = ledger();
        commit(&mut ledger, 1, decision(TurnAction::Wait, 0, 1));
        ledger.apply(event(2, "close", EventKind::FinalizeTurn {
            session_id: "session-1".into(),
            turn_id: "turn-1".into(),
            chunk_count: 1,
        })).unwrap();
        assert_eq!(ledger.summary().wait_chunks, 1);
        assert_eq!(ledger.summary().enqueued, 0);
    }

    #[test]
    fn missing_chunk_does_not_default_to_wait() {
        let mut ledger = ledger();
        commit(&mut ledger, 1, decision(TurnAction::Wait, 1, 1));
        assert_eq!(
            ledger.apply(event(2, "close", EventKind::FinalizeTurn {
                session_id: "session-1".into(),
                turn_id: "turn-1".into(),
                chunk_count: 2,
            })),
            Err(LedgerError::IncompleteTurn { expected: 2, observed: 1 })
        );
    }

    #[test]
    fn respond_binds_plan_artifact_forecast_epoch_and_window() {
        let mut ledger = ledger();
        commit(&mut ledger, 1, decision(TurnAction::Respond, 0, 1));
        ledger.apply(respond_event("decision-0", 2)).unwrap();
        assert_eq!(ledger.summary().respond_applied, 1);
        assert_eq!(ledger.state().active_commitment_id.as_deref(), Some("commit-respond"));
    }

    #[test]
    fn respond_rejects_plan_substitution_and_out_of_window_effect() {
        let mut ledger = ledger();
        commit(&mut ledger, 1, decision(TurnAction::Respond, 0, 1));
        let mut bad_plan = respond_event("decision-0", 2);
        if let EventKind::RespondApplied { plan_sha256, .. } = &mut bad_plan.kind {
            *plan_sha256 = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee".into();
        }
        assert_eq!(ledger.apply(bad_plan), Err(LedgerError::ResponseBindingMismatch));
        let mut bad_frame = respond_event("decision-0", 2);
        if let EventKind::RespondApplied { effect_frame, .. } = &mut bad_frame.kind {
            *effect_frame = 250;
        }
        assert_eq!(ledger.apply(bad_frame), Err(LedgerError::EffectOutsideAuthorizedWindow));
    }

    #[test]
    fn respond_rejects_expiry_and_runner_clock_origin_mismatch() {
        let mut ledger = ledger();
        commit(&mut ledger, 1, decision(TurnAction::Respond, 0, 1));
        let mut expired = respond_event("decision-0", 2);
        if let EventKind::RespondApplied { applied_at_ns, .. } = &mut expired.kind {
            *applied_at_ns = 9_000;
        }
        assert_eq!(ledger.apply(expired), Err(LedgerError::AuthorizationExpired));
        let mut wrong_origin = respond_event("decision-0", 2);
        if let EventKind::RespondApplied { runner_clock_origin_id, .. } = &mut wrong_origin.kind {
            *runner_clock_origin_id = "other-origin".into();
        }
        assert_eq!(ledger.apply(wrong_origin), Err(LedgerError::ClockOriginMismatch));
    }

    #[test]
    fn continue_requires_exact_state_preservation() {
        let mut ledger = ledger();
        commit(&mut ledger, 1, decision(TurnAction::Continue, 0, 1));
        let before = initial_state();
        let mut after = before.clone();
        after.note_cursor += 1;
        assert_eq!(
            ledger.apply(event(2, "continue", EventKind::ContinueApplied(ContinueReceipt {
                decision_id: "decision-0".into(),
                commitment_id: "active-0".into(),
                before,
                after,
            }))),
            Err(LedgerError::ContinueMutatedState)
        );
    }

    #[test]
    fn continue_receipt_preserves_commitment_cursor_queue_and_epoch() {
        let mut ledger = ledger();
        commit(&mut ledger, 1, decision(TurnAction::Continue, 0, 1));
        let before = initial_state();
        ledger.apply(event(2, "continue", EventKind::ContinueApplied(ContinueReceipt {
            decision_id: "decision-0".into(),
            commitment_id: "active-0".into(),
            after: before.clone(),
            before,
        }))).unwrap();
        assert_eq!(ledger.summary().continue_preserved, 1);
    }

    #[test]
    fn yield_advances_epoch_and_accounts_exact_pending_work() {
        let mut ledger = ledger();
        commit(&mut ledger, 1, decision(TurnAction::Yield, 0, 1));
        let before = initial_state();
        let after = invalidate_state(&before);
        ledger.apply(event(2, "yield", EventKind::YieldApplied(YieldReceipt {
            decision_id: "decision-0".into(),
            before,
            after,
            discarded_unconsumed_notes: 5,
            discarded_application_frames: 480,
        }))).unwrap();
        assert_eq!(ledger.state().epoch, 1);
        assert_eq!(ledger.summary().yield_applied, 1);
    }

    #[test]
    fn yield_does_not_claim_callback_admission_or_acoustic_silence() {
        let mut ledger = ledger();
        commit(&mut ledger, 1, decision(TurnAction::Yield, 0, 1));
        let before = initial_state();
        let after = invalidate_state(&before);
        ledger.apply(event(2, "yield", EventKind::YieldApplied(YieldReceipt {
            decision_id: "decision-0".into(),
            before,
            after,
            discarded_unconsumed_notes: 5,
            discarded_application_frames: 480,
        }))).unwrap();
        let summary = ledger.summary();
        assert_eq!(summary.callback_observed, 0);
        assert_eq!(summary.cpal_predicted, 0);
        assert_eq!(summary.loopback_observed, 0);
    }

    #[test]
    fn invalid_veto_is_rejected_without_state_mutation() {
        let mut ledger = ledger();
        let before = initial_state();
        let after = invalidate_state(&before);
        assert_eq!(
            ledger.apply(event(1, "veto", EventKind::VetoApplied(VetoReceipt {
                session_id: "session-1".into(),
                turn_id: "turn-1".into(),
                participant_id: "performer-1".into(),
                authority_valid: false,
                control_sequence: 1,
                before,
                after,
                discarded_unconsumed_notes: 5,
                discarded_application_frames: 480,
            }))),
            Err(LedgerError::VetoAuthorityRejected)
        );
        assert_eq!(ledger.state().epoch, 0);
    }

    #[test]
    fn accepted_veto_revokes_pending_response_and_rejects_late_application() {
        let mut ledger = ledger();
        commit(&mut ledger, 1, decision(TurnAction::Respond, 0, 1));
        let before = initial_state();
        let after = invalidate_state(&before);
        ledger.apply(event(2, "veto", EventKind::VetoApplied(VetoReceipt {
            session_id: "session-1".into(),
            turn_id: "turn-1".into(),
            participant_id: "performer-1".into(),
            authority_valid: true,
            control_sequence: 2,
            before,
            after,
            discarded_unconsumed_notes: 5,
            discarded_application_frames: 480,
        }))).unwrap();
        assert_eq!(ledger.decision_revoked("decision-0"), Some(true));
        assert_eq!(ledger.apply(respond_event("decision-0", 3)), Err(LedgerError::RevokedDecision));
    }

    #[test]
    fn stale_epoch_decision_cannot_commit_after_yield() {
        let mut ledger = ledger();
        commit(&mut ledger, 1, decision(TurnAction::Yield, 0, 1));
        let before = initial_state();
        let after = invalidate_state(&before);
        ledger.apply(event(2, "yield", EventKind::YieldApplied(YieldReceipt {
            decision_id: "decision-0".into(),
            before,
            after,
            discarded_unconsumed_notes: 5,
            discarded_application_frames: 480,
        }))).unwrap();
        let mut stale = decision(TurnAction::Wait, 1, 2);
        stale.output_epoch = 0;
        assert_eq!(
            ledger.apply(event(3, "stale", EventKind::DecisionCommitted(stale))),
            Err(LedgerError::StaleOutputEpoch { expected: 1, received: 0 })
        );
    }

    #[test]
    fn exact_event_replay_is_idempotent_and_conflicting_replay_fails() {
        let mut ledger = ledger();
        let original = event(1, "same-id", EventKind::DecisionCommitted(decision(TurnAction::Wait, 0, 1)));
        ledger.apply(original.clone()).unwrap();
        ledger.apply(original).unwrap();
        let conflict = event(1, "same-id", EventKind::DecisionCommitted(decision(TurnAction::Yield, 0, 1)));
        assert_eq!(
            ledger.apply(conflict),
            Err(LedgerError::ConflictingDuplicateEvent { event_id: "same-id".into() })
        );
    }

    #[test]
    fn evidence_cannot_skip_enqueue_or_callback() {
        let mut ledger = ledger();
        commit(&mut ledger, 1, decision(TurnAction::Respond, 0, 1));
        ledger.apply(respond_event("decision-0", 2)).unwrap();
        let predicted = stage_event("decision-0", "predicted", 3, EffectStage::CpalPredictedPlayback,
            ClockDomain::CpalPlayback, "cpal-playback-1", 2_000);
        assert_eq!(ledger.apply(predicted), Err(LedgerError::InvalidStageTransition));
    }

    #[test]
    fn callback_prediction_and_loopback_are_separate_evidence_stages() {
        let mut ledger = ledger();
        commit(&mut ledger, 1, decision(TurnAction::Respond, 0, 1));
        ledger.apply(respond_event("decision-0", 2)).unwrap();
        ledger.apply(stage_event("decision-0", "enqueue", 3, EffectStage::EnqueuedToApplicationOutput,
            ClockDomain::RunnerMonotonic, RUNNER, 1_200)).unwrap();
        ledger.apply(stage_event("decision-0", "callback", 4, EffectStage::CallbackObserved,
            ClockDomain::CpalCallback, "cpal-callback-1", 1_000)).unwrap();
        ledger.apply(stage_event("decision-0", "prediction", 5, EffectStage::CpalPredictedPlayback,
            ClockDomain::CpalPlayback, "cpal-playback-1", 1_100)).unwrap();
        ledger.apply(stage_event("decision-0", "loopback", 6, EffectStage::LoopbackObserved,
            ClockDomain::IndependentLoopback, "loopback-1", 500)).unwrap();
        let summary = ledger.summary();
        assert_eq!(summary.enqueued, 1);
        assert_eq!(summary.callback_observed, 1);
        assert_eq!(summary.cpal_predicted, 1);
        assert_eq!(summary.loopback_observed, 1);
    }

    #[test]
    fn loopback_may_be_observed_without_cpal_prediction_but_not_without_callback() {
        let mut ledger = ledger();
        commit(&mut ledger, 1, decision(TurnAction::Respond, 0, 1));
        ledger.apply(respond_event("decision-0", 2)).unwrap();
        ledger.apply(stage_event("decision-0", "enqueue", 3, EffectStage::EnqueuedToApplicationOutput,
            ClockDomain::RunnerMonotonic, RUNNER, 1_200)).unwrap();
        ledger.apply(stage_event("decision-0", "callback", 4, EffectStage::CallbackObserved,
            ClockDomain::CpalCallback, "cpal-callback-1", 1_000)).unwrap();
        ledger.apply(stage_event("decision-0", "loopback", 5, EffectStage::LoopbackObserved,
            ClockDomain::IndependentLoopback, "loopback-1", 500)).unwrap();
        assert_eq!(ledger.summary().cpal_predicted, 0);
        assert_eq!(ledger.summary().loopback_observed, 1);
    }

    #[test]
    fn predicted_playback_requires_clock_bridge_digest() {
        let mut ledger = ledger();
        commit(&mut ledger, 1, decision(TurnAction::Respond, 0, 1));
        ledger.apply(respond_event("decision-0", 2)).unwrap();
        ledger.apply(stage_event("decision-0", "enqueue", 3, EffectStage::EnqueuedToApplicationOutput,
            ClockDomain::RunnerMonotonic, RUNNER, 1_200)).unwrap();
        ledger.apply(stage_event("decision-0", "callback", 4, EffectStage::CallbackObserved,
            ClockDomain::CpalCallback, "cpal-callback-1", 1_000)).unwrap();
        let mut predicted = stage_event("decision-0", "prediction", 5, EffectStage::CpalPredictedPlayback,
            ClockDomain::CpalPlayback, "cpal-playback-1", 1_100);
        if let EventKind::EffectObserved(receipt) = &mut predicted.kind {
            receipt.clock_bridge_sha256 = None;
        }
        assert_eq!(ledger.apply(predicted), Err(LedgerError::MissingEvidenceDigest));
    }

    #[test]
    fn loopback_requires_independent_evidence_digest() {
        let mut ledger = ledger();
        commit(&mut ledger, 1, decision(TurnAction::Respond, 0, 1));
        ledger.apply(respond_event("decision-0", 2)).unwrap();
        ledger.apply(stage_event("decision-0", "enqueue", 3, EffectStage::EnqueuedToApplicationOutput,
            ClockDomain::RunnerMonotonic, RUNNER, 1_200)).unwrap();
        ledger.apply(stage_event("decision-0", "callback", 4, EffectStage::CallbackObserved,
            ClockDomain::CpalCallback, "cpal-callback-1", 1_000)).unwrap();
        let mut loopback = stage_event("decision-0", "loopback", 5, EffectStage::LoopbackObserved,
            ClockDomain::IndependentLoopback, "loopback-1", 500);
        if let EventKind::EffectObserved(receipt) = &mut loopback.kind {
            receipt.loopback_evidence_sha256 = None;
        }
        assert_eq!(ledger.apply(loopback), Err(LedgerError::MissingEvidenceDigest));
    }

    #[test]
    fn clock_domains_are_not_interchangeable() {
        let mut ledger = ledger();
        commit(&mut ledger, 1, decision(TurnAction::Respond, 0, 1));
        ledger.apply(respond_event("decision-0", 2)).unwrap();
        let wrong = stage_event("decision-0", "wrong", 3, EffectStage::EnqueuedToApplicationOutput,
            ClockDomain::CpalCallback, "cpal-callback-1", 1_000);
        assert_eq!(ledger.apply(wrong), Err(LedgerError::WrongClockDomain));
    }

    #[test]
    fn callback_admitted_before_veto_can_still_receive_later_playback_evidence() {
        let mut ledger = ledger();
        commit(&mut ledger, 1, decision(TurnAction::Respond, 0, 1));
        ledger.apply(respond_event("decision-0", 2)).unwrap();
        ledger.apply(stage_event("decision-0", "enqueue", 3, EffectStage::EnqueuedToApplicationOutput,
            ClockDomain::RunnerMonotonic, RUNNER, 1_200)).unwrap();
        ledger.apply(stage_event("decision-0", "callback", 4, EffectStage::CallbackObserved,
            ClockDomain::CpalCallback, "cpal-callback-1", 1_000)).unwrap();
        let before = ledger.state().clone();
        let after = invalidate_state(&before);
        ledger.apply(event(5, "veto", EventKind::VetoApplied(VetoReceipt {
            session_id: "session-1".into(),
            turn_id: "turn-1".into(),
            participant_id: "performer-1".into(),
            authority_valid: true,
            control_sequence: 2,
            before,
            after,
            discarded_unconsumed_notes: 0,
            discarded_application_frames: 0,
        }))).unwrap();
        ledger.apply(stage_event("decision-0", "prediction", 6, EffectStage::CpalPredictedPlayback,
            ClockDomain::CpalPlayback, "cpal-playback-1", 1_100)).unwrap();
        assert_eq!(ledger.summary().cpal_predicted, 1);
    }

    #[test]
    fn enqueue_observed_before_veto_does_not_authorize_late_callback_admission() {
        let mut ledger = ledger();
        commit(&mut ledger, 1, decision(TurnAction::Respond, 0, 1));
        ledger.apply(respond_event("decision-0", 2)).unwrap();
        ledger.apply(stage_event("decision-0", "enqueue", 3, EffectStage::EnqueuedToApplicationOutput,
            ClockDomain::RunnerMonotonic, RUNNER, 1_200)).unwrap();
        let before = ledger.state().clone();
        let after = invalidate_state(&before);
        ledger.apply(event(4, "veto", EventKind::VetoApplied(VetoReceipt {
            session_id: "session-1".into(),
            turn_id: "turn-1".into(),
            participant_id: "performer-1".into(),
            authority_valid: true,
            control_sequence: 2,
            before,
            after,
            discarded_unconsumed_notes: 0,
            discarded_application_frames: 0,
        }))).unwrap();
        let callback = stage_event("decision-0", "callback", 5, EffectStage::CallbackObserved,
            ClockDomain::CpalCallback, "cpal-callback-1", 1_000);
        assert_eq!(ledger.apply(callback), Err(LedgerError::RevokedDecision));
    }

    #[test]
    fn turn_close_requires_exact_contiguous_chunk_accounting() {
        let mut ledger = ledger();
        commit(&mut ledger, 1, decision(TurnAction::Wait, 0, 1));
        commit(&mut ledger, 2, decision(TurnAction::Continue, 1, 2));
        ledger.apply(event(3, "close", EventKind::FinalizeTurn {
            session_id: "session-1".into(),
            turn_id: "turn-1".into(),
            chunk_count: 2,
        })).unwrap();
    }

    #[test]
    fn decision_rejects_declared_lookahead_and_wrong_clock_origin() {
        let mut ledger = ledger();
        let mut lookahead = decision(TurnAction::Wait, 0, 1);
        lookahead.observed_through_frame -= 1;
        assert_eq!(
            ledger.apply(event(1, "lookahead", EventKind::DecisionCommitted(lookahead))),
            Err(LedgerError::LookaheadOrObservedThroughMismatch)
        );
        let mut wrong_clock = decision(TurnAction::Wait, 0, 1);
        wrong_clock.runner_clock_origin_id = "different-runner".into();
        assert_eq!(
            ledger.apply(event(1, "wrong-clock", EventKind::DecisionCommitted(wrong_clock))),
            Err(LedgerError::ClockOriginMismatch)
        );
    }
}
