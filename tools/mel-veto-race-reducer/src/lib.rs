//! Dependency-free reference reducer for the Melothaea veto/pending-output race.
//!
//! This models the application-queue epoch boundary. It is not the production
//! CPAL callback and does not prove physical cancellation or acoustic silence.
//! Trace order is the modeled linearization order of control updates and
//! callback epoch observations.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RaceError {
    EmptyIdentity,
    InvalidDigest,
    InvalidFrameRange,
    OutOfOrderEvent {
        previous: u64,
        received: u64,
    },
    ConflictingDuplicateEvent {
        event_id: String,
    },
    DuplicateAuthorization {
        authorization_id: String,
    },
    UnknownAuthorization {
        authorization_id: String,
    },
    RevokedAuthorization {
        authorization_id: String,
    },
    AuthorizationExpired {
        authorization_id: String,
        at_ns: u64,
        expires_at_ns: u64,
    },
    ExpirationBeforeDeadline {
        authorization_id: String,
        observed_at_ns: u64,
        expires_at_ns: u64,
    },
    InvalidAuthorizationTime {
        authorized_at_ns: u64,
        expires_at_ns: u64,
    },
    ProducerTimeBeforeAuthorization {
        producer_id: String,
        started_at_ns: u64,
        authorized_at_ns: u64,
    },
    ProducerTimeAfterExpiry {
        producer_id: String,
        started_at_ns: u64,
        expires_at_ns: u64,
    },
    RenderCompletionBeforeProducerStart {
        producer_id: String,
        started_at_ns: u64,
        completed_at_ns: u64,
    },
    EnqueueBeforeRenderCompletion {
        frame_id: u64,
        completed_at_ns: u64,
        enqueued_at_ns: u64,
    },
    ControlSequenceRegression {
        previous: u64,
        received: u64,
    },
    AuthorizationEpochMismatch {
        expected: u64,
        received: u64,
    },
    VetoAuthorityRejected,
    VetoTurnMismatch,
    VetoEpochMismatch {
        expected: u64,
        prior: u64,
        next: u64,
    },
    EpochExhausted,
    DuplicateFrame {
        frame_id: u64,
    },
    OverlappingFrameRange {
        existing_frame_id: u64,
        incoming_frame_id: u64,
    },
    DuplicateProducerOperation {
        producer_id: String,
    },
    UnknownProducerOperation {
        producer_id: String,
    },
    ProducerAlreadyCompleted {
        producer_id: String,
    },
    ProducerBindingMismatch {
        producer_id: String,
    },
    UnknownFrame {
        frame_id: u64,
    },
    FrameBindingMismatch {
        frame_id: u64,
    },
    FrameAlreadyEnqueued {
        frame_id: u64,
    },
    FrameNotEnqueued {
        frame_id: u64,
    },
    FrameAlreadyTerminal {
        frame_id: u64,
    },
    CallbackEpochRegression {
        expected_at_least: u64,
        observed: u64,
    },
    CallbackObservedFutureEpoch {
        current: u64,
        observed: u64,
    },
}

impl std::fmt::Display for RaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for RaceError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameRange {
    pub start: u64,
    pub end: u64,
}

impl FrameRange {
    pub fn is_valid(self) -> bool {
        self.start < self.end
    }

    pub fn len(self) -> u64 {
        self.end - self.start
    }

    pub fn is_empty(self) -> bool {
        self.start >= self.end
    }

    pub fn overlaps(self, other: Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Authorization {
    pub authorization_id: String,
    pub session_id: String,
    pub turn_id: String,
    pub decision_id: String,
    pub plan_sha256: String,
    pub epoch: u64,
    pub control_sequence: u64,
    pub authorized_at_ns: u64,
    pub expires_at_ns: u64,
    pub revoked: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameDisposition {
    CallbackAdmitted { observed_epoch: u64 },
    StaleDiscarded { observed_epoch: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedFrameRange {
    pub frame_id: u64,
    pub range: FrameRange,
    pub authorization_id: String,
    pub plan_sha256: String,
    pub captured_epoch: u64,
    pub completed_at_ns: u64,
    pub enqueued_at_ns: Option<u64>,
    pub enqueued: bool,
    pub disposition: Option<FrameDisposition>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProducerOperation {
    pub producer_id: String,
    pub authorization_id: String,
    pub plan_sha256: String,
    pub captured_epoch: u64,
    pub started_order: u64,
    pub started_at_ns: u64,
    pub completed_at_ns: Option<u64>,
    pub completed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventKind {
    Authorize {
        authorization_id: String,
        session_id: String,
        turn_id: String,
        decision_id: String,
        plan_sha256: String,
        epoch: u64,
        control_sequence: u64,
        authorized_at_ns: u64,
        expires_at_ns: u64,
    },
    ProducerStarted {
        producer_id: String,
        authorization_id: String,
        plan_sha256: String,
        captured_epoch: u64,
        started_at_ns: u64,
    },
    RenderCompleted {
        producer_id: String,
        frame_id: u64,
        range: FrameRange,
        completed_at_ns: u64,
    },
    Enqueue {
        frame_id: u64,
        enqueued_at_ns: u64,
    },
    Veto {
        authorization_id: String,
        session_id: String,
        turn_id: String,
        participant_id: String,
        authority_valid: bool,
        control_sequence: u64,
        prior_epoch: u64,
        new_epoch: u64,
    },
    ExpireAuthorization {
        authorization_id: String,
        session_id: String,
        turn_id: String,
        control_sequence: u64,
        observed_at_ns: u64,
        prior_epoch: u64,
        new_epoch: u64,
    },
    CallbackEpochCheck {
        frame_id: u64,
        observed_epoch: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LedgerEvent {
    /// Strictly increasing modeled linearization order; not wall-clock time.
    pub order: u64,
    /// Stable idempotency identity for exact replay.
    pub event_id: String,
    pub kind: EventKind,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RaceSummary {
    pub current_epoch: u64,
    pub authorization_count: usize,
    pub revoked_authorization_count: usize,
    pub producer_started_count: usize,
    pub producer_completed_count: usize,
    pub producer_in_flight_count: usize,
    pub rendered_frame_ranges: usize,
    pub rendered_frames: u64,
    pub rendered_not_enqueued_ranges: usize,
    pub rendered_not_enqueued_frames: u64,
    pub enqueued_frame_ranges: usize,
    pub enqueued_frames: u64,
    pub callback_admitted_ranges: usize,
    pub callback_admitted_frames: u64,
    pub stale_discarded_ranges: usize,
    pub stale_discarded_frames: u64,
    pub pending_enqueued_ranges: usize,
    pub pending_enqueued_frames: u64,
}

#[derive(Debug, Default)]
pub struct RaceLedger {
    current_epoch: u64,
    last_event_order: Option<u64>,
    last_control_sequence: u64,
    authorizations: BTreeMap<String, Authorization>,
    producers: BTreeMap<String, ProducerOperation>,
    frames: BTreeMap<u64, RenderedFrameRange>,
    seen_events: BTreeMap<String, LedgerEvent>,
}

impl RaceLedger {
    pub fn new(initial_epoch: u64) -> Self {
        Self {
            current_epoch: initial_epoch,
            ..Self::default()
        }
    }

    pub fn current_epoch(&self) -> u64 {
        self.current_epoch
    }

    pub fn authorization(&self, authorization_id: &str) -> Option<&Authorization> {
        self.authorizations.get(authorization_id)
    }

    pub fn frame(&self, frame_id: u64) -> Option<&RenderedFrameRange> {
        self.frames.get(&frame_id)
    }

    pub fn producer(&self, producer_id: &str) -> Option<&ProducerOperation> {
        self.producers.get(producer_id)
    }

    /// Apply one event in modeled linearization order.
    ///
    /// Exact duplicate event IDs are idempotent even if replayed after the
    /// original. Reusing an event ID with a different payload is rejected.
    pub fn apply(&mut self, event: LedgerEvent) -> Result<(), RaceError> {
        if event.event_id.trim().is_empty() {
            return Err(RaceError::EmptyIdentity);
        }

        if let Some(previous) = self.seen_events.get(&event.event_id) {
            return if previous == &event {
                Ok(())
            } else {
                Err(RaceError::ConflictingDuplicateEvent {
                    event_id: event.event_id,
                })
            };
        }

        if let Some(previous) = self.last_event_order {
            if event.order <= previous {
                return Err(RaceError::OutOfOrderEvent {
                    previous,
                    received: event.order,
                });
            }
        }

        self.apply_kind(&event.kind, event.order)?;
        self.last_event_order = Some(event.order);
        self.seen_events.insert(event.event_id.clone(), event);
        Ok(())
    }

    fn apply_kind(&mut self, kind: &EventKind, event_order: u64) -> Result<(), RaceError> {
        match kind {
            EventKind::Authorize {
                authorization_id,
                session_id,
                turn_id,
                decision_id,
                plan_sha256,
                epoch,
                control_sequence,
                authorized_at_ns,
                expires_at_ns,
            } => {
                require_identity(authorization_id)?;
                require_identity(session_id)?;
                require_identity(turn_id)?;
                require_identity(decision_id)?;
                require_digest(plan_sha256)?;

                if self.authorizations.contains_key(authorization_id) {
                    return Err(RaceError::DuplicateAuthorization {
                        authorization_id: authorization_id.clone(),
                    });
                }
                self.check_control_sequence(*control_sequence)?;
                if *epoch != self.current_epoch {
                    return Err(RaceError::AuthorizationEpochMismatch {
                        expected: self.current_epoch,
                        received: *epoch,
                    });
                }
                if authorized_at_ns >= expires_at_ns {
                    return Err(RaceError::InvalidAuthorizationTime {
                        authorized_at_ns: *authorized_at_ns,
                        expires_at_ns: *expires_at_ns,
                    });
                }

                self.last_control_sequence = *control_sequence;
                self.authorizations.insert(
                    authorization_id.clone(),
                    Authorization {
                        authorization_id: authorization_id.clone(),
                        session_id: session_id.clone(),
                        turn_id: turn_id.clone(),
                        decision_id: decision_id.clone(),
                        plan_sha256: plan_sha256.clone(),
                        epoch: *epoch,
                        control_sequence: *control_sequence,
                        authorized_at_ns: *authorized_at_ns,
                        expires_at_ns: *expires_at_ns,
                        revoked: false,
                    },
                );
            }
            EventKind::ProducerStarted {
                producer_id,
                authorization_id,
                plan_sha256,
                captured_epoch,
                started_at_ns,
            } => {
                require_identity(producer_id)?;
                if self.producers.contains_key(producer_id) {
                    return Err(RaceError::DuplicateProducerOperation {
                        producer_id: producer_id.clone(),
                    });
                }
                let auth = self.authorizations.get(authorization_id).ok_or_else(|| {
                    RaceError::UnknownAuthorization {
                        authorization_id: authorization_id.clone(),
                    }
                })?;
                if auth.revoked {
                    return Err(RaceError::RevokedAuthorization {
                        authorization_id: authorization_id.clone(),
                    });
                }
                if *started_at_ns < auth.authorized_at_ns {
                    return Err(RaceError::ProducerTimeBeforeAuthorization {
                        producer_id: producer_id.clone(),
                        started_at_ns: *started_at_ns,
                        authorized_at_ns: auth.authorized_at_ns,
                    });
                }
                if *started_at_ns >= auth.expires_at_ns {
                    return Err(RaceError::ProducerTimeAfterExpiry {
                        producer_id: producer_id.clone(),
                        started_at_ns: *started_at_ns,
                        expires_at_ns: auth.expires_at_ns,
                    });
                }
                if auth.control_sequence != self.last_control_sequence {
                    return Err(RaceError::RevokedAuthorization {
                        authorization_id: authorization_id.clone(),
                    });
                }
                if auth.plan_sha256 != *plan_sha256
                    || auth.epoch != *captured_epoch
                    || *captured_epoch != self.current_epoch
                {
                    return Err(RaceError::ProducerBindingMismatch {
                        producer_id: producer_id.clone(),
                    });
                }
                self.producers.insert(
                    producer_id.clone(),
                    ProducerOperation {
                        producer_id: producer_id.clone(),
                        authorization_id: authorization_id.clone(),
                        plan_sha256: plan_sha256.clone(),
                        captured_epoch: *captured_epoch,
                        started_order: event_order,
                        started_at_ns: *started_at_ns,
                        completed_at_ns: None,
                        completed: false,
                    },
                );
            }
            EventKind::RenderCompleted {
                producer_id,
                frame_id,
                range,
                completed_at_ns,
            } => {
                if !range.is_valid() {
                    return Err(RaceError::InvalidFrameRange);
                }
                if self.frames.contains_key(frame_id) {
                    return Err(RaceError::DuplicateFrame {
                        frame_id: *frame_id,
                    });
                }
                let producer = self
                    .producers
                    .get_mut(producer_id)
                    .ok_or_else(|| RaceError::UnknownProducerOperation {
                        producer_id: producer_id.clone(),
                    })?;
                if producer.completed {
                    return Err(RaceError::ProducerAlreadyCompleted {
                        producer_id: producer_id.clone(),
                    });
                }
                let authorization_id = producer.authorization_id.clone();
                let plan_sha256 = producer.plan_sha256.clone();
                let captured_epoch = producer.captured_epoch;
                let started_at_ns = producer.started_at_ns;
                if *completed_at_ns < started_at_ns {
                    return Err(RaceError::RenderCompletionBeforeProducerStart {
                        producer_id: producer_id.clone(),
                        started_at_ns,
                        completed_at_ns: *completed_at_ns,
                    });
                }
                // Completion is allowed after veto only because this producer
                // captured the active authorization/epoch before revocation.
                if let Some((existing_frame_id, _)) = self.frames.iter().find(|(_, existing)| {
                    existing.range.overlaps(*range)
                }) {
                    return Err(RaceError::OverlappingFrameRange {
                        existing_frame_id: *existing_frame_id,
                        incoming_frame_id: *frame_id,
                    });
                }
                producer.completed = true;
                producer.completed_at_ns = Some(*completed_at_ns);
                self.frames.insert(
                    *frame_id,
                    RenderedFrameRange {
                        frame_id: *frame_id,
                        range: *range,
                        authorization_id,
                        plan_sha256,
                        captured_epoch,
                        completed_at_ns: *completed_at_ns,
                        enqueued_at_ns: None,
                        enqueued: false,
                        disposition: None,
                    },
                );
            }
            EventKind::Enqueue {
                frame_id,
                enqueued_at_ns,
            } => {
                let (authorization_id, completed_at_ns, already_enqueued) = {
                    let frame = self
                        .frames
                        .get(frame_id)
                        .ok_or(RaceError::UnknownFrame {
                            frame_id: *frame_id,
                        })?;
                    (
                        frame.authorization_id.clone(),
                        frame.completed_at_ns,
                        frame.enqueued,
                    )
                };
                if already_enqueued {
                    return Err(RaceError::FrameAlreadyEnqueued {
                        frame_id: *frame_id,
                    });
                }
                if *enqueued_at_ns < completed_at_ns {
                    return Err(RaceError::EnqueueBeforeRenderCompletion {
                        frame_id: *frame_id,
                        completed_at_ns,
                        enqueued_at_ns: *enqueued_at_ns,
                    });
                }
                let auth = self
                    .authorizations
                    .get(&authorization_id)
                    .ok_or_else(|| RaceError::UnknownAuthorization {
                        authorization_id: authorization_id.clone(),
                    })?;
                // A producer that finishes after veto may still publish its
                // captured old-epoch work so the callback can explicitly drain
                // and account for it. Ordinary expiry rejects enqueue rather
                // than silently extending authorization.
                if !auth.revoked && *enqueued_at_ns >= auth.expires_at_ns {
                    return Err(RaceError::AuthorizationExpired {
                        authorization_id,
                        at_ns: *enqueued_at_ns,
                        expires_at_ns: auth.expires_at_ns,
                    });
                }
                let frame = self.frames.get_mut(frame_id).expect("frame checked above");
                frame.enqueued = true;
                frame.enqueued_at_ns = Some(*enqueued_at_ns);
            }
            EventKind::Veto {
                authorization_id,
                session_id,
                turn_id,
                participant_id,
                authority_valid,
                control_sequence,
                prior_epoch,
                new_epoch,
            } => {
                require_identity(participant_id)?;
                if !authority_valid {
                    return Err(RaceError::VetoAuthorityRejected);
                }
                self.check_control_sequence(*control_sequence)?;
                let auth = self
                    .authorizations
                    .get_mut(authorization_id)
                    .ok_or_else(|| RaceError::UnknownAuthorization {
                        authorization_id: authorization_id.clone(),
                    })?;
                if auth.session_id != *session_id || auth.turn_id != *turn_id {
                    return Err(RaceError::VetoTurnMismatch);
                }
                if auth.revoked {
                    return Err(RaceError::RevokedAuthorization {
                        authorization_id: authorization_id.clone(),
                    });
                }
                let expected_next = self
                    .current_epoch
                    .checked_add(1)
                    .ok_or(RaceError::EpochExhausted)?;
                if *prior_epoch != self.current_epoch || *new_epoch != expected_next {
                    return Err(RaceError::VetoEpochMismatch {
                        expected: self.current_epoch,
                        prior: *prior_epoch,
                        next: *new_epoch,
                    });
                }
                auth.revoked = true;
                self.current_epoch = *new_epoch;
                self.last_control_sequence = *control_sequence;
            }
            EventKind::ExpireAuthorization {
                authorization_id,
                session_id,
                turn_id,
                control_sequence,
                observed_at_ns,
                prior_epoch,
                new_epoch,
            } => {
                self.check_control_sequence(*control_sequence)?;
                let auth = self
                    .authorizations
                    .get_mut(authorization_id)
                    .ok_or_else(|| RaceError::UnknownAuthorization {
                        authorization_id: authorization_id.clone(),
                    })?;
                if auth.session_id != *session_id || auth.turn_id != *turn_id {
                    return Err(RaceError::VetoTurnMismatch);
                }
                if auth.revoked {
                    return Err(RaceError::RevokedAuthorization {
                        authorization_id: authorization_id.clone(),
                    });
                }
                if *observed_at_ns < auth.expires_at_ns {
                    return Err(RaceError::ExpirationBeforeDeadline {
                        authorization_id: authorization_id.clone(),
                        observed_at_ns: *observed_at_ns,
                        expires_at_ns: auth.expires_at_ns,
                    });
                }
                let expected_next = self
                    .current_epoch
                    .checked_add(1)
                    .ok_or(RaceError::EpochExhausted)?;
                if *prior_epoch != self.current_epoch || *new_epoch != expected_next {
                    return Err(RaceError::VetoEpochMismatch {
                        expected: self.current_epoch,
                        prior: *prior_epoch,
                        next: *new_epoch,
                    });
                }
                auth.revoked = true;
                self.current_epoch = *new_epoch;
                self.last_control_sequence = *control_sequence;
            }
            EventKind::CallbackEpochCheck {
                frame_id,
                observed_epoch,
            } => {
                let frame = self
                    .frames
                    .get_mut(frame_id)
                    .ok_or(RaceError::UnknownFrame {
                        frame_id: *frame_id,
                    })?;
                if !frame.enqueued {
                    return Err(RaceError::FrameNotEnqueued {
                        frame_id: *frame_id,
                    });
                }
                if frame.disposition.is_some() {
                    return Err(RaceError::FrameAlreadyTerminal {
                        frame_id: *frame_id,
                    });
                }
                if *observed_epoch < self.current_epoch {
                    return Err(RaceError::CallbackEpochRegression {
                        expected_at_least: self.current_epoch,
                        observed: *observed_epoch,
                    });
                }
                if *observed_epoch > self.current_epoch {
                    return Err(RaceError::CallbackObservedFutureEpoch {
                        current: self.current_epoch,
                        observed: *observed_epoch,
                    });
                }
                if frame.captured_epoch == *observed_epoch {
                    frame.disposition = Some(FrameDisposition::CallbackAdmitted {
                        observed_epoch: *observed_epoch,
                    });
                } else if frame.captured_epoch < *observed_epoch {
                    frame.disposition = Some(FrameDisposition::StaleDiscarded {
                        observed_epoch: *observed_epoch,
                    });
                } else {
                    return Err(RaceError::CallbackEpochRegression {
                        expected_at_least: frame.captured_epoch,
                        observed: *observed_epoch,
                    });
                }
            }
        }
        Ok(())
    }

    fn check_control_sequence(&self, received: u64) -> Result<(), RaceError> {
        if received <= self.last_control_sequence {
            return Err(RaceError::ControlSequenceRegression {
                previous: self.last_control_sequence,
                received,
            });
        }
        Ok(())
    }

    pub fn summary(&self) -> RaceSummary {
        let mut summary = RaceSummary {
            current_epoch: self.current_epoch,
            authorization_count: self.authorizations.len(),
            revoked_authorization_count: self.authorizations.values().filter(|a| a.revoked).count(),
            producer_started_count: self.producers.len(),
            producer_completed_count: self.producers.values().filter(|p| p.completed).count(),
            producer_in_flight_count: self.producers.values().filter(|p| !p.completed).count(),
            rendered_frame_ranges: self.frames.len(),
            ..RaceSummary::default()
        };

        for frame in self.frames.values() {
            let count = frame.range.len();
            summary.rendered_frames = summary.rendered_frames.saturating_add(count);
            if frame.enqueued {
                summary.enqueued_frame_ranges += 1;
                summary.enqueued_frames = summary.enqueued_frames.saturating_add(count);
            } else {
                summary.rendered_not_enqueued_ranges += 1;
                summary.rendered_not_enqueued_frames =
                    summary.rendered_not_enqueued_frames.saturating_add(count);
            }
            match frame.disposition {
                Some(FrameDisposition::CallbackAdmitted { .. }) => {
                    summary.callback_admitted_ranges += 1;
                    summary.callback_admitted_frames =
                        summary.callback_admitted_frames.saturating_add(count);
                }
                Some(FrameDisposition::StaleDiscarded { .. }) => {
                    summary.stale_discarded_ranges += 1;
                    summary.stale_discarded_frames =
                        summary.stale_discarded_frames.saturating_add(count);
                }
                None if frame.enqueued => {
                    summary.pending_enqueued_ranges += 1;
                    summary.pending_enqueued_frames =
                        summary.pending_enqueued_frames.saturating_add(count);
                }
                None => {}
            }
        }
        summary
    }
}

fn require_identity(value: &str) -> Result<(), RaceError> {
    if value.trim().is_empty() {
        Err(RaceError::EmptyIdentity)
    } else {
        Ok(())
    }
}

fn require_digest(value: &str) -> Result<(), RaceError> {
    if value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        Ok(())
    } else {
        Err(RaceError::InvalidDigest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PLAN_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn event(order: u64, event_id: &str, kind: EventKind) -> LedgerEvent {
        LedgerEvent {
            order,
            event_id: event_id.into(),
            kind,
        }
    }

    fn authorization_event(control_sequence: u64) -> LedgerEvent {
        authorization_event_at(control_sequence, control_sequence)
    }

    fn authorization_event_at(order: u64, control_sequence: u64) -> LedgerEvent {
        event(
            order,
            &format!("authorize-{control_sequence}"),
            EventKind::Authorize {
                authorization_id: format!("auth-{control_sequence}"),
                session_id: "session-1".into(),
                turn_id: "turn-1".into(),
                decision_id: format!("decision-{control_sequence}"),
                plan_sha256: PLAN_A.into(),
                epoch: if control_sequence == 1 { 0 } else { 1 },
                control_sequence,
                authorized_at_ns: 100,
                expires_at_ns: 10_000_000_000,
            },
        )
    }

    fn producer_start_event(order: u64, producer_id: &str, epoch: u64) -> LedgerEvent {
        event(
            order,
            &format!("producer-start-{producer_id}"),
            EventKind::ProducerStarted {
                producer_id: producer_id.into(),
                authorization_id: "auth-1".into(),
                plan_sha256: PLAN_A.into(),
                captured_epoch: epoch,
                started_at_ns: 200,
            },
        )
    }

    fn render_complete_event(order: u64, producer_id: &str, frame_id: u64) -> LedgerEvent {
        event(
            order,
            &format!("render-complete-{frame_id}"),
            EventKind::RenderCompleted {
                producer_id: producer_id.into(),
                frame_id,
                range: FrameRange {
                    start: frame_id * 100,
                    end: frame_id * 100 + 100,
                },
                completed_at_ns: 300,
            },
        )
    }

    fn enqueue_event(order: u64, frame_id: u64) -> LedgerEvent {
        event(
            order,
            &format!("enqueue-{frame_id}"),
            EventKind::Enqueue {
                frame_id,
                enqueued_at_ns: 400,
            },
        )
    }

    fn veto_event(order: u64) -> LedgerEvent {
        event(
            order,
            "veto-1",
            EventKind::Veto {
                authorization_id: "auth-1".into(),
                session_id: "session-1".into(),
                turn_id: "turn-1".into(),
                participant_id: "performer-1".into(),
                authority_valid: true,
                control_sequence: 2,
                prior_epoch: 0,
                new_epoch: 1,
            },
        )
    }

    fn expiration_event(order: u64, observed_at_ns: u64) -> LedgerEvent {
        event(
            order,
            "expire-auth-1",
            EventKind::ExpireAuthorization {
                authorization_id: "auth-1".into(),
                session_id: "session-1".into(),
                turn_id: "turn-1".into(),
                control_sequence: 2,
                observed_at_ns,
                prior_epoch: 0,
                new_epoch: 1,
            },
        )
    }

    fn callback_event(order: u64, frame_id: u64, observed_epoch: u64) -> LedgerEvent {
        event(
            order,
            &format!("callback-{frame_id}"),
            EventKind::CallbackEpochCheck {
                frame_id,
                observed_epoch,
            },
        )
    }

    fn authorized_ledger() -> RaceLedger {
        let mut ledger = RaceLedger::new(0);
        ledger.apply(authorization_event(1)).unwrap();
        ledger
    }

    #[test]
    fn veto_before_callback_discards_late_enqueued_old_epoch_frames() {
        let mut ledger = authorized_ledger();
        ledger.apply(producer_start_event(2, "producer-7", 0)).unwrap();
        ledger.apply(veto_event(3)).unwrap();
        ledger.apply(render_complete_event(4, "producer-7", 7)).unwrap();
        ledger.apply(enqueue_event(5, 7)).unwrap();
        ledger.apply(callback_event(6, 7, 1)).unwrap();

        let frame = ledger.frame(7).unwrap();
        assert_eq!(frame.captured_epoch, 0);
        assert_eq!(
            frame.disposition,
            Some(FrameDisposition::StaleDiscarded { observed_epoch: 1 })
        );
        assert_eq!(
            ledger.summary(),
            RaceSummary {
                current_epoch: 1,
                authorization_count: 1,
                revoked_authorization_count: 1,
                producer_started_count: 1,
                producer_completed_count: 1,
                producer_in_flight_count: 0,
                rendered_frame_ranges: 1,
                rendered_frames: 100,
                rendered_not_enqueued_ranges: 0,
                rendered_not_enqueued_frames: 0,
                enqueued_frame_ranges: 1,
                enqueued_frames: 100,
                callback_admitted_ranges: 0,
                callback_admitted_frames: 0,
                stale_discarded_ranges: 1,
                stale_discarded_frames: 100,
                pending_enqueued_ranges: 0,
                pending_enqueued_frames: 0,
            }
        );
    }

    #[test]
    fn callback_admitted_before_veto_remains_admitted_and_is_not_retroactively_cancelled() {
        let mut ledger = authorized_ledger();
        ledger.apply(producer_start_event(2, "producer-7", 0)).unwrap();
        ledger.apply(render_complete_event(3, "producer-7", 7)).unwrap();
        ledger.apply(enqueue_event(4, 7)).unwrap();
        ledger.apply(callback_event(5, 7, 0)).unwrap();
        ledger.apply(veto_event(6)).unwrap();

        assert_eq!(
            ledger.frame(7).unwrap().disposition,
            Some(FrameDisposition::CallbackAdmitted { observed_epoch: 0 })
        );
        assert_eq!(ledger.summary().callback_admitted_frames, 100);
        assert_eq!(ledger.summary().stale_discarded_frames, 0);
        assert_eq!(ledger.summary().revoked_authorization_count, 1);
    }

    #[test]
    fn callback_that_observes_new_epoch_never_admits_old_frame() {
        let mut ledger = authorized_ledger();
        ledger.apply(producer_start_event(2, "producer-7", 0)).unwrap();
        ledger.apply(veto_event(3)).unwrap();
        ledger.apply(render_complete_event(4, "producer-7", 7)).unwrap();
        ledger.apply(enqueue_event(5, 7)).unwrap();
        ledger.apply(callback_event(6, 7, 1)).unwrap();

        assert_eq!(
            ledger.frame(7).unwrap().disposition,
            Some(FrameDisposition::StaleDiscarded { observed_epoch: 1 })
        );
        assert_eq!(ledger.summary().callback_admitted_frames, 0);
    }

    #[test]
    fn producer_cannot_retag_old_render_with_new_epoch() {
        let mut ledger = authorized_ledger();
        ledger.apply(producer_start_event(2, "producer-7", 0)).unwrap();
        ledger.apply(veto_event(3)).unwrap();
        // Already-started work may complete, but its captured epoch is fixed.
        ledger.apply(render_complete_event(4, "producer-7", 7)).unwrap();

        let result = ledger.apply(event(
            5,
            "producer-start-after-veto",
            EventKind::ProducerStarted {
                producer_id: "producer-8".into(),
                authorization_id: "auth-1".into(),
                plan_sha256: PLAN_A.into(),
                captured_epoch: 1,
                started_at_ns: 500,
            },
        ));
        assert_eq!(
            result,
            Err(RaceError::RevokedAuthorization {
                authorization_id: "auth-1".into(),
            })
        );
    }

    #[test]
    fn conflicting_duplicate_event_id_is_rejected_but_exact_replay_is_idempotent() {
        let mut ledger = authorized_ledger();
        let original = authorization_event(1);
        // The original authorization has already been applied; exact replay
        // is accepted even though its order is no longer the next event.
        ledger.apply(original.clone()).unwrap();

        let mut conflicting = original;
        conflicting.kind = EventKind::Authorize {
            authorization_id: "auth-other".into(),
            session_id: "session-1".into(),
            turn_id: "turn-1".into(),
            decision_id: "decision-other".into(),
            plan_sha256: PLAN_A.into(),
            epoch: 0,
            control_sequence: 1,
            authorized_at_ns: 100,
            expires_at_ns: 10_000_000_000,
        };
        assert!(matches!(
            ledger.apply(conflicting),
            Err(RaceError::ConflictingDuplicateEvent { .. })
        ));
    }

    #[test]
    fn new_authorization_uses_new_epoch_and_old_frames_stay_stale() {
        let mut ledger = authorized_ledger();
        ledger.apply(producer_start_event(2, "producer-7", 0)).unwrap();
        ledger.apply(veto_event(3)).unwrap();
        ledger.apply(render_complete_event(4, "producer-7", 7)).unwrap();
        ledger.apply(enqueue_event(5, 7)).unwrap();
        ledger.apply(callback_event(6, 7, 1)).unwrap();

        ledger.apply(authorization_event_at(7, 6)).unwrap();
        let new_authorization = ledger.authorization("auth-6").unwrap();
        assert_eq!(new_authorization.epoch, 1);
        assert!(!new_authorization.revoked);
        assert_eq!(
            ledger.frame(7).unwrap().disposition,
            Some(FrameDisposition::StaleDiscarded { observed_epoch: 1 })
        );
    }

    #[test]
    fn veto_from_wrong_turn_or_without_authority_is_rejected() {
        let mut wrong_turn = authorized_ledger();
        let wrong_turn_result = wrong_turn.apply(event(
            2,
            "wrong-turn-veto",
            EventKind::Veto {
                authorization_id: "auth-1".into(),
                session_id: "session-1".into(),
                turn_id: "other-turn".into(),
                participant_id: "performer-1".into(),
                authority_valid: true,
                control_sequence: 2,
                prior_epoch: 0,
                new_epoch: 1,
            },
        ));
        assert_eq!(wrong_turn_result, Err(RaceError::VetoTurnMismatch));

        let mut no_authority = authorized_ledger();
        let no_authority_result = no_authority.apply(event(
            2,
            "unauthorized-veto",
            EventKind::Veto {
                authorization_id: "auth-1".into(),
                session_id: "session-1".into(),
                turn_id: "turn-1".into(),
                participant_id: "performer-1".into(),
                authority_valid: false,
                control_sequence: 2,
                prior_epoch: 0,
                new_epoch: 1,
            },
        ));
        assert_eq!(no_authority_result, Err(RaceError::VetoAuthorityRejected));
    }

    #[test]
    fn missing_enqueue_or_duplicate_terminal_check_fails_closed() {
        let mut no_enqueue = authorized_ledger();
        no_enqueue
            .apply(producer_start_event(2, "producer-7", 0))
            .unwrap();
        no_enqueue
            .apply(render_complete_event(3, "producer-7", 7))
            .unwrap();
        assert_eq!(
            no_enqueue.apply(callback_event(4, 7, 0)),
            Err(RaceError::FrameNotEnqueued { frame_id: 7 })
        );

        let mut duplicate_terminal = authorized_ledger();
        duplicate_terminal
            .apply(producer_start_event(2, "producer-7", 0))
            .unwrap();
        duplicate_terminal
            .apply(render_complete_event(3, "producer-7", 7))
            .unwrap();
        duplicate_terminal.apply(enqueue_event(4, 7)).unwrap();
        duplicate_terminal.apply(callback_event(5, 7, 0)).unwrap();
        assert_eq!(
            duplicate_terminal.apply(callback_event(6, 7, 0)),
            Err(RaceError::FrameAlreadyTerminal { frame_id: 7 })
        );
    }

    #[test]
    fn overlapping_rendered_ranges_reject_without_consuming_the_producer_operation() {
        let mut ledger = authorized_ledger();
        ledger
            .apply(producer_start_event(2, "producer-1", 0))
            .unwrap();
        ledger
            .apply(render_complete_event(3, "producer-1", 1))
            .unwrap();
        ledger
            .apply(producer_start_event(4, "producer-2", 0))
            .unwrap();

        let overlap = event(
            5,
            "overlap-attempt",
            EventKind::RenderCompleted {
                producer_id: "producer-2".into(),
                frame_id: 2,
                range: FrameRange {
                    start: 150,
                    end: 250,
                },
                completed_at_ns: 300,
            },
        );
        assert_eq!(
            ledger.apply(overlap),
            Err(RaceError::OverlappingFrameRange {
                existing_frame_id: 1,
                incoming_frame_id: 2,
            })
        );
        assert!(!ledger.producer("producer-2").unwrap().completed);

        // A rejected event must not partially consume the operation or the
        // linearization order; corrected evidence can be replayed at order 5.
        ledger
            .apply(event(
                5,
                "non-overlapping-completion",
                EventKind::RenderCompleted {
                    producer_id: "producer-2".into(),
                    frame_id: 2,
                    range: FrameRange {
                        start: 200,
                        end: 300,
                    },
                    completed_at_ns: 300,
                },
            ))
            .unwrap();
        assert!(ledger.producer("producer-2").unwrap().completed);
        assert_eq!(ledger.summary().rendered_frames, 200);
    }

    #[test]
    fn expiry_invalidates_queued_old_epoch_frames_before_callback_admission() {
        let mut ledger = authorized_ledger();
        ledger
            .apply(producer_start_event(2, "producer-7", 0))
            .unwrap();
        ledger
            .apply(render_complete_event(3, "producer-7", 7))
            .unwrap();
        ledger.apply(enqueue_event(4, 7)).unwrap();
        ledger.apply(expiration_event(5, 10_000_000_000)).unwrap();
        ledger.apply(callback_event(6, 7, 1)).unwrap();

        assert_eq!(
            ledger.frame(7).unwrap().disposition,
            Some(FrameDisposition::StaleDiscarded { observed_epoch: 1 })
        );
        assert_eq!(ledger.summary().stale_discarded_frames, 100);
        assert_eq!(ledger.summary().callback_admitted_frames, 0);
        assert_eq!(ledger.summary().revoked_authorization_count, 1);
    }

    #[test]
    fn callback_admitted_before_expiry_is_not_retroactively_cancelled() {
        let mut ledger = authorized_ledger();
        ledger
            .apply(producer_start_event(2, "producer-7", 0))
            .unwrap();
        ledger
            .apply(render_complete_event(3, "producer-7", 7))
            .unwrap();
        ledger.apply(enqueue_event(4, 7)).unwrap();
        ledger.apply(callback_event(5, 7, 0)).unwrap();
        ledger.apply(expiration_event(6, 10_000_000_000)).unwrap();

        assert_eq!(
            ledger.frame(7).unwrap().disposition,
            Some(FrameDisposition::CallbackAdmitted { observed_epoch: 0 })
        );
        assert_eq!(ledger.summary().callback_admitted_frames, 100);
        assert_eq!(ledger.summary().stale_discarded_frames, 0);
    }

    #[test]
    fn producer_cannot_start_before_authorization_or_at_expiry() {
        let mut ledger = authorized_ledger();
        let before_authorization = ledger.apply(event(
            2,
            "producer-too-early",
            EventKind::ProducerStarted {
                producer_id: "producer-early".into(),
                authorization_id: "auth-1".into(),
                plan_sha256: PLAN_A.into(),
                captured_epoch: 0,
                started_at_ns: 99,
            },
        ));
        assert_eq!(
            before_authorization,
            Err(RaceError::ProducerTimeBeforeAuthorization {
                producer_id: "producer-early".into(),
                started_at_ns: 99,
                authorized_at_ns: 100,
            })
        );

        let at_expiry = ledger.apply(event(
            2,
            "producer-at-expiry",
            EventKind::ProducerStarted {
                producer_id: "producer-expired".into(),
                authorization_id: "auth-1".into(),
                plan_sha256: PLAN_A.into(),
                captured_epoch: 0,
                started_at_ns: 10_000_000_000,
            },
        ));
        assert_eq!(
            at_expiry,
            Err(RaceError::ProducerTimeAfterExpiry {
                producer_id: "producer-expired".into(),
                started_at_ns: 10_000_000_000,
                expires_at_ns: 10_000_000_000,
            })
        );
    }

    #[test]
    fn producer_may_finish_after_expiry_but_cannot_enqueue_without_revocation() {
        let mut ledger = authorized_ledger();
        ledger
            .apply(producer_start_event(2, "producer-7", 0))
            .unwrap();
        ledger
            .apply(event(
                3,
                "complete-after-expiry",
                EventKind::RenderCompleted {
                    producer_id: "producer-7".into(),
                    frame_id: 7,
                    range: FrameRange {
                        start: 700,
                        end: 800,
                    },
                    completed_at_ns: 10_000_000_001,
                },
            ))
            .unwrap();

        let enqueue = event(
            4,
            "enqueue-after-expiry",
            EventKind::Enqueue {
                frame_id: 7,
                enqueued_at_ns: 10_000_000_002,
            },
        );
        assert_eq!(
            ledger.apply(enqueue),
            Err(RaceError::AuthorizationExpired {
                authorization_id: "auth-1".into(),
                at_ns: 10_000_000_002,
                expires_at_ns: 10_000_000_000,
            })
        );
        assert!(!ledger.frame(7).unwrap().enqueued);
        assert_eq!(ledger.summary().rendered_not_enqueued_frames, 100);
    }

    #[test]
    fn epoch_exhaustion_rejects_veto_without_wrapping() {
        let mut ledger = RaceLedger::new(u64::MAX);
        ledger
            .apply(event(
                1,
                "authorize-max",
                EventKind::Authorize {
                    authorization_id: "auth-max".into(),
                    session_id: "session-1".into(),
                    turn_id: "turn-1".into(),
                    decision_id: "decision-max".into(),
                    plan_sha256: PLAN_A.into(),
                    epoch: u64::MAX,
                    control_sequence: 1,
                    authorized_at_ns: 1,
                    expires_at_ns: 10,
                },
            ))
            .unwrap();
        let result = ledger.apply(event(
            2,
            "veto-max",
            EventKind::Veto {
                authorization_id: "auth-max".into(),
                session_id: "session-1".into(),
                turn_id: "turn-1".into(),
                participant_id: "performer-1".into(),
                authority_valid: true,
                control_sequence: 2,
                prior_epoch: u64::MAX,
                new_epoch: 0,
            },
        ));
        assert_eq!(result, Err(RaceError::EpochExhausted));
        assert_eq!(ledger.current_epoch(), u64::MAX);
    }
}
