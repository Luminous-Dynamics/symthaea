// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Append-only presentation outcome receipts for Symthaea embodiments.
//!
//! A permit proves only that local policy allowed an output attempt. This
//! ledger records whether an adapter accepted submission, an observer reported
//! output, and cessation was observed. Missing observation remains unresolved;
//! it is never upgraded to success.
//!
//! Digests form an unkeyed BLAKE3 integrity chain. They are not signatures,
//! trusted timestamps, hardware attestation, or proof of observer independence.
//! Anchor/export chain heads to a separately trusted system for persistence.

use super::embodiment_event::{EpistemicStatus, EventKind, EventSource, PrivacyClass};
use super::embodiment_policy::{
    DenialReason, MotionMode, OutputChannel, OutputDestination, PresentationPermit,
    PresentationPolicy,
};
use super::embodiment_projection::EmbodimentForm;
use serde::Serialize;
use std::collections::HashMap;
use std::str::FromStr;

pub const PRESENTATION_RECEIPT_SCHEMA_V1: &str = "symthaea.presentation-receipt.v1";
pub const MAX_RECEIPT_ATTEMPTS: usize = 65_536;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReceiptPhase { Permitted, Submitted, Observed, OutcomeUnknown, Stopped, Failed, NotSubmitted }

impl ReceiptPhase {
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Stopped | Self::Failed | Self::NotSubmitted)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationSource {
    AdapterCallback,
    Compositor,
    AudioDeviceLoopback,
    HapticDriverAcknowledgement,
    ExternalObserver,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservedOutputState { Started, Stopped }

/// Monotonic value meaningful only inside the named clock domain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReceiptStamp {
    pub clock_domain_id: String,
    pub monotonic_ns: String,
}

impl ReceiptStamp {
    pub fn new(
        clock_domain_id: impl Into<String>,
        monotonic_ns: impl Into<String>,
    ) -> Result<Self, ReceiptError> {
        let clock_domain_id = clock_domain_id.into();
        let monotonic_ns = monotonic_ns.into();
        validate_identifier(&clock_domain_id)?;
        parse_monotonic_ns(&monotonic_ns)?;
        Ok(Self { clock_domain_id, monotonic_ns })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AdapterIdentity {
    pub adapter_id: String,
    pub build_digest: String,
}

impl AdapterIdentity {
    pub fn new(
        adapter_id: impl Into<String>,
        build_digest: impl Into<String>,
    ) -> Result<Self, ReceiptError> {
        let adapter_id = adapter_id.into();
        let build_digest = build_digest.into();
        validate_identifier(&adapter_id)?;
        validate_identifier_max(&build_digest, 256)?;
        Ok(Self { adapter_id, build_digest })
    }
}

/// Observer kind is disclosed; the enum does not certify independence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OutputObservation {
    pub observer_id: String,
    pub observer_build_digest: String,
    pub source: ObservationSource,
    pub output_state: ObservedOutputState,
}

impl OutputObservation {
    pub fn new(
        observer_id: impl Into<String>,
        observer_build_digest: impl Into<String>,
        source: ObservationSource,
        output_state: ObservedOutputState,
    ) -> Result<Self, ReceiptError> {
        let observer_id = observer_id.into();
        let observer_build_digest = observer_build_digest.into();
        validate_identifier(&observer_id)?;
        validate_identifier_max(&observer_build_digest, 256)?;
        Ok(Self { observer_id, observer_build_digest, source, output_state })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NotSubmittedReason { PermitExpired, PolicyDenied, UserCancelled, RendererUnavailable }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureCode {
    SubmissionRejected,
    RendererUnavailable,
    DeviceLost,
    ObservationFailed,
    OutputInterrupted,
    AdapterShutdown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReceiptError {
    InvalidIdentifier,
    InvalidTimestamp,
    WrongSession,
    WrongDecisionClock,
    PermitExpired,
    DuplicateAttempt,
    AttemptNotFound,
    AttemptCapacityReached,
    InvalidTransition { from: Option<ReceiptPhase>, to: ReceiptPhase },
    ClockRegression,
    PermitIdentityMismatch,
    MissingObservation,
    UnexpectedObservation,
    MissingFailureCode,
    SerializationFailure,
    PolicyDenied(DenialReason),
}

/// Immutable receipt fields are private and exposed read-only.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PresentationReceipt {
    schema: String,
    policy_version: String,
    receipt_id: String,
    attempt_id: String,
    transition_index: u32,
    phase: ReceiptPhase,
    session_id: String,
    source_clock_domain_id: String,
    event_id: String,
    grant_id: String,
    event_sequence: u64,
    event_kind: EventKind,
    source: EventSource,
    epistemic_status: EpistemicStatus,
    privacy_class: PrivacyClass,
    form: EmbodimentForm,
    channel: OutputChannel,
    destination: OutputDestination,
    motion_mode: MotionMode,
    requested_intensity: f32,
    effective_intensity: f32,
    semantic_key: String,
    decision_clock: ReceiptStamp,
    output_clock: ReceiptStamp,
    adapter: AdapterIdentity,
    observation: Option<OutputObservation>,
    not_submitted_reason: Option<NotSubmittedReason>,
    failure_code: Option<FailureCode>,
    previous_digest: Option<String>,
    digest: String,
}

impl PresentationReceipt {
    pub fn schema(&self) -> &str { &self.schema }
    pub fn receipt_id(&self) -> &str { &self.receipt_id }
    pub fn attempt_id(&self) -> &str { &self.attempt_id }
    pub fn transition_index(&self) -> u32 { self.transition_index }
    pub fn phase(&self) -> ReceiptPhase { self.phase }
    pub fn event_id(&self) -> &str { &self.event_id }
    pub fn grant_id(&self) -> &str { &self.grant_id }
    pub fn previous_digest(&self) -> Option<&str> { self.previous_digest.as_deref() }
    pub fn digest(&self) -> &str { &self.digest }
    pub fn decision_clock(&self) -> &ReceiptStamp { &self.decision_clock }
    pub fn output_clock(&self) -> &ReceiptStamp { &self.output_clock }
    pub fn observation(&self) -> Option<&OutputObservation> { self.observation.as_ref() }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttemptStatus {
    pub attempt_id: String,
    pub phase: ReceiptPhase,
    pub transition_count: u32,
    pub event_id: String,
    pub grant_id: String,
}

#[derive(Debug, Clone)]
pub struct PresentationReceiptLedger {
    session_id: String,
    policy_version: String,
    adapter: AdapterIdentity,
    max_attempts: usize,
    attempts: HashMap<String, Attempt>,
    receipts: Vec<PresentationReceipt>,
    latest_digest: Option<String>,
}

#[derive(Debug, Clone)]
struct Attempt {
    permit: PresentationPermit,
    phase: Option<ReceiptPhase>,
    next_transition: u32,
    clock_high_water: HashMap<String, u64>,
}

impl PresentationReceiptLedger {
    pub fn new(
        session_id: impl Into<String>,
        policy_version: impl Into<String>,
        adapter: AdapterIdentity,
    ) -> Result<Self, ReceiptError> {
        Self::with_max_attempts(session_id, policy_version, adapter, MAX_RECEIPT_ATTEMPTS)
    }

    pub fn with_max_attempts(
        session_id: impl Into<String>,
        policy_version: impl Into<String>,
        adapter: AdapterIdentity,
        max_attempts: usize,
    ) -> Result<Self, ReceiptError> {
        let session_id = session_id.into();
        let policy_version = policy_version.into();
        validate_identifier(&session_id)?;
        validate_identifier(&policy_version)?;
        AdapterIdentity::new(adapter.adapter_id.clone(), adapter.build_digest.clone())?;
        if max_attempts == 0 || max_attempts > MAX_RECEIPT_ATTEMPTS {
            return Err(ReceiptError::AttemptCapacityReached);
        }
        Ok(Self {
            session_id,
            policy_version,
            adapter,
            max_attempts,
            attempts: HashMap::with_capacity(max_attempts.min(1024)),
            receipts: Vec::new(),
            latest_digest: None,
        })
    }

    /// Record the policy decision, not proof of renderer submission/output.
    pub fn record_permitted(
        &mut self,
        attempt_id: impl Into<String>,
        permit: &PresentationPermit,
        decision_stamp: ReceiptStamp,
    ) -> Result<PresentationReceipt, ReceiptError> {
        let attempt_id = attempt_id.into();
        validate_identifier_max(&attempt_id, 96)?;
        validate_stamp(&decision_stamp)?;
        if self.attempts.contains_key(&attempt_id) {
            return Err(ReceiptError::DuplicateAttempt);
        }
        if self.attempts.len() >= self.max_attempts {
            return Err(ReceiptError::AttemptCapacityReached);
        }
        if permit.session_id() != self.session_id {
            return Err(ReceiptError::WrongSession);
        }
        if decision_stamp.clock_domain_id != permit.clock_domain_id() {
            return Err(ReceiptError::WrongDecisionClock);
        }
        let now = parse_monotonic_ns(&decision_stamp.monotonic_ns)?;
        let issued = parse_monotonic_ns(permit.issued_at_monotonic_ns())?;
        let expires = parse_monotonic_ns(permit.expires_at_monotonic_ns())?;
        if issued > now || expires <= now {
            return Err(ReceiptError::PermitExpired);
        }

        self.attempts.insert(attempt_id.clone(), Attempt {
            permit: permit.clone(),
            phase: None,
            next_transition: 0,
            clock_high_water: HashMap::new(),
        });
        match self.append_transition(
            &attempt_id, ReceiptPhase::Permitted, decision_stamp,
            None, None, None, None,
        ) {
            Ok(receipt) => Ok(receipt),
            Err(error) => {
                self.attempts.remove(&attempt_id);
                Err(error)
            }
        }
    }

    /// Re-check policy before submission. Hold the same serialization lock
    /// around this call and the renderer's actual submission to prevent a
    /// concurrent revocation race. Policy and renderer clocks are separate.
    pub fn record_submitted(
        &mut self,
        attempt_id: &str,
        policy: &PresentationPolicy,
        policy_now: ReceiptStamp,
        submitted_at: ReceiptStamp,
    ) -> Result<(PresentationPermit, PresentationReceipt), ReceiptError> {
        validate_stamp(&policy_now)?;
        validate_stamp(&submitted_at)?;
        let attempt = self.attempts.get(attempt_id).ok_or(ReceiptError::AttemptNotFound)?;
        let original = attempt.permit.clone();
        if policy_now.clock_domain_id != original.clock_domain_id() {
            return Err(ReceiptError::WrongDecisionClock);
        }
        let refreshed = policy
            .recheck_permit(&original, &policy_now.monotonic_ns, &policy_now.clock_domain_id)
            .map_err(ReceiptError::PolicyDenied)?;
        let receipt = self.append_transition(
            attempt_id, ReceiptPhase::Submitted, submitted_at,
            None, None, None, Some(refreshed.clone()),
        )?;
        Ok((refreshed, receipt))
    }

    pub fn record_observed(
        &mut self,
        attempt_id: &str,
        stamp: ReceiptStamp,
        observation: OutputObservation,
    ) -> Result<PresentationReceipt, ReceiptError> {
        if observation.output_state != ObservedOutputState::Started {
            return Err(ReceiptError::UnexpectedObservation);
        }
        self.append_transition(
            attempt_id, ReceiptPhase::Observed, stamp,
            Some(observation), None, None, None,
        )
    }

    /// A cancel command is not evidence that output cessation occurred.
    pub fn record_stopped(
        &mut self,
        attempt_id: &str,
        stamp: ReceiptStamp,
        observation: OutputObservation,
    ) -> Result<PresentationReceipt, ReceiptError> {
        if observation.output_state != ObservedOutputState::Stopped {
            return Err(ReceiptError::UnexpectedObservation);
        }
        self.append_transition(
            attempt_id, ReceiptPhase::Stopped, stamp,
            Some(observation), None, None, None,
        )
    }

    /// Missing output observation remains unresolved, never successful. A
    /// delayed observer may later resolve this to Observed or Failed.
    pub fn record_outcome_unknown(
        &mut self,
        attempt_id: &str,
        stamp: ReceiptStamp,
    ) -> Result<PresentationReceipt, ReceiptError> {
        self.append_transition(
            attempt_id, ReceiptPhase::OutcomeUnknown, stamp,
            None, None, None, None,
        )
    }

    pub fn record_not_submitted(
        &mut self,
        attempt_id: &str,
        stamp: ReceiptStamp,
        reason: NotSubmittedReason,
    ) -> Result<PresentationReceipt, ReceiptError> {
        self.append_transition(
            attempt_id, ReceiptPhase::NotSubmitted, stamp,
            None, Some(reason), None, None,
        )
    }

    pub fn record_failed(
        &mut self,
        attempt_id: &str,
        stamp: ReceiptStamp,
        failure_code: FailureCode,
    ) -> Result<PresentationReceipt, ReceiptError> {
        self.append_transition(
            attempt_id, ReceiptPhase::Failed, stamp,
            None, None, Some(failure_code), None,
        )
    }

    fn append_transition(
        &mut self,
        attempt_id: &str,
        next_phase: ReceiptPhase,
        stamp: ReceiptStamp,
        observation: Option<OutputObservation>,
        not_submitted_reason: Option<NotSubmittedReason>,
        failure_code: Option<FailureCode>,
        permit_override: Option<PresentationPermit>,
    ) -> Result<PresentationReceipt, ReceiptError> {
        validate_stamp(&stamp)?;
        if let Some(evidence) = observation.as_ref() {
            validate_identifier(&evidence.observer_id)?;
            validate_identifier_max(&evidence.observer_build_digest, 256)?;
        }
        let mut attempt = self.attempts.get(attempt_id)
            .cloned()
            .ok_or(ReceiptError::AttemptNotFound)?;
        if !transition_allowed(attempt.phase, next_phase) {
            return Err(ReceiptError::InvalidTransition { from: attempt.phase, to: next_phase });
        }
        let timestamp = parse_monotonic_ns(&stamp.monotonic_ns)?;
        if let Some(previous) = attempt.clock_high_water.get(&stamp.clock_domain_id) {
            if timestamp < *previous {
                return Err(ReceiptError::ClockRegression);
            }
        }
        match next_phase {
            ReceiptPhase::Observed | ReceiptPhase::Stopped if observation.is_none() =>
                return Err(ReceiptError::MissingObservation),
            ReceiptPhase::NotSubmitted if not_submitted_reason.is_none() =>
                return Err(ReceiptError::MissingFailureCode),
            ReceiptPhase::Failed if failure_code.is_none() =>
                return Err(ReceiptError::MissingFailureCode),
            _ => {}
        }

        let permit = permit_override.unwrap_or_else(|| attempt.permit.clone());
        if !same_permit_identity(&attempt.permit, &permit) {
            return Err(ReceiptError::PermitIdentityMismatch);
        }
        attempt.permit = permit.clone();
        attempt.clock_high_water.insert(stamp.clock_domain_id.clone(), timestamp);

        let transition_index = attempt.next_transition;
        let receipt_id = format!("{attempt_id}:{transition_index}");
        let previous_digest = self.latest_digest.clone();
        let mut receipt = PresentationReceipt {
            schema: PRESENTATION_RECEIPT_SCHEMA_V1.to_owned(),
            policy_version: self.policy_version.clone(),
            receipt_id,
            attempt_id: attempt_id.to_owned(),
            transition_index,
            phase: next_phase,
            session_id: permit.session_id().to_owned(),
            source_clock_domain_id: permit.clock_domain_id().to_owned(),
            event_id: permit.event_id().to_owned(),
            grant_id: permit.grant_id().to_owned(),
            event_sequence: permit.sequence(),
            event_kind: permit.event_kind(),
            source: permit.source().clone(),
            epistemic_status: permit.epistemic_status(),
            privacy_class: permit.privacy_class(),
            form: permit.form(),
            channel: permit.channel(),
            destination: permit.destination(),
            motion_mode: permit.motion_mode(),
            requested_intensity: permit.requested_intensity(),
            effective_intensity: permit.effective_intensity(),
            semantic_key: permit.semantic_key().to_owned(),
            decision_clock: ReceiptStamp {
                clock_domain_id: permit.clock_domain_id().to_owned(),
                monotonic_ns: permit.issued_at_monotonic_ns().to_owned(),
            },
            output_clock: stamp,
            adapter: self.adapter.clone(),
            observation,
            not_submitted_reason,
            failure_code,
            previous_digest,
            digest: String::new(),
        };
        receipt.digest = compute_digest(&receipt)?;
        self.latest_digest = Some(receipt.digest.clone());
        attempt.phase = Some(next_phase);
        attempt.next_transition = attempt.next_transition.saturating_add(1);
        self.attempts.insert(attempt_id.to_owned(), attempt);
        self.receipts.push(receipt.clone());
        Ok(receipt)
    }

    pub fn status(&self, attempt_id: &str) -> Option<AttemptStatus> {
        let attempt = self.attempts.get(attempt_id)?;
        Some(AttemptStatus {
            attempt_id: attempt_id.to_owned(),
            phase: attempt.phase?,
            transition_count: attempt.next_transition,
            event_id: attempt.permit.event_id().to_owned(),
            grant_id: attempt.permit.grant_id().to_owned(),
        })
    }

    pub fn receipts(&self) -> &[PresentationReceipt] { &self.receipts }
    pub fn latest_digest(&self) -> Option<&str> { self.latest_digest.as_deref() }

    /// Verify internal chain linkage and each unkeyed digest. This detects
    /// accidental modification, not a ledger rewritten with a new chain head.
    pub fn verify_chain(&self) -> Result<(), ReceiptError> {
        let mut previous: Option<String> = None;
        for receipt in &self.receipts {
            if receipt.previous_digest != previous {
                return Err(ReceiptError::SerializationFailure);
            }
            let expected = compute_digest(receipt)?;
            if expected != receipt.digest {
                return Err(ReceiptError::SerializationFailure);
            }
            previous = Some(receipt.digest.clone());
        }
        if previous != self.latest_digest {
            return Err(ReceiptError::SerializationFailure);
        }
        Ok(())
    }
}

fn transition_allowed(from: Option<ReceiptPhase>, to: ReceiptPhase) -> bool {
    match (from, to) {
        (None, ReceiptPhase::Permitted) => true,
        (Some(ReceiptPhase::Permitted), ReceiptPhase::Submitted | ReceiptPhase::NotSubmitted) => true,
        (Some(ReceiptPhase::Submitted), ReceiptPhase::Observed | ReceiptPhase::OutcomeUnknown | ReceiptPhase::Failed) => true,
        (Some(ReceiptPhase::OutcomeUnknown), ReceiptPhase::Observed | ReceiptPhase::Failed) => true,
        (Some(ReceiptPhase::Observed), ReceiptPhase::Stopped | ReceiptPhase::Failed) => true,
        _ => false,
    }
}

fn same_permit_identity(a: &PresentationPermit, b: &PresentationPermit) -> bool {
    a.session_id() == b.session_id()
        && a.clock_domain_id() == b.clock_domain_id()
        && a.event_id() == b.event_id()
        && a.grant_id() == b.grant_id()
        && a.sequence() == b.sequence()
        && a.event_kind() == b.event_kind()
        && a.form() == b.form()
        && a.channel() == b.channel()
        && a.destination() == b.destination()
        && a.semantic_key() == b.semantic_key()
}

fn compute_digest(receipt: &PresentationReceipt) -> Result<String, ReceiptError> {
    let mut material = receipt.clone();
    material.digest.clear();
    let bytes = serde_json::to_vec(&material).map_err(|_| ReceiptError::SerializationFailure)?;
    let mut hasher = blake3::Hasher::new_derive_key("symthaea.presentation-receipt.v1");
    hasher.update(&bytes);
    Ok(hasher.finalize().to_hex().to_string())
}

fn validate_stamp(stamp: &ReceiptStamp) -> Result<(), ReceiptError> {
    validate_identifier(&stamp.clock_domain_id)?;
    parse_monotonic_ns(&stamp.monotonic_ns)?;
    Ok(())
}

fn validate_identifier(value: &str) -> Result<(), ReceiptError> {
    validate_identifier_max(value, 128)
}

fn validate_identifier_max(value: &str, max_len: usize) -> Result<(), ReceiptError> {
    if value.is_empty()
        || value.len() > max_len
        || !value.bytes().all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
    {
        return Err(ReceiptError::InvalidIdentifier);
    }
    Ok(())
}

fn parse_monotonic_ns(value: &str) -> Result<u64, ReceiptError> {
    if value.is_empty()
        || (value.len() > 1 && value.starts_with('0'))
        || !value.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(ReceiptError::InvalidTimestamp);
    }
    u64::from_str(value).map_err(|_| ReceiptError::InvalidTimestamp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embodiment_event::{EventKind, EventSource};
    use crate::embodiment_policy::{ConsentGrant, PresentationRequest};
    use crate::embodiment_projection::{CanonicalCue, SemanticCue};

    const SESSION: &str = "receipt-session";
    const CLOCK: &str = "clock:dialogue-epoch-1";
    const OUTPUT_CLOCK: &str = "clock:renderer-epoch-1";

    fn fixture() -> (PresentationPolicy, PresentationPermit, PresentationReceiptLedger) {
        let mut policy = PresentationPolicy::new(SESSION, CLOCK).unwrap();
        policy.grant(
            ConsentGrant {
                grant_id: "grant-v1".to_owned(),
                session_id: SESSION.to_owned(),
                clock_domain_id: CLOCK.to_owned(),
                channel: OutputChannel::Visual,
                destination: OutputDestination::LocalPrivate,
                allowed_forms: vec![EmbodimentForm::LivingConstellation],
                expires_at_monotonic_ns: "10000".to_owned(),
                max_intensity: 0.7,
            },
            "100",
            CLOCK,
        ).unwrap();
        policy.set_paused(false);
        let cue = CanonicalCue {
            session_id: SESSION.to_owned(),
            clock_domain_id: CLOCK.to_owned(),
            event_id: "event-1".to_owned(),
            sequence: 1,
            event_kind: EventKind::ConversationListening,
            semantic: SemanticCue::Listening { active: true },
            source: EventSource {
                component: "dialogue".to_owned(),
                instance_id: "dialogue-1".to_owned(),
                source_event_id: None,
                build_id: Some("build-1".to_owned()),
            },
            observed_at_monotonic_ns: "90".to_owned(),
            expires_at_monotonic_ns: "9000".to_owned(),
            epistemic_status: EpistemicStatus::Measured,
            confidence: None,
            evidence_refs: Vec::new(),
            privacy_class: PrivacyClass::Public,
        };
        let permit = policy.evaluate(
            &cue,
            &PresentationRequest {
                form: EmbodimentForm::LivingConstellation,
                channel: OutputChannel::Visual,
                destination: OutputDestination::LocalPrivate,
                requested_intensity: 0.9,
            },
            "110",
            CLOCK,
        ).unwrap();
        let adapter = AdapterIdentity::new("constellation", "sha256:renderer-build-1").unwrap();
        let ledger = PresentationReceiptLedger::new(SESSION, "policy-v1", adapter).unwrap();
        (policy, permit, ledger)
    }

    fn stamp(clock: &str, time: &str) -> ReceiptStamp {
        ReceiptStamp::new(clock, time).unwrap()
    }

    fn started() -> OutputObservation {
        OutputObservation::new(
            "compositor-1", "sha256:compositor-build-1",
            ObservationSource::Compositor, ObservedOutputState::Started,
        ).unwrap()
    }

    fn stopped() -> OutputObservation {
        OutputObservation::new(
            "compositor-1", "sha256:compositor-build-1",
            ObservationSource::Compositor, ObservedOutputState::Stopped,
        ).unwrap()
    }

    fn begin(ledger: &mut PresentationReceiptLedger, permit: &PresentationPermit) {
        ledger.record_permitted("attempt-1", permit, stamp(CLOCK, "120")).unwrap();
    }

    fn submit(
        ledger: &mut PresentationReceiptLedger,
        policy: &PresentationPolicy,
        permit: &PresentationPermit,
    ) {
        begin(ledger, permit);
        ledger.record_submitted(
            "attempt-1", policy, stamp(CLOCK, "121"), stamp(OUTPUT_CLOCK, "10"),
        ).unwrap();
    }

    #[test]
    fn permit_is_not_counted_as_submitted_or_observed() {
        let (_policy, permit, mut ledger) = fixture();
        begin(&mut ledger, &permit);
        let status = ledger.status("attempt-1").unwrap();
        assert_eq!(status.phase, ReceiptPhase::Permitted);
        assert_eq!(ledger.receipts().len(), 1);
        assert_eq!(ledger.receipts()[0].phase(), ReceiptPhase::Permitted);
    }

    #[test]
    fn submission_requires_a_fresh_policy_recheck() {
        let (mut policy, permit, mut ledger) = fixture();
        begin(&mut ledger, &permit);
        policy.revoke_channel(OutputChannel::Visual);
        assert!(matches!(
            ledger.record_submitted("attempt-1", &policy, stamp(CLOCK, "121"), stamp(OUTPUT_CLOCK, "10")),
            Err(ReceiptError::PolicyDenied(DenialReason::NoMatchingGrant))
        ));
        assert_eq!(ledger.status("attempt-1").unwrap().phase, ReceiptPhase::Permitted);
    }

    #[test]
    fn output_is_not_success_until_observed_and_stopped() {
        let (policy, permit, mut ledger) = fixture();
        submit(&mut ledger, &policy, &permit);
        assert_eq!(ledger.status("attempt-1").unwrap().phase, ReceiptPhase::Submitted);
        ledger.record_observed("attempt-1", stamp(OUTPUT_CLOCK, "12"), started()).unwrap();
        assert_eq!(ledger.status("attempt-1").unwrap().phase, ReceiptPhase::Observed);
        ledger.record_stopped("attempt-1", stamp(OUTPUT_CLOCK, "20"), stopped()).unwrap();
        assert_eq!(ledger.status("attempt-1").unwrap().phase, ReceiptPhase::Stopped);
        assert!(ledger.status("attempt-1").unwrap().phase.is_terminal());
        ledger.verify_chain().unwrap();
    }

    #[test]
    fn missing_observation_stays_unknown_and_can_be_resolved_later() {
        let (policy, permit, mut ledger) = fixture();
        submit(&mut ledger, &policy, &permit);
        ledger.record_outcome_unknown("attempt-1", stamp(OUTPUT_CLOCK, "50")).unwrap();
        assert_eq!(ledger.status("attempt-1").unwrap().phase, ReceiptPhase::OutcomeUnknown);
        ledger.record_observed("attempt-1", stamp(OUTPUT_CLOCK, "60"), started()).unwrap();
        assert_eq!(ledger.status("attempt-1").unwrap().phase, ReceiptPhase::Observed);
    }

    #[test]
    fn unknown_outcome_can_resolve_to_failure() {
        let (policy, permit, mut ledger) = fixture();
        submit(&mut ledger, &policy, &permit);
        ledger.record_outcome_unknown("attempt-1", stamp(OUTPUT_CLOCK, "50")).unwrap();
        ledger.record_failed("attempt-1", stamp(OUTPUT_CLOCK, "70"), FailureCode::ObservationFailed).unwrap();
        assert_eq!(ledger.status("attempt-1").unwrap().phase, ReceiptPhase::Failed);
    }

    #[test]
    fn stopped_requires_a_stopped_observation_not_a_command() {
        let (policy, permit, mut ledger) = fixture();
        submit(&mut ledger, &policy, &permit);
        ledger.record_observed("attempt-1", stamp(OUTPUT_CLOCK, "12"), started()).unwrap();
        assert_eq!(
            ledger.record_stopped("attempt-1", stamp(OUTPUT_CLOCK, "15"), started()),
            Err(ReceiptError::UnexpectedObservation)
        );
        assert_eq!(ledger.status("attempt-1").unwrap().phase, ReceiptPhase::Observed);
    }

    #[test]
    fn terminal_states_cannot_be_reopened_or_duplicated() {
        let (policy, permit, mut ledger) = fixture();
        submit(&mut ledger, &policy, &permit);
        ledger.record_observed("attempt-1", stamp(OUTPUT_CLOCK, "12"), started()).unwrap();
        ledger.record_stopped("attempt-1", stamp(OUTPUT_CLOCK, "20"), stopped()).unwrap();
        assert!(matches!(
            ledger.record_failed("attempt-1", stamp(OUTPUT_CLOCK, "21"), FailureCode::OutputInterrupted),
            Err(ReceiptError::InvalidTransition { .. })
        ));
    }

    #[test]
    fn not_submitted_is_a_distinct_terminal_outcome() {
        let (_policy, permit, mut ledger) = fixture();
        begin(&mut ledger, &permit);
        ledger.record_not_submitted(
            "attempt-1", stamp(OUTPUT_CLOCK, "15"), NotSubmittedReason::UserCancelled,
        ).unwrap();
        assert_eq!(ledger.status("attempt-1").unwrap().phase, ReceiptPhase::NotSubmitted);
        assert!(ledger.status("attempt-1").unwrap().phase.is_terminal());
    }

    #[test]
    fn duplicate_attempt_ids_are_rejected() {
        let (_policy, permit, mut ledger) = fixture();
        begin(&mut ledger, &permit);
        assert_eq!(
            ledger.record_permitted("attempt-1", &permit, stamp(CLOCK, "121")),
            Err(ReceiptError::DuplicateAttempt)
        );
    }

    #[test]
    fn clock_regression_is_checked_per_clock_domain_only() {
        let (policy, permit, mut ledger) = fixture();
        submit(&mut ledger, &policy, &permit);
        ledger.record_observed("attempt-1", stamp(OUTPUT_CLOCK, "12"), started()).unwrap();
        assert_eq!(
            ledger.record_stopped("attempt-1", stamp(OUTPUT_CLOCK, "11"), stopped()),
            Err(ReceiptError::ClockRegression)
        );
        ledger.record_stopped(
            "attempt-1", stamp("clock:external-observer", "1"), stopped(),
        ).unwrap();
        assert_eq!(ledger.status("attempt-1").unwrap().phase, ReceiptPhase::Stopped);
    }

    #[test]
    fn policy_denial_does_not_advance_the_receipt_state() {
        let (mut policy, permit, mut ledger) = fixture();
        begin(&mut ledger, &permit);
        policy.set_paused(true);
        assert!(ledger.record_submitted(
            "attempt-1", &policy, stamp(CLOCK, "121"), stamp(OUTPUT_CLOCK, "10"),
        ).is_err());
        assert_eq!(ledger.status("attempt-1").unwrap().phase, ReceiptPhase::Permitted);
        assert_eq!(ledger.receipts().len(), 1);
    }

    #[test]
    fn receipts_form_a_verifiable_ledger_wide_digest_chain() {
        let (policy, permit, mut ledger) = fixture();
        submit(&mut ledger, &policy, &permit);
        ledger.record_observed("attempt-1", stamp(OUTPUT_CLOCK, "12"), started()).unwrap();
        assert_eq!(ledger.receipts()[0].previous_digest(), None);
        assert_eq!(
            ledger.receipts()[1].previous_digest(),
            Some(ledger.receipts()[0].digest())
        );
        ledger.verify_chain().unwrap();
    }

    #[test]
    fn receipt_retains_separate_decision_and_renderer_clock_domains() {
        let (policy, permit, mut ledger) = fixture();
        submit(&mut ledger, &policy, &permit);
        let receipt = &ledger.receipts()[1];
        assert_eq!(receipt.decision_clock().clock_domain_id, CLOCK);
        assert_eq!(receipt.output_clock().clock_domain_id, OUTPUT_CLOCK);
        assert_eq!(receipt.decision_clock().monotonic_ns, "110");
        assert_eq!(receipt.output_clock().monotonic_ns, "10");
    }
}
