// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Correlation of Wayland presentation-time feedback with authorized scene output.
//!
//! This module is deliberately limited to one claim: the compositor reported
//! that a particular surface commit was presented or discarded. A Presented
//! result can provide evidence for ReceiptPhase::Observed. A Discarded result
//! cannot. This module never emits Stopped; presentation of a frame does not,
//! by itself, prove that a previous frame has ceased to be visible.
//!
//! The compositor timestamp belongs to the clock domain advertised by
//! wp_presentation.clock_id. It is not compared with the source-event clock.
//! Cross-clock latency requires a separately measured and bounded conversion.

use super::embodiment_receipt::{
    ObservationSource, ObservedOutputState, OutputObservation,
};
use serde::Serialize;
use std::collections::{HashMap, HashSet};

pub const MAX_PENDING_PRESENTATION_FEEDBACK: usize = 65_536;
pub const PRESENTATION_FLAG_VSYNC: u32 = 0x01;
pub const PRESENTATION_FLAG_HW_CLOCK: u32 = 0x02;
pub const PRESENTATION_FLAG_HW_COMPLETION: u32 = 0x04;
pub const PRESENTATION_FLAG_ZERO_COPY: u32 = 0x08;
pub const PRESENTATION_FLAG_FIXED_RATE: u32 = 0x10;
pub const PRESENTATION_FLAG_VARIABLE_RATE: u32 = 0x20;
pub const KNOWN_PRESENTATION_FLAGS: u32 = 0x3f;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PresentationFeedbackError {
    InvalidIdentifier,
    InvalidDigest,
    InvalidTimestamp,
    TimestampOverflow,
    NanosecondsOutOfRange,
    DuplicateRequest,
    RequestNotFound,
    DuplicateTerminalFeedback,
    PendingCapacityReached,
    RecordCapacityReached,
    SurfaceMismatch,
    FrameMismatch,
    ClockDomainMismatch,
    TimestampBeforeSubmission,
    ReceiptEvidenceInvalid,
}

/// Exact renderer/backend commit correlation known before feedback arrives.
/// submitted_at_monotonic_ns must use the same clock domain later reported by
/// wp_presentation.clock_id. It is not the source-event clock.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PresentationSubmission {
    pub request_id: String,
    pub session_id: String,
    pub event_id: String,
    pub canonical_cue_digest: String,
    pub grant_id: String,
    pub policy_version: String,
    pub surface_id: String,
    pub frame_id: String,
    pub backend_commit_id: String,
    pub clock_domain_id: String,
    pub submitted_at_monotonic_ns: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WaylandFeedbackResult {
    Presented {
        output_id: Option<String>,
        presentation_timestamp_ns: String,
        refresh_interval_ns: Option<u64>,
        sequence: Option<u64>,
        flags: u32,
    },
    Discarded,
}

/// Normalized callback from wp_presentation_feedback. The Wayland adapter is
/// responsible for correlating this with the wl_surface.commit that requested
/// feedback; this ledger independently checks all supplied identities.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WaylandPresentationFeedback {
    pub request_id: String,
    pub surface_id: String,
    pub frame_id: String,
    pub clock_domain_id: String,
    pub result: WaylandFeedbackResult,
}

impl WaylandPresentationFeedback {
    /// Normalize the raw wp_presentation_feedback.presented event fields.
    ///
    /// The caller supplies the output learned from the preceding sync_output
    /// event, if one was delivered. Zero refresh means the compositor did not
    /// provide a usable refresh interval; sequence words are preserved exactly.
    pub fn presented_from_protocol(
        request_id: impl Into<String>,
        surface_id: impl Into<String>,
        frame_id: impl Into<String>,
        clock_domain_id: impl Into<String>,
        output_id: Option<String>,
        seconds_high: u32,
        seconds_low: u32,
        nanoseconds: u32,
        refresh_interval_ns: u32,
        sequence_high: u32,
        sequence_low: u32,
        flags: u32,
    ) -> Result<Self, PresentationFeedbackError> {
        let presentation_timestamp_ns =
            wayland_timestamp_ns(seconds_high, seconds_low, nanoseconds)?.to_string();
        let sequence = ((sequence_high as u64) << 32) | sequence_low as u64;
        Ok(Self {
            request_id: request_id.into(),
            surface_id: surface_id.into(),
            frame_id: frame_id.into(),
            clock_domain_id: clock_domain_id.into(),
            result: WaylandFeedbackResult::Presented {
                output_id,
                presentation_timestamp_ns,
                refresh_interval_ns: (refresh_interval_ns != 0)
                    .then_some(refresh_interval_ns as u64),
                sequence: Some(sequence),
                flags,
            },
        })
    }

    /// Normalize the raw wp_presentation_feedback.discarded event.
    pub fn discarded_from_protocol(
        request_id: impl Into<String>,
        surface_id: impl Into<String>,
        frame_id: impl Into<String>,
        clock_domain_id: impl Into<String>,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            surface_id: surface_id.into(),
            frame_id: frame_id.into(),
            clock_domain_id: clock_domain_id.into(),
            result: WaylandFeedbackResult::Discarded,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PresentationOutcome {
    Presented,
    Discarded,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PresentationFeedbackRecord {
    pub request_id: String,
    pub session_id: String,
    pub event_id: String,
    pub canonical_cue_digest: String,
    pub grant_id: String,
    pub policy_version: String,
    pub surface_id: String,
    pub output_id: Option<String>,
    pub frame_id: String,
    pub backend_commit_id: String,
    pub clock_domain_id: String,
    pub outcome: PresentationOutcome,
    pub presentation_timestamp_ns: Option<String>,
    pub refresh_interval_ns: Option<u64>,
    pub sequence: Option<u64>,
    /// Raw protocol flags are preserved, including unknown future bits.
    pub flags: Option<u32>,
    /// Present only for a compositor-reported Presented outcome. The adapter
    /// may pass this to PresentationReceiptLedger::record_observed().
    pub output_observation: Option<OutputObservation>,
    pub was_pending_at_feedback: bool,
}

#[derive(Debug, Clone)]
pub struct PresentationFeedbackLedger {
    observer_id: String,
    observer_build_digest: String,
    max_pending: usize,
    pending: HashMap<String, PresentationSubmission>,
    terminal_request_ids: HashSet<String>,
    records: Vec<PresentationFeedbackRecord>,
}

impl PresentationFeedbackLedger {
    pub fn new(
        observer_id: impl Into<String>,
        observer_build_digest: impl Into<String>,
        max_pending: usize,
    ) -> Result<Self, PresentationFeedbackError> {
        let observer_id = observer_id.into();
        let observer_build_digest = observer_build_digest.into();
        validate_identifier(&observer_id)?;
        validate_identifier_max(&observer_build_digest, 256)?;
        if max_pending == 0 || max_pending > MAX_PENDING_PRESENTATION_FEEDBACK {
            return Err(PresentationFeedbackError::PendingCapacityReached);
        }
        Ok(Self {
            observer_id,
            observer_build_digest,
            max_pending,
            pending: HashMap::with_capacity(max_pending.min(1024)),
            terminal_request_ids: HashSet::new(),
            records: Vec::new(),
        })
    }

    pub fn register_submission(
        &mut self,
        submission: PresentationSubmission,
    ) -> Result<(), PresentationFeedbackError> {
        validate_identifier(&submission.request_id)?;
        validate_identifier(&submission.session_id)?;
        validate_identifier(&submission.event_id)?;
        validate_hex_digest(&submission.canonical_cue_digest)?;
        validate_identifier(&submission.grant_id)?;
        validate_identifier(&submission.policy_version)?;
        validate_identifier(&submission.surface_id)?;
        validate_identifier(&submission.frame_id)?;
        validate_identifier(&submission.backend_commit_id)?;
        validate_identifier(&submission.clock_domain_id)?;
        let _submitted_at = parse_monotonic_ns(&submission.submitted_at_monotonic_ns)?;

        if self.pending.contains_key(&submission.request_id)
            || self.terminal_request_ids.contains(&submission.request_id)
        {
            return Err(PresentationFeedbackError::DuplicateRequest);
        }
        if self.pending.len() >= self.max_pending {
            return Err(PresentationFeedbackError::PendingCapacityReached);
        }
        self.pending.insert(submission.request_id.clone(), submission);
        Ok(())
    }

    /// Accept exactly one terminal compositor result for a registered commit.
    /// A malformed callback does not consume the pending submission, allowing
    /// the host to preserve the unresolved outcome instead of recording success.
    pub fn record_feedback(
        &mut self,
        feedback: WaylandPresentationFeedback,
    ) -> Result<PresentationFeedbackRecord, PresentationFeedbackError> {
        validate_identifier(&feedback.request_id)?;
        validate_identifier(&feedback.surface_id)?;
        validate_identifier(&feedback.frame_id)?;
        validate_identifier(&feedback.clock_domain_id)?;

        if self.terminal_request_ids.contains(&feedback.request_id) {
            return Err(PresentationFeedbackError::DuplicateTerminalFeedback);
        }
        let submission = self
            .pending
            .get(&feedback.request_id)
            .cloned()
            .ok_or(PresentationFeedbackError::RequestNotFound)?;

        if feedback.surface_id != submission.surface_id {
            return Err(PresentationFeedbackError::SurfaceMismatch);
        }
        if feedback.frame_id != submission.frame_id {
            return Err(PresentationFeedbackError::FrameMismatch);
        }
        if feedback.clock_domain_id != submission.clock_domain_id {
            return Err(PresentationFeedbackError::ClockDomainMismatch);
        }

        let (outcome, output_id, timestamp, refresh, sequence, flags, observation) =
            match feedback.result {
                WaylandFeedbackResult::Presented {
                    output_id,
                    presentation_timestamp_ns,
                    refresh_interval_ns,
                    sequence,
                    flags,
                } => {
                    if let Some(output_id) = output_id.as_deref() {
                        validate_identifier(output_id)?;
                    }
                    let presented_at = parse_monotonic_ns(&presentation_timestamp_ns)?;
                    let submitted_at = parse_monotonic_ns(&submission.submitted_at_monotonic_ns)?;
                    if presented_at < submitted_at {
                        return Err(PresentationFeedbackError::TimestampBeforeSubmission);
                    }
                    let observation = OutputObservation::new(
                        self.observer_id.clone(),
                        self.observer_build_digest.clone(),
                        ObservationSource::Compositor,
                        ObservedOutputState::Started,
                    )
                    .map_err(|_| PresentationFeedbackError::ReceiptEvidenceInvalid)?;
                    (
                        PresentationOutcome::Presented,
                        output_id,
                        Some(presentation_timestamp_ns),
                        refresh_interval_ns,
                        sequence,
                        Some(flags),
                        Some(observation),
                    )
                }
                WaylandFeedbackResult::Discarded => (
                    PresentationOutcome::Discarded,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
            };

        if self.records.len() >= MAX_PENDING_PRESENTATION_FEEDBACK {
            return Err(PresentationFeedbackError::RecordCapacityReached);
        }

        let record = PresentationFeedbackRecord {
            request_id: submission.request_id.clone(),
            session_id: submission.session_id.clone(),
            event_id: submission.event_id.clone(),
            canonical_cue_digest: submission.canonical_cue_digest.clone(),
            grant_id: submission.grant_id.clone(),
            policy_version: submission.policy_version.clone(),
            surface_id: submission.surface_id.clone(),
            output_id,
            frame_id: submission.frame_id.clone(),
            backend_commit_id: submission.backend_commit_id.clone(),
            clock_domain_id: submission.clock_domain_id.clone(),
            outcome,
            presentation_timestamp_ns: timestamp,
            refresh_interval_ns: refresh,
            sequence,
            flags,
            output_observation: observation,
            was_pending_at_feedback: true,
        };

        self.pending.remove(&submission.request_id);
        self.terminal_request_ids.insert(submission.request_id);
        self.records.push(record.clone());
        Ok(record)
    }

    /// The caller may record OutcomeUnknown in the receipt ledger while the
    /// callback remains pending here. A delayed compositor result can still
    /// resolve that unknown receipt to Observed or a discarded outcome.
    pub fn is_pending(&self, request_id: &str) -> bool {
        self.pending.contains_key(request_id)
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    pub fn records(&self) -> &[PresentationFeedbackRecord] {
        &self.records
    }

    pub fn observer_id(&self) -> &str {
        &self.observer_id
    }

    pub fn observer_build_digest(&self) -> &str {
        &self.observer_build_digest
    }
}

/// Convert wp_presentation.presented's seconds high/low words and nanoseconds
/// into one checked nanosecond value without mixing it with another clock.
pub fn wayland_timestamp_ns(
    seconds_high: u32,
    seconds_low: u32,
    nanoseconds: u32,
) -> Result<u64, PresentationFeedbackError> {
    if nanoseconds >= 1_000_000_000 {
        return Err(PresentationFeedbackError::NanosecondsOutOfRange);
    }
    let seconds = ((seconds_high as u64) << 32) | seconds_low as u64;
    seconds
        .checked_mul(1_000_000_000)
        .and_then(|value| value.checked_add(nanoseconds as u64))
        .ok_or(PresentationFeedbackError::TimestampOverflow)
}

fn validate_identifier(value: &str) -> Result<(), PresentationFeedbackError> {
    validate_identifier_max(value, 128)
}

fn validate_identifier_max(
    value: &str,
    max_len: usize,
) -> Result<(), PresentationFeedbackError> {
    if value.is_empty()
        || value.len() > max_len
        || !value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || b"._:-".contains(&byte)
        })
    {
        return Err(PresentationFeedbackError::InvalidIdentifier);
    }
    Ok(())
}

fn validate_hex_digest(value: &str) -> Result<(), PresentationFeedbackError> {
    if value.len() != 64
        || !value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(PresentationFeedbackError::InvalidDigest);
    }
    Ok(())
}

fn parse_monotonic_ns(value: &str) -> Result<u64, PresentationFeedbackError> {
    if value.is_empty()
        || (value.len() > 1 && value.starts_with('0'))
        || !value.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(PresentationFeedbackError::InvalidTimestamp);
    }
    value
        .parse::<u64>()
        .map_err(|_| PresentationFeedbackError::InvalidTimestamp)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLOCK: &str = "clock:wayland-monotonic";

    fn submission(request_id: &str) -> PresentationSubmission {
        PresentationSubmission {
            request_id: request_id.to_owned(),
            session_id: "session-1".to_owned(),
            event_id: "event-1".to_owned(),
            canonical_cue_digest: "a".repeat(64),
            grant_id: "grant-1".to_owned(),
            policy_version: "policy-v1".to_owned(),
            surface_id: "surface-1".to_owned(),
            frame_id: format!("frame-{request_id}"),
            backend_commit_id: format!("commit-{request_id}"),
            clock_domain_id: CLOCK.to_owned(),
            submitted_at_monotonic_ns: "1000".to_owned(),
        }
    }

    fn ledger() -> PresentationFeedbackLedger {
        PresentationFeedbackLedger::new(
            "wayland-observer",
            "sha256:observer-build",
            8,
        )
        .unwrap()
    }

    fn presented(request_id: &str) -> WaylandPresentationFeedback {
        WaylandPresentationFeedback {
            request_id: request_id.to_owned(),
            surface_id: "surface-1".to_owned(),
            frame_id: format!("frame-{request_id}"),
            clock_domain_id: CLOCK.to_owned(),
            result: WaylandFeedbackResult::Presented {
                output_id: Some("output-1".to_owned()),
                presentation_timestamp_ns: "1100".to_owned(),
                refresh_interval_ns: Some(16_666_667),
                sequence: Some(42),
                flags: PRESENTATION_FLAG_VSYNC | PRESENTATION_FLAG_HW_CLOCK,
            },
        }
    }

    #[test]
    fn raw_wayland_presented_fields_are_combined_without_loss() {
        let feedback = WaylandPresentationFeedback::presented_from_protocol(
            "request-raw",
            "surface-1",
            "frame-request-raw",
            CLOCK,
            Some("output-1".to_owned()),
            1,
            2,
            3,
            16_666_667,
            4,
            5,
            PRESENTATION_FLAG_VSYNC | PRESENTATION_FLAG_HW_CLOCK | 0x80,
        ).unwrap();
        match feedback.result {
            WaylandFeedbackResult::Presented {
                output_id,
                presentation_timestamp_ns,
                refresh_interval_ns,
                sequence,
                flags,
            } => {
                assert_eq!(output_id.as_deref(), Some("output-1"));
                assert_eq!(presentation_timestamp_ns, "4294967298000000003");
                assert_eq!(refresh_interval_ns, Some(16_666_667));
                assert_eq!(sequence, Some((4_u64 << 32) | 5));
                assert_eq!(flags, PRESENTATION_FLAG_VSYNC | PRESENTATION_FLAG_HW_CLOCK | 0x80);
            }
            WaylandFeedbackResult::Discarded => panic!("expected presented feedback"),
        }
    }

    #[test]
    fn zero_refresh_is_preserved_as_unavailable() {
        let feedback = WaylandPresentationFeedback::presented_from_protocol(
            "request-zero-refresh",
            "surface-1",
            "frame-request-zero-refresh",
            CLOCK,
            None,
            0, 0, 100, 0, 0, 0, 0,
        ).unwrap();
        match feedback.result {
            WaylandFeedbackResult::Presented { refresh_interval_ns, sequence, output_id, .. } => {
                assert_eq!(refresh_interval_ns, None);
                assert_eq!(sequence, Some(0));
                assert_eq!(output_id, None);
            }
            WaylandFeedbackResult::Discarded => panic!("expected presented feedback"),
        }
    }

    #[test]
    fn malformed_raw_wayland_timestamp_is_rejected() {
        assert_eq!(
            WaylandPresentationFeedback::presented_from_protocol(
                "request-invalid",
                "surface-1",
                "frame-invalid",
                CLOCK,
                None,
                0, 1, 1_000_000_000, 0, 0, 0, 0,
            ),
            Err(PresentationFeedbackError::NanosecondsOutOfRange)
        );
    }

    #[test]
    fn discarded_feedback_never_becomes_observed() {
        let mut ledger = ledger();
        ledger.register_submission(submission("request-1")).unwrap();
        let record = ledger.record_feedback(presented("request-1")).unwrap();
        assert_eq!(record.outcome, PresentationOutcome::Presented);
        assert_eq!(record.output_id.as_deref(), Some("output-1"));
        assert_eq!(record.presentation_timestamp_ns.as_deref(), Some("1100"));
        assert_eq!(record.refresh_interval_ns, Some(16_666_667));
        assert_eq!(record.sequence, Some(42));
        assert_eq!(record.flags, Some(PRESENTATION_FLAG_VSYNC | PRESENTATION_FLAG_HW_CLOCK));
        let observation = record.output_observation.as_ref().unwrap();
        assert_eq!(observation.source, ObservationSource::Compositor);
        assert_eq!(observation.output_state, ObservedOutputState::Started);
        assert_eq!(ledger.pending_count(), 0);
    }

    #[test]
    fn discarded_feedback_never_becomes_observed() {
        let mut ledger = ledger();
        ledger.register_submission(submission("request-1")).unwrap();
        let mut feedback = presented("request-1");
        feedback.result = WaylandFeedbackResult::Discarded;
        let record = ledger.record_feedback(feedback).unwrap();
        assert_eq!(record.outcome, PresentationOutcome::Discarded);
        assert!(record.output_observation.is_none());
        assert!(record.presentation_timestamp_ns.is_none());
    }

    #[test]
    fn duplicate_terminal_feedback_is_rejected() {
        let mut ledger = ledger();
        ledger.register_submission(submission("request-1")).unwrap();
        ledger.record_feedback(presented("request-1")).unwrap();
        assert_eq!(
            ledger.record_feedback(presented("request-1")),
            Err(PresentationFeedbackError::DuplicateTerminalFeedback)
        );
    }

    #[test]
    fn malformed_surface_feedback_does_not_consume_pending_request() {
        let mut ledger = ledger();
        ledger.register_submission(submission("request-1")).unwrap();
        let mut feedback = presented("request-1");
        feedback.surface_id = "surface-other".to_owned();
        assert_eq!(
            ledger.record_feedback(feedback),
            Err(PresentationFeedbackError::SurfaceMismatch)
        );
        assert!(ledger.is_pending("request-1"));
    }

    #[test]
    fn wrong_clock_domain_feedback_does_not_consume_pending_request() {
        let mut ledger = ledger();
        ledger.register_submission(submission("request-1")).unwrap();
        let mut feedback = presented("request-1");
        feedback.clock_domain_id = "clock:other".to_owned();
        assert_eq!(
            ledger.record_feedback(feedback),
            Err(PresentationFeedbackError::ClockDomainMismatch)
        );
        assert!(ledger.is_pending("request-1"));
    }

    #[test]
    fn presentation_before_submission_is_rejected() {
        let mut ledger = ledger();
        ledger.register_submission(submission("request-1")).unwrap();
        let mut feedback = presented("request-1");
        feedback.result = WaylandFeedbackResult::Presented {
            output_id: Some("output-1".to_owned()),
            presentation_timestamp_ns: "999".to_owned(),
            refresh_interval_ns: None,
            sequence: None,
            flags: 0,
        };
        assert_eq!(
            ledger.record_feedback(feedback),
            Err(PresentationFeedbackError::TimestampBeforeSubmission)
        );
        assert!(ledger.is_pending("request-1"));
    }

    #[test]
    fn duplicate_request_identifiers_are_rejected_after_completion() {
        let mut ledger = ledger();
        ledger.register_submission(submission("request-1")).unwrap();
        ledger.record_feedback(presented("request-1")).unwrap();
        assert_eq!(
            ledger.register_submission(submission("request-1")),
            Err(PresentationFeedbackError::DuplicateRequest)
        );
    }

    #[test]
    fn pending_capacity_is_bounded() {
        let mut ledger = PresentationFeedbackLedger::new(
            "wayland-observer",
            "sha256:observer-build",
            1,
        )
        .unwrap();
        ledger.register_submission(submission("request-1")).unwrap();
        assert_eq!(
            ledger.register_submission(submission("request-2")),
            Err(PresentationFeedbackError::PendingCapacityReached)
        );
    }

    #[test]
    fn wayland_timestamp_conversion_checks_nanoseconds_and_overflow() {
        assert_eq!(wayland_timestamp_ns(0, 1, 25), Ok(1_000_000_025));
        assert_eq!(
            wayland_timestamp_ns(0, 1, 1_000_000_000),
            Err(PresentationFeedbackError::NanosecondsOutOfRange)
        );
        assert_eq!(
            wayland_timestamp_ns(u32::MAX, u32::MAX, 999_999_999),
            Err(PresentationFeedbackError::TimestampOverflow)
        );
    }

    #[test]
    fn unknown_future_flag_bits_are_preserved() {
        let mut ledger = ledger();
        ledger.register_submission(submission("request-1")).unwrap();
        let mut feedback = presented("request-1");
        feedback.result = WaylandFeedbackResult::Presented {
            output_id: Some("output-1".to_owned()),
            presentation_timestamp_ns: "1100".to_owned(),
            refresh_interval_ns: None,
            sequence: None,
            flags: KNOWN_PRESENTATION_FLAGS | 0x80,
        };
        let record = ledger.record_feedback(feedback).unwrap();
        assert_eq!(record.flags, Some(KNOWN_PRESENTATION_FLAGS | 0x80));
    }
}
