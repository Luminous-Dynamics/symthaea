// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Correlated timestamped transport traces for Embodiment Contract v2.
//!
//! A delivery trace binds one physical command proposition to one transport
//! attempt without conflating delivery, acknowledgement, backend application,
//! or measured execution. Event timestamps are observation times in explicitly
//! named clock domains; cross-clock subtraction is rejected unless a separate
//! synchronization/alignment layer first establishes comparability.

use std::collections::HashMap;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::embodiment_action::{
    ActionStageBindingV1, ActionStageValidationError, PhysicalActionStageV1,
};
use crate::embodiment_evidence::{EvidenceValidationError, TimestampV1};
use crate::embodiment_transport::{
    TransportDeliveryBindingV1, TransportDeliveryStateV1, TransportDeliveryValidationError,
};

/// Schema version for [`TransportDeliveryTraceV1`].
pub const TRANSPORT_DELIVERY_TRACE_SCHEMA_V1: u16 = 1;
const TRANSPORT_TRACE_COMMITMENT_DOMAIN_V1: &[u8] =
    b"symthaea.embodiment.transport-delivery-trace.v1\0";

/// One timestamped delivery observation in a correlated transport attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransportDeliveryEventV1 {
    /// Contiguous 1-based event sequence within this transport attempt.
    pub sequence: u64,
    /// Delivery/application proposition observed at this event.
    pub binding: TransportDeliveryBindingV1,
    /// Observation time in the stated clock domain.
    ///
    /// For remote responses this is not automatically the remote occurrence time;
    /// it is the timestamp supplied by the evidence-producing observation path.
    pub observed_at: TimestampV1,
}

impl TransportDeliveryEventV1 {
    /// Validate sequence, nested delivery binding, and timestamp identity.
    pub fn validate(&self) -> Result<(), TransportTraceValidationError> {
        if self.sequence == 0 {
            return Err(TransportTraceValidationError::InvalidEventSequence {
                expected: 1,
                actual: 0,
            });
        }
        self.binding
            .validate()
            .map_err(TransportTraceValidationError::DeliveryBinding)?;
        self.observed_at
            .validate()
            .map_err(TransportTraceValidationError::Timestamp)?;
        Ok(())
    }
}

/// One correlated command-delivery attempt.
///
/// `source_action` identifies the physical causal stage of the payload entering
/// transport. Only requested and safety-projected physical actions are valid
/// transport sources. Backend-applied and measured-execution propositions are
/// downstream facts and cannot be transported "backward" into this trace.
///
/// The trace carries a domain-separated BLAKE3 content commitment over its
/// identity, source-action binding, delivery events, evidence references, clock
/// domains, and timestamps. That commitment detects post-capture mutation; it is
/// not a signature and does not authenticate the evidence producer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransportDeliveryTraceV1 {
    /// Schema version. Must equal [`TRANSPORT_DELIVERY_TRACE_SCHEMA_V1`].
    pub schema_version: u16,
    /// Transport/profile identity defining protocol-specific interpretation.
    pub transport_profile_id: String,
    /// Correlation/attempt identity. Retries should use a distinct trace identity.
    pub correlation_id: String,
    /// Identity of the exact payload/content submitted for this attempt.
    pub payload_id: String,
    /// Physical action stage of the payload entering transport.
    pub source_action: ActionStageBindingV1,
    /// Ordered observations for this single transport attempt.
    pub events: Vec<TransportDeliveryEventV1>,
    /// Domain-separated content commitment over every semantic field above.
    pub trace_digest_hex: String,
}

impl TransportDeliveryTraceV1 {
    /// Construct, commit, and validate one transport-delivery trace.
    pub fn new(
        transport_profile_id: impl Into<String>,
        correlation_id: impl Into<String>,
        payload_id: impl Into<String>,
        source_action: ActionStageBindingV1,
        events: Vec<TransportDeliveryEventV1>,
    ) -> Result<Self, TransportTraceValidationError> {
        let mut trace = Self {
            schema_version: TRANSPORT_DELIVERY_TRACE_SCHEMA_V1,
            transport_profile_id: transport_profile_id.into(),
            correlation_id: correlation_id.into(),
            payload_id: payload_id.into(),
            source_action,
            events,
            trace_digest_hex: String::new(),
        };
        trace.validate_without_digest()?;
        trace.trace_digest_hex = trace.compute_digest_hex();
        trace.validate()?;
        Ok(trace)
    }

    /// Validate structure plus the content commitment.
    pub fn validate(&self) -> Result<(), TransportTraceValidationError> {
        self.validate_without_digest()?;
        if self.trace_digest_hex != self.compute_digest_hex() {
            return Err(TransportTraceValidationError::DigestMismatch);
        }
        Ok(())
    }

    /// Stable evidence identifier for this exact trace content.
    ///
    /// The identifier is content-addressed but not an authentication claim.
    pub fn evidence_id(&self) -> Result<String, TransportTraceValidationError> {
        self.validate()?;
        Ok(format!(
            "symthaea.embodiment.transport-delivery-trace.v1:{}",
            self.trace_digest_hex
        ))
    }

    fn validate_without_digest(&self) -> Result<(), TransportTraceValidationError> {
        if self.schema_version != TRANSPORT_DELIVERY_TRACE_SCHEMA_V1 {
            return Err(TransportTraceValidationError::UnsupportedSchemaVersion {
                found: self.schema_version,
            });
        }

        validate_identifier(&self.transport_profile_id, "transport_profile_id")?;
        validate_identifier(&self.correlation_id, "correlation_id")?;
        validate_identifier(&self.payload_id, "payload_id")?;

        self.source_action
            .validate()
            .map_err(TransportTraceValidationError::SourceAction)?;
        if !matches!(
            self.source_action.stage,
            PhysicalActionStageV1::RequestedPhysicalProjection
                | PhysicalActionStageV1::SafetyProjectedPhysical
        ) {
            return Err(TransportTraceValidationError::InvalidSourceActionStage(
                self.source_action.stage,
            ));
        }

        if self.events.is_empty() {
            return Err(TransportTraceValidationError::EmptyTrace);
        }
        if self.events[0].binding.state != TransportDeliveryStateV1::DispatchRequested {
            return Err(TransportTraceValidationError::DispatchMustBeFirst);
        }

        let mut last_by_clock: HashMap<&str, u64> = HashMap::new();
        let mut terminal_state: Option<TransportDeliveryStateV1> = None;

        for (index, event) in self.events.iter().enumerate() {
            let expected = index as u64 + 1;
            if event.sequence != expected {
                return Err(TransportTraceValidationError::InvalidEventSequence {
                    expected,
                    actual: event.sequence,
                });
            }
            event.validate()?;

            if index > 0 && event.binding.state == TransportDeliveryStateV1::DispatchRequested {
                return Err(TransportTraceValidationError::DuplicateDispatch);
            }
            if let Some(terminal) = terminal_state {
                return Err(TransportTraceValidationError::EventAfterTerminalState {
                    terminal,
                    later: event.binding.state,
                });
            }

            let clock = event.observed_at.clock_domain.as_str();
            if let Some(previous) = last_by_clock.get(clock) {
                if event.observed_at.nanoseconds < *previous {
                    return Err(TransportTraceValidationError::NonMonotonicClock {
                        clock_domain: clock.to_string(),
                    });
                }
            }
            last_by_clock.insert(clock, event.observed_at.nanoseconds);

            if is_terminal_negative(event.binding.state) {
                terminal_state = Some(event.binding.state);
            }
        }

        Ok(())
    }

    /// Borrow one event by its 1-based sequence number.
    pub fn event(&self, sequence: u64) -> Option<&TransportDeliveryEventV1> {
        if sequence == 0 {
            return None;
        }
        let index = usize::try_from(sequence - 1).ok()?;
        self.events
            .get(index)
            .filter(|event| event.sequence == sequence)
    }

    /// Elapsed observation time between two events in the same clock domain.
    ///
    /// The whole trace is revalidated before arithmetic so public-field mutation or
    /// untrusted deserialization cannot obtain plausible latency from an invalid trace.
    /// This method deliberately fails across clock domains instead of assuming
    /// synchronization between host, middleware, device, or remote clocks.
    pub fn observed_elapsed_ns(
        &self,
        earlier_sequence: u64,
        later_sequence: u64,
    ) -> Result<u64, TransportTraceValidationError> {
        self.validate()?;
        if later_sequence <= earlier_sequence {
            return Err(TransportTraceValidationError::InvalidLatencyOrder {
                earlier: earlier_sequence,
                later: later_sequence,
            });
        }
        let earlier = self
            .event(earlier_sequence)
            .ok_or(TransportTraceValidationError::MissingEvent(earlier_sequence))?;
        let later = self
            .event(later_sequence)
            .ok_or(TransportTraceValidationError::MissingEvent(later_sequence))?;
        later
            .observed_at
            .elapsed_since(&earlier.observed_at)
            .map_err(TransportTraceValidationError::Timestamp)
    }

    fn compute_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(TRANSPORT_TRACE_COMMITMENT_DOMAIN_V1);
        hasher.update(&self.schema_version.to_le_bytes());
        feed_str(&mut hasher, &self.transport_profile_id);
        feed_str(&mut hasher, &self.correlation_id);
        feed_str(&mut hasher, &self.payload_id);

        hasher.update(&self.source_action.schema_version.to_le_bytes());
        feed_str(&mut hasher, self.source_action.stage.wire_token());
        feed_optional_str(&mut hasher, self.source_action.evidence_id.as_deref());

        hasher.update(&(self.events.len() as u64).to_le_bytes());
        for event in &self.events {
            hasher.update(&event.sequence.to_le_bytes());
            hasher.update(&event.binding.schema_version.to_le_bytes());
            feed_str(&mut hasher, event.binding.state.wire_token());
            feed_optional_str(&mut hasher, event.binding.evidence_id.as_deref());
            feed_str(&mut hasher, event.observed_at.clock_domain.as_str());
            hasher.update(&event.observed_at.nanoseconds.to_le_bytes());
        }

        digest_hex(hasher.finalize().as_bytes())
    }
}

fn is_terminal_negative(state: TransportDeliveryStateV1) -> bool {
    matches!(
        state,
        TransportDeliveryStateV1::RemoteApplicationRejected
            | TransportDeliveryStateV1::RemoteApplicationFailed
            | TransportDeliveryStateV1::TimedOut
            | TransportDeliveryStateV1::Cancelled
    )
}

fn validate_identifier(
    value: &str,
    field: &'static str,
) -> Result<(), TransportTraceValidationError> {
    let trimmed = value.trim();
    if trimmed.is_empty()
        || trimmed.len() != value.len()
        || value.chars().any(char::is_control)
    {
        return Err(TransportTraceValidationError::InvalidIdentifier(field));
    }
    Ok(())
}

fn feed_str(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn feed_optional_str(hasher: &mut blake3::Hasher, value: Option<&str>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            feed_str(hasher, value);
        }
        None => hasher.update(&[0]),
    }
}

fn digest_hex(bytes: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

/// Validation failure for [`TransportDeliveryTraceV1`].
#[derive(Debug, Clone, PartialEq)]
pub enum TransportTraceValidationError {
    /// Trace uses an unsupported schema version.
    UnsupportedSchemaVersion {
        /// Unsupported version encountered.
        found: u16,
    },
    /// Required trace identifier is empty, padded, or contains controls.
    InvalidIdentifier(&'static str),
    /// Nested source-action binding is invalid.
    SourceAction(ActionStageValidationError),
    /// Source action is downstream of command transport and is therefore invalid here.
    InvalidSourceActionStage(PhysicalActionStageV1),
    /// Trace contains no events.
    EmptyTrace,
    /// First event was not a dispatch request.
    DispatchMustBeFirst,
    /// A second dispatch request appeared in the same attempt.
    DuplicateDispatch,
    /// Event sequence was not contiguous and 1-based.
    InvalidEventSequence {
        /// Sequence expected at this position.
        expected: u64,
        /// Sequence actually present.
        actual: u64,
    },
    /// Nested delivery binding is invalid.
    DeliveryBinding(TransportDeliveryValidationError),
    /// Timestamp or same-clock elapsed-time validation failed.
    Timestamp(EvidenceValidationError),
    /// Observation time moved backward within one named clock domain.
    NonMonotonicClock {
        /// Clock domain whose observation sequence moved backward.
        clock_domain: String,
    },
    /// A later event appeared after a terminal negative outcome.
    EventAfterTerminalState {
        /// Earlier terminal state.
        terminal: TransportDeliveryStateV1,
        /// Invalid later state.
        later: TransportDeliveryStateV1,
    },
    /// Requested event sequence is absent.
    MissingEvent(u64),
    /// Latency helper was called with non-forward event order.
    InvalidLatencyOrder {
        /// Intended earlier event sequence.
        earlier: u64,
        /// Intended later event sequence.
        later: u64,
    },
    /// Content commitment does not match the trace fields.
    DigestMismatch,
}

impl std::fmt::Display for TransportTraceValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => {
                write!(f, "unsupported transport trace schema version {found}")
            }
            Self::InvalidIdentifier(field) => write!(f, "invalid {field} identifier"),
            Self::SourceAction(error) => write!(f, "invalid source action: {error}"),
            Self::InvalidSourceActionStage(stage) => {
                write!(f, "physical action stage {stage:?} cannot originate command transport")
            }
            Self::EmptyTrace => write!(f, "transport trace must contain at least one event"),
            Self::DispatchMustBeFirst => {
                write!(f, "transport trace must begin with dispatch_requested")
            }
            Self::DuplicateDispatch => write!(
                f,
                "one transport attempt may contain only one dispatch_requested event"
            ),
            Self::InvalidEventSequence { expected, actual } => write!(
                f,
                "invalid transport event sequence: expected {expected}, found {actual}"
            ),
            Self::DeliveryBinding(error) => {
                write!(f, "invalid transport delivery binding: {error}")
            }
            Self::Timestamp(error) => write!(f, "invalid transport timestamp: {error}"),
            Self::NonMonotonicClock { clock_domain } => write!(
                f,
                "transport observations moved backward in clock domain {clock_domain}"
            ),
            Self::EventAfterTerminalState { terminal, later } => write!(
                f,
                "transport event {later:?} appears after terminal state {terminal:?}"
            ),
            Self::MissingEvent(sequence) => {
                write!(f, "transport event sequence {sequence} is missing")
            }
            Self::InvalidLatencyOrder { earlier, later } => write!(
                f,
                "latency requires later sequence > earlier sequence ({later} <= {earlier})"
            ),
            Self::DigestMismatch => write!(f, "transport trace content commitment mismatch"),
        }
    }
}

impl std::error::Error for TransportTraceValidationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embodiment_action::ActionStageBindingV1;
    use crate::embodiment_evidence::ClockDomainId;

    fn ts(clock: &str, nanoseconds: u64) -> TimestampV1 {
        TimestampV1::new(ClockDomainId::new(clock).unwrap(), nanoseconds)
    }

    fn event(
        sequence: u64,
        state: TransportDeliveryStateV1,
        clock: &str,
        nanoseconds: u64,
    ) -> TransportDeliveryEventV1 {
        let evidence_id = if state == TransportDeliveryStateV1::DispatchRequested {
            None
        } else {
            Some(format!("transport:event:{sequence}"))
        };
        TransportDeliveryEventV1 {
            sequence,
            binding: TransportDeliveryBindingV1::new(state, evidence_id).unwrap(),
            observed_at: ts(clock, nanoseconds),
        }
    }

    fn requested_source() -> ActionStageBindingV1 {
        ActionStageBindingV1::requested_projection()
    }

    #[test]
    fn valid_same_clock_trace_and_latency() {
        let trace = TransportDeliveryTraceV1::new(
            "ros2.action.follow-joint-trajectory.v1",
            "goal:abc",
            "payload:sha256:01",
            requested_source(),
            vec![
                event(
                    1,
                    TransportDeliveryStateV1::DispatchRequested,
                    "host.monotonic",
                    100,
                ),
                event(
                    2,
                    TransportDeliveryStateV1::LocalTransportAccepted,
                    "host.monotonic",
                    130,
                ),
                event(
                    3,
                    TransportDeliveryStateV1::RemoteApplicationAccepted,
                    "host.monotonic",
                    190,
                ),
            ],
        )
        .unwrap();

        assert_eq!(trace.observed_elapsed_ns(1, 3).unwrap(), 90);
        assert!(trace
            .evidence_id()
            .unwrap()
            .starts_with("symthaea.embodiment.transport-delivery-trace.v1:"));
    }

    #[test]
    fn cross_clock_trace_validates_but_latency_fails_closed() {
        let trace = TransportDeliveryTraceV1::new(
            "mavlink.command.v1",
            "command:42",
            "payload:command-long:42",
            requested_source(),
            vec![
                event(
                    1,
                    TransportDeliveryStateV1::DispatchRequested,
                    "host.monotonic",
                    100,
                ),
                event(
                    2,
                    TransportDeliveryStateV1::RemoteReceiptConfirmed,
                    "autopilot.boot",
                    5_000,
                ),
            ],
        )
        .unwrap();

        assert_eq!(
            trace.observed_elapsed_ns(1, 2),
            Err(TransportTraceValidationError::Timestamp(
                EvidenceValidationError::ClockDomainMismatch
            ))
        );
    }

    #[test]
    fn backward_time_in_same_clock_fails() {
        let error = TransportDeliveryTraceV1::new(
            "ros2.topic.v1",
            "attempt:1",
            "payload:1",
            requested_source(),
            vec![
                event(
                    1,
                    TransportDeliveryStateV1::DispatchRequested,
                    "host.monotonic",
                    200,
                ),
                event(
                    2,
                    TransportDeliveryStateV1::LocalTransportAccepted,
                    "host.monotonic",
                    199,
                ),
            ],
        )
        .unwrap_err();

        assert_eq!(
            error,
            TransportTraceValidationError::NonMonotonicClock {
                clock_domain: "host.monotonic".to_string()
            }
        );
    }

    #[test]
    fn terminal_negative_state_admits_no_later_event() {
        for terminal in [
            TransportDeliveryStateV1::RemoteApplicationRejected,
            TransportDeliveryStateV1::RemoteApplicationFailed,
            TransportDeliveryStateV1::TimedOut,
            TransportDeliveryStateV1::Cancelled,
        ] {
            let error = TransportDeliveryTraceV1::new(
                "test.transport.v1",
                format!("attempt:{terminal:?}"),
                "payload:1",
                requested_source(),
                vec![
                    event(
                        1,
                        TransportDeliveryStateV1::DispatchRequested,
                        "host",
                        1,
                    ),
                    event(2, terminal, "remote", 20),
                    event(
                        3,
                        TransportDeliveryStateV1::RemoteReceiptConfirmed,
                        "another-clock",
                        1,
                    ),
                ],
            )
            .unwrap_err();

            assert!(matches!(
                error,
                TransportTraceValidationError::EventAfterTerminalState { .. }
            ));
        }
    }

    #[test]
    fn non_contiguous_sequence_fails() {
        let error = TransportDeliveryTraceV1::new(
            "test.transport.v1",
            "attempt:gap",
            "payload:1",
            requested_source(),
            vec![
                event(1, TransportDeliveryStateV1::DispatchRequested, "host", 1),
                event(
                    3,
                    TransportDeliveryStateV1::RemoteReceiptConfirmed,
                    "host",
                    3,
                ),
            ],
        )
        .unwrap_err();

        assert_eq!(
            error,
            TransportTraceValidationError::InvalidEventSequence {
                expected: 2,
                actual: 3
            }
        );
    }

    #[test]
    fn dispatch_must_be_first_and_unique() {
        let missing = TransportDeliveryTraceV1::new(
            "test.transport.v1",
            "attempt:no-dispatch",
            "payload:1",
            requested_source(),
            vec![event(
                1,
                TransportDeliveryStateV1::LocalTransportAccepted,
                "host",
                1,
            )],
        )
        .unwrap_err();
        assert_eq!(missing, TransportTraceValidationError::DispatchMustBeFirst);

        let duplicate = TransportDeliveryTraceV1::new(
            "test.transport.v1",
            "attempt:duplicate",
            "payload:1",
            requested_source(),
            vec![
                event(1, TransportDeliveryStateV1::DispatchRequested, "host", 1),
                event(2, TransportDeliveryStateV1::DispatchRequested, "host", 2),
            ],
        )
        .unwrap_err();
        assert_eq!(duplicate, TransportTraceValidationError::DuplicateDispatch);
    }

    #[test]
    fn downstream_physical_stages_cannot_originate_transport_trace() {
        for stage in [
            PhysicalActionStageV1::BackendAppliedPhysical,
            PhysicalActionStageV1::MeasuredExecution,
        ] {
            let source = ActionStageBindingV1::new(
                stage,
                Some("physical-evidence:01".to_string()),
            )
            .unwrap();
            let error = TransportDeliveryTraceV1::new(
                "test.transport.v1",
                format!("attempt:{stage:?}"),
                "payload:1",
                source,
                vec![event(1, TransportDeliveryStateV1::DispatchRequested, "host", 1)],
            )
            .unwrap_err();
            assert_eq!(
                error,
                TransportTraceValidationError::InvalidSourceActionStage(stage)
            );
        }
    }

    #[test]
    fn safety_projected_source_is_allowed_when_evidence_bound() {
        let source = ActionStageBindingV1::new(
            PhysicalActionStageV1::SafetyProjectedPhysical,
            Some("safety-projection:01".to_string()),
        )
        .unwrap();
        TransportDeliveryTraceV1::new(
            "test.transport.v1",
            "attempt:safety-projected",
            "payload:1",
            source,
            vec![event(1, TransportDeliveryStateV1::DispatchRequested, "host", 1)],
        )
        .unwrap();
    }

    #[test]
    fn latency_requires_forward_event_order() {
        let trace = TransportDeliveryTraceV1::new(
            "test.transport.v1",
            "attempt:latency-order",
            "payload:1",
            requested_source(),
            vec![
                event(1, TransportDeliveryStateV1::DispatchRequested, "host", 1),
                event(
                    2,
                    TransportDeliveryStateV1::LocalTransportAccepted,
                    "host",
                    2,
                ),
            ],
        )
        .unwrap();

        assert_eq!(
            trace.observed_elapsed_ns(2, 1),
            Err(TransportTraceValidationError::InvalidLatencyOrder {
                earlier: 2,
                later: 1
            })
        );
    }

    #[test]
    fn latency_revalidates_trace_before_arithmetic() {
        let mut trace = TransportDeliveryTraceV1::new(
            "test.transport.v1",
            "attempt:tamper",
            "payload:1",
            requested_source(),
            vec![
                event(1, TransportDeliveryStateV1::DispatchRequested, "host", 10),
                event(
                    2,
                    TransportDeliveryStateV1::LocalTransportAccepted,
                    "host",
                    20,
                ),
            ],
        )
        .unwrap();

        trace.events[1].sequence = 3;
        assert_eq!(
            trace.observed_elapsed_ns(1, 2),
            Err(TransportTraceValidationError::InvalidEventSequence {
                expected: 2,
                actual: 3
            })
        );
    }

    #[test]
    fn semantically_valid_payload_mutation_breaks_commitment() {
        let mut trace = TransportDeliveryTraceV1::new(
            "test.transport.v1",
            "attempt:digest-payload",
            "payload:1",
            requested_source(),
            vec![event(1, TransportDeliveryStateV1::DispatchRequested, "host", 10)],
        )
        .unwrap();

        trace.payload_id = "payload:2".to_string();
        assert_eq!(trace.validate(), Err(TransportTraceValidationError::DigestMismatch));
        assert_eq!(
            trace.evidence_id(),
            Err(TransportTraceValidationError::DigestMismatch)
        );
    }

    #[test]
    fn semantically_valid_timestamp_mutation_breaks_commitment_before_latency() {
        let mut trace = TransportDeliveryTraceV1::new(
            "test.transport.v1",
            "attempt:digest-time",
            "payload:1",
            requested_source(),
            vec![
                event(1, TransportDeliveryStateV1::DispatchRequested, "host", 10),
                event(
                    2,
                    TransportDeliveryStateV1::LocalTransportAccepted,
                    "host",
                    20,
                ),
            ],
        )
        .unwrap();

        trace.events[1].observed_at.nanoseconds = 21;
        assert_eq!(
            trace.observed_elapsed_ns(1, 2),
            Err(TransportTraceValidationError::DigestMismatch)
        );
    }

    #[test]
    fn impossible_platform_sequence_lookup_fails_without_integer_truncation() {
        let trace = TransportDeliveryTraceV1::new(
            "test.transport.v1",
            "attempt:lookup",
            "payload:1",
            requested_source(),
            vec![event(1, TransportDeliveryStateV1::DispatchRequested, "host", 1)],
        )
        .unwrap();

        assert!(trace.event(u64::MAX).is_none());
    }
}
