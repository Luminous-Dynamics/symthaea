// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Transport-delivery evidence primitives for Embodiment Contract v2.
//!
//! Transport delivery and physical execution are different causal propositions.
//! A ROS/DDS publish, remote receipt, ROS action goal response, or MAVLink command
//! acknowledgement must not silently become evidence that a backend applied a
//! physical action. This module therefore models transport/application delivery
//! independently from [`crate::embodiment_action::PhysicalActionStageV1`].

use serde::{Deserialize, Serialize};

/// Schema version for [`TransportDeliveryBindingV1`].
pub const TRANSPORT_DELIVERY_BINDING_SCHEMA_V1: u16 = 1;

/// Evidence state for one transport/application delivery attempt.
///
/// The declaration order is descriptive only. This enum deliberately does not
/// derive `Ord`: delivery states are not a trust, safety, or execution ranking,
/// and protocols do not necessarily traverse every state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransportDeliveryStateV1 {
    /// Local code requested that a payload be dispatched.
    DispatchRequested,
    /// The local middleware/driver accepted the payload for transmission.
    LocalTransportAccepted,
    /// Protocol evidence confirms receipt by the intended remote endpoint.
    RemoteReceiptConfirmed,
    /// The remote application/controller explicitly accepted the request.
    RemoteApplicationAccepted,
    /// The remote application/controller explicitly rejected or denied the request.
    RemoteApplicationRejected,
    /// The remote application/controller reported that processing the request failed.
    ///
    /// This remains an application/transport proposition. It does not establish
    /// whether any physical action was partially or fully applied before failure.
    RemoteApplicationFailed,
    /// The expected transport/application response did not arrive before the applicable deadline.
    TimedOut,
    /// The transport/application request was cancelled.
    Cancelled,
}

impl TransportDeliveryStateV1 {
    /// Stable snake-case token for evidence/wire interoperability.
    pub const fn wire_token(self) -> &'static str {
        match self {
            Self::DispatchRequested => "dispatch_requested",
            Self::LocalTransportAccepted => "local_transport_accepted",
            Self::RemoteReceiptConfirmed => "remote_receipt_confirmed",
            Self::RemoteApplicationAccepted => "remote_application_accepted",
            Self::RemoteApplicationRejected => "remote_application_rejected",
            Self::RemoteApplicationFailed => "remote_application_failed",
            Self::TimedOut => "timed_out",
            Self::Cancelled => "cancelled",
        }
    }

    /// Whether this state makes a boundary claim requiring an evidence reference.
    ///
    /// `DispatchRequested` may be a deterministic local derivation. Every stronger
    /// transport/application claim must bind the record that established it.
    pub const fn requires_evidence_id(self) -> bool {
        !matches!(self, Self::DispatchRequested)
    }
}

/// Evidence binding for one transport/application delivery state.
///
/// This binding intentionally carries no physical-action stage and exposes no API
/// that promotes transport evidence into `BackendAppliedPhysical` or
/// `MeasuredExecution`. Physical application and independent execution feedback
/// must be established by their own evidence records.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransportDeliveryBindingV1 {
    /// Schema version. Must equal [`TRANSPORT_DELIVERY_BINDING_SCHEMA_V1`].
    pub schema_version: u16,
    /// Transport/application delivery state being claimed.
    pub state: TransportDeliveryStateV1,
    /// Evidence record supporting the state; required after `DispatchRequested`.
    pub evidence_id: Option<String>,
}

impl TransportDeliveryBindingV1 {
    /// Construct and validate one transport-delivery binding.
    pub fn new(
        state: TransportDeliveryStateV1,
        evidence_id: Option<String>,
    ) -> Result<Self, TransportDeliveryValidationError> {
        let value = Self {
            schema_version: TRANSPORT_DELIVERY_BINDING_SCHEMA_V1,
            state,
            evidence_id,
        };
        value.validate()?;
        Ok(value)
    }

    /// Convenience constructor for the local dispatch-request proposition.
    pub fn dispatch_requested() -> Self {
        Self {
            schema_version: TRANSPORT_DELIVERY_BINDING_SCHEMA_V1,
            state: TransportDeliveryStateV1::DispatchRequested,
            evidence_id: None,
        }
    }

    /// Validate schema and evidence-strength invariants.
    pub fn validate(&self) -> Result<(), TransportDeliveryValidationError> {
        if self.schema_version != TRANSPORT_DELIVERY_BINDING_SCHEMA_V1 {
            return Err(TransportDeliveryValidationError::UnsupportedSchemaVersion {
                found: self.schema_version,
            });
        }

        if let Some(evidence_id) = &self.evidence_id {
            validate_evidence_id(evidence_id)?;
        }

        if self.state.requires_evidence_id() && self.evidence_id.is_none() {
            return Err(TransportDeliveryValidationError::MissingEvidenceForState(
                self.state,
            ));
        }

        Ok(())
    }
}

fn validate_evidence_id(value: &str) -> Result<(), TransportDeliveryValidationError> {
    let trimmed = value.trim();
    if trimmed.is_empty()
        || trimmed.len() != value.len()
        || value.chars().any(char::is_control)
    {
        return Err(TransportDeliveryValidationError::InvalidEvidenceId);
    }
    Ok(())
}

/// Validation failure for [`TransportDeliveryBindingV1`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransportDeliveryValidationError {
    /// Binding uses an unsupported schema version.
    UnsupportedSchemaVersion {
        /// Unsupported version encountered.
        found: u16,
    },
    /// A transport/application boundary claim omitted its evidence reference.
    MissingEvidenceForState(TransportDeliveryStateV1),
    /// Evidence reference was empty, padded, or contained control characters.
    InvalidEvidenceId,
}

impl std::fmt::Display for TransportDeliveryValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => {
                write!(f, "unsupported transport-delivery schema version {found}")
            }
            Self::MissingEvidenceForState(state) => {
                write!(f, "transport-delivery state {state:?} requires evidence")
            }
            Self::InvalidEvidenceId => write!(f, "invalid transport-delivery evidence identifier"),
        }
    }
}

impl std::error::Error for TransportDeliveryValidationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_tokens_are_stable() {
        let cases = [
            (TransportDeliveryStateV1::DispatchRequested, "dispatch_requested"),
            (
                TransportDeliveryStateV1::LocalTransportAccepted,
                "local_transport_accepted",
            ),
            (
                TransportDeliveryStateV1::RemoteReceiptConfirmed,
                "remote_receipt_confirmed",
            ),
            (
                TransportDeliveryStateV1::RemoteApplicationAccepted,
                "remote_application_accepted",
            ),
            (
                TransportDeliveryStateV1::RemoteApplicationRejected,
                "remote_application_rejected",
            ),
            (
                TransportDeliveryStateV1::RemoteApplicationFailed,
                "remote_application_failed",
            ),
            (TransportDeliveryStateV1::TimedOut, "timed_out"),
            (TransportDeliveryStateV1::Cancelled, "cancelled"),
        ];

        for (state, expected) in cases {
            assert_eq!(state.wire_token(), expected);
            assert_eq!(serde_json::to_string(&state).unwrap(), format!("\"{expected}\""));
        }
    }

    #[test]
    fn dispatch_request_needs_no_external_evidence() {
        let binding = TransportDeliveryBindingV1::dispatch_requested();
        binding.validate().unwrap();
        assert_eq!(binding.state, TransportDeliveryStateV1::DispatchRequested);
        assert_eq!(binding.evidence_id, None);
    }

    #[test]
    fn every_stronger_state_requires_evidence() {
        for state in [
            TransportDeliveryStateV1::LocalTransportAccepted,
            TransportDeliveryStateV1::RemoteReceiptConfirmed,
            TransportDeliveryStateV1::RemoteApplicationAccepted,
            TransportDeliveryStateV1::RemoteApplicationRejected,
            TransportDeliveryStateV1::RemoteApplicationFailed,
            TransportDeliveryStateV1::TimedOut,
            TransportDeliveryStateV1::Cancelled,
        ] {
            assert_eq!(
                TransportDeliveryBindingV1::new(state, None),
                Err(TransportDeliveryValidationError::MissingEvidenceForState(state))
            );
            TransportDeliveryBindingV1::new(
                state,
                Some("transport-evidence:trace:01".to_string()),
            )
            .unwrap();
        }
    }

    #[test]
    fn malformed_evidence_ids_are_rejected() {
        for invalid in ["", "   ", " padded", "padded ", "bad\nvalue"] {
            assert_eq!(
                TransportDeliveryBindingV1::new(
                    TransportDeliveryStateV1::RemoteReceiptConfirmed,
                    Some(invalid.to_string()),
                ),
                Err(TransportDeliveryValidationError::InvalidEvidenceId)
            );
        }
    }

    #[test]
    fn local_dispatch_may_be_evidence_bound() {
        let binding = TransportDeliveryBindingV1::new(
            TransportDeliveryStateV1::DispatchRequested,
            Some("dispatch:local-call:42".to_string()),
        )
        .unwrap();
        assert_eq!(binding.evidence_id.as_deref(), Some("dispatch:local-call:42"));
    }

    #[test]
    fn serialization_round_trip_preserves_binding() {
        let original = TransportDeliveryBindingV1::new(
            TransportDeliveryStateV1::RemoteApplicationAccepted,
            Some("ros-action:goal-response:abc".to_string()),
        )
        .unwrap();
        let bytes = serde_json::to_vec(&original).unwrap();
        let restored: TransportDeliveryBindingV1 = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(restored, original);
        restored.validate().unwrap();
    }

    #[test]
    fn rejected_and_failed_remain_distinct_remote_outcomes() {
        assert_ne!(
            TransportDeliveryStateV1::RemoteApplicationRejected,
            TransportDeliveryStateV1::RemoteApplicationFailed
        );
        assert_ne!(
            TransportDeliveryStateV1::RemoteApplicationRejected.wire_token(),
            TransportDeliveryStateV1::RemoteApplicationFailed.wire_token()
        );
    }

    #[test]
    fn unsupported_schema_version_fails_closed() {
        let mut binding = TransportDeliveryBindingV1::dispatch_requested();
        binding.schema_version = 2;
        assert_eq!(
            binding.validate(),
            Err(TransportDeliveryValidationError::UnsupportedSchemaVersion { found: 2 })
        );
    }

    #[test]
    fn transport_and_physical_stage_vocabularies_remain_distinct() {
        use crate::embodiment_action::PhysicalActionStageV1;

        assert_ne!(
            TransportDeliveryStateV1::RemoteApplicationAccepted.wire_token(),
            PhysicalActionStageV1::BackendAppliedPhysical.wire_token()
        );
        assert_ne!(
            TransportDeliveryStateV1::RemoteReceiptConfirmed.wire_token(),
            PhysicalActionStageV1::MeasuredExecution.wire_token()
        );
    }
}
