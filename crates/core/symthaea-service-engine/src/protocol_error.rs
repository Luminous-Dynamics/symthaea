// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Stable transport-facing failures for the owner-backed service runtime.
//!
//! Transports should not need to understand mailbox implementation details,
//! state-plane error enums, or owner lifecycle internals. This module collapses
//! those details into a small protocol taxonomy while preserving the daemon-v1
//! `{type:"error", message:...}` response for compatibility.

use std::fmt;

use serde::Serialize;

use crate::host::{ServiceHostCommandError, ServiceHostReadError};
use crate::wire::ServiceWireResponse;

/// Broad failure class suitable for transport policy and telemetry aggregation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceProtocolFailureClass {
    Overload,
    Availability,
    Exhaustion,
    Observation,
    Internal,
}

/// Stable failure independent of runtime implementation details.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServiceProtocolFailure {
    pub code: &'static str,
    pub class: ServiceProtocolFailureClass,
    pub message: &'static str,
    pub retryable: bool,
}

impl ServiceProtocolFailure {
    pub fn legacy_response(self) -> ServiceWireResponse {
        ServiceWireResponse::Error {
            message: self.message.to_string(),
        }
    }

    pub fn v2_response(self) -> ServiceProtocolErrorV2 {
        ServiceProtocolErrorV2 {
            response_type: "error_v2",
            code: self.code,
            class: self.class,
            message: self.message,
            retryable: self.retryable,
        }
    }
}

impl fmt::Display for ServiceProtocolFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message)
    }
}

impl std::error::Error for ServiceProtocolFailure {}

impl From<ServiceHostCommandError> for ServiceProtocolFailure {
    fn from(error: ServiceHostCommandError) -> Self {
        match error {
            ServiceHostCommandError::Busy => Self {
                code: "runtime_busy",
                class: ServiceProtocolFailureClass::Overload,
                message: "Symthaea is busy; retry the request after backoff",
                retryable: true,
            },
            ServiceHostCommandError::Closed => Self {
                code: "runtime_closed",
                class: ServiceProtocolFailureClass::Availability,
                message: "Symthaea runtime is not accepting commands",
                retryable: false,
            },
            ServiceHostCommandError::SequenceExhausted => Self {
                code: "runtime_sequence_exhausted",
                class: ServiceProtocolFailureClass::Exhaustion,
                message: "Symthaea runtime command sequence is exhausted",
                retryable: false,
            },
            ServiceHostCommandError::OwnerStopped => Self {
                code: "runtime_owner_stopped",
                class: ServiceProtocolFailureClass::Availability,
                message: "Symthaea runtime stopped before completing the request",
                retryable: false,
            },
            ServiceHostCommandError::UnexpectedReply => Self {
                code: "runtime_reply_mismatch",
                class: ServiceProtocolFailureClass::Internal,
                message: "Symthaea runtime produced an unexpected reply",
                retryable: false,
            },
        }
    }
}

impl From<ServiceHostReadError> for ServiceProtocolFailure {
    fn from(_error: ServiceHostReadError) -> Self {
        // Subscribe/read implementation details are intentionally collapsed here.
        // They may be recorded at the host boundary for local diagnostics, but the
        // transport contract exposes only a stable state-unavailable condition.
        Self {
            code: "runtime_state_unavailable",
            class: ServiceProtocolFailureClass::Observation,
            message: "Symthaea runtime state is temporarily unavailable",
            retryable: true,
        }
    }
}

/// Versioned machine-readable error for new transports/clients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ServiceProtocolErrorV2 {
    #[serde(rename = "type")]
    pub response_type: &'static str,
    pub code: &'static str,
    pub class: ServiceProtocolFailureClass,
    pub message: &'static str,
    pub retryable: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_interface_runtime::StatePlaneError;

    #[test]
    fn command_failures_have_stable_codes_and_retry_policy() {
        let cases = [
            (
                ServiceHostCommandError::Busy,
                "runtime_busy",
                ServiceProtocolFailureClass::Overload,
                true,
            ),
            (
                ServiceHostCommandError::Closed,
                "runtime_closed",
                ServiceProtocolFailureClass::Availability,
                false,
            ),
            (
                ServiceHostCommandError::SequenceExhausted,
                "runtime_sequence_exhausted",
                ServiceProtocolFailureClass::Exhaustion,
                false,
            ),
            (
                ServiceHostCommandError::OwnerStopped,
                "runtime_owner_stopped",
                ServiceProtocolFailureClass::Availability,
                false,
            ),
            (
                ServiceHostCommandError::UnexpectedReply,
                "runtime_reply_mismatch",
                ServiceProtocolFailureClass::Internal,
                false,
            ),
        ];

        for (source, code, class, retryable) in cases {
            let failure = ServiceProtocolFailure::from(source);
            assert_eq!(failure.code, code);
            assert_eq!(failure.class, class);
            assert_eq!(failure.retryable, retryable);
            assert_eq!(failure.to_string(), failure.message);
        }
    }

    #[test]
    fn v2_error_is_machine_readable_but_v1_shape_remains_simple() {
        let failure = ServiceProtocolFailure::from(ServiceHostCommandError::Busy);
        let v2 = serde_json::to_value(failure.v2_response()).unwrap();
        assert_eq!(v2["type"], "error_v2");
        assert_eq!(v2["code"], "runtime_busy");
        assert_eq!(v2["class"], "overload");
        assert_eq!(v2["retryable"], true);

        let legacy = serde_json::to_value(failure.legacy_response()).unwrap();
        assert_eq!(legacy["type"], "error");
        assert!(legacy.get("code").is_none());
        assert!(legacy.get("retryable").is_none());
    }

    #[test]
    fn read_plane_details_are_not_disclosed_on_wire() {
        let failure = ServiceProtocolFailure::from(ServiceHostReadError::Subscribe(
            StatePlaneError::RevisionExhausted,
        ));
        assert_eq!(failure.code, "runtime_state_unavailable");
        assert_eq!(failure.class, ServiceProtocolFailureClass::Observation);
        assert!(failure.retryable);
        assert_eq!(failure.to_string(), "Symthaea runtime state is temporarily unavailable");
    }
}
