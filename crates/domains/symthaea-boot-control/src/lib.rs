// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Typed presentation-mode semantics for the Spore boot experience.
//!
//! This crate deliberately contains no VT ioctls, DRM operations, input-device
//! access, systemd control, or journal access. It answers only: given a user's
//! requested visibility and an automatic minimum visibility, what presentation
//! mode should the UI use?

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_boot_ecology_live::DiagnosticFloor;

/// v2 adds the non-interactive `Status` layer between Ambient and Diagnostics.
pub const CONTROL_PROTOCOL_VERSION: u16 = 2;
pub const MAX_CONTROL_WIRE_BYTES: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PresentationMode {
    /// Beautiful visual boot with no structured details in the foreground.
    Ambient,
    /// One restrained human-readable status cue while the ecology remains primary.
    Status,
    /// Full normalized boot domains/health diagnostics overlay.
    Diagnostics,
    /// Request handoff to the genuine Linux log VT.
    RawLogs,
}

impl PresentationMode {
    pub const fn rank(self) -> u8 {
        match self {
            Self::Ambient => 0,
            Self::Status => 1,
            Self::Diagnostics => 2,
            Self::RawLogs => 3,
        }
    }
}

/// Canonical automatic-visibility mapping shared by ecology and controls.
/// `DiagnosticFloor` can never request RawLogs.
impl From<DiagnosticFloor> for PresentationMode {
    fn from(value: DiagnosticFloor) -> Self {
        match value {
            DiagnosticFloor::Ambient => Self::Ambient,
            DiagnosticFloor::Status => Self::Status,
            DiagnosticFloor::Diagnostics => Self::Diagnostics,
        }
    }
}

/// Message accepted from the user-input side of the presentation controller.
///
/// This is a request, not an instruction to perform a privileged action. In
/// particular `RawLogs` still requires a separate coordinator to safely quiesce
/// DRM and perform a VT handoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PresentationRequest {
    pub protocol_version: u16,
    pub sequence: u64,
    pub requested: PresentationMode,
}

impl PresentationRequest {
    pub const fn new(sequence: u64, requested: PresentationMode) -> Self {
        Self {
            protocol_version: CONTROL_PROTOCOL_VERSION,
            sequence,
            requested,
        }
    }

    pub fn validate(&self) -> Result<(), ControlError> {
        if self.protocol_version != CONTROL_PROTOCOL_VERSION {
            return Err(ControlError::UnsupportedVersion(self.protocol_version));
        }
        if self.sequence == 0 {
            return Err(ControlError::ZeroSequence);
        }
        Ok(())
    }
}

/// Resolves user preference against automatic safety/diagnostic visibility.
///
/// The policy floor is intentionally capped at `Diagnostics`: ordinary health
/// policy can make trouble visible, but it cannot automatically throw the user
/// into a raw console. Raw VT handoff remains an explicit user/recovery action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PresentationArbiter {
    user_request: PresentationMode,
    policy_floor: PresentationMode,
    last_user_sequence: u64,
}

impl Default for PresentationArbiter {
    fn default() -> Self {
        Self {
            user_request: PresentationMode::Ambient,
            policy_floor: PresentationMode::Ambient,
            last_user_sequence: 0,
        }
    }
}

impl PresentationArbiter {
    pub const fn user_request(&self) -> PresentationMode {
        self.user_request
    }

    pub const fn policy_floor(&self) -> PresentationMode {
        self.policy_floor
    }

    pub const fn effective(&self) -> PresentationMode {
        if self.user_request.rank() >= self.policy_floor.rank() {
            self.user_request
        } else {
            self.policy_floor
        }
    }

    /// Apply a validated, monotonically newer user request.
    /// Returns `Ok(false)` for duplicate/stale sequence numbers.
    pub fn apply_user_request(
        &mut self,
        request: PresentationRequest,
    ) -> Result<bool, ControlError> {
        request.validate()?;
        if request.sequence <= self.last_user_sequence {
            return Ok(false);
        }
        self.last_user_sequence = request.sequence;
        self.user_request = request.requested;
        Ok(true)
    }

    /// Set automatic minimum visibility. Policy may request Ambient, Status, or
    /// Diagnostics, never RawLogs.
    pub fn set_policy_floor(&mut self, floor: PresentationMode) -> Result<(), ControlError> {
        if floor == PresentationMode::RawLogs {
            return Err(ControlError::RawLogsCannotBeAutomatic);
        }
        self.policy_floor = floor;
        Ok(())
    }

    /// Apply the canonical automatic floor derived from live boot semantics.
    pub fn set_diagnostic_floor(&mut self, floor: DiagnosticFloor) {
        self.policy_floor = floor.into();
    }
}

pub fn validate_control_datagram(bytes: &[u8]) -> Result<(), ControlError> {
    if bytes.len() > MAX_CONTROL_WIRE_BYTES {
        return Err(ControlError::DatagramTooLarge(bytes.len()));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlError {
    UnsupportedVersion(u16),
    ZeroSequence,
    RawLogsCannotBeAutomatic,
    DatagramTooLarge(usize),
}

impl std::fmt::Display for ControlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported presentation-control version {version}")
            }
            Self::ZeroSequence => write!(f, "presentation-control sequence must be non-zero"),
            Self::RawLogsCannotBeAutomatic => {
                write!(f, "raw-log mode cannot be established by automatic policy")
            }
            Self::DatagramTooLarge(bytes) => write!(
                f,
                "presentation-control datagram exceeds {MAX_CONTROL_WIRE_BYTES} bytes: {bytes}"
            ),
        }
    }
}

impl std::error::Error for ControlError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reducer_status_floor_surfaces_only_status() {
        let mut arbiter = PresentationArbiter::default();
        arbiter.set_diagnostic_floor(DiagnosticFloor::Status);
        assert_eq!(arbiter.policy_floor(), PresentationMode::Status);
        assert_eq!(arbiter.effective(), PresentationMode::Status);
    }

    #[test]
    fn diagnostics_floor_overrides_ambient_user_request() {
        let mut arbiter = PresentationArbiter::default();
        arbiter.set_diagnostic_floor(DiagnosticFloor::Diagnostics);
        assert_eq!(arbiter.effective(), PresentationMode::Diagnostics);
    }

    #[test]
    fn canonical_diagnostic_mapping_cannot_produce_raw_logs() {
        let mappings = [
            (DiagnosticFloor::Ambient, PresentationMode::Ambient),
            (DiagnosticFloor::Status, PresentationMode::Status),
            (DiagnosticFloor::Diagnostics, PresentationMode::Diagnostics),
        ];
        for (floor, expected) in mappings {
            assert_eq!(PresentationMode::from(floor), expected);
            assert_ne!(PresentationMode::from(floor), PresentationMode::RawLogs);
        }
    }

    #[test]
    fn user_can_request_raw_logs_explicitly() {
        let mut arbiter = PresentationArbiter::default();
        arbiter
            .apply_user_request(PresentationRequest::new(1, PresentationMode::RawLogs))
            .unwrap();
        assert_eq!(arbiter.effective(), PresentationMode::RawLogs);
    }

    #[test]
    fn automatic_policy_cannot_force_raw_logs() {
        let mut arbiter = PresentationArbiter::default();
        assert_eq!(
            arbiter.set_policy_floor(PresentationMode::RawLogs),
            Err(ControlError::RawLogsCannotBeAutomatic)
        );
        assert_eq!(arbiter.effective(), PresentationMode::Ambient);
    }

    #[test]
    fn user_diagnostics_request_stays_above_status_floor() {
        let mut arbiter = PresentationArbiter::default();
        arbiter
            .apply_user_request(PresentationRequest::new(1, PresentationMode::Diagnostics))
            .unwrap();
        arbiter.set_diagnostic_floor(DiagnosticFloor::Status);
        assert_eq!(arbiter.effective(), PresentationMode::Diagnostics);
    }

    #[test]
    fn stale_user_request_does_not_rewind_visibility() {
        let mut arbiter = PresentationArbiter::default();
        assert!(arbiter
            .apply_user_request(PresentationRequest::new(2, PresentationMode::Diagnostics))
            .unwrap());
        assert!(!arbiter
            .apply_user_request(PresentationRequest::new(1, PresentationMode::Ambient))
            .unwrap());
        assert_eq!(arbiter.user_request(), PresentationMode::Diagnostics);
    }

    #[test]
    fn recovery_can_lower_policy_floor_without_overwriting_user_choice() {
        let mut arbiter = PresentationArbiter::default();
        arbiter
            .apply_user_request(PresentationRequest::new(1, PresentationMode::Diagnostics))
            .unwrap();
        arbiter.set_diagnostic_floor(DiagnosticFloor::Diagnostics);
        arbiter.set_diagnostic_floor(DiagnosticFloor::Ambient);
        assert_eq!(arbiter.effective(), PresentationMode::Diagnostics);
    }

    #[test]
    fn control_datagram_has_tiny_hard_limit() {
        assert!(validate_control_datagram(&vec![0; MAX_CONTROL_WIRE_BYTES]).is_ok());
        assert!(validate_control_datagram(&vec![0; MAX_CONTROL_WIRE_BYTES + 1]).is_err());
    }
}
