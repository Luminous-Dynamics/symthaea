// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Explicit physical-hardware execution boundary.
//!
//! Simulation and physical actuation are different constructors and states.
//! A missing backend, stale sensor frame, expired authority token, sequence
//! regression, or watchdog timeout disarms the bridge; it never silently falls
//! back to simulation while claiming that commands reached hardware.

use serde::{Deserialize, Serialize};

use crate::types::{HelicopterCommand, HelicopterState};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HardwareBackendKind {
    SimulationOnly,
    PhysicalHardware,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HardwareBridgeState {
    Disarmed,
    Armed,
    Faulted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareIoError {
    Unavailable,
    SelfTestFailed,
    ReadFailed,
    WriteFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareBridgeError {
    WrongBackendKind,
    InvalidConfiguration,
    InvalidAuthorityToken,
    AuthorityExpired,
    BackendSelfTestFailed,
    BridgeNotArmed,
    NonFiniteTime,
    TimeWentBackwards,
    SequenceDidNotIncrease,
    SensorUnavailable,
    SensorStale,
    SensorNonFinite,
    CommandNonFinite,
    CommandWatchdogExpired,
    HardwareWriteFailed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HardwareAuthorityToken {
    pub authority_id: String,
    pub issued_at_s: f64,
    pub expires_at_s: f64,
    pub mission_id: String,
}

impl HardwareAuthorityToken {
    pub fn validate_at(&self, now_s: f64) -> Result<(), HardwareBridgeError> {
        if self.authority_id.trim().is_empty()
            || self.mission_id.trim().is_empty()
            || !self.issued_at_s.is_finite()
            || !self.expires_at_s.is_finite()
            || !now_s.is_finite()
            || self.expires_at_s <= self.issued_at_s
            || now_s < self.issued_at_s
        {
            return Err(HardwareBridgeError::InvalidAuthorityToken);
        }
        if now_s >= self.expires_at_s {
            return Err(HardwareBridgeError::AuthorityExpired);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareSensorFrame {
    pub sequence: u64,
    pub monotonic_time_s: f64,
    pub state: HelicopterState,
    pub backend_health_ok: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HardwareCommandFrame {
    pub sequence: u64,
    pub monotonic_time_s: f64,
    pub command: HelicopterCommand,
}

pub trait HelicopterHardwareIo {
    fn backend_kind(&self) -> HardwareBackendKind;
    fn self_test(&mut self) -> Result<(), HardwareIoError>;
    fn read_sensor_frame(&mut self) -> Result<HardwareSensorFrame, HardwareIoError>;
    fn write_command_frame(&mut self, frame: HardwareCommandFrame) -> Result<(), HardwareIoError>;
    fn disarm_outputs(&mut self) -> Result<(), HardwareIoError>;
}

/// Deliberately unavailable backend used when physical I/O is not installed.
/// It identifies itself as simulation-only and rejects every physical action.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullHardwareIo;

impl HelicopterHardwareIo for NullHardwareIo {
    fn backend_kind(&self) -> HardwareBackendKind {
        HardwareBackendKind::SimulationOnly
    }

    fn self_test(&mut self) -> Result<(), HardwareIoError> {
        Err(HardwareIoError::Unavailable)
    }

    fn read_sensor_frame(&mut self) -> Result<HardwareSensorFrame, HardwareIoError> {
        Err(HardwareIoError::Unavailable)
    }

    fn write_command_frame(&mut self, _frame: HardwareCommandFrame) -> Result<(), HardwareIoError> {
        Err(HardwareIoError::Unavailable)
    }

    fn disarm_outputs(&mut self) -> Result<(), HardwareIoError> {
        Err(HardwareIoError::Unavailable)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HardwareSafetyConfig {
    pub maximum_sensor_age_s: f64,
    pub maximum_command_gap_s: f64,
    pub maximum_authority_duration_s: f64,
}

impl Default for HardwareSafetyConfig {
    fn default() -> Self {
        Self {
            maximum_sensor_age_s: 0.050,
            maximum_command_gap_s: 0.100,
            maximum_authority_duration_s: 24.0 * 60.0 * 60.0,
        }
    }
}

impl HardwareSafetyConfig {
    pub fn validate(&self) -> bool {
        self.maximum_sensor_age_s.is_finite()
            && self.maximum_sensor_age_s > 0.0
            && self.maximum_command_gap_s.is_finite()
            && self.maximum_command_gap_s > 0.0
            && self.maximum_authority_duration_s.is_finite()
            && self.maximum_authority_duration_s > 0.0
    }
}

#[derive(Debug)]
pub struct HelicopterHardwareBridge<I: HelicopterHardwareIo> {
    io: I,
    config: HardwareSafetyConfig,
    state: HardwareBridgeState,
    authority: Option<HardwareAuthorityToken>,
    last_time_s: Option<f64>,
    last_sensor_sequence: Option<u64>,
    last_command_sequence: Option<u64>,
    last_command_time_s: Option<f64>,
}

impl<I: HelicopterHardwareIo> HelicopterHardwareBridge<I> {
    /// Construct a physical bridge. Simulation-only backends are rejected.
    pub fn new_physical(io: I, config: HardwareSafetyConfig) -> Result<Self, HardwareBridgeError> {
        if !config.validate() {
            return Err(HardwareBridgeError::InvalidConfiguration);
        }
        if io.backend_kind() != HardwareBackendKind::PhysicalHardware {
            return Err(HardwareBridgeError::WrongBackendKind);
        }
        Ok(Self {
            io,
            config,
            state: HardwareBridgeState::Disarmed,
            authority: None,
            last_time_s: None,
            last_sensor_sequence: None,
            last_command_sequence: None,
            last_command_time_s: None,
        })
    }

    pub fn state(&self) -> HardwareBridgeState {
        self.state
    }

    pub fn arm(
        &mut self,
        authority: HardwareAuthorityToken,
        now_s: f64,
    ) -> Result<(), HardwareBridgeError> {
        authority.validate_at(now_s)?;
        if authority.expires_at_s - authority.issued_at_s > self.config.maximum_authority_duration_s
        {
            return Err(HardwareBridgeError::InvalidAuthorityToken);
        }
        self.io
            .self_test()
            .map_err(|_| HardwareBridgeError::BackendSelfTestFailed)?;
        self.authority = Some(authority);
        self.state = HardwareBridgeState::Armed;
        self.last_time_s = Some(now_s);
        self.last_sensor_sequence = None;
        self.last_command_sequence = None;
        self.last_command_time_s = Some(now_s);
        Ok(())
    }

    fn validate_time(&mut self, now_s: f64) -> Result<(), HardwareBridgeError> {
        if !now_s.is_finite() {
            return Err(HardwareBridgeError::NonFiniteTime);
        }
        if self.last_time_s.is_some_and(|previous| now_s < previous) {
            self.fault_and_disarm();
            return Err(HardwareBridgeError::TimeWentBackwards);
        }
        self.last_time_s = Some(now_s);
        Ok(())
    }

    fn validate_authority(&mut self, now_s: f64) -> Result<(), HardwareBridgeError> {
        let result = self
            .authority
            .as_ref()
            .ok_or(HardwareBridgeError::BridgeNotArmed)
            .and_then(|authority| authority.validate_at(now_s));
        if result.is_err() {
            self.fault_and_disarm();
        }
        result
    }

    /// Read one fresh, finite, strictly sequenced hardware sensor frame.
    pub fn read_state(&mut self, now_s: f64) -> Result<HardwareSensorFrame, HardwareBridgeError> {
        if self.state != HardwareBridgeState::Armed {
            return Err(HardwareBridgeError::BridgeNotArmed);
        }
        self.validate_time(now_s)?;
        self.validate_authority(now_s)?;
        let frame = match self.io.read_sensor_frame() {
            Ok(frame) => frame,
            Err(_) => {
                self.fault_and_disarm();
                return Err(HardwareBridgeError::SensorUnavailable);
            }
        };
        if !frame.monotonic_time_s.is_finite()
            || now_s < frame.monotonic_time_s
            || now_s - frame.monotonic_time_s > self.config.maximum_sensor_age_s
        {
            self.fault_and_disarm();
            return Err(HardwareBridgeError::SensorStale);
        }
        if !frame.state.is_finite() || !frame.backend_health_ok {
            self.fault_and_disarm();
            return Err(HardwareBridgeError::SensorNonFinite);
        }
        if self
            .last_sensor_sequence
            .is_some_and(|previous| frame.sequence <= previous)
        {
            self.fault_and_disarm();
            return Err(HardwareBridgeError::SequenceDidNotIncrease);
        }
        self.last_sensor_sequence = Some(frame.sequence);
        Ok(frame)
    }

    /// Write one finite, strictly sequenced command after authority/watchdog checks.
    pub fn write_command(
        &mut self,
        frame: HardwareCommandFrame,
    ) -> Result<(), HardwareBridgeError> {
        if self.state != HardwareBridgeState::Armed {
            return Err(HardwareBridgeError::BridgeNotArmed);
        }
        self.validate_time(frame.monotonic_time_s)?;
        self.validate_authority(frame.monotonic_time_s)?;
        if self
            .last_command_sequence
            .is_some_and(|previous| frame.sequence <= previous)
        {
            self.fault_and_disarm();
            return Err(HardwareBridgeError::SequenceDidNotIncrease);
        }
        if self.last_command_time_s.is_some_and(|previous| {
            frame.monotonic_time_s - previous > self.config.maximum_command_gap_s
        }) {
            self.fault_and_disarm();
            return Err(HardwareBridgeError::CommandWatchdogExpired);
        }
        if !frame
            .command
            .to_ctrl()
            .iter()
            .all(|value| value.is_finite())
        {
            self.fault_and_disarm();
            return Err(HardwareBridgeError::CommandNonFinite);
        }
        self.io
            .write_command_frame(HardwareCommandFrame {
                command: frame.command.clamped(),
                ..frame
            })
            .map_err(|_| {
                self.fault_and_disarm();
                HardwareBridgeError::HardwareWriteFailed
            })?;
        self.last_command_sequence = Some(frame.sequence);
        self.last_command_time_s = Some(frame.monotonic_time_s);
        Ok(())
    }

    pub fn disarm(&mut self) {
        let _ = self.io.disarm_outputs();
        self.state = HardwareBridgeState::Disarmed;
        self.authority = None;
    }

    fn fault_and_disarm(&mut self) {
        let _ = self.io.disarm_outputs();
        self.state = HardwareBridgeState::Faulted;
        self.authority = None;
    }

    pub fn into_inner(self) -> I {
        self.io
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct MockPhysicalIo {
        frame: HardwareSensorFrame,
        writes: Vec<HardwareCommandFrame>,
        disarmed: bool,
    }

    impl HelicopterHardwareIo for MockPhysicalIo {
        fn backend_kind(&self) -> HardwareBackendKind {
            HardwareBackendKind::PhysicalHardware
        }

        fn self_test(&mut self) -> Result<(), HardwareIoError> {
            Ok(())
        }

        fn read_sensor_frame(&mut self) -> Result<HardwareSensorFrame, HardwareIoError> {
            Ok(self.frame.clone())
        }

        fn write_command_frame(
            &mut self,
            frame: HardwareCommandFrame,
        ) -> Result<(), HardwareIoError> {
            self.writes.push(frame);
            Ok(())
        }

        fn disarm_outputs(&mut self) -> Result<(), HardwareIoError> {
            self.disarmed = true;
            Ok(())
        }
    }

    fn token() -> HardwareAuthorityToken {
        HardwareAuthorityToken {
            authority_id: "operator-key-hash".to_string(),
            issued_at_s: 0.0,
            expires_at_s: 60.0,
            mission_id: "SAR-001".to_string(),
        }
    }

    fn io() -> MockPhysicalIo {
        MockPhysicalIo {
            frame: HardwareSensorFrame {
                sequence: 1,
                monotonic_time_s: 0.01,
                state: HelicopterState::hover(20.0),
                backend_health_ok: true,
            },
            writes: Vec::new(),
            disarmed: false,
        }
    }

    #[test]
    fn null_backend_cannot_be_constructed_as_physical() {
        assert!(matches!(
            HelicopterHardwareBridge::new_physical(NullHardwareIo, HardwareSafetyConfig::default(),),
            Err(HardwareBridgeError::WrongBackendKind)
        ));
    }

    #[test]
    fn valid_authority_allows_fresh_read_and_command() {
        let mut bridge =
            HelicopterHardwareBridge::new_physical(io(), HardwareSafetyConfig::default()).unwrap();
        bridge.arm(token(), 0.0).unwrap();
        assert!(bridge.read_state(0.02).is_ok());
        bridge
            .write_command(HardwareCommandFrame {
                sequence: 1,
                monotonic_time_s: 0.03,
                command: HelicopterCommand::hover(),
            })
            .unwrap();
        assert_eq!(bridge.state(), HardwareBridgeState::Armed);
        assert_eq!(bridge.into_inner().writes.len(), 1);
    }

    #[test]
    fn stale_sensor_faults_and_disarms() {
        let mut backend = io();
        backend.frame.monotonic_time_s = 0.0;
        let mut bridge =
            HelicopterHardwareBridge::new_physical(backend, HardwareSafetyConfig::default())
                .unwrap();
        bridge.arm(token(), 0.0).unwrap();
        assert!(matches!(
            bridge.read_state(1.0),
            Err(HardwareBridgeError::SensorStale)
        ));
        assert_eq!(bridge.state(), HardwareBridgeState::Faulted);
        assert!(bridge.into_inner().disarmed);
    }

    #[test]
    fn command_gap_trips_watchdog() {
        let mut bridge =
            HelicopterHardwareBridge::new_physical(io(), HardwareSafetyConfig::default()).unwrap();
        bridge.arm(token(), 0.0).unwrap();
        assert_eq!(
            bridge.write_command(HardwareCommandFrame {
                sequence: 1,
                monotonic_time_s: 0.2,
                command: HelicopterCommand::hover(),
            }),
            Err(HardwareBridgeError::CommandWatchdogExpired)
        );
        assert_eq!(bridge.state(), HardwareBridgeState::Faulted);
    }

    #[test]
    fn expired_authority_never_writes() {
        let mut bridge =
            HelicopterHardwareBridge::new_physical(io(), HardwareSafetyConfig::default()).unwrap();
        bridge.arm(token(), 0.0).unwrap();
        assert_eq!(
            bridge.write_command(HardwareCommandFrame {
                sequence: 1,
                monotonic_time_s: 60.0,
                command: HelicopterCommand::hover(),
            }),
            Err(HardwareBridgeError::AuthorityExpired)
        );
        assert!(bridge.into_inner().writes.is_empty());
    }
}
