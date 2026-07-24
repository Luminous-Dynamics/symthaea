// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Machine-profile validation and staged G-code authority.
//!
//! A [`GCodeProgram`] is descriptive output. A [`ValidatedGCode`] is the
//! capability-bearing form that has passed explicit build-volume, feed-rate,
//! temperature, homing, and retraction checks for one machine profile.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::printer_control::{PrinterApi, PrinterError};
use crate::toolpath::{GCodeCommand, GCodeProgram};
use serde::{Deserialize, Serialize};

pub const MAX_MACHINE_ID_BYTES: usize = 256;
pub const MAX_SESSION_NONCE_BYTES: usize = 256;
pub const MAX_MACHINE_PROFILE_NAME_BYTES: usize = 256;

/// Limits enforced before a G-code program may be submitted to a printer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MachineProfile {
    pub name: String,
    /// Inclusive minimum machine coordinates in millimetres.
    pub build_min_mm: [f32; 3],
    /// Inclusive maximum machine coordinates in millimetres.
    pub build_max_mm: [f32; 3],
    /// Maximum commanded feed rate in millimetres per minute.
    pub max_feed_rate_mm_min: f32,
    pub max_nozzle_temp_c: u16,
    pub max_bed_temp_c: u16,
    /// Largest permitted single decrease in absolute extrusion position.
    pub max_retraction_mm: f32,
    /// Require a home command before the first motion command.
    pub require_homing: bool,
}

impl Default for MachineProfile {
    fn default() -> Self {
        Self {
            name: "generic-fdm-220".into(),
            build_min_mm: [0.0, 0.0, 0.0],
            build_max_mm: [220.0, 220.0, 250.0],
            max_feed_rate_mm_min: 18_000.0,
            max_nozzle_temp_c: 300,
            max_bed_temp_c: 130,
            max_retraction_mm: 8.0,
            require_homing: true,
        }
    }
}

impl MachineProfile {
    /// Validate profile invariants before using it as an authority boundary.
    pub fn validate(&self) -> Result<(), &'static str> {
        for axis in 0..3 {
            let min = self.build_min_mm[axis];
            let max = self.build_max_mm[axis];
            if !min.is_finite() || !max.is_finite() || min >= max {
                return Err("build bounds must be finite and ordered");
            }
        }
        if !self.max_feed_rate_mm_min.is_finite() || self.max_feed_rate_mm_min <= 0.0 {
            return Err("maximum feed rate must be finite and positive");
        }
        if !self.max_retraction_mm.is_finite() || self.max_retraction_mm < 0.0 {
            return Err("maximum retraction must be finite and non-negative");
        }
        if !canonical_identifier(&self.name, MAX_MACHINE_PROFILE_NAME_BYTES) {
            return Err("machine profile name must be canonical and bounded");
        }
        if self.max_nozzle_temp_c == 0 {
            return Err("maximum nozzle temperature must be positive");
        }
        if self.max_bed_temp_c == 0 {
            return Err("maximum bed temperature must be positive");
        }
        Ok(())
    }
}

/// Capabilities advertised by one concrete machine endpoint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MachineCapabilities {
    pub machine_id: String,
    pub build_min_mm: [f32; 3],
    pub build_max_mm: [f32; 3],
    pub max_feed_rate_mm_min: f32,
    pub max_nozzle_temp_c: u16,
    pub max_bed_temp_c: u16,
    pub max_retraction_mm: f32,
    pub supports_homing: bool,
}

impl MachineCapabilities {
    pub fn from_profile(machine_id: impl Into<String>, profile: &MachineProfile) -> Self {
        Self {
            machine_id: machine_id.into(),
            build_min_mm: profile.build_min_mm,
            build_max_mm: profile.build_max_mm,
            max_feed_rate_mm_min: profile.max_feed_rate_mm_min,
            max_nozzle_temp_c: profile.max_nozzle_temp_c,
            max_bed_temp_c: profile.max_bed_temp_c,
            max_retraction_mm: profile.max_retraction_mm,
            supports_homing: true,
        }
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if !canonical_identifier(&self.machine_id, MAX_MACHINE_ID_BYTES) {
            return Err("machine identity must be canonical and bounded");
        }
        let profile = MachineProfile {
            name: "advertised-capabilities".into(),
            build_min_mm: self.build_min_mm,
            build_max_mm: self.build_max_mm,
            max_feed_rate_mm_min: self.max_feed_rate_mm_min,
            max_nozzle_temp_c: self.max_nozzle_temp_c,
            max_bed_temp_c: self.max_bed_temp_c,
            max_retraction_mm: self.max_retraction_mm,
            require_homing: false,
        };
        profile.validate()
    }
}

/// One freshness-scoped capability advertisement from a machine connection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MachineSession {
    pub session_nonce: String,
    pub capabilities: MachineCapabilities,
}

pub const TIMED_MACHINE_SESSION_SCHEMA: &str = "symthaea.fabrication.timed-machine-session.v1";

/// A machine capability advertisement with an explicit freshness window.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TimedMachineSession {
    pub schema_version: String,
    pub session: MachineSession,
    pub session_sequence: u64,
    pub issued_at_unix_s: u64,
    pub expires_at_unix_s: u64,
}

impl TimedMachineSession {
    pub fn new(
        session: MachineSession,
        session_sequence: u64,
        issued_at_unix_s: u64,
        expires_at_unix_s: u64,
    ) -> Self {
        Self {
            schema_version: TIMED_MACHINE_SESSION_SCHEMA.into(),
            session,
            session_sequence,
            issued_at_unix_s,
            expires_at_unix_s,
        }
    }

    pub fn is_fresh_at(&self, unix_s: u64) -> bool {
        unix_s >= self.issued_at_unix_s && unix_s < self.expires_at_unix_s
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MachineSessionWindow {
    pub sequence: u64,
    pub issued_at_unix_s: u64,
    pub expires_at_unix_s: u64,
    pub digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MachineNegotiationViolation {
    InvalidProfile(&'static str),
    InvalidCapabilities(&'static str),
    InvalidSessionNonce,
    UnsupportedSessionSchema,
    SessionSequenceZero,
    InvalidSessionWindow,
    SessionNotYetValid {
        evaluation_time_unix_s: u64,
        issued_at_unix_s: u64,
    },
    SessionExpired {
        evaluation_time_unix_s: u64,
        expires_at_unix_s: u64,
    },
    SessionEncoding(String),
    BuildMinimumUnsupported {
        axis: char,
        requested_mm: f32,
        supported_mm: f32,
    },
    BuildMaximumUnsupported {
        axis: char,
        requested_mm: f32,
        supported_mm: f32,
    },
    FeedRateUnsupported {
        requested: f32,
        supported: f32,
    },
    NozzleTemperatureUnsupported {
        requested: u16,
        supported: u16,
    },
    BedTemperatureUnsupported {
        requested: u16,
        supported: u16,
    },
    RetractionUnsupported {
        requested: f32,
        supported: f32,
    },
    HomingUnsupported,
}

/// Exact negotiated machine authority retained for a single connection session.
#[derive(Debug, Clone)]
pub struct NegotiatedMachine {
    session: MachineSession,
    profile: MachineProfile,
    session_window: Option<MachineSessionWindow>,
}

impl NegotiatedMachine {
    pub fn machine_id(&self) -> &str {
        &self.session.capabilities.machine_id
    }

    pub fn session_nonce(&self) -> &str {
        &self.session.session_nonce
    }

    pub fn profile(&self) -> &MachineProfile {
        &self.profile
    }

    pub fn capabilities(&self) -> &MachineCapabilities {
        &self.session.capabilities
    }

    pub fn session_window(&self) -> Option<MachineSessionWindow> {
        self.session_window
    }

    pub fn is_time_bound(&self) -> bool {
        self.session_window.is_some()
    }

    pub fn is_fresh_at(&self, unix_s: u64) -> bool {
        self.session_window.is_some_and(|window| {
            unix_s >= window.issued_at_unix_s && unix_s < window.expires_at_unix_s
        })
    }
}

/// Negotiate a requested safety profile against live machine capabilities.
pub fn negotiate_machine_profile(
    profile: &MachineProfile,
    session: MachineSession,
) -> Result<NegotiatedMachine, Vec<MachineNegotiationViolation>> {
    let mut violations = Vec::new();
    if let Err(reason) = profile.validate() {
        violations.push(MachineNegotiationViolation::InvalidProfile(reason));
    }
    if let Err(reason) = session.capabilities.validate() {
        violations.push(MachineNegotiationViolation::InvalidCapabilities(reason));
    }
    if !canonical_identifier(&session.session_nonce, MAX_SESSION_NONCE_BYTES) {
        violations.push(MachineNegotiationViolation::InvalidSessionNonce);
    }
    for (axis, name) in [(0, 'X'), (1, 'Y'), (2, 'Z')] {
        if profile.build_min_mm[axis] < session.capabilities.build_min_mm[axis] {
            violations.push(MachineNegotiationViolation::BuildMinimumUnsupported {
                axis: name,
                requested_mm: profile.build_min_mm[axis],
                supported_mm: session.capabilities.build_min_mm[axis],
            });
        }
        if profile.build_max_mm[axis] > session.capabilities.build_max_mm[axis] {
            violations.push(MachineNegotiationViolation::BuildMaximumUnsupported {
                axis: name,
                requested_mm: profile.build_max_mm[axis],
                supported_mm: session.capabilities.build_max_mm[axis],
            });
        }
    }
    if profile.max_feed_rate_mm_min > session.capabilities.max_feed_rate_mm_min {
        violations.push(MachineNegotiationViolation::FeedRateUnsupported {
            requested: profile.max_feed_rate_mm_min,
            supported: session.capabilities.max_feed_rate_mm_min,
        });
    }
    if profile.max_nozzle_temp_c > session.capabilities.max_nozzle_temp_c {
        violations.push(MachineNegotiationViolation::NozzleTemperatureUnsupported {
            requested: profile.max_nozzle_temp_c,
            supported: session.capabilities.max_nozzle_temp_c,
        });
    }
    if profile.max_bed_temp_c > session.capabilities.max_bed_temp_c {
        violations.push(MachineNegotiationViolation::BedTemperatureUnsupported {
            requested: profile.max_bed_temp_c,
            supported: session.capabilities.max_bed_temp_c,
        });
    }
    if profile.max_retraction_mm > session.capabilities.max_retraction_mm {
        violations.push(MachineNegotiationViolation::RetractionUnsupported {
            requested: profile.max_retraction_mm,
            supported: session.capabilities.max_retraction_mm,
        });
    }
    if profile.require_homing && !session.capabilities.supports_homing {
        violations.push(MachineNegotiationViolation::HomingUnsupported);
    }

    if violations.is_empty() {
        Ok(NegotiatedMachine {
            session,
            profile: profile.clone(),
            session_window: None,
        })
    } else {
        Err(violations)
    }
}

/// Negotiate against a capability advertisement with explicit issue and expiry times.
pub fn negotiate_machine_profile_at(
    profile: &MachineProfile,
    timed: TimedMachineSession,
    evaluation_time_unix_s: u64,
) -> Result<NegotiatedMachine, Vec<MachineNegotiationViolation>> {
    let mut violations = Vec::new();
    if timed.schema_version != TIMED_MACHINE_SESSION_SCHEMA {
        violations.push(MachineNegotiationViolation::UnsupportedSessionSchema);
    }
    if timed.session_sequence == 0 {
        violations.push(MachineNegotiationViolation::SessionSequenceZero);
    }
    if timed.issued_at_unix_s >= timed.expires_at_unix_s {
        violations.push(MachineNegotiationViolation::InvalidSessionWindow);
    } else if evaluation_time_unix_s < timed.issued_at_unix_s {
        violations.push(MachineNegotiationViolation::SessionNotYetValid {
            evaluation_time_unix_s,
            issued_at_unix_s: timed.issued_at_unix_s,
        });
    } else if evaluation_time_unix_s >= timed.expires_at_unix_s {
        violations.push(MachineNegotiationViolation::SessionExpired {
            evaluation_time_unix_s,
            expires_at_unix_s: timed.expires_at_unix_s,
        });
    }
    let digest = match digest_timed_machine_session(&timed) {
        Ok(digest) => Some(digest),
        Err(error) => {
            violations.push(MachineNegotiationViolation::SessionEncoding(error));
            None
        }
    };
    let legacy = negotiate_machine_profile(profile, timed.session.clone());
    match legacy {
        Ok(mut machine) if violations.is_empty() => {
            machine.session_window = Some(MachineSessionWindow {
                sequence: timed.session_sequence,
                issued_at_unix_s: timed.issued_at_unix_s,
                expires_at_unix_s: timed.expires_at_unix_s,
                digest: digest.expect("digest present when session validation succeeds"),
            });
            Ok(machine)
        }
        Ok(_) => Err(violations),
        Err(mut legacy_violations) => {
            violations.append(&mut legacy_violations);
            Err(violations)
        }
    }
}

pub fn digest_timed_machine_session(timed: &TimedMachineSession) -> Result<Sha256Digest, String> {
    if timed.schema_version != TIMED_MACHINE_SESSION_SCHEMA {
        return Err("unsupported timed machine session schema".into());
    }
    if timed.session_sequence == 0 {
        return Err("session sequence must be positive".into());
    }
    if timed.issued_at_unix_s >= timed.expires_at_unix_s {
        return Err("session window must be ordered".into());
    }
    if !canonical_identifier(&timed.session.session_nonce, MAX_SESSION_NONCE_BYTES) {
        return Err("session nonce must be canonical and bounded".into());
    }
    timed
        .session
        .capabilities
        .validate()
        .map_err(str::to_string)?;
    let bytes = serde_json::to_vec(timed).map_err(|error| error.to_string())?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.timed-machine-session-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

/// One machine-policy violation tied to a command index when applicable.
#[derive(Debug, Clone, PartialEq)]
pub struct MachineViolation {
    pub command_index: Option<usize>,
    pub reason: MachineViolationReason,
}

/// Failures that prevent machine authority from being granted.
#[derive(Debug, Clone, PartialEq)]
pub enum MachineViolationReason {
    InvalidProfile(&'static str),
    EmptyProgram,
    NoExtrusionMotion,
    MotionBeforeHoming,
    NonFiniteValue(&'static str),
    CoordinateOutOfBounds {
        axis: char,
        value_mm: f32,
        minimum_mm: f32,
        maximum_mm: f32,
    },
    FeedRateExceeded {
        value_mm_min: f32,
        maximum_mm_min: f32,
    },
    NozzleTemperatureExceeded {
        value_c: u16,
        maximum_c: u16,
    },
    BedTemperatureExceeded {
        value_c: u16,
        maximum_c: u16,
    },
    RetractionExceeded {
        distance_mm: f32,
        maximum_mm: f32,
    },
}

/// Complete evidence from validating a program for one machine profile.
#[derive(Debug, Clone, PartialEq)]
pub struct MachineValidationReport {
    pub profile_name: String,
    pub command_count: usize,
    pub violations: Vec<MachineViolation>,
}

impl MachineValidationReport {
    pub fn is_safe_to_submit(&self) -> bool {
        self.violations.is_empty()
    }
}

fn validate_axis(
    violations: &mut Vec<MachineViolation>,
    command_index: usize,
    axis: char,
    value: Option<f32>,
    minimum: f32,
    maximum: f32,
) {
    let Some(value) = value else {
        return;
    };
    if !value.is_finite() {
        violations.push(MachineViolation {
            command_index: Some(command_index),
            reason: MachineViolationReason::NonFiniteValue("coordinate"),
        });
    } else if value < minimum || value > maximum {
        violations.push(MachineViolation {
            command_index: Some(command_index),
            reason: MachineViolationReason::CoordinateOutOfBounds {
                axis,
                value_mm: value,
                minimum_mm: minimum,
                maximum_mm: maximum,
            },
        });
    }
}

fn validate_feed(
    violations: &mut Vec<MachineViolation>,
    command_index: usize,
    feed: Option<f32>,
    maximum: f32,
) {
    let Some(feed) = feed else {
        return;
    };
    if !feed.is_finite() || feed < 0.0 {
        violations.push(MachineViolation {
            command_index: Some(command_index),
            reason: MachineViolationReason::NonFiniteValue("feed rate"),
        });
    } else if feed > maximum {
        violations.push(MachineViolation {
            command_index: Some(command_index),
            reason: MachineViolationReason::FeedRateExceeded {
                value_mm_min: feed,
                maximum_mm_min: maximum,
            },
        });
    }
}

/// Validate G-code commands against one immutable machine profile.
pub fn validate_gcode_for_machine(
    program: &GCodeProgram,
    profile: &MachineProfile,
) -> MachineValidationReport {
    let mut violations = Vec::new();
    if let Err(reason) = profile.validate() {
        violations.push(MachineViolation {
            command_index: None,
            reason: MachineViolationReason::InvalidProfile(reason),
        });
    }
    if program.commands.is_empty() {
        violations.push(MachineViolation {
            command_index: None,
            reason: MachineViolationReason::EmptyProgram,
        });
    }

    let mut homed = !profile.require_homing;
    let mut previous_extrusion: Option<f32> = None;
    let mut has_positive_extrusion = false;

    for (command_index, command) in program.commands.iter().enumerate() {
        match command {
            GCodeCommand::G28 => homed = true,
            GCodeCommand::G0 { x, y, z, f } => {
                if !homed {
                    violations.push(MachineViolation {
                        command_index: Some(command_index),
                        reason: MachineViolationReason::MotionBeforeHoming,
                    });
                }
                validate_axis(
                    &mut violations,
                    command_index,
                    'X',
                    *x,
                    profile.build_min_mm[0],
                    profile.build_max_mm[0],
                );
                validate_axis(
                    &mut violations,
                    command_index,
                    'Y',
                    *y,
                    profile.build_min_mm[1],
                    profile.build_max_mm[1],
                );
                validate_axis(
                    &mut violations,
                    command_index,
                    'Z',
                    *z,
                    profile.build_min_mm[2],
                    profile.build_max_mm[2],
                );
                validate_feed(
                    &mut violations,
                    command_index,
                    *f,
                    profile.max_feed_rate_mm_min,
                );
            }
            GCodeCommand::G1 { x, y, z, e, f } => {
                if !homed {
                    violations.push(MachineViolation {
                        command_index: Some(command_index),
                        reason: MachineViolationReason::MotionBeforeHoming,
                    });
                }
                validate_axis(
                    &mut violations,
                    command_index,
                    'X',
                    *x,
                    profile.build_min_mm[0],
                    profile.build_max_mm[0],
                );
                validate_axis(
                    &mut violations,
                    command_index,
                    'Y',
                    *y,
                    profile.build_min_mm[1],
                    profile.build_max_mm[1],
                );
                validate_axis(
                    &mut violations,
                    command_index,
                    'Z',
                    *z,
                    profile.build_min_mm[2],
                    profile.build_max_mm[2],
                );
                validate_feed(
                    &mut violations,
                    command_index,
                    *f,
                    profile.max_feed_rate_mm_min,
                );
                if let Some(extrusion) = e {
                    if !extrusion.is_finite() {
                        violations.push(MachineViolation {
                            command_index: Some(command_index),
                            reason: MachineViolationReason::NonFiniteValue("extrusion"),
                        });
                    } else if let Some(previous) = previous_extrusion {
                        let delta = extrusion - previous;
                        if delta > 0.0 {
                            has_positive_extrusion = true;
                        }
                        let retraction = -delta;
                        if retraction > profile.max_retraction_mm {
                            violations.push(MachineViolation {
                                command_index: Some(command_index),
                                reason: MachineViolationReason::RetractionExceeded {
                                    distance_mm: retraction,
                                    maximum_mm: profile.max_retraction_mm,
                                },
                            });
                        }
                    } else if *extrusion > 0.0 {
                        has_positive_extrusion = true;
                    }
                    previous_extrusion = Some(*extrusion);
                }
            }
            GCodeCommand::M104 { s } | GCodeCommand::M109 { s } => {
                if *s > profile.max_nozzle_temp_c {
                    violations.push(MachineViolation {
                        command_index: Some(command_index),
                        reason: MachineViolationReason::NozzleTemperatureExceeded {
                            value_c: *s,
                            maximum_c: profile.max_nozzle_temp_c,
                        },
                    });
                }
            }
            GCodeCommand::M140 { s } | GCodeCommand::M190 { s } => {
                if *s > profile.max_bed_temp_c {
                    violations.push(MachineViolation {
                        command_index: Some(command_index),
                        reason: MachineViolationReason::BedTemperatureExceeded {
                            value_c: *s,
                            maximum_c: profile.max_bed_temp_c,
                        },
                    });
                }
            }
            GCodeCommand::Comment(_) => {}
        }
    }

    if !program.commands.is_empty() && !has_positive_extrusion {
        violations.push(MachineViolation {
            command_index: None,
            reason: MachineViolationReason::NoExtrusionMotion,
        });
    }

    MachineValidationReport {
        profile_name: profile.name.clone(),
        command_count: program.commands.len(),
        violations,
    }
}

/// A G-code program that has passed one named machine profile.
#[derive(Debug, Clone)]
pub struct ValidatedGCode {
    program: GCodeProgram,
    profile: MachineProfile,
}

impl ValidatedGCode {
    pub fn try_new(
        program: GCodeProgram,
        profile: &MachineProfile,
    ) -> Result<Self, MachineValidationReport> {
        let report = validate_gcode_for_machine(&program, profile);
        if !report.is_safe_to_submit() {
            return Err(report);
        }
        Ok(Self {
            program,
            profile: profile.clone(),
        })
    }

    pub fn program(&self) -> &GCodeProgram {
        &self.program
    }

    pub fn profile(&self) -> &MachineProfile {
        &self.profile
    }

    pub fn profile_name(&self) -> &str {
        &self.profile.name
    }

    pub fn into_program(self) -> GCodeProgram {
        self.program
    }
}

/// Submit only a program that has already crossed the machine-policy boundary.
pub fn submit_validated_gcode(
    printer: &mut dyn PrinterApi,
    program: &ValidatedGCode,
) -> Result<String, PrinterError> {
    printer.submit_gcode(&program.program.to_gcode_string())
}

fn canonical_identifier(value: &str, maximum_bytes: usize) -> bool {
    !value.trim().is_empty() && value == value.trim() && value.len() <= maximum_bytes
}

#[cfg(test)]
mod timed_session_tests {
    use super::*;

    #[test]
    fn timed_session_is_fresh_only_inside_half_open_window() {
        let profile = MachineProfile::default();
        let timed = TimedMachineSession::new(
            MachineSession {
                session_nonce: "fresh-session".into(),
                capabilities: MachineCapabilities::from_profile("machine-1", &profile),
            },
            7,
            100,
            200,
        );
        assert!(negotiate_machine_profile_at(&profile, timed.clone(), 100).is_ok());
        assert!(negotiate_machine_profile_at(&profile, timed.clone(), 199).is_ok());
        assert!(matches!(
            negotiate_machine_profile_at(&profile, timed, 200),
            Err(violations) if violations.iter().any(|violation| matches!(
                violation,
                MachineNegotiationViolation::SessionExpired { .. }
            ))
        ));
    }

    #[test]
    fn timed_session_digest_changes_with_nonce_and_sequence() {
        let profile = MachineProfile::default();
        let base = TimedMachineSession::new(
            MachineSession {
                session_nonce: "nonce-a".into(),
                capabilities: MachineCapabilities::from_profile("machine-1", &profile),
            },
            1,
            100,
            200,
        );
        let mut changed = base.clone();
        changed.session.session_nonce = "nonce-b".into();
        assert_ne!(
            digest_timed_machine_session(&base).unwrap(),
            digest_timed_machine_session(&changed).unwrap()
        );
        changed = base.clone();
        changed.session_sequence = 2;
        assert_ne!(
            digest_timed_machine_session(&base).unwrap(),
            digest_timed_machine_session(&changed).unwrap()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slicer::{Contour, Point2, SliceConfig, SliceLayer};
    use crate::toolpath::{ToolpathConfig, try_generate_gcode};

    fn valid_program() -> GCodeProgram {
        let layers = vec![SliceLayer {
            z: 0.2,
            outer_contours: vec![Contour {
                points: vec![
                    Point2::new(1.0, 1.0),
                    Point2::new(10.0, 1.0),
                    Point2::new(10.0, 10.0),
                    Point2::new(1.0, 10.0),
                ],
            }],
            inner_contours: vec![],
            infill_lines: vec![],
        }];
        try_generate_gcode(&layers, &SliceConfig::default(), &ToolpathConfig::default()).unwrap()
    }

    #[test]
    fn exact_profile_negotiates_against_matching_capabilities() {
        let profile = MachineProfile::default();
        let session = MachineSession {
            session_nonce: "session-1".into(),
            capabilities: MachineCapabilities::from_profile("printer-1", &profile),
        };
        let negotiated = negotiate_machine_profile(&profile, session).unwrap();
        assert_eq!(negotiated.machine_id(), "printer-1");
        assert_eq!(negotiated.profile(), &profile);
    }

    #[test]
    fn negotiation_rejects_capability_overclaim() {
        let profile = MachineProfile::default();
        let mut capabilities = MachineCapabilities::from_profile("printer-1", &profile);
        capabilities.max_nozzle_temp_c = profile.max_nozzle_temp_c - 1;
        let result = negotiate_machine_profile(
            &profile,
            MachineSession {
                session_nonce: "session-1".into(),
                capabilities,
            },
        );
        assert!(matches!(
            result,
            Err(violations) if violations.iter().any(|violation| matches!(
                violation,
                MachineNegotiationViolation::NozzleTemperatureUnsupported { .. }
            ))
        ));
    }

    #[test]
    fn generated_program_passes_default_profile() {
        let program = valid_program();
        let report = validate_gcode_for_machine(&program, &MachineProfile::default());
        assert!(report.is_safe_to_submit(), "{:#?}", report.violations);
        assert!(ValidatedGCode::try_new(program, &MachineProfile::default()).is_ok());
    }

    #[test]
    fn rejects_coordinate_outside_build_volume() {
        let program = GCodeProgram {
            commands: vec![
                GCodeCommand::G28,
                GCodeCommand::G0 {
                    x: Some(221.0),
                    y: Some(0.0),
                    z: Some(0.0),
                    f: Some(1000.0),
                },
                GCodeCommand::G1 {
                    x: None,
                    y: None,
                    z: None,
                    e: Some(1.0),
                    f: Some(1000.0),
                },
            ],
            total_extrusion_mm: 0.0,
        };
        let report = validate_gcode_for_machine(&program, &MachineProfile::default());
        assert!(report.violations.iter().any(|violation| matches!(
            &violation.reason,
            MachineViolationReason::CoordinateOutOfBounds { axis: 'X', .. }
        )));
    }

    #[test]
    fn rejects_motion_before_homing() {
        let program = GCodeProgram {
            commands: vec![
                GCodeCommand::G0 {
                    x: Some(1.0),
                    y: None,
                    z: None,
                    f: Some(1000.0),
                },
                GCodeCommand::G1 {
                    x: None,
                    y: None,
                    z: None,
                    e: Some(1.0),
                    f: Some(1000.0),
                },
            ],
            total_extrusion_mm: 0.0,
        };
        let report = validate_gcode_for_machine(&program, &MachineProfile::default());
        assert!(
            report.violations.iter().any(|violation| {
                violation.reason == MachineViolationReason::MotionBeforeHoming
            })
        );
    }

    #[test]
    fn rejects_temperature_and_feed_overrides() {
        let program = GCodeProgram {
            commands: vec![
                GCodeCommand::G28,
                GCodeCommand::M109 { s: 350 },
                GCodeCommand::G1 {
                    x: None,
                    y: None,
                    z: None,
                    e: Some(1.0),
                    f: Some(20_000.0),
                },
            ],
            total_extrusion_mm: 0.0,
        };
        let report = validate_gcode_for_machine(&program, &MachineProfile::default());
        assert_eq!(report.violations.len(), 2);
    }

    #[test]
    fn rejects_motionless_temperature_only_program() {
        let program = GCodeProgram {
            commands: vec![
                GCodeCommand::G28,
                GCodeCommand::M190 { s: 60 },
                GCodeCommand::M109 { s: 210 },
            ],
            total_extrusion_mm: 0.0,
        };
        let report = validate_gcode_for_machine(&program, &MachineProfile::default());
        assert!(
            report
                .violations
                .iter()
                .any(|violation| { violation.reason == MachineViolationReason::NoExtrusionMotion })
        );
    }

    #[test]
    fn rejects_excessive_retraction() {
        let program = GCodeProgram {
            commands: vec![
                GCodeCommand::G28,
                GCodeCommand::G1 {
                    x: None,
                    y: None,
                    z: None,
                    e: Some(10.0),
                    f: Some(1000.0),
                },
                GCodeCommand::G1 {
                    x: None,
                    y: None,
                    z: None,
                    e: Some(0.0),
                    f: Some(1000.0),
                },
            ],
            total_extrusion_mm: 10.0,
        };
        let report = validate_gcode_for_machine(&program, &MachineProfile::default());
        assert!(report.violations.iter().any(|violation| matches!(
            &violation.reason,
            MachineViolationReason::RetractionExceeded { .. }
        )));
    }

    #[test]
    fn validated_program_retains_exact_machine_profile() {
        let program = valid_program();
        let mut profile = MachineProfile::default();
        profile.max_feed_rate_mm_min = 12_345.0;
        let validated = ValidatedGCode::try_new(program, &profile).unwrap();
        assert_eq!(validated.profile(), &profile);
        assert_eq!(validated.profile_name(), profile.name.as_str());
    }
}
