// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Verifier-owned current physical-health witness for humanoid execution.
//!
//! `HumanoidPhysicalHealthFrame` is serializable evidence. This module owns the
//! additional runtime state required before that evidence can contribute positive
//! physical authority: the exact qualified calibration identity, the backend
//! profile/session lineage, and a strictly monotonic accepted sequence frontier.
//!
//! Backend profile/session identifiers are owner-bound local lineage labels; this
//! module does not claim they cryptographically authenticate a transport. A later
//! HAL integration must bind the verifier instance to the actual authenticated
//! backend channel/session.

use crate::execution::HumanoidAuthorityEnvelope;
use crate::morphology::HumanoidMorphology;
use crate::physical_health::{
    HumanoidPhysicalHealthEnvelope, HumanoidPhysicalHealthFrame, PhysicalHealthAuthorityConfig,
    PhysicalHealthError,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HumanoidPhysicalHealthSourceBinding {
    backend_profile_id: String,
    backend_session_id: String,
    qualified_calibration_fingerprint: u64,
}

impl HumanoidPhysicalHealthSourceBinding {
    pub fn new(
        backend_profile_id: impl Into<String>,
        backend_session_id: impl Into<String>,
        qualified_calibration_fingerprint: u64,
    ) -> Result<Self, VerifiedPhysicalHealthError> {
        let backend_profile_id = backend_profile_id.into();
        let backend_session_id = backend_session_id.into();
        if backend_profile_id.trim().is_empty() {
            return Err(VerifiedPhysicalHealthError::InvalidBackendProfile);
        }
        if backend_session_id.trim().is_empty() {
            return Err(VerifiedPhysicalHealthError::InvalidBackendSession);
        }
        if qualified_calibration_fingerprint == 0 {
            return Err(VerifiedPhysicalHealthError::InvalidQualifiedCalibration);
        }
        Ok(Self {
            backend_profile_id,
            backend_session_id,
            qualified_calibration_fingerprint,
        })
    }

    pub fn backend_profile_id(&self) -> &str {
        &self.backend_profile_id
    }

    pub fn backend_session_id(&self) -> &str {
        &self.backend_session_id
    }

    pub const fn qualified_calibration_fingerprint(&self) -> u64 {
        self.qualified_calibration_fingerprint
    }
}

/// Process-local proof that one physical-health frame was accepted by the owner
/// of the current backend/session sequence frontier.
///
/// Intentionally not `Serialize`, `Deserialize`, `Clone`, or `Copy`. Persisted
/// health data must be re-established through a fresh verifier after restart.
#[derive(Debug)]
pub struct VerifiedCurrentHumanoidPhysicalHealth {
    source: HumanoidPhysicalHealthSourceBinding,
    sequence: u64,
    envelope: HumanoidPhysicalHealthEnvelope,
}

impl VerifiedCurrentHumanoidPhysicalHealth {
    pub fn source(&self) -> &HumanoidPhysicalHealthSourceBinding {
        &self.source
    }

    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    pub const fn envelope(&self) -> HumanoidPhysicalHealthEnvelope {
        self.envelope
    }

    pub fn physical_authority(&self) -> f32 {
        self.envelope.physical_authority()
    }

    /// Physical-health evidence may only tighten an existing authority envelope.
    pub fn restrict_authority(
        &self,
        authority: HumanoidAuthorityEnvelope,
    ) -> HumanoidAuthorityEnvelope {
        self.envelope.restrict_authority(authority)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifiedPhysicalHealthError {
    InvalidBackendProfile,
    InvalidBackendSession,
    InvalidQualifiedCalibration,
    CalibrationMismatch,
    ZeroSequence,
    SequenceReplayOrRegression,
    Health(PhysicalHealthError),
}

impl std::fmt::Display for VerifiedPhysicalHealthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::InvalidBackendProfile => "physical-health backend profile identity is empty",
            Self::InvalidBackendSession => "physical-health backend session identity is empty",
            Self::InvalidQualifiedCalibration => {
                "qualified physical-health calibration identity is invalid"
            }
            Self::CalibrationMismatch => {
                "physical-health frame does not match the qualified calibration identity"
            }
            Self::ZeroSequence => "physical-health sequence zero is not admissible",
            Self::SequenceReplayOrRegression => {
                "physical-health sequence replay or regression detected"
            }
            Self::Health(_) => "physical-health frame failed current health evaluation",
        };
        f.write_str(message)
    }
}

impl std::error::Error for VerifiedPhysicalHealthError {}

/// Stateful owner of one backend/session physical-health sequence frontier.
///
/// A new backend session must create a new verifier instance with a new session
/// identity. Successful frames advance the accepted sequence frontier; rejected
/// frames do not consume sequence numbers.
#[derive(Debug)]
pub struct HumanoidPhysicalHealthVerifier {
    expected_morphology: HumanoidMorphology,
    source: HumanoidPhysicalHealthSourceBinding,
    config: PhysicalHealthAuthorityConfig,
    last_accepted_sequence: Option<u64>,
}

impl HumanoidPhysicalHealthVerifier {
    pub fn new(
        expected_morphology: HumanoidMorphology,
        source: HumanoidPhysicalHealthSourceBinding,
        config: PhysicalHealthAuthorityConfig,
    ) -> Self {
        Self {
            expected_morphology,
            source,
            config,
            last_accepted_sequence: None,
        }
    }

    pub fn expected_morphology(&self) -> HumanoidMorphology {
        self.expected_morphology
    }

    pub fn source(&self) -> &HumanoidPhysicalHealthSourceBinding {
        &self.source
    }

    pub const fn last_accepted_sequence(&self) -> Option<u64> {
        self.last_accepted_sequence
    }

    /// Verify one frame against exact calibration identity, monotonic session
    /// sequencing, morphology, freshness, and the existing physical-health
    /// evaluator. The sequence frontier advances only after every check passes.
    pub fn verify(
        &mut self,
        frame: &HumanoidPhysicalHealthFrame,
        now_s: f64,
    ) -> Result<VerifiedCurrentHumanoidPhysicalHealth, VerifiedPhysicalHealthError> {
        if frame.calibration_fingerprint != self.source.qualified_calibration_fingerprint {
            return Err(VerifiedPhysicalHealthError::CalibrationMismatch);
        }
        if frame.sequence == 0 {
            return Err(VerifiedPhysicalHealthError::ZeroSequence);
        }
        if self
            .last_accepted_sequence
            .is_some_and(|previous| frame.sequence <= previous)
        {
            return Err(VerifiedPhysicalHealthError::SequenceReplayOrRegression);
        }

        let envelope = frame
            .evaluate(self.expected_morphology, now_s, self.config)
            .map_err(VerifiedPhysicalHealthError::Health)?;

        self.last_accepted_sequence = Some(frame.sequence);
        Ok(VerifiedCurrentHumanoidPhysicalHealth {
            source: self.source.clone(),
            sequence: frame.sequence,
            envelope,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physical_health::{ActuatorHealthSample, PowerHealthSample};

    fn binding(session: &str) -> HumanoidPhysicalHealthSourceBinding {
        HumanoidPhysicalHealthSourceBinding::new("dmc21-hardware-v1", session, 42).unwrap()
    }

    fn nominal_frame(sequence: u64) -> HumanoidPhysicalHealthFrame {
        let morphology = HumanoidMorphology::Dmc21;
        HumanoidPhysicalHealthFrame {
            morphology,
            sequence,
            sampled_at_s: 1.0,
            received_at_s: 1.0,
            calibration_fingerprint: 42,
            actuators: vec![
                ActuatorHealthSample {
                    enabled: true,
                    feedback_valid: true,
                    current_a: 0.2,
                    current_limit_a: 2.0,
                    temperature_c: Some(40.0),
                    warning_temperature_c: 70.0,
                    shutdown_temperature_c: 90.0,
                };
                morphology.num_actuators()
            ],
            power: PowerHealthSample {
                bus_voltage_v: 48.0,
                minimum_bus_voltage_v: 40.0,
                nominal_bus_voltage_v: 48.0,
                pack_current_a: 5.0,
                pack_current_limit_a: 40.0,
                state_of_charge: 0.8,
            },
            latched_fault: false,
        }
    }

    fn verifier(session: &str) -> HumanoidPhysicalHealthVerifier {
        HumanoidPhysicalHealthVerifier::new(
            HumanoidMorphology::Dmc21,
            binding(session),
            PhysicalHealthAuthorityConfig::default(),
        )
    }

    #[test]
    fn exact_current_frame_produces_process_local_witness() {
        let mut verifier = verifier("boot-17");
        let verified = verifier.verify(&nominal_frame(7), 1.0).unwrap();
        assert_eq!(verified.sequence(), 7);
        assert_eq!(verified.source().backend_profile_id(), "dmc21-hardware-v1");
        assert_eq!(verified.source().backend_session_id(), "boot-17");
        assert_eq!(verified.source().qualified_calibration_fingerprint(), 42);
        assert_eq!(verified.physical_authority(), 1.0);
        assert_eq!(verifier.last_accepted_sequence(), Some(7));
    }

    #[test]
    fn nonzero_but_wrong_calibration_cannot_be_verified() {
        let mut verifier = verifier("boot-17");
        let mut frame = nominal_frame(7);
        frame.calibration_fingerprint = 41;
        assert_eq!(
            verifier.verify(&frame, 1.0).unwrap_err(),
            VerifiedPhysicalHealthError::CalibrationMismatch
        );
        assert_eq!(verifier.last_accepted_sequence(), None);
    }

    #[test]
    fn replay_and_regression_are_rejected_within_one_backend_session() {
        let mut verifier = verifier("boot-17");
        verifier.verify(&nominal_frame(7), 1.0).unwrap();
        assert_eq!(
            verifier.verify(&nominal_frame(7), 1.0).unwrap_err(),
            VerifiedPhysicalHealthError::SequenceReplayOrRegression
        );
        assert_eq!(
            verifier.verify(&nominal_frame(6), 1.0).unwrap_err(),
            VerifiedPhysicalHealthError::SequenceReplayOrRegression
        );
        assert_eq!(verifier.last_accepted_sequence(), Some(7));
    }

    #[test]
    fn rejected_stale_frame_does_not_advance_sequence_frontier() {
        let mut verifier = verifier("boot-17");
        verifier.verify(&nominal_frame(7), 1.0).unwrap();
        assert!(matches!(
            verifier.verify(&nominal_frame(8), 1.2),
            Err(VerifiedPhysicalHealthError::Health(_))
        ));
        assert_eq!(verifier.last_accepted_sequence(), Some(7));
        verifier.verify(&nominal_frame(8), 1.0).unwrap();
        assert_eq!(verifier.last_accepted_sequence(), Some(8));
    }

    #[test]
    fn new_backend_session_has_an_independent_sequence_frontier() {
        let mut first = verifier("boot-17");
        first.verify(&nominal_frame(7), 1.0).unwrap();

        let mut second = verifier("boot-18");
        let verified = second.verify(&nominal_frame(1), 1.0).unwrap();
        assert_eq!(verified.sequence(), 1);
        assert_eq!(verified.source().backend_session_id(), "boot-18");
    }

    #[test]
    fn verified_health_only_restricts_existing_physical_authority() {
        let mut verifier = verifier("boot-17");
        let mut frame = nominal_frame(7);
        frame.actuators[0].temperature_c = Some(80.0);
        let verified = verifier.verify(&frame, 1.0).unwrap();
        let authority = verified.restrict_authority(HumanoidAuthorityEnvelope {
            physical: 0.2,
            ..HumanoidAuthorityEnvelope::fully_admitted()
        });
        assert!((authority.physical - 0.2).abs() < 1.0e-6);
    }
}
