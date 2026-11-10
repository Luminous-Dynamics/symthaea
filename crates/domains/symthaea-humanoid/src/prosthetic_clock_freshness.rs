// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Conservative source-capture freshness qualification for prosthetic feedback.
//!
//! This module is a pure reference model. It correlates an explicitly identified
//! source-clock epoch to a trusted receiver monotonic clock, propagates bounded
//! offset/drift/quantization uncertainty, and reports a source-age interval.
//! Receive age and source-capture age are different quantities; this module
//! never treats one as proof of the other.
//!
//! The caller must supply a clock correlation already qualified by the owning
//! provenance/clock-verification path. This module does not authenticate the
//! sensor, verify calibration, establish physical clock quality, persist replay
//! state, render haptics, issue motor commands, or grant actuation authority.
//! Indeterminate assessments must not be admitted as fresh.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Immutable, exact clock-correlation profile for one sensor and reboot epoch.
///
/// All source times are normalized nanoseconds under the referenced profile.
/// The profile must bind the sensor, firmware, acquisition configuration,
/// source-clock profile revision, and epoch. The validity interval is inclusive.
/// drift_bound_ppm is an absolute relative-rate-error bound from the anchor;
/// synchronization_uncertainty_ns bounds the receiver/source relation at the
/// anchor, and timestamp_quantization_bound_ns bounds source timestamp error.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceClockCorrelationV1 {
    /// Exact sensor identity.
    pub sensor_id: String,
    /// Immutable source-clock profile revision reference.
    pub source_clock_profile_ref: String,
    /// Unique source-clock boot/reset epoch reference.
    pub source_clock_epoch_ref: String,
    /// Exact sensor firmware revision reference.
    pub firmware_revision_ref: String,
    /// Exact DAQ/acquisition configuration reference.
    pub acquisition_configuration_ref: String,
    /// Source-clock anchor, normalized to nanoseconds.
    pub source_anchor_ns: u64,
    /// Immutable identity of the receiver monotonic clock profile.
    pub receiver_clock_profile_ref: String,
    /// Unique receiver monotonic clock epoch (for example, boot/session epoch).
    pub receiver_clock_epoch_ref: String,
    /// Same anchor expressed in the trusted receiver monotonic domain.
    pub receiver_anchor_monotonic_ns: u64,
    /// Earliest source timestamp covered by this correlation.
    pub valid_source_from_ns: u64,
    /// Latest source timestamp covered by this correlation.
    pub valid_source_until_ns: u64,
    /// Absolute synchronization uncertainty at the anchor.
    pub synchronization_uncertainty_ns: u64,
    /// Absolute relative clock-rate error bound in parts per million.
    pub drift_bound_ppm: u32,
    /// Maximum absolute source timestamp quantization error.
    pub timestamp_quantization_bound_ns: u64,
}

/// One source observation with receive and evaluation times in the same
/// trusted receiver monotonic domain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceClockObservationV1 {
    /// Exact sensor identity.
    pub sensor_id: String,
    /// Immutable source-clock profile revision reference.
    pub source_clock_profile_ref: String,
    /// Unique source-clock boot/reset epoch reference.
    pub source_clock_epoch_ref: String,
    /// Exact sensor firmware revision reference.
    pub firmware_revision_ref: String,
    /// Exact DAQ/acquisition configuration reference.
    pub acquisition_configuration_ref: String,
    /// Sequence number, monotonic only within the exact source epoch.
    pub sequence: u64,
    /// Source capture timestamp, normalized under the referenced clock profile.
    pub source_timestamp_ns: u64,
    /// Immutable identity of the receiver monotonic clock profile.
    pub receiver_clock_profile_ref: String,
    /// Unique receiver monotonic clock epoch for receive/evaluation times.
    pub receiver_clock_epoch_ref: String,
    /// Trusted receiver monotonic time when this observation arrived.
    pub received_monotonic_ns: u64,
    /// Trusted receiver monotonic time when freshness is evaluated.
    pub evaluated_monotonic_ns: u64,
}

/// Conservative interval for source-capture age at evaluation time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceAgeIntervalNs {
    /// Earliest age consistent with the qualified clock bounds.
    pub earliest_ns: u64,
    /// Latest age consistent with the qualified clock bounds.
    pub latest_ns: u64,
}

/// Reasons why source-capture freshness cannot be established.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceFreshnessIndeterminateReason {
    /// A required identity/profile/epoch reference is absent.
    MissingIdentity,
    /// The observation's sensor identity differs from the correlation.
    SensorIdentityMismatch,
    /// The immutable source-clock profile revision differs.
    ClockProfileMismatch,
    /// The trusted receiver monotonic clock profile differs.
    ReceiverClockProfileMismatch,
    /// The trusted receiver monotonic clock epoch differs.
    ReceiverClockEpochMismatch,
    /// The source-clock epoch differs, including a reboot/reset epoch mismatch.
    ClockEpochMismatch,
    /// The sensor firmware revision differs from the correlated revision.
    FirmwareRevisionMismatch,
    /// The DAQ/acquisition configuration differs from the correlated revision.
    AcquisitionConfigurationMismatch,
    /// The correlation interval is malformed or does not contain its anchor.
    InvalidCorrelationInterval,
    /// The declared drift bound is too large to guarantee a positive clock rate.
    InvalidDriftBound,
    /// The source timestamp is outside the correlation's declared validity.
    ObservationOutsideCorrelationValidity,
    /// Evaluation time precedes trusted receive time.
    EvaluationPrecedesReceive,
    /// The mapped source capture is definitely later than receiver arrival.
    CaptureTimeAfterReceive,
    /// The mapped source-capture interval overlaps receiver arrival.
    CorrelationOverlapsReceive,
    /// A conservative mapping or age calculation exceeded representable bounds.
    ArithmeticOverflow,
    /// The age interval overlaps the deadline; freshness is not provable.
    DeadlineBoundaryUncertain,
}

/// Source-capture freshness assessment.
///
/// Only Fresh establishes that the worst-case source age is within the
/// declared deadline. Stale and Indeterminate must both suppress feedback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceFreshnessAssessment {
    /// The worst-case source-capture age is within the deadline.
    Fresh {
        /// Conservative source-age interval at evaluation.
        age: SourceAgeIntervalNs,
    },
    /// Even the most optimistic source-capture age exceeds the deadline.
    Stale {
        /// Conservative source-age interval at evaluation.
        age: SourceAgeIntervalNs,
    },
    /// The correlation or observation cannot prove either freshness or staleness.
    Indeterminate {
        /// Specific reason the observation is not eligible for freshness admission.
        reason: SourceFreshnessIndeterminateReason,
        /// Available age interval, when enough information was available to derive it.
        age: Option<SourceAgeIntervalNs>,
    },
}

impl SourceFreshnessAssessment {
    /// Returns the source-age interval when clock mapping reached that stage.
    pub const fn age_interval(self) -> Option<SourceAgeIntervalNs> {
        match self {
            Self::Fresh { age } | Self::Stale { age } => Some(age),
            Self::Indeterminate { age, .. } => age,
        }
    }

    /// Returns true only when worst-case source-capture age satisfies the deadline.
    pub const fn is_fresh(self) -> bool {
        matches!(self, Self::Fresh { .. })
    }
}

/// Assess source-capture freshness using a qualified source-to-receiver mapping.
///
/// The mapped capture interval is the receiver anchor plus source-time delta,
/// expanded by uncertainty. Uncertainty is synchronization uncertainty at the
/// anchor, plus absolute drift growth over the source-time distance from the
/// anchor, plus timestamp quantization error. Source age is then computed
/// against evaluated_monotonic_ns, never by subtracting raw source and host
/// clock values.
///
/// received_monotonic_ns is an independent consistency boundary: the entire
/// mapped capture interval must be no later than receiver arrival. Actual
/// transport/buffering delay remains visible in the resulting source age.
///
/// The correlation is an input claim from an external verifier. This function
/// only checks structural identity equality and arithmetic. It does not verify
/// signatures, calibration, clock synchronization hardware, or source truth.
pub fn assess_source_capture_freshness(
    correlation: &SourceClockCorrelationV1,
    observation: &SourceClockObservationV1,
    deadline_ns: u64,
) -> SourceFreshnessAssessment {
    let indeterminate = |reason, age| SourceFreshnessAssessment::Indeterminate { reason, age };

    if correlation.sensor_id.trim().is_empty()
        || correlation.source_clock_profile_ref.trim().is_empty()
        || correlation.source_clock_epoch_ref.trim().is_empty()
        || correlation.firmware_revision_ref.trim().is_empty()
        || correlation.acquisition_configuration_ref.trim().is_empty()
        || correlation.receiver_clock_profile_ref.trim().is_empty()
        || correlation.receiver_clock_epoch_ref.trim().is_empty()
        || observation.sensor_id.trim().is_empty()
        || observation.source_clock_profile_ref.trim().is_empty()
        || observation.source_clock_epoch_ref.trim().is_empty()
        || observation.firmware_revision_ref.trim().is_empty()
        || observation.acquisition_configuration_ref.trim().is_empty()
        || observation.receiver_clock_profile_ref.trim().is_empty()
        || observation.receiver_clock_epoch_ref.trim().is_empty()
    {
        return indeterminate(SourceFreshnessIndeterminateReason::MissingIdentity, None);
    }

    if observation.sensor_id != correlation.sensor_id {
        return indeterminate(SourceFreshnessIndeterminateReason::SensorIdentityMismatch, None);
    }
    if observation.source_clock_profile_ref != correlation.source_clock_profile_ref {
        return indeterminate(SourceFreshnessIndeterminateReason::ClockProfileMismatch, None);
    }
    if observation.receiver_clock_profile_ref != correlation.receiver_clock_profile_ref {
        return indeterminate(
            SourceFreshnessIndeterminateReason::ReceiverClockProfileMismatch,
            None,
        );
    }
    if observation.receiver_clock_epoch_ref != correlation.receiver_clock_epoch_ref {
        return indeterminate(SourceFreshnessIndeterminateReason::ReceiverClockEpochMismatch, None);
    }
    if observation.source_clock_epoch_ref != correlation.source_clock_epoch_ref {
        return indeterminate(SourceFreshnessIndeterminateReason::ClockEpochMismatch, None);
    }
    if observation.firmware_revision_ref != correlation.firmware_revision_ref {
        return indeterminate(
            SourceFreshnessIndeterminateReason::FirmwareRevisionMismatch,
            None,
        );
    }
    if observation.acquisition_configuration_ref != correlation.acquisition_configuration_ref {
        return indeterminate(
            SourceFreshnessIndeterminateReason::AcquisitionConfigurationMismatch,
            None,
        );
    }

    if correlation.valid_source_from_ns > correlation.source_anchor_ns
        || correlation.source_anchor_ns > correlation.valid_source_until_ns
        || correlation.valid_source_from_ns > correlation.valid_source_until_ns
    {
        return indeterminate(
            SourceFreshnessIndeterminateReason::InvalidCorrelationInterval,
            None,
        );
    }
    if correlation.drift_bound_ppm >= 1_000_000 {
        return indeterminate(SourceFreshnessIndeterminateReason::InvalidDriftBound, None);
    }
    if observation.source_timestamp_ns < correlation.valid_source_from_ns
        || observation.source_timestamp_ns > correlation.valid_source_until_ns
    {
        return indeterminate(
            SourceFreshnessIndeterminateReason::ObservationOutsideCorrelationValidity,
            None,
        );
    }
    if observation.evaluated_monotonic_ns < observation.received_monotonic_ns {
        return indeterminate(
            SourceFreshnessIndeterminateReason::EvaluationPrecedesReceive,
            None,
        );
    }

    // Keep clock-domain conversion in signed wide arithmetic. A source event
    // may precede or follow the synchronization anchor within the declared
    // validity interval; drift uncertainty grows with absolute distance.
    let source_delta_ns = i128::from(observation.source_timestamp_ns)
        - i128::from(correlation.source_anchor_ns);
    let distance_from_anchor_ns = source_delta_ns.unsigned_abs();
    let drift_numerator =
        distance_from_anchor_ns * u128::from(correlation.drift_bound_ppm);
    let drift_uncertainty_ns = (drift_numerator + 999_999) / 1_000_000;
    let total_uncertainty_ns = u128::from(correlation.synchronization_uncertainty_ns)
        + drift_uncertainty_ns
        + u128::from(correlation.timestamp_quantization_bound_ns);

    let uncertainty = match i128::try_from(total_uncertainty_ns) {
        Ok(value) => value,
        Err(_) => {
            return indeterminate(SourceFreshnessIndeterminateReason::ArithmeticOverflow, None);
        }
    };
    let mapped_capture_center =
        i128::from(correlation.receiver_anchor_monotonic_ns) + source_delta_ns;
    let mapped_capture_lower = mapped_capture_center - uncertainty;
    let mapped_capture_upper = mapped_capture_center + uncertainty;

    if mapped_capture_lower < 0
        || mapped_capture_upper > i128::from(u64::MAX)
        || mapped_capture_lower > mapped_capture_upper
    {
        return indeterminate(SourceFreshnessIndeterminateReason::ArithmeticOverflow, None);
    }

    let capture_lower_ns = mapped_capture_lower as u64;
    let capture_upper_ns = mapped_capture_upper as u64;
    if capture_lower_ns > observation.received_monotonic_ns {
        return indeterminate(
            SourceFreshnessIndeterminateReason::CaptureTimeAfterReceive,
            None,
        );
    }
    if capture_upper_ns > observation.received_monotonic_ns {
        return indeterminate(
            SourceFreshnessIndeterminateReason::CorrelationOverlapsReceive,
            None,
        );
    }

    let age = SourceAgeIntervalNs {
        earliest_ns: observation.evaluated_monotonic_ns - capture_upper_ns,
        latest_ns: observation.evaluated_monotonic_ns - capture_lower_ns,
    };

    if age.latest_ns <= deadline_ns {
        SourceFreshnessAssessment::Fresh { age }
    } else if age.earliest_ns > deadline_ns {
        SourceFreshnessAssessment::Stale { age }
    } else {
        indeterminate(
            SourceFreshnessIndeterminateReason::DeadlineBoundaryUncertain,
            Some(age),
        )
    }
}

/// Epoch-scoped sequence replay guard.
///
/// The guard deliberately has no implicit reset operation. A new epoch must
/// be installed through an explicit transition after the owning verifier has
/// qualified the reboot/clock-reset evidence. This in-memory reference guard
/// does not authenticate transitions or persist state across process restart;
/// production use must durably preserve active-epoch and sequence state.
#[derive(Debug, Default)]
pub struct SourceEpochReplayGuard {
    active_epochs: HashMap<(String, String), String>,
    seen_epochs: HashSet<(String, String, String)>,
    last_sequences: HashMap<(String, String, String), u64>,
}

/// Errors returned by the epoch-scoped replay guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceEpochReplayError {
    /// Sensor, clock-profile, or epoch identity is missing.
    MissingIdentity,
    /// An initial epoch is already installed for this sensor/profile stream.
    InitialEpochAlreadyInstalled,
    /// A requested transition does not match the currently active epoch.
    UnexpectedPreviousEpoch,
    /// An epoch identity has already been used for this sensor/profile stream.
    EpochIdentityReused,
    /// No initial epoch has been installed for this sensor/profile stream.
    UnknownStream,
    /// The observation belongs to an inactive or unknown epoch.
    InactiveOrUnknownEpoch,
    /// Sequence is not strictly increasing within its exact epoch.
    NonIncreasingSequence,
}

impl SourceEpochReplayGuard {
    /// Install the first externally qualified epoch for a sensor/profile stream.
    ///
    /// The caller is responsible for authenticating the epoch/profile evidence
    /// before calling this method.
    pub fn install_initial_epoch(
        &mut self,
        sensor_id: &str,
        profile_ref: &str,
        epoch_ref: &str,
    ) -> Result<(), SourceEpochReplayError> {
        if sensor_id.trim().is_empty() || profile_ref.trim().is_empty() || epoch_ref.trim().is_empty()
        {
            return Err(SourceEpochReplayError::MissingIdentity);
        }
        let stream = (sensor_id.to_owned(), profile_ref.to_owned());
        if self.active_epochs.contains_key(&stream) {
            return Err(SourceEpochReplayError::InitialEpochAlreadyInstalled);
        }
        let epoch_key = (stream.0.clone(), stream.1.clone(), epoch_ref.to_owned());
        if self.seen_epochs.contains(&epoch_key) {
            return Err(SourceEpochReplayError::EpochIdentityReused);
        }
        self.seen_epochs.insert(epoch_key);
        self.active_epochs.insert(stream, epoch_ref.to_owned());
        Ok(())
    }

    /// Transition from the active epoch to a new, previously unseen epoch.
    ///
    /// Call only after the exact transition has been qualified by the external
    /// clock/provenance owner. Old epochs are never made active again.
    pub fn transition_epoch(
        &mut self,
        sensor_id: &str,
        profile_ref: &str,
        previous_epoch_ref: &str,
        next_epoch_ref: &str,
    ) -> Result<(), SourceEpochReplayError> {
        if sensor_id.trim().is_empty()
            || profile_ref.trim().is_empty()
            || previous_epoch_ref.trim().is_empty()
            || next_epoch_ref.trim().is_empty()
        {
            return Err(SourceEpochReplayError::MissingIdentity);
        }
        let stream = (sensor_id.to_owned(), profile_ref.to_owned());
        let active = self
            .active_epochs
            .get(&stream)
            .ok_or(SourceEpochReplayError::UnknownStream)?;
        if active.as_str() != previous_epoch_ref {
            return Err(SourceEpochReplayError::UnexpectedPreviousEpoch);
        }
        let next_key = (
            stream.0.clone(),
            stream.1.clone(),
            next_epoch_ref.to_owned(),
        );
        if self.seen_epochs.contains(&next_key) {
            return Err(SourceEpochReplayError::EpochIdentityReused);
        }
        self.seen_epochs.insert(next_key);
        self.active_epochs.insert(stream, next_epoch_ref.to_owned());
        Ok(())
    }

    /// Admit a strictly increasing sequence from the currently active epoch.
    ///
    /// Call after external identity/provenance checks. A frame rejected for a
    /// later freshness reason should still consume its sequence so it cannot
    /// be replayed repeatedly.
    pub fn admit_sequence(
        &mut self,
        observation: &SourceClockObservationV1,
    ) -> Result<(), SourceEpochReplayError> {
        if observation.sensor_id.trim().is_empty()
            || observation.source_clock_profile_ref.trim().is_empty()
            || observation.source_clock_epoch_ref.trim().is_empty()
        {
            return Err(SourceEpochReplayError::MissingIdentity);
        }
        let stream = (
            observation.sensor_id.clone(),
            observation.source_clock_profile_ref.clone(),
        );
        let active = self
            .active_epochs
            .get(&stream)
            .ok_or(SourceEpochReplayError::UnknownStream)?;
        if active != &observation.source_clock_epoch_ref {
            return Err(SourceEpochReplayError::InactiveOrUnknownEpoch);
        }

        let epoch_key = (
            stream.0,
            stream.1,
            observation.source_clock_epoch_ref.clone(),
        );
        if self
            .last_sequences
            .get(&epoch_key)
            .is_some_and(|previous| observation.sequence <= *previous)
        {
            return Err(SourceEpochReplayError::NonIncreasingSequence);
        }
        self.last_sequences.insert(epoch_key, observation.sequence);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn correlation() -> SourceClockCorrelationV1 {
        SourceClockCorrelationV1 {
            sensor_id: "sensor-1".into(),
            source_clock_profile_ref: "clock-profile-v1".into(),
            source_clock_epoch_ref: "boot-7".into(),
            firmware_revision_ref: "firmware-3".into(),
            acquisition_configuration_ref: "daq-config-2".into(),
            source_anchor_ns: 100,
            receiver_anchor_monotonic_ns: 1_000,
            receiver_clock_profile_ref: "receiver-monotonic-v1".into(),
            receiver_clock_epoch_ref: "receiver-boot-12".into(),
            valid_source_from_ns: 0,
            valid_source_until_ns: 1_000,
            synchronization_uncertainty_ns: 2,
            drift_bound_ppm: 0,
            timestamp_quantization_bound_ns: 1,
        }
    }

    fn observation() -> SourceClockObservationV1 {
        SourceClockObservationV1 {
            sensor_id: "sensor-1".into(),
            source_clock_profile_ref: "clock-profile-v1".into(),
            source_clock_epoch_ref: "boot-7".into(),
            firmware_revision_ref: "firmware-3".into(),
            acquisition_configuration_ref: "daq-config-2".into(),
            sequence: 1,
            source_timestamp_ns: 110,
            receiver_clock_profile_ref: "receiver-monotonic-v1".into(),
            receiver_clock_epoch_ref: "receiver-boot-12".into(),
            received_monotonic_ns: 1_020,
            evaluated_monotonic_ns: 1_025,
        }
    }

    #[test]
    fn admits_only_when_worst_case_source_age_meets_deadline() {
        let assessment = assess_source_capture_freshness(&correlation(), &observation(), 20);
        assert_eq!(
            assessment,
            SourceFreshnessAssessment::Fresh {
                age: SourceAgeIntervalNs {
                    earliest_ns: 12,
                    latest_ns: 18,
                },
            }
        );
        assert!(assessment.is_fresh());
    }

    #[test]
    fn recently_received_but_old_capture_is_stale() {
        let mut sample = observation();
        sample.received_monotonic_ns = 5_010;
        sample.evaluated_monotonic_ns = 5_020;

        let assessment = assess_source_capture_freshness(&correlation(), &sample, 100);
        assert_eq!(
            assessment,
            SourceFreshnessAssessment::Stale {
                age: SourceAgeIntervalNs {
                    earliest_ns: 4_007,
                    latest_ns: 4_013,
                },
            }
        );
        assert!(!assessment.is_fresh());
    }

    #[test]
    fn uncertainty_straddling_deadline_is_indeterminate() {
        let mut sample = observation();
        sample.evaluated_monotonic_ns = 1_033;

        let assessment = assess_source_capture_freshness(&correlation(), &sample, 23);
        assert_eq!(
            assessment,
            SourceFreshnessAssessment::Indeterminate {
                reason: SourceFreshnessIndeterminateReason::DeadlineBoundaryUncertain,
                age: Some(SourceAgeIntervalNs {
                    earliest_ns: 20,
                    latest_ns: 26,
                }),
            }
        );
        assert!(!assessment.is_fresh());
    }

    #[test]
    fn drift_uncertainty_grows_with_distance_from_anchor() {
        let mut clock = correlation();
        clock.source_anchor_ns = 0;
        clock.receiver_anchor_monotonic_ns = 1_000_000;
        clock.valid_source_until_ns = 2_000_000_000;
        clock.synchronization_uncertainty_ns = 0;
        clock.timestamp_quantization_bound_ns = 0;
        clock.drift_bound_ppm = 1_000;

        let mut sample = observation();
        sample.source_timestamp_ns = 1_000_000_000;
        sample.received_monotonic_ns = 1_003_000_000;
        sample.evaluated_monotonic_ns = 1_004_000_000;

        let assessment = assess_source_capture_freshness(&clock, &sample, 4_000_000);
        assert_eq!(
            assessment.age_interval(),
            Some(SourceAgeIntervalNs {
                earliest_ns: 2_000_000,
                latest_ns: 4_000_000,
            })
        );
        assert!(assessment.is_fresh());
    }

    #[test]
    fn rejects_profile_epoch_firmware_and_configuration_substitution() {
        let mut sample = observation();
        sample.source_clock_profile_ref = "clock-profile-v2".into();
        assert_eq!(
            assess_source_capture_freshness(&correlation(), &sample, 20),
            SourceFreshnessAssessment::Indeterminate {
                reason: SourceFreshnessIndeterminateReason::ClockProfileMismatch,
                age: None,
            }
        );

        let mut sample = observation();
        sample.receiver_clock_profile_ref = "receiver-monotonic-v2".into();
        assert!(matches!(
            assess_source_capture_freshness(&correlation(), &sample, 20),
            SourceFreshnessAssessment::Indeterminate {
                reason: SourceFreshnessIndeterminateReason::ReceiverClockProfileMismatch,
                ..
            }
        ));

        let mut sample = observation();
        sample.receiver_clock_epoch_ref = "receiver-boot-13".into();
        assert!(matches!(
            assess_source_capture_freshness(&correlation(), &sample, 20),
            SourceFreshnessAssessment::Indeterminate {
                reason: SourceFreshnessIndeterminateReason::ReceiverClockEpochMismatch,
                ..
            }
        ));

        let mut sample = observation();
        sample.source_clock_epoch_ref = "boot-8".into();
        assert!(matches!(
            assess_source_capture_freshness(&correlation(), &sample, 20),
            SourceFreshnessAssessment::Indeterminate {
                reason: SourceFreshnessIndeterminateReason::ClockEpochMismatch,
                ..
            }
        ));

        let mut sample = observation();
        sample.firmware_revision_ref = "firmware-4".into();
        assert!(matches!(
            assess_source_capture_freshness(&correlation(), &sample, 20),
            SourceFreshnessAssessment::Indeterminate {
                reason: SourceFreshnessIndeterminateReason::FirmwareRevisionMismatch,
                ..
            }
        ));

        let mut sample = observation();
        sample.acquisition_configuration_ref = "daq-config-3".into();
        assert!(matches!(
            assess_source_capture_freshness(&correlation(), &sample, 20),
            SourceFreshnessAssessment::Indeterminate {
                reason: SourceFreshnessIndeterminateReason::AcquisitionConfigurationMismatch,
                ..
            }
        ));
    }

    #[test]
    fn rejects_sensor_substitution_and_missing_identity() {
        let mut sample = observation();
        sample.sensor_id = "sensor-2".into();
        assert!(matches!(
            assess_source_capture_freshness(&correlation(), &sample, 20),
            SourceFreshnessAssessment::Indeterminate {
                reason: SourceFreshnessIndeterminateReason::SensorIdentityMismatch,
                ..
            }
        ));

        let mut sample = observation();
        sample.source_clock_epoch_ref = " ".into();
        assert!(matches!(
            assess_source_capture_freshness(&correlation(), &sample, 20),
            SourceFreshnessAssessment::Indeterminate {
                reason: SourceFreshnessIndeterminateReason::MissingIdentity,
                ..
            }
        ));
    }

    #[test]
    fn rejects_malformed_or_expired_correlation_intervals() {
        let mut clock = correlation();
        clock.valid_source_from_ns = 101;
        assert!(matches!(
            assess_source_capture_freshness(&clock, &observation(), 20),
            SourceFreshnessAssessment::Indeterminate {
                reason: SourceFreshnessIndeterminateReason::InvalidCorrelationInterval,
                ..
            }
        ));

        let mut clock = correlation();
        clock.valid_source_until_ns = 109;
        assert!(matches!(
            assess_source_capture_freshness(&clock, &observation(), 20),
            SourceFreshnessAssessment::Indeterminate {
                reason: SourceFreshnessIndeterminateReason::ObservationOutsideCorrelationValidity,
                ..
            }
        ));
    }

    #[test]
    fn rejects_unbounded_drift_and_evaluation_before_receive() {
        let mut clock = correlation();
        clock.drift_bound_ppm = 1_000_000;
        assert!(matches!(
            assess_source_capture_freshness(&clock, &observation(), 20),
            SourceFreshnessAssessment::Indeterminate {
                reason: SourceFreshnessIndeterminateReason::InvalidDriftBound,
                ..
            }
        ));

        let mut sample = observation();
        sample.evaluated_monotonic_ns = 1_019;
        assert!(matches!(
            assess_source_capture_freshness(&correlation(), &sample, 20),
            SourceFreshnessAssessment::Indeterminate {
                reason: SourceFreshnessIndeterminateReason::EvaluationPrecedesReceive,
                ..
            }
        ));
    }

    #[test]
    fn refuses_capture_mapping_after_or_overlapping_receiver_arrival() {
        let mut sample = observation();
        sample.received_monotonic_ns = 1_005;
        assert!(matches!(
            assess_source_capture_freshness(&correlation(), &sample, 20),
            SourceFreshnessAssessment::Indeterminate {
                reason: SourceFreshnessIndeterminateReason::CaptureTimeAfterReceive,
                ..
            }
        ));

        let mut sample = observation();
        sample.received_monotonic_ns = 1_010;
        assert!(matches!(
            assess_source_capture_freshness(&correlation(), &sample, 20),
            SourceFreshnessAssessment::Indeterminate {
                reason: SourceFreshnessIndeterminateReason::CorrelationOverlapsReceive,
                ..
            }
        ));
    }

    #[test]
    fn replay_guard_accepts_new_epoch_sequence_and_rejects_replays() {
        let mut guard = SourceEpochReplayGuard::default();
        guard
            .install_initial_epoch("sensor-1", "clock-profile-v1", "boot-7")
            .unwrap();

        let first = observation();
        guard.admit_sequence(&first).unwrap();
        assert_eq!(
            guard.admit_sequence(&first),
            Err(SourceEpochReplayError::NonIncreasingSequence)
        );

        let mut next = observation();
        next.sequence = 2;
        guard.admit_sequence(&next).unwrap();

        guard
            .transition_epoch("sensor-1", "clock-profile-v1", "boot-7", "boot-8")
            .unwrap();
        let mut rebooted = observation();
        rebooted.source_clock_epoch_ref = "boot-8".into();
        rebooted.sequence = 1;
        guard.admit_sequence(&rebooted).unwrap();

        let mut old_epoch = observation();
        old_epoch.sequence = 3;
        assert_eq!(
            guard.admit_sequence(&old_epoch),
            Err(SourceEpochReplayError::InactiveOrUnknownEpoch)
        );
    }

    #[test]
    fn replay_guard_never_reactivates_or_reuses_an_old_epoch() {
        let mut guard = SourceEpochReplayGuard::default();
        guard
            .install_initial_epoch("sensor-1", "clock-profile-v1", "boot-7")
            .unwrap();
        guard
            .transition_epoch("sensor-1", "clock-profile-v1", "boot-7", "boot-8")
            .unwrap();

        assert_eq!(
            guard.transition_epoch("sensor-1", "clock-profile-v1", "boot-8", "boot-7"),
            Err(SourceEpochReplayError::EpochIdentityReused)
        );
        assert_eq!(
            guard.install_initial_epoch("sensor-1", "clock-profile-v1", "boot-9"),
            Err(SourceEpochReplayError::InitialEpochAlreadyInstalled)
        );
        assert_eq!(
            guard.transition_epoch("sensor-1", "clock-profile-v1", "boot-7", "boot-9"),
            Err(SourceEpochReplayError::UnexpectedPreviousEpoch)
        );
    }

    #[test]
    fn replay_guard_requires_an_explicit_initial_epoch() {
        let mut guard = SourceEpochReplayGuard::default();
        assert_eq!(
            guard.admit_sequence(&observation()),
            Err(SourceEpochReplayError::UnknownStream)
        );
        assert_eq!(
            guard.install_initial_epoch("sensor-1", " ", "boot-7"),
            Err(SourceEpochReplayError::MissingIdentity)
        );
    }
}
