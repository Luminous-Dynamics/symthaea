// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Read-only semantic feedback encoding for prosthetic-hand research.
//!
//! This module converts explicitly qualified sensor observations into
//! user-configurable semantic haptic cue identifiers. It does not read sensors,
//! verify provenance itself, drive a haptic actuator, emit motor commands, or
//! establish clinical/physical truth. The caller must populate provenance and
//! calibration dispositions from the appropriate external verification path.
//!
//! Synthetic, simulated, bench-measured and human-observed inputs remain
//! explicitly labelled in every output frame. A synthetic or simulated cue
//! must never be re-labelled as measured touch by a downstream adapter.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Evidence class inherited from the observation source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SensoryEvidenceClass {
    Synthetic,
    Simulated,
    BenchMeasured,
    HumanObserved,
}

/// Provenance disposition supplied by an external verifier. A receipt
/// reference without Verified disposition is never sufficient for cue encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProvenanceDisposition {
    Verified,
    Missing,
    Invalid,
    Indeterminate,
}

/// Calibration disposition supplied by an external qualification/verifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibrationDisposition {
    VerifiedCurrent,
    Missing,
    Expired,
    Revoked,
    Indeterminate,
}

/// Distinguishes direct measurements from derived estimates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObservationOrigin {
    DirectMeasurement,
    DerivedEstimate,
}

/// Named prosthetic-hand digit; no anatomical inference is performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HandDigit {
    Thumb,
    Index,
    Middle,
    Ring,
    Little,
}

impl HandDigit {
    const fn index(self) -> usize {
        match self {
            Self::Thumb => 0,
            Self::Index => 1,
            Self::Middle => 2,
            Self::Ring => 3,
            Self::Little => 4,
        }
    }
}

/// A contact observation with an explicit measured/derived origin.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContactReading {
    pub digit: HandDigit,
    pub in_contact: bool,
    pub origin: ObservationOrigin,
}

/// Grip-force assessment against a task-specific, externally defined envelope.
///
/// The envelope is a reference supplied by the caller; this module does not
/// decide what force is clinically, mechanically or contextually safe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GripForceAssessment {
    pub force_newtons: f32,
    pub preferred_upper_newtons: f32,
    pub maximum_allowed_newtons: f32,
    pub task_envelope_ref: String,
    pub origin: ObservationOrigin,
}

/// Slip probability from an identified estimator; it is not direct touch data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlipEstimate {
    pub probability: f32,
    pub estimator_profile_ref: String,
}

/// A single provenance-bound tactile observation frame.
///
/// source_timestamp_ns belongs to source_clock_profile_ref. Age checks use
/// received_monotonic_ns against the caller's current monotonic clock; source
/// and host clock values are never compared or silently treated as equivalent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProstheticSensoryObservationV1 {
    pub schema_version: u16,
    pub sequence: u64,
    pub sensor_id: String,
    pub provenance_receipt_ref: Option<String>,
    pub provenance: ProvenanceDisposition,
    pub calibration_ref: Option<String>,
    pub calibration: CalibrationDisposition,
    pub evidence_class: SensoryEvidenceClass,
    pub source_clock_profile_ref: String,
    pub source_timestamp_ns: u64,
    pub received_monotonic_ns: u64,
    pub contacts: Vec<ContactReading>,
    pub grip: Option<GripForceAssessment>,
    pub slip: Option<SlipEstimate>,
}

impl ProstheticSensoryObservationV1 {
    pub const SCHEMA_VERSION: u16 = 1;

    fn validate(&self) -> Result<(), ObservationValidationError> {
        if self.schema_version != Self::SCHEMA_VERSION {
            return Err(ObservationValidationError::UnsupportedSchema {
                actual: self.schema_version,
            });
        }
        if self.sensor_id.trim().is_empty()
            || self.source_clock_profile_ref.trim().is_empty()
        {
            return Err(ObservationValidationError::MissingSourceIdentity);
        }

        let mut seen_digits = HashSet::new();
        for contact in &self.contacts {
            if !seen_digits.insert(contact.digit) {
                return Err(ObservationValidationError::DuplicateDigit {
                    digit: contact.digit,
                });
            }
        }

        if let Some(grip) = &self.grip {
            if !grip.force_newtons.is_finite()
                || !grip.preferred_upper_newtons.is_finite()
                || !grip.maximum_allowed_newtons.is_finite()
                || grip.force_newtons < 0.0
                || grip.preferred_upper_newtons <= 0.0
                || grip.maximum_allowed_newtons < grip.preferred_upper_newtons
                || grip.task_envelope_ref.trim().is_empty()
            {
                return Err(ObservationValidationError::InvalidGripAssessment);
            }
        }

        if let Some(slip) = &self.slip {
            if !slip.probability.is_finite()
                || !(0.0..=1.0).contains(&slip.probability)
                || slip.estimator_profile_ref.trim().is_empty()
            {
                return Err(ObservationValidationError::InvalidSlipEstimate);
            }
        }

        Ok(())
    }
}

/// Semantic grip band relative to the task envelope, not an absolute safety class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GripBand {
    WithinPreferred,
    NearDeclaredLimit,
    AboveDeclaredLimit,
}

/// Semantic haptic event. It names information to communicate, not a waveform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HapticCueSemantic {
    Contact(HandDigit),
    Grip(GripBand),
    SlipRisk,
    DataUnavailable,
}

/// User-configurable mapping from semantic events to opaque platform cue IDs.
///
/// These IDs are not PWM values, motor torques, actuator setpoints or safety
/// commands. A separate output adapter must own waveform rendering, intensity,
/// location, rate limits, device checks and the physical disable path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HapticCodebookV1 {
    pub mapping_id: String,
    pub contact_codes: [u16; 5],
    pub grip_codes: [u16; 3],
    pub slip_risk_code: u16,
    pub data_unavailable_code: u16,
}

impl HapticCodebookV1 {
    pub fn validate(&self) -> Result<(), CodebookValidationError> {
        if self.mapping_id.trim().is_empty() {
            return Err(CodebookValidationError::MissingMappingId);
        }
        let mut unique = HashSet::new();
        for code in self
            .contact_codes
            .iter()
            .chain(self.grip_codes.iter())
            .chain([&self.slip_risk_code, &self.data_unavailable_code])
        {
            if !unique.insert(*code) {
                return Err(CodebookValidationError::DuplicateCueCode { code: *code });
            }
        }
        Ok(())
    }
}

/// Policy contains only semantic encoding and receive-age controls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProstheticFeedbackPolicyV1 {
    pub enabled: bool,
    pub maximum_receive_age_ns: u64,
    pub slip_alert_threshold: f32,
    pub emit_data_unavailable_cue: bool,
}

impl ProstheticFeedbackPolicyV1 {
    pub fn validate(&self) -> Result<(), PolicyValidationError> {
        if self.maximum_receive_age_ns == 0
            || !self.slip_alert_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.slip_alert_threshold)
        {
            return Err(PolicyValidationError::InvalidThresholdOrAge);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CueEvidenceOrigin {
    DirectMeasurement,
    DerivedEstimate,
    AvailabilityGate,
}

impl From<ObservationOrigin> for CueEvidenceOrigin {
    fn from(origin: ObservationOrigin) -> Self {
        match origin {
            ObservationOrigin::DirectMeasurement => Self::DirectMeasurement,
            ObservationOrigin::DerivedEstimate => Self::DerivedEstimate,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HapticCue {
    pub code_id: u16,
    pub semantic: HapticCueSemantic,
    pub origin: CueEvidenceOrigin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeedbackUnavailableReason {
    MissingProvenanceReceipt,
    ProvenanceNotVerified,
    CalibrationNotCurrent,
    StaleReceiveTimestamp,
    FutureReceiveTimestamp,
    ReplayOrOutOfOrder,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeedbackFrameDisposition {
    CuesEncoded,
    NoCue,
    DataUnavailable(FeedbackUnavailableReason),
    Disabled,
}

/// Every output retains the source evidence class and mapping identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProstheticFeedbackFrameV1 {
    /// Immutable user-codebook identity. The codebook owner must assign a new
    /// identity whenever any semantic-to-code assignment changes.
    pub mapping_id: String,
    pub source_sequence: u64,
    pub source_sensor_id: String,
    pub provenance_receipt_ref: Option<String>,
    pub provenance: ProvenanceDisposition,
    pub calibration_ref: Option<String>,
    pub source_clock_profile_ref: String,
    pub source_timestamp_ns: u64,
    pub received_monotonic_ns: u64,
    pub evidence_class: SensoryEvidenceClass,
    pub cues: Vec<HapticCue>,
    pub disposition: FeedbackFrameDisposition,
}

/// Structural errors fail closed rather than being converted to tactile claims.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObservationValidationError {
    UnsupportedSchema { actual: u16 },
    MissingSourceIdentity,
    DuplicateDigit { digit: HandDigit },
    InvalidGripAssessment,
    InvalidSlipEstimate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodebookValidationError {
    MissingMappingId,
    DuplicateCueCode { code: u16 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyValidationError {
    InvalidThresholdOrAge,
}

/// Stateful replay guard and semantic-only feedback encoder.
///
/// This object has no hardware handles, transport, motor command type or
/// actuator callback. The returned cue frame must be consumed by a separately
/// reviewed haptic renderer.
pub struct ProstheticFeedbackEncoder {
    policy: ProstheticFeedbackPolicyV1,
    codebook: HapticCodebookV1,
    last_sequences: HashMap<(String, String), u64>,
}

impl ProstheticFeedbackEncoder {
    pub fn new(
        policy: ProstheticFeedbackPolicyV1,
        codebook: HapticCodebookV1,
    ) -> Result<Self, EncoderConfigurationError> {
        policy
            .validate()
            .map_err(EncoderConfigurationError::Policy)?;
        codebook
            .validate()
            .map_err(EncoderConfigurationError::Codebook)?;
        Ok(Self {
            policy,
            codebook,
            last_sequences: HashMap::new(),
        })
    }

    pub fn reset_replay_guard(&mut self) {
        self.last_sequences.clear();
    }

    pub fn encode(
        &mut self,
        observation: &ProstheticSensoryObservationV1,
        now_monotonic_ns: u64,
    ) -> Result<ProstheticFeedbackFrameV1, ObservationValidationError> {
        observation.validate()?;

        if !self.policy.enabled {
            return Ok(self.frame(
                observation,
                Vec::new(),
                FeedbackFrameDisposition::Disabled,
            ));
        }

        // Untrusted or uncalibrated frames must not advance the replay
        // watermark: otherwise a forged high sequence could suppress a later
        // qualified observation.
        if observation.provenance != ProvenanceDisposition::Verified {
            let reason = if observation.provenance == ProvenanceDisposition::Missing
                || observation
                    .provenance_receipt_ref
                    .as_deref()
                    .is_none_or(|reference| reference.trim().is_empty())
            {
                FeedbackUnavailableReason::MissingProvenanceReceipt
            } else {
                FeedbackUnavailableReason::ProvenanceNotVerified
            };
            return Ok(self.unavailable(observation, reason));
        }

        if observation
            .provenance_receipt_ref
            .as_deref()
            .is_none_or(|reference| reference.trim().is_empty())
        {
            return Ok(self.unavailable(
                observation,
                FeedbackUnavailableReason::MissingProvenanceReceipt,
            ));
        }

        if observation.calibration != CalibrationDisposition::VerifiedCurrent
            || observation
                .calibration_ref
                .as_deref()
                .is_none_or(|reference| reference.trim().is_empty())
        {
            return Ok(self.unavailable(
                observation,
                FeedbackUnavailableReason::CalibrationNotCurrent,
            ));
        }

        // Sequence numbers are scoped to a sensor + source-clock stream.
        // Multiple independent sensors must not suppress each other's frames.
        let stream_key = (
            observation.sensor_id.clone(),
            observation.source_clock_profile_ref.clone(),
        );
        if self
            .last_sequences
            .get(&stream_key)
            .is_some_and(|previous| observation.sequence <= *previous)
        {
            return Ok(self.unavailable(
                observation,
                FeedbackUnavailableReason::ReplayOrOutOfOrder,
            ));
        }

        // Once external provenance/calibration gates admit this sample,
        // consume its sequence even if receive-time freshness fails.
        self.last_sequences
            .insert(stream_key, observation.sequence);

        if now_monotonic_ns < observation.received_monotonic_ns {
            return Ok(self.unavailable(
                observation,
                FeedbackUnavailableReason::FutureReceiveTimestamp,
            ));
        }

        if now_monotonic_ns - observation.received_monotonic_ns
            > self.policy.maximum_receive_age_ns
        {
            return Ok(self.unavailable(
                observation,
                FeedbackUnavailableReason::StaleReceiveTimestamp,
            ));
        }

        let mut cues = Vec::new();

        // Slip has the highest semantic priority in the output ordering. This
        // is not a safety intervention; it is a wearer-information cue only.
        if let Some(slip) = &observation.slip {
            if slip.probability >= self.policy.slip_alert_threshold {
                cues.push(HapticCue {
                    code_id: self.codebook.slip_risk_code,
                    semantic: HapticCueSemantic::SlipRisk,
                    origin: CueEvidenceOrigin::DerivedEstimate,
                });
            }
        }

        if let Some(grip) = &observation.grip {
            let band = if grip.force_newtons <= grip.preferred_upper_newtons {
                GripBand::WithinPreferred
            } else if grip.force_newtons <= grip.maximum_allowed_newtons {
                GripBand::NearDeclaredLimit
            } else {
                GripBand::AboveDeclaredLimit
            };
            let code_index = match band {
                GripBand::WithinPreferred => 0,
                GripBand::NearDeclaredLimit => 1,
                GripBand::AboveDeclaredLimit => 2,
            };
            cues.push(HapticCue {
                code_id: self.codebook.grip_codes[code_index],
                semantic: HapticCueSemantic::Grip(band),
                origin: grip.origin.into(),
            });
        }

        for contact in &observation.contacts {
            if contact.in_contact {
                cues.push(HapticCue {
                    code_id: self.codebook.contact_codes[contact.digit.index()],
                    semantic: HapticCueSemantic::Contact(contact.digit),
                    origin: contact.origin.into(),
                });
            }
        }

        if cues.is_empty() {
            return Ok(self.frame(
                observation,
                Vec::new(),
                FeedbackFrameDisposition::NoCue,
            ));
        }

        Ok(self.frame(
            observation,
            cues,
            FeedbackFrameDisposition::CuesEncoded,
        ))
    }

    fn unavailable(
        &self,
        observation: &ProstheticSensoryObservationV1,
        reason: FeedbackUnavailableReason,
    ) -> ProstheticFeedbackFrameV1 {
        // Replayed/out-of-order frames are deliberately silent. A repeated
        // replay must not repeatedly buzz the wearer; the disposition remains
        // available to logs and benchmark evaluation.
        let cues = if self.policy.emit_data_unavailable_cue
            && reason != FeedbackUnavailableReason::ReplayOrOutOfOrder
        {
            vec![HapticCue {
                code_id: self.codebook.data_unavailable_code,
                semantic: HapticCueSemantic::DataUnavailable,
                origin: CueEvidenceOrigin::AvailabilityGate,
            }]
        } else {
            Vec::new()
        };
        self.frame(
            observation,
            cues,
            FeedbackFrameDisposition::DataUnavailable(reason),
        )
    }

    fn frame(
        &self,
        observation: &ProstheticSensoryObservationV1,
        cues: Vec<HapticCue>,
        disposition: FeedbackFrameDisposition,
    ) -> ProstheticFeedbackFrameV1 {
        ProstheticFeedbackFrameV1 {
            mapping_id: self.codebook.mapping_id.clone(),
            source_sequence: observation.sequence,
            source_sensor_id: observation.sensor_id.clone(),
            provenance_receipt_ref: observation.provenance_receipt_ref.clone(),
            provenance: observation.provenance,
            calibration_ref: observation.calibration_ref.clone(),
            source_clock_profile_ref: observation.source_clock_profile_ref.clone(),
            source_timestamp_ns: observation.source_timestamp_ns,
            received_monotonic_ns: observation.received_monotonic_ns,
            evidence_class: observation.evidence_class,
            cues,
            disposition,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncoderConfigurationError {
    Policy(PolicyValidationError),
    Codebook(CodebookValidationError),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoder() -> ProstheticFeedbackEncoder {
        ProstheticFeedbackEncoder::new(
            ProstheticFeedbackPolicyV1 {
                enabled: true,
                maximum_receive_age_ns: 100,
                slip_alert_threshold: 0.7,
                emit_data_unavailable_cue: true,
            },
            HapticCodebookV1 {
                mapping_id: "user-map-v1".into(),
                contact_codes: [10, 11, 12, 13, 14],
                grip_codes: [20, 21, 22],
                slip_risk_code: 30,
                data_unavailable_code: 99,
            },
        )
        .expect("test encoder configuration is valid")
    }

    fn observation(sequence: u64) -> ProstheticSensoryObservationV1 {
        ProstheticSensoryObservationV1 {
            schema_version: ProstheticSensoryObservationV1::SCHEMA_VERSION,
            sequence,
            sensor_id: "hand-sensor-array-1".into(),
            provenance_receipt_ref: Some("receipt:observation-1".into()),
            provenance: ProvenanceDisposition::Verified,
            calibration_ref: Some("calibration:epoch-4".into()),
            calibration: CalibrationDisposition::VerifiedCurrent,
            evidence_class: SensoryEvidenceClass::BenchMeasured,
            source_clock_profile_ref: "clock-profile:daq-1".into(),
            source_timestamp_ns: 10,
            received_monotonic_ns: 100,
            contacts: Vec::new(),
            grip: None,
            slip: None,
        }
    }

    #[test]
    fn maps_contact_to_user_configured_digit_code() {
        let mut encoder = encoder();
        let mut sample = observation(1);
        sample.contacts.push(ContactReading {
            digit: HandDigit::Thumb,
            in_contact: true,
            origin: ObservationOrigin::DirectMeasurement,
        });

        let frame = encoder.encode(&sample, 150).unwrap();
        assert_eq!(frame.disposition, FeedbackFrameDisposition::CuesEncoded);
        assert_eq!(frame.mapping_id, "user-map-v1");
        assert_eq!(frame.cues[0].code_id, 10);
        assert_eq!(
            frame.cues[0].semantic,
            HapticCueSemantic::Contact(HandDigit::Thumb)
        );
        assert_eq!(frame.evidence_class, SensoryEvidenceClass::BenchMeasured);
    }

    #[test]
    fn output_preserves_sensor_calibration_and_clock_lineage() {
        let mut encoder = encoder();
        let mut sample = observation(42);
        sample.contacts.push(ContactReading {
            digit: HandDigit::Thumb,
            in_contact: true,
            origin: ObservationOrigin::DirectMeasurement,
        });

        let frame = encoder.encode(&sample, 150).unwrap();
        assert_eq!(frame.source_sensor_id, sample.sensor_id);
        assert_eq!(frame.provenance_receipt_ref, sample.provenance_receipt_ref);
        assert_eq!(frame.calibration_ref, sample.calibration_ref);
        assert_eq!(frame.source_clock_profile_ref, sample.source_clock_profile_ref);
        assert_eq!(frame.source_timestamp_ns, sample.source_timestamp_ns);
        assert_eq!(frame.received_monotonic_ns, sample.received_monotonic_ns);
        assert_eq!(frame.source_sequence, sample.sequence);
    }

    #[test]
    fn maps_grip_force_against_task_specific_envelope() {
        let mut encoder = encoder();
        let mut sample = observation(1);
        sample.grip = Some(GripForceAssessment {
            force_newtons: 8.0,
            preferred_upper_newtons: 5.0,
            maximum_allowed_newtons: 10.0,
            task_envelope_ref: "task-envelope:glass-v2".into(),
            origin: ObservationOrigin::DirectMeasurement,
        });

        let frame = encoder.encode(&sample, 150).unwrap();
        assert_eq!(frame.cues[0].code_id, 21);
        assert_eq!(
            frame.cues[0].semantic,
            HapticCueSemantic::Grip(GripBand::NearDeclaredLimit)
        );
    }

    #[test]
    fn slip_alert_retains_derived_estimate_origin() {
        let mut encoder = encoder();
        let mut sample = observation(1);
        sample.slip = Some(SlipEstimate {
            probability: 0.9,
            estimator_profile_ref: "slip-estimator:v3".into(),
        });

        let frame = encoder.encode(&sample, 150).unwrap();
        assert_eq!(frame.cues[0].code_id, 30);
        assert_eq!(frame.cues[0].semantic, HapticCueSemantic::SlipRisk);
        assert_eq!(frame.cues[0].origin, ObservationOrigin::DerivedEstimate);
    }

    #[test]
    fn stale_sample_never_emits_a_touch_or_grip_claim() {
        let mut encoder = encoder();
        let mut sample = observation(1);
        sample.contacts.push(ContactReading {
            digit: HandDigit::Index,
            in_contact: true,
            origin: ObservationOrigin::DirectMeasurement,
        });

        let frame = encoder.encode(&sample, 201).unwrap();
        assert_eq!(
            frame.disposition,
            FeedbackFrameDisposition::DataUnavailable(
                FeedbackUnavailableReason::StaleReceiveTimestamp
            )
        );
        assert_eq!(frame.cues.len(), 1);
        assert_eq!(frame.cues[0].semantic, HapticCueSemantic::DataUnavailable);
    }

    #[test]
    fn unqualified_calibration_never_emits_touch() {
        let mut encoder = encoder();
        let mut sample = observation(1);
        sample.calibration = CalibrationDisposition::Indeterminate;
        sample.contacts.push(ContactReading {
            digit: HandDigit::Middle,
            in_contact: true,
            origin: ObservationOrigin::DirectMeasurement,
        });

        let frame = encoder.encode(&sample, 150).unwrap();
        assert_eq!(
            frame.disposition,
            FeedbackFrameDisposition::DataUnavailable(
                FeedbackUnavailableReason::CalibrationNotCurrent
            )
        );
        assert_eq!(frame.cues[0].semantic, HapticCueSemantic::DataUnavailable);
    }

    #[test]
    fn unprovenanced_high_sequence_does_not_poison_qualified_sequence() {
        let mut encoder = encoder();
        let mut untrusted = observation(u64::MAX);
        untrusted.provenance_receipt_ref = None;
        untrusted.provenance = ProvenanceDisposition::Indeterminate;
        let rejected = encoder.encode(&untrusted, 150).unwrap();
        assert_eq!(
            rejected.disposition,
            FeedbackFrameDisposition::DataUnavailable(
                FeedbackUnavailableReason::MissingProvenanceReceipt
            )
        );

        let mut qualified = observation(1);
        qualified.contacts.push(ContactReading {
            digit: HandDigit::Thumb,
            in_contact: true,
            origin: ObservationOrigin::DirectMeasurement,
        });
        let accepted = encoder.encode(&qualified, 150).unwrap();
        assert_eq!(accepted.disposition, FeedbackFrameDisposition::CuesEncoded);
    }

    #[test]
    fn nonempty_but_unverified_provenance_is_rejected() {
        let mut encoder = encoder();
        let mut sample = observation(1);
        sample.provenance = ProvenanceDisposition::Indeterminate;
        let frame = encoder.encode(&sample, 150).unwrap();
        assert_eq!(
            frame.disposition,
            FeedbackFrameDisposition::DataUnavailable(
                FeedbackUnavailableReason::ProvenanceNotVerified
            )
        );
        assert!(frame
            .cues
            .iter()
            .all(|cue| cue.origin == CueEvidenceOrigin::AvailabilityGate));
    }

    #[test]
    fn replay_sequences_are_scoped_per_sensor_clock_stream() {
        let mut encoder = encoder();
        let mut sensor_a = observation(1);
        sensor_a.contacts.push(ContactReading {
            digit: HandDigit::Thumb,
            in_contact: true,
            origin: ObservationOrigin::DirectMeasurement,
        });
        assert_eq!(
            encoder.encode(&sensor_a, 150).unwrap().disposition,
            FeedbackFrameDisposition::CuesEncoded
        );

        let mut sensor_b = observation(1);
        sensor_b.sensor_id = "hand-sensor-array-2".into();
        sensor_b.contacts.push(ContactReading {
            digit: HandDigit::Index,
            in_contact: true,
            origin: ObservationOrigin::DirectMeasurement,
        });
        assert_eq!(
            encoder.encode(&sensor_b, 150).unwrap().disposition,
            FeedbackFrameDisposition::CuesEncoded
        );
    }

    #[test]
    fn whitespace_provenance_and_calibration_references_are_rejected() {
        let mut encoder = encoder();
        let mut sample = observation(1);
        sample.provenance_receipt_ref = Some("   ".into());
        let missing_receipt = encoder.encode(&sample, 150).unwrap();
        assert_eq!(
            missing_receipt.disposition,
            FeedbackFrameDisposition::DataUnavailable(
                FeedbackUnavailableReason::MissingProvenanceReceipt
            )
        );

        let mut sample = observation(2);
        sample.calibration_ref = Some("  ".into());
        let missing_calibration = encoder.encode(&sample, 150).unwrap();
        assert_eq!(
            missing_calibration.disposition,
            FeedbackFrameDisposition::DataUnavailable(
                FeedbackUnavailableReason::CalibrationNotCurrent
            )
        );
    }

    #[test]
    fn repeated_stale_frame_does_not_repeat_unavailable_cue() {
        let mut encoder = encoder();
        let mut sample = observation(1);
        sample.received_monotonic_ns = 0;
        sample.contacts.push(ContactReading {
            digit: HandDigit::Thumb,
            in_contact: true,
            origin: ObservationOrigin::DirectMeasurement,
        });

        let first = encoder.encode(&sample, 200).unwrap();
        assert_eq!(
            first.disposition,
            FeedbackFrameDisposition::DataUnavailable(
                FeedbackUnavailableReason::StaleReceiveTimestamp
            )
        );
        assert_eq!(first.cues.len(), 1);

        let replay = encoder.encode(&sample, 200).unwrap();
        assert_eq!(
            replay.disposition,
            FeedbackFrameDisposition::DataUnavailable(
                FeedbackUnavailableReason::ReplayOrOutOfOrder
            )
        );
        assert!(replay.cues.is_empty());
    }

    #[test]
    fn missing_provenance_never_emits_touch() {
        let mut encoder = encoder();
        let mut sample = observation(1);
        sample.provenance_receipt_ref = None;
        sample.contacts.push(ContactReading {
            digit: HandDigit::Ring,
            in_contact: true,
            origin: ObservationOrigin::DirectMeasurement,
        });

        let frame = encoder.encode(&sample, 150).unwrap();
        assert_eq!(
            frame.disposition,
            FeedbackFrameDisposition::DataUnavailable(
                FeedbackUnavailableReason::MissingProvenanceReceipt
            )
        );
    }

    #[test]
    fn replayed_or_out_of_order_sequence_is_suppressed() {
        let mut encoder = encoder();
        let mut first = observation(7);
        first.contacts.push(ContactReading {
            digit: HandDigit::Little,
            in_contact: true,
            origin: ObservationOrigin::DirectMeasurement,
        });
        assert_eq!(
            encoder.encode(&first, 150).unwrap().disposition,
            FeedbackFrameDisposition::CuesEncoded
        );

        let mut replay = observation(7);
        replay.contacts.push(ContactReading {
            digit: HandDigit::Little,
            in_contact: true,
            origin: ObservationOrigin::DirectMeasurement,
        });
        let frame = encoder.encode(&replay, 150).unwrap();
        assert_eq!(
            frame.disposition,
            FeedbackFrameDisposition::DataUnavailable(
                FeedbackUnavailableReason::ReplayOrOutOfOrder
            )
        );
    }

    #[test]
    fn disabled_feedback_emits_no_cues() {
        let mut disabled = ProstheticFeedbackEncoder::new(
            ProstheticFeedbackPolicyV1 {
                enabled: false,
                maximum_receive_age_ns: 100,
                slip_alert_threshold: 0.7,
                emit_data_unavailable_cue: true,
            },
            HapticCodebookV1 {
                mapping_id: "disabled-map".into(),
                contact_codes: [1, 2, 3, 4, 5],
                grip_codes: [6, 7, 8],
                slip_risk_code: 9,
                data_unavailable_code: 10,
            },
        )
        .unwrap();
        let mut sample = observation(1);
        sample.contacts.push(ContactReading {
            digit: HandDigit::Thumb,
            in_contact: true,
            origin: ObservationOrigin::DirectMeasurement,
        });
        let frame = disabled.encode(&sample, 150).unwrap();
        assert!(frame.cues.is_empty());
        assert_eq!(frame.disposition, FeedbackFrameDisposition::Disabled);
    }

    #[test]
    fn valid_quiet_sample_is_not_reported_as_missing_data() {
        let mut encoder = encoder();
        let sample = observation(1);

        let frame = encoder.encode(&sample, 150).unwrap();
        assert_eq!(frame.disposition, FeedbackFrameDisposition::NoCue);
        assert!(frame.cues.is_empty());

        let replay = encoder.encode(&sample, 150).unwrap();
        assert_eq!(
            replay.disposition,
            FeedbackFrameDisposition::DataUnavailable(
                FeedbackUnavailableReason::ReplayOrOutOfOrder
            )
        );
    }

    #[test]
    fn future_receive_timestamp_is_rejected() {
        let mut encoder = encoder();
        let mut sample = observation(1);
        sample.received_monotonic_ns = 200;

        let frame = encoder.encode(&sample, 150).unwrap();
        assert_eq!(
            frame.disposition,
            FeedbackFrameDisposition::DataUnavailable(
                FeedbackUnavailableReason::FutureReceiveTimestamp
            )
        );
    }

    #[test]
    fn duplicate_digit_readings_are_rejected() {
        let mut encoder = encoder();
        let mut sample = observation(1);
        sample.contacts = vec![
            ContactReading {
                digit: HandDigit::Index,
                in_contact: true,
                origin: ObservationOrigin::DirectMeasurement,
            },
            ContactReading {
                digit: HandDigit::Index,
                in_contact: false,
                origin: ObservationOrigin::DirectMeasurement,
            },
        ];
        assert_eq!(
            encoder.encode(&sample, 150),
            Err(ObservationValidationError::DuplicateDigit {
                digit: HandDigit::Index
            })
        );
    }

    #[test]
    fn non_finite_grip_measurement_fails_closed() {
        let mut encoder = encoder();
        let mut sample = observation(1);
        sample.grip = Some(GripForceAssessment {
            force_newtons: f32::NAN,
            preferred_upper_newtons: 5.0,
            maximum_allowed_newtons: 10.0,
            task_envelope_ref: "task-envelope:glass-v2".into(),
            origin: ObservationOrigin::DirectMeasurement,
        });
        assert_eq!(
            encoder.encode(&sample, 150),
            Err(ObservationValidationError::InvalidGripAssessment)
        );
    }

    #[test]
    fn duplicate_code_ids_are_rejected() {
        let duplicate = HapticCodebookV1 {
            mapping_id: "bad-map".into(),
            contact_codes: [1, 1, 3, 4, 5],
            grip_codes: [6, 7, 8],
            slip_risk_code: 9,
            data_unavailable_code: 10,
        };
        assert_eq!(
            duplicate.validate(),
            Err(CodebookValidationError::DuplicateCueCode { code: 1 })
        );
    }
}
