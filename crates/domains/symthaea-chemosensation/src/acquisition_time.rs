// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-bound acquisition-time normalization for chemical percepts.
//!
//! This boundary deliberately does not accept a raw offset, drift scalar, target
//! timestamp, or caller-created comparison receipt. A physical acquisition path
//! supplies the source timestamp evidence already attached to its sensor clock
//! plus a self-verifying [`BoundedHoldoverTransform`] derived by the generic time
//! stack. The generic normalization contract then produces the only comparison
//! timestamp/receipt accepted here.
//!
//! The exact calibration decision + holdover authority is content-addressed and
//! attached beside the normalized comparison time. This preserves the distinction
//! between "this transform is valid" and "this exact evidence chain authorized
//! its use for acquisition" without rewriting the chemical observation itself.
//!
//! The raw acquisition timestamp, legacy clock metadata, calibration provenance,
//! and observation content address remain the evidence of what the sensor emitted.

use std::fmt;

use symthaea_evidence_plane::ContentAddressError;
use symthaea_time_holdover::{BoundedHoldoverTransform, HoldoverError};
use symthaea_time_integrity::TimeIntegrityReceipt;
use symthaea_time_normalization::{ClockTransformError, normalize_timestamp_us};

use crate::{
    ChemicalAcquisitionTimeAuthorizationError, ChemicalAcquisitionTimeAuthorizationId,
    ChemicalPercept, ChemicalTimeAlignmentError, TimedChemicalPercept,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChemicalAcquisitionTimeError {
    /// The calibration + holdover capability no longer verifies against its own
    /// stored evidence and derivation.
    Holdover(HoldoverError),
    /// The verified authority digest could not be wrapped in the generic content
    /// address contract.
    ContentAddress(ContentAddressError),
    /// The raw timestamp/receipt cannot be normalized under the verified finite
    /// transform window.
    Normalization(ClockTransformError),
    /// The normalized result is inconsistent with the immutable chemical source
    /// evidence (for example, legacy source clock metadata contradicts it).
    Alignment(ChemicalTimeAlignmentError),
}

impl fmt::Display for ChemicalAcquisitionTimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Holdover(error) => write!(f, "chemical acquisition holdover evidence invalid: {error}"),
            Self::ContentAddress(error) => write!(
                f,
                "chemical acquisition authority could not be content-addressed: {error}"
            ),
            Self::Normalization(error) => {
                write!(f, "chemical acquisition timestamp cannot be normalized: {error}")
            }
            Self::Alignment(error) => {
                write!(f, "normalized chemical time contradicts source evidence: {error}")
            }
        }
    }
}

impl std::error::Error for ChemicalAcquisitionTimeError {}

impl From<HoldoverError> for ChemicalAcquisitionTimeError {
    fn from(value: HoldoverError) -> Self {
        Self::Holdover(value)
    }
}

impl From<ContentAddressError> for ChemicalAcquisitionTimeError {
    fn from(value: ContentAddressError) -> Self {
        Self::ContentAddress(value)
    }
}

impl From<ClockTransformError> for ChemicalAcquisitionTimeError {
    fn from(value: ClockTransformError) -> Self {
        Self::Normalization(value)
    }
}

impl From<ChemicalTimeAlignmentError> for ChemicalAcquisitionTimeError {
    fn from(value: ChemicalTimeAlignmentError) -> Self {
        Self::Alignment(value)
    }
}

/// Normalize one chemical acquisition timestamp through a self-verifying,
/// evidence-bound finite holdover transform.
///
/// This function intentionally re-runs [`BoundedHoldoverTransform::verify_self`]
/// at the consumer boundary. It then content-addresses the exact calibration
/// decision bundle + holdover claim + derived transform, delegates all domain,
/// epoch, continuity, validity-window, and uncertainty propagation checks to
/// [`normalize_timestamp_us`], and attaches both normalized provenance and the
/// authority address to the resulting timed percept.
///
/// No timing metadata is copied into the raw [`crate::ChemicalObservation`].
pub fn bind_evidence_bound_acquisition_time(
    percept: ChemicalPercept,
    source_time: TimeIntegrityReceipt,
    holdover: &BoundedHoldoverTransform,
) -> Result<TimedChemicalPercept, ChemicalAcquisitionTimeError> {
    // Preserve the existing consumer-facing Holdover error boundary explicitly.
    holdover.verify_self()?;
    let authorization = ChemicalAcquisitionTimeAuthorizationId::from_holdover(holdover)
        .map_err(|error| match error {
            ChemicalAcquisitionTimeAuthorizationError::Holdover(inner) => {
                ChemicalAcquisitionTimeError::Holdover(inner)
            }
        })?;
    let authorization = authorization.content_address()?;

    let normalized = normalize_timestamp_us(
        percept.timestamp_us(),
        &source_time,
        holdover.transform(),
    )?;
    Ok(TimedChemicalPercept::from_evidence_bound_normalized(
        percept,
        normalized,
        authorization,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ChemicalClockDomainId, ChemicalEncodingSpaceId, ChemicalFingerprint, ChemicalModality,
        ChemicalObservation,
    };
    use symthaea_core::hdc::{HDC_DIMENSION, unified_hv::ContinuousHV};
    use symthaea_time_calibration::{
        CalibrationConsensus, ClockCalibrationEvidence, FourTimestampExchange, TimestampEvidence,
    };
    use symthaea_time_calibration_bundle::CalibrationDecisionBundle;
    use symthaea_time_calibration_policy::{CalibrationDecisionPolicy, CalibrationPolicyId};
    use symthaea_time_holdover::HoldoverClaim;
    use symthaea_time_integrity::{
        ClockDomainId, ClockEpochId, ContinuityStatus, TimeUncertainty,
    };

    fn source_domain() -> ClockDomainId {
        ClockDomainId::new("sensor-a/monotonic").unwrap()
    }

    fn source_epoch() -> ClockEpochId {
        ClockEpochId::new("sensor-a-boot-7").unwrap()
    }

    fn target_domain() -> ClockDomainId {
        ClockDomainId::new("capture-host/monotonic").unwrap()
    }

    fn target_epoch() -> ClockEpochId {
        ClockEpochId::new("capture-host-boot-3").unwrap()
    }

    fn receipt(domain: ClockDomainId, epoch: ClockEpochId, error_us: u64) -> TimeIntegrityReceipt {
        TimeIntegrityReceipt::declared(domain)
            .with_epoch(epoch)
            .with_continuity(ContinuityStatus::Continuous)
            .with_uncertainty(TimeUncertainty::bounded(error_us))
    }

    fn stamp(timestamp_us: u64, domain: ClockDomainId, epoch: ClockEpochId) -> TimestampEvidence {
        TimestampEvidence::new(timestamp_us, receipt(domain, epoch, 0))
    }

    fn calibration_evidence() -> ClockCalibrationEvidence {
        // Source->target offset is 500 us, one-way delays are 10 us, and target
        // processing time is 10 us. Source envelope is [1000, 1030].
        let exchange = FourTimestampExchange::new(
            stamp(1_000, source_domain(), source_epoch()),
            stamp(1_510, target_domain(), target_epoch()),
            stamp(1_520, target_domain(), target_epoch()),
            stamp(1_030, source_domain(), source_epoch()),
        )
        .unwrap();
        ClockCalibrationEvidence::derive(exchange).unwrap()
    }

    fn holdover() -> BoundedHoldoverTransform {
        let evidence = vec![calibration_evidence()];
        let consensus = CalibrationConsensus::from_evidence(&evidence).unwrap();
        let policy = CalibrationDecisionPolicy::new(
            CalibrationPolicyId::new("chemical-acquisition-v1").unwrap(),
            1,
            20,
            Some(100),
        )
        .unwrap();
        let decision = policy.evaluate(&consensus);
        let bundle = CalibrationDecisionBundle::new(decision, evidence).unwrap();
        let claim = HoldoverClaim::new(
            900,
            1_200,
            1_000,
            2,
            ContinuityStatus::Continuous,
            ContinuityStatus::Continuous,
            ContinuityStatus::Continuous,
        )
        .unwrap();
        BoundedHoldoverTransform::new(bundle, claim).unwrap()
    }

    fn percept(timestamp_us: u64, legacy_clock: &str) -> ChemicalPercept {
        let mut evidence = ChemicalObservation::new(
            timestamp_us,
            ChemicalModality::Olfactory,
            "nose-a",
            vec![],
        );
        evidence.clock_domain = Some(ChemicalClockDomainId::new(legacy_clock).unwrap());
        ChemicalPercept {
            evidence,
            fingerprint: ChemicalFingerprint {
                vector: ContinuousHV::random(HDC_DIMENSION, 7),
                confidence: 0.9,
                used_channels: 1,
                ignored_channels: 0,
                encoding_space_id: ChemicalEncodingSpaceId::from_bytes([9; 32]),
            },
        }
    }

    fn acquisition_receipt() -> TimeIntegrityReceipt {
        receipt(source_domain(), source_epoch(), 2)
    }

    #[test]
    fn verified_holdover_produces_normalized_time_and_exact_authority_without_rewriting_raw_evidence() {
        let percept = percept(1_000, "sensor-a/monotonic");
        let observation_id = percept.observation_id();
        let holdover = holdover();
        let expected_authority = ChemicalAcquisitionTimeAuthorizationId::from_holdover(&holdover)
            .unwrap()
            .content_address()
            .unwrap();
        let timed = bind_evidence_bound_acquisition_time(
            percept,
            acquisition_receipt(),
            &holdover,
        )
        .unwrap();

        assert_eq!(timed.percept().timestamp_us(), 1_000);
        assert_eq!(timed.comparison_timestamp_us(), 1_500);
        assert_eq!(timed.observation_id(), observation_id);
        assert_eq!(timed.time().clock_domain, target_domain());
        assert_eq!(timed.time().clock_epoch.as_ref(), Some(&target_epoch()));
        assert!(timed.time().supports_bounded_comparison());
        assert_eq!(timed.acquisition_authorization(), Some(&expected_authority));
        let normalization = timed.normalization().expect("normalized provenance retained");
        assert_eq!(normalization.source_timestamp_us(), 1_000);
        assert_eq!(normalization.target_timestamp_us(), 1_500);
        assert_eq!(normalization.transform(), holdover.transform());
    }

    #[test]
    fn source_timestamp_outside_authorized_holdover_window_fails_closed() {
        let error = bind_evidence_bound_acquisition_time(
            percept(1_300, "sensor-a/monotonic"),
            acquisition_receipt(),
            &holdover(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ChemicalAcquisitionTimeError::Normalization(
                ClockTransformError::SourceTimestampOutsideValidity { .. }
            )
        ));
    }

    #[test]
    fn unbounded_source_time_cannot_be_upgraded_by_a_valid_holdover_transform() {
        let weak = TimeIntegrityReceipt::declared(source_domain())
            .with_epoch(source_epoch())
            .with_continuity(ContinuityStatus::Continuous);
        let error = bind_evidence_bound_acquisition_time(
            percept(1_000, "sensor-a/monotonic"),
            weak,
            &holdover(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ChemicalAcquisitionTimeError::Normalization(
                ClockTransformError::UnboundedUncertainty { .. }
            )
        ));
    }

    #[test]
    fn legacy_source_clock_contradiction_survives_normalization_and_fails_closed() {
        let error = bind_evidence_bound_acquisition_time(
            percept(1_000, "different-sensor/monotonic"),
            acquisition_receipt(),
            &holdover(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ChemicalAcquisitionTimeError::Alignment(
                ChemicalTimeAlignmentError::LegacyClockDomainMismatch { .. }
            )
        ));
    }
}
