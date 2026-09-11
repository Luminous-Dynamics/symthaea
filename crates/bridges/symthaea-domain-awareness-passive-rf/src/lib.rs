// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Receive-only passive RF evidence for domain awareness.
//!
//! This bridge deliberately contains no transmit, jamming, spoofing, protocol
//! manipulation, or electronic-attack API. It publishes measured RF evidence
//! and scan coverage only. RF silence is not identity, intent, or safety.

#![deny(unsafe_code)]

use symthaea_domain_awareness::{
    Domain, EvidenceLineage, IntegrityStatus, Measurement, MeasurementUncertainty, Modality,
    ObservationEnvelope, SensorHealth, TimeEvidence,
};
use uuid::Uuid;

/// Raw receive-only spectrum sample compatible with the existing SpectrumManager
/// measurement fields (`frequency_hz`, `noise_floor_dbm`, `snr_db`).
#[derive(Debug, Clone, PartialEq)]
pub struct PassiveSpectrumSample {
    pub sample_id: String,
    pub sequence: u64,
    pub observed_at_ms: u64,
    pub received_at_ms: u64,
    pub center_frequency_hz: u64,
    pub noise_floor_dbm: f64,
    pub snr_db: f64,
    /// Exact receiver/calibration release used to interpret this sample.
    pub calibration_ref: String,
    /// Reference to the richer raw SDR/spectrum record in the evidence store.
    pub raw_evidence_ref: String,
}

impl PassiveSpectrumSample {
    pub fn validate(&self) -> bool {
        !self.sample_id.trim().is_empty()
            && self.center_frequency_hz > 0
            && self.noise_floor_dbm.is_finite()
            && self.snr_db.is_finite()
            && !self.calibration_ref.trim().is_empty()
            && !self.raw_evidence_ref.trim().is_empty()
            && self.observed_at_ms <= self.received_at_ms
    }

    /// Constructor matching the measured fields already emitted by Symthaea's
    /// existing `SpectrumObservation`. The existing interpreted `jammed` flag is
    /// intentionally not accepted here; domain awareness receives measurements,
    /// not that subsystem's downstream interpretation.
    #[allow(clippy::too_many_arguments)]
    pub fn from_spectrum_manager_fields(
        sample_id: impl Into<String>,
        sequence: u64,
        observed_at_ms: u64,
        received_at_ms: u64,
        frequency_hz: u64,
        noise_floor_dbm: f32,
        snr_db: f32,
        calibration_ref: impl Into<String>,
        raw_evidence_ref: impl Into<String>,
    ) -> Self {
        Self {
            sample_id: sample_id.into(),
            sequence,
            observed_at_ms,
            received_at_ms,
            center_frequency_hz: frequency_hz,
            noise_floor_dbm: f64::from(noise_floor_dbm),
            snr_db: f64::from(snr_db),
            calibration_ref: calibration_ref.into(),
            raw_evidence_ref: raw_evidence_ref.into(),
        }
    }
}

/// Authenticated/provenance context for one passive RF receiver pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassiveRfObservationContext {
    pub source_id: String,
    pub physical_receiver_id: String,
    pub processor_id: String,
    pub network_path: String,
    pub clock_domain: String,
    pub clock_source: String,
    pub domain: Domain,
    pub maximum_valid_age_ms: u64,
    pub clock_uncertainty_ms: u64,
}

impl PassiveRfObservationContext {
    pub fn validate(&self) -> bool {
        !self.source_id.trim().is_empty()
            && !self.physical_receiver_id.trim().is_empty()
            && !self.processor_id.trim().is_empty()
            && !self.network_path.trim().is_empty()
            && !self.clock_domain.trim().is_empty()
            && !self.clock_source.trim().is_empty()
            && self.maximum_valid_age_ms > 0
    }

    /// Publish one normalized receive-only observation. Exact center frequency,
    /// noise floor, and calibration remain recoverable through the raw evidence
    /// reference; the normalized scalar is SNR for generic fusion/quality logic.
    pub fn to_observation(
        &self,
        sample: &PassiveSpectrumSample,
        integrity: IntegrityStatus,
        health: SensorHealth,
        producer_confidence: f64,
    ) -> Result<ObservationEnvelope, PassiveRfError> {
        if !self.validate() {
            return Err(PassiveRfError::InvalidContext);
        }
        if !sample.validate() {
            return Err(PassiveRfError::InvalidSample);
        }
        if !producer_confidence.is_finite() || !(0.0..=1.0).contains(&producer_confidence) {
            return Err(PassiveRfError::InvalidConfidence);
        }

        let observation = ObservationEnvelope {
            observation_id: Uuid::new_v4(),
            source_id: self.source_id.clone(),
            sequence: sample.sequence,
            domain: self.domain,
            modality: Modality::RadioFrequency,
            coordinate_frame: "receiver-local-spectrum".to_string(),
            measurement: Measurement::Scalar {
                quantity: "passive-rf-snr".to_string(),
                value: sample.snr_db,
                unit: "dB".to_string(),
            },
            uncertainty: MeasurementUncertainty {
                position_sigma_m: None,
                velocity_sigma_mps: None,
                bearing_sigma_deg: None,
            },
            time: TimeEvidence {
                observed_at_ms: sample.observed_at_ms,
                received_at_ms: sample.received_at_ms,
                clock_source: self.clock_source.clone(),
                clock_uncertainty_ms: self.clock_uncertainty_ms,
                maximum_valid_age_ms: self.maximum_valid_age_ms,
            },
            lineage: EvidenceLineage {
                physical_source_id: self.physical_receiver_id.clone(),
                processor_id: self.processor_id.clone(),
                network_path: self.network_path.clone(),
                clock_domain: self.clock_domain.clone(),
            },
            integrity,
            sensor_health: health,
            confidence: producer_confidence.min(health.trust_cap()),
            evidence_refs: vec![
                sample.raw_evidence_ref.clone(),
                sample.calibration_ref.clone(),
                format!("passive-rf-sample:{}", sample.sample_id),
            ],
        };

        observation
            .validate()
            .then_some(observation)
            .ok_or(PassiveRfError::InvalidObservation)
    }
}

/// Evidence that a receive-only scanner actually covered a declared frequency
/// interval. A zero-emitter scan is still only evidence of *no emission observed
/// by this receiver under these conditions*.
#[derive(Debug, Clone, PartialEq)]
pub struct PassiveRfScanCoverage {
    pub scan_id: String,
    pub observed_at_ms: u64,
    pub start_frequency_hz: u64,
    pub end_frequency_hz: u64,
    pub bins_observed: usize,
    pub samples_with_signal_evidence: usize,
    pub receiver_health: SensorHealth,
    pub calibration_ref: String,
    pub evidence_ref: String,
}

impl PassiveRfScanCoverage {
    pub fn validate(&self) -> bool {
        !self.scan_id.trim().is_empty()
            && self.start_frequency_hz > 0
            && self.end_frequency_hz >= self.start_frequency_hz
            && self.bins_observed > 0
            && self.samples_with_signal_evidence <= self.bins_observed
            && !self.calibration_ref.trim().is_empty()
            && !self.evidence_ref.trim().is_empty()
    }

    pub fn emission_observed(&self) -> bool {
        self.samples_with_signal_evidence > 0
    }

    /// Explicit epistemic boundary: silence cannot establish object identity.
    pub const fn silence_establishes_identity(&self) -> bool {
        false
    }

    /// Explicit epistemic boundary: silence cannot establish benign intent/safety.
    pub const fn silence_establishes_safety(&self) -> bool {
        false
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassiveRfError {
    InvalidContext,
    InvalidSample,
    InvalidConfidence,
    InvalidObservation,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> PassiveRfObservationContext {
        PassiveRfObservationContext {
            source_id: "rtl-sdr-a-spectrum".into(),
            physical_receiver_id: "rtl-sdr-a".into(),
            processor_id: "spectrum-manager-v1".into(),
            network_path: "local-usb".into(),
            clock_domain: "host-monotonic".into(),
            clock_source: "host-monotonic".into(),
            domain: Domain::Air,
            maximum_valid_age_ms: 500,
            clock_uncertainty_ms: 5,
        }
    }

    fn sample() -> PassiveSpectrumSample {
        PassiveSpectrumSample::from_spectrum_manager_fields(
            "scan-42-bin-7",
            42,
            1_000,
            1_010,
            915_000_000,
            -90.0,
            20.0,
            "calibration:rtl-sdr-a:2026-09",
            "rfraw:scan-42-bin-7",
        )
    }

    #[test]
    fn spectrum_sample_becomes_receive_only_rf_evidence() {
        let observation = context()
            .to_observation(
                &sample(),
                IntegrityStatus::Verified,
                SensorHealth::Nominal,
                0.8,
            )
            .unwrap();
        assert_eq!(observation.modality, Modality::RadioFrequency);
        assert_eq!(observation.lineage.physical_source_id, "rtl-sdr-a");
        assert!(matches!(
            observation.measurement,
            Measurement::Scalar { ref quantity, value, ref unit }
                if quantity == "passive-rf-snr" && value == 20.0 && unit == "dB"
        ));
    }

    #[test]
    fn degraded_receiver_can_only_reduce_confidence() {
        let observation = context()
            .to_observation(
                &sample(),
                IntegrityStatus::Verified,
                SensorHealth::Suspect,
                0.95,
            )
            .unwrap();
        assert_eq!(observation.confidence, SensorHealth::Suspect.trust_cap());
    }

    #[test]
    fn receiver_derivatives_share_one_physical_fault_domain() {
        let mut alternate = context();
        alternate.source_id = "rtl-sdr-a-alt-processor".into();
        alternate.processor_id = "alternate-spectrum-model".into();
        let a = context()
            .to_observation(
                &sample(),
                IntegrityStatus::Verified,
                SensorHealth::Nominal,
                0.8,
            )
            .unwrap();
        let b = alternate
            .to_observation(
                &sample(),
                IntegrityStatus::Verified,
                SensorHealth::Nominal,
                0.8,
            )
            .unwrap();
        assert_eq!(a.lineage.physical_source_id, b.lineage.physical_source_id);
    }

    #[test]
    fn silent_scan_never_establishes_identity_safety_or_authority() {
        let coverage = PassiveRfScanCoverage {
            scan_id: "scan-42".into(),
            observed_at_ms: 1_000,
            start_frequency_hz: 24_000_000,
            end_frequency_hz: 1_766_000_000,
            bins_observed: 1_024,
            samples_with_signal_evidence: 0,
            receiver_health: SensorHealth::Nominal,
            calibration_ref: "calibration:rtl-sdr-a:2026-09".into(),
            evidence_ref: "rfscan:42".into(),
        };
        assert!(coverage.validate());
        assert!(!coverage.emission_observed());
        assert!(!coverage.silence_establishes_identity());
        assert!(!coverage.silence_establishes_safety());
        assert!(!coverage.grants_physical_authority());
    }

    #[test]
    fn interpreted_jamming_flag_is_not_part_of_bridge_constructor() {
        // Constructor accepts only the measured SpectrumManager fields. This
        // compile-time API test documents that the existing `jammed` bool is not
        // promoted as a domain-awareness fact.
        let sample = PassiveSpectrumSample::from_spectrum_manager_fields(
            "sample",
            1,
            1,
            1,
            100_000_000,
            -100.0,
            -5.0,
            "calibration:1",
            "rfraw:1",
        );
        assert_eq!(sample.snr_db, -5.0);
        assert!(sample.validate());
    }
}
