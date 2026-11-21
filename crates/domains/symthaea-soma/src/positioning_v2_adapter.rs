//! SOMA adapter for Mycelix Position's versioned navigation claims.
//!
//! This is a provider-facing mapping boundary, not a second positioning
//! estimator, provenance service, clock service or navigation-health engine.
//! It only packages source observations and the evidence references supplied
//! by the mobile/platform layer. Missing source metadata remains unqualified.

use std::fmt;

use positioning::measurements::{MeasurementModality, MeasurementValue};
use positioning::navigation_v2::{
    CalibrationRef, ConfigurationRef, ContentCommitment, ContractError,
    EvidenceClassRef, FailureDomainRef, NavigationClaimV2, PositionObservationV2,
    ProvenanceRef, Qualification, QualificationReason, SourceEvidenceRefsV2,
    SourceRef, SpatialReferenceRef, TimeEvidenceV2, TimeUnitV2,
};
use serde::Serialize;

#[derive(Debug, Clone, PartialEq)]
pub enum SomaAdapterError {
    Contract(ContractError),
    Serialization(String),
    SequenceExhausted,
}

impl fmt::Display for SomaAdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(error) => write!(f, "Mycelix Position claim contract rejected SOMA input: {error}"),
            Self::Serialization(error) => write!(f, "unable to serialize SOMA observation identity: {error}"),
            Self::SequenceExhausted => write!(f, "SOMA adapter observation sequence exhausted"),
        }
    }
}
impl std::error::Error for SomaAdapterError {}
impl From<ContractError> for SomaAdapterError {
    fn from(value: ContractError) -> Self {
        Self::Contract(value)
    }
}

/// Evidence this adapter receives about a mobile source.
///
/// source_id identifies the adapter-visible source endpoint. It is not a
/// cryptographic device identity or calibration credential. Every remaining
/// field is an opaque reference to evidence owned by Mycelix's existing
/// Position/evidence subsystems.
#[derive(Debug, Clone, PartialEq)]
pub struct SomaSourceEvidence {
    pub source_id: String,
    pub provenance: Qualification<ProvenanceRef>,
    pub calibration: Qualification<CalibrationRef>,
    pub configuration: Qualification<ConfigurationRef>,
    pub failure_domains: Vec<FailureDomainRef>,
    pub raw_payload: Option<ContentCommitment>,
}

impl SomaSourceEvidence {
    fn into_position_refs(self) -> Result<SourceEvidenceRefsV2, SomaAdapterError> {
        let refs = SourceEvidenceRefsV2 {
            source: SourceRef::new(self.source_id)?,
            provenance: self.provenance,
            calibration: self.calibration,
            configuration: self.configuration,
            failure_domains: self.failure_domains,
            raw_payload: self.raw_payload,
        };
        refs.validate()?;
        Ok(refs)
    }

    /// Honest default for a legacy mobile provider that exposes no device
    /// credential, calibration record, configuration commitment or dependency
    /// graph. Empty failure domains means unknown, not independent.
    pub fn unqualified_provider(source_id: impl Into<String>) -> Self {
        Self {
            source_id: source_id.into(),
            provenance: Qualification::Unknown,
            calibration: Qualification::Unqualified {
                reason: QualificationReason::CalibrationUnavailable,
            },
            configuration: Qualification::Unknown,
            failure_domains: Vec::new(),
            raw_payload: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SomaObservationInput {
    pub modality: MeasurementModality,
    pub value: MeasurementValue,
    pub observed_at: TimeEvidenceV2,
    pub received_at: Option<TimeEvidenceV2>,
    pub spatial_reference: Qualification<SpatialReferenceRef>,
    pub source_evidence: SomaSourceEvidence,
    pub evidence_class: Qualification<EvidenceClassRef>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct ObservationIdentityMaterial<'a> {
    sequence: u64,
    modality: MeasurementModality,
    value: &'a MeasurementValue,
    observed_at: &'a TimeEvidenceV2,
    received_at: &'a Option<TimeEvidenceV2>,
    source: &'a SourceEvidenceRefsV2,
    spatial_reference: &'a Qualification<SpatialReferenceRef>,
    evidence_class: &'a Qualification<EvidenceClassRef>,
}

/// Minimal SOMA-to-Mycelix v2 observation queue.
///
/// It never estimates position. It packages provider observations, validates
/// the Mycelix contract, and preserves their evidence state. A downstream
/// estimator remains the owner of PositionEstimateV2.
#[derive(Debug, Default)]
pub struct SomaPositioningV2Adapter {
    next_sequence: u64,
    pending: Vec<NavigationClaimV2>,
}

impl SomaPositioningV2Adapter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one provider observation with source observation time kept
    /// distinct from adapter receipt time.
    pub fn record_observation(
        &mut self,
        input: SomaObservationInput,
    ) -> Result<ContentCommitment, SomaAdapterError> {
        let SomaObservationInput {
            modality,
            value,
            observed_at,
            received_at,
            spatial_reference,
            source_evidence,
            evidence_class,
        } = input;
        let source_refs = source_evidence.into_position_refs()?;
        let identity = ObservationIdentityMaterial {
            sequence: self.next_sequence,
            modality,
            value: &value,
            observed_at: &observed_at,
            received_at: &received_at,
            source: &source_refs,
            spatial_reference: &spatial_reference,
            evidence_class: &evidence_class,
        };
        let encoded = serde_json::to_vec(&identity)
            .map_err(|error| SomaAdapterError::Serialization(error.to_string()))?;
        let observation_id = ContentCommitment::new(
            "blake3",
            blake3::hash(&encoded).to_hex().to_string(),
        )?;

        let observation = PositionObservationV2 {
            observation_id: observation_id.clone(),
            modality,
            value,
            observed_at,
            received_at,
            spatial_reference,
            source_evidence: source_refs,
            evidence_class,
        };
        observation.validate()?;

        let next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(SomaAdapterError::SequenceExhausted)?;
        self.pending
            .push(NavigationClaimV2::Observation(observation));
        self.next_sequence = next_sequence;

        Ok(observation_id)
    }

    /// Convenience for mobile providers that expose no source acquisition
    /// timestamp or qualified coordinate reference.
    ///
    /// Source time is explicitly absent. A supplied local tick is recorded
    /// only as unqualified receipt time with DeviceTicks units; it never
    /// substitutes for source observation time.
    pub fn record_legacy_provider_sample(
        &mut self,
        source_id: impl Into<String>,
        modality: MeasurementModality,
        value: MeasurementValue,
        local_receipt_tick: Option<f64>,
    ) -> Result<ContentCommitment, SomaAdapterError> {
        let observed_at = TimeEvidenceV2 {
            raw_value: None,
            unit: None,
            basis: Qualification::Unqualified {
                reason: QualificationReason::SourceMetadataUnavailable,
            },
            uncertainty_seconds: None,
            normalization: None,
        };
        let received_at = local_receipt_tick.map(|tick| TimeEvidenceV2 {
            raw_value: Some(tick),
            unit: Some(TimeUnitV2::DeviceTicks),
            basis: Qualification::Unqualified {
                reason: QualificationReason::SourceMetadataUnavailable,
            },
            uncertainty_seconds: None,
            normalization: None,
        });

        self.record_observation(SomaObservationInput {
            modality,
            value,
            observed_at,
            received_at,
            spatial_reference: Qualification::Unqualified {
                reason: QualificationReason::ReferenceSystemUnresolved,
            },
            source_evidence: SomaSourceEvidence::unqualified_provider(source_id),
            evidence_class: Qualification::Unknown,
        })
    }

    /// Drain typed claims exactly once. No claim is silently promoted to an
    /// estimate, prediction, healthy navigation state or admitted source.
    pub fn drain_claims(&mut self) -> Vec<NavigationClaimV2> {
        std::mem::take(&mut self.pending)
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    pub fn next_sequence(&self) -> u64 {
        self.next_sequence
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use positioning::navigation_v2::TimeBasisRef;

    #[test]
    fn missing_source_time_is_not_filled_with_receipt_tick() {
        let mut adapter = SomaPositioningV2Adapter::new();
        adapter
            .record_legacy_provider_sample(
                "soma/mobile/gnss",
                MeasurementModality::Gps,
                MeasurementValue::Position {
                    xyz: [1.0, 2.0, 3.0],
                    sigma: [2.0, 2.0, 4.0],
                },
                Some(77.0),
            )
            .unwrap();

        let claims = adapter.drain_claims();
        let NavigationClaimV2::Observation(observation) = &claims[0] else {
            panic!("provider sample must remain an observation");
        };

        assert_eq!(observation.observed_at.raw_value, None);
        assert_eq!(observation.observed_at.unit, None);
        assert_eq!(
            observation.observed_at.basis,
            Qualification::Unqualified {
                reason: QualificationReason::SourceMetadataUnavailable
            }
        );
        assert_eq!(
            observation.received_at.as_ref().unwrap().raw_value,
            Some(77.0)
        );
        assert_eq!(
            observation.received_at.as_ref().unwrap().unit,
            Some(TimeUnitV2::DeviceTicks)
        );
        assert!(!observation.received_at.as_ref().unwrap().basis.is_qualified());
        assert!(!observation.has_qualified_time_and_frame());
    }

    #[test]
    fn source_provenance_calibration_and_frame_are_not_inferred() {
        let mut adapter = SomaPositioningV2Adapter::new();
        adapter
            .record_legacy_provider_sample(
                "soma/mobile/uwb",
                MeasurementModality::UwbToF,
                MeasurementValue::Range {
                    distance_m: 4.5,
                    sigma_m: 0.2,
                },
                None,
            )
            .unwrap();

        let claims = adapter.drain_claims();
        let NavigationClaimV2::Observation(observation) = &claims[0] else {
            panic!("provider sample must remain an observation");
        };

        assert_eq!(observation.source_evidence.provenance, Qualification::Unknown);
        assert_eq!(
            observation.source_evidence.calibration,
            Qualification::Unqualified {
                reason: QualificationReason::CalibrationUnavailable
            }
        );
        assert!(observation.source_evidence.failure_domains.is_empty());
        assert!(matches!(
            &observation.spatial_reference,
            Qualification::Unqualified {
                reason: QualificationReason::ReferenceSystemUnresolved
            }
        ));
    }

    #[test]
    fn invalid_provider_measurement_is_rejected_before_queueing() {
        let mut adapter = SomaPositioningV2Adapter::new();
        let result = adapter.record_legacy_provider_sample(
            "soma/mobile/uwb",
            MeasurementModality::UwbToF,
            MeasurementValue::Range {
                distance_m: 4.5,
                sigma_m: 0.0,
            },
            None,
        );
        assert!(matches!(
            result,
            Err(SomaAdapterError::Contract(
                ContractError::InvalidMeasurementValue
            ))
        ));
        assert_eq!(adapter.pending_count(), 0);
        assert_eq!(adapter.next_sequence(), 0);
    }

    #[test]
    fn identical_payload_events_receive_distinct_content_commitments() {
        let mut adapter = SomaPositioningV2Adapter::new();
        let first = adapter
            .record_legacy_provider_sample(
                "soma/mobile/gnss",
                MeasurementModality::Gps,
                MeasurementValue::Position {
                    xyz: [1.0, 2.0, 3.0],
                    sigma: [2.0, 2.0, 4.0],
                },
                None,
            )
            .unwrap();
        let second = adapter
            .record_legacy_provider_sample(
                "soma/mobile/gnss",
                MeasurementModality::Gps,
                MeasurementValue::Position {
                    xyz: [1.0, 2.0, 3.0],
                    sigma: [2.0, 2.0, 4.0],
                },
                None,
            )
            .unwrap();

        assert_ne!(first, second);
        assert_eq!(adapter.pending_count(), 2);
        assert_eq!(adapter.next_sequence(), 2);
    }


    #[test]
    fn invalid_receipt_tick_is_rejected_without_consuming_sequence() {
        let mut adapter = SomaPositioningV2Adapter::new();
        let result = adapter.record_legacy_provider_sample(
            "soma/mobile/gnss",
            MeasurementModality::Gps,
            MeasurementValue::Position {
                xyz: [1.0, 2.0, 3.0],
                sigma: [1.0, 1.0, 2.0],
            },
            Some(f64::NAN),
        );

        assert!(matches!(
            result,
            Err(SomaAdapterError::Contract(ContractError::InvalidTimeValue))
        ));
        assert_eq!(adapter.pending_count(), 0);
        assert_eq!(adapter.next_sequence(), 0);
    }

    #[test]
    fn qualified_source_time_is_preserved_without_being_rewritten() {
        let mut adapter = SomaPositioningV2Adapter::new();
        let observed_at = TimeEvidenceV2 {
            raw_value: Some(1234.0),
            unit: Some(TimeUnitV2::Microseconds),
            basis: Qualification::Qualified(TimeBasisRef::new("clock/tai").unwrap()),
            uncertainty_seconds: Some(0.000_001),
            normalization: None,
        };
        adapter
            .record_observation(SomaObservationInput {
                modality: MeasurementModality::Gps,
                value: MeasurementValue::Position {
                    xyz: [0.0, 0.0, 0.0],
                    sigma: [1.0, 1.0, 2.0],
                },
                observed_at: observed_at.clone(),
                received_at: None,
                spatial_reference: Qualification::Qualified(
                    SpatialReferenceRef::new("crs/earth-ecef-v1").unwrap(),
                ),
                source_evidence: SomaSourceEvidence::unqualified_provider("soma/mobile/gnss"),
                evidence_class: Qualification::Unknown,
            })
            .unwrap();

        let claims = adapter.drain_claims();
        let NavigationClaimV2::Observation(observation) = &claims[0] else {
            panic!("provider sample must remain an observation");
        };
        assert_eq!(observation.observed_at, observed_at);
        assert!(observation.observed_at.basis.is_qualified());
        assert!(observation.received_at.is_none());
    }
}
