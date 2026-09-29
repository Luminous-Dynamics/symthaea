// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Modality-neutral observation envelope for physical and information sensors.
//!
//! The fabric separates what was observed, its context, its producing sensor,
//! disclosure policy, and the immutable asset from which it was derived.
//! Adapters can map this model to OGC API - Connected Systems and STAC without
//! making either external standard a dependency of the cognitive core.

use serde::{Deserialize, Serialize};

/// Sensor modality at the observation boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObservationModality {
    Optical, Infrared, Multispectral, Hyperspectral, Sar, Radar,
    RadioFrequency, Acoustic, Inertial, Gnss, Weather, Maritime,
    Aviation, Document, Web, Human, Other,
}

/// Coarse disclosure class for privacy-preserving sharing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum DisclosureClass {
    Public, Restricted, Private, Secret,
}

/// What an observation is allowed to disclose.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DisclosurePolicy {
    pub class: DisclosureClass,
    pub share_exact_time: bool,
    pub share_exact_location: bool,
    pub share_raw_asset: bool,
    pub share_derived_features: bool,
}

impl DisclosurePolicy {
    pub const fn restricted() -> Self {
        Self {
            class: DisclosureClass::Restricted,
            share_exact_time: false,
            share_exact_location: false,
            share_raw_asset: false,
            share_derived_features: true,
        }
    }

    pub const fn private() -> Self {
        Self {
            class: DisclosureClass::Private,
            share_exact_time: false,
            share_exact_location: false,
            share_raw_asset: false,
            share_derived_features: false,
        }
    }
}

impl Default for DisclosurePolicy {
    fn default() -> Self { Self::restricted() }
}

/// Stable sensor/platform identity without embedding private credential material.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SensorIdentity {
    pub sensor_id: String,
    pub platform_id: Option<String>,
    pub procedure_id: Option<String>,
    pub credential_fingerprint: Option<String>,
}

impl SensorIdentity {
    pub fn new(sensor_id: impl Into<String>) -> Self {
        Self {
            sensor_id: sensor_id.into(),
            platform_id: None,
            procedure_id: None,
            credential_fingerprint: None,
        }
    }
}

/// Reference to immutable bytes from which an observation was derived.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssetRef {
    pub hash_algorithm: String,
    pub content_hash: String,
    pub media_type: Option<String>,
    pub catalog_id: Option<String>,
}

impl AssetRef {
    /// Create a content-addressed reference over exact source bytes.
    pub fn blake3(bytes: &[u8]) -> Self {
        Self {
            hash_algorithm: "blake3".to_string(),
            content_hash: blake3::hash(bytes).to_hex().to_string(),
            media_type: None,
            catalog_id: None,
        }
    }

    pub fn validate(&self) -> Result<(), ObservationValidationError> {
        if self.hash_algorithm != "blake3" {
            return Err(ObservationValidationError::UnsupportedHashAlgorithm(
                self.hash_algorithm.clone(),
            ));
        }
        if self.content_hash.len() != 64
            || !self.content_hash.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err(ObservationValidationError::InvalidContentHash);
        }
        Ok(())
    }
}

/// Sensor event time plus explicit clock uncertainty.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationTime {
    pub observed_at_unix_ns: i128,
    pub time_uncertainty_ns: u64,
}

/// Geographic position plus explicit uncertainty radius.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ObservationLocation {
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    pub uncertainty_m: f64,
}

impl ObservationLocation {
    pub fn validate(&self) -> Result<(), ObservationValidationError> {
        if !self.latitude_deg.is_finite()
            || !self.longitude_deg.is_finite()
            || !self.uncertainty_m.is_finite()
            || self.latitude_deg < -90.0
            || self.latitude_deg > 90.0
            || self.longitude_deg < -180.0
            || self.longitude_deg > 180.0
            || self.uncertainty_m < 0.0
        {
            return Err(ObservationValidationError::InvalidLocation);
        }
        Ok(())
    }
}

/// Measurement quality. Confidence is source confidence, not truth probability.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ObservationQuality {
    pub confidence: f32,
    pub measurement_uncertainty: Option<f64>,
    pub calibrated: bool,
}

impl ObservationQuality {
    pub fn validate(&self) -> Result<(), ObservationValidationError> {
        if !self.confidence.is_finite()
            || !(0.0..=1.0).contains(&self.confidence)
            || self.measurement_uncertainty
                .is_some_and(|v| !v.is_finite() || v < 0.0)
        {
            return Err(ObservationValidationError::InvalidQuality);
        }
        Ok(())
    }
}

/// Provenance linking an observation to its producer and processing lineage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationProvenance {
    pub source: SensorIdentity,
    pub acquired_by: Option<String>,
    pub parent_observation_ids: Vec<String>,
    pub processing_fingerprint: Option<String>,
}

/// Modality-neutral observation envelope.
///
/// This is not a claim, detection, or hypothesis. Derived inferences should
/// reference one or more observations and carry their own inference metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    pub id: String,
    pub modality: ObservationModality,
    pub time: ObservationTime,
    pub location: Option<ObservationLocation>,
    pub quality: ObservationQuality,
    pub provenance: ObservationProvenance,
    pub asset: Option<AssetRef>,
    pub disclosure: DisclosurePolicy,
}

impl Observation {
    /// Validate invariants before entering fusion or an evidence graph.
    pub fn validate(&self) -> Result<(), ObservationValidationError> {
        if self.id.trim().is_empty() {
            return Err(ObservationValidationError::EmptyId);
        }
        if self.provenance.source.sensor_id.trim().is_empty() {
            return Err(ObservationValidationError::EmptySensorId);
        }
        if let Some(location) = self.location {
            location.validate()?;
        }
        self.quality.validate()?;
        if let Some(asset) = &self.asset {
            asset.validate()?;
        }
        if self.disclosure.class == DisclosureClass::Secret
            && self.disclosure.share_raw_asset
        {
            return Err(ObservationValidationError::SecretRawAssetExport);
        }
        Ok(())
    }
}

/// Directed relationship between two observations in the evidence graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObservationRelationKind {
    /// The source observation provides positive evidence for the target.
    Supports,
    /// The source observation provides evidence against the target.
    Contradicts,
    /// The source independently agrees with the target observation.
    Corroborates,
    /// The source was computationally derived from the target.
    DerivedFrom,
    /// The observations support an identity match.
    SameEntity,
    /// The observations may refer to the same entity, but the match is unresolved.
    PossibleSameEntity,
}

/// Auditable edge between observations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceIndependence {
    /// The observations have no declared shared upstream source.
    Independent,
    /// The observations share an upstream source, platform, or processing chain.
    SharedUpstream,
    /// The source observation is computationally derived from the target lineage.
    Derived,
    /// Independence has not been established.
    Unknown,
}

/// Auditable edge between observations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationRelation {
    pub source_observation_id: String,
    pub target_observation_id: String,
    pub kind: ObservationRelationKind,
    pub independence: EvidenceIndependence,
}

impl ObservationRelation {
    /// Validate graph-edge invariants before insertion.
    pub fn validate(&self) -> Result<(), ObservationValidationError> {
        if self.source_observation_id.trim().is_empty()
            || self.target_observation_id.trim().is_empty()
        {
            return Err(ObservationValidationError::EmptyRelationId);
        }
        if self.source_observation_id == self.target_observation_id {
            return Err(ObservationValidationError::SelfRelation);
        }
        if matches!(self.kind, ObservationRelationKind::DerivedFrom)
            && !matches!(self.independence, EvidenceIndependence::Derived | EvidenceIndependence::SharedUpstream)
        {
            return Err(ObservationValidationError::DerivedRelationIndependenceMismatch);
        }
        Ok(())
    }
}

/// Fail-closed validation errors for malformed sensor metadata.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ObservationValidationError {
    #[error("observation id must not be empty")]
    EmptyId,
    #[error("sensor id must not be empty")]
    EmptySensorId,
    #[error("location is outside valid geographic bounds or has invalid uncertainty")]
    InvalidLocation,
    #[error("quality/confidence is outside valid bounds")]
    InvalidQuality,
    #[error("unsupported content hash algorithm: {0}")]
    UnsupportedHashAlgorithm(String),
    #[error("content hash must be 64 hexadecimal characters")]
    InvalidContentHash,
    #[error("secret observations cannot permit raw-asset export")]
    SecretRawAssetExport,
    #[error("relation observation ids must not be empty")]
    EmptyRelationId,
    #[error("an observation relation cannot point to itself")]
    SelfRelation,
    #[error("derived-from relations require derived or shared-upstream independence")]
    DerivedRelationIndependenceMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Observation {
        Observation {
            id: "obs-001".into(),
            modality: ObservationModality::Optical,
            time: ObservationTime {
                observed_at_unix_ns: 1_700_000_000_000_000_000,
                time_uncertainty_ns: 2_000_000,
            },
            location: Some(ObservationLocation {
                latitude_deg: 10.0,
                longitude_deg: 20.0,
                uncertainty_m: 4.0,
            }),
            quality: ObservationQuality {
                confidence: 0.92,
                measurement_uncertainty: Some(0.3),
                calibrated: true,
            },
            provenance: ObservationProvenance {
                source: SensorIdentity::new("camera-1"),
                acquired_by: None,
                parent_observation_ids: Vec::new(),
                processing_fingerprint: Some("proc-v1".into()),
            },
            asset: Some(AssetRef::blake3(b"frame-bytes")),
            disclosure: DisclosurePolicy::restricted(),
        }
    }

    #[test]
    fn valid_observation_passes() {
        assert!(fixture().validate().is_ok());
    }

    #[test]
    fn invalid_location_fails_closed() {
        let mut observation = fixture();
        observation.location = Some(ObservationLocation {
            latitude_deg: 91.0,
            longitude_deg: 0.0,
            uncertainty_m: 1.0,
        });
        assert_eq!(
            observation.validate(),
            Err(ObservationValidationError::InvalidLocation)
        );
    }

    #[test]
    fn invalid_confidence_fails_closed() {
        let mut observation = fixture();
        observation.quality.confidence = 1.5;
        assert_eq!(
            observation.validate(),
            Err(ObservationValidationError::InvalidQuality)
        );
    }

    #[test]
    fn secret_raw_export_fails_closed() {
        let mut observation = fixture();
        observation.disclosure = DisclosurePolicy {
            class: DisclosureClass::Secret,
            share_exact_time: false,
            share_exact_location: false,
            share_raw_asset: true,
            share_derived_features: false,
        };
        assert_eq!(
            observation.validate(),
            Err(ObservationValidationError::SecretRawAssetExport)
        );
    }

    #[test]
    fn asset_hash_is_content_addressed() {
        let a = AssetRef::blake3(b"same");
        let b = AssetRef::blake3(b"same");
        let c = AssetRef::blake3(b"different");
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a.validate().is_ok());
    }

    #[test]
    fn relation_rejects_empty_ids() {
        let relation = ObservationRelation {
            source_observation_id: String::new(),
            target_observation_id: "obs-2".into(),
            kind: ObservationRelationKind::Supports,
            independence: EvidenceIndependence::Unknown,
        };
        assert_eq!(
            relation.validate(),
            Err(ObservationValidationError::EmptyRelationId)
        );
    }

    #[test]
    fn relation_rejects_self_edge() {
        let relation = ObservationRelation {
            source_observation_id: "obs-1".into(),
            target_observation_id: "obs-1".into(),
            kind: ObservationRelationKind::Corroborates,
            independence: EvidenceIndependence::Independent,
        };
        assert_eq!(
            relation.validate(),
            Err(ObservationValidationError::SelfRelation)
        );
    }

    #[test]
    fn derived_relation_requires_lineage_independence_marker() {
        let relation = ObservationRelation {
            source_observation_id: "derived".into(),
            target_observation_id: "source".into(),
            kind: ObservationRelationKind::DerivedFrom,
            independence: EvidenceIndependence::Independent,
        };
        assert_eq!(
            relation.validate(),
            Err(ObservationValidationError::DerivedRelationIndependenceMismatch)
        );
    }

    #[test]
    fn observation_round_trips_json() {
        let original = fixture();
        let encoded = serde_json::to_string(&original).expect("serialize");
        let decoded: Observation = serde_json::from_str(&encoded).expect("deserialize");
        assert_eq!(original, decoded);
    }
}
