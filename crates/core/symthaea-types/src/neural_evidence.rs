// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Canonical neural-evidence provenance types for Symthaea.
//!
//! This module separates four things that must never be conflated:
//!
//! 1. synthetic software fixtures,
//! 2. simulated model state,
//! 3. predictions from external neural surrogate models,
//! 4. observed neural recordings.
//!
//! Representational resemblance does not grant evidential authority. Evidence
//! use is authorized only by [`ValidatedProvenance`], never by a raw authority
//! enum alone. Spatial coordinate changes are likewise valid only when an
//! explicit, versioned, digest-bound transform chain connects the native and
//! current coordinate systems.

use serde::{Deserialize, Deserializer, Serialize};

/// Current serialized schema version for [`NeuralEvidenceProvenance`].
pub const NEURAL_EVIDENCE_SCHEMA_VERSION: u16 = 1;

/// Evidential authority of a neural representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvidenceAuthority {
    /// Deterministic or stochastic data generated solely for software tests.
    SyntheticFixture,
    /// State produced by Symthaea or another explicit simulation/model.
    SimulatedModel,
    /// Prediction from an external model trained to predict neural data.
    ExternalSurrogate,
    /// Neural data measured from a participant or organism.
    EmpiricalObserved,
}

impl EvidenceAuthority {
    /// Whether this authority represents an observed biological measurement.
    pub const fn is_empirical_observation(self) -> bool {
        matches!(self, Self::EmpiricalObserved)
    }

    /// Whether this authority represents an external neural prediction model.
    pub const fn is_external_surrogate(self) -> bool {
        matches!(self, Self::ExternalSurrogate)
    }
}

/// Intended use of an evidence artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvidenceUse {
    /// Software-level format, reproducibility, or invariant checking.
    SoftwareQualification,
    /// Analysis of the behavior of an explicit simulation/model.
    ModelBehavior,
    /// Comparison against an external brain-encoding or neural surrogate model.
    SurrogateAlignment,
    /// Analysis that claims correspondence with observed neural measurements.
    EmpiricalNeuralAnalysis,
    /// A claim about consciousness itself.
    ConsciousnessInference,
}

/// Neural recording or prediction modality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NeuralModality {
    Fmri,
    Eeg,
    Meg,
    Ieeg,
    TmsEeg,
}

/// Coordinate system in which neural values live.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CoordinateSystem {
    #[serde(rename = "symthaea12")]
    Symthaea12,
    #[serde(rename = "fsaverage5")]
    FsAverage5,
    #[serde(rename = "hcp_mmp1")]
    HcpMmp1,
    #[serde(rename = "glasser360")]
    Glasser360,
    #[serde(rename = "desikan_killiany68")]
    DesikanKilliany68,
    #[serde(rename = "schaefer100")]
    Schaefer100,
    #[serde(rename = "sensor_space")]
    SensorSpace,
}

/// Temporal reduction applied independently of spatial coordinate transforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TemporalAggregation {
    #[serde(rename = "native")]
    Native,
    #[serde(rename = "temporal_mean")]
    TemporalMean,
}

/// Explicit provenance for one spatial coordinate-system transformation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoordinateTransformProvenance {
    pub from: CoordinateSystem,
    pub to: CoordinateSystem,
    pub transform_id: String,
    pub transform_version: String,
    /// Immutable digest of the atlas/mapping artifact or controlled transform source.
    pub source_digest: String,
}

impl CoordinateTransformProvenance {
    /// Validate this transformation independently.
    pub fn validate(self) -> Result<ValidatedCoordinateTransform, ProvenanceError> {
        validate_transform_fields(&self)?;
        Ok(ValidatedCoordinateTransform(self))
    }
}

/// A coordinate transform whose identity, version, digest, and endpoints are valid.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ValidatedCoordinateTransform(CoordinateTransformProvenance);

impl ValidatedCoordinateTransform {
    pub const fn from(&self) -> CoordinateSystem {
        self.0.from
    }

    pub const fn to(&self) -> CoordinateSystem {
        self.0.to
    }

    pub fn as_raw(&self) -> &CoordinateTransformProvenance {
        &self.0
    }
}

impl<'de> Deserialize<'de> for ValidatedCoordinateTransform {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = CoordinateTransformProvenance::deserialize(deserializer)?;
        raw.validate().map_err(serde::de::Error::custom)
    }
}

/// Raw provenance payload before validation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeuralEvidenceProvenance {
    /// Serialization contract version. Unsupported versions fail closed.
    pub schema_version: u16,
    pub authority: EvidenceAuthority,
    pub modality: Option<NeuralModality>,
    /// Human-readable producer/source identity.
    pub source: String,
    /// Model identity for simulated and surrogate evidence.
    pub model: Option<String>,
    /// Explicit model/runtime revision for reproducibility.
    pub model_version: Option<String>,
    /// Dataset identity for observed empirical evidence.
    pub dataset: Option<String>,
    pub subject_id: Option<String>,
    pub stimulus_id: Option<String>,
    /// Coordinate system in which the source representation originally lived.
    pub native_coordinate_system: CoordinateSystem,
    /// Coordinate system in which the current data payload lives.
    pub coordinate_system: CoordinateSystem,
    /// Ordered spatial lineage from native to current coordinates.
    #[serde(default)]
    pub coordinate_transforms: Vec<CoordinateTransformProvenance>,
    /// Temporal reduction, independent from spatial transformation lineage.
    pub temporal_aggregation: TemporalAggregation,
    /// Optional immutable source/content digest, e.g. `sha256:<hex>`.
    pub source_digest: Option<String>,
}

impl NeuralEvidenceProvenance {
    /// Validate provenance invariants and return a wrapper safe for evidence use.
    pub fn validate(self) -> Result<ValidatedProvenance, ProvenanceError> {
        ValidatedProvenance::try_from(self)
    }
}

/// Provenance that has passed authority, metadata, and coordinate-lineage invariants.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ValidatedProvenance(NeuralEvidenceProvenance);

impl ValidatedProvenance {
    pub const fn authority(&self) -> EvidenceAuthority {
        self.0.authority
    }

    pub const fn native_coordinate_system(&self) -> CoordinateSystem {
        self.0.native_coordinate_system
    }

    pub const fn coordinate_system(&self) -> CoordinateSystem {
        self.0.coordinate_system
    }

    pub const fn modality(&self) -> Option<NeuralModality> {
        self.0.modality
    }

    pub fn coordinate_transforms(&self) -> &[CoordinateTransformProvenance] {
        &self.0.coordinate_transforms
    }

    pub fn as_raw(&self) -> &NeuralEvidenceProvenance {
        &self.0
    }

    /// Whether this *validated provenance* may be used for the requested purpose.
    ///
    /// Raw [`EvidenceAuthority`] values deliberately expose no equivalent method.
    /// No validated provenance automatically permits consciousness inference.
    pub const fn permits(&self, use_case: EvidenceUse) -> bool {
        match (self.0.authority, use_case) {
            (_, EvidenceUse::SoftwareQualification) => true,
            (EvidenceAuthority::SimulatedModel, EvidenceUse::ModelBehavior) => true,
            (EvidenceAuthority::ExternalSurrogate, EvidenceUse::SurrogateAlignment) => true,
            (EvidenceAuthority::EmpiricalObserved, EvidenceUse::EmpiricalNeuralAnalysis) => true,
            (_, EvidenceUse::ConsciousnessInference) => false,
            _ => false,
        }
    }
}

impl TryFrom<NeuralEvidenceProvenance> for ValidatedProvenance {
    type Error = ProvenanceError;

    fn try_from(value: NeuralEvidenceProvenance) -> Result<Self, Self::Error> {
        if value.schema_version != NEURAL_EVIDENCE_SCHEMA_VERSION {
            return Err(ProvenanceError::UnsupportedSchemaVersion {
                found: value.schema_version,
            });
        }
        if value.source.trim().is_empty() {
            return Err(ProvenanceError::MissingSource);
        }
        if let Some(digest) = value.source_digest.as_deref() {
            validate_digest(digest)?;
        }

        match value.authority {
            EvidenceAuthority::SyntheticFixture => {}
            EvidenceAuthority::SimulatedModel => {
                require_model_identity(&value, ProvenanceError::SimulatedMissingModel)?;
                require_model_version(&value, ProvenanceError::SimulatedMissingModelVersion)?;
            }
            EvidenceAuthority::ExternalSurrogate => {
                require_model_identity(&value, ProvenanceError::SurrogateMissingModel)?;
                require_model_version(&value, ProvenanceError::SurrogateMissingModelVersion)?;
                if value.modality.is_none() {
                    return Err(ProvenanceError::SurrogateMissingModality);
                }
            }
            EvidenceAuthority::EmpiricalObserved => {
                if value.dataset.as_deref().is_none_or(|v| v.trim().is_empty()) {
                    return Err(ProvenanceError::EmpiricalMissingDataset);
                }
                if value.modality.is_none() {
                    return Err(ProvenanceError::EmpiricalMissingModality);
                }
            }
        }

        validate_coordinate_chain(&value)?;
        Ok(Self(value))
    }
}

impl<'de> Deserialize<'de> for ValidatedProvenance {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = NeuralEvidenceProvenance::deserialize(deserializer)?;
        raw.validate().map_err(serde::de::Error::custom)
    }
}

/// A neural representation paired with validated provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NeuralObservation<T> {
    data: T,
    provenance: ValidatedProvenance,
}

impl<T> NeuralObservation<T> {
    pub fn new(data: T, provenance: NeuralEvidenceProvenance) -> Result<Self, ProvenanceError> {
        Ok(Self {
            data,
            provenance: provenance.validate()?,
        })
    }

    pub fn data(&self) -> &T {
        &self.data
    }

    pub fn provenance(&self) -> &ValidatedProvenance {
        &self.provenance
    }

    pub const fn permits(&self, use_case: EvidenceUse) -> bool {
        self.provenance.permits(use_case)
    }

    pub fn into_parts(self) -> (T, ValidatedProvenance) {
        (self.data, self.provenance)
    }
}

fn require_model_identity(
    value: &NeuralEvidenceProvenance,
    error: ProvenanceError,
) -> Result<(), ProvenanceError> {
    if value.model.as_deref().is_none_or(|v| v.trim().is_empty()) {
        return Err(error);
    }
    Ok(())
}

fn require_model_version(
    value: &NeuralEvidenceProvenance,
    error: ProvenanceError,
) -> Result<(), ProvenanceError> {
    if value
        .model_version
        .as_deref()
        .is_none_or(|v| v.trim().is_empty())
    {
        return Err(error);
    }
    Ok(())
}

fn validate_coordinate_chain(value: &NeuralEvidenceProvenance) -> Result<(), ProvenanceError> {
    let mut current = value.native_coordinate_system;

    for (index, transform) in value.coordinate_transforms.iter().enumerate() {
        validate_transform_fields(transform)?;
        if transform.from != current {
            return Err(ProvenanceError::TransformChainSourceMismatch {
                index,
                expected: current,
                found: transform.from,
            });
        }
        current = transform.to;
    }

    if current != value.coordinate_system {
        return Err(ProvenanceError::TransformChainTargetMismatch {
            expected: value.coordinate_system,
            found: current,
        });
    }

    Ok(())
}

fn validate_transform_fields(transform: &CoordinateTransformProvenance) -> Result<(), ProvenanceError> {
    if transform.from == transform.to {
        return Err(ProvenanceError::IdentityCoordinateTransform);
    }
    if transform.transform_id.trim().is_empty() {
        return Err(ProvenanceError::MissingTransformId);
    }
    if transform.transform_version.trim().is_empty() {
        return Err(ProvenanceError::MissingTransformVersion);
    }
    validate_digest(&transform.source_digest)
}

fn validate_digest(digest: &str) -> Result<(), ProvenanceError> {
    let Some((algorithm, value)) = digest.split_once(':') else {
        return Err(ProvenanceError::MalformedDigest);
    };
    if algorithm.trim().is_empty() || value.trim().is_empty() {
        return Err(ProvenanceError::MalformedDigest);
    }
    Ok(())
}

/// Validation errors for neural evidence and coordinate-transform provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProvenanceError {
    UnsupportedSchemaVersion { found: u16 },
    MissingSource,
    SimulatedMissingModel,
    SimulatedMissingModelVersion,
    SurrogateMissingModel,
    SurrogateMissingModelVersion,
    SurrogateMissingModality,
    EmpiricalMissingDataset,
    EmpiricalMissingModality,
    MalformedDigest,
    IdentityCoordinateTransform,
    MissingTransformId,
    MissingTransformVersion,
    TransformChainSourceMismatch {
        index: usize,
        expected: CoordinateSystem,
        found: CoordinateSystem,
    },
    TransformChainTargetMismatch {
        expected: CoordinateSystem,
        found: CoordinateSystem,
    },
}

impl std::fmt::Display for ProvenanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => write!(
                f,
                "unsupported neural evidence schema version {found}; expected {NEURAL_EVIDENCE_SCHEMA_VERSION}"
            ),
            Self::MissingSource => f.write_str("neural evidence source must be explicit"),
            Self::SimulatedMissingModel => {
                f.write_str("simulated model evidence requires an explicit model identifier")
            }
            Self::SimulatedMissingModelVersion => {
                f.write_str("simulated model evidence requires an explicit model version")
            }
            Self::SurrogateMissingModel => {
                f.write_str("external surrogate evidence requires an explicit model identifier")
            }
            Self::SurrogateMissingModelVersion => {
                f.write_str("external surrogate evidence requires an explicit model version")
            }
            Self::SurrogateMissingModality => {
                f.write_str("external surrogate evidence requires an explicit neural modality")
            }
            Self::EmpiricalMissingDataset => {
                f.write_str("empirical observed evidence requires an explicit dataset identifier")
            }
            Self::EmpiricalMissingModality => {
                f.write_str("empirical observed evidence requires an explicit neural modality")
            }
            Self::MalformedDigest => {
                f.write_str("source digest must use a non-empty algorithm:value form")
            }
            Self::IdentityCoordinateTransform => {
                f.write_str("coordinate transforms must change coordinate systems")
            }
            Self::MissingTransformId => f.write_str("coordinate transform id must be explicit"),
            Self::MissingTransformVersion => {
                f.write_str("coordinate transform version must be explicit")
            }
            Self::TransformChainSourceMismatch {
                index,
                expected,
                found,
            } => write!(
                f,
                "coordinate transform {index} expected input {expected:?}, found {found:?}"
            ),
            Self::TransformChainTargetMismatch { expected, found } => write!(
                f,
                "coordinate transform chain ended at {found:?}, expected {expected:?}"
            ),
        }
    }
}

impl std::error::Error for ProvenanceError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic() -> NeuralEvidenceProvenance {
        NeuralEvidenceProvenance {
            schema_version: NEURAL_EVIDENCE_SCHEMA_VERSION,
            authority: EvidenceAuthority::SyntheticFixture,
            modality: Some(NeuralModality::Fmri),
            source: "unit-test".into(),
            model: Some("symthaea-neural-fixture".into()),
            model_version: Some("1".into()),
            dataset: None,
            subject_id: None,
            stimulus_id: Some("fixture-01".into()),
            native_coordinate_system: CoordinateSystem::Symthaea12,
            coordinate_system: CoordinateSystem::Symthaea12,
            coordinate_transforms: vec![],
            temporal_aggregation: TemporalAggregation::Native,
            source_digest: Some("sha256:fixture".into()),
        }
    }

    fn simulated() -> NeuralEvidenceProvenance {
        NeuralEvidenceProvenance {
            schema_version: NEURAL_EVIDENCE_SCHEMA_VERSION,
            authority: EvidenceAuthority::SimulatedModel,
            modality: None,
            source: "Symthaea".into(),
            model: Some("cognitive-loop-cortical-map".into()),
            model_version: Some("git:2a8b8fd".into()),
            dataset: None,
            subject_id: None,
            stimulus_id: Some("sim-01".into()),
            native_coordinate_system: CoordinateSystem::Symthaea12,
            coordinate_system: CoordinateSystem::Symthaea12,
            coordinate_transforms: vec![],
            temporal_aggregation: TemporalAggregation::Native,
            source_digest: None,
        }
    }

    fn surrogate() -> NeuralEvidenceProvenance {
        NeuralEvidenceProvenance {
            schema_version: NEURAL_EVIDENCE_SCHEMA_VERSION,
            authority: EvidenceAuthority::ExternalSurrogate,
            modality: Some(NeuralModality::Fmri),
            source: "TRIBE v2".into(),
            model: Some("facebook/tribev2".into()),
            model_version: Some("released-api-v2".into()),
            dataset: None,
            subject_id: None,
            stimulus_id: Some("movie-01".into()),
            native_coordinate_system: CoordinateSystem::FsAverage5,
            coordinate_system: CoordinateSystem::FsAverage5,
            coordinate_transforms: vec![],
            temporal_aggregation: TemporalAggregation::TemporalMean,
            source_digest: Some("sha256:tribe-output".into()),
        }
    }

    fn empirical() -> NeuralEvidenceProvenance {
        NeuralEvidenceProvenance {
            schema_version: NEURAL_EVIDENCE_SCHEMA_VERSION,
            authority: EvidenceAuthority::EmpiricalObserved,
            modality: Some(NeuralModality::Fmri),
            source: "Natural Scenes Dataset".into(),
            model: None,
            model_version: None,
            dataset: Some("NSD".into()),
            subject_id: Some("subj01".into()),
            stimulus_id: Some("nsd-00001".into()),
            native_coordinate_system: CoordinateSystem::FsAverage5,
            coordinate_system: CoordinateSystem::FsAverage5,
            coordinate_transforms: vec![],
            temporal_aggregation: TemporalAggregation::Native,
            source_digest: Some("sha256:example".into()),
        }
    }

    fn fsaverage_to_symthaea_chain() -> Vec<CoordinateTransformProvenance> {
        vec![
            CoordinateTransformProvenance {
                from: CoordinateSystem::FsAverage5,
                to: CoordinateSystem::Glasser360,
                transform_id: "fsaverage5-to-glasser360".into(),
                transform_version: "1".into(),
                source_digest: "sha256:atlas".into(),
            },
            CoordinateTransformProvenance {
                from: CoordinateSystem::Glasser360,
                to: CoordinateSystem::Symthaea12,
                transform_id: "glasser360-to-symthaea12".into(),
                transform_version: "1".into(),
                source_digest: "sha256:mapping".into(),
            },
        ]
    }

    #[test]
    fn synthetic_never_permits_empirical_or_surrogate_analysis() {
        let p = synthetic().validate().unwrap();
        assert!(p.permits(EvidenceUse::SoftwareQualification));
        assert!(!p.permits(EvidenceUse::EmpiricalNeuralAnalysis));
        assert!(!p.permits(EvidenceUse::SurrogateAlignment));
    }

    #[test]
    fn simulated_only_permits_model_behavior_beyond_qualification() {
        let p = simulated().validate().unwrap();
        assert!(p.permits(EvidenceUse::ModelBehavior));
        assert!(!p.permits(EvidenceUse::SurrogateAlignment));
        assert!(!p.permits(EvidenceUse::EmpiricalNeuralAnalysis));
    }

    #[test]
    fn external_surrogate_is_not_empirical_observation() {
        let p = surrogate().validate().unwrap();
        assert!(p.authority().is_external_surrogate());
        assert!(!p.authority().is_empirical_observation());
        assert!(p.permits(EvidenceUse::SurrogateAlignment));
        assert!(!p.permits(EvidenceUse::EmpiricalNeuralAnalysis));
    }

    #[test]
    fn no_validated_authority_directly_permits_consciousness_inference() {
        for raw in [synthetic(), simulated(), surrogate(), empirical()] {
            let p = raw.validate().unwrap();
            assert!(!p.permits(EvidenceUse::ConsciousnessInference));
        }
    }

    #[test]
    fn unsupported_schema_version_fails_closed() {
        let mut p = empirical();
        p.schema_version += 1;
        assert_eq!(
            p.validate(),
            Err(ProvenanceError::UnsupportedSchemaVersion { found: 2 })
        );
    }

    #[test]
    fn empirical_observation_requires_dataset_and_modality() {
        let mut p = empirical();
        p.dataset = None;
        assert_eq!(p.validate(), Err(ProvenanceError::EmpiricalMissingDataset));

        let mut p = empirical();
        p.modality = None;
        assert_eq!(p.validate(), Err(ProvenanceError::EmpiricalMissingModality));
    }

    #[test]
    fn surrogate_requires_model_version_and_modality() {
        let mut p = surrogate();
        p.model = None;
        assert_eq!(p.validate(), Err(ProvenanceError::SurrogateMissingModel));

        let mut p = surrogate();
        p.model_version = Some("  ".into());
        assert_eq!(
            p.validate(),
            Err(ProvenanceError::SurrogateMissingModelVersion)
        );

        let mut p = surrogate();
        p.modality = None;
        assert_eq!(p.validate(), Err(ProvenanceError::SurrogateMissingModality));
    }

    #[test]
    fn simulated_requires_model_identity_and_version() {
        let mut p = simulated();
        p.model = Some("  ".into());
        assert_eq!(p.validate(), Err(ProvenanceError::SimulatedMissingModel));

        let mut p = simulated();
        p.model_version = None;
        assert_eq!(
            p.validate(),
            Err(ProvenanceError::SimulatedMissingModelVersion)
        );
    }

    #[test]
    fn temporal_mean_does_not_imply_spatial_transform() {
        let p = surrogate().validate().unwrap();
        assert_eq!(p.native_coordinate_system(), CoordinateSystem::FsAverage5);
        assert_eq!(p.coordinate_system(), CoordinateSystem::FsAverage5);
        assert!(p.coordinate_transforms().is_empty());
        assert_eq!(
            p.as_raw().temporal_aggregation,
            TemporalAggregation::TemporalMean
        );
    }

    #[test]
    fn coordinate_relabel_without_transform_fails_closed() {
        let mut p = surrogate();
        p.coordinate_system = CoordinateSystem::Glasser360;
        assert_eq!(
            p.validate(),
            Err(ProvenanceError::TransformChainTargetMismatch {
                expected: CoordinateSystem::Glasser360,
                found: CoordinateSystem::FsAverage5,
            })
        );
    }

    #[test]
    fn valid_coordinate_chain_binds_native_to_current_space() {
        let mut p = surrogate();
        p.coordinate_system = CoordinateSystem::Symthaea12;
        p.coordinate_transforms = fsaverage_to_symthaea_chain();
        let validated = p.validate().unwrap();
        assert_eq!(
            validated.native_coordinate_system(),
            CoordinateSystem::FsAverage5
        );
        assert_eq!(validated.coordinate_system(), CoordinateSystem::Symthaea12);
        assert_eq!(validated.coordinate_transforms().len(), 2);
    }

    #[test]
    fn broken_coordinate_chain_is_rejected() {
        let mut p = surrogate();
        p.coordinate_system = CoordinateSystem::Symthaea12;
        let mut chain = fsaverage_to_symthaea_chain();
        chain[1].from = CoordinateSystem::HcpMmp1;
        p.coordinate_transforms = chain;
        assert_eq!(
            p.validate(),
            Err(ProvenanceError::TransformChainSourceMismatch {
                index: 1,
                expected: CoordinateSystem::Glasser360,
                found: CoordinateSystem::HcpMmp1,
            })
        );
    }

    #[test]
    fn standalone_coordinate_transform_requires_identity_version_and_digest() {
        let valid = CoordinateTransformProvenance {
            from: CoordinateSystem::FsAverage5,
            to: CoordinateSystem::Glasser360,
            transform_id: "fsaverage5-to-glasser360".into(),
            transform_version: "1".into(),
            source_digest: "sha256:atlas".into(),
        };
        let validated = valid.clone().validate().unwrap();
        assert_eq!(validated.from(), CoordinateSystem::FsAverage5);
        assert_eq!(validated.to(), CoordinateSystem::Glasser360);

        let invalid = CoordinateTransformProvenance {
            to: CoordinateSystem::FsAverage5,
            ..valid.clone()
        };
        assert_eq!(
            invalid.validate(),
            Err(ProvenanceError::IdentityCoordinateTransform)
        );

        let mut invalid_digest = valid;
        invalid_digest.source_digest.clear();
        assert_eq!(
            invalid_digest.validate(),
            Err(ProvenanceError::MalformedDigest)
        );
    }

    #[test]
    fn malformed_digest_is_rejected() {
        let mut p = empirical();
        p.source_digest = Some("not-a-digest".into());
        assert_eq!(p.validate(), Err(ProvenanceError::MalformedDigest));
    }

    #[test]
    fn validated_provenance_rejects_invalid_deserialization() {
        let mut raw = surrogate();
        raw.coordinate_system = CoordinateSystem::Glasser360;
        let json = serde_json::to_string(&raw).unwrap();
        let parsed = serde_json::from_str::<ValidatedProvenance>(&json);
        assert!(parsed.is_err());
    }

    #[test]
    fn validated_transform_rejects_invalid_deserialization() {
        let raw = CoordinateTransformProvenance {
            from: CoordinateSystem::FsAverage5,
            to: CoordinateSystem::FsAverage5,
            transform_id: "bad".into(),
            transform_version: "1".into(),
            source_digest: "sha256:atlas".into(),
        };
        let json = serde_json::to_string(&raw).unwrap();
        let parsed = serde_json::from_str::<ValidatedCoordinateTransform>(&json);
        assert!(parsed.is_err());
    }

    #[test]
    fn neural_observation_roundtrip_preserves_authority_and_lineage() {
        let mut p = surrogate();
        p.coordinate_system = CoordinateSystem::Symthaea12;
        p.coordinate_transforms = fsaverage_to_symthaea_chain();
        let observation = NeuralObservation::new(vec![0.1_f32, 0.2, 0.3], p).unwrap();
        let json = serde_json::to_string(&observation).unwrap();
        let restored: NeuralObservation<Vec<f32>> = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.data(), &[0.1, 0.2, 0.3]);
        assert_eq!(
            restored.provenance().authority(),
            EvidenceAuthority::ExternalSurrogate
        );
        assert_eq!(restored.provenance().modality(), Some(NeuralModality::Fmri));
        assert_eq!(restored.provenance().coordinate_transforms().len(), 2);
    }

    #[test]
    fn modality_serialization_is_stable_snake_case() {
        assert_eq!(
            serde_json::to_string(&NeuralModality::TmsEeg).unwrap(),
            "\"tms_eeg\""
        );
    }
}
