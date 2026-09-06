// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Generic empirical-observation evidence authority for Symthaea.
//!
//! This module deliberately separates four transitions:
//!
//! ```text
//! self-described observation record
//!       !=
//! structurally validated observation
//!       !=
//! application-policy-admitted direct observation
//!       !=
//! belief support / canonical belief / action authority
//! ```
//!
//! A record cannot grant itself empirical authority merely by declaring that it
//! came from a sensor. Admission requires a separate runtime policy that names
//! the exact trusted source generation and permitted `(modality, payload_format)`
//! pair. The contract is source-agnostic: cameras, RGB-D sensors, lidar, event
//! cameras, IMUs, microphones, tactile arrays, scientific instruments, and
//! future sensor classes can use the same boundary without inheriting neural-only
//! vocabulary.
//!
//! Activating a policy is the explicit application/owning-adapter trust-root
//! operation. This module does not cryptographically authenticate a physical
//! device or prove ownership of `source_id`; source identities are logical
//! labels. Digest fields are canonical, structurally validated content identities
//! unless an owning boundary separately verifies their relationship to authentic
//! bytes.
//!
//! Structural validation and policy admission still do not prove that an
//! observation is true, current, belief-worthy, or action-authorizing.

use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashSet;
use std::num::NonZeroU64;

/// Current serialized schema version for generic observation evidence.
pub const OBSERVATION_EVIDENCE_SCHEMA_VERSION: u16 = 1;

/// Declared origin of one observation-shaped record.
///
/// This is provenance metadata, not admission authority. In particular,
/// `ExternalObservation` cannot authorize itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationOriginV1 {
    /// Deterministic or stochastic software/test fixture.
    SyntheticFixture,
    /// Observation-shaped output produced by a simulator or predictive model.
    SimulatedObservation,
    /// Observation declared to originate at an external empirical boundary.
    ExternalObservation,
}

impl ObservationOriginV1 {
    const fn requires_model_identity(self) -> bool {
        matches!(self, Self::SyntheticFixture | Self::SimulatedObservation)
    }
}

/// Explicit downstream use requested from the admission policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationEvidenceUseV1 {
    /// Admit one exact external observation for a downstream owning adapter.
    ObservationAdmission,
    /// Reserved transition. Observation admission is not belief-support authority.
    BeliefSupport,
    /// Reserved transition. Observations do not directly create canonical beliefs.
    CanonicalBeliefAdmission,
    /// Reserved transition. Observations are never external-effect authority.
    ActionAuthority,
}

/// Raw source-agnostic observation reference before structural validation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationEvidenceRefV1 {
    pub schema_version: u16,
    /// Self-described provenance only; this field cannot mint admission.
    pub origin: ObservationOriginV1,
    /// Stable identity of the producer/acquisition boundary.
    pub source_id: String,
    /// Non-zero source generation. Reset/reuse/resemanticization requires a new generation.
    pub source_generation: NonZeroU64,
    /// Optional firmware/software/instrument revision. Metadata only.
    pub source_revision: Option<String>,
    /// Producer-local identity of this exact observation within the source generation.
    pub observation_id: String,
    /// Source-agnostic descriptive modality, e.g. `rgbd`, `lidar`, `imu`, `audio`.
    pub modality: String,
    /// Exact payload representation identity/version whose bytes are named by `claim_digest`.
    ///
    /// This is intentionally distinct from broad modality. For example, two
    /// `rgbd` streams using optical-z f64 and integer-millimetre range payloads
    /// must have different payload-format identities.
    pub payload_format: String,
    /// Digest of the exact observation statement/payload represented by this record.
    pub claim_digest: String,
    /// Optional digest of a source artifact/configuration used to produce the observation.
    pub source_artifact_digest: Option<String>,
    /// Optional digest of calibration data required to interpret the observation.
    pub calibration_digest: Option<String>,
    /// Explicit generator/model identity for synthetic or simulated records.
    pub model_id: Option<String>,
    /// Explicit generator/model revision for synthetic or simulated records.
    pub model_version: Option<String>,
    /// Optional source-owned epoch. Metadata only; this is not proof of currentness.
    pub freshness_epoch: Option<u64>,
}

impl ObservationEvidenceRefV1 {
    pub fn validate(
        self,
    ) -> Result<ValidatedObservationEvidenceRefV1, ObservationEvidenceError> {
        ValidatedObservationEvidenceRefV1::try_from(self)
    }
}

/// Observation reference whose schema, identity, provenance, and digests validated.
///
/// This type remains non-authorizing. It intentionally has no `authorize` or
/// `admit` method; admission belongs to [`ObservationAdmissionPolicyV1`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ValidatedObservationEvidenceRefV1(ObservationEvidenceRefV1);

impl ValidatedObservationEvidenceRefV1 {
    pub const fn origin(&self) -> ObservationOriginV1 {
        self.0.origin
    }

    pub fn source_id(&self) -> &str {
        &self.0.source_id
    }

    pub const fn source_generation(&self) -> NonZeroU64 {
        self.0.source_generation
    }

    pub fn observation_id(&self) -> &str {
        &self.0.observation_id
    }

    pub fn claim_digest(&self) -> &str {
        &self.0.claim_digest
    }

    pub fn modality(&self) -> &str {
        &self.0.modality
    }

    pub fn payload_format(&self) -> &str {
        &self.0.payload_format
    }

    pub fn as_raw(&self) -> &ObservationEvidenceRefV1 {
        &self.0
    }
}

impl TryFrom<ObservationEvidenceRefV1> for ValidatedObservationEvidenceRefV1 {
    type Error = ObservationEvidenceError;

    fn try_from(value: ObservationEvidenceRefV1) -> Result<Self, Self::Error> {
        if value.schema_version != OBSERVATION_EVIDENCE_SCHEMA_VERSION {
            return Err(ObservationEvidenceError::UnsupportedSchemaVersion {
                found: value.schema_version,
            });
        }

        require_nonempty(&value.source_id, ObservationEvidenceError::MissingSourceId)?;
        require_canonical_text("source_id", &value.source_id)?;
        require_nonempty(
            &value.observation_id,
            ObservationEvidenceError::MissingObservationId,
        )?;
        require_canonical_text("observation_id", &value.observation_id)?;
        require_nonempty(&value.modality, ObservationEvidenceError::MissingModality)?;
        require_canonical_text("modality", &value.modality)?;
        require_nonempty(
            &value.payload_format,
            ObservationEvidenceError::MissingPayloadFormat,
        )?;
        require_canonical_text("payload_format", &value.payload_format)?;
        validate_digest(&value.claim_digest)?;

        validate_optional_nonempty(
            value.source_revision.as_deref(),
            ObservationEvidenceError::EmptySourceRevision,
        )?;
        validate_optional_canonical_text("source_revision", value.source_revision.as_deref())?;
        validate_optional_digest(value.source_artifact_digest.as_deref())?;
        validate_optional_digest(value.calibration_digest.as_deref())?;

        if value.origin.requires_model_identity() {
            require_optional_nonempty(
                value.model_id.as_deref(),
                ObservationEvidenceError::MissingModelId {
                    origin: value.origin,
                },
            )?;
            require_optional_nonempty(
                value.model_version.as_deref(),
                ObservationEvidenceError::MissingModelVersion {
                    origin: value.origin,
                },
            )?;
            validate_optional_canonical_text("model_id", value.model_id.as_deref())?;
            validate_optional_canonical_text("model_version", value.model_version.as_deref())?;
        } else if value.model_id.is_some() || value.model_version.is_some() {
            return Err(ObservationEvidenceError::ExternalRecordCarriesModelIdentity);
        }

        Ok(Self(value))
    }
}

impl<'de> Deserialize<'de> for ValidatedObservationEvidenceRefV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        ObservationEvidenceRefV1::deserialize(deserializer)?
            .validate()
            .map_err(serde::de::Error::custom)
    }
}

/// One exact `(modality, payload_format)` pair allowed by a trusted source policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PermittedObservationFormatV1 {
    pub modality: String,
    pub payload_format: String,
}

/// One exact source generation trusted by an observation-admission policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrustedObservationSourceV1 {
    pub source_id: String,
    pub source_generation: NonZeroU64,
    /// When present, the observation must carry this exact source revision.
    pub required_source_revision: Option<String>,
    /// When present, the observation must bind this exact source artifact.
    pub required_source_artifact_digest: Option<String>,
    /// When present, the observation must bind this exact calibration artifact.
    pub required_calibration_digest: Option<String>,
    /// Closed set of exact modality/payload-format pairs this source may admit.
    pub permitted_formats: Vec<PermittedObservationFormatV1>,
}

/// Serializable policy configuration. This is configuration, not a runtime
/// admission capability by itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationAdmissionPolicyConfigV1 {
    pub schema_version: u16,
    /// Human/operator-owned policy identity.
    pub policy_id: String,
    /// Non-zero policy generation; semantic policy changes require a new generation.
    pub policy_generation: NonZeroU64,
    /// Declared immutable digest identity of the policy source/configuration artifact.
    ///
    /// This module validates digest structure but does not independently prove
    /// that the supplied digest was computed from an authentic policy artifact.
    pub policy_source_digest: String,
    pub trusted_sources: Vec<TrustedObservationSourceV1>,
}

impl ObservationAdmissionPolicyConfigV1 {
    /// Validate and explicitly activate this configuration as an application trust root.
    ///
    /// Calling this method is the trust decision. It must occur only at an owning
    /// application/adapter boundary that is authorized to choose trusted sources.
    /// It does not authenticate hardware merely because a logical `source_id`
    /// matches the policy.
    pub fn activate_as_trust_root(
        self,
    ) -> Result<ObservationAdmissionPolicyV1, ObservationPolicyError> {
        validate_policy_config(&self)?;
        Ok(ObservationAdmissionPolicyV1 { config: self })
    }
}

/// Runtime observation-admission policy.
///
/// Deliberately not serializable/deserializable or cloneable: persistence stores
/// policy configuration, not an already-active admission capability. The owning
/// application may activate a fresh policy explicitly at its trust boundary.
#[derive(Debug, PartialEq, Eq)]
#[must_use = "an activated observation policy is a runtime trust capability"]
pub struct ObservationAdmissionPolicyV1 {
    config: ObservationAdmissionPolicyConfigV1,
}

impl ObservationAdmissionPolicyV1 {
    pub fn policy_id(&self) -> &str {
        &self.config.policy_id
    }

    pub const fn policy_generation(&self) -> NonZeroU64 {
        self.config.policy_generation
    }

    pub fn policy_source_digest(&self) -> &str {
        &self.config.policy_source_digest
    }

    /// Consume one validated observation and admit it for one explicitly
    /// requested use if this runtime policy authorizes the exact source
    /// generation and observation-format pair.
    pub fn authorize(
        &self,
        evidence: ValidatedObservationEvidenceRefV1,
        use_case: ObservationEvidenceUseV1,
    ) -> Result<AdmittedObservationEvidenceV1, ObservationAdmissionError> {
        if use_case != ObservationEvidenceUseV1::ObservationAdmission {
            return Err(ObservationAdmissionError::ReservedUse { requested: use_case });
        }
        if evidence.origin() != ObservationOriginV1::ExternalObservation {
            return Err(ObservationAdmissionError::NonExternalOrigin {
                found: evidence.origin(),
            });
        }

        let Some(source) = self.config.trusted_sources.iter().find(|source| {
            source.source_id == evidence.source_id()
                && source.source_generation == evidence.source_generation()
        }) else {
            return Err(ObservationAdmissionError::UntrustedSourceGeneration);
        };

        if !source.permitted_formats.iter().any(|format| {
            format.modality == evidence.modality()
                && format.payload_format == evidence.payload_format()
        }) {
            return Err(ObservationAdmissionError::ObservationFormatNotPermitted);
        }

        if let Some(expected) = source.required_source_revision.as_deref() {
            if evidence.as_raw().source_revision.as_deref() != Some(expected) {
                return Err(ObservationAdmissionError::SourceRevisionMismatch);
            }
        }
        if let Some(expected) = source.required_source_artifact_digest.as_deref() {
            if evidence.as_raw().source_artifact_digest.as_deref() != Some(expected) {
                return Err(ObservationAdmissionError::SourceArtifactMismatch);
            }
        }
        if let Some(expected) = source.required_calibration_digest.as_deref() {
            if evidence.as_raw().calibration_digest.as_deref() != Some(expected) {
                return Err(ObservationAdmissionError::CalibrationMismatch);
            }
        }

        Ok(AdmittedObservationEvidenceV1 {
            evidence,
            policy_id: self.config.policy_id.clone(),
            policy_generation: self.config.policy_generation,
            policy_source_digest: self.config.policy_source_digest.clone(),
        })
    }
}

/// Runtime capability proving that one exact validated observation was admitted
/// by one explicit runtime policy generation.
///
/// This capability is intentionally non-serializable, non-deserializable, and
/// non-cloneable. It is not belief support, canonical-belief authority, truth,
/// freshness proof, device authentication, or action authority.
#[derive(Debug, PartialEq, Eq)]
#[must_use = "admitted observation evidence is a runtime capability"]
pub struct AdmittedObservationEvidenceV1 {
    evidence: ValidatedObservationEvidenceRefV1,
    policy_id: String,
    policy_generation: NonZeroU64,
    policy_source_digest: String,
}

impl AdmittedObservationEvidenceV1 {
    pub fn evidence(&self) -> &ValidatedObservationEvidenceRefV1 {
        &self.evidence
    }

    pub fn policy_id(&self) -> &str {
        &self.policy_id
    }

    pub const fn policy_generation(&self) -> NonZeroU64 {
        self.policy_generation
    }

    pub fn policy_source_digest(&self) -> &str {
        &self.policy_source_digest
    }

    pub fn into_evidence(self) -> ValidatedObservationEvidenceRefV1 {
        self.evidence
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObservationEvidenceError {
    UnsupportedSchemaVersion {
        found: u16,
    },
    MissingSourceId,
    MissingObservationId,
    MissingModality,
    MissingPayloadFormat,
    EmptySourceRevision,
    NonCanonicalText {
        field: &'static str,
    },
    MissingModelId {
        origin: ObservationOriginV1,
    },
    MissingModelVersion {
        origin: ObservationOriginV1,
    },
    ExternalRecordCarriesModelIdentity,
    MalformedDigest,
}

impl std::fmt::Display for ObservationEvidenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => write!(
                f,
                "unsupported observation evidence schema version {found}; expected {OBSERVATION_EVIDENCE_SCHEMA_VERSION}"
            ),
            Self::MissingSourceId => f.write_str("observation source id must be explicit"),
            Self::MissingObservationId => {
                f.write_str("observation id must be explicit within the source generation")
            }
            Self::MissingModality => f.write_str("observation modality must be explicit"),
            Self::MissingPayloadFormat => {
                f.write_str("observation payload format identity must be explicit")
            }
            Self::EmptySourceRevision => {
                f.write_str("source revision must be non-empty when supplied")
            }
            Self::NonCanonicalText { field } => write!(
                f,
                "{field} must not contain leading or trailing whitespace"
            ),
            Self::MissingModelId { origin } => {
                write!(f, "{origin:?} requires an explicit model/generator id")
            }
            Self::MissingModelVersion { origin } => {
                write!(f, "{origin:?} requires an explicit model/generator version")
            }
            Self::ExternalRecordCarriesModelIdentity => f.write_str(
                "direct external observation records must not carry simulation/model identity",
            ),
            Self::MalformedDigest => f.write_str(
                "digest must be canonical sha256:<64 lowercase hex> or blake3:<64 lowercase hex>",
            ),
        }
    }
}

impl std::error::Error for ObservationEvidenceError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObservationPolicyError {
    UnsupportedSchemaVersion {
        found: u16,
    },
    MissingPolicyId,
    NonCanonicalPolicyText {
        field: &'static str,
    },
    MalformedPolicyDigest,
    EmptyTrustedSources,
    EmptyTrustedSourceId,
    EmptyRequiredSourceRevision,
    MalformedRequiredDigest,
    EmptyPermittedFormats,
    EmptyPermittedModality,
    EmptyPermittedPayloadFormat,
    DuplicateTrustedSourceGeneration,
    DuplicatePermittedFormat,
}

impl std::fmt::Display for ObservationPolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => write!(
                f,
                "unsupported observation admission policy schema version {found}; expected {OBSERVATION_EVIDENCE_SCHEMA_VERSION}"
            ),
            Self::MissingPolicyId => f.write_str("observation admission policy id must be explicit"),
            Self::NonCanonicalPolicyText { field } => write!(
                f,
                "policy field {field} must not contain leading or trailing whitespace"
            ),
            Self::MalformedPolicyDigest => f.write_str(
                "observation admission policy source digest must use canonical lowercase sha256/blake3 syntax",
            ),
            Self::EmptyTrustedSources => {
                f.write_str("observation admission policy requires at least one trusted source")
            }
            Self::EmptyTrustedSourceId => {
                f.write_str("trusted observation source id must be explicit")
            }
            Self::EmptyRequiredSourceRevision => {
                f.write_str("required source revision must be non-empty when supplied")
            }
            Self::MalformedRequiredDigest => f.write_str(
                "required source/calibration digest must use canonical lowercase sha256/blake3 syntax",
            ),
            Self::EmptyPermittedFormats => {
                f.write_str("trusted observation source requires at least one permitted format")
            }
            Self::EmptyPermittedModality => {
                f.write_str("permitted observation modality must be non-empty")
            }
            Self::EmptyPermittedPayloadFormat => {
                f.write_str("permitted payload format must be non-empty")
            }
            Self::DuplicateTrustedSourceGeneration => {
                f.write_str("observation policy contains duplicate source id/generation entries")
            }
            Self::DuplicatePermittedFormat => f.write_str(
                "trusted observation source contains duplicate modality/payload-format pairs",
            ),
        }
    }
}

impl std::error::Error for ObservationPolicyError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationAdmissionError {
    ReservedUse {
        requested: ObservationEvidenceUseV1,
    },
    NonExternalOrigin {
        found: ObservationOriginV1,
    },
    UntrustedSourceGeneration,
    ObservationFormatNotPermitted,
    SourceRevisionMismatch,
    SourceArtifactMismatch,
    CalibrationMismatch,
}

impl std::fmt::Display for ObservationAdmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReservedUse { requested } => {
                write!(f, "observation admission policy cannot grant {requested:?}")
            }
            Self::NonExternalOrigin { found } => {
                write!(f, "{found:?} cannot be admitted as a direct external observation")
            }
            Self::UntrustedSourceGeneration => {
                f.write_str("observation source id/generation is not trusted by this policy")
            }
            Self::ObservationFormatNotPermitted => f.write_str(
                "observation modality/payload-format pair is not permitted for this source generation",
            ),
            Self::SourceRevisionMismatch => {
                f.write_str("observation source revision does not match policy")
            }
            Self::SourceArtifactMismatch => {
                f.write_str("observation source artifact digest does not match policy")
            }
            Self::CalibrationMismatch => {
                f.write_str("observation calibration digest does not match policy")
            }
        }
    }
}

impl std::error::Error for ObservationAdmissionError {}

fn validate_policy_config(
    config: &ObservationAdmissionPolicyConfigV1,
) -> Result<(), ObservationPolicyError> {
    if config.schema_version != OBSERVATION_EVIDENCE_SCHEMA_VERSION {
        return Err(ObservationPolicyError::UnsupportedSchemaVersion {
            found: config.schema_version,
        });
    }
    if config.policy_id.trim().is_empty() {
        return Err(ObservationPolicyError::MissingPolicyId);
    }
    validate_policy_canonical_text("policy_id", &config.policy_id)?;
    validate_digest(&config.policy_source_digest)
        .map_err(|_| ObservationPolicyError::MalformedPolicyDigest)?;
    if config.trusted_sources.is_empty() {
        return Err(ObservationPolicyError::EmptyTrustedSources);
    }

    let mut source_generations = HashSet::new();
    for source in &config.trusted_sources {
        if source.source_id.trim().is_empty() {
            return Err(ObservationPolicyError::EmptyTrustedSourceId);
        }
        validate_policy_canonical_text("trusted_source.source_id", &source.source_id)?;
        if !source_generations.insert((source.source_id.as_str(), source.source_generation)) {
            return Err(ObservationPolicyError::DuplicateTrustedSourceGeneration);
        }
        if source
            .required_source_revision
            .as_deref()
            .is_some_and(|revision| revision.trim().is_empty())
        {
            return Err(ObservationPolicyError::EmptyRequiredSourceRevision);
        }
        if let Some(revision) = source.required_source_revision.as_deref() {
            validate_policy_canonical_text("required_source_revision", revision)?;
        }
        for digest in [
            source.required_source_artifact_digest.as_deref(),
            source.required_calibration_digest.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            validate_digest(digest)
                .map_err(|_| ObservationPolicyError::MalformedRequiredDigest)?;
        }
        if source.permitted_formats.is_empty() {
            return Err(ObservationPolicyError::EmptyPermittedFormats);
        }
        let mut formats = HashSet::new();
        for format in &source.permitted_formats {
            if format.modality.trim().is_empty() {
                return Err(ObservationPolicyError::EmptyPermittedModality);
            }
            if format.payload_format.trim().is_empty() {
                return Err(ObservationPolicyError::EmptyPermittedPayloadFormat);
            }
            validate_policy_canonical_text("permitted_format.modality", &format.modality)?;
            validate_policy_canonical_text(
                "permitted_format.payload_format",
                &format.payload_format,
            )?;
            if !formats.insert((format.modality.as_str(), format.payload_format.as_str())) {
                return Err(ObservationPolicyError::DuplicatePermittedFormat);
            }
        }
    }

    Ok(())
}

fn require_nonempty(
    value: &str,
    error: ObservationEvidenceError,
) -> Result<(), ObservationEvidenceError> {
    if value.trim().is_empty() {
        Err(error)
    } else {
        Ok(())
    }
}

fn require_canonical_text(
    field: &'static str,
    value: &str,
) -> Result<(), ObservationEvidenceError> {
    if value == value.trim() {
        Ok(())
    } else {
        Err(ObservationEvidenceError::NonCanonicalText { field })
    }
}

fn validate_optional_canonical_text(
    field: &'static str,
    value: Option<&str>,
) -> Result<(), ObservationEvidenceError> {
    if let Some(value) = value {
        require_canonical_text(field, value)?;
    }
    Ok(())
}

fn validate_policy_canonical_text(
    field: &'static str,
    value: &str,
) -> Result<(), ObservationPolicyError> {
    if value == value.trim() {
        Ok(())
    } else {
        Err(ObservationPolicyError::NonCanonicalPolicyText { field })
    }
}

fn require_optional_nonempty(
    value: Option<&str>,
    error: ObservationEvidenceError,
) -> Result<(), ObservationEvidenceError> {
    match value {
        Some(value) if !value.trim().is_empty() => Ok(()),
        _ => Err(error),
    }
}

fn validate_optional_nonempty(
    value: Option<&str>,
    error: ObservationEvidenceError,
) -> Result<(), ObservationEvidenceError> {
    match value {
        Some(value) if value.trim().is_empty() => Err(error),
        _ => Ok(()),
    }
}

fn validate_optional_digest(value: Option<&str>) -> Result<(), ObservationEvidenceError> {
    if let Some(value) = value {
        validate_digest(value)?;
    }
    Ok(())
}

fn validate_digest(digest: &str) -> Result<(), ObservationEvidenceError> {
    let Some((algorithm, hex)) = digest.split_once(':') else {
        return Err(ObservationEvidenceError::MalformedDigest);
    };
    if !matches!(algorithm, "sha256" | "blake3")
        || hex.len() != 64
        || !hex
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Err(ObservationEvidenceError::MalformedDigest);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA_A: &str =
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str =
        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const BLAKE_C: &str =
        "blake3:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const RGBD_FORMAT: &str = "rgbd:rectified-optical-z-f64le:v1";
    const DEPTH_FORMAT: &str = "depth:optical-z-f64le:v1";

    fn external() -> ObservationEvidenceRefV1 {
        ObservationEvidenceRefV1 {
            schema_version: OBSERVATION_EVIDENCE_SCHEMA_VERSION,
            origin: ObservationOriginV1::ExternalObservation,
            source_id: "camera:front-left".into(),
            source_generation: NonZeroU64::new(7).unwrap(),
            source_revision: Some("firmware:4.2.1".into()),
            observation_id: "frame:0000042".into(),
            modality: "rgbd".into(),
            payload_format: RGBD_FORMAT.into(),
            claim_digest: SHA_A.into(),
            source_artifact_digest: Some(SHA_B.into()),
            calibration_digest: Some(BLAKE_C.into()),
            model_id: None,
            model_version: None,
            freshness_epoch: Some(91),
        }
    }

    fn simulated() -> ObservationEvidenceRefV1 {
        ObservationEvidenceRefV1 {
            origin: ObservationOriginV1::SimulatedObservation,
            source_id: "simulator:symworld".into(),
            source_generation: NonZeroU64::new(2).unwrap(),
            source_revision: Some("git:abc123".into()),
            observation_id: "render:42".into(),
            modality: "rgbd".into(),
            payload_format: RGBD_FORMAT.into(),
            model_id: Some("symworld-renderer".into()),
            model_version: Some("2".into()),
            ..external()
        }
    }

    fn fixture() -> ObservationEvidenceRefV1 {
        ObservationEvidenceRefV1 {
            origin: ObservationOriginV1::SyntheticFixture,
            source_id: "test:observation-fixture".into(),
            source_generation: NonZeroU64::new(1).unwrap(),
            source_revision: None,
            observation_id: "fixture:1".into(),
            modality: "depth".into(),
            payload_format: DEPTH_FORMAT.into(),
            source_artifact_digest: None,
            calibration_digest: None,
            model_id: Some("fixture-generator".into()),
            model_version: Some("1".into()),
            freshness_epoch: None,
            ..external()
        }
    }

    fn permitted(modality: &str, payload_format: &str) -> PermittedObservationFormatV1 {
        PermittedObservationFormatV1 {
            modality: modality.into(),
            payload_format: payload_format.into(),
        }
    }

    fn policy_config() -> ObservationAdmissionPolicyConfigV1 {
        ObservationAdmissionPolicyConfigV1 {
            schema_version: OBSERVATION_EVIDENCE_SCHEMA_VERSION,
            policy_id: "lab-camera-admission".into(),
            policy_generation: NonZeroU64::new(3).unwrap(),
            policy_source_digest: SHA_A.into(),
            trusted_sources: vec![TrustedObservationSourceV1 {
                source_id: "camera:front-left".into(),
                source_generation: NonZeroU64::new(7).unwrap(),
                required_source_revision: Some("firmware:4.2.1".into()),
                required_source_artifact_digest: Some(SHA_B.into()),
                required_calibration_digest: Some(BLAKE_C.into()),
                permitted_formats: vec![
                    permitted("rgb", "rgb:srgb8-row-major:v1"),
                    permitted("rgbd", RGBD_FORMAT),
                    permitted("depth", DEPTH_FORMAT),
                ],
            }],
        }
    }

    fn policy() -> ObservationAdmissionPolicyV1 {
        policy_config().activate_as_trust_root().unwrap()
    }

    #[test]
    fn self_labeled_external_record_cannot_mint_admission_without_policy() {
        let validated = external().validate().unwrap();
        assert_eq!(validated.origin(), ObservationOriginV1::ExternalObservation);
        assert_eq!(validated.source_id(), "camera:front-left");
        assert_eq!(validated.observation_id(), "frame:0000042");
        assert_eq!(validated.modality(), "rgbd");
        assert_eq!(validated.payload_format(), RGBD_FORMAT);
        assert_eq!(validated.claim_digest(), SHA_A);
    }

    #[test]
    fn exact_trusted_source_generation_can_earn_observation_admission() {
        let admitted = policy()
            .authorize(
                external().validate().unwrap(),
                ObservationEvidenceUseV1::ObservationAdmission,
            )
            .unwrap();
        assert_eq!(admitted.policy_id(), "lab-camera-admission");
        assert_eq!(admitted.policy_generation(), NonZeroU64::new(3).unwrap());
        assert_eq!(admitted.policy_source_digest(), SHA_A);
        assert_eq!(admitted.evidence().claim_digest(), SHA_A);
        assert_eq!(admitted.evidence().payload_format(), RGBD_FORMAT);
    }

    #[test]
    fn simulated_and_fixture_records_cannot_masquerade_as_direct_observations() {
        let policy = policy();
        for raw in [fixture(), simulated()] {
            let err = policy
                .authorize(
                    raw.validate().unwrap(),
                    ObservationEvidenceUseV1::ObservationAdmission,
                )
                .unwrap_err();
            assert!(matches!(
                err,
                ObservationAdmissionError::NonExternalOrigin { .. }
            ));
        }
    }

    #[test]
    fn observation_policy_cannot_grant_belief_or_action_transitions() {
        let policy = policy();
        for use_case in [
            ObservationEvidenceUseV1::BeliefSupport,
            ObservationEvidenceUseV1::CanonicalBeliefAdmission,
            ObservationEvidenceUseV1::ActionAuthority,
        ] {
            assert!(matches!(
                policy.authorize(external().validate().unwrap(), use_case),
                Err(ObservationAdmissionError::ReservedUse { .. })
            ));
        }
    }

    #[test]
    fn source_generation_reset_breaks_admission_even_with_reused_local_id() {
        let mut raw = external();
        raw.source_generation = NonZeroU64::new(8).unwrap();
        assert_eq!(
            policy()
                .authorize(
                    raw.validate().unwrap(),
                    ObservationEvidenceUseV1::ObservationAdmission,
                )
                .unwrap_err(),
            ObservationAdmissionError::UntrustedSourceGeneration
        );
    }

    #[test]
    fn policy_binds_exact_modality_payload_pair_revision_artifact_and_calibration() {
        let mut raw = external();
        raw.modality = "depth".into();
        assert_eq!(
            policy()
                .authorize(
                    raw.validate().unwrap(),
                    ObservationEvidenceUseV1::ObservationAdmission,
                )
                .unwrap_err(),
            ObservationAdmissionError::ObservationFormatNotPermitted
        );

        let mut raw = external();
        raw.payload_format = DEPTH_FORMAT.into();
        assert_eq!(
            policy()
                .authorize(
                    raw.validate().unwrap(),
                    ObservationEvidenceUseV1::ObservationAdmission,
                )
                .unwrap_err(),
            ObservationAdmissionError::ObservationFormatNotPermitted
        );

        let mut raw = external();
        raw.source_revision = Some("firmware:4.3.0".into());
        assert_eq!(
            policy()
                .authorize(
                    raw.validate().unwrap(),
                    ObservationEvidenceUseV1::ObservationAdmission,
                )
                .unwrap_err(),
            ObservationAdmissionError::SourceRevisionMismatch
        );

        let mut raw = external();
        raw.source_artifact_digest = Some(BLAKE_C.into());
        assert_eq!(
            policy()
                .authorize(
                    raw.validate().unwrap(),
                    ObservationEvidenceUseV1::ObservationAdmission,
                )
                .unwrap_err(),
            ObservationAdmissionError::SourceArtifactMismatch
        );

        let mut raw = external();
        raw.calibration_digest = Some(SHA_B.into());
        assert_eq!(
            policy()
                .authorize(
                    raw.validate().unwrap(),
                    ObservationEvidenceUseV1::ObservationAdmission,
                )
                .unwrap_err(),
            ObservationAdmissionError::CalibrationMismatch
        );
    }

    #[test]
    fn synthetic_and_simulated_records_require_explicit_model_identity() {
        for mut raw in [fixture(), simulated()] {
            raw.model_id = None;
            assert!(matches!(
                raw.validate(),
                Err(ObservationEvidenceError::MissingModelId { .. })
            ));
        }

        for mut raw in [fixture(), simulated()] {
            raw.model_version = None;
            assert!(matches!(
                raw.validate(),
                Err(ObservationEvidenceError::MissingModelVersion { .. })
            ));
        }
    }

    #[test]
    fn external_record_rejects_simulation_model_identity() {
        let mut raw = external();
        raw.model_id = Some("prediction-model".into());
        raw.model_version = Some("1".into());
        assert_eq!(
            raw.validate(),
            Err(ObservationEvidenceError::ExternalRecordCarriesModelIdentity)
        );
    }

    #[test]
    fn digests_use_one_canonical_lowercase_text_identity() {
        for bad in [
            "sha256:abc",
            "sha512:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "sha256:zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
            "sha256:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            " sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ",
        ] {
            let mut raw = external();
            raw.claim_digest = bad.into();
            assert_eq!(raw.validate(), Err(ObservationEvidenceError::MalformedDigest));
        }

        let mut raw = external();
        raw.calibration_digest = Some("decorative:calibration".into());
        assert_eq!(raw.validate(), Err(ObservationEvidenceError::MalformedDigest));

        let mut config = policy_config();
        config.policy_source_digest =
            "sha256:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA".into();
        assert_eq!(
            config.activate_as_trust_root(),
            Err(ObservationPolicyError::MalformedPolicyDigest)
        );
    }

    #[test]
    fn identity_and_format_text_reject_whitespace_aliases() {
        for field in ["source_id", "observation_id", "modality", "payload_format"] {
            let mut raw = external();
            match field {
                "source_id" => raw.source_id = " camera:front-left".into(),
                "observation_id" => raw.observation_id = "frame:0000042 ".into(),
                "modality" => raw.modality = " rgbd".into(),
                "payload_format" => raw.payload_format = format!("{RGBD_FORMAT} "),
                _ => unreachable!(),
            }
            assert!(matches!(
                raw.validate(),
                Err(ObservationEvidenceError::NonCanonicalText { .. })
            ));
        }

        let mut config = policy_config();
        config.trusted_sources[0].source_id = "camera:front-left ".into();
        assert!(matches!(
            config.activate_as_trust_root(),
            Err(ObservationPolicyError::NonCanonicalPolicyText { .. })
        ));
    }

    #[test]
    fn changed_observation_content_changes_claim_identity() {
        let a = external().validate().unwrap();
        let mut raw_b = external();
        raw_b.claim_digest = SHA_B.into();
        let b = raw_b.validate().unwrap();

        assert_ne!(a.claim_digest(), b.claim_digest());
        assert_eq!(a.source_id(), b.source_id());
        assert_eq!(a.observation_id(), b.observation_id());
    }

    #[test]
    fn changed_payload_format_changes_observation_meaning_and_policy_match() {
        let a = external().validate().unwrap();
        let mut raw_b = external();
        raw_b.payload_format = "rgbd:rectified-optical-z-f32le:v1".into();
        let b = raw_b.validate().unwrap();
        assert_ne!(a.payload_format(), b.payload_format());
        assert_eq!(a.claim_digest(), b.claim_digest());
        assert_eq!(
            policy()
                .authorize(b, ObservationEvidenceUseV1::ObservationAdmission)
                .unwrap_err(),
            ObservationAdmissionError::ObservationFormatNotPermitted
        );
    }

    #[test]
    fn policy_validation_rejects_ambiguous_or_malformed_trust_sets() {
        let mut duplicate_source = policy_config();
        duplicate_source
            .trusted_sources
            .push(duplicate_source.trusted_sources[0].clone());
        assert_eq!(
            duplicate_source.activate_as_trust_root(),
            Err(ObservationPolicyError::DuplicateTrustedSourceGeneration)
        );

        let mut duplicate_format = policy_config();
        let duplicate = duplicate_format.trusted_sources[0].permitted_formats[0].clone();
        duplicate_format.trusted_sources[0]
            .permitted_formats
            .push(duplicate);
        assert_eq!(
            duplicate_format.activate_as_trust_root(),
            Err(ObservationPolicyError::DuplicatePermittedFormat)
        );

        let mut malformed_digest = policy_config();
        malformed_digest.policy_source_digest = "sha256:bad".into();
        assert_eq!(
            malformed_digest.activate_as_trust_root(),
            Err(ObservationPolicyError::MalformedPolicyDigest)
        );
    }

    #[test]
    fn validated_reference_revalidates_on_deserialization() {
        let mut raw = external();
        raw.claim_digest = "sha256:bad".into();
        let json = serde_json::to_string(&raw).unwrap();
        assert!(serde_json::from_str::<ValidatedObservationEvidenceRefV1>(&json).is_err());
    }

    #[test]
    fn unknown_fields_and_schema_versions_fail_closed() {
        let json = serde_json::to_value(external()).unwrap();
        let mut object = json.as_object().unwrap().clone();
        object.insert("self_authorized".into(), serde_json::json!(true));
        assert!(
            serde_json::from_value::<ObservationEvidenceRefV1>(serde_json::Value::Object(object))
                .is_err()
        );

        let mut raw = external();
        raw.schema_version += 1;
        assert_eq!(
            raw.validate(),
            Err(ObservationEvidenceError::UnsupportedSchemaVersion { found: 2 })
        );
    }

    #[test]
    fn freshness_epoch_is_metadata_not_currentness_proof() {
        let mut raw = external();
        raw.freshness_epoch = None;
        assert!(
            policy()
                .authorize(
                    raw.validate().unwrap(),
                    ObservationEvidenceUseV1::ObservationAdmission,
                )
                .is_ok()
        );
    }

    #[test]
    fn modality_and_payload_format_vocabularies_remain_generic() {
        for (modality, payload_format) in [
            ("rgb", "rgb:srgb8-row-major:v1"),
            ("rgbd", RGBD_FORMAT),
            ("lidar", "lidar:xyz-f32le:v1"),
            ("event_camera", "events:xytp-packed:v1"),
            ("imu", "imu:accel-gyro-f64le:v1"),
            ("audio", "audio:pcm-s16le-mono:v1"),
            ("tactile", "tactile:pressure-f32le:v1"),
        ] {
            let mut raw = external();
            raw.modality = modality.into();
            raw.payload_format = payload_format.into();
            let validated = raw.validate().unwrap();
            assert_eq!(validated.modality(), modality);
            assert_eq!(validated.payload_format(), payload_format);
        }
    }
}
