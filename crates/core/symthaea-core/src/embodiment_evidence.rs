// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence primitives for Embodiment Contract v2 migration.
//!
//! This module is deliberately behavior-neutral. It gives embodiment, simulator,
//! transport, and HIL code a shared vocabulary for saying **what is known**, **how
//! it is known**, and **which clock produced the observation** without changing the
//! legacy [`crate::embodiment::EmbodimentBridge`] numerical compatibility surface.
//!
//! The central epistemic rule is that unavailable data, compatibility assumptions,
//! measured observations, estimates, derived metrics, and simulator truth are
//! different propositions. A finite number does not become evidence merely because
//! it is convenient for downstream arithmetic.

use serde::{Deserialize, Serialize};

/// Schema version emitted by the v1 evidence records in this module.
pub const EMBODIMENT_EVIDENCE_SCHEMA_V1: u16 = 1;

/// Whether an evidence value is currently usable as reported by its producer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvidenceAvailability {
    /// A value is present and the producer considers it current under its profile.
    Available,
    /// A prior value is retained for diagnosis/replay but is no longer current.
    Stale,
    /// No value is established.
    Unavailable,
}

/// How an evidence value was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvidenceSourceClass {
    /// Direct measurement from a sensor, device, or physical telemetry source.
    Measured,
    /// State estimated from measurements by an estimator/filter/model.
    Estimated,
    /// Metric deterministically derived from other evidence (for example a residual).
    Derived,
    /// Privileged simulator state unavailable as equivalent truth on real hardware.
    SimulatorTruth,
    /// Static/profile value declared as nominal rather than observed at runtime.
    DeclaredNominal,
    /// Compatibility prior retained only to preserve legacy numerical behavior.
    CompatibilityAssumption,
}

impl EvidenceSourceClass {
    /// Whether this source class represents runtime evidence that must carry a timestamp.
    pub const fn requires_runtime_timestamp(self) -> bool {
        matches!(
            self,
            Self::Measured | Self::Estimated | Self::Derived | Self::SimulatorTruth
        )
    }

    /// Whether this source is explicitly an assumption rather than runtime evidence.
    pub const fn is_assumption(self) -> bool {
        matches!(
            self,
            Self::DeclaredNominal | Self::CompatibilityAssumption
        )
    }
}

/// Stable identifier for one clock/time domain.
///
/// The numeric timestamp carried by [`TimestampV1`] is interpreted only inside this
/// domain. The identifier does **not** imply Unix epoch time, wall-clock time, or
/// synchronization with any other domain. Examples could include host monotonic time,
/// simulator model time, or an autopilot boot clock.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClockDomainId(String);

impl ClockDomainId {
    /// Construct a validated clock-domain identifier.
    pub fn new(id: impl Into<String>) -> Result<Self, EvidenceValidationError> {
        let value = Self(id.into());
        value.validate()?;
        Ok(value)
    }

    /// Borrow the canonical identifier string.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Validate that the identifier is non-empty, trimmed, and contains no controls.
    pub fn validate(&self) -> Result<(), EvidenceValidationError> {
        let trimmed = self.0.trim();
        if trimmed.is_empty()
            || trimmed.len() != self.0.len()
            || self.0.chars().any(char::is_control)
        {
            return Err(EvidenceValidationError::InvalidClockDomain);
        }
        Ok(())
    }
}

impl std::fmt::Display for ClockDomainId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Timestamp in nanoseconds relative to an explicitly named clock-domain origin.
///
/// `nanoseconds` is intentionally **not** called epoch time. The origin and any
/// cross-clock synchronization belong to the clock/profile evidence, not this scalar.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TimestampV1 {
    /// Clock domain in which `nanoseconds` is meaningful.
    pub clock_domain: ClockDomainId,
    /// Nanoseconds since that domain's declared origin.
    pub nanoseconds: u64,
}

impl TimestampV1 {
    /// Construct a timestamp in a validated clock domain.
    pub fn new(clock_domain: ClockDomainId, nanoseconds: u64) -> Self {
        Self {
            clock_domain,
            nanoseconds,
        }
    }

    /// Validate the timestamp and its clock-domain identifier.
    pub fn validate(&self) -> Result<(), EvidenceValidationError> {
        self.clock_domain.validate()
    }

    /// Return elapsed nanoseconds since an earlier timestamp in the same clock domain.
    ///
    /// Cross-domain subtraction is rejected rather than silently assuming clock
    /// synchronization. A timestamp that moves backward in the same domain is also
    /// rejected rather than wrapping into a plausible duration.
    pub fn elapsed_since(&self, earlier: &Self) -> Result<u64, EvidenceValidationError> {
        self.validate()?;
        earlier.validate()?;
        if self.clock_domain != earlier.clock_domain {
            return Err(EvidenceValidationError::ClockDomainMismatch);
        }
        self.nanoseconds
            .checked_sub(earlier.nanoseconds)
            .ok_or(EvidenceValidationError::NonMonotonicTimestamp)
    }
}

/// Shared metadata attached to normalized embodiment evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceMetadataV1 {
    /// Evidence schema version. Must equal [`EMBODIMENT_EVIDENCE_SCHEMA_V1`].
    pub schema_version: u16,
    /// Current availability state of the value.
    pub availability: EvidenceAvailability,
    /// Source class when a value exists; absent when unavailable.
    pub source: Option<EvidenceSourceClass>,
    /// Source/derivation timestamp in its explicit clock domain.
    pub observed_at: Option<TimestampV1>,
    /// Optional maximum validity duration declared by the producer/profile.
    pub valid_for_ns: Option<u64>,
    /// Optional provenance/evidence reference. Cryptographic semantics are external.
    pub provenance_id: Option<String>,
    /// Optional calibration/normalization/profile reference.
    pub profile_id: Option<String>,
}

impl EvidenceMetadataV1 {
    /// Construct explicit unavailable metadata with no invented source or value.
    pub fn unavailable() -> Self {
        Self {
            schema_version: EMBODIMENT_EVIDENCE_SCHEMA_V1,
            availability: EvidenceAvailability::Unavailable,
            source: None,
            observed_at: None,
            valid_for_ns: None,
            provenance_id: None,
            profile_id: None,
        }
    }

    /// Construct available metadata and validate source/timestamp semantics.
    pub fn available(
        source: EvidenceSourceClass,
        observed_at: Option<TimestampV1>,
    ) -> Result<Self, EvidenceValidationError> {
        let value = Self {
            schema_version: EMBODIMENT_EVIDENCE_SCHEMA_V1,
            availability: EvidenceAvailability::Available,
            source: Some(source),
            observed_at,
            valid_for_ns: None,
            provenance_id: None,
            profile_id: None,
        };
        value.validate()?;
        Ok(value)
    }

    /// Construct stale runtime metadata while retaining its original timestamp.
    pub fn stale(
        source: EvidenceSourceClass,
        observed_at: TimestampV1,
    ) -> Result<Self, EvidenceValidationError> {
        let value = Self {
            schema_version: EMBODIMENT_EVIDENCE_SCHEMA_V1,
            availability: EvidenceAvailability::Stale,
            source: Some(source),
            observed_at: Some(observed_at),
            valid_for_ns: None,
            provenance_id: None,
            profile_id: None,
        };
        value.validate()?;
        Ok(value)
    }

    /// Validate source, availability, clock, validity-window, and identifier invariants.
    pub fn validate(&self) -> Result<(), EvidenceValidationError> {
        if self.schema_version != EMBODIMENT_EVIDENCE_SCHEMA_V1 {
            return Err(EvidenceValidationError::UnsupportedSchemaVersion {
                found: self.schema_version,
            });
        }

        if let Some(timestamp) = &self.observed_at {
            timestamp.validate()?;
        }
        if self.valid_for_ns == Some(0) {
            return Err(EvidenceValidationError::ZeroValidityWindow);
        }
        if self.valid_for_ns.is_some() && self.observed_at.is_none() {
            return Err(EvidenceValidationError::ValidityWindowRequiresTimestamp);
        }
        validate_optional_identifier(&self.provenance_id, "provenance_id")?;
        validate_optional_identifier(&self.profile_id, "profile_id")?;

        match self.availability {
            EvidenceAvailability::Unavailable => {
                if self.source.is_some() {
                    return Err(EvidenceValidationError::SourceForbiddenForUnavailable);
                }
            }
            EvidenceAvailability::Available | EvidenceAvailability::Stale => {
                let source = self.source.ok_or(EvidenceValidationError::SourceRequired)?;
                if source.requires_runtime_timestamp() && self.observed_at.is_none() {
                    return Err(EvidenceValidationError::RuntimeTimestampRequired(source));
                }
                if self.availability == EvidenceAvailability::Stale
                    && !source.requires_runtime_timestamp()
                {
                    return Err(EvidenceValidationError::StaleRequiresRuntimeSource);
                }
            }
        }

        Ok(())
    }

    /// Evaluate producer-declared availability at `now` when a validity window exists.
    ///
    /// This method never compares timestamps from different clock domains. If no
    /// validity window is declared, the producer's current availability classification
    /// is returned unchanged. Consumers may still impose stricter local freshness
    /// policies above this primitive.
    pub fn effective_availability_at(
        &self,
        now: &TimestampV1,
    ) -> Result<EvidenceAvailability, EvidenceValidationError> {
        self.validate()?;
        if self.availability != EvidenceAvailability::Available {
            return Ok(self.availability);
        }
        let Some(valid_for_ns) = self.valid_for_ns else {
            return Ok(EvidenceAvailability::Available);
        };
        let observed_at = self
            .observed_at
            .as_ref()
            .ok_or(EvidenceValidationError::ValidityWindowRequiresTimestamp)?;
        let age_ns = now.elapsed_since(observed_at)?;
        if age_ns >= valid_for_ns {
            Ok(EvidenceAvailability::Stale)
        } else {
            Ok(EvidenceAvailability::Available)
        }
    }
}

/// Cardinality required by an embodiment evidence semantic kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvidenceCardinality {
    /// Exactly one normalized scalar value.
    Scalar,
    /// One or more normalized values whose positions are defined by a profile.
    Vector,
}

/// Semantic meaning of a normalized embodiment evidence value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmbodimentEvidenceKind {
    /// Per-actuator health/reliability vector. This kind is vector-valued.
    ActuatorHealth,
    /// Qualified sensorimotor accuracy score.
    SensorimotorAccuracy,
    /// Qualified physical reliability score for a declared subsystem/profile.
    PhysicalReliability,
    /// Error between a model prediction and the resulting observation.
    PredictiveResidual,
    /// Inter-frame or inter-representation change/novelty.
    TemporalStateNovelty,
    /// Error between an actual state/output and a target/setpoint.
    TaskTrackingError,
    /// Qualified model-versus-plant divergence under a declared profile.
    ModelDivergence,
    /// Confidence/quality of an observation source or estimate.
    ObservationConfidence,
}

impl EmbodimentEvidenceKind {
    /// Required scalar/vector cardinality for this semantic kind.
    pub const fn cardinality(self) -> EvidenceCardinality {
        match self {
            Self::ActuatorHealth => EvidenceCardinality::Vector,
            Self::SensorimotorAccuracy
            | Self::PhysicalReliability
            | Self::PredictiveResidual
            | Self::TemporalStateNovelty
            | Self::TaskTrackingError
            | Self::ModelDivergence
            | Self::ObservationConfidence => EvidenceCardinality::Scalar,
        }
    }
}

/// One normalized scalar evidence value in the inclusive range `[0, 1]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalizedScalarEvidenceV1 {
    /// Semantic meaning of this scalar.
    pub kind: EmbodimentEvidenceKind,
    /// Value when available/stale; `None` when explicitly unavailable.
    pub value: Option<f32>,
    /// Evidence source/availability/time metadata.
    pub metadata: EvidenceMetadataV1,
}

impl NormalizedScalarEvidenceV1 {
    /// Construct explicit unavailable scalar evidence for a scalar semantic kind.
    pub fn unavailable(kind: EmbodimentEvidenceKind) -> Result<Self, EvidenceValidationError> {
        let value = Self {
            kind,
            value: None,
            metadata: EvidenceMetadataV1::unavailable(),
        };
        value.validate()?;
        Ok(value)
    }

    /// Construct and validate a normalized scalar evidence value.
    pub fn new(
        kind: EmbodimentEvidenceKind,
        value: Option<f32>,
        metadata: EvidenceMetadataV1,
    ) -> Result<Self, EvidenceValidationError> {
        let evidence = Self {
            kind,
            value,
            metadata,
        };
        evidence.validate()?;
        Ok(evidence)
    }

    /// Validate semantic cardinality, availability, and normalized value range.
    pub fn validate(&self) -> Result<(), EvidenceValidationError> {
        self.metadata.validate()?;
        if self.kind.cardinality() != EvidenceCardinality::Scalar {
            return Err(EvidenceValidationError::CardinalityMismatch {
                kind: self.kind,
                expected: EvidenceCardinality::Scalar,
            });
        }
        validate_scalar_payload(self.value, self.metadata.availability)
    }
}

/// One normalized vector evidence value with every element in `[0, 1]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalizedVectorEvidenceV1 {
    /// Semantic meaning of this vector.
    pub kind: EmbodimentEvidenceKind,
    /// Values when available/stale; `None` when explicitly unavailable.
    pub values: Option<Vec<f32>>,
    /// Evidence source/availability/time metadata.
    pub metadata: EvidenceMetadataV1,
}

impl NormalizedVectorEvidenceV1 {
    /// Construct explicit unavailable vector evidence for a vector semantic kind.
    pub fn unavailable(kind: EmbodimentEvidenceKind) -> Result<Self, EvidenceValidationError> {
        let value = Self {
            kind,
            values: None,
            metadata: EvidenceMetadataV1::unavailable(),
        };
        value.validate()?;
        Ok(value)
    }

    /// Construct and validate a normalized vector evidence value.
    pub fn new(
        kind: EmbodimentEvidenceKind,
        values: Option<Vec<f32>>,
        metadata: EvidenceMetadataV1,
    ) -> Result<Self, EvidenceValidationError> {
        let evidence = Self {
            kind,
            values,
            metadata,
        };
        evidence.validate()?;
        Ok(evidence)
    }

    /// Validate semantic cardinality, profile identity, availability, and value range.
    pub fn validate(&self) -> Result<(), EvidenceValidationError> {
        self.metadata.validate()?;
        if self.kind.cardinality() != EvidenceCardinality::Vector {
            return Err(EvidenceValidationError::CardinalityMismatch {
                kind: self.kind,
                expected: EvidenceCardinality::Vector,
            });
        }

        match self.metadata.availability {
            EvidenceAvailability::Unavailable => {
                if self.values.is_some() {
                    return Err(EvidenceValidationError::ValueForbiddenForUnavailable);
                }
            }
            EvidenceAvailability::Available | EvidenceAvailability::Stale => {
                if self.metadata.profile_id.is_none() {
                    return Err(EvidenceValidationError::ProfileRequiredForVector);
                }
                let values = self
                    .values
                    .as_ref()
                    .ok_or(EvidenceValidationError::ValueRequired)?;
                if values.is_empty() {
                    return Err(EvidenceValidationError::EmptyVector);
                }
                for (index, value) in values.iter().copied().enumerate() {
                    validate_normalized_value(value, Some(index))?;
                }
            }
        }
        Ok(())
    }
}

fn validate_scalar_payload(
    value: Option<f32>,
    availability: EvidenceAvailability,
) -> Result<(), EvidenceValidationError> {
    match availability {
        EvidenceAvailability::Unavailable => {
            if value.is_some() {
                return Err(EvidenceValidationError::ValueForbiddenForUnavailable);
            }
        }
        EvidenceAvailability::Available | EvidenceAvailability::Stale => {
            let value = value.ok_or(EvidenceValidationError::ValueRequired)?;
            validate_normalized_value(value, None)?;
        }
    }
    Ok(())
}

fn validate_normalized_value(
    value: f32,
    index: Option<usize>,
) -> Result<(), EvidenceValidationError> {
    if !value.is_finite() {
        return Err(EvidenceValidationError::NonFiniteValue { index });
    }
    if !(0.0..=1.0).contains(&value) {
        return Err(EvidenceValidationError::OutOfRangeValue { index, value });
    }
    Ok(())
}

fn validate_optional_identifier(
    value: &Option<String>,
    field: &'static str,
) -> Result<(), EvidenceValidationError> {
    if let Some(value) = value {
        let trimmed = value.trim();
        if trimmed.is_empty()
            || trimmed.len() != value.len()
            || value.chars().any(char::is_control)
        {
            return Err(EvidenceValidationError::InvalidIdentifier(field));
        }
    }
    Ok(())
}

/// Validation failure for Embodiment Evidence v1 records.
#[derive(Debug, Clone, PartialEq)]
pub enum EvidenceValidationError {
    /// Record uses a schema version not understood by this v1 validator.
    UnsupportedSchemaVersion {
        /// Unsupported schema version encountered.
        found: u16,
    },
    /// Clock-domain identifier is empty, padded, or contains control characters.
    InvalidClockDomain,
    /// Timestamp arithmetic attempted across unrelated clock domains.
    ClockDomainMismatch,
    /// A later timestamp was numerically earlier in the same clock domain.
    NonMonotonicTimestamp,
    /// Optional identifier is present but invalid.
    InvalidIdentifier(&'static str),
    /// Available/stale evidence omitted its source class.
    SourceRequired,
    /// Unavailable evidence incorrectly claims a source class.
    SourceForbiddenForUnavailable,
    /// Runtime evidence omitted a source timestamp.
    RuntimeTimestampRequired(EvidenceSourceClass),
    /// Stale evidence used a non-runtime assumption source.
    StaleRequiresRuntimeSource,
    /// A validity window was declared without a source timestamp.
    ValidityWindowRequiresTimestamp,
    /// A validity window of zero nanoseconds is meaningless.
    ZeroValidityWindow,
    /// Available/stale evidence omitted its value payload.
    ValueRequired,
    /// Unavailable evidence incorrectly carried a value payload.
    ValueForbiddenForUnavailable,
    /// Available/stale vector evidence was empty.
    EmptyVector,
    /// Available/stale vector evidence omitted the profile defining vector positions.
    ProfileRequiredForVector,
    /// Scalar/vector representation does not match the semantic kind.
    CardinalityMismatch {
        /// Semantic kind being represented.
        kind: EmbodimentEvidenceKind,
        /// Cardinality required by the representation API.
        expected: EvidenceCardinality,
    },
    /// A normalized value was NaN or infinite.
    NonFiniteValue {
        /// Vector index, or `None` for scalar evidence.
        index: Option<usize>,
    },
    /// A normalized value fell outside the inclusive range `[0, 1]`.
    OutOfRangeValue {
        /// Vector index, or `None` for scalar evidence.
        index: Option<usize>,
        /// Invalid value encountered.
        value: f32,
    },
}

impl std::fmt::Display for EvidenceValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => {
                write!(f, "unsupported embodiment evidence schema version {found}")
            }
            Self::InvalidClockDomain => write!(f, "invalid clock-domain identifier"),
            Self::ClockDomainMismatch => write!(f, "cannot compare different clock domains"),
            Self::NonMonotonicTimestamp => write!(f, "timestamp moved backward within one clock domain"),
            Self::InvalidIdentifier(field) => write!(f, "invalid {field} identifier"),
            Self::SourceRequired => write!(f, "available/stale evidence requires a source class"),
            Self::SourceForbiddenForUnavailable => {
                write!(f, "unavailable evidence must not claim a source class")
            }
            Self::RuntimeTimestampRequired(source) => {
                write!(f, "runtime source {source:?} requires an explicit timestamp")
            }
            Self::StaleRequiresRuntimeSource => {
                write!(f, "stale evidence requires a timestamped runtime source")
            }
            Self::ValidityWindowRequiresTimestamp => {
                write!(f, "validity window requires an explicit source timestamp")
            }
            Self::ZeroValidityWindow => write!(f, "validity window must be greater than zero"),
            Self::ValueRequired => write!(f, "available/stale evidence requires a value"),
            Self::ValueForbiddenForUnavailable => {
                write!(f, "unavailable evidence must not carry a value")
            }
            Self::EmptyVector => write!(f, "available/stale vector evidence must not be empty"),
            Self::ProfileRequiredForVector => write!(
                f,
                "available/stale vector evidence requires a profile defining vector positions"
            ),
            Self::CardinalityMismatch { kind, expected } => {
                write!(f, "evidence kind {kind:?} does not have {expected:?} cardinality")
            }
            Self::NonFiniteValue { index: Some(index) } => {
                write!(f, "normalized value at index {index} is non-finite")
            }
            Self::NonFiniteValue { index: None } => write!(f, "normalized scalar is non-finite"),
            Self::OutOfRangeValue {
                index: Some(index),
                value,
            } => write!(
                f,
                "normalized value at index {index} is outside [0, 1]: {value}"
            ),
            Self::OutOfRangeValue { index: None, value } => {
                write!(f, "normalized scalar is outside [0, 1]: {value}")
            }
        }
    }
}

impl std::error::Error for EvidenceValidationError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn host_time(ns: u64) -> TimestampV1 {
        TimestampV1::new(ClockDomainId::new("host.monotonic").unwrap(), ns)
    }

    fn sim_time(ns: u64) -> TimestampV1 {
        TimestampV1::new(ClockDomainId::new("sim.model").unwrap(), ns)
    }

    #[test]
    fn unavailable_scalar_carries_no_value_or_source() {
        let evidence =
            NormalizedScalarEvidenceV1::unavailable(EmbodimentEvidenceKind::PhysicalReliability)
                .unwrap();
        assert_eq!(evidence.value, None);
        assert_eq!(evidence.metadata.source, None);
        assert_eq!(
            evidence.metadata.availability,
            EvidenceAvailability::Unavailable
        );
    }

    #[test]
    fn unavailable_value_is_rejected() {
        let evidence = NormalizedScalarEvidenceV1 {
            kind: EmbodimentEvidenceKind::PhysicalReliability,
            value: Some(1.0),
            metadata: EvidenceMetadataV1::unavailable(),
        };
        assert_eq!(
            evidence.validate(),
            Err(EvidenceValidationError::ValueForbiddenForUnavailable)
        );
    }

    #[test]
    fn measured_runtime_evidence_requires_timestamp() {
        assert_eq!(
            EvidenceMetadataV1::available(EvidenceSourceClass::Measured, None),
            Err(EvidenceValidationError::RuntimeTimestampRequired(
                EvidenceSourceClass::Measured
            ))
        );

        assert!(EvidenceMetadataV1::available(
            EvidenceSourceClass::Measured,
            Some(host_time(42))
        )
        .is_ok());
    }

    #[test]
    fn compatibility_assumption_is_explicit_and_may_be_timeless() {
        let metadata = EvidenceMetadataV1::available(
            EvidenceSourceClass::CompatibilityAssumption,
            None,
        )
        .unwrap();
        assert!(metadata.source.unwrap().is_assumption());
        assert_eq!(metadata.observed_at, None);
    }

    #[test]
    fn stale_assumption_is_rejected() {
        assert_eq!(
            EvidenceMetadataV1::stale(
                EvidenceSourceClass::CompatibilityAssumption,
                host_time(10)
            ),
            Err(EvidenceValidationError::StaleRequiresRuntimeSource)
        );
    }

    #[test]
    fn normalized_scalar_rejects_non_finite_and_out_of_range_values() {
        let metadata = EvidenceMetadataV1::available(
            EvidenceSourceClass::Derived,
            Some(host_time(100)),
        )
        .unwrap();
        assert!(matches!(
            NormalizedScalarEvidenceV1::new(
                EmbodimentEvidenceKind::PredictiveResidual,
                Some(f32::NAN),
                metadata.clone()
            ),
            Err(EvidenceValidationError::NonFiniteValue { index: None })
        ));
        assert!(matches!(
            NormalizedScalarEvidenceV1::new(
                EmbodimentEvidenceKind::PredictiveResidual,
                Some(1.01),
                metadata
            ),
            Err(EvidenceValidationError::OutOfRangeValue {
                index: None,
                ..
            })
        ));
    }

    #[test]
    fn actuator_health_is_vector_valued_and_non_empty_when_available() {
        let mut metadata = EvidenceMetadataV1::available(
            EvidenceSourceClass::Measured,
            Some(host_time(200)),
        )
        .unwrap();
        metadata.profile_id = Some("actuator-map.v1".into());
        assert!(NormalizedVectorEvidenceV1::new(
            EmbodimentEvidenceKind::ActuatorHealth,
            Some(vec![1.0, 0.75, 0.5]),
            metadata.clone()
        )
        .is_ok());
        assert_eq!(
            NormalizedVectorEvidenceV1::new(
                EmbodimentEvidenceKind::ActuatorHealth,
                Some(Vec::new()),
                metadata
            ),
            Err(EvidenceValidationError::EmptyVector)
        );
    }

    #[test]
    fn actuator_health_requires_profile_identity() {
        let metadata = EvidenceMetadataV1::available(
            EvidenceSourceClass::Measured,
            Some(host_time(200)),
        )
        .unwrap();
        assert_eq!(
            NormalizedVectorEvidenceV1::new(
                EmbodimentEvidenceKind::ActuatorHealth,
                Some(vec![1.0]),
                metadata
            ),
            Err(EvidenceValidationError::ProfileRequiredForVector)
        );
    }

    #[test]
    fn semantic_kind_cannot_use_wrong_scalar_vector_shape() {
        assert!(matches!(
            NormalizedScalarEvidenceV1::unavailable(EmbodimentEvidenceKind::ActuatorHealth),
            Err(EvidenceValidationError::CardinalityMismatch {
                expected: EvidenceCardinality::Scalar,
                ..
            })
        ));
        assert!(matches!(
            NormalizedVectorEvidenceV1::unavailable(
                EmbodimentEvidenceKind::ObservationConfidence
            ),
            Err(EvidenceValidationError::CardinalityMismatch {
                expected: EvidenceCardinality::Vector,
                ..
            })
        ));
    }

    #[test]
    fn predictive_residual_and_temporal_novelty_are_distinct_semantics() {
        assert_ne!(
            EmbodimentEvidenceKind::PredictiveResidual,
            EmbodimentEvidenceKind::TemporalStateNovelty
        );
        assert_eq!(
            EmbodimentEvidenceKind::PredictiveResidual.cardinality(),
            EvidenceCardinality::Scalar
        );
        assert_eq!(
            EmbodimentEvidenceKind::TemporalStateNovelty.cardinality(),
            EvidenceCardinality::Scalar
        );
    }

    #[test]
    fn invalid_clock_and_identifier_strings_are_rejected() {
        assert_eq!(
            ClockDomainId::new("  "),
            Err(EvidenceValidationError::InvalidClockDomain)
        );
        assert_eq!(
            ClockDomainId::new(" host.monotonic"),
            Err(EvidenceValidationError::InvalidClockDomain)
        );

        let mut metadata = EvidenceMetadataV1::available(
            EvidenceSourceClass::Measured,
            Some(host_time(1)),
        )
        .unwrap();
        metadata.profile_id = Some(" ".into());
        assert_eq!(
            metadata.validate(),
            Err(EvidenceValidationError::InvalidIdentifier("profile_id"))
        );
    }

    #[test]
    fn validity_window_requires_timestamp() {
        let mut metadata = EvidenceMetadataV1::available(
            EvidenceSourceClass::CompatibilityAssumption,
            None,
        )
        .unwrap();
        metadata.valid_for_ns = Some(10);
        assert_eq!(
            metadata.validate(),
            Err(EvidenceValidationError::ValidityWindowRequiresTimestamp)
        );
    }

    #[test]
    fn freshness_uses_only_same_monotonic_clock_domain() {
        let mut metadata = EvidenceMetadataV1::available(
            EvidenceSourceClass::Measured,
            Some(host_time(100)),
        )
        .unwrap();
        metadata.valid_for_ns = Some(50);

        assert_eq!(
            metadata.effective_availability_at(&host_time(149)).unwrap(),
            EvidenceAvailability::Available
        );
        assert_eq!(
            metadata.effective_availability_at(&host_time(150)).unwrap(),
            EvidenceAvailability::Stale
        );
        assert_eq!(
            metadata.effective_availability_at(&sim_time(150)),
            Err(EvidenceValidationError::ClockDomainMismatch)
        );
        assert_eq!(
            metadata.effective_availability_at(&host_time(99)),
            Err(EvidenceValidationError::NonMonotonicTimestamp)
        );
    }

    #[test]
    fn stale_runtime_evidence_retains_value_but_is_not_available() {
        let metadata = EvidenceMetadataV1::stale(EvidenceSourceClass::Estimated, host_time(5))
            .unwrap();
        let evidence = NormalizedScalarEvidenceV1::new(
            EmbodimentEvidenceKind::SensorimotorAccuracy,
            Some(0.8),
            metadata,
        )
        .unwrap();
        assert_eq!(evidence.value, Some(0.8));
        assert_eq!(evidence.metadata.availability, EvidenceAvailability::Stale);
    }
}
