// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Detached observation types for the RCA shadow boundary.
//!
//! This module deliberately contains values only. It owns no runtime handles,
//! mutable cognitive state, workspace references, action capabilities, or
//! recursive-improvement authority.

use serde::{Deserialize, Deserializer, Serialize};

use crate::cognitive_evidence::COGNITIVE_PROBABILITY_SCALE;

pub const FROZEN_CYCLE_OBSERVATION_SCHEMA_VERSION: u16 = 1;

/// Owned, detached projection of one completed cognitive cycle.
///
/// The eventual runtime adapter is expected to construct this only after the
/// authoritative cycle has completed. The RCA shadow plane receives this value,
/// never `CycleResult` or `CognitiveLoopService` itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenCycleObservationV1 {
    pub schema_version: u16,
    /// Digest identifying the exact architecture/configuration generation that
    /// produced this observation.
    pub source_generation_digest: String,
    /// Monotonic cycle number within the producing execution lineage.
    pub cycle_index: u64,
    /// Duration of the completed cycle.
    pub cycle_time_us: u64,
    /// Fixed-point prediction error in [0, 1_000_000].
    pub prediction_error_ppm: u32,
    /// Fixed-point peak attention in [0, 1_000_000].
    pub peak_attention_ppm: u32,
    pub learning_occurred: bool,
    pub detected_primitive_count: u32,
    /// Commitment to the exact cycle output vector/materialization.
    pub output_digest: String,
    /// Commitment to the exact thought-vector materialization.
    pub thought_digest: String,
    /// Commitment to the exact `CycleMetadata` projection used by the adapter.
    pub metadata_digest: String,
    /// Commitment to exact language output when present. Raw language need not
    /// cross the shadow boundary merely to identify the observation.
    pub language_output_digest: Option<String>,
    /// Producing language subsystem/model identity when language output exists.
    pub language_source: Option<String>,
}

impl FrozenCycleObservationV1 {
    pub fn validate(
        self,
    ) -> Result<ValidatedFrozenCycleObservationV1, FrozenCycleObservationError> {
        ValidatedFrozenCycleObservationV1::try_from(self)
    }
}

/// Structurally validated detached cycle observation.
///
/// Fields remain inaccessible for mutation through this wrapper. Persistence
/// crosses validation again through the custom `Deserialize` implementation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ValidatedFrozenCycleObservationV1(FrozenCycleObservationV1);

impl ValidatedFrozenCycleObservationV1 {
    pub fn as_raw(&self) -> &FrozenCycleObservationV1 {
        &self.0
    }

    pub const fn cycle_index(&self) -> u64 {
        self.0.cycle_index
    }

    pub fn source_generation_digest(&self) -> &str {
        &self.0.source_generation_digest
    }
}

impl TryFrom<FrozenCycleObservationV1> for ValidatedFrozenCycleObservationV1 {
    type Error = FrozenCycleObservationError;

    fn try_from(value: FrozenCycleObservationV1) -> Result<Self, Self::Error> {
        if value.schema_version != FROZEN_CYCLE_OBSERVATION_SCHEMA_VERSION {
            return Err(FrozenCycleObservationError::UnsupportedSchemaVersion {
                found: value.schema_version,
            });
        }

        for digest in [
            value.source_generation_digest.as_str(),
            value.output_digest.as_str(),
            value.thought_digest.as_str(),
            value.metadata_digest.as_str(),
        ] {
            validate_digest(digest)?;
        }

        if let Some(digest) = value.language_output_digest.as_deref() {
            validate_digest(digest)?;
            if value
                .language_source
                .as_deref()
                .is_none_or(|source| source.trim().is_empty())
            {
                return Err(FrozenCycleObservationError::MissingLanguageSource);
            }
        } else if value.language_source.is_some() {
            return Err(FrozenCycleObservationError::DanglingLanguageSource);
        }

        if value.prediction_error_ppm > COGNITIVE_PROBABILITY_SCALE {
            return Err(FrozenCycleObservationError::PredictionErrorOutOfRange {
                found: value.prediction_error_ppm,
            });
        }
        if value.peak_attention_ppm > COGNITIVE_PROBABILITY_SCALE {
            return Err(FrozenCycleObservationError::PeakAttentionOutOfRange {
                found: value.peak_attention_ppm,
            });
        }

        Ok(Self(value))
    }
}

impl<'de> Deserialize<'de> for ValidatedFrozenCycleObservationV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        FrozenCycleObservationV1::deserialize(deserializer)?
            .validate()
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrozenCycleObservationError {
    UnsupportedSchemaVersion { found: u16 },
    MalformedDigest,
    PredictionErrorOutOfRange { found: u32 },
    PeakAttentionOutOfRange { found: u32 },
    MissingLanguageSource,
    DanglingLanguageSource,
}

impl std::fmt::Display for FrozenCycleObservationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => write!(
                f,
                "unsupported frozen-cycle observation schema version {found}; expected {FROZEN_CYCLE_OBSERVATION_SCHEMA_VERSION}"
            ),
            Self::MalformedDigest => {
                f.write_str("digest must be sha256:<64 hex> or blake3:<64 hex>")
            }
            Self::PredictionErrorOutOfRange { found } => write!(
                f,
                "prediction error {found} exceeds fixed-point scale {COGNITIVE_PROBABILITY_SCALE}"
            ),
            Self::PeakAttentionOutOfRange { found } => write!(
                f,
                "peak attention {found} exceeds fixed-point scale {COGNITIVE_PROBABILITY_SCALE}"
            ),
            Self::MissingLanguageSource => {
                f.write_str("language output commitment requires explicit language source identity")
            }
            Self::DanglingLanguageSource => {
                f.write_str("language source cannot be present without language output commitment")
            }
        }
    }
}

impl std::error::Error for FrozenCycleObservationError {}

fn validate_digest(digest: &str) -> Result<(), FrozenCycleObservationError> {
    let Some((algorithm, hex)) = digest.split_once(':') else {
        return Err(FrozenCycleObservationError::MalformedDigest);
    };
    if !matches!(algorithm, "sha256" | "blake3")
        || hex.len() != 64
        || !hex.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(FrozenCycleObservationError::MalformedDigest);
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

    fn observation() -> FrozenCycleObservationV1 {
        FrozenCycleObservationV1 {
            schema_version: FROZEN_CYCLE_OBSERVATION_SCHEMA_VERSION,
            source_generation_digest: SHA_A.into(),
            cycle_index: 42,
            cycle_time_us: 18_000,
            prediction_error_ppm: 125_000,
            peak_attention_ppm: 810_000,
            learning_occurred: true,
            detected_primitive_count: 3,
            output_digest: SHA_B.into(),
            thought_digest: BLAKE_C.into(),
            metadata_digest: SHA_A.into(),
            language_output_digest: Some(SHA_B.into()),
            language_source: Some("broca-lite@generation-7".into()),
        }
    }

    #[test]
    fn valid_detached_observation_passes() {
        assert!(observation().validate().is_ok());
    }

    #[test]
    fn strict_digests_fail_closed() {
        let mut raw = observation();
        raw.metadata_digest = "decorative-hash".into();
        assert_eq!(
            raw.validate(),
            Err(FrozenCycleObservationError::MalformedDigest)
        );
    }

    #[test]
    fn fixed_point_metrics_are_bounded() {
        let mut raw = observation();
        raw.prediction_error_ppm = COGNITIVE_PROBABILITY_SCALE + 1;
        assert_eq!(
            raw.validate(),
            Err(FrozenCycleObservationError::PredictionErrorOutOfRange {
                found: COGNITIVE_PROBABILITY_SCALE + 1
            })
        );

        let mut raw = observation();
        raw.peak_attention_ppm = COGNITIVE_PROBABILITY_SCALE + 1;
        assert_eq!(
            raw.validate(),
            Err(FrozenCycleObservationError::PeakAttentionOutOfRange {
                found: COGNITIVE_PROBABILITY_SCALE + 1
            })
        );
    }

    #[test]
    fn language_commitment_requires_source_identity() {
        let mut raw = observation();
        raw.language_source = None;
        assert_eq!(
            raw.validate(),
            Err(FrozenCycleObservationError::MissingLanguageSource)
        );
    }

    #[test]
    fn language_source_without_output_is_rejected() {
        let mut raw = observation();
        raw.language_output_digest = None;
        assert_eq!(
            raw.validate(),
            Err(FrozenCycleObservationError::DanglingLanguageSource)
        );
    }

    #[test]
    fn validated_observation_revalidates_after_persistence() {
        let valid = observation().validate().unwrap();
        let encoded = serde_json::to_string(&valid).unwrap();
        let decoded: ValidatedFrozenCycleObservationV1 = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, valid);

        let mut raw = observation();
        raw.output_digest = "bad".into();
        let encoded = serde_json::to_string(&raw).unwrap();
        assert!(serde_json::from_str::<ValidatedFrozenCycleObservationV1>(&encoded).is_err());
    }
}
