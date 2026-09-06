// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Causally inert observation boundary for Recursive Cognitive Architecture v1.
//!
//! The production cognitive loop must eventually project a completed cycle into
//! [`FrozenCycleObservationV1`] and pass only that owned value across this
//! boundary. This crate deliberately has no dependency on the root `symthaea`
//! crate, GWT, MetaRouter, memory, learning, action, networking, or recursive
//! improvement infrastructure.
//!
//! RCA-002 does not perform epistemic admission. It proves that an observation
//! can be detached, validated, cryptographically committed, and replayed without
//! acquiring a path back into authoritative cognition.

#![deny(unsafe_code)]

use serde::{Deserialize, Deserializer, Serialize};

pub const FROZEN_CYCLE_OBSERVATION_SCHEMA_VERSION: u16 = 1;
pub const SHADOW_OBSERVATION_RECEIPT_SCHEMA_VERSION: u16 = 1;
pub const SHADOW_OBSERVER_PROFILE_V1: &str = "rca-shadow-observer-v1";
pub const COGNITIVE_PROBABILITY_SCALE: u32 = 1_000_000;

/// Owned, detached projection of one already-completed cognitive cycle.
///
/// This is intentionally a small observation surface rather than a clone of
/// `CycleResult`. Large/raw cognitive artifacts are represented by content
/// commitments, preventing shadow analysis from accidentally becoming another
/// runtime object graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenCycleObservationV1 {
    pub schema_version: u16,
    /// Exact architecture/configuration generation that produced this cycle.
    pub source_generation_digest: String,
    /// Exact execution/session lineage. `cycle_index` is meaningful only within
    /// this lineage and must never be treated as globally unique by itself.
    pub execution_lineage_digest: String,
    /// Named projection implementation that created this detached observation.
    pub adapter_profile: String,
    /// Commitment to the exact adapter/projection contract semantics.
    pub adapter_contract_digest: String,
    /// Monotonic cycle number within the producing execution lineage.
    pub cycle_index: u64,
    pub cycle_time_us: u64,
    /// Fixed-point value in [0, 1_000_000].
    pub prediction_error_ppm: u32,
    /// Fixed-point value in [0, 1_000_000].
    pub peak_attention_ppm: u32,
    pub learning_occurred: bool,
    pub detected_primitive_count: u32,
    /// Exact cycle output commitment.
    pub output_digest: String,
    /// Exact thought-vector commitment.
    pub thought_digest: String,
    /// Commitment to the exact metadata projection selected by the adapter.
    pub metadata_digest: String,
    /// Exact language output commitment when language was emitted.
    pub language_output_digest: Option<String>,
    /// Explicit producing language subsystem/model identity when output exists.
    pub language_source: Option<String>,
}

impl FrozenCycleObservationV1 {
    pub fn validate(
        self,
    ) -> Result<ValidatedFrozenCycleObservationV1, ShadowObservationError> {
        ValidatedFrozenCycleObservationV1::try_from(self)
    }
}

/// Validated detached observation. Persistence must revalidate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ValidatedFrozenCycleObservationV1(FrozenCycleObservationV1);

impl ValidatedFrozenCycleObservationV1 {
    pub fn as_raw(&self) -> &FrozenCycleObservationV1 {
        &self.0
    }
}

impl TryFrom<FrozenCycleObservationV1> for ValidatedFrozenCycleObservationV1 {
    type Error = ShadowObservationError;

    fn try_from(value: FrozenCycleObservationV1) -> Result<Self, Self::Error> {
        if value.schema_version != FROZEN_CYCLE_OBSERVATION_SCHEMA_VERSION {
            return Err(ShadowObservationError::UnsupportedObservationSchema {
                found: value.schema_version,
            });
        }

        for digest in [
            value.source_generation_digest.as_str(),
            value.execution_lineage_digest.as_str(),
            value.adapter_contract_digest.as_str(),
            value.output_digest.as_str(),
            value.thought_digest.as_str(),
            value.metadata_digest.as_str(),
        ] {
            validate_digest(digest)?;
        }

        if value.adapter_profile.trim().is_empty() {
            return Err(ShadowObservationError::MissingAdapterProfile);
        }

        if value.prediction_error_ppm > COGNITIVE_PROBABILITY_SCALE {
            return Err(ShadowObservationError::PredictionErrorOutOfRange {
                found: value.prediction_error_ppm,
            });
        }
        if value.peak_attention_ppm > COGNITIVE_PROBABILITY_SCALE {
            return Err(ShadowObservationError::PeakAttentionOutOfRange {
                found: value.peak_attention_ppm,
            });
        }

        match (
            value.language_output_digest.as_deref(),
            value.language_source.as_deref(),
        ) {
            (Some(digest), Some(source)) => {
                validate_digest(digest)?;
                if source.trim().is_empty() {
                    return Err(ShadowObservationError::MissingLanguageSource);
                }
            }
            (Some(_), None) => return Err(ShadowObservationError::MissingLanguageSource),
            (None, Some(_)) => return Err(ShadowObservationError::DanglingLanguageSource),
            (None, None) => {}
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

/// Replayable, observational receipt from the shadow boundary.
///
/// This type is deliberately data-only. It is not admitted evidence, a belief
/// token, an action capability, a GWT submission, or an improvement permit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowObservationReceiptV1 {
    pub schema_version: u16,
    pub observer_profile: String,
    /// BLAKE3 commitment over the exact validated observation serialization.
    pub observation_commitment: String,
    pub source_generation_digest: String,
    pub execution_lineage_digest: String,
    pub adapter_profile: String,
    pub adapter_contract_digest: String,
    pub cycle_index: u64,
    pub cycle_time_us: u64,
    pub prediction_error_ppm: u32,
    pub peak_attention_ppm: u32,
    pub learning_occurred: bool,
    pub detected_primitive_count: u32,
    pub has_language_output: bool,
}

/// Pure one-way RCA-002 observation function.
///
/// There is no clock, RNG, filesystem, network, mutable singleton, runtime
/// handle, or callback here. Equal validated observations produce equal receipts.
pub fn observe(
    observation: &ValidatedFrozenCycleObservationV1,
) -> Result<ShadowObservationReceiptV1, ShadowObservationError> {
    let raw = observation.as_raw();
    let bytes = serde_json::to_vec(raw).map_err(ShadowObservationError::Serialization)?;
    let commitment = format!("blake3:{}", blake3::hash(&bytes).to_hex());

    Ok(ShadowObservationReceiptV1 {
        schema_version: SHADOW_OBSERVATION_RECEIPT_SCHEMA_VERSION,
        observer_profile: SHADOW_OBSERVER_PROFILE_V1.to_string(),
        observation_commitment: commitment,
        source_generation_digest: raw.source_generation_digest.clone(),
        execution_lineage_digest: raw.execution_lineage_digest.clone(),
        adapter_profile: raw.adapter_profile.clone(),
        adapter_contract_digest: raw.adapter_contract_digest.clone(),
        cycle_index: raw.cycle_index,
        cycle_time_us: raw.cycle_time_us,
        prediction_error_ppm: raw.prediction_error_ppm,
        peak_attention_ppm: raw.peak_attention_ppm,
        learning_occurred: raw.learning_occurred,
        detected_primitive_count: raw.detected_primitive_count,
        has_language_output: raw.language_output_digest.is_some(),
    })
}

#[derive(Debug)]
pub enum ShadowObservationError {
    UnsupportedObservationSchema { found: u16 },
    MalformedDigest,
    MissingAdapterProfile,
    PredictionErrorOutOfRange { found: u32 },
    PeakAttentionOutOfRange { found: u32 },
    MissingLanguageSource,
    DanglingLanguageSource,
    Serialization(serde_json::Error),
}

impl PartialEq for ShadowObservationError {
    fn eq(&self, other: &Self) -> bool {
        use ShadowObservationError::*;
        match (self, other) {
            (
                UnsupportedObservationSchema { found: a },
                UnsupportedObservationSchema { found: b },
            ) => a == b,
            (MalformedDigest, MalformedDigest) => true,
            (MissingAdapterProfile, MissingAdapterProfile) => true,
            (PredictionErrorOutOfRange { found: a }, PredictionErrorOutOfRange { found: b }) => {
                a == b
            }
            (PeakAttentionOutOfRange { found: a }, PeakAttentionOutOfRange { found: b }) => a == b,
            (MissingLanguageSource, MissingLanguageSource) => true,
            (DanglingLanguageSource, DanglingLanguageSource) => true,
            (Serialization(a), Serialization(b)) => a.to_string() == b.to_string(),
            _ => false,
        }
    }
}

impl Eq for ShadowObservationError {}

impl std::fmt::Display for ShadowObservationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedObservationSchema { found } => write!(
                f,
                "unsupported frozen-cycle observation schema version {found}; expected {FROZEN_CYCLE_OBSERVATION_SCHEMA_VERSION}"
            ),
            Self::MalformedDigest => {
                f.write_str("digest must be sha256:<64 hex> or blake3:<64 hex>")
            }
            Self::MissingAdapterProfile => {
                f.write_str("shadow observation requires explicit adapter profile identity")
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
            Self::Serialization(error) => write!(f, "cannot serialize shadow observation: {error}"),
        }
    }
}

impl std::error::Error for ShadowObservationError {}

fn validate_digest(digest: &str) -> Result<(), ShadowObservationError> {
    let Some((algorithm, hex)) = digest.split_once(':') else {
        return Err(ShadowObservationError::MalformedDigest);
    };
    if !matches!(algorithm, "sha256" | "blake3")
        || hex.len() != 64
        || !hex.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(ShadowObservationError::MalformedDigest);
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

    fn raw() -> FrozenCycleObservationV1 {
        FrozenCycleObservationV1 {
            schema_version: FROZEN_CYCLE_OBSERVATION_SCHEMA_VERSION,
            source_generation_digest: SHA_A.into(),
            execution_lineage_digest: BLAKE_C.into(),
            adapter_profile: "cycle-result-shadow-adapter-v1".into(),
            adapter_contract_digest: SHA_B.into(),
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
    fn valid_observation_produces_receipt() {
        let validated = raw().validate().unwrap();
        let receipt = observe(&validated).unwrap();
        assert_eq!(receipt.cycle_index, 42);
        assert_eq!(receipt.observer_profile, SHADOW_OBSERVER_PROFILE_V1);
        assert_eq!(receipt.execution_lineage_digest, BLAKE_C);
        assert_eq!(receipt.adapter_profile, "cycle-result-shadow-adapter-v1");
        assert!(receipt.observation_commitment.starts_with("blake3:"));
        assert!(receipt.has_language_output);
    }

    #[test]
    fn observer_is_deterministic() {
        let validated = raw().validate().unwrap();
        assert_eq!(observe(&validated).unwrap(), observe(&validated).unwrap());
    }

    #[test]
    fn changing_observation_changes_commitment() {
        let a = raw().validate().unwrap();
        let mut b_raw = raw();
        b_raw.cycle_index += 1;
        let b = b_raw.validate().unwrap();
        assert_ne!(
            observe(&a).unwrap().observation_commitment,
            observe(&b).unwrap().observation_commitment
        );
    }

    #[test]
    fn execution_lineage_is_commitment_bound() {
        let a = raw().validate().unwrap();
        let mut b_raw = raw();
        b_raw.execution_lineage_digest = SHA_B.into();
        let b = b_raw.validate().unwrap();
        assert_ne!(
            observe(&a).unwrap().observation_commitment,
            observe(&b).unwrap().observation_commitment
        );
    }

    #[test]
    fn adapter_semantics_are_commitment_bound() {
        let a = raw().validate().unwrap();
        let mut b_raw = raw();
        b_raw.adapter_contract_digest = BLAKE_C.into();
        let b = b_raw.validate().unwrap();
        assert_ne!(
            observe(&a).unwrap().observation_commitment,
            observe(&b).unwrap().observation_commitment
        );
    }

    #[test]
    fn adapter_profile_must_be_explicit() {
        let mut invalid = raw();
        invalid.adapter_profile = " ".into();
        assert_eq!(
            invalid.validate(),
            Err(ShadowObservationError::MissingAdapterProfile)
        );
    }

    #[test]
    fn invalid_metrics_fail_before_shadow_observation() {
        let mut invalid = raw();
        invalid.prediction_error_ppm = COGNITIVE_PROBABILITY_SCALE + 1;
        assert_eq!(
            invalid.validate(),
            Err(ShadowObservationError::PredictionErrorOutOfRange {
                found: COGNITIVE_PROBABILITY_SCALE + 1
            })
        );
    }

    #[test]
    fn invalid_digest_fails_closed() {
        let mut invalid = raw();
        invalid.metadata_digest = "not-a-commitment".into();
        assert_eq!(
            invalid.validate(),
            Err(ShadowObservationError::MalformedDigest)
        );
    }

    #[test]
    fn language_output_requires_source_identity() {
        let mut invalid = raw();
        invalid.language_source = None;
        assert_eq!(
            invalid.validate(),
            Err(ShadowObservationError::MissingLanguageSource)
        );
    }

    #[test]
    fn validated_observation_revalidates_after_persistence() {
        let validated = raw().validate().unwrap();
        let encoded = serde_json::to_string(&validated).unwrap();
        let decoded: ValidatedFrozenCycleObservationV1 = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, validated);

        let mut invalid = raw();
        invalid.output_digest = "bad".into();
        let encoded = serde_json::to_string(&invalid).unwrap();
        assert!(serde_json::from_str::<ValidatedFrozenCycleObservationV1>(&encoded).is_err());
    }

    #[test]
    fn receipt_contains_no_raw_language_payload() {
        let receipt = observe(&raw().validate().unwrap()).unwrap();
        let encoded = serde_json::to_string(&receipt).unwrap();
        assert!(!encoded.contains("broca-lite"));
        assert!(!encoded.contains("language_output_digest"));
    }
}
