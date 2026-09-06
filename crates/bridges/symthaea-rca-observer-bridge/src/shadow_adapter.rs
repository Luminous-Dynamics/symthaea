// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Pure RCA-002.1 projection from a provenance-bound completed cycle into the
//! isolated shadow-observation contract.
//!
//! This module has no route back into `RcaObservableCognitiveLoopV1`. It borrows
//! one already-completed cycle, produces owned detached data, validates that
//! data through `symthaea-rca-shadow`, and returns only the validated observation.

use symthaea_rca_shadow::{
    COGNITIVE_PROBABILITY_SCALE, FROZEN_CYCLE_OBSERVATION_SCHEMA_VERSION,
    FrozenCycleObservationV1, ShadowObservationError, ValidatedFrozenCycleObservationV1,
};
use thiserror::Error;

use super::{RcaCompletedCycleV1, encode_value_v1};

pub const RCA_CYCLE_ADAPTER_PROFILE_V1: &str = "rca-cycle-result-shadow-projection-v1";

/// Normative semantics of the first production-to-shadow projection.
///
/// Anything not named here remains outside RCA shadow evidence v1. Adding a
/// field later requires a new adapter profile/contract rather than silently
/// changing what an existing observation commitment means.
pub const RCA_CYCLE_ADAPTER_CONTRACT_V1: &str = concat!(
    "rca-cycle-result-shadow-projection-v1\n",
    "source_generation_digest=RcaCompletedCycleV1.source_generation_digest\n",
    "execution_lineage_digest=RcaCompletedCycleV1.execution_lineage_digest\n",
    "cycle_index=wrapper_owned_monotonic_cycle_index\n",
    "cycle_time_us=CycleResult.cycle_time_us\n",
    "prediction_error_ppm=reject_nonfinite_or_outside_[0,1];round(f64(value)*1000000)\n",
    "peak_attention_bits=exact_f32_to_bits;shadow_validation_rejects_nonfinite_or_negative;no_clamp_or_normalization\n",
    "learning_occurred=CycleResult.learning_occurred\n",
    "detected_primitive_count=checked_u32_len_only;primitive_identities_excluded_v1\n",
    "output_digest=blake3_domain|u64le_len|ordered_f32_to_bits_le\n",
    "thought_digest=blake3_domain|u64le_len|ordered_f32_to_bits_le\n",
    "metadata_digest=blake3_domain|canonical_value_tree_v1(CycleMetadata)\n",
    "language_output_digest=optional_blake3_domain|u64le_utf8_len|exact_utf8\n",
    "language_source=CycleResult.language_source_exact_when_output_present\n",
    "excluded_v1=training_loss,bits_saved_persist,bits_saved_zero,bits_kappa,recall_fired,recall_similarity,recall_matched_timestamp,wisdom_hv,feature_gated_cycle_payloads,detected_primitive_identities\n",
    "no_shadow_receipt_or_control_value_is_returned_to_cognition\n",
);

const ADAPTER_CONTRACT_DOMAIN: &[u8] = b"symthaea:rca-cycle-adapter-contract:v1\0";
const OUTPUT_DOMAIN: &[u8] = b"symthaea:rca-cycle-output:v1\0";
const THOUGHT_DOMAIN: &[u8] = b"symthaea:rca-cycle-thought:v1\0";
const METADATA_DOMAIN: &[u8] = b"symthaea:rca-cycle-metadata:v1\0";
const LANGUAGE_DOMAIN: &[u8] = b"symthaea:rca-cycle-language:v1\0";

pub fn cycle_adapter_contract_digest_v1() -> String {
    hash_bytes(ADAPTER_CONTRACT_DOMAIN, RCA_CYCLE_ADAPTER_CONTRACT_V1.as_bytes())
}

/// Project one already-completed, lineage-bound cycle into validated detached
/// shadow data.
///
/// This function is pure with respect to live cognition: it receives `&` only,
/// allocates owned projection data, and returns no handle, callback, receipt, or
/// control value that the cognitive wrapper consumes.
pub fn adapt_completed_cycle_v1(
    completed: &RcaCompletedCycleV1,
) -> Result<ValidatedFrozenCycleObservationV1, RcaShadowAdapterError> {
    let result = completed.result();

    let prediction_error_ppm = prediction_error_to_ppm(result.prediction_error)?;
    let peak_attention_bits = result.peak_attention.to_bits();
    let detected_primitive_count = u32::try_from(result.detected_primitives.len()).map_err(|_| {
        RcaShadowAdapterError::PrimitiveCountOverflow {
            found: result.detected_primitives.len(),
        }
    })?;

    let output_digest = hash_f32_slice(OUTPUT_DOMAIN, &result.output);
    let thought_digest = hash_f32_slice(THOUGHT_DOMAIN, &result.thought_vector);

    let metadata_value = serde_json::to_value(&result.metadata)?;
    let mut metadata_bytes = Vec::new();
    encode_value_v1(&metadata_value, &mut metadata_bytes);
    let metadata_digest = hash_bytes(METADATA_DOMAIN, &metadata_bytes);

    let language_output_digest = result
        .language_output
        .as_deref()
        .map(|text| hash_bytes(LANGUAGE_DOMAIN, text.as_bytes()));

    FrozenCycleObservationV1 {
        schema_version: FROZEN_CYCLE_OBSERVATION_SCHEMA_VERSION,
        source_generation_digest: completed.source_generation_digest().to_string(),
        execution_lineage_digest: completed.execution_lineage_digest().to_string(),
        adapter_profile: RCA_CYCLE_ADAPTER_PROFILE_V1.to_string(),
        adapter_contract_digest: cycle_adapter_contract_digest_v1(),
        cycle_index: completed.cycle_index(),
        cycle_time_us: result.cycle_time_us,
        prediction_error_ppm,
        peak_attention_bits,
        learning_occurred: result.learning_occurred,
        detected_primitive_count,
        output_digest,
        thought_digest,
        metadata_digest,
        language_output_digest,
        language_source: result.language_source.clone(),
    }
    .validate()
    .map_err(RcaShadowAdapterError::ShadowValidation)
}

#[derive(Debug, Error)]
pub enum RcaShadowAdapterError {
    #[error("prediction_error must be finite and in [0,1]; found f32 bits 0x{value_bits:08x}")]
    PredictionErrorOutsideUnitInterval { value_bits: u32 },
    #[error("detected primitive count {found} does not fit in shadow schema u32")]
    PrimitiveCountOverflow { found: usize },
    #[error("cannot serialize CycleMetadata for the declared v1 projection: {0}")]
    MetadataProjection(#[from] serde_json::Error),
    #[error("detached shadow observation failed validation: {0}")]
    ShadowValidation(ShadowObservationError),
}

fn prediction_error_to_ppm(value: f32) -> Result<u32, RcaShadowAdapterError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(RcaShadowAdapterError::PredictionErrorOutsideUnitInterval {
            value_bits: value.to_bits(),
        });
    }

    let scaled = (f64::from(value) * f64::from(COGNITIVE_PROBABILITY_SCALE)).round();
    Ok(scaled as u32)
}

fn hash_f32_slice(domain: &[u8], values: &[f32]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&(values.len() as u64).to_le_bytes());
    for value in values {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn hash_bytes(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    format!("blake3:{}", hasher.finalize().to_hex())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RcaObservableCognitiveLoopV1;
    use symthaea::cognitive_loop::CognitiveLoopConfig;

    const SOURCE: &str =
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn completed_cycle() -> RcaCompletedCycleV1 {
        let mut observable = RcaObservableCognitiveLoopV1::new_shadow_observable(
            CognitiveLoopConfig::default(),
            SOURCE,
        )
        .expect("observable execution");
        observable.cycle("RCA shadow adapter qualification cycle")
    }

    #[test]
    fn projection_is_deterministic_for_same_completed_cycle() {
        let completed = completed_cycle();
        let first = adapt_completed_cycle_v1(&completed).unwrap();
        let second = adapt_completed_cycle_v1(&completed).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn projection_preserves_authoritative_identity() {
        let completed = completed_cycle();
        let observation = adapt_completed_cycle_v1(&completed).unwrap();
        let raw = observation.as_raw();
        assert_eq!(raw.source_generation_digest, SOURCE);
        assert_eq!(
            raw.execution_lineage_digest,
            completed.execution_lineage_digest()
        );
        assert_eq!(raw.cycle_index, completed.cycle_index());
        assert_eq!(raw.adapter_profile, RCA_CYCLE_ADAPTER_PROFILE_V1);
        assert_eq!(
            raw.adapter_contract_digest,
            cycle_adapter_contract_digest_v1()
        );
    }

    #[test]
    fn output_and_thought_commitments_are_bit_sensitive() {
        let mut completed = completed_cycle();
        if completed.result.output.is_empty() {
            completed.result.output.push(0.0);
        }
        if completed.result.thought_vector.is_empty() {
            completed.result.thought_vector.push(0.0);
        }
        let original = adapt_completed_cycle_v1(&completed).unwrap();

        completed.result.output[0] = f32::from_bits(completed.result.output[0].to_bits() ^ 1);
        completed.result.thought_vector[0] =
            f32::from_bits(completed.result.thought_vector[0].to_bits() ^ 1);
        let changed = adapt_completed_cycle_v1(&completed).unwrap();

        assert_ne!(
            original.as_raw().output_digest,
            changed.as_raw().output_digest
        );
        assert_ne!(
            original.as_raw().thought_digest,
            changed.as_raw().thought_digest
        );
    }

    #[test]
    fn metadata_commitment_changes_with_admitted_metadata_projection() {
        let mut completed = completed_cycle();
        let original = adapt_completed_cycle_v1(&completed).unwrap();
        completed.result.metadata.surprise_triggered =
            !completed.result.metadata.surprise_triggered;
        let changed = adapt_completed_cycle_v1(&completed).unwrap();
        assert_ne!(
            original.as_raw().metadata_digest,
            changed.as_raw().metadata_digest
        );
    }

    #[test]
    fn peak_attention_above_one_is_preserved_bit_exact() {
        let mut completed = completed_cycle();
        completed.result.peak_attention = 2.75;
        let observation = adapt_completed_cycle_v1(&completed).unwrap();
        assert_eq!(observation.as_raw().peak_attention_bits, 2.75_f32.to_bits());
    }

    #[test]
    fn invalid_peak_attention_fails_without_clamping() {
        let mut completed = completed_cycle();
        completed.result.peak_attention = f32::NAN;
        assert!(matches!(
            adapt_completed_cycle_v1(&completed),
            Err(RcaShadowAdapterError::ShadowValidation(
                ShadowObservationError::InvalidPeakAttention { .. }
            ))
        ));

        completed.result.peak_attention = -0.25;
        assert!(matches!(
            adapt_completed_cycle_v1(&completed),
            Err(RcaShadowAdapterError::ShadowValidation(
                ShadowObservationError::InvalidPeakAttention { .. }
            ))
        ));
    }

    #[test]
    fn excluded_training_loss_does_not_silently_widen_v1_projection() {
        let mut completed = completed_cycle();
        let original = adapt_completed_cycle_v1(&completed).unwrap();
        completed.result.training_loss = Some(0.123456);
        let changed = adapt_completed_cycle_v1(&completed).unwrap();
        assert_eq!(original, changed);
    }

    #[test]
    fn primitive_identity_is_excluded_while_count_is_admitted() {
        let mut completed = completed_cycle();
        completed.result.detected_primitives = vec!["ALPHA".into(), "BETA".into()];
        let original = adapt_completed_cycle_v1(&completed).unwrap();
        completed.result.detected_primitives = vec!["GAMMA".into(), "DELTA".into()];
        let changed = adapt_completed_cycle_v1(&completed).unwrap();
        assert_eq!(original, changed);
        assert_eq!(changed.as_raw().detected_primitive_count, 2);
    }

    #[test]
    fn out_of_range_prediction_error_fails_instead_of_clamping() {
        let mut completed = completed_cycle();
        completed.result.prediction_error = 1.01;
        assert!(matches!(
            adapt_completed_cycle_v1(&completed),
            Err(RcaShadowAdapterError::PredictionErrorOutsideUnitInterval { .. })
        ));

        completed.result.prediction_error = f32::NAN;
        assert!(matches!(
            adapt_completed_cycle_v1(&completed),
            Err(RcaShadowAdapterError::PredictionErrorOutsideUnitInterval { .. })
        ));
    }

    #[test]
    fn language_commitment_requires_language_provenance() {
        let mut completed = completed_cycle();
        completed.result.language_output = Some("committed language".to_string());
        completed.result.language_source = None;
        assert!(matches!(
            adapt_completed_cycle_v1(&completed),
            Err(RcaShadowAdapterError::ShadowValidation(
                ShadowObservationError::MissingLanguageSource
            ))
        ));
    }

    #[test]
    fn exact_language_bytes_change_commitment() {
        let mut completed = completed_cycle();
        completed.result.language_source = Some("qualified-test-language-source".into());
        completed.result.language_output = Some("alpha".into());
        let alpha = adapt_completed_cycle_v1(&completed).unwrap();
        completed.result.language_output = Some("Alpha".into());
        let capitalized = adapt_completed_cycle_v1(&completed).unwrap();
        assert_ne!(
            alpha.as_raw().language_output_digest,
            capitalized.as_raw().language_output_digest
        );
    }

    #[test]
    fn adapter_contract_has_strict_identity() {
        let digest = cycle_adapter_contract_digest_v1();
        assert!(digest.starts_with("blake3:"));
        assert_eq!(digest.len(), "blake3:".len() + 64);
        assert_eq!(digest, cycle_adapter_contract_digest_v1());
    }
}
