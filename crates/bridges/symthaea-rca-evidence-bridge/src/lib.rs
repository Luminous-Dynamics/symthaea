// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! One-way bridge from detached RCA observations to **candidate** evidence.
//!
//! An RCA frozen observation is an instrumented observation of Symthaea's own
//! runtime. It is not an external empirical observation, an inference, a
//! canonical belief, or action authority. The current canonical cognitive
//! evidence enum has no honest authority variant for this origin, so this crate
//! does not mislabel it.
//!
//! Instead, the bridge extracts only lossless field claims from an already
//! validated [`ValidatedFrozenCycleObservationV1`] and produces
//! [`InstrumentedRuntimeEvidenceCandidateV1`]. Candidate evidence can receive
//! provenance/currentness bookkeeping, but it cannot be converted here into
//! canonical cognitive evidence or admitted evidence-use authority.

#![deny(unsafe_code)]

use serde::{Deserialize, Deserializer, Serialize};
use symthaea_epistemic_governance::currentness::{
    CurrentnessError, EvidenceCurrentnessModeV1, EvidenceCurrentnessV1,
    ValidatedEvidenceCurrentnessV1, COGNITIVE_CURRENTNESS_SCHEMA_VERSION,
};
use symthaea_epistemic_governance::lineage::{
    CognitiveDerivationKindV1, CognitiveLineageError, EvidenceLineageNodeV1,
    ValidatedEvidenceLineageNodeV1, COGNITIVE_LINEAGE_SCHEMA_VERSION,
};
use symthaea_rca_shadow::{
    observe, ValidatedFrozenCycleObservationV1, SHADOW_OBSERVER_PROFILE_V1,
};

pub const RUNTIME_EVIDENCE_CANDIDATE_SCHEMA_VERSION: u16 = 1;
pub const RUNTIME_EVIDENCE_CANDIDATE_PROFILE_V1: &str =
    "rca-instrumented-runtime-evidence-candidate-v1";

/// Normative semantics for candidate evidence extracted from the detached
/// runtime observation boundary.
///
/// The claim set is deliberately closed. A caller chooses which already-frozen
/// field to expose; it cannot inject an arbitrary proposition or claim digest.
pub const RUNTIME_EVIDENCE_CANDIDATE_CONTRACT_V1: &str = concat!(
    "rca-instrumented-runtime-evidence-candidate-v1\n",
    "origin=instrumented_cognitive_runtime_observation\n",
    "input=ValidatedFrozenCycleObservationV1_only\n",
    "claim_selection=closed_lossless_field_projection_only\n",
    "fields=cycle_time_us,prediction_error_ppm,peak_attention_bits,learning_occurred,detected_primitive_count,output_digest,thought_digest,metadata_digest,language_output\n",
    "claim_digest=domain_separated_explicit_variant_encoding\n",
    "candidate_id=blake3(profile_digest|observer_contract_digest|observation_commitment|claim_digest)\n",
    "candidate_is_not_CognitiveEvidenceRefV1\n",
    "candidate_is_not_AdmittedCognitiveEvidenceV1\n",
    "lineage_root_is_provenance_only_not_epistemic_admission\n",
    "immutable_historical_currentness_does_not_mean_current_system_relevance\n",
    "no_gwt_workspace_action_or_self_improvement_authority\n",
);

const PROFILE_DOMAIN: &[u8] = b"symthaea:rca-runtime-evidence-candidate-contract:v1\0";
const CLAIM_DOMAIN: &[u8] = b"symthaea:rca-runtime-observed-claim:v1\0";
const CANDIDATE_DOMAIN: &[u8] = b"symthaea:rca-runtime-evidence-candidate:v1\0";

/// Closed set of exact observation fields that RCA-002.2 may expose as
/// candidate evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShadowObservationFieldV1 {
    CycleTimeUs,
    PredictionErrorPpm,
    PeakAttentionBits,
    LearningOccurred,
    DetectedPrimitiveCount,
    OutputDigest,
    ThoughtDigest,
    MetadataDigest,
    LanguageOutput,
}

/// Lossless claim extracted from one frozen observation field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InstrumentedRuntimeClaimV1 {
    CycleTimeUs { value: u64 },
    PredictionErrorPpm { value: u32 },
    PeakAttentionBits { value_bits: u32 },
    LearningOccurred { value: bool },
    DetectedPrimitiveCount { value: u32 },
    OutputDigest { digest: String },
    ThoughtDigest { digest: String },
    MetadataDigest { digest: String },
    LanguageOutput {
        output_digest: Option<String>,
        source: Option<String>,
    },
}

/// Persistable candidate evidence derived mechanically from a detached
/// observation.
///
/// Deserialization revalidates the nested observation, re-extracts the selected
/// field, recomputes observer identity, claim identity, and candidate identity,
/// and rejects any mismatch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InstrumentedRuntimeEvidenceCandidateV1 {
    schema_version: u16,
    candidate_profile: String,
    candidate_profile_digest: String,
    observer_profile: String,
    observer_contract_digest: String,
    observation_commitment: String,
    field: ShadowObservationFieldV1,
    claim: InstrumentedRuntimeClaimV1,
    claim_digest: String,
    candidate_id: String,
    observation: ValidatedFrozenCycleObservationV1,
}

impl InstrumentedRuntimeEvidenceCandidateV1 {
    pub fn new(
        observation: ValidatedFrozenCycleObservationV1,
        field: ShadowObservationFieldV1,
    ) -> Self {
        let receipt = observe(&observation);
        let candidate_profile_digest = runtime_evidence_candidate_profile_digest_v1();
        let claim = extract_claim(&observation, field);
        let claim_digest = runtime_claim_digest_v1(&claim);
        let candidate_id = runtime_evidence_candidate_id_v1(
            &candidate_profile_digest,
            &receipt.observer_contract_digest,
            &receipt.observation_commitment,
            &claim_digest,
        );

        Self {
            schema_version: RUNTIME_EVIDENCE_CANDIDATE_SCHEMA_VERSION,
            candidate_profile: RUNTIME_EVIDENCE_CANDIDATE_PROFILE_V1.to_string(),
            candidate_profile_digest,
            observer_profile: receipt.observer_profile,
            observer_contract_digest: receipt.observer_contract_digest,
            observation_commitment: receipt.observation_commitment,
            field,
            claim,
            claim_digest,
            candidate_id,
            observation,
        }
    }

    pub fn candidate_id(&self) -> &str {
        &self.candidate_id
    }

    pub fn claim_digest(&self) -> &str {
        &self.claim_digest
    }

    pub fn claim(&self) -> &InstrumentedRuntimeClaimV1 {
        &self.claim
    }

    pub const fn field(&self) -> ShadowObservationFieldV1 {
        self.field
    }

    pub fn observation_commitment(&self) -> &str {
        &self.observation_commitment
    }

    pub fn observation(&self) -> &ValidatedFrozenCycleObservationV1 {
        &self.observation
    }

    /// Represent this candidate as a root in the generic evidence-lineage DAG.
    ///
    /// This records provenance only. A lineage root is not canonical cognitive
    /// evidence admission and does not grant any downstream use.
    pub fn lineage_root(
        &self,
    ) -> Result<ValidatedEvidenceLineageNodeV1, CognitiveLineageError> {
        EvidenceLineageNodeV1 {
            schema_version: COGNITIVE_LINEAGE_SCHEMA_VERSION,
            evidence_id: self.candidate_id.clone(),
            parent_ids: Vec::new(),
            derivation_kind: CognitiveDerivationKindV1::RootObservation,
        }
        .validate()
    }

    /// Currentness of the exact historical observation claim itself.
    ///
    /// A statement such as "cycle N had prediction error X" does not become
    /// false because time advances. This must not be interpreted as evidence
    /// that the same state holds *now*; proposition relevance remains a later
    /// admission/relation decision.
    pub fn immutable_historical_currentness(
        &self,
    ) -> Result<ValidatedEvidenceCurrentnessV1, CurrentnessError> {
        EvidenceCurrentnessV1 {
            schema_version: COGNITIVE_CURRENTNESS_SCHEMA_VERSION,
            evidence_id: self.candidate_id.clone(),
            mode: EvidenceCurrentnessModeV1::Immutable,
            observed_at_unix_ms: None,
            valid_until_unix_ms: None,
            source_generation: None,
            model_generation: None,
            environment_generation: None,
            superseded_by: None,
        }
        .validate()
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InstrumentedRuntimeEvidenceCandidateWireV1 {
    schema_version: u16,
    candidate_profile: String,
    candidate_profile_digest: String,
    observer_profile: String,
    observer_contract_digest: String,
    observation_commitment: String,
    field: ShadowObservationFieldV1,
    claim: InstrumentedRuntimeClaimV1,
    claim_digest: String,
    candidate_id: String,
    observation: ValidatedFrozenCycleObservationV1,
}

impl<'de> Deserialize<'de> for InstrumentedRuntimeEvidenceCandidateV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = InstrumentedRuntimeEvidenceCandidateWireV1::deserialize(deserializer)?;
        if wire.schema_version != RUNTIME_EVIDENCE_CANDIDATE_SCHEMA_VERSION {
            return Err(serde::de::Error::custom(
                RuntimeEvidenceCandidateError::UnsupportedSchemaVersion {
                    found: wire.schema_version,
                },
            ));
        }
        if wire.candidate_profile != RUNTIME_EVIDENCE_CANDIDATE_PROFILE_V1 {
            return Err(serde::de::Error::custom(
                RuntimeEvidenceCandidateError::UnexpectedCandidateProfile,
            ));
        }
        validate_digest(&wire.candidate_profile_digest).map_err(serde::de::Error::custom)?;
        validate_digest(&wire.observer_contract_digest).map_err(serde::de::Error::custom)?;
        validate_digest(&wire.observation_commitment).map_err(serde::de::Error::custom)?;
        validate_digest(&wire.claim_digest).map_err(serde::de::Error::custom)?;
        validate_digest(&wire.candidate_id).map_err(serde::de::Error::custom)?;

        let expected_profile_digest = runtime_evidence_candidate_profile_digest_v1();
        if wire.candidate_profile_digest != expected_profile_digest {
            return Err(serde::de::Error::custom(
                RuntimeEvidenceCandidateError::CandidateProfileDigestMismatch,
            ));
        }

        let receipt = observe(&wire.observation);
        if wire.observer_profile != SHADOW_OBSERVER_PROFILE_V1
            || wire.observer_profile != receipt.observer_profile
            || wire.observer_contract_digest != receipt.observer_contract_digest
            || wire.observation_commitment != receipt.observation_commitment
        {
            return Err(serde::de::Error::custom(
                RuntimeEvidenceCandidateError::ObservationBindingMismatch,
            ));
        }

        let expected_claim = extract_claim(&wire.observation, wire.field);
        if wire.claim != expected_claim {
            return Err(serde::de::Error::custom(
                RuntimeEvidenceCandidateError::ClaimProjectionMismatch,
            ));
        }

        let expected_claim_digest = runtime_claim_digest_v1(&expected_claim);
        if wire.claim_digest != expected_claim_digest {
            return Err(serde::de::Error::custom(
                RuntimeEvidenceCandidateError::ClaimDigestMismatch,
            ));
        }

        let expected_candidate_id = runtime_evidence_candidate_id_v1(
            &expected_profile_digest,
            &receipt.observer_contract_digest,
            &receipt.observation_commitment,
            &expected_claim_digest,
        );
        if wire.candidate_id != expected_candidate_id {
            return Err(serde::de::Error::custom(
                RuntimeEvidenceCandidateError::CandidateIdentityMismatch,
            ));
        }

        Ok(Self {
            schema_version: wire.schema_version,
            candidate_profile: wire.candidate_profile,
            candidate_profile_digest: wire.candidate_profile_digest,
            observer_profile: wire.observer_profile,
            observer_contract_digest: wire.observer_contract_digest,
            observation_commitment: wire.observation_commitment,
            field: wire.field,
            claim: wire.claim,
            claim_digest: wire.claim_digest,
            candidate_id: wire.candidate_id,
            observation: wire.observation,
        })
    }
}

pub fn runtime_evidence_candidate_profile_digest_v1() -> String {
    domain_hash(
        PROFILE_DOMAIN,
        RUNTIME_EVIDENCE_CANDIDATE_CONTRACT_V1.as_bytes(),
    )
}

fn extract_claim(
    observation: &ValidatedFrozenCycleObservationV1,
    field: ShadowObservationFieldV1,
) -> InstrumentedRuntimeClaimV1 {
    let raw = observation.as_raw();
    match field {
        ShadowObservationFieldV1::CycleTimeUs => InstrumentedRuntimeClaimV1::CycleTimeUs {
            value: raw.cycle_time_us,
        },
        ShadowObservationFieldV1::PredictionErrorPpm => {
            InstrumentedRuntimeClaimV1::PredictionErrorPpm {
                value: raw.prediction_error_ppm,
            }
        }
        ShadowObservationFieldV1::PeakAttentionBits => {
            InstrumentedRuntimeClaimV1::PeakAttentionBits {
                value_bits: raw.peak_attention_bits,
            }
        }
        ShadowObservationFieldV1::LearningOccurred => {
            InstrumentedRuntimeClaimV1::LearningOccurred {
                value: raw.learning_occurred,
            }
        }
        ShadowObservationFieldV1::DetectedPrimitiveCount => {
            InstrumentedRuntimeClaimV1::DetectedPrimitiveCount {
                value: raw.detected_primitive_count,
            }
        }
        ShadowObservationFieldV1::OutputDigest => InstrumentedRuntimeClaimV1::OutputDigest {
            digest: raw.output_digest.clone(),
        },
        ShadowObservationFieldV1::ThoughtDigest => InstrumentedRuntimeClaimV1::ThoughtDigest {
            digest: raw.thought_digest.clone(),
        },
        ShadowObservationFieldV1::MetadataDigest => {
            InstrumentedRuntimeClaimV1::MetadataDigest {
                digest: raw.metadata_digest.clone(),
            }
        }
        ShadowObservationFieldV1::LanguageOutput => {
            InstrumentedRuntimeClaimV1::LanguageOutput {
                output_digest: raw.language_output_digest.clone(),
                source: raw.language_source.clone(),
            }
        }
    }
}

fn runtime_claim_digest_v1(claim: &InstrumentedRuntimeClaimV1) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CLAIM_DOMAIN);
    match claim {
        InstrumentedRuntimeClaimV1::CycleTimeUs { value } => {
            hash_field(&mut hasher, b"kind", b"cycle_time_us");
            hash_field(&mut hasher, b"value", &value.to_le_bytes());
        }
        InstrumentedRuntimeClaimV1::PredictionErrorPpm { value } => {
            hash_field(&mut hasher, b"kind", b"prediction_error_ppm");
            hash_field(&mut hasher, b"value", &value.to_le_bytes());
        }
        InstrumentedRuntimeClaimV1::PeakAttentionBits { value_bits } => {
            hash_field(&mut hasher, b"kind", b"peak_attention_bits");
            hash_field(&mut hasher, b"value_bits", &value_bits.to_le_bytes());
        }
        InstrumentedRuntimeClaimV1::LearningOccurred { value } => {
            hash_field(&mut hasher, b"kind", b"learning_occurred");
            hash_field(&mut hasher, b"value", &[u8::from(*value)]);
        }
        InstrumentedRuntimeClaimV1::DetectedPrimitiveCount { value } => {
            hash_field(&mut hasher, b"kind", b"detected_primitive_count");
            hash_field(&mut hasher, b"value", &value.to_le_bytes());
        }
        InstrumentedRuntimeClaimV1::OutputDigest { digest } => {
            hash_field(&mut hasher, b"kind", b"output_digest");
            hash_field(&mut hasher, b"digest", digest.as_bytes());
        }
        InstrumentedRuntimeClaimV1::ThoughtDigest { digest } => {
            hash_field(&mut hasher, b"kind", b"thought_digest");
            hash_field(&mut hasher, b"digest", digest.as_bytes());
        }
        InstrumentedRuntimeClaimV1::MetadataDigest { digest } => {
            hash_field(&mut hasher, b"kind", b"metadata_digest");
            hash_field(&mut hasher, b"digest", digest.as_bytes());
        }
        InstrumentedRuntimeClaimV1::LanguageOutput {
            output_digest,
            source,
        } => {
            hash_field(&mut hasher, b"kind", b"language_output");
            hash_option_text(&mut hasher, b"output_digest", output_digest.as_deref());
            hash_option_text(&mut hasher, b"source", source.as_deref());
        }
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn runtime_evidence_candidate_id_v1(
    profile_digest: &str,
    observer_contract_digest: &str,
    observation_commitment: &str,
    claim_digest: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CANDIDATE_DOMAIN);
    hash_field(&mut hasher, b"profile_digest", profile_digest.as_bytes());
    hash_field(
        &mut hasher,
        b"observer_contract_digest",
        observer_contract_digest.as_bytes(),
    );
    hash_field(
        &mut hasher,
        b"observation_commitment",
        observation_commitment.as_bytes(),
    );
    hash_field(&mut hasher, b"claim_digest", claim_digest.as_bytes());
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn hash_field(hasher: &mut blake3::Hasher, label: &[u8], value: &[u8]) {
    hasher.update(&(label.len() as u64).to_le_bytes());
    hasher.update(label);
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn hash_option_text(hasher: &mut blake3::Hasher, label: &[u8], value: Option<&str>) {
    hasher.update(&(label.len() as u64).to_le_bytes());
    hasher.update(label);
    match value {
        None => {
            hasher.update(&[0]);
        }
        Some(text) => {
            hasher.update(&[1]);
            hasher.update(&(text.len() as u64).to_le_bytes());
            hasher.update(text.as_bytes());
        }
    }
}

fn domain_hash(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn validate_digest(digest: &str) -> Result<(), RuntimeEvidenceCandidateError> {
    let Some((algorithm, hex)) = digest.split_once(':') else {
        return Err(RuntimeEvidenceCandidateError::MalformedDigest);
    };
    if !matches!(algorithm, "sha256" | "blake3")
        || hex.len() != 64
        || !hex.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(RuntimeEvidenceCandidateError::MalformedDigest);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeEvidenceCandidateError {
    UnsupportedSchemaVersion { found: u16 },
    UnexpectedCandidateProfile,
    MalformedDigest,
    CandidateProfileDigestMismatch,
    ObservationBindingMismatch,
    ClaimProjectionMismatch,
    ClaimDigestMismatch,
    CandidateIdentityMismatch,
}

impl std::fmt::Display for RuntimeEvidenceCandidateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => write!(
                f,
                "unsupported runtime-evidence candidate schema version {found}; expected {RUNTIME_EVIDENCE_CANDIDATE_SCHEMA_VERSION}"
            ),
            Self::UnexpectedCandidateProfile => write!(
                f,
                "unexpected runtime-evidence candidate profile; expected {RUNTIME_EVIDENCE_CANDIDATE_PROFILE_V1:?}"
            ),
            Self::MalformedDigest => {
                f.write_str("digest must be sha256:<64 hex> or blake3:<64 hex>")
            }
            Self::CandidateProfileDigestMismatch => {
                f.write_str("runtime-evidence candidate profile digest mismatch")
            }
            Self::ObservationBindingMismatch => {
                f.write_str("runtime-evidence candidate does not match its validated observation")
            }
            Self::ClaimProjectionMismatch => {
                f.write_str("runtime-evidence claim is not the declared lossless observation-field projection")
            }
            Self::ClaimDigestMismatch => {
                f.write_str("runtime-evidence claim digest does not match the extracted claim")
            }
            Self::CandidateIdentityMismatch => {
                f.write_str("runtime-evidence candidate id does not match observation and claim identity")
            }
        }
    }
}

impl std::error::Error for RuntimeEvidenceCandidateError {}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_epistemic_governance::currentness::{
        CurrentnessAssessmentV1, CurrentnessContextV1,
    };
    use symthaea_rca_shadow::{
        FrozenCycleObservationV1, FROZEN_CYCLE_OBSERVATION_SCHEMA_VERSION,
    };

    const SHA_A: &str =
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str =
        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const BLAKE_C: &str =
        "blake3:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn observation() -> ValidatedFrozenCycleObservationV1 {
        FrozenCycleObservationV1 {
            schema_version: FROZEN_CYCLE_OBSERVATION_SCHEMA_VERSION,
            source_generation_digest: SHA_A.into(),
            execution_lineage_digest: BLAKE_C.into(),
            adapter_profile: "adapter-v1".into(),
            adapter_contract_digest: SHA_B.into(),
            cycle_index: 9,
            cycle_time_us: 12_345,
            prediction_error_ppm: 250_000,
            peak_attention_bits: 2.5_f32.to_bits(),
            learning_occurred: true,
            detected_primitive_count: 4,
            output_digest: SHA_B.into(),
            thought_digest: BLAKE_C.into(),
            metadata_digest: SHA_A.into(),
            language_output_digest: Some(SHA_B.into()),
            language_source: Some("broca-lite@g7".into()),
        }
        .validate()
        .unwrap()
    }

    #[test]
    fn candidate_is_deterministic_for_same_observation_and_field() {
        let a = InstrumentedRuntimeEvidenceCandidateV1::new(
            observation(),
            ShadowObservationFieldV1::PredictionErrorPpm,
        );
        let b = InstrumentedRuntimeEvidenceCandidateV1::new(
            observation(),
            ShadowObservationFieldV1::PredictionErrorPpm,
        );
        assert_eq!(a, b);
    }

    #[test]
    fn selected_claim_is_lossless_observation_field() {
        let candidate = InstrumentedRuntimeEvidenceCandidateV1::new(
            observation(),
            ShadowObservationFieldV1::PeakAttentionBits,
        );
        assert_eq!(
            candidate.claim(),
            &InstrumentedRuntimeClaimV1::PeakAttentionBits {
                value_bits: 2.5_f32.to_bits()
            }
        );
    }

    #[test]
    fn different_fields_are_different_candidate_evidence() {
        let pe = InstrumentedRuntimeEvidenceCandidateV1::new(
            observation(),
            ShadowObservationFieldV1::PredictionErrorPpm,
        );
        let learning = InstrumentedRuntimeEvidenceCandidateV1::new(
            observation(),
            ShadowObservationFieldV1::LearningOccurred,
        );
        assert_ne!(pe.claim_digest(), learning.claim_digest());
        assert_ne!(pe.candidate_id(), learning.candidate_id());
    }

    #[test]
    fn observation_identity_is_candidate_identity_bearing() {
        let a = InstrumentedRuntimeEvidenceCandidateV1::new(
            observation(),
            ShadowObservationFieldV1::PredictionErrorPpm,
        );
        let mut raw = observation().as_raw().clone();
        raw.cycle_index += 1;
        let b = InstrumentedRuntimeEvidenceCandidateV1::new(
            raw.validate().unwrap(),
            ShadowObservationFieldV1::PredictionErrorPpm,
        );
        assert_ne!(a.observation_commitment(), b.observation_commitment());
        assert_ne!(a.candidate_id(), b.candidate_id());
    }

    #[test]
    fn candidate_id_is_not_observation_or_claim_identity() {
        let candidate = InstrumentedRuntimeEvidenceCandidateV1::new(
            observation(),
            ShadowObservationFieldV1::PredictionErrorPpm,
        );
        assert_ne!(candidate.candidate_id(), candidate.observation_commitment());
        assert_ne!(candidate.candidate_id(), candidate.claim_digest());
    }

    #[test]
    fn persistence_revalidates_complete_projection() {
        let candidate = InstrumentedRuntimeEvidenceCandidateV1::new(
            observation(),
            ShadowObservationFieldV1::PredictionErrorPpm,
        );
        let encoded = serde_json::to_string(&candidate).unwrap();
        let decoded: InstrumentedRuntimeEvidenceCandidateV1 =
            serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, candidate);
    }

    #[test]
    fn tampered_claim_fails_closed() {
        let candidate = InstrumentedRuntimeEvidenceCandidateV1::new(
            observation(),
            ShadowObservationFieldV1::PredictionErrorPpm,
        );
        let mut value = serde_json::to_value(&candidate).unwrap();
        value["claim"]["value"] = serde_json::Value::from(999_999_u64);
        assert!(
            serde_json::from_value::<InstrumentedRuntimeEvidenceCandidateV1>(value).is_err()
        );
    }

    #[test]
    fn tampered_candidate_identity_fails_closed() {
        let candidate = InstrumentedRuntimeEvidenceCandidateV1::new(
            observation(),
            ShadowObservationFieldV1::PredictionErrorPpm,
        );
        let mut value = serde_json::to_value(&candidate).unwrap();
        value["candidate_id"] = serde_json::Value::String(SHA_A.into());
        assert!(
            serde_json::from_value::<InstrumentedRuntimeEvidenceCandidateV1>(value).is_err()
        );
    }

    #[test]
    fn candidate_can_be_a_provenance_root_without_becoming_admitted_evidence() {
        let candidate = InstrumentedRuntimeEvidenceCandidateV1::new(
            observation(),
            ShadowObservationFieldV1::LearningOccurred,
        );
        let root = candidate.lineage_root().unwrap();
        assert_eq!(root.evidence_id(), candidate.candidate_id());
        assert_eq!(root.derivation_kind(), CognitiveDerivationKindV1::RootObservation);
        assert!(root.parent_ids().is_empty());
    }

    #[test]
    fn exact_historical_claim_is_immutable_not_a_current_state_assertion() {
        let candidate = InstrumentedRuntimeEvidenceCandidateV1::new(
            observation(),
            ShadowObservationFieldV1::PredictionErrorPpm,
        );
        let currentness = candidate.immutable_historical_currentness().unwrap();
        assert_eq!(
            currentness.as_raw().mode,
            EvidenceCurrentnessModeV1::Immutable
        );
        assert_eq!(
            currentness.assess(&CurrentnessContextV1 {
                now_unix_ms: u64::MAX,
                source_generation: Some(999),
                model_generation: Some(999),
                environment_generation: Some(999),
            }),
            CurrentnessAssessmentV1::Current
        );
    }

    #[test]
    fn language_absence_is_an_exact_observed_claim() {
        let mut raw = observation().as_raw().clone();
        raw.language_output_digest = None;
        raw.language_source = None;
        let candidate = InstrumentedRuntimeEvidenceCandidateV1::new(
            raw.validate().unwrap(),
            ShadowObservationFieldV1::LanguageOutput,
        );
        assert_eq!(
            candidate.claim(),
            &InstrumentedRuntimeClaimV1::LanguageOutput {
                output_digest: None,
                source: None
            }
        );
    }

    #[test]
    fn candidate_profile_has_strict_identity() {
        let digest = runtime_evidence_candidate_profile_digest_v1();
        assert!(digest.starts_with("blake3:"));
        assert_eq!(digest.len(), "blake3:".len() + 64);
        assert_eq!(digest, runtime_evidence_candidate_profile_digest_v1());
    }
}
