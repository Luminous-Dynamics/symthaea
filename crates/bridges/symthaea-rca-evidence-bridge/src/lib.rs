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
//! provenance bookkeeping, but currentness, canonical cognitive evidence, and
//! admitted evidence-use authority remain later policy boundaries.

#![deny(unsafe_code)]

use serde::{Deserialize, Deserializer, Serialize};
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
pub const RUNTIME_OBSERVATION_ROOT_PROFILE_V1: &str = "rca-runtime-observation-root-v1";

/// Normative identity for the shared provenance root of one frozen observation.
pub const RUNTIME_OBSERVATION_ROOT_CONTRACT_V1: &str = concat!(
    "rca-runtime-observation-root-v1\n",
    "root=one_exact_validated_frozen_cycle_observation\n",
    "root_id=blake3(profile_digest|observer_contract_digest|observation_commitment)\n",
    "all_field_candidates_from_same_observation_share_root\n",
    "observation_root_is_provenance_only_not_epistemic_admission\n",
);

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
    "candidate_id=blake3(profile_digest|observation_root_id|claim_digest)\n",
    "candidate_lineage=transformation_child_of_shared_observation_root\n",
    "same_observation_field_candidates_are_not_independent_roots\n",
    "candidate_is_not_CognitiveEvidenceRefV1\n",
    "candidate_is_not_AdmittedCognitiveEvidenceV1\n",
    "candidate_does_not_self_declare_currentness\n",
    "no_gwt_workspace_action_or_self_improvement_authority\n",
);

const ROOT_PROFILE_DOMAIN: &[u8] = b"symthaea:rca-runtime-observation-root-contract:v1\0";
const ROOT_ID_DOMAIN: &[u8] = b"symthaea:rca-runtime-observation-root:v1\0";
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

/// Recomputed lineage fragment for one runtime field candidate.
///
/// The observation event is the root. The selected field candidate is a
/// deterministic `Transformation` child of that shared root. This prevents two
/// fields extracted from the same cycle from masquerading as independent
/// observations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeCandidateLineageV1 {
    observation_root: ValidatedEvidenceLineageNodeV1,
    candidate_node: ValidatedEvidenceLineageNodeV1,
}

impl RuntimeCandidateLineageV1 {
    pub fn observation_root(&self) -> &ValidatedEvidenceLineageNodeV1 {
        &self.observation_root
    }

    pub fn candidate_node(&self) -> &ValidatedEvidenceLineageNodeV1 {
        &self.candidate_node
    }
}

/// Persistable candidate evidence derived mechanically from a detached
/// observation.
///
/// Deserialization revalidates the nested observation, re-extracts the selected
/// field, recomputes observer identity, shared observation-root identity, claim
/// identity, and candidate identity, and rejects any mismatch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InstrumentedRuntimeEvidenceCandidateV1 {
    schema_version: u16,
    candidate_profile: String,
    candidate_profile_digest: String,
    observer_profile: String,
    observer_contract_digest: String,
    observation_commitment: String,
    observation_root_id: String,
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
        let observation_root_id = runtime_observation_root_id_v1_from_receipt(
            &receipt.observer_contract_digest,
            &receipt.observation_commitment,
        );
        let claim = extract_claim(&observation, field);
        let claim_digest = runtime_claim_digest_v1(&claim);
        let candidate_id = runtime_evidence_candidate_id_v1(
            &candidate_profile_digest,
            &observation_root_id,
            &claim_digest,
        );

        Self {
            schema_version: RUNTIME_EVIDENCE_CANDIDATE_SCHEMA_VERSION,
            candidate_profile: RUNTIME_EVIDENCE_CANDIDATE_PROFILE_V1.to_string(),
            candidate_profile_digest,
            observer_profile: receipt.observer_profile,
            observer_contract_digest: receipt.observer_contract_digest,
            observation_commitment: receipt.observation_commitment,
            observation_root_id,
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

    pub fn observation_root_id(&self) -> &str {
        &self.observation_root_id
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

    /// Recompute the generic lineage fragment for this candidate.
    ///
    /// The candidate itself is **not** a root observation. It is a deterministic
    /// transformation of the exact frozen observation event represented by the
    /// shared `observation_root_id`.
    pub fn lineage_fragment(&self) -> Result<RuntimeCandidateLineageV1, CognitiveLineageError> {
        let observation_root = EvidenceLineageNodeV1 {
            schema_version: COGNITIVE_LINEAGE_SCHEMA_VERSION,
            evidence_id: self.observation_root_id.clone(),
            parent_ids: Vec::new(),
            derivation_kind: CognitiveDerivationKindV1::RootObservation,
        }
        .validate()?;

        let candidate_node = EvidenceLineageNodeV1 {
            schema_version: COGNITIVE_LINEAGE_SCHEMA_VERSION,
            evidence_id: self.candidate_id.clone(),
            parent_ids: vec![self.observation_root_id.clone()],
            derivation_kind: CognitiveDerivationKindV1::Transformation,
        }
        .validate()?;

        Ok(RuntimeCandidateLineageV1 {
            observation_root,
            candidate_node,
        })
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
    observation_root_id: String,
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
        validate_digest(&wire.observation_root_id).map_err(serde::de::Error::custom)?;
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

        let expected_observation_root_id = runtime_observation_root_id_v1_from_receipt(
            &receipt.observer_contract_digest,
            &receipt.observation_commitment,
        );
        if wire.observation_root_id != expected_observation_root_id {
            return Err(serde::de::Error::custom(
                RuntimeEvidenceCandidateError::ObservationRootIdentityMismatch,
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
            &expected_observation_root_id,
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
            observation_root_id: wire.observation_root_id,
            field: wire.field,
            claim: wire.claim,
            claim_digest: wire.claim_digest,
            candidate_id: wire.candidate_id,
            observation: wire.observation,
        })
    }
}

pub fn runtime_observation_root_profile_digest_v1() -> String {
    domain_hash(
        ROOT_PROFILE_DOMAIN,
        RUNTIME_OBSERVATION_ROOT_CONTRACT_V1.as_bytes(),
    )
}

pub fn runtime_evidence_candidate_profile_digest_v1() -> String {
    domain_hash(
        PROFILE_DOMAIN,
        RUNTIME_EVIDENCE_CANDIDATE_CONTRACT_V1.as_bytes(),
    )
}

fn runtime_observation_root_id_v1_from_receipt(
    observer_contract_digest: &str,
    observation_commitment: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(ROOT_ID_DOMAIN);
    hash_field(
        &mut hasher,
        b"root_profile_digest",
        runtime_observation_root_profile_digest_v1().as_bytes(),
    );
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
    format!("blake3:{}", hasher.finalize().to_hex())
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
    observation_root_id: &str,
    claim_digest: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CANDIDATE_DOMAIN);
    hash_field(&mut hasher, b"profile_digest", profile_digest.as_bytes());
    hash_field(
        &mut hasher,
        b"observation_root_id",
        observation_root_id.as_bytes(),
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
    ObservationRootIdentityMismatch,
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
            Self::ObservationRootIdentityMismatch => {
                f.write_str("runtime-evidence candidate does not match its shared observation-root identity")
            }
            Self::ClaimProjectionMismatch => {
                f.write_str("runtime-evidence claim is not the declared lossless observation-field projection")
            }
            Self::ClaimDigestMismatch => {
                f.write_str("runtime-evidence claim digest does not match the extracted claim")
            }
            Self::CandidateIdentityMismatch => {
                f.write_str("runtime-evidence candidate id does not match observation-root and claim identity")
            }
        }
    }
}

impl std::error::Error for RuntimeEvidenceCandidateError {}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_epistemic_governance::lineage::{
        EvidenceIndependenceV1, EvidenceLineageGraphV1,
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

    fn observation_at_cycle(cycle_index: u64) -> ValidatedFrozenCycleObservationV1 {
        let mut raw = observation().as_raw().clone();
        raw.cycle_index = cycle_index;
        raw.validate().unwrap()
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
    fn different_fields_are_different_candidates_but_share_observation_root() {
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
        assert_eq!(pe.observation_root_id(), learning.observation_root_id());
    }

    #[test]
    fn observation_identity_is_candidate_and_root_identity_bearing() {
        let a = InstrumentedRuntimeEvidenceCandidateV1::new(
            observation(),
            ShadowObservationFieldV1::PredictionErrorPpm,
        );
        let b = InstrumentedRuntimeEvidenceCandidateV1::new(
            observation_at_cycle(10),
            ShadowObservationFieldV1::PredictionErrorPpm,
        );
        assert_ne!(a.observation_commitment(), b.observation_commitment());
        assert_ne!(a.observation_root_id(), b.observation_root_id());
        assert_ne!(a.candidate_id(), b.candidate_id());
    }

    #[test]
    fn candidate_id_is_not_root_observation_or_claim_identity() {
        let candidate = InstrumentedRuntimeEvidenceCandidateV1::new(
            observation(),
            ShadowObservationFieldV1::PredictionErrorPpm,
        );
        assert_ne!(candidate.candidate_id(), candidate.observation_root_id());
        assert_ne!(candidate.candidate_id(), candidate.observation_commitment());
        assert_ne!(candidate.candidate_id(), candidate.claim_digest());
    }

    #[test]
    fn persistence_revalidates_complete_projection_and_shared_root() {
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
    fn tampered_observation_root_fails_closed() {
        let candidate = InstrumentedRuntimeEvidenceCandidateV1::new(
            observation(),
            ShadowObservationFieldV1::PredictionErrorPpm,
        );
        let mut value = serde_json::to_value(&candidate).unwrap();
        value["observation_root_id"] = serde_json::Value::String(SHA_A.into());
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
    fn lineage_fragment_makes_candidate_a_transformation_child() {
        let candidate = InstrumentedRuntimeEvidenceCandidateV1::new(
            observation(),
            ShadowObservationFieldV1::LearningOccurred,
        );
        let lineage = candidate.lineage_fragment().unwrap();
        assert_eq!(lineage.observation_root().evidence_id(), candidate.observation_root_id());
        assert_eq!(
            lineage.observation_root().derivation_kind(),
            CognitiveDerivationKindV1::RootObservation
        );
        assert!(lineage.observation_root().parent_ids().is_empty());
        assert_eq!(lineage.candidate_node().evidence_id(), candidate.candidate_id());
        assert_eq!(
            lineage.candidate_node().derivation_kind(),
            CognitiveDerivationKindV1::Transformation
        );
        assert_eq!(
            lineage.candidate_node().parent_ids(),
            &[candidate.observation_root_id().to_string()]
        );
    }

    #[test]
    fn same_observation_field_candidates_are_not_independent() {
        let pe = InstrumentedRuntimeEvidenceCandidateV1::new(
            observation(),
            ShadowObservationFieldV1::PredictionErrorPpm,
        );
        let learning = InstrumentedRuntimeEvidenceCandidateV1::new(
            observation(),
            ShadowObservationFieldV1::LearningOccurred,
        );
        let pe_lineage = pe.lineage_fragment().unwrap();
        let learning_lineage = learning.lineage_fragment().unwrap();

        let graph = EvidenceLineageGraphV1 {
            schema_version: COGNITIVE_LINEAGE_SCHEMA_VERSION,
            graph_id: SHA_B.into(),
            nodes: vec![
                pe_lineage.observation_root().clone(),
                pe_lineage.candidate_node().clone(),
                learning_lineage.candidate_node().clone(),
            ],
        }
        .validate()
        .unwrap();

        assert_eq!(
            graph
                .assess_independence(pe.candidate_id(), learning.candidate_id())
                .unwrap(),
            EvidenceIndependenceV1::SameRoot
        );
    }

    #[test]
    fn different_observation_events_can_be_independent() {
        let a = InstrumentedRuntimeEvidenceCandidateV1::new(
            observation_at_cycle(9),
            ShadowObservationFieldV1::PredictionErrorPpm,
        );
        let b = InstrumentedRuntimeEvidenceCandidateV1::new(
            observation_at_cycle(10),
            ShadowObservationFieldV1::PredictionErrorPpm,
        );
        let a_lineage = a.lineage_fragment().unwrap();
        let b_lineage = b.lineage_fragment().unwrap();
        let graph = EvidenceLineageGraphV1 {
            schema_version: COGNITIVE_LINEAGE_SCHEMA_VERSION,
            graph_id: SHA_A.into(),
            nodes: vec![
                a_lineage.observation_root().clone(),
                a_lineage.candidate_node().clone(),
                b_lineage.observation_root().clone(),
                b_lineage.candidate_node().clone(),
            ],
        }
        .validate()
        .unwrap();

        assert_eq!(
            graph.assess_independence(a.candidate_id(), b.candidate_id()).unwrap(),
            EvidenceIndependenceV1::Independent
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
    fn observation_root_profile_has_strict_identity() {
        let digest = runtime_observation_root_profile_digest_v1();
        assert!(digest.starts_with("blake3:"));
        assert_eq!(digest.len(), "blake3:".len() + 64);
        assert_eq!(digest, runtime_observation_root_profile_digest_v1());
    }

    #[test]
    fn candidate_profile_has_strict_identity() {
        let digest = runtime_evidence_candidate_profile_digest_v1();
        assert!(digest.starts_with("blake3:"));
        assert_eq!(digest.len(), "blake3:".len() + 64);
        assert_eq!(digest, runtime_evidence_candidate_profile_digest_v1());
    }
}
