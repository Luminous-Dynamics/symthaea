// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Shadow-only relevance policy for instrumented runtime evidence candidates.
//!
//! A historical runtime observation may be perfectly authentic while being
//! irrelevant to the current cognitive execution. RCA must not turn a source
//! digest into a numeric epoch or assume that "historically true" means "true
//! of this execution now".
//!
//! This crate therefore performs one narrow assessment:
//!
//! ```text
//! instrumented runtime evidence candidate
//!         +
//! explicit current execution/use context
//!         ↓
//! current-runtime relevance assessment
//! ```
//!
//! Relevance is still not canonical evidence admission, proposition support,
//! belief/workspace admission, action authority, or self-improvement promotion.

#![deny(unsafe_code)]

use serde::{Deserialize, Deserializer, Serialize};
use symthaea_rca_evidence_bridge::InstrumentedRuntimeEvidenceCandidateV1;

pub const RUNTIME_RELEVANCE_SCHEMA_VERSION: u16 = 1;
pub const RUNTIME_RELEVANCE_PROFILE_V1: &str = "rca-current-runtime-relevance-v1";

/// Normative current-runtime relevance semantics.
///
/// Exact content identities remain exact content identities; no digest is
/// truncated or reinterpreted as a numeric generation. Cycle freshness is
/// evaluated only inside the exact execution lineage and against an explicit
/// caller-supplied maximum lag.
pub const RUNTIME_RELEVANCE_CONTRACT_V1: &str = concat!(
    "rca-current-runtime-relevance-v1\n",
    "purpose=shadow_only_current_execution_relevance\n",
    "source_identity=exact_digest_equality\n",
    "execution_identity=exact_digest_equality\n",
    "adapter_profile=exact_string_equality\n",
    "adapter_contract=exact_digest_equality\n",
    "cycle_comparison_requires_exact_execution_lineage\n",
    "cycle_rule=observed_cycle_must_not_exceed_current_cycle\n",
    "lag_rule=current_cycle-observed_cycle<=explicit_max_cycle_lag\n",
    "all_semantically_valid_detected_defects_are_preserved\n",
    "historical_truth_is_not_current_runtime_relevance\n",
    "assessment_is_issued_non_deserializable_shadow_result\n",
    "persistence_does_not_recreate_relevance_assessment_authority\n",
    "relevance_is_not_evidence_admission_or_proposition_support\n",
    "relevance_is_not_workspace_action_or_self_improvement_authority\n",
);

const PROFILE_DOMAIN: &[u8] = b"symthaea:rca-current-runtime-relevance-contract:v1\0";
const CONTEXT_DOMAIN: &[u8] = b"symthaea:rca-current-runtime-relevance-context:v1\0";

/// Explicit use context against which one runtime observation is evaluated.
///
/// `max_cycle_lag` is policy, not a universal property of evidence. Callers must
/// choose/freeze it for the proposition/use under evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CurrentRuntimeRelevanceContextV1 {
    pub schema_version: u16,
    pub source_generation_digest: String,
    pub execution_lineage_digest: String,
    pub adapter_profile: String,
    pub adapter_contract_digest: String,
    pub current_cycle_index: u64,
    pub max_cycle_lag: u64,
}

impl CurrentRuntimeRelevanceContextV1 {
    pub fn validate(
        self,
    ) -> Result<ValidatedCurrentRuntimeRelevanceContextV1, RuntimeRelevanceError> {
        ValidatedCurrentRuntimeRelevanceContextV1::try_from(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ValidatedCurrentRuntimeRelevanceContextV1(CurrentRuntimeRelevanceContextV1);

impl ValidatedCurrentRuntimeRelevanceContextV1 {
    pub fn as_raw(&self) -> &CurrentRuntimeRelevanceContextV1 {
        &self.0
    }

    pub fn commitment(&self) -> String {
        context_commitment_v1(&self.0)
    }
}

impl TryFrom<CurrentRuntimeRelevanceContextV1> for ValidatedCurrentRuntimeRelevanceContextV1 {
    type Error = RuntimeRelevanceError;

    fn try_from(value: CurrentRuntimeRelevanceContextV1) -> Result<Self, Self::Error> {
        if value.schema_version != RUNTIME_RELEVANCE_SCHEMA_VERSION {
            return Err(RuntimeRelevanceError::UnsupportedSchemaVersion {
                found: value.schema_version,
            });
        }
        validate_digest(&value.source_generation_digest)?;
        validate_digest(&value.execution_lineage_digest)?;
        validate_digest(&value.adapter_contract_digest)?;
        if value.adapter_profile.trim().is_empty() {
            return Err(RuntimeRelevanceError::MissingAdapterProfile);
        }
        Ok(Self(value))
    }
}

impl<'de> Deserialize<'de> for ValidatedCurrentRuntimeRelevanceContextV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        CurrentRuntimeRelevanceContextV1::deserialize(deserializer)?
            .validate()
            .map_err(serde::de::Error::custom)
    }
}

/// Every reason a candidate fails to describe the requested current execution.
///
/// The assessment preserves all independent defects it can determine rather
/// than stopping at the first mismatch. Cycle-age defects are meaningful only
/// when the execution lineage itself matches.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RuntimeRelevanceDefectV1 {
    SourceGenerationMismatch {
        observed: String,
        current: String,
    },
    ExecutionLineageMismatch {
        observed: String,
        current: String,
    },
    AdapterProfileMismatch {
        observed: String,
        current: String,
    },
    AdapterContractMismatch {
        observed: String,
        current: String,
    },
    FutureObservation {
        observed_cycle: u64,
        current_cycle: u64,
    },
    StaleByCycleLag {
        lag: u64,
        max_cycle_lag: u64,
    },
}

/// Issued result of one pure shadow relevance assessment.
///
/// The fields are private and this type deliberately does not implement
/// `Deserialize`. An archived JSON representation is audit data only; loading
/// bytes must not recreate a trusted "relevant" result. Downstream code that
/// needs a current relevance decision must retain/reload the candidate and
/// validated context and call [`assess_current_runtime_relevance`] again.
#[must_use = "runtime relevance assessments are shadow evidence and should be inspected"]
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RuntimeRelevanceAssessmentV1 {
    schema_version: u16,
    profile: String,
    profile_contract_digest: String,
    candidate_id: String,
    context_commitment: String,
    observed_cycle_index: u64,
    current_cycle_index: u64,
    defects: Vec<RuntimeRelevanceDefectV1>,
}

impl RuntimeRelevanceAssessmentV1 {
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub fn profile(&self) -> &str {
        &self.profile
    }

    pub fn profile_contract_digest(&self) -> &str {
        &self.profile_contract_digest
    }

    pub fn candidate_id(&self) -> &str {
        &self.candidate_id
    }

    pub fn context_commitment(&self) -> &str {
        &self.context_commitment
    }

    pub const fn observed_cycle_index(&self) -> u64 {
        self.observed_cycle_index
    }

    pub const fn current_cycle_index(&self) -> u64 {
        self.current_cycle_index
    }

    pub fn defects(&self) -> &[RuntimeRelevanceDefectV1] {
        &self.defects
    }

    pub fn is_relevant(&self) -> bool {
        self.defects.is_empty()
    }

    /// Cycle lag exists only when both cycles belong to the same execution
    /// lineage and the observation is not from the future.
    pub fn cycle_lag(&self) -> Option<u64> {
        if self.defects.iter().any(|defect| {
            matches!(
                defect,
                RuntimeRelevanceDefectV1::ExecutionLineageMismatch { .. }
                    | RuntimeRelevanceDefectV1::FutureObservation { .. }
            )
        }) {
            return None;
        }
        self.current_cycle_index
            .checked_sub(self.observed_cycle_index)
    }
}

pub fn runtime_relevance_profile_digest_v1() -> String {
    domain_hash(PROFILE_DOMAIN, RUNTIME_RELEVANCE_CONTRACT_V1.as_bytes())
}

/// Pure shadow-only assessment of whether a candidate can describe the current
/// RCA execution under one explicit cycle-lag policy.
///
/// This function is the only public issuance path for
/// [`RuntimeRelevanceAssessmentV1`]. It does not convert the candidate to
/// canonical evidence, attach a proposition relation, or return any
/// authority-bearing capability.
pub fn assess_current_runtime_relevance(
    candidate: &InstrumentedRuntimeEvidenceCandidateV1,
    context: &ValidatedCurrentRuntimeRelevanceContextV1,
) -> RuntimeRelevanceAssessmentV1 {
    let observed = candidate.observation().as_raw();
    let current = context.as_raw();
    let mut defects = Vec::new();

    if observed.source_generation_digest != current.source_generation_digest {
        defects.push(RuntimeRelevanceDefectV1::SourceGenerationMismatch {
            observed: observed.source_generation_digest.clone(),
            current: current.source_generation_digest.clone(),
        });
    }

    let same_execution_lineage =
        observed.execution_lineage_digest == current.execution_lineage_digest;
    if !same_execution_lineage {
        defects.push(RuntimeRelevanceDefectV1::ExecutionLineageMismatch {
            observed: observed.execution_lineage_digest.clone(),
            current: current.execution_lineage_digest.clone(),
        });
    }

    if observed.adapter_profile != current.adapter_profile {
        defects.push(RuntimeRelevanceDefectV1::AdapterProfileMismatch {
            observed: observed.adapter_profile.clone(),
            current: current.adapter_profile.clone(),
        });
    }
    if observed.adapter_contract_digest != current.adapter_contract_digest {
        defects.push(RuntimeRelevanceDefectV1::AdapterContractMismatch {
            observed: observed.adapter_contract_digest.clone(),
            current: current.adapter_contract_digest.clone(),
        });
    }

    // A cycle index is scoped to its execution lineage. Never compare cycle
    // numbers across different lineages, even if the integers happen to match.
    if same_execution_lineage {
        match current.current_cycle_index.checked_sub(observed.cycle_index) {
            None => defects.push(RuntimeRelevanceDefectV1::FutureObservation {
                observed_cycle: observed.cycle_index,
                current_cycle: current.current_cycle_index,
            }),
            Some(lag) if lag > current.max_cycle_lag => {
                defects.push(RuntimeRelevanceDefectV1::StaleByCycleLag {
                    lag,
                    max_cycle_lag: current.max_cycle_lag,
                });
            }
            Some(_) => {}
        }
    }

    RuntimeRelevanceAssessmentV1 {
        schema_version: RUNTIME_RELEVANCE_SCHEMA_VERSION,
        profile: RUNTIME_RELEVANCE_PROFILE_V1.to_string(),
        profile_contract_digest: runtime_relevance_profile_digest_v1(),
        candidate_id: candidate.candidate_id().to_string(),
        context_commitment: context.commitment(),
        observed_cycle_index: observed.cycle_index,
        current_cycle_index: current.current_cycle_index,
        defects,
    }
}

fn context_commitment_v1(context: &CurrentRuntimeRelevanceContextV1) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CONTEXT_DOMAIN);
    hash_field(
        &mut hasher,
        b"profile_contract_digest",
        runtime_relevance_profile_digest_v1().as_bytes(),
    );
    hash_field(
        &mut hasher,
        b"schema_version",
        &context.schema_version.to_le_bytes(),
    );
    hash_field(
        &mut hasher,
        b"source_generation_digest",
        context.source_generation_digest.as_bytes(),
    );
    hash_field(
        &mut hasher,
        b"execution_lineage_digest",
        context.execution_lineage_digest.as_bytes(),
    );
    hash_field(
        &mut hasher,
        b"adapter_profile",
        context.adapter_profile.as_bytes(),
    );
    hash_field(
        &mut hasher,
        b"adapter_contract_digest",
        context.adapter_contract_digest.as_bytes(),
    );
    hash_field(
        &mut hasher,
        b"current_cycle_index",
        &context.current_cycle_index.to_le_bytes(),
    );
    hash_field(
        &mut hasher,
        b"max_cycle_lag",
        &context.max_cycle_lag.to_le_bytes(),
    );
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn hash_field(hasher: &mut blake3::Hasher, label: &[u8], value: &[u8]) {
    hasher.update(&(label.len() as u64).to_le_bytes());
    hasher.update(label);
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn domain_hash(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn validate_digest(digest: &str) -> Result<(), RuntimeRelevanceError> {
    let Some((algorithm, hex)) = digest.split_once(':') else {
        return Err(RuntimeRelevanceError::MalformedDigest);
    };
    if !matches!(algorithm, "sha256" | "blake3")
        || hex.len() != 64
        || !hex.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(RuntimeRelevanceError::MalformedDigest);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeRelevanceError {
    UnsupportedSchemaVersion { found: u16 },
    MalformedDigest,
    MissingAdapterProfile,
}

impl std::fmt::Display for RuntimeRelevanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => write!(
                f,
                "unsupported runtime-relevance schema version {found}; expected {RUNTIME_RELEVANCE_SCHEMA_VERSION}"
            ),
            Self::MalformedDigest => {
                f.write_str("digest must be sha256:<64 hex> or blake3:<64 hex>")
            }
            Self::MissingAdapterProfile => {
                f.write_str("current runtime relevance requires an explicit adapter profile")
            }
        }
    }
}

impl std::error::Error for RuntimeRelevanceError {}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_rca_evidence_bridge::{
        InstrumentedRuntimeEvidenceCandidateV1, ShadowObservationFieldV1,
    };
    use symthaea_rca_shadow::{
        FrozenCycleObservationV1, FROZEN_CYCLE_OBSERVATION_SCHEMA_VERSION,
    };

    const SOURCE_A: &str =
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SOURCE_B: &str =
        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const LINEAGE_A: &str =
        "blake3:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const LINEAGE_B: &str =
        "blake3:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const ADAPTER_A: &str =
        "blake3:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const ADAPTER_B: &str =
        "blake3:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    fn candidate(cycle: u64) -> InstrumentedRuntimeEvidenceCandidateV1 {
        let observation = FrozenCycleObservationV1 {
            schema_version: FROZEN_CYCLE_OBSERVATION_SCHEMA_VERSION,
            source_generation_digest: SOURCE_A.into(),
            execution_lineage_digest: LINEAGE_A.into(),
            adapter_profile: "adapter-v1".into(),
            adapter_contract_digest: ADAPTER_A.into(),
            cycle_index: cycle,
            cycle_time_us: 10_000,
            prediction_error_ppm: 200_000,
            peak_attention_bits: 1.5_f32.to_bits(),
            learning_occurred: false,
            detected_primitive_count: 2,
            output_digest: SOURCE_B.into(),
            thought_digest: LINEAGE_B.into(),
            metadata_digest: ADAPTER_B.into(),
            language_output_digest: None,
            language_source: None,
        }
        .validate()
        .unwrap();
        InstrumentedRuntimeEvidenceCandidateV1::new(
            observation,
            ShadowObservationFieldV1::PredictionErrorPpm,
        )
    }

    fn context(current_cycle: u64, max_lag: u64) -> ValidatedCurrentRuntimeRelevanceContextV1 {
        CurrentRuntimeRelevanceContextV1 {
            schema_version: RUNTIME_RELEVANCE_SCHEMA_VERSION,
            source_generation_digest: SOURCE_A.into(),
            execution_lineage_digest: LINEAGE_A.into(),
            adapter_profile: "adapter-v1".into(),
            adapter_contract_digest: ADAPTER_A.into(),
            current_cycle_index: current_cycle,
            max_cycle_lag: max_lag,
        }
        .validate()
        .unwrap()
    }

    #[test]
    fn exact_current_execution_within_lag_is_relevant() {
        let assessment = assess_current_runtime_relevance(&candidate(40), &context(42, 2));
        assert!(assessment.is_relevant());
        assert_eq!(assessment.cycle_lag(), Some(2));
        assert_eq!(
            assessment.profile_contract_digest(),
            runtime_relevance_profile_digest_v1()
        );
    }

    #[test]
    fn zero_lag_policy_requires_same_cycle() {
        assert!(assess_current_runtime_relevance(&candidate(42), &context(42, 0)).is_relevant());
        let stale = assess_current_runtime_relevance(&candidate(41), &context(42, 0));
        assert_eq!(
            stale.defects(),
            &[RuntimeRelevanceDefectV1::StaleByCycleLag {
                lag: 1,
                max_cycle_lag: 0
            }]
        );
    }

    #[test]
    fn source_digest_is_compared_exactly_not_as_numeric_epoch() {
        let mut raw = context(42, 2).as_raw().clone();
        raw.source_generation_digest = SOURCE_B.into();
        let assessment =
            assess_current_runtime_relevance(&candidate(42), &raw.validate().unwrap());
        assert!(assessment.defects().iter().any(|defect| matches!(
            defect,
            RuntimeRelevanceDefectV1::SourceGenerationMismatch { .. }
        )));
    }

    #[test]
    fn same_source_but_different_execution_is_not_currently_relevant() {
        let mut raw = context(42, 2).as_raw().clone();
        raw.execution_lineage_digest = LINEAGE_B.into();
        let assessment =
            assess_current_runtime_relevance(&candidate(42), &raw.validate().unwrap());
        assert!(assessment.defects().iter().any(|defect| matches!(
            defect,
            RuntimeRelevanceDefectV1::ExecutionLineageMismatch { .. }
        )));
        assert_eq!(assessment.cycle_lag(), None);
    }

    #[test]
    fn different_lineage_never_compares_cycle_indices() {
        let mut raw = context(1_000, 0).as_raw().clone();
        raw.execution_lineage_digest = LINEAGE_B.into();
        let assessment =
            assess_current_runtime_relevance(&candidate(1), &raw.validate().unwrap());
        assert_eq!(
            assessment.defects(),
            &[RuntimeRelevanceDefectV1::ExecutionLineageMismatch {
                observed: LINEAGE_A.into(),
                current: LINEAGE_B.into(),
            }]
        );
        assert_eq!(assessment.cycle_lag(), None);
    }

    #[test]
    fn adapter_semantics_are_relevance_bearing() {
        let mut raw = context(42, 2).as_raw().clone();
        raw.adapter_profile = "adapter-v2".into();
        raw.adapter_contract_digest = ADAPTER_B.into();
        let assessment =
            assess_current_runtime_relevance(&candidate(42), &raw.validate().unwrap());
        assert!(assessment.defects().iter().any(|defect| matches!(
            defect,
            RuntimeRelevanceDefectV1::AdapterProfileMismatch { .. }
        )));
        assert!(assessment.defects().iter().any(|defect| matches!(
            defect,
            RuntimeRelevanceDefectV1::AdapterContractMismatch { .. }
        )));
    }

    #[test]
    fn stale_observation_fails_explicit_lag_policy() {
        let assessment = assess_current_runtime_relevance(&candidate(30), &context(42, 5));
        assert_eq!(
            assessment.defects(),
            &[RuntimeRelevanceDefectV1::StaleByCycleLag {
                lag: 12,
                max_cycle_lag: 5
            }]
        );
    }

    #[test]
    fn future_observation_fails_closed() {
        let assessment = assess_current_runtime_relevance(&candidate(43), &context(42, 5));
        assert_eq!(
            assessment.defects(),
            &[RuntimeRelevanceDefectV1::FutureObservation {
                observed_cycle: 43,
                current_cycle: 42
            }]
        );
        assert_eq!(assessment.cycle_lag(), None);
    }

    #[test]
    fn independent_identity_defects_are_preserved_without_cross_lineage_lag_math() {
        let mut raw = context(42, 1).as_raw().clone();
        raw.source_generation_digest = SOURCE_B.into();
        raw.execution_lineage_digest = LINEAGE_B.into();
        raw.adapter_profile = "adapter-v2".into();
        raw.adapter_contract_digest = ADAPTER_B.into();
        let assessment =
            assess_current_runtime_relevance(&candidate(30), &raw.validate().unwrap());
        assert_eq!(assessment.defects().len(), 4);
        assert!(assessment.defects().iter().all(|defect| !matches!(
            defect,
            RuntimeRelevanceDefectV1::StaleByCycleLag { .. }
                | RuntimeRelevanceDefectV1::FutureObservation { .. }
        )));
        assert!(!assessment.is_relevant());
    }

    #[test]
    fn context_commitment_binds_lag_policy() {
        assert_ne!(context(42, 1).commitment(), context(42, 2).commitment());
    }

    #[test]
    fn relevance_profile_has_strict_identity() {
        let digest = runtime_relevance_profile_digest_v1();
        assert!(digest.starts_with("blake3:"));
        assert_eq!(digest.len(), "blake3:".len() + 64);
        assert_eq!(digest, runtime_relevance_profile_digest_v1());
    }

    #[test]
    fn issued_assessment_serializes_for_audit_but_is_not_a_persisted_capability() {
        let assessment = assess_current_runtime_relevance(&candidate(40), &context(42, 2));
        let encoded = serde_json::to_string(&assessment).unwrap();
        assert!(encoded.contains(assessment.candidate_id()));
        assert!(encoded.contains(assessment.profile_contract_digest()));
    }

    #[test]
    fn validated_context_revalidates_after_persistence() {
        let validated = context(42, 2);
        let encoded = serde_json::to_string(&validated).unwrap();
        let decoded: ValidatedCurrentRuntimeRelevanceContextV1 =
            serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, validated);
    }

    #[test]
    fn malformed_context_digest_fails_closed() {
        let raw = CurrentRuntimeRelevanceContextV1 {
            schema_version: RUNTIME_RELEVANCE_SCHEMA_VERSION,
            source_generation_digest: "generation-7".into(),
            execution_lineage_digest: LINEAGE_A.into(),
            adapter_profile: "adapter-v1".into(),
            adapter_contract_digest: ADAPTER_A.into(),
            current_cycle_index: 42,
            max_cycle_lag: 2,
        };
        assert_eq!(raw.validate(), Err(RuntimeRelevanceError::MalformedDigest));
    }
}
