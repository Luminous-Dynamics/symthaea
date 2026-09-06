// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! RCA-003b.3d: preregister the exact shadow-disposition evaluation surface.
//!
//! The base/effective disposition policies predate the raw and canonical-lineage
//! preflight contracts. This wrapper makes those exact profile semantics
//! preregistration-bearing before any result-bearing disposition engine exists.
//!
//! It accepts no case, preflight, witness, or lineage instance.

#![deny(unsafe_code)]

use serde::{Deserialize, Deserializer, Serialize};
use symthaea_rca_effective_disposition_policy::RegisteredEffectiveShadowDispositionPolicyV1;
use symthaea_rca_lineage_bound_disposition_preflight::lineage_bound_preflight_profile_digest_v1;
use symthaea_rca_shadow_disposition_preflight::shadow_disposition_preflight_profile_digest_v1;

pub const SHADOW_DISPOSITION_EVALUATION_POLICY_SCHEMA_VERSION: u16 = 1;
pub const SHADOW_DISPOSITION_EVALUATION_POLICY_PROFILE_V1: &str =
    "rca-shadow-disposition-evaluation-policy-v1";

pub const SHADOW_DISPOSITION_EVALUATION_POLICY_CONTRACT_V1: &str = concat!(
    "rca-shadow-disposition-evaluation-policy-v1\n",
    "input=registered_effective_shadow_disposition_policy_v1_only\n",
    "binding=exact_effective_policy+raw_preflight_profile+canonical_lineage_bound_preflight_profile\n",
    "raw_preflight_profile_is_preregistration_bearing\n",
    "lineage_bound_preflight_profile_is_preregistration_bearing\n",
    "profile_drift_requires_new_evaluation_policy_identity\n",
    "evaluation_policy_id=blake3_explicit_complete_binding_v1\n",
    "persistence=revalidate_effective_policy+current_preflight_profiles+identity\n",
    "registration_precedes_result_bearing_disposition_evaluation\n",
    "evaluation_policy_accepts_no_case_preflight_witness_or_lineage_instance\n",
    "evaluation_policy_is_not_disposition_belief_workspace_action_or_promotion_authority\n",
);

const PROFILE_DOMAIN: &[u8] = b"symthaea:rca-shadow-disposition-evaluation-policy-contract:v1\0";
const POLICY_ID_DOMAIN: &[u8] = b"symthaea:rca-shadow-disposition-evaluation-policy:v1\0";

/// Persistable preregistration artifact for the exact evaluation surface that a
/// later pure shadow-disposition engine may consume.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RegisteredShadowDispositionEvaluationPolicyV1 {
    schema_version: u16,
    profile: String,
    profile_contract_digest: String,
    evaluation_policy_id: String,
    raw_preflight_profile_digest: String,
    lineage_bound_preflight_profile_digest: String,
    effective_policy: RegisteredEffectiveShadowDispositionPolicyV1,
}

impl RegisteredShadowDispositionEvaluationPolicyV1 {
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub fn profile(&self) -> &str {
        &self.profile
    }

    pub fn profile_contract_digest(&self) -> &str {
        &self.profile_contract_digest
    }

    pub fn evaluation_policy_id(&self) -> &str {
        &self.evaluation_policy_id
    }

    pub fn raw_preflight_profile_digest(&self) -> &str {
        &self.raw_preflight_profile_digest
    }

    pub fn lineage_bound_preflight_profile_digest(&self) -> &str {
        &self.lineage_bound_preflight_profile_digest
    }

    pub fn effective_policy(&self) -> &RegisteredEffectiveShadowDispositionPolicyV1 {
        &self.effective_policy
    }
}

pub fn shadow_disposition_evaluation_policy_profile_digest_v1() -> String {
    domain_hash(
        PROFILE_DOMAIN,
        SHADOW_DISPOSITION_EVALUATION_POLICY_CONTRACT_V1.as_bytes(),
    )
}

pub fn register_shadow_disposition_evaluation_policy_v1(
    effective_policy: RegisteredEffectiveShadowDispositionPolicyV1,
) -> RegisteredShadowDispositionEvaluationPolicyV1 {
    let profile_contract_digest = shadow_disposition_evaluation_policy_profile_digest_v1();
    let raw_preflight_profile_digest = shadow_disposition_preflight_profile_digest_v1();
    let lineage_bound_preflight_profile_digest = lineage_bound_preflight_profile_digest_v1();
    let evaluation_policy_id = evaluation_policy_id_v1(
        &profile_contract_digest,
        &effective_policy,
        &raw_preflight_profile_digest,
        &lineage_bound_preflight_profile_digest,
    );

    RegisteredShadowDispositionEvaluationPolicyV1 {
        schema_version: SHADOW_DISPOSITION_EVALUATION_POLICY_SCHEMA_VERSION,
        profile: SHADOW_DISPOSITION_EVALUATION_POLICY_PROFILE_V1.to_string(),
        profile_contract_digest,
        evaluation_policy_id,
        raw_preflight_profile_digest,
        lineage_bound_preflight_profile_digest,
        effective_policy,
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RegisteredShadowDispositionEvaluationPolicyWireV1 {
    schema_version: u16,
    profile: String,
    profile_contract_digest: String,
    evaluation_policy_id: String,
    raw_preflight_profile_digest: String,
    lineage_bound_preflight_profile_digest: String,
    effective_policy: RegisteredEffectiveShadowDispositionPolicyV1,
}

impl<'de> Deserialize<'de> for RegisteredShadowDispositionEvaluationPolicyV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RegisteredShadowDispositionEvaluationPolicyWireV1::deserialize(deserializer)?;
        if wire.schema_version != SHADOW_DISPOSITION_EVALUATION_POLICY_SCHEMA_VERSION {
            return Err(serde::de::Error::custom(
                ShadowDispositionEvaluationPolicyError::UnsupportedSchemaVersion {
                    found: wire.schema_version,
                },
            ));
        }
        if wire.profile != SHADOW_DISPOSITION_EVALUATION_POLICY_PROFILE_V1 {
            return Err(serde::de::Error::custom(
                ShadowDispositionEvaluationPolicyError::UnexpectedProfile,
            ));
        }
        validate_digest(&wire.profile_contract_digest).map_err(serde::de::Error::custom)?;
        validate_digest(&wire.evaluation_policy_id).map_err(serde::de::Error::custom)?;
        validate_digest(&wire.raw_preflight_profile_digest).map_err(serde::de::Error::custom)?;
        validate_digest(&wire.lineage_bound_preflight_profile_digest)
            .map_err(serde::de::Error::custom)?;

        let expected =
            register_shadow_disposition_evaluation_policy_v1(wire.effective_policy.clone());
        if wire.profile_contract_digest != expected.profile_contract_digest
            || wire.evaluation_policy_id != expected.evaluation_policy_id
            || wire.raw_preflight_profile_digest != expected.raw_preflight_profile_digest
            || wire.lineage_bound_preflight_profile_digest
                != expected.lineage_bound_preflight_profile_digest
            || wire.effective_policy != expected.effective_policy
        {
            return Err(serde::de::Error::custom(
                ShadowDispositionEvaluationPolicyError::EvaluationPolicyIdentityMismatch,
            ));
        }
        Ok(expected)
    }
}

fn evaluation_policy_id_v1(
    profile_contract_digest: &str,
    effective_policy: &RegisteredEffectiveShadowDispositionPolicyV1,
    raw_preflight_profile_digest: &str,
    lineage_bound_preflight_profile_digest: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(POLICY_ID_DOMAIN);
    hash_text(
        &mut hasher,
        b"profile_contract_digest",
        profile_contract_digest,
    );
    hash_bytes(
        &mut hasher,
        b"schema_version",
        &SHADOW_DISPOSITION_EVALUATION_POLICY_SCHEMA_VERSION.to_le_bytes(),
    );
    hash_text(
        &mut hasher,
        b"effective_policy_id",
        effective_policy.effective_policy_id(),
    );
    hash_text(
        &mut hasher,
        b"effective_policy_profile_contract_digest",
        effective_policy.profile_contract_digest(),
    );
    hash_text(
        &mut hasher,
        b"raw_preflight_profile_digest",
        raw_preflight_profile_digest,
    );
    hash_text(
        &mut hasher,
        b"lineage_bound_preflight_profile_digest",
        lineage_bound_preflight_profile_digest,
    );
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn validate_digest(digest: &str) -> Result<(), ShadowDispositionEvaluationPolicyError> {
    let Some((algorithm, hex)) = digest.split_once(':') else {
        return Err(ShadowDispositionEvaluationPolicyError::MalformedDigest);
    };
    if !matches!(algorithm, "sha256" | "blake3")
        || hex.len() != 64
        || !hex.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(ShadowDispositionEvaluationPolicyError::MalformedDigest);
    }
    Ok(())
}

fn hash_text(hasher: &mut blake3::Hasher, label: &[u8], value: &str) {
    hash_bytes(hasher, label, value.as_bytes());
}

fn hash_bytes(hasher: &mut blake3::Hasher, label: &[u8], value: &[u8]) {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShadowDispositionEvaluationPolicyError {
    UnsupportedSchemaVersion { found: u16 },
    UnexpectedProfile,
    MalformedDigest,
    EvaluationPolicyIdentityMismatch,
}

impl std::fmt::Display for ShadowDispositionEvaluationPolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => write!(
                f,
                "unsupported shadow-disposition evaluation-policy schema {found}; expected {SHADOW_DISPOSITION_EVALUATION_POLICY_SCHEMA_VERSION}"
            ),
            Self::UnexpectedProfile => {
                f.write_str("unexpected shadow-disposition evaluation-policy profile")
            }
            Self::MalformedDigest => {
                f.write_str("digest must be sha256:<64 hex> or blake3:<64 hex>")
            }
            Self::EvaluationPolicyIdentityMismatch => {
                f.write_str("shadow-disposition evaluation-policy identity mismatch")
            }
        }
    }
}

impl std::error::Error for ShadowDispositionEvaluationPolicyError {}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_epistemic_governance::experiment_contract::EXPERIMENT_CONTRACT_SCHEMA_VERSION;
    use symthaea_rca_effective_disposition_policy::register_effective_shadow_disposition_policy_v1;
    use symthaea_rca_shadow_disposition_policy::{
        current_bound_case_profile_digest_v1, current_evidence_set_witness_profile_digest_v1,
        current_interpretation_lineage_profile_digest_v1,
        current_relation_eligibility_profile_digest_v1, DefeaterModeV1,
        DispositionPolicyEvaluationBindingV1, DispositionPolicyResourceCeilingsV1,
        EvidenceSetSemanticsV1, InterpretationRootSetSemanticsV1,
        OutcomeTopologyRequirementsV1, RelationStrengthTreatmentV1,
        ShadowDispositionPolicyV1, UnknownInterpretationIndependenceModeV1,
        SHADOW_DISPOSITION_POLICY_SCHEMA_VERSION,
    };

    const PROPOSITION: &str =
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const EXPERIMENT: &str =
        "blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn requirements(evidence: u16, interpretations: u16) -> OutcomeTopologyRequirementsV1 {
        OutcomeTopologyRequirementsV1 {
            min_pairwise_independent_evidence_items: evidence,
            min_pairwise_independent_interpretation_roots: interpretations,
        }
    }

    fn effective_policy() -> RegisteredEffectiveShadowDispositionPolicyV1 {
        let base = ShadowDispositionPolicyV1 {
            schema_version: SHADOW_DISPOSITION_POLICY_SCHEMA_VERSION,
            proposition_id: PROPOSITION.into(),
            bound_case_profile_digest: current_bound_case_profile_digest_v1(),
            evidence_set_witness_profile_digest: current_evidence_set_witness_profile_digest_v1(),
            relation_eligibility_profile_digest: current_relation_eligibility_profile_digest_v1(),
            interpretation_lineage_profile_digest: current_interpretation_lineage_profile_digest_v1(),
            evidence_set_semantics: EvidenceSetSemanticsV1::IssuedPairwiseIndependentItems,
            interpretation_root_set_semantics:
                InterpretationRootSetSemanticsV1::PairwiseIndependentRoots,
            tentative_support_requirements: requirements(1, 1),
            support_requirements: requirements(2, 2),
            tentative_opposition_requirements: requirements(1, 1),
            opposition_requirements: requirements(2, 2),
            defeater_requirements: requirements(1, 1),
            contested_side_requirements: requirements(1, 1),
            strength_treatment: RelationStrengthTreatmentV1::DiagnosticOnly,
            defeater_mode: DefeaterModeV1::QualifiedCurrentBlocker,
            unknown_interpretation_independence_mode:
                UnknownInterpretationIndependenceModeV1::ForceUnderdetermined,
            contested_requires_qualified_support_and_opposition: true,
            evaluation: DispositionPolicyEvaluationBindingV1 {
                experiment_contract_schema_version: EXPERIMENT_CONTRACT_SCHEMA_VERSION,
                registered_experiment_contract_digest: EXPERIMENT.into(),
            },
            resources: DispositionPolicyResourceCeilingsV1 {
                max_case_items: 32,
                max_interpretation_pairs: 496,
            },
        }
        .register()
        .unwrap();
        register_effective_shadow_disposition_policy_v1(base)
    }

    #[test]
    fn evaluation_policy_binds_both_preflight_profiles() {
        let registered = register_shadow_disposition_evaluation_policy_v1(effective_policy());
        assert_eq!(
            registered.raw_preflight_profile_digest(),
            shadow_disposition_preflight_profile_digest_v1()
        );
        assert_eq!(
            registered.lineage_bound_preflight_profile_digest(),
            lineage_bound_preflight_profile_digest_v1()
        );
        assert!(registered.evaluation_policy_id().starts_with("blake3:"));
    }

    #[test]
    fn effective_policy_identity_is_evaluation_policy_bearing() {
        let first = register_shadow_disposition_evaluation_policy_v1(effective_policy());
        let mut changed_base = effective_policy().base_policy().policy().clone();
        changed_base.resources.max_case_items = 64;
        let second_effective = register_effective_shadow_disposition_policy_v1(
            changed_base.register().unwrap(),
        );
        let second = register_shadow_disposition_evaluation_policy_v1(second_effective);
        assert_ne!(first.evaluation_policy_id(), second.evaluation_policy_id());
    }

    #[test]
    fn evaluation_policy_revalidates_after_persistence() {
        let registered = register_shadow_disposition_evaluation_policy_v1(effective_policy());
        let encoded = serde_json::to_string(&registered).unwrap();
        let decoded: RegisteredShadowDispositionEvaluationPolicyV1 =
            serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, registered);
    }

    #[test]
    fn tampered_raw_preflight_profile_fails_closed() {
        let registered = register_shadow_disposition_evaluation_policy_v1(effective_policy());
        let mut value = serde_json::to_value(&registered).unwrap();
        value["raw_preflight_profile_digest"] = serde_json::Value::String(
            "blake3:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
                .into(),
        );
        assert!(
            serde_json::from_value::<RegisteredShadowDispositionEvaluationPolicyV1>(value).is_err()
        );
    }

    #[test]
    fn tampered_lineage_bound_preflight_profile_fails_closed() {
        let registered = register_shadow_disposition_evaluation_policy_v1(effective_policy());
        let mut value = serde_json::to_value(&registered).unwrap();
        value["lineage_bound_preflight_profile_digest"] = serde_json::Value::String(
            "blake3:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
                .into(),
        );
        assert!(
            serde_json::from_value::<RegisteredShadowDispositionEvaluationPolicyV1>(value).is_err()
        );
    }
}
