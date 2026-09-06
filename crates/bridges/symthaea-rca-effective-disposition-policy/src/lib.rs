// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! RCA-003b.3a: effective preregistered shadow-disposition policy binding.
//!
//! `RegisteredShadowDispositionPolicyV1` predates the independent interpretation
//! root-set witness contract. This crate does not mutate that reviewed policy
//! body. Instead it wraps the registered policy and binds the exact current
//! interpretation-set-witness profile into a new effective policy identity.
//!
//! The resulting artifact is still policy only. It evaluates no case, issues no
//! disposition, and grants no belief, workspace, action, or promotion authority.

#![deny(unsafe_code)]

use serde::{Deserialize, Deserializer, Serialize};
use symthaea_epistemic_governance::interpretation_set_witness::
    independent_interpretation_root_set_witness_profile_digest_v1;
use symthaea_rca_shadow_disposition_policy::RegisteredShadowDispositionPolicyV1;

pub const EFFECTIVE_SHADOW_DISPOSITION_POLICY_SCHEMA_VERSION: u16 = 1;
pub const EFFECTIVE_SHADOW_DISPOSITION_POLICY_PROFILE_V1: &str =
    "rca-effective-shadow-disposition-policy-v1";

pub const EFFECTIVE_SHADOW_DISPOSITION_POLICY_CONTRACT_V1: &str = concat!(
    "rca-effective-shadow-disposition-policy-v1\n",
    "input=registered_shadow_disposition_policy_v1\n",
    "binding=exact_base_policy_id+base_policy_profile+interpretation_set_witness_profile\n",
    "interpretation_set_witness_profile_is_preregistration_bearing\n",
    "profile_change_requires_new_effective_policy_identity\n",
    "effective_policy_id=blake3_explicit_complete_binding_v1\n",
    "persistence=revalidate_base_policy+current_interpretation_witness_profile+identity\n",
    "effective_policy_registration_precedes_result_bearing_evaluation\n",
    "effective_policy_is_not_case_evaluation_or_disposition\n",
    "effective_policy_is_not_belief_workspace_action_or_promotion_authority\n",
);

const PROFILE_DOMAIN: &[u8] =
    b"symthaea:rca-effective-shadow-disposition-policy-contract:v1\0";
const EFFECTIVE_POLICY_ID_DOMAIN: &[u8] =
    b"symthaea:rca-effective-shadow-disposition-policy:v1\0";

/// Persistable effective policy binding. The wrapped base policy revalidates on
/// deserialization, and this wrapper additionally rechecks the exact current
/// interpretation-set-witness profile before restoring trusted policy state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RegisteredEffectiveShadowDispositionPolicyV1 {
    schema_version: u16,
    profile: String,
    profile_contract_digest: String,
    effective_policy_id: String,
    interpretation_set_witness_profile_digest: String,
    base_policy: RegisteredShadowDispositionPolicyV1,
}

impl RegisteredEffectiveShadowDispositionPolicyV1 {
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub fn profile(&self) -> &str {
        &self.profile
    }

    pub fn profile_contract_digest(&self) -> &str {
        &self.profile_contract_digest
    }

    pub fn effective_policy_id(&self) -> &str {
        &self.effective_policy_id
    }

    pub fn interpretation_set_witness_profile_digest(&self) -> &str {
        &self.interpretation_set_witness_profile_digest
    }

    pub fn base_policy(&self) -> &RegisteredShadowDispositionPolicyV1 {
        &self.base_policy
    }
}

pub fn effective_shadow_disposition_policy_profile_digest_v1() -> String {
    domain_hash(
        PROFILE_DOMAIN,
        EFFECTIVE_SHADOW_DISPOSITION_POLICY_CONTRACT_V1.as_bytes(),
    )
}

pub fn register_effective_shadow_disposition_policy_v1(
    base_policy: RegisteredShadowDispositionPolicyV1,
) -> RegisteredEffectiveShadowDispositionPolicyV1 {
    let profile_contract_digest = effective_shadow_disposition_policy_profile_digest_v1();
    let interpretation_set_witness_profile_digest =
        independent_interpretation_root_set_witness_profile_digest_v1();
    let effective_policy_id = effective_policy_id_v1(
        &profile_contract_digest,
        &base_policy,
        &interpretation_set_witness_profile_digest,
    );

    RegisteredEffectiveShadowDispositionPolicyV1 {
        schema_version: EFFECTIVE_SHADOW_DISPOSITION_POLICY_SCHEMA_VERSION,
        profile: EFFECTIVE_SHADOW_DISPOSITION_POLICY_PROFILE_V1.to_string(),
        profile_contract_digest,
        effective_policy_id,
        interpretation_set_witness_profile_digest,
        base_policy,
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RegisteredEffectiveShadowDispositionPolicyWireV1 {
    schema_version: u16,
    profile: String,
    profile_contract_digest: String,
    effective_policy_id: String,
    interpretation_set_witness_profile_digest: String,
    base_policy: RegisteredShadowDispositionPolicyV1,
}

impl<'de> Deserialize<'de> for RegisteredEffectiveShadowDispositionPolicyV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RegisteredEffectiveShadowDispositionPolicyWireV1::deserialize(deserializer)?;
        if wire.schema_version != EFFECTIVE_SHADOW_DISPOSITION_POLICY_SCHEMA_VERSION {
            return Err(serde::de::Error::custom(
                EffectiveShadowDispositionPolicyError::UnsupportedSchemaVersion {
                    found: wire.schema_version,
                },
            ));
        }
        if wire.profile != EFFECTIVE_SHADOW_DISPOSITION_POLICY_PROFILE_V1 {
            return Err(serde::de::Error::custom(
                EffectiveShadowDispositionPolicyError::UnexpectedProfile,
            ));
        }
        validate_digest(&wire.profile_contract_digest).map_err(serde::de::Error::custom)?;
        validate_digest(&wire.effective_policy_id).map_err(serde::de::Error::custom)?;
        validate_digest(&wire.interpretation_set_witness_profile_digest)
            .map_err(serde::de::Error::custom)?;

        let expected = register_effective_shadow_disposition_policy_v1(wire.base_policy.clone());
        if wire.profile_contract_digest != expected.profile_contract_digest
            || wire.effective_policy_id != expected.effective_policy_id
            || wire.interpretation_set_witness_profile_digest
                != expected.interpretation_set_witness_profile_digest
            || wire.base_policy != expected.base_policy
        {
            return Err(serde::de::Error::custom(
                EffectiveShadowDispositionPolicyError::EffectivePolicyIdentityMismatch,
            ));
        }
        Ok(expected)
    }
}

fn effective_policy_id_v1(
    profile_contract_digest: &str,
    base_policy: &RegisteredShadowDispositionPolicyV1,
    interpretation_set_witness_profile_digest: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(EFFECTIVE_POLICY_ID_DOMAIN);
    hash_text(
        &mut hasher,
        b"profile_contract_digest",
        profile_contract_digest,
    );
    hash_bytes(
        &mut hasher,
        b"schema_version",
        &EFFECTIVE_SHADOW_DISPOSITION_POLICY_SCHEMA_VERSION.to_le_bytes(),
    );
    hash_text(&mut hasher, b"base_policy_id", base_policy.policy_id());
    hash_text(
        &mut hasher,
        b"base_policy_profile_contract_digest",
        base_policy.profile_contract_digest(),
    );
    hash_text(
        &mut hasher,
        b"interpretation_set_witness_profile_digest",
        interpretation_set_witness_profile_digest,
    );
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn validate_digest(digest: &str) -> Result<(), EffectiveShadowDispositionPolicyError> {
    let Some((algorithm, hex)) = digest.split_once(':') else {
        return Err(EffectiveShadowDispositionPolicyError::MalformedDigest);
    };
    if !matches!(algorithm, "sha256" | "blake3")
        || hex.len() != 64
        || !hex.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(EffectiveShadowDispositionPolicyError::MalformedDigest);
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
pub enum EffectiveShadowDispositionPolicyError {
    UnsupportedSchemaVersion { found: u16 },
    UnexpectedProfile,
    MalformedDigest,
    EffectivePolicyIdentityMismatch,
}

impl std::fmt::Display for EffectiveShadowDispositionPolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => write!(
                f,
                "unsupported effective shadow-disposition policy schema {found}; expected {EFFECTIVE_SHADOW_DISPOSITION_POLICY_SCHEMA_VERSION}"
            ),
            Self::UnexpectedProfile => {
                f.write_str("unexpected effective shadow-disposition policy profile")
            }
            Self::MalformedDigest => {
                f.write_str("digest must be sha256:<64 hex> or blake3:<64 hex>")
            }
            Self::EffectivePolicyIdentityMismatch => {
                f.write_str("effective shadow-disposition policy identity mismatch")
            }
        }
    }
}

impl std::error::Error for EffectiveShadowDispositionPolicyError {}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_epistemic_governance::experiment_contract::EXPERIMENT_CONTRACT_SCHEMA_VERSION;
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

    fn base_policy() -> RegisteredShadowDispositionPolicyV1 {
        ShadowDispositionPolicyV1 {
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
        .unwrap()
    }

    #[test]
    fn effective_policy_binds_current_interpretation_witness_profile() {
        let effective = register_effective_shadow_disposition_policy_v1(base_policy());
        assert_eq!(
            effective.interpretation_set_witness_profile_digest(),
            independent_interpretation_root_set_witness_profile_digest_v1()
        );
        assert!(effective.effective_policy_id().starts_with("blake3:"));
    }

    #[test]
    fn base_policy_identity_is_effective_policy_bearing() {
        let first = register_effective_shadow_disposition_policy_v1(base_policy());
        let mut changed = base_policy().policy().clone();
        changed.resources.max_case_items = 64;
        let second = register_effective_shadow_disposition_policy_v1(changed.register().unwrap());
        assert_ne!(first.effective_policy_id(), second.effective_policy_id());
    }

    #[test]
    fn effective_policy_revalidates_after_persistence() {
        let effective = register_effective_shadow_disposition_policy_v1(base_policy());
        let encoded = serde_json::to_string(&effective).unwrap();
        let decoded: RegisteredEffectiveShadowDispositionPolicyV1 =
            serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, effective);
    }

    #[test]
    fn tampered_interpretation_witness_profile_fails_closed() {
        let effective = register_effective_shadow_disposition_policy_v1(base_policy());
        let mut value = serde_json::to_value(&effective).unwrap();
        value["interpretation_set_witness_profile_digest"] = serde_json::Value::String(
            "blake3:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
                .into(),
        );
        assert!(
            serde_json::from_value::<RegisteredEffectiveShadowDispositionPolicyV1>(value).is_err()
        );
    }
}
