// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! RCA-003b.3: preregistered shadow-disposition policy identity.
//!
//! This crate intentionally contains **no disposition engine**. It freezes the
//! complete policy surface that a later pure engine may consume so thresholds,
//! profile bindings, abstention behavior, defeater treatment, evaluation identity,
//! and resource ceilings cannot be introduced as hidden implementation constants.

#![deny(unsafe_code)]

use serde::{Deserialize, Deserializer, Serialize};
use symthaea_epistemic_governance::{
    interpretation_lineage::interpretation_lineage_profile_digest_v1,
    relation_qualification::relation_declaration_eligibility_profile_digest_v1,
};
use symthaea_rca_bound_shadow_case::bound_shadow_evidence_case_profile_digest_v1;

pub const SHADOW_DISPOSITION_POLICY_SCHEMA_VERSION: u16 = 1;
pub const SHADOW_DISPOSITION_POLICY_PROFILE_V1: &str = "rca-shadow-disposition-policy-v1";

pub const SHADOW_DISPOSITION_POLICY_CONTRACT_V1: &str = concat!(
    "rca-shadow-disposition-policy-v1\n",
    "policy_is_registered_before_result_bearing_case_evaluation\n",
    "scope=one_exact_proposition_id_v1\n",
    "profiles=exact_bound_case+relation_eligibility+interpretation_lineage\n",
    "thresholds=explicit_per_outcome_evidence_root+interpretation_root_requirements\n",
    "strength_treatment=diagnostic_only_v1_no_arithmetic\n",
    "defeater_mode=qualified_current_blocker_only_v1\n",
    "unknown_interpretation_independence=force_underdetermined_v1\n",
    "contestation=qualified_support_and_opposition_must_survive\n",
    "evaluation_binding=preregistration+corpus+seed+metric+evaluator_identity\n",
    "resource_ceilings=explicit_case_items+interpretation_pairs\n",
    "policy_id=blake3_explicit_complete_normalized_policy_v1\n",
    "post_result_policy_change_requires_new_policy_identity\n",
    "policy_registration_is_not_case_evaluation_or_disposition\n",
    "policy_is_not_belief_workspace_action_or_promotion_authority\n",
);

const PROFILE_DOMAIN: &[u8] = b"symthaea:rca-shadow-disposition-policy-contract:v1\0";
const POLICY_ID_DOMAIN: &[u8] = b"symthaea:rca-shadow-disposition-policy:v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationStrengthTreatmentV1 {
    /// Relation strength is retained as diagnostic metadata only. V1 exposes no
    /// arithmetic semantics for summing, averaging, multiplying, normalizing,
    /// voting, or Bayesian updating declared relation strengths.
    DiagnosticOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DefeaterModeV1 {
    /// A later engine may block positive support only when the defeater relation
    /// is current, case-joined, declarer-eligible, and satisfies the exact root
    /// requirements registered in this policy.
    QualifiedCurrentBlocker,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnknownInterpretationIndependenceModeV1 {
    /// Missing interpretation independence never becomes an independent vote.
    /// The later engine must preserve underdetermination where the registered
    /// root requirements cannot be established.
    ForceUnderdetermined,
}

/// Root requirements for one future shadow-disposition outcome condition.
///
/// These are topology requirements, not candidate/module vote counts and not
/// calibrated probabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeRootRequirementsV1 {
    pub min_independent_evidence_roots: u16,
    pub min_qualified_interpretation_roots: u16,
}

/// Exact evaluation/preregistration lineage under which this policy was frozen.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DispositionPolicyEvaluationBindingV1 {
    pub preregistration_contract_digest: String,
    pub evaluation_corpus_digest: String,
    pub seed_plan_digest: String,
    pub metric_contract_digest: String,
    pub evaluator_id: String,
    pub evaluator_version: Option<String>,
}

/// Resource ceilings are part of policy identity even before an engine exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DispositionPolicyResourceCeilingsV1 {
    pub max_case_items: u32,
    pub max_interpretation_pairs: u32,
}

/// Raw preregistered policy before normalization/registration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowDispositionPolicyV1 {
    pub schema_version: u16,
    pub proposition_id: String,

    /// Exact lower-layer semantic profiles this policy is allowed to interpret.
    pub bound_case_profile_digest: String,
    pub relation_eligibility_profile_digest: String,
    pub interpretation_lineage_profile_digest: String,

    pub tentative_support_requirements: OutcomeRootRequirementsV1,
    pub support_requirements: OutcomeRootRequirementsV1,
    pub tentative_opposition_requirements: OutcomeRootRequirementsV1,
    pub opposition_requirements: OutcomeRootRequirementsV1,
    /// Requirements that a qualified current `Defeats` relation must satisfy
    /// before a later engine may block positive support.
    pub defeater_requirements: OutcomeRootRequirementsV1,
    /// Requirements on each surviving side before a later engine may emit
    /// `Contested` rather than collapsing disagreement.
    pub contested_side_requirements: OutcomeRootRequirementsV1,

    pub strength_treatment: RelationStrengthTreatmentV1,
    pub defeater_mode: DefeaterModeV1,
    pub unknown_interpretation_independence_mode: UnknownInterpretationIndependenceModeV1,
    pub contested_requires_qualified_support_and_opposition: bool,

    pub evaluation: DispositionPolicyEvaluationBindingV1,
    pub resources: DispositionPolicyResourceCeilingsV1,
}

impl ShadowDispositionPolicyV1 {
    pub fn register(self) -> Result<RegisteredShadowDispositionPolicyV1, ShadowDispositionPolicyError> {
        RegisteredShadowDispositionPolicyV1::try_from(self)
    }
}

/// Persistable/revalidated preregistered policy.
///
/// This is policy data only. There is deliberately no method that accepts a case
/// or emits `Supported`, `Contested`, `Defeated`, or any other disposition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RegisteredShadowDispositionPolicyV1 {
    schema_version: u16,
    profile: String,
    profile_contract_digest: String,
    policy_id: String,
    policy: ShadowDispositionPolicyV1,
}

impl RegisteredShadowDispositionPolicyV1 {
    pub const fn schema_version(&self) -> u16 {
        self.schema_version
    }

    pub fn profile(&self) -> &str {
        &self.profile
    }

    pub fn profile_contract_digest(&self) -> &str {
        &self.profile_contract_digest
    }

    pub fn policy_id(&self) -> &str {
        &self.policy_id
    }

    pub fn policy(&self) -> &ShadowDispositionPolicyV1 {
        &self.policy
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RegisteredShadowDispositionPolicyWireV1 {
    schema_version: u16,
    profile: String,
    profile_contract_digest: String,
    policy_id: String,
    policy: ShadowDispositionPolicyV1,
}

impl<'de> Deserialize<'de> for RegisteredShadowDispositionPolicyV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RegisteredShadowDispositionPolicyWireV1::deserialize(deserializer)?;
        if wire.schema_version != SHADOW_DISPOSITION_POLICY_SCHEMA_VERSION {
            return Err(serde::de::Error::custom(
                ShadowDispositionPolicyError::UnsupportedSchemaVersion {
                    found: wire.schema_version,
                },
            ));
        }
        if wire.profile != SHADOW_DISPOSITION_POLICY_PROFILE_V1 {
            return Err(serde::de::Error::custom(
                ShadowDispositionPolicyError::UnexpectedProfile,
            ));
        }
        validate_digest(&wire.profile_contract_digest).map_err(serde::de::Error::custom)?;
        validate_digest(&wire.policy_id).map_err(serde::de::Error::custom)?;

        let expected = wire
            .policy
            .clone()
            .register()
            .map_err(serde::de::Error::custom)?;
        if wire.profile_contract_digest != expected.profile_contract_digest
            || wire.policy_id != expected.policy_id
            || wire.policy != expected.policy
        {
            return Err(serde::de::Error::custom(
                ShadowDispositionPolicyError::PolicyIdentityMismatch,
            ));
        }
        Ok(expected)
    }
}

pub fn shadow_disposition_policy_profile_digest_v1() -> String {
    domain_hash(
        PROFILE_DOMAIN,
        SHADOW_DISPOSITION_POLICY_CONTRACT_V1.as_bytes(),
    )
}

pub fn current_bound_case_profile_digest_v1() -> String {
    bound_shadow_evidence_case_profile_digest_v1()
}

pub fn current_relation_eligibility_profile_digest_v1() -> String {
    relation_declaration_eligibility_profile_digest_v1()
}

pub fn current_interpretation_lineage_profile_digest_v1() -> String {
    interpretation_lineage_profile_digest_v1()
}

impl TryFrom<ShadowDispositionPolicyV1> for RegisteredShadowDispositionPolicyV1 {
    type Error = ShadowDispositionPolicyError;

    fn try_from(value: ShadowDispositionPolicyV1) -> Result<Self, Self::Error> {
        validate_policy_v1(&value)?;
        let profile_contract_digest = shadow_disposition_policy_profile_digest_v1();
        let policy_id = shadow_disposition_policy_id_v1(&profile_contract_digest, &value);
        Ok(Self {
            schema_version: SHADOW_DISPOSITION_POLICY_SCHEMA_VERSION,
            profile: SHADOW_DISPOSITION_POLICY_PROFILE_V1.to_string(),
            profile_contract_digest,
            policy_id,
            policy: value,
        })
    }
}

fn validate_policy_v1(value: &ShadowDispositionPolicyV1) -> Result<(), ShadowDispositionPolicyError> {
    if value.schema_version != SHADOW_DISPOSITION_POLICY_SCHEMA_VERSION {
        return Err(ShadowDispositionPolicyError::UnsupportedSchemaVersion {
            found: value.schema_version,
        });
    }
    validate_digest(&value.proposition_id)?;

    let expected_case = current_bound_case_profile_digest_v1();
    if value.bound_case_profile_digest != expected_case {
        return Err(ShadowDispositionPolicyError::BoundCaseProfileMismatch);
    }
    let expected_eligibility = current_relation_eligibility_profile_digest_v1();
    if value.relation_eligibility_profile_digest != expected_eligibility {
        return Err(ShadowDispositionPolicyError::RelationEligibilityProfileMismatch);
    }
    let expected_interpretation = current_interpretation_lineage_profile_digest_v1();
    if value.interpretation_lineage_profile_digest != expected_interpretation {
        return Err(ShadowDispositionPolicyError::InterpretationLineageProfileMismatch);
    }

    validate_root_requirements("tentative_support", value.tentative_support_requirements)?;
    validate_root_requirements("support", value.support_requirements)?;
    validate_root_requirements(
        "tentative_opposition",
        value.tentative_opposition_requirements,
    )?;
    validate_root_requirements("opposition", value.opposition_requirements)?;
    validate_root_requirements("defeater", value.defeater_requirements)?;
    validate_root_requirements("contested_side", value.contested_side_requirements)?;

    if !requirements_at_least(
        value.support_requirements,
        value.tentative_support_requirements,
    ) {
        return Err(ShadowDispositionPolicyError::SupportWeakerThanTentativeSupport);
    }
    if !requirements_at_least(
        value.opposition_requirements,
        value.tentative_opposition_requirements,
    ) {
        return Err(ShadowDispositionPolicyError::OppositionWeakerThanTentativeOpposition);
    }
    if !value.contested_requires_qualified_support_and_opposition {
        return Err(ShadowDispositionPolicyError::ContestedMustPreserveBothSides);
    }

    // V1 intentionally has only these fail-closed variants. Keeping the matches
    // explicit prevents later enum growth from silently acquiring semantics.
    match value.strength_treatment {
        RelationStrengthTreatmentV1::DiagnosticOnly => {}
    }
    match value.defeater_mode {
        DefeaterModeV1::QualifiedCurrentBlocker => {}
    }
    match value.unknown_interpretation_independence_mode {
        UnknownInterpretationIndependenceModeV1::ForceUnderdetermined => {}
    }

    validate_evaluation_binding(&value.evaluation)?;
    if value.resources.max_case_items == 0 {
        return Err(ShadowDispositionPolicyError::ZeroMaxCaseItems);
    }
    if value.resources.max_interpretation_pairs == 0 {
        return Err(ShadowDispositionPolicyError::ZeroMaxInterpretationPairs);
    }

    let max_required_evidence_roots = [
        value.tentative_support_requirements,
        value.support_requirements,
        value.tentative_opposition_requirements,
        value.opposition_requirements,
        value.defeater_requirements,
        value.contested_side_requirements,
    ]
    .into_iter()
    .map(|requirement| requirement.min_independent_evidence_roots as u32)
    .max()
    .unwrap_or(0);
    if value.resources.max_case_items < max_required_evidence_roots {
        return Err(ShadowDispositionPolicyError::CaseCeilingBelowEvidenceRootRequirement);
    }

    Ok(())
}

fn validate_root_requirements(
    outcome: &'static str,
    value: OutcomeRootRequirementsV1,
) -> Result<(), ShadowDispositionPolicyError> {
    if value.min_independent_evidence_roots == 0 {
        return Err(ShadowDispositionPolicyError::ZeroEvidenceRootRequirement { outcome });
    }
    if value.min_qualified_interpretation_roots == 0 {
        return Err(ShadowDispositionPolicyError::ZeroInterpretationRootRequirement { outcome });
    }
    Ok(())
}

fn requirements_at_least(
    stronger: OutcomeRootRequirementsV1,
    weaker: OutcomeRootRequirementsV1,
) -> bool {
    stronger.min_independent_evidence_roots >= weaker.min_independent_evidence_roots
        && stronger.min_qualified_interpretation_roots >= weaker.min_qualified_interpretation_roots
}

fn validate_evaluation_binding(
    value: &DispositionPolicyEvaluationBindingV1,
) -> Result<(), ShadowDispositionPolicyError> {
    validate_digest(&value.preregistration_contract_digest)?;
    validate_digest(&value.evaluation_corpus_digest)?;
    validate_digest(&value.seed_plan_digest)?;
    validate_digest(&value.metric_contract_digest)?;
    if value.evaluator_id.trim().is_empty() {
        return Err(ShadowDispositionPolicyError::MissingEvaluatorId);
    }
    if value
        .evaluator_version
        .as_deref()
        .is_some_and(|version| version.trim().is_empty())
    {
        return Err(ShadowDispositionPolicyError::EmptyEvaluatorVersion);
    }
    Ok(())
}

fn shadow_disposition_policy_id_v1(
    profile_contract_digest: &str,
    value: &ShadowDispositionPolicyV1,
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
        &value.schema_version.to_le_bytes(),
    );
    hash_text(&mut hasher, b"proposition_id", &value.proposition_id);
    hash_text(
        &mut hasher,
        b"bound_case_profile_digest",
        &value.bound_case_profile_digest,
    );
    hash_text(
        &mut hasher,
        b"relation_eligibility_profile_digest",
        &value.relation_eligibility_profile_digest,
    );
    hash_text(
        &mut hasher,
        b"interpretation_lineage_profile_digest",
        &value.interpretation_lineage_profile_digest,
    );

    hash_requirements(
        &mut hasher,
        b"tentative_support",
        value.tentative_support_requirements,
    );
    hash_requirements(&mut hasher, b"support", value.support_requirements);
    hash_requirements(
        &mut hasher,
        b"tentative_opposition",
        value.tentative_opposition_requirements,
    );
    hash_requirements(&mut hasher, b"opposition", value.opposition_requirements);
    hash_requirements(&mut hasher, b"defeater", value.defeater_requirements);
    hash_requirements(
        &mut hasher,
        b"contested_side",
        value.contested_side_requirements,
    );

    hash_text(
        &mut hasher,
        b"strength_treatment",
        strength_treatment_tag(value.strength_treatment),
    );
    hash_text(
        &mut hasher,
        b"defeater_mode",
        defeater_mode_tag(value.defeater_mode),
    );
    hash_text(
        &mut hasher,
        b"unknown_interpretation_independence_mode",
        unknown_independence_tag(value.unknown_interpretation_independence_mode),
    );
    hash_bool(
        &mut hasher,
        b"contested_requires_qualified_support_and_opposition",
        value.contested_requires_qualified_support_and_opposition,
    );

    hash_text(
        &mut hasher,
        b"preregistration_contract_digest",
        &value.evaluation.preregistration_contract_digest,
    );
    hash_text(
        &mut hasher,
        b"evaluation_corpus_digest",
        &value.evaluation.evaluation_corpus_digest,
    );
    hash_text(
        &mut hasher,
        b"seed_plan_digest",
        &value.evaluation.seed_plan_digest,
    );
    hash_text(
        &mut hasher,
        b"metric_contract_digest",
        &value.evaluation.metric_contract_digest,
    );
    hash_text(
        &mut hasher,
        b"evaluator_id",
        &value.evaluation.evaluator_id,
    );
    hash_option_text(
        &mut hasher,
        b"evaluator_version",
        value.evaluation.evaluator_version.as_deref(),
    );

    hash_bytes(
        &mut hasher,
        b"max_case_items",
        &value.resources.max_case_items.to_le_bytes(),
    );
    hash_bytes(
        &mut hasher,
        b"max_interpretation_pairs",
        &value.resources.max_interpretation_pairs.to_le_bytes(),
    );

    format!("blake3:{}", hasher.finalize().to_hex())
}

fn hash_requirements(
    hasher: &mut blake3::Hasher,
    label: &[u8],
    value: OutcomeRootRequirementsV1,
) {
    let evidence_label = [label, b"_min_independent_evidence_roots"].concat();
    let interpretation_label = [label, b"_min_qualified_interpretation_roots"].concat();
    hash_bytes(
        hasher,
        &evidence_label,
        &value.min_independent_evidence_roots.to_le_bytes(),
    );
    hash_bytes(
        hasher,
        &interpretation_label,
        &value.min_qualified_interpretation_roots.to_le_bytes(),
    );
}

fn strength_treatment_tag(value: RelationStrengthTreatmentV1) -> &'static str {
    match value {
        RelationStrengthTreatmentV1::DiagnosticOnly => "diagnostic_only",
    }
}

fn defeater_mode_tag(value: DefeaterModeV1) -> &'static str {
    match value {
        DefeaterModeV1::QualifiedCurrentBlocker => "qualified_current_blocker",
    }
}

fn unknown_independence_tag(value: UnknownInterpretationIndependenceModeV1) -> &'static str {
    match value {
        UnknownInterpretationIndependenceModeV1::ForceUnderdetermined => "force_underdetermined",
    }
}

fn validate_digest(digest: &str) -> Result<(), ShadowDispositionPolicyError> {
    let Some((algorithm, hex)) = digest.split_once(':') else {
        return Err(ShadowDispositionPolicyError::MalformedDigest);
    };
    if !matches!(algorithm, "sha256" | "blake3")
        || hex.len() != 64
        || !hex.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(ShadowDispositionPolicyError::MalformedDigest);
    }
    Ok(())
}

fn hash_bool(hasher: &mut blake3::Hasher, label: &[u8], value: bool) {
    hash_bytes(hasher, label, &[u8::from(value)]);
}

fn hash_text(hasher: &mut blake3::Hasher, label: &[u8], value: &str) {
    hash_bytes(hasher, label, value.as_bytes());
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
pub enum ShadowDispositionPolicyError {
    UnsupportedSchemaVersion { found: u16 },
    UnexpectedProfile,
    PolicyIdentityMismatch,
    MalformedDigest,
    BoundCaseProfileMismatch,
    RelationEligibilityProfileMismatch,
    InterpretationLineageProfileMismatch,
    ZeroEvidenceRootRequirement { outcome: &'static str },
    ZeroInterpretationRootRequirement { outcome: &'static str },
    SupportWeakerThanTentativeSupport,
    OppositionWeakerThanTentativeOpposition,
    ContestedMustPreserveBothSides,
    MissingEvaluatorId,
    EmptyEvaluatorVersion,
    ZeroMaxCaseItems,
    ZeroMaxInterpretationPairs,
    CaseCeilingBelowEvidenceRootRequirement,
}

impl std::fmt::Display for ShadowDispositionPolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found } => write!(
                f,
                "unsupported shadow-disposition policy schema version {found}; expected {SHADOW_DISPOSITION_POLICY_SCHEMA_VERSION}"
            ),
            Self::UnexpectedProfile => f.write_str("unexpected shadow-disposition policy profile"),
            Self::PolicyIdentityMismatch => {
                f.write_str("registered shadow-disposition policy identity mismatch")
            }
            Self::MalformedDigest => {
                f.write_str("digest must be sha256:<64 hex> or blake3:<64 hex>")
            }
            Self::BoundCaseProfileMismatch => f.write_str("policy does not bind the current bound-case profile"),
            Self::RelationEligibilityProfileMismatch => f.write_str("policy does not bind the current relation-eligibility profile"),
            Self::InterpretationLineageProfileMismatch => f.write_str("policy does not bind the current interpretation-lineage profile"),
            Self::ZeroEvidenceRootRequirement { outcome } => {
                write!(f, "{outcome} requires at least one independent evidence root")
            }
            Self::ZeroInterpretationRootRequirement { outcome } => {
                write!(f, "{outcome} requires at least one qualified interpretation root")
            }
            Self::SupportWeakerThanTentativeSupport => {
                f.write_str("Supported requirements cannot be weaker than TentativelySupported requirements")
            }
            Self::OppositionWeakerThanTentativeOpposition => {
                f.write_str("Opposed requirements cannot be weaker than TentativelyOpposed requirements")
            }
            Self::ContestedMustPreserveBothSides => {
                f.write_str("v1 Contested must require qualified support and opposition")
            }
            Self::MissingEvaluatorId => f.write_str("policy evaluation binding requires evaluator id"),
            Self::EmptyEvaluatorVersion => {
                f.write_str("policy evaluator version cannot be empty when present")
            }
            Self::ZeroMaxCaseItems => f.write_str("policy max_case_items must be non-zero"),
            Self::ZeroMaxInterpretationPairs => {
                f.write_str("policy max_interpretation_pairs must be non-zero")
            }
            Self::CaseCeilingBelowEvidenceRootRequirement => f.write_str(
                "policy max_case_items cannot be below its largest evidence-root requirement",
            ),
        }
    }
}

impl std::error::Error for ShadowDispositionPolicyError {}

#[cfg(test)]
mod tests {
    use super::*;

    const PROPOSITION: &str =
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const PRE_REG: &str =
        "blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const CORPUS: &str =
        "blake3:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const SEEDS: &str =
        "blake3:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const METRIC: &str =
        "blake3:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";

    fn requirements(evidence: u16, interpretations: u16) -> OutcomeRootRequirementsV1 {
        OutcomeRootRequirementsV1 {
            min_independent_evidence_roots: evidence,
            min_qualified_interpretation_roots: interpretations,
        }
    }

    fn policy() -> ShadowDispositionPolicyV1 {
        ShadowDispositionPolicyV1 {
            schema_version: SHADOW_DISPOSITION_POLICY_SCHEMA_VERSION,
            proposition_id: PROPOSITION.into(),
            bound_case_profile_digest: current_bound_case_profile_digest_v1(),
            relation_eligibility_profile_digest: current_relation_eligibility_profile_digest_v1(),
            interpretation_lineage_profile_digest: current_interpretation_lineage_profile_digest_v1(),
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
                preregistration_contract_digest: PRE_REG.into(),
                evaluation_corpus_digest: CORPUS.into(),
                seed_plan_digest: SEEDS.into(),
                metric_contract_digest: METRIC.into(),
                evaluator_id: "shadow-disposition-policy-evaluator".into(),
                evaluator_version: Some("v1".into()),
            },
            resources: DispositionPolicyResourceCeilingsV1 {
                max_case_items: 32,
                max_interpretation_pairs: 496,
            },
        }
    }

    #[test]
    fn policy_registers_against_exact_current_profiles() {
        let registered = policy().register().unwrap();
        assert_eq!(registered.policy().proposition_id, PROPOSITION);
        assert_eq!(
            registered.policy().bound_case_profile_digest,
            current_bound_case_profile_digest_v1()
        );
        assert_eq!(
            registered.policy().relation_eligibility_profile_digest,
            current_relation_eligibility_profile_digest_v1()
        );
        assert_eq!(
            registered.policy().interpretation_lineage_profile_digest,
            current_interpretation_lineage_profile_digest_v1()
        );
    }

    #[test]
    fn strength_semantics_are_diagnostic_only_v1() {
        assert_eq!(
            policy().strength_treatment,
            RelationStrengthTreatmentV1::DiagnosticOnly
        );
        assert!(SHADOW_DISPOSITION_POLICY_CONTRACT_V1.contains(
            "strength_treatment=diagnostic_only_v1_no_arithmetic"
        ));
    }

    #[test]
    fn supported_cannot_be_weaker_than_tentative_support() {
        let mut raw = policy();
        raw.tentative_support_requirements = requirements(2, 2);
        raw.support_requirements = requirements(1, 2);
        assert_eq!(
            raw.register(),
            Err(ShadowDispositionPolicyError::SupportWeakerThanTentativeSupport)
        );
    }

    #[test]
    fn opposed_cannot_be_weaker_than_tentative_opposition() {
        let mut raw = policy();
        raw.tentative_opposition_requirements = requirements(2, 2);
        raw.opposition_requirements = requirements(2, 1);
        assert_eq!(
            raw.register(),
            Err(ShadowDispositionPolicyError::OppositionWeakerThanTentativeOpposition)
        );
    }

    #[test]
    fn zero_root_requirements_fail_closed() {
        let mut raw = policy();
        raw.defeater_requirements = requirements(0, 1);
        assert!(matches!(
            raw.register(),
            Err(ShadowDispositionPolicyError::ZeroEvidenceRootRequirement {
                outcome: "defeater"
            })
        ));
    }

    #[test]
    fn lower_layer_profile_drift_requires_new_policy() {
        let mut raw = policy();
        raw.interpretation_lineage_profile_digest = PRE_REG.into();
        assert_eq!(
            raw.register(),
            Err(ShadowDispositionPolicyError::InterpretationLineageProfileMismatch)
        );
    }

    #[test]
    fn contested_must_preserve_qualified_support_and_opposition() {
        let mut raw = policy();
        raw.contested_requires_qualified_support_and_opposition = false;
        assert_eq!(
            raw.register(),
            Err(ShadowDispositionPolicyError::ContestedMustPreserveBothSides)
        );
    }

    #[test]
    fn evaluation_and_resource_identity_are_policy_bearing() {
        let a = policy().register().unwrap();
        let mut changed = policy();
        changed.evaluation.metric_contract_digest = PRE_REG.into();
        let b = changed.register().unwrap();
        assert_ne!(a.policy_id(), b.policy_id());

        let mut changed = policy();
        changed.resources.max_case_items = 64;
        let c = changed.register().unwrap();
        assert_ne!(a.policy_id(), c.policy_id());
    }

    #[test]
    fn threshold_change_requires_new_policy_identity() {
        let a = policy().register().unwrap();
        let mut changed = policy();
        changed.support_requirements = requirements(3, 2);
        let b = changed.register().unwrap();
        assert_ne!(a.policy_id(), b.policy_id());
    }

    #[test]
    fn registered_policy_revalidates_after_persistence() {
        let registered = policy().register().unwrap();
        let encoded = serde_json::to_string(&registered).unwrap();
        let decoded: RegisteredShadowDispositionPolicyV1 =
            serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, registered);
    }

    #[test]
    fn policy_tampering_fails_closed() {
        let registered = policy().register().unwrap();
        let mut value = serde_json::to_value(&registered).unwrap();
        value["policy"]["support_requirements"]["min_independent_evidence_roots"] =
            serde_json::Value::from(7_u64);
        assert!(serde_json::from_value::<RegisteredShadowDispositionPolicyV1>(value).is_err());
    }

    #[test]
    fn resource_ceiling_cannot_be_below_required_evidence_roots() {
        let mut raw = policy();
        raw.support_requirements = requirements(33, 2);
        assert_eq!(
            raw.register(),
            Err(ShadowDispositionPolicyError::CaseCeilingBelowEvidenceRootRequirement)
        );
    }

    #[test]
    fn policy_profile_has_strict_identity() {
        let digest = shadow_disposition_policy_profile_digest_v1();
        assert!(digest.starts_with("blake3:"));
        assert_eq!(digest.len(), "blake3:".len() + 64);
        assert_eq!(digest, shadow_disposition_policy_profile_digest_v1());
    }
}
