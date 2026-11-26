// SPDX-License-Identifier: AGPL-3.0-or-later

//! Consumer-side verification of independently issued DecisionSufficiency
//! attestations. Qualification answers whether evidence may be considered;
//! sufficiency answers whether the qualified set supports one exact decision.
//! Neither receipt authorizes an action.

use std::collections::BTreeSet;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::qualification_dsse::{
    DSSE_PAYLOAD_TYPE_V1,
    IN_TOTO_STATEMENT_TYPE_V1,
    MACE_DECISION_SUFFICIENCY_PREDICATE_TYPE_V1,
    parse_dsse_envelope_v1,
    parse_strict_json_bounded,
};
use crate::qualification_trust::PinnedEd25519QualificationVerifierV1;
use crate::qualified_candidate_evidence::QualifiedCandidateEvidenceBatchV1;
use crate::evidence_qualification_gate::{
    EvidenceQualificationBindingV1, EvidenceQualificationRejectV1,
    EvidenceQualificationSignatureVerifierV1,
};

pub const DECISION_SUFFICIENCY_VERIFIER_ID_V1: &str =
    "mace-independent-decision-sufficiency-evaluator";

/// One independently resolved source-root family and the receipt that
/// commits to its transitive lineage/correlation analysis. The consumer checks
/// identity, ordering and cardinality; the evaluator must independently verify
/// the referenced provenance receipts before asserting family independence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionSufficiencyRootFamilyV1 {
    pub root_family_sha256: String,
    pub provenance_receipt_sha256: String,
}

/// One explicit obligation that must be met before a particular decision can
/// be considered sufficient. Independent-root counts refer to verified source
/// families, not the number of copied documents or derived chunks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionNeedRequirementV1 {
    pub requirement_id: String,
    pub evidence_class_sha256: String,
    pub minimum_independent_root_families: u32,
    pub maximum_evidence_age_seconds: u64,
}

/// Consumer-owned, versioned specification of one decision.
///
/// This is loaded from a trusted consumer policy registry, not from candidate
/// memory or the qualifier response. Its digest pins the allowed action set,
/// utility model, risk limit, confidence floor, calibration artifact, resource
/// budget and complete sorted requirement set. The evaluator may assess this
/// contract but may not choose or weaken it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionNeedV1 {
    pub decision_need_id: String,
    pub domain_id: String,
    pub subject_scope_sha256: String,
    pub purpose_sha256: String,
    pub action_set_sha256: String,
    pub utility_profile_sha256: String,
    pub risk_limit_ppm: u32,
    pub minimum_confidence_level_ppm: u32,
    pub regret_tolerance_micros: i64,
    pub sufficiency_policy_sha256: String,
    pub calibration_profile_sha256: String,
    pub calibration_receipt_sha256: String,
    pub maximum_evidence_age_seconds: u64,
    pub maximum_context_tokens: u64,
    pub maximum_tool_calls: u32,
    pub maximum_cost_micros: u64,
    pub maximum_latency_ms: u64,
    pub requirements: Vec<DecisionNeedRequirementV1>,
}

impl DecisionNeedV1 {
    pub fn validate(&self) -> Result<(), DecisionSufficiencyRejectV1> {
        if !valid_atom(&self.decision_need_id)
            || self.decision_need_id.len() > 256
            || !valid_atom(&self.domain_id)
            || self.domain_id.len() > 512
        {
            return Err(DecisionSufficiencyRejectV1::InvalidIdentifier);
        }
        for digest in [
            self.subject_scope_sha256.as_str(),
            self.purpose_sha256.as_str(),
            self.action_set_sha256.as_str(),
            self.utility_profile_sha256.as_str(),
            self.sufficiency_policy_sha256.as_str(),
            self.calibration_profile_sha256.as_str(),
            self.calibration_receipt_sha256.as_str(),
        ] {
            if !valid_sha256(digest) {
                return Err(DecisionSufficiencyRejectV1::InvalidDigest);
            }
        }
        if self.risk_limit_ppm > 1_000_000
            || !(500_000..=999_999).contains(&self.minimum_confidence_level_ppm)
            || self.regret_tolerance_micros < 0
            || self.regret_tolerance_micros > 1_000_000_000_000
            || self.maximum_cost_micros == 0
            || self.maximum_cost_micros > 1_000_000_000_000
            || self.maximum_evidence_age_seconds == 0
            || self.maximum_evidence_age_seconds > 315_360_000
            || self.maximum_context_tokens == 0
            || self.maximum_context_tokens > 1_000_000_000_000
            || self.maximum_tool_calls == 0
            || self.maximum_tool_calls > 1_000_000
            || self.maximum_latency_ms == 0
            || self.maximum_latency_ms > 86_400_000
            || self.requirements.is_empty()
            || self.requirements.len() > 1024
        {
            return Err(DecisionSufficiencyRejectV1::InvalidDecisionNeed);
        }
        if self.requirements.windows(2).any(|pair| {
            pair[0].requirement_id >= pair[1].requirement_id
        }) {
            return Err(DecisionSufficiencyRejectV1::RequirementOrderInvalid);
        }
        for requirement in &self.requirements {
            if !valid_atom(&requirement.requirement_id)
                || requirement.requirement_id.len() > 256
                || !valid_sha256(&requirement.evidence_class_sha256)
                || requirement.minimum_independent_root_families == 0
                || requirement.minimum_independent_root_families > 10_000
                || requirement.maximum_evidence_age_seconds == 0
                || requirement.maximum_evidence_age_seconds > 315_360_000
                || requirement.maximum_evidence_age_seconds > self.maximum_evidence_age_seconds
            {
                return Err(DecisionSufficiencyRejectV1::InvalidDecisionNeed);
            }
        }
        Ok(())
    }

    /// Domain-separated digest over the complete typed decision contract.
    pub fn digest(&self) -> Result<String, DecisionSufficiencyRejectV1> {
        self.validate()?;
        let mut hasher = Sha256::new();
        hasher.update(b"luminous:mace:decision-need:v1\0");
        for value in [
            self.decision_need_id.as_str(),
            self.domain_id.as_str(),
            self.subject_scope_sha256.as_str(),
            self.purpose_sha256.as_str(),
            self.action_set_sha256.as_str(),
            self.utility_profile_sha256.as_str(),
            self.sufficiency_policy_sha256.as_str(),
            self.calibration_profile_sha256.as_str(),
            self.calibration_receipt_sha256.as_str(),
        ] {
            hash_field(&mut hasher, value);
        }
        hasher.update(self.risk_limit_ppm.to_be_bytes());
        hasher.update(self.minimum_confidence_level_ppm.to_be_bytes());
        hasher.update(self.regret_tolerance_micros.to_be_bytes());
        hasher.update(self.maximum_evidence_age_seconds.to_be_bytes());
        hasher.update(self.maximum_context_tokens.to_be_bytes());
        hasher.update(self.maximum_tool_calls.to_be_bytes());
        hasher.update(self.maximum_cost_micros.to_be_bytes());
        hasher.update(self.maximum_latency_ms.to_be_bytes());
        hasher.update((self.requirements.len() as u64).to_be_bytes());
        for requirement in &self.requirements {
            hash_field(&mut hasher, &requirement.requirement_id);
            hash_field(&mut hasher, &requirement.evidence_class_sha256);
            hasher.update(requirement.minimum_independent_root_families.to_be_bytes());
            hasher.update(requirement.maximum_evidence_age_seconds.to_be_bytes());
        }
        Ok(hex_digest(&hasher.finalize()))
    }

    /// Decode a consumer-owned manifest from strict duplicate-safe JSON.
    /// The caller must obtain this document from its authorized policy registry,
    /// not from candidate memory or the qualification issuer.
    pub fn from_json_str(input: &str) -> Result<Self, DecisionSufficiencyRejectV1> {
        let value = parse_strict_json_bounded(input.as_bytes(), 1024 * 1024)
            .map_err(|_| DecisionSufficiencyRejectV1::InvalidJson)?;
        Self::from_json_value(&value)
    }

    pub fn from_json_value(value: &Value) -> Result<Self, DecisionSufficiencyRejectV1> {
        let object = value.as_object().ok_or(DecisionSufficiencyRejectV1::InvalidJson)?;
        const FIELDS: &[&str] = &[
            "schema_version", "decision_need_id", "domain_id", "subject_scope_sha256",
            "purpose_sha256", "action_set_sha256", "utility_profile_sha256",
            "risk_limit_ppm", "minimum_confidence_level_ppm", "regret_tolerance_micros",
            "sufficiency_policy_sha256", "calibration_profile_sha256",
            "calibration_receipt_sha256", "maximum_evidence_age_seconds",
            "maximum_context_tokens", "maximum_tool_calls", "maximum_cost_micros",
            "maximum_latency_ms", "requirements",
        ];
        if object.len() != FIELDS.len()
            || object.keys().any(|key| !FIELDS.contains(&key.as_str()))
            || object.get("schema_version").and_then(Value::as_str)
                != Some("mace_decision_need_v1")
        {
            return Err(DecisionSufficiencyRejectV1::InvalidJson);
        }
        let string = |field: &str| -> Result<String, DecisionSufficiencyRejectV1> {
            object.get(field).and_then(Value::as_str).map(str::to_owned)
                .ok_or(DecisionSufficiencyRejectV1::InvalidJson)
        };
        let u64_value = |field: &str| -> Result<u64, DecisionSufficiencyRejectV1> {
            object.get(field).and_then(Value::as_u64)
                .ok_or(DecisionSufficiencyRejectV1::InvalidJson)
        };
        let u32_value = |field: &str| -> Result<u32, DecisionSufficiencyRejectV1> {
            u32::try_from(u64_value(field)?)
                .map_err(|_| DecisionSufficiencyRejectV1::InvalidJson)
        };
        let i64_value = |field: &str| -> Result<i64, DecisionSufficiencyRejectV1> {
            object.get(field).and_then(Value::as_i64)
                .ok_or(DecisionSufficiencyRejectV1::InvalidJson)
        };
        let values = object.get("requirements").and_then(Value::as_array)
            .ok_or(DecisionSufficiencyRejectV1::InvalidJson)?;
        if values.is_empty() || values.len() > 1024 {
            return Err(DecisionSufficiencyRejectV1::InvalidDecisionNeed);
        }
        let mut requirements = Vec::with_capacity(values.len());
        for value in values {
            let item = value.as_object().ok_or(DecisionSufficiencyRejectV1::InvalidJson)?;
            if item.len() != 4
                || item.keys().any(|key| ![
                    "requirement_id", "evidence_class_sha256",
                    "minimum_independent_root_families", "maximum_evidence_age_seconds",
                ].contains(&key.as_str()))
            {
                return Err(DecisionSufficiencyRejectV1::InvalidJson);
            }
            requirements.push(DecisionNeedRequirementV1 {
                requirement_id: item.get("requirement_id").and_then(Value::as_str)
                    .ok_or(DecisionSufficiencyRejectV1::InvalidJson)?.to_owned(),
                evidence_class_sha256: item.get("evidence_class_sha256").and_then(Value::as_str)
                    .ok_or(DecisionSufficiencyRejectV1::InvalidJson)?.to_owned(),
                minimum_independent_root_families: u32::try_from(
                    item.get("minimum_independent_root_families").and_then(Value::as_u64)
                        .ok_or(DecisionSufficiencyRejectV1::InvalidJson)?
                ).map_err(|_| DecisionSufficiencyRejectV1::InvalidJson)?,
                maximum_evidence_age_seconds: item.get("maximum_evidence_age_seconds")
                    .and_then(Value::as_u64)
                    .ok_or(DecisionSufficiencyRejectV1::InvalidJson)?,
            });
        }
        let result = Self {
            decision_need_id: string("decision_need_id")?,
            domain_id: string("domain_id")?,
            subject_scope_sha256: string("subject_scope_sha256")?,
            purpose_sha256: string("purpose_sha256")?,
            action_set_sha256: string("action_set_sha256")?,
            utility_profile_sha256: string("utility_profile_sha256")?,
            risk_limit_ppm: u32_value("risk_limit_ppm")?,
            minimum_confidence_level_ppm: u32_value("minimum_confidence_level_ppm")?,
            regret_tolerance_micros: i64_value("regret_tolerance_micros")?,
            sufficiency_policy_sha256: string("sufficiency_policy_sha256")?,
            calibration_profile_sha256: string("calibration_profile_sha256")?,
            calibration_receipt_sha256: string("calibration_receipt_sha256")?,
            maximum_evidence_age_seconds: u64_value("maximum_evidence_age_seconds")?,
            maximum_context_tokens: u64_value("maximum_context_tokens")?,
            maximum_tool_calls: u32_value("maximum_tool_calls")?,
            maximum_cost_micros: u64_value("maximum_cost_micros")?,
            maximum_latency_ms: u64_value("maximum_latency_ms")?,
            requirements,
        };
        result.validate()?;
        Ok(result)
    }

    pub fn requirement_ids(&self) -> Vec<String> {
        self.requirements.iter().map(|item| item.requirement_id.clone()).collect()
    }

    pub fn to_json_value(&self) -> Result<Value, DecisionSufficiencyRejectV1> {
        self.validate()?;
        Ok(json!({
            "schema_version": "mace_decision_need_v1",
            "decision_need_id": self.decision_need_id,
            "domain_id": self.domain_id,
            "subject_scope_sha256": self.subject_scope_sha256,
            "purpose_sha256": self.purpose_sha256,
            "action_set_sha256": self.action_set_sha256,
            "utility_profile_sha256": self.utility_profile_sha256,
            "risk_limit_ppm": self.risk_limit_ppm,
            "minimum_confidence_level_ppm": self.minimum_confidence_level_ppm,
            "regret_tolerance_micros": self.regret_tolerance_micros,
            "sufficiency_policy_sha256": self.sufficiency_policy_sha256,
            "calibration_profile_sha256": self.calibration_profile_sha256,
            "calibration_receipt_sha256": self.calibration_receipt_sha256,
            "maximum_evidence_age_seconds": self.maximum_evidence_age_seconds,
            "maximum_context_tokens": self.maximum_context_tokens,
            "maximum_tool_calls": self.maximum_tool_calls,
            "maximum_cost_micros": self.maximum_cost_micros,
            "maximum_latency_ms": self.maximum_latency_ms,
            "requirements": self.requirements.iter().map(|item| json!({
                "requirement_id": item.requirement_id,
                "evidence_class_sha256": item.evidence_class_sha256,
                "minimum_independent_root_families": item.minimum_independent_root_families,
                "maximum_evidence_age_seconds": item.maximum_evidence_age_seconds
            })).collect::<Vec<_>>()
        }))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionSufficiencyStatusV1 {
    Sufficient,
    Insufficient,
    Indeterminate,
    Abstained,
}

impl DecisionSufficiencyStatusV1 {
    fn as_str(self) -> &'static str {
        match self {
            Self::Sufficient => "sufficient",
            Self::Insufficient => "insufficient",
            Self::Indeterminate => "indeterminate",
            Self::Abstained => "abstained",
        }
    }

    fn parse(value: &str) -> Result<Self, DecisionSufficiencyRejectV1> {
        match value {
            "sufficient" => Ok(Self::Sufficient),
            "insufficient" => Ok(Self::Insufficient),
            "indeterminate" => Ok(Self::Indeterminate),
            "abstained" => Ok(Self::Abstained),
            _ => Err(DecisionSufficiencyRejectV1::InvalidReceipt),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionRequirementStatusV1 {
    Met,
    Unmet,
    Unknown,
}

impl DecisionRequirementStatusV1 {
    fn as_str(self) -> &'static str {
        match self {
            Self::Met => "met",
            Self::Unmet => "unmet",
            Self::Unknown => "unknown",
        }
    }

    fn parse(value: &str) -> Result<Self, DecisionSufficiencyRejectV1> {
        match value {
            "met" => Ok(Self::Met),
            "unmet" => Ok(Self::Unmet),
            "unknown" => Ok(Self::Unknown),
            _ => Err(DecisionSufficiencyRejectV1::InvalidReceipt),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionSufficiencyRequirementV1 {
    pub requirement_id: String,
    /// Must exactly match the consumer-owned DecisionNeed evidence class.
    pub evidence_class_sha256: String,
    pub status: DecisionRequirementStatusV1,
    /// Supporting evidence/assessment receipts. Empty is permitted for unmet
    /// or unknown requirements, but never for a requirement marked met.
    pub basis_receipt_sha256: Vec<String>,
    /// Sorted, unique root-family commitments with their independently
    /// checked transitive-lineage receipt commitments.
    pub independent_root_families: Vec<DecisionSufficiencyRootFamilyV1>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionSufficiencyReceiptV1 {
    pub run_id: String,
    pub query_invocation_id: String,
    pub domain_id: String,
    pub subject_scope_sha256: String,
    pub environment_revision_sha256: String,
    pub tool_profile_sha256: String,
    pub query_sha256: String,
    pub candidate_set_sha256: String,
    pub qualified_evidence_set_sha256: String,
    pub qualification_receipt_sha256: String,
    pub decision_need_sha256: String,
    pub purpose_sha256: String,
    pub sufficiency_policy_sha256: String,
    pub calibration_profile_sha256: String,
    pub status: DecisionSufficiencyStatusV1,
    pub requirements: Vec<DecisionSufficiencyRequirementV1>,
    pub risk_upper_bound_ppm: Option<u32>,
    pub risk_limit_ppm: Option<u32>,
    pub confidence_level_ppm: Option<u32>,
    pub calibration_receipt_sha256: Option<String>,
    /// Evaluator's conservative held-out decision-regret upper bound, in
    /// micro-utility units, and the consumer-defined regret tolerance.
    pub regret_upper_bound_micros: Option<u64>,
    pub regret_limit_micros: Option<u64>,
    /// Measured resource use for the exact evaluated decision/evidence set.
    pub observed_cost_micros: Option<u64>,
    pub observed_latency_ms: Option<u64>,
    pub observed_context_tokens: Option<u64>,
    pub observed_tool_calls: Option<u32>,
    pub maximum_evidence_age_seconds: Option<u64>,
    pub issued_at_unix: u64,
    pub expires_at_unix: u64,
    pub issuer_profile_sha256: String,
    pub key_id: String,
    /// DSSE carries this signature in its outer envelope. It is reconstructed
    /// here only to preserve the existing typed receipt schema.
    pub signature: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecisionSufficiencyRejectV1 {
    InvalidJson,
    InvalidReceipt,
    InvalidDecisionNeed,
    InvalidDigest,
    InvalidIdentifier,
    RequirementOrderInvalid,
    DuplicateRequirement,
    EmptyBasis,
    EvidenceClassMismatch,
    InsufficientIndependentRootFamilies,
    SufficientRequirementsNotMet,
    SufficientRiskBoundInvalid,
    BindingMismatch,
    SubjectMismatch,
    WrongPredicateType,
    WrongEvaluatorRole,
    SharedQualificationAuthority,
    ProfileDigestMismatch,
    UnknownOrRevokedSigner,
    NotYetValid,
    Expired,
    InvalidSignature,
    Qualification(EvidenceQualificationRejectV1),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionSufficiencyUseBindingV1<'a> {
    pub run_id: &'a str,
    pub query_invocation_id: &'a str,
    pub domain_id: &'a str,
    pub subject_scope_sha256: &'a str,
    pub environment_revision_sha256: &'a str,
    pub tool_profile_sha256: &'a str,
    pub query_sha256: &'a str,
    pub candidate_set_sha256: &'a str,
    pub qualified_evidence_set_sha256: &'a str,
    pub qualification_receipt_sha256: &'a str,
    pub decision_need_sha256: &'a str,
    pub purpose_sha256: &'a str,
    pub lifecycle_snapshot_sha256: &'a str,
    pub decision_need: &'a DecisionNeedV1,
    pub sufficiency_policy_sha256: &'a str,
    pub calibration_profile_sha256: &'a str,
    pub calibration_receipt_sha256: &'a str,
    pub required_requirement_ids: &'a [String],
    pub risk_limit_ppm: u32,
    pub minimum_confidence_level_ppm: u32,
    pub issuer_profile_sha256: &'a str,
    pub now_unix: u64,
}

impl DecisionSufficiencyReceiptV1 {
    pub fn from_json_value(value: &Value) -> Result<Self, DecisionSufficiencyRejectV1> {
        let object = value.as_object().ok_or(DecisionSufficiencyRejectV1::InvalidJson)?;
        let known = [
            "schema_version", "run_id", "query_invocation_id", "domain_id",
            "subject_scope_sha256", "environment_revision_sha256", "tool_profile_sha256",
            "query_sha256", "candidate_set_sha256", "qualified_evidence_set_sha256",
            "qualification_receipt_sha256", "decision_need_sha256", "purpose_sha256",
            "sufficiency_policy_sha256", "calibration_profile_sha256", "status",
            "requirements", "risk_upper_bound_ppm", "risk_limit_ppm",
            "confidence_level_ppm", "calibration_receipt_sha256",
            "regret_upper_bound_micros", "regret_limit_micros",
            "observed_cost_micros", "observed_latency_ms",
            "observed_context_tokens", "observed_tool_calls",
            "maximum_evidence_age_seconds", "issued_at_unix",
            "expires_at_unix", "issuer_profile_sha256", "key_id", "signature",
        ];
        if object.keys().any(|key| !known.contains(&key.as_str()))
            || object.get("schema_version").and_then(Value::as_str)
                != Some("mace_decision_sufficiency_receipt_v1")
        {
            return Err(DecisionSufficiencyRejectV1::InvalidJson);
        }

        let required_string = |field: &str| -> Result<String, DecisionSufficiencyRejectV1> {
            object.get(field).and_then(Value::as_str).map(str::to_owned)
                .ok_or(DecisionSufficiencyRejectV1::InvalidJson)
        };
        let required_u64 = |field: &str| -> Result<u64, DecisionSufficiencyRejectV1> {
            object.get(field).and_then(Value::as_u64)
                .ok_or(DecisionSufficiencyRejectV1::InvalidJson)
        };
        let optional_u32 = |field: &str| -> Result<Option<u32>, DecisionSufficiencyRejectV1> {
            match object.get(field) {
                None => Ok(None),
                Some(value) => value.as_u64()
                    .and_then(|number| u32::try_from(number).ok())
                    .map(Some)
                    .ok_or(DecisionSufficiencyRejectV1::InvalidJson),
            }
        };
        let optional_u64 = |field: &str| -> Result<Option<u64>, DecisionSufficiencyRejectV1> {
            match object.get(field) {
                None => Ok(None),
                Some(value) => value.as_u64().map(Some)
                    .ok_or(DecisionSufficiencyRejectV1::InvalidJson),
            }
        };
        let optional_digest = |field: &str| -> Result<Option<String>, DecisionSufficiencyRejectV1> {
            match object.get(field) {
                None => Ok(None),
                Some(Value::String(value)) => Ok(Some(value.clone())),
                _ => Err(DecisionSufficiencyRejectV1::InvalidJson),
            }
        };

        let requirement_values = object.get("requirements").and_then(Value::as_array)
            .ok_or(DecisionSufficiencyRejectV1::InvalidJson)?;
        if requirement_values.is_empty() || requirement_values.len() > 1024 {
            return Err(DecisionSufficiencyRejectV1::InvalidReceipt);
        }
        let mut requirements = Vec::with_capacity(requirement_values.len());
        for item in requirement_values {
            let entry = item.as_object().ok_or(DecisionSufficiencyRejectV1::InvalidJson)?;
            if entry.len() != 5
                || entry.keys().any(|key| ![
                    "requirement_id", "evidence_class_sha256", "status",
                    "basis_receipt_sha256", "independent_root_families"
                ].contains(&key.as_str()))
            {
                return Err(DecisionSufficiencyRejectV1::InvalidJson);
            }
            let requirement_id = entry.get("requirement_id").and_then(Value::as_str)
                .ok_or(DecisionSufficiencyRejectV1::InvalidJson)?.to_owned();
            let evidence_class_sha256 = entry.get("evidence_class_sha256")
                .and_then(Value::as_str).ok_or(DecisionSufficiencyRejectV1::InvalidJson)?
                .to_owned();
            let status = DecisionRequirementStatusV1::parse(
                entry.get("status").and_then(Value::as_str)
                    .ok_or(DecisionSufficiencyRejectV1::InvalidJson)?
            )?;
            let basis_values = entry.get("basis_receipt_sha256").and_then(Value::as_array)
                .ok_or(DecisionSufficiencyRejectV1::InvalidJson)?;
            if basis_values.len() > 256 {
                return Err(DecisionSufficiencyRejectV1::InvalidReceipt);
            }
            if status == DecisionRequirementStatusV1::Met && basis_values.is_empty() {
                return Err(DecisionSufficiencyRejectV1::EmptyBasis);
            }
            let basis_receipt_sha256 = basis_values.iter()
                .map(|value| value.as_str().map(str::to_owned)
                    .ok_or(DecisionSufficiencyRejectV1::InvalidJson))
                .collect::<Result<Vec<_>, _>>()?;
            let family_values = entry.get("independent_root_families")
                .and_then(Value::as_array).ok_or(DecisionSufficiencyRejectV1::InvalidJson)?;
            if family_values.len() > 10_000 {
                return Err(DecisionSufficiencyRejectV1::InvalidReceipt);
            }
            let mut independent_root_families = Vec::with_capacity(family_values.len());
            for family_value in family_values {
                let family = family_value.as_object()
                    .ok_or(DecisionSufficiencyRejectV1::InvalidJson)?;
                if family.len() != 2 || family.keys().any(|key| ![
                    "root_family_sha256", "provenance_receipt_sha256"
                ].contains(&key.as_str())) {
                    return Err(DecisionSufficiencyRejectV1::InvalidJson);
                }
                independent_root_families.push(DecisionSufficiencyRootFamilyV1 {
                    root_family_sha256: family.get("root_family_sha256")
                        .and_then(Value::as_str).ok_or(DecisionSufficiencyRejectV1::InvalidJson)?
                        .to_owned(),
                    provenance_receipt_sha256: family.get("provenance_receipt_sha256")
                        .and_then(Value::as_str).ok_or(DecisionSufficiencyRejectV1::InvalidJson)?
                        .to_owned(),
                });
            }
            requirements.push(DecisionSufficiencyRequirementV1 {
                requirement_id,
                evidence_class_sha256,
                status,
                basis_receipt_sha256,
                independent_root_families,
            });
        }

        let signature_values = object.get("signature").and_then(Value::as_array)
            .ok_or(DecisionSufficiencyRejectV1::InvalidJson)?;
        if signature_values.is_empty() || signature_values.len() > 4096 {
            return Err(DecisionSufficiencyRejectV1::InvalidReceipt);
        }
        let signature = signature_values.iter()
            .map(|value| value.as_u64().and_then(|byte| u8::try_from(byte).ok())
                .ok_or(DecisionSufficiencyRejectV1::InvalidJson))
            .collect::<Result<Vec<_>, _>>()?;

        let receipt = Self {
            run_id: required_string("run_id")?,
            query_invocation_id: required_string("query_invocation_id")?,
            domain_id: required_string("domain_id")?,
            subject_scope_sha256: required_string("subject_scope_sha256")?,
            environment_revision_sha256: required_string("environment_revision_sha256")?,
            tool_profile_sha256: required_string("tool_profile_sha256")?,
            query_sha256: required_string("query_sha256")?,
            candidate_set_sha256: required_string("candidate_set_sha256")?,
            qualified_evidence_set_sha256: required_string("qualified_evidence_set_sha256")?,
            qualification_receipt_sha256: required_string("qualification_receipt_sha256")?,
            decision_need_sha256: required_string("decision_need_sha256")?,
            purpose_sha256: required_string("purpose_sha256")?,
            sufficiency_policy_sha256: required_string("sufficiency_policy_sha256")?,
            calibration_profile_sha256: required_string("calibration_profile_sha256")?,
            status: DecisionSufficiencyStatusV1::parse(
                object.get("status").and_then(Value::as_str)
                    .ok_or(DecisionSufficiencyRejectV1::InvalidJson)?
            )?,
            requirements,
            risk_upper_bound_ppm: optional_u32("risk_upper_bound_ppm")?,
            risk_limit_ppm: optional_u32("risk_limit_ppm")?,
            confidence_level_ppm: optional_u32("confidence_level_ppm")?,
            calibration_receipt_sha256: optional_digest("calibration_receipt_sha256")?,
            regret_upper_bound_micros: optional_u64("regret_upper_bound_micros")?,
            regret_limit_micros: optional_u64("regret_limit_micros")?,
            observed_cost_micros: optional_u64("observed_cost_micros")?,
            observed_latency_ms: optional_u64("observed_latency_ms")?,
            observed_context_tokens: optional_u64("observed_context_tokens")?,
            observed_tool_calls: optional_u32("observed_tool_calls")?,
            maximum_evidence_age_seconds: optional_u64("maximum_evidence_age_seconds")?,
            issued_at_unix: required_u64("issued_at_unix")?,
            expires_at_unix: required_u64("expires_at_unix")?,
            issuer_profile_sha256: required_string("issuer_profile_sha256")?,
            key_id: required_string("key_id")?,
            signature,
        };
        receipt.validate_shape()?;
        Ok(receipt)
    }

    pub fn to_json_value(&self) -> Value {
        let mut value = json!({
            "schema_version": "mace_decision_sufficiency_receipt_v1",
            "run_id": self.run_id,
            "query_invocation_id": self.query_invocation_id,
            "domain_id": self.domain_id,
            "subject_scope_sha256": self.subject_scope_sha256,
            "environment_revision_sha256": self.environment_revision_sha256,
            "tool_profile_sha256": self.tool_profile_sha256,
            "query_sha256": self.query_sha256,
            "candidate_set_sha256": self.candidate_set_sha256,
            "qualified_evidence_set_sha256": self.qualified_evidence_set_sha256,
            "qualification_receipt_sha256": self.qualification_receipt_sha256,
            "decision_need_sha256": self.decision_need_sha256,
            "purpose_sha256": self.purpose_sha256,
            "sufficiency_policy_sha256": self.sufficiency_policy_sha256,
            "calibration_profile_sha256": self.calibration_profile_sha256,
            "status": self.status.as_str(),
            "requirements": self.requirements.iter().map(|requirement| json!({
                "requirement_id": requirement.requirement_id,
                "evidence_class_sha256": requirement.evidence_class_sha256,
                "status": requirement.status.as_str(),
                "basis_receipt_sha256": requirement.basis_receipt_sha256,
                "independent_root_families": requirement.independent_root_families.iter().map(|family| json!({
                    "root_family_sha256": family.root_family_sha256,
                    "provenance_receipt_sha256": family.provenance_receipt_sha256,
                })).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
            "issued_at_unix": self.issued_at_unix,
            "expires_at_unix": self.expires_at_unix,
            "issuer_profile_sha256": self.issuer_profile_sha256,
            "key_id": self.key_id,
            "signature": self.signature.iter().copied().map(Value::from).collect::<Vec<_>>(),
        });
        let object = value.as_object_mut().expect("json macro returns object");
        if let Some(v) = self.risk_upper_bound_ppm {
            object.insert("risk_upper_bound_ppm".into(), Value::from(v));
        }
        if let Some(v) = self.risk_limit_ppm {
            object.insert("risk_limit_ppm".into(), Value::from(v));
        }
        if let Some(v) = self.confidence_level_ppm {
            object.insert("confidence_level_ppm".into(), Value::from(v));
        }
        if let Some(v) = &self.calibration_receipt_sha256 {
            object.insert("calibration_receipt_sha256".into(), Value::String(v.clone()));
        }
        for (field, value) in [
            ("regret_upper_bound_micros", self.regret_upper_bound_micros),
            ("regret_limit_micros", self.regret_limit_micros),
            ("observed_cost_micros", self.observed_cost_micros),
            ("observed_latency_ms", self.observed_latency_ms),
            ("observed_context_tokens", self.observed_context_tokens),
            ("maximum_evidence_age_seconds", self.maximum_evidence_age_seconds),
        ] {
            if let Some(v) = value {
                object.insert(field.into(), Value::from(v));
            }
        }
        if let Some(v) = self.observed_tool_calls {
            object.insert("observed_tool_calls".into(), Value::from(v));
        }
        value
    }

    pub fn unsigned_predicate_value(&self) -> Value {
        let mut value = self.to_json_value();
        value.as_object_mut().expect("receipt object").remove("signature");
        value
    }

    pub fn receipt_sha256(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"luminous:mace:decision-sufficiency-receipt:v1\0");
        for value in [
            self.run_id.as_str(),
            self.query_invocation_id.as_str(),
            self.domain_id.as_str(),
            self.subject_scope_sha256.as_str(),
            self.environment_revision_sha256.as_str(),
            self.tool_profile_sha256.as_str(),
            self.query_sha256.as_str(),
            self.candidate_set_sha256.as_str(),
            self.qualified_evidence_set_sha256.as_str(),
            self.qualification_receipt_sha256.as_str(),
            self.decision_need_sha256.as_str(),
            self.purpose_sha256.as_str(),
            self.sufficiency_policy_sha256.as_str(),
            self.calibration_profile_sha256.as_str(),
            self.issuer_profile_sha256.as_str(),
            self.key_id.as_str(),
        ] {
            hash_field(&mut hasher, value);
        }
        hasher.update([match self.status {
            DecisionSufficiencyStatusV1::Sufficient => 1,
            DecisionSufficiencyStatusV1::Insufficient => 2,
            DecisionSufficiencyStatusV1::Indeterminate => 3,
            DecisionSufficiencyStatusV1::Abstained => 4,
        }]);
        hasher.update(self.issued_at_unix.to_be_bytes());
        hasher.update(self.expires_at_unix.to_be_bytes());
        for value in [
            self.risk_upper_bound_ppm,
            self.risk_limit_ppm,
            self.confidence_level_ppm,
        ] {
            match value {
                Some(v) => {
                    hasher.update([1]);
                    hasher.update(v.to_be_bytes());
                }
                None => hasher.update([0]),
            }
        }
        if let Some(value) = &self.calibration_receipt_sha256 {
            hasher.update([1]);
            hash_field(&mut hasher, value);
        } else {
            hasher.update([0]);
        }
        for value in [
            self.regret_upper_bound_micros,
            self.regret_limit_micros,
            self.observed_cost_micros,
            self.observed_latency_ms,
            self.observed_context_tokens,
            self.maximum_evidence_age_seconds,
        ] {
            match value {
                Some(number) => {
                    hasher.update([1]);
                    hasher.update(number.to_be_bytes());
                }
                None => hasher.update([0]),
            }
        }
        match self.observed_tool_calls {
            Some(number) => {
                hasher.update([1]);
                hasher.update(number.to_be_bytes());
            }
            None => hasher.update([0]),
        }
        hasher.update((self.requirements.len() as u64).to_be_bytes());
        for requirement in &self.requirements {
            hash_field(&mut hasher, &requirement.requirement_id);
            hash_field(&mut hasher, &requirement.evidence_class_sha256);
            hasher.update([match requirement.status {
                DecisionRequirementStatusV1::Met => 1,
                DecisionRequirementStatusV1::Unmet => 2,
                DecisionRequirementStatusV1::Unknown => 3,
            }]);
            hasher.update((requirement.basis_receipt_sha256.len() as u64).to_be_bytes());
            for digest in &requirement.basis_receipt_sha256 {
                hash_field(&mut hasher, digest);
            }
            hasher.update((requirement.independent_root_families.len() as u64).to_be_bytes());
            for family in &requirement.independent_root_families {
                hash_field(&mut hasher, &family.root_family_sha256);
                hash_field(&mut hasher, &family.provenance_receipt_sha256);
            }
        }
        hex_digest(&hasher.finalize())
    }

    fn validate_shape(&self) -> Result<(), DecisionSufficiencyRejectV1> {
        for value in [
            self.subject_scope_sha256.as_str(),
            self.environment_revision_sha256.as_str(),
            self.tool_profile_sha256.as_str(),
            self.query_sha256.as_str(),
            self.candidate_set_sha256.as_str(),
            self.qualified_evidence_set_sha256.as_str(),
            self.qualification_receipt_sha256.as_str(),
            self.decision_need_sha256.as_str(),
            self.purpose_sha256.as_str(),
            self.sufficiency_policy_sha256.as_str(),
            self.calibration_profile_sha256.as_str(),
            self.issuer_profile_sha256.as_str(),
            self.key_id.as_str(),
        ] {
            if !valid_sha256(value) {
                return Err(DecisionSufficiencyRejectV1::InvalidDigest);
            }
        }
        for (value, maximum) in [
            (self.run_id.as_str(), 256),
            (self.query_invocation_id.as_str(), 256),
            (self.domain_id.as_str(), 512),
        ] {
            if !valid_atom(value) || value.len() > maximum {
                return Err(DecisionSufficiencyRejectV1::InvalidIdentifier);
            }
        }
        if self.issued_at_unix >= self.expires_at_unix
            || self.requirements.is_empty()
            || self.requirements.len() > 1024
            || self.signature.len() != 64
        {
            return Err(DecisionSufficiencyRejectV1::InvalidReceipt);
        }
        if self.requirements.windows(2).any(|pair| {
            pair[0].requirement_id >= pair[1].requirement_id
        }) {
            return Err(DecisionSufficiencyRejectV1::RequirementOrderInvalid);
        }
        for requirement in &self.requirements {
            if !valid_atom(&requirement.requirement_id)
                || requirement.requirement_id.len() > 256
            {
                return Err(DecisionSufficiencyRejectV1::InvalidIdentifier);
            }
            let mut lineage_receipts = BTreeSet::new();
            if requirement.independent_root_families.iter()
                .any(|family| !lineage_receipts.insert(family.provenance_receipt_sha256.as_str()))
            {
                return Err(DecisionSufficiencyRejectV1::InvalidReceipt);
            }
            if requirement.basis_receipt_sha256.len() > 256 {
                return Err(DecisionSufficiencyRejectV1::InvalidReceipt);
            }
            if requirement.status == DecisionRequirementStatusV1::Met
                && requirement.basis_receipt_sha256.is_empty()
            {
                return Err(DecisionSufficiencyRejectV1::EmptyBasis);
            }
            if !valid_sha256(&requirement.evidence_class_sha256)
                || requirement.basis_receipt_sha256.windows(2).any(|pair| pair[0] >= pair[1])
                || requirement.basis_receipt_sha256.iter().any(|v| !valid_sha256(v))
                || requirement.independent_root_families.len() > 10_000
                || requirement.independent_root_families.windows(2).any(|pair| {
                    pair[0].root_family_sha256 >= pair[1].root_family_sha256
                })
                || requirement.independent_root_families.iter().any(|family| {
                    !valid_sha256(&family.root_family_sha256)
                        || !valid_sha256(&family.provenance_receipt_sha256)
                })
            {
                return Err(DecisionSufficiencyRejectV1::InvalidDigest);
            }
        }
        for value in [
            self.risk_upper_bound_ppm,
            self.risk_limit_ppm,
        ].into_iter().flatten() {
            if value > 1_000_000 {
                return Err(DecisionSufficiencyRejectV1::InvalidReceipt);
            }
        }
        if let Some(confidence) = self.confidence_level_ppm {
            if !(500_000..=999_999).contains(&confidence) {
                return Err(DecisionSufficiencyRejectV1::InvalidReceipt);
            }
        }
        if self.calibration_receipt_sha256.as_ref().is_some_and(|v| !valid_sha256(v)) {
            return Err(DecisionSufficiencyRejectV1::InvalidDigest);
        }
        if self.regret_upper_bound_micros.is_some_and(|v| v > 1_000_000_000_000)
            || self.regret_limit_micros.is_some_and(|v| v > 1_000_000_000_000)
            || self.observed_cost_micros.is_some_and(|v| v > 1_000_000_000_000)
            || self.observed_latency_ms.is_some_and(|v| v > 86_400_000)
            || self.observed_context_tokens.is_some_and(|v| v > 1_000_000_000_000)
            || self.observed_tool_calls.is_some_and(|v| v > 1_000_000)
            || self.maximum_evidence_age_seconds.is_some_and(|v| v > 315_360_000)
        {
            return Err(DecisionSufficiencyRejectV1::InvalidReceipt);
        }
        if self.status == DecisionSufficiencyStatusV1::Sufficient {
            if self.requirements.iter().any(|r| r.status != DecisionRequirementStatusV1::Met) {
                return Err(DecisionSufficiencyRejectV1::SufficientRequirementsNotMet);
            }
            let (Some(upper), Some(limit), Some(confidence), Some(_calibration)) = (
                self.risk_upper_bound_ppm,
                self.risk_limit_ppm,
                self.confidence_level_ppm,
                self.calibration_receipt_sha256.as_ref(),
            ) else {
                return Err(DecisionSufficiencyRejectV1::SufficientRiskBoundInvalid);
            };
            if upper > limit || confidence < 500_000 {
                return Err(DecisionSufficiencyRejectV1::SufficientRiskBoundInvalid);
            }
            let (
                Some(regret_upper),
                Some(regret_limit),
                Some(_cost),
                Some(_latency),
                Some(_tokens),
                Some(_tool_calls),
                Some(_max_age),
            ) = (
                self.regret_upper_bound_micros,
                self.regret_limit_micros,
                self.observed_cost_micros,
                self.observed_latency_ms,
                self.observed_context_tokens,
                self.observed_tool_calls,
                self.maximum_evidence_age_seconds,
            ) else {
                return Err(DecisionSufficiencyRejectV1::SufficientRiskBoundInvalid);
            };
            if regret_upper > regret_limit {
                return Err(DecisionSufficiencyRejectV1::SufficientRiskBoundInvalid);
            }
        }
        Ok(())
    }
}

/// Verified evaluator assessment. This is evidence that one exact DecisionNeed
/// met the configured sufficiency policy; it is never an action authorization.
#[derive(Clone, Debug, PartialEq)]
pub struct VerifiedDecisionSufficiencyV1 {
    receipt: DecisionSufficiencyReceiptV1,
    pae: Vec<u8>,
    signature: Vec<u8>,
}

impl VerifiedDecisionSufficiencyV1 {
    /// Expose the immutable, already verified receipt to follow-on verifiers.
    /// Callers must still perform validate_at_use with current trust profiles.
    pub fn receipt(&self) -> &DecisionSufficiencyReceiptV1 {
        &self.receipt
    }

    pub fn status(&self) -> DecisionSufficiencyStatusV1 {
        self.receipt.status
    }

    pub fn is_sufficient(&self) -> bool {
        self.receipt.status == DecisionSufficiencyStatusV1::Sufficient
    }

    pub fn receipt_sha256(&self) -> String {
        self.receipt.receipt_sha256()
    }

    pub fn validate_at_use(
        &self,
        qualified: &QualifiedCandidateEvidenceBatchV1,
        qualification_binding: &EvidenceQualificationBindingV1<'_>,
        current_qualification_verifier: &impl EvidenceQualificationSignatureVerifierV1,
        binding: &DecisionSufficiencyUseBindingV1<'_>,
        current_evaluator_profile: &PinnedEd25519QualificationVerifierV1,
    ) -> Result<(), DecisionSufficiencyRejectV1> {
        qualified.validate_at_use(qualification_binding, current_qualification_verifier)
            .map_err(DecisionSufficiencyRejectV1::Qualification)?;
        if qualification_binding.now_unix != binding.now_unix {
            return Err(DecisionSufficiencyRejectV1::BindingMismatch);
        }
        validate_binding_against_qualified(qualified, binding)?;
        validate_evaluator_role(current_evaluator_profile, qualified, &self.receipt)?;
        validate_receipt_binding(&self.receipt, binding)?;
        if binding.now_unix < self.receipt.issued_at_unix {
            return Err(DecisionSufficiencyRejectV1::NotYetValid);
        }
        if binding.now_unix >= self.receipt.expires_at_unix {
            return Err(DecisionSufficiencyRejectV1::Expired);
        }
        current_evaluator_profile.verify_message(
            &self.receipt.issuer_profile_sha256,
            &self.receipt.key_id,
            &self.pae,
            &self.signature,
            self.receipt.issued_at_unix,
            binding.now_unix,
        ).map_err(|error| match error {
            crate::evidence_qualification_gate::EvidenceQualificationRejectV1::RevokedKey
            | crate::evidence_qualification_gate::EvidenceQualificationRejectV1::UnknownKey =>
                DecisionSufficiencyRejectV1::UnknownOrRevokedSigner,
            crate::evidence_qualification_gate::EvidenceQualificationRejectV1::TrustProfileDigestMismatch =>
                DecisionSufficiencyRejectV1::ProfileDigestMismatch,
            crate::evidence_qualification_gate::EvidenceQualificationRejectV1::KeyNotYetValid =>
                DecisionSufficiencyRejectV1::NotYetValid,
            crate::evidence_qualification_gate::EvidenceQualificationRejectV1::KeyExpired =>
                DecisionSufficiencyRejectV1::Expired,
            _ => DecisionSufficiencyRejectV1::InvalidSignature,
        })
    }
}

/// Verify a separately issued DecisionSufficiency DSSE/in-toto attestation
/// against one exact qualified projection and the consumer's DecisionNeed.
pub fn verify_decision_sufficiency_v1(
    qualified: &QualifiedCandidateEvidenceBatchV1,
    qualification_binding: &EvidenceQualificationBindingV1<'_>,
    current_qualification_verifier: &impl EvidenceQualificationSignatureVerifierV1,
    binding: &DecisionSufficiencyUseBindingV1<'_>,
    envelope_json: &str,
    evaluator_profile: &PinnedEd25519QualificationVerifierV1,
) -> Result<VerifiedDecisionSufficiencyV1, DecisionSufficiencyRejectV1> {
    qualified.validate_at_use(qualification_binding, current_qualification_verifier)
        .map_err(DecisionSufficiencyRejectV1::Qualification)?;
    if qualification_binding.now_unix != binding.now_unix {
        return Err(DecisionSufficiencyRejectV1::BindingMismatch);
    }
    validate_binding_against_qualified(qualified, binding)?;
    if evaluator_profile.profile_sha256() != binding.issuer_profile_sha256 {
        return Err(DecisionSufficiencyRejectV1::ProfileDigestMismatch);
    }
    if evaluator_profile.verifier_id() != DECISION_SUFFICIENCY_VERIFIER_ID_V1 {
        return Err(DecisionSufficiencyRejectV1::WrongEvaluatorRole);
    }
    if evaluator_profile.profile_sha256() == qualified.qualification_verifier_profile_sha256() {
        return Err(DecisionSufficiencyRejectV1::SharedQualificationAuthority);
    }

    let envelope = parse_dsse_envelope_v1(envelope_json, DSSE_PAYLOAD_TYPE_V1)
        .map_err(|_| DecisionSufficiencyRejectV1::InvalidJson)?;
    let statement_value = parse_strict_json_bounded(&envelope.payload, 8 * 1024 * 1024)
        .map_err(|_| DecisionSufficiencyRejectV1::InvalidJson)?;
    let statement = statement_value.as_object()
        .ok_or(DecisionSufficiencyRejectV1::InvalidJson)?;
    if statement.get("_type").and_then(Value::as_str) != Some(IN_TOTO_STATEMENT_TYPE_V1)
        || statement.get("predicateType").and_then(Value::as_str)
            != Some(MACE_DECISION_SUFFICIENCY_PREDICATE_TYPE_V1)
    {
        return Err(DecisionSufficiencyRejectV1::WrongPredicateType);
    }
    let subjects = statement.get("subject").and_then(Value::as_array)
        .ok_or(DecisionSufficiencyRejectV1::InvalidJson)?;
    if subjects.len() != 1 {
        return Err(DecisionSufficiencyRejectV1::SubjectMismatch);
    }
    let subject = subjects[0].as_object()
        .ok_or(DecisionSufficiencyRejectV1::InvalidJson)?;
    let expected_name = format!(
        "urn:luminous:mace:qualified-evidence-set:v1:{}",
        binding.qualified_evidence_set_sha256
    );
    if subject.get("name").and_then(Value::as_str) != Some(expected_name.as_str()) {
        return Err(DecisionSufficiencyRejectV1::SubjectMismatch);
    }
    let digest = subject.get("digest").and_then(Value::as_object)
        .ok_or(DecisionSufficiencyRejectV1::InvalidJson)?;
    if digest.get("sha256").and_then(Value::as_str)
        != Some(binding.qualified_evidence_set_sha256)
    {
        return Err(DecisionSufficiencyRejectV1::SubjectMismatch);
    }

    let predicate = statement.get("predicate")
        .ok_or(DecisionSufficiencyRejectV1::InvalidJson)?;
    let mut receipt_value = predicate.clone();
    receipt_value.as_object_mut()
        .ok_or(DecisionSufficiencyRejectV1::InvalidJson)?
        .insert("signature".into(), Value::Array(
            envelope.signature.iter().copied().map(Value::from).collect()
        ));
    let receipt = DecisionSufficiencyReceiptV1::from_json_value(&receipt_value)?;
    if envelope.key_id_hint.as_ref().is_some_and(|hint| hint != &receipt.key_id) {
        return Err(DecisionSufficiencyRejectV1::InvalidSignature);
    }
    if receipt.unsigned_predicate_value() != *predicate {
        return Err(DecisionSufficiencyRejectV1::InvalidReceipt);
    }
    validate_receipt_binding(&receipt, binding)?;
    validate_evaluator_role(evaluator_profile, qualified, &receipt)?;
    if binding.now_unix < receipt.issued_at_unix {
        return Err(DecisionSufficiencyRejectV1::NotYetValid);
    }
    if binding.now_unix >= receipt.expires_at_unix {
        return Err(DecisionSufficiencyRejectV1::Expired);
    }
    evaluator_profile.verify_message(
        &receipt.issuer_profile_sha256,
        &receipt.key_id,
        &envelope.pae,
        &envelope.signature,
        receipt.issued_at_unix,
        binding.now_unix,
    ).map_err(|error| match error {
        crate::evidence_qualification_gate::EvidenceQualificationRejectV1::RevokedKey
        | crate::evidence_qualification_gate::EvidenceQualificationRejectV1::UnknownKey =>
            DecisionSufficiencyRejectV1::UnknownOrRevokedSigner,
        crate::evidence_qualification_gate::EvidenceQualificationRejectV1::TrustProfileDigestMismatch =>
            DecisionSufficiencyRejectV1::ProfileDigestMismatch,
        crate::evidence_qualification_gate::EvidenceQualificationRejectV1::KeyNotYetValid =>
            DecisionSufficiencyRejectV1::NotYetValid,
        crate::evidence_qualification_gate::EvidenceQualificationRejectV1::KeyExpired =>
            DecisionSufficiencyRejectV1::Expired,
        _ => DecisionSufficiencyRejectV1::InvalidSignature,
    })?;

    Ok(VerifiedDecisionSufficiencyV1 {
        receipt,
        pae: envelope.pae,
        signature: envelope.signature,
    })
}

fn validate_binding_against_qualified(
    qualified: &QualifiedCandidateEvidenceBatchV1,
    binding: &DecisionSufficiencyUseBindingV1<'_>,
) -> Result<(), DecisionSufficiencyRejectV1> {
    let need = binding.decision_need;
    let need_digest = need.digest()?;
    if qualified.decision_need_sha256() != need_digest.as_str()
        || qualified.domain_id() != need.domain_id.as_str()
        || qualified.subject_scope_sha256() != need.subject_scope_sha256.as_str()
        || qualified.purpose_sha256() != need.purpose_sha256.as_str()
        || binding.decision_need_sha256 != need_digest.as_str()
        || binding.purpose_sha256 != need.purpose_sha256.as_str()
        || binding.sufficiency_policy_sha256 != need.sufficiency_policy_sha256.as_str()
        || binding.calibration_profile_sha256 != need.calibration_profile_sha256.as_str()
        || binding.calibration_receipt_sha256 != need.calibration_receipt_sha256.as_str()
        || binding.risk_limit_ppm != need.risk_limit_ppm
        || binding.minimum_confidence_level_ppm != need.minimum_confidence_level_ppm
        || binding.required_requirement_ids.iter().map(String::as_str)
            .ne(need.requirements.iter().map(|item| item.requirement_id.as_str()))
    {
        return Err(DecisionSufficiencyRejectV1::BindingMismatch);
    }
    for (actual, expected) in [
        (qualified.run_id(), binding.run_id),
        (qualified.query_invocation_id(), binding.query_invocation_id),
        (qualified.domain_id(), binding.domain_id),
        (qualified.subject_scope_sha256(), binding.subject_scope_sha256),
        (qualified.environment_revision_sha256(), binding.environment_revision_sha256),
        (qualified.tool_profile_sha256(), binding.tool_profile_sha256),
        (qualified.batch().query_sha256.as_str(), binding.query_sha256),
        (qualified.source_candidate_set_sha256(), binding.candidate_set_sha256),
        (qualified.qualified_candidate_set_sha256(), binding.qualified_evidence_set_sha256),
        (qualified.qualification_receipt_sha256(), binding.qualification_receipt_sha256),
        (qualified.decision_need_sha256(), binding.decision_need_sha256),
        (qualified.purpose_sha256(), binding.purpose_sha256),
        (qualified.lifecycle_snapshot_sha256(), binding.lifecycle_snapshot_sha256),
    ] {
        if actual != expected {
            return Err(DecisionSufficiencyRejectV1::BindingMismatch);
        }
    }
    for value in [
        binding.subject_scope_sha256, binding.environment_revision_sha256,
        binding.tool_profile_sha256, binding.query_sha256, binding.candidate_set_sha256,
        binding.qualified_evidence_set_sha256, binding.qualification_receipt_sha256,
        binding.decision_need_sha256, binding.purpose_sha256, binding.lifecycle_snapshot_sha256,
        binding.sufficiency_policy_sha256, binding.calibration_profile_sha256,
        binding.issuer_profile_sha256,
    ] {
        if !valid_sha256(value) {
            return Err(DecisionSufficiencyRejectV1::InvalidDigest);
        }
    }
    if !valid_atom(binding.run_id) || !valid_atom(binding.query_invocation_id)
        || !valid_atom(binding.domain_id)
    {
        return Err(DecisionSufficiencyRejectV1::InvalidIdentifier);
    }
    Ok(())
}

fn validate_receipt_binding(
    receipt: &DecisionSufficiencyReceiptV1,
    binding: &DecisionSufficiencyUseBindingV1<'_>,
) -> Result<(), DecisionSufficiencyRejectV1> {
    if binding.required_requirement_ids.is_empty()
        || binding.required_requirement_ids.windows(2).any(|pair| pair[0] >= pair[1])
        || receipt.requirements.iter().map(|item| item.requirement_id.as_str())
            .ne(binding.required_requirement_ids.iter().map(String::as_str))
        || receipt.requirements.len() != binding.decision_need.requirements.len()
    {
        return Err(DecisionSufficiencyRejectV1::BindingMismatch);
    }
    for (claim, required) in receipt.requirements.iter()
        .zip(binding.decision_need.requirements.iter())
    {
        if claim.evidence_class_sha256 != required.evidence_class_sha256 {
            return Err(DecisionSufficiencyRejectV1::EvidenceClassMismatch);
        }
        if claim.status == DecisionRequirementStatusV1::Met
            && claim.independent_root_families.len()
                < required.minimum_independent_root_families as usize
        {
            return Err(DecisionSufficiencyRejectV1::InsufficientIndependentRootFamilies);
        }
    }
    if binding.risk_limit_ppm > 1_000_000
        || !(500_000..=999_999).contains(&binding.minimum_confidence_level_ppm)
        || !valid_sha256(binding.calibration_receipt_sha256)
    {
        return Err(DecisionSufficiencyRejectV1::InvalidReceipt);
    }
    if receipt.risk_limit_ppm.is_some_and(|risk| risk != binding.risk_limit_ppm)
        || receipt.regret_limit_micros.is_some_and(|limit| {
            limit != binding.decision_need.regret_tolerance_micros as u64
        })
        || receipt.calibration_receipt_sha256.as_deref()
            .is_some_and(|digest| digest != binding.calibration_receipt_sha256)
    {
        return Err(DecisionSufficiencyRejectV1::BindingMismatch);
    }
    if receipt.status == DecisionSufficiencyStatusV1::Sufficient {
        let need = binding.decision_need;
        if receipt.risk_limit_ppm != Some(need.risk_limit_ppm)
            || receipt.confidence_level_ppm
                .is_none_or(|confidence| confidence < need.minimum_confidence_level_ppm)
            || receipt.calibration_receipt_sha256.as_deref()
                != Some(need.calibration_receipt_sha256.as_str())
            || receipt.regret_limit_micros != Some(need.regret_tolerance_micros as u64)
            || receipt.regret_upper_bound_micros
                .is_none_or(|regret| regret > need.regret_tolerance_micros as u64)
            || receipt.observed_cost_micros
                .is_none_or(|value| value > need.maximum_cost_micros)
            || receipt.observed_latency_ms
                .is_none_or(|value| value > need.maximum_latency_ms)
            || receipt.observed_context_tokens
                .is_none_or(|value| value > need.maximum_context_tokens)
            || receipt.observed_tool_calls
                .is_none_or(|value| value > need.maximum_tool_calls)
            || receipt.maximum_evidence_age_seconds
                .is_none_or(|value| value > need.maximum_evidence_age_seconds)
        {
            return Err(DecisionSufficiencyRejectV1::SufficientRiskBoundInvalid);
        }
    }

    for (actual, expected) in [
        (receipt.run_id.as_str(), binding.run_id),
        (receipt.query_invocation_id.as_str(), binding.query_invocation_id),
        (receipt.domain_id.as_str(), binding.domain_id),
        (receipt.subject_scope_sha256.as_str(), binding.subject_scope_sha256),
        (receipt.environment_revision_sha256.as_str(), binding.environment_revision_sha256),
        (receipt.tool_profile_sha256.as_str(), binding.tool_profile_sha256),
        (receipt.query_sha256.as_str(), binding.query_sha256),
        (receipt.candidate_set_sha256.as_str(), binding.candidate_set_sha256),
        (receipt.qualified_evidence_set_sha256.as_str(), binding.qualified_evidence_set_sha256),
        (receipt.qualification_receipt_sha256.as_str(), binding.qualification_receipt_sha256),
        (receipt.decision_need_sha256.as_str(), binding.decision_need_sha256),
        (receipt.purpose_sha256.as_str(), binding.purpose_sha256),
        (receipt.sufficiency_policy_sha256.as_str(), binding.sufficiency_policy_sha256),
        (receipt.calibration_profile_sha256.as_str(), binding.calibration_profile_sha256),
        (receipt.issuer_profile_sha256.as_str(), binding.issuer_profile_sha256),
    ] {
        if actual != expected {
            return Err(DecisionSufficiencyRejectV1::BindingMismatch);
        }
    }
    Ok(())
}

fn validate_evaluator_role(
    profile: &PinnedEd25519QualificationVerifierV1,
    qualified: &QualifiedCandidateEvidenceBatchV1,
    receipt: &DecisionSufficiencyReceiptV1,
) -> Result<(), DecisionSufficiencyRejectV1> {
    if profile.verifier_id() != DECISION_SUFFICIENCY_VERIFIER_ID_V1 {
        return Err(DecisionSufficiencyRejectV1::WrongEvaluatorRole);
    }
    if profile.profile_sha256() != receipt.issuer_profile_sha256 {
        return Err(DecisionSufficiencyRejectV1::ProfileDigestMismatch);
    }
    if profile.profile_sha256() == qualified.qualification_verifier_profile_sha256()
        || receipt.key_id == qualified.qualification_signer_key_id()
    {
        return Err(DecisionSufficiencyRejectV1::SharedQualificationAuthority);
    }
    Ok(())
}

fn valid_atom(value: &str) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && !value.chars().any(char::is_control)
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn hash_field(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}


#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    use crate::candidate_evidence::{
        CandidateEvidenceV1, EvidenceModalityV1, encode_candidate_evidence_jsonl,
    };
    use crate::evidence_qualification_gate::{
        EvidenceDispositionV1, EvidenceQualificationBindingV1,
        EvidenceQualificationEntryV1, EvidenceQualificationReceiptV1,
        EvidenceQualificationSignatureVerifierV1, VerifiedEvidenceQualificationV1,
    };
    use crate::qualified_candidate_evidence::{
        CandidateEvidenceBatchExpectationV1, CandidateEvidenceProjectionV1,
        project_qualified_candidate_evidence_v1,
    };
    use crate::qualification_dsse::{
        DSSE_PAYLOAD_TYPE_V1, IN_TOTO_STATEMENT_TYPE_V1,
        encode_dsse_envelope_v1, dsse_pae_v1, in_toto_qualification_payload_v1,
    };
    use crate::source_lineage_receipt::{
        MACE_SOURCE_LINEAGE_PREDICATE_TYPE_V1, SOURCE_LINEAGE_VERIFIER_ID_V1,
        SourceLineageEnvelopeInputV1, verify_source_lineage_receipts_v1,
    };
    use crate::qualification_trust::{
        PinnedEd25519QualificationVerifierV1, QualificationKeyStatusV1,
        QualificationTrustKeyV1, QualificationTrustProfileV1, qualification_key_id_v1,
    };

    const SCOPE: &str = "11";
    const ENV: &str = "12";
    const TOOL: &str = "13";
    const QUERY: &str = "14";
    const NEED: &str = "15";
    const PURPOSE: &str = "16";
    const QUAL_POLICY: &str = "17";
    const LIFECYCLE: &str = "18";
    const QUAL_PROFILE: &str = "19";
    const SUFF_POLICY: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const CALIBRATION: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const CALIBRATION_RECEIPT: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

    struct TestQualificationVerifier;
    struct RevokedQualificationVerifier;

    impl EvidenceQualificationSignatureVerifierV1 for RevokedQualificationVerifier {
        fn verify_signature(
            &self,
            _profile: &str,
            _key_id: &str,
            _payload_sha256: &str,
            _signature: &[u8],
            _issued_at_unix: u64,
            _now_unix: u64,
        ) -> Result<(), crate::evidence_qualification_gate::EvidenceQualificationRejectV1> {
            Err(crate::evidence_qualification_gate::EvidenceQualificationRejectV1::RevokedKey)
        }
    }

    impl EvidenceQualificationSignatureVerifierV1 for TestQualificationVerifier {
        fn verify_signature(
            &self,
            profile: &str,
            key_id: &str,
            payload_sha256: &str,
            signature: &[u8],
            _issued_at_unix: u64,
            _now_unix: u64,
        ) -> Result<(), crate::evidence_qualification_gate::EvidenceQualificationRejectV1> {
            if valid_sha256(profile)
                && valid_sha256(key_id)
                && signature == payload_sha256.as_bytes()
            {
                Ok(())
            } else {
                Err(crate::evidence_qualification_gate::EvidenceQualificationRejectV1::SignatureRejected)
            }
        }
    }

    fn repeat(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn candidate() -> CandidateEvidenceV1 {
        CandidateEvidenceV1 {
            run_id: "run-sufficiency".into(),
            candidate_id: "candidate-sufficiency".into(),
            backend_id: "bm25-test".into(),
            backend_revision_sha256: repeat('3'),
            query_scope: "tenant:scope-a".into(),
            query_purpose: "decision:workflow".into(),
            query_sha256: repeat('4'),
            evidence_id: "evidence-1".into(),
            source_id: "source-1".into(),
            source_sha256: repeat('5'),
            payload_sha256: repeat('6'),
            payload_ref: "cas://fixture/evidence-1".into(),
            modality: EvidenceModalityV1::Text,
            retrieval_channel: "bm25".into(),
            retrieval_rank: 1,
            retrieval_score: Some(1.0),
            parent_evidence_ids: Vec::new(),
            derivation_receipt_sha256: None,
        }
    }

    fn candidate_ids_digest(ids: &[String]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"luminous:mace:candidate-evidence-ids:v1\0");
        hasher.update((ids.len() as u64).to_be_bytes());
        for id in ids {
            hash_field(&mut hasher, id);
        }
        hex_digest(&hasher.finalize())
    }

    fn qualified_batch(
        qualification_key_id: &str,
    ) -> QualifiedCandidateEvidenceBatchV1 {
        let candidate = candidate();
        let batch = encode_candidate_evidence_jsonl(&[candidate.clone()])
            .expect("canonical candidate batch");
        let ids = vec![candidate.evidence_id.clone()];
        let mut qualification = EvidenceQualificationReceiptV1 {
            run_id: batch.run_id.clone(),
            query_invocation_id: "invocation-sufficiency".into(),
            domain_id: "web".into(),
            subject_scope_sha256: repeat('1'),
            environment_revision_sha256: repeat('2'),
            tool_profile_sha256: repeat('3'),
            query_sha256: batch.query_sha256.clone(),
            candidate_set_sha256: batch.candidate_set_sha256.clone(),
            candidate_evidence_ids_sha256: candidate_ids_digest(&ids),
            decision_need_sha256: test_decision_need().digest().unwrap(),
            purpose_sha256: test_decision_need().purpose_sha256.clone(),
            qualification_policy_sha256: repeat('7'),
            lifecycle_snapshot_sha256: repeat('8'),
            verifier_profile_sha256: repeat('9'),
            key_id: qualification_key_id.to_owned(),
            issued_at_unix: 100,
            expires_at_unix: 250,
            entries: vec![EvidenceQualificationEntryV1 {
                evidence_id: ids[0].clone(),
                disposition: EvidenceDispositionV1::Admissible,
                basis_receipt_sha256: repeat('a'),
            }],
            signature: Vec::new(),
        };
        qualification.signature = qualification.signing_payload_sha256().into_bytes();

        let binding = EvidenceQualificationBindingV1 {
            run_id: &qualification.run_id,
            query_invocation_id: &qualification.query_invocation_id,
            domain_id: &qualification.domain_id,
            subject_scope_sha256: &qualification.subject_scope_sha256,
            environment_revision_sha256: &qualification.environment_revision_sha256,
            tool_profile_sha256: &qualification.tool_profile_sha256,
            query_sha256: &qualification.query_sha256,
            candidate_set_sha256: &qualification.candidate_set_sha256,
            candidate_evidence_ids: &ids,
            decision_need_sha256: &qualification.decision_need_sha256,
            purpose_sha256: &qualification.purpose_sha256,
            qualification_policy_sha256: &qualification.qualification_policy_sha256,
            lifecycle_snapshot_sha256: &qualification.lifecycle_snapshot_sha256,
            verifier_profile_sha256: &qualification.verifier_profile_sha256,
            now_unix: 150,
        };
        let verified = VerifiedEvidenceQualificationV1::verify_and_bind(
            &qualification,
            &binding,
            &TestQualificationVerifier,
        ).expect("qualification receipt verifies");

        let expected = CandidateEvidenceBatchExpectationV1 {
            candidate_id: &batch.candidate_id,
            backend_id: &batch.backend_id,
            backend_revision_sha256: &batch.backend_revision_sha256,
            query_scope: &batch.query_scope,
            query_purpose: &batch.query_purpose,
        };
        match project_qualified_candidate_evidence_v1(
            &batch,
            &verified,
            &binding,
            &expected,
            &TestQualificationVerifier,
        ).expect("qualified projection") {
            CandidateEvidenceProjectionV1::Qualified(qualified) => qualified,
            CandidateEvidenceProjectionV1::Abstained(_) => panic!("fixture should qualify"),
        }
    }

    fn evaluator_profile(
        signing_key: &SigningKey,
        status: QualificationKeyStatusV1,
        profile_version: u64,
        verifier_id: &str,
    ) -> PinnedEd25519QualificationVerifierV1 {
        let public_key = signing_key.verifying_key().to_bytes();
        let key = QualificationTrustKeyV1 {
            key_id: qualification_key_id_v1(&public_key),
            public_key,
            status,
            not_before_unix: 1,
            not_after_unix: 1000,
        };
        let profile = QualificationTrustProfileV1 {
            schema_version: 1,
            verifier_id: verifier_id.to_owned(),
            profile_version,
            keys: vec![key],
        };
        let digest = profile.profile_sha256().unwrap();
        PinnedEd25519QualificationVerifierV1::from_pinned_profile(
            &profile,
            &digest,
            profile_version,
        ).unwrap()
    }

    fn qualification_binding<'a>(
        qualified: &'a QualifiedCandidateEvidenceBatchV1,
        candidate_ids: &'a [String],
        now_unix: u64,
    ) -> EvidenceQualificationBindingV1<'a> {
        EvidenceQualificationBindingV1 {
            run_id: qualified.run_id(),
            query_invocation_id: qualified.query_invocation_id(),
            domain_id: qualified.domain_id(),
            subject_scope_sha256: qualified.subject_scope_sha256(),
            environment_revision_sha256: qualified.environment_revision_sha256(),
            tool_profile_sha256: qualified.tool_profile_sha256(),
            query_sha256: qualified.batch().query_sha256.as_str(),
            candidate_set_sha256: qualified.source_candidate_set_sha256(),
            candidate_evidence_ids: candidate_ids,
            decision_need_sha256: qualified.decision_need_sha256(),
            purpose_sha256: qualified.purpose_sha256(),
            qualification_policy_sha256: qualified.qualification_policy_sha256(),
            lifecycle_snapshot_sha256: qualified.lifecycle_snapshot_sha256(),
            verifier_profile_sha256: qualified.qualification_verifier_profile_sha256(),
            now_unix,
        }
    }

    fn test_decision_need() -> &'static DecisionNeedV1 {
        static NEED: OnceLock<DecisionNeedV1> = OnceLock::new();
        NEED.get_or_init(|| DecisionNeedV1 {
            decision_need_id: "workflow-completion".into(),
            domain_id: "web".into(),
            subject_scope_sha256: repeat('1'),
            purpose_sha256: repeat('6'),
            action_set_sha256: repeat('c'),
            utility_profile_sha256: repeat('e'),
            risk_limit_ppm: 5000,
            minimum_confidence_level_ppm: 900_000,
            regret_tolerance_micros: 1000,
            sufficiency_policy_sha256: SUFF_POLICY.into(),
            calibration_profile_sha256: CALIBRATION.into(),
            calibration_receipt_sha256: CALIBRATION_RECEIPT.into(),
            maximum_evidence_age_seconds: 86_400,
            maximum_context_tokens: 16_384,
            maximum_tool_calls: 8,
            maximum_cost_micros: 1_000_000,
            maximum_latency_ms: 5_000,
            requirements: vec![DecisionNeedRequirementV1 {
                requirement_id: "minimum-independent-evidence".into(),
                evidence_class_sha256: repeat('c'),
                minimum_independent_root_families: 2,
                maximum_evidence_age_seconds: 3_600,
            }],
        })
    }

    fn use_binding<'a>(
        qualified: &'a QualifiedCandidateEvidenceBatchV1,
        issuer_profile_sha256: &'a str,
        now_unix: u64,
    ) -> DecisionSufficiencyUseBindingV1<'a> {
        static REQUIRED_IDS: OnceLock<Vec<String>> = OnceLock::new();
        let required_ids = REQUIRED_IDS.get_or_init(|| {
            vec!["minimum-independent-evidence".to_owned()]
        });
        DecisionSufficiencyUseBindingV1 {
            decision_need: test_decision_need(),
            run_id: qualified.run_id(),
            query_invocation_id: qualified.query_invocation_id(),
            domain_id: qualified.domain_id(),
            subject_scope_sha256: qualified.subject_scope_sha256(),
            environment_revision_sha256: qualified.environment_revision_sha256(),
            tool_profile_sha256: qualified.tool_profile_sha256(),
            query_sha256: qualified.batch().query_sha256.as_str(),
            candidate_set_sha256: qualified.source_candidate_set_sha256(),
            qualified_evidence_set_sha256: qualified.qualified_candidate_set_sha256(),
            qualification_receipt_sha256: qualified.qualification_receipt_sha256(),
            decision_need_sha256: qualified.decision_need_sha256(),
            purpose_sha256: qualified.purpose_sha256(),
            lifecycle_snapshot_sha256: qualified.lifecycle_snapshot_sha256(),
            sufficiency_policy_sha256: SUFF_POLICY,
            calibration_profile_sha256: CALIBRATION,
            calibration_receipt_sha256: CALIBRATION_RECEIPT,
            required_requirement_ids: required_ids.as_slice(),
            risk_limit_ppm: 5000,
            minimum_confidence_level_ppm: 900_000,
            issuer_profile_sha256,
            now_unix,
        }
    }

    fn receipt(
        qualified: &QualifiedCandidateEvidenceBatchV1,
        binding: &DecisionSufficiencyUseBindingV1<'_>,
        evaluator_key_id: &str,
        status: DecisionSufficiencyStatusV1,
        requirement_status: DecisionRequirementStatusV1,
        risk_upper: Option<u32>,
        risk_limit: Option<u32>,
    ) -> DecisionSufficiencyReceiptV1 {
        DecisionSufficiencyReceiptV1 {
            run_id: qualified.run_id().into(),
            query_invocation_id: qualified.query_invocation_id().into(),
            domain_id: qualified.domain_id().into(),
            subject_scope_sha256: qualified.subject_scope_sha256().into(),
            environment_revision_sha256: qualified.environment_revision_sha256().into(),
            tool_profile_sha256: qualified.tool_profile_sha256().into(),
            query_sha256: qualified.batch().query_sha256.clone(),
            candidate_set_sha256: qualified.source_candidate_set_sha256().into(),
            qualified_evidence_set_sha256: qualified.qualified_candidate_set_sha256().into(),
            qualification_receipt_sha256: qualified.qualification_receipt_sha256().into(),
            decision_need_sha256: qualified.decision_need_sha256().into(),
            purpose_sha256: qualified.purpose_sha256().into(),
            sufficiency_policy_sha256: binding.sufficiency_policy_sha256.into(),
            calibration_profile_sha256: binding.calibration_profile_sha256.into(),
            status,
            requirements: vec![DecisionSufficiencyRequirementV1 {
                requirement_id: "minimum-independent-evidence".into(),
                evidence_class_sha256: repeat('c'),
                status: requirement_status,
                basis_receipt_sha256: if requirement_status == DecisionRequirementStatusV1::Met {
                    vec![repeat('c')]
                } else {
                    Vec::new()
                },
                independent_root_families: if requirement_status == DecisionRequirementStatusV1::Met {
                    vec![
                        DecisionSufficiencyRootFamilyV1 {
                            root_family_sha256: repeat('1'),
                            provenance_receipt_sha256: repeat('2'),
                        },
                        DecisionSufficiencyRootFamilyV1 {
                            root_family_sha256: repeat('3'),
                            provenance_receipt_sha256: repeat('4'),
                        },
                    ]
                } else {
                    Vec::new()
                },
            }],
            risk_upper_bound_ppm: risk_upper,
            risk_limit_ppm: risk_limit,
            confidence_level_ppm: (status == DecisionSufficiencyStatusV1::Sufficient).then_some(950_000),
            calibration_receipt_sha256: (status == DecisionSufficiencyStatusV1::Sufficient)
                .then(|| CALIBRATION_RECEIPT.to_owned()),
            regret_upper_bound_micros: (status == DecisionSufficiencyStatusV1::Sufficient)
                .then_some(500),
            regret_limit_micros: (status == DecisionSufficiencyStatusV1::Sufficient)
                .then_some(1000),
            observed_cost_micros: (status == DecisionSufficiencyStatusV1::Sufficient)
                .then_some(50_000),
            observed_latency_ms: (status == DecisionSufficiencyStatusV1::Sufficient)
                .then_some(250),
            observed_context_tokens: (status == DecisionSufficiencyStatusV1::Sufficient)
                .then_some(2048),
            observed_tool_calls: (status == DecisionSufficiencyStatusV1::Sufficient)
                .then_some(2),
            maximum_evidence_age_seconds: (status == DecisionSufficiencyStatusV1::Sufficient)
                .then_some(900),
            issued_at_unix: 150,
            expires_at_unix: 200,
            issuer_profile_sha256: binding.issuer_profile_sha256.into(),
            key_id: evaluator_key_id.to_owned(),
            signature: Vec::new(),
        }
    }

    fn signed_envelope(
        receipt: &DecisionSufficiencyReceiptV1,
        signing_key: &SigningKey,
    ) -> String {
        let subject_name = format!(
            "urn:luminous:mace:qualified-evidence-set:v1:{}",
            receipt.qualified_evidence_set_sha256
        );
        let statement = json!({
            "_type": IN_TOTO_STATEMENT_TYPE_V1,
            "subject": [{
                "name": subject_name,
                "digest": {
                    "sha256": receipt.qualified_evidence_set_sha256
                }
            }],
            "predicateType": MACE_DECISION_SUFFICIENCY_PREDICATE_TYPE_V1,
            "predicate": receipt.unsigned_predicate_value()
        });
        let payload = serde_json::to_vec(&statement).unwrap();
        let pae = dsse_pae_v1(DSSE_PAYLOAD_TYPE_V1, &payload);
        let signature = signing_key.sign(&pae).to_bytes();
        encode_dsse_envelope_v1(&payload, &receipt.key_id, &signature).unwrap()
    }

    fn signed_source_lineage_envelope(
        signing_key: &SigningKey,
        profile_sha256: &str,
        qualified_evidence_set_sha256: &str,
        decision_need_sha256: &str,
        run_id: &str,
        query_invocation_id: &str,
        requirement_id: &str,
        evidence_class_sha256: &str,
        root_family_sha256: &str,
    ) -> String {
        let key_id = qualification_key_id_v1(&signing_key.verifying_key().to_bytes());
        let predicate = json!({
            "schema_version": "mace_source_lineage_receipt_v1",
            "run_id": run_id,
            "query_invocation_id": query_invocation_id,
            "qualified_evidence_set_sha256": qualified_evidence_set_sha256,
            "decision_need_sha256": decision_need_sha256,
            "requirement_id": requirement_id,
            "evidence_class_sha256": evidence_class_sha256,
            "root_family_sha256": root_family_sha256,
            "source_graph_sha256": repeat('a'),
            "lineage_closure_sha256": repeat('b'),
            "source_node_count": 2,
            "dependency_edge_count": 1,
            "unresolved_ancestor_count": 0,
            "status": "independent",
            "issued_at_unix": 100,
            "expires_at_unix": 500,
            "issuer_profile_sha256": profile_sha256,
            "key_id": key_id,
        });
        let subject_name = format!(
            "urn:luminous:mace:source-root-family:v1:{}",
            root_family_sha256
        );
        let statement = json!({
            "_type": IN_TOTO_STATEMENT_TYPE_V1,
            "subject": [{
                "name": subject_name,
                "digest": {"sha256": root_family_sha256}
            }],
            "predicateType": MACE_SOURCE_LINEAGE_PREDICATE_TYPE_V1,
            "predicate": predicate,
        });
        let payload = serde_json::to_vec(&statement).unwrap();
        let pae = dsse_pae_v1(DSSE_PAYLOAD_TYPE_V1, &payload);
        let signature = signing_key.sign(&pae).to_bytes();
        encode_dsse_envelope_v1(&payload, &key_id, &signature).unwrap()
    }

    fn raw_envelope_sha256(envelope: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(envelope.as_bytes());
        hex_digest(&hasher.finalize())
    }

    fn build_valid_assessment<'a>(
        qualified: &'a QualifiedCandidateEvidenceBatchV1,
        status: DecisionSufficiencyStatusV1,
        requirement_status: DecisionRequirementStatusV1,
        risk_upper: Option<u32>,
        risk_limit: Option<u32>,
    ) -> (
        DecisionSufficiencyUseBindingV1<'a>,
        PinnedEd25519QualificationVerifierV1,
        String,
    ) {
        // Only the immutable evaluator-profile digest is leaked in this test
        // helper; all evidence and request fields borrow from the projection.
        let signing_key = SigningKey::from_bytes(&[73; 32]);
        let profile = evaluator_profile(
            &signing_key,
            QualificationKeyStatusV1::Active,
            1,
            DECISION_SUFFICIENCY_VERIFIER_ID_V1,
        );
        let profile_digest = profile.profile_sha256().to_owned();
        let issuer_profile: &'a str = Box::leak(profile_digest.into_boxed_str());
        let binding = use_binding(qualified, issuer_profile, 160);
        let evaluator_key_id = qualification_key_id_v1(&signing_key.verifying_key().to_bytes());
        let assessment = receipt(
            qualified, &binding, &evaluator_key_id, status, requirement_status,
            risk_upper, risk_limit,
        );
        let envelope = signed_envelope(&assessment, &signing_key);
        (binding, profile, envelope)
    }

    #[test]
    fn three_authority_lineage_chain_verifies_and_revalidates_at_use() {
        let qualified = qualified_batch(&repeat('f'));
        let candidate_ids = vec!["evidence-1".to_owned()];
        let qualification_binding = qualification_binding(&qualified, &candidate_ids, 160);

        let evaluator_key = SigningKey::from_bytes(&[73; 32]);
        let evaluator_profile_value = evaluator_profile(
            &evaluator_key,
            QualificationKeyStatusV1::Active,
            1,
            DECISION_SUFFICIENCY_VERIFIER_ID_V1,
        );
        let evaluator_profile_sha256 = evaluator_profile_value.profile_sha256().to_owned();
        let evaluator_profile_pin = evaluator_profile_sha256.clone();
        let evaluator_profile_pin: &'static str =
            Box::leak(evaluator_profile_pin.into_boxed_str());
        let binding = use_binding(&qualified, evaluator_profile_pin, 160);
        let evaluator_key_id =
            qualification_key_id_v1(&evaluator_key.verifying_key().to_bytes());
        let mut assessment = receipt(
            &qualified,
            &binding,
            &evaluator_key_id,
            DecisionSufficiencyStatusV1::Sufficient,
            DecisionRequirementStatusV1::Met,
            Some(1000),
            Some(5000),
        );

        let resolver_key = SigningKey::from_bytes(&[99; 32]);
        let resolver_profile = evaluator_profile(
            &resolver_key,
            QualificationKeyStatusV1::Active,
            1,
            SOURCE_LINEAGE_VERIFIER_ID_V1,
        );
        let resolver_profile_sha256 = resolver_profile.profile_sha256().to_owned();
        let root_a = assessment.requirements[0].independent_root_families[0]
            .root_family_sha256.clone();
        let root_b = assessment.requirements[0].independent_root_families[1]
            .root_family_sha256.clone();
        let need = test_decision_need();
        let need_sha256 = need.digest().unwrap();
        let envelope_a = signed_source_lineage_envelope(
            &resolver_key,
            &resolver_profile_sha256,
            qualified.qualified_candidate_set_sha256(),
            &need_sha256,
            qualified.run_id(),
            qualified.query_invocation_id(),
            "minimum-independent-evidence",
            &need.requirements[0].evidence_class_sha256,
            &root_a,
        );
        let envelope_b = signed_source_lineage_envelope(
            &resolver_key,
            &resolver_profile_sha256,
            qualified.qualified_candidate_set_sha256(),
            &need_sha256,
            qualified.run_id(),
            qualified.query_invocation_id(),
            "minimum-independent-evidence",
            &need.requirements[0].evidence_class_sha256,
            &root_b,
        );
        assessment.requirements[0].independent_root_families[0]
            .provenance_receipt_sha256 = raw_envelope_sha256(&envelope_a);
        assessment.requirements[0].independent_root_families[1]
            .provenance_receipt_sha256 = raw_envelope_sha256(&envelope_b);
        let evaluator_envelope = signed_envelope(&assessment, &evaluator_key);
        let verified_sufficiency = verify_decision_sufficiency_v1(
            &qualified,
            &qualification_binding,
            &TestQualificationVerifier,
            &binding,
            &evaluator_envelope,
            &evaluator_profile_value,
        ).expect("independent evaluator receipt");

        let lineage_inputs = vec![
            SourceLineageEnvelopeInputV1 {
                requirement_id: "minimum-independent-evidence",
                root_family_sha256: &root_a,
                envelope_json: &envelope_a,
            },
            SourceLineageEnvelopeInputV1 {
                requirement_id: "minimum-independent-evidence",
                root_family_sha256: &root_b,
                envelope_json: &envelope_b,
            },
        ];
        let verified_lineage = verify_source_lineage_receipts_v1(
            &qualified,
            &qualification_binding,
            &TestQualificationVerifier,
            &verified_sufficiency,
            &binding,
            &evaluator_profile_value,
            need,
            &lineage_inputs,
            &resolver_profile_sha256,
            &resolver_profile,
        ).expect("third-authority lineage receipts");
        assert_eq!(verified_lineage.len(), 2);
        verified_lineage.validate_at_use(
            &qualified,
            &qualification_binding,
            &TestQualificationVerifier,
            &verified_sufficiency,
            &binding,
            &evaluator_profile_value,
            need,
            &resolver_profile,
        ).expect("all three authorities remain valid at one use-time sample");
    }

    #[test]
    fn consumer_decision_need_manifest_roundtrips_and_binds_every_policy_field() {
        let need = test_decision_need();
        let value = need.to_json_value().expect("valid DecisionNeed");
        let wire = serde_json::to_string(&value).unwrap();
        let decoded = DecisionNeedV1::from_json_str(&wire).expect("strict manifest decode");
        assert_eq!(&decoded, need);
        assert_eq!(decoded.digest().unwrap(), need.digest().unwrap());

        let mut changed = value;
        changed["risk_limit_ppm"] = Value::from(6000);
        let altered = DecisionNeedV1::from_json_value(&changed).expect("altered but valid manifest");
        assert_ne!(altered.digest().unwrap(), need.digest().unwrap());

        let duplicate = wire.replacen(
            "\"risk_limit_ppm\":",
            "\"risk_limit_ppm\":5000,\"risk_limit_ppm\":",
            1,
        );
        assert_eq!(
            DecisionNeedV1::from_json_str(&duplicate),
            Err(DecisionSufficiencyRejectV1::InvalidJson)
        );
    }

    #[test]
    fn independently_signed_sufficiency_is_bound_to_the_exact_qualified_set() {
        let qualified = qualified_batch(&repeat('f'));
        let (binding, profile, envelope) = build_valid_assessment(
            &qualified,
            DecisionSufficiencyStatusV1::Sufficient,
            DecisionRequirementStatusV1::Met,
            Some(1000),
            Some(5000),
        );
        let candidate_ids = vec!["evidence-1".to_owned()];
        let qualification_binding = qualification_binding(&qualified, &candidate_ids, 160);
        let verified = verify_decision_sufficiency_v1(
            &qualified,
            &qualification_binding,
            &TestQualificationVerifier,
            &binding,
            &envelope,
            &profile,
        ).expect("separately signed sufficient receipt");
        assert!(verified.is_sufficient());
        assert_eq!(verified.status(), DecisionSufficiencyStatusV1::Sufficient);
        verified.validate_at_use(
            &qualified,
            &qualification_binding,
            &TestQualificationVerifier,
            &binding,
            &profile,
        ).expect("same DecisionNeed and current evaluator profile remain valid");
    }

    #[test]
    fn insufficient_is_a_valid_assessment_but_never_a_sufficient_verdict() {
        let qualified = qualified_batch(&repeat('f'));
        let (binding, profile, envelope) = build_valid_assessment(
            &qualified,
            DecisionSufficiencyStatusV1::Insufficient,
            DecisionRequirementStatusV1::Unmet,
            Some(10_000),
            Some(5000),
        );
        let candidate_ids = vec!["evidence-1".to_owned()];
        let qualification_binding = qualification_binding(&qualified, &candidate_ids, 160);
        let verified = verify_decision_sufficiency_v1(
            &qualified,
            &qualification_binding,
            &TestQualificationVerifier,
            &binding,
            &envelope,
            &profile,
        ).expect("honest insufficient assessment");
        assert!(!verified.is_sufficient());
        assert_eq!(verified.status(), DecisionSufficiencyStatusV1::Insufficient);
    }

    #[test]
    fn sufficient_receipt_with_unmet_requirement_is_rejected() {
        let qualified = qualified_batch(&repeat('f'));
        let (binding, profile, envelope) = build_valid_assessment(
            &qualified,
            DecisionSufficiencyStatusV1::Sufficient,
            DecisionRequirementStatusV1::Unmet,
            Some(1000),
            Some(5000),
        );
        assert_eq!(
            verify_decision_sufficiency_v1(
                &qualified,
                &qualification_binding(&qualified, &["evidence-1".to_owned()], 160),
                &TestQualificationVerifier,
                &binding,
                &envelope,
                &profile,
            ),
            Err(DecisionSufficiencyRejectV1::SufficientRequirementsNotMet)
        );
    }

    #[test]
    fn sufficient_receipt_with_too_few_independent_root_families_is_rejected() {
        let qualified = qualified_batch(&repeat('f'));
        let (binding, profile, _) = build_valid_assessment(
            &qualified,
            DecisionSufficiencyStatusV1::Sufficient,
            DecisionRequirementStatusV1::Met,
            Some(1000),
            Some(5000),
        );
        let signing_key = SigningKey::from_bytes(&[73; 32]);
        let evaluator_key_id = qualification_key_id_v1(&signing_key.verifying_key().to_bytes());
        let mut assessment = receipt(
            &qualified,
            &binding,
            &evaluator_key_id,
            DecisionSufficiencyStatusV1::Sufficient,
            DecisionRequirementStatusV1::Met,
            Some(1000),
            Some(5000),
        );
        assessment.requirements[0].independent_root_families.pop();
        let envelope = signed_envelope(&assessment, &signing_key);
        assert_eq!(
            verify_decision_sufficiency_v1(
                &qualified,
                &qualification_binding(&qualified, &["evidence-1".to_owned()], 160),
                &TestQualificationVerifier,
                &binding,
                &envelope,
                &profile,
            ),
            Err(DecisionSufficiencyRejectV1::InsufficientIndependentRootFamilies)
        );
    }

    #[test]
    fn one_lineage_receipt_cannot_attest_multiple_root_families() {
        let qualified = qualified_batch(&repeat('f'));
        let (binding, profile, _) = build_valid_assessment(
            &qualified,
            DecisionSufficiencyStatusV1::Sufficient,
            DecisionRequirementStatusV1::Met,
            Some(1000),
            Some(5000),
        );
        let signing_key = SigningKey::from_bytes(&[73; 32]);
        let evaluator_key_id = qualification_key_id_v1(&signing_key.verifying_key().to_bytes());
        let mut assessment = receipt(
            &qualified,
            &binding,
            &evaluator_key_id,
            DecisionSufficiencyStatusV1::Sufficient,
            DecisionRequirementStatusV1::Met,
            Some(1000),
            Some(5000),
        );
        let first_receipt = assessment.requirements[0].independent_root_families[0]
            .provenance_receipt_sha256.clone();
        assessment.requirements[0].independent_root_families[1]
            .provenance_receipt_sha256 = first_receipt;
        let envelope = signed_envelope(&assessment, &signing_key);
        assert_eq!(
            verify_decision_sufficiency_v1(
                &qualified,
                &qualification_binding(&qualified, &["evidence-1".to_owned()], 160),
                &TestQualificationVerifier,
                &binding,
                &envelope,
                &profile,
            ),
            Err(DecisionSufficiencyRejectV1::InvalidReceipt)
        );
    }

    #[test]
    fn sufficient_receipt_with_wrong_evidence_class_is_rejected() {
        let qualified = qualified_batch(&repeat('f'));
        let (binding, profile, _) = build_valid_assessment(
            &qualified,
            DecisionSufficiencyStatusV1::Sufficient,
            DecisionRequirementStatusV1::Met,
            Some(1000),
            Some(5000),
        );
        let signing_key = SigningKey::from_bytes(&[73; 32]);
        let evaluator_key_id = qualification_key_id_v1(&signing_key.verifying_key().to_bytes());
        let mut assessment = receipt(
            &qualified,
            &binding,
            &evaluator_key_id,
            DecisionSufficiencyStatusV1::Sufficient,
            DecisionRequirementStatusV1::Met,
            Some(1000),
            Some(5000),
        );
        assessment.requirements[0].evidence_class_sha256 = repeat('d');
        let envelope = signed_envelope(&assessment, &signing_key);
        assert_eq!(
            verify_decision_sufficiency_v1(
                &qualified,
                &qualification_binding(&qualified, &["evidence-1".to_owned()], 160),
                &TestQualificationVerifier,
                &binding,
                &envelope,
                &profile,
            ),
            Err(DecisionSufficiencyRejectV1::EvidenceClassMismatch)
        );
    }

    #[test]
    fn sufficient_receipt_with_risk_bound_over_limit_is_rejected() {
        let qualified = qualified_batch(&repeat('f'));
        let (binding, profile, envelope) = build_valid_assessment(
            &qualified,
            DecisionSufficiencyStatusV1::Sufficient,
            DecisionRequirementStatusV1::Met,
            Some(6000),
            Some(5000),
        );
        assert_eq!(
            verify_decision_sufficiency_v1(
                &qualified,
                &qualification_binding(&qualified, &["evidence-1".to_owned()], 160),
                &TestQualificationVerifier,
                &binding,
                &envelope,
                &profile,
            ),
            Err(DecisionSufficiencyRejectV1::SufficientRiskBoundInvalid)
        );
    }

    #[test]
    fn sufficient_receipt_with_regret_upper_bound_over_tolerance_is_rejected() {
        let qualified = qualified_batch(&repeat('f'));
        let (binding, profile, _) = build_valid_assessment(
            &qualified,
            DecisionSufficiencyStatusV1::Sufficient,
            DecisionRequirementStatusV1::Met,
            Some(1000),
            Some(5000),
        );
        let signing_key = SigningKey::from_bytes(&[73; 32]);
        let evaluator_key_id = qualification_key_id_v1(&signing_key.verifying_key().to_bytes());
        let mut assessment = receipt(
            &qualified,
            &binding,
            &evaluator_key_id,
            DecisionSufficiencyStatusV1::Sufficient,
            DecisionRequirementStatusV1::Met,
            Some(1000),
            Some(5000),
        );
        assessment.regret_upper_bound_micros = Some(2000);
        let envelope = signed_envelope(&assessment, &signing_key);
        assert_eq!(
            verify_decision_sufficiency_v1(
                &qualified,
                &qualification_binding(&qualified, &["evidence-1".to_owned()], 160),
                &TestQualificationVerifier,
                &binding,
                &envelope,
                &profile,
            ),
            Err(DecisionSufficiencyRejectV1::SufficientRiskBoundInvalid)
        );
    }

    #[test]
    fn sufficient_receipt_cannot_exceed_declared_resource_budget() {
        let qualified = qualified_batch(&repeat('f'));
        let (binding, profile, _) = build_valid_assessment(
            &qualified,
            DecisionSufficiencyStatusV1::Sufficient,
            DecisionRequirementStatusV1::Met,
            Some(1000),
            Some(5000),
        );
        let signing_key = SigningKey::from_bytes(&[73; 32]);
        let evaluator_key_id = qualification_key_id_v1(&signing_key.verifying_key().to_bytes());
        let mut assessment = receipt(
            &qualified,
            &binding,
            &evaluator_key_id,
            DecisionSufficiencyStatusV1::Sufficient,
            DecisionRequirementStatusV1::Met,
            Some(1000),
            Some(5000),
        );
        assessment.observed_cost_micros = Some(binding.decision_need.maximum_cost_micros + 1);
        let envelope = signed_envelope(&assessment, &signing_key);
        assert_eq!(
            verify_decision_sufficiency_v1(
                &qualified,
                &qualification_binding(&qualified, &["evidence-1".to_owned()], 160),
                &TestQualificationVerifier,
                &binding,
                &envelope,
                &profile,
            ),
            Err(DecisionSufficiencyRejectV1::SufficientRiskBoundInvalid)
        );
    }

    #[test]
    fn decision_sufficiency_cannot_reuse_qualification_signer() {
        let signing_key = SigningKey::from_bytes(&[73; 32]);
        let key_id = qualification_key_id_v1(&signing_key.verifying_key().to_bytes());
        let qualified = qualified_batch(&key_id);
        let (binding, profile, envelope) = build_valid_assessment(
            &qualified,
            DecisionSufficiencyStatusV1::Sufficient,
            DecisionRequirementStatusV1::Met,
            Some(1000),
            Some(5000),
        );
        assert_eq!(
            verify_decision_sufficiency_v1(
                &qualified,
                &qualification_binding(&qualified, &["evidence-1".to_owned()], 160),
                &TestQualificationVerifier,
                &binding,
                &envelope,
                &profile,
            ),
            Err(DecisionSufficiencyRejectV1::SharedQualificationAuthority)
        );
    }

    #[test]
    fn sufficiency_cannot_outlive_original_qualification_expiry() {
        let qualified = qualified_batch(&repeat('f'));
        let (binding, profile, envelope) = build_valid_assessment(
            &qualified,
            DecisionSufficiencyStatusV1::Sufficient,
            DecisionRequirementStatusV1::Met,
            Some(1000),
            Some(5000),
        );
        let candidate_ids = vec!["evidence-1".to_owned()];
        let qualification_binding = qualification_binding(&qualified, &candidate_ids, 250);
        let mut later_binding = binding.clone();
        later_binding.now_unix = 250;
        assert_eq!(
            verify_decision_sufficiency_v1(
                &qualified,
                &qualification_binding,
                &TestQualificationVerifier,
                &later_binding,
                &envelope,
                &profile,
            ),
            Err(DecisionSufficiencyRejectV1::Qualification(
                EvidenceQualificationRejectV1::ReceiptExpired
            ))
        );
    }

    #[test]
    fn sufficiency_rechecks_original_qualification_signer_at_use() {
        let qualified = qualified_batch(&repeat('f'));
        let (binding, profile, envelope) = build_valid_assessment(
            &qualified,
            DecisionSufficiencyStatusV1::Sufficient,
            DecisionRequirementStatusV1::Met,
            Some(1000),
            Some(5000),
        );
        let candidate_ids = vec!["evidence-1".to_owned()];
        let qualification_binding = qualification_binding(&qualified, &candidate_ids, 160);
        assert_eq!(
            verify_decision_sufficiency_v1(
                &qualified,
                &qualification_binding,
                &RevokedQualificationVerifier,
                &binding,
                &envelope,
                &profile,
            ),
            Err(DecisionSufficiencyRejectV1::Qualification(
                EvidenceQualificationRejectV1::RevokedKey
            ))
        );
    }

    #[test]
    fn rotated_or_revoked_evaluator_profile_invalidates_previous_sufficiency() {
        let qualified = qualified_batch(&repeat('f'));
        let (binding, active_profile, envelope) = build_valid_assessment(
            &qualified,
            DecisionSufficiencyStatusV1::Sufficient,
            DecisionRequirementStatusV1::Met,
            Some(1000),
            Some(5000),
        );
        let candidate_ids = vec!["evidence-1".to_owned()];
        let qualification_binding = qualification_binding(&qualified, &candidate_ids, 160);
        let verified = verify_decision_sufficiency_v1(
            &qualified,
            &qualification_binding,
            &TestQualificationVerifier,
            &binding,
            &envelope,
            &active_profile,
        ).unwrap();

        let signing_key = SigningKey::from_bytes(&[73; 32]);
        let revoked_profile = evaluator_profile(
            &signing_key,
            QualificationKeyStatusV1::Revoked,
            2,
            DECISION_SUFFICIENCY_VERIFIER_ID_V1,
        );
        assert_eq!(
            verified.validate_at_use(
                &qualified,
                &qualification_binding,
                &TestQualificationVerifier,
                &binding,
                &revoked_profile,
            ),
            Err(DecisionSufficiencyRejectV1::ProfileDigestMismatch)
        );
    }
}
