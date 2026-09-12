// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fmt;
use thiserror::Error;

pub(crate) const REQUIREMENT_DOMAIN_V1: &[u8] = b"symthaea.etk-accepted-requirement.v1\0";
pub(crate) const SUBJECT_DOMAIN_V1: &[u8] = b"symthaea.etk-engineering-subject.v1\0";
pub(crate) const TWIN_DOMAIN_V1: &[u8] = b"symthaea.etk-twin-revision.v1\0";
pub(crate) const VALIDITY_DOMAIN_V1: &[u8] = b"symthaea.etk-validity-domain.v1\0";
pub(crate) const CURRENTNESS_DOMAIN_V1: &[u8] = b"symthaea.etk-currentness-assertion.v1\0";
pub(crate) const OBLIGATION_DOMAIN_V1: &[u8] = b"symthaea.etk-proof-obligation-snapshot.v1\0";
pub(crate) const METHOD_DOMAIN_V1: &[u8] = b"symthaea.etk-native-analytical-method.v1\0";
pub(crate) const INPUT_DOMAIN_V1: &[u8] = b"symthaea.etk-native-analytical-input.v1\0";
pub(crate) const POLICY_DOMAIN_V1: &[u8] = b"symthaea.etk-native-analytical-policy.v1\0";
pub(crate) const PLAN_DOMAIN_V1: &[u8] = b"symthaea.etk-native-analytical-plan.v1\0";
pub(crate) const ADMITTED_DOMAIN_V1: &[u8] = b"symthaea.etk-native-analytical-admitted-evidence.v1\0";
pub(crate) const RECEIPT_DOMAIN_V1: &[u8] = b"symthaea.etk-native-analytical-discharge-receipt.v1\0";
pub(crate) const FACT_DOMAIN_V1: &[u8] = b"symthaea.etk-current-native-analytical-discharge-fact.v1\0";
pub(crate) const EQUATION_RELATIVE_TOLERANCE: f64 = 1e-12;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AnalysisTrustErrorV1 {
    #[error("{0} cannot be empty or have leading/trailing whitespace")]
    InvalidText(&'static str),
    #[error("{0} is not a canonical SHA-256 identity")]
    InvalidDigest(&'static str),
    #[error("{0} must be finite")]
    NonFinite(&'static str),
    #[error("{0} must be greater than zero")]
    NonPositive(&'static str),
    #[error("{0} must be within [0, 1]")]
    UnitInterval(&'static str),
    #[error("duplicate validity-domain dimension: {0}")]
    DuplicateValidityDimension(String),
    #[error("twin revision does not belong to the supplied subject revision")]
    TwinSubjectMismatch,
    #[error("twin parent belongs to a different subject or twin kind")]
    TwinLineageMismatch,
    #[error("validity domain does not bind the supplied subject/twin revisions")]
    ValidityContextMismatch,
    #[error("currentness assertion does not bind the supplied twin/validity revisions")]
    CurrentnessContextMismatch,
    #[error("proof obligation does not expect Analysis evidence")]
    NotAnalysisObligation,
    #[error("analytical input was created for a different method revision")]
    MethodInputMismatch,
    #[error("analytical result does not target the exact bound method/input/policy")]
    PlanBindingMismatch,
    #[error("analytical policy is not conservative enough to discharge the accepted requirement")]
    PolicyDoesNotDischargeRequirement,
    #[error("reported {0} is inconsistent with the bound analytical input")]
    AnalyticalEquationMismatch(&'static str),
    #[error("reported factor of safety is inconsistent with yield strength / bending stress")]
    InconsistentFactorOfSafety,
    #[error("model-relative-error bound exceeds the admission policy")]
    ModelErrorBudgetExceeded,
    #[error("conservative analytical acceptance predicate failed")]
    AcceptancePredicateFailed,
    #[error("admitted evidence targets a different analytical plan")]
    AdmittedPlanMismatch,
    #[error("discharge receipt targets a historical analytical plan")]
    HistoricalPlan,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sha256DigestV1(String);

impl Sha256DigestV1 {
    pub fn parse(value: impl Into<String>) -> Result<Self, AnalysisTrustErrorV1> {
        let value = value.into();
        let Some(hex_part) = value.strip_prefix("sha256:") else {
            return Err(AnalysisTrustErrorV1::InvalidDigest("SHA-256 digest"));
        };
        if hex_part.len() != 64
            || !hex_part
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(AnalysisTrustErrorV1::InvalidDigest("SHA-256 digest"));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Sha256DigestV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

macro_rules! premise_id {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(Sha256DigestV1);

        impl $name {
            pub fn parse(value: impl Into<String>) -> Result<Self, AnalysisTrustErrorV1> {
                Sha256DigestV1::parse(value).map(Self)
            }

            pub fn from_digest(digest: Sha256DigestV1) -> Self {
                Self(digest)
            }

            pub fn as_str(&self) -> &str {
                self.0.as_str()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }
    };
}

macro_rules! semantic_id {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(Sha256DigestV1);

        impl $name {
            pub(crate) fn from_digest(digest: Sha256DigestV1) -> Self {
                Self(digest)
            }

            pub fn as_str(&self) -> &str {
                self.0.as_str()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }
    };
}

premise_id!(ModelQualificationRecordDigestV1);
premise_id!(ExecutionArtifactDigestV1);

semantic_id!(SubjectRevisionIdV1);
semantic_id!(TwinRevisionIdV1);
semantic_id!(ValidityDomainRevisionIdV1);
semantic_id!(CurrentnessAssertionIdV1);
semantic_id!(AnalysisRequirementRevisionIdV1);
semantic_id!(ObligationRevisionIdV1);
semantic_id!(AnalyticalMethodRevisionIdV1);
semantic_id!(AnalyticalInputRevisionIdV1);
semantic_id!(AnalyticalPolicyRevisionIdV1);
semantic_id!(AnalyticalPlanIdV1);
semantic_id!(AdmittedAnalyticalEvidenceIdV1);
semantic_id!(NativeAnalyticalDischargeReceiptIdV1);
semantic_id!(CurrentNativeAnalyticalDischargeFactIdV1);

/// Cross-language identity encoding for one finite IEEE-754 binary64 value.
/// Signed zero is intentionally normalized.
pub fn canonical_binary64_v1(value: f64) -> Result<String, AnalysisTrustErrorV1> {
    if !value.is_finite() {
        return Err(AnalysisTrustErrorV1::NonFinite("binary64"));
    }
    let normalized = if value == 0.0 { 0.0 } else { value };
    Ok(format!("f64:{:016x}", normalized.to_bits()))
}

pub(crate) fn canonical_text(
    value: String,
    field: &'static str,
) -> Result<String, AnalysisTrustErrorV1> {
    if value.is_empty() || value.trim() != value {
        Err(AnalysisTrustErrorV1::InvalidText(field))
    } else {
        Ok(value)
    }
}

pub(crate) fn domain_hash(domain: &[u8], value: &Value) -> Sha256DigestV1 {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(canonical_json(value).as_bytes());
    Sha256DigestV1::parse(format!("sha256:{}", hex::encode(hasher.finalize())))
        .expect("SHA-256 output is canonical lowercase hex")
}

fn canonical_json(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => {
            serde_json::to_string(value).expect("serializing an in-memory JSON string cannot fail")
        }
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",")
        ),
        Value::Object(map) => {
            let mut keys = map.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            let body = keys
                .into_iter()
                .map(|key| {
                    let encoded = serde_json::to_string(key)
                        .expect("serializing an in-memory JSON key cannot fail");
                    format!("{encoded}:{}", canonical_json(&map[key]))
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{body}}}")
        }
    }
}
