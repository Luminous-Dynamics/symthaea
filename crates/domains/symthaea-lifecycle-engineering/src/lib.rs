// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Provenance-bound lifecycle design contracts.
//!
//! This crate binds design intent and exact external subject references. It does not own
//! manufacturing execution, lifecycle events, material truth, reliability evidence, impact
//! calculations, circularity certification, or compliance authority.
//!
//! ```text
//! lifecycle strategy referenced
//! != strategy feasible
//! != strategy executed
//! != lifecycle event observed
//! != environmental benefit established
//! ```

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use thiserror::Error;

const REF_ID_DOMAIN: &str = "symthaea-lifecycle-engineering::exact-subject-ref-v1";
const PROFILE_ID_DOMAIN: &str = "symthaea-lifecycle-engineering::lifecycle-design-profile-v1";

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LifecycleContractError {
    #[error("{field} must be a canonical non-empty token")]
    NonCanonical { field: &'static str },
    #[error("content digest must be exactly 64 lowercase hexadecimal characters")]
    InvalidDigest,
    #[error("lifecycle design profile requires at least one manufacturing-plan reference")]
    MissingManufacturingPlan,
    #[error("lifecycle design profile requires at least one lifecycle strategy")]
    MissingStrategy,
    #[error("duplicate reference in {field}: {value}")]
    DuplicateReference { field: &'static str, value: String },
}

fn canonical(field: &'static str, value: &str) -> Result<(), LifecycleContractError> {
    if value.is_empty() || value.trim() != value || value.chars().any(char::is_control) {
        return Err(LifecycleContractError::NonCanonical { field });
    }
    Ok(())
}

fn digest64(value: &str) -> Result<(), LifecycleContractError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(LifecycleContractError::InvalidDigest);
    }
    Ok(())
}

fn hash_field(hasher: &mut blake3::Hasher, value: &str) {
    let bytes = value.as_bytes();
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ExactLifecycleSubjectRefV1 {
    pub namespace: String,
    pub subject_id: String,
    pub semantic_version: String,
    pub content_blake3: String,
}

impl ExactLifecycleSubjectRefV1 {
    pub fn validate(&self) -> Result<(), LifecycleContractError> {
        canonical("subject.namespace", &self.namespace)?;
        canonical("subject.subject_id", &self.subject_id)?;
        canonical("subject.semantic_version", &self.semantic_version)?;
        digest64(&self.content_blake3)
    }

    /// Stable identity of the external-reference tuple. A valid ID does not prove the external
    /// subject exists or that its owning system admits it.
    pub fn ref_id(&self) -> Result<String, LifecycleContractError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hash_field(&mut hasher, REF_ID_DOMAIN);
        hash_field(&mut hasher, &self.namespace);
        hash_field(&mut hasher, &self.subject_id);
        hash_field(&mut hasher, &self.semantic_version);
        hash_field(&mut hasher, &self.content_blake3);
        Ok(hasher.finalize().to_hex().to_string())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LifecycleStrategyKindV1 {
    Disassembly,
    Repair,
    Reuse,
    Refurbishment,
    Remanufacture,
    ComponentHarvest,
    Recycling,
    MaterialRecovery,
    Disposal,
}

impl LifecycleStrategyKindV1 {
    fn tag(self) -> &'static str {
        match self {
            Self::Disassembly => "disassembly",
            Self::Repair => "repair",
            Self::Reuse => "reuse",
            Self::Refurbishment => "refurbishment",
            Self::Remanufacture => "remanufacture",
            Self::ComponentHarvest => "component-harvest",
            Self::Recycling => "recycling",
            Self::MaterialRecovery => "material-recovery",
            Self::Disposal => "disposal",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LifecycleStrategyRefV1 {
    pub kind: LifecycleStrategyKindV1,
    pub subject: ExactLifecycleSubjectRefV1,
}

impl LifecycleStrategyRefV1 {
    pub fn validate(&self) -> Result<(), LifecycleContractError> {
        self.subject.validate()
    }

    fn identity_key(&self) -> Result<String, LifecycleContractError> {
        Ok(format!("{}:{}", self.kind.tag(), self.subject.ref_id()?))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LifecycleAssessmentKindV1 {
    MaterialFlowAccount,
    EnvironmentalInventory,
    LifecycleImpactAssessment,
    CircularityAssessment,
    CriticalMaterialAssessment,
    ServiceLifeAssumption,
    LifecycleExposureEvidence,
}

impl LifecycleAssessmentKindV1 {
    fn tag(self) -> &'static str {
        match self {
            Self::MaterialFlowAccount => "material-flow-account",
            Self::EnvironmentalInventory => "environmental-inventory",
            Self::LifecycleImpactAssessment => "lifecycle-impact-assessment",
            Self::CircularityAssessment => "circularity-assessment",
            Self::CriticalMaterialAssessment => "critical-material-assessment",
            Self::ServiceLifeAssumption => "service-life-assumption",
            Self::LifecycleExposureEvidence => "lifecycle-exposure-evidence",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LifecycleAssessmentRefV1 {
    pub kind: LifecycleAssessmentKindV1,
    pub subject: ExactLifecycleSubjectRefV1,
}

impl LifecycleAssessmentRefV1 {
    pub fn validate(&self) -> Result<(), LifecycleContractError> {
        self.subject.validate()
    }

    fn identity_key(&self) -> Result<String, LifecycleContractError> {
        Ok(format!("{}:{}", self.kind.tag(), self.subject.ref_id()?))
    }
}

/// Hard authority ceiling for this first lifecycle contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LifecycleDesignAuthorityCeilingV1 {
    DesignIntentAndAssessmentReferencesOnly,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LifecycleDesignProfileV1 {
    pub semantic_version: String,
    pub subject_configuration: ExactLifecycleSubjectRefV1,
    pub functional_unit: ExactLifecycleSubjectRefV1,
    #[serde(default)]
    pub manufacturing_plan_refs: Vec<ExactLifecycleSubjectRefV1>,
    #[serde(default)]
    pub strategies: Vec<LifecycleStrategyRefV1>,
    #[serde(default)]
    pub assessments: Vec<LifecycleAssessmentRefV1>,
    pub authority_ceiling: LifecycleDesignAuthorityCeilingV1,
    pub display_label: Option<String>,
    pub notes: Option<String>,
}

impl LifecycleDesignProfileV1 {
    pub fn validate(&self) -> Result<(), LifecycleContractError> {
        canonical("profile.semantic_version", &self.semantic_version)?;
        self.subject_configuration.validate()?;
        self.functional_unit.validate()?;
        if self.manufacturing_plan_refs.is_empty() {
            return Err(LifecycleContractError::MissingManufacturingPlan);
        }
        if self.strategies.is_empty() {
            return Err(LifecycleContractError::MissingStrategy);
        }

        validate_unique_exact_refs("manufacturing_plan_refs", &self.manufacturing_plan_refs)?;
        validate_unique_strategy_refs(&self.strategies)?;
        validate_unique_assessment_refs(&self.assessments)?;

        if let Some(label) = &self.display_label {
            canonical("profile.display_label", label)?;
        }
        if let Some(notes) = &self.notes {
            canonical("profile.notes", notes)?;
        }
        Ok(())
    }

    pub fn profile_id(&self) -> Result<String, LifecycleContractError> {
        self.validate()?;
        let mut plans = self.manufacturing_plan_refs.clone();
        plans.sort();
        let mut strategies = self.strategies.clone();
        strategies.sort();
        let mut assessments = self.assessments.clone();
        assessments.sort();

        let mut hasher = blake3::Hasher::new();
        hash_field(&mut hasher, PROFILE_ID_DOMAIN);
        hash_field(&mut hasher, &self.semantic_version);
        hash_field(&mut hasher, &self.subject_configuration.ref_id()?);
        hash_field(&mut hasher, &self.functional_unit.ref_id()?);
        hash_field(&mut hasher, "manufacturing-plans");
        for reference in plans {
            hash_field(&mut hasher, &reference.ref_id()?);
        }
        hash_field(&mut hasher, "strategies");
        for strategy in strategies {
            hash_field(&mut hasher, &strategy.identity_key()?);
        }
        hash_field(&mut hasher, "assessments");
        for assessment in assessments {
            hash_field(&mut hasher, &assessment.identity_key()?);
        }
        hash_field(
            &mut hasher,
            "authority:design-intent-and-assessment-references-only",
        );
        Ok(hasher.finalize().to_hex().to_string())
    }
}

fn validate_unique_exact_refs(
    field: &'static str,
    refs: &[ExactLifecycleSubjectRefV1],
) -> Result<(), LifecycleContractError> {
    let mut seen = BTreeSet::new();
    for reference in refs {
        reference.validate()?;
        let key = reference.ref_id()?;
        if !seen.insert(key.clone()) {
            return Err(LifecycleContractError::DuplicateReference { field, value: key });
        }
    }
    Ok(())
}

fn validate_unique_strategy_refs(
    refs: &[LifecycleStrategyRefV1],
) -> Result<(), LifecycleContractError> {
    let mut seen = BTreeSet::new();
    for reference in refs {
        reference.validate()?;
        let key = reference.identity_key()?;
        if !seen.insert(key.clone()) {
            return Err(LifecycleContractError::DuplicateReference {
                field: "strategies",
                value: key,
            });
        }
    }
    Ok(())
}

fn validate_unique_assessment_refs(
    refs: &[LifecycleAssessmentRefV1],
) -> Result<(), LifecycleContractError> {
    let mut seen = BTreeSet::new();
    for reference in refs {
        reference.validate()?;
        let key = reference.identity_key()?;
        if !seen.insert(key.clone()) {
            return Err(LifecycleContractError::DuplicateReference {
                field: "assessments",
                value: key,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn subject(namespace: &str, id: &str, fill: char) -> ExactLifecycleSubjectRefV1 {
        ExactLifecycleSubjectRefV1 {
            namespace: namespace.into(),
            subject_id: id.into(),
            semantic_version: "1".into(),
            content_blake3: std::iter::repeat_n(fill, 64).collect(),
        }
    }

    fn profile() -> LifecycleDesignProfileV1 {
        LifecycleDesignProfileV1 {
            semantic_version: "1.0.0".into(),
            subject_configuration: subject("symthaea.configuration", "product-a", 'a'),
            functional_unit: subject("symthaea.functional-unit", "service-unit-a", 'b'),
            manufacturing_plan_refs: vec![subject("symthaea.mfg-plan", "plan-a", 'c')],
            strategies: vec![
                LifecycleStrategyRefV1 {
                    kind: LifecycleStrategyKindV1::Repair,
                    subject: subject("symthaea.lifecycle-strategy", "repair-a", 'd'),
                },
                LifecycleStrategyRefV1 {
                    kind: LifecycleStrategyKindV1::MaterialRecovery,
                    subject: subject("symthaea.lifecycle-strategy", "recovery-a", 'e'),
                },
            ],
            assessments: vec![LifecycleAssessmentRefV1 {
                kind: LifecycleAssessmentKindV1::MaterialFlowAccount,
                subject: subject("mycelix.material-flow", "account-a", 'f'),
            }],
            authority_ceiling:
                LifecycleDesignAuthorityCeilingV1::DesignIntentAndAssessmentReferencesOnly,
            display_label: Some("Serviceable product A".into()),
            notes: Some("navigation-only note".into()),
        }
    }

    #[test]
    fn display_metadata_does_not_change_profile_identity() {
        let a = profile();
        let mut b = a.clone();
        b.display_label = Some("renamed".into());
        b.notes = Some("different prose".into());
        assert_eq!(a.profile_id().unwrap(), b.profile_id().unwrap());
    }

    #[test]
    fn changing_recovery_subject_changes_profile_identity() {
        let a = profile();
        let mut b = a.clone();
        b.strategies[1].subject = subject("symthaea.lifecycle-strategy", "recovery-b", '9');
        assert_ne!(a.profile_id().unwrap(), b.profile_id().unwrap());
    }

    #[test]
    fn collection_order_is_canonicalized() {
        let a = profile();
        let mut b = a.clone();
        b.strategies.reverse();
        assert_eq!(a.profile_id().unwrap(), b.profile_id().unwrap());
    }

    #[test]
    fn duplicate_semantic_ref_rejects() {
        let mut p = profile();
        p.manufacturing_plan_refs.push(p.manufacturing_plan_refs[0].clone());
        assert!(matches!(
            p.validate(),
            Err(LifecycleContractError::DuplicateReference { .. })
        ));
    }

    #[test]
    fn same_subject_under_different_strategy_kind_is_semantically_distinct() {
        let a = subject("symthaea.lifecycle-strategy", "shared-subject", '7');
        let repair = LifecycleStrategyRefV1 {
            kind: LifecycleStrategyKindV1::Repair,
            subject: a.clone(),
        };
        let remanufacture = LifecycleStrategyRefV1 {
            kind: LifecycleStrategyKindV1::Remanufacture,
            subject: a,
        };
        assert_ne!(repair.identity_key().unwrap(), remanufacture.identity_key().unwrap());
    }

    #[test]
    fn service_life_assumption_is_distinct_from_lifecycle_exposure_evidence() {
        let a = subject("symthaea.se-vv", "service-life-subject", '8');
        let assumption = LifecycleAssessmentRefV1 {
            kind: LifecycleAssessmentKindV1::ServiceLifeAssumption,
            subject: a.clone(),
        };
        let evidence = LifecycleAssessmentRefV1 {
            kind: LifecycleAssessmentKindV1::LifecycleExposureEvidence,
            subject: a,
        };
        assert_ne!(assumption.identity_key().unwrap(), evidence.identity_key().unwrap());
    }

    #[test]
    fn malformed_digest_fails_closed() {
        let mut p = profile();
        p.functional_unit.content_blake3 = "BAD".into();
        assert_eq!(p.validate(), Err(LifecycleContractError::InvalidDigest));
    }

    #[test]
    fn serialization_round_trip_preserves_identity() {
        let p = profile();
        let encoded = serde_json::to_string(&p).unwrap();
        let decoded: LifecycleDesignProfileV1 = serde_json::from_str(&encoded).unwrap();
        assert_eq!(p.profile_id().unwrap(), decoded.profile_id().unwrap());
    }

    #[test]
    fn public_authority_ceiling_is_design_only() {
        assert_eq!(
            profile().authority_ceiling,
            LifecycleDesignAuthorityCeilingV1::DesignIntentAndAssessmentReferencesOnly
        );
    }
}
