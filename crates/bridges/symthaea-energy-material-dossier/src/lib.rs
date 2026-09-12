// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Identity-bound, multi-source energy-material evidence dossiers.
//!
//! This crate composes generic discovery predictions from independent evidence
//! adapters without trusting adapter-specific structs. Every contribution must
//! bind to one explicit candidate identity assertion and source-receipt digest.
//! Partial dossiers remain partial; missing evidence is never synthesized.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_discovery::{CandidateId, Prediction};
use symthaea_energy_material_screening::{
    EnergyMaterialEvidenceBundle, EnergyMaterialScreeningAssessment,
    EnergyMaterialScreeningPolicy, EvidenceDimension,
};
use thiserror::Error;

pub const CAPABILITY_CLASSIFICATION: &str =
    "ENERGY-MATERIAL EVIDENCE DOSSIER ONLY -- identity-bound evidence assembly is not scientific validation, synthesis authorization, certification, procurement, manufacturing, or deployment authority.";
const DOSSIER_DIGEST_DOMAIN: &[u8] = b"symthaea.energy-material-dossier.v0\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdentityAssertionBasis {
    /// Evidence is generated directly from the internal Symthaea candidate.
    InternalCandidate,
    /// External subject identifier exactly equals the candidate id.
    ExactExternalId,
    /// Caller asserts that a differently named external subject refers to this
    /// candidate; the note remains an assumption rather than proof.
    ExplicitMapping { note: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdentityAssertion {
    pub assertion_id: String,
    pub candidate_id: CandidateId,
    pub namespace: String,
    pub subject_id: String,
    pub basis: IdentityAssertionBasis,
    pub source_receipt_sha256: String,
}

impl IdentityAssertion {
    pub fn validate(&self) -> Result<(), DossierError> {
        CandidateId::new(self.candidate_id.0.clone())?;
        if self.assertion_id.trim().is_empty()
            || self.namespace.trim().is_empty()
            || self.subject_id.trim().is_empty()
        {
            return Err(DossierError::InvalidIdentity(
                "assertion_id, namespace and subject_id must be non-empty".into(),
            ));
        }
        validate_sha256(&self.source_receipt_sha256, "identity source-receipt SHA-256")?;
        match &self.basis {
            IdentityAssertionBasis::InternalCandidate => {
                if self.namespace != "symthaea" || self.subject_id != self.candidate_id.0 {
                    return Err(DossierError::InvalidIdentity(
                        "InternalCandidate requires namespace='symthaea' and subject_id == candidate_id"
                            .into(),
                    ));
                }
            }
            IdentityAssertionBasis::ExactExternalId => {
                if self.subject_id != self.candidate_id.0 {
                    return Err(DossierError::InvalidIdentity(
                        "ExactExternalId requires subject_id == candidate_id".into(),
                    ));
                }
            }
            IdentityAssertionBasis::ExplicitMapping { note } => {
                if note.trim().is_empty() {
                    return Err(DossierError::InvalidIdentity(
                        "ExplicitMapping requires a non-empty review note".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceContribution {
    pub dimension: EvidenceDimension,
    pub candidate_id: CandidateId,
    pub identity_assertion_id: String,
    pub source_receipt_sha256: String,
    pub prediction: Prediction,
}

impl EvidenceContribution {
    pub fn validate(&self) -> Result<(), DossierError> {
        CandidateId::new(self.candidate_id.0.clone())?;
        if self.identity_assertion_id.trim().is_empty() {
            return Err(DossierError::InvalidContribution(
                "identity_assertion_id cannot be empty".into(),
            ));
        }
        validate_sha256(&self.source_receipt_sha256, "contribution source-receipt SHA-256")?;
        self.prediction.validate()?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnergyMaterialDossier {
    pub schema: String,
    pub capability_classification: String,
    pub candidate_id: CandidateId,
    pub policy_id: String,
    pub policy_sha256: String,
    pub identity_assertions: Vec<IdentityAssertion>,
    pub contributions: Vec<EvidenceContribution>,
    pub screening_assessment: EnergyMaterialScreeningAssessment,
}

impl EnergyMaterialDossier {
    pub fn sha256(&self) -> Result<String, DossierError> {
        self.validate_integrity()?;
        let bytes = serde_json::to_vec(self)?;
        let mut hasher = Sha256::new();
        hasher.update(DOSSIER_DIGEST_DOMAIN);
        hasher.update(bytes);
        Ok(hex_lower(&hasher.finalize()))
    }

    pub fn validate_integrity(&self) -> Result<(), DossierError> {
        if self.schema != "symthaea.energy-material-dossier.v0"
            || self.capability_classification != CAPABILITY_CLASSIFICATION
        {
            return Err(DossierError::InvalidDossier(
                "dossier schema/capability classification was altered".into(),
            ));
        }
        CandidateId::new(self.candidate_id.0.clone())?;
        if self.policy_id.trim().is_empty() {
            return Err(DossierError::InvalidDossier("policy_id cannot be empty".into()));
        }
        validate_sha256(&self.policy_sha256, "policy SHA-256")?;
        if self.screening_assessment.candidate_id != self.candidate_id
            || self.screening_assessment.policy_id != self.policy_id
            || self.screening_assessment.policy_sha256 != self.policy_sha256
        {
            return Err(DossierError::InvalidDossier(
                "screening assessment identity/policy differs from dossier".into(),
            ));
        }
        for assertion in &self.identity_assertions {
            assertion.validate()?;
            if assertion.candidate_id != self.candidate_id {
                return Err(DossierError::CandidateMismatch {
                    expected: self.candidate_id.0.clone(),
                    found: assertion.candidate_id.0.clone(),
                });
            }
        }
        for contribution in &self.contributions {
            contribution.validate()?;
            if contribution.candidate_id != self.candidate_id {
                return Err(DossierError::CandidateMismatch {
                    expected: self.candidate_id.0.clone(),
                    found: contribution.candidate_id.0.clone(),
                });
            }
        }
        Ok(())
    }
}

pub fn assemble_dossier(
    candidate_id: CandidateId,
    policy: &EnergyMaterialScreeningPolicy,
    mut identity_assertions: Vec<IdentityAssertion>,
    mut contributions: Vec<EvidenceContribution>,
) -> Result<EnergyMaterialDossier, DossierError> {
    CandidateId::new(candidate_id.0.clone())?;
    policy.validate()?;
    let policy_sha256 = policy.sha256()?;

    let mut assertion_ids = BTreeSet::new();
    for assertion in &identity_assertions {
        assertion.validate()?;
        if assertion.candidate_id != candidate_id {
            return Err(DossierError::CandidateMismatch {
                expected: candidate_id.0.clone(),
                found: assertion.candidate_id.0.clone(),
            });
        }
        if !assertion_ids.insert(assertion.assertion_id.as_str()) {
            return Err(DossierError::DuplicateIdentityAssertion(
                assertion.assertion_id.clone(),
            ));
        }
    }
    identity_assertions.sort_by(|left, right| left.assertion_id.cmp(&right.assertion_id));
    let assertions: BTreeMap<&str, &IdentityAssertion> = identity_assertions
        .iter()
        .map(|assertion| (assertion.assertion_id.as_str(), assertion))
        .collect();

    let contracts: BTreeMap<u8, _> = policy
        .contracts
        .iter()
        .map(|contract| (dimension_code(contract.dimension), contract))
        .collect();
    let mut seen_dimensions = BTreeSet::new();
    for contribution in &contributions {
        contribution.validate()?;
        if contribution.candidate_id != candidate_id {
            return Err(DossierError::CandidateMismatch {
                expected: candidate_id.0.clone(),
                found: contribution.candidate_id.0.clone(),
            });
        }
        let code = dimension_code(contribution.dimension);
        if !seen_dimensions.insert(code) {
            return Err(DossierError::DuplicateDimension(contribution.dimension));
        }
        let contract = contracts
            .get(&code)
            .copied()
            .ok_or(DossierError::UnknownDimension(contribution.dimension))?;
        if contribution.prediction.metric != contract.metric
            || contribution.prediction.unit != contract.unit
        {
            return Err(DossierError::MetricContractMismatch {
                dimension: contribution.dimension,
                expected_metric: contract.metric.clone(),
                expected_unit: contract.unit.clone(),
                found_metric: contribution.prediction.metric.clone(),
                found_unit: contribution.prediction.unit.clone(),
            });
        }
        let assertion = assertions
            .get(contribution.identity_assertion_id.as_str())
            .copied()
            .ok_or_else(|| {
                DossierError::MissingIdentityAssertion(contribution.identity_assertion_id.clone())
            })?;
        if assertion.source_receipt_sha256 != contribution.source_receipt_sha256 {
            return Err(DossierError::ReceiptBindingMismatch {
                assertion_id: assertion.assertion_id.clone(),
            });
        }
    }
    contributions.sort_by_key(|contribution| dimension_code(contribution.dimension));

    let bundle = EnergyMaterialEvidenceBundle {
        candidate_id: candidate_id.clone(),
        predictions: contributions
            .iter()
            .map(|contribution| contribution.prediction.clone())
            .collect(),
    };
    let screening_assessment = bundle.assess(policy)?;

    let dossier = EnergyMaterialDossier {
        schema: "symthaea.energy-material-dossier.v0".into(),
        capability_classification: CAPABILITY_CLASSIFICATION.into(),
        candidate_id,
        policy_id: policy.policy_id.clone(),
        policy_sha256,
        identity_assertions,
        contributions,
        screening_assessment,
    };
    dossier.validate_integrity()?;
    Ok(dossier)
}

#[derive(Debug, Error)]
pub enum DossierError {
    #[error("invalid identity assertion: {0}")]
    InvalidIdentity(String),
    #[error("invalid evidence contribution: {0}")]
    InvalidContribution(String),
    #[error("invalid dossier: {0}")]
    InvalidDossier(String),
    #[error("candidate mismatch: expected {expected:?}, found {found:?}")]
    CandidateMismatch { expected: String, found: String },
    #[error("duplicate identity assertion id {0:?}")]
    DuplicateIdentityAssertion(String),
    #[error("duplicate evidence dimension {0:?}")]
    DuplicateDimension(EvidenceDimension),
    #[error("dimension {0:?} is not declared by the screening policy")]
    UnknownDimension(EvidenceDimension),
    #[error("missing identity assertion {0:?}")]
    MissingIdentityAssertion(String),
    #[error("identity assertion {assertion_id:?} and contribution bind different source receipts")]
    ReceiptBindingMismatch { assertion_id: String },
    #[error(
        "metric contract mismatch for {dimension:?}: expected {expected_metric:?}/{expected_unit:?}, found {found_metric:?}/{found_unit:?}"
    )]
    MetricContractMismatch {
        dimension: EvidenceDimension,
        expected_metric: String,
        expected_unit: String,
        found_metric: String,
        found_unit: String,
    },
    #[error(transparent)]
    Screening(#[from] symthaea_energy_material_screening::ScreeningError),
    #[error(transparent)]
    Discovery(#[from] symthaea_discovery::DiscoveryError),
    #[error("dossier serialization failed: {0}")]
    Json(#[from] serde_json::Error),
}

fn validate_sha256(value: &str, name: &str) -> Result<(), DossierError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(DossierError::InvalidDossier(format!(
            "{name} must be exactly 64 hexadecimal characters"
        )));
    }
    Ok(())
}

fn dimension_code(dimension: EvidenceDimension) -> u8 {
    match dimension {
        EvidenceDimension::FunctionalPerformance => 0,
        EvidenceDimension::ThermodynamicStability => 1,
        EvidenceDimension::CriticalMaterialBurden => 2,
        EvidenceDimension::SupplyResilience => 3,
        EvidenceDimension::HumanEnvironmentalHazard => 4,
        EvidenceDimension::Circularity => 5,
        EvidenceDimension::Manufacturability => 6,
    }
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_discovery::{
        EvidenceKind, EvidenceRef, FidelityLevel, ModelProvenance, ObjectiveDirection,
        UncertaintyEstimate,
    };
    use symthaea_energy_material_screening::{
        EvidenceCompleteness, MetricContract,
    };

    fn policy() -> EnergyMaterialScreeningPolicy {
        let dims = [
            (EvidenceDimension::FunctionalPerformance, "functional", "score"),
            (EvidenceDimension::ThermodynamicStability, "stability", "score"),
            (EvidenceDimension::CriticalMaterialBurden, "critical", "score"),
            (EvidenceDimension::SupplyResilience, "supply", "score"),
            (EvidenceDimension::HumanEnvironmentalHazard, "hazard", "score"),
            (EvidenceDimension::Circularity, "circular", "score"),
            (EvidenceDimension::Manufacturability, "manufacturing", "score"),
        ];
        EnergyMaterialScreeningPolicy {
            policy_id: "dossier-test-policy".into(),
            contracts: dims
                .into_iter()
                .map(|(dimension, metric, unit)| MetricContract {
                    dimension,
                    metric: metric.into(),
                    unit: unit.into(),
                    direction: ObjectiveDirection::Minimize,
                    minimum_fidelity: FidelityLevel::Surrogate,
                    accepted_evidence_kinds: vec![EvidenceKind::Dataset],
                })
                .collect(),
            constraints: vec![],
        }
    }

    fn prediction(metric: &str) -> Prediction {
        Prediction {
            metric: metric.into(),
            value: 1.0,
            unit: "score".into(),
            uncertainty: UncertaintyEstimate::new(1.0, 0.0).unwrap(),
            fidelity: FidelityLevel::Surrogate,
            model: ModelProvenance::named("fixture").unwrap(),
            assumptions: vec![],
            evidence: vec![EvidenceRef {
                id: format!("evidence-{metric}"),
                kind: EvidenceKind::Dataset,
                uri: None,
                digest: Some("sha256:fixture".into()),
                note: None,
            }],
        }
    }

    fn sha() -> String {
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into()
    }

    #[test]
    fn partial_dossier_remains_incomplete_instead_of_filling_missing_dimensions() {
        let candidate = CandidateId::new("candidate-a").unwrap();
        let assertion = IdentityAssertion {
            assertion_id: "internal".into(),
            candidate_id: candidate.clone(),
            namespace: "symthaea".into(),
            subject_id: candidate.0.clone(),
            basis: IdentityAssertionBasis::InternalCandidate,
            source_receipt_sha256: sha(),
        };
        let contribution = EvidenceContribution {
            dimension: EvidenceDimension::FunctionalPerformance,
            candidate_id: candidate.clone(),
            identity_assertion_id: "internal".into(),
            source_receipt_sha256: sha(),
            prediction: prediction("functional"),
        };
        let dossier = assemble_dossier(candidate, &policy(), vec![assertion], vec![contribution])
            .unwrap();
        assert_eq!(
            dossier.screening_assessment.completeness,
            EvidenceCompleteness::Incomplete
        );
        assert!(dossier.screening_assessment.evaluation.is_none());
    }

    #[test]
    fn contribution_must_match_policy_metric_and_bound_receipt() {
        let candidate = CandidateId::new("candidate-a").unwrap();
        let assertion = IdentityAssertion {
            assertion_id: "a".into(),
            candidate_id: candidate.clone(),
            namespace: "external".into(),
            subject_id: "external-1".into(),
            basis: IdentityAssertionBasis::ExplicitMapping {
                note: "fixture".into(),
            },
            source_receipt_sha256: sha(),
        };
        let wrong_metric = EvidenceContribution {
            dimension: EvidenceDimension::FunctionalPerformance,
            candidate_id: candidate.clone(),
            identity_assertion_id: "a".into(),
            source_receipt_sha256: sha(),
            prediction: prediction("wrong"),
        };
        assert!(assemble_dossier(
            candidate.clone(),
            &policy(),
            vec![assertion.clone()],
            vec![wrong_metric],
        )
        .is_err());

        let mut wrong_receipt = sha();
        wrong_receipt.replace_range(0..1, "b");
        let contribution = EvidenceContribution {
            dimension: EvidenceDimension::FunctionalPerformance,
            candidate_id: candidate.clone(),
            identity_assertion_id: "a".into(),
            source_receipt_sha256: wrong_receipt,
            prediction: prediction("functional"),
        };
        assert!(assemble_dossier(candidate, &policy(), vec![assertion], vec![contribution]).is_err());
    }

    #[test]
    fn dossier_digest_is_input_order_independent() {
        let candidate = CandidateId::new("candidate-a").unwrap();
        let assertions = vec![
            IdentityAssertion {
                assertion_id: "z".into(),
                candidate_id: candidate.clone(),
                namespace: "external".into(),
                subject_id: "z-subject".into(),
                basis: IdentityAssertionBasis::ExplicitMapping { note: "z".into() },
                source_receipt_sha256: sha(),
            },
            IdentityAssertion {
                assertion_id: "a".into(),
                candidate_id: candidate.clone(),
                namespace: "external".into(),
                subject_id: "a-subject".into(),
                basis: IdentityAssertionBasis::ExplicitMapping { note: "a".into() },
                source_receipt_sha256: sha(),
            },
        ];
        let contributions = vec![
            EvidenceContribution {
                dimension: EvidenceDimension::SupplyResilience,
                candidate_id: candidate.clone(),
                identity_assertion_id: "z".into(),
                source_receipt_sha256: sha(),
                prediction: prediction("supply"),
            },
            EvidenceContribution {
                dimension: EvidenceDimension::FunctionalPerformance,
                candidate_id: candidate.clone(),
                identity_assertion_id: "a".into(),
                source_receipt_sha256: sha(),
                prediction: prediction("functional"),
            },
        ];
        let first = assemble_dossier(
            candidate.clone(),
            &policy(),
            assertions.clone(),
            contributions.clone(),
        )
        .unwrap();
        let second = assemble_dossier(
            candidate,
            &policy(),
            assertions.into_iter().rev().collect(),
            contributions.into_iter().rev().collect(),
        )
        .unwrap();
        assert_eq!(first.sha256().unwrap(), second.sha256().unwrap());
    }
}
