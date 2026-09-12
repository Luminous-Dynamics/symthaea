// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Candidate-version binding for energy-material evidence dossiers.
//!
//! A textual candidate id is not a sufficient scientific identity: the same id
//! could be reused after composition, structure, process, or origin metadata
//! changes. This crate content-addresses the complete domain-neutral `Candidate`
//! and requires one explicit version-review attestation for every dossier
//! contribution before the dossier can move downstream.
//!
//! V0 deliberately treats those attestations as review records, not signatures
//! or cryptographic proof that an older source receipt was generated from the
//! exact candidate version. Future receipt schemas should embed the candidate
//! version digest natively and can then retire this compatibility review step.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_discovery::{Candidate, CandidateId, CandidateOrigin};
use symthaea_energy_material_dossier::EnergyMaterialDossier;
use symthaea_energy_material_screening::EvidenceDimension;
use thiserror::Error;

pub const CAPABILITY_CLASSIFICATION: &str =
    "CANDIDATE-VERSION DOSSIER BINDING ONLY -- review attestations bind evidence assembly to one immutable candidate specification but are not signatures, experimental validation, certification, synthesis authority, or deployment approval.";

const CANDIDATE_DIGEST_DOMAIN: &[u8] = b"symthaea.discovery-candidate.version.v0\0";
const BOUND_DOSSIER_DIGEST_DOMAIN: &[u8] = b"symthaea.energy-material.candidate-bound-dossier.v0\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateVersionAnchor {
    pub schema: String,
    pub candidate: Candidate,
    pub candidate_sha256: String,
}

impl CandidateVersionAnchor {
    pub fn validate(&self) -> Result<(), CandidateVersionError> {
        if self.schema != "symthaea.discovery-candidate.version.v0" {
            return Err(CandidateVersionError::InvalidCandidate(
                "candidate-version schema was altered".into(),
            ));
        }
        validate_candidate(&self.candidate)?;
        validate_sha256(&self.candidate_sha256, "candidate SHA-256")?;
        let expected = candidate_sha256(&self.candidate)?;
        if self.candidate_sha256 != expected {
            return Err(CandidateVersionError::CandidateDigestMismatch {
                expected,
                actual: self.candidate_sha256.clone(),
            });
        }
        Ok(())
    }
}

pub fn anchor_candidate(candidate: Candidate) -> Result<CandidateVersionAnchor, CandidateVersionError> {
    validate_candidate(&candidate)?;
    let candidate_sha256 = candidate_sha256(&candidate)?;
    Ok(CandidateVersionAnchor {
        schema: "symthaea.discovery-candidate.version.v0".into(),
        candidate,
        candidate_sha256,
    })
}

pub fn candidate_sha256(candidate: &Candidate) -> Result<String, CandidateVersionError> {
    validate_candidate(candidate)?;
    let encoded = serde_json::to_vec(candidate)?;
    let mut hasher = Sha256::new();
    hasher.update(CANDIDATE_DIGEST_DOMAIN);
    hasher.update(encoded);
    Ok(hex_lower(&hasher.finalize()))
}

/// Explicit compatibility attestation binding one already-authored source
/// receipt to one immutable candidate version.
///
/// This is review provenance, not independent proof. The attestation therefore
/// requires both a reviewer identifier and a human-readable basis note.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReceiptVersionAttestation {
    pub attestation_id: String,
    pub dimension: EvidenceDimension,
    pub source_receipt_sha256: String,
    pub candidate_sha256: String,
    pub reviewer: String,
    pub review_note: String,
}

impl ReceiptVersionAttestation {
    pub fn validate(&self) -> Result<(), CandidateVersionError> {
        if self.attestation_id.trim().is_empty()
            || self.reviewer.trim().is_empty()
            || self.review_note.trim().is_empty()
        {
            return Err(CandidateVersionError::InvalidAttestation(
                "attestation_id, reviewer and review_note must be non-empty".into(),
            ));
        }
        validate_sha256(&self.source_receipt_sha256, "attestation source-receipt SHA-256")?;
        validate_sha256(&self.candidate_sha256, "attestation candidate SHA-256")?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateVersionBoundDossier {
    pub schema: String,
    pub capability_classification: String,
    pub candidate_anchor: CandidateVersionAnchor,
    pub dossier_sha256: String,
    pub dossier: EnergyMaterialDossier,
    pub receipt_version_attestations: Vec<ReceiptVersionAttestation>,
}

impl CandidateVersionBoundDossier {
    pub fn validate_integrity(&self) -> Result<(), CandidateVersionError> {
        if self.schema != "symthaea.energy-material.candidate-bound-dossier.v0"
            || self.capability_classification != CAPABILITY_CLASSIFICATION
        {
            return Err(CandidateVersionError::InvalidBoundDossier(
                "bound-dossier schema/capability classification was altered".into(),
            ));
        }
        self.candidate_anchor.validate()?;
        self.dossier.validate_integrity()?;
        validate_sha256(&self.dossier_sha256, "dossier SHA-256")?;
        let expected_dossier_sha = self.dossier.sha256()?;
        if self.dossier_sha256 != expected_dossier_sha {
            return Err(CandidateVersionError::DossierDigestMismatch {
                expected: expected_dossier_sha,
                actual: self.dossier_sha256.clone(),
            });
        }
        if self.dossier.candidate_id != self.candidate_anchor.candidate.id {
            return Err(CandidateVersionError::CandidateIdMismatch {
                expected: self.candidate_anchor.candidate.id.0.clone(),
                found: self.dossier.candidate_id.0.clone(),
            });
        }

        let contributions: BTreeMap<u8, (&EvidenceDimension, &str)> = self
            .dossier
            .contributions
            .iter()
            .map(|contribution| {
                (
                    dimension_code(contribution.dimension),
                    (&contribution.dimension, contribution.source_receipt_sha256.as_str()),
                )
            })
            .collect();

        if contributions.len() != self.dossier.contributions.len() {
            return Err(CandidateVersionError::InvalidBoundDossier(
                "dossier contains duplicate evidence dimensions".into(),
            ));
        }

        let mut attestation_ids = BTreeSet::new();
        let mut seen_dimensions = BTreeSet::new();
        for attestation in &self.receipt_version_attestations {
            attestation.validate()?;
            if !attestation_ids.insert(attestation.attestation_id.as_str()) {
                return Err(CandidateVersionError::DuplicateAttestationId(
                    attestation.attestation_id.clone(),
                ));
            }
            let code = dimension_code(attestation.dimension);
            if !seen_dimensions.insert(code) {
                return Err(CandidateVersionError::DuplicateAttestationDimension(
                    attestation.dimension,
                ));
            }
            if attestation.candidate_sha256 != self.candidate_anchor.candidate_sha256 {
                return Err(CandidateVersionError::AttestationCandidateMismatch {
                    dimension: attestation.dimension,
                });
            }
            let (_, expected_receipt) = contributions
                .get(&code)
                .copied()
                .ok_or(CandidateVersionError::AttestationWithoutContribution(
                    attestation.dimension,
                ))?;
            if attestation.source_receipt_sha256 != expected_receipt {
                return Err(CandidateVersionError::AttestationReceiptMismatch {
                    dimension: attestation.dimension,
                });
            }
        }

        for contribution in &self.dossier.contributions {
            let code = dimension_code(contribution.dimension);
            if !seen_dimensions.contains(&code) {
                return Err(CandidateVersionError::MissingVersionAttestation(
                    contribution.dimension,
                ));
            }
        }

        if self.receipt_version_attestations.len() != self.dossier.contributions.len() {
            return Err(CandidateVersionError::InvalidBoundDossier(
                "version-attestation set must be a one-to-one mapping with dossier contributions"
                    .into(),
            ));
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<String, CandidateVersionError> {
        self.validate_integrity()?;
        let encoded = serde_json::to_vec(self)?;
        let mut hasher = Sha256::new();
        hasher.update(BOUND_DOSSIER_DIGEST_DOMAIN);
        hasher.update(encoded);
        Ok(hex_lower(&hasher.finalize()))
    }
}

pub fn bind_dossier_to_candidate_version(
    candidate: Candidate,
    dossier: EnergyMaterialDossier,
    mut receipt_version_attestations: Vec<ReceiptVersionAttestation>,
) -> Result<CandidateVersionBoundDossier, CandidateVersionError> {
    let candidate_anchor = anchor_candidate(candidate)?;
    if dossier.candidate_id != candidate_anchor.candidate.id {
        return Err(CandidateVersionError::CandidateIdMismatch {
            expected: candidate_anchor.candidate.id.0.clone(),
            found: dossier.candidate_id.0.clone(),
        });
    }
    dossier.validate_integrity()?;
    let dossier_sha256 = dossier.sha256()?;

    receipt_version_attestations.sort_by(|left, right| {
        dimension_code(left.dimension)
            .cmp(&dimension_code(right.dimension))
            .then_with(|| left.attestation_id.cmp(&right.attestation_id))
    });

    let bound = CandidateVersionBoundDossier {
        schema: "symthaea.energy-material.candidate-bound-dossier.v0".into(),
        capability_classification: CAPABILITY_CLASSIFICATION.into(),
        candidate_anchor,
        dossier_sha256,
        dossier,
        receipt_version_attestations,
    };
    bound.validate_integrity()?;
    Ok(bound)
}

#[derive(Debug, Error)]
pub enum CandidateVersionError {
    #[error("invalid candidate version: {0}")]
    InvalidCandidate(String),
    #[error("invalid receipt-version attestation: {0}")]
    InvalidAttestation(String),
    #[error("invalid candidate-bound dossier: {0}")]
    InvalidBoundDossier(String),
    #[error("candidate SHA-256 mismatch: expected {expected}, got {actual}")]
    CandidateDigestMismatch { expected: String, actual: String },
    #[error("dossier SHA-256 mismatch: expected {expected}, got {actual}")]
    DossierDigestMismatch { expected: String, actual: String },
    #[error("candidate id mismatch: expected {expected:?}, found {found:?}")]
    CandidateIdMismatch { expected: String, found: String },
    #[error("duplicate version-attestation id {0:?}")]
    DuplicateAttestationId(String),
    #[error("duplicate version attestation for dimension {0:?}")]
    DuplicateAttestationDimension(EvidenceDimension),
    #[error("missing candidate-version attestation for dossier dimension {0:?}")]
    MissingVersionAttestation(EvidenceDimension),
    #[error("version attestation exists for absent dossier dimension {0:?}")]
    AttestationWithoutContribution(EvidenceDimension),
    #[error("version attestation candidate digest differs for dimension {dimension:?}")]
    AttestationCandidateMismatch { dimension: EvidenceDimension },
    #[error("version attestation source receipt differs for dimension {dimension:?}")]
    AttestationReceiptMismatch { dimension: EvidenceDimension },
    #[error(transparent)]
    Discovery(#[from] symthaea_discovery::DiscoveryError),
    #[error(transparent)]
    Dossier(#[from] symthaea_energy_material_dossier::DossierError),
    #[error("candidate-version JSON failed to parse/serialize: {0}")]
    Json(#[from] serde_json::Error),
}

fn validate_candidate(candidate: &Candidate) -> Result<(), CandidateVersionError> {
    CandidateId::new(candidate.id.0.clone())?;
    if candidate.kind.trim().is_empty() {
        return Err(CandidateVersionError::InvalidCandidate(
            "candidate kind cannot be empty".into(),
        ));
    }
    for (key, value) in &candidate.specification {
        if key.trim().is_empty() || value.trim().is_empty() {
            return Err(CandidateVersionError::InvalidCandidate(
                "candidate specification keys and values must be non-empty".into(),
            ));
        }
        if key.contains('\0') || value.contains('\0') {
            return Err(CandidateVersionError::InvalidCandidate(
                "candidate specification cannot contain NUL characters".into(),
            ));
        }
    }
    match &candidate.origin {
        CandidateOrigin::UserProposed => {}
        CandidateOrigin::Generated { generator, version } => {
            if generator.trim().is_empty() {
                return Err(CandidateVersionError::InvalidCandidate(
                    "generated candidate requires a non-empty generator".into(),
                ));
            }
            if version.as_deref().is_some_and(|value| value.trim().is_empty()) {
                return Err(CandidateVersionError::InvalidCandidate(
                    "generated candidate version cannot be blank when present".into(),
                ));
            }
        }
        CandidateOrigin::Imported { source } => {
            if source.trim().is_empty() {
                return Err(CandidateVersionError::InvalidCandidate(
                    "imported candidate requires a non-empty source".into(),
                ));
            }
        }
    }
    Ok(())
}

fn validate_sha256(value: &str, name: &str) -> Result<(), CandidateVersionError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(CandidateVersionError::InvalidBoundDossier(format!(
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
    use std::collections::BTreeMap;
    use symthaea_discovery::{
        CandidateOrigin, EvidenceKind, EvidenceRef, FidelityLevel, ModelProvenance,
        ObjectiveDirection, Prediction, UncertaintyEstimate,
    };
    use symthaea_energy_material_dossier::{
        assemble_dossier, EvidenceContribution, IdentityAssertion, IdentityAssertionBasis,
    };
    use symthaea_energy_material_screening::{
        EnergyMaterialScreeningPolicy, MetricContract,
    };

    fn candidate(spec_value: &str) -> Candidate {
        let mut specification = BTreeMap::new();
        specification.insert("formula".into(), spec_value.into());
        Candidate {
            id: CandidateId::new("candidate-a").unwrap(),
            kind: "energy_material".into(),
            specification,
            origin: CandidateOrigin::Generated {
                generator: "fixture-generator".into(),
                version: Some("v1".into()),
            },
        }
    }

    fn policy() -> EnergyMaterialScreeningPolicy {
        let dims = [
            (EvidenceDimension::FunctionalPerformance, "functional"),
            (EvidenceDimension::ThermodynamicStability, "stability"),
            (EvidenceDimension::CriticalMaterialBurden, "critical"),
            (EvidenceDimension::SupplyResilience, "supply"),
            (EvidenceDimension::HumanEnvironmentalHazard, "hazard"),
            (EvidenceDimension::Circularity, "circular"),
            (EvidenceDimension::Manufacturability, "manufacturing"),
        ];
        EnergyMaterialScreeningPolicy {
            policy_id: "candidate-version-test".into(),
            contracts: dims
                .into_iter()
                .map(|(dimension, metric)| MetricContract {
                    dimension,
                    metric: metric.into(),
                    unit: "score".into(),
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
                id: format!("fixture-{metric}"),
                kind: EvidenceKind::Dataset,
                uri: None,
                digest: Some("sha256:fixture".into()),
                note: None,
            }],
        }
    }

    fn receipt_sha(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn partial_dossier() -> EnergyMaterialDossier {
        let candidate_id = CandidateId::new("candidate-a").unwrap();
        let receipt = receipt_sha('a');
        assemble_dossier(
            candidate_id.clone(),
            &policy(),
            vec![IdentityAssertion {
                assertion_id: "internal-performance".into(),
                candidate_id: candidate_id.clone(),
                namespace: "symthaea".into(),
                subject_id: candidate_id.0.clone(),
                basis: IdentityAssertionBasis::InternalCandidate,
                source_receipt_sha256: receipt.clone(),
            }],
            vec![EvidenceContribution {
                dimension: EvidenceDimension::FunctionalPerformance,
                candidate_id,
                identity_assertion_id: "internal-performance".into(),
                source_receipt_sha256: receipt,
                prediction: prediction("functional"),
            }],
        )
        .unwrap()
    }

    fn attestation(candidate_sha: &str) -> ReceiptVersionAttestation {
        ReceiptVersionAttestation {
            attestation_id: "review-performance-v0".into(),
            dimension: EvidenceDimension::FunctionalPerformance,
            source_receipt_sha256: receipt_sha('a'),
            candidate_sha256: candidate_sha.into(),
            reviewer: "fixture-reviewer".into(),
            review_note: "Reviewed the source receipt against the exact formula specification.".into(),
        }
    }

    #[test]
    fn candidate_spec_change_changes_version_digest_even_when_id_is_same() {
        let first = candidate_sha256(&candidate("LiFePO4")).unwrap();
        let second = candidate_sha256(&candidate("NaFePO4")).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn candidate_map_insertion_order_does_not_change_digest() {
        let mut first = candidate("LiFePO4");
        first.specification.insert("phase".into(), "olivine".into());
        let mut second_spec = BTreeMap::new();
        second_spec.insert("phase".into(), "olivine".into());
        second_spec.insert("formula".into(), "LiFePO4".into());
        let second = Candidate {
            specification: second_spec,
            ..candidate("LiFePO4")
        };
        assert_eq!(candidate_sha256(&first).unwrap(), candidate_sha256(&second).unwrap());
    }

    #[test]
    fn contribution_requires_exactly_one_version_attestation() {
        let candidate = candidate("LiFePO4");
        let dossier = partial_dossier();
        assert!(matches!(
            bind_dossier_to_candidate_version(candidate, dossier, vec![]),
            Err(CandidateVersionError::MissingVersionAttestation(
                EvidenceDimension::FunctionalPerformance
            ))
        ));
    }

    #[test]
    fn wrong_candidate_version_digest_fails_closed() {
        let candidate = candidate("LiFePO4");
        let candidate_sha = candidate_sha256(&candidate("NaFePO4")).unwrap();
        let dossier = partial_dossier();
        assert!(matches!(
            bind_dossier_to_candidate_version(candidate, dossier, vec![attestation(&candidate_sha)]),
            Err(CandidateVersionError::AttestationCandidateMismatch { .. })
        ));
    }

    #[test]
    fn version_bound_dossier_round_trip_is_deterministic() {
        let candidate = candidate("LiFePO4");
        let candidate_sha = candidate_sha256(&candidate).unwrap();
        let first = bind_dossier_to_candidate_version(
            candidate.clone(),
            partial_dossier(),
            vec![attestation(&candidate_sha)],
        )
        .unwrap();
        let second = bind_dossier_to_candidate_version(
            candidate,
            partial_dossier(),
            vec![attestation(&candidate_sha)],
        )
        .unwrap();
        assert_eq!(first.sha256().unwrap(), second.sha256().unwrap());
        let encoded = serde_json::to_vec(&first).unwrap();
        let decoded: CandidateVersionBoundDossier = serde_json::from_slice(&encoded).unwrap();
        decoded.validate_integrity().unwrap();
    }

    #[test]
    fn blank_candidate_spec_values_fail_closed() {
        let mut invalid = candidate("LiFePO4");
        invalid.specification.insert("phase".into(), "   ".into());
        assert!(matches!(
            anchor_candidate(invalid),
            Err(CandidateVersionError::InvalidCandidate(_))
        ));
    }
}
