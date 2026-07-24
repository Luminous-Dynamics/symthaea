// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Machine-readable assurance and claim-scope ledger.
//!
//! Passing simulator tests does not imply hardware readiness, flight safety, or
//! certification. Claims declare a maximum assurance level and the evidence
//! categories required at that level. Missing evidence is incomplete; a claim
//! above its declared ceiling is refused even when unrelated evidence exists.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AssuranceLevel {
    ResearchPrototype,
    DeterministicSimulation,
    SoftwareInTheLoop,
    HardwareInTheLoop,
    GroundTest,
    FlightTest,
    RegulatoryApproved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ClaimEvidenceKind {
    SourceReview,
    StaticAnalysis,
    AutomatedTests,
    ScenarioCampaign,
    DeterministicReplay,
    CryptographicSignature,
    TraceableCalibration,
    RealtimeTiming,
    HardwareInTheLoopRun,
    GroundTestRun,
    FlightTestRun,
    IndependentSafetyReview,
    RegulatoryApproval,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimEvidenceRequirement {
    pub kind: ClaimEvidenceKind,
    pub minimum_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssuranceClaim {
    pub claim_id: String,
    pub statement: String,
    pub maximum_level: AssuranceLevel,
    pub requirements: Vec<ClaimEvidenceRequirement>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimEvidenceArtifact {
    pub artifact_id: String,
    pub kind: ClaimEvidenceKind,
    pub verified: bool,
    pub digest: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClaimAssessmentStatus {
    Supported,
    Incomplete,
    Refused,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimAssessment {
    pub claim_id: String,
    pub requested_level: AssuranceLevel,
    pub status: ClaimAssessmentStatus,
    pub maximum_level: AssuranceLevel,
    pub missing_evidence: Vec<ClaimEvidenceRequirement>,
    pub unverified_artifact_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimLedgerError {
    InvalidClaim,
    DuplicateClaim,
    UnknownClaim,
    InvalidEvidence,
    SerializationFailed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimLedger {
    pub schema_version: String,
    pub claims: Vec<AssuranceClaim>,
}

impl ClaimLedger {
    pub fn new(
        schema_version: impl Into<String>,
        claims: Vec<AssuranceClaim>,
    ) -> Result<Self, ClaimLedgerError> {
        let ledger = Self {
            schema_version: schema_version.into(),
            claims,
        };
        ledger.validate()?;
        Ok(ledger)
    }

    pub fn validate(&self) -> Result<(), ClaimLedgerError> {
        if self.schema_version.trim().is_empty() || self.claims.is_empty() {
            return Err(ClaimLedgerError::InvalidClaim);
        }
        for (index, claim) in self.claims.iter().enumerate() {
            if claim.claim_id.trim().is_empty()
                || claim.statement.trim().is_empty()
                || claim.requirements.is_empty()
                || claim
                    .requirements
                    .iter()
                    .any(|requirement| requirement.minimum_count == 0)
            {
                return Err(ClaimLedgerError::InvalidClaim);
            }
            if self.claims[..index]
                .iter()
                .any(|previous| previous.claim_id == claim.claim_id)
            {
                return Err(ClaimLedgerError::DuplicateClaim);
            }
            for (requirement_index, requirement) in claim.requirements.iter().enumerate() {
                if claim.requirements[..requirement_index]
                    .iter()
                    .any(|previous| previous.kind == requirement.kind)
                {
                    return Err(ClaimLedgerError::InvalidClaim);
                }
            }
            if claim.maximum_level == AssuranceLevel::RegulatoryApproved
                && !claim
                    .requirements
                    .iter()
                    .any(|requirement| requirement.kind == ClaimEvidenceKind::RegulatoryApproval)
            {
                return Err(ClaimLedgerError::InvalidClaim);
            }
        }
        Ok(())
    }

    pub fn assess(
        &self,
        claim_id: &str,
        requested_level: AssuranceLevel,
        artifacts: &[ClaimEvidenceArtifact],
    ) -> Result<ClaimAssessment, ClaimLedgerError> {
        self.validate()?;
        if artifacts.iter().any(|artifact| {
            artifact.artifact_id.trim().is_empty()
                || artifact
                    .digest
                    .as_ref()
                    .is_some_and(|digest| digest.trim().is_empty())
        }) {
            return Err(ClaimLedgerError::InvalidEvidence);
        }
        let claim = self
            .claims
            .iter()
            .find(|claim| claim.claim_id == claim_id)
            .ok_or(ClaimLedgerError::UnknownClaim)?;
        if requested_level > claim.maximum_level {
            return Ok(ClaimAssessment {
                claim_id: claim.claim_id.clone(),
                requested_level,
                status: ClaimAssessmentStatus::Refused,
                maximum_level: claim.maximum_level,
                missing_evidence: claim.requirements.clone(),
                unverified_artifact_ids: Vec::new(),
            });
        }

        let unverified_artifact_ids = artifacts
            .iter()
            .filter(|artifact| !artifact.verified)
            .map(|artifact| artifact.artifact_id.clone())
            .collect::<Vec<_>>();
        let missing_evidence = claim
            .requirements
            .iter()
            .filter(|requirement| {
                artifacts
                    .iter()
                    .filter(|artifact| artifact.verified && artifact.kind == requirement.kind)
                    .count()
                    < requirement.minimum_count
            })
            .cloned()
            .collect::<Vec<_>>();
        Ok(ClaimAssessment {
            claim_id: claim.claim_id.clone(),
            requested_level,
            status: if missing_evidence.is_empty() {
                ClaimAssessmentStatus::Supported
            } else {
                ClaimAssessmentStatus::Incomplete
            },
            maximum_level: claim.maximum_level,
            missing_evidence,
            unverified_artifact_ids,
        })
    }

    pub fn helicopter_default() -> Self {
        Self::new(
            "symthaea-helicopter-claim-ledger-v1",
            vec![
                AssuranceClaim {
                    claim_id: "reduced-order-simulator".to_string(),
                    statement: "The reduced-order simulator is deterministic for a bound scenario, seed, and configuration.".to_string(),
                    maximum_level: AssuranceLevel::SoftwareInTheLoop,
                    requirements: vec![
                        requirement(ClaimEvidenceKind::AutomatedTests, 1),
                        requirement(ClaimEvidenceKind::ScenarioCampaign, 1),
                        requirement(ClaimEvidenceKind::DeterministicReplay, 1),
                    ],
                },
                AssuranceClaim {
                    claim_id: "physical-control-boundary".to_string(),
                    statement: "The fail-closed hardware boundary met its declared HIL timing and authority contracts.".to_string(),
                    maximum_level: AssuranceLevel::HardwareInTheLoop,
                    requirements: vec![
                        requirement(ClaimEvidenceKind::HardwareInTheLoopRun, 1),
                        requirement(ClaimEvidenceKind::RealtimeTiming, 1),
                        requirement(ClaimEvidenceKind::CryptographicSignature, 1),
                    ],
                },
                AssuranceClaim {
                    claim_id: "named-airframe-model".to_string(),
                    statement: "The model parameters are traceable to a named research airframe within declared uncertainty.".to_string(),
                    maximum_level: AssuranceLevel::GroundTest,
                    requirements: vec![
                        requirement(ClaimEvidenceKind::TraceableCalibration, 1),
                        requirement(ClaimEvidenceKind::GroundTestRun, 1),
                        requirement(ClaimEvidenceKind::IndependentSafetyReview, 1),
                    ],
                },
                AssuranceClaim {
                    claim_id: "airworthy-or-certified".to_string(),
                    statement: "The aircraft is airworthy or approved for regulated operation.".to_string(),
                    maximum_level: AssuranceLevel::RegulatoryApproved,
                    requirements: vec![
                        requirement(ClaimEvidenceKind::FlightTestRun, 1),
                        requirement(ClaimEvidenceKind::IndependentSafetyReview, 1),
                        requirement(ClaimEvidenceKind::RegulatoryApproval, 1),
                    ],
                },
            ],
        )
        .expect("default helicopter claim ledger is internally valid")
    }

    pub fn canonical_json(&self) -> Result<Vec<u8>, ClaimLedgerError> {
        self.validate()?;
        let mut canonical = self.clone();
        canonical
            .claims
            .sort_by(|left, right| left.claim_id.cmp(&right.claim_id));
        serde_json::to_vec(&canonical).map_err(|_| ClaimLedgerError::SerializationFailed)
    }
}

fn requirement(kind: ClaimEvidenceKind, minimum_count: usize) -> ClaimEvidenceRequirement {
    ClaimEvidenceRequirement {
        kind,
        minimum_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact(id: &str, kind: ClaimEvidenceKind) -> ClaimEvidenceArtifact {
        ClaimEvidenceArtifact {
            artifact_id: id.to_string(),
            kind,
            verified: true,
            digest: Some(format!("digest:{id}")),
        }
    }

    #[test]
    fn simulator_claim_cannot_be_promoted_to_hardware_level() {
        let ledger = ClaimLedger::helicopter_default();
        let assessment = ledger
            .assess(
                "reduced-order-simulator",
                AssuranceLevel::HardwareInTheLoop,
                &[],
            )
            .unwrap();
        assert_eq!(assessment.status, ClaimAssessmentStatus::Refused);
    }

    #[test]
    fn complete_hil_evidence_supports_only_hil_claim() {
        let ledger = ClaimLedger::helicopter_default();
        let evidence = vec![
            artifact("hil-run", ClaimEvidenceKind::HardwareInTheLoopRun),
            artifact("timing", ClaimEvidenceKind::RealtimeTiming),
            artifact("signature", ClaimEvidenceKind::CryptographicSignature),
        ];
        let assessment = ledger
            .assess(
                "physical-control-boundary",
                AssuranceLevel::HardwareInTheLoop,
                &evidence,
            )
            .unwrap();
        assert_eq!(assessment.status, ClaimAssessmentStatus::Supported);
    }

    #[test]
    fn certification_claim_without_regulatory_artifact_is_incomplete() {
        let ledger = ClaimLedger::helicopter_default();
        let evidence = vec![
            artifact("flight", ClaimEvidenceKind::FlightTestRun),
            artifact("review", ClaimEvidenceKind::IndependentSafetyReview),
        ];
        let assessment = ledger
            .assess(
                "airworthy-or-certified",
                AssuranceLevel::RegulatoryApproved,
                &evidence,
            )
            .unwrap();
        assert_eq!(assessment.status, ClaimAssessmentStatus::Incomplete);
        assert!(
            assessment
                .missing_evidence
                .iter()
                .any(|requirement| requirement.kind == ClaimEvidenceKind::RegulatoryApproval)
        );
    }
}
