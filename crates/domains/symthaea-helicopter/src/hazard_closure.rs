// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Machine-evaluable hazard closure.
//!
//! This module does not certify an aircraft. It makes the evidence required to
//! close a declared hazard explicit and refuses to collapse missing evidence,
//! failed verification, and accepted residual risk into one boolean.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HazardSeverity {
    Minor,
    Major,
    Hazardous,
    Catastrophic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HazardVerificationStatus {
    Passed,
    Failed,
    Missing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HazardEvidenceKind {
    Analysis,
    Inspection,
    Simulation,
    HardwareInLoop,
    GroundTest,
    FlightTest,
    IndependentReview,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HazardVerificationEvidence {
    pub evidence_id: String,
    pub requirement_id: String,
    pub kind: HazardEvidenceKind,
    pub status: HazardVerificationStatus,
    pub independent: bool,
    pub artifact_digest: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResidualRiskAcceptance {
    pub accepted: bool,
    pub authority_id: Option<String>,
    pub decision_evidence_id: Option<String>,
    pub expires_at_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HazardClosureRecord {
    pub hazard_id: String,
    pub title: String,
    pub severity: HazardSeverity,
    pub safety_objective_ids: Vec<String>,
    pub mitigation_ids: Vec<String>,
    pub verification: Vec<HazardVerificationEvidence>,
    pub residual_risk: ResidualRiskAcceptance,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HazardClosurePolicy {
    pub minimum_mitigations: BTreeMap<HazardSeverity, usize>,
    pub require_independent_for: BTreeSet<HazardSeverity>,
    pub require_digest_for: BTreeSet<HazardEvidenceKind>,
    pub require_unexpired_acceptance: bool,
}

impl Default for HazardClosurePolicy {
    fn default() -> Self {
        Self {
            minimum_mitigations: BTreeMap::from([
                (HazardSeverity::Minor, 1),
                (HazardSeverity::Major, 1),
                (HazardSeverity::Hazardous, 2),
                (HazardSeverity::Catastrophic, 2),
            ]),
            require_independent_for: BTreeSet::from([
                HazardSeverity::Hazardous,
                HazardSeverity::Catastrophic,
            ]),
            require_digest_for: BTreeSet::from([
                HazardEvidenceKind::HardwareInLoop,
                HazardEvidenceKind::GroundTest,
                HazardEvidenceKind::FlightTest,
            ]),
            require_unexpired_acceptance: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HazardClosureStatus {
    Pass,
    Fail,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HazardClosureIssue {
    MissingRequiredHazard {
        hazard_id: String,
    },
    DuplicateHazard {
        hazard_id: String,
    },
    MissingSafetyObjective {
        hazard_id: String,
    },
    InsufficientMitigations {
        hazard_id: String,
        required: usize,
        observed: usize,
    },
    MissingVerification {
        hazard_id: String,
    },
    FailedVerification {
        hazard_id: String,
        evidence_id: String,
    },
    MissingVerificationArtifact {
        hazard_id: String,
        evidence_id: String,
    },
    MissingIndependentVerification {
        hazard_id: String,
    },
    ResidualRiskNotAccepted {
        hazard_id: String,
    },
    MissingAcceptanceAuthority {
        hazard_id: String,
    },
    MissingAcceptanceEvidence {
        hazard_id: String,
    },
    AcceptanceExpired {
        hazard_id: String,
        expires_at_ms: u64,
        now_ms: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HazardClosureAssessment {
    pub status: HazardClosureStatus,
    pub assessed_hazards: usize,
    pub closed_hazards: usize,
    pub issues: Vec<HazardClosureIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HazardClosureError {
    EmptyHazardId,
    EmptyRequiredHazardId,
}

pub struct HazardClosureGate {
    policy: HazardClosurePolicy,
}

impl HazardClosureGate {
    pub fn new(policy: HazardClosurePolicy) -> Self {
        Self { policy }
    }

    pub fn assess(
        &self,
        required_hazard_ids: &[String],
        records: &[HazardClosureRecord],
        now_ms: u64,
    ) -> Result<HazardClosureAssessment, HazardClosureError> {
        if required_hazard_ids.iter().any(|id| id.trim().is_empty()) {
            return Err(HazardClosureError::EmptyRequiredHazardId);
        }
        if records
            .iter()
            .any(|record| record.hazard_id.trim().is_empty())
        {
            return Err(HazardClosureError::EmptyHazardId);
        }

        let mut issues = Vec::new();
        let mut by_id = BTreeMap::<&str, Vec<&HazardClosureRecord>>::new();
        for record in records {
            by_id
                .entry(record.hazard_id.as_str())
                .or_default()
                .push(record);
        }

        for (hazard_id, matching) in &by_id {
            if matching.len() > 1 {
                issues.push(HazardClosureIssue::DuplicateHazard {
                    hazard_id: (*hazard_id).to_string(),
                });
            }
        }

        let required: BTreeSet<&str> = required_hazard_ids.iter().map(String::as_str).collect();
        for hazard_id in &required {
            if !by_id.contains_key(hazard_id) {
                issues.push(HazardClosureIssue::MissingRequiredHazard {
                    hazard_id: (*hazard_id).to_string(),
                });
            }
        }

        let mut closed_hazards = 0usize;
        for record in records {
            let before = issues.len();
            self.assess_record(record, now_ms, &mut issues);
            if issues.len() == before && by_id[record.hazard_id.as_str()].len() == 1 {
                closed_hazards += 1;
            }
        }

        issues.sort_by_key(issue_sort_key);
        let has_fail = issues.iter().any(is_failure);
        let status = if has_fail {
            HazardClosureStatus::Fail
        } else if issues.is_empty() {
            HazardClosureStatus::Pass
        } else {
            HazardClosureStatus::Incomplete
        };

        Ok(HazardClosureAssessment {
            status,
            assessed_hazards: records.len(),
            closed_hazards,
            issues,
        })
    }

    fn assess_record(
        &self,
        record: &HazardClosureRecord,
        now_ms: u64,
        issues: &mut Vec<HazardClosureIssue>,
    ) {
        if record.safety_objective_ids.is_empty() {
            issues.push(HazardClosureIssue::MissingSafetyObjective {
                hazard_id: record.hazard_id.clone(),
            });
        }

        let required_mitigations = self
            .policy
            .minimum_mitigations
            .get(&record.severity)
            .copied()
            .unwrap_or(1);
        let unique_mitigations = record
            .mitigation_ids
            .iter()
            .filter(|id| !id.trim().is_empty())
            .collect::<BTreeSet<_>>()
            .len();
        if unique_mitigations < required_mitigations {
            issues.push(HazardClosureIssue::InsufficientMitigations {
                hazard_id: record.hazard_id.clone(),
                required: required_mitigations,
                observed: unique_mitigations,
            });
        }

        if record.verification.is_empty() {
            issues.push(HazardClosureIssue::MissingVerification {
                hazard_id: record.hazard_id.clone(),
            });
        }
        for evidence in &record.verification {
            match evidence.status {
                HazardVerificationStatus::Failed => {
                    issues.push(HazardClosureIssue::FailedVerification {
                        hazard_id: record.hazard_id.clone(),
                        evidence_id: evidence.evidence_id.clone(),
                    });
                }
                HazardVerificationStatus::Missing => {
                    issues.push(HazardClosureIssue::MissingVerificationArtifact {
                        hazard_id: record.hazard_id.clone(),
                        evidence_id: evidence.evidence_id.clone(),
                    });
                }
                HazardVerificationStatus::Passed => {
                    if self.policy.require_digest_for.contains(&evidence.kind)
                        && evidence.artifact_digest.as_deref().unwrap_or("").is_empty()
                    {
                        issues.push(HazardClosureIssue::MissingVerificationArtifact {
                            hazard_id: record.hazard_id.clone(),
                            evidence_id: evidence.evidence_id.clone(),
                        });
                    }
                }
            }
        }

        if self
            .policy
            .require_independent_for
            .contains(&record.severity)
            && !record
                .verification
                .iter()
                .any(|e| e.independent && e.status == HazardVerificationStatus::Passed)
        {
            issues.push(HazardClosureIssue::MissingIndependentVerification {
                hazard_id: record.hazard_id.clone(),
            });
        }

        if !record.residual_risk.accepted {
            issues.push(HazardClosureIssue::ResidualRiskNotAccepted {
                hazard_id: record.hazard_id.clone(),
            });
        } else {
            if record
                .residual_risk
                .authority_id
                .as_deref()
                .unwrap_or("")
                .is_empty()
            {
                issues.push(HazardClosureIssue::MissingAcceptanceAuthority {
                    hazard_id: record.hazard_id.clone(),
                });
            }
            if record
                .residual_risk
                .decision_evidence_id
                .as_deref()
                .unwrap_or("")
                .is_empty()
            {
                issues.push(HazardClosureIssue::MissingAcceptanceEvidence {
                    hazard_id: record.hazard_id.clone(),
                });
            }
            if self.policy.require_unexpired_acceptance {
                if let Some(expires_at_ms) = record.residual_risk.expires_at_ms {
                    if expires_at_ms < now_ms {
                        issues.push(HazardClosureIssue::AcceptanceExpired {
                            hazard_id: record.hazard_id.clone(),
                            expires_at_ms,
                            now_ms,
                        });
                    }
                }
            }
        }
    }
}

fn is_failure(issue: &HazardClosureIssue) -> bool {
    matches!(
        issue,
        HazardClosureIssue::DuplicateHazard { .. }
            | HazardClosureIssue::InsufficientMitigations { .. }
            | HazardClosureIssue::FailedVerification { .. }
            | HazardClosureIssue::ResidualRiskNotAccepted { .. }
            | HazardClosureIssue::AcceptanceExpired { .. }
    )
}

fn issue_sort_key(issue: &HazardClosureIssue) -> (String, u8, String) {
    match issue {
        HazardClosureIssue::MissingRequiredHazard { hazard_id } => {
            (hazard_id.clone(), 0, String::new())
        }
        HazardClosureIssue::DuplicateHazard { hazard_id } => (hazard_id.clone(), 1, String::new()),
        HazardClosureIssue::MissingSafetyObjective { hazard_id } => {
            (hazard_id.clone(), 2, String::new())
        }
        HazardClosureIssue::InsufficientMitigations { hazard_id, .. } => {
            (hazard_id.clone(), 3, String::new())
        }
        HazardClosureIssue::MissingVerification { hazard_id } => {
            (hazard_id.clone(), 4, String::new())
        }
        HazardClosureIssue::FailedVerification {
            hazard_id,
            evidence_id,
        } => (hazard_id.clone(), 5, evidence_id.clone()),
        HazardClosureIssue::MissingVerificationArtifact {
            hazard_id,
            evidence_id,
        } => (hazard_id.clone(), 6, evidence_id.clone()),
        HazardClosureIssue::MissingIndependentVerification { hazard_id } => {
            (hazard_id.clone(), 7, String::new())
        }
        HazardClosureIssue::ResidualRiskNotAccepted { hazard_id } => {
            (hazard_id.clone(), 8, String::new())
        }
        HazardClosureIssue::MissingAcceptanceAuthority { hazard_id } => {
            (hazard_id.clone(), 9, String::new())
        }
        HazardClosureIssue::MissingAcceptanceEvidence { hazard_id } => {
            (hazard_id.clone(), 10, String::new())
        }
        HazardClosureIssue::AcceptanceExpired { hazard_id, .. } => {
            (hazard_id.clone(), 11, String::new())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn closed_record() -> HazardClosureRecord {
        HazardClosureRecord {
            hazard_id: "HZ-001".into(),
            title: "Loss of lift".into(),
            severity: HazardSeverity::Catastrophic,
            safety_objective_ids: vec!["SO-001".into()],
            mitigation_ids: vec!["MIT-001".into(), "MIT-002".into()],
            verification: vec![HazardVerificationEvidence {
                evidence_id: "EV-001".into(),
                requirement_id: "REQ-001".into(),
                kind: HazardEvidenceKind::FlightTest,
                status: HazardVerificationStatus::Passed,
                independent: true,
                artifact_digest: Some("sha256:abc".into()),
            }],
            residual_risk: ResidualRiskAcceptance {
                accepted: true,
                authority_id: Some("chief-engineer".into()),
                decision_evidence_id: Some("RISK-001".into()),
                expires_at_ms: Some(2_000),
            },
        }
    }

    #[test]
    fn closes_complete_catastrophic_hazard() {
        let gate = HazardClosureGate::new(HazardClosurePolicy::default());
        let result = gate
            .assess(&["HZ-001".into()], &[closed_record()], 1_000)
            .unwrap();
        assert_eq!(result.status, HazardClosureStatus::Pass);
        assert_eq!(result.closed_hazards, 1);
    }

    #[test]
    fn failed_verification_is_failure() {
        let mut record = closed_record();
        record.verification[0].status = HazardVerificationStatus::Failed;
        let gate = HazardClosureGate::new(HazardClosurePolicy::default());
        let result = gate.assess(&["HZ-001".into()], &[record], 1_000).unwrap();
        assert_eq!(result.status, HazardClosureStatus::Fail);
        assert!(
            result
                .issues
                .iter()
                .any(|issue| matches!(issue, HazardClosureIssue::FailedVerification { .. }))
        );
    }

    #[test]
    fn missing_independent_evidence_is_incomplete() {
        let mut record = closed_record();
        record.verification[0].independent = false;
        let gate = HazardClosureGate::new(HazardClosurePolicy::default());
        let result = gate.assess(&["HZ-001".into()], &[record], 1_000).unwrap();
        assert_eq!(result.status, HazardClosureStatus::Incomplete);
    }

    #[test]
    fn expired_acceptance_fails() {
        let gate = HazardClosureGate::new(HazardClosurePolicy::default());
        let result = gate
            .assess(&["HZ-001".into()], &[closed_record()], 3_000)
            .unwrap();
        assert_eq!(result.status, HazardClosureStatus::Fail);
    }
}
