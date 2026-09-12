// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Common-cause diversity assurance for safety-evidence verifiers.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use symthaea_formal_safety::{
    SafetyCase, SafetyEvidenceReceipt, StrictSafetyCaseReport, StrictSafetyCaseStatus,
    assess_strict_safety_case,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierFaultDomainProfile {
    pub verifier_ref: String,
    pub organization_domain: String,
    pub review_process_domain: String,
    pub toolchain_domain: String,
    pub evidence_source_domain: String,
    pub evidence_refs: Vec<String>,
}

impl VerifierFaultDomainProfile {
    pub fn validate(&self) -> bool {
        !self.verifier_ref.trim().is_empty()
            && !self.organization_domain.trim().is_empty()
            && !self.review_process_domain.trim().is_empty()
            && !self.toolchain_domain.trim().is_empty()
            && !self.evidence_source_domain.trim().is_empty()
            && !self.evidence_refs.is_empty()
            && self.evidence_refs.iter().all(|value| !value.trim().is_empty())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierDiversityRequirement {
    pub obligation_key: String,
    pub minimum_distinct_verifiers: usize,
    pub minimum_organization_domains: usize,
    pub minimum_review_process_domains: usize,
    pub minimum_toolchain_domains: usize,
    pub minimum_evidence_source_domains: usize,
    pub evidence_refs: Vec<String>,
}

impl VerifierDiversityRequirement {
    pub fn validate(&self) -> bool {
        !self.obligation_key.trim().is_empty()
            && self.minimum_distinct_verifiers >= 1
            && self.minimum_organization_domains >= 1
            && self.minimum_review_process_domains >= 1
            && self.minimum_toolchain_domains >= 1
            && self.minimum_evidence_source_domains >= 1
            && !self.evidence_refs.is_empty()
            && self.evidence_refs.iter().all(|value| !value.trim().is_empty())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierDiversityPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub requirements: Vec<VerifierDiversityRequirement>,
    pub evidence_refs: Vec<String>,
}

impl VerifierDiversityPolicy {
    pub fn validate(&self) -> bool {
        !self.schema_version.trim().is_empty()
            && !self.policy_id.trim().is_empty()
            && !self.requirements.is_empty()
            && self.requirements.iter().all(VerifierDiversityRequirement::validate)
            && !self.evidence_refs.is_empty()
            && self.evidence_refs.iter().all(|value| !value.trim().is_empty())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerifierDiversityIssue {
    InvalidPolicy,
    DuplicateRequirement(String),
    UnknownObligation(String),
    InvalidVerifierProfile(String),
    DuplicateVerifierProfile(String),
    MissingVerifierProfile {
        receipt_id: String,
        verifier_ref: String,
    },
    InsufficientDistinctVerifiers {
        obligation_key: String,
        observed: usize,
        required: usize,
    },
    InsufficientOrganizationDomains {
        obligation_key: String,
        observed: usize,
        required: usize,
    },
    InsufficientReviewProcessDomains {
        obligation_key: String,
        observed: usize,
        required: usize,
    },
    InsufficientToolchainDomains {
        obligation_key: String,
        observed: usize,
        required: usize,
    },
    InsufficientEvidenceSourceDomains {
        obligation_key: String,
        observed: usize,
        required: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObligationVerifierDiversityReport {
    pub obligation_key: String,
    pub receipt_count: usize,
    pub distinct_verifiers: usize,
    pub organization_domains: usize,
    pub review_process_domains: usize,
    pub toolchain_domains: usize,
    pub evidence_source_domains: usize,
    pub satisfied: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierDiversityReport {
    pub policy_id: String,
    pub status: StrictSafetyCaseStatus,
    pub issues: Vec<VerifierDiversityIssue>,
    pub obligation_reports: Vec<ObligationVerifierDiversityReport>,
    pub strict_report: StrictSafetyCaseReport,
}

impl VerifierDiversityReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

/// Apply reviewed verifier-diversity requirements on top of ordinary strict readiness.
///
/// This gate is monotonic: it cannot upgrade a `Blocked` or `Invalid` strict case.
pub fn assess_verifier_diversity(
    safety_case: &SafetyCase,
    receipts: &[SafetyEvidenceReceipt],
    profiles: &[VerifierFaultDomainProfile],
    policy: &VerifierDiversityPolicy,
) -> VerifierDiversityReport {
    let strict_report = assess_strict_safety_case(safety_case, receipts);
    let mut issues = Vec::new();
    let mut obligation_reports = Vec::new();

    if !policy.validate() {
        issues.push(VerifierDiversityIssue::InvalidPolicy);
    }

    let obligation_keys = safety_case
        .obligations
        .iter()
        .map(|obligation| obligation.stable_key())
        .collect::<BTreeSet<_>>();

    let mut requirements = BTreeMap::<String, &VerifierDiversityRequirement>::new();
    for requirement in &policy.requirements {
        if !requirement.validate() {
            continue;
        }
        if !obligation_keys.contains(&requirement.obligation_key) {
            issues.push(VerifierDiversityIssue::UnknownObligation(
                requirement.obligation_key.clone(),
            ));
            continue;
        }
        if requirements
            .insert(requirement.obligation_key.clone(), requirement)
            .is_some()
        {
            issues.push(VerifierDiversityIssue::DuplicateRequirement(
                requirement.obligation_key.clone(),
            ));
        }
    }

    let mut profile_map = BTreeMap::<String, &VerifierFaultDomainProfile>::new();
    for profile in profiles {
        if !profile.validate() {
            issues.push(VerifierDiversityIssue::InvalidVerifierProfile(
                profile.verifier_ref.clone(),
            ));
            continue;
        }
        if profile_map
            .insert(profile.verifier_ref.clone(), profile)
            .is_some()
        {
            issues.push(VerifierDiversityIssue::DuplicateVerifierProfile(
                profile.verifier_ref.clone(),
            ));
        }
    }

    for (obligation_key, requirement) in requirements {
        let matching = receipts
            .iter()
            .filter(|receipt| receipt.validate() && receipt.obligation_key == obligation_key)
            .collect::<Vec<_>>();

        let mut verifier_refs = BTreeSet::new();
        let mut organization_domains = BTreeSet::new();
        let mut review_process_domains = BTreeSet::new();
        let mut toolchain_domains = BTreeSet::new();
        let mut evidence_source_domains = BTreeSet::new();

        for receipt in &matching {
            let Some(profile) = profile_map.get(&receipt.verifier_ref) else {
                issues.push(VerifierDiversityIssue::MissingVerifierProfile {
                    receipt_id: receipt.receipt_id.clone(),
                    verifier_ref: receipt.verifier_ref.clone(),
                });
                continue;
            };
            verifier_refs.insert(profile.verifier_ref.clone());
            organization_domains.insert(profile.organization_domain.clone());
            review_process_domains.insert(profile.review_process_domain.clone());
            toolchain_domains.insert(profile.toolchain_domain.clone());
            evidence_source_domains.insert(profile.evidence_source_domain.clone());
        }

        let mut satisfied = true;
        let checks = [
            (
                verifier_refs.len(),
                requirement.minimum_distinct_verifiers,
                0usize,
            ),
            (
                organization_domains.len(),
                requirement.minimum_organization_domains,
                1usize,
            ),
            (
                review_process_domains.len(),
                requirement.minimum_review_process_domains,
                2usize,
            ),
            (
                toolchain_domains.len(),
                requirement.minimum_toolchain_domains,
                3usize,
            ),
            (
                evidence_source_domains.len(),
                requirement.minimum_evidence_source_domains,
                4usize,
            ),
        ];
        for (observed, required, kind) in checks {
            if observed >= required {
                continue;
            }
            satisfied = false;
            let issue = match kind {
                0 => VerifierDiversityIssue::InsufficientDistinctVerifiers {
                    obligation_key: obligation_key.clone(),
                    observed,
                    required,
                },
                1 => VerifierDiversityIssue::InsufficientOrganizationDomains {
                    obligation_key: obligation_key.clone(),
                    observed,
                    required,
                },
                2 => VerifierDiversityIssue::InsufficientReviewProcessDomains {
                    obligation_key: obligation_key.clone(),
                    observed,
                    required,
                },
                3 => VerifierDiversityIssue::InsufficientToolchainDomains {
                    obligation_key: obligation_key.clone(),
                    observed,
                    required,
                },
                _ => VerifierDiversityIssue::InsufficientEvidenceSourceDomains {
                    obligation_key: obligation_key.clone(),
                    observed,
                    required,
                },
            };
            issues.push(issue);
        }

        obligation_reports.push(ObligationVerifierDiversityReport {
            obligation_key,
            receipt_count: matching.len(),
            distinct_verifiers: verifier_refs.len(),
            organization_domains: organization_domains.len(),
            review_process_domains: review_process_domains.len(),
            toolchain_domains: toolchain_domains.len(),
            evidence_source_domains: evidence_source_domains.len(),
            satisfied,
        });
    }

    obligation_reports.sort_by(|a, b| a.obligation_key.cmp(&b.obligation_key));

    let structurally_invalid = issues.iter().any(|issue| {
        matches!(
            issue,
            VerifierDiversityIssue::InvalidPolicy
                | VerifierDiversityIssue::DuplicateRequirement(_)
                | VerifierDiversityIssue::UnknownObligation(_)
                | VerifierDiversityIssue::InvalidVerifierProfile(_)
                | VerifierDiversityIssue::DuplicateVerifierProfile(_)
        )
    });
    let diversity_blocked = issues.iter().any(|issue| {
        matches!(
            issue,
            VerifierDiversityIssue::MissingVerifierProfile { .. }
                | VerifierDiversityIssue::InsufficientDistinctVerifiers { .. }
                | VerifierDiversityIssue::InsufficientOrganizationDomains { .. }
                | VerifierDiversityIssue::InsufficientReviewProcessDomains { .. }
                | VerifierDiversityIssue::InsufficientToolchainDomains { .. }
                | VerifierDiversityIssue::InsufficientEvidenceSourceDomains { .. }
        )
    });

    let status = if structurally_invalid || strict_report.status == StrictSafetyCaseStatus::Invalid {
        StrictSafetyCaseStatus::Invalid
    } else if strict_report.status != StrictSafetyCaseStatus::Ready || diversity_blocked {
        StrictSafetyCaseStatus::Blocked
    } else {
        StrictSafetyCaseStatus::Ready
    };

    VerifierDiversityReport {
        policy_id: policy.policy_id.clone(),
        status,
        issues,
        obligation_reports,
        strict_report,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_formal_safety::{EvidenceKind, ProofObligation};

    fn case() -> SafetyCase {
        let mut case = SafetyCase::new("subject");
        case.add_obligation(ProofObligation::new("claim", EvidenceKind::Test).discharge("test:x"));
        case
    }

    fn receipt(case: &SafetyCase, id: &str, verifier: &str) -> SafetyEvidenceReceipt {
        let obligation = &case.obligations[0];
        SafetyEvidenceReceipt {
            receipt_id: id.into(),
            obligation_key: obligation.stable_key(),
            evidence_kind: obligation.expected_evidence,
            evidence_ref: format!("artifact:{id}"),
            evidence_digest: format!("blake3:{id}"),
            verifier_ref: verifier.into(),
            verified_at_ms: 100,
        }
    }

    fn profile(verifier: &str, org: &str, process: &str, tool: &str, source: &str) -> VerifierFaultDomainProfile {
        VerifierFaultDomainProfile {
            verifier_ref: verifier.into(),
            organization_domain: org.into(),
            review_process_domain: process.into(),
            toolchain_domain: tool.into(),
            evidence_source_domain: source.into(),
            evidence_refs: vec![format!("profile:{verifier}")],
        }
    }

    fn policy(case: &SafetyCase, minimum: usize) -> VerifierDiversityPolicy {
        VerifierDiversityPolicy {
            schema_version: "1".into(),
            policy_id: "diversity-v1".into(),
            requirements: vec![VerifierDiversityRequirement {
                obligation_key: case.obligations[0].stable_key(),
                minimum_distinct_verifiers: minimum,
                minimum_organization_domains: minimum,
                minimum_review_process_domains: minimum,
                minimum_toolchain_domains: minimum,
                minimum_evidence_source_domains: minimum,
                evidence_refs: vec!["review:diversity-policy".into()],
            }],
            evidence_refs: vec!["policy:diversity-v1".into()],
        }
    }

    #[test]
    fn one_profile_can_satisfy_one_verifier_policy() {
        let case = case();
        let receipts = vec![receipt(&case, "r1", "verifier:a")];
        let profiles = vec![profile("verifier:a", "org:a", "process:a", "tool:a", "source:a")];
        let report = assess_verifier_diversity(&case, &receipts, &profiles, &policy(&case, 1));
        assert_eq!(report.status, StrictSafetyCaseStatus::Ready);
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn multiple_receipts_from_one_verifier_do_not_create_diversity() {
        let case = case();
        let receipts = vec![
            receipt(&case, "r1", "verifier:a"),
            receipt(&case, "r2", "verifier:a"),
        ];
        let profiles = vec![profile("verifier:a", "org:a", "process:a", "tool:a", "source:a")];
        let report = assess_verifier_diversity(&case, &receipts, &profiles, &policy(&case, 2));
        assert_eq!(report.status, StrictSafetyCaseStatus::Blocked);
        assert_eq!(report.obligation_reports[0].distinct_verifiers, 1);
    }

    #[test]
    fn distinct_verifier_ids_in_one_organization_do_not_create_org_diversity() {
        let case = case();
        let receipts = vec![
            receipt(&case, "r1", "verifier:a"),
            receipt(&case, "r2", "verifier:b"),
        ];
        let profiles = vec![
            profile("verifier:a", "org:shared", "process:a", "tool:a", "source:a"),
            profile("verifier:b", "org:shared", "process:b", "tool:b", "source:b"),
        ];
        let report = assess_verifier_diversity(&case, &receipts, &profiles, &policy(&case, 2));
        assert_eq!(report.status, StrictSafetyCaseStatus::Blocked);
        assert_eq!(report.obligation_reports[0].distinct_verifiers, 2);
        assert_eq!(report.obligation_reports[0].organization_domains, 1);
    }

    #[test]
    fn fully_diverse_verifiers_can_satisfy_policy() {
        let case = case();
        let receipts = vec![
            receipt(&case, "r1", "verifier:a"),
            receipt(&case, "r2", "verifier:b"),
        ];
        let profiles = vec![
            profile("verifier:a", "org:a", "process:a", "tool:a", "source:a"),
            profile("verifier:b", "org:b", "process:b", "tool:b", "source:b"),
        ];
        let report = assess_verifier_diversity(&case, &receipts, &profiles, &policy(&case, 2));
        assert_eq!(report.status, StrictSafetyCaseStatus::Ready);
    }

    #[test]
    fn missing_verifier_profile_blocks_readiness() {
        let case = case();
        let receipts = vec![receipt(&case, "r1", "verifier:a")];
        let report = assess_verifier_diversity(&case, &receipts, &[], &policy(&case, 1));
        assert_eq!(report.status, StrictSafetyCaseStatus::Blocked);
    }

    #[test]
    fn duplicate_profiles_are_invalid() {
        let case = case();
        let receipts = vec![receipt(&case, "r1", "verifier:a")];
        let p = profile("verifier:a", "org:a", "process:a", "tool:a", "source:a");
        let report = assess_verifier_diversity(&case, &receipts, &[p.clone(), p], &policy(&case, 1));
        assert_eq!(report.status, StrictSafetyCaseStatus::Invalid);
    }

    #[test]
    fn diversity_gate_cannot_upgrade_blocked_strict_case() {
        let mut case = case();
        case.obligations[0].status = symthaea_formal_safety::ObligationStatus::Open;
        let receipts = vec![receipt(&case, "r1", "verifier:a")];
        let profiles = vec![profile("verifier:a", "org:a", "process:a", "tool:a", "source:a")];
        let report = assess_verifier_diversity(&case, &receipts, &profiles, &policy(&case, 1));
        assert_eq!(report.status, StrictSafetyCaseStatus::Blocked);
    }
}
