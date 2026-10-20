//! Explicit-cost, evidence-bounded multi-fidelity evaluation ledger.
//!
//! This is a campaign ledger and promotion gate, not an acquisition function.
//! It does not rank designs, infer that a fidelity class is universally more
//! trustworthy, estimate information gain, or validate the underlying physics.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EvaluationClass {
    Analytical,
    LowFidelitySimulation,
    HighFidelitySimulation,
    PhysicalTest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EvidenceOrigin {
    SyntheticFixture,
    AnalyticalModel,
    CfdSimulation,
    ExperimentalMeasurement,
    IndependentPhysicalValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EvidenceUseStatus {
    PermittedForDeclaredUse,
    Restricted,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AttemptStatus {
    Complete,
    Failed,
    Censored,
    OutOfDomain,
    InvalidResult,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FidelityRequirement {
    pub requirement_id: String,
    pub candidate_id: String,
    /// Frozen SHA-256 identity for the exact geometry/configuration.
    pub configuration_sha256: String,
    pub operating_case_id: String,
    /// These are exact allowed class/origin identities, not an implicit universal ranking.
    pub evaluation_class: EvaluationClass,
    pub evidence_origin: EvidenceOrigin,
    pub method_id: String,
    pub method_revision: String,
    /// Zero permits/records this evaluation type without making it a release gate.
    pub required_successes: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FidelityProfile {
    pub profile_id: String,
    pub campaign_id: String,
    /// Digest of the frozen profile and all its requirements.
    pub manifest_sha256: String,
    /// All evaluation costs use these declared integer budget units.
    pub cost_basis_id: String,
    pub requirements: Vec<FidelityRequirement>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluationReceipt {
    /// Unique identity for one attempt; retries require new receipt IDs.
    pub receipt_id: String,
    pub campaign_id: String,
    pub requirement_id: String,
    pub candidate_id: String,
    pub configuration_sha256: String,
    pub operating_case_id: String,
    pub evaluation_class: EvaluationClass,
    pub evidence_origin: EvidenceOrigin,
    pub method_id: String,
    pub method_revision: String,
    /// Digest/identifier of the input data, reference corpus or test protocol.
    pub source_id: String,
    pub source_reference_sha256: String,
    pub evidence_use_status: EvidenceUseStatus,
    pub cost_basis_id: String,
    /// Missing cost is not zero; zero is rejected to block fabricated free information.
    pub cost_units: Option<u64>,
    pub status: AttemptStatus,
    /// Required for a complete result; optional for failed or censored attempts.
    pub output_sha256: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FidelityAuditDisposition {
    EvidenceCoverageComplete,
    Incomplete,
    InvalidProfile,
    InvalidReceipts,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FidelityIssue {
    EmptyProfileIdentity(&'static str),
    InvalidManifestDigest,
    EmptyRequirements,
    NoRequiredReleaseGate,
    DuplicateRequirementId { requirement_id: String },
    CandidateConfigurationMismatch {
        candidate_id: String,
        first_sha256: String,
        observed_sha256: String,
    },
    InvalidRequirement { requirement_id: String, field: &'static str },
    MissingRequirementEvidence {
        requirement_id: String,
        required_successes: u32,
        completed_successes: u32,
    },
    EmptyReceiptIdentity { receipt_id: String, field: &'static str },
    InvalidReceiptDigest { receipt_id: String, field: &'static str },
    EvidenceRightsNotPermitted {
        receipt_id: String,
        status: EvidenceUseStatus,
    },
    MissingOutputDigest { receipt_id: String },
    DuplicateReceiptId { receipt_id: String, count: usize },
    MissingCost { receipt_id: String },
    ZeroCost { receipt_id: String },
    CostBasisMismatch { receipt_id: String },
    CampaignMismatch { receipt_id: String },
    UnregisteredRequirement {
        receipt_id: String,
        requirement_id: String,
    },
    ReceiptRequirementMismatch {
        receipt_id: String,
        requirement_id: String,
        field: &'static str,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequirementProgress {
    pub requirement_id: String,
    pub candidate_id: String,
    pub operating_case_id: String,
    pub evaluation_class: EvaluationClass,
    pub evidence_origin: EvidenceOrigin,
    pub required_successes: u32,
    pub completed_successes: u32,
    pub failed_attempts: u32,
    pub censored_attempts: u32,
    pub out_of_domain_attempts: u32,
    pub invalid_result_attempts: u32,
    pub total_attempts: u32,
    pub covered: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FidelityAudit {
    pub profile_id: String,
    pub campaign_id: String,
    /// Declared frozen profile identity copied into the audit receipt. Equality is checked, but
    /// the caller-supplied digest is not cryptographically tied to profile fields by this module.
    pub profile_manifest_sha256: String,
    pub cost_basis_id: String,
    pub disposition: FidelityAuditDisposition,
    pub input_receipt_count: usize,
    /// Only unique receipt IDs contribute to attempt counters; duplicate IDs are invalid.
    pub unique_receipt_count: usize,
    pub duplicate_receipt_record_count: usize,
    pub complete_attempts: u32,
    pub failed_attempts: u32,
    pub censored_attempts: u32,
    pub out_of_domain_attempts: u32,
    pub invalid_result_attempts: u32,
    pub unmatched_receipt_count: usize,
    /// None means total campaign cost is not auditable under this profile.
    pub total_cost_units: Option<u128>,
    pub requirements: Vec<RequirementProgress>,
    /// Canonically sorted so record-input permutation does not alter the audit.
    pub issues: Vec<FidelityIssue>,
}

#[derive(Debug, Default, Clone)]
struct AttemptCounts {
    complete: u32,
    failed: u32,
    censored: u32,
    out_of_domain: u32,
    invalid_result: u32,
}

impl AttemptCounts {
    fn add(&mut self, status: AttemptStatus) {
        match status {
            AttemptStatus::Complete => self.complete += 1,
            AttemptStatus::Failed => self.failed += 1,
            AttemptStatus::Censored => self.censored += 1,
            AttemptStatus::OutOfDomain => self.out_of_domain += 1,
            AttemptStatus::InvalidResult => self.invalid_result += 1,
        }
    }

    fn total(&self) -> u32 {
        self.complete + self.failed + self.censored + self.out_of_domain + self.invalid_result
    }
}

fn is_nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.as_bytes().iter().all(u8::is_ascii_hexdigit)
}

fn validate_profile(profile: &FidelityProfile, issues: &mut Vec<FidelityIssue>) -> bool {
    let mut invalid = false;

    for (value, field) in [
        (profile.profile_id.as_str(), "profile_id"),
        (profile.campaign_id.as_str(), "campaign_id"),
        (profile.cost_basis_id.as_str(), "cost_basis_id"),
    ] {
        if !is_nonempty(value) {
            issues.push(FidelityIssue::EmptyProfileIdentity(field));
            invalid = true;
        }
    }
    if !valid_sha256_hex(&profile.manifest_sha256) {
        issues.push(FidelityIssue::InvalidManifestDigest);
        invalid = true;
    }
    if profile.requirements.is_empty() {
        issues.push(FidelityIssue::EmptyRequirements);
        invalid = true;
    }
    if !profile.requirements.iter().any(|r| r.required_successes > 0) {
        issues.push(FidelityIssue::NoRequiredReleaseGate);
        invalid = true;
    }

    let mut seen = BTreeSet::new();
    let mut candidate_configurations: BTreeMap<String, String> = BTreeMap::new();
    for requirement in &profile.requirements {
        if let Some(first_digest) = candidate_configurations.get(&requirement.candidate_id) {
            if first_digest != &requirement.configuration_sha256 {
                issues.push(FidelityIssue::CandidateConfigurationMismatch {
                    candidate_id: requirement.candidate_id.clone(),
                    first_sha256: first_digest.clone(),
                    observed_sha256: requirement.configuration_sha256.clone(),
                });
                invalid = true;
            }
        } else {
            candidate_configurations.insert(
                requirement.candidate_id.clone(),
                requirement.configuration_sha256.clone(),
            );
        }
        if !seen.insert(requirement.requirement_id.clone()) {
            issues.push(FidelityIssue::DuplicateRequirementId {
                requirement_id: requirement.requirement_id.clone(),
            });
            invalid = true;
        }
        for (value, field) in [
            (requirement.requirement_id.as_str(), "requirement_id"),
            (requirement.candidate_id.as_str(), "candidate_id"),
            (requirement.operating_case_id.as_str(), "operating_case_id"),
            (requirement.method_id.as_str(), "method_id"),
            (requirement.method_revision.as_str(), "method_revision"),
        ] {
            if !is_nonempty(value) {
                issues.push(FidelityIssue::InvalidRequirement {
                    requirement_id: requirement.requirement_id.clone(),
                    field,
                });
                invalid = true;
            }
        }
        if !valid_sha256_hex(&requirement.configuration_sha256) {
            issues.push(FidelityIssue::InvalidRequirement {
                requirement_id: requirement.requirement_id.clone(),
                field: "configuration_sha256",
            });
            invalid = true;
        }
    }

    invalid
}

fn validate_receipt(receipt: &EvaluationReceipt, issues: &mut Vec<FidelityIssue>) -> bool {
    let mut invalid = false;

    for (value, field) in [
        (receipt.receipt_id.as_str(), "receipt_id"),
        (receipt.campaign_id.as_str(), "campaign_id"),
        (receipt.requirement_id.as_str(), "requirement_id"),
        (receipt.candidate_id.as_str(), "candidate_id"),
        (receipt.operating_case_id.as_str(), "operating_case_id"),
        (receipt.method_id.as_str(), "method_id"),
        (receipt.method_revision.as_str(), "method_revision"),
        (receipt.source_id.as_str(), "source_id"),
        (receipt.cost_basis_id.as_str(), "cost_basis_id"),
    ] {
        if !is_nonempty(value) {
            issues.push(FidelityIssue::EmptyReceiptIdentity {
                receipt_id: receipt.receipt_id.clone(),
                field,
            });
            invalid = true;
        }
    }

    for (value, field) in [
        (receipt.configuration_sha256.as_str(), "configuration_sha256"),
        (receipt.source_reference_sha256.as_str(), "source_reference_sha256"),
    ] {
        if !valid_sha256_hex(value) {
            issues.push(FidelityIssue::InvalidReceiptDigest {
                receipt_id: receipt.receipt_id.clone(),
                field,
            });
            invalid = true;
        }
    }

    match &receipt.output_sha256 {
        Some(digest) if !valid_sha256_hex(digest) => {
            issues.push(FidelityIssue::InvalidReceiptDigest {
                receipt_id: receipt.receipt_id.clone(),
                field: "output_sha256",
            });
            invalid = true;
        }
        None if receipt.status == AttemptStatus::Complete => {
            issues.push(FidelityIssue::MissingOutputDigest {
                receipt_id: receipt.receipt_id.clone(),
            });
            invalid = true;
        }
        _ => {}
    }

    match receipt.cost_units {
        None => {
            issues.push(FidelityIssue::MissingCost {
                receipt_id: receipt.receipt_id.clone(),
            });
            invalid = true;
        }
        Some(0) => {
            issues.push(FidelityIssue::ZeroCost {
                receipt_id: receipt.receipt_id.clone(),
            });
            invalid = true;
        }
        Some(_) => {}
    }

    invalid
}

fn requirement_mismatch(
    requirement: &FidelityRequirement,
    receipt: &EvaluationReceipt,
) -> Option<&'static str> {
    if requirement.candidate_id != receipt.candidate_id {
        return Some("candidate_id");
    }
    if requirement.configuration_sha256 != receipt.configuration_sha256 {
        return Some("configuration_sha256");
    }
    if requirement.operating_case_id != receipt.operating_case_id {
        return Some("operating_case_id");
    }
    if requirement.evaluation_class != receipt.evaluation_class {
        return Some("evaluation_class");
    }
    if requirement.evidence_origin != receipt.evidence_origin {
        return Some("evidence_origin");
    }
    if requirement.method_id != receipt.method_id {
        return Some("method_id");
    }
    if requirement.method_revision != receipt.method_revision {
        return Some("method_revision");
    }
    None
}

/// Audit all evaluation attempts against one frozen profile.
///
/// Required-release tiers are explicit per requirement. There is no built-in
/// assumption that CFD, measurement or any named class is universally superior.
/// Failed/censored/out-of-domain runs retain their cost. Missing, zero or
/// mismatched cost invalidates cost accounting instead of becoming a free trial.
pub fn audit_fidelity_campaign(
    profile: &FidelityProfile,
    receipts: &[EvaluationReceipt],
) -> FidelityAudit {
    let mut issues = Vec::new();
    let invalid_profile = validate_profile(profile, &mut issues);

    let mut id_counts: BTreeMap<String, usize> = BTreeMap::new();
    for receipt in receipts {
        *id_counts.entry(receipt.receipt_id.clone()).or_default() += 1;
    }
    let duplicate_ids: BTreeSet<String> = id_counts
        .iter()
        .filter(|(_, count)| **count > 1)
        .map(|(id, _)| id.clone())
        .collect();
    for (receipt_id, count) in &id_counts {
        if *count > 1 {
            issues.push(FidelityIssue::DuplicateReceiptId {
                receipt_id: receipt_id.clone(),
                count: *count,
            });
        }
    }

    let unique_receipt_count = id_counts.len() - duplicate_ids.len();
    let duplicate_receipt_record_count: usize = duplicate_ids
        .iter()
        .filter_map(|id| id_counts.get(id))
        .sum();

    let mut requirement_by_id: BTreeMap<String, &FidelityRequirement> = BTreeMap::new();
    for requirement in &profile.requirements {
        requirement_by_id
            .entry(requirement.requirement_id.clone())
            .or_insert(requirement);
    }

    let mut progress: BTreeMap<String, AttemptCounts> = BTreeMap::new();
    for requirement in &profile.requirements {
        progress.entry(requirement.requirement_id.clone()).or_default();
    }

    let mut unmatched_receipt_count = 0;
    let mut any_invalid_receipt = !duplicate_ids.is_empty();
    let mut all_attempt_counts = AttemptCounts::default();
    let mut total_cost_units: u128 = 0;
    let mut cost_auditable = duplicate_ids.is_empty();

    // Sorting makes tie handling deterministic. Duplicate IDs are excluded in full,
    // because choosing one duplicate payload by input order would create ambiguity.
    let mut sorted_receipts: Vec<&EvaluationReceipt> = receipts.iter().collect();
    sorted_receipts.sort_by(|a, b| {
        a.receipt_id
            .cmp(&b.receipt_id)
            .then_with(|| a.requirement_id.cmp(&b.requirement_id))
            .then_with(|| a.candidate_id.cmp(&b.candidate_id))
            .then_with(|| a.configuration_sha256.cmp(&b.configuration_sha256))
            .then_with(|| a.operating_case_id.cmp(&b.operating_case_id))
    });

    for receipt in sorted_receipts {
        if duplicate_ids.contains(&receipt.receipt_id) {
            continue;
        }

        all_attempt_counts.add(receipt.status);
        if validate_receipt(receipt, &mut issues) {
            any_invalid_receipt = true;
        }
        if receipt.campaign_id != profile.campaign_id {
            issues.push(FidelityIssue::CampaignMismatch {
                receipt_id: receipt.receipt_id.clone(),
            });
            unmatched_receipt_count += 1;
            cost_auditable = false;
            continue;
        }

        match (receipt.cost_units, receipt.cost_basis_id == profile.cost_basis_id) {
            (Some(cost), true) if cost > 0 => {
                match total_cost_units.checked_add(cost as u128) {
                    Some(next) => total_cost_units = next,
                    None => {
                        cost_auditable = false;
                        any_invalid_receipt = true;
                    }
                }
            }
            (_, false) => {
                issues.push(FidelityIssue::CostBasisMismatch {
                    receipt_id: receipt.receipt_id.clone(),
                });
                any_invalid_receipt = true;
                cost_auditable = false;
            }
            _ => {
                cost_auditable = false;
            }
        }

        let Some(requirement) = requirement_by_id.get(&receipt.requirement_id) else {
            issues.push(FidelityIssue::UnregisteredRequirement {
                receipt_id: receipt.receipt_id.clone(),
                requirement_id: receipt.requirement_id.clone(),
            });
            unmatched_receipt_count += 1;
            continue;
        };

        if let Some(field) = requirement_mismatch(requirement, receipt) {
            issues.push(FidelityIssue::ReceiptRequirementMismatch {
                receipt_id: receipt.receipt_id.clone(),
                requirement_id: requirement.requirement_id.clone(),
                field,
            });
            unmatched_receipt_count += 1;
            continue;
        }

        if receipt.evidence_use_status != EvidenceUseStatus::PermittedForDeclaredUse {
            issues.push(FidelityIssue::EvidenceRightsNotPermitted {
                receipt_id: receipt.receipt_id.clone(),
                status: receipt.evidence_use_status,
            });
            unmatched_receipt_count += 1;
            continue;
        }

        progress
            .entry(requirement.requirement_id.clone())
            .or_default()
            .add(receipt.status);
    }

    let mut requirement_progress = Vec::new();
    let mut missing_successes = 0;
    for requirement in &profile.requirements {
        let counts = progress
            .get(&requirement.requirement_id)
            .cloned()
            .unwrap_or_default();
        let covered = counts.complete >= requirement.required_successes;
        if !covered {
            missing_successes += 1;
            issues.push(FidelityIssue::MissingRequirementEvidence {
                requirement_id: requirement.requirement_id.clone(),
                required_successes: requirement.required_successes,
                completed_successes: counts.complete,
            });
        }

        requirement_progress.push(RequirementProgress {
            requirement_id: requirement.requirement_id.clone(),
            candidate_id: requirement.candidate_id.clone(),
            operating_case_id: requirement.operating_case_id.clone(),
            evaluation_class: requirement.evaluation_class,
            evidence_origin: requirement.evidence_origin,
            required_successes: requirement.required_successes,
            completed_successes: counts.complete,
            failed_attempts: counts.failed,
            censored_attempts: counts.censored,
            out_of_domain_attempts: counts.out_of_domain,
            invalid_result_attempts: counts.invalid_result,
            total_attempts: counts.total(),
            covered,
        });
    }
    requirement_progress.sort_by(|a, b| a.requirement_id.cmp(&b.requirement_id));
    issues.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));

    let complete_attempts = all_attempt_counts.complete;
    let failed_attempts = all_attempt_counts.failed;
    let censored_attempts = all_attempt_counts.censored;
    let out_of_domain_attempts = all_attempt_counts.out_of_domain;
    let invalid_result_attempts = all_attempt_counts.invalid_result;

    let disposition = if invalid_profile {
        FidelityAuditDisposition::InvalidProfile
    } else if any_invalid_receipt {
        FidelityAuditDisposition::InvalidReceipts
    } else if missing_successes > 0 || unmatched_receipt_count > 0 || issues.iter().any(|i| matches!(i, FidelityIssue::CampaignMismatch { .. } | FidelityIssue::UnregisteredRequirement { .. } | FidelityIssue::ReceiptRequirementMismatch { .. } | FidelityIssue::CostBasisMismatch { .. })) {
        FidelityAuditDisposition::Incomplete
    } else {
        FidelityAuditDisposition::EvidenceCoverageComplete
    };

    FidelityAudit {
        profile_id: profile.profile_id.clone(),
        campaign_id: profile.campaign_id.clone(),
        profile_manifest_sha256: profile.manifest_sha256.clone(),
        cost_basis_id: profile.cost_basis_id.clone(),
        disposition,
        input_receipt_count: receipts.len(),
        unique_receipt_count,
        duplicate_receipt_record_count,
        complete_attempts,
        failed_attempts,
        censored_attempts,
        out_of_domain_attempts,
        invalid_result_attempts,
        unmatched_receipt_count,
        total_cost_units: cost_auditable.then_some(total_cost_units),
        requirements: requirement_progress,
        issues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn requirement(
        id: &str,
        case_id: &str,
        class: EvaluationClass,
        origin: EvidenceOrigin,
        required_successes: u32,
    ) -> FidelityRequirement {
        FidelityRequirement {
            requirement_id: id.to_owned(),
            candidate_id: "hull-prop-v1".to_owned(),
            configuration_sha256: "a".repeat(64),
            operating_case_id: case_id.to_owned(),
            evaluation_class: class,
            evidence_origin: origin,
            method_id: format!("method-{id}"),
            method_revision: "rev-1".to_owned(),
            required_successes,
        }
    }

    fn profile() -> FidelityProfile {
        FidelityProfile {
            profile_id: "marine-profile-A".to_owned(),
            campaign_id: "campaign-A".to_owned(),
            manifest_sha256: "b".repeat(64),
            cost_basis_id: "cost-units-v1".to_owned(),
            requirements: vec![
                requirement("screen-low", "low-speed", EvaluationClass::LowFidelitySimulation, EvidenceOrigin::CfdSimulation, 0),
                requirement("release-cfd-low", "low-speed", EvaluationClass::HighFidelitySimulation, EvidenceOrigin::CfdSimulation, 1),
                requirement("release-test-low", "low-speed", EvaluationClass::PhysicalTest, EvidenceOrigin::IndependentPhysicalValidation, 1),
                requirement("release-cfd-cruise", "cruise", EvaluationClass::HighFidelitySimulation, EvidenceOrigin::CfdSimulation, 1),
            ],
        }
    }

    fn receipt(
        req: &FidelityRequirement,
        receipt_id: &str,
        status: AttemptStatus,
        cost: Option<u64>,
    ) -> EvaluationReceipt {
        EvaluationReceipt {
            receipt_id: receipt_id.to_owned(),
            campaign_id: "campaign-A".to_owned(),
            requirement_id: req.requirement_id.clone(),
            candidate_id: req.candidate_id.clone(),
            configuration_sha256: req.configuration_sha256.clone(),
            operating_case_id: req.operating_case_id.clone(),
            evaluation_class: req.evaluation_class,
            evidence_origin: req.evidence_origin,
            method_id: req.method_id.clone(),
            method_revision: req.method_revision.clone(),
            source_id: "synthetic-source-permitted-for-benchmark".to_owned(),
            source_reference_sha256: "c".repeat(64),
            evidence_use_status: EvidenceUseStatus::PermittedForDeclaredUse,
            cost_basis_id: "cost-units-v1".to_owned(),
            cost_units: cost,
            status,
            output_sha256: (status == AttemptStatus::Complete).then(|| "d".repeat(64)),
        }
    }

    fn complete_receipts(p: &FidelityProfile) -> Vec<EvaluationReceipt> {
        p.requirements.iter()
            .filter(|r| r.required_successes > 0)
            .enumerate()
            .map(|(i, r)| receipt(r, &format!("receipt-{i}"), AttemptStatus::Complete, Some((i as u64 + 1) * 10)))
            .collect()
    }

    fn has_issue(audit: &FidelityAudit, issue: &FidelityIssue) -> bool {
        audit.issues.contains(issue)
    }

    #[test]
    fn complete_declared_fidelity_requirements_pass_coverage_only() {
        let p = profile();
        let audit = audit_fidelity_campaign(&p, &complete_receipts(&p));
        assert_eq!(audit.disposition, FidelityAuditDisposition::EvidenceCoverageComplete);
        assert_eq!(audit.total_cost_units, Some(60));
        assert_eq!(audit.requirements.iter().filter(|r| r.covered).count(), 4);
    }

    #[test]
    fn low_fidelity_cannot_satisfy_high_fidelity_requirement() {
        let p = profile();
        let low = p.requirements.iter().find(|r| r.requirement_id == "screen-low").unwrap();
        let audit = audit_fidelity_campaign(&p, &[receipt(low, "screen-result", AttemptStatus::Complete, Some(1))]);
        assert_ne!(audit.disposition, FidelityAuditDisposition::EvidenceCoverageComplete);
        assert!(has_issue(&audit, &FidelityIssue::MissingRequirementEvidence {
            requirement_id: "release-cfd-low".to_owned(),
            required_successes: 1,
            completed_successes: 0,
        }));
    }

    #[test]
    fn physical_validation_is_separate_from_high_fidelity_cfd() {
        let p = profile();
        let receipts = p.requirements.iter()
            .filter(|r| r.required_successes > 0 && r.requirement_id != "release-test-low")
            .enumerate()
            .map(|(i,r)| receipt(r, &format!("cfd-{i}"), AttemptStatus::Complete, Some(15)))
            .collect::<Vec<_>>();
        let audit = audit_fidelity_campaign(&p, &receipts);
        assert!(has_issue(&audit, &FidelityIssue::MissingRequirementEvidence {
            requirement_id: "release-test-low".to_owned(),
            required_successes: 1,
            completed_successes: 0,
        }));
    }

    #[test]
    fn missing_cost_is_invalid_not_free() {
        let p = profile();
        let r = &p.requirements[1];
        let audit = audit_fidelity_campaign(&p, &[receipt(r, "receipt-no-cost", AttemptStatus::Complete, None)]);
        assert_eq!(audit.disposition, FidelityAuditDisposition::InvalidReceipts);
        assert_eq!(audit.total_cost_units, None);
        assert!(has_issue(&audit, &FidelityIssue::MissingCost { receipt_id: "receipt-no-cost".to_owned() }));
    }

    #[test]
    fn zero_cost_is_invalid() {
        let p = profile();
        let r = &p.requirements[1];
        let audit = audit_fidelity_campaign(&p, &[receipt(r, "receipt-zero-cost", AttemptStatus::Complete, Some(0))]);
        assert_eq!(audit.disposition, FidelityAuditDisposition::InvalidReceipts);
        assert!(has_issue(&audit, &FidelityIssue::ZeroCost { receipt_id: "receipt-zero-cost".to_owned() }));
    }

    #[test]
    fn mixed_cost_basis_invalidates_total_cost() {
        let p = profile();
        let r = &p.requirements[1];
        let mut record = receipt(r, "receipt-cost-basis", AttemptStatus::Complete, Some(20));
        record.cost_basis_id = "currency-dollars".to_owned();
        let audit = audit_fidelity_campaign(&p, &[record]);
        assert_eq!(audit.total_cost_units, None);
        assert!(has_issue(&audit, &FidelityIssue::CostBasisMismatch { receipt_id: "receipt-cost-basis".to_owned() }));
    }

    #[test]
    fn failed_attempt_retains_cost_and_does_not_satisfy_release() {
        let p = profile();
        let r = &p.requirements[1];
        let audit = audit_fidelity_campaign(&p, &[receipt(r, "failed-cfd", AttemptStatus::Failed, Some(42))]);
        assert_eq!(audit.failed_attempts, 1);
        assert_eq!(audit.total_cost_units, Some(42));
        assert!(has_issue(&audit, &FidelityIssue::MissingRequirementEvidence {
            requirement_id: r.requirement_id.clone(),
            required_successes: 1,
            completed_successes: 0,
        }));
    }

    #[test]
    fn out_of_domain_attempt_is_counted_but_never_release_evidence() {
        let p = profile();
        let r = &p.requirements[1];
        let audit = audit_fidelity_campaign(&p, &[receipt(r, "ood-cfd", AttemptStatus::OutOfDomain, Some(30))]);
        assert_eq!(audit.out_of_domain_attempts, 1);
        assert_eq!(audit.total_cost_units, Some(30));
        assert_eq!(audit.disposition, FidelityAuditDisposition::Incomplete);
    }

    #[test]
    fn duplicate_receipt_ids_are_not_double_counted() {
        let p = profile();
        let r = &p.requirements[1];
        let duplicate = receipt(r, "receipt-dup", AttemptStatus::Complete, Some(20));
        let audit = audit_fidelity_campaign(&p, &[duplicate.clone(), duplicate]);
        assert_eq!(audit.duplicate_receipt_record_count, 2);
        assert_eq!(audit.unique_receipt_count, 0);
        assert_eq!(audit.total_cost_units, None);
        assert_eq!(audit.disposition, FidelityAuditDisposition::InvalidReceipts);
    }

    #[test]
    fn stale_revision_cannot_satisfy_release_requirement() {
        let p = profile();
        let r = &p.requirements[1];
        let mut record = receipt(r, "stale-method", AttemptStatus::Complete, Some(10));
        record.method_revision = "rev-old".to_owned();
        let audit = audit_fidelity_campaign(&p, &[record]);
        assert!(has_issue(&audit, &FidelityIssue::ReceiptRequirementMismatch {
            receipt_id: "stale-method".to_owned(),
            requirement_id: r.requirement_id.clone(),
            field: "method_revision",
        }));
    }

    #[test]
    fn changed_geometry_digest_cannot_satisfy_requirement() {
        let p = profile();
        let r = &p.requirements[1];
        let mut record = receipt(r, "changed-geometry", AttemptStatus::Complete, Some(10));
        record.configuration_sha256 = "e".repeat(64);
        let audit = audit_fidelity_campaign(&p, &[record]);
        assert!(has_issue(&audit, &FidelityIssue::ReceiptRequirementMismatch {
            receipt_id: "changed-geometry".to_owned(),
            requirement_id: r.requirement_id.clone(),
            field: "configuration_sha256",
        }));
    }

    #[test]
    fn missing_complete_output_digest_is_invalid() {
        let p = profile();
        let r = &p.requirements[1];
        let mut record = receipt(r, "missing-output-digest", AttemptStatus::Complete, Some(10));
        record.output_sha256 = None;
        let audit = audit_fidelity_campaign(&p, &[record]);
        assert_eq!(audit.disposition, FidelityAuditDisposition::InvalidReceipts);
        assert!(has_issue(&audit, &FidelityIssue::MissingOutputDigest { receipt_id: "missing-output-digest".to_owned() }));
    }

    #[test]
    fn unknown_source_rights_cannot_satisfy_a_release_requirement() {
        let p = profile();
        let requirement = &p.requirements[1];
        let mut record = receipt(requirement, "unknown-rights", AttemptStatus::Complete, Some(20));
        record.evidence_use_status = EvidenceUseStatus::Unknown;
        let audit = audit_fidelity_campaign(&p, &[record]);
        assert_ne!(audit.disposition, FidelityAuditDisposition::EvidenceCoverageComplete);
        assert!(has_issue(&audit, &FidelityIssue::EvidenceRightsNotPermitted {
            receipt_id: "unknown-rights".to_owned(),
            status: EvidenceUseStatus::Unknown,
        }));
        assert_eq!(audit.total_cost_units, Some(20));
    }

    #[test]
    fn unregistered_requirement_remains_visible() {
        let p = profile();
        let mut record = receipt(&p.requirements[1], "unregistered", AttemptStatus::Complete, Some(10));
        record.requirement_id = "not-in-profile".to_owned();
        let audit = audit_fidelity_campaign(&p, &[record]);
        assert_eq!(audit.unmatched_receipt_count, 1);
        assert!(has_issue(&audit, &FidelityIssue::UnregisteredRequirement {
            receipt_id: "unregistered".to_owned(),
            requirement_id: "not-in-profile".to_owned(),
        }));
    }

    #[test]
    fn input_permutation_does_not_change_audit() {
        let p = profile();
        let receipts = complete_receipts(&p);
        let reversed = receipts.iter().cloned().rev().collect::<Vec<_>>();
        assert_eq!(
            audit_fidelity_campaign(&p, &receipts),
            audit_fidelity_campaign(&p, &reversed)
        );
    }

    #[test]
    fn profile_with_only_optional_requirements_is_invalid() {
        let mut p = profile();
        for r in &mut p.requirements {
            r.required_successes = 0;
        }
        let audit = audit_fidelity_campaign(&p, &[]);
        assert_eq!(audit.disposition, FidelityAuditDisposition::InvalidProfile);
        assert!(has_issue(&audit, &FidelityIssue::NoRequiredReleaseGate));
    }

    #[test]
    fn candidate_geometry_digest_must_be_stable_across_operating_cases() {
        let mut p = profile();
        p.requirements[3].configuration_sha256 = "e".repeat(64);
        let audit = audit_fidelity_campaign(&p, &complete_receipts(&p));
        assert_eq!(audit.disposition, FidelityAuditDisposition::InvalidProfile);
        assert!(audit.issues.iter().any(|issue| matches!(
            issue,
            FidelityIssue::CandidateConfigurationMismatch { candidate_id, .. }
                if candidate_id == "hull-prop-v1"
        )));
    }

    #[test]
    fn duplicate_requirement_ids_invalidate_profile() {
        let mut p = profile();
        p.requirements[1].requirement_id = p.requirements[0].requirement_id.clone();
        let audit = audit_fidelity_campaign(&p, &complete_receipts(&p));
        assert_eq!(audit.disposition, FidelityAuditDisposition::InvalidProfile);
    }

    #[test]
    fn campaign_mismatch_is_not_counted_as_profile_cost() {
        let p = profile();
        let r = &p.requirements[1];
        let mut record = receipt(r, "wrong-campaign", AttemptStatus::Complete, Some(10));
        record.campaign_id = "other-campaign".to_owned();
        let audit = audit_fidelity_campaign(&p, &[record]);
        assert_eq!(audit.total_cost_units, None);
        assert!(has_issue(&audit, &FidelityIssue::CampaignMismatch { receipt_id: "wrong-campaign".to_owned() }));
    }
}
