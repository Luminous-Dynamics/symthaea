//! Frozen multi-operating-point evidence coverage audit.
//!
//! This module never computes energy savings or hydrodynamics. It verifies that
//! each operating case declared before design comparison has exactly one
//! eligible baseline/candidate pair with the expected mission/service/condition
//! fingerprints, and that the same two design IDs are compared across the full
//! envelope. Coverage is necessary for a fair comparison, not proof of one.

use super::paired_trial::{
    EligiblePair, PairAssessment, PairDisposition, PairingIssue,
};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperatingCaseSpec {
    /// Stable ID; also used as the paired-trial pair ID.
    pub case_id: String,
    pub mission_id: String,
    /// SHA-256 digest of the delivered-service contract (format checked, not authenticated).
    pub service_digest: String,
    /// Exact external condition/profile identity. This module checks equality only.
    pub condition_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperatingEnvelopeSpec {
    pub profile_id: String,
    /// SHA-256 digest of the frozen profile manifest (format checked, not authenticated).
    pub manifest_sha256: String,
    pub campaign_id: String,
    pub cases: Vec<OperatingCaseSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvelopeIssue {
    EmptyIdentity(&'static str),
    InvalidManifestDigest,
    EmptyCaseSet,
    EmptyCaseId,
    DuplicateCaseId { case_id: String },
    EmptyMissionId { case_id: String },
    InvalidServiceDigest { case_id: String },
    EmptyConditionDigest { case_id: String },
    MissingCase { case_id: String },
    DuplicateCaseResult { case_id: String, count: usize },
    UnexpectedCase { case_id: String },
    PairNotEligible {
        case_id: String,
        issues: Vec<PairingIssue>,
    },
    EligiblePairPayloadMissing { case_id: String },
    PairCampaignMismatch { case_id: String },
    PairIdentityMismatch { case_id: String },
    MissionMismatch { case_id: String },
    ServiceDigestMismatch { case_id: String },
    ConditionMismatch { case_id: String },
    EmptyDesignId { case_id: String },
    SameDesignForBaselineAndCandidate { case_id: String },
    BaselineDesignMismatch {
        case_id: String,
        expected: String,
        observed: String,
    },
    CandidateDesignMismatch {
        case_id: String,
        expected: String,
        observed: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvelopeDisposition {
    CompleteForEnvelopeComparison,
    Incomplete,
    InvalidProfile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaseDisposition {
    Eligible,
    Missing,
    Duplicate,
    Unexpected,
    Incomparable,
    FingerprintMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaseCoverage {
    pub case_id: String,
    pub disposition: CaseDisposition,
    pub result_count: usize,
    pub issues: Vec<EnvelopeIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvelopeAudit {
    pub profile_id: String,
    pub campaign_id: String,
    pub disposition: EnvelopeDisposition,
    pub expected_case_count: usize,
    pub supplied_result_count: usize,
    pub eligible_case_count: usize,
    pub missing_case_count: usize,
    pub duplicate_case_count: usize,
    pub unexpected_case_count: usize,
    pub incomparable_case_count: usize,
    pub baseline_design_id: Option<String>,
    pub candidate_design_id: Option<String>,
    /// Sorted by case ID. Includes unexpected result groups as well as expected cases.
    pub cases: Vec<CaseCoverage>,
    /// Includes all profile and per-case reasons; output ordering is canonical.
    pub issues: Vec<EnvelopeIssue>,
}

fn is_nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.as_bytes().iter().all(u8::is_ascii_hexdigit)
}

fn fingerprint_issues(
    spec: &OperatingCaseSpec,
    pair: &EligiblePair,
) -> Vec<EnvelopeIssue> {
    let mut issues = Vec::new();

    if pair.baseline_candidate_id.trim().is_empty()
        || pair.candidate_candidate_id.trim().is_empty()
    {
        issues.push(EnvelopeIssue::EmptyDesignId {
            case_id: spec.case_id.clone(),
        });
    }
    if pair.baseline_candidate_id == pair.candidate_candidate_id {
        issues.push(EnvelopeIssue::SameDesignForBaselineAndCandidate {
            case_id: spec.case_id.clone(),
        });
    }
    if pair.campaign_id.trim().is_empty()
        || pair.campaign_id != pair.baseline_energy.trace_id && pair.campaign_id.is_empty()
    {
        // This deliberately only checks for a non-empty campaign identity here.
        // Campaign-to-profile matching is checked by the caller against PairAssessment.
        issues.push(EnvelopeIssue::PairCampaignMismatch {
            case_id: spec.case_id.clone(),
        });
    }
    if pair.baseline_energy.mission_id != spec.mission_id
        || pair.candidate_energy.mission_id != spec.mission_id
    {
        issues.push(EnvelopeIssue::MissionMismatch {
            case_id: spec.case_id.clone(),
        });
    }
    if pair.baseline_energy.service_digest != spec.service_digest
        || pair.candidate_energy.service_digest != spec.service_digest
    {
        issues.push(EnvelopeIssue::ServiceDigestMismatch {
            case_id: spec.case_id.clone(),
        });
    }
    if pair.baseline_energy.condition_digest != spec.condition_digest
        || pair.candidate_energy.condition_digest != spec.condition_digest
    {
        issues.push(EnvelopeIssue::ConditionMismatch {
            case_id: spec.case_id.clone(),
        });
    }

    issues
}

/// Audit evidence coverage for a frozen required operating envelope.
///
/// The function uses case IDs as the pairing key. Every expected case must
/// have exactly one eligible pair, and every pair must belong to the declared
/// campaign. Candidate identities cannot change between cases. Inputs from
/// distinct campaigns, unexpected cases, failed pairs, duplicates, and
/// missing cases are retained in the result and block a complete disposition.
pub fn audit_envelope(
    spec: &OperatingEnvelopeSpec,
    pair_assessments: &[PairAssessment],
) -> EnvelopeAudit {
    let mut issues = Vec::new();
    let mut invalid_profile = false;

    if !is_nonempty(&spec.profile_id) {
        issues.push(EnvelopeIssue::EmptyIdentity("profile_id"));
        invalid_profile = true;
    }
    if !is_nonempty(&spec.campaign_id) {
        issues.push(EnvelopeIssue::EmptyIdentity("campaign_id"));
        invalid_profile = true;
    }
    if !valid_sha256_hex(&spec.manifest_sha256) {
        issues.push(EnvelopeIssue::InvalidManifestDigest);
        invalid_profile = true;
    }
    if spec.cases.is_empty() {
        issues.push(EnvelopeIssue::EmptyCaseSet);
        invalid_profile = true;
    }

    // Sort first so even malformed profiles with repeated IDs produce stable
    // audit output under input-order permutations.
    let mut sorted_specs: Vec<&OperatingCaseSpec> = spec.cases.iter().collect();
    sorted_specs.sort_by(|a, b| {
        a.case_id
            .cmp(&b.case_id)
            .then_with(|| a.mission_id.cmp(&b.mission_id))
            .then_with(|| a.service_digest.cmp(&b.service_digest))
            .then_with(|| a.condition_digest.cmp(&b.condition_digest))
    });

    let mut expected: BTreeMap<String, &OperatingCaseSpec> = BTreeMap::new();
    for case in sorted_specs {
        if !is_nonempty(&case.case_id) {
            issues.push(EnvelopeIssue::EmptyCaseId);
            invalid_profile = true;
        }
        if !is_nonempty(&case.mission_id) {
            issues.push(EnvelopeIssue::EmptyMissionId {
                case_id: case.case_id.clone(),
            });
            invalid_profile = true;
        }
        if !valid_sha256_hex(&case.service_digest) {
            issues.push(EnvelopeIssue::InvalidServiceDigest {
                case_id: case.case_id.clone(),
            });
            invalid_profile = true;
        }
        if !is_nonempty(&case.condition_digest) {
            issues.push(EnvelopeIssue::EmptyConditionDigest {
                case_id: case.case_id.clone(),
            });
            invalid_profile = true;
        }

        if expected.contains_key(&case.case_id) {
            issues.push(EnvelopeIssue::DuplicateCaseId {
                case_id: case.case_id.clone(),
            });
            invalid_profile = true;
        } else {
            expected.insert(case.case_id.clone(), case);
        }
    }

    let mut results_by_case: BTreeMap<String, Vec<&PairAssessment>> = BTreeMap::new();
    for result in pair_assessments {
        results_by_case
            .entry(result.pair_id.clone())
            .or_default()
            .push(result);
    }

    let mut cases = Vec::new();
    let mut eligible_case_count = 0;
    let mut missing_case_count = 0;
    let mut duplicate_case_count = 0;
    let mut unexpected_case_count = 0;
    let mut incomparable_case_count = 0;
    let mut baseline_design_id: Option<String> = None;
    let mut candidate_design_id: Option<String> = None;

    for (case_id, case_spec) in &expected {
        let mut case_issues = Vec::new();
        let group = results_by_case.remove(case_id);

        let (disposition, result_count) = match group {
            None => {
                case_issues.push(EnvelopeIssue::MissingCase {
                    case_id: case_id.clone(),
                });
                missing_case_count += 1;
                (CaseDisposition::Missing, 0)
            }
            Some(group) if group.len() != 1 => {
                case_issues.push(EnvelopeIssue::DuplicateCaseResult {
                    case_id: case_id.clone(),
                    count: group.len(),
                });
                duplicate_case_count += 1;
                (CaseDisposition::Duplicate, group.len())
            }
            Some(group) => {
                let result = group[0];

                if result.campaign_id != spec.campaign_id {
                    case_issues.push(EnvelopeIssue::PairCampaignMismatch {
                        case_id: case_id.clone(),
                    });
                }

                if result.disposition != PairDisposition::EligibleForEnergyComparison {
                    case_issues.push(EnvelopeIssue::PairNotEligible {
                        case_id: case_id.clone(),
                        issues: result.issues.clone(),
                    });
                    incomparable_case_count += 1;
                    (CaseDisposition::Incomparable, 1)
                } else if let Some(pair) = &result.eligible_pair {
                    if pair.pair_id != *case_id
                        || pair.campaign_id != result.campaign_id
                        || pair.baseline_candidate_id.trim().is_empty()
                        || pair.candidate_candidate_id.trim().is_empty()
                    {
                        case_issues.push(EnvelopeIssue::PairIdentityMismatch {
                            case_id: case_id.clone(),
                        });
                    }

                    case_issues.extend(fingerprint_issues(case_spec, pair));

                    if case_issues.is_empty() {
                        match (&baseline_design_id, &candidate_design_id) {
                            (Some(expected_baseline), Some(expected_candidate)) => {
                                if expected_baseline != &pair.baseline_candidate_id {
                                    case_issues.push(EnvelopeIssue::BaselineDesignMismatch {
                                        case_id: case_id.clone(),
                                        expected: expected_baseline.clone(),
                                        observed: pair.baseline_candidate_id.clone(),
                                    });
                                }
                                if expected_candidate != &pair.candidate_candidate_id {
                                    case_issues.push(EnvelopeIssue::CandidateDesignMismatch {
                                        case_id: case_id.clone(),
                                        expected: expected_candidate.clone(),
                                        observed: pair.candidate_candidate_id.clone(),
                                    });
                                }
                            }
                            _ => {
                                baseline_design_id = Some(pair.baseline_candidate_id.clone());
                                candidate_design_id = Some(pair.candidate_candidate_id.clone());
                            }
                        }
                    }

                    if case_issues.is_empty() {
                        eligible_case_count += 1;
                        (CaseDisposition::Eligible, 1)
                    } else {
                        (CaseDisposition::FingerprintMismatch, 1)
                    }
                } else {
                    case_issues.push(EnvelopeIssue::EligiblePairPayloadMissing {
                        case_id: case_id.clone(),
                    });
                    incomparable_case_count += 1;
                    (CaseDisposition::Incomparable, 1)
                }
            }
        };

        issues.extend(case_issues.iter().cloned());
        cases.push(CaseCoverage {
            case_id: case_id.clone(),
            disposition,
            result_count,
            issues: case_issues,
        });
    }

    // Anything not in the frozen manifest remains visible as an unexpected case.
    for (case_id, group) in results_by_case {
        let mut case_issues = vec![EnvelopeIssue::UnexpectedCase {
            case_id: case_id.clone(),
        }];
        unexpected_case_count += 1;
        if group.len() > 1 {
            case_issues.push(EnvelopeIssue::DuplicateCaseResult {
                case_id: case_id.clone(),
                count: group.len(),
            });
            duplicate_case_count += 1;
        }
        issues.extend(case_issues.iter().cloned());
        cases.push(CaseCoverage {
            case_id,
            disposition: CaseDisposition::Unexpected,
            result_count: group.len(),
            issues: case_issues,
        });
    }
    cases.sort_by(|a, b| a.case_id.cmp(&b.case_id));

    EnvelopeAudit {
        profile_id: spec.profile_id.clone(),
        campaign_id: spec.campaign_id.clone(),
        disposition: if invalid_profile {
            EnvelopeDisposition::InvalidProfile
        } else if issues.is_empty() && eligible_case_count == expected.len() {
            EnvelopeDisposition::CompleteForEnvelopeComparison
        } else {
            EnvelopeDisposition::Incomplete
        },
        expected_case_count: spec.cases.len(),
        supplied_result_count: pair_assessments.len(),
        eligible_case_count,
        missing_case_count,
        duplicate_case_count,
        unexpected_case_count,
        incomparable_case_count,
        baseline_design_id,
        candidate_design_id,
        cases,
        issues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::marine_voyage_trace_bench::paired_trial;
