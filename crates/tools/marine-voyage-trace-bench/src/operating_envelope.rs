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
    use crate::paired_trial::EligiblePair;
    use crate::{EnergyBand, TraceEnergy};

    fn energy(trace_id: &str, case: &OperatingCaseSpec) -> TraceEnergy {
        TraceEnergy {
            trace_id: trace_id.to_owned(),
            raw_trace_sha256: "a".repeat(64),
            mission_id: case.mission_id.clone(),
            condition_digest: case.condition_digest.clone(),
            service_digest: case.service_digest.clone(),
            calibration_epoch_id: "meter-epoch-A".to_owned(),
            energy_boundary_id: "dc-input-boundary-A".to_owned(),
            uncertainty_method_id: "absolute-bound-v1".to_owned(),
            start_ms: 0,
            end_ms: 60_000,
            duration_ms: 60_000,
            sample_count: 61,
            max_observed_gap_ms: 1_000,
            propulsion_wh: EnergyBand { lower_wh: 80.0, nominal_wh: 100.0, upper_wh: 120.0 },
            auxiliary_wh: EnergyBand { lower_wh: 8.0, nominal_wh: 10.0, upper_wh: 12.0 },
            total_wh: EnergyBand { lower_wh: 88.0, nominal_wh: 110.0, upper_wh: 132.0 },
        }
    }

    fn case(case_id: &str, service_suffix: char, condition: &str) -> OperatingCaseSpec {
        OperatingCaseSpec {
            case_id: case_id.to_owned(),
            mission_id: "mission-A".to_owned(),
            service_digest: service_suffix.to_string().repeat(64),
            condition_digest: condition.to_owned(),
        }
    }

    fn profile() -> OperatingEnvelopeSpec {
        OperatingEnvelopeSpec {
            profile_id: "profile-A".to_owned(),
            manifest_sha256: "f".repeat(64),
            campaign_id: "campaign-A".to_owned(),
            cases: vec![
                case("low-speed", 'b', "condition-low-speed"),
                case("cruise", 'c', "condition-cruise"),
            ],
        }
    }

    fn result(
        case: &OperatingCaseSpec,
        campaign_id: &str,
        baseline_id: &str,
        candidate_id: &str,
    ) -> PairAssessment {
        PairAssessment {
            campaign_id: campaign_id.to_owned(),
            pair_id: case.case_id.clone(),
            disposition: PairDisposition::EligibleForEnergyComparison,
            issues: vec![],
            eligible_pair: Some(EligiblePair {
                campaign_id: campaign_id.to_owned(),
                pair_id: case.case_id.clone(),
                baseline_candidate_id: baseline_id.to_owned(),
                candidate_candidate_id: candidate_id.to_owned(),
                baseline_order: crate::paired_trial::TrialOrder::First,
                candidate_order: crate::paired_trial::TrialOrder::Second,
                baseline_energy: energy(&format!("base-{}", case.case_id), case),
                candidate_energy: energy(&format!("candidate-{}", case.case_id), case),
            }),
            run_count: 2,
        }
    }

    fn complete_results(p: &OperatingEnvelopeSpec) -> Vec<PairAssessment> {
        p.cases.iter()
            .map(|case| result(case, &p.campaign_id, "baseline-v1", "candidate-v2"))
            .collect()
    }

    fn has_issue(audit: &EnvelopeAudit, expected: &EnvelopeIssue) -> bool {
        audit.issues.contains(expected)
    }

    #[test]
    fn complete_matching_envelope_is_eligible_for_comparison() {
        let p = profile();
        let audit = audit_envelope(&p, &complete_results(&p));
        assert_eq!(audit.disposition, EnvelopeDisposition::CompleteForEnvelopeComparison);
        assert_eq!(audit.expected_case_count, 2);
        assert_eq!(audit.eligible_case_count, 2);
        assert_eq!(audit.baseline_design_id.as_deref(), Some("baseline-v1"));
        assert_eq!(audit.candidate_design_id.as_deref(), Some("candidate-v2"));
    }

    #[test]
    fn missing_required_case_never_passes() {
        let p = profile();
        let mut results = complete_results(&p);
        results.pop();
        let audit = audit_envelope(&p, &results);
        assert_eq!(audit.disposition, EnvelopeDisposition::Incomplete);
        assert_eq!(audit.missing_case_count, 1);
    }

    #[test]
    fn unexpected_case_is_visible_and_blocks_complete_status() {
        let p = profile();
        let mut results = complete_results(&p);
        let extra = case("unregistered-condition", 'd', "condition-extra");
        results.push(result(&extra, "campaign-A", "baseline-v1", "candidate-v2"));
        let audit = audit_envelope(&p, &results);
        assert_eq!(audit.disposition, EnvelopeDisposition::Incomplete);
        assert_eq!(audit.unexpected_case_count, 1);
        assert!(has_issue(&audit, &EnvelopeIssue::UnexpectedCase { case_id: "unregistered-condition".to_owned() }));
    }

    #[test]
    fn duplicate_profile_case_id_invalidates_profile() {
        let mut p = profile();
        p.cases.push(p.cases[0].clone());
        let audit = audit_envelope(&p, &complete_results(&p));
        assert_eq!(audit.disposition, EnvelopeDisposition::InvalidProfile);
        assert!(has_issue(&audit, &EnvelopeIssue::DuplicateCaseId { case_id: "low-speed".to_owned() }));
    }

    #[test]
    fn duplicate_result_for_case_blocks_profile() {
        let p = profile();
        let mut results = complete_results(&p);
        results.push(results[0].clone());
        let audit = audit_envelope(&p, &results);
        assert_eq!(audit.disposition, EnvelopeDisposition::Incomplete);
        assert_eq!(audit.duplicate_case_count, 1);
        assert!(has_issue(&audit, &EnvelopeIssue::DuplicateCaseResult { case_id: "low-speed".to_owned(), count: 2 }));
    }

    #[test]
    fn mission_mismatch_is_rejected() {
        let mut p = profile();
        p.cases[0].mission_id = "mission-other".to_owned();
        let audit = audit_envelope(&p, &complete_results(&p));
        assert!(has_issue(&audit, &EnvelopeIssue::MissionMismatch { case_id: "low-speed".to_owned() }));
        assert_ne!(audit.disposition, EnvelopeDisposition::CompleteForEnvelopeComparison);
    }

    #[test]
    fn service_fingerprint_mismatch_is_rejected() {
        let p = profile();
        let mut results = complete_results(&p);
        if let Some(pair) = results[0].eligible_pair.as_mut() {
            pair.candidate_energy.service_digest = "d".repeat(64);
        }
        let audit = audit_envelope(&p, &results);
        assert!(has_issue(&audit, &EnvelopeIssue::ServiceDigestMismatch { case_id: "low-speed".to_owned() }));
    }

    #[test]
    fn condition_mismatch_is_rejected() {
        let p = profile();
        let mut results = complete_results(&p);
        if let Some(pair) = results[0].eligible_pair.as_mut() {
            pair.candidate_energy.condition_digest = "different-condition".to_owned();
        }
        let audit = audit_envelope(&p, &results);
        assert!(has_issue(&audit, &EnvelopeIssue::ConditionMismatch { case_id: "low-speed".to_owned() }));
    }

    #[test]
    fn mixed_baseline_designs_across_cases_are_rejected() {
        let p = profile();
        let mut results = complete_results(&p);
        if let Some(pair) = results[1].eligible_pair.as_mut() {
            pair.baseline_candidate_id = "another-baseline".to_owned();
        }
        let audit = audit_envelope(&p, &results);
        assert!(audit.issues.iter().any(|issue| matches!(issue, EnvelopeIssue::BaselineDesignMismatch { .. })));
    }

    #[test]
    fn mixed_candidate_designs_across_cases_are_rejected() {
        let p = profile();
        let mut results = complete_results(&p);
        if let Some(pair) = results[1].eligible_pair.as_mut() {
            pair.candidate_candidate_id = "another-candidate".to_owned();
        }
        let audit = audit_envelope(&p, &results);
        assert!(audit.issues.iter().any(|issue| matches!(issue, EnvelopeIssue::CandidateDesignMismatch { .. })));
    }

    #[test]
    fn incomparable_pair_does_not_count_as_coverage() {
        let p = profile();
        let mut results = complete_results(&p);
        results[0].disposition = PairDisposition::Incomparable;
        results[0].eligible_pair = None;
        results[0].issues = vec![crate::paired_trial::PairingIssue::ConditionMismatch];
        let audit = audit_envelope(&p, &results);
        assert_eq!(audit.eligible_case_count, 1);
        assert_eq!(audit.incomparable_case_count, 1);
        assert_ne!(audit.disposition, EnvelopeDisposition::CompleteForEnvelopeComparison);
    }

    #[test]
    fn wrong_campaign_is_rejected() {
        let p = profile();
        let mut results = complete_results(&p);
        results[0].campaign_id = "campaign-other".to_owned();
        let audit = audit_envelope(&p, &results);
        assert!(has_issue(&audit, &EnvelopeIssue::PairCampaignMismatch { case_id: "low-speed".to_owned() }));
    }

    #[test]
    fn invalid_profile_digest_is_rejected() {
        let mut p = profile();
        p.manifest_sha256 = "not-a-sha256".to_owned();
        let audit = audit_envelope(&p, &complete_results(&p));
        assert_eq!(audit.disposition, EnvelopeDisposition::InvalidProfile);
        assert!(has_issue(&audit, &EnvelopeIssue::InvalidManifestDigest));
    }

    #[test]
    fn empty_case_set_is_invalid() {
        let mut p = profile();
        p.cases.clear();
        let audit = audit_envelope(&p, &[]);
        assert_eq!(audit.disposition, EnvelopeDisposition::InvalidProfile);
        assert!(has_issue(&audit, &EnvelopeIssue::EmptyCaseSet));
    }

    #[test]
    fn malformed_service_digest_is_invalid_profile() {
        let mut p = profile();
        p.cases[0].service_digest = "not-hex".to_owned();
        let audit = audit_envelope(&p, &complete_results(&p));
        assert_eq!(audit.disposition, EnvelopeDisposition::InvalidProfile);
        assert!(audit.issues.iter().any(|issue| matches!(issue, EnvelopeIssue::InvalidServiceDigest { case_id } if case_id == "low-speed")));
    }

    #[test]
    fn same_baseline_and_candidate_is_rejected() {
        let p = profile();
        let results = p.cases.iter()
            .map(|case| result(case, "campaign-A", "design-A", "design-A"))
            .collect::<Vec<_>>();
        let audit = audit_envelope(&p, &results);
        assert!(audit.issues.iter().any(|issue| matches!(issue, EnvelopeIssue::SameDesignForBaselineAndCandidate { .. })));
    }

    #[test]
    fn input_order_permutation_produces_same_audit() {
        let p = profile();
        let results = complete_results(&p);
        let mut reversed_profile = p.clone();
        reversed_profile.cases.reverse();
        let reversed_results = results.iter().cloned().rev().collect::<Vec<_>>();
        assert_eq!(audit_envelope(&p, &results), audit_envelope(&reversed_profile, &reversed_results));
    }

    #[test]
    fn eligible_disposition_without_pair_payload_is_rejected() {
        let p = profile();
        let mut results = complete_results(&p);
        results[0].eligible_pair = None;
        let audit = audit_envelope(&p, &results);
        assert!(has_issue(&audit, &EnvelopeIssue::EligiblePairPayloadMissing { case_id: "low-speed".to_owned() }));
        assert_ne!(audit.disposition, EnvelopeDisposition::CompleteForEnvelopeComparison);
    }
}
