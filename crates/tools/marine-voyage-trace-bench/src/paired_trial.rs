//! Matched-trial eligibility and campaign order diagnostics.
//!
//! This module deliberately does not calculate a saving, correct environmental
//! conditions, infer randomization, or estimate a causal effect. It validates
//! that two already-integrated trace records are eligible for a separate
//! mission-energy comparison.

use super::{EnergyBand, TraceDisposition, TraceEnergy, TraceOutcome};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrialArm {
    Baseline,
    Candidate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrialOrder {
    First,
    Second,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrialRun {
    pub campaign_id: String,
    pub pair_id: String,
    pub candidate_id: String,
    pub arm: TrialArm,
    pub order: TrialOrder,
    pub trace_outcome: TraceOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairingIssue {
    EmptyIdentity(&'static str),
    SameCandidateId,
    CampaignMismatch,
    PairIdMismatch,
    BaselineArmRequired,
    CandidateArmRequired,
    SameRunOrder,
    TraceNotComplete {
        candidate_id: String,
        disposition: TraceDisposition,
    },
    MissionMismatch,
    ServiceDigestMismatch,
    ConditionMismatch,
    EnergyBoundaryMismatch,
    CalibrationEpochMismatch,
    UncertaintyMethodMismatch,
    PairCardinality {
        count: usize,
    },
    RoleAssignmentMismatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairDisposition {
    EligibleForEnergyComparison,
    Incomparable,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EligiblePair {
    pub campaign_id: String,
    pub pair_id: String,
    pub baseline_candidate_id: String,
    pub candidate_candidate_id: String,
    pub baseline_order: TrialOrder,
    pub candidate_order: TrialOrder,
    pub baseline_energy: TraceEnergy,
    pub candidate_energy: TraceEnergy,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PairAssessment {
    pub campaign_id: String,
    pub pair_id: String,
    pub disposition: PairDisposition,
    pub issues: Vec<PairingIssue>,
    pub eligible_pair: Option<EligiblePair>,
    pub run_count: usize,
}

impl PairAssessment {
    fn incomparable(
        campaign_id: String,
        pair_id: String,
        run_count: usize,
        issues: Vec<PairingIssue>,
    ) -> Self {
        Self {
            campaign_id,
            pair_id,
            disposition: PairDisposition::Incomparable,
            issues,
            eligible_pair: None,
            run_count,
        }
    }
}

/// Validate a baseline/candidate pair. This function only decides whether
/// the records may be passed to the independent energy comparator.
pub fn assess_pair(baseline: &TrialRun, candidate: &TrialRun) -> PairAssessment {
    let mut issues = Vec::new();

    for (value, field) in [
        (baseline.campaign_id.as_str(), "baseline.campaign_id"),
        (candidate.campaign_id.as_str(), "candidate.campaign_id"),
        (baseline.pair_id.as_str(), "baseline.pair_id"),
        (candidate.pair_id.as_str(), "candidate.pair_id"),
        (baseline.candidate_id.as_str(), "baseline.candidate_id"),
        (candidate.candidate_id.as_str(), "candidate.candidate_id"),
    ] {
        if value.trim().is_empty() {
            issues.push(PairingIssue::EmptyIdentity(field));
        }
    }

    if baseline.campaign_id != candidate.campaign_id {
        issues.push(PairingIssue::CampaignMismatch);
    }
    if baseline.pair_id != candidate.pair_id {
        issues.push(PairingIssue::PairIdMismatch);
    }
    if baseline.arm != TrialArm::Baseline {
        issues.push(PairingIssue::BaselineArmRequired);
    }
    if candidate.arm != TrialArm::Candidate {
        issues.push(PairingIssue::CandidateArmRequired);
    }
    if baseline.candidate_id == candidate.candidate_id {
        issues.push(PairingIssue::SameCandidateId);
    }
    if baseline.order == candidate.order {
        issues.push(PairingIssue::SameRunOrder);
    }

    let baseline_energy = match &baseline.trace_outcome {
        TraceOutcome::Complete(energy) => Some(energy),
        outcome => {
            issues.push(PairingIssue::TraceNotComplete {
                candidate_id: baseline.candidate_id.clone(),
                disposition: outcome.disposition(),
            });
            None
        }
    };
    let candidate_energy = match &candidate.trace_outcome {
        TraceOutcome::Complete(energy) => Some(energy),
        outcome => {
            issues.push(PairingIssue::TraceNotComplete {
                candidate_id: candidate.candidate_id.clone(),
                disposition: outcome.disposition(),
            });
            None
        }
    };

    if let (Some(base), Some(challenger)) = (baseline_energy, candidate_energy) {
        if base.mission_id != challenger.mission_id {
            issues.push(PairingIssue::MissionMismatch);
        }
        if base.service_digest != challenger.service_digest {
            issues.push(PairingIssue::ServiceDigestMismatch);
        }
        if base.condition_digest != challenger.condition_digest {
            issues.push(PairingIssue::ConditionMismatch);
        }
        if base.energy_boundary_id != challenger.energy_boundary_id {
            issues.push(PairingIssue::EnergyBoundaryMismatch);
        }
        if base.calibration_epoch_id != challenger.calibration_epoch_id {
            issues.push(PairingIssue::CalibrationEpochMismatch);
        }
        if base.uncertainty_method_id != challenger.uncertainty_method_id {
            issues.push(PairingIssue::UncertaintyMethodMismatch);
        }
    }

    if !issues.is_empty() {
        return PairAssessment::incomparable(
            baseline.campaign_id.clone(),
            baseline.pair_id.clone(),
            2,
            issues,
        );
    }

    let baseline_energy = baseline_energy.expect("complete baseline was checked");
    let candidate_energy = candidate_energy.expect("complete candidate was checked");

    PairAssessment {
        campaign_id: baseline.campaign_id.clone(),
        pair_id: baseline.pair_id.clone(),
        disposition: PairDisposition::EligibleForEnergyComparison,
        issues,
        eligible_pair: Some(EligiblePair {
            campaign_id: baseline.campaign_id.clone(),
            pair_id: baseline.pair_id.clone(),
            baseline_candidate_id: baseline.candidate_id.clone(),
            candidate_candidate_id: candidate.candidate_id.clone(),
            baseline_order: baseline.order,
            candidate_order: candidate.order,
            baseline_energy: baseline_energy.clone(),
            candidate_energy: candidate_energy.clone(),
        }),
        run_count: 2,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderCoverage {
    NoEligiblePairs,
    SingleOrderOnly,
    BothOrdersObserved,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComparisonOrderCounts {
    pub baseline_candidate_id: String,
    pub candidate_candidate_id: String,
    pub baseline_first_pairs: usize,
    pub candidate_first_pairs: usize,
    pub order_coverage: OrderCoverage,
    pub order_count_difference: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CampaignAudit {
    pub campaign_id: String,
    pub total_runs: usize,
    pub complete_runs: usize,
    pub indeterminate_runs: usize,
    pub invalid_runs: usize,
    pub pair_assessments: Vec<PairAssessment>,
    pub eligible_pair_count: usize,
    pub incomparable_pair_count: usize,
    pub baseline_first_eligible_pairs: usize,
    pub candidate_first_eligible_pairs: usize,
    pub ineligible_runs: usize,
    /// Order counts keyed by the exact baseline/candidate pair. Campaign-wide
    /// coverage is considered balanced only when every compared design pair
    /// has observations in both orders.
    pub comparison_order_counts: Vec<ComparisonOrderCounts>,
    pub order_coverage: OrderCoverage,
    pub order_count_difference: usize,
}

/// Audit all runs for one declared campaign. Pair groups and output order are
/// canonicalized by identifiers, so input permutation cannot silently change
/// the audit. A campaign with both orders observed is not proof of randomization
/// or sufficient statistical power; it is only a coverage diagnostic.
pub fn audit_campaign(campaign_id: &str, runs: &[TrialRun]) -> CampaignAudit {
    let mut complete_runs = 0;
    let mut indeterminate_runs = 0;
    let mut invalid_runs = 0;
    let mut groups: BTreeMap<(String, String), Vec<&TrialRun>> = BTreeMap::new();

    for run in runs {
        match run.trace_outcome.disposition() {
            TraceDisposition::Complete => complete_runs += 1,
            TraceDisposition::Indeterminate => indeterminate_runs += 1,
            TraceDisposition::Invalid => invalid_runs += 1,
        }
        groups
            .entry((run.campaign_id.clone(), run.pair_id.clone()))
            .or_default()
            .push(run);
    }

    let mut pair_assessments = Vec::new();
    let mut eligible_pair_count = 0;
    let mut incomparable_pair_count = 0;
    let mut baseline_first_eligible_pairs = 0;
    let mut candidate_first_eligible_pairs = 0;
    let mut ineligible_runs = 0;
    let mut order_counts: BTreeMap<(String, String), (usize, usize)> = BTreeMap::new();

    for ((group_campaign_id, pair_id), mut group) in groups {
        group.sort_by(|a, b| {
            a.arm
                .cmp(&b.arm)
                .then_with(|| a.candidate_id.cmp(&b.candidate_id))
                .then_with(|| a.order.cmp(&b.order))
        });

        let mut assessment = if group.len() != 2 {
            PairAssessment::incomparable(
                group_campaign_id.clone(),
                pair_id.clone(),
                group.len(),
                vec![PairingIssue::PairCardinality { count: group.len() }],
            )
        } else {
            let baseline = group.iter().find(|run| run.arm == TrialArm::Baseline);
            let candidate = group.iter().find(|run| run.arm == TrialArm::Candidate);
            match (baseline, candidate) {
                (Some(baseline), Some(candidate)) => assess_pair(baseline, candidate),
                _ => PairAssessment::incomparable(
                    group_campaign_id.clone(),
                    pair_id.clone(),
                    group.len(),
                    vec![PairingIssue::RoleAssignmentMismatch],
                ),
            }
        };

        if group_campaign_id != campaign_id {
            if assessment.eligible_pair.take().is_some() {
                assessment.disposition = PairDisposition::Incomparable;
            }
            if !assessment.issues.contains(&PairingIssue::CampaignMismatch) {
                assessment.issues.push(PairingIssue::CampaignMismatch);
            }
        }

        match assessment.disposition {
            PairDisposition::EligibleForEnergyComparison => {
                eligible_pair_count += 1;
                if let Some(pair) = &assessment.eligible_pair {
                    let key = (
                        pair.baseline_candidate_id.clone(),
                        pair.candidate_candidate_id.clone(),
                    );
                    let counts = order_counts.entry(key).or_insert((0, 0));
                    if pair.baseline_order == TrialOrder::First {
                        baseline_first_eligible_pairs += 1;
                        counts.0 += 1;
                    } else {
                        candidate_first_eligible_pairs += 1;
                        counts.1 += 1;
                    }
                }
            }
            PairDisposition::Incomparable => {
                incomparable_pair_count += 1;
                ineligible_runs += assessment.run_count;
            }
        }
        pair_assessments.push(assessment);
    }

    let comparison_order_counts: Vec<ComparisonOrderCounts> = order_counts
        .into_iter()
        .map(|((baseline_candidate_id, candidate_candidate_id), (baseline_first_pairs, candidate_first_pairs))| {
            let order_coverage = match (baseline_first_pairs > 0, candidate_first_pairs > 0) {
                (false, false) => OrderCoverage::NoEligiblePairs,
                (true, true) => OrderCoverage::BothOrdersObserved,
                _ => OrderCoverage::SingleOrderOnly,
            };
            ComparisonOrderCounts {
                baseline_candidate_id,
                candidate_candidate_id,
                baseline_first_pairs,
                candidate_first_pairs,
                order_coverage,
                order_count_difference: baseline_first_pairs.abs_diff(candidate_first_pairs),
            }
        })
        .collect();

    let order_coverage = if comparison_order_counts.is_empty() {
        OrderCoverage::NoEligiblePairs
    } else if comparison_order_counts
        .iter()
        .all(|counts| counts.order_coverage == OrderCoverage::BothOrdersObserved)
    {
        OrderCoverage::BothOrdersObserved
    } else {
        OrderCoverage::SingleOrderOnly
    };

    CampaignAudit {
        campaign_id: campaign_id.to_owned(),
        total_runs: runs.len(),
        complete_runs,
        indeterminate_runs,
        invalid_runs,
        pair_assessments,
        eligible_pair_count,
        incomparable_pair_count,
        baseline_first_eligible_pairs,
        candidate_first_eligible_pairs,
        ineligible_runs,
        comparison_order_counts,
        order_coverage,
        order_count_difference: baseline_first_eligible_pairs.abs_diff(candidate_first_eligible_pairs),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn band(wh: f64) -> EnergyBand {
        EnergyBand {
            lower_wh: wh,
            nominal_wh: wh,
            upper_wh: wh,
        }
    }

    fn energy(trace_id: &str) -> TraceEnergy {
        TraceEnergy {
            trace_id: trace_id.to_owned(),
            raw_trace_sha256: "a".repeat(64),
            mission_id: "mission-A".to_owned(),
            condition_digest: "environment-condition-sha256".to_owned(),
            service_digest: "b".repeat(64),
            calibration_epoch_id: "meter-cal-epoch-A".to_owned(),
            energy_boundary_id: "dc-input-boundary-A".to_owned(),
            uncertainty_method_id: "absolute-bound-v1".to_owned(),
            start_ms: 0,
            end_ms: 3_600_000,
            duration_ms: 3_600_000,
            sample_count: 361,
            max_observed_gap_ms: 10_000,
            propulsion_wh: band(800.0),
            auxiliary_wh: band(200.0),
            total_wh: band(1_000.0),
        }
    }

    fn run(
        pair_id: &str,
        candidate_id: &str,
        arm: TrialArm,
        order: TrialOrder,
        trace_outcome: TraceOutcome,
    ) -> TrialRun {
        TrialRun {
            campaign_id: "campaign-A".to_owned(),
            pair_id: pair_id.to_owned(),
            candidate_id: candidate_id.to_owned(),
            arm,
            order,
            trace_outcome,
        }
    }

    fn complete_run(
        pair_id: &str,
        candidate_id: &str,
        arm: TrialArm,
        order: TrialOrder,
    ) -> TrialRun {
        run(
            pair_id,
            candidate_id,
            arm,
            order,
            TraceOutcome::Complete(energy(candidate_id)),
        )
    }

    #[test]
    fn matching_service_and_condition_is_eligible() {
        let base = complete_run("pair-1", "baseline-v1", TrialArm::Baseline, TrialOrder::First);
        let candidate = complete_run("pair-1", "candidate-v2", TrialArm::Candidate, TrialOrder::Second);
        let result = assess_pair(&base, &candidate);
        assert_eq!(result.disposition, PairDisposition::EligibleForEnergyComparison);
        assert!(result.eligible_pair.is_some());
        assert!(result.issues.is_empty());
    }

    #[test]
    fn service_digest_mismatch_is_incomparable() {
        let base = complete_run("pair-1", "baseline-v1", TrialArm::Baseline, TrialOrder::First);
        let mut candidate = complete_run("pair-1", "candidate-v2", TrialArm::Candidate, TrialOrder::Second);
        if let TraceOutcome::Complete(ref mut e) = candidate.trace_outcome {
            e.service_digest = "c".repeat(64);
        }
        assert!(assess_pair(&base, &candidate).issues.contains(&PairingIssue::ServiceDigestMismatch));
    }

    #[test]
    fn environmental_condition_mismatch_is_incomparable() {
        let base = complete_run("pair-1", "baseline-v1", TrialArm::Baseline, TrialOrder::First);
        let mut candidate = complete_run("pair-1", "candidate-v2", TrialArm::Candidate, TrialOrder::Second);
        if let TraceOutcome::Complete(ref mut e) = candidate.trace_outcome {
            e.condition_digest = "different-condition".to_owned();
        }
        assert!(assess_pair(&base, &candidate).issues.contains(&PairingIssue::ConditionMismatch));
    }

    #[test]
    fn boundary_mismatch_is_incomparable() {
        let base = complete_run("pair-1", "baseline-v1", TrialArm::Baseline, TrialOrder::First);
        let mut candidate = complete_run("pair-1", "candidate-v2", TrialArm::Candidate, TrialOrder::Second);
        if let TraceOutcome::Complete(ref mut e) = candidate.trace_outcome {
            e.energy_boundary_id = "shaft-power".to_owned();
        }
        assert!(assess_pair(&base, &candidate).issues.contains(&PairingIssue::EnergyBoundaryMismatch));
    }

    #[test]
    fn calibration_epoch_mismatch_is_incomparable() {
        let base = complete_run("pair-1", "baseline-v1", TrialArm::Baseline, TrialOrder::First);
        let mut candidate = complete_run("pair-1", "candidate-v2", TrialArm::Candidate, TrialOrder::Second);
        if let TraceOutcome::Complete(ref mut e) = candidate.trace_outcome {
            e.calibration_epoch_id = "meter-cal-epoch-B".to_owned();
        }
        assert!(assess_pair(&base, &candidate).issues.contains(&PairingIssue::CalibrationEpochMismatch));
    }

    #[test]
    fn uncertainty_method_mismatch_is_incomparable() {
        let base = complete_run("pair-1", "baseline-v1", TrialArm::Baseline, TrialOrder::First);
        let mut candidate = complete_run("pair-1", "candidate-v2", TrialArm::Candidate, TrialOrder::Second);
        if let TraceOutcome::Complete(ref mut e) = candidate.trace_outcome {
            e.uncertainty_method_id = "statistical-confidence-interval".to_owned();
        }
        assert!(assess_pair(&base, &candidate).issues.contains(&PairingIssue::UncertaintyMethodMismatch));
    }

    #[test]
    fn mission_mismatch_is_incomparable() {
        let base = complete_run("pair-1", "baseline-v1", TrialArm::Baseline, TrialOrder::First);
        let mut candidate = complete_run("pair-1", "candidate-v2", TrialArm::Candidate, TrialOrder::Second);
        if let TraceOutcome::Complete(ref mut e) = candidate.trace_outcome {
            e.mission_id = "mission-B".to_owned();
        }
        assert!(assess_pair(&base, &candidate).issues.contains(&PairingIssue::MissionMismatch));
    }

    #[test]
    fn campaign_mismatch_is_incomparable() {
        let base = complete_run("pair-1", "baseline-v1", TrialArm::Baseline, TrialOrder::First);
        let mut candidate = complete_run("pair-1", "candidate-v2", TrialArm::Candidate, TrialOrder::Second);
        candidate.campaign_id = "campaign-B".to_owned();
        assert!(assess_pair(&base, &candidate).issues.contains(&PairingIssue::CampaignMismatch));
    }

    #[test]
    fn pair_id_mismatch_is_incomparable() {
        let base = complete_run("pair-1", "baseline-v1", TrialArm::Baseline, TrialOrder::First);
        let candidate = complete_run("pair-2", "candidate-v2", TrialArm::Candidate, TrialOrder::Second);
        assert!(assess_pair(&base, &candidate).issues.contains(&PairingIssue::PairIdMismatch));
    }

    #[test]
    fn duplicate_candidate_id_is_incomparable() {
        let base = complete_run("pair-1", "same-id", TrialArm::Baseline, TrialOrder::First);
        let candidate = complete_run("pair-1", "same-id", TrialArm::Candidate, TrialOrder::Second);
        assert!(assess_pair(&base, &candidate).issues.contains(&PairingIssue::SameCandidateId));
    }

    #[test]
    fn same_order_is_rejected() {
        let base = complete_run("pair-1", "baseline-v1", TrialArm::Baseline, TrialOrder::First);
        let candidate = complete_run("pair-1", "candidate-v2", TrialArm::Candidate, TrialOrder::First);
        assert!(assess_pair(&base, &candidate).issues.contains(&PairingIssue::SameRunOrder));
    }

    #[test]
    fn incomplete_trace_is_not_eligible() {
        let base = complete_run("pair-1", "baseline-v1", TrialArm::Baseline, TrialOrder::First);
        let candidate = run(
            "pair-1",
            "candidate-v2",
            TrialArm::Candidate,
            TrialOrder::Second,
            TraceOutcome::Indeterminate(vec![]),
        );
        let result = assess_pair(&base, &candidate);
        assert_eq!(result.disposition, PairDisposition::Incomparable);
        assert!(result.issues.iter().any(|issue| matches!(issue, PairingIssue::TraceNotComplete { disposition: TraceDisposition::Indeterminate, .. })));
    }

    #[test]
    fn campaign_audit_counts_both_orders() {
        let runs = vec![
            complete_run("pair-1", "baseline-v1", TrialArm::Baseline, TrialOrder::First),
            complete_run("pair-1", "candidate-v2", TrialArm::Candidate, TrialOrder::Second),
            complete_run("pair-2", "baseline-v1", TrialArm::Baseline, TrialOrder::Second),
            complete_run("pair-2", "candidate-v2", TrialArm::Candidate, TrialOrder::First),
        ];
        let audit = audit_campaign("campaign-A", &runs);
        assert_eq!(audit.eligible_pair_count, 2);
        assert_eq!(audit.baseline_first_eligible_pairs, 1);
        assert_eq!(audit.candidate_first_eligible_pairs, 1);
        assert_eq!(audit.order_coverage, OrderCoverage::BothOrdersObserved);
        assert_eq!(audit.order_count_difference, 0);
    }

    #[test]
    fn single_order_campaign_is_not_counterbalanced() {
        let runs = vec![
            complete_run("pair-1", "baseline-v1", TrialArm::Baseline, TrialOrder::First),
            complete_run("pair-1", "candidate-v2", TrialArm::Candidate, TrialOrder::Second),
            complete_run("pair-2", "baseline-v1", TrialArm::Baseline, TrialOrder::First),
            complete_run("pair-2", "candidate-v2", TrialArm::Candidate, TrialOrder::Second),
        ];
        let audit = audit_campaign("campaign-A", &runs);
        assert_eq!(audit.order_coverage, OrderCoverage::SingleOrderOnly);
        assert_eq!(audit.order_count_difference, 2);
    }

    #[test]
    fn unpaired_runs_are_visible_in_audit() {
        let runs = vec![
            complete_run("pair-1", "baseline-v1", TrialArm::Baseline, TrialOrder::First),
            complete_run("pair-2", "candidate-v2", TrialArm::Candidate, TrialOrder::Second),
        ];
        let audit = audit_campaign("campaign-A", &runs);
        assert_eq!(audit.total_runs, 2);
        assert_eq!(audit.incomparable_pair_count, 2);
        assert_eq!(audit.ineligible_runs, 2);
    }

    #[test]
    fn order_coverage_is_assessed_per_exact_candidate_pair() {
        let runs = vec![
            complete_run("pair-1", "baseline-v1", TrialArm::Baseline, TrialOrder::First),
            complete_run("pair-1", "candidate-v2", TrialArm::Candidate, TrialOrder::Second),
            complete_run("pair-2", "baseline-v1", TrialArm::Baseline, TrialOrder::Second),
            complete_run("pair-2", "candidate-v3", TrialArm::Candidate, TrialOrder::First),
        ];
        let audit = audit_campaign("campaign-A", &runs);
        assert_eq!(audit.order_coverage, OrderCoverage::SingleOrderOnly);
        assert_eq!(audit.comparison_order_counts.len(), 2);
        assert!(audit.comparison_order_counts.iter().all(|counts| {
            counts.order_coverage == OrderCoverage::SingleOrderOnly
        }));
    }

    #[test]
    fn input_permutation_produces_same_audit() {
        let runs = vec![
            complete_run("pair-2", "baseline-v1", TrialArm::Baseline, TrialOrder::Second),
            complete_run("pair-2", "candidate-v2", TrialArm::Candidate, TrialOrder::First),
            complete_run("pair-1", "baseline-v1", TrialArm::Baseline, TrialOrder::First),
            complete_run("pair-1", "candidate-v2", TrialArm::Candidate, TrialOrder::Second),
        ];
        let reversed = runs.iter().cloned().rev().collect::<Vec<_>>();
        assert_eq!(audit_campaign("campaign-A", &runs), audit_campaign("campaign-A", &reversed));
    }

    #[test]
    fn invalid_and_indeterminate_counts_are_preserved() {
        let runs = vec![
            run("pair-1", "baseline-v1", TrialArm::Baseline, TrialOrder::First, TraceOutcome::Invalid(vec![])),
            run("pair-1", "candidate-v2", TrialArm::Candidate, TrialOrder::Second, TraceOutcome::Indeterminate(vec![])),
        ];
        let audit = audit_campaign("campaign-A", &runs);
        assert_eq!(audit.invalid_runs, 1);
        assert_eq!(audit.indeterminate_runs, 1);
        assert_eq!(audit.complete_runs, 0);
        assert_eq!(audit.total_runs, 2);
    }
}
