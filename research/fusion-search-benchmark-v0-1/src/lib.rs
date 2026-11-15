use std::collections::BTreeSet;

pub const GRID_SIDE: u8 = 64;
pub const CANDIDATE_COUNT: usize = GRID_SIDE as usize * GRID_SIDE as usize;
const CENTER: i32 = 24;
const FEASIBLE_RADIUS: i32 = 22;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scenario {
    Smooth,
    RareFeasibleIsland,
}

impl Scenario {
    pub fn name(self) -> &'static str {
        match self {
            Self::Smooth => "smooth",
            Self::RareFeasibleIsland => "rare_island",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Policy {
    Exhaustive,
    LowFidelityThreshold,
    UncertaintyGuarded,
}

impl Policy {
    pub fn name(self) -> &'static str {
        match self {
            Self::Exhaustive => "exhaustive",
            Self::LowFidelityThreshold => "low_fidelity",
            Self::UncertaintyGuarded => "uncertainty_guarded",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Disposition {
    NotEvaluated,
    Passed,
    Rejected,
    Inconclusive,
    Failed,
    NonConverged,
    Unsupported,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Candidate {
    pub id: u32,
    pub x: u8,
    pub y: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReferenceResult {
    pub feasible: bool,
    pub service_a: i32,
    pub service_b: i32,
}

impl ReferenceResult {
    pub fn objective_sum(self) -> i32 {
        self.service_a + self.service_b
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CandidateRecord {
    pub candidate: Candidate,
    pub reference: ReferenceResult,
    pub analytical: Disposition,
    pub high_fidelity: Disposition,
}

#[derive(Clone, Debug)]
pub struct Scorecard {
    pub candidate_universe_coverage: f64,
    pub exact_reference_metrics: bool,
    pub generated_records: usize,
    pub screened_candidates: usize,
    pub reference_feasible_candidates: usize,
    pub feasible_recall: Option<f64>,
    pub reference_pareto_candidates: usize,
    pub pareto_recall: Option<f64>,
    pub usable_pareto_recall: Option<f64>,
    pub best_objective_regret: Option<i32>,
    pub high_fidelity_attempts: usize,
    pub high_fidelity_converged: usize,
    pub high_fidelity_failed: usize,
    pub high_fidelity_non_converged: usize,
    pub high_fidelity_unsupported: usize,
    pub high_fidelity_inconclusive: usize,
}

pub fn candidate_universe() -> Vec<Candidate> {
    let mut candidates = Vec::with_capacity(CANDIDATE_COUNT);
    for x in 0..GRID_SIDE {
        for y in 0..GRID_SIDE {
            candidates.push(Candidate {
                id: u32::from(x) * u32::from(GRID_SIDE) + u32::from(y),
                x,
                y,
            });
        }
    }
    candidates
}

fn manhattan_distance(candidate: Candidate) -> i32 {
    (i32::from(candidate.x) - CENTER).abs() + (i32::from(candidate.y) - CENTER).abs()
}

fn low_fidelity_margin(candidate: Candidate) -> i32 {
    FEASIBLE_RADIUS - manhattan_distance(candidate)
}

pub fn reference_result(scenario: Scenario, candidate: Candidate) -> ReferenceResult {
    let ordinary_feasible = manhattan_distance(candidate) <= FEASIBLE_RADIUS;
    let rare_island =
        scenario == Scenario::RareFeasibleIsland && candidate.x == 61 && candidate.y == 61;
    let feasible = ordinary_feasible || rare_island;

    let (service_a, service_b) = if rare_island {
        (1_600, 1_600)
    } else {
        let ax = (i32::from(candidate.x) - 44).abs();
        let ay = (i32::from(candidate.y) - 12).abs();
        let bx = (i32::from(candidate.x) - 12).abs();
        let by = (i32::from(candidate.y) - 44).abs();
        (1_000 - ax * ax - ay * ay, 1_000 - bx * bx - by * by)
    };

    ReferenceResult {
        feasible,
        service_a,
        service_b,
    }
}

fn uncertainty_radius(scenario: Scenario, candidate: Candidate) -> i32 {
    if scenario == Scenario::RareFeasibleIsland && candidate.x >= 60 && candidate.y >= 60 {
        64
    } else {
        0
    }
}

fn selected_by_policy(scenario: Scenario, policy: Policy, candidate: Candidate) -> bool {
    match policy {
        Policy::Exhaustive => true,
        Policy::LowFidelityThreshold => low_fidelity_margin(candidate) >= 0,
        Policy::UncertaintyGuarded => {
            low_fidelity_margin(candidate) + uncertainty_radius(scenario, candidate) >= 0
        }
    }
}

fn synthetic_solver_disposition(candidate: Candidate) -> Disposition {
    if candidate.id.is_multiple_of(307) {
        Disposition::Inconclusive
    } else if candidate.id.is_multiple_of(211) {
        Disposition::Failed
    } else if candidate.id.is_multiple_of(113) {
        Disposition::NonConverged
    } else if candidate.id.is_multiple_of(509) {
        Disposition::Unsupported
    } else {
        Disposition::Passed
    }
}

pub fn evaluate_funnel(scenario: Scenario, policy: Policy) -> Vec<CandidateRecord> {
    candidate_universe()
        .into_iter()
        .map(|candidate| {
            let selected = selected_by_policy(scenario, policy, candidate);
            CandidateRecord {
                candidate,
                reference: reference_result(scenario, candidate),
                analytical: if selected {
                    Disposition::Passed
                } else {
                    Disposition::Rejected
                },
                high_fidelity: if selected {
                    synthetic_solver_disposition(candidate)
                } else {
                    Disposition::NotEvaluated
                },
            }
        })
        .collect()
}

fn dominates(left: ReferenceResult, right: ReferenceResult) -> bool {
    left.service_a >= right.service_a
        && left.service_b >= right.service_b
        && (left.service_a > right.service_a || left.service_b > right.service_b)
}

fn is_complete_universe(records: &[CandidateRecord]) -> bool {
    if records.len() != CANDIDATE_COUNT {
        return false;
    }
    let identities: BTreeSet<u32> = records.iter().map(|record| record.candidate.id).collect();
    identities.len() == CANDIDATE_COUNT
        && identities
            .iter()
            .all(|identity| (*identity as usize) < CANDIDATE_COUNT)
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

pub fn summarize(records: &[CandidateRecord], reference_labels_exhaustive: bool) -> Scorecard {
    let identities: BTreeSet<u32> = records.iter().map(|record| record.candidate.id).collect();
    let candidate_universe_coverage = identities.len() as f64 / CANDIDATE_COUNT as f64;
    let exact_reference_metrics = reference_labels_exhaustive && is_complete_universe(records);

    let screened_candidates = records
        .iter()
        .filter(|record| record.analytical == Disposition::Passed)
        .count();
    let reference_feasible_candidates = records
        .iter()
        .filter(|record| record.reference.feasible)
        .count();

    let frontier: Vec<&CandidateRecord> = records
        .iter()
        .filter(|record| record.reference.feasible)
        .filter(|candidate| {
            !records.iter().any(|other| {
                other.reference.feasible
                    && other.candidate.id != candidate.candidate.id
                    && dominates(other.reference, candidate.reference)
            })
        })
        .collect();

    let retained_feasible = records
        .iter()
        .filter(|record| record.reference.feasible && record.analytical == Disposition::Passed)
        .count();
    let retained_frontier = frontier
        .iter()
        .filter(|record| record.analytical == Disposition::Passed)
        .count();
    let usable_frontier = frontier
        .iter()
        .filter(|record| record.high_fidelity == Disposition::Passed)
        .count();

    let best_reference_sum = records
        .iter()
        .filter(|record| record.reference.feasible)
        .map(|record| record.reference.objective_sum())
        .max();
    let best_observed_sum = records
        .iter()
        .filter(|record| record.reference.feasible && record.high_fidelity == Disposition::Passed)
        .map(|record| record.reference.objective_sum())
        .max();

    let high_fidelity_attempts = records
        .iter()
        .filter(|record| record.high_fidelity != Disposition::NotEvaluated)
        .count();

    Scorecard {
        candidate_universe_coverage,
        exact_reference_metrics,
        generated_records: records.len(),
        screened_candidates,
        reference_feasible_candidates,
        feasible_recall: if exact_reference_metrics {
            ratio(retained_feasible, reference_feasible_candidates)
        } else {
            None
        },
        reference_pareto_candidates: frontier.len(),
        pareto_recall: if exact_reference_metrics {
            ratio(retained_frontier, frontier.len())
        } else {
            None
        },
        usable_pareto_recall: if exact_reference_metrics {
            ratio(usable_frontier, frontier.len())
        } else {
            None
        },
        best_objective_regret: if exact_reference_metrics {
            best_reference_sum
                .zip(best_observed_sum)
                .map(|(best, found)| best - found)
        } else {
            None
        },
        high_fidelity_attempts,
        high_fidelity_converged: records
            .iter()
            .filter(|record| record.high_fidelity == Disposition::Passed)
            .count(),
        high_fidelity_failed: records
            .iter()
            .filter(|record| record.high_fidelity == Disposition::Failed)
            .count(),
        high_fidelity_non_converged: records
            .iter()
            .filter(|record| record.high_fidelity == Disposition::NonConverged)
            .count(),
        high_fidelity_unsupported: records
            .iter()
            .filter(|record| record.high_fidelity == Disposition::Unsupported)
            .count(),
        high_fidelity_inconclusive: records
            .iter()
            .filter(|record| record.high_fidelity == Disposition::Inconclusive)
            .count(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_universe_has_4096_unique_stable_identities() {
        let candidates = candidate_universe();
        let identities: BTreeSet<u32> = candidates.iter().map(|candidate| candidate.id).collect();
        assert_eq!(candidates.len(), CANDIDATE_COUNT);
        assert_eq!(identities.len(), CANDIDATE_COUNT);
        assert_eq!(candidates.first().unwrap().id, 0);
        assert_eq!(candidates.last().unwrap().id, 4_095);
    }

    #[test]
    fn count_recall_does_not_hide_a_lost_pareto_best_design() {
        let records = evaluate_funnel(Scenario::RareFeasibleIsland, Policy::LowFidelityThreshold);
        let scorecard = summarize(&records, true);

        assert!(scorecard.feasible_recall.unwrap() >= 0.99);
        assert_eq!(scorecard.reference_pareto_candidates, 1);
        assert_eq!(scorecard.pareto_recall, Some(0.0));
        assert!(scorecard.best_objective_regret.unwrap() > 0);
    }

    #[test]
    fn uncertainty_guard_recovers_rare_frontier_with_fewer_calls_than_exhaustive() {
        let guarded = evaluate_funnel(Scenario::RareFeasibleIsland, Policy::UncertaintyGuarded);
        let exhaustive = evaluate_funnel(Scenario::RareFeasibleIsland, Policy::Exhaustive);
        let guarded_score = summarize(&guarded, true);
        let exhaustive_score = summarize(&exhaustive, true);

        assert_eq!(guarded_score.pareto_recall, Some(1.0));
        assert_eq!(guarded_score.best_objective_regret, Some(0));
        assert!(guarded_score.high_fidelity_attempts < exhaustive_score.high_fidelity_attempts);
    }

    #[test]
    fn solver_failures_and_non_convergence_remain_explicit() {
        let records = evaluate_funnel(Scenario::Smooth, Policy::Exhaustive);
        let scorecard = summarize(&records, true);

        assert_eq!(scorecard.high_fidelity_attempts, CANDIDATE_COUNT);
        assert!(scorecard.high_fidelity_failed > 0);
        assert!(scorecard.high_fidelity_non_converged > 0);
        assert!(scorecard.high_fidelity_unsupported > 0);
        assert!(scorecard.high_fidelity_inconclusive > 0);
    }

    #[test]
    fn partial_reference_labels_never_emit_exact_recall_or_regret() {
        let records = evaluate_funnel(Scenario::Smooth, Policy::LowFidelityThreshold);
        let scorecard = summarize(&records, false);

        assert!(!scorecard.exact_reference_metrics);
        assert!(scorecard.feasible_recall.is_none());
        assert!(scorecard.pareto_recall.is_none());
        assert!(scorecard.best_objective_regret.is_none());
    }

    #[test]
    fn incomplete_candidate_universe_never_emits_exact_recall_or_regret() {
        let records = evaluate_funnel(Scenario::Smooth, Policy::LowFidelityThreshold);
        let scorecard = summarize(&records[..records.len() - 1], true);

        assert!(!scorecard.exact_reference_metrics);
        assert!(scorecard.candidate_universe_coverage < 1.0);
        assert!(scorecard.feasible_recall.is_none());
        assert!(scorecard.pareto_recall.is_none());
        assert!(scorecard.best_objective_regret.is_none());
    }

    #[test]
    fn smooth_proxy_case_retains_all_feasible_and_pareto_candidates() {
        let records = evaluate_funnel(Scenario::Smooth, Policy::LowFidelityThreshold);
        let scorecard = summarize(&records, true);

        assert_eq!(scorecard.feasible_recall, Some(1.0));
        assert_eq!(scorecard.pareto_recall, Some(1.0));
    }
}
