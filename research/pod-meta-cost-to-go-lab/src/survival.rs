use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CensorReason {
    BudgetExhausted,
    Timeout,
    InfrastructureInterrupted,
    ActionOutcomeUnknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CompetingReason {
    PermissionRevoked,
    ExplicitAbort,
    PolicyAbandoned,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerminalOutcome {
    SufficientProof,
    RightCensored(CensorReason),
    CompetingTerminal(CompetingReason),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimeToProofObservation {
    pub task_family: u32,
    pub observed_work_units: u64,
    pub terminal: TerminalOutcome,
}

impl TimeToProofObservation {
    pub fn new(
        task_family: u32,
        observed_work_units: u64,
        terminal: TerminalOutcome,
    ) -> Result<Self, SurvivalError> {
        if observed_work_units == 0 {
            return Err(SurvivalError::ZeroElapsedWork);
        }
        Ok(Self {
            task_family,
            observed_work_units,
            terminal,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurvivalError {
    EmptyCohort,
    NoObservationsForFamily,
    ZeroElapsedWork,
}

#[derive(Clone, Copy, Debug, Default)]
struct TimeCounts {
    completed: usize,
    competing: usize,
    right_censored: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurvivalPoint {
    pub work_units: u64,
    pub at_risk: usize,
    pub completed: usize,
    pub competing_terminal: usize,
    pub right_censored: usize,
    pub completion_hazard: f64,
    pub competing_hazard: f64,
    /// Product-limit survival for the completion cause, treating competing
    /// terminal events as censoring (the cause-specific net-survival view).
    pub net_survival: f64,
    /// All-cause event-free survival used by the Aalen-Johansen estimator.
    pub event_free_survival: f64,
    /// Cumulative incidence of sufficient proof in the presence of
    /// competing terminal events.
    pub completion_cumulative_incidence: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SurvivalSummary {
    pub task_family: Option<u32>,
    pub episodes: usize,
    pub completed: usize,
    pub right_censored: usize,
    pub competing_terminal: usize,
    pub net_completion_probability: f64,
    pub completion_cumulative_incidence: f64,
    pub median_net_work_units: Option<u64>,
    pub q90_net_work_units: Option<u64>,
    pub censor_counts: BTreeMap<CensorReason, usize>,
    pub competing_counts: BTreeMap<CompetingReason, usize>,
    pub curve: Vec<SurvivalPoint>,
}

impl SurvivalSummary {
    /// Estimated completion cumulative incidence by an observed work horizon.
    /// No extrapolation is performed beyond the final observed endpoint.
    pub fn completion_probability_within(&self, horizon_work_units: u64) -> f64 {
        self.curve
            .iter()
            .take_while(|point| point.work_units <= horizon_work_units)
            .last()
            .map(|point| point.completion_cumulative_incidence)
            .unwrap_or(0.0)
    }
}

/// Non-parametric time-to-sufficient-proof baseline.
///
/// This combines a cause-specific product-limit curve with an
/// Aalen-Johansen cumulative-incidence estimate. Right-censored observations
/// remain in each risk set through their observed endpoint but are never
/// counted as completion failures. Competing terminal events are kept
/// separate and reduce the all-cause event-free survival.
///
/// Interpretation requires censoring to be independent of latent completion
/// time conditional on the declared cohort. Policy-induced or otherwise
/// informative censoring violates that assumption; this function does not
/// correct for informative censoring or identify a counterfactual completion
/// time after a competing terminal event.
pub fn estimate_time_to_proof(
    observations: &[TimeToProofObservation],
    task_family: Option<u32>,
) -> Result<SurvivalSummary, SurvivalError> {
    if observations.is_empty() {
        return Err(SurvivalError::EmptyCohort);
    }

    let selected = observations
        .iter()
        .copied()
        .filter(|observation| {
            task_family.is_none_or(|family| observation.task_family == family)
        })
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Err(SurvivalError::NoObservationsForFamily);
    }
    if selected
        .iter()
        .any(|observation| observation.observed_work_units == 0)
    {
        return Err(SurvivalError::ZeroElapsedWork);
    }

    let mut by_time = BTreeMap::<u64, TimeCounts>::new();
    let mut censor_counts = BTreeMap::<CensorReason, usize>::new();
    let mut competing_counts = BTreeMap::<CompetingReason, usize>::new();
    let mut completed = 0usize;
    let mut right_censored = 0usize;
    let mut competing_terminal = 0usize;

    for observation in &selected {
        let counts = by_time.entry(observation.observed_work_units).or_default();
        match observation.terminal {
            TerminalOutcome::SufficientProof => {
                counts.completed += 1;
                completed += 1;
            }
            TerminalOutcome::RightCensored(reason) => {
                counts.right_censored += 1;
                right_censored += 1;
                *censor_counts.entry(reason).or_default() += 1;
            }
            TerminalOutcome::CompetingTerminal(reason) => {
                counts.competing += 1;
                competing_terminal += 1;
                *competing_counts.entry(reason).or_default() += 1;
            }
        }
    }

    let mut at_risk = selected.len();
    let mut net_survival = 1.0;
    let mut event_free_survival = 1.0;
    let mut completion_cumulative_incidence = 0.0;
    let mut curve = Vec::with_capacity(by_time.len());

    for (work_units, counts) in by_time {
        if at_risk == 0 {
            break;
        }
        let completion_hazard = counts.completed as f64 / at_risk as f64;
        let competing_hazard = counts.competing as f64 / at_risk as f64;

        completion_cumulative_incidence += event_free_survival * completion_hazard;
        net_survival *= 1.0 - completion_hazard;
        event_free_survival *= 1.0 - completion_hazard - competing_hazard;

        curve.push(SurvivalPoint {
            work_units,
            at_risk,
            completed: counts.completed,
            competing_terminal: counts.competing,
            right_censored: counts.right_censored,
            completion_hazard,
            competing_hazard,
            net_survival,
            event_free_survival,
            completion_cumulative_incidence,
        });

        at_risk = at_risk.saturating_sub(
            counts.completed + counts.competing + counts.right_censored,
        );
    }

    let median_net_work_units = net_quantile(&curve, 0.50);
    let q90_net_work_units = net_quantile(&curve, 0.90);
    Ok(SurvivalSummary {
        task_family,
        episodes: selected.len(),
        completed,
        right_censored,
        competing_terminal,
        net_completion_probability: 1.0 - net_survival,
        completion_cumulative_incidence,
        median_net_work_units,
        q90_net_work_units,
        censor_counts,
        competing_counts,
        curve,
    })
}

fn net_quantile(curve: &[SurvivalPoint], quantile: f64) -> Option<u64> {
    let survival_threshold = 1.0 - quantile;
    curve
        .iter()
        .find(|point| point.net_survival <= survival_threshold)
        .map(|point| point.work_units)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(
        family: u32,
        work: u64,
        terminal: TerminalOutcome,
    ) -> TimeToProofObservation {
        TimeToProofObservation::new(family, work, terminal).unwrap()
    }

    #[test]
    fn right_censoring_is_not_counted_as_a_completion_failure() {
        let observations = [
            observation(1, 1, TerminalOutcome::SufficientProof),
            observation(
                1,
                2,
                TerminalOutcome::RightCensored(CensorReason::BudgetExhausted),
            ),
            observation(1, 3, TerminalOutcome::SufficientProof),
        ];
        let summary = estimate_time_to_proof(&observations, Some(1)).unwrap();

        assert_eq!(summary.episodes, 3);
        assert_eq!(summary.completed, 2);
        assert_eq!(summary.right_censored, 1);
        assert_eq!(summary.curve.len(), 3);
        assert!((summary.curve[1].net_survival - (2.0 / 3.0)).abs() < 1.0e-12);
        assert_eq!(summary.median_net_work_units, Some(3));
        assert_eq!(summary.q90_net_work_units, Some(3));
        assert!((summary.completion_cumulative_incidence - 1.0).abs() < 1.0e-12);
        assert_eq!(
            summary.censor_counts.get(&CensorReason::BudgetExhausted),
            Some(&1)
        );
    }

    #[test]
    fn competing_terminals_reduce_cumulative_incidence_without_becoming_completions() {
        let observations = [
            observation(2, 1, TerminalOutcome::SufficientProof),
            observation(
                2,
                2,
                TerminalOutcome::CompetingTerminal(CompetingReason::PermissionRevoked),
            ),
            observation(
                2,
                3,
                TerminalOutcome::RightCensored(CensorReason::BudgetExhausted),
            ),
        ];
        let summary = estimate_time_to_proof(&observations, Some(2)).unwrap();

        assert_eq!(summary.completed, 1);
        assert_eq!(summary.competing_terminal, 1);
        assert_eq!(summary.right_censored, 1);
        assert!((summary.completion_cumulative_incidence - (1.0 / 3.0)).abs() < 1.0e-12);
        assert!((summary.net_completion_probability - (1.0 / 3.0)).abs() < 1.0e-12);
        assert_eq!(summary.median_net_work_units, None);
        assert_eq!(summary.q90_net_work_units, None);
        assert_eq!(
            summary
                .competing_counts
                .get(&CompetingReason::PermissionRevoked),
            Some(&1)
        );
    }

    #[test]
    fn family_filtering_prevents_cross_family_pooling() {
        let observations = [
            observation(10, 1, TerminalOutcome::SufficientProof),
            observation(
                11,
                1,
                TerminalOutcome::RightCensored(CensorReason::Timeout),
            ),
        ];
        let success = estimate_time_to_proof(&observations, Some(10)).unwrap();
        let timeout = estimate_time_to_proof(&observations, Some(11)).unwrap();

        assert_eq!(success.episodes, 1);
        assert_eq!(success.median_net_work_units, Some(1));
        assert_eq!(timeout.episodes, 1);
        assert_eq!(timeout.median_net_work_units, None);
        assert_eq!(timeout.completion_probability_within(10), 0.0);
    }

    #[test]
    fn quantiles_are_unavailable_when_the_observed_tail_does_not_support_them() {
        let observations = [
            observation(3, 1, TerminalOutcome::SufficientProof),
            observation(
                3,
                2,
                TerminalOutcome::RightCensored(CensorReason::InfrastructureInterrupted),
            ),
            observation(
                3,
                3,
                TerminalOutcome::CompetingTerminal(CompetingReason::ExplicitAbort),
            ),
        ];
        let summary = estimate_time_to_proof(&observations, Some(3)).unwrap();

        assert_eq!(summary.median_net_work_units, None);
        assert_eq!(summary.q90_net_work_units, None);
        assert!((summary.completion_probability_within(1) - (1.0 / 3.0)).abs() < 1.0e-12);
        assert!((summary.completion_probability_within(100) - (1.0 / 3.0)).abs() < 1.0e-12);
    }

    #[test]
    fn censor_reasons_remain_distinguishable() {
        let observations = [
            observation(
                4,
                2,
                TerminalOutcome::RightCensored(CensorReason::BudgetExhausted),
            ),
            observation(4, 3, TerminalOutcome::RightCensored(CensorReason::Timeout)),
            observation(
                4,
                4,
                TerminalOutcome::RightCensored(CensorReason::ActionOutcomeUnknown),
            ),
        ];
        let summary = estimate_time_to_proof(&observations, Some(4)).unwrap();

        assert_eq!(summary.right_censored, 3);
        assert_eq!(summary.censor_counts.len(), 3);
        assert_eq!(summary.completed, 0);
        assert_eq!(summary.net_completion_probability, 0.0);
    }

    #[test]
    fn empty_cohorts_and_zero_duration_are_rejected() {
        assert_eq!(
            estimate_time_to_proof(&[], None),
            Err(SurvivalError::EmptyCohort)
        );
        assert_eq!(
            TimeToProofObservation::new(1, 0, TerminalOutcome::SufficientProof),
            Err(SurvivalError::ZeroElapsedWork)
        );
        let observations = [observation(1, 1, TerminalOutcome::SufficientProof)];
        assert_eq!(
            estimate_time_to_proof(&observations, Some(2)),
            Err(SurvivalError::NoObservationsForFamily)
        );
    }
}
