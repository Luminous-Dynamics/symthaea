//! Greedy R-I / R-IDeA design-portfolio selection with sequential representativeness updates.
//!
//! Build from repository root:
//!   rustc --edition=2024 --test scripts/research/sdcb_ridea_portfolio.rs -o /tmp/sdcb-ridea-portfolio
//!
//! Important boundary: EIG and current/proxy predictions are frozen candidate
//! inputs. This is a greedy design-only batch planner. It recomputes MMD
//! representativeness after each selected design, but it does not update a
//! posterior from observed outcomes or recompute a joint EIG.

#[path = "sdcb_ridea.rs"]
mod ridea_reference;

use ridea_reference::{
    mmd_squared_rbf, rank_ridea_candidates, AcquisitionKind, RideaCandidate, RideaConfig,
    RideaError, RideaScore,
};

const EPS: f64 = 1e-12;
const MAX_PORTFOLIO_CANDIDATES: usize = 16;
const MAX_PORTFOLIO_SAMPLES: usize = 128;
const MAX_PORTFOLIO_STEPS: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PortfolioError {
    NoCandidates,
    InvalidConfiguration,
    InvalidHistoryOrTarget,
    InvalidCandidate,
    DuplicateCandidateId,
    SearchSpaceTooLarge,
    Ridea(RideaError),
}

impl From<RideaError> for PortfolioError {
    fn from(value: RideaError) -> Self {
        Self::Ridea(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortfolioStopReason {
    TargetDistributionMatched,
    BudgetExhausted,
    NoPositiveAcquisition,
    CandidatePoolExhausted,
    ExperimentLimitReached,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PortfolioStep {
    pub candidate_id: String,
    pub cost_before: u64,
    pub cost_after: u64,
    pub mmd2_before: f64,
    pub mmd2_after: f64,
    pub score: RideaScore,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RideaPortfolioResult {
    pub selected_candidate_ids: Vec<String>,
    pub steps: Vec<PortfolioStep>,
    pub initial_mmd2: f64,
    pub final_mmd2: f64,
    pub total_cost_units: u64,
    pub remaining_budget_units: u64,
    pub stop_reason: PortfolioStopReason,
}

fn validate_config(config: &RideaConfig) -> Result<(), PortfolioError> {
    if !config.representativeness_weight.is_finite()
        || !(0.0..=1.0).contains(&config.representativeness_weight)
        || !config.deamplification_threshold.is_finite()
        || config.deamplification_threshold < 0.0
        || !config.sigmoid_scale.is_finite()
        || config.sigmoid_scale <= 0.0
        || !config.rbf_sigma.is_finite()
        || config.rbf_sigma <= 0.0
    {
        return Err(PortfolioError::InvalidConfiguration);
    }
    Ok(())
}

fn validate_inputs(
    history: &[Vec<f64>],
    target: &[Vec<f64>],
    candidates: &[RideaCandidate],
    config: &RideaConfig,
) -> Result<usize, PortfolioError> {
    validate_config(config)?;
    if history.is_empty() || target.is_empty() {
        return Err(PortfolioError::InvalidHistoryOrTarget);
    }
    if history.len() > MAX_PORTFOLIO_SAMPLES
        || target.len() > MAX_PORTFOLIO_SAMPLES
        || history.len().saturating_add(candidates.len()) > MAX_PORTFOLIO_SAMPLES
    {
        return Err(PortfolioError::SearchSpaceTooLarge);
    }
    let dimension = history[0].len();
    if dimension == 0
        || history.iter().chain(target).any(|row| {
            row.len() != dimension || row.iter().any(|x| !x.is_finite())
        })
    {
        return Err(PortfolioError::InvalidHistoryOrTarget);
    }
    if candidates.is_empty() {
        return Err(PortfolioError::NoCandidates);
    }
    if candidates.len() > MAX_PORTFOLIO_CANDIDATES {
        return Err(PortfolioError::SearchSpaceTooLarge);
    }
    for (index, candidate) in candidates.iter().enumerate() {
        if candidate.id.trim().is_empty()
            || candidate.cost_units == 0
            || candidate.design.len() != dimension
            || candidate.design.iter().any(|x| !x.is_finite())
            || !candidate.expected_information_gain.is_finite()
            || candidate.expected_information_gain < 0.0
            || !candidate.current_prediction.is_finite()
            || !candidate.proxy_prediction.is_finite()
        {
            return Err(PortfolioError::InvalidCandidate);
        }
        if candidates[..index].iter().any(|old| old.id == candidate.id) {
            return Err(PortfolioError::DuplicateCandidateId);
        }
    }
    Ok(dimension)
}

fn acquisition_is_positive(score: &RideaScore, kind: AcquisitionKind) -> bool {
    match kind {
        AcquisitionKind::RepresentativeInformative => {
            score.representative_informative > EPS
        }
        AcquisitionKind::RepresentativeInformativeDeAmplifying => score.ridea_score > EPS,
    }
}

/// Select a greedy design portfolio. After every selected design, the history
/// is updated and all remaining candidates are rescored against the new MMD²
/// representativeness baseline.
///
/// Candidate EIG and proxy predictions are frozen for the full portfolio.
/// This intentionally isolates the batch-coverage effect; an adaptive
/// sequential policy must replace these with outcome-conditioned model
/// updates and recomputed joint/posterior information gain.
pub fn greedy_ridea_portfolio(
    initial_history: &[Vec<f64>],
    target_distribution: &[Vec<f64>],
    candidates: &[RideaCandidate],
    config: &RideaConfig,
    kind: AcquisitionKind,
    budget_units: u64,
    max_experiments: usize,
) -> Result<RideaPortfolioResult, PortfolioError> {
    if max_experiments > MAX_PORTFOLIO_STEPS {
        return Err(PortfolioError::SearchSpaceTooLarge);
    }
    validate_inputs(initial_history, target_distribution, candidates, config)?;

    let initial_mmd2 =
        mmd_squared_rbf(initial_history, target_distribution, config.rbf_sigma)?;
    let mut history = initial_history.to_vec();
    let mut remaining = candidates.to_vec();
    let mut selected_candidate_ids = Vec::new();
    let mut steps = Vec::new();
    let mut total_cost_units = 0u64;
    let mut current_mmd2 = initial_mmd2;

    let mut stop_reason = if current_mmd2 <= EPS {
        PortfolioStopReason::TargetDistributionMatched
    } else if budget_units == 0 {
        PortfolioStopReason::BudgetExhausted
    } else if max_experiments == 0 {
        PortfolioStopReason::ExperimentLimitReached
    } else {
        PortfolioStopReason::CandidatePoolExhausted
    };

    while current_mmd2 > EPS
        && total_cost_units < budget_units
        && steps.len() < max_experiments
        && !remaining.is_empty()
    {
        let remaining_budget = budget_units - total_cost_units;
        let affordable: Vec<RideaCandidate> = remaining
            .iter()
            .filter(|candidate| candidate.cost_units <= remaining_budget)
            .cloned()
            .collect();

        if affordable.is_empty() {
            stop_reason = PortfolioStopReason::BudgetExhausted;
            break;
        }

        // The existing scorer recalculates each candidate's R-I/R-IDeA value
        // against the current selected-design history.
        let ranked = rank_ridea_candidates(
            &history,
            target_distribution,
            &affordable,
            config,
            kind,
        )?;

        let Some(selected_score) = ranked
            .into_iter()
            .find(|score| acquisition_is_positive(score, kind))
        else {
            stop_reason = PortfolioStopReason::NoPositiveAcquisition;
            break;
        };

        let selected_index = remaining
            .iter()
            .position(|candidate| candidate.id == selected_score.candidate_id)
            .ok_or(PortfolioError::InvalidCandidate)?;
        let selected = remaining.remove(selected_index);

        let mmd2_before = current_mmd2;
        history.push(selected.design.clone());
        current_mmd2 = mmd_squared_rbf(
            &history,
            target_distribution,
            config.rbf_sigma,
        )?;

        let cost_before = total_cost_units;
        total_cost_units = total_cost_units
            .checked_add(selected.cost_units)
            .ok_or(PortfolioError::SearchSpaceTooLarge)?;

        selected_candidate_ids.push(selected.id.clone());
        steps.push(PortfolioStep {
            candidate_id: selected.id,
            cost_before,
            cost_after: total_cost_units,
            mmd2_before,
            mmd2_after: current_mmd2,
            score: selected_score,
        });

        stop_reason = if current_mmd2 <= EPS {
            PortfolioStopReason::TargetDistributionMatched
        } else if total_cost_units >= budget_units {
            PortfolioStopReason::BudgetExhausted
        } else if steps.len() >= max_experiments {
            PortfolioStopReason::ExperimentLimitReached
        } else if remaining.is_empty() {
            PortfolioStopReason::CandidatePoolExhausted
        } else {
            PortfolioStopReason::NoPositiveAcquisition
        };
    }

    Ok(RideaPortfolioResult {
        selected_candidate_ids,
        steps,
        initial_mmd2,
        final_mmd2: current_mmd2,
        total_cost_units,
        remaining_budget_units: budget_units - total_cost_units,
        stop_reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> RideaConfig {
        RideaConfig {
            representativeness_weight: 0.5,
            deamplification_threshold: 0.5,
            sigmoid_scale: 0.5,
            rbf_sigma: 1.0,
        }
    }

    fn history() -> Vec<Vec<f64>> {
        vec![vec![0.0], vec![3.0]]
    }

    fn target() -> Vec<Vec<f64>> {
        vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]
    }

    fn candidates() -> Vec<RideaCandidate> {
        vec![
            RideaCandidate {
                id: "cover-a".into(),
                design: vec![1.0],
                expected_information_gain: 0.70,
                cost_units: 1,
                current_prediction: 2.0,
                proxy_prediction: 0.5,
            },
            RideaCandidate {
                id: "cover-b".into(),
                design: vec![2.0],
                expected_information_gain: 0.65,
                cost_units: 1,
                current_prediction: 1.7,
                proxy_prediction: 0.5,
            },
            RideaCandidate {
                id: "high-eig-extrapolation".into(),
                design: vec![4.0],
                expected_information_gain: 1.30,
                cost_units: 1,
                current_prediction: 0.9,
                proxy_prediction: 0.8,
            },
            RideaCandidate {
                id: "redundant-low-eig".into(),
                design: vec![0.5],
                expected_information_gain: 0.10,
                cost_units: 1,
                current_prediction: 0.9,
                proxy_prediction: 0.8,
            },
        ]
    }

    #[test]
    fn portfolio_updates_mmd_after_each_selected_design() {
        let result = greedy_ridea_portfolio(
            &history(), &target(), &candidates(), &config(),
            AcquisitionKind::RepresentativeInformativeDeAmplifying, 2, 2,
        ).unwrap();

        assert_eq!(result.selected_candidate_ids, vec!["cover-a", "cover-b"]);
        assert!(result.steps[0].mmd2_after < result.steps[0].mmd2_before);
        assert!(result.steps[1].mmd2_after < result.steps[1].mmd2_before);
        assert!(result.final_mmd2.abs() < EPS);
        assert_eq!(result.stop_reason, PortfolioStopReason::TargetDistributionMatched);
    }

    #[test]
    fn a_high_eig_outlier_does_not_win_over_representative_designs() {
        let result = greedy_ridea_portfolio(
            &history(), &target(), &candidates(), &config(),
            AcquisitionKind::RepresentativeInformativeDeAmplifying, 2, 1,
        ).unwrap();
        assert_eq!(result.selected_candidate_ids, vec!["cover-a"]);
        assert!(result.steps[0].score.expected_information_gain < 1.30);
        assert_eq!(result.stop_reason, PortfolioStopReason::ExperimentLimitReached);
    }

    #[test]
    fn portfolio_abstains_when_all_acquisition_scores_are_zero() {
        let mut candidate = candidates()[0].clone();
        candidate.expected_information_gain = 0.0;
        let result = greedy_ridea_portfolio(
            &history(), &target(), &[candidate], &config(),
            AcquisitionKind::RepresentativeInformativeDeAmplifying, 2, 2,
        ).unwrap();
        assert!(result.steps.is_empty());
        assert_eq!(result.stop_reason, PortfolioStopReason::NoPositiveAcquisition);
    }

    #[test]
    fn portfolio_respects_cost_budget() {
        let result = greedy_ridea_portfolio(
            &history(), &target(), &candidates(), &config(),
            AcquisitionKind::RepresentativeInformativeDeAmplifying, 1, 4,
        ).unwrap();
        assert_eq!(result.total_cost_units, 1);
        assert_eq!(result.remaining_budget_units, 0);
        assert_eq!(result.stop_reason, PortfolioStopReason::BudgetExhausted);
    }

    #[test]
    fn zero_initial_mmd_stops_without_undefined_ratio() {
        let already_covered = target();
        let result = greedy_ridea_portfolio(
            &already_covered, &target(), &candidates(), &config(),
            AcquisitionKind::RepresentativeInformativeDeAmplifying, 4, 4,
        ).unwrap();
        assert!(result.steps.is_empty());
        assert_eq!(result.stop_reason, PortfolioStopReason::TargetDistributionMatched);
    }

    #[test]
    fn zero_budget_and_zero_experiment_limit_are_explicit() {
        let by_budget = greedy_ridea_portfolio(
            &history(), &target(), &candidates(), &config(),
            AcquisitionKind::RepresentativeInformativeDeAmplifying, 0, 4,
        ).unwrap();
        assert_eq!(by_budget.stop_reason, PortfolioStopReason::BudgetExhausted);

        let by_limit = greedy_ridea_portfolio(
            &history(), &target(), &candidates(), &config(),
            AcquisitionKind::RepresentativeInformativeDeAmplifying, 4, 0,
        ).unwrap();
        assert_eq!(by_limit.stop_reason, PortfolioStopReason::ExperimentLimitReached);
    }

    #[test]
    fn duplicate_candidate_ids_are_rejected() {
        let mut values = candidates();
        values[1].id = values[0].id.clone();
        assert_eq!(
            greedy_ridea_portfolio(
                &history(), &target(), &values, &config(),
                AcquisitionKind::RepresentativeInformativeDeAmplifying, 4, 4,
            ),
            Err(PortfolioError::DuplicateCandidateId)
        );
    }

    #[test]
    fn incompatible_design_dimensions_are_rejected() {
        let mut values = candidates();
        values[0].design.push(3.0);
        assert_eq!(
            greedy_ridea_portfolio(
                &history(), &target(), &values, &config(),
                AcquisitionKind::RepresentativeInformativeDeAmplifying, 4, 4,
            ),
            Err(PortfolioError::InvalidCandidate)
        );
    }

    #[test]
    fn non_finite_design_values_are_rejected() {
        let mut values = candidates();
        values[0].design[0] = f64::NAN;
        assert_eq!(
            greedy_ridea_portfolio(
                &history(), &target(), &values, &config(),
                AcquisitionKind::RepresentativeInformativeDeAmplifying, 4, 4,
            ),
            Err(PortfolioError::InvalidCandidate)
        );
    }

    #[test]
    fn invalid_rbf_or_deamplification_config_is_rejected() {
        let mut cfg = config();
        cfg.rbf_sigma = 0.0;
        assert_eq!(
            greedy_ridea_portfolio(
                &history(), &target(), &candidates(), &cfg,
                AcquisitionKind::RepresentativeInformativeDeAmplifying, 4, 4,
            ),
            Err(PortfolioError::InvalidConfiguration)
        );
    }

    #[test]
    fn initial_distribution_distance_is_preserved_in_result() {
        let result = greedy_ridea_portfolio(
            &history(), &target(), &candidates(), &config(),
            AcquisitionKind::RepresentativeInformativeDeAmplifying, 2, 2,
        ).unwrap();
        assert!((result.initial_mmd2 - 0.14173847129404793).abs() < 1e-12);
        assert!(result.final_mmd2.abs() < EPS);
        assert_eq!(result.total_cost_units, 2);
        assert_eq!(result.remaining_budget_units, 0);
    }

    #[test]
    fn candidate_pool_cap_is_enforced() {
        let mut values = candidates();
        while values.len() < MAX_PORTFOLIO_CANDIDATES + 1 {
            let mut next = candidates()[0].clone();
            next.id = format!("extra-{}", values.len());
            values.push(next);
        }
        assert_eq!(
            greedy_ridea_portfolio(
                &history(), &target(), &values, &config(),
                AcquisitionKind::RepresentativeInformativeDeAmplifying, 4, 4,
            ),
            Err(PortfolioError::SearchSpaceTooLarge)
        );
    }

    #[test]
    fn sample_and_step_caps_are_enforced() {
        let long_history = vec![vec![0.0]; MAX_PORTFOLIO_SAMPLES];
        assert_eq!(
            greedy_ridea_portfolio(
                &long_history, &target(), &[candidates()[0].clone()], &config(),
                AcquisitionKind::RepresentativeInformativeDeAmplifying, 4, 1,
            ),
            Err(PortfolioError::SearchSpaceTooLarge)
        );
        assert_eq!(
            greedy_ridea_portfolio(
                &history(), &target(), &candidates(), &config(),
                AcquisitionKind::RepresentativeInformativeDeAmplifying, 4, MAX_PORTFOLIO_STEPS + 1,
            ),
            Err(PortfolioError::SearchSpaceTooLarge)
        );
    }

    #[test]
    fn r_i_portfolio_variant_uses_the_same_updated_history() {
        let result = greedy_ridea_portfolio(
            &history(), &target(), &candidates(), &config(),
            AcquisitionKind::RepresentativeInformative, 2, 2,
        ).unwrap();
        assert_eq!(result.selected_candidate_ids, vec!["cover-a", "cover-b"]);
        assert!(result.final_mmd2.abs() < EPS);
    }
}
