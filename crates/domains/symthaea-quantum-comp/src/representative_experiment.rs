//! Representative, informative, and de-amplifying experiment-design ablations.
//!
//! This module implements a transparent R-I-D acquisition proxy inspired by,
//! but not equivalent to, the R-IDeA method from Tang, Sloman, and Kaski
//! (AISTATS 2026). It composes:
//! - decision informativeness from QARCH-005 EVSI per cost;
//! - representativeness as one minus total-variation distance between a
//!   preregistered target-regime distribution and the experiment's regime mix;
//! - a de-amplification proxy as one minus the absolute gap between validation
//!   error under the target distribution and under the experiment's sampling
//!   distribution.
//!
//! Validation errors are declared inputs from a validation artifact, not
//! evaluator-only truth. The module reports all components and ablations; it
//! does not claim that these proxies reproduce a published acquisition rule
//! or guarantee generalization under unrepresented shifts.

use crate::active_experiment::{ActiveExperimentProblem, ExperimentDesignError};
use std::error::Error;
use std::fmt;

/// Eligibility status imported from the upstream physical/evidence qualification gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentEligibilityStatus {
    /// Candidate has passed all preregistered eligibility checks for this benchmark.
    Eligible,
    /// No qualified optical/physical model is available.
    NotCalibrated,
    /// Calibration does not match the current hardware configuration.
    StaleCalibration,
    /// Requested operating point is outside the model's validated domain.
    OutOfDomain,
    /// Evidence is insufficient to establish eligibility.
    InsufficientEvidence,
}

/// Validation evidence for one experiment's operating-regime coverage.
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentRegimeProfile {
    /// Stable candidate experiment ID.
    pub experiment_id: String,
    /// Eligibility result from the upstream qualification gate.
    pub eligibility_status: ExperimentEligibilityStatus,
    /// Digest/immutable ID of the artifact supporting the eligibility result.
    pub eligibility_evidence_digest: String,
    /// Fraction of this experiment's measurement budget in each regime.
    /// The values are ordered exactly as RepresentativeExperimentContext::regime_labels.
    /// Must be empty when eligibility_status is not Eligible.
    pub sampling_regime_probabilities: Vec<f64>,
    /// Validation error estimate in each regime, normalized to [0, 1].
    /// These values must come from the declared validation artifact, not held-out truth.
    /// Must be empty when eligibility_status is not Eligible.
    pub validation_error_by_regime: Vec<f64>,
}

/// Candidate that failed the upstream eligibility gate and was excluded from rankings.
#[derive(Debug, Clone, PartialEq)]
pub struct RidaIneligibleCandidate {
    /// Stable candidate experiment ID.
    pub experiment_id: String,
    /// Exact rejection status supplied by the upstream gate.
    pub eligibility_status: ExperimentEligibilityStatus,
    /// Digest/immutable ID of the evidence supporting the rejection.
    pub eligibility_evidence_digest: String,
}

/// Target operating distribution and split provenance shared by candidate profiles.
#[derive(Debug, Clone, PartialEq)]
pub struct RepresentativeExperimentContext {
    /// Stable operating-regime labels in vector order.
    pub regime_labels: Vec<String>,
    /// Preregistered target-population probability for each regime.
    pub target_regime_probabilities: Vec<f64>,
    /// Digest or immutable ID of the validation artifact used for regime error estimates.
    pub validation_artifact_digest: String,
    /// Digest or immutable ID of the evaluator-only artifact used downstream.
    pub evaluation_artifact_digest: String,
    /// Candidate profiles, one per experiment in the decision problem.
    /// Ineligible profiles must carry empty scoring vectors.
    pub profiles: Vec<ExperimentRegimeProfile>,
}

/// Experiment selection policy in the component-ablation bake-off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RidaAblationPolicy {
    /// Rank by target-regime representativeness only.
    RepresentativenessOnly,
    /// Rank by normalized nominal EVSI per cost only.
    InformativenessOnly,
    /// Rank by the validation error de-amplification proxy only.
    DeAmplificationOnly,
    /// Geometric mean of representativeness and informativeness.
    RepresentativenessAndInformativeness,
    /// Geometric mean of representativeness and de-amplification.
    RepresentativenessAndDeAmplification,
    /// Geometric mean of informativeness and de-amplification.
    InformativenessAndDeAmplification,
    /// Three-component geometric mean; an R-I-D proxy, not published R-IDeA.
    FullRidaProxy,
}

/// Per-candidate component values. Every component is independently reportable.
#[derive(Debug, Clone, PartialEq)]
pub struct RidaExperimentComponentScore {
    /// Stable experiment ID.
    pub experiment_id: String,
    /// Nominal EVSI divided by cost in the decision problem's units.
    pub nominal_evsi_per_cost: f64,
    /// Nominal information gain divided by cost, diagnostic only.
    pub information_gain_per_cost: f64,
    /// EVSI/cost normalized by the maximum across this candidate pool.
    pub informativeness_score: f64,
    /// One minus total-variation distance to the target regime distribution.
    pub representativeness_score: f64,
    /// Validation error weighted by the target-population distribution.
    pub target_weighted_validation_error: f64,
    /// Validation error weighted by this experiment's sampling distribution.
    pub sampling_weighted_validation_error: f64,
    /// Absolute difference between the two weighted validation errors.
    pub validation_error_distribution_gap: f64,
    /// One minus the validation-error distribution gap.
    /// This is a bounded proxy, not a universal measure of error amplification.
    pub de_amplification_score: f64,
}

/// One candidate and its score/rank for a named ablation.
#[derive(Debug, Clone, PartialEq)]
pub struct RidaPolicyScore {
    /// Candidate experiment ID.
    pub experiment_id: String,
    /// Policy score in [0, 1].
    pub score: f64,
    /// One-based deterministic rank.
    pub rank: usize,
}

/// Ranked candidate list for one ablation/policy.
#[derive(Debug, Clone, PartialEq)]
pub struct RidaPolicyRanking {
    /// Policy being ranked.
    pub policy: RidaAblationPolicy,
    /// Descending score, then ascending experiment ID.
    pub candidates: Vec<RidaPolicyScore>,
}

/// Transparent component and ablation report.
#[derive(Debug, Clone, PartialEq)]
pub struct RidaAblationReport {
    /// Declared regime labels.
    pub regime_labels: Vec<String>,
    /// Preregistered target regime distribution.
    pub target_regime_probabilities: Vec<f64>,
    /// Validation artifact identifier.
    pub validation_artifact_digest: String,
    /// Downstream evaluator artifact identifier; never used in policy scoring.
    pub evaluation_artifact_digest: String,
    /// Utility unit from QARCH-005.
    pub utility_unit: String,
    /// Measurement-cost unit from QARCH-005.
    pub measurement_cost_unit: String,
    /// Candidate IDs/statuses excluded by the upstream eligibility gate.
    pub ineligible_candidates: Vec<RidaIneligibleCandidate>,
    /// Component metrics for eligible candidates only.
    pub component_scores: Vec<RidaExperimentComponentScore>,
    /// Rankings for all one-, two-, and three-component policies.
    pub policy_rankings: Vec<RidaPolicyRanking>,
}

/// Validation and calculation errors in the R-I-D ablation benchmark.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RidaAblationError {
    /// A required identifier or artifact digest is blank.
    EmptyMetadata,
    /// Validation and evaluation artifacts must be distinct.
    ValidationEvaluationArtifactOverlap,
    /// Regime labels must be unique and non-empty.
    InvalidRegimeLabels,
    /// Target, sampling, or validation-error dimensions differ.
    RegimeDimensionMismatch,
    /// A target or sampling vector is not a finite probability distribution.
    InvalidRegimeDistribution,
    /// A validation error is not finite or outside [0, 1].
    InvalidValidationError,
    /// Profiles contain duplicate candidate IDs.
    DuplicateCandidateProfile,
    /// Profiles do not cover exactly the decision problem's candidates.
    CandidateProfileSetMismatch,
    /// A derived score is non-finite or outside its declared range.
    InvalidDerivedScore,
    /// QARCH-005 could not score the candidate experiments.
    DecisionScoringFailed,
}

impl fmt::Display for RidaAblationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::EmptyMetadata => "required regime or artifact metadata is empty",
            Self::ValidationEvaluationArtifactOverlap => {
                "validation and evaluator-only artifacts must be distinct"
            }
            Self::NoEligibleCandidates => "no experiment candidate passed the eligibility gate",
            Self::IneligibleCandidateContainsMetrics => {
                "ineligible candidates must not carry numerical scoring data"
            }
            Self::MissingEligibilityEvidence => {
                "every candidate eligibility result must bind an evidence artifact"
            }
            Self::InvalidRegimeLabels => "regime labels must be unique and non-empty",
            Self::RegimeDimensionMismatch => {
                "target, sampling, and validation-error regime dimensions must match"
            }
            Self::InvalidRegimeDistribution => {
                "target and sampling regime vectors must be valid probability distributions"
            }
            Self::InvalidValidationError => "validation errors must be finite values in [0, 1]",
            Self::DuplicateCandidateProfile => "candidate regime profiles contain duplicate IDs",
            Self::CandidateProfileSetMismatch => {
                "candidate regime profiles must match the decision problem exactly"
            }
            Self::InvalidDerivedScore => "a derived component or policy score is invalid",
            Self::DecisionScoringFailed => "QARCH-005 decision scoring failed",
        };
        f.write_str(message)
    }
}

impl Error for RidaAblationError {}

impl From<ExperimentDesignError> for RidaAblationError {
    fn from(_: ExperimentDesignError) -> Self {
        Self::DecisionScoringFailed
    }
}

/// Rank the same candidate pool under component-only, paired, and full R-I-D policies.
///
/// The full policy is the cube root of R * I * D, where each component is in
/// [0, 1]. This transparent geometric mean prevents a zero component from
/// being hidden by a large value in another component. No evaluator-only
/// outcomes are accepted by this API.
pub fn rank_rida_ablations(
    problem: &ActiveExperimentProblem,
    context: &RepresentativeExperimentContext,
) -> Result<RidaAblationReport, RidaAblationError> {
    validate_context(problem, context)?;
    let nominal_ranking = problem.rank_experiments()?;
    let maximum_evsi_per_cost = nominal_ranking
        .scores
        .iter()
        .map(|score| score.value_per_cost)
        .fold(0.0_f64, f64::max);

    let mut component_scores = Vec::with_capacity(context.profiles.len());
    for profile in context
        .profiles
        .iter()
        .filter(|profile| profile.eligibility_status == ExperimentEligibilityStatus::Eligible)
    {
        let nominal = nominal_ranking
            .scores
            .iter()
            .find(|score| score.experiment_id == profile.experiment_id)
            .ok_or(RidaAblationError::CandidateProfileSetMismatch)?;
        let representativeness_score = (1.0
            - total_variation_distance(
                &context.target_regime_probabilities,
                &profile.sampling_regime_probabilities,
            ))
        .clamp(0.0, 1.0);
        let target_weighted_validation_error = weighted_sum(
            &context.target_regime_probabilities,
            &profile.validation_error_by_regime,
        );
        let sampling_weighted_validation_error = weighted_sum(
            &profile.sampling_regime_probabilities,
            &profile.validation_error_by_regime,
        );
        let validation_error_distribution_gap =
            (target_weighted_validation_error - sampling_weighted_validation_error).abs();
        let de_amplification_score =
            (1.0 - validation_error_distribution_gap).clamp(0.0, 1.0);
        let informativeness_score = if maximum_evsi_per_cost == 0.0 {
            0.0
        } else {
            (nominal.value_per_cost / maximum_evsi_per_cost).clamp(0.0, 1.0)
        };

        let values = [
            representativeness_score,
            target_weighted_validation_error,
            sampling_weighted_validation_error,
            validation_error_distribution_gap,
            de_amplification_score,
            informativeness_score,
        ];
        if values.iter().any(|value| !value.is_finite())
            || values.iter().any(|value| !(0.0..=1.0).contains(value))
        {
            return Err(RidaAblationError::InvalidDerivedScore);
        }
        component_scores.push(RidaExperimentComponentScore {
            experiment_id: profile.experiment_id.clone(),
            nominal_evsi_per_cost: nominal.value_per_cost,
            information_gain_per_cost: nominal.information_gain_per_cost,
            informativeness_score,
            representativeness_score,
            target_weighted_validation_error,
            sampling_weighted_validation_error,
            validation_error_distribution_gap,
            de_amplification_score,
        });
    }

    if component_scores.is_empty() {
        return Err(RidaAblationError::NoEligibleCandidates);
    }

    let policies = [
        RidaAblationPolicy::RepresentativenessOnly,
        RidaAblationPolicy::InformativenessOnly,
        RidaAblationPolicy::DeAmplificationOnly,
        RidaAblationPolicy::RepresentativenessAndInformativeness,
        RidaAblationPolicy::RepresentativenessAndDeAmplification,
        RidaAblationPolicy::InformativenessAndDeAmplification,
        RidaAblationPolicy::FullRidaProxy,
    ];
    let mut policy_rankings = Vec::with_capacity(policies.len());
    for policy in policies {
        let mut candidates = component_scores
            .iter()
            .map(|component| {
                (
                    component.experiment_id.clone(),
                    policy_score(component, policy),
                )
            })
            .collect::<Vec<_>>();
        if candidates.iter().any(|(_, score)| {
            !score.is_finite() || !(0.0..=1.0).contains(score)
        }) {
            return Err(RidaAblationError::InvalidDerivedScore);
        }
        candidates.sort_by(|left, right| {
            right.1.total_cmp(&left.1).then_with(|| left.0.cmp(&right.0))
        });
        let candidates = candidates
            .into_iter()
            .enumerate()
            .map(|(index, (experiment_id, score))| RidaPolicyScore {
                experiment_id,
                score,
                rank: index + 1,
            })
            .collect();
        policy_rankings.push(RidaPolicyRanking { policy, candidates });
    }

    Ok(RidaAblationReport {
        regime_labels: context.regime_labels.clone(),
        target_regime_probabilities: context.target_regime_probabilities.clone(),
        validation_artifact_digest: context.validation_artifact_digest.clone(),
        evaluation_artifact_digest: context.evaluation_artifact_digest.clone(),
        utility_unit: nominal_ranking.utility_unit,
        measurement_cost_unit: nominal_ranking.measurement_cost_unit,
        ineligible_candidates: context
            .profiles
            .iter()
            .filter(|profile| profile.eligibility_status != ExperimentEligibilityStatus::Eligible)
            .map(|profile| RidaIneligibleCandidate {
                experiment_id: profile.experiment_id.clone(),
                eligibility_status: profile.eligibility_status,
                eligibility_evidence_digest: profile.eligibility_evidence_digest.clone(),
            })
            .collect(),
        component_scores,
        policy_rankings,
    })
}

fn validate_context(
    problem: &ActiveExperimentProblem,
    context: &RepresentativeExperimentContext,
) -> Result<(), RidaAblationError> {
    if context.validation_artifact_digest.trim().is_empty()
        || context.evaluation_artifact_digest.trim().is_empty()
        || context.regime_labels.is_empty()
    {
        return Err(RidaAblationError::EmptyMetadata);
    }
    if context.validation_artifact_digest == context.evaluation_artifact_digest {
        return Err(RidaAblationError::ValidationEvaluationArtifactOverlap);
    }
    if context.regime_labels.iter().any(|label| label.trim().is_empty())
        || has_duplicates(&context.regime_labels)
    {
        return Err(RidaAblationError::InvalidRegimeLabels);
    }
    let regime_count = context.regime_labels.len();
    if context.target_regime_probabilities.len() != regime_count {
        return Err(RidaAblationError::RegimeDimensionMismatch);
    }
    if context.profiles.len() != problem.experiments().len() {
        return Err(RidaAblationError::CandidateProfileSetMismatch);
    }
    validate_distribution(&context.target_regime_probabilities)?;

    let candidate_ids = problem
        .experiments()
        .iter()
        .map(|candidate| candidate.id())
        .collect::<Vec<_>>();
    let mut profile_ids = Vec::with_capacity(context.profiles.len());
    for profile in &context.profiles {
        profile_ids.push(profile.experiment_id.as_str());
        if profile.experiment_id.trim().is_empty() {
            return Err(RidaAblationError::EmptyMetadata);
        }
        if profile.eligibility_evidence_digest.trim().is_empty() {
            return Err(RidaAblationError::MissingEligibilityEvidence);
        }
        if profile.eligibility_status != ExperimentEligibilityStatus::Eligible {
            if !profile.sampling_regime_probabilities.is_empty()
                || !profile.validation_error_by_regime.is_empty()
            {
                return Err(RidaAblationError::IneligibleCandidateContainsMetrics);
            }
            continue;
        }
        if profile.sampling_regime_probabilities.len() != regime_count
            || profile.validation_error_by_regime.len() != regime_count
        {
            return Err(RidaAblationError::RegimeDimensionMismatch);
        }
        validate_distribution(&profile.sampling_regime_probabilities)?;
        if profile
            .validation_error_by_regime
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(RidaAblationError::InvalidValidationError);
        }
    }
    if has_duplicates(&profile_ids) {
        return Err(RidaAblationError::DuplicateCandidateProfile);
    }
    if candidate_ids.len() != profile_ids.len()
        || candidate_ids.iter().any(|id| !profile_ids.contains(id))
    {
        return Err(RidaAblationError::CandidateProfileSetMismatch);
    }
    if !context.profiles.iter().any(|profile| {
        profile.eligibility_status == ExperimentEligibilityStatus::Eligible
    }) {
        return Err(RidaAblationError::NoEligibleCandidates);
    }
    Ok(())
}

fn validate_distribution(values: &[f64]) -> Result<(), RidaAblationError> {
    if values.is_empty()
        || values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0 || *value > 1.0)
        || (values.iter().sum::<f64>() - 1.0).abs() > 1.0e-9
    {
        return Err(RidaAblationError::InvalidRegimeDistribution);
    }
    Ok(())
}

fn has_duplicates<T: AsRef<str>>(values: &[T]) -> bool {
    values.iter().enumerate().any(|(index, value)| {
        values[..index]
            .iter()
            .any(|previous| previous.as_ref() == value.as_ref())
    })
}

fn total_variation_distance(left: &[f64], right: &[f64]) -> f64 {
    0.5 * left
        .iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
}

fn weighted_sum(weights: &[f64], values: &[f64]) -> f64 {
    weights
        .iter()
        .zip(values)
        .map(|(weight, value)| weight * value)
        .sum::<f64>()
}

fn policy_score(component: &RidaExperimentComponentScore, policy: RidaAblationPolicy) -> f64 {
    let r = component.representativeness_score;
    let i = component.informativeness_score;
    let d = component.de_amplification_score;
    match policy {
        RidaAblationPolicy::RepresentativenessOnly => r,
        RidaAblationPolicy::InformativenessOnly => i,
        RidaAblationPolicy::DeAmplificationOnly => d,
        RidaAblationPolicy::RepresentativenessAndInformativeness => (r * i).sqrt(),
        RidaAblationPolicy::RepresentativenessAndDeAmplification => (r * d).sqrt(),
        RidaAblationPolicy::InformativenessAndDeAmplification => (i * d).sqrt(),
        RidaAblationPolicy::FullRidaProxy => (r * i * d).cbrt(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::active_experiment::DecisionExperiment;

    fn candidate(id: &str, accuracy: f64) -> DecisionExperiment {
        DecisionExperiment::new(
            id,
            1.0,
            vec!["outcome-0".into(), "outcome-1".into()],
            vec![
                vec![accuracy, 1.0 - accuracy],
                vec![1.0 - accuracy, accuracy],
            ],
        )
        .unwrap()
    }

    fn problem() -> ActiveExperimentProblem {
        ActiveExperimentProblem::new(
            vec!["H0".into(), "H1".into()],
            vec![0.5, 0.5],
            vec!["choose-H0".into(), "choose-H1".into()],
            vec![vec![2.0, 0.0], vec![0.0, 2.0]],
            "utility_points",
            "measurement_seconds",
            vec![
                candidate("A-high-info-out-of-distribution", 0.95),
                candidate("B-representative-low-info", 0.65),
                candidate("C-balanced-rid", 0.80),
                candidate("D-alternative-mix", 0.725),
            ],
        )
        .unwrap()
    }

    fn profile(
        experiment_id: &str,
        sampling: Vec<f64>,
        errors: Vec<f64>,
    ) -> ExperimentRegimeProfile {
        ExperimentRegimeProfile {
            experiment_id: experiment_id.into(),
            eligibility_status: ExperimentEligibilityStatus::Eligible,
            eligibility_evidence_digest: format!("eligibility-evidence-{experiment_id}"),
            sampling_regime_probabilities: sampling,
            validation_error_by_regime: errors,
        }
    }

    fn context() -> RepresentativeExperimentContext {
        RepresentativeExperimentContext {
            regime_labels: vec!["lab-nominal".into(), "warm-drift".into(), "low-count".into()],
            target_regime_probabilities: vec![0.5, 0.3, 0.2],
            validation_artifact_digest: "validation-split-sha256-v1".into(),
            evaluation_artifact_digest: "evaluator-only-split-sha256-v1".into(),
            profiles: vec![
                profile(
                    "A-high-info-out-of-distribution",
                    vec![0.05, 0.05, 0.90],
                    vec![0.1, 0.1, 0.9],
                ),
                profile(
                    "B-representative-low-info",
                    vec![0.5, 0.3, 0.2],
                    vec![0.2, 0.2, 0.2],
                ),
                profile(
                    "C-balanced-rid",
                    vec![0.4, 0.4, 0.2],
                    vec![0.1, 0.7, 0.2],
                ),
                profile(
                    "D-alternative-mix",
                    vec![0.2, 0.4, 0.4],
                    vec![0.4, 0.1, 0.1],
                ),
            ],
        }
    }

    fn ranking<'a>(
        report: &'a RidaAblationReport,
        policy: RidaAblationPolicy,
    ) -> &'a RidaPolicyRanking {
        report.policy_rankings.iter().find(|rank| rank.policy == policy).unwrap()
    }

    #[test]
    fn full_rid_proxy_and_component_ablations_have_expected_rankings() {
        let report = rank_rida_ablations(&problem(), &context()).unwrap();

        assert_eq!(
            ranking(&report, RidaAblationPolicy::InformativenessOnly).candidates[0].experiment_id,
            "A-high-info-out-of-distribution"
        );
        assert_eq!(
            ranking(&report, RidaAblationPolicy::RepresentativenessOnly).candidates[0].experiment_id,
            "B-representative-low-info"
        );
        assert_eq!(
            ranking(&report, RidaAblationPolicy::DeAmplificationOnly).candidates[0].experiment_id,
            "B-representative-low-info"
        );
        assert_eq!(
            ranking(&report, RidaAblationPolicy::FullRidaProxy).candidates[0].experiment_id,
            "C-balanced-rid"
        );
        let representative_and_informative = ranking(
            &report,
            RidaAblationPolicy::RepresentativenessAndInformativeness,
        );
        assert_eq!(
            representative_and_informative.candidates[0].experiment_id,
            "C-balanced-rid"
        );
        let informative_and_de_amplifying = ranking(
            &report,
            RidaAblationPolicy::InformativenessAndDeAmplification,
        );
        assert_eq!(
            informative_and_de_amplifying.candidates[0].experiment_id,
            "C-balanced-rid"
        );
    }

    #[test]
    fn component_metrics_match_independent_hand_calculation() {
        let report = rank_rida_ablations(&problem(), &context()).unwrap();
        let a = report.component_scores.iter()
            .find(|score| score.experiment_id == "A-high-info-out-of-distribution").unwrap();
        assert!((a.nominal_evsi_per_cost - 0.9).abs() < 1e-12);
        assert!((a.representativeness_score - 0.3).abs() < 1e-12);
        assert!((a.target_weighted_validation_error - 0.26).abs() < 1e-12);
        assert!((a.sampling_weighted_validation_error - 0.82).abs() < 1e-12);
        assert!((a.validation_error_distribution_gap - 0.56).abs() < 1e-12);
        assert!((a.de_amplification_score - 0.44).abs() < 1e-12);

        let c = report.component_scores.iter()
            .find(|score| score.experiment_id == "C-balanced-rid").unwrap();
        assert!((c.nominal_evsi_per_cost - 0.6).abs() < 1e-12);
        assert!((c.representativeness_score - 0.9).abs() < 1e-12);
        assert!((c.de_amplification_score - 0.94).abs() < 1e-12);
    }

    #[test]
    fn full_policy_ranking_adapts_to_a_preregistered_target_shift() {
        let problem = problem();
        let mut shifted = context();
        shifted.target_regime_probabilities = vec![0.1, 0.2, 0.7];
        let report = rank_rida_ablations(&problem, &shifted).unwrap();

        assert_eq!(
            ranking(&report, RidaAblationPolicy::FullRidaProxy).candidates[0]
                .experiment_id,
            "A-high-info-out-of-distribution"
        );
    }

    #[test]
    fn zero_information_candidates_have_zero_information_component() {
        let uninformative_problem = ActiveExperimentProblem::new(
            vec!["H0".into(), "H1".into()],
            vec![0.5, 0.5],
            vec!["choose-H0".into(), "choose-H1".into()],
            vec![vec![2.0, 0.0], vec![0.0, 2.0]],
            "utility_points",
            "measurement_seconds",
            vec![
                candidate("A-zero-information", 0.5),
                candidate("B-zero-information", 0.5),
            ],
        )
        .unwrap();
        let mut data = context();
        data.profiles = vec![
            profile(
                "A-zero-information",
                vec![0.5, 0.3, 0.2],
                vec![0.2, 0.2, 0.2],
            ),
            profile(
                "B-zero-information",
                vec![0.5, 0.3, 0.2],
                vec![0.2, 0.2, 0.2],
            ),
        ];
        let report = rank_rida_ablations(&uninformative_problem, &data).unwrap();

        assert!(report
            .component_scores
            .iter()
            .all(|score| score.informativeness_score == 0.0));
        assert!(ranking(&report, RidaAblationPolicy::FullRidaProxy)
            .candidates
            .iter()
            .all(|score| score.score == 0.0));
        assert_eq!(
            ranking(&report, RidaAblationPolicy::FullRidaProxy).candidates[0]
                .experiment_id,
            "A-zero-information"
        );
    }

    #[test]
    fn stale_calibration_candidate_is_ineligible_even_with_high_evsi() {
        let mut data = context();
        data.profiles[0].eligibility_status = ExperimentEligibilityStatus::StaleCalibration;
        data.profiles[0].sampling_regime_probabilities.clear();
        data.profiles[0].validation_error_by_regime.clear();
        let report = rank_rida_ablations(&problem(), &data).unwrap();

        assert_eq!(report.component_scores.len(), 3);
        assert!(report
            .component_scores
            .iter()
            .all(|score| score.experiment_id != "A-high-info-out-of-distribution"));
        assert_eq!(
            report.ineligible_candidates,
            vec![RidaIneligibleCandidate {
                experiment_id: "A-high-info-out-of-distribution".into(),
                eligibility_status: ExperimentEligibilityStatus::StaleCalibration,
                eligibility_evidence_digest:
                    "eligibility-evidence-A-high-info-out-of-distribution".into(),
            }]
        );
    }

    #[test]
    fn no_eligible_candidates_is_a_hard_error() {
        let mut data = context();
        for profile in &mut data.profiles {
            profile.eligibility_status = ExperimentEligibilityStatus::OutOfDomain;
            profile.sampling_regime_probabilities.clear();
            profile.validation_error_by_regime.clear();
        }
        assert_eq!(
            rank_rida_ablations(&problem(), &data),
            Err(RidaAblationError::NoEligibleCandidates)
        );
    }

    #[test]
    fn evaluator_only_artifact_cannot_equal_validation_artifact() {
        let mut data = context();
        data.evaluation_artifact_digest = data.validation_artifact_digest.clone();
        assert_eq!(
            rank_rida_ablations(&problem(), &data),
            Err(RidaAblationError::ValidationEvaluationArtifactOverlap)
        );
    }

    #[test]
    fn malformed_regime_distributions_fail_closed() {
        let mut data = context();
        data.profiles[0].sampling_regime_probabilities = vec![0.2, 0.2, 0.2];
        assert_eq!(
            rank_rida_ablations(&problem(), &data),
            Err(RidaAblationError::InvalidRegimeDistribution)
        );
    }

    #[test]
    fn validation_error_outside_unit_interval_is_rejected() {
        let mut data = context();
        data.profiles[0].validation_error_by_regime[0] = 1.01;
        assert_eq!(
            rank_rida_ablations(&problem(), &data),
            Err(RidaAblationError::InvalidValidationError)
        );
    }

    #[test]
    fn unknown_or_duplicate_candidate_profiles_are_rejected() {
        let mut data = context();
        data.profiles[0].experiment_id = "not-a-candidate".into();
        assert_eq!(
            rank_rida_ablations(&problem(), &data),
            Err(RidaAblationError::CandidateProfileSetMismatch)
        );

        let mut duplicate = context();
        duplicate.profiles[1].experiment_id = duplicate.profiles[0].experiment_id.clone();
        assert_eq!(
            rank_rida_ablations(&problem(), &duplicate),
            Err(RidaAblationError::DuplicateCandidateProfile)
        );
    }

    #[test]
    fn profile_order_does_not_change_policy_rankings() {
        let problem = problem();
        let forward = rank_rida_ablations(&problem, &context()).unwrap();
        let mut reversed_context = context();
        reversed_context.profiles.reverse();
        let reverse = rank_rida_ablations(&problem, &reversed_context).unwrap();

        for forward_policy in &forward.policy_rankings {
            let reverse_policy = ranking(&reverse, forward_policy.policy);
            assert_eq!(
                forward_policy.candidates.iter().map(|entry| &entry.experiment_id).collect::<Vec<_>>(),
                reverse_policy.candidates.iter().map(|entry| &entry.experiment_id).collect::<Vec<_>>()
            );
        }
    }
}