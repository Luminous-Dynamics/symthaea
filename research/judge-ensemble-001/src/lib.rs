//! Core metrics and policy comparisons for SCI-JUDGE-ENSEMBLE-001.
//!
//! This first implementation slice is deliberately std-only and operates on
//! typed in-memory records. It does not authenticate judge identity, prove
//! reference truth, fit a latent-truth model, estimate population effects, or
//! claim matched-budget superiority. Family-capped voting is a transparent
//! dependence stress comparator, not a full covariance-aware optimal ensemble.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    Calibration,
    Evaluation,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Response {
    Decision {
        label: bool,
        /// Judge's stated probability that its selected decision is correct.
        confidence_correct_probability: Option<f64>,
    },
    Abstain,
    Failed,
    Missing,
}

#[derive(Debug, Clone, PartialEq)]
pub struct JudgeSpec {
    pub judge_id: String,
    /// Declared common dependency lineage (e.g. base model/provider/prompt family).
    pub dependency_group_id: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    pub judge_id: String,
    pub response: Response,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Trial {
    pub trial_id: String,
    pub task_family_id: String,
    pub split: Split,
    /// None preserves an unadjudicated / unavailable reference denominator.
    pub reference_label: Option<bool>,
    /// Exactly one observation slot per preregistered judge, including failures.
    pub observations: Vec<Observation>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct JudgeProfile {
    pub judge_id: String,
    pub dependency_group_id: String,
    pub planned_trials: u64,
    pub reference_known_trials: u64,
    pub answered_trials: u64,
    pub scored_decisions: u64,
    pub correct_decisions: u64,
    pub positive_reference_trials: u64,
    pub negative_reference_trials: u64,
    pub true_positive_count: u64,
    pub false_negative_count: u64,
    pub true_negative_count: u64,
    pub false_positive_count: u64,
    pub sensitivity_on_answered_positive: Option<f64>,
    pub specificity_on_answered_negative: Option<f64>,
    pub balanced_accuracy_on_answered: Option<f64>,
    pub abstentions: u64,
    pub failures: u64,
    pub missing_responses: u64,
    pub confidence_scored_decisions: u64,
    pub accuracy_on_answered: Option<f64>,
    pub confidence_brier_score: Option<f64>,
    /// Non-negative log-odds weight from Laplace-smoothed calibration accuracy.
    /// This is a simple comparator, not a fitted multivariate reliability model.
    pub calibration_weight: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PairwiseDependence {
    pub left_judge_id: String,
    pub right_judge_id: String,
    pub shared_dependency_group: bool,
    pub co_observed_scored_decisions: u64,
    pub double_faults: u64,
    pub double_fault_rate: Option<f64>,
    /// Phi correlation of binary error indicators; None when either has zero variance.
    pub error_phi_correlation: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnsemblePolicy {
    calibration_trial_count: u64,
    /// Exact calibration item IDs, used to reject calibration/evaluation leakage.
    calibration_trial_ids: BTreeSet<String>,
    profiles: Vec<JudgeProfile>,
    pairwise_dependence: Vec<PairwiseDependence>,
    best_single_judge_id: String,
}

impl EnsemblePolicy {
    pub fn calibration_trial_count(&self) -> u64 {
        self.calibration_trial_count
    }

    pub fn profiles(&self) -> &[JudgeProfile] {
        &self.profiles
    }

    pub fn pairwise_dependence(&self) -> &[PairwiseDependence] {
        &self.pairwise_dependence
    }

    pub fn best_single_judge_id(&self) -> &str {
        &self.best_single_judge_id
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationMethod {
    BestSingle,
    Majority,
    FamilyCapped,
    CalibrationWeighted,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolicySummary {
    pub method: AggregationMethod,
    pub planned_trials: u64,
    pub reference_known_trials: u64,
    pub reference_missing_trials: u64,
    pub answered_trials: u64,
    pub correct_decisions: u64,
    pub incorrect_decisions: u64,
    pub abstained_trials: u64,
    pub coverage: Option<f64>,
    pub selective_accuracy: Option<f64>,
    pub selective_error_rate: Option<f64>,
    /// Count of planned judge invocations, not measured monetary/latency cost.
    pub planned_judge_calls: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationReport {
    pub policy_calibration_trials: u64,
    pub summaries: Vec<PolicySummary>,
    pub claim_boundary: &'static str,
}

fn validate_panel(judges: &[JudgeSpec], trials: &[Trial]) -> Result<(), String> {
    if judges.is_empty() {
        return Err("judge panel must not be empty".to_owned());
    }
    if trials.is_empty() {
        return Err("trial set must not be empty".to_owned());
    }

    let mut judge_ids = BTreeSet::new();
    for judge in judges {
        if judge.judge_id.trim().is_empty() {
            return Err("judge_id must not be empty".to_owned());
        }
        if judge.dependency_group_id.trim().is_empty() {
            return Err(format!(
                "judge {} has an empty dependency_group_id",
                judge.judge_id
            ));
        }
        if !judge_ids.insert(judge.judge_id.as_str()) {
            return Err(format!("duplicate judge_id {}", judge.judge_id));
        }
    }

    let mut trial_ids = BTreeSet::new();
    for trial in trials {
        if trial.trial_id.trim().is_empty() || trial.task_family_id.trim().is_empty() {
            return Err("trial_id and task_family_id must be non-empty".to_owned());
        }
        if !trial_ids.insert(trial.trial_id.as_str()) {
            return Err(format!("duplicate trial_id {}", trial.trial_id));
        }
        if trial.observations.len() != judges.len() {
            return Err(format!(
                "trial {} must contain exactly {} judge slots; got {}",
                trial.trial_id,
                judges.len(),
                trial.observations.len()
            ));
        }

        let mut observed = BTreeSet::new();
        for observation in &trial.observations {
            if !judge_ids.contains(observation.judge_id.as_str()) {
                return Err(format!(
                    "trial {} contains unknown judge {}",
                    trial.trial_id, observation.judge_id
                ));
            }
            if !observed.insert(observation.judge_id.as_str()) {
                return Err(format!(
                    "trial {} contains duplicate judge slot {}",
                    trial.trial_id, observation.judge_id
                ));
            }
            if let Response::Decision {
                confidence_correct_probability: Some(probability),
                ..
            } = &observation.response
            {
                if !probability.is_finite() || !(0.0..=1.0).contains(probability) {
                    return Err(format!(
                        "trial {} judge {} has invalid confidence probability",
                        trial.trial_id, observation.judge_id
                    ));
                }
            }
        }
        if observed.len() != judges.len() {
            return Err(format!(
                "trial {} is missing one or more preregistered judge slots",
                trial.trial_id
            ));
        }
    }
    Ok(())
}

fn response_for<'a>(trial: &'a Trial, judge_id: &str) -> &'a Response {
    // Panel validation guarantees exactly one slot for every judge.
    &trial
        .observations
        .iter()
        .find(|observation| observation.judge_id == judge_id)
        .expect("validated panel has a complete judge slot")
        .response
}

fn decision_from_response(response: &Response) -> Option<bool> {
    match response {
        Response::Decision { label, .. } => Some(*label),
        Response::Abstain | Response::Failed | Response::Missing => None,
    }
}

fn make_profile(judge: &JudgeSpec, trials: &[Trial]) -> JudgeProfile {
    let mut answered_trials = 0_u64;
    let mut reference_known_trials = 0_u64;
    let mut scored_decisions = 0_u64;
    let mut correct_decisions = 0_u64;
    let mut positive_reference_trials = 0_u64;
    let mut negative_reference_trials = 0_u64;
    let mut true_positive_count = 0_u64;
    let mut false_negative_count = 0_u64;
    let mut true_negative_count = 0_u64;
    let mut false_positive_count = 0_u64;
    let mut abstentions = 0_u64;
    let mut failures = 0_u64;
    let mut missing_responses = 0_u64;
    let mut confidence_scored_decisions = 0_u64;
    let mut brier_sum = 0.0_f64;

    for trial in trials {
        match trial.reference_label {
            Some(true) => {
                reference_known_trials += 1;
                positive_reference_trials += 1;
            }
            Some(false) => {
                reference_known_trials += 1;
                negative_reference_trials += 1;
            }
            None => {}
        }
        match response_for(trial, &judge.judge_id) {
            Response::Decision {
                label,
                confidence_correct_probability,
            } => {
                answered_trials += 1;
                if let Some(reference) = trial.reference_label {
                    scored_decisions += 1;
                    let correct = *label == reference;
                    if correct {
                        correct_decisions += 1;
                    }
                    match (reference, *label) {
                        (true, true) => true_positive_count += 1,
                        (true, false) => false_negative_count += 1,
                        (false, false) => true_negative_count += 1,
                        (false, true) => false_positive_count += 1,
                    }
                    if let Some(probability) = confidence_correct_probability {
                        let target = if correct { 1.0 } else { 0.0 };
                        brier_sum += (*probability - target).powi(2);
                        confidence_scored_decisions += 1;
                    }
                }
            }
            Response::Abstain => abstentions += 1,
            Response::Failed => failures += 1,
            Response::Missing => missing_responses += 1,
        }
    }

    let accuracy = (scored_decisions > 0)
        .then_some(correct_decisions as f64 / scored_decisions as f64);
    let sensitivity_denominator = true_positive_count + false_negative_count;
    let specificity_denominator = true_negative_count + false_positive_count;
    let sensitivity = (sensitivity_denominator > 0)
        .then_some(true_positive_count as f64 / sensitivity_denominator as f64);
    let specificity = (specificity_denominator > 0)
        .then_some(true_negative_count as f64 / specificity_denominator as f64);
    let balanced_accuracy = sensitivity.zip(specificity).map(|(tpr, tnr)| (tpr + tnr) / 2.0);
    let brier = (confidence_scored_decisions > 0)
        .then_some(brier_sum / confidence_scored_decisions as f64);

    // Balanced-accuracy weighting avoids silently rewarding a judge simply
    // because the calibration set over-represents its favored class. Each
    // class-specific rate receives independent Beta(1,1) smoothing. This is
    // still a simple comparator, not a fitted sensitivity/specificity model.
    let calibration_weight = if sensitivity_denominator > 0 && specificity_denominator > 0 {
        let smoothed_sensitivity =
            (true_positive_count as f64 + 1.0) / (sensitivity_denominator as f64 + 2.0);
        let smoothed_specificity =
            (true_negative_count as f64 + 1.0) / (specificity_denominator as f64 + 2.0);
        let smoothed_balanced_accuracy =
            (smoothed_sensitivity + smoothed_specificity) / 2.0;
        (smoothed_balanced_accuracy / (1.0 - smoothed_balanced_accuracy))
            .ln()
            .max(0.0)
    } else {
        0.0
    };

    JudgeProfile {
        judge_id: judge.judge_id.clone(),
        dependency_group_id: judge.dependency_group_id.clone(),
        planned_trials: trials.len() as u64,
        reference_known_trials,
        answered_trials,
        scored_decisions,
        correct_decisions,
        positive_reference_trials,
        negative_reference_trials,
        true_positive_count,
        false_negative_count,
        true_negative_count,
        false_positive_count,
        sensitivity_on_answered_positive: sensitivity,
        specificity_on_answered_negative: specificity,
        balanced_accuracy_on_answered: balanced_accuracy,
        abstentions,
        failures,
        missing_responses,
        confidence_scored_decisions,
        accuracy_on_answered: accuracy,
        confidence_brier_score: brier,
        calibration_weight,
    }
}

fn pairwise_dependence(
    judges: &[JudgeSpec],
    trials: &[Trial],
) -> Vec<PairwiseDependence> {
    let mut output = Vec::new();
    for left_index in 0..judges.len() {
        for right_index in (left_index + 1)..judges.len() {
            let left = &judges[left_index];
            let right = &judges[right_index];
            let mut n11 = 0_u64;
            let mut n10 = 0_u64;
            let mut n01 = 0_u64;
            let mut n00 = 0_u64;

            for trial in trials {
                let Some(reference) = trial.reference_label else {
                    continue;
                };
                let (Some(left_label), Some(right_label)) = (
                    decision_from_response(response_for(trial, &left.judge_id)),
                    decision_from_response(response_for(trial, &right.judge_id)),
                ) else {
                    continue;
                };
                let left_error = left_label != reference;
                let right_error = right_label != reference;
                match (left_error, right_error) {
                    (true, true) => n11 += 1,
                    (true, false) => n10 += 1,
                    (false, true) => n01 += 1,
                    (false, false) => n00 += 1,
                }
            }

            let comparable = n11 + n10 + n01 + n00;
            let double_fault_rate =
                (comparable > 0).then_some(n11 as f64 / comparable as f64);
            let phi_denominator = ((n11 + n10) as f64
                * (n01 + n00) as f64
                * (n11 + n01) as f64
                * (n10 + n00) as f64)
                .sqrt();
            let error_phi_correlation = (phi_denominator > 0.0).then_some(
                (n11 as f64 * n00 as f64 - n10 as f64 * n01 as f64)
                    / phi_denominator,
            );

            output.push(PairwiseDependence {
                left_judge_id: left.judge_id.clone(),
                right_judge_id: right.judge_id.clone(),
                shared_dependency_group: left.dependency_group_id
                    == right.dependency_group_id,
                co_observed_scored_decisions: comparable,
                double_faults: n11,
                double_fault_rate,
                error_phi_correlation,
            });
        }
    }
    output
}

/// Fit transparent baseline policies using calibration trials only.
/// Evaluation records are rejected rather than silently used for weight fitting.
pub fn fit_ensemble(
    judges: &[JudgeSpec],
    calibration_trials: &[Trial],
) -> Result<EnsemblePolicy, String> {
    validate_panel(judges, calibration_trials)?;
    if calibration_trials
        .iter()
        .any(|trial| trial.split != Split::Calibration)
    {
        return Err("fit_ensemble accepts calibration trials only".to_owned());
    }
    let has_positive = calibration_trials
        .iter()
        .any(|trial| trial.reference_label == Some(true));
    let has_negative = calibration_trials
        .iter()
        .any(|trial| trial.reference_label == Some(false));
    if !has_positive || !has_negative {
        return Err(
            "calibration requires adjudicated positive and negative reference examples".to_owned(),
        );
    }

    let mut profiles: Vec<JudgeProfile> = judges
        .iter()
        .map(|judge| make_profile(judge, calibration_trials))
        .collect();
    profiles.sort_by(|left, right| left.judge_id.cmp(&right.judge_id));

    let best_single_judge_id = profiles
        .iter()
        .filter(|profile| profile.scored_decisions > 0)
        .max_by(|left, right| {
            left.balanced_accuracy_on_answered
                .unwrap_or(-1.0)
                .total_cmp(&right.balanced_accuracy_on_answered.unwrap_or(-1.0))
                .then_with(|| right.judge_id.cmp(&left.judge_id))
        })
        .map(|profile| profile.judge_id.clone())
        .ok_or_else(|| "no judge has a scored calibration decision".to_owned())?;

    let calibration_trial_ids = calibration_trials
        .iter()
        .map(|trial| trial.trial_id.clone())
        .collect();

    Ok(EnsemblePolicy {
        calibration_trial_count: calibration_trials.len() as u64,
        calibration_trial_ids,
        profiles,
        pairwise_dependence: pairwise_dependence(judges, calibration_trials),
        best_single_judge_id,
    })
}

fn majority_decision<'a>(
    judge_ids: impl Iterator<Item = &'a str>,
    trial: &Trial,
) -> Option<bool> {
    let mut yes = 0_u64;
    let mut no = 0_u64;
    for judge_id in judge_ids {
        match decision_from_response(response_for(trial, judge_id)) {
            Some(true) => yes += 1,
            Some(false) => no += 1,
            None => {}
        }
    }
    match yes.cmp(&no) {
        std::cmp::Ordering::Greater => Some(true),
        std::cmp::Ordering::Less => Some(false),
        std::cmp::Ordering::Equal => None,
    }
}

fn family_capped_decision(judges: &[JudgeSpec], trial: &Trial) -> Option<bool> {
    let mut groups: BTreeMap<&str, (u64, u64)> = BTreeMap::new();
    for judge in judges {
        let counts = groups
            .entry(judge.dependency_group_id.as_str())
            .or_insert((0, 0));
        match decision_from_response(response_for(trial, &judge.judge_id)) {
            Some(true) => counts.0 += 1,
            Some(false) => counts.1 += 1,
            None => {}
        }
    }

    let mut group_yes = 0_u64;
    let mut group_no = 0_u64;
    for (yes, no) in groups.values() {
        match yes.cmp(no) {
            std::cmp::Ordering::Greater => group_yes += 1,
            std::cmp::Ordering::Less => group_no += 1,
            std::cmp::Ordering::Equal => {}
        }
    }
    match group_yes.cmp(&group_no) {
        std::cmp::Ordering::Greater => Some(true),
        std::cmp::Ordering::Less => Some(false),
        std::cmp::Ordering::Equal => None,
    }
}

fn weighted_decision(policy: &EnsemblePolicy, trial: &Trial) -> Option<bool> {
    let mut yes = 0.0_f64;
    let mut no = 0.0_f64;
    for profile in &policy.profiles {
        match decision_from_response(response_for(trial, &profile.judge_id)) {
            Some(true) => yes += profile.calibration_weight,
            Some(false) => no += profile.calibration_weight,
            None => {}
        }
    }
    if yes + no == 0.0 || yes == no {
        None
    } else {
        Some(yes > no)
    }
}

fn summarize(
    method: AggregationMethod,
    decisions: &[Option<bool>],
    trials: &[Trial],
    planned_calls_per_trial: u64,
) -> PolicySummary {
    let mut reference_known_trials = 0_u64;
    let mut answered_trials = 0_u64;
    let mut correct_decisions = 0_u64;
    let mut incorrect_decisions = 0_u64;
    let mut abstained_trials = 0_u64;

    for (trial, decision) in trials.iter().zip(decisions) {
        let Some(reference) = trial.reference_label else {
            continue;
        };
        reference_known_trials += 1;
        match decision {
            Some(label) => {
                answered_trials += 1;
                if *label == reference {
                    correct_decisions += 1;
                } else {
                    incorrect_decisions += 1;
                }
            }
            None => abstained_trials += 1,
        }
    }

    PolicySummary {
        method,
        planned_trials: trials.len() as u64,
        reference_known_trials,
        reference_missing_trials: trials.len() as u64 - reference_known_trials,
        answered_trials,
        correct_decisions,
        incorrect_decisions,
        abstained_trials,
        coverage: (reference_known_trials > 0)
            .then_some(answered_trials as f64 / reference_known_trials as f64),
        selective_accuracy: (answered_trials > 0)
            .then_some(correct_decisions as f64 / answered_trials as f64),
        selective_error_rate: (answered_trials > 0)
            .then_some(incorrect_decisions as f64 / answered_trials as f64),
        planned_judge_calls: trials.len() as u64 * planned_calls_per_trial,
    }
}

/// Score frozen policy arms on evaluation trials. Calibration records are rejected.
pub fn evaluate_ensemble(
    judges: &[JudgeSpec],
    policy: &EnsemblePolicy,
    evaluation_trials: &[Trial],
) -> Result<EvaluationReport, String> {
    validate_panel(judges, evaluation_trials)?;
    if evaluation_trials
        .iter()
        .any(|trial| trial.split != Split::Evaluation)
    {
        return Err("evaluate_ensemble accepts evaluation trials only".to_owned());
    }

    let judge_ids: BTreeSet<&str> = judges.iter().map(|judge| judge.judge_id.as_str()).collect();
    let profile_ids: BTreeSet<&str> = policy
        .profiles
        .iter()
        .map(|profile| profile.judge_id.as_str())
        .collect();
    if judge_ids != profile_ids || judges.len() != policy.profiles.len() {
        return Err("evaluation judge panel differs from the frozen calibration panel".to_owned());
    }
    if !judge_ids.contains(policy.best_single_judge_id.as_str()) {
        return Err("best-single judge is not in the frozen panel".to_owned());
    }
    for profile in &policy.profiles {
        if profile.dependency_group_id.trim().is_empty()
            || !profile.calibration_weight.is_finite()
            || profile.calibration_weight < 0.0
        {
            return Err(format!("judge {} has an invalid calibrated profile", profile.judge_id));
        }
    }
    for judge in judges {
        let profile = policy
            .profiles
            .iter()
            .find(|profile| profile.judge_id == judge.judge_id)
            .expect("panel set equality was validated");
        if profile.dependency_group_id != judge.dependency_group_id {
            return Err(format!(
                "judge {} dependency lineage differs from calibration",
                judge.judge_id
            ));
        }
    }
    if policy.calibration_trial_count == 0
        || policy.calibration_trial_ids.len() as u64 != policy.calibration_trial_count
    {
        return Err("policy calibration trial IDs do not match its committed count".to_owned());
    }
    if evaluation_trials
        .iter()
        .any(|trial| policy.calibration_trial_ids.contains(&trial.trial_id))
    {
        return Err("calibration/evaluation trial IDs overlap".to_owned());
    }

    let single_id = policy.best_single_judge_id.as_str();
    let single = evaluation_trials
        .iter()
        .map(|trial| decision_from_response(response_for(trial, single_id)))
        .collect::<Vec<_>>();
    let majority = evaluation_trials
        .iter()
        .map(|trial| {
            majority_decision(judges.iter().map(|judge| judge.judge_id.as_str()), trial)
        })
        .collect::<Vec<_>>();
    let family_capped = evaluation_trials
        .iter()
        .map(|trial| family_capped_decision(judges, trial))
        .collect::<Vec<_>>();
    let weighted = evaluation_trials
        .iter()
        .map(|trial| weighted_decision(policy, trial))
        .collect::<Vec<_>>();

    let panel_calls = judges.len() as u64;
    Ok(EvaluationReport {
        policy_calibration_trials: policy.calibration_trial_count,
        summaries: vec![
            summarize(AggregationMethod::BestSingle, &single, evaluation_trials, 1),
            summarize(AggregationMethod::Majority, &majority, evaluation_trials, panel_calls),
            summarize(
                AggregationMethod::FamilyCapped,
                &family_capped,
                evaluation_trials,
                panel_calls,
            ),
            summarize(
                AggregationMethod::CalibrationWeighted,
                &weighted,
                evaluation_trials,
                panel_calls,
            ),
        ],
        claim_boundary: "Fixed evaluation-set descriptive metrics only. No population inference, judge identity authentication, independent reference truth, matched-budget claim, or generalization is established.",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn judges() -> Vec<JudgeSpec> {
        vec![
            JudgeSpec {
                judge_id: "clone-a".to_owned(),
                dependency_group_id: "shared-foundation".to_owned(),
            },
            JudgeSpec {
                judge_id: "clone-b".to_owned(),
                dependency_group_id: "shared-foundation".to_owned(),
            },
            JudgeSpec {
                judge_id: "clone-c".to_owned(),
                dependency_group_id: "shared-foundation".to_owned(),
            },
            JudgeSpec {
                judge_id: "independent-a".to_owned(),
                dependency_group_id: "family-a".to_owned(),
            },
            JudgeSpec {
                judge_id: "independent-b".to_owned(),
                dependency_group_id: "family-b".to_owned(),
            },
        ]
    }

    fn trial(index: usize, split: Split, reference: Option<bool>) -> Trial {
        let specs = judges();
        let observations = specs
            .iter()
            .map(|judge| {
                // The shared family has a systematic positive-label bias. The
                // two independent judges follow the known-answer reference.
                let label = if judge.dependency_group_id == "shared-foundation" {
                    true
                } else {
                    reference.unwrap_or(true)
                };
                Observation {
                    judge_id: judge.judge_id.clone(),
                    response: Response::Decision {
                        label,
                        confidence_correct_probability: Some(0.95),
                    },
                }
            })
            .collect();
        let split_prefix = match split {
            Split::Calibration => "calibration",
            Split::Evaluation => "evaluation",
        };
        Trial {
            trial_id: format!("{split_prefix}-item-{index:03}"),
            task_family_id: if index % 4 < 2 {
                "family-x".to_owned()
            } else {
                "family-y".to_owned()
            },
            split,
            reference_label: reference,
            observations,
        }
    }

    fn data() -> (Vec<JudgeSpec>, Vec<Trial>, Vec<Trial>) {
        let panel = judges();
        let calibration = (0..12)
            .map(|index| trial(index, Split::Calibration, Some(index % 2 == 0)))
            .collect();
        let evaluation = (0..12)
            .map(|index| trial(index, Split::Evaluation, Some(index % 2 == 0)))
            .collect();
        (panel, calibration, evaluation)
    }

    fn summary(report: &EvaluationReport, method: AggregationMethod) -> &PolicySummary {
        report
            .summaries
            .iter()
            .find(|summary| summary.method == method)
            .expect("expected aggregation summary")
    }

    #[test]
    fn calibration_weighting_and_family_caps_resist_clone_vote_amplification() {
        let (panel, calibration, evaluation) = data();
        let policy = fit_ensemble(&panel, &calibration).unwrap();
        let report = evaluate_ensemble(&panel, &policy, &evaluation).unwrap();

        let majority = summary(&report, AggregationMethod::Majority);
        let family_capped = summary(&report, AggregationMethod::FamilyCapped);
        let weighted = summary(&report, AggregationMethod::CalibrationWeighted);
        let best_single = summary(&report, AggregationMethod::BestSingle);

        assert_eq!(majority.incorrect_decisions, 6);
        assert_eq!(majority.correct_decisions, 6);
        assert_eq!(family_capped.incorrect_decisions, 0);
        assert_eq!(family_capped.correct_decisions, 12);
        assert_eq!(family_capped.abstained_trials, 0);
        assert_eq!(weighted.incorrect_decisions, 0);
        assert_eq!(weighted.correct_decisions, 12);
        assert_eq!(best_single.incorrect_decisions, 0);
        assert_eq!(best_single.correct_decisions, 12);
        assert_eq!(family_capped.planned_judge_calls, 60);
        assert_eq!(best_single.planned_judge_calls, 12);
    }

    #[test]
    fn perfectly_shared_errors_are_reported_as_double_faults() {
        let (panel, calibration, _) = data();
        let policy = fit_ensemble(&panel, &calibration).unwrap();
        let pair = policy
            .pairwise_dependence
            .iter()
            .find(|pair| {
                pair.left_judge_id == "clone-a" && pair.right_judge_id == "clone-b"
            })
            .unwrap();
        assert!(pair.shared_dependency_group);
        assert_eq!(pair.co_observed_scored_decisions, 12);
        assert_eq!(pair.double_faults, 6);
        assert_eq!(pair.double_fault_rate, Some(0.5));
        assert_eq!(pair.error_phi_correlation, Some(1.0));
    }

    #[test]
    fn judge_weights_are_derived_from_calibration_only() {
        let (panel, calibration, mut evaluation) = data();
        let before = fit_ensemble(&panel, &calibration).unwrap();
        for trial in &mut evaluation {
            trial.reference_label = Some(false);
        }
        let after = fit_ensemble(&panel, &calibration).unwrap();
        assert_eq!(before, after);
        assert_ne!(
            evaluate_ensemble(&panel, &before, &data().2).unwrap(),
            evaluate_ensemble(&panel, &before, &evaluation).unwrap()
        );
    }

    #[test]
    fn calibration_rejects_evaluation_split() {
        let (panel, _, evaluation) = data();
        assert!(fit_ensemble(&panel, &evaluation)
            .unwrap_err()
            .contains("calibration trials only"));
    }

    #[test]
    fn evaluation_rejects_calibration_split() {
        let (panel, calibration, _) = data();
        let policy = fit_ensemble(&panel, &calibration).unwrap();
        assert!(evaluate_ensemble(&panel, &policy, &calibration)
            .unwrap_err()
            .contains("evaluation trials only"));
    }

    #[test]
    fn evaluation_rejects_reused_calibration_item_ids() {
        let (panel, calibration, mut evaluation) = data();
        let policy = fit_ensemble(&panel, &calibration).unwrap();
        evaluation[0].trial_id = calibration[0].trial_id.clone();
        assert!(evaluate_ensemble(&panel, &policy, &evaluation)
            .unwrap_err()
            .contains("calibration/evaluation trial IDs overlap"));
    }

    #[test]
    fn calibration_requires_both_reference_classes() {
        let (panel, mut calibration, _) = data();
        for trial in &mut calibration {
            trial.reference_label = Some(true);
        }
        assert!(fit_ensemble(&panel, &calibration)
            .unwrap_err()
            .contains("positive and negative reference examples"));
    }

    #[test]
    fn unadjudicated_trials_remain_in_denominators_but_not_accuracy() {
        let (panel, calibration, mut evaluation) = data();
        let policy = fit_ensemble(&panel, &calibration).unwrap();
        evaluation[0].reference_label = None;
        evaluation[1].observations[0].response = Response::Missing;
        let report = evaluate_ensemble(&panel, &policy, &evaluation).unwrap();
        let single = summary(&report, AggregationMethod::BestSingle);
        assert_eq!(single.planned_trials, 6);
        assert_eq!(single.reference_known_trials, 5);
        assert_eq!(single.reference_missing_trials, 1);
        assert_eq!(single.answered_trials, 5);
    }

    #[test]
    fn malformed_panel_slots_and_confidence_are_rejected() {
        let (panel, calibration, _) = data();
        let mut malformed = calibration.clone();
        malformed[0].observations.pop();
        assert!(validate_panel(&panel, &malformed)
            .unwrap_err()
            .contains("exactly 5 judge slots"));

        let mut invalid_confidence = calibration;
        invalid_confidence[0].observations[0].response = Response::Decision {
            label: false,
            confidence_correct_probability: Some(1.5),
        };
        assert!(validate_panel(&panel, &invalid_confidence)
            .unwrap_err()
            .contains("invalid confidence"));
    }

    #[test]
    fn confidence_brier_score_is_separate_from_vote_accuracy() {
        let (panel, calibration, _) = data();
        let policy = fit_ensemble(&panel, &calibration).unwrap();
        let good = policy
            .profiles
            .iter()
            .find(|profile| profile.judge_id == "independent-a")
            .unwrap();
        assert_eq!(good.accuracy_on_answered, Some(1.0));
        assert_eq!(good.sensitivity_on_answered_positive, Some(1.0));
        assert_eq!(good.specificity_on_answered_negative, Some(1.0));
        assert_eq!(good.balanced_accuracy_on_answered, Some(1.0));
        let brier = good.confidence_brier_score.unwrap();
        assert!((brier - 0.05_f64.powi(2)).abs() < 1.0e-12);

        let clone = policy
            .profiles
            .iter()
            .find(|profile| profile.judge_id == "clone-a")
            .unwrap();
        assert_eq!(clone.sensitivity_on_answered_positive, Some(1.0));
        assert_eq!(clone.specificity_on_answered_negative, Some(0.0));
        assert_eq!(clone.balanced_accuracy_on_answered, Some(0.5));
        assert_eq!(clone.calibration_weight, 0.0);
    }
}
