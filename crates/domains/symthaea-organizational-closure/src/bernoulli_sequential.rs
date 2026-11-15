//! Anytime-valid two-arm Bernoulli assay evaluator for AOBench.
//!
//! This is a narrow stochastic pilot. It scores an independently assigned
//! control/treatment assay under iid Bernoulli outcomes and predictable
//! allocation. It is not a general causal-discovery or missing-data engine.
//!
//! alpha_arm = alpha / 2 and alpha_(arm,n) = 6 * alpha_arm / (pi^2 * n^2).
//! Hoeffding's inequality gives radius sqrt(log(2 / alpha_(arm,n)) / (2n)).
//! A union bound across both arms and every n >= 1 gives simultaneous
//! coverage of at least 1 - alpha at every interim look.

use super::experiment_design::ExperimentEvidenceStatus;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BernoulliArm {
    Control,
    Treatment,
}

/// Frozen agent-visible task contract. Oracle probabilities are supplied
/// separately to the evaluator and are never embedded in this structure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BernoulliSequentialScenario {
    pub alpha: f64,
    pub minimum_effect: f64,
    pub max_total_samples: usize,
    pub max_samples_per_arm: usize,
    pub sample_cost_units: u64,
    pub max_cost_units: u64,
    pub sample_risk_units: u64,
    pub max_risk_units: u64,
}

impl BernoulliSequentialScenario {
    pub fn new(
        alpha: f64,
        minimum_effect: f64,
        max_total_samples: usize,
        max_samples_per_arm: usize,
        sample_cost_units: u64,
        max_cost_units: u64,
        sample_risk_units: u64,
        max_risk_units: u64,
    ) -> Self {
        Self {
            alpha,
            minimum_effect,
            max_total_samples,
            max_samples_per_arm,
            sample_cost_units,
            max_cost_units,
            sample_risk_units,
            max_risk_units,
        }
    }
}

/// Evidence statuses are supplied by the harness. This module does not
/// authenticate receipts, assignment logs or physical randomization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BernoulliTrial {
    pub arm: BernoulliArm,
    pub outcome: bool,
    pub assignment: ExperimentEvidenceStatus,
    pub delivery: ExperimentEvidenceStatus,
    pub isolation: ExperimentEvidenceStatus,
    pub observation: ExperimentEvidenceStatus,
}

impl BernoulliTrial {
    pub fn qualified(arm: BernoulliArm, outcome: bool) -> Self {
        Self {
            arm,
            outcome,
            assignment: ExperimentEvidenceStatus::Verified,
            delivery: ExperimentEvidenceStatus::Verified,
            isolation: ExperimentEvidenceStatus::Verified,
            observation: ExperimentEvidenceStatus::Verified,
        }
    }
}

/// Evaluator-only truth. Deliberately not Serialize/Deserialize and never
/// embedded in the agent-facing scenario or trial trace.
#[derive(Debug, Clone, PartialEq)]
pub struct BernoulliOracle {
    control_probability: f64,
    treatment_probability: f64,
}

impl BernoulliOracle {
    pub fn new(control_probability: f64, treatment_probability: f64) -> Self {
        Self { control_probability, treatment_probability }
    }

    pub fn true_effect(&self) -> f64 {
        self.treatment_probability - self.control_probability
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BernoulliClaim {
    Benefit,
    Harm,
    NoMaterialEffect,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case", deny_unknown_fields)]
pub enum BernoulliTerminalDecision {
    Stop { claim: BernoulliClaim },
    Abstain { reason: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BernoulliSequentialTrace {
    pub trials: Vec<BernoulliTrial>,
    pub terminal: BernoulliTerminalDecision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BernoulliQualification {
    Qualified,
    NotQualified,
    Inconclusive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BernoulliBoundary {
    Continue,
    Benefit,
    Harm,
    NoMaterialEffect,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BernoulliInterval {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BernoulliArmSummary {
    pub attempted_samples: usize,
    pub qualified_samples: usize,
    pub successes: usize,
    pub mean: Option<f64>,
    pub interval: BernoulliInterval,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BernoulliTrialAssessment {
    pub trial_index: usize,
    pub arm: BernoulliArm,
    pub evidence_qualified: bool,
    pub qualified_control_samples: usize,
    pub qualified_treatment_samples: usize,
    pub treatment_minus_control: BernoulliInterval,
    pub boundary: BernoulliBoundary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BernoulliFindingCode {
    InvalidAlpha,
    InvalidMinimumEffect,
    EmptySampleBudget,
    InvalidOracleProbability,
    TotalSampleBudgetExceeded,
    PerArmSampleBudgetExceeded,
    CostBudgetExceeded,
    RiskBudgetExceeded,
    AssignmentRejected,
    DeliveryRejected,
    IsolationRejected,
    ObservationRejected,
    EvidenceUnavailable,
    SamplingAfterBoundary,
    PrematureStop,
    WrongClaim,
    AbstainedAfterBoundary,
    EmptyAbstentionReason,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BernoulliFinding {
    pub code: BernoulliFindingCode,
    pub trial_index: Option<usize>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BernoulliSequentialReport {
    pub qualification: BernoulliQualification,
    pub boundary: BernoulliBoundary,
    pub control: BernoulliArmSummary,
    pub treatment: BernoulliArmSummary,
    pub effect_interval: BernoulliInterval,
    pub observed_effect: Option<f64>,
    pub attempted_samples: usize,
    pub unqualified_samples: usize,
    pub total_cost_units: u64,
    pub total_risk_units: u64,
    pub remaining_sample_budget: usize,
    pub claim: Option<BernoulliClaim>,
    /// Evaluator-only score fields, consumed only after the episode closes.
    pub true_effect_for_scoring: f64,
    pub confidence_interval_covers_truth: bool,
    pub claim_correct: Option<bool>,
    pub false_positive_claim: bool,
    pub trial_assessments: Vec<BernoulliTrialAssessment>,
    pub findings: Vec<BernoulliFinding>,
}

impl BernoulliSequentialReport {
    pub fn correctly_claims(&self) -> bool {
        self.qualification == BernoulliQualification::Qualified
            && self.claim_correct == Some(true)
            && self.findings.is_empty()
    }

    pub fn made_false_positive_claim(&self) -> bool {
        self.false_positive_claim
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct ArmCounts {
    attempted: usize,
    qualified: usize,
    successes: usize,
}

fn push_finding(
    findings: &mut Vec<BernoulliFinding>,
    code: BernoulliFindingCode,
    trial_index: Option<usize>,
    detail: impl Into<String>,
) {
    findings.push(BernoulliFinding { code, trial_index, detail: detail.into() });
}

/// Two-sided, time-uniform Hoeffding interval for one iid Bernoulli stream.
/// The caller passes alpha/2 to preserve total coverage across two arms.
fn arm_confidence_interval(successes: usize, samples: usize, alpha_arm: f64) -> BernoulliInterval {
    if samples == 0 {
        return BernoulliInterval { lower: 0.0, upper: 1.0 };
    }
    let n = samples as f64;
    let mean = successes as f64 / n;
    let alpha_at_n = alpha_arm * 6.0 / (PI * PI * n * n);
    let radius = ((2.0 / alpha_at_n).ln() / (2.0 * n)).sqrt().min(1.0);
    BernoulliInterval {
        lower: (mean - radius).max(0.0),
        upper: (mean + radius).min(1.0),
    }
}

fn effect_interval(
    control: &ArmCounts,
    treatment: &ArmCounts,
    alpha: f64,
) -> BernoulliInterval {
    let alpha_arm = alpha / 2.0;
    let control_interval =
        arm_confidence_interval(control.successes, control.qualified, alpha_arm);
    let treatment_interval =
        arm_confidence_interval(treatment.successes, treatment.qualified, alpha_arm);
    BernoulliInterval {
        lower: treatment_interval.lower - control_interval.upper,
        upper: treatment_interval.upper - control_interval.lower,
    }
}

fn boundary(interval: BernoulliInterval, minimum_effect: f64) -> BernoulliBoundary {
    if interval.lower > minimum_effect {
        BernoulliBoundary::Benefit
    } else if interval.upper < -minimum_effect {
        BernoulliBoundary::Harm
    } else if interval.lower >= -minimum_effect && interval.upper <= minimum_effect {
        BernoulliBoundary::NoMaterialEffect
    } else {
        BernoulliBoundary::Continue
    }
}

fn mean(successes: usize, samples: usize) -> Option<f64> {
    (samples > 0).then(|| successes as f64 / samples as f64)
}

fn arm_summary(counts: ArmCounts, alpha_arm: f64) -> BernoulliArmSummary {
    BernoulliArmSummary {
        attempted_samples: counts.attempted,
        qualified_samples: counts.qualified,
        successes: counts.successes,
        mean: mean(counts.successes, counts.qualified),
        interval: arm_confidence_interval(counts.successes, counts.qualified, alpha_arm),
    }
}

fn terminal_claim(terminal: &BernoulliTerminalDecision) -> Option<BernoulliClaim> {
    match terminal {
        BernoulliTerminalDecision::Stop { claim } => Some(*claim),
        BernoulliTerminalDecision::Abstain { .. } => None,
    }
}

fn claim_for_boundary(value: BernoulliBoundary) -> Option<BernoulliClaim> {
    match value {
        BernoulliBoundary::Benefit => Some(BernoulliClaim::Benefit),
        BernoulliBoundary::Harm => Some(BernoulliClaim::Harm),
        BernoulliBoundary::NoMaterialEffect => Some(BernoulliClaim::NoMaterialEffect),
        BernoulliBoundary::Continue => None,
    }
}

fn oracle_class(true_effect: f64, minimum_effect: f64) -> BernoulliClaim {
    if true_effect > minimum_effect {
        BernoulliClaim::Benefit
    } else if true_effect < -minimum_effect {
        BernoulliClaim::Harm
    } else {
        BernoulliClaim::NoMaterialEffect
    }
}

fn qualification(findings: &[BernoulliFinding], final_boundary: BernoulliBoundary) -> BernoulliQualification {
    if findings.is_empty() {
        return if final_boundary == BernoulliBoundary::Continue {
            BernoulliQualification::Inconclusive
        } else {
            BernoulliQualification::Qualified
        };
    }
    if findings.iter().all(|finding| {
        finding.code == BernoulliFindingCode::EvidenceUnavailable
    }) {
        BernoulliQualification::Inconclusive
    } else {
        BernoulliQualification::NotQualified
    }
}

/// Evaluate a sequential two-arm Bernoulli trace.
///
/// Attempted trials always consume sample, cost and risk budgets. Only trials
/// whose assignment, delivery, isolation and observation statuses are all
/// Verified contribute to either arm's confidence sequence.
pub fn evaluate_bernoulli_sequential(
    scenario: &BernoulliSequentialScenario,
    oracle: &BernoulliOracle,
    trace: &BernoulliSequentialTrace,
) -> BernoulliSequentialReport {
    let mut findings = Vec::new();
    if !scenario.alpha.is_finite() || scenario.alpha <= 0.0 || scenario.alpha >= 1.0 {
        push_finding(&mut findings, BernoulliFindingCode::InvalidAlpha, None,
            "alpha must be finite and strictly between zero and one");
    }
    if !scenario.minimum_effect.is_finite()
        || scenario.minimum_effect <= 0.0
        || scenario.minimum_effect > 1.0
    {
        push_finding(&mut findings, BernoulliFindingCode::InvalidMinimumEffect, None,
            "minimum_effect must be finite and in (0, 1]");
    }
    if scenario.max_total_samples == 0 || scenario.max_samples_per_arm == 0 {
        push_finding(&mut findings, BernoulliFindingCode::EmptySampleBudget, None,
            "sample budgets must both be positive");
    }
    if !oracle.control_probability.is_finite()
        || !(0.0..=1.0).contains(&oracle.control_probability)
        || !oracle.treatment_probability.is_finite()
        || !(0.0..=1.0).contains(&oracle.treatment_probability)
    {
        push_finding(&mut findings, BernoulliFindingCode::InvalidOracleProbability, None,
            "evaluator-only probabilities must be finite and in [0, 1]");
    }

    if !findings.is_empty() {
        let control = ArmCounts::default();
        let treatment = ArmCounts::default();
        let alpha = if scenario.alpha.is_finite() && scenario.alpha > 0.0 && scenario.alpha < 1.0 {
            scenario.alpha
        } else { 0.05 };
        let delta = if scenario.minimum_effect.is_finite()
            && scenario.minimum_effect > 0.0 && scenario.minimum_effect <= 1.0
        { scenario.minimum_effect } else { 0.1 };
        let interval = effect_interval(&control, &treatment, alpha);
        let true_effect = if oracle.control_probability.is_finite()
            && oracle.treatment_probability.is_finite()
            && (0.0..=1.0).contains(&oracle.control_probability)
            && (0.0..=1.0).contains(&oracle.treatment_probability)
        { oracle.true_effect() } else { 0.0 };
        return BernoulliSequentialReport {
            qualification: BernoulliQualification::Inconclusive,
            boundary: boundary(interval, delta),
            control: arm_summary(control, alpha / 2.0),
            treatment: arm_summary(treatment, alpha / 2.0),
            effect_interval: interval,
            observed_effect: None,
            attempted_samples: 0,
            unqualified_samples: 0,
            total_cost_units: 0,
            total_risk_units: 0,
            remaining_sample_budget: scenario.max_total_samples,
            claim: terminal_claim(&trace.terminal),
            true_effect_for_scoring: true_effect,
            confidence_interval_covers_truth: interval.lower <= true_effect && true_effect <= interval.upper,
            claim_correct: None,
            false_positive_claim: false,
            trial_assessments: Vec::new(),
            findings,
        };
    }

    let mut control = ArmCounts::default();
    let mut treatment = ArmCounts::default();
    let mut total_cost = 0_u64;
    let mut total_risk = 0_u64;
    let mut unqualified_samples = 0_usize;
    let mut assessments = Vec::new();

    for (index, trial) in trace.trials.iter().enumerate() {
        let trial_index = index + 1;
        let boundary_before = boundary(
            effect_interval(&control, &treatment, scenario.alpha),
            scenario.minimum_effect,
        );

        match trial.arm {
            BernoulliArm::Control => control.attempted += 1,
            BernoulliArm::Treatment => treatment.attempted += 1,
        }

        let next_cost = total_cost.checked_add(scenario.sample_cost_units);
        let next_risk = total_risk.checked_add(scenario.sample_risk_units);
        let cost_exceeded = next_cost.is_none_or(|value| value > scenario.max_cost_units);
        let risk_exceeded = next_risk.is_none_or(|value| value > scenario.max_risk_units);
        // Charge both dimensions for every attempted sample, even when either
        // budget is exceeded. Cost and risk are deliberately non-compensatory.
        total_cost = next_cost.unwrap_or(u64::MAX);
        total_risk = next_risk.unwrap_or(u64::MAX);
        if cost_exceeded {
            push_finding(&mut findings, BernoulliFindingCode::CostBudgetExceeded,
                Some(trial_index), "attempted sample exceeds cumulative cost budget");
        }
        if risk_exceeded {
            push_finding(&mut findings, BernoulliFindingCode::RiskBudgetExceeded,
                Some(trial_index), "attempted sample exceeds cumulative risk budget");
        }
        if cost_exceeded || risk_exceeded {
            break;
        }

        if trial_index > scenario.max_total_samples {
            push_finding(&mut findings, BernoulliFindingCode::TotalSampleBudgetExceeded,
                Some(trial_index), "trace exceeds total attempted-sample budget");
            break;
        }
        let arm_attempted = match trial.arm {
            BernoulliArm::Control => control.attempted,
            BernoulliArm::Treatment => treatment.attempted,
        };
        if arm_attempted > scenario.max_samples_per_arm {
            push_finding(&mut findings, BernoulliFindingCode::PerArmSampleBudgetExceeded,
                Some(trial_index), "trace exceeds per-arm attempted-sample budget");
            break;
        }
        if boundary_before != BernoulliBoundary::Continue {
            push_finding(&mut findings, BernoulliFindingCode::SamplingAfterBoundary,
                Some(trial_index), "a stopping boundary was crossed before this attempted sample");
            break;
        }

        let mut evidence_qualified = true;
        for (label, status) in [
            ("assignment", trial.assignment),
            ("delivery", trial.delivery),
            ("isolation", trial.isolation),
            ("observation", trial.observation),
        ] {
            match status {
                ExperimentEvidenceStatus::Verified => {}
                ExperimentEvidenceStatus::Rejected => {
                    evidence_qualified = false;
                    let code = match label {
                        "assignment" => BernoulliFindingCode::AssignmentRejected,
                        "delivery" => BernoulliFindingCode::DeliveryRejected,
                        "isolation" => BernoulliFindingCode::IsolationRejected,
                        _ => BernoulliFindingCode::ObservationRejected,
                    };
                    push_finding(&mut findings, code, Some(trial_index),
                        format!("{label} evidence was rejected"));
                }
                ExperimentEvidenceStatus::Unavailable => {
                    evidence_qualified = false;
                    push_finding(&mut findings, BernoulliFindingCode::EvidenceUnavailable,
                        Some(trial_index),
                        format!("{label} evidence unavailable; outcome cannot update the confidence sequence"));
                }
            }
        }

        if evidence_qualified {
            match trial.arm {
                BernoulliArm::Control => {
                    control.qualified += 1;
                    if trial.outcome { control.successes += 1; }
                }
                BernoulliArm::Treatment => {
                    treatment.qualified += 1;
                    if trial.outcome { treatment.successes += 1; }
                }
            }
        } else {
            unqualified_samples += 1;
        }

        let interval = effect_interval(&control, &treatment, scenario.alpha);
        let current_boundary = boundary(interval, scenario.minimum_effect);
        assessments.push(BernoulliTrialAssessment {
            trial_index,
            arm: trial.arm,
            evidence_qualified,
            qualified_control_samples: control.qualified,
            qualified_treatment_samples: treatment.qualified,
            treatment_minus_control: interval,
            boundary: current_boundary,
        });
    }

    let interval = effect_interval(&control, &treatment, scenario.alpha);
    let final_boundary = boundary(interval, scenario.minimum_effect);
    let claim = terminal_claim(&trace.terminal);

    match &trace.terminal {
        BernoulliTerminalDecision::Stop { claim: declared } => {
            if final_boundary == BernoulliBoundary::Continue {
                push_finding(&mut findings, BernoulliFindingCode::PrematureStop, None,
                    "final confidence sequence does not support a stopping claim");
            } else if claim_for_boundary(final_boundary) != Some(*declared) {
                push_finding(&mut findings, BernoulliFindingCode::WrongClaim, None,
                    "declared claim differs from the confidence-sequence boundary");
            }
        }
        BernoulliTerminalDecision::Abstain { reason } => {
            if reason.trim().is_empty() {
                push_finding(&mut findings, BernoulliFindingCode::EmptyAbstentionReason, None,
                    "abstention requires an explicit non-empty reason");
            }
            if final_boundary != BernoulliBoundary::Continue {
                push_finding(&mut findings, BernoulliFindingCode::AbstainedAfterBoundary, None,
                    "trace abstained after a preregistered decision boundary was crossed");
            }
        }
    }

    let true_effect = oracle.true_effect();
    let actual_class = oracle_class(true_effect, scenario.minimum_effect);
    let claim_correct = claim.map(|declared| declared == actual_class);
    let false_positive_claim = match claim {
        Some(BernoulliClaim::Benefit) => actual_class != BernoulliClaim::Benefit,
        Some(BernoulliClaim::Harm) => actual_class != BernoulliClaim::Harm,
        Some(BernoulliClaim::NoMaterialEffect) | None => false,
    };
    let control_summary = arm_summary(control, scenario.alpha / 2.0);
    let treatment_summary = arm_summary(treatment, scenario.alpha / 2.0);
    let observed_effect = match (control_summary.mean, treatment_summary.mean) {
        (Some(control_mean), Some(treatment_mean)) => Some(treatment_mean - control_mean),
        _ => None,
    };
    let attempted_samples = control.attempted.saturating_add(treatment.attempted);
    let qualification = qualification(&findings, final_boundary);

    BernoulliSequentialReport {
        qualification,
        boundary: final_boundary,
        control: control_summary,
        treatment: treatment_summary,
        effect_interval: interval,
        observed_effect,
        attempted_samples,
        unqualified_samples,
        total_cost_units: total_cost,
        total_risk_units: total_risk,
        remaining_sample_budget: scenario.max_total_samples.saturating_sub(attempted_samples),
        claim,
        true_effect_for_scoring: true_effect,
        confidence_interval_covers_truth: interval.lower <= true_effect && true_effect <= interval.upper,
        claim_correct,
        false_positive_claim,
        trial_assessments: assessments,
        findings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scenario() -> BernoulliSequentialScenario {
        BernoulliSequentialScenario::new(0.05, 0.10, 500, 250, 1, 500, 1, 500)
    }

    fn positive_trace_until_boundary(
        scenario: &BernoulliSequentialScenario,
        add_extra_sample_after_boundary: bool,
    ) -> BernoulliSequentialTrace {
        let mut control = ArmCounts::default();
        let mut treatment = ArmCounts::default();
        let mut trials = Vec::new();
        for _ in 0..scenario.max_samples_per_arm {
            trials.push(BernoulliTrial::qualified(BernoulliArm::Control, false));
            control.attempted += 1;
            control.qualified += 1;

            trials.push(BernoulliTrial::qualified(BernoulliArm::Treatment, true));
            treatment.attempted += 1;
            treatment.qualified += 1;
            treatment.successes += 1;

            let interval = effect_interval(&control, &treatment, scenario.alpha);
            if boundary(interval, scenario.minimum_effect) != BernoulliBoundary::Continue {
                break;
            }
        }
        if add_extra_sample_after_boundary {
            trials.push(BernoulliTrial::qualified(BernoulliArm::Treatment, true));
        }
        BernoulliSequentialTrace {
            trials,
            terminal: BernoulliTerminalDecision::Stop { claim: BernoulliClaim::Benefit },
        }
    }

    #[test]
    fn zero_and_small_sample_intervals_remain_wide() {
        let empty = arm_confidence_interval(0, 0, 0.025);
        assert_eq!(empty, BernoulliInterval { lower: 0.0, upper: 1.0 });
        let one = arm_confidence_interval(1, 1, 0.025);
        assert_eq!(one.lower, 0.0);
        assert_eq!(one.upper, 1.0);
        let few = arm_confidence_interval(2, 3, 0.025);
        assert!(few.lower <= 2.0 / 3.0);
        assert!(few.upper >= 2.0 / 3.0);
    }

    #[test]
    fn unavailable_assignment_does_not_update_but_still_consumes_resources() {
        let mut scenario = scenario();
        scenario.max_total_samples = 1;
        scenario.max_samples_per_arm = 1;
        let mut trial = BernoulliTrial::qualified(BernoulliArm::Treatment, true);
        trial.assignment = ExperimentEvidenceStatus::Unavailable;
        let trace = BernoulliSequentialTrace {
            trials: vec![trial],
            terminal: BernoulliTerminalDecision::Abstain { reason: "assignment receipt unavailable".into() },
        };
        let report = evaluate_bernoulli_sequential(&scenario, &BernoulliOracle::new(0.5, 0.8), &trace);
        assert_eq!(report.qualification, BernoulliQualification::Inconclusive);
        assert_eq!(report.attempted_samples, 1);
        assert_eq!(report.unqualified_samples, 1);
        assert_eq!(report.treatment.qualified_samples, 0);
        assert_eq!(report.total_cost_units, 1);
        assert_eq!(report.total_risk_units, 1);
        assert_eq!(report.boundary, BernoulliBoundary::Continue);
    }

    #[test]
    fn rejected_isolation_does_not_contribute_to_a_claim() {
        let mut scenario = scenario();
        scenario.max_total_samples = 1;
        scenario.max_samples_per_arm = 1;
        let mut trial = BernoulliTrial::qualified(BernoulliArm::Treatment, true);
        trial.isolation = ExperimentEvidenceStatus::Rejected;
        let trace = BernoulliSequentialTrace {
            trials: vec![trial],
            terminal: BernoulliTerminalDecision::Stop { claim: BernoulliClaim::Benefit },
        };
        let report = evaluate_bernoulli_sequential(&scenario, &BernoulliOracle::new(0.5, 0.8), &trace);
        assert_eq!(report.qualification, BernoulliQualification::NotQualified);
        assert_eq!(report.treatment.qualified_samples, 0);
        assert!(report.findings.iter().any(|f| f.code == BernoulliFindingCode::PrematureStop));
    }

    #[test]
    fn positive_control_stops_only_after_anytime_boundary_and_covers_truth() {
        let scenario = scenario();
        let oracle = BernoulliOracle::new(0.0, 1.0);
        let trace = positive_trace_until_boundary(&scenario, false);
        let report = evaluate_bernoulli_sequential(&scenario, &oracle, &trace);
        assert_eq!(report.qualification, BernoulliQualification::Qualified);
        assert_eq!(report.boundary, BernoulliBoundary::Benefit);
        assert!(report.confidence_interval_covers_truth);
        assert_eq!(report.claim_correct, Some(true));
        assert!(!report.false_positive_claim);
    }

    #[test]
    fn sampling_after_crossing_the_boundary_is_rejected() {
        let scenario = scenario();
        let trace = positive_trace_until_boundary(&scenario, true);
        let report = evaluate_bernoulli_sequential(&scenario, &BernoulliOracle::new(0.0, 1.0), &trace);
        assert_eq!(report.qualification, BernoulliQualification::NotQualified);
        assert!(report.findings.iter().any(|f| f.code == BernoulliFindingCode::SamplingAfterBoundary));
    }

    #[test]
    fn claim_before_boundary_is_rejected() {
        let scenario = scenario();
        let trace = BernoulliSequentialTrace {
            trials: Vec::new(),
            terminal: BernoulliTerminalDecision::Stop { claim: BernoulliClaim::Benefit },
        };
        let report = evaluate_bernoulli_sequential(&scenario, &BernoulliOracle::new(0.4, 0.8), &trace);
        assert_eq!(report.qualification, BernoulliQualification::NotQualified);
        assert!(report.findings.iter().any(|f| f.code == BernoulliFindingCode::PrematureStop));
    }

    #[test]
    fn wrong_claim_is_rejected_even_when_another_boundary_is_crossed() {
        let scenario = scenario();
        let mut trace = positive_trace_until_boundary(&scenario, false);
        trace.terminal = BernoulliTerminalDecision::Stop { claim: BernoulliClaim::Harm };
        let report = evaluate_bernoulli_sequential(&scenario, &BernoulliOracle::new(0.0, 1.0), &trace);
        assert_eq!(report.qualification, BernoulliQualification::NotQualified);
        assert!(report.findings.iter().any(|f| f.code == BernoulliFindingCode::WrongClaim));
    }

    #[test]
    fn false_positive_claim_is_scored_against_hidden_oracle_only_after_episode() {
        let scenario = scenario();
        let trace = positive_trace_until_boundary(&scenario, false);
        let report = evaluate_bernoulli_sequential(&scenario, &BernoulliOracle::new(0.5, 0.5), &trace);
        assert_eq!(report.boundary, BernoulliBoundary::Benefit);
        assert_eq!(report.claim_correct, Some(false));
        assert!(report.false_positive_claim);
    }


    #[test]
    fn cost_and_risk_are_both_charged_and_fail_independently() {
        let mut scenario = scenario();
        scenario.max_total_samples = 1;
        scenario.max_samples_per_arm = 1;
        scenario.max_cost_units = 0;
        scenario.max_risk_units = 0;
        let trace = BernoulliSequentialTrace {
            trials: vec![BernoulliTrial::qualified(BernoulliArm::Control, false)],
            terminal: BernoulliTerminalDecision::Abstain { reason: "budgets exhausted".into() },
        };
        let report = evaluate_bernoulli_sequential(&scenario, &BernoulliOracle::new(0.5, 0.5), &trace);
        assert_eq!(report.total_cost_units, 1);
        assert_eq!(report.total_risk_units, 1);
        assert!(report.findings.iter().any(|f| f.code == BernoulliFindingCode::CostBudgetExceeded));
        assert!(report.findings.iter().any(|f| f.code == BernoulliFindingCode::RiskBudgetExceeded));
    }

    #[test]
    fn harm_control_stops_only_after_the_negative_anytime_boundary() {
        let scenario = scenario();
        let mut control = ArmCounts::default();
        let mut treatment = ArmCounts::default();
        let mut trials = Vec::new();
        for _ in 0..scenario.max_samples_per_arm {
            trials.push(BernoulliTrial::qualified(BernoulliArm::Control, true));
            control.attempted += 1;
            control.qualified += 1;
            control.successes += 1;
            trials.push(BernoulliTrial::qualified(BernoulliArm::Treatment, false));
            treatment.attempted += 1;
            treatment.qualified += 1;
            if boundary(effect_interval(&control, &treatment, scenario.alpha),
                scenario.minimum_effect) != BernoulliBoundary::Continue {
                break;
            }
        }
        let trace = BernoulliSequentialTrace {
            trials,
            terminal: BernoulliTerminalDecision::Stop { claim: BernoulliClaim::Harm },
        };
        let report = evaluate_bernoulli_sequential(&scenario, &BernoulliOracle::new(1.0, 0.0), &trace);
        assert_eq!(report.qualification, BernoulliQualification::Qualified);
        assert_eq!(report.boundary, BernoulliBoundary::Harm);
        assert_eq!(report.claim_correct, Some(true));
    }

    #[test]
    fn no_material_effect_requires_the_full_interval_inside_the_equivalence_region() {
        let scenario = BernoulliSequentialScenario::new(0.05, 0.50, 500, 250, 1, 500, 1, 500);
        let mut control = ArmCounts::default();
        let mut treatment = ArmCounts::default();
        let mut trials = Vec::new();
        for i in 0..scenario.max_samples_per_arm {
            let outcome = i % 2 == 0;
            trials.push(BernoulliTrial::qualified(BernoulliArm::Control, outcome));
            control.attempted += 1;
            control.qualified += 1;
            if outcome { control.successes += 1; }
            trials.push(BernoulliTrial::qualified(BernoulliArm::Treatment, outcome));
            treatment.attempted += 1;
            treatment.qualified += 1;
            if outcome { treatment.successes += 1; }
            if boundary(effect_interval(&control, &treatment, scenario.alpha),
                scenario.minimum_effect) == BernoulliBoundary::NoMaterialEffect {
                break;
            }
        }
        let trace = BernoulliSequentialTrace {
            trials,
            terminal: BernoulliTerminalDecision::Stop { claim: BernoulliClaim::NoMaterialEffect },
        };
        let report = evaluate_bernoulli_sequential(&scenario, &BernoulliOracle::new(0.5, 0.5), &trace);
        assert_eq!(report.qualification, BernoulliQualification::Qualified);
        assert_eq!(report.boundary, BernoulliBoundary::NoMaterialEffect);
        assert_eq!(report.claim_correct, Some(true));
        assert!(report.effect_interval.lower >= -0.50);
        assert!(report.effect_interval.upper <= 0.50);
    }

    #[test]
    fn unqualified_attempts_consume_both_sample_and_arm_budgets() {
        let mut scenario = scenario();
        scenario.max_total_samples = 2;
        scenario.max_samples_per_arm = 1;
        let mut first = BernoulliTrial::qualified(BernoulliArm::Control, false);
        first.observation = ExperimentEvidenceStatus::Unavailable;
        let second = BernoulliTrial::qualified(BernoulliArm::Control, false);
        let trace = BernoulliSequentialTrace {
            trials: vec![first, second],
            terminal: BernoulliTerminalDecision::Abstain { reason: "unavailable first observation".into() },
        };
        let report = evaluate_bernoulli_sequential(&scenario, &BernoulliOracle::new(0.5, 0.5), &trace);
        assert_eq!(report.qualification, BernoulliQualification::NotQualified);
        assert_eq!(report.control.attempted_samples, 2);
        assert_eq!(report.control.qualified_samples, 0);
        assert!(report.findings.iter().any(|f| f.code == BernoulliFindingCode::PerArmSampleBudgetExceeded));
    }
}
