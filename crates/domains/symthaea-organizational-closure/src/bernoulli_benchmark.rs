//! Suite-level operating-characteristic and process-integrity metrics for
//! the AOBENCH stochastic Bernoulli sequential assay.

use super::bernoulli_sequential::{
    BernoulliFindingCode, BernoulliQualification, BernoulliSequentialReport,
};

const WILSON_Z_95: f64 = 1.959_963_984_540_054;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RateEstimate {
    pub successes: usize,
    pub episodes: usize,
    pub estimate: Option<f64>,
    pub lower_95: Option<f64>,
    pub upper_95: Option<f64>,
}

impl RateEstimate {
    fn from_counts(successes: usize, episodes: usize) -> Self {
        if episodes == 0 {
            return Self {
                successes,
                episodes,
                estimate: None,
                lower_95: None,
                upper_95: None,
            };
        }
        let n = episodes as f64;
        let p = successes as f64 / n;
        let z = WILSON_Z_95;
        let z2 = z * z;
        let denominator = 1.0 + z2 / n;
        let center = (p + z2 / (2.0 * n)) / denominator;
        let half_width = (z / denominator)
            * ((p * (1.0 - p) / n) + z2 / (4.0 * n * n)).sqrt();
        Self {
            successes,
            episodes,
            estimate: Some(p),
            lower_95: Some((center - half_width).max(0.0)),
            upper_95: Some((center + half_width).min(1.0)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BernoulliBenchmarkError {
    InvalidMinimumEffect,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BernoulliBenchmarkSummary {
    pub episode_count: usize,
    pub oracle_valid_episodes: usize,
    pub oracle_invalid_episodes: usize,
    pub qualified_episodes: usize,
    pub not_qualified_episodes: usize,
    pub inconclusive_episodes: usize,
    pub null_episodes: usize,
    pub material_effect_episodes: usize,
    pub coverage_eligible_episodes: usize,
    /// False directional claims divided by valid-oracle null episodes.
    pub false_positive_rate_on_null: RateEstimate,
    /// Qualified, correct directional claims divided by materially non-null episodes.
    pub qualified_power_on_material_effect: RateEstimate,
    /// Correct terminal claims divided by all valid-oracle episodes.
    pub correct_claim_rate: RateEstimate,
    /// Final-look coverage divided by coverage-eligible episodes.
    pub final_interval_coverage: RateEstimate,
    /// All-interim-look coverage divided by coverage-eligible episodes.
    pub all_look_confidence_sequence_coverage: RateEstimate,
    /// Process qualification divided by all attempted episodes.
    pub process_qualification_rate: RateEstimate,
    pub mean_attempted_samples: Option<f64>,
    pub mean_cost_units: Option<f64>,
    pub mean_risk_units: Option<f64>,
}

fn has_finding(report: &BernoulliSequentialReport, code: BernoulliFindingCode) -> bool {
    report.findings.iter().any(|finding| finding.code == code)
}

fn valid_oracle(report: &BernoulliSequentialReport) -> bool {
    !has_finding(report, BernoulliFindingCode::InvalidOracleProbability)
        && report.true_effect_for_scoring.is_finite()
        && (-1.0..=1.0).contains(&report.true_effect_for_scoring)
}

fn coverage_eligible(report: &BernoulliSequentialReport) -> bool {
    valid_oracle(report)
        && !has_finding(report, BernoulliFindingCode::InvalidAlpha)
        && report.unqualified_samples == 0
}

/// Summarize a frozen episode batch without collapsing distinct scientific
/// outcomes into one score. Invalid-oracle episodes remain in process totals
/// but are excluded from truth-conditioned and coverage rates.
pub fn summarize_bernoulli_episodes(
    reports: &[BernoulliSequentialReport],
    minimum_effect: f64,
) -> Result<BernoulliBenchmarkSummary, BernoulliBenchmarkError> {
    if !minimum_effect.is_finite() || minimum_effect <= 0.0 || minimum_effect > 1.0 {
        return Err(BernoulliBenchmarkError::InvalidMinimumEffect);
    }

    let mut oracle_valid = 0_usize;
    let mut oracle_invalid = 0_usize;
    let mut qualified = 0_usize;
    let mut not_qualified = 0_usize;
    let mut inconclusive = 0_usize;
    let mut null_episodes = 0_usize;
    let mut material_episodes = 0_usize;
    let mut false_positives = 0_usize;
    let mut qualified_power = 0_usize;
    let mut correct_claims = 0_usize;
    let mut coverage_eligible_count = 0_usize;
    let mut final_coverage = 0_usize;
    let mut all_look_coverage = 0_usize;
    let mut attempted_samples = 0_u128;
    let mut cost_units = 0_u128;
    let mut risk_units = 0_u128;

    for report in reports {
        match report.qualification {
            BernoulliQualification::Qualified => qualified += 1,
            BernoulliQualification::NotQualified => not_qualified += 1,
            BernoulliQualification::Inconclusive => inconclusive += 1,
        }
        attempted_samples += report.attempted_samples as u128;
        cost_units += report.total_cost_units as u128;
        risk_units += report.total_risk_units as u128;

        if !valid_oracle(report) {
            oracle_invalid += 1;
            continue;
        }
        oracle_valid += 1;

        let is_null = report.true_effect_for_scoring.abs() <= minimum_effect;
        if is_null {
            null_episodes += 1;
            if report.false_positive_claim {
                false_positives += 1;
            }
        } else {
            material_episodes += 1;
            if report.correctly_claims() {
                qualified_power += 1;
            }
        }
        if report.claim_correct == Some(true) {
            correct_claims += 1;
        }

        if coverage_eligible(report) {
            coverage_eligible_count += 1;
            if report.confidence_interval_covers_truth {
                final_coverage += 1;
            }
            if report.confidence_sequence_covers_truth_at_all_looks {
                all_look_coverage += 1;
            }
        }
    }

    let episode_count = reports.len();
    let average = |total: u128| {
        (episode_count > 0).then(|| total as f64 / episode_count as f64)
    };

    Ok(BernoulliBenchmarkSummary {
        episode_count,
        oracle_valid_episodes: oracle_valid,
        oracle_invalid_episodes: oracle_invalid,
        qualified_episodes: qualified,
        not_qualified_episodes: not_qualified,
        inconclusive_episodes: inconclusive,
        null_episodes,
        material_effect_episodes: material_episodes,
        coverage_eligible_episodes: coverage_eligible_count,
        false_positive_rate_on_null: RateEstimate::from_counts(false_positives, null_episodes),
        qualified_power_on_material_effect: RateEstimate::from_counts(
            qualified_power,
            material_episodes,
        ),
        correct_claim_rate: RateEstimate::from_counts(correct_claims, oracle_valid),
        final_interval_coverage: RateEstimate::from_counts(final_coverage, coverage_eligible_count),
        all_look_confidence_sequence_coverage: RateEstimate::from_counts(
            all_look_coverage,
            coverage_eligible_count,
        ),
        process_qualification_rate: RateEstimate::from_counts(qualified, episode_count),
        mean_attempted_samples: average(attempted_samples),
        mean_cost_units: average(cost_units),
        mean_risk_units: average(risk_units),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bernoulli_sequential::{
        BernoulliArm, BernoulliBoundary, BernoulliClaim, BernoulliOracle,
        BernoulliSequentialScenario, BernoulliSequentialTrace, BernoulliTerminalDecision,
        BernoulliTrial, evaluate_bernoulli_sequential,
    };
    use crate::bernoulli_sequential::{BernoulliInterval, BernoulliQualification};

    fn scenario() -> BernoulliSequentialScenario {
        BernoulliSequentialScenario::new(0.05, 0.10, 500, 250, 1, 500, 1, 500)
    }

    fn positive_trace() -> BernoulliSequentialTrace {
        let scenario = scenario();
        let mut control_successes = 0_usize;
        let mut treatment_successes = 0_usize;
        let mut control_n = 0_usize;
        let mut treatment_n = 0_usize;
        let mut trials = Vec::new();
        for _ in 0..scenario.max_samples_per_arm {
            trials.push(BernoulliTrial::qualified(BernoulliArm::Control, false));
            control_n += 1;
            trials.push(BernoulliTrial::qualified(BernoulliArm::Treatment, true));
            treatment_n += 1;
            treatment_successes += 1;
            let control = crate::bernoulli_sequential::BernoulliInterval {
                lower: (control_successes as f64 / control_n as f64),
                upper: (control_successes as f64 / control_n as f64),
            };
            let treatment = crate::bernoulli_sequential::BernoulliInterval {
                lower: (treatment_successes as f64 / treatment_n as f64),
                upper: (treatment_successes as f64 / treatment_n as f64),
            };
            let _ = (control, treatment);
            // This fixture has deterministic 0/1 streams; the evaluator's
            // actual confidence sequence determines the stop point below.
            if trials.len() >= 100 {
                break;
            }
        }
        // The chosen conservative boundary is crossed well before 100 trials.
        BernoulliSequentialTrace {
            trials,
            terminal: BernoulliTerminalDecision::Stop { claim: BernoulliClaim::Benefit },
        }
    }

    #[test]
    fn wilson_interval_handles_zero_all_and_middle_counts() {
        let empty = RateEstimate::from_counts(0, 0);
        assert_eq!(empty.estimate, None);
        assert_eq!(empty.lower_95, None);

        let zero = RateEstimate::from_counts(0, 10);
        let all = RateEstimate::from_counts(10, 10);
        let half = RateEstimate::from_counts(5, 10);
        assert_eq!(zero.estimate, Some(0.0));
        assert_eq!(all.estimate, Some(1.0));
        assert_eq!(half.estimate, Some(0.5));
        assert_eq!(zero.lower_95, Some(0.0));
        assert_eq!(all.upper_95, Some(1.0));
        assert!(half.lower_95.unwrap() < 0.5);
        assert!(half.upper_95.unwrap() > 0.5);
    }

    #[test]
    fn summary_keeps_null_power_coverage_and_process_denominators_separate() {
        let scenario = scenario();
        let positive = positive_trace();
        let false_positive = evaluate_bernoulli_sequential(
            &scenario,
            &BernoulliOracle::new(0.5, 0.5),
            &positive,
        );
        let correct_positive = evaluate_bernoulli_sequential(
            &scenario,
            &BernoulliOracle::new(0.0, 1.0),
            &positive,
        );
        let abstain = BernoulliSequentialTrace {
            trials: Vec::new(),
            terminal: BernoulliTerminalDecision::Abstain { reason: "no samples requested".into() },
        };
        let inconclusive_null = evaluate_bernoulli_sequential(
            &scenario,
            &BernoulliOracle::new(0.5, 0.5),
            &abstain,
        );
        let invalid_oracle = evaluate_bernoulli_sequential(
            &scenario,
            &BernoulliOracle::new(1.5, 0.5),
            &abstain,
        );

        let summary = summarize_bernoulli_episodes(
            &[false_positive, correct_positive, inconclusive_null, invalid_oracle],
            scenario.minimum_effect,
        ).unwrap();

        assert_eq!(summary.episode_count, 4);
        assert_eq!(summary.oracle_valid_episodes, 3);
        assert_eq!(summary.oracle_invalid_episodes, 1);
        assert_eq!(summary.qualified_episodes, 2);
        assert_eq!(summary.inconclusive_episodes, 2);
        assert_eq!(summary.null_episodes, 2);
        assert_eq!(summary.material_effect_episodes, 1);
        assert_eq!(summary.false_positive_rate_on_null.successes, 1);
        assert_eq!(summary.false_positive_rate_on_null.episodes, 2);
        assert_eq!(summary.qualified_power_on_material_effect.successes, 1);
        assert_eq!(summary.qualified_power_on_material_effect.episodes, 1);
        assert_eq!(summary.correct_claim_rate.successes, 1);
        assert_eq!(summary.correct_claim_rate.episodes, 3);
        assert_eq!(summary.all_look_confidence_sequence_coverage.episodes, 3);
        assert_eq!(summary.process_qualification_rate.episodes, 4);
        assert!(summary.mean_attempted_samples.is_some());
    }

    #[test]
    fn invalid_margin_is_rejected_and_invalid_oracle_is_not_truth_scored() {
        assert_eq!(
            summarize_bernoulli_episodes(&[], 0.0),
            Err(BernoulliBenchmarkError::InvalidMinimumEffect)
        );
        let invalid = BernoulliSequentialReport {
            qualification: BernoulliQualification::Inconclusive,
            boundary: BernoulliBoundary::Continue,
            control: crate::bernoulli_sequential::BernoulliArmSummary {
                attempted_samples: 0,
                qualified_samples: 0,
                successes: 0,
                mean: None,
                interval: BernoulliInterval { lower: 0.0, upper: 1.0 },
            },
            treatment: crate::bernoulli_sequential::BernoulliArmSummary {
                attempted_samples: 0,
                qualified_samples: 0,
                successes: 0,
                mean: None,
                interval: BernoulliInterval { lower: 0.0, upper: 1.0 },
            },
            effect_interval: BernoulliInterval { lower: -1.0, upper: 1.0 },
            observed_effect: None,
            attempted_samples: 0,
            unqualified_samples: 0,
            total_cost_units: 0,
            total_risk_units: 0,
            remaining_sample_budget: 0,
            claim: None,
            true_effect_for_scoring: 0.0,
            confidence_interval_covers_truth: true,
            confidence_sequence_covers_truth_at_all_looks: true,
            claim_correct: None,
            false_positive_claim: false,
            trial_assessments: Vec::new(),
            findings: vec![crate::bernoulli_sequential::BernoulliFinding {
                code: BernoulliFindingCode::InvalidOracleProbability,
                trial_index: None,
                detail: "invalid fixture".into(),
            }],
        };
        let summary = summarize_bernoulli_episodes(&[invalid], 0.1).unwrap();
        assert_eq!(summary.oracle_invalid_episodes, 1);
        assert_eq!(summary.null_episodes, 0);
        assert_eq!(summary.false_positive_rate_on_null.episodes, 0);
        assert_eq!(summary.false_positive_rate_on_null.estimate, None);
    }
}
