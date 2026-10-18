//! Probability calibration for recovery and intervention signals.
//!
//! Forecasts must be recorded before the outcome is observed. An invalid run,
//! timeout, or safety stop is treated as a failure of end-to-end task completion
//! when building the binary outcome; it must not be silently dropped.
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq)]
pub struct SuccessForecast {
    pub task_id: String,
    pub variant_id: String,
    pub trial_id: String,
    /// Probability emitted before the trial of achieving the benchmark's full
    /// completion criterion. Must be finite and lie in [0, 1].
    pub predicted_success_probability: f64,
    /// True only if the trial achieved the full task completion criterion.
    pub success: bool,
    /// False if the trial timed out, errored, was interrupted, or had no valid run.
    pub valid_run: bool,
}

impl SuccessForecast {
    pub fn validate(&self) -> Result<(), CalibrationError> {
        if self.task_id.trim().is_empty()
            || self.variant_id.trim().is_empty()
            || self.trial_id.trim().is_empty()
        {
            return Err(CalibrationError::EmptyIdentifier);
        }
        let p = self.predicted_success_probability;
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(CalibrationError::ProbabilityOutOfRange);
        }
        if self.success && !self.valid_run {
            return Err(CalibrationError::SuccessOnInvalidRun);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationError {
    EmptyForecasts,
    EmptyIdentifier,
    ProbabilityOutOfRange,
    SuccessOnInvalidRun,
    DuplicateForecast,
    InvalidBinCount,
    InvalidConfidenceThreshold,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReliabilityBin {
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub count: usize,
    pub mean_predicted_probability: f64,
    pub observed_success_rate: f64,
    pub absolute_calibration_gap: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RiskCoveragePoint {
    /// Fraction of trials retained, starting at 1 / N and ending at 1.
    pub coverage: f64,
    /// Failure fraction among the retained highest-confidence trials.
    pub risk: f64,
    pub retained_trials: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationSummary {
    pub task_id: String,
    pub variant_id: String,
    /// Includes operationally invalid runs. Invalid runs are failures, not missing labels.
    pub total_trials: usize,
    pub valid_runs: usize,
    pub successes: usize,
    pub failures: usize,
    pub mean_predicted_probability: f64,
    pub observed_success_rate: f64,
    pub brier_score: f64,
    pub log_loss: f64,
    pub expected_calibration_error: f64,
    pub confidence_threshold: f64,
    pub high_confidence_trials: usize,
    pub high_confidence_failure_rate: Option<f64>,
    /// Area under the risk-versus-coverage curve; lower is better when confidence ranks failures.
    pub risk_coverage_auc: f64,
    pub reliability_bins: Vec<ReliabilityBin>,
    pub risk_coverage: Vec<RiskCoveragePoint>,
}

/// Summarize full-completion probability forecasts per task and policy variant.
///
/// ECE uses fixed-width probability bins. The risk-coverage curve retains trials
/// in descending predicted-success probability; ties are ordered by trial ID.
/// The curve and all error metrics include invalid runs, which count as failures
/// for end-to-end reliability. Probabilities must have been generated before
/// outcomes are known; this module cannot enforce that data-collection property.
pub fn summarize_calibration(
    forecasts: &[SuccessForecast],
    bin_count: usize,
    confidence_threshold: f64,
) -> Result<Vec<CalibrationSummary>, CalibrationError> {
    if forecasts.is_empty() { return Err(CalibrationError::EmptyForecasts); }
    if bin_count == 0 { return Err(CalibrationError::InvalidBinCount); }
    if !confidence_threshold.is_finite() || !(0.0..=1.0).contains(&confidence_threshold) {
        return Err(CalibrationError::InvalidConfidenceThreshold);
    }

    let mut groups: BTreeMap<(String, String), Vec<&SuccessForecast>> = BTreeMap::new();
    let mut seen = BTreeSet::new();
    for record in forecasts {
        record.validate()?;
        let key = (record.task_id.clone(), record.variant_id.clone(), record.trial_id.clone());
        if !seen.insert(key) { return Err(CalibrationError::DuplicateForecast); }
        groups.entry((record.task_id.clone(), record.variant_id.clone())).or_default().push(record);
    }

    let mut summaries = Vec::with_capacity(groups.len());
    for ((task_id, variant_id), rows) in groups {
        let n = rows.len();
        let mut bins: Vec<Vec<&SuccessForecast>> = (0..bin_count).map(|_| Vec::new()).collect();
        let mut probability_sum = 0.0;
        let mut brier_sum = 0.0;
        let mut log_loss_sum = 0.0;
        let mut successes = 0usize;
        let mut high_count = 0usize;
        let mut high_failures = 0usize;
        let mut ranked: Vec<&SuccessForecast> = rows.clone();

        for row in &rows {
            let p = row.predicted_success_probability;
            let outcome = if row.success { 1.0 } else { 0.0 };
            probability_sum += p;
            brier_sum += (p - outcome) * (p - outcome);
            let clipped = p.clamp(1e-15, 1.0 - 1e-15);
            log_loss_sum -= outcome * clipped.ln() + (1.0 - outcome) * (1.0 - clipped).ln();
            if row.success { successes += 1; }
            if p >= confidence_threshold {
                high_count += 1;
                if !row.success { high_failures += 1; }
            }
            let bin_index = ((p * bin_count as f64).floor() as usize).min(bin_count - 1);
            bins[bin_index].push(*row);
        }

        let reliability_bins: Vec<ReliabilityBin> = bins.into_iter().enumerate().filter_map(|(i, bin)| {
            if bin.is_empty() { return None; }
            let count = bin.len();
            let mean_p = bin.iter().map(|r| r.predicted_success_probability).sum::<f64>() / count as f64;
            let observed = bin.iter().filter(|r| r.success).count() as f64 / count as f64;
            Some(ReliabilityBin {
                lower_bound: i as f64 / bin_count as f64,
                upper_bound: (i + 1) as f64 / bin_count as f64,
                count,
                mean_predicted_probability: mean_p,
                observed_success_rate: observed,
                absolute_calibration_gap: (mean_p - observed).abs(),
            })
        }).collect();

        let expected_calibration_error = reliability_bins.iter()
            .map(|bin| bin.count as f64 / n as f64 * bin.absolute_calibration_gap)
            .sum::<f64>();

        ranked.sort_by(|a, b| {
            b.predicted_success_probability.total_cmp(&a.predicted_success_probability)
                .then_with(|| a.trial_id.cmp(&b.trial_id))
        });
        let mut risk_coverage = Vec::with_capacity(n);
        let mut failures_seen = 0usize;
        let mut previous_coverage = 0.0;
        let mut previous_risk = 0.0;
        let mut risk_coverage_auc = 0.0;
        for (index, row) in ranked.iter().enumerate() {
            if !row.success { failures_seen += 1; }
            let retained = index + 1;
            let coverage = retained as f64 / n as f64;
            let risk = failures_seen as f64 / retained as f64;
            risk_coverage_auc += (coverage - previous_coverage) * (risk + previous_risk) / 2.0;
            risk_coverage.push(RiskCoveragePoint { coverage, risk, retained_trials: retained });
            previous_coverage = coverage;
            previous_risk = risk;
        }

        summaries.push(CalibrationSummary {
            task_id,
            variant_id,
            total_trials: n,
            valid_runs: rows.iter().filter(|r| r.valid_run).count(),
            successes,
            failures: n - successes,
            mean_predicted_probability: probability_sum / n as f64,
            observed_success_rate: successes as f64 / n as f64,
            brier_score: brier_sum / n as f64,
            log_loss: log_loss_sum / n as f64,
            expected_calibration_error,
            confidence_threshold,
            high_confidence_trials: high_count,
            high_confidence_failure_rate: (high_count > 0).then_some(high_failures as f64 / high_count as f64),
            risk_coverage_auc,
            reliability_bins,
            risk_coverage,
        });
    }
    Ok(summaries)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn forecast(id: &str, probability: f64, success: bool, valid_run: bool) -> SuccessForecast {
        SuccessForecast {
            task_id: "store_in_safe".to_owned(),
            variant_id: "sidecar".to_owned(),
            trial_id: id.to_owned(),
            predicted_success_probability: probability,
            success,
            valid_run,
        }
    }

    #[test]
    fn perfect_forecasts_have_zero_brier_and_calibration_error() {
        let rows = vec![forecast("a", 0.0, false, true), forecast("b", 1.0, true, true)];
        let summary = summarize_calibration(&rows, 5, 0.8).unwrap();
        assert_eq!(summary.len(), 1);
        assert_eq!(summary[0].brier_score, 0.0);
        assert!(summary[0].expected_calibration_error.abs() < 1e-12);
        assert!(summary[0].log_loss < 1e-12);
        assert_eq!(summary[0].high_confidence_failure_rate, Some(0.0));
    }

    #[test]
    fn confidently_wrong_forecast_is_penalized() {
        let summary = summarize_calibration(&[forecast("a", 0.99, false, true)], 10, 0.8).unwrap();
        assert!(summary[0].brier_score > 0.9);
        assert!(summary[0].log_loss > 4.0);
        assert!(summary[0].expected_calibration_error > 0.9);
        assert_eq!(summary[0].high_confidence_failure_rate, Some(1.0));
    }

    #[test]
    fn invalid_run_counts_as_failure_in_calibration() {
        let summary = summarize_calibration(&[forecast("timeout", 0.9, false, false)], 10, 0.8).unwrap();
        assert_eq!(summary[0].total_trials, 1);
        assert_eq!(summary[0].valid_runs, 0);
        assert_eq!(summary[0].failures, 1);
        assert_eq!(summary[0].high_confidence_failure_rate, Some(1.0));
    }

    #[test]
    fn confidence_ranks_success_before_failure_on_risk_coverage_curve() {
        let rows = vec![
            forecast("low-success", 0.2, true, true),
            forecast("high-failure", 0.1, false, true),
            forecast("high-success", 0.9, true, true),
        ];
        let summary = summarize_calibration(&rows, 5, 0.8).unwrap();
        let curve = &summary[0].risk_coverage;
        assert_eq!(curve.len(), 3);
        assert_eq!(curve[0].risk, 0.0);
        assert_eq!(curve[0].retained_trials, 1);
        assert!(curve[2].risk > 0.0);
    }

    #[test]
    fn duplicate_forecasts_are_rejected() {
        let row = forecast("duplicate", 0.8, true, true);
        assert_eq!(
            summarize_calibration(&[row.clone(), row], 10, 0.8),
            Err(CalibrationError::DuplicateForecast)
        );
    }

    #[test]
    fn invalid_probabilities_and_success_on_invalid_run_are_rejected() {
        assert_eq!(
            summarize_calibration(&[forecast("nan", f64::NAN, false, true)], 10, 0.8),
            Err(CalibrationError::ProbabilityOutOfRange)
        );
        assert_eq!(
            summarize_calibration(&[forecast("invalid", 0.5, true, false)], 10, 0.8),
            Err(CalibrationError::SuccessOnInvalidRun)
        );
        assert_eq!(
            summarize_calibration(&[forecast("valid", 0.5, false, true)], 0, 0.8),
            Err(CalibrationError::InvalidBinCount)
        );
    }

    #[test]
    fn calibration_groups_are_deterministic_and_isolated_by_variant() {
        let rows = vec![
            forecast("a", 0.4, false, true),
            SuccessForecast {
                variant_id: "baseline".to_owned(),
                ..forecast("b", 0.6, true, true)
            },
            forecast("c", 0.7, true, true),
        ];
        let a = summarize_calibration(&rows, 5, 0.8).unwrap();
        let b = summarize_calibration(&rows, 5, 0.8).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 2);
    }
}
