//! Anytime-valid measurement-channel bounds for SCI-ANYTIME-IMPERFECT-CERT-001.
//!
//! This std-only kernel implements summable-alpha Hoeffding bounds for four
//! Bernoulli streams: TPR lower, FPR lower, FPR upper, and test-positive
//! prevalence upper. It does not authenticate references or execution, fit
//! a population model, or prove that the required iid stream assumptions hold.

#![forbid(unsafe_code)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CertificationStatus {
    Certified,
    Inconclusive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelCounts {
    pub true_positive: u64,
    pub false_negative: u64,
    pub true_negative: u64,
    pub false_positive: u64,
    pub test_positive: u64,
    pub test_total: u64,
}

impl ChannelCounts {
    pub fn positive_calibration_total(self) -> Result<u64, String> {
        self.true_positive
            .checked_add(self.false_negative)
            .ok_or_else(|| "positive calibration count overflow".to_owned())
    }

    pub fn negative_calibration_total(self) -> Result<u64, String> {
        self.true_negative
            .checked_add(self.false_positive)
            .ok_or_else(|| "negative calibration count overflow".to_owned())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChannelBounds {
    counts: ChannelCounts,
    total_one_sided_streams: u64,
    tpr_lower: f64,
    fpr_lower: f64,
    fpr_upper: f64,
    test_positive_upper: f64,
    channel_separation_established: bool,
    /// False when the simultaneous test-positive and FPR bounds are
    /// incompatible with q = f + (t - f)p for any p in [0, 1].
    measurement_channel_consistent: bool,
    /// None means measurement-channel inversion is not justified.
    failure_prevalence_upper: Option<f64>,
}

impl ChannelBounds {
    pub fn counts(&self) -> ChannelCounts {
        self.counts
    }

    pub fn total_one_sided_streams(&self) -> u64 {
        self.total_one_sided_streams
    }

    pub fn tpr_lower(&self) -> f64 {
        self.tpr_lower
    }

    pub fn fpr_lower(&self) -> f64 {
        self.fpr_lower
    }

    pub fn fpr_upper(&self) -> f64 {
        self.fpr_upper
    }

    pub fn test_positive_upper(&self) -> f64 {
        self.test_positive_upper
    }

    pub fn channel_separation_established(&self) -> bool {
        self.channel_separation_established
    }

    pub fn measurement_channel_consistent(&self) -> bool {
        self.measurement_channel_consistent
    }

    pub fn failure_prevalence_upper(&self) -> Option<f64> {
        self.failure_prevalence_upper
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CertificationDecision {
    pub status: CertificationStatus,
    pub failure_prevalence_upper: Option<f64>,
    pub target_failure_rate: f64,
    pub reason: String,
}

fn validate_alpha(alpha: f64) -> Result<(), String> {
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err("alpha must be finite and strictly between zero and one".to_owned());
    }
    Ok(())
}

/// Allocate a one-sided error budget to one stream at its nth observation.
///
/// Across S predeclared one-sided streams and all n >= 1, the sum of
/// alpha / (S * n * (n + 1)) is at most alpha.
pub fn alpha_for_look(
    alpha: f64,
    total_streams: u64,
    observations: u64,
) -> Result<f64, String> {
    validate_alpha(alpha)?;
    if total_streams == 0 {
        return Err("total_streams must be positive".to_owned());
    }
    if observations == 0 {
        return Err("observations must be positive".to_owned());
    }

    let n = observations as f64;
    let denominator = total_streams as f64 * n * (n + 1.0);
    let allocated = alpha / denominator;
    if !denominator.is_finite() || !allocated.is_finite() || allocated <= 0.0 || allocated >= 1.0 {
        return Err("alpha allocation is outside the representable open interval (0, 1)".to_owned());
    }
    Ok(allocated)
}

fn validate_binomial(successes: u64, observations: u64, delta: f64) -> Result<(), String> {
    if observations == 0 {
        return Err("Bernoulli stream must contain at least one observation".to_owned());
    }
    if successes > observations {
        return Err("successes must not exceed observations".to_owned());
    }
    if !delta.is_finite() || delta <= 0.0 || delta >= 1.0 {
        return Err("one-sided delta must be finite and strictly between zero and one".to_owned());
    }
    Ok(())
}

fn hoeffding_radius(observations: u64, delta: f64) -> Result<f64, String> {
    if observations == 0 {
        return Err("Bernoulli stream must contain at least one observation".to_owned());
    }
    if !delta.is_finite() || delta <= 0.0 || delta >= 1.0 {
        return Err("one-sided delta must be finite and strictly between zero and one".to_owned());
    }
    let radius = ((1.0 / delta).ln() / (2.0 * observations as f64)).sqrt();
    if !radius.is_finite() {
        return Err("Hoeffding radius is not finite".to_owned());
    }
    Ok(radius)
}

/// One-sided time-uniform lower confidence bound for a Bernoulli mean.
pub fn hoeffding_lower(
    successes: u64,
    observations: u64,
    delta: f64,
) -> Result<f64, String> {
    validate_binomial(successes, observations, delta)?;
    let mean = successes as f64 / observations as f64;
    Ok((mean - hoeffding_radius(observations, delta)?).clamp(0.0, 1.0))
}

/// One-sided time-uniform upper confidence bound for a Bernoulli mean.
pub fn hoeffding_upper(
    successes: u64,
    observations: u64,
    delta: f64,
) -> Result<f64, String> {
    validate_binomial(successes, observations, delta)?;
    let mean = successes as f64 / observations as f64;
    Ok((mean + hoeffding_radius(observations, delta)?).clamp(0.0, 1.0))
}

/// Compute the four simultaneous one-sided bounds for one arm × family.
///
/// total_one_sided_streams must include all predeclared streams across every
/// judge-policy arm and task family. For one arm and family it is at least
/// four. Calibration abstentions must already be counted as positive actions
/// in TP/FP, and test abstentions as test-positive outcomes.
pub fn measurement_channel_bounds(
    counts: ChannelCounts,
    alpha: f64,
    total_one_sided_streams: u64,
) -> Result<ChannelBounds, String> {
    validate_alpha(alpha)?;
    if total_one_sided_streams < 4 {
        return Err("at least four one-sided streams are required per arm and family".to_owned());
    }

    let positive_n = counts.positive_calibration_total()?;
    let negative_n = counts.negative_calibration_total()?;
    let test_n = counts.test_total;
    if positive_n == 0 {
        return Err("positive-reference calibration stream is empty".to_owned());
    }
    if negative_n == 0 {
        return Err("negative-reference calibration stream is empty".to_owned());
    }
    if test_n == 0 {
        return Err("test-prevalence stream is empty".to_owned());
    }
    if counts.test_positive > test_n {
        return Err("test_positive must not exceed test_total".to_owned());
    }

    // Each line corresponds to a distinct one-sided confidence-sequence stream.
    let tpr_delta = alpha_for_look(alpha, total_one_sided_streams, positive_n)?;
    let fpr_lower_delta = alpha_for_look(alpha, total_one_sided_streams, negative_n)?;
    let fpr_upper_delta = alpha_for_look(alpha, total_one_sided_streams, negative_n)?;
    let test_delta = alpha_for_look(alpha, total_one_sided_streams, test_n)?;

    let tpr_lower = hoeffding_lower(counts.true_positive, positive_n, tpr_delta)?;
    let fpr_lower = hoeffding_lower(counts.false_positive, negative_n, fpr_lower_delta)?;
    let fpr_upper = hoeffding_upper(counts.false_positive, negative_n, fpr_upper_delta)?;
    let test_positive_upper = hoeffding_upper(counts.test_positive, test_n, test_delta)?;

    // Inversion is permitted only if every channel in the simultaneous
    // confidence set has TPR > FPR.
    let channel_separation_established = tpr_lower > fpr_upper;
    let measurement_channel_consistent = test_positive_upper >= fpr_lower;
    let failure_prevalence_upper =
        if channel_separation_established && measurement_channel_consistent {
            let denominator = tpr_lower - fpr_lower;
            if denominator <= 0.0 {
                return Err("separation gate passed with a non-positive inversion denominator".to_owned());
            }
            Some(((test_positive_upper - fpr_lower) / denominator).clamp(0.0, 1.0))
        } else {
            None
        };

    Ok(ChannelBounds {
        counts,
        total_one_sided_streams,
        tpr_lower,
        fpr_lower,
        fpr_upper,
        test_positive_upper,
        channel_separation_established,
        measurement_channel_consistent,
        failure_prevalence_upper,
    })
}

/// Return an affirmative certificate only when all minimums and the risk
/// threshold are met. Every other valid statistical state is inconclusive.
pub fn decide_certification(
    bounds: &ChannelBounds,
    target_failure_rate: f64,
    minimum_positive_calibration: u64,
    minimum_negative_calibration: u64,
    minimum_test_observations: u64,
) -> Result<CertificationDecision, String> {
    if !target_failure_rate.is_finite()
        || target_failure_rate <= 0.0
        || target_failure_rate >= 1.0
    {
        return Err("target_failure_rate must be finite and strictly between zero and one".to_owned());
    }

    let positive_n = bounds.counts.positive_calibration_total()?;
    let negative_n = bounds.counts.negative_calibration_total()?;
    let test_n = bounds.counts.test_total;
    if positive_n < minimum_positive_calibration
        || negative_n < minimum_negative_calibration
        || test_n < minimum_test_observations
    {
        return Ok(CertificationDecision {
            status: CertificationStatus::Inconclusive,
            failure_prevalence_upper: bounds.failure_prevalence_upper,
            target_failure_rate,
            reason: "one or more predeclared minimum sample counts are unmet".to_owned(),
        });
    }

    if !bounds.channel_separation_established {
        return Ok(CertificationDecision {
            status: CertificationStatus::Inconclusive,
            failure_prevalence_upper: None,
            target_failure_rate,
            reason: "TPR lower bound does not exceed FPR upper bound".to_owned(),
        });
    }
    if !bounds.measurement_channel_consistent {
        return Ok(CertificationDecision {
            status: CertificationStatus::Inconclusive,
            failure_prevalence_upper: None,
            target_failure_rate,
            reason: "test-positive upper bound is below FPR lower bound; measurement channel is inconsistent".to_owned(),
        });
    }

    let prevalence_upper = bounds
        .failure_prevalence_upper
        .ok_or_else(|| "separated channel is missing its prevalence upper bound".to_owned())?;
    if prevalence_upper < target_failure_rate {
        Ok(CertificationDecision {
            status: CertificationStatus::Certified,
            failure_prevalence_upper: Some(prevalence_upper),
            target_failure_rate,
            reason: "all minimums are met and the simultaneous prevalence upper bound is below target".to_owned(),
        })
    } else {
        Ok(CertificationDecision {
            status: CertificationStatus::Inconclusive,
            failure_prevalence_upper: Some(prevalence_upper),
            target_failure_rate,
            reason: "failure prevalence upper bound does not fall strictly below target".to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn separated_counts() -> ChannelCounts {
        ChannelCounts {
            true_positive: 9_500,
            false_negative: 500,
            true_negative: 9_800,
            false_positive: 200,
            test_positive: 500,
            test_total: 10_000,
        }
    }

    #[test]
    fn alpha_spending_is_summable_and_bounded() {
        let alpha = 0.05;
        let streams = 12;
        let mut per_stream_sum = 0.0;
        for n in 1..=100_000 {
            per_stream_sum += alpha_for_look(alpha, streams, n).unwrap();
        }
        let family_sum = per_stream_sum * streams as f64;
        assert!(family_sum <= alpha);
        assert!(family_sum > alpha * (1.0 - 2.0 / 100_000.0));
    }

    #[test]
    fn clearly_separated_judge_gets_a_prevalence_upper_bound() {
        let bounds = measurement_channel_bounds(separated_counts(), 0.05, 4).unwrap();
        assert!(bounds.tpr_lower > bounds.fpr_upper);
        let prevalence_upper = bounds.failure_prevalence_upper.unwrap();
        assert!(prevalence_upper < 0.12, "{prevalence_upper}");
        let decision = decide_certification(&bounds, 0.12, 1_000, 1_000, 1_000).unwrap();
        assert_eq!(decision.status, CertificationStatus::Certified);
    }

    #[test]
    fn overlapping_tpr_and_fpr_intervals_remain_inconclusive() {
        let counts = ChannelCounts {
            true_positive: 8_000,
            false_negative: 2_000,
            true_negative: 2_200,
            false_positive: 7_800,
            test_positive: 7_000,
            test_total: 10_000,
        };
        let bounds = measurement_channel_bounds(counts, 0.05, 4).unwrap();
        // The lower-vs-lower check would pass, but the simultaneous intervals overlap.
        assert!(bounds.tpr_lower > bounds.fpr_lower);
        assert!(bounds.tpr_lower <= bounds.fpr_upper);
        assert!(!bounds.channel_separation_established);
        assert_eq!(bounds.failure_prevalence_upper, None);
        let decision = decide_certification(&bounds, 0.10, 1_000, 1_000, 1_000).unwrap();
        assert_eq!(decision.status, CertificationStatus::Inconclusive);
    }

    #[test]
    fn incompatible_test_prevalence_and_fpr_are_inconclusive_not_zero_risk() {
        let counts = ChannelCounts {
            true_positive: 9_900,
            false_negative: 100,
            true_negative: 8_500,
            false_positive: 1_500,
            test_positive: 0,
            test_total: 10_000,
        };
        let bounds = measurement_channel_bounds(counts, 0.05, 4).unwrap();
        assert!(bounds.channel_separation_established);
        assert!(!bounds.measurement_channel_consistent);
        assert_eq!(bounds.failure_prevalence_upper, None);
        let decision = decide_certification(&bounds, 0.12, 1, 1, 1).unwrap();
        assert_eq!(decision.status, CertificationStatus::Inconclusive);
        assert!(decision.reason.contains("inconsistent"));
    }

    #[test]
    fn count_overflow_is_rejected() {
        let counts = ChannelCounts {
            true_positive: u64::MAX,
            false_negative: 1,
            true_negative: 1,
            false_positive: 1,
            test_positive: 0,
            test_total: 1,
        };
        assert!(measurement_channel_bounds(counts, 0.05, 4)
            .unwrap_err()
            .contains("overflow"));
    }

    #[test]
    fn inverted_judge_is_never_certified() {
        let counts = ChannelCounts {
            true_positive: 1_000,
            false_negative: 9_000,
            true_negative: 2_000,
            false_positive: 8_000,
            test_positive: 5_000,
            test_total: 10_000,
        };
        let bounds = measurement_channel_bounds(counts, 0.05, 4).unwrap();
        assert!(!bounds.channel_separation_established);
        assert_eq!(bounds.failure_prevalence_upper, None);
    }

    #[test]
    fn insufficient_calibration_or_test_counts_are_inconclusive() {
        let bounds = measurement_channel_bounds(separated_counts(), 0.05, 4).unwrap();
        let decision = decide_certification(&bounds, 0.12, 20_000, 1, 1).unwrap();
        assert_eq!(decision.status, CertificationStatus::Inconclusive);
    }

    #[test]
    fn absent_class_stream_or_test_stream_is_rejected() {
        let mut counts = separated_counts();
        counts.false_negative = 0;
        counts.true_positive = 0;
        assert!(measurement_channel_bounds(counts, 0.05, 4)
            .unwrap_err()
            .contains("positive-reference"));

        let mut counts = separated_counts();
        counts.false_positive = 0;
        counts.true_negative = 0;
        assert!(measurement_channel_bounds(counts, 0.05, 4)
            .unwrap_err()
            .contains("negative-reference"));

        let mut counts = separated_counts();
        counts.test_total = 0;
        assert!(measurement_channel_bounds(counts, 0.05, 4)
            .unwrap_err()
            .contains("test-prevalence"));
    }

    #[test]
    fn invalid_alpha_stream_count_and_binomial_counts_are_rejected() {
        assert!(alpha_for_look(0.0, 4, 1).is_err());
        assert!(alpha_for_look(0.05, 0, 1).is_err());
        assert!(alpha_for_look(0.05, 4, 0).is_err());
        assert!(hoeffding_lower(3, 2, 0.05).is_err());
        assert!(hoeffding_upper(1, 2, 1.0).is_err());
        assert!(measurement_channel_bounds(separated_counts(), 0.05, 3).is_err());
    }

    #[test]
    fn strict_target_and_minimum_counts_are_enforced() {
        let bounds = measurement_channel_bounds(separated_counts(), 0.05, 4).unwrap();
        assert_eq!(
            decide_certification(&bounds, 0.12, 1, 1, 1).unwrap().status,
            CertificationStatus::Certified
        );
        let exact_upper = bounds.failure_prevalence_upper.unwrap();
        assert_eq!(
            decide_certification(&bounds, exact_upper, 1, 1, 1).unwrap().status,
            CertificationStatus::Inconclusive
        );
    }

    #[test]
    fn deterministic_sequential_null_simulation_checks_anytime_coverage() {
        fn next_unit(seed: &mut u64) -> f64 {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            ((*seed >> 11) as f64) / ((1_u64 << 53) as f64)
        }

        const RUNS: u64 = 2_000;
        const MAX_N: u64 = 64;
        let alpha = 0.10;
        let true_tpr = 0.90;
        let true_fpr = 0.05;
        let true_test_positive = 0.07;
        let mut any_sequence_misses = 0_u64;

        for run in 0..RUNS {
            let mut seed = 0x9e37_79b9_7f4a_7c15_u64 ^ run.wrapping_add(1);
            let mut tp = 0_u64;
            let mut fn_count = 0_u64;
            let mut tn = 0_u64;
            let mut fp = 0_u64;
            let mut test_positive = 0_u64;
            let mut missed = false;

            for _ in 1..=MAX_N {
                if next_unit(&mut seed) < true_tpr {
                    tp += 1;
                } else {
                    fn_count += 1;
                }
                if next_unit(&mut seed) < true_fpr {
                    fp += 1;
                } else {
                    tn += 1;
                }
                if next_unit(&mut seed) < true_test_positive {
                    test_positive += 1;
                }

                let positive_n = tp + fn_count;
                let counts = ChannelCounts {
                    true_positive: tp,
                    false_negative: fn_count,
                    true_negative: tn,
                    false_positive: fp,
                    test_positive,
                    test_total: positive_n,
                };
                let bounds = measurement_channel_bounds(counts, alpha, 4).unwrap();
                if true_tpr < bounds.tpr_lower
                    || true_fpr < bounds.fpr_lower
                    || true_fpr > bounds.fpr_upper
                    || true_test_positive > bounds.test_positive_upper
                {
                    missed = true;
                }
            }
            if missed {
                any_sequence_misses += 1;
            }
        }

        // Loose Monte Carlo guard around nominal 10%; this is a smoke test, not proof.
        let observed_rate = any_sequence_misses as f64 / RUNS as f64;
        assert!(observed_rate < 0.14, "observed miss rate {observed_rate}");
    }
}
