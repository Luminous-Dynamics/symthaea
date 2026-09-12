// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Small statistical assurance primitives for zero-event validation evidence.
//!
//! These functions quantify what can be inferred after observing zero events.
//! They do not establish independence, stationarity, representativeness, or
//! deployment equivalence; those assumptions must be justified separately.

#![deny(unsafe_code)]

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BernoulliZeroEventBound {
    pub independent_exposures: u64,
    pub confidence: f64,
    pub upper_event_probability: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoissonZeroEventBound {
    pub exposure: f64,
    pub confidence: f64,
    pub upper_event_rate_per_exposure_unit: f64,
}

fn valid_confidence(confidence: f64) -> bool {
    confidence.is_finite() && confidence > 0.0 && confidence < 1.0
}

/// Exact one-sided upper confidence bound for a Bernoulli event probability after
/// zero observed events in `independent_exposures` independent trials.
///
/// `p_upper = 1 - (1 - confidence)^(1/n)`.
pub fn bernoulli_zero_event_upper_bound(
    independent_exposures: u64,
    confidence: f64,
) -> Option<BernoulliZeroEventBound> {
    if independent_exposures == 0 || !valid_confidence(confidence) {
        return None;
    }
    let alpha = 1.0 - confidence;
    let exponent = alpha.ln() / independent_exposures as f64;
    // Numerically stable form of 1 - exp(exponent) for small magnitudes.
    let upper = -exponent.exp_m1();
    Some(BernoulliZeroEventBound {
        independent_exposures,
        confidence,
        upper_event_probability: upper.clamp(0.0, 1.0),
    })
}

/// Minimum number of independent zero-event Bernoulli exposures required for the
/// one-sided upper bound to be at or below `target_upper_probability`.
pub fn required_zero_event_bernoulli_exposures(
    target_upper_probability: f64,
    confidence: f64,
) -> Option<u64> {
    if !valid_confidence(confidence)
        || !target_upper_probability.is_finite()
        || target_upper_probability <= 0.0
        || target_upper_probability >= 1.0
    {
        return None;
    }
    let alpha = 1.0 - confidence;
    let denominator = (-target_upper_probability).ln_1p(); // ln(1-p), negative
    let required = (alpha.ln() / denominator).ceil();
    if !required.is_finite() || required <= 0.0 || required > u64::MAX as f64 {
        return None;
    }
    Some(required as u64)
}

/// Exact one-sided Poisson event-rate upper bound after zero observed events over
/// positive exposure `T`.
///
/// `lambda_upper = -ln(1 - confidence) / T`.
pub fn poisson_zero_event_upper_rate(
    exposure: f64,
    confidence: f64,
) -> Option<PoissonZeroEventBound> {
    if !valid_confidence(confidence) || !exposure.is_finite() || exposure <= 0.0 {
        return None;
    }
    let alpha = 1.0 - confidence;
    Some(PoissonZeroEventBound {
        exposure,
        confidence,
        upper_event_rate_per_exposure_unit: -alpha.ln() / exposure,
    })
}

/// Minimum positive exposure required for a zero-event Poisson campaign to place
/// the one-sided upper event rate at or below `target_upper_rate`.
pub fn required_zero_event_poisson_exposure(
    target_upper_rate: f64,
    confidence: f64,
) -> Option<f64> {
    if !valid_confidence(confidence)
        || !target_upper_rate.is_finite()
        || target_upper_rate <= 0.0
    {
        return None;
    }
    let alpha = 1.0 - confidence;
    Some(-alpha.ln() / target_upper_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_95_percent_zero_event_bound_matches_rule_of_three_scale() {
        let bound = bernoulli_zero_event_upper_bound(100, 0.95).unwrap();
        assert!((bound.upper_event_probability - 0.0295130496).abs() < 1e-9);
        // The familiar 3/n approximation at 95% is close but deliberately not used.
        assert!((bound.upper_event_probability - 0.03).abs() < 0.001);
    }

    #[test]
    fn required_exposures_round_up_and_meet_target() {
        let n = required_zero_event_bernoulli_exposures(0.001, 0.95).unwrap();
        let bound = bernoulli_zero_event_upper_bound(n, 0.95).unwrap();
        assert!(bound.upper_event_probability <= 0.001);
        if n > 1 {
            let previous = bernoulli_zero_event_upper_bound(n - 1, 0.95).unwrap();
            assert!(previous.upper_event_probability > 0.001);
        }
    }

    #[test]
    fn poisson_zero_event_bound_is_exact() {
        let bound = poisson_zero_event_upper_rate(100.0, 0.95).unwrap();
        assert!((bound.upper_event_rate_per_exposure_unit - 0.0299573227).abs() < 1e-9);
    }

    #[test]
    fn required_poisson_exposure_meets_target_rate() {
        let exposure = required_zero_event_poisson_exposure(1e-4, 0.99).unwrap();
        let bound = poisson_zero_event_upper_rate(exposure, 0.99).unwrap();
        assert!((bound.upper_event_rate_per_exposure_unit - 1e-4).abs() < 1e-12);
    }

    #[test]
    fn invalid_inputs_fail_closed() {
        assert!(bernoulli_zero_event_upper_bound(0, 0.95).is_none());
        assert!(bernoulli_zero_event_upper_bound(10, 1.0).is_none());
        assert!(required_zero_event_bernoulli_exposures(0.0, 0.95).is_none());
        assert!(poisson_zero_event_upper_rate(0.0, 0.95).is_none());
        assert!(required_zero_event_poisson_exposure(-1.0, 0.95).is_none());
    }
}
