// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! One-sided exact binomial upper bounds for explicitly justified IID rare-event trials.
//!
//! This utility is intentionally not applied automatically to NANO-013's unequal-probability
//! window sample. It is only estimable when the caller can justify one complete binary outcome
//! per independent, equal-exposure, identically distributed unit and equal-probability sampling.
//! These declarations are claims to audit, not facts this function can verify from the counts.

use serde::{Deserialize, Serialize};

const MAX_SUPPORTED_TRIALS: u64 = 10_000_000;
const BISECTION_ITERATIONS: usize = 80;
const TAIL_RELATIVE_TOLERANCE: f64 = 1.0e-15;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RareEventEvidence {
    /// Number of independent binary endpoints actually evaluated.
    pub observed_units: u64,
    /// Number of those endpoints with the preregistered rare event observed.
    pub observed_events: u64,
    /// Example: parent-system-at-fixed-horizon; all units must share its definition.
    pub unit_label: String,
    /// Desired one-sided confidence level, e.g. 0.95.
    pub confidence_level: f64,
    /// Must be justified by design/domain review; windows from one trajectory are not independent.
    pub independent_units_justified: bool,
    /// Must be justified; heterogeneous event probabilities cannot be pooled as one binomial model.
    pub common_event_probability_justified: bool,
    /// Every unit must represent the same preregistered exposure horizon / endpoint opportunity.
    pub common_exposure_horizon: bool,
    /// No unit may have a missing or failed reference outcome for this endpoint.
    pub all_outcomes_observed: bool,
    /// The observed units must have the sampling mechanism required by the IID binomial model.
    pub equal_probability_sampling: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum RareEventAssessment {
    BoundUnderDeclaredAssumptions {
        unit_label: String,
        observed_units: u64,
        observed_events: u64,
        confidence_level: f64,
        observed_rate: f64,
        upper_confidence_bound: f64,
        method: String,
        assumptions: Vec<String>,
        claim_ceiling: String,
    },
    NotEstimable {
        unit_label: String,
        observed_units: u64,
        observed_events: u64,
        confidence_level: f64,
        reasons: Vec<String>,
        claim_ceiling: String,
    },
}

/// Assess a one-sided exact Clopper-Pearson upper confidence bound.
///
/// An Ok(NotEstimable { .. }) is a valid result when design assumptions are not satisfied.
/// An Err is reserved for malformed counts or numerical inputs. A bound is not emitted merely
/// because a caller supplied counts: all design flags must be true and independently justified.
pub fn assess_rare_event_risk(
    evidence: &RareEventEvidence,
) -> Result<RareEventAssessment, String> {
    if evidence.observed_units == 0 {
        return Err("observed_units must be greater than zero".to_owned());
    }
    if evidence.observed_units > MAX_SUPPORTED_TRIALS {
        return Err(format!(
            "observed_units exceeds the supported maximum of {MAX_SUPPORTED_TRIALS}"
        ));
    }
    if evidence.observed_events > evidence.observed_units {
        return Err("observed_events cannot exceed observed_units".to_owned());
    }
    if evidence.unit_label.trim().is_empty() {
        return Err("unit_label must not be empty".to_owned());
    }
    if !evidence.confidence_level.is_finite()
        || evidence.confidence_level <= 0.0
        || evidence.confidence_level >= 1.0
    {
        return Err("confidence_level must be finite and strictly between 0 and 1".to_owned());
    }

    let mut reasons = Vec::new();
    if !evidence.independent_units_justified {
        reasons.push(
            "independence of the declared units has not been justified; clustered windows cannot be counted as independent trials".to_owned(),
        );
    }
    if !evidence.common_event_probability_justified {
        reasons.push(
            "a common event probability across the units has not been justified; heterogeneous parent systems must not be pooled into one IID binomial model".to_owned(),
        );
    }
    if !evidence.common_exposure_horizon {
        reasons.push(
            "units do not have a common preregistered exposure horizon or endpoint opportunity".to_owned(),
        );
    }
    if !evidence.all_outcomes_observed {
        reasons.push(
            "one or more unit outcomes are missing, failed, or unavailable; missing outcomes cannot be treated as event-free".to_owned(),
        );
    }
    if !evidence.equal_probability_sampling {
        reasons.push(
            "sampling is unequal-probability or otherwise incompatible with the IID binomial model; use a design-appropriate estimator instead".to_owned(),
        );
    }

    if !reasons.is_empty() {
        return Ok(RareEventAssessment::NotEstimable {
            unit_label: evidence.unit_label.clone(),
            observed_units: evidence.observed_units,
            observed_events: evidence.observed_events,
            confidence_level: evidence.confidence_level,
            reasons,
            claim_ceiling: "No risk upper bound is estimable from these declarations. This is not evidence of zero risk.".to_owned(),
        });
    }

    let n = evidence.observed_units;
    let k = evidence.observed_events;
    let alpha = 1.0 - evidence.confidence_level;
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err("confidence_level is too close to 0 or 1 for stable f64 significance arithmetic".to_owned());
    }
    let upper = clopper_pearson_upper(k, n, alpha);
    let observed_rate = k as f64 / n as f64;

    Ok(RareEventAssessment::BoundUnderDeclaredAssumptions {
        unit_label: evidence.unit_label.clone(),
        observed_units: n,
        observed_events: k,
        confidence_level: evidence.confidence_level,
        observed_rate,
        upper_confidence_bound: upper,
        method: "one-sided exact Clopper-Pearson binomial upper confidence bound; numerical inversion uses bisection of the binomial CDF, with the exact closed form for zero events".to_owned(),
        assumptions: vec![
            "one binary endpoint per declared unit".to_owned(),
            "independent units".to_owned(),
            "common event probability across units".to_owned(),
            "common preregistered exposure horizon".to_owned(),
            "all endpoint outcomes observed".to_owned(),
            "equal-probability sampling compatible with the binomial model".to_owned(),
        ],
        claim_ceiling: "Conditional on independently justified IID binomial assumptions only. This does not bound transfer risk to unseen material families, does not certify per-step risk from a fixed-horizon endpoint, and does not replace worst-group or design-based analysis.".to_owned(),
    })
}

fn clopper_pearson_upper(events: u64, units: u64, alpha: f64) -> f64 {
    if events == 0 {
        // P(X = 0 | p_U) = (1 - p_U)^n = alpha.
        // -expm1 avoids catastrophic cancellation when the upper bound is small.
        return -((alpha.ln()) / units as f64).exp_m1();
    }
    if events == units {
        return 1.0;
    }

    let observed_rate = events as f64 / units as f64;
    let mut low = observed_rate;
    let mut high = 1.0;

    for _ in 0..BISECTION_ITERATIONS {
        let midpoint = low + (high - low) * 0.5;
        if midpoint <= observed_rate {
            low = midpoint;
            continue;
        }
        if binomial_cdf(events, units, midpoint) > alpha {
            low = midpoint;
        } else {
            high = midpoint;
        }
    }
    high
}

/// This is only called at p > events / units, where the terms from k down to zero
/// decrease monotonically. Starting at P(X=k) and recurring downward avoids summing
/// tiny terms from j=0 upward and is stable for the supported count range.
fn binomial_cdf(events: u64, units: u64, probability: f64) -> f64 {
    if events >= units {
        return 1.0;
    }
    if probability <= 0.0 {
        return 1.0;
    }
    if probability >= 1.0 {
        return 0.0;
    }

    let log_mass_at_k = log_binomial_coefficient(units, events)
        + events as f64 * probability.ln()
        + (units - events) as f64 * (-probability).ln_1p();
    let mut mass = log_mass_at_k.exp();
    let mut sum = mass;
    let mut j = events;

    while j > 0 {
        let ratio = (j as f64 / (units - j + 1) as f64)
            * ((1.0 - probability) / probability);
        mass *= ratio;
        sum += mass;
        if mass <= sum * TAIL_RELATIVE_TOLERANCE {
            break;
        }
        j -= 1;
    }
    sum.clamp(0.0, 1.0)
}

fn log_binomial_coefficient(units: u64, events: u64) -> f64 {
    let smaller = events.min(units - events);
    let mut sum = 0.0;
    for i in 1..=smaller {
        sum += ((units - smaller + i) as f64).ln() - (i as f64).ln();
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(events: u64, units: u64) -> RareEventEvidence {
        RareEventEvidence {
            observed_units: units,
            observed_events: events,
            unit_label: "independent-parent-at-fixed-horizon".to_owned(),
            confidence_level: 0.95,
            independent_units_justified: true,
            common_event_probability_justified: true,
            common_exposure_horizon: true,
            all_outcomes_observed: true,
            equal_probability_sampling: true,
        }
    }

    fn upper(result: RareEventAssessment) -> f64 {
        match result {
            RareEventAssessment::BoundUnderDeclaredAssumptions {
                upper_confidence_bound,
                ..
            } => upper_confidence_bound,
            other => panic!("expected conditional bound, got {other:?}"),
        }
    }

    #[test]
    fn zero_events_never_imply_zero_risk() {
        let result = assess_rare_event_risk(&evidence(0, 10)).unwrap();
        assert!((upper(result) - 0.258_865_550_893_052_2).abs() < 1.0e-12);
    }

    #[test]
    fn zero_events_in_299_independent_units_bound_risk_below_one_percent() {
        let result = assess_rare_event_risk(&evidence(0, 299)).unwrap();
        let bound = upper(result);
        assert!(bound > 0.0);
        assert!(bound < 0.01);
        assert!((bound - 0.009_969_146_792_899_269).abs() < 1.0e-12);
    }

    #[test]
    fn positive_event_count_uses_exact_binomial_cdf_inversion() {
        let result = assess_rare_event_risk(&evidence(1, 100)).unwrap();
        assert!((upper(result) - 0.046_559_811_453_539).abs() < 1.0e-10);
    }

    #[test]
    fn upper_bound_increases_with_event_count_and_confidence() {
        let one_event = upper(assess_rare_event_risk(&evidence(1, 100)).unwrap());
        let five_events = upper(assess_rare_event_risk(&evidence(5, 100)).unwrap());
        assert!(five_events > one_event);

        let mut higher_confidence = evidence(1, 100);
        higher_confidence.confidence_level = 0.99;
        let stricter = upper(assess_rare_event_risk(&higher_confidence).unwrap());
        assert!(stricter > one_event);
    }

    #[test]
    fn all_units_with_event_produces_upper_bound_one() {
        assert_eq!(upper(assess_rare_event_risk(&evidence(20, 20)).unwrap()), 1.0);
    }

    #[test]
    fn incomplete_outcomes_make_a_bound_not_estimable() {
        let mut input = evidence(0, 299);
        input.all_outcomes_observed = false;
        let report = assess_rare_event_risk(&input).unwrap();
        match report {
            RareEventAssessment::NotEstimable { reasons, .. } => {
                assert!(reasons.iter().any(|reason| reason.contains("missing")));
            }
            other => panic!("expected not-estimable, got {other:?}"),
        }
    }

    #[test]
    fn clustered_windows_and_unequal_probability_samples_are_not_binomial_trials() {
        let mut clustered = evidence(0, 299);
        clustered.independent_units_justified = false;
        let report = assess_rare_event_risk(&clustered).unwrap();
        assert!(matches!(report, RareEventAssessment::NotEstimable { .. }));

        let mut unequal = evidence(0, 299);
        unequal.equal_probability_sampling = false;
        let report = assess_rare_event_risk(&unequal).unwrap();
        assert!(matches!(report, RareEventAssessment::NotEstimable { .. }));
    }

    #[test]
    fn malformed_counts_and_confidence_fail_closed() {
        assert!(assess_rare_event_risk(&evidence(2, 1)).is_err());
        assert!(assess_rare_event_risk(&evidence(0, 0)).is_err());

        let mut invalid_confidence = evidence(0, 10);
        invalid_confidence.confidence_level = 1.0;
        assert!(assess_rare_event_risk(&invalid_confidence).is_err());
    }
}
