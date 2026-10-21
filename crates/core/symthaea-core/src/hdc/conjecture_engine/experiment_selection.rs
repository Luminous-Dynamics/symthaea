// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! FEP-flavored active experiment selection: given several live candidate
//! hypotheses that all still fit the data seen so far, choose which
//! experiment to run next by picking the one that maximally *discriminates*
//! between them.
//!
//! This is the second capability in the agreed longer sequence for the
//! Ramanujan Protocol (HDC → constrained physical reasoning → **this** →
//! CfC → IIT). The idea was scoped back when M2 (flux discovery) was still
//! open: "FEP for active experiment selection (choosing initial conditions
//! that maximally discriminate between surviving candidate hypotheses --
//! needs multiple live candidates to discriminate between, which the search
//! doesn't have yet)". M2's factorized search (`flux_discovery.rs`) and the
//! new `typed_generation` module (`symthaea-physics-bridge`) both now
//! routinely produce multiple structurally-diverse surviving candidates, so
//! this capability is buildable.
//!
//! ## Why not reuse `symthaea-fep::ExpectedFreeEnergyComputer` directly
//!
//! `symthaea-fep` already has a real, working expected-free-energy
//! computation (`free_energy.rs`), and its `epistemic_value` there is
//! exactly the right *concept* (uncertainty reduction), but not directly
//! reusable machinery: it computes the entropy of one continuous
//! `HiddenState` before and after a predicted transition under a single
//! `GenerativeModel` -- built for the cognitive loop's own perception-action
//! domain. What this module needs is different: not "how much does one
//! model's own uncertainty shrink," but "how much do *several independent
//! discrete symbolic hypotheses' predictions disagree* for a given
//! experiment" -- the classic optimal-experimental-design / query-by-
//! committee framing, which happens to be the multi-hypothesis analogue of
//! the same FEP epistemic-value idea. `epistemic_value` below is a new,
//! small, honestly-scoped implementation of that analogue, not a
//! reimplementation of `symthaea-fep`'s machinery.
//!
//! ## Design
//!
//! Deliberately generic over the hypothesis and experiment representations
//! (via a `predict` closure) rather than hardcoded to `Expr` -- this makes
//! it reusable for whatever the next discovery task looks like, not just
//! the closed M2 wave-chain problem. [`epistemic_value`] scores a single
//! candidate experiment; [`select_most_informative_experiment`] picks the
//! best of a candidate pool. `predict` returning `None` for a hypothesis
//! (e.g. the expression is undefined/non-finite at that experiment) means
//! that hypothesis contributes no signal for this experiment, not that the
//! experiment is uninformative -- it's simply excluded from that
//! experiment's variance computation.

/// A typed result for scoring a candidate experiment against live hypotheses.
///
/// Keeping unsupported and unrepresentable scores distinct prevents callers
/// from mistaking an invalid experiment for a legitimate zero-disagreement one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EpistemicAssessment {
    /// Finite, representable population variance over at least two predictions.
    Score {
        value: f64,
        valid_predictions: usize,
    },
    /// Fewer than two hypotheses produced finite predictions for this experiment.
    InsufficientValidPredictions {
        valid_predictions: usize,
    },
    /// Predictions were finite, but the variance was not representable as f64.
    UnrepresentableScore {
        valid_predictions: usize,
    },
}

/// Population variance (not sample variance). Scaling before reduction avoids
/// overflowing the mean when several large finite predictions share a sign.
fn variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let scale = values
        .iter()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
    if scale == 0.0 {
        return 0.0;
    }

    let normalized: Vec<f64> = values.iter().map(|value| value / scale).collect();
    let mean = normalized.iter().sum::<f64>() / normalized.len() as f64;
    let normalized_variance = normalized
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / normalized.len() as f64;

    // Multiplication order matters: normalized_variance * scale * scale can
    // retain a representable result when scale*scale alone would overflow.
    normalized_variance * scale * scale
}

/// Score a candidate experiment while preserving validity information.
///
/// Non-finite predictions are excluded from the comparison. Fewer than two
/// finite predictions return `InsufficientValidPredictions`; an overflowed
/// variance returns `UnrepresentableScore` instead of silently becoming a
/// high score or contaminating the candidate ranking.
pub fn assess_epistemic_value<H, E>(
    experiment: &E,
    hypotheses: &[H],
    predict: impl Fn(&H, &E) -> Option<f64>,
) -> EpistemicAssessment {
    let predictions: Vec<f64> = hypotheses
        .iter()
        .filter_map(|hypothesis| predict(hypothesis, experiment))
        .filter(|prediction| prediction.is_finite())
        .collect();

    if predictions.len() < 2 {
        return EpistemicAssessment::InsufficientValidPredictions {
            valid_predictions: predictions.len(),
        };
    }

    let valid_predictions = predictions.len();
    let value = variance(&predictions);
    if !value.is_finite() || value < 0.0 {
        EpistemicAssessment::UnrepresentableScore { valid_predictions }
    } else {
        EpistemicAssessment::Score {
            value,
            valid_predictions,
        }
    }
}

/// Compatibility score for one candidate experiment.
///
/// High population variance means candidate hypotheses disagree; zero means
/// either that the predictions agree, fewer than two finite predictions remain,
/// or the score is unrepresentable. Callers that need to distinguish those cases
/// should use [`assess_epistemic_value`] instead.
pub fn epistemic_value<H, E>(
    experiment: &E,
    hypotheses: &[H],
    predict: impl Fn(&H, &E) -> Option<f64>,
) -> f64 {
    match assess_epistemic_value(experiment, hypotheses, predict) {
        EpistemicAssessment::Score { value, .. } => value,
        EpistemicAssessment::InsufficientValidPredictions { .. }
        | EpistemicAssessment::UnrepresentableScore { .. } => 0.0,
    }
}

/// Select the highest-scoring candidate with finite, representable disagreement
/// across at least two hypotheses. Experiments with no valid comparison or zero
/// disagreement are omitted; returns `None` if no informative candidate remains.
/// Ties keep the first candidate, preserving deterministic input-order semantics.
pub fn select_most_informative_experiment<'a, H, E>(
    candidates: &'a [E],
    hypotheses: &[H],
    predict: impl Fn(&H, &E) -> Option<f64> + Copy,
) -> Option<(&'a E, f64)> {
    let mut best: Option<(&'a E, f64)> = None;

    for candidate in candidates {
        let EpistemicAssessment::Score { value, .. } =
            assess_epistemic_value(candidate, hypotheses, predict)
        else {
            continue;
        };

        if value <= 0.0 {
            continue;
        }

        if best.map_or(true, |(_, best_value)| value > best_value) {
            best = Some((candidate, value));
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::conjecture_engine::Expr;

    #[test]
    fn variance_of_identical_values_is_zero() {
        assert_eq!(variance(&[1.0, 1.0, 1.0]), 0.0);
    }

    #[test]
    fn variance_matches_hand_computation() {
        // [1, 2, 3]: mean=2, population variance = ((1)^2+(0)^2+(1)^2)/3 = 2/3
        let v = variance(&[1.0, 2.0, 3.0]);
        assert!((v - (2.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn epistemic_value_is_zero_when_hypotheses_agree() {
        // Four synthetic "laws" that all happen to predict 0 at x=0.
        let hypotheses: Vec<fn(f64) -> f64> = vec![|x| x, |x| x * x, |x| 2.0 * x, |x: f64| x.sin()];
        let predict = |h: &fn(f64) -> f64, x: &f64| Some(h(*x));
        let v = epistemic_value(&0.0, &hypotheses, predict);
        assert!(v < 1e-9, "expected near-zero disagreement at x=0, got {v}");
    }

    #[test]
    fn epistemic_value_is_high_when_hypotheses_diverge() {
        let hypotheses: Vec<fn(f64) -> f64> = vec![|x| x, |x| x * x, |x| 2.0 * x, |x: f64| x.sin()];
        let predict = |h: &fn(f64) -> f64, x: &f64| Some(h(*x));
        let at_zero = epistemic_value(&0.0, &hypotheses, predict);
        let at_three = epistemic_value(&3.0, &hypotheses, predict);
        assert!(
            at_three > at_zero,
            "x=3 (predictions 3, 9, 6, sin(3)≈0.14) should disagree far more than x=0 \
             (all predict 0), got at_zero={at_zero}, at_three={at_three}"
        );
    }

    #[test]
    fn selector_avoids_the_degenerate_all_agree_point() {
        let hypotheses: Vec<fn(f64) -> f64> = vec![|x| x, |x| x * x, |x| 2.0 * x, |x: f64| x.sin()];
        let predict = |h: &fn(f64) -> f64, x: &f64| Some(h(*x));
        // Candidate pool deliberately includes the degenerate x=0 point
        // alongside genuinely discriminating ones.
        let candidates = [0.0, -3.0, -1.0, 1.0, 3.0];
        let (chosen, value) = select_most_informative_experiment(&candidates, &hypotheses, predict)
            .expect("non-empty candidate pool");
        assert_ne!(
            *chosen, 0.0,
            "selector should not pick the point where every hypothesis agrees"
        );
        assert!(value > 0.0);
    }

    #[test]
    fn selector_returns_none_on_empty_candidate_pool() {
        let hypotheses: Vec<fn(f64) -> f64> = vec![|x| x];
        let predict = |h: &fn(f64) -> f64, x: &f64| Some(h(*x));
        let candidates: [f64; 0] = [];
        assert!(select_most_informative_experiment(&candidates, &hypotheses, predict).is_none());
    }

    #[test]
    fn none_predictions_are_excluded_not_treated_as_zero_disagreement() {
        // A hypothesis that can't produce a prediction for some experiments
        // (e.g. undefined there) should simply not count toward that
        // experiment's variance -- not silently contribute a 0.0 that could
        // suppress a real signal from the hypotheses that DO predict there.
        let hypotheses = vec!["always_valid", "invalid_at_zero"];
        let predict = |h: &&str, x: &f64| match *h {
            "always_valid" => Some(*x),
            "invalid_at_zero" if *x == 0.0 => None,
            "invalid_at_zero" => Some(*x * 10.0),
            _ => None,
        };
        // At x=0.0, only one hypothesis contributes a prediction -> variance 0.0
        // (not because they "agree", but because there's nothing to compare).
        let v = epistemic_value(&0.0, &hypotheses, predict);
        assert_eq!(v, 0.0);
        // At x=1.0, both contribute (1.0 vs 10.0) -> real disagreement.
        let v2 = epistemic_value(&1.0, &hypotheses, predict);
        assert!(v2 > 0.0);
    }

    /// Integration check: the same mechanism applied to actual
    /// `conjecture_engine::Expr` candidates (the Ramanujan Protocol's real
    /// hypothesis representation), not just closures -- confirms this is
    /// directly usable against the discovery engine's own candidate type,
    /// without depending on any specific domain (wave-chain or otherwise).
    #[test]
    fn works_directly_against_expr_hypotheses() {
        use crate::hdc::conjecture_engine::BinOp;
        let var = |n: &str| Expr::Var(n.to_string());
        // Two candidate "laws": y = x, and y = x^2.
        let h1 = var("x");
        let h2 = Expr::BinOp(BinOp::Pow, Box::new(var("x")), Box::new(Expr::Const(2.0)));
        let hypotheses = vec![h1, h2];
        let predict = |h: &Expr, x: &f64| {
            let v = h.eval(&[("x", *x)]);
            v.is_finite().then_some(v)
        };
        // x=1: both predict 1 (agree). x=3: predict 3 vs 9 (disagree a lot).
        let candidates = [1.0, 3.0];
        let (chosen, _) = select_most_informative_experiment(&candidates, &hypotheses, predict)
            .expect("non-empty pool");
        assert_eq!(
            *chosen, 3.0,
            "should prefer the point where x vs x^2 diverge most"
        );
    }

    #[test]
    fn selector_skips_non_finite_and_under_supported_candidates() {
        let hypotheses = [0_i32, 1_i32, 2_i32];
        let predict = |hypothesis: &i32, experiment: &f64| {
            if *experiment == 0.0 {
                if *hypothesis == 0 {
                    Some(1.0)
                } else {
                    Some(f64::NAN)
                }
            } else if *experiment == 1.0 {
                match *hypothesis {
                    0 => Some(0.0),
                    1 => Some(10.0),
                    _ => Some(f64::INFINITY),
                }
            } else {
                None
            }
        };

        assert_eq!(
            assess_epistemic_value(&0.0, &hypotheses, predict),
            EpistemicAssessment::InsufficientValidPredictions {
                valid_predictions: 1
            }
        );

        let candidates = [0.0, 1.0];
        let (chosen, score) = select_most_informative_experiment(
            &candidates,
            &hypotheses,
            predict,
        )
        .expect("the second candidate has two finite, disagreeing predictions");
        assert_eq!(*chosen, 1.0);
        assert!((score - 25.0).abs() < 1e-12);
        assert!(score.is_finite());
    }

    #[test]
    fn assessment_distinguishes_unrepresentable_variance() {
        let extreme = [-1.0e308_f64, 1.0e308_f64];
        assert_eq!(
            assess_epistemic_value(&(), &extreme, |value, _| Some(*value)),
            EpistemicAssessment::UnrepresentableScore {
                valid_predictions: 2
            }
        );

        // The scaled calculation still returns zero for equal large values,
        // instead of overflowing while summing the mean.
        let equal_large = [1.0e308_f64, 1.0e308_f64];
        assert_eq!(
            assess_epistemic_value(&(), &equal_large, |value, _| Some(*value)),
            EpistemicAssessment::Score {
                value: 0.0,
                valid_predictions: 2
            }
        );
    }

    #[test]
    fn selector_returns_none_when_no_candidate_has_positive_disagreement() {
        let hypotheses = [1_i32, 2_i32, 3_i32];
        let predict = |_: &i32, _: &f64| Some(4.0);
        let candidates = [0.0, 1.0, 2.0];
        assert!(
            select_most_informative_experiment(&candidates, &hypotheses, predict).is_none()
        );
    }

}
