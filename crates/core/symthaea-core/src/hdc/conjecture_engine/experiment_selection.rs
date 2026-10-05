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

/// Population variance (not sample variance -- deliberate: this scores
/// *disagreement across the hypothesis set itself*, not an estimate of a
/// variance parameter from a sample of some larger population).
fn variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
}

/// How informative would running `experiment` be for discriminating among
/// `hypotheses`? Scored as the variance of their predictions for that
/// experiment -- high variance means the hypotheses disagree a lot, so
/// whichever one turns out to match reality, this experiment clearly rules
/// out the others; low variance (including the degenerate case where every
/// hypothesis predicts the same thing) means this experiment teaches
/// nothing about which hypothesis is right.
///
/// Returns `0.0` if fewer than 2 hypotheses produce a prediction for this
/// experiment (nothing to disagree about).
pub fn epistemic_value<H, E>(
    experiment: &E,
    hypotheses: &[H],
    predict: impl Fn(&H, &E) -> Option<f64>,
) -> f64 {
    let predictions: Vec<f64> = hypotheses
        .iter()
        .filter_map(|h| predict(h, experiment))
        .collect();
    variance(&predictions)
}

/// Select, from a pool of candidate experiments, the one with maximum
/// [`epistemic_value`] against `hypotheses` -- i.e. the single most
/// discriminating experiment to run next. Returns `None` if `candidates` is
/// empty. Ties broken by first occurrence (stable, deterministic given a
/// fixed candidate ordering).
pub fn select_most_informative_experiment<'a, H, E>(
    candidates: &'a [E],
    hypotheses: &[H],
    predict: impl Fn(&H, &E) -> Option<f64> + Copy,
) -> Option<(&'a E, f64)> {
    candidates
        .iter()
        .enumerate()
        .map(|(index, experiment)| {
            (index, experiment, epistemic_value(experiment, hypotheses, predict))
        })
        .max_by(|a, b| {
            a.2.partial_cmp(&b.2)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.0.cmp(&a.0))
        })
        .map(|(_, experiment, value)| (experiment, value))
}


/// Select an experiment and emit an evidence-neutral reproducibility receipt.
pub fn select_most_informative_experiment_with_receipt<'a, H, E, P, D>(
    candidates: &'a [E],
    hypotheses: &[H],
    predict: P,
    hypothesis_handoff_digest: impl Into<String>,
    hypothesis_set_digest: impl Into<String>,
    challenge_space_digest: impl Into<String>,
    selector_revision: impl Into<String>,
    selection_seed: u64,
    challenge_digest: D,
) -> Result<Option<(&'a E, ScientificInquirySelectionReceipt)>, String>
where
    P: Fn(&H, &E) -> Option<f64> + Copy,
    D: Fn(&E) -> String,
{
    let Some((selected, predicted)) =
        select_most_informative_experiment(candidates, hypotheses, predict)
    else {
        return Ok(None);
    };

    let receipt = ScientificInquirySelectionReceipt::new(
        hypothesis_handoff_digest,
        hypothesis_set_digest,
        challenge_space_digest,
        challenge_digest(selected),
        selector_revision,
        selection_seed,
        predicted,
    )?;
    Ok(Some((selected, receipt)))
}

#[cfg(test)]
mod tests {
    use super::*;
use symthaea_types::ScientificInquirySelectionReceipt;
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
    fn selector_uses_first_candidate_on_exact_tie() {
        #[derive(Debug, PartialEq)]
        struct Experiment(u8);

        let hypotheses = [0u8, 1u8];
        let candidates = [Experiment(1), Experiment(2), Experiment(3)];
        let predict = |_h: &u8, _e: &Experiment| Some(1.0);
        let (chosen, value) =
            select_most_informative_experiment(&candidates, &hypotheses, predict)
                .expect("non-empty candidate pool");
        assert_eq!(chosen, &Experiment(1));
        assert_eq!(value, 0.0);
    }

    #[test]
    fn selection_receipt_binds_the_chosen_experiment() {
        let hypotheses = [0u8, 1u8];
        let candidates = [1.0f64, 3.0f64];
        let predict = |h: &u8, x: &f64| Some(if *h == 0 { *x } else { *x * 2.0 });
        let (chosen, receipt) = select_most_informative_experiment_with_receipt(
            &candidates,
            &hypotheses,
            predict,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            "selector-v1",
            17,
            |x| format!("experiment:{x:.1}"),
        )
        .unwrap()
        .expect("candidate pool is non-empty");

        assert_eq!(*chosen, 3.0);
        assert!(receipt.validate().is_ok());
        assert_eq!(
            receipt.selected_challenge_digest,
            "experiment:3.0"
        );
        assert!(receipt.predicted_information_gain() > 0.0);
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
}
