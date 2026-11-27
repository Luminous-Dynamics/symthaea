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
//! experiment" -- the classic query-by-committee / experimental-design
//! framing. The score below is explicitly a disagreement heuristic, not a
//! Shannon-information calculation, because no hypothesis prior/posterior
//! distribution is supplied. It is not a reimplementation of
//! `symthaea-fep`'s machinery.
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
    discriminative_value(experiment, hypotheses, predict)
        .map(|(score, _)| score)
        .unwrap_or(0.0)
}

/// Return the finite prediction-disagreement score and the number of hypotheses
/// directly compared for one candidate experiment.
///
/// This is deliberately *not* Shannon information gain: the selector has no
/// prior/posterior hypothesis probabilities, so population variance is only a
/// disagreement heuristic. A candidate with fewer than two finite predictions
/// is not discriminative and is therefore rejected from the strict inquiry path.
pub fn discriminative_value<H, E>(
    experiment: &E,
    hypotheses: &[H],
    predict: impl Fn(&H, &E) -> Option<f64>,
) -> Option<(f64, u32)> {
    let predictions: Vec<f64> = hypotheses
        .iter()
        .filter_map(|h| predict(h, experiment).filter(|value| value.is_finite()))
        .collect();

    if predictions.len() < 2 {
        return None;
    }

    let score = variance(&predictions);
    if !score.is_finite() || score < 0.0 || predictions.len() > u32::MAX as usize {
        return None;
    }

    Some((score, predictions.len() as u32))
}

/// Select the candidate experiment that most strongly discriminates between
/// the surviving hypotheses. Only candidates with at least two finite
/// hypothesis predictions are eligible.
///
/// Ties are broken by first occurrence, making the result deterministic for a
/// fixed candidate ordering.
pub fn select_most_discriminative_experiment<'a, H, E>(
    candidates: &'a [E],
    hypotheses: &[H],
    predict: impl Fn(&H, &E) -> Option<f64> + Copy,
) -> Option<(&'a E, f64, u32)> {
    candidates
        .iter()
        .enumerate()
        .filter_map(|(index, experiment)| {
            discriminative_value(experiment, hypotheses, predict)
                .map(|(score, count)| (index, experiment, score, count))
        })
        .max_by(|a, b| {
            a.2.partial_cmp(&b.2)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.0.cmp(&a.0))
        })
        .map(|(_, experiment, score, count)| (experiment, score, count))
}

/// Backward-compatible selection name. The implementation now uses the
/// strict discriminative path so active-inquiry callers cannot select a
/// challenge where only one hypothesis can make a finite prediction.
pub fn select_most_informative_experiment<'a, H, E>(
    candidates: &'a [E],
    hypotheses: &[H],
    predict: impl Fn(&H, &E) -> Option<f64> + Copy,
) -> Option<(&'a E, f64)> {
    select_most_discriminative_experiment(candidates, hypotheses, predict)
        .map(|(experiment, score, _)| (experiment, score))
}

/// Select a discriminative experiment after converting every finite hypothesis
/// prediction into one explicit physical frame. A candidate is eligible only
/// when every live hypothesis produces a finite, physically compatible value.
///
/// This closes the unit-scale loophole in raw variance: `1000 J` and `1 kJ`
/// must compare as equal before disagreement is scored.
pub fn discriminative_value_typed<H, E>(
    experiment: &E,
    hypotheses: &[H],
    prediction_frame: &symthaea_types::PhysicalType,
    predict: impl Fn(&H, &E) -> Option<(f64, symthaea_types::PhysicalType)>,
) -> Result<Option<(f64, u32)>, String> {
    prediction_frame
        .validate()
        .map_err(|error| format!("invalid prediction frame: {}", error.reason))?;

    if hypotheses.len() < 2 {
        return Ok(None);
    }

    let mut predictions = Vec::with_capacity(hypotheses.len());
    for hypothesis in hypotheses {
        let Some((value, source_type)) = predict(hypothesis, experiment) else {
            return Ok(None);
        };
        let symthaea_types::TypeJudgement::Valid(normalized) =
            source_type.convert_value_to(value, prediction_frame)
        else {
            return Ok(None);
        };
        if !normalized.is_finite() {
            return Ok(None);
        }
        predictions.push(normalized);
    }

    let score = variance(&predictions);
    if !score.is_finite() || score < 0.0 || predictions.len() > u32::MAX as usize {
        return Ok(None);
    }

    Ok(Some((score, predictions.len() as u32)))
}

pub fn select_most_discriminative_experiment_typed<'a, H, E>(
    candidates: &'a [E],
    hypotheses: &[H],
    prediction_frame: &symthaea_types::PhysicalType,
    predict: impl Fn(&H, &E) -> Option<(f64, symthaea_types::PhysicalType)> + Copy,
) -> Result<Option<(&'a E, f64, u32)>, String> {
    let mut best: Option<(&'a E, f64, u32, usize)> = None;

    for (index, candidate) in candidates.iter().enumerate() {
        let Some((score, count)) = discriminative_value_typed(
            candidate,
            hypotheses,
            prediction_frame,
            predict,
        )? else {
            continue;
        };

        let replace = match best {
            None => true,
            Some((_, best_score, _, best_index)) => {
                score > best_score || (score == best_score && index < best_index)
            }
        };
        if replace {
            best = Some((candidate, score, count, index));
        }
    }

    Ok(best.map(|(candidate, score, count, _)| (candidate, score, count)))
}

/// Select a typed discriminative experiment and emit an evidence-neutral
/// reproducibility receipt. The receipt records the canonical physical frame,
/// prediction coverage, and disagreement score—not realized evidence.
pub fn select_most_discriminative_experiment_with_typed_receipt<'a, H, E, P, D>(
    candidates: &'a [E],
    hypotheses: &[H],
    prediction_frame: &symthaea_types::PhysicalType,
    predict: P,
    hypothesis_handoff_digest: impl Into<String>,
    hypothesis_set_digest: impl Into<String>,
    challenge_space_digest: impl Into<String>,
    selector_revision: impl Into<String>,
    selection_seed: u64,
    challenge_digest: D,
) -> Result<Option<(&'a E, ScientificInquirySelectionReceipt)>, String>
where
    P: Fn(&H, &E) -> Option<(f64, symthaea_types::PhysicalType)> + Copy,
    D: Fn(&E) -> String,
{
    let Some((selected, predicted, prediction_count)) =
        select_most_discriminative_experiment_typed(
            candidates,
            hypotheses,
            prediction_frame,
            predict,
        )?
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
        prediction_frame.digest_hex(),
        predicted,
        prediction_count,
        hypotheses.len() as u32,
    )?;
    Ok(Some((selected, receipt)))
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
    fn typed_selector_normalizes_prediction_units_before_scoring() {
        let frame = PhysicalType::with_kind(
            symthaea_types::QuantityKind::Energy,
            symthaea_types::PhysicalDimension::ENERGY,
        )
        .with_unit(symthaea_types::UnitRef {
            symbol: "J".into(),
            dimension: symthaea_types::PhysicalDimension::ENERGY,
            transform_to_si: symthaea_types::UnitTransform::IDENTITY,
            semantic_id: None,
        });
        let kilojoule = frame.clone().with_unit(symthaea_types::UnitRef {
            symbol: "kJ".into(),
            dimension: symthaea_types::PhysicalDimension::ENERGY,
            transform_to_si: symthaea_types::UnitTransform::new(
                symthaea_types::RationalScale { numerator: 1000, denominator: 1 },
                symthaea_types::RationalScale { numerator: 0, denominator: 1 },
            ),
            semantic_id: None,
        });

        let hypotheses = [0u8, 1u8];
        let candidates = [0u8, 1u8];
        let predict = |h: &u8, candidate: &u8| {
            match (*h, *candidate) {
                (0, 0) => Some((1000.0, frame.clone())),
                (1, 0) => Some((1.0, kilojoule.clone())),
                (0, 1) => Some((1000.0, frame.clone())),
                (1, 1) => Some((2.0, kilojoule.clone())),
                _ => None,
            }
        };

        let (chosen, score, count) =
            select_most_discriminative_experiment_typed(
                &candidates, &hypotheses, &frame, predict
            )
            .unwrap()
            .expect("complete physical predictions");

        assert_eq!(*chosen, 1);
        assert!(score > 0.0);
        assert_eq!(count, 2);
    }

    #[test]
    fn typed_selector_rejects_semantically_incompatible_prediction() {
        let frame = PhysicalType::with_kind(
            symthaea_types::QuantityKind::Energy,
            symthaea_types::PhysicalDimension::ENERGY,
        );
        let torque = PhysicalType::with_kind(
            symthaea_types::QuantityKind::Torque,
            symthaea_types::PhysicalDimension::ENERGY,
        );
        let hypotheses = [0u8, 1u8];
        let candidates = [0u8];
        let predict = |h: &u8, _candidate: &u8| {
            if *h == 0 {
                Some((1.0, frame.clone()))
            } else {
                Some((1.0, torque.clone()))
            }
        };

        assert!(
            select_most_discriminative_experiment_typed(
                &candidates, &hypotheses, &frame, predict
            )
            .unwrap()
            .is_none()
        );
    }

    #[test]
    fn typed_receipt_binds_prediction_frame_and_full_coverage() {
        let frame = PhysicalType::with_kind(
            symthaea_types::QuantityKind::Length,
            symthaea_types::PhysicalDimension::LENGTH,
        );
        let hypotheses = [0u8, 1u8];
        let candidates = [1u8];
        let predict = |_h: &u8, _candidate: &u8| Some((1.0, frame.clone()));

        let (_, receipt) = select_most_discriminative_experiment_with_typed_receipt(
            &candidates, &hypotheses, &frame, predict,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            "selector-v3", 17,
            |candidate| format!("experiment:{candidate}"),
        )
        .unwrap()
        .expect("typed candidate should be eligible");

        assert_eq!(receipt.prediction_count, 2);
        assert_eq!(receipt.hypothesis_count, 2);
        assert_eq!(receipt.prediction_frame_digest, frame.digest_hex());
        assert!(receipt.validate().is_ok());
        assert!(receipt.validate_against_prediction_frame(&frame).is_ok());
    }


    #[test]
    fn strict_selector_rejects_single_prediction_candidates() {
        let hypotheses = [0u8, 1u8];
        let candidates = [0.0f64, 2.0f64];
        let predict = |h: &u8, x: &f64| {
            if *h == 0 {
                Some(*x)
            } else if *x == 0.0 {
                None
            } else {
                Some(*x * 2.0)
            }
        };
        let (chosen, score, count) =
            select_most_discriminative_experiment(&candidates, &hypotheses, predict)
                .expect("at least one candidate has two finite predictions");
        assert_eq!(*chosen, 2.0);
        assert!(score > 0.0);
        assert_eq!(count, 2);
    }

    #[test]
    fn strict_selector_rejects_nonfinite_predictions() {
        let hypotheses = [0u8, 1u8];
        let candidates = [1.0f64, 2.0f64];
        let predict = |h: &u8, x: &f64| {
            Some(match (*h, *x as u8) {
                (0, 1) => f64::NAN,
                (1, 1) => 1.0,
                (0, 2) => 2.0,
                (1, 2) => 4.0,
                _ => 0.0,
            })
        };
        let (chosen, score, count) =
            select_most_discriminative_experiment(&candidates, &hypotheses, predict)
                .expect("finite candidate should remain eligible");
        assert_eq!(*chosen, 2.0);
        assert!(score > 0.0);
        assert_eq!(count, 2);
    }

    #[test]
    fn receipt_selector_returns_none_without_two_finite_predictions() {
        let hypotheses = [0u8, 1u8];
        let candidates = [0.0f64];
        let predict = |h: &u8, _x: &f64| (*h == 0).then_some(1.0);

        let result = select_most_discriminative_experiment_with_receipt(
            &candidates,
            &hypotheses,
            predict,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            "selector-v2",
            17,
            |x| format!("experiment:{x:.1}"),
        )
        .expect("structural selector inputs are valid");

        assert!(result.is_none());
    }

    #[test]
    fn selection_receipt_records_prediction_coverage() {
        let hypotheses = [0u8, 1u8];
        let candidates = [1.0f64, 3.0f64];
        let predict = |h: &u8, x: &f64| Some(if *h == 0 { *x } else { *x * 2.0 });
        let (_, receipt) = select_most_discriminative_experiment_with_receipt(
            &candidates,
            &hypotheses,
            predict,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            "selector-v2",
            17,
            |x| format!("experiment:{x:.1}"),
        )
        .unwrap()
        .expect("candidate pool is non-empty");
        assert_eq!(receipt.prediction_count, 2);
        assert!(receipt.predicted_disagreement_score() > 0.0);
        assert!(receipt.validate().is_ok());
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
