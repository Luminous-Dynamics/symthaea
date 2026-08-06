// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-futures-calibration
//!
//! Proper scoring rules and calibration diagnostics for the Symthaea Futures Laboratory
//! (`SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md`).
//!
//! Per the plan: this is deliberately *not* the hard part of Phase 1 — CRPS/Brier/log-score are
//! well-trodden. What the plan requires beyond "implement the formulas" is property tests
//! against known-calibrated synthetic cases, not just unit tests on hand-picked numbers — that's
//! what the test module below does.
//!
//! ## Which rule for which outcome space
//!
//! [`BrierScore`] and [`LogScore`] are for [`OutcomeRegion::Boolean`]/[`OutcomeRegion::Discrete`]
//! outcome spaces (Phase 1's extinction-within-horizon target) — they match branches to the
//! realized outcome by exact equality, which is meaningful for booleans/labels but not for
//! continuous values. [`Crps`] is for [`OutcomeRegion::Interval`] outcome spaces (Phase 1's
//! time-to-extinction target) — it never uses exact equality on floats.
//!
//! ## A documented scoring-convention choice (read before comparing numbers to another source)
//!
//! [`BrierScore`] uses the classic **multi-class** formulation (Brier, 1950): sum over every
//! distinct outcome region appearing among the forecast's branches — plus, if
//! `unsupported_mass > 0`, an implicit "not covered" pseudo-class — of
//! `(assigned_probability - indicator(actual))²`. For a two-branch Boolean forecast with
//! complementary branches (`p_true + p_false == 1`, no unsupported mass), this is **exactly
//! twice** the more commonly cited single-value binary Brier score `(p_true - indicator)²` seen
//! in weather forecasting — a well-known relationship between the two conventions (both
//! branches' squared errors are algebraically equal when they're complements), not a bug. The
//! property tests below are written against *this* crate's multi-class convention; don't compare
//! its numbers to a source using the binary convention without doubling/halving first.

use symthaea_futures_core::{ForecastDistribution, OutcomeRegion};

/// Which proper scoring rule to apply. Kept as an explicit enum (rather than one hardcoded
/// choice) so the evidence ledger can record exactly which rule scored a given forecast —
/// different outcome-space shapes need different rules (Brier/log-score for the discrete
/// extinction-within-horizon target, CRPS for the continuous time-to-extinction target; see the
/// plan's "First experiment" section).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoringRuleKind {
    Brier,
    Crps,
    LogScore,
}

/// A score that is guaranteed finite. Private storage: the whole point is that a caller
/// ranking forecasters can compare these without first re-checking for `NaN`/`inf`.
///
/// Pre-validation, `NaN` scores propagated silently into comparisons (`a < b` is always false
/// for `NaN`, so a `NaN`-producing forecaster's rank depended on iteration order and comparison
/// direction rather than on its accuracy) and `-inf` was reachable and optimal. Both are now
/// unrepresentable.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct FiniteScore(f64);

impl FiniteScore {
    /// Fails rather than producing a non-comparable score.
    pub fn new(value: f64) -> Result<Self, ScoringError> {
        if value.is_finite() {
            Ok(Self(value))
        } else {
            Err(ScoringError::NonFiniteScore { value })
        }
    }

    pub fn get(self) -> f64 {
        self.0
    }
}

impl From<FiniteScore> for f64 {
    fn from(s: FiniteScore) -> f64 {
        s.0
    }
}

/// Why a forecast could not be scored. These are *semantic* mismatches that survive
/// construction-time validation — a forecast can be perfectly well-formed and still be the
/// wrong shape for a given rule.
#[derive(Debug, Clone, PartialEq)]
pub enum ScoringError {
    /// [`Crps`] was asked to score against a non-`Interval` outcome. Previously returned `NaN`,
    /// which silently contaminated any aggregate it entered.
    ActualNotIntervalShaped,
    /// [`Crps`] found no `Interval`-shaped branches to form point atoms from. Previously `NaN`.
    NoIntervalAtoms,
    /// A score came out non-finite despite validated inputs — should be unreachable for Brier
    /// and log score, and is retained as a hard backstop rather than a silent pass-through.
    NonFiniteScore { value: f64 },
}

impl std::fmt::Display for ScoringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ActualNotIntervalShaped => {
                write!(f, "CRPS requires an Interval-shaped actual outcome")
            }
            Self::NoIntervalAtoms => {
                write!(f, "CRPS requires at least one Interval-shaped branch")
            }
            Self::NonFiniteScore { value } => write!(f, "score was not finite: {value}"),
        }
    }
}

impl std::error::Error for ScoringError {}

/// Implemented once per [`ScoringRuleKind`].
pub trait ScoringRule {
    fn kind(&self) -> ScoringRuleKind;
    /// Scores a **validated** forecast. Returns [`ScoringError`] for semantic mismatches rather
    /// than a sentinel `NaN`.
    fn score(
        &self,
        forecast: &ForecastDistribution,
        actual: &OutcomeRegion,
    ) -> Result<FiniteScore, ScoringError>;
}

/// Whether `region` covers the realized outcome `actual`.
///
/// `Boolean`/`Discrete` match by equality, as they always have. `Interval` regions match by
/// **containment** of the outcome's representative point, which is its midpoint — the convention
/// [`Crps`] already used, adopted here so all three rules share one event model.
///
/// Before 2026-07-31 this was exact equality for every region kind. For intervals that is not a
/// scoring rule but a lookup that almost always misses: an outcome of 4.0 against a forecast of
/// `[0,10]` matched nothing, so a *correct* forecast scored at the log-score epsilon floor while
/// `Crps` scored the same forecast as nearly right. See
/// `docs/FORECAST_INTERVAL_OVERLAP_AUDIT_2026-07-31.md`.
///
/// This is a no-op on degenerate intervals: a point atom `[t, t]` contains `y` iff `t == y`.
fn region_contains(region: &OutcomeRegion, actual: &OutcomeRegion) -> bool {
    match (region, actual) {
        (OutcomeRegion::Interval(bin), OutcomeRegion::Interval(a)) => bin.contains(a.midpoint()),
        (x, y) => x == y,
    }
}

/// Total probability mass the forecast assigns to regions covering `target` (0.0 if none does).
///
/// At most one branch can match, because `try_new` rejects overlapping intervals — so this sum
/// cannot double-count. That guarantee is why containment is safe here.
fn probability_of(forecast: &ForecastDistribution, target: &OutcomeRegion) -> f64 {
    forecast
        .branches()
        .iter()
        .filter(|b| region_contains(&b.outcome, target))
        .map(|b| b.probability.get())
        .sum()
}

/// The set of distinct outcome regions the forecast actually considered — the classes the
/// multi-class Brier score sums over (see module docs on the convention).
fn distinct_outcomes(forecast: &ForecastDistribution) -> Vec<OutcomeRegion> {
    let mut seen: Vec<OutcomeRegion> = Vec::new();
    for b in forecast.branches() {
        if !seen.contains(&b.outcome) {
            seen.push(b.outcome.clone());
        }
    }
    seen
}

/// Classic multi-class Brier score (Brier, 1950) — see module docs for the convention and its
/// relationship to the single-value binary form.
pub struct BrierScore;

impl ScoringRule for BrierScore {
    fn kind(&self) -> ScoringRuleKind {
        ScoringRuleKind::Brier
    }

    fn score(
        &self,
        forecast: &ForecastDistribution,
        actual: &OutcomeRegion,
    ) -> Result<FiniteScore, ScoringError> {
        // The classes are the forecast's own regions — its bins. The realized outcome is mapped
        // onto whichever bin covers it rather than being appended as its own class.
        //
        // Appending it was correct under exact-equality matching, where every class was disjoint
        // by construction. Under containment it is not: an outcome of 4.0 and a bin of [0,10] are
        // not disjoint, so scoring both would count the same 0.9 of mass twice — once as the bin
        // and once as the outcome — and inflate the score.
        let classes = distinct_outcomes(forecast);
        let covered = classes.iter().any(|c| region_contains(c, actual));

        let mut sum = 0.0;
        for class in &classes {
            let p = probability_of(forecast, class);
            let o = if region_contains(class, actual) {
                1.0
            } else {
                0.0
            };
            sum += (p - o).powi(2);
        }

        // An outcome no bin covers is a real miss and must be penalised: the forecast assigned it
        // nothing, so it enters as a class with p = 0 and indicator 1.
        if !covered {
            sum += 1.0;
        }

        // The implicit "not covered" pseudo-class: mass the forecast declined to assign to any
        // enumerated branch. Its indicator is always 0 (the realized outcome, by definition,
        // isn't "uncovered" from the ground truth's perspective — it happened).
        //
        // The `> 0.0` guard is now a pure fast-path: `unsupported_mass` is a `Probability`, so
        // it can no longer be negative. Before validation it could be, and this guard silently
        // discarded it — a forecast declaring `-0.5` uncovered mass took no penalty at all.
        let unsupported = forecast.unsupported_mass().get();
        if unsupported > 0.0 {
            sum += unsupported.powi(2);
        }

        FiniteScore::new(sum)
    }
}

/// `-ln(clamped probability assigned to the realized outcome)`. For
/// `Boolean`/`Discrete` outcome spaces (see module docs) — matches by exact equality.
pub struct LogScore {
    /// Floor on the matched probability before taking `ln`, so a forecast that assigned exactly
    /// zero probability to what actually happened doesn't produce literal `f64::INFINITY`.
    /// Default `1e-9`.
    pub epsilon: f64,
}

impl Default for LogScore {
    fn default() -> Self {
        Self { epsilon: 1e-9 }
    }
}

impl ScoringRule for LogScore {
    fn kind(&self) -> ScoringRuleKind {
        ScoringRuleKind::LogScore
    }

    fn score(
        &self,
        forecast: &ForecastDistribution,
        actual: &OutcomeRegion,
    ) -> Result<FiniteScore, ScoringError> {
        // `p` is a sum of validated `Probability` values, so it lies in [0, 1] and the result is
        // nonnegative. Before validation, `p > 1` yielded a *negative* log score and `p = inf`
        // yielded `-inf` — i.e. an invalid forecaster beat every honest one. Now unrepresentable.
        let p = probability_of(forecast, actual).max(self.epsilon);
        FiniteScore::new(-p.ln())
    }
}

/// Midpoint of an `Interval` region, used as its representative location for [`Crps`]. `None`
/// for `Boolean`/`Discrete` — those aren't `Crps`'s intended input (see module docs).
fn interval_midpoint(region: &OutcomeRegion) -> Option<f64> {
    match region {
        OutcomeRegion::Interval(iv) => Some(iv.midpoint()),
        _ => None,
    }
}

/// Continuous Ranked Probability Score, computed via the discrete/ensemble estimator (Gneiting &
/// Raftery 2007, "Strictly Proper Scoring Rules, Prediction, and Estimation" — the standard
/// closed form for a weighted set of scenario/point predictions):
///
/// `CRPS(F, y) = E_F|X - y| - 0.5 * E_F|X - X'|`, for independent `X, X' ~ F`.
///
/// Each branch is treated as a point atom at its interval midpoint with weight
/// `branch.probability`; `actual` is likewise reduced to its interval's midpoint (a true point
/// observation should be passed as `Interval { low: x, high: x }`). **Disclosed limitation**:
/// `unsupported_mass` has no location and is excluded from the atom set entirely — if it's
/// non-negligible, this CRPS underestimates the forecast's true predictive uncertainty. Not
/// silently corrected; a future revision could fold it in at a scenario-declared default
/// location if that turns out to matter in practice.
pub struct Crps;

impl ScoringRule for Crps {
    fn kind(&self) -> ScoringRuleKind {
        ScoringRuleKind::Crps
    }

    fn score(
        &self,
        forecast: &ForecastDistribution,
        actual: &OutcomeRegion,
    ) -> Result<FiniteScore, ScoringError> {
        let Some(y) = interval_midpoint(actual) else {
            // actual isn't Interval-shaped — Crps has nothing to compute against. Previously
            // returned NaN, which silently contaminated any aggregate it entered.
            return Err(ScoringError::ActualNotIntervalShaped);
        };

        let atoms: Vec<(f64, f64)> = forecast
            .branches()
            .iter()
            .filter_map(|b| interval_midpoint(&b.outcome).map(|loc| (loc, b.probability.get())))
            .collect();

        if atoms.is_empty() {
            return Err(ScoringError::NoIntervalAtoms);
        }

        let expected_abs_error: f64 = atoms.iter().map(|&(x, p)| p * (x - y).abs()).sum();

        let mut expected_pairwise_spread = 0.0;
        for &(xi, pi) in &atoms {
            for &(xj, pj) in &atoms {
                expected_pairwise_spread += pi * pj * (xi - xj).abs();
            }
        }

        FiniteScore::new(expected_abs_error - 0.5 * expected_pairwise_spread)
    }
}

/// One bucket of a [`ReliabilityDiagram`]: how well-calibrated the forecaster is among
/// predictions that fell in `[bucket_low, bucket_high)` (the last bucket includes `1.0`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReliabilityBucket {
    pub bucket_low: f64,
    pub bucket_high: f64,
    /// Mean predicted `P(true)` among predictions in this bucket.
    pub mean_predicted_probability: f64,
    /// Fraction of predictions in this bucket where the actual outcome was `true`.
    pub empirical_frequency: f64,
    pub count: usize,
}

/// A reliability (calibration) curve: `num_buckets` equal-width buckets over `[0, 1]`, each
/// comparing mean predicted probability against empirical frequency. A well-calibrated
/// forecaster has `mean_predicted_probability ≈ empirical_frequency` in every bucket. Empty
/// buckets are omitted, not padded with `NaN`/zero — an empty bucket has no meaningful predicted
/// probability to report.
#[derive(Debug, Clone, PartialEq)]
pub struct ReliabilityDiagram {
    pub buckets: Vec<ReliabilityBucket>,
}

impl ReliabilityDiagram {
    /// Expected Calibration Error: count-weighted mean absolute gap between predicted
    /// probability and empirical frequency across non-empty buckets. `NaN` if there are no
    /// predictions at all (rather than silently reporting `0.0`, which would misrepresent "no
    /// data" as "perfectly calibrated").
    pub fn expected_calibration_error(&self) -> f64 {
        let total: usize = self.buckets.iter().map(|b| b.count).sum();
        if total == 0 {
            return f64::NAN;
        }
        self.buckets
            .iter()
            .map(|b| {
                (b.count as f64 / total as f64)
                    * (b.mean_predicted_probability - b.empirical_frequency).abs()
            })
            .sum()
    }
}

/// Builds a [`ReliabilityDiagram`] from `(predicted P(true), actual outcome)` pairs.
/// `num_buckets` is clamped to at least 1.
pub fn reliability_diagram(predictions: &[(f64, bool)], num_buckets: usize) -> ReliabilityDiagram {
    let num_buckets = num_buckets.max(1);
    let width = 1.0 / num_buckets as f64;
    // (sum of predicted probabilities, count where actual was true, total count)
    let mut sums = vec![(0.0f64, 0usize, 0usize); num_buckets];

    for &(p, actual) in predictions {
        let p = p.clamp(0.0, 1.0);
        let idx = ((p / width) as usize).min(num_buckets - 1);
        sums[idx].0 += p;
        sums[idx].2 += 1;
        if actual {
            sums[idx].1 += 1;
        }
    }

    let buckets = sums
        .into_iter()
        .enumerate()
        .filter(|(_, (_, _, count))| *count > 0)
        .map(|(i, (sum_p, true_count, count))| ReliabilityBucket {
            bucket_low: i as f64 * width,
            bucket_high: (i + 1) as f64 * width,
            mean_predicted_probability: sum_p / count as f64,
            empirical_frequency: true_count as f64 / count as f64,
            count,
        })
        .collect();

    ReliabilityDiagram { buckets }
}

/// Convenience: extracts `(predicted P(true), actual)` from a Boolean-outcome forecast and its
/// realized outcome, ready to feed into [`reliability_diagram`]. Returns `None` only if `actual`
/// isn't [`OutcomeRegion::Boolean`] — a forecast with no `Boolean(true)` branch legitimately
/// yields `p = 0.0` (an omitted branch means "zero probability assigned"), not `None`.
pub fn boolean_prediction_pair(
    forecast: &ForecastDistribution,
    actual: &OutcomeRegion,
) -> Option<(f64, bool)> {
    let OutcomeRegion::Boolean(actual_bool) = actual else {
        return None;
    };
    let p_true = forecast
        .branches()
        .iter()
        .find(|b| b.outcome == OutcomeRegion::Boolean(true))
        .map(|b| b.probability.get())
        .unwrap_or(0.0);
    Some((p_true, *actual_bool))
}

/// A post-hoc calibration correction: histogram binning (fit each equal-width bucket's
/// calibrated output to its empirical frequency ON THE TRAINING SET, reusing
/// [`reliability_diagram`]'s own bucketing rather than a second implementation of it), fit on
/// one set of predictions and applied to different ones.
///
/// **Why this exists**: a forecaster can produce a real, directionally-correct signal that is
/// nonetheless systematically mis-scaled (e.g. `symthaea-futures-ensemble`'s `FepDrivenGenerator`
/// was found to plateau at ~0.74 instead of ~1.0 once truly extinct — see
/// `SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md`'s rung-5 convergence probe for the traced
/// mechanism). Fixing the underlying model may be out of scope (as it was there — a shared core
/// crate); a calibration-correction layer fixes the *reported probability*, not the model,
/// entirely downstream of it.
///
/// **The non-negotiable rule this type exists to make easy to follow**: [`Self::fit`] and
/// evaluation must use *disjoint* data. Fitting and evaluating a correction on the same
/// predictions is look-ahead bias — it can only ever look calibrated on data it has already
/// seen, telling you nothing about held-out performance. This type doesn't enforce that at the
/// type level (Rust can't distinguish "training data" from "test data" by type), but callers
/// must — see `SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md` for the train/test seed split
/// this was built to require.
pub struct HistogramCalibrator {
    num_buckets: usize,
    /// bucket index -> calibrated probability, only for buckets the training data covered.
    calibrated: std::collections::HashMap<usize, f64>,
}

impl HistogramCalibrator {
    /// Fits a calibrator from `(raw predicted P(true), actual outcome)` **training** pairs.
    pub fn fit(training_predictions: &[(f64, bool)], num_buckets: usize) -> Self {
        let num_buckets = num_buckets.max(1);
        let width = 1.0 / num_buckets as f64;
        let diagram = reliability_diagram(training_predictions, num_buckets);

        let calibrated = diagram
            .buckets
            .iter()
            .map(|b| {
                let index = (b.bucket_low / width).round() as usize;
                (index, b.empirical_frequency)
            })
            .collect();

        Self {
            num_buckets,
            calibrated,
        }
    }

    /// Maps a raw predicted probability to its calibrated value. Falls back to the raw value
    /// unchanged — a disclosed, honest fallback, not a guess — for a bucket the training data
    /// never covered (empty buckets are omitted by [`reliability_diagram`], so there is no
    /// empirical frequency to correct toward).
    pub fn calibrate(&self, raw_p: f64) -> f64 {
        let raw_p = raw_p.clamp(0.0, 1.0);
        let width = 1.0 / self.num_buckets as f64;
        let index = ((raw_p / width) as usize).min(self.num_buckets - 1);
        self.calibrated.get(&index).copied().unwrap_or(raw_p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_futures_core::{Horizon, OutcomeSpaceId};

    fn distribution(
        branches: Vec<(f64, OutcomeRegion)>,
        unsupported_mass: f64,
    ) -> ForecastDistribution {
        ForecastDistribution::try_from_raw(
            0,
            Horizon(1),
            OutcomeSpaceId("test".to_string()),
            branches
                .into_iter()
                .map(|(probability, outcome)| (probability, outcome, Vec::new()))
                .collect(),
            unsupported_mass,
        )
        .expect("test fixture must be a valid forecast")
    }

    /// Scores and unwraps to a raw f64, for the many tests asserting exact numeric values.
    fn score_of(rule: &dyn ScoringRule, f: &ForecastDistribution, actual: &OutcomeRegion) -> f64 {
        rule.score(f, actual)
            .expect("valid forecast must score")
            .get()
    }

    // ---- BrierScore ----

    #[test]
    fn brier_perfect_forecast_scores_zero() {
        let f = distribution(
            vec![
                (1.0, OutcomeRegion::Boolean(true)),
                (0.0, OutcomeRegion::Boolean(false)),
            ],
            0.0,
        );
        assert_eq!(
            score_of(&BrierScore, &f, &OutcomeRegion::Boolean(true)),
            0.0
        );
    }

    #[test]
    fn brier_maximally_wrong_forecast_scores_two_for_complementary_binary_branches() {
        // Multi-class convention (see module docs): 1^2 + 1^2 = 2.0, not 1.0.
        let f = distribution(
            vec![
                (1.0, OutcomeRegion::Boolean(false)),
                (0.0, OutcomeRegion::Boolean(true)),
            ],
            0.0,
        );
        assert_eq!(
            score_of(&BrierScore, &f, &OutcomeRegion::Boolean(true)),
            2.0
        );
    }

    #[test]
    fn brier_matches_hand_computed_intermediate_value() {
        // p_true=0.7, p_false=0.3, actual=true: (0.7-1)^2 + (0.3-0)^2 = 0.09 + 0.09 = 0.18.
        let f = distribution(
            vec![
                (0.7, OutcomeRegion::Boolean(true)),
                (0.3, OutcomeRegion::Boolean(false)),
            ],
            0.0,
        );
        let score = score_of(&BrierScore, &f, &OutcomeRegion::Boolean(true));
        assert!((score - 0.18).abs() < 1e-12, "got {score}");
    }

    #[test]
    fn brier_penalizes_a_true_outcome_that_was_never_enumerated() {
        // Forecast only ever considered "false" — actual is "true", a class it never assigned
        // any probability to at all: (0-1)^2 for the never-considered true class, plus (1-0)^2
        // for the false class it was fully confident in = 2.0, the same maximal penalty as the
        // "actively wrong" case above -- correctly indistinguishable from confidently wrong.
        let f = distribution(vec![(1.0, OutcomeRegion::Boolean(false))], 0.0);
        assert_eq!(
            score_of(&BrierScore, &f, &OutcomeRegion::Boolean(true)),
            2.0
        );
    }

    #[test]
    fn brier_includes_unsupported_mass_as_its_own_penalty_term() {
        let f = distribution(vec![(0.6, OutcomeRegion::Boolean(true))], 0.4);
        // classes = {true}; (0.6-1)^2 = 0.16, plus unsupported_mass^2 = 0.16 -> 0.32.
        let score = score_of(&BrierScore, &f, &OutcomeRegion::Boolean(true));
        assert!((score - 0.32).abs() < 1e-12, "got {score}");
    }

    // ---- LogScore ----

    #[test]
    fn log_score_perfect_forecast_scores_zero() {
        let f = distribution(vec![(1.0, OutcomeRegion::Boolean(true))], 0.0);
        assert_eq!(
            score_of(&LogScore::default(), &f, &OutcomeRegion::Boolean(true)),
            0.0
        );
    }

    #[test]
    fn log_score_completely_missed_outcome_is_clamped_not_infinite() {
        let f = distribution(vec![(1.0, OutcomeRegion::Boolean(false))], 0.0);
        let score = score_of(&LogScore::default(), &f, &OutcomeRegion::Boolean(true));
        assert!(score.is_finite());
        assert!(score > 15.0, "expected a heavy penalty, got {score}"); // -ln(1e-9) ~= 20.7
    }

    #[test]
    fn log_score_matches_hand_computed_value() {
        let f = distribution(
            vec![
                (0.25, OutcomeRegion::Boolean(true)),
                (0.75, OutcomeRegion::Boolean(false)),
            ],
            0.0,
        );
        let score = score_of(&LogScore::default(), &f, &OutcomeRegion::Boolean(true));
        assert!((score - (0.25f64.ln() * -1.0)).abs() < 1e-12, "got {score}");
    }

    // ---- Crps ----

    #[test]
    fn crps_single_atom_reduces_to_absolute_error() {
        // With exactly one atom, E|X-X'| = 0 (X and X' are always the same value), so
        // CRPS = |v - y| exactly -- the degenerate case that reduces to plain MAE.
        let f = distribution(
            vec![(1.0, OutcomeRegion::interval(40.0, 40.0).unwrap())],
            0.0,
        );
        let actual = OutcomeRegion::interval(50.0, 50.0).unwrap();
        assert_eq!(score_of(&Crps, &f, &actual), 10.0);
    }

    #[test]
    fn crps_perfect_point_forecast_scores_zero() {
        let f = distribution(
            vec![(1.0, OutcomeRegion::interval(30.0, 30.0).unwrap())],
            0.0,
        );
        let actual = OutcomeRegion::interval(30.0, 30.0).unwrap();
        assert_eq!(score_of(&Crps, &f, &actual), 0.0);
    }

    #[test]
    fn crps_two_atom_matches_hand_computed_value() {
        // Atoms at 0 (p=0.5) and 10 (p=0.5), actual=0.
        // E|X-y| = 0.5*|0-0| + 0.5*|10-0| = 5.0
        // E|X-X'| = 0.25*|0-0| + 0.25*|0-10| + 0.25*|10-0| + 0.25*|10-10| = 2.5 + 2.5 = 5.0
        // CRPS = 5.0 - 0.5*5.0 = 2.5
        let f = distribution(
            vec![
                (0.5, OutcomeRegion::interval(0.0, 0.0).unwrap()),
                (0.5, OutcomeRegion::interval(10.0, 10.0).unwrap()),
            ],
            0.0,
        );
        let actual = OutcomeRegion::interval(0.0, 0.0).unwrap();
        let score = score_of(&Crps, &f, &actual);
        assert!((score - 2.5).abs() < 1e-12, "got {score}");
    }

    #[test]
    fn crps_is_never_negative_across_a_spread_of_random_looking_cases() {
        // CRPS is a proper scoring rule and must be >= 0 for any valid distribution/actual
        // pair -- a cheap but real property check, not just hand-picked numbers.
        let cases: Vec<(Vec<(f64, f64)>, f64)> = vec![
            (vec![(0.2, -5.0), (0.5, 0.0), (0.3, 8.0)], 3.0),
            (vec![(1.0, 0.0)], -100.0),
            (vec![(0.1, 1.0), (0.1, 2.0), (0.8, 3.0)], 2.5),
        ];
        for (atoms, y) in cases {
            let f = distribution(
                atoms
                    .into_iter()
                    .map(|(p, v)| (p, OutcomeRegion::interval(v, v).unwrap()))
                    .collect(),
                0.0,
            );
            let actual = OutcomeRegion::interval(y, y).unwrap();
            let score = score_of(&Crps, &f, &actual);
            assert!(score >= -1e-12, "CRPS was negative: {score}");
        }
    }

    // ---- ReliabilityDiagram ----

    /// Deterministic (not random) synthetic calibration data: for each `p` in
    /// `[0.15, 0.35, 0.55, 0.75, 0.95]`, 100 predictions at exactly that probability, with the
    /// actual outcome set to `true` for exactly the first `round(p * 100)` of them. This is a
    /// known-calibrated dataset by construction, not hand-picked numbers -- a perfectly
    /// calibrated forecaster over this data should show `mean_predicted_probability ≈
    /// empirical_frequency` in every bucket and near-zero ECE.
    fn perfectly_calibrated_predictions() -> Vec<(f64, bool)> {
        let mut predictions = Vec::new();
        for &p in &[0.15f64, 0.35, 0.55, 0.75, 0.95] {
            let true_count = (p * 100.0).round() as usize;
            for i in 0..100 {
                predictions.push((p, i < true_count));
            }
        }
        predictions
    }

    #[test]
    fn reliability_diagram_perfectly_calibrated_data_has_near_zero_ece() {
        let diagram = reliability_diagram(&perfectly_calibrated_predictions(), 10);
        let ece = diagram.expected_calibration_error();
        assert!(
            ece < 1e-9,
            "expected near-zero ECE for known-calibrated data, got {ece}"
        );

        // Each bucket's mean predicted probability should equal its empirical frequency exactly
        // (both are literally 0.15/0.35/0.55/0.75/0.95 by construction).
        assert_eq!(diagram.buckets.len(), 5);
        for bucket in &diagram.buckets {
            assert!(
                (bucket.mean_predicted_probability - bucket.empirical_frequency).abs() < 1e-12,
                "bucket [{}, {}) miscalibrated: predicted={}, empirical={}",
                bucket.bucket_low,
                bucket.bucket_high,
                bucket.mean_predicted_probability,
                bucket.empirical_frequency
            );
        }
    }

    #[test]
    fn reliability_diagram_overconfident_forecaster_has_a_large_ece() {
        // Always predicts p=0.9, but only actually true half the time: a real, known
        // miscalibration (overconfidence), not a fit to noise.
        let predictions: Vec<(f64, bool)> = (0..100).map(|i| (0.9, i % 2 == 0)).collect();
        let diagram = reliability_diagram(&predictions, 10);
        assert_eq!(diagram.buckets.len(), 1);
        let bucket = diagram.buckets[0];
        assert!((bucket.mean_predicted_probability - 0.9).abs() < 1e-12);
        assert!((bucket.empirical_frequency - 0.5).abs() < 1e-12);
        let ece = diagram.expected_calibration_error();
        assert!(
            (ece - 0.4).abs() < 1e-9,
            "expected ECE=0.4 (|0.9-0.5|), got {ece}"
        );
    }

    #[test]
    fn reliability_diagram_reports_nan_ece_with_no_predictions() {
        let diagram = reliability_diagram(&[], 10);
        assert!(diagram.buckets.is_empty());
        assert!(diagram.expected_calibration_error().is_nan());
    }

    #[test]
    fn boolean_prediction_pair_extracts_zero_probability_for_an_omitted_branch() {
        let f = distribution(vec![(1.0, OutcomeRegion::Boolean(false))], 0.0);
        let pair = boolean_prediction_pair(&f, &OutcomeRegion::Boolean(true));
        assert_eq!(pair, Some((0.0, true)));
    }

    #[test]
    fn boolean_prediction_pair_returns_none_for_a_non_boolean_actual() {
        let f = distribution(vec![(1.0, OutcomeRegion::Boolean(true))], 0.0);
        let actual = OutcomeRegion::interval(1.0, 1.0).unwrap();
        assert_eq!(boolean_prediction_pair(&f, &actual), None);
    }

    // ---- HistogramCalibrator ----

    #[test]
    fn calibrator_corrects_a_known_systematic_bias() {
        // Training data: raw prediction always 0.74 (matching the real rung-5 finding), but
        // actual is true only 30% of the time -- a real, known miscalibration to correct, not a
        // fit to noise.
        let training: Vec<(f64, bool)> = (0..100).map(|i| (0.74, i < 30)).collect();
        let calibrator = HistogramCalibrator::fit(&training, 10);
        assert!((calibrator.calibrate(0.74) - 0.30).abs() < 1e-9);
    }

    #[test]
    fn calibrator_applies_the_training_correction_to_different_held_out_values() {
        // The whole point: fit on one set of predictions, apply to genuinely different ones in
        // the same bucket (not the exact values seen during fit).
        let training: Vec<(f64, bool)> = (0..100).map(|i| (0.74, i < 30)).collect();
        let calibrator = HistogramCalibrator::fit(&training, 10);

        // 0.71 and 0.79 both fall in the same [0.7, 0.8) bucket as the training data's 0.74.
        assert!((calibrator.calibrate(0.71) - 0.30).abs() < 1e-9);
        assert!((calibrator.calibrate(0.79) - 0.30).abs() < 1e-9);
    }

    #[test]
    fn calibrator_falls_back_to_the_raw_value_for_an_uncovered_bucket() {
        let training: Vec<(f64, bool)> = (0..100).map(|i| (0.74, i < 30)).collect();
        let calibrator = HistogramCalibrator::fit(&training, 10);

        // No training data ever fell in [0.0, 0.1) -- honest fallback, not a fabricated guess.
        assert!((calibrator.calibrate(0.05) - 0.05).abs() < 1e-9);
    }

    #[test]
    fn calibrator_is_a_no_op_on_already_well_calibrated_data() {
        let training = perfectly_calibrated_predictions();
        let calibrator = HistogramCalibrator::fit(&training, 10);
        for &p in &[0.15, 0.35, 0.55, 0.75, 0.95] {
            assert!((calibrator.calibrate(p) - p).abs() < 1e-9);
        }
    }
}

#[cfg(test)]
mod validated_input_properties {
    //! Property tests over the *validated* input space: every forecast that
    //! `ForecastDistribution::try_from_raw` accepts must yield a finite, nonnegative Brier and
    //! log score. Deterministic grid sweep rather than randomized sampling — exhaustive over a
    //! discretization is both stronger here and immune to flaky-seed reports.
    //!
    //! The pre-validation counterexamples (`p > 1` → negative log score, `p = inf` → `-inf`,
    //! inverted interval → perfect 0.0 Brier) are no longer expressible: they are rejected at
    //! construction and covered by `symthaea-futures-core`'s own rejection tests.

    use super::*;
    use symthaea_futures_core::{Horizon, OutcomeSpaceId};

    fn boolean_forecast(p_true: f64, unsupported: f64) -> Option<ForecastDistribution> {
        let p_false = 1.0 - p_true - unsupported;
        if p_false < 0.0 {
            return None;
        }
        ForecastDistribution::try_from_raw(
            0,
            Horizon(1),
            OutcomeSpaceId("grid".into()),
            vec![
                (p_true, OutcomeRegion::Boolean(true), vec![]),
                (p_false, OutcomeRegion::Boolean(false), vec![]),
            ],
            unsupported,
        )
        .ok()
    }

    #[test]
    fn every_accepted_forecast_scores_finite_and_nonnegative() {
        let mut checked = 0usize;
        for i in 0..=100 {
            for u in 0..=20 {
                let p_true = i as f64 / 100.0;
                let unsupported = u as f64 / 20.0;
                let Some(f) = boolean_forecast(p_true, unsupported) else {
                    continue;
                };
                for actual in [OutcomeRegion::Boolean(true), OutcomeRegion::Boolean(false)] {
                    let brier = BrierScore
                        .score(&f, &actual)
                        .expect("validated forecast must yield a Brier score")
                        .get();
                    assert!(
                        brier.is_finite() && brier >= 0.0,
                        "Brier {brier} for p_true={p_true} unsupported={unsupported}"
                    );

                    let log = LogScore::default()
                        .score(&f, &actual)
                        .expect("validated forecast must yield a log score")
                        .get();
                    assert!(
                        log.is_finite() && log >= 0.0,
                        "LogScore {log} for p_true={p_true} unsupported={unsupported}"
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 1000, "grid too small to be meaningful: {checked}");
    }

    #[test]
    fn honest_forecast_scores_are_unchanged_by_validation() {
        // Exact pre-change values from the reproduction table in
        // `symthaea_futures_core::validated`'s module docs. Validation must not perturb the
        // score of a forecast that was already valid.
        let f = boolean_forecast(0.7, 0.0).expect("honest forecast is valid");
        let actual = OutcomeRegion::Boolean(true);

        let brier = BrierScore.score(&f, &actual).unwrap().get();
        assert!((brier - 0.1800).abs() < 1e-12, "Brier changed: {brier}");

        let log = LogScore::default().score(&f, &actual).unwrap().get();
        assert!(
            (log - 0.35667494393873245).abs() < 1e-12,
            "LogScore changed: {log}"
        );
    }

    #[test]
    fn crps_reports_shape_mismatch_instead_of_nan() {
        let f = boolean_forecast(0.5, 0.0).unwrap();
        assert_eq!(
            Crps.score(&f, &OutcomeRegion::Boolean(true)),
            Err(ScoringError::ActualNotIntervalShaped)
        );
        // A Boolean-only forecast has no interval atoms to compare an interval actual against.
        assert_eq!(
            Crps.score(&f, &OutcomeRegion::interval(0.0, 1.0).unwrap()),
            Err(ScoringError::NoIntervalAtoms)
        );
    }
}
