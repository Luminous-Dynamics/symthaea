//! Interval scoring must use containment, not exact equality.
//!
//! See `docs/FORECAST_INTERVAL_OVERLAP_AUDIT_2026-07-31.md` (Q4) and
//! `docs/FORECAST_CONTAINMENT_PREREGISTRATION_2026-07-31.md`.
//!
//! Before the containment change, `probability_of` matched regions by exact equality, so an
//! outcome that fell *inside* a forecast interval was scored as if the forecaster had assigned it
//! no probability at all. Measured: forecast `[0,10]` at p=0.9, actual 4.0 gave Brier 1.82 (max
//! for two classes is 2.0) and log 20.72 (exactly `-ln(1e-9)`, the epsilon floor). CRPS, which
//! already used the interval midpoint, gave 0.9.
//!
//! Each test below was run against the unfixed implementation first; the two that must fail did.

use symthaea_futures_calibration::*;
use symthaea_futures_core::*;

fn dist(branches: Vec<(f64, (f64, f64))>, unsupported: f64) -> ForecastDistribution {
    let bs = branches
        .into_iter()
        .map(|(p, (lo, hi))| ForecastBranch {
            probability: Probability::new(p).unwrap(),
            outcome: OutcomeRegion::interval(lo, hi).unwrap(),
            assumptions: vec![],
        })
        .collect();
    ForecastDistribution::try_new(
        0,
        Horizon(10),
        OutcomeSpaceId("s".into()),
        bs,
        Probability::new(unsupported).unwrap(),
    )
    .expect("valid forecast")
}

fn at(x: f64) -> OutcomeRegion {
    OutcomeRegion::interval(x, x).unwrap()
}

/// The direct Q4 regression guard.
///
/// A forecaster that put 0.9 on a bin the outcome actually landed in has been *right*. It must not
/// receive the score reserved for one that assigned essentially zero probability to what happened.
#[test]
fn value_inside_forecast_interval_is_not_a_total_miss() {
    let f = dist(vec![(0.9, (0.0, 10.0))], 0.1);
    let actual = at(4.0);

    let log = LogScore::default().score(&f, &actual).unwrap().get();
    let brier = BrierScore.score(&f, &actual).unwrap().get();

    // -ln(0.9) is about 0.105. The pre-fix value was 20.72, the epsilon floor.
    assert!(
        log < 1.0,
        "log score {log:.4} for an outcome INSIDE the forecast interval. The epsilon floor is \
         -ln(1e-9) = 20.72; anything near it means the forecast's mass was never matched at all."
    );
    assert!(
        brier < 0.5,
        "Brier {brier:.4} for an outcome INSIDE the forecast interval (max for two classes is 2.0)"
    );
}

/// The paired control: all three rules must agree which of two forecasts is better.
///
/// Under exact-equality matching Brier and log could not distinguish these at all — both
/// forecasts scored identically, because neither's interval equalled the point outcome — so this
/// test fails against the unfixed code by construction.
#[test]
fn cross_rule_ordering_agrees() {
    // `good` concentrates on the bin containing the outcome; `bad` concentrates on the other bin.
    let good = dist(vec![(0.9, (0.0, 10.0)), (0.1, (10.1, 20.0))], 0.0);
    let bad = dist(vec![(0.1, (0.0, 10.0)), (0.9, (10.1, 20.0))], 0.0);
    let actual = at(4.0);

    let (bg, bb) = (
        BrierScore.score(&good, &actual).unwrap().get(),
        BrierScore.score(&bad, &actual).unwrap().get(),
    );
    let (lg, lb) = (
        LogScore::default().score(&good, &actual).unwrap().get(),
        LogScore::default().score(&bad, &actual).unwrap().get(),
    );
    let (cg, cb) = (
        Crps.score(&good, &actual).unwrap().get(),
        Crps.score(&bad, &actual).unwrap().get(),
    );

    // All three are loss functions: lower is better.
    assert!(bg < bb, "Brier disagrees: good {bg:.4} !< bad {bb:.4}");
    assert!(lg < lb, "log disagrees: good {lg:.4} !< bad {lb:.4}");
    assert!(cg < cb, "CRPS disagrees: good {cg:.4} !< bad {cb:.4}");
}

/// Pins prediction 1 of the pre-registration: the change is a no-op on degenerate intervals.
///
/// A point atom `[t, t]` contains midpoint `y` iff `t == y`, which is exactly what exact equality
/// did. `interval_atoms_distribution` in `symthaea-futures-ensemble` emits only point atoms, so
/// this is what guarantees its numbers do not move.
#[test]
fn point_atoms_score_identically_to_exact_equality() {
    let f = dist(vec![(0.7, (3.0, 3.0)), (0.3, (8.0, 8.0))], 0.0);

    // Matching atom: mass 0.7 -> -ln(0.7).
    let hit = LogScore::default().score(&f, &at(3.0)).unwrap().get();
    assert!(
        (hit - (-0.7f64.ln())).abs() < 1e-9,
        "point atom [3,3] should match outcome 3.0 with p=0.7, got log {hit:.6}"
    );

    // Non-matching outcome: no atom contains it, so it stays a total miss. That is correct here —
    // the forecast genuinely assigned it nothing.
    let miss = LogScore::default().score(&f, &at(5.0)).unwrap().get();
    assert!(
        miss > 10.0,
        "outcome 5.0 is in no atom and must remain unmatched, got log {miss:.6}"
    );
}

/// The partition property the non-overlap gate buys, now that containment makes it load-bearing.
///
/// Under exact equality the overlap check was defensive only (Q3 of the audit). With containment,
/// two bins containing the same point would double-count mass — so this asserts the gate holds.
#[test]
fn containment_matches_at_most_one_bin() {
    let f = dist(
        vec![(0.25, (0.0, 5.0)), (0.25, (5.1, 10.0)), (0.5, (10.1, 20.0))],
        0.0,
    );

    // Sweep the covered range; matched mass may never exceed the largest single bin's mass.
    let mut x = 0.0;
    while x <= 20.0 {
        let p = LogScore { epsilon: 1e-12 }.score(&f, &at(x)).unwrap().get();
        let matched = (-p).exp();
        assert!(
            matched <= 0.5 + 1e-9,
            "outcome {x} matched {matched:.4} of the mass — more than any single bin holds, so \
             two bins contained it and mass was double-counted"
        );
        x += 0.25;
    }
}
