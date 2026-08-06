// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Validated forecast types — the enforcement half of the neutral contest boundary.
//!
//! ## Why these types have private storage
//!
//! Before this module existed, [`ForecastDistribution`] was a plain struct with public `f64`
//! fields and no constructor. The scoring layer (`symthaea-futures-calibration`) consumed those
//! fields directly, which meant the *ordering relation the whole laboratory depends on* was
//! defined over an input space that included values no probability can take. Empirically, on the
//! real scorers:
//!
//! | constructed forecast          | Brier   | LogScore |
//! |-------------------------------|---------|----------|
//! | honest: p(true)=0.7           |  0.1800 |   0.3567 |
//! | mass sums to 3.2 (p > 1)      |  1.7000 |  -0.5878 |
//! | p = INFINITY                  |     inf |     -inf |
//! | inverted interval low>high    |  0.0000 |  -0.0000 |
//!
//! Log score is `-ln(p)`. For `p > 1` that is **negative**, and for `p = ∞` it is `-∞`. Lower is
//! better. So an invalid forecaster did not merely produce a meaningless number — it *strictly
//! beat every honest forecaster*, and an infinite one attained the theoretical optimum. An
//! inverted interval scored a perfect 0.0 Brier by matching itself under exact equality.
//!
//! That is not an input-hygiene problem. It is an attack on the experiment's ordering relation,
//! reachable by ordinary numerical corruption as much as by malice. Hence: private storage,
//! validated construction, and validation on the deserialization path too (a `serde` round-trip
//! must not be a way to smuggle in what the constructor rejects).
//!
//! ## What is deliberately NOT enforced here
//!
//! Scores are not guaranteed *comparable across scenario families* — that is the evidence
//! ledger's job, not this type's. And [`crate::Crps`]-style rules may legitimately return
//! negative values; only Brier and log score are guaranteed nonnegative (see
//! `symthaea-futures-calibration`'s property tests).

use serde::{Deserialize, Serialize};

use crate::{AssumptionId, Horizon, OutcomeSpaceId};

/// Absolute tolerance on the total-mass invariant (`Σ branch probabilities + unsupported_mass
/// == 1`). Named rather than inlined so the number is reviewable and so tests can reference the
/// same constant the constructor uses. Chosen to absorb ordinary f64 summation error over a
/// realistic branch count without admitting a forecast that is meaningfully unnormalized.
pub const MASS_TOLERANCE: f64 = 1e-9;

/// Why a forecast was rejected at construction. Every variant carries the offending value so a
/// failure is diagnosable from the error alone, without re-deriving it from the input.
#[derive(Debug, Clone, PartialEq)]
pub enum ForecastError {
    /// A probability was `NaN` or infinite.
    ProbabilityNotFinite { value: f64 },
    /// A probability was finite but outside `[0, 1]`.
    ProbabilityOutOfRange { value: f64 },
    /// An interval bound was `NaN` or infinite.
    IntervalBoundNotFinite { low: f64, high: f64 },
    /// An interval had `low > high`.
    IntervalInverted { low: f64, high: f64 },
    /// A distribution had no branches. Absence of a forecast must be expressed as
    /// [`crate::ForecastOutput::Abstain`], never as an empty distribution — an empty
    /// distribution is silently scoreable (it produced a finite, competitive-looking Brier of
    /// 1.0) and would let a generator dodge abstention accounting.
    EmptyDistribution,
    /// `Σ branch probabilities + unsupported_mass` was not within [`MASS_TOLERANCE`] of 1.
    MassNotNormalized { total: f64, tolerance: f64 },
    /// Two branches carried the same `Boolean`/`Discrete` outcome region. Rejected rather than
    /// canonicalized: the scorer sums matching branches, so a duplicate silently double-counts
    /// mass, and merging them here would hide a generator bug rather than surface it.
    DuplicateOutcomeRegion,
    /// Two branches carried overlapping `Interval` regions. Rejected for the same reason —
    /// overlapping support means the branch masses are not a partition and the scorer's
    /// exact-match semantics stop being meaningful.
    OverlappingIntervals {
        first: (f64, f64),
        second: (f64, f64),
    },
}

impl std::fmt::Display for ForecastError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ProbabilityNotFinite { value } => {
                write!(f, "probability must be finite, got {value}")
            }
            Self::ProbabilityOutOfRange { value } => {
                write!(f, "probability must lie in [0, 1], got {value}")
            }
            Self::IntervalBoundNotFinite { low, high } => {
                write!(f, "interval bounds must be finite, got [{low}, {high}]")
            }
            Self::IntervalInverted { low, high } => {
                write!(f, "interval requires low <= high, got [{low}, {high}]")
            }
            Self::EmptyDistribution => write!(
                f,
                "a forecast distribution needs at least one branch; use ForecastOutput::Abstain \
                 to express the absence of a forecast"
            ),
            Self::MassNotNormalized { total, tolerance } => write!(
                f,
                "branch probabilities plus unsupported_mass must sum to 1 within {tolerance}, \
                 got {total}"
            ),
            Self::DuplicateOutcomeRegion => {
                write!(f, "two branches share the same outcome region")
            }
            Self::OverlappingIntervals { first, second } => write!(
                f,
                "branch intervals overlap: [{}, {}] and [{}, {}]",
                first.0, first.1, second.0, second.1
            ),
        }
    }
}

impl std::error::Error for ForecastError {}

/// A probability: finite and within `[0, 1]`, enforced at construction and on deserialization.
///
/// Storage is private specifically so that `Probability` cannot be produced by a struct literal
/// anywhere — including elsewhere inside this crate, since private fields are module-scoped and
/// this type lives in its own module.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "f64", into = "f64")]
pub struct Probability(f64);

impl Probability {
    /// Probability zero.
    pub const ZERO: Self = Self(0.0);
    /// Probability one.
    pub const ONE: Self = Self(1.0);

    /// The only way to build a `Probability` from a raw float.
    pub fn new(value: f64) -> Result<Self, ForecastError> {
        if !value.is_finite() {
            return Err(ForecastError::ProbabilityNotFinite { value });
        }
        if !(0.0..=1.0).contains(&value) {
            return Err(ForecastError::ProbabilityOutOfRange { value });
        }
        Ok(Self(value))
    }

    /// The underlying value, guaranteed finite and in `[0, 1]`.
    pub fn get(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for Probability {
    type Error = ForecastError;
    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<Probability> for f64 {
    fn from(p: Probability) -> f64 {
        p.0
    }
}

mod interval_repr {
    use serde::{Deserialize, Serialize};

    /// Wire representation of an [`super::Interval`]. Kept structurally identical to the old
    /// public `Interval { low, high }` enum-variant fields so this change is wire-compatible
    /// with forecasts already recorded in the evidence ledger.
    #[derive(Serialize, Deserialize)]
    pub struct IntervalRepr {
        pub low: f64,
        pub high: f64,
    }
}
use interval_repr::IntervalRepr;

/// A closed interval `[low, high]` with finite bounds and `low <= high`.
///
/// Private storage for the same reason as [`Probability`]: an inverted interval previously
/// scored a *perfect* 0.0 Brier, because exact-equality matching compared it against itself.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "IntervalRepr", into = "IntervalRepr")]
pub struct Interval {
    low: f64,
    high: f64,
}

impl Interval {
    /// The only way to build an `Interval`.
    pub fn new(low: f64, high: f64) -> Result<Self, ForecastError> {
        if !low.is_finite() || !high.is_finite() {
            return Err(ForecastError::IntervalBoundNotFinite { low, high });
        }
        if low > high {
            return Err(ForecastError::IntervalInverted { low, high });
        }
        Ok(Self { low, high })
    }

    /// A degenerate interval representing a point observation.
    pub fn point(at: f64) -> Result<Self, ForecastError> {
        Self::new(at, at)
    }

    pub fn low(&self) -> f64 {
        self.low
    }

    pub fn high(&self) -> f64 {
        self.high
    }

    /// Representative location, used by CRPS to reduce a branch to a point atom.
    pub fn midpoint(&self) -> f64 {
        // Both bounds are finite by construction, so this cannot overflow to inf for any
        // realistic magnitudes and cannot be NaN.
        (self.low + self.high) / 2.0
    }

    /// Whether two intervals share any point. Closed-interval semantics: `[0,1]` and `[1,2]`
    /// overlap at `1`.
    pub fn overlaps(&self, other: &Self) -> bool {
        self.low <= other.high && other.low <= self.high
    }

    /// Whether `x` lies in this closed interval.
    ///
    /// This is what makes interval scoring a scoring rule rather than a lookup. Because
    /// [`ForecastDistribution::try_new`] rejects overlapping intervals, at most one branch of a
    /// validated distribution can contain any given `x` — so summing matched mass cannot
    /// double-count. That non-overlap gate was merely defensive under the old exact-equality
    /// matching; it became load-bearing when `probability_of` moved to containment on 2026-07-31.
    pub fn contains(&self, x: f64) -> bool {
        self.low <= x && x <= self.high
    }
}

impl TryFrom<IntervalRepr> for Interval {
    type Error = ForecastError;
    fn try_from(r: IntervalRepr) -> Result<Self, Self::Error> {
        Self::new(r.low, r.high)
    }
}

impl From<Interval> for IntervalRepr {
    fn from(i: Interval) -> IntervalRepr {
        IntervalRepr {
            low: i.low,
            high: i.high,
        }
    }
}

/// One branch of a forecast: probability mass assigned to a region of the outcome space, with
/// the assumptions it depends on recorded alongside it.
///
/// Fields are public because every one of them is already a validated type — there is no way to
/// put an invalid value in a `ForecastBranch` once [`Probability`] and [`Interval`] enforce
/// their own invariants.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastBranch {
    pub probability: Probability,
    pub outcome: crate::OutcomeRegion,
    pub assumptions: Vec<AssumptionId>,
}

impl ForecastBranch {
    /// Convenience constructor taking a raw probability, validated on the way in.
    pub fn new(
        probability: f64,
        outcome: crate::OutcomeRegion,
        assumptions: Vec<AssumptionId>,
    ) -> Result<Self, ForecastError> {
        Ok(Self {
            probability: Probability::new(probability)?,
            outcome,
            assumptions,
        })
    }
}

/// Deserialization shadow of [`ForecastDistribution`]. Exists so the `Deserialize` path is
/// forced through [`ForecastDistribution::try_new`] rather than reconstructing private fields
/// directly — otherwise a recorded-artifact round-trip would be a hole straight through the
/// constructor.
#[derive(Deserialize)]
struct ForecastDistributionRepr {
    issued_at_tick: u64,
    horizon: Horizon,
    outcome_space: OutcomeSpaceId,
    branches: Vec<ForecastBranch>,
    unsupported_mass: Probability,
}

/// The canonical forecast representation — what gets scored and ledgered.
///
/// Construct with [`ForecastDistribution::try_new`]. Fields are private; read them through the
/// accessors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "ForecastDistributionRepr")]
pub struct ForecastDistribution {
    issued_at_tick: u64,
    horizon: Horizon,
    outcome_space: OutcomeSpaceId,
    branches: Vec<ForecastBranch>,
    unsupported_mass: Probability,
}

impl ForecastDistribution {
    /// Build a validated forecast. Enforces, in order: non-emptiness, per-branch validity
    /// (already guaranteed by the branch types), region distinctness, interval
    /// non-overlap, and the total-mass invariant.
    pub fn try_new(
        issued_at_tick: u64,
        horizon: Horizon,
        outcome_space: OutcomeSpaceId,
        branches: Vec<ForecastBranch>,
        unsupported_mass: Probability,
    ) -> Result<Self, ForecastError> {
        if branches.is_empty() {
            return Err(ForecastError::EmptyDistribution);
        }

        // Distinctness for exact-match regions, non-overlap for intervals. Both matter because
        // `probability_of` sums every branch whose region matches, so a non-partition silently
        // changes what a score means.
        for (i, a) in branches.iter().enumerate() {
            for b in &branches[i + 1..] {
                match (&a.outcome, &b.outcome) {
                    (crate::OutcomeRegion::Interval(x), crate::OutcomeRegion::Interval(y)) => {
                        if x.overlaps(y) {
                            return Err(ForecastError::OverlappingIntervals {
                                first: (x.low(), x.high()),
                                second: (y.low(), y.high()),
                            });
                        }
                    }
                    (x, y) if x == y => return Err(ForecastError::DuplicateOutcomeRegion),
                    _ => {}
                }
            }
        }

        let total: f64 =
            branches.iter().map(|b| b.probability.get()).sum::<f64>() + unsupported_mass.get();
        if (total - 1.0).abs() > MASS_TOLERANCE {
            return Err(ForecastError::MassNotNormalized {
                total,
                tolerance: MASS_TOLERANCE,
            });
        }

        Ok(Self {
            issued_at_tick,
            horizon,
            outcome_space,
            branches,
            unsupported_mass,
        })
    }

    /// Ergonomic form of [`try_new`](Self::try_new) taking raw floats, for call sites that build
    /// a forecast from computed values and would otherwise wrap each one individually.
    pub fn try_from_raw(
        issued_at_tick: u64,
        horizon: Horizon,
        outcome_space: OutcomeSpaceId,
        branches: Vec<(f64, crate::OutcomeRegion, Vec<AssumptionId>)>,
        unsupported_mass: f64,
    ) -> Result<Self, ForecastError> {
        let branches = branches
            .into_iter()
            .map(|(p, o, a)| ForecastBranch::new(p, o, a))
            .collect::<Result<Vec<_>, _>>()?;
        Self::try_new(
            issued_at_tick,
            horizon,
            outcome_space,
            branches,
            Probability::new(unsupported_mass)?,
        )
    }

    pub fn issued_at_tick(&self) -> u64 {
        self.issued_at_tick
    }

    pub fn horizon(&self) -> Horizon {
        self.horizon
    }

    pub fn outcome_space(&self) -> &OutcomeSpaceId {
        &self.outcome_space
    }

    pub fn branches(&self) -> &[ForecastBranch] {
        &self.branches
    }

    /// Probability mass NOT assigned to any enumerated branch. A generator that always reports
    /// `0.0` here is claiming full coverage of the outcome space — that claim should be
    /// scrutinized, not assumed true by construction.
    pub fn unsupported_mass(&self) -> Probability {
        self.unsupported_mass
    }
}

impl TryFrom<ForecastDistributionRepr> for ForecastDistribution {
    type Error = ForecastError;
    fn try_from(r: ForecastDistributionRepr) -> Result<Self, Self::Error> {
        Self::try_new(
            r.issued_at_tick,
            r.horizon,
            r.outcome_space,
            r.branches,
            r.unsupported_mass,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OutcomeRegion;

    fn space() -> OutcomeSpaceId {
        OutcomeSpaceId("extinct_within_horizon".into())
    }

    fn build(
        branches: Vec<(f64, OutcomeRegion)>,
        unsupported: f64,
    ) -> Result<ForecastDistribution, ForecastError> {
        ForecastDistribution::try_from_raw(
            0,
            Horizon(10),
            space(),
            branches.into_iter().map(|(p, o)| (p, o, vec![])).collect(),
            unsupported,
        )
    }

    // --- Every adversarial case from the pre-fix reproduction is now a typed error. ---
    // Pre-fix, each of these constructed successfully and produced a *scoreable* number;
    // three of them scored better than an honest forecast. See this module's docs.

    #[test]
    fn rejects_negative_probability() {
        let e = build(vec![(-5.0, OutcomeRegion::Boolean(true))], 0.0).unwrap_err();
        assert_eq!(e, ForecastError::ProbabilityOutOfRange { value: -5.0 });
    }

    #[test]
    fn rejects_probability_above_one() {
        let e = build(vec![(1.8, OutcomeRegion::Boolean(true))], 0.0).unwrap_err();
        assert_eq!(e, ForecastError::ProbabilityOutOfRange { value: 1.8 });
    }

    #[test]
    fn rejects_nan_probability() {
        let e = build(vec![(f64::NAN, OutcomeRegion::Boolean(true))], 0.0).unwrap_err();
        assert!(matches!(e, ForecastError::ProbabilityNotFinite { .. }));
    }

    #[test]
    fn rejects_infinite_probability() {
        // Pre-fix this scored -inf on log score: the theoretical optimum, beating every
        // honest forecaster outright.
        let e = build(vec![(f64::INFINITY, OutcomeRegion::Boolean(true))], 0.0).unwrap_err();
        assert!(matches!(e, ForecastError::ProbabilityNotFinite { .. }));
    }

    #[test]
    fn rejects_negative_unsupported_mass() {
        // Pre-fix this was silently ignored by the `> 0.0` guard in BrierScore — no penalty,
        // no error, mass simply vanished.
        let e = build(vec![(0.5, OutcomeRegion::Boolean(true))], -0.5).unwrap_err();
        assert_eq!(e, ForecastError::ProbabilityOutOfRange { value: -0.5 });
    }

    #[test]
    fn rejects_empty_distribution() {
        let e = build(vec![], 0.0).unwrap_err();
        assert_eq!(e, ForecastError::EmptyDistribution);
    }

    #[test]
    fn rejects_inverted_interval() {
        // Pre-fix this scored a *perfect* 0.0 Brier, by matching itself under exact equality.
        let e = Interval::new(100.0, 0.0).unwrap_err();
        assert_eq!(
            e,
            ForecastError::IntervalInverted {
                low: 100.0,
                high: 0.0
            }
        );
    }

    #[test]
    fn rejects_nonfinite_interval_bounds() {
        assert!(matches!(
            Interval::new(0.0, f64::INFINITY).unwrap_err(),
            ForecastError::IntervalBoundNotFinite { .. }
        ));
        assert!(matches!(
            Interval::new(f64::NAN, 1.0).unwrap_err(),
            ForecastError::IntervalBoundNotFinite { .. }
        ));
    }

    #[test]
    fn rejects_unnormalized_mass() {
        let e = build(
            vec![
                (0.9, OutcomeRegion::Boolean(true)),
                (0.9, OutcomeRegion::Boolean(false)),
            ],
            0.5,
        )
        .unwrap_err();
        match e {
            ForecastError::MassNotNormalized { total, tolerance } => {
                assert!((total - 2.3).abs() < 1e-12, "got {total}");
                assert_eq!(tolerance, MASS_TOLERANCE);
            }
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn rejects_duplicate_outcome_region() {
        let e = build(
            vec![
                (0.5, OutcomeRegion::Boolean(true)),
                (0.5, OutcomeRegion::Boolean(true)),
            ],
            0.0,
        )
        .unwrap_err();
        assert_eq!(e, ForecastError::DuplicateOutcomeRegion);
    }

    #[test]
    fn rejects_overlapping_intervals() {
        let e = build(
            vec![
                (0.5, OutcomeRegion::interval(0.0, 10.0).unwrap()),
                (0.5, OutcomeRegion::interval(5.0, 15.0).unwrap()),
            ],
            0.0,
        )
        .unwrap_err();
        assert!(matches!(e, ForecastError::OverlappingIntervals { .. }));
    }

    // --- Valid forecasts still construct, unchanged. ---

    #[test]
    fn accepts_honest_forecast() {
        let f = build(
            vec![
                (0.7, OutcomeRegion::Boolean(true)),
                (0.3, OutcomeRegion::Boolean(false)),
            ],
            0.0,
        )
        .expect("honest forecast must construct");
        assert_eq!(f.branches().len(), 2);
        assert_eq!(f.unsupported_mass().get(), 0.0);
        assert_eq!(f.horizon(), Horizon(10));
    }

    #[test]
    fn accepts_forecast_with_declared_unsupported_mass() {
        let f = build(vec![(0.6, OutcomeRegion::Boolean(true))], 0.4)
            .expect("partial coverage is legitimate and must construct");
        assert_eq!(f.unsupported_mass().get(), 0.4);
    }

    #[test]
    fn accepts_adjacent_nonoverlapping_intervals() {
        build(
            vec![
                (0.5, OutcomeRegion::interval(0.0, 5.0).unwrap()),
                (0.5, OutcomeRegion::interval(5.000001, 10.0).unwrap()),
            ],
            0.0,
        )
        .expect("disjoint intervals are a legitimate partition");
    }

    // --- Deserialization must not bypass the constructor. ---

    #[test]
    fn deserialization_rejects_what_construction_rejects() {
        // Hand-written JSON matching the pre-fix wire format exactly, with p > 1.
        let json = r#"{
            "issued_at_tick": 0,
            "horizon": 10,
            "outcome_space": "s",
            "branches": [{"probability": 1.8, "outcome": {"Boolean": true}, "assumptions": []}],
            "unsupported_mass": 0.0
        }"#;
        assert!(
            serde_json::from_str::<ForecastDistribution>(json).is_err(),
            "serde must not be a hole through the constructor"
        );
    }

    #[test]
    fn deserialization_rejects_unnormalized_mass() {
        let json = r#"{
            "issued_at_tick": 0,
            "horizon": 10,
            "outcome_space": "s",
            "branches": [{"probability": 0.2, "outcome": {"Boolean": true}, "assumptions": []}],
            "unsupported_mass": 0.1
        }"#;
        assert!(serde_json::from_str::<ForecastDistribution>(json).is_err());
    }

    #[test]
    fn deserialization_rejects_inverted_interval() {
        let json = r#"{"Interval":{"low":100.0,"high":0.0}}"#;
        assert!(serde_json::from_str::<OutcomeRegion>(json).is_err());
    }

    #[test]
    fn valid_forecast_round_trips_and_wire_format_is_unchanged() {
        let f = build(
            vec![
                (0.5, OutcomeRegion::interval(0.0, 5.0).unwrap()),
                (0.5, OutcomeRegion::Boolean(true)),
            ],
            0.0,
        )
        .unwrap();
        let json = serde_json::to_string(&f).unwrap();
        // The interval must still serialize with named low/high fields, so artifacts
        // recorded before this change still deserialize.
        assert!(
            json.contains(r#"{"Interval":{"low":0.0,"high":5.0}}"#),
            "wire format changed: {json}"
        );
        let back: ForecastDistribution = serde_json::from_str(&json).unwrap();
        assert_eq!(back, f);
    }
}
