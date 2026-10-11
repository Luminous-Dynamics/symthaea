// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-futures-core
//!
//! Neutral forecast representation for the Symthaea Futures Laboratory. See
//! `SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md` at the repository root for the full plan;
//! this crate implements the neutral forecast boundary it describes, with validation enforced.
//!
//! ## Why this crate has (and must keep) zero non-std, non-serde dependencies
//!
//! Every predictor — FEP-driven (`symthaea-futures-state`), naive statistical baselines
//! (`symthaea-futures-ensemble`), and later a gated quantum lane (Phase 3B) — implements
//! [`TrajectoryGenerator`] and competes on equal terms. If this crate depended on
//! `symthaea-fep` or any specific backend, that backend would quietly become privileged
//! infrastructure rather than one candidate among several, which is the exact failure mode
//! the plan's Phase 0 section calls out. Do not add a dependency here without updating the
//! plan doc's reasoning for why the boundary moved.
//!
//! ## What's real here vs. what's still Phase 1 work
//!
//! The types below are the canonical, scoreable forecast representation the plan specifies —
//! not placeholders. Phase 1 has since landed: `symthaea-futures-state` (belief-state
//! estimation), `symthaea-futures-ensemble` (trajectory generation),
//! `symthaea-futures-calibration` (scoring), `-ledger` and `-analysis` all produce or consume
//! these types today. (This paragraph previously said no such code existed yet; it was stale in
//! exactly the direction that let the validation gap described in [`validated`] go unnoticed.)
//!
//! ## Validity is enforced, not assumed
//!
//! [`ForecastDistribution`], [`ForecastPayload`], [`ForecastBranch`], [`Probability`] and
//! [`Interval`] have private storage and validated constructors. `ForecastDistribution` retains
//! the existing tick-indexed simulation contract; `ForecastPayload` contains only the validated
//! probability/outcome surface so external or calendar-time scenario families do not need fake
//! ticks merely to participate in scoring and evidence recording.

use serde::{Deserialize, Serialize};

pub mod payload;
pub mod validated;
pub use payload::ForecastPayload;
pub use validated::{
    ForecastBranch, ForecastDistribution, ForecastError, Interval, MASS_TOLERANCE, Probability,
};

/// Identifies which outcome space a forecast is defined over (e.g. "population extinct within
/// horizon: bool" vs. "time-to-extinction: continuous"). Opaque on purpose — `futures-core`
/// doesn't know what outcome spaces exist; scenario-family crates define and register their own.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OutcomeSpaceId(pub String);

/// A region of an outcome space a branch's probability mass is assigned to (e.g. a boolean
/// value, a numeric interval, a discrete class). Deliberately unopinionated at this layer —
/// see the plan's "First experiment" section for the two concrete outcome spaces Phase 1 uses
/// (extinction-within-horizon: bool; time-to-extinction: interval).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OutcomeRegion {
    Boolean(bool),
    /// A validated closed interval. Holds an [`Interval`] rather than bare `low`/`high` fields
    /// so an inverted or nonfinite range cannot be constructed at all — build one with
    /// [`OutcomeRegion::interval`] or [`Interval::new`]. Wire format is unchanged
    /// (`{"Interval":{"low":..,"high":..}}`), so previously recorded artifacts still deserialize
    /// — and are now validated on the way in.
    Interval(Interval),
    Discrete(String),
}

impl OutcomeRegion {
    /// Build an `Interval` region from raw bounds, validating them on the way in.
    pub fn interval(low: f64, high: f64) -> Result<Self, ForecastError> {
        Ok(Self::Interval(Interval::new(low, high)?))
    }

    /// Build a degenerate `Interval` region representing a point observation.
    pub fn point(at: f64) -> Result<Self, ForecastError> {
        Ok(Self::Interval(Interval::point(at)?))
    }
}

/// Identifies an assumption a forecast branch depends on (e.g. "observation policy version",
/// "model X's generative assumptions"), so the evidence ledger can trace *why* a branch exists,
/// not just what probability it was assigned.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AssumptionId(pub String);

/// How many ticks ahead a legacy simulation forecast reaches. The neutral v2 ledger does not
/// require this type; calendar/external scenario families bind time semantics separately from
/// [`ForecastPayload`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Horizon(pub u64);

/// Why a generator declined to produce a forecast instead of guessing. Abstention is a
/// first-class output, not an error case — see the plan's "Abstention, built in from the
/// start" deliverable. A generator that never returns this, or returns it uniformly at random
/// with respect to how it would otherwise have scored, has produced a null result for this
/// mechanism specifically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AbstentionReason {
    InsufficientObservationHistory,
    OutOfDistributionScenario,
    ModelDisagreementTooHigh,
    HorizonBeyondValidatedRange,
    UnresolvedOutcomeSpace,
    ObservationPolicyTooLossy,
}

/// A generator's output: either a scoreable forecast, or a typed refusal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ForecastOutput {
    Distribution(ForecastDistribution),
    Abstain(AbstentionReason),
}

/// Implemented by every predictor competing in the legacy tick-indexed lab — the naive
/// persistence baseline, the FEP-driven ensemble, the oracle upper bound, and (Phase 3B, gated)
/// any quantum lane. Generic over `Observation` so this crate never needs to know what a
/// scenario family's observation type looks like; `symthaea-futures-symtropy` owns that (see its
/// `ObservationPolicy`). Calendar/external adapters can emit [`ForecastPayload`] without being
/// forced through this tick-specific trait.
pub trait TrajectoryGenerator {
    type Observation;

    fn generate(&self, observation: &Self::Observation, horizon: Horizon) -> ForecastOutput;
}
