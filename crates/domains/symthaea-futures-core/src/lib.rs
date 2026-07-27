// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-futures-core
//!
//! Neutral forecast representation for the Symthaea Futures Laboratory. See
//! `SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md` at the repository root for the full plan;
//! this crate implements only the Phase 0 skeleton it describes.
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
//! not placeholders. What's *not* here yet: any code that produces or consumes them. That's
//! `symthaea-futures-state` (belief-state estimation), `symthaea-futures-ensemble` (trajectory
//! generation), and `symthaea-futures-calibration` (scoring), all Phase 1 work.

use serde::{Deserialize, Serialize};

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
    Interval { low: f64, high: f64 },
    Discrete(String),
}

/// Identifies an assumption a forecast branch depends on (e.g. "observation policy version",
/// "model X's generative assumptions"), so the evidence ledger can trace *why* a branch exists,
/// not just what probability it was assigned.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AssumptionId(pub String);

/// How many ticks ahead a forecast reaches. A plain newtype, not a duration type — Symtropy-side
/// scenarios (see `symthaea-futures-symtropy`) are tick-indexed, not wall-clock-indexed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Horizon(pub u64);

/// One branch of a forecast: a probability mass assigned to a region of the outcome space,
/// with the assumptions it depends on recorded alongside it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastBranch {
    pub probability: f64,
    pub outcome: OutcomeRegion,
    pub assumptions: Vec<AssumptionId>,
}

/// The canonical forecast representation. This is what gets scored and ledgered — natural
/// language explanations, if any exist downstream, are generated *from* this struct, never
/// the other way around (see the plan's "Forecasts are typed objects, not narration" section).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastDistribution {
    pub issued_at_tick: u64,
    pub horizon: Horizon,
    pub outcome_space: OutcomeSpaceId,
    pub branches: Vec<ForecastBranch>,
    /// Probability mass NOT assigned to any enumerated branch. Forces "we didn't cover this"
    /// to be a visible number rather than silently redistributed across branches that were
    /// thought of. A generator that always reports `0.0` here is claiming full coverage of the
    /// outcome space — that claim should be scrutinized, not assumed true by construction.
    pub unsupported_mass: f64,
}

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

/// Implemented by every predictor competing in the lab — the naive persistence baseline, the
/// FEP-driven ensemble, the oracle upper bound, and (Phase 3B, gated) any quantum lane. Generic
/// over `Observation` so this crate never needs to know what a scenario family's observation
/// type looks like; `symthaea-futures-symtropy` owns that (see its `ObservationPolicy`).
pub trait TrajectoryGenerator {
    type Observation;

    fn generate(&self, observation: &Self::Observation, horizon: Horizon) -> ForecastOutput;
}
