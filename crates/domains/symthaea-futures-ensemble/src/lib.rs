// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-futures-ensemble
//!
//! The baseline hierarchy for the Symthaea Futures Laboratory
//! (`SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md`, Phase 1 deliverables).
//!
//! No [`symthaea_futures_core::TrajectoryGenerator`] implementation is trusted to mean
//! anything in isolation — each rung below only means something once compared against the
//! rungs before it, and bounded above by the oracle. [`BaselineRung`] exists so an evidence
//! record can say *which* rung produced a given forecast without ambiguity.
//!
//! ## What's real here vs. what's still open
//!
//! [`ecological`] has all six rungs implemented and real-run against the ecological-collapse
//! scenario family (see that module's docs and `SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md`
//! for the design decisions and disclosed findings behind each one — in particular rung 5's
//! "perceive-only teaches nothing" trap and rung 6's corrected design-gap retraction).
//! [`predator_prey`] has rungs 1-6 implemented against the second scenario family (predator
//! extinction forecasting).

use symthaea_futures_core::{ForecastDistribution, Horizon, OutcomeRegion, OutcomeSpaceId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BaselineRung {
    /// "Whatever's true now continues."
    Persistence,
    /// Base rate across all training seeds; ignores the specific observation entirely.
    HistoricalFrequency,
    /// Linear or autoregressive fit on the observed series.
    SimpleStatistical,
    /// A simplified closed-form model of the scenario family (a real equation, not a fit).
    ScenarioMechanistic,
    /// The FEP-driven ensemble — the actual system under test.
    FepDriven,
    /// Linear/OLS fit on an evolving-trait signal alone (ignoring population count entirely).
    /// Added for the evolutionary-rescue family's Phase 2.2B rung hierarchy — the plan's own
    /// "trait-trend statistical predictor," distinct from [`Self::SimpleStatistical`]'s
    /// population-count trend.
    TraitTrend,
    /// The FEP-driven ensemble reading population count *and* a trait signal jointly (2D belief
    /// state) — the evolutionary-rescue family's "trait-augmented model" the Phase 2.2B
    /// acceptance gate compares against [`Self::FepDriven`] run census-only.
    FepCensusPlusTrait,
    /// Full `SymtropyGroundTruth` access, evaluation-only. Isolates whether weak performance
    /// at rung 5 comes from partial observability (rungs 1–5 close the gap to this as more is
    /// observed) or from the model itself being weak (rung 5 doesn't approach this even with
    /// full observation).
    OracleUpperBound,
}

/// Shared by every scenario family's rungs: builds a two-branch `Boolean` forecast for an
/// "X happens within horizon" target. `outcome_space` names the specific target (e.g.
/// `"ecological_extinction_within_horizon"` vs. `"predator_extinction_within_horizon"`) so two
/// families' forecasts are never confusable downstream even though the branch shape is
/// identical.
pub(crate) fn boolean_distribution(
    issued_at_tick: u64,
    horizon: Horizon,
    p_true: f64,
    outcome_space: &str,
) -> ForecastDistribution {
    let p_true = p_true.clamp(0.0, 1.0);
    // Cannot fail by construction: `p_true` is clamped into [0, 1], `1.0 - p_true` is therefore
    // also in [0, 1], the two Boolean regions are distinct, and the masses sum to 1.0 to within
    // one ULP — far inside `MASS_TOLERANCE`. The `.expect` is a hard backstop, not a hope.
    //
    // NOTE for artifact auditing: this clamp is why every forecast this crate has ever emitted
    // is already valid under the new construction rules — the clamp silently absorbed any
    // out-of-range `p_true` before it reached the scorer. That is a reason past Futures results
    // are *likely* unaffected, but it is not a substitute for scanning recorded artifacts.
    ForecastDistribution::try_from_raw(
        issued_at_tick,
        horizon,
        OutcomeSpaceId(outcome_space.to_string()),
        vec![
            (p_true, OutcomeRegion::Boolean(true), Vec::new()),
            (1.0 - p_true, OutcomeRegion::Boolean(false), Vec::new()),
        ],
        0.0,
    )
    .expect("clamped complementary boolean masses are valid by construction")
}

pub mod ecological;
pub mod evolutionary_rescue;
pub mod predator_prey;
