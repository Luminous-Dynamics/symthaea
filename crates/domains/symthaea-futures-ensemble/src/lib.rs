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

use symthaea_futures_core::{
    ForecastBranch, ForecastDistribution, Horizon, OutcomeRegion, OutcomeSpaceId,
};

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
    ForecastDistribution {
        issued_at_tick,
        horizon,
        outcome_space: OutcomeSpaceId(outcome_space.to_string()),
        branches: vec![
            ForecastBranch {
                probability: p_true,
                outcome: OutcomeRegion::Boolean(true),
                assumptions: Vec::new(),
            },
            ForecastBranch {
                probability: 1.0 - p_true,
                outcome: OutcomeRegion::Boolean(false),
                assumptions: Vec::new(),
            },
        ],
        unsupported_mass: 0.0,
    }
}

pub mod ecological;
pub mod predator_prey;
