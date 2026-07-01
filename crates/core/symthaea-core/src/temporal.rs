// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared temporal prediction trait for O(1) CfC-based forecasting.
//!
//! All genesis crates (fusion, hydrology, materials, nuclear, multiscale)
//! share the same prediction pattern: given a current HDC state and a time
//! horizon, produce a predicted state using closed-form CfC evolution.
//!
//! This trait unifies that pattern so cross-domain bridges can work with
//! any predictor generically.

use crate::hdc::unified_hv::ContinuousHV;

/// A temporal predictor that can forecast HDC state at any time horizon with O(1) cost.
///
/// Implementors wrap a CfC neuron and use `evolve_closed_form()` for prediction.
/// The key invariant is that prediction cost is independent of the time horizon.
pub trait TemporalPredictor {
    /// Predict state at the given time horizon (seconds). O(1) cost.
    ///
    /// # Arguments
    /// * `current_state` — The current HDC-encoded state.
    /// * `horizon_seconds` — Time horizon in seconds (must be finite and positive).
    fn predict_at(&self, current_state: &ContinuousHV, horizon_seconds: f32) -> ContinuousHV;

    /// Feed an observation to update internal state.
    fn observe(&mut self, state: &ContinuousHV, dt_seconds: f32);

    /// Domain name for this predictor (e.g., "plasma", "hydrology", "materials").
    fn domain(&self) -> &'static str;

    /// Characteristic timescale (tau_base) in seconds.
    fn tau_base(&self) -> f32;

    /// Standard prediction horizons for this domain (seconds).
    ///
    /// Returns the domain-specific horizon array (e.g., `PLASMA_HORIZONS`,
    /// `AGING_HORIZONS`). Defaults to empty for predictors without fixed horizons.
    fn default_horizons(&self) -> &'static [f32] {
        &[]
    }

    /// Human-readable labels for [`Self::default_horizons`], in matching order.
    ///
    /// Returns labels like `["1 ms", "10 ms", "100 ms", "1 s", "10 s"]`.
    /// Defaults to empty.
    fn horizon_labels(&self) -> &'static [&'static str] {
        &[]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verify the trait is object-safe (can be used as dyn TemporalPredictor)
    fn _assert_object_safe(_: &dyn TemporalPredictor) {}
}
