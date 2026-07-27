// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-futures-state
//!
//! Observation → belief-state estimation for the Symthaea Futures Laboratory
//! (`SYMTHAEA_FUTURES_LABORATORY_PLAN_2026-07-25.md`).
//!
//! ## Reuse spike — DONE, verdict: reuse, with two real (not blocking) gaps
//!
//! Per the plan's "Do not duplicate `symthaea-fep`" section, this had to be resolved before
//! writing any real belief-state code. Read `agent.rs`, `types.rs`, `generative_model.rs`, and
//! `markov_blanket.rs` in `symthaea-fep`, plus `symthaea-alife::Organism` (the one existing
//! consumer that already drives `ActiveInferenceAgent` from a real simulation loop).
//!
//! **Verdict: reuse `ActiveInferenceAgent`/[`HiddenState`] directly as the belief-state
//! estimator. Do not build a parallel stack.** `perceive()` is a real, working variational
//! belief update (gradient descent on free energy, precision-weighted), and its posterior
//! (`HiddenState`) is exactly the belief-state type `symthaea-futures-core` needs — see
//! [`BeliefState`] below, a type alias, not a hand-designed struct, per the plan's own framing.
//!
//! **Gap 1 — `Observation` has no missing-data / partial-visibility representation.**
//! `symthaea_fep::types::Observation::values` is a fixed-width dense `Vec<f64>`; `obs_dim` is
//! fixed at `ActiveInferenceAgent` construction and `GenerativeModel::prediction_error` just
//! `zip()`s it against the predicted vector — there is no per-dimension mask. This means the
//! Futures Laboratory's observation-firewall crate (`symthaea-futures-symtropy`) cannot hand
//! FEP a "some dimensions withheld" vector natively; something has to encode "not observed" as
//! an actual `f64` value before it reaches `perceive()`. That encoding is real new work, not
//! decorative glue — see [`mask_observation`] below for the primitive that does it.
//!
//! **Where the masking primitive came from — and why it's not `MarkovBoundaryOperator`.**
//! `symthaea-fep::markov_blanket::MarkovBoundaryOperator::gate_observation` looked like an
//! off-the-shelf answer: it already lerps a raw observation toward a prior/belief mean by a
//! permeability scalar `p`, and at `p = 0` the output is *exactly* the belief mean —
//! independent of the true observed value, which is the property a leakage-safe mask needs.
//! But its `p` is computed from the *organism's own physiological deficit* (energy Set-point
//! gap → boundary thickness) — a trust/noise model for "how much do I let environmental signal
//! move my beliefs," not a scenario-designer's declared visibility policy. Reusing it wholesale
//! would conflate "sensor is noisy" with "ground truth is deliberately withheld," and — worse
//! for the leakage-test requirement — its `p` is never guaranteed to sit exactly at `0.0` for a
//! field a scenario author intends to hide; it drifts with organism state. [`mask_observation`]
//! below reuses only the *lerp formula* (`markov_blanket.rs:1030`, reimplemented here rather
//! than made `pub` upstream to avoid coupling this crate to organism-physiology internals it
//! doesn't want), driven instead by a `visibility: &[f64]` vector the `ObservationPolicy`
//! supplies directly — `1.0` fully visible, `0.0` fully withheld (leakage-test-safe by
//! construction: output at that index is a pure function of the current belief, never of the
//! raw value), fractional values reserved for a future noised-but-visible case.
//!
//! **Gap 2 (lower priority, not blocking Phase 1) — no native irregular-interval handling.**
//! `Observation.timestamp` is accepted but never read by `perceive()`; `ActiveInferenceAgent`
//! advances its own internal `timestamp` as a plain per-call counter. Symtropy-family scenarios
//! are already integer-tick-indexed and regularly sampled at the policy layer, so this doesn't
//! block Phase 1's first experiment — flagged here so a future scenario family that genuinely
//! needs true irregular real-time sampling doesn't assume this is already handled.
//!
//! **Dead-code finding, worth knowing before Phase 1 leans on it**: `Observation.precision` and
//! `.modality` are constructed by every existing caller (including `Organism`) but never read by
//! any FEP computation — `FreeEnergyCalculator::compute_accuracy` uses `GenerativeModel::
//! observation_precision` (one fixed scalar on the model), not the per-observation field. If
//! Phase 1 wants per-dimension observation confidence to actually affect inference (e.g. a
//! "noised but visible" reading, not fully hidden), that's a third real gap — most naturally
//! implemented as a fractional entry in [`mask_observation`]'s `visibility` vector, not by
//! setting `Observation.precision` and expecting it to do anything.

pub use symthaea_fep::types::Observation;
pub use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig, HiddenState};

/// The Futures Laboratory's belief-state type. A type alias, deliberately not a wrapper struct —
/// the reuse spike's whole point is that `ActiveInferenceAgent`'s own posterior already *is*
/// the belief state; adding a wrapper here would be exactly the "second belief-state stack"
/// the plan says not to build.
pub type BeliefState = HiddenState;

/// The leakage-safe masking primitive `symthaea-futures-symtropy`'s `ObservationPolicy`
/// implementations should use to turn "here's what's visible" into an `Observation` FEP can
/// consume. `visibility[i] == 0.0` makes index `i` of the output a pure function of
/// `current_belief.mean[i]` — never of `raw.values[i]` — which is exactly the property the
/// plan's leakage-test fixture (feed a deliberately wrong hidden value, assert byte-identical
/// output) is designed to catch a violation of.
///
/// `visibility` shorter than `raw.values` treats missing trailing entries as `0.0` (fully
/// withheld) rather than panicking — an `ObservationPolicy` that forgets to declare visibility
/// for a trailing dimension fails safe (hides it) rather than leaking it.
pub fn mask_observation(
    raw: &Observation,
    current_belief: &BeliefState,
    visibility: &[f64],
) -> Observation {
    let masked_values: Vec<f64> = raw
        .values
        .iter()
        .enumerate()
        .map(|(i, &raw_val)| {
            let belief_val = current_belief.mean.get(i).copied().unwrap_or(0.5);
            let p = visibility.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
            lerp(belief_val, raw_val, p)
        })
        .collect();

    Observation {
        values: masked_values,
        precision: raw.precision,
        timestamp: raw.timestamp,
        modality: raw.modality.clone(),
    }
}

/// Linear interpolation: a + t × (b - a). Reimplemented locally (see module docs) rather than
/// depending on `symthaea-fep::markov_blanket`'s private helper of the same name.
#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fully_withheld_dimension_is_independent_of_raw_value() {
        let belief = BeliefState::new(2);
        let visibility = [1.0, 0.0]; // dim 0 visible, dim 1 withheld

        let raw_a = Observation::new(vec![0.9, 0.1], 1.0, "test");
        let raw_b = Observation::new(vec![0.9, 0.99], 1.0, "test"); // only the hidden dim differs

        let masked_a = mask_observation(&raw_a, &belief, &visibility);
        let masked_b = mask_observation(&raw_b, &belief, &visibility);

        // The masking primitive is expected to make the withheld dimension byte-identical
        // regardless of the raw value behind it — the same property the plan's real leakage
        // test (feeding a deliberately wrong ground-truth value) checks end-to-end once a real
        // ObservationPolicy exists.
        assert_eq!(masked_a.values[1], masked_b.values[1]);
        assert_eq!(masked_a.values[1], belief.mean[1]);
        // The visible dimension still passes the raw value through untouched at full visibility.
        assert_eq!(masked_a.values[0], raw_a.values[0]);
    }

    #[test]
    fn missing_visibility_entries_fail_safe_to_hidden() {
        let belief = BeliefState::new(3);
        let visibility = [1.0]; // no entry for dims 1, 2

        let raw = Observation::new(vec![0.7, 0.7, 0.7], 1.0, "test");
        let masked = mask_observation(&raw, &belief, &visibility);

        assert_eq!(masked.values[0], raw.values[0]);
        assert_eq!(masked.values[1], belief.mean[1]);
        assert_eq!(masked.values[2], belief.mean[2]);
    }
}
