// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Ion-channel reversal potentials, derived from the real Nernst equation
//! (`symthaea_core::physics::biophysics::BiophysicsEncoder::nernst_potential_mv`)
//! rather than the hand-picked constants (`VMEM_DEPOLARIZED`/
//! `VMEM_HYPERPOLARIZED`) [`crate::bioelectric`] uses by default.
//!
//! This backs the opt-in ion-channel conductance model in
//! [`crate::bioelectric`] (`NeuralOrganoid::set_ion_channel_model_enabled`):
//! rather than every cell relaxing at one fixed rate toward one of two
//! hardcoded target voltages, cells instead carry per-cell K+ (hyperpolarizing)
//! and Na+-like (depolarizing) channel *conductances* — the biologically real,
//! cell-to-cell-varying quantity (Levin-lab findings are about differential
//! channel *expression* during differentiation, not per-cell ion
//! *concentration* gradients) — and Vmem settles toward a conductance-weighted
//! average of each channel's reversal potential, exactly the Goldman-style
//! logic real membranes follow.
//!
//! Concentrations are treated as shared, literature-typical bath/cytoplasm
//! values (not per-cell state) at physiological temperature (310K / 37°C) —
//! the same K+ values `biophysics.rs`'s own Nernst test uses.

use symthaea_core::physics::BiophysicsEncoder;

/// Valence of both channel species modeled here (monovalent cations).
const ION_VALENCE: i32 = 1;
/// Physiological temperature (37°C), matching `biophysics.rs`'s own test.
const PHYSIOLOGICAL_TEMP_K: f64 = 310.0;

/// K+ leak channel concentrations (mM) — extracellular/intracellular.
/// Textbook values; identical to `biophysics.rs`'s own Nernst test, which
/// verifies these produce ≈ -90 mV.
const K_CONC_OUT_MM: f64 = 5.0;
const K_CONC_IN_MM: f64 = 140.0;

/// Na+-like depolarizing channel concentrations (mM) — extracellular/
/// intracellular. Textbook values (e.g. Kandel et al.).
const NA_CONC_OUT_MM: f64 = 145.0;
const NA_CONC_IN_MM: f64 = 12.0;

/// Converts real millivolts into this crate's normalized Vmem convention
/// (`VMEM_HYPERPOLARIZED = -1.0` .. `VMEM_DEPOLARIZED = 0.0`, with headroom
/// above 0.0 already established by `VMEM_WOUND_SPIKE = 0.6`). Chosen so the
/// K+ reversal potential (≈ -89 mV) lands at ≈ -1.0, matching
/// `VMEM_HYPERPOLARIZED` almost exactly — the Na+-like reversal potential
/// then lands at ≈ +0.74, a modest, intentional overshoot past
/// `VMEM_DEPOLARIZED` consistent with the existing wound-spike precedent.
/// This keeps the ion-channel model's Vmem range compatible with
/// `TargetMorphology`'s existing `vmem_span` normalization without any
/// changes there.
const MV_TO_NORMALIZED_SCALE: f32 = 90.0;

/// Real, Nernst-derived reversal potentials for the two channel species this
/// model uses, in this crate's normalized Vmem units: `(e_k_norm, e_na_norm)`.
pub(crate) fn reversal_potentials_normalized() -> (f32, f32) {
    let e_k_mv = BiophysicsEncoder::nernst_potential_mv(
        ION_VALENCE,
        K_CONC_OUT_MM,
        K_CONC_IN_MM,
        PHYSIOLOGICAL_TEMP_K,
    );
    let e_na_mv = BiophysicsEncoder::nernst_potential_mv(
        ION_VALENCE,
        NA_CONC_OUT_MM,
        NA_CONC_IN_MM,
        PHYSIOLOGICAL_TEMP_K,
    );
    (
        (e_k_mv as f32) / MV_TO_NORMALIZED_SCALE,
        (e_na_mv as f32) / MV_TO_NORMALIZED_SCALE,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reversal_potentials_are_physiologically_ordered() {
        let (e_k, e_na) = reversal_potentials_normalized();
        assert!(
            e_k < e_na,
            "K+ reversal potential should be hyperpolarizing relative to Na+'s depolarizing one, got e_k={e_k} e_na={e_na}"
        );
    }

    #[test]
    fn reversal_potentials_reuse_nernst_formula() {
        // Hand-computed from the same formula biophysics.rs's own test
        // validates (E ~= -90mV for these K+ concentrations at 310K):
        // this cross-checks that the rescaling here is applied correctly,
        // not that the underlying Nernst math is correct (that's covered by
        // `symthaea-core`'s own `test_nernst_potential`).
        let (e_k, _) = reversal_potentials_normalized();
        let expected_e_k_mv = BiophysicsEncoder::nernst_potential_mv(1, 5.0, 140.0, 310.0);
        let expected_e_k_norm = (expected_e_k_mv as f32) / MV_TO_NORMALIZED_SCALE;
        assert!(
            (e_k - expected_e_k_norm).abs() < 1e-6,
            "expected {expected_e_k_norm}, got {e_k}"
        );
    }

    #[test]
    fn k_reversal_potential_lands_near_hyperpolarized_constant() {
        let (e_k, _) = reversal_potentials_normalized();
        // Should land close to VMEM_HYPERPOLARIZED (-1.0) by construction of
        // MV_TO_NORMALIZED_SCALE -- not exactly, since the scale constant is
        // fixed and the Nernst output isn't exactly -90mV.
        assert!(
            (e_k - (-1.0)).abs() < 0.2,
            "expected e_k near -1.0, got {e_k}"
        );
    }
}
