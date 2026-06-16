// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Neuromodulator Bath — thin re-export from `symthaea-neuromodulators` crate.
//!
//! All core types, structs, and logic live in the extracted sub-crate.
//! This module re-exports everything and adds:
//! - `CircadianPhase` conversion from `crate::chronobiology`
//! - `NeuromodulatorBathExt` trait with `to_hormone_state()` bridge

#[allow(unused_imports)]
pub use symthaea_neuromodulators::*;

// ── CircadianPhase bridge ──────────────────────────────────────────────────
// The sub-crate has its own CircadianPhase enum to avoid coupling.
// This conversion lets callers pass the main crate's CircadianPhase transparently.

impl From<crate::chronobiology::CircadianPhase> for symthaea_neuromodulators::CircadianPhase {
    fn from(phase: crate::chronobiology::CircadianPhase) -> Self {
        match phase {
            crate::chronobiology::CircadianPhase::Dawn => Self::Dawn,
            crate::chronobiology::CircadianPhase::Day => Self::Day,
            crate::chronobiology::CircadianPhase::Dusk => Self::Dusk,
            crate::chronobiology::CircadianPhase::Night => Self::Night,
        }
    }
}

// ── HormoneState bridge (Sapolsky 2004, McEwen 2007) ───────────────────────
// This method depends on crate::physiology::endocrine::HormoneState,
// so it lives here as an extension trait rather than in the sub-crate.

pub(crate) trait NeuromodulatorBathExt {
    /// Export bath state as HormoneState for downstream physiology consumers.
    fn to_hormone_state(&self) -> crate::physiology::endocrine::HormoneState;
}

impl NeuromodulatorBathExt for NeuromodulatorBath {
    fn to_hormone_state(&self) -> crate::physiology::endocrine::HormoneState {
        let ne = self.noradrenaline.effective() as f64;
        let sht = self.serotonin.effective() as f64;
        let da = self.dopamine.effective().min(1.0) as f64;
        // Cortisol: sustained NE + 5-HT depletion (Sapolsky 2004)
        let cortisol = (ne * 0.6 + (1.0 - sht) * 0.4) * 0.8;
        crate::physiology::endocrine::HormoneState {
            dopamine: da,
            norepinephrine: ne,
            serotonin: sht,
            acetylcholine: self.acetylcholine.effective() as f64,
            cortisol: cortisol.clamp(0.0, 1.0),
            oxytocin: self.oxytocin.effective() as f64,
        }
    }
}
