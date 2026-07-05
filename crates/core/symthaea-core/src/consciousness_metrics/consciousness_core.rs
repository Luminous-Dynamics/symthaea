// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Consciousness Core — shared Φ measurement, cadence-disciplined
//!
//! Wraps [`SpectralMIPFinder`] — the same O(n³) Fiedler-ordering IIT Φ
//! approximation that drives `MotorSafetyLevel::from_phi()` across every
//! robotics platform in the cognitive loop — with the push/compute cadence
//! discipline that makes it cheap: the loop doesn't call `compute()` every
//! cycle, it pushes every 2 cycles and computes every 47 (see
//! `cognitive_loop/consciousness_engine/measure.rs`). `ConsciousnessCore`
//! generalizes that same discipline to any caller with its own tick/call
//! cadence, so callers outside the cognitive loop (e.g. the `Symthaea::process()`
//! facade) can compute Φ the same validated way instead of inventing their own
//! ad hoc formula.
//!
//! ## Why separate instances, not a shared one
//!
//! The cognitive loop and the facade have fundamentally different cadences —
//! the loop pushes from a continuous ~31Hz tick stream, the facade pushes once
//! per user-facing `process()` call, which may be seconds or minutes apart.
//! Feeding both into one shared sliding window would mix two unrelated
//! sampling rates into a single covariance estimate, which is not
//! well-defined. Each caller should own its own `ConsciousnessCore` instance;
//! what's shared is the *type* and the *algorithm*, not the *state*.

use super::spectral_mip::{SpectralMIPConfig, SpectralMIPFinder};
use crate::hdc::unified_hv::ContinuousHV;

/// Cadence configuration for [`ConsciousnessCore`].
#[derive(Debug, Clone)]
pub struct ConsciousnessCoreConfig {
    /// Underlying spectral MIP window/computation parameters.
    pub spectral_mip: SpectralMIPConfig,
    /// Push a sample into the window only every `push_every` calls to
    /// [`ConsciousnessCore::measure`]. `1` pushes every call.
    pub push_every: u64,
    /// Recompute Φ only every `compute_every` calls to
    /// [`ConsciousnessCore::measure`] (the expensive O(n³) step). Between
    /// computes, `measure()` returns the last cached value. `1` computes
    /// every call.
    pub compute_every: u64,
}

impl Default for ConsciousnessCoreConfig {
    /// Tuned for low-frequency, per-request callers (e.g. a conversational
    /// facade), not the cognitive loop's own high-frequency tick cadence.
    /// A conversational window rarely has dozens of turns, so `window_size`/
    /// `min_samples` are reduced from `SpectralMIPConfig::default()` (50/10)
    /// so Φ becomes available within a handful of turns rather than never.
    fn default() -> Self {
        Self {
            spectral_mip: SpectralMIPConfig {
                window_size: 20,
                min_samples: 5,
                ..SpectralMIPConfig::default()
            },
            push_every: 1,
            compute_every: 1,
        }
    }
}

/// Shared Φ measurement core: [`SpectralMIPFinder`] plus cadence discipline.
///
/// Call [`measure`](Self::measure) once per caller-defined "tick" (a cognitive
/// loop cycle, a conversational turn, whatever the caller's own unit of time
/// is). Returns `None` until the window has accumulated `min_samples` pushes.
#[derive(Debug, Clone)]
pub struct ConsciousnessCore {
    finder: SpectralMIPFinder,
    config: ConsciousnessCoreConfig,
    calls: u64,
    last_phi: Option<f64>,
}

impl ConsciousnessCore {
    /// Construct with explicit cadence configuration.
    pub fn new(config: ConsciousnessCoreConfig) -> Self {
        Self {
            finder: SpectralMIPFinder::new(config.spectral_mip.clone()),
            config,
            calls: 0,
            last_phi: None,
        }
    }

    /// Construct with defaults tuned for a low-frequency, per-request caller.
    pub fn with_defaults() -> Self {
        Self::new(ConsciousnessCoreConfig::default())
    }

    /// Push `state` (on cadence) and recompute Φ (on cadence). Returns the
    /// current Φ — freshly computed this call, or cached from a previous
    /// compute — or `None` if the window hasn't reached `min_samples` yet.
    pub fn measure(&mut self, state: &ContinuousHV) -> Option<f64> {
        self.calls += 1;

        if self.calls % self.config.push_every.max(1) == 0 {
            self.finder.push(state);
        }

        if self.calls % self.config.compute_every.max(1) == 0
            && let Some(result) = self.finder.compute()
        {
            self.last_phi = Some(result.phi);
        }

        self.last_phi
    }

    /// Number of samples currently in the sliding window.
    pub fn window_len(&self) -> usize {
        self.finder.window_len()
    }

    /// Reset all accumulated state (window, cadence counter, cached Φ).
    pub fn reset(&mut self) {
        self.finder.reset();
        self.calls = 0;
        self.last_phi = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hv(seed: u64) -> ContinuousHV {
        ContinuousHV::random(128, seed)
    }

    #[test]
    fn returns_none_before_min_samples() {
        let mut core = ConsciousnessCore::with_defaults();
        for i in 0..4 {
            assert_eq!(core.measure(&hv(i)), None);
        }
    }

    #[test]
    fn returns_some_after_min_samples() {
        let mut core = ConsciousnessCore::with_defaults();
        let mut last = None;
        for i in 0..10 {
            last = core.measure(&hv(i));
        }
        assert!(last.is_some());
        assert!(last.unwrap().is_finite());
    }

    #[test]
    fn respects_compute_cadence() {
        let mut core = ConsciousnessCore::new(ConsciousnessCoreConfig {
            spectral_mip: SpectralMIPConfig {
                window_size: 20,
                min_samples: 5,
                ..SpectralMIPConfig::default()
            },
            push_every: 1,
            compute_every: 3,
        });
        // Calls 1,2 don't compute (not multiples of 3); nothing pushed enough yet either.
        assert_eq!(core.measure(&hv(0)), None);
        assert_eq!(core.measure(&hv(1)), None);
        // Call 3 computes but window only has 3 samples (< min_samples=5) -> still None.
        assert_eq!(core.measure(&hv(2)), None);
        for i in 3..12 {
            core.measure(&hv(i));
        }
        // By now min_samples is satisfied and a compute cadence has fired.
        assert!(core.window_len() >= 5);
    }

    #[test]
    fn reset_clears_state() {
        let mut core = ConsciousnessCore::with_defaults();
        for i in 0..10 {
            core.measure(&hv(i));
        }
        assert!(core.window_len() > 0);
        core.reset();
        assert_eq!(core.window_len(), 0);
        assert_eq!(core.measure(&hv(999)), None);
    }
}
