// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use serde::{Deserialize, Serialize};

/// Circadian phase for modulating transmitter baselines.
///
/// Duplicated from `symthaea::chronobiology::CircadianPhase` to avoid
/// cross-crate coupling. The main crate re-export provides `From` conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CircadianPhase {
    Dawn,
    Day,
    Dusk,
    Night,
}

/// A single neuromodulator channel with production/reuptake dynamics.
///
/// Each transmitter tracks both a tonic level (slow baseline) and a phasic burst
/// component (fast-decaying transient). The distinction enables downstream consumers
/// to differentiate RPE bursts (phasic DA) from sustained motivational tone (tonic DA).
///
/// Science: Grace (1991) — phasic DA bursts encode RPE; tonic DA sets motivational tone.
/// Aston-Jones & Cohen (2005) — LC-NE phasic/tonic modes govern exploit/explore.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transmitter {
    /// Current level (0.0 = depleted, 1.0 = saturated)
    pub level: f32,
    /// Receptor sensitivity (adapts to sustained high/low levels).
    /// Range: [0.5, 2.0]. Down-regulates under sustained high level, up-regulates under low.
    pub receptor_sensitivity: f32,
    /// Legacy linear clearance rate. **This field no longer drives clearance.**
    ///
    /// Clearance is now Michaelis-Menten — see `mm_v_max` / `mm_k_m`. `reuptake()`
    /// does not read this field at all, so writing it (including setting it to 0.0)
    /// has **no effect on level dynamics**. It is retained only because
    /// `boost_reuptake()` / `reset_reuptake()` — and therefore
    /// `NeuromodulatorBath::engage_anomaly_recovery()` — still write it, and because
    /// it is part of the serialized form.
    ///
    /// To hold a level constant in a test, re-assert `level` each cycle (model-agnostic)
    /// or set `mm_v_max: 0.0`. Setting `reuptake_rate: 0.0` will silently do nothing.
    pub(crate) reuptake_rate: f32,
    /// Tonic baseline level (what the system returns to at rest)
    pub(crate) baseline: f32,
    /// Fast-decaying burst component (0.0–1.0). Tracks recent production spikes.
    /// Decays at `phasic_decay` rate per cycle (~5-cycle half-life at 0.3).
    pub(crate) phasic: f32,
    /// Phasic decay rate per cycle (default 0.3 → ~5-cycle half-life via ×0.7).
    pub(crate) phasic_decay: f32,
    /// Consecutive cycles with level > baseline+0.2 (fast tachyphylaxis tracking).
    /// Science: Gainetdinov et al. (2004) — rapid GPCR desensitization.
    #[serde(default)]
    pub high_exposure_cycles: u32,
    /// Remaining rebound sensitization cycles after withdrawal from sustained high.
    #[serde(default)]
    pub withdrawal_cycles: u32,
    // ── Per-transmitter tolerance/withdrawal curves (Koob & Le Moal 2001) ──
    /// Cycles of high exposure before tolerance onset (default 20).
    #[serde(default = "default_tolerance_onset")]
    pub tolerance_onset_cycles: u32,
    /// Receptor sensitivity decay rate during tolerance (default 0.99).
    #[serde(default = "default_tolerance_decay")]
    pub tolerance_decay_rate: f32,
    /// Withdrawal rebound duration in cycles (default 30).
    #[serde(default = "default_withdrawal_duration")]
    pub withdrawal_duration: u32,
    /// Withdrawal sensitization rate (default 1.01).
    #[serde(default = "default_withdrawal_recovery")]
    pub withdrawal_recovery_rate: f32,
    /// High-exposure threshold offset above baseline (default 0.2).
    #[serde(default = "default_tolerance_threshold")]
    pub tolerance_threshold: f32,
    /// Michaelis-Menten V_max: maximum clearance rate (default 0.15).
    /// Science: Torres et al. (2003) — monoamine transporter kinetics.
    #[serde(default = "default_mm_v_max")]
    pub mm_v_max: f32,
    /// Michaelis-Menten K_m: half-saturation constant (default 0.4).
    /// Science: Torres et al. (2003) — DAT K_m ≈ 0.2μM normalized.
    #[serde(default = "default_mm_k_m")]
    pub mm_k_m: f32,
}

fn default_tolerance_onset() -> u32 {
    20
}
fn default_tolerance_decay() -> f32 {
    0.99
}
fn default_withdrawal_duration() -> u32 {
    30
}
fn default_withdrawal_recovery() -> f32 {
    1.01
}
fn default_tolerance_threshold() -> f32 {
    0.2
}
fn default_mm_v_max() -> f32 {
    0.15
}
fn default_mm_k_m() -> f32 {
    0.4
}

impl Default for Transmitter {
    fn default() -> Self {
        Self {
            level: 0.5,
            receptor_sensitivity: 1.0,
            reuptake_rate: 0.1,
            baseline: 0.5,
            phasic: 0.0,
            phasic_decay: 0.3,
            high_exposure_cycles: 0,
            withdrawal_cycles: 0,
            tolerance_onset_cycles: 20,
            tolerance_decay_rate: 0.99,
            withdrawal_duration: 30,
            withdrawal_recovery_rate: 1.01,
            tolerance_threshold: 0.2,
            mm_v_max: 0.15,
            mm_k_m: 0.4,
        }
    }
}

impl Transmitter {
    /// Effective signal = level * receptor_sensitivity (what downstream reads).
    #[inline]
    pub fn effective(&self) -> f32 {
        (self.level * self.receptor_sensitivity).clamp(0.0, 2.0)
    }

    /// Produce: add to level from input signal.
    /// Also tracks phasic burst magnitude (positive production only).
    #[inline]
    pub fn produce(&mut self, amount: f32) {
        self.level = (self.level + amount).clamp(0.0, 1.0);
        // Track phasic burst from positive production (negative = dip, not a burst)
        if amount > 0.0 {
            self.phasic = (self.phasic + amount).clamp(0.0, 1.0);
        }
    }

    /// Current phasic burst magnitude (fast-decaying transient signal).
    #[inline]
    pub fn phasic(&self) -> f32 {
        self.phasic
    }

    /// Set the tonic baseline level (clamped to [0.2, 0.8]).
    /// Used by circadian modulation to shift the resting point.
    #[inline]
    pub fn set_baseline(&mut self, baseline: f32) {
        self.baseline = baseline.clamp(0.2, 0.8);
    }

    /// Read-only access to baseline (non-test, for allostatic load and other callers).
    #[inline]
    pub fn baseline_val(&self) -> f32 {
        self.baseline
    }

    /// Read-only access to baseline for testing.
    #[cfg(test)]
    pub fn baseline_for_test(&self) -> f32 {
        self.baseline
    }

    /// Reuptake: exponential decay toward baseline + receptor sensitivity adaptation.
    ///
    /// Receptor adaptation uses baseline-relative thresholds (±0.2) so that
    /// circadian baseline shifts (e.g. Night NE=0.30) don't cause spurious
    /// sensitization/tolerance when the level simply hovers near its new baseline.
    ///
    /// Phasic component decays fast (×(1-phasic_decay) per cycle, ~5-cycle half-life).
    pub fn reuptake(&mut self) {
        // Michaelis-Menten clearance toward baseline (Torres et al. 2003).
        // At low deviation this is ~linear with slope mm_v_max/mm_k_m (0.375 at
        // defaults). That is NOT the old `reuptake_rate` default of 0.1 — clearance
        // near baseline is ~3.75x faster than the pre-MM model, so this is not
        // backward-compatible with it. `reuptake_rate` is not read here at all.
        // At high deviation: saturates at V_max (transporters fully occupied).
        let delta = self.level - self.baseline;
        let abs_delta = delta.abs();
        if abs_delta > 1e-6 {
            let clearance = self.mm_v_max * abs_delta / (self.mm_k_m + abs_delta);
            if delta > 0.0 {
                self.level -= clearance;
            } else {
                self.level += clearance;
            }
        }
        self.level = self.level.clamp(0.0, 1.0);
        // Fast phasic decay: Grace (1991) — burst signals are transient
        self.phasic *= 1.0 - self.phasic_decay;
        // Receptor adaptation (slow): baseline-relative thresholds
        // Gated: only active before tolerance onset. Once slow tachyphylaxis
        // (receptor internalization) takes over, fast GPCR phosphorylation
        // is superseded. Prevents 0.998 × 0.985 double-decay.
        // Science: Gainetdinov et al. (2004) — GPCR phosphorylation precedes internalization.
        let high = self.baseline + 0.2;
        let low = self.baseline - 0.2;
        if self.high_exposure_cycles <= self.tolerance_onset_cycles {
            if self.level > high {
                self.receptor_sensitivity *= 0.998; // fast adaptation (pre-tolerance)
            } else if self.level < low {
                self.receptor_sensitivity *= 1.002; // sensitization
            }
        }
        // ── Slow tachyphylaxis (Koob & Le Moal 2001) ─────────────────────
        // Receptor internalization under sustained high exposure.
        // Per-transmitter curves: allostatic addiction model.
        let high_thresh = self.baseline + self.tolerance_threshold;
        if self.level > high_thresh {
            self.high_exposure_cycles = self.high_exposure_cycles.saturating_add(1);
            if self.high_exposure_cycles > self.tolerance_onset_cycles {
                self.receptor_sensitivity *= self.tolerance_decay_rate;
            }
        } else {
            if self.high_exposure_cycles > self.tolerance_onset_cycles && self.level < self.baseline
            {
                self.withdrawal_cycles = self.withdrawal_duration;
            }
            self.high_exposure_cycles = 0;
        }
        if self.withdrawal_cycles > 0 {
            self.receptor_sensitivity *= self.withdrawal_recovery_rate;
            self.withdrawal_cycles -= 1;
        }
        self.receptor_sensitivity = self.receptor_sensitivity.clamp(0.5, 2.0);
    }

    /// Whether receptor is in tolerance state (sustained high exposure > onset threshold).
    /// Science: Gainetdinov et al. (2004) — GPCR desensitization.
    #[inline]
    pub fn is_tolerant(&self) -> bool {
        self.high_exposure_cycles > self.tolerance_onset_cycles
    }

    /// Whether transmitter is in withdrawal rebound (sensitization after high-exposure drop).
    #[inline]
    pub fn is_in_withdrawal(&self) -> bool {
        self.withdrawal_cycles > 0
    }

    /// Boost reuptake rate by factor (clamped 0.01–0.5).
    /// Science: Turrigiano (2008) — homeostatic plasticity adjusts clearance.
    ///
    /// ⚠️ **Currently inert.** This writes `reuptake_rate`, which `reuptake()` no
    /// longer reads (clearance is Michaelis-Menten). Calling this does not change
    /// how fast the level actually returns to baseline. To make homeostatic
    /// recovery real again it would need to scale `mm_v_max` instead — a behaviour
    /// change that has not been made here.
    pub fn boost_reuptake(&mut self, factor: f32) {
        self.reuptake_rate = (self.reuptake_rate * factor).clamp(0.01, 0.5);
    }

    /// Reset reuptake rate to default (0.1).
    ///
    /// ⚠️ **Currently inert** — see [`Transmitter::boost_reuptake`].
    pub fn reset_reuptake(&mut self) {
        self.reuptake_rate = 0.1;
    }

    /// Adjust tonic baseline by delta within custom bounds.
    /// Science: Schultz (2016) — sustained prediction errors shift DA tonic level.
    pub fn adjust_baseline(&mut self, delta: f32, lo: f32, hi: f32) {
        self.baseline = (self.baseline + delta).clamp(lo, hi);
    }

    /// Read-only access to reuptake rate for testing.
    #[cfg(test)]
    pub fn reuptake_rate_for_test(&self) -> f32 {
        self.reuptake_rate
    }
}
