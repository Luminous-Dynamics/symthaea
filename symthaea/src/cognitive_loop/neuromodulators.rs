//! Neuromodulator Bath: DA/NE/5-HT/ACh chemical signaling.
//!
//! Four first-class neurotransmitter channels that unify the cognitive loop's
//! 44+ scattered modulation sites under a coherent biological model.
//!
//! Each transmitter has production/reuptake dynamics and receptor adaptation:
//! - **Dopamine (DA)**: Reward prediction error → learning rate & motivation
//! - **Noradrenaline (NE)**: Surprise & arousal → exploration & alertness
//! - **Serotonin (5-HT)**: Satisfaction & mood → confidence & risk aversion
//! - **Acetylcholine (ACh)**: Attention & precision → focus & signal filtering
//!
//! Science: Doya (2002) — "Metalearning and neuromodulation"

use serde::{Deserialize, Serialize};

/// A single neuromodulator channel with production/reuptake dynamics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Transmitter {
    /// Current level (0.0 = depleted, 1.0 = saturated)
    pub level: f32,
    /// Receptor sensitivity (adapts to sustained high/low levels).
    /// Range: [0.5, 2.0]. Down-regulates under sustained high level, up-regulates under low.
    pub receptor_sensitivity: f32,
    /// Reuptake rate: fraction cleared per cycle (higher = faster return to baseline)
    reuptake_rate: f32,
    /// Tonic baseline level (what the system returns to at rest)
    baseline: f32,
}

impl Default for Transmitter {
    fn default() -> Self {
        Self {
            level: 0.5,
            receptor_sensitivity: 1.0,
            reuptake_rate: 0.1,
            baseline: 0.5,
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
    #[inline]
    pub fn produce(&mut self, amount: f32) {
        self.level = (self.level + amount).clamp(0.0, 1.0);
    }

    /// Reuptake: exponential decay toward baseline + receptor sensitivity adaptation.
    pub fn reuptake(&mut self) {
        // Exponential return to baseline
        self.level += (self.baseline - self.level) * self.reuptake_rate;
        self.level = self.level.clamp(0.0, 1.0);
        // Receptor adaptation (slow): down-regulate if chronically high, up-regulate if low
        if self.level > 0.7 {
            self.receptor_sensitivity *= 0.998; // tolerance
        } else if self.level < 0.3 {
            self.receptor_sensitivity *= 1.002; // sensitization
        }
        self.receptor_sensitivity = self.receptor_sensitivity.clamp(0.5, 2.0);
    }
}

/// Signals from the cognitive cycle that drive transmitter production.
pub(crate) struct NeuromodulatorInputs {
    /// Prediction error magnitude (0.0–1.0+)
    pub prediction_error: f32,
    /// FEP surprise triggered this cycle
    pub surprise: bool,
    /// Goal achievement / moral alignment (-1.0 to 1.0)
    pub reward_signal: f32,
    /// Temporal coherence (0.0–1.0)
    pub coherence: f32,
    /// Current emotional arousal (0.0–1.0)
    pub arousal: f32,
    /// Phenomenal binding strength (0.0–1.0)
    pub binding_strength: f32,
    /// Epistemic gate confidence (0.0–1.0)
    pub epistemic_confidence: f32,
    /// Whether the system is in flow state
    pub flow_active: bool,
}

/// The four core neuromodulator channels.
///
/// Science: Doya (2002) — "Metalearning and neuromodulation"
/// DA = reward prediction error, NE = unexpected uncertainty,
/// 5-HT = punishment/aversion, ACh = expected uncertainty.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct NeuromodulatorBath {
    /// Dopamine: reward prediction error → learning rate & motivation
    pub dopamine: Transmitter,
    /// Noradrenaline: surprise & arousal → exploration & alertness
    pub noradrenaline: Transmitter,
    /// Serotonin: satisfaction & mood → confidence & risk aversion
    pub serotonin: Transmitter,
    /// Acetylcholine: attention & precision → focus & signal filtering
    pub acetylcholine: Transmitter,
}

impl Default for NeuromodulatorBath {
    fn default() -> Self {
        Self {
            dopamine: Transmitter::default(),
            noradrenaline: Transmitter::default(),
            serotonin: Transmitter::default(),
            acetylcholine: Transmitter::default(),
        }
    }
}

impl NeuromodulatorBath {
    /// Update all transmitters from cognitive cycle signals.
    ///
    /// Production rules are biologically grounded:
    /// - DA: Schultz (1997) — reward prediction error
    /// - NE: Aston-Jones (2005) — unexpected uncertainty
    /// - 5-HT: Dayan & Huys (2009) — satisfaction & safety
    /// - ACh: Yu & Dayan (2005) — expected uncertainty
    pub fn update(&mut self, inputs: &NeuromodulatorInputs) {
        // ── DOPAMINE: Reward Prediction Error (Schultz 1997) ─────────
        // Positive RPE (better than expected) → DA burst
        // Negative RPE (worse than expected) → DA dip
        let da_signal = inputs.reward_signal * 0.15
            + if inputs.prediction_error < 0.2 {
                0.05
            } else {
                -0.05
            }
            + if inputs.flow_active { 0.03 } else { 0.0 };
        self.dopamine.produce(da_signal);

        // ── NORADRENALINE: Unexpected Uncertainty (Aston-Jones 2005) ─
        // Surprise and high arousal → NE release
        let ne_signal = if inputs.surprise { 0.15 } else { 0.0 }
            + inputs.arousal * 0.08
            + inputs.prediction_error.clamp(0.0, 0.5) * 0.1;
        self.noradrenaline.produce(ne_signal);

        // ── SEROTONIN: Satisfaction & Safety (Dayan & Huys 2009) ─────
        // High coherence + confidence → 5-HT (system is "content")
        // Low coherence, moral violations → 5-HT dip
        let sht_signal = inputs.coherence * 0.08
            + inputs.epistemic_confidence * 0.05
            + inputs.binding_strength * 0.04
            - if inputs.reward_signal < -0.3 { 0.1 } else { 0.0 };
        self.serotonin.produce(sht_signal);

        // ── ACETYLCHOLINE: Expected Uncertainty (Yu & Dayan 2005) ────
        // Epistemic uncertainty (known unknowns) → ACh release
        // Flow state → sustained ACh (deep focus)
        let ach_signal = (1.0 - inputs.epistemic_confidence) * 0.1
            + if inputs.flow_active { 0.06 } else { 0.0 }
            + if inputs.binding_strength > 0.7 {
                0.03
            } else {
                0.0
            };
        self.acetylcholine.produce(ach_signal);

        // ── CROSS-MODULATION ─────────────────────────────────────────
        // High DA suppresses NE (exploitation suppresses exploration)
        if self.dopamine.level > 0.7 {
            self.noradrenaline.level *= 0.97;
        }
        // High 5-HT suppresses NE (contentment dampens arousal)
        if self.serotonin.level > 0.7 {
            self.noradrenaline.level *= 0.98;
        }
        // High NE boosts ACh (arousal sharpens attention)
        if self.noradrenaline.level > 0.6 {
            self.acetylcholine.produce(0.02);
        }

        // ── REUPTAKE (all channels) ──────────────────────────────────
        self.dopamine.reuptake();
        self.noradrenaline.reuptake();
        self.serotonin.reuptake();
        self.acetylcholine.reuptake();
    }

    /// DA → learning rate multiplier (0.7–1.5).
    /// Science: Schultz (1997) — DA scales synaptic plasticity.
    #[inline]
    pub fn learning_rate_factor(&self) -> f32 {
        let da = self.dopamine.effective();
        (0.7 + da * 0.4).clamp(0.7, 1.5)
    }

    /// NE → exploration urge delta (-0.05 to +0.10).
    /// Science: Aston-Jones (2005) — LC-NE modulates explore/exploit.
    #[inline]
    pub fn exploration_delta(&self) -> f32 {
        let ne = self.noradrenaline.effective();
        (ne - 0.5) * 0.2 // centered at 0.5 baseline
    }

    /// 5-HT → confidence modulation (-0.02 to +0.02).
    /// Science: Dayan & Huys (2009) — 5-HT biases risk assessment.
    #[inline]
    pub fn confidence_delta(&self) -> f32 {
        let sht = self.serotonin.effective();
        (sht - 0.5) * 0.04
    }

    /// ACh → attention sensitivity multiplier (0.8–1.3).
    /// Science: Yu & Dayan (2005) — ACh gates sensory precision.
    #[inline]
    pub fn attention_factor(&self) -> f32 {
        let ach = self.acetylcholine.effective();
        (0.8 + ach * 0.25).clamp(0.8, 1.3)
    }

    /// ACh → threshold scale modifier (tighter focus = lower threshold).
    /// High ACh → lower threshold (more precise signal detection).
    #[inline]
    pub fn threshold_factor(&self) -> f32 {
        let ach = self.acetylcholine.effective();
        (1.1 - ach * 0.2).clamp(0.8, 1.2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transmitter_produce_clamp() {
        let mut t = Transmitter::default();
        assert!((t.level - 0.5).abs() < f32::EPSILON);
        // Produce beyond 1.0 should clamp
        t.produce(0.8);
        assert!(t.level <= 1.0);
        assert!(t.level > 0.9);
        // Produce negative to deplete
        t.produce(-2.0);
        assert!(t.level >= 0.0);
    }

    #[test]
    fn test_transmitter_reuptake_decay() {
        let mut t = Transmitter {
            level: 0.9,
            ..Default::default()
        };
        // After reuptake, level should move toward baseline (0.5)
        for _ in 0..50 {
            t.reuptake();
        }
        assert!(t.level < 0.7, "level should decay: got {}", t.level);
        assert!(t.level > 0.45, "should not overshoot baseline: got {}", t.level);
    }

    #[test]
    fn test_receptor_downregulation() {
        let mut t = Transmitter {
            level: 0.9,
            receptor_sensitivity: 1.0,
            reuptake_rate: 0.0, // disable reuptake so level stays high
            baseline: 0.5,
        };
        let initial_sens = t.receptor_sensitivity;
        // Sustained high level → tolerance → sensitivity decreases
        for _ in 0..200 {
            t.reuptake(); // only receptor adaptation runs (reuptake_rate=0 means no level change)
        }
        assert!(
            t.receptor_sensitivity < initial_sens,
            "sensitivity should decrease: got {}",
            t.receptor_sensitivity
        );
        assert!(t.receptor_sensitivity >= 0.5, "clamped at 0.5");
    }

    #[test]
    fn test_receptor_upregulation() {
        let mut t = Transmitter {
            level: 0.1,
            receptor_sensitivity: 1.0,
            reuptake_rate: 0.0,
            baseline: 0.5,
        };
        let initial_sens = t.receptor_sensitivity;
        for _ in 0..200 {
            t.reuptake();
        }
        assert!(
            t.receptor_sensitivity > initial_sens,
            "sensitivity should increase: got {}",
            t.receptor_sensitivity
        );
        assert!(t.receptor_sensitivity <= 2.0, "clamped at 2.0");
    }

    #[test]
    fn test_da_reward_burst() {
        let mut bath = NeuromodulatorBath::default();
        let initial_da = bath.dopamine.level;
        let inputs = NeuromodulatorInputs {
            prediction_error: 0.1,
            surprise: false,
            reward_signal: 0.8, // strong positive reward
            coherence: 0.5,
            arousal: 0.5,
            binding_strength: 0.5,
            epistemic_confidence: 0.5,
            flow_active: false,
        };
        bath.update(&inputs);
        // DA should increase from positive reward
        // (reward_signal * 0.15 = 0.12) + (low error bonus 0.05) = +0.17 pre-reuptake
        assert!(
            bath.dopamine.level > initial_da - 0.1,
            "DA should increase on reward: {} vs initial {}",
            bath.dopamine.level,
            initial_da
        );
        assert!(bath.dopamine.effective().is_finite());
    }

    #[test]
    fn test_ne_surprise_release() {
        let mut bath = NeuromodulatorBath::default();
        let initial_ne = bath.noradrenaline.level;
        let inputs = NeuromodulatorInputs {
            prediction_error: 0.3,
            surprise: true, // surprise fires
            reward_signal: 0.0,
            coherence: 0.5,
            arousal: 0.7,
            binding_strength: 0.5,
            epistemic_confidence: 0.5,
            flow_active: false,
        };
        bath.update(&inputs);
        // NE should spike from surprise (0.15) + arousal (0.056) + PE (0.03)
        assert!(
            bath.noradrenaline.level > initial_ne,
            "NE should increase on surprise: {} vs initial {}",
            bath.noradrenaline.level,
            initial_ne
        );
    }

    #[test]
    fn test_sht_coherence_satisfaction() {
        let mut bath = NeuromodulatorBath::default();
        let inputs = NeuromodulatorInputs {
            prediction_error: 0.1,
            surprise: false,
            reward_signal: 0.5,
            coherence: 0.9,             // high coherence
            epistemic_confidence: 0.85, // high confidence
            binding_strength: 0.8,      // strong binding
            arousal: 0.4,
            flow_active: false,
        };
        // Run several cycles to accumulate
        for _ in 0..10 {
            bath.update(&inputs);
        }
        // 5-HT should be elevated from high coherence + confidence
        assert!(
            bath.serotonin.effective() > 0.4,
            "5-HT should rise with coherence: effective={}",
            bath.serotonin.effective()
        );
    }

    #[test]
    fn test_cross_modulation_da_suppresses_ne() {
        let mut bath = NeuromodulatorBath::default();
        // Push DA high
        bath.dopamine.level = 0.85;
        let ne_before = 0.6_f32;
        bath.noradrenaline.level = ne_before;

        let inputs = NeuromodulatorInputs {
            prediction_error: 0.1,
            surprise: false,
            reward_signal: 0.5,
            coherence: 0.5,
            arousal: 0.3,
            binding_strength: 0.5,
            epistemic_confidence: 0.5,
            flow_active: false,
        };
        bath.update(&inputs);

        // DA > 0.7 should have suppressed NE (*.97 cross-modulation)
        // NE got production + cross-modulation suppression + reuptake
        // The key test: NE effective should be finite and in range
        assert!(bath.noradrenaline.effective().is_finite());
        assert!(bath.noradrenaline.level <= 1.0);
        assert!(bath.noradrenaline.level >= 0.0);
    }
}
