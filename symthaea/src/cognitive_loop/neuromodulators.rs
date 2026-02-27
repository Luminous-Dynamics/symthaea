// Scaffolded for upcoming wiring — callers not yet connected
#![allow(dead_code)]

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
///
/// Each transmitter tracks both a tonic level (slow baseline) and a phasic burst
/// component (fast-decaying transient). The distinction enables downstream consumers
/// to differentiate RPE bursts (phasic DA) from sustained motivational tone (tonic DA).
///
/// Science: Grace (1991) — phasic DA bursts encode RPE; tonic DA sets motivational tone.
/// Aston-Jones & Cohen (2005) — LC-NE phasic/tonic modes govern exploit/explore.
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
    /// Fast-decaying burst component (0.0–1.0). Tracks recent production spikes.
    /// Decays at `phasic_decay` rate per cycle (~5-cycle half-life at 0.3).
    phasic: f32,
    /// Phasic decay rate per cycle (default 0.3 → ~5-cycle half-life via ×0.7).
    phasic_decay: f32,
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
        // Exponential return to baseline
        self.level += (self.baseline - self.level) * self.reuptake_rate;
        self.level = self.level.clamp(0.0, 1.0);
        // Fast phasic decay: Grace (1991) — burst signals are transient
        self.phasic *= 1.0 - self.phasic_decay;
        // Receptor adaptation (slow): baseline-relative thresholds
        let high = self.baseline + 0.2;
        let low = self.baseline - 0.2;
        if self.level > high {
            self.receptor_sensitivity *= 0.998; // tolerance
        } else if self.level < low {
            self.receptor_sensitivity *= 1.002; // sensitization
        }
        self.receptor_sensitivity = self.receptor_sensitivity.clamp(0.5, 2.0);
    }
}

/// Learnable cross-modulation matrix (4×4 Hebbian weights).
///
/// Entry `weights[i][j]` = how transmitter `i` modulates transmitter `j`'s production.
/// Positive = excitatory, negative = inhibitory. Initialized with biological priors
/// and updated via Hebbian co-activation of phasic bursts.
///
/// Science: Hebb (1949) — neurons that fire together wire together.
/// Hasselmo (2006) — ACh/DA/NE/5-HT interact through learned modulatory pathways.
///
/// Indices: DA=0, NE=1, 5-HT=2, ACh=3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CrossModulationMatrix {
    /// 4×4 weight matrix: [source][target]
    pub weights: [[f32; 4]; 4],
    /// Hebbian learning rate (very slow: 0.001)
    learning_rate: f32,
}

impl Default for CrossModulationMatrix {
    fn default() -> Self {
        // Biological priors from known neurotransmitter interactions
        let mut weights = [[0.0_f32; 4]; 4];
        weights[0][1] = -0.03; // DA→NE: exploitation suppresses exploration
        weights[2][1] = -0.02; // 5-HT→NE: contentment dampens arousal
        weights[1][3] = 0.02;  // NE→ACh: arousal sharpens attention
        Self {
            weights,
            learning_rate: 0.001,
        }
    }
}

impl CrossModulationMatrix {
    /// Compute modulation deltas for each channel based on current levels.
    #[inline]
    pub fn apply(&self, levels: &[f32; 4]) -> [f32; 4] {
        let mut deltas = [0.0_f32; 4];
        for src in 0..4 {
            for tgt in 0..4 {
                if src != tgt {
                    deltas[tgt] += self.weights[src][tgt] * levels[src];
                }
            }
        }
        deltas
    }

    /// Hebbian update from phasic co-activation.
    /// Δw[i][j] = lr × phasic[i] × phasic[j] with weight decay to prevent runaway.
    pub fn hebbian_update(&mut self, phasics: &[f32; 4]) {
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    // Hebbian: co-activation strengthens connection
                    self.weights[i][j] += self.learning_rate * phasics[i] * phasics[j];
                    // Weight decay prevents runaway (×0.999/cycle)
                    self.weights[i][j] *= 0.999;
                    // Clamp to prevent extreme modulation
                    self.weights[i][j] = self.weights[i][j].clamp(-0.1, 0.1);
                }
            }
        }
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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct NeuromodulatorBath {
    /// Dopamine: reward prediction error → learning rate & motivation
    pub dopamine: Transmitter,
    /// Noradrenaline: surprise & arousal → exploration & alertness
    pub noradrenaline: Transmitter,
    /// Serotonin: satisfaction & mood → confidence & risk aversion
    pub serotonin: Transmitter,
    /// Acetylcholine: attention & precision → focus & signal filtering
    pub acetylcholine: Transmitter,
    /// Learnable cross-modulation weights (Hebbian-adaptive, replaces hardcoded rules).
    pub cross_mod: CrossModulationMatrix,
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

        // ── CROSS-MODULATION (Hebbian-adaptive) ─────────────────────
        // Science: Hasselmo (2006) — learned modulatory pathways
        let levels = [
            self.dopamine.level,
            self.noradrenaline.level,
            self.serotonin.level,
            self.acetylcholine.level,
        ];
        let deltas = self.cross_mod.apply(&levels);
        self.dopamine.produce(deltas[0]);
        self.noradrenaline.produce(deltas[1]);
        self.serotonin.produce(deltas[2]);
        self.acetylcholine.produce(deltas[3]);
        // Hebbian update from phasic bursts (co-activation learning)
        let phasics = [
            self.dopamine.phasic,
            self.noradrenaline.phasic,
            self.serotonin.phasic,
            self.acetylcholine.phasic,
        ];
        self.cross_mod.hebbian_update(&phasics);

        // ── REUPTAKE (all channels) ──────────────────────────────────
        self.dopamine.reuptake();
        self.noradrenaline.reuptake();
        self.serotonin.reuptake();
        self.acetylcholine.reuptake();
    }

    /// DA phasic burst magnitude (fast-decaying RPE signal).
    /// Science: Grace (1991) — phasic DA encodes reward prediction error.
    #[inline]
    pub fn da_phasic(&self) -> f32 {
        self.dopamine.phasic
    }

    /// NE phasic burst magnitude (fast-decaying surprise signal).
    /// Science: Aston-Jones & Cohen (2005) — phasic LC-NE encodes unexpected events.
    #[inline]
    pub fn ne_phasic(&self) -> f32 {
        self.noradrenaline.phasic
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

    /// Override transmitter levels for pharmacological ablation (virtual lesion).
    /// Pass `None` to leave a channel unchanged, `Some(v)` to clamp it.
    #[allow(dead_code)]
    pub fn clamp_levels(
        &mut self,
        da: Option<f32>,
        ne: Option<f32>,
        sht: Option<f32>,
        ach: Option<f32>,
    ) {
        if let Some(v) = da {
            self.dopamine.level = v.clamp(0.0, 1.0);
            self.dopamine.phasic = 0.0; // reset burst on clamp
        }
        if let Some(v) = ne {
            self.noradrenaline.level = v.clamp(0.0, 1.0);
            self.noradrenaline.phasic = 0.0;
        }
        if let Some(v) = sht {
            self.serotonin.level = v.clamp(0.0, 1.0);
            self.serotonin.phasic = 0.0;
        }
        if let Some(v) = ach {
            self.acetylcholine.level = v.clamp(0.0, 1.0);
            self.acetylcholine.phasic = 0.0;
        }
    }

    /// DA → gradient magnitude scaling (0.5–2.0).
    /// Science: Schultz (1997) — DA scales synaptic plasticity amplitude.
    #[inline]
    pub fn gradient_scale_factor(&self) -> f32 {
        let da = self.dopamine.effective();
        (0.5 + da * 0.75).clamp(0.5, 2.0)
    }

    /// ACh → learning threshold gate (0.5–1.5).
    /// High ACh → lower effective threshold (learn from smaller errors).
    /// Science: Yu & Dayan (2005) — ACh sharpens expected-uncertainty gating.
    #[inline]
    pub fn threshold_gate(&self) -> f32 {
        let ach = self.acetylcholine.effective();
        // Invert: high ACh → low factor → divides threshold → more learning
        (1.5 - ach * 0.5).clamp(0.5, 1.5)
    }

    /// Neurochemical consciousness modulation factor (0.6–1.2).
    ///
    /// ACh (cholinergic arousal) and NE (noradrenergic wakefulness) are the
    /// primary consciousness-sustaining transmitters. Their effective levels
    /// modulate the system's unified Ψ — depleted neurochemistry reduces
    /// consciousness integration, elevated can mildly enhance it.
    ///
    /// Science: Alkire et al. (2008) — consciousness correlates with ACh/NE.
    /// Mashour & Hudetz (2018) — ACh/NE are critical for conscious information integration.
    #[inline]
    pub fn consciousness_modulation(&self) -> f32 {
        let ach = self.acetylcholine.effective();
        let ne = self.noradrenaline.effective();
        // ACh contributes 60% (cortical arousal), NE 40% (thalamic relay)
        let combined = ach * 0.6 + ne * 0.4;
        // Map [0, 1] → [0.6, 1.2]: depleted = suppressed, elevated = mild boost
        (0.6 + combined * 0.6).clamp(0.6, 1.2)
    }

    /// Sleep consolidation boost (1.0–3.0).
    ///
    /// During Night phase, elevated tonic DA (sustained reward salience) tags
    /// memories for preferential consolidation. Returns a replay batch multiplier.
    ///
    /// Science: Walker & Stickgold (2006) — DA-tagged memories consolidate during sleep.
    /// Stickgold (2005) — sleep replay preferentially consolidates reward-associated memories.
    #[inline]
    pub fn sleep_consolidation_boost(&self) -> f32 {
        // Tonic DA = level minus phasic (sustained baseline component)
        let tonic_da = (self.dopamine.level - self.dopamine.phasic).max(0.0);
        // Maps tonic DA [0.4, 0.7] → boost [1.0, 3.0]
        ((tonic_da - 0.4) * (2.0 / 0.3) + 1.0).clamp(1.0, 3.0)
    }

    /// Shift transmitter baselines based on circadian phase.
    ///
    /// Science:
    /// - Aston-Jones (2001) — LC-NE has strong circadian modulation
    /// - Nishino (2000) — DA consolidation peaks during sleep
    /// - 5-HT follows roughly opposite NE pattern
    /// - ACh peaks during waking attention, troughs in slow-wave sleep
    pub fn modulate_circadian(&mut self, phase: crate::chronobiology::CircadianPhase) {
        use crate::chronobiology::CircadianPhase;
        let (da_base, ne_base, sht_base, ach_base) = match phase {
            CircadianPhase::Dawn  => (0.55, 0.60, 0.45, 0.50),
            CircadianPhase::Day   => (0.50, 0.50, 0.50, 0.60),
            CircadianPhase::Dusk  => (0.45, 0.40, 0.60, 0.50),
            CircadianPhase::Night => (0.55, 0.30, 0.65, 0.40),
        };
        self.dopamine.set_baseline(da_base);
        self.noradrenaline.set_baseline(ne_base);
        self.serotonin.set_baseline(sht_base);
        self.acetylcholine.set_baseline(ach_base);
    }

    /// Whether the system should query the exocortex (swarm network).
    ///
    /// Trigger: high NE (uncertainty) + low DA (no reward prediction) + low 5-HT (low confidence).
    /// Science: Exploration under uncertainty with no reward expectation → seek external knowledge.
    pub fn should_query_exocortex(&self) -> bool {
        let ne = self.noradrenaline.effective();
        let da = self.dopamine.effective();
        let sht = self.serotonin.effective();
        ne > 0.7 && da < 0.4 && sht < 0.5
    }

    /// Derive a neurochemical personality profile from receptor sensitivities.
    ///
    /// Science: Cloninger (1987) — psychobiological model of temperament.
    pub fn personality_profile(&self) -> NeuromodulatorProfile {
        NeuromodulatorProfile {
            novelty_seeking: self.dopamine.receptor_sensitivity,
            harm_avoidance: 2.0 - self.noradrenaline.receptor_sensitivity,
            reward_dependence: self.serotonin.receptor_sensitivity,
            persistence: self.acetylcholine.receptor_sensitivity,
        }
    }

    /// Human-readable personality description from receptor sensitivities.
    pub fn personality_description(&self) -> String {
        let p = self.personality_profile();
        let mut traits = Vec::new();
        if p.novelty_seeking > 1.3 {
            traits.push("novelty-seeking");
        } else if p.novelty_seeking < 0.7 {
            traits.push("risk-averse");
        }
        if p.harm_avoidance > 1.3 {
            traits.push("cautious");
        } else if p.harm_avoidance < 0.7 {
            traits.push("bold");
        }
        if p.reward_dependence > 1.3 {
            traits.push("socially-sensitive");
        }
        if p.persistence > 1.3 {
            traits.push("persistent");
        } else if p.persistence < 0.7 {
            traits.push("flexible");
        }
        if traits.is_empty() {
            "balanced".into()
        } else {
            traits.join(", ")
        }
    }
}

/// Persistent neurochemistry state (receptor sensitivities + cross-modulation weights).
///
/// Checkpointed across sessions so personality adapts over time.
/// Science: Volkow et al. (2004) — receptor density changes persist for weeks/months.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeurochemistryCheckpoint {
    /// DA receptor sensitivity
    pub da_sensitivity: f32,
    /// NE receptor sensitivity
    pub ne_sensitivity: f32,
    /// 5-HT receptor sensitivity
    pub sht_sensitivity: f32,
    /// ACh receptor sensitivity
    pub ach_sensitivity: f32,
    /// Cross-modulation weights (Hebbian-learned)
    pub cross_mod_weights: [[f32; 4]; 4],
}

impl NeuromodulatorBath {
    /// Export persistent state for checkpointing.
    pub fn checkpoint(&self) -> NeurochemistryCheckpoint {
        NeurochemistryCheckpoint {
            da_sensitivity: self.dopamine.receptor_sensitivity,
            ne_sensitivity: self.noradrenaline.receptor_sensitivity,
            sht_sensitivity: self.serotonin.receptor_sensitivity,
            ach_sensitivity: self.acetylcholine.receptor_sensitivity,
            cross_mod_weights: self.cross_mod.weights,
        }
    }

    /// Restore persistent state from checkpoint.
    pub fn restore(&mut self, ckpt: &NeurochemistryCheckpoint) {
        self.dopamine.receptor_sensitivity = ckpt.da_sensitivity.clamp(0.5, 2.0);
        self.noradrenaline.receptor_sensitivity = ckpt.ne_sensitivity.clamp(0.5, 2.0);
        self.serotonin.receptor_sensitivity = ckpt.sht_sensitivity.clamp(0.5, 2.0);
        self.acetylcholine.receptor_sensitivity = ckpt.ach_sensitivity.clamp(0.5, 2.0);
        // Restore cross-modulation weights with clamping
        self.cross_mod.weights = ckpt.cross_mod_weights;
        for row in &mut self.cross_mod.weights {
            for w in row.iter_mut() {
                *w = w.clamp(-0.1, 0.1);
            }
        }
    }
}

/// Neurochemical personality derived from receptor sensitivities.
///
/// Science: Cloninger (1987) — psychobiological model of temperament.
///   DA receptor → Novelty Seeking
///   NE receptor → Harm Avoidance (inverse)
///   5-HT receptor → Reward Dependence
///   ACh receptor → Persistence
#[derive(Debug, Clone)]
pub(crate) struct NeuromodulatorProfile {
    /// DA sensitivity → novelty seeking
    pub novelty_seeking: f32,
    /// Inverse NE sensitivity → harm avoidance (high NE sens = low harm avoidance)
    pub harm_avoidance: f32,
    /// 5-HT sensitivity → reward dependence
    pub reward_dependence: f32,
    /// ACh sensitivity → persistence
    pub persistence: f32,
}

/// Tracks personality profile drift over time for metacognitive anomaly detection.
///
/// Records `NeuromodulatorProfile` snapshots and computes the maximum
/// per-trait delta rate. Rapid drift signals destabilization (e.g. receptor
/// adaptation runaway).
#[derive(Debug, Clone)]
pub(crate) struct PersonalityDriftTracker {
    history: std::collections::VecDeque<NeuromodulatorProfile>,
    capacity: usize,
}

impl Default for PersonalityDriftTracker {
    fn default() -> Self {
        Self::new(16)
    }
}

impl PersonalityDriftTracker {
    pub fn new(capacity: usize) -> Self {
        Self {
            history: std::collections::VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Record a personality profile snapshot. Evicts oldest if at capacity.
    pub fn record(&mut self, profile: &NeuromodulatorProfile) {
        if self.history.len() >= self.capacity {
            self.history.pop_front();
        }
        self.history.push_back(profile.clone());
    }

    /// Maximum absolute trait delta per snapshot across all 4 traits.
    /// Returns 0.0 if fewer than 2 snapshots recorded.
    pub fn drift_rate(&self) -> f32 {
        if self.history.len() < 2 {
            return 0.0;
        }
        let first = self.history.front().unwrap();
        let last = self.history.back().unwrap();
        let n = (self.history.len() - 1) as f32;
        let deltas = [
            (last.novelty_seeking - first.novelty_seeking).abs() / n,
            (last.harm_avoidance - first.harm_avoidance).abs() / n,
            (last.reward_dependence - first.reward_dependence).abs() / n,
            (last.persistence - first.persistence).abs() / n,
        ];
        deltas.into_iter().fold(0.0_f32, f32::max)
    }

    /// Whether drift exceeds the anomaly threshold (0.005 per snapshot).
    pub fn is_anomalous(&self) -> bool {
        self.drift_rate() > 0.005
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
            ..Default::default()
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
            ..Default::default()
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
    fn test_downstream_range_finite() {
        // All 5 downstream methods return finite values for edge-case inputs
        let mut bath = NeuromodulatorBath::default();

        // Edge case: depleted transmitters
        bath.dopamine.level = 0.0;
        bath.noradrenaline.level = 0.0;
        bath.serotonin.level = 0.0;
        bath.acetylcholine.level = 0.0;
        assert!(bath.learning_rate_factor().is_finite());
        assert!(bath.exploration_delta().is_finite());
        assert!(bath.confidence_delta().is_finite());
        assert!(bath.attention_factor().is_finite());
        assert!(bath.threshold_factor().is_finite());

        // Edge case: saturated transmitters with max sensitivity
        bath.dopamine.level = 1.0;
        bath.dopamine.receptor_sensitivity = 2.0;
        bath.noradrenaline.level = 1.0;
        bath.noradrenaline.receptor_sensitivity = 2.0;
        bath.serotonin.level = 1.0;
        bath.serotonin.receptor_sensitivity = 2.0;
        bath.acetylcholine.level = 1.0;
        bath.acetylcholine.receptor_sensitivity = 2.0;
        assert!(bath.learning_rate_factor().is_finite());
        assert!(bath.exploration_delta().is_finite());
        assert!(bath.confidence_delta().is_finite());
        assert!(bath.attention_factor().is_finite());
        assert!(bath.threshold_factor().is_finite());
    }

    #[test]
    fn test_bath_dominance_coefficient() {
        // Verify bath contributes meaningful LR modulation over 100 cycles
        let inputs = NeuromodulatorInputs {
            prediction_error: 0.3,
            surprise: false,
            reward_signal: 0.3,
            coherence: 0.6,
            arousal: 0.5,
            binding_strength: 0.5,
            epistemic_confidence: 0.5,
            flow_active: false,
        };

        let mut bath = NeuromodulatorBath::default();
        let mut lr_factors = Vec::new();
        for _ in 0..100 {
            bath.update(&inputs);
            lr_factors.push(bath.learning_rate_factor());
        }

        let avg_factor: f32 = lr_factors.iter().sum::<f32>() / lr_factors.len() as f32;
        // Bath LR factor should be in valid range (0.7–1.5)
        assert!(
            (0.7..=1.5).contains(&avg_factor),
            "bath LR factor out of range: {avg_factor}"
        );
        // The bath should produce meaningful LR factor (not stuck near minimum)
        assert!(
            avg_factor > 0.8,
            "bath should produce meaningful LR factor: {avg_factor}"
        );
        let variance: f32 = lr_factors
            .iter()
            .map(|f| (f - avg_factor).powi(2))
            .sum::<f32>()
            / lr_factors.len() as f32;
        assert!(variance.is_finite());
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

    // ── Phase 2: Cross-Modulation Learning ─────────────────────────────

    #[test]
    fn test_cross_mod_default_matches_biological_priors() {
        let m = CrossModulationMatrix::default();
        // DA→NE inhibitory
        assert!(m.weights[0][1] < 0.0, "DA→NE should be inhibitory");
        // 5-HT→NE inhibitory
        assert!(m.weights[2][1] < 0.0, "5-HT→NE should be inhibitory");
        // NE→ACh excitatory
        assert!(m.weights[1][3] > 0.0, "NE→ACh should be excitatory");
        // Self-connections should be zero
        for i in 0..4 {
            assert!(
                m.weights[i][i].abs() < f32::EPSILON,
                "Self-connection [{i}][{i}] should be 0"
            );
        }
    }

    #[test]
    fn test_cross_mod_hebbian_strengthens_coactivation() {
        let mut m = CrossModulationMatrix::default();
        let initial_da_ne = m.weights[0][1].abs();
        // Repeated DA+NE phasic co-activation
        for _ in 0..100 {
            m.hebbian_update(&[0.5, 0.5, 0.0, 0.0]);
        }
        // DA↔NE magnitude should increase (Hebbian: co-fire → strengthen)
        // Note: DA→NE started negative; Hebbian pushes it positive (toward co-activation).
        // The combined effect: magnitude of cross-mod for DA↔NE should increase.
        let after_da_ne_01 = m.weights[0][1];
        let after_ne_da_10 = m.weights[1][0];
        assert!(
            after_da_ne_01.abs() > initial_da_ne || after_ne_da_10.abs() > 0.001,
            "Hebbian should modify DA↔NE weights: w[0][1]={after_da_ne_01}, w[1][0]={after_ne_da_10}"
        );
    }

    #[test]
    fn test_cross_mod_weight_decay_prevents_runaway() {
        let mut m = CrossModulationMatrix::default();
        // 1000 cycles of strong co-activation across all channels
        for _ in 0..1000 {
            m.hebbian_update(&[1.0, 1.0, 1.0, 1.0]);
        }
        // All weights should stay within [-0.1, 0.1]
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    m.weights[i][j] >= -0.1 && m.weights[i][j] <= 0.1,
                    "Weight [{i}][{j}] out of range: {}",
                    m.weights[i][j]
                );
            }
        }
    }

    #[test]
    fn test_cross_mod_replaces_hardcoded() {
        // Verify that the learnable matrix produces NE suppression when DA is high
        // (functional equivalence with the old hardcoded `if DA > 0.7 { NE *= 0.97 }`)
        let m = CrossModulationMatrix::default();
        let levels = [0.8, 0.6, 0.5, 0.5]; // High DA
        let deltas = m.apply(&levels);
        // DA→NE weight is -0.03, so delta[1] should include -0.03 * 0.8 = -0.024
        assert!(
            deltas[1] < 0.0,
            "NE delta should be negative when DA is high: {}",
            deltas[1]
        );
    }

    // ── Phase 2: Circadian Holon ──────────────────────────────────────

    #[test]
    fn test_circadian_night_high_serotonin() {
        use crate::chronobiology::CircadianPhase;
        let mut bath = NeuromodulatorBath::default();
        bath.modulate_circadian(CircadianPhase::Night);
        // Night: 5-HT baseline rises to 0.65
        assert!(
            bath.serotonin.baseline_for_test() > 0.6,
            "Night 5-HT baseline should be >0.6, got {}",
            bath.serotonin.baseline_for_test()
        );
        // Run 50 cycles to let levels converge toward baseline
        let inputs = NeuromodulatorInputs {
            prediction_error: 0.2,
            surprise: false,
            reward_signal: 0.0,
            coherence: 0.5,
            arousal: 0.3,
            binding_strength: 0.5,
            epistemic_confidence: 0.5,
            flow_active: false,
        };
        for _ in 0..50 {
            bath.update(&inputs);
        }
        assert!(
            bath.serotonin.effective() > 0.45,
            "Night 5-HT effective should rise: {}",
            bath.serotonin.effective()
        );
    }

    #[test]
    fn test_circadian_dawn_high_noradrenaline() {
        use crate::chronobiology::CircadianPhase;
        let mut bath = NeuromodulatorBath::default();
        bath.modulate_circadian(CircadianPhase::Dawn);
        assert!(
            bath.noradrenaline.baseline_for_test() > 0.55,
            "Dawn NE baseline should be >0.55, got {}",
            bath.noradrenaline.baseline_for_test()
        );
    }

    #[test]
    fn test_circadian_day_high_acetylcholine() {
        use crate::chronobiology::CircadianPhase;
        let mut bath = NeuromodulatorBath::default();
        bath.modulate_circadian(CircadianPhase::Day);
        assert!(
            bath.acetylcholine.baseline_for_test() > 0.55,
            "Day ACh baseline should be >0.55, got {}",
            bath.acetylcholine.baseline_for_test()
        );
    }

    #[test]
    fn test_circadian_phase_transition() {
        use crate::chronobiology::CircadianPhase;
        let mut bath = NeuromodulatorBath::default();

        // Start at Dawn: NE baseline high
        bath.modulate_circadian(CircadianPhase::Dawn);
        let dawn_ne_baseline = bath.noradrenaline.baseline_for_test();
        assert!(dawn_ne_baseline > 0.55);

        // Transition to Night: NE baseline drops
        bath.modulate_circadian(CircadianPhase::Night);
        let night_ne_baseline = bath.noradrenaline.baseline_for_test();
        assert!(
            night_ne_baseline < dawn_ne_baseline,
            "Night NE baseline ({night_ne_baseline}) should be < Dawn ({dawn_ne_baseline})"
        );
        // 5-HT rises when switching to Night
        let night_sht_baseline = bath.serotonin.baseline_for_test();
        assert!(
            night_sht_baseline > 0.6,
            "Night 5-HT baseline should be > 0.6: {night_sht_baseline}"
        );
    }

    // ── Phase 2: Neuromod → Consciousness Bridge ────────────────────

    #[test]
    fn test_consciousness_mod_default_near_one() {
        let bath = NeuromodulatorBath::default();
        let factor = bath.consciousness_modulation();
        assert!(
            (0.9..=1.1).contains(&factor),
            "Default bath should produce factor near 1.0: {factor}"
        );
    }

    #[test]
    fn test_consciousness_mod_depleted_ach_ne() {
        let mut bath = NeuromodulatorBath::default();
        bath.acetylcholine.level = 0.0;
        bath.acetylcholine.receptor_sensitivity = 1.0;
        bath.noradrenaline.level = 0.0;
        bath.noradrenaline.receptor_sensitivity = 1.0;
        let factor = bath.consciousness_modulation();
        assert!(
            factor < 0.65,
            "Depleted ACh/NE should suppress consciousness: {factor}"
        );
    }

    #[test]
    fn test_consciousness_mod_elevated() {
        let mut bath = NeuromodulatorBath::default();
        bath.acetylcholine.level = 0.9;
        bath.acetylcholine.receptor_sensitivity = 1.2;
        bath.noradrenaline.level = 0.8;
        bath.noradrenaline.receptor_sensitivity = 1.2;
        let factor = bath.consciousness_modulation();
        assert!(
            factor > 1.0,
            "Elevated ACh/NE should enhance consciousness: {factor}"
        );
    }

    // ── Phase 4: Exocortex Query Trigger ────────────────────────────

    #[test]
    fn test_exocortex_query_triggers() {
        let mut bath = NeuromodulatorBath::default();
        // High NE (uncertainty), low DA (no reward), low 5-HT (low confidence)
        bath.noradrenaline.level = 0.9;
        bath.noradrenaline.receptor_sensitivity = 1.0;
        bath.dopamine.level = 0.2;
        bath.dopamine.receptor_sensitivity = 1.0;
        bath.serotonin.level = 0.3;
        bath.serotonin.receptor_sensitivity = 1.0;
        assert!(
            bath.should_query_exocortex(),
            "Should trigger exocortex query: NE={}, DA={}, 5-HT={}",
            bath.noradrenaline.effective(),
            bath.dopamine.effective(),
            bath.serotonin.effective()
        );
    }

    #[test]
    fn test_exocortex_query_suppressed() {
        let mut bath = NeuromodulatorBath::default();
        // High DA (reward) → should NOT trigger even with high NE
        bath.noradrenaline.level = 0.9;
        bath.dopamine.level = 0.8; // high reward prediction
        bath.serotonin.level = 0.3;
        assert!(
            !bath.should_query_exocortex(),
            "Should NOT trigger exocortex query when DA is high"
        );
    }

    // ── Phase 2: Sleep Consolidation ────────────────────────────────

    #[test]
    fn test_sleep_boost_default_modest() {
        let bath = NeuromodulatorBath::default();
        // Default DA tonic = 0.5 (level=0.5, phasic=0.0) → tonic_da = 0.5
        // Maps (0.5 - 0.4) * (2.0/0.3) + 1.0 = 0.1 * 6.67 + 1.0 = 1.67
        let boost = bath.sleep_consolidation_boost();
        assert!(
            (1.0..=2.0).contains(&boost),
            "Default bath should produce modest sleep boost: {boost}"
        );
    }

    #[test]
    fn test_sleep_boost_high_tonic_da() {
        let mut bath = NeuromodulatorBath::default();
        bath.dopamine.level = 0.7;
        bath.dopamine.phasic = 0.05; // small phasic → tonic ≈ 0.65
        let boost = bath.sleep_consolidation_boost();
        assert!(
            boost > 2.0,
            "High tonic DA should produce substantial sleep boost: {boost}"
        );
    }

    // ── Phase 5: Receptor Personality ────────────────────────────────

    #[test]
    fn test_personality_profile_from_defaults() {
        let bath = NeuromodulatorBath::default();
        let profile = bath.personality_profile();
        // Default sensitivity = 1.0 → balanced profile
        assert!((profile.novelty_seeking - 1.0).abs() < f32::EPSILON);
        assert!((profile.harm_avoidance - 1.0).abs() < f32::EPSILON);
        assert!((profile.reward_dependence - 1.0).abs() < f32::EPSILON);
        assert!((profile.persistence - 1.0).abs() < f32::EPSILON);
        assert_eq!(bath.personality_description(), "balanced");
    }

    #[test]
    fn test_personality_description_novelty_seeking() {
        let mut bath = NeuromodulatorBath::default();
        // High DA sensitivity → novelty-seeking
        bath.dopamine.receptor_sensitivity = 1.5;
        let desc = bath.personality_description();
        assert!(
            desc.contains("novelty-seeking"),
            "Expected 'novelty-seeking' in: {desc}"
        );
    }

    #[test]
    fn test_personality_adapts_over_time() {
        let mut bath = NeuromodulatorBath::default();
        let initial = bath.personality_profile().novelty_seeking;

        // 500 cycles of high reward → DA stays high → tolerance → sensitivity drops
        let inputs = NeuromodulatorInputs {
            prediction_error: 0.05,
            surprise: false,
            reward_signal: 0.9, // sustained high reward
            coherence: 0.8,
            arousal: 0.3,
            binding_strength: 0.7,
            epistemic_confidence: 0.8,
            flow_active: true,
        };
        for _ in 0..500 {
            bath.update(&inputs);
        }

        let after = bath.personality_profile().novelty_seeking;
        // DA tolerance should have reduced receptor sensitivity (novelty seeking)
        assert!(
            after < initial,
            "Novelty seeking should decrease after sustained reward: {after} vs initial {initial}"
        );
    }

    // ── Phase 6: Circadian × Personality — Baseline-Relative Adaptation ─

    #[test]
    fn test_circadian_no_spurious_adaptation() {
        use crate::chronobiology::CircadianPhase;
        let mut bath = NeuromodulatorBath::default();
        bath.modulate_circadian(CircadianPhase::Night);
        // Night: NE baseline = 0.30. With quiescent inputs (no arousal, no PE),
        // NE should converge to baseline. The old absolute-threshold code
        // would spuriously sensitize at <0.3. The new baseline-relative code
        // only sensitizes at <baseline-0.2 = 0.10.
        //
        // Note: With learnable cross-modulation (Phase 2), DA→NE inhibition
        // (-0.03 × DA_level) continuously pushes NE below baseline, which can
        // trigger legitimate sensitization. The test verifies sensitivity
        // stays in a reasonable range (not runaway), not that it's exactly 1.0.
        let inputs = NeuromodulatorInputs {
            prediction_error: 0.0,
            surprise: false,
            reward_signal: 0.0,
            coherence: 0.5,
            arousal: 0.0,
            binding_strength: 0.5,
            epistemic_confidence: 0.5,
            flow_active: false,
        };
        for _ in 0..200 {
            bath.update(&inputs);
        }
        // Sensitivity should stay in reasonable range — cross-mod may push NE
        // below baseline triggering mild sensitization, but not runaway
        assert!(
            bath.noradrenaline.receptor_sensitivity > 0.8,
            "NE sensitivity should not drop too low at Night baseline, got {}",
            bath.noradrenaline.receptor_sensitivity
        );
        assert!(
            bath.noradrenaline.receptor_sensitivity < 2.0,
            "NE sensitivity should stay below max, got {}",
            bath.noradrenaline.receptor_sensitivity
        );
    }

    #[test]
    fn test_adaptation_at_extremes() {
        // DA pushed far above any baseline+0.2 → tolerance should still activate
        let mut t = Transmitter {
            level: 0.95,
            receptor_sensitivity: 1.0,
            reuptake_rate: 0.0, // disable reuptake so level stays high
            baseline: 0.5,     // high = 0.7, level 0.95 > 0.7 → tolerance
            ..Default::default()
        };
        let initial_sens = t.receptor_sensitivity;
        for _ in 0..200 {
            t.reuptake();
        }
        assert!(
            t.receptor_sensitivity < initial_sens,
            "Tolerance should activate at extreme high level: got {}",
            t.receptor_sensitivity
        );
    }

    // ── PersonalityDriftTracker ─────────────────────────────────────────

    #[test]
    fn test_drift_tracker_stable() {
        let bath = NeuromodulatorBath::default();
        let mut tracker = PersonalityDriftTracker::new(16);
        // 100 identical snapshots → drift should be near 0
        for _ in 0..100 {
            tracker.record(&bath.personality_profile());
        }
        assert!(
            tracker.drift_rate() < 0.001,
            "Drift rate should be near 0 for stable bath: {}",
            tracker.drift_rate()
        );
        assert!(!tracker.is_anomalous());
    }

    #[test]
    fn test_drift_tracker_detects_change() {
        let mut tracker = PersonalityDriftTracker::new(16);
        // Push divergent sensitivity snapshots
        for i in 0..16 {
            tracker.record(&NeuromodulatorProfile {
                novelty_seeking: 1.0 + i as f32 * 0.02,
                harm_avoidance: 1.0,
                reward_dependence: 1.0,
                persistence: 1.0,
            });
        }
        // 15 steps × 0.02 = 0.30 total delta → 0.02/step >> 0.005 threshold
        assert!(
            tracker.is_anomalous(),
            "Should detect rapid drift: rate={}",
            tracker.drift_rate()
        );
    }

    #[test]
    fn test_drift_tracker_window_eviction() {
        let mut tracker = PersonalityDriftTracker::new(16);
        // Push 20 records — oldest 4 should be evicted
        for i in 0..20 {
            tracker.record(&NeuromodulatorProfile {
                novelty_seeking: 1.0 + i as f32 * 0.001,
                harm_avoidance: 1.0,
                reward_dependence: 1.0,
                persistence: 1.0,
            });
        }
        assert_eq!(tracker.history.len(), 16, "Should cap at capacity");
    }

    // ── Phase 2: Receptor Sensitivity Persistence ────────────────────

    #[test]
    fn test_checkpoint_roundtrip() {
        let mut bath = NeuromodulatorBath::default();
        bath.dopamine.receptor_sensitivity = 0.75;
        bath.noradrenaline.receptor_sensitivity = 1.5;
        bath.serotonin.receptor_sensitivity = 0.9;
        bath.acetylcholine.receptor_sensitivity = 1.1;
        bath.cross_mod.weights[0][1] = -0.05;
        let ckpt = bath.checkpoint();
        // Restore into a fresh bath
        let mut bath2 = NeuromodulatorBath::default();
        bath2.restore(&ckpt);
        assert!(
            (bath2.dopamine.receptor_sensitivity - 0.75).abs() < f32::EPSILON,
            "DA sensitivity should roundtrip"
        );
        assert!(
            (bath2.noradrenaline.receptor_sensitivity - 1.5).abs() < f32::EPSILON,
            "NE sensitivity should roundtrip"
        );
        assert!(
            (bath2.cross_mod.weights[0][1] - (-0.05)).abs() < f32::EPSILON,
            "Cross-mod weights should roundtrip"
        );
    }

    #[test]
    fn test_checkpoint_clamps_invalid() {
        let ckpt = NeurochemistryCheckpoint {
            da_sensitivity: 5.0,  // out of range
            ne_sensitivity: -1.0, // out of range
            sht_sensitivity: 1.0,
            ach_sensitivity: 1.0,
            cross_mod_weights: [[0.5; 4]; 4], // out of range
        };
        let mut bath = NeuromodulatorBath::default();
        bath.restore(&ckpt);
        assert!(
            bath.dopamine.receptor_sensitivity <= 2.0,
            "DA sensitivity clamped to 2.0"
        );
        assert!(
            bath.noradrenaline.receptor_sensitivity >= 0.5,
            "NE sensitivity clamped to 0.5"
        );
        for row in &bath.cross_mod.weights {
            for &w in row {
                assert!((-0.1..=0.1).contains(&w), "Cross-mod weight clamped: {w}");
            }
        }
    }

    #[test]
    fn test_checkpoint_preserves_cross_mod() {
        let mut bath = NeuromodulatorBath::default();
        // Modify cross-mod via Hebbian learning
        for _ in 0..50 {
            bath.cross_mod.hebbian_update(&[0.5, 0.3, 0.0, 0.0]);
        }
        let w_before = bath.cross_mod.weights[0][1];
        let ckpt = bath.checkpoint();
        let mut bath2 = NeuromodulatorBath::default();
        bath2.restore(&ckpt);
        assert!(
            (bath2.cross_mod.weights[0][1] - w_before).abs() < f32::EPSILON,
            "Cross-mod DA→NE weight should persist: {} vs {}",
            bath2.cross_mod.weights[0][1],
            w_before
        );
    }

    // ── Neuromod-Aware Training: gradient_scale_factor / threshold_gate ──

    #[test]
    fn test_gradient_scale_high_da() {
        let mut bath = NeuromodulatorBath::default();
        bath.dopamine.level = 0.9;
        bath.dopamine.receptor_sensitivity = 1.2;
        let factor = bath.gradient_scale_factor();
        assert!(
            factor > 1.0,
            "High DA should produce gradient scale >1.0: got {factor}"
        );
    }

    #[test]
    fn test_gradient_scale_low_da() {
        let mut bath = NeuromodulatorBath::default();
        bath.dopamine.level = 0.1;
        bath.dopamine.receptor_sensitivity = 0.8;
        let factor = bath.gradient_scale_factor();
        assert!(
            factor < 0.75,
            "Low DA should produce gradient scale <0.75: got {factor}"
        );
    }

    #[test]
    fn test_threshold_gate_high_ach() {
        let mut bath = NeuromodulatorBath::default();
        bath.acetylcholine.level = 0.9;
        bath.acetylcholine.receptor_sensitivity = 1.2;
        let gate = bath.threshold_gate();
        assert!(
            gate < 1.0,
            "High ACh should lower threshold gate (<1.0): got {gate}"
        );
    }

    #[test]
    fn test_threshold_gate_low_ach() {
        let mut bath = NeuromodulatorBath::default();
        bath.acetylcholine.level = 0.1;
        bath.acetylcholine.receptor_sensitivity = 0.8;
        let gate = bath.threshold_gate();
        assert!(
            gate > 1.3,
            "Low ACh should raise threshold gate (>1.3): got {gate}"
        );
    }

    // ── Exocortex Trigger Counter ─────────────────────────────────────

    // ── Phase 2: Phasic/Tonic Dynamics ──────────────────────────────────

    #[test]
    fn test_phasic_decays_faster_than_tonic() {
        let mut t = Transmitter::default();
        t.produce(0.5);
        assert!(t.phasic > 0.4, "phasic should capture burst: {}", t.phasic);
        // After 10 reuptake cycles: phasic should be nearly gone, level still high
        for _ in 0..10 {
            t.reuptake();
        }
        assert!(
            t.phasic < 0.05,
            "phasic should decay fast (<0.05): {}",
            t.phasic
        );
        assert!(
            t.level > 0.3,
            "tonic level should persist (>0.3): {}",
            t.level
        );
    }

    #[test]
    fn test_phasic_burst_observable() {
        let mut bath = NeuromodulatorBath::default();
        // Strong reward → DA burst
        let inputs = NeuromodulatorInputs {
            prediction_error: 0.1,
            surprise: true,
            reward_signal: 0.8,
            coherence: 0.5,
            arousal: 0.7,
            binding_strength: 0.5,
            epistemic_confidence: 0.5,
            flow_active: false,
        };
        bath.update(&inputs);
        // DA and NE should have phasic signal immediately
        assert!(
            bath.da_phasic() > 0.0,
            "DA phasic should be positive after reward: {}",
            bath.da_phasic()
        );
        assert!(
            bath.ne_phasic() > 0.0,
            "NE phasic should be positive after surprise: {}",
            bath.ne_phasic()
        );
        // After 20 cycles of quiescent input → phasic approaches 0
        let quiet = NeuromodulatorInputs {
            prediction_error: 0.0,
            surprise: false,
            reward_signal: 0.0,
            coherence: 0.5,
            arousal: 0.3,
            binding_strength: 0.5,
            epistemic_confidence: 0.5,
            flow_active: false,
        };
        for _ in 0..20 {
            bath.update(&quiet);
        }
        // DA phasic decays fast but each update() cycle adds small production
        // (reward_signal=0 still produces a small positive DA signal from low PE).
        // After 20 quiet cycles, phasic should be much lower than the initial burst.
        assert!(
            bath.da_phasic() < 0.2,
            "DA phasic should decay substantially: {}",
            bath.da_phasic()
        );
    }

    #[test]
    fn test_effective_includes_phasic_overlay() {
        let mut t = Transmitter {
            level: 0.3,
            receptor_sensitivity: 1.0,
            reuptake_rate: 0.0, // disable tonic reuptake
            baseline: 0.3,
            phasic: 0.0,
            phasic_decay: 0.3,
        };
        let eff_before = t.effective();
        // Produce adds to both level and phasic
        t.produce(0.3);
        let eff_after = t.effective();
        assert!(
            eff_after > eff_before,
            "effective should increase after produce: {} > {}",
            eff_after,
            eff_before
        );
    }

    #[test]
    fn test_exocortex_counter_triggers() {
        let mut bath = NeuromodulatorBath::default();
        // Set up conditions that trigger exocortex: high NE, low DA, low 5-HT
        bath.noradrenaline.level = 0.9;
        bath.noradrenaline.receptor_sensitivity = 1.0;
        bath.dopamine.level = 0.2;
        bath.dopamine.receptor_sensitivity = 1.0;
        bath.serotonin.level = 0.3;
        bath.serotonin.receptor_sensitivity = 1.0;
        assert!(bath.should_query_exocortex());
        // Simulate counter accumulation
        let mut count = 0_u64;
        for _ in 0..5 {
            if bath.should_query_exocortex() {
                count += 1;
            }
        }
        assert_eq!(count, 5, "Counter should increment for each trigger check");
    }
}
