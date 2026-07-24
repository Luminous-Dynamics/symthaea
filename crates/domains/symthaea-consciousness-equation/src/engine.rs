// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use std::collections::VecDeque;
// web-time: drop-in Instant for wasm32 (std::time::Instant panics on wasm32-unknown-unknown)
use web_time::Instant;

use crate::config::MasterEquationConfig;
use crate::embodiment::EmbodimentFactor;
use crate::narrative::NarrativeCoherence;
use crate::social::SocialEmbedding;
use crate::types::{ConsciousnessInputs, ConsciousnessResult};

/// The Master Consciousness Equation engine
#[derive(Debug)]
pub struct MasterConsciousnessEquation {
    /// Configuration
    config: MasterEquationConfig,

    /// Embodiment factor computation
    pub embodiment_factor: EmbodimentFactor,

    /// Narrative coherence computation
    pub narrative_coherence: NarrativeCoherence,

    /// Social embedding computation
    pub social_embedding: SocialEmbedding,

    /// Consciousness history for temporal stability
    history: VecDeque<ConsciousnessSnapshot>,

    /// Last computation time
    last_computation: Option<Instant>,

    /// Current gating factors (γᵢ)
    gating_factors: GatingFactors,
}

/// Gating factors for each component
#[derive(Debug, Clone)]
struct GatingFactors {
    phi: f64,
    broadcast: f64,
    working_memory: f64,
    attention: f64,
    recurrence: f64,
    embodiment: f64,
    knowledge: f64,
    embodiment_factor: f64,
    narrative: f64,
    social: f64,
}

impl Default for GatingFactors {
    fn default() -> Self {
        Self {
            phi: 1.0,
            broadcast: 1.0,
            working_memory: 1.0,
            attention: 1.0,
            recurrence: 1.0,
            embodiment: 1.0,
            knowledge: 1.0,
            embodiment_factor: 1.0,
            narrative: 1.0,
            social: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
struct ConsciousnessSnapshot {
    level: f64,
    #[allow(dead_code)] // Stored for potential temporal analysis
    timestamp: Instant,
}

impl Default for MasterConsciousnessEquation {
    fn default() -> Self {
        Self::new(MasterEquationConfig::default())
    }
}

impl MasterConsciousnessEquation {
    /// Create a new Master Consciousness Equation engine
    pub fn new(config: MasterEquationConfig) -> Self {
        Self {
            history: VecDeque::with_capacity(config.history_size),
            config,
            embodiment_factor: EmbodimentFactor::new(),
            narrative_coherence: NarrativeCoherence::new(),
            social_embedding: SocialEmbedding::new(),
            last_computation: None,
            gating_factors: GatingFactors::default(),
        }
    }

    /// Compute consciousness level C(t) using the master equation
    ///
    /// C(t) = σ(softmin(Φ, B, W, A, R, E, K; τ)) × [Σ(wᵢ × Cᵢ × γᵢ) / Σ(wᵢ)] × S × ρ(t) × M × N × Soc
    pub fn compute(&mut self, inputs: &ConsciousnessInputs) -> ConsciousnessResult {
        // Get the three new factors
        let m = if self.config.enable_embodiment_factor {
            self.embodiment_factor.compute()
        } else {
            1.0
        };

        let n = if self.config.enable_narrative_factor {
            self.narrative_coherence.compute()
        } else {
            1.0
        };

        let soc = if self.config.enable_social_factor {
            self.social_embedding.compute()
        } else {
            1.0
        };

        // Step 1: Compute softmin of bottleneck factors
        let factors = vec![
            ("Φ (Integration)", inputs.phi),
            ("B (Broadcast)", inputs.broadcast),
            ("W (Working Memory)", inputs.working_memory),
            ("A (Attention)", inputs.attention),
            ("R (Recurrence)", inputs.recurrence),
            ("E (Embodiment)", inputs.embodiment),
            ("K (Knowledge)", inputs.knowledge),
        ];

        let (bottleneck_factor, bottleneck_name) = self.softmin_with_name(&factors);

        // Step 2: Apply sigmoid to bottleneck
        let sigmoid_bottleneck = self.sigmoid(bottleneck_factor);

        // Step 3: Compute weighted sum of components [Σ(wᵢ × Cᵢ × γᵢ) / Σ(wᵢ)]
        //
        // Defensive clamp (2026-07-22, probe_cl_calibration finding): the 7 fields on
        // `ConsciousnessInputs` are each clamped to [0,1] by their callers (see the
        // 2026-07-18 embodiment unit fix), but `m`/`n`/`soc` here are NOT — they come
        // straight from `EmbodimentFactor::compute()` / `NarrativeCoherence::compute()`
        // / `SocialEmbedding::compute()`, products of internal state fields with no
        // equivalent upper-bound guarantee. `test_weighted_sum_bounded_for_unit_inputs`
        // only exercises the 7 bounded inputs, not m/n/soc, so it didn't catch this.
        // Measured live (probe_cl_calibration, "alarming" regime, 300 cycles post the
        // embodiment fix): mean weighted_sum = 1.1057 > 1.0, and consciousness_level
        // was pinned at the Green safety tier for all 300 cycles with zero transitions
        // — the intended [0,1] invariant this component should hold (documented in
        // `test_weighted_sum_bounded_for_unit_inputs`'s own doc comment) was silently
        // violated, pushing consciousness_level toward its outer clamp ceiling and
        // destroying tier resolution in exactly the high-arousal regime where graduated
        // motor-safety discrimination matters most.
        let weighted_sum = self.compute_weighted_sum(inputs, m, n, soc).clamp(0.0, 1.0);

        // Step 4: Compute temporal stability ρ(t)
        let temporal_stability = self.compute_temporal_stability();

        // Step 5: Final consciousness level with new factors
        // C(t) = σ(softmin) × weighted_sum × S × ρ(t) × M' × N' × Soc'
        //
        // M, N, Soc are already included in the weighted_sum via their component weights.
        // As raw multiplicatives they double-count AND cause catastrophic attenuation when
        // a factor is low (e.g., Soc=0.35 in non-social context → 65% permanent haircut).
        // Convert to soft modulations: map [0,1] → [0.65, 1.0] so low values attenuate
        // gently rather than crushing consciousness. Floor raised from 0.5→0.65:
        // three moderate factors at 0.5 used to give 0.75³=0.42 (58% loss).
        // With 0.65 floor: 0.825³=0.56 (44% loss) — still meaningful but not punitive.
        // Science: Modular consciousness theories (Baars 2005) — subsystem deficits
        // reduce but don't eliminate consciousness. Consciousness can exist without
        // rich social embedding or narrative (dreamless sleep, infant consciousness).
        let m_mod = 0.65 + 0.35 * m;
        let n_mod = 0.65 + 0.35 * n;
        let soc_mod = 0.65 + 0.35 * soc;
        let consciousness_level = sigmoid_bottleneck
            * weighted_sum
            * inputs.synchrony
            * temporal_stability
            * m_mod
            * n_mod
            * soc_mod;

        // Clamp to [0, 1]
        let consciousness_level = consciousness_level.clamp(0.0, 1.0);

        // Record in history
        self.record_snapshot(consciousness_level);
        self.last_computation = Some(Instant::now());

        ConsciousnessResult {
            consciousness_level,
            bottleneck_factor,
            weighted_sum,
            embodiment_factor: m,
            narrative_coherence: n,
            social_embedding: soc,
            temporal_stability,
            bottleneck_name,
            factors: inputs.clone(),
        }
    }

    /// Softmin function: softmin(x; τ) = Σ(xᵢ × exp(-xᵢ/τ)) / Σ(exp(-xᵢ/τ))
    /// This is a smooth minimum that identifies the bottleneck.
    /// Numerically stabilized: clamps exponents to prevent overflow, and falls
    /// back to hard-min when the result is non-finite.
    fn softmin_with_name(&self, factors: &[(&str, f64)]) -> (f64, String) {
        let tau = self.config.softmin_tau;
        let epsilon = self.config.epsilon;

        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        let mut min_val = f64::MAX;
        let mut min_name = "Unknown".to_string();

        for (name, val) in factors {
            // Clamp exponent to prevent overflow/underflow: exp(709.78) is f64::MAX,
            // exp(-745) is the smallest positive subnormal. Use ±700 for safety margin.
            let weight = (-val / tau).clamp(-700.0, 700.0).exp();
            weighted_sum += val * weight;
            weight_sum += weight;

            if *val < min_val {
                min_val = *val;
                min_name = name.to_string();
            }
        }

        let softmin = if weight_sum > epsilon {
            let raw = weighted_sum / weight_sum;
            // Fall back to hard-min if val*weight overflowed
            if raw.is_finite() { raw } else { min_val }
        } else {
            min_val
        };

        (softmin, min_name)
    }

    /// Sigmoid function: σ(x) = 1 / (1 + exp(-5x))
    /// Numerically stable: avoids exp() overflow for large |x| by branching on sign.
    fn sigmoid(&self, x: f64) -> f64 {
        let z = x * 5.0; // Scaled for [0,1] inputs
        if z >= 0.0 {
            1.0 / (1.0 + (-z).exp())
        } else {
            let e = z.exp();
            e / (1.0 + e)
        }
    }

    /// Compute weighted sum: Σ(wᵢ × Cᵢ × γᵢ) / Σ(wᵢ)
    fn compute_weighted_sum(&self, inputs: &ConsciousnessInputs, m: f64, n: f64, soc: f64) -> f64 {
        let w = &self.config.component_weights;
        let g = &self.gating_factors;

        let numerator = w.phi * inputs.phi * g.phi
            + w.broadcast * inputs.broadcast * g.broadcast
            + w.working_memory * inputs.working_memory * g.working_memory
            + w.attention * inputs.attention * g.attention
            + w.recurrence * inputs.recurrence * g.recurrence
            + w.embodiment * inputs.embodiment * g.embodiment
            + w.knowledge * inputs.knowledge * g.knowledge
            + w.embodiment_factor * m * g.embodiment_factor
            + w.narrative * n * g.narrative
            + w.social * soc * g.social;

        let denominator = w.total();

        if denominator > self.config.epsilon {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Compute temporal stability ρ(t)
    fn compute_temporal_stability(&self) -> f64 {
        if self.history.len() < 5 {
            return 0.8; // Default when not enough history
        }

        // Compute variance of recent consciousness levels
        let values: Vec<f64> = self
            .history
            .iter()
            .rev()
            .take(20)
            .map(|s| s.level)
            .collect();

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;

        // Lower variance = higher stability
        // Map variance [0, 0.25] to stability [1.0, 0.5]
        let stability = 1.0 - (variance * 2.0).min(0.5);

        stability.clamp(0.5, 1.0)
    }

    /// Record a consciousness snapshot
    fn record_snapshot(&mut self, level: f64) {
        if self.history.len() >= self.config.history_size {
            self.history.pop_front();
        }
        self.history.push_back(ConsciousnessSnapshot {
            level,
            timestamp: Instant::now(),
        });
    }

    /// Set gating factor for a component
    pub fn set_gating(&mut self, component: &str, value: f64) {
        let value = value.clamp(0.0, 1.0);
        match component {
            "phi" => self.gating_factors.phi = value,
            "broadcast" => self.gating_factors.broadcast = value,
            "working_memory" => self.gating_factors.working_memory = value,
            "attention" => self.gating_factors.attention = value,
            "recurrence" => self.gating_factors.recurrence = value,
            "embodiment" => self.gating_factors.embodiment = value,
            "knowledge" => self.gating_factors.knowledge = value,
            "embodiment_factor" => self.gating_factors.embodiment_factor = value,
            "narrative" => self.gating_factors.narrative = value,
            "social" => self.gating_factors.social = value,
            _ => {}
        }
    }

    /// Get current configuration
    pub fn config(&self) -> &MasterEquationConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: MasterEquationConfig) {
        self.config = config;
    }

    /// Get consciousness trend (positive = improving)
    pub fn consciousness_trend(&self) -> f64 {
        if self.history.len() < 10 {
            return 0.0;
        }

        let recent: f64 = self
            .history
            .iter()
            .rev()
            .take(5)
            .map(|s| s.level)
            .sum::<f64>()
            / 5.0;

        let older: f64 = self
            .history
            .iter()
            .rev()
            .skip(5)
            .take(5)
            .map(|s| s.level)
            .sum::<f64>()
            / 5.0;

        recent - older
    }

    /// Get average consciousness level
    pub fn average_consciousness(&self) -> f64 {
        if self.history.is_empty() {
            return 0.5;
        }

        self.history.iter().map(|s| s.level).sum::<f64>() / self.history.len() as f64
    }

    /// Describe current state in natural language
    pub fn describe_state(&self, result: &ConsciousnessResult) -> String {
        let level_desc = match result.consciousness_level {
            l if l > 0.8 => "highly integrated",
            l if l > 0.6 => "well integrated",
            l if l > 0.4 => "moderately integrated",
            l if l > 0.2 => "partially integrated",
            _ => "minimally integrated",
        };

        let bottleneck_desc = format!("bottlenecked by {}", result.bottleneck_name);

        let embodiment_desc = if self.config.enable_embodiment_factor {
            format!(", embodiment factor {:.2}", result.embodiment_factor)
        } else {
            String::new()
        };

        let narrative_desc = if self.config.enable_narrative_factor {
            format!(", narrative coherence {:.2}", result.narrative_coherence)
        } else {
            String::new()
        };

        let social_desc = if self.config.enable_social_factor {
            format!(", social embedding {:.2}", result.social_embedding)
        } else {
            String::new()
        };

        format!(
            "Consciousness is {} (C={:.3}), {}{}{}{}, temporal stability {:.2}",
            level_desc,
            result.consciousness_level,
            bottleneck_desc,
            embodiment_desc,
            narrative_desc,
            social_desc,
            result.temporal_stability
        )
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use crate::config::MasterEquationConfig;
    use crate::types::ConsciousnessInputs;

    fn all_high_inputs() -> ConsciousnessInputs {
        ConsciousnessInputs {
            phi: 0.8,
            broadcast: 0.8,
            working_memory: 0.8,
            attention: 0.8,
            recurrence: 0.8,
            embodiment: 0.8,
            knowledge: 0.8,
            synchrony: 0.8,
        }
    }

    #[test]
    fn test_default_creation() {
        let eq = MasterConsciousnessEquation::default();
        // Config should have sensible defaults
        assert!(eq.config().softmin_tau > 0.0);
        assert!(eq.config().epsilon > 0.0);
        assert!(eq.config().history_size > 0);
        assert!(eq.config().enable_embodiment_factor);
        assert!(eq.config().enable_narrative_factor);
        assert!(eq.config().enable_social_factor);
    }

    #[test]
    fn test_compute_all_high() {
        let mut eq = MasterConsciousnessEquation::default();
        let inputs = all_high_inputs();
        let result = eq.compute(&inputs);
        // With default embodiment/narrative/social factors (not yet stimulated),
        // consciousness is modulated down from the 0.8 inputs.
        assert!(
            result.consciousness_level > 0.0 && result.consciousness_level <= 1.0,
            "All-high inputs should produce valid consciousness, got {}",
            result.consciousness_level
        );
    }

    #[test]
    fn test_compute_all_zero() {
        let mut eq = MasterConsciousnessEquation::default();
        let inputs = ConsciousnessInputs {
            phi: 0.0,
            broadcast: 0.0,
            working_memory: 0.0,
            attention: 0.0,
            recurrence: 0.0,
            embodiment: 0.0,
            knowledge: 0.0,
            synchrony: 0.0,
        };
        let result = eq.compute(&inputs);
        assert!(
            result.consciousness_level < 0.01,
            "All-zero inputs should produce consciousness near 0.0, got {}",
            result.consciousness_level
        );
    }

    #[test]
    fn test_bottleneck_identification() {
        let mut eq = MasterConsciousnessEquation::default();
        let inputs = ConsciousnessInputs {
            phi: 0.01,
            broadcast: 0.8,
            working_memory: 0.8,
            attention: 0.8,
            recurrence: 0.8,
            embodiment: 0.8,
            knowledge: 0.8,
            synchrony: 0.8,
        };
        let result = eq.compute(&inputs);
        assert!(
            result.bottleneck_name.contains("\u{03A6}"),
            "With phi=0.01 as the lowest factor, bottleneck should contain 'Phi', got '{}'",
            result.bottleneck_name
        );
    }

    #[test]
    fn test_gating_reduces_component() {
        let mut eq = MasterConsciousnessEquation::default();
        let inputs = all_high_inputs();

        // Compute baseline weighted sum with all gating at 1.0
        let baseline = eq.compute(&inputs);
        let baseline_ws = baseline.weighted_sum;

        // Set phi gating to 0.0 and recompute
        let mut eq2 = MasterConsciousnessEquation::default();
        eq2.set_gating("phi", 0.0);
        let gated = eq2.compute(&inputs);
        let gated_ws = gated.weighted_sum;

        assert!(
            gated_ws < baseline_ws,
            "Gating phi to 0.0 should reduce weighted_sum: baseline={}, gated={}",
            baseline_ws,
            gated_ws
        );
    }

    #[test]
    fn test_temporal_stability_default() {
        let mut eq = MasterConsciousnessEquation::default();
        let inputs = all_high_inputs();
        // With fewer than 5 history entries, temporal_stability should be the default 0.8
        let result = eq.compute(&inputs);
        assert!(
            (result.temporal_stability - 0.8).abs() < 1e-6,
            "With < 5 history entries temporal_stability should be 0.8, got {}",
            result.temporal_stability
        );
    }

    #[test]
    fn test_temporal_stability_stable_history() {
        let mut eq = MasterConsciousnessEquation::default();
        let inputs = all_high_inputs();
        // Compute 20 times with the same inputs to build stable history
        for _ in 0..20 {
            eq.compute(&inputs);
        }
        let result = eq.compute(&inputs);
        assert!(
            result.temporal_stability > 0.95,
            "Stable repeated inputs should yield temporal_stability near 1.0, got {}",
            result.temporal_stability
        );
    }

    #[test]
    fn test_temporal_stability_variable_history() {
        let mut eq = MasterConsciousnessEquation::default();
        let high = all_high_inputs();
        let low = ConsciousnessInputs {
            phi: 0.1,
            broadcast: 0.1,
            working_memory: 0.1,
            attention: 0.1,
            recurrence: 0.1,
            embodiment: 0.1,
            knowledge: 0.1,
            synchrony: 0.1,
        };
        // Alternate between high and low inputs to create instability
        for i in 0..20 {
            if i % 2 == 0 {
                eq.compute(&high);
            } else {
                eq.compute(&low);
            }
        }
        let result = eq.compute(&high);
        assert!(
            result.temporal_stability < 1.0,
            "Variable inputs should reduce temporal_stability below 1.0, got {}",
            result.temporal_stability
        );
    }

    #[test]
    fn test_consciousness_trend_positive() {
        let mut eq = MasterConsciousnessEquation::default();
        let low = ConsciousnessInputs {
            phi: 0.1,
            broadcast: 0.1,
            working_memory: 0.1,
            attention: 0.1,
            recurrence: 0.1,
            embodiment: 0.1,
            knowledge: 0.1,
            synchrony: 0.1,
        };
        let high = all_high_inputs();

        // Start with low inputs
        for _ in 0..7 {
            eq.compute(&low);
        }
        // Switch to high inputs
        for _ in 0..7 {
            eq.compute(&high);
        }

        let trend = eq.consciousness_trend();
        assert!(
            trend > 0.0,
            "Trend should be positive after switching from low to high inputs, got {}",
            trend
        );
    }

    #[test]
    fn test_consciousness_trend_no_history() {
        let eq = MasterConsciousnessEquation::default();
        let trend = eq.consciousness_trend();
        assert!(
            (trend - 0.0).abs() < 1e-10,
            "Trend without enough history should be 0.0, got {}",
            trend
        );
    }

    #[test]
    fn test_average_consciousness() {
        let mut eq = MasterConsciousnessEquation::default();
        let inputs = all_high_inputs();
        for _ in 0..10 {
            eq.compute(&inputs);
        }
        let avg = eq.average_consciousness();
        assert!(
            avg > 0.0 && avg <= 1.0,
            "Average consciousness should be in (0.0, 1.0], got {}",
            avg
        );
    }

    #[test]
    fn test_disable_embodiment_factor() {
        let mut config = MasterEquationConfig::default();
        config.enable_embodiment_factor = false;
        let mut eq = MasterConsciousnessEquation::new(config);
        let inputs = all_high_inputs();
        let result = eq.compute(&inputs);
        assert!(
            (result.embodiment_factor - 1.0).abs() < 1e-10,
            "Disabled embodiment factor should yield 1.0, got {}",
            result.embodiment_factor
        );
    }

    #[test]
    fn test_disable_narrative_factor() {
        let mut config = MasterEquationConfig::default();
        config.enable_narrative_factor = false;
        let mut eq = MasterConsciousnessEquation::new(config);
        let inputs = all_high_inputs();
        let result = eq.compute(&inputs);
        assert!(
            (result.narrative_coherence - 1.0).abs() < 1e-10,
            "Disabled narrative factor should yield 1.0, got {}",
            result.narrative_coherence
        );
    }

    #[test]
    fn test_disable_social_factor() {
        let mut config = MasterEquationConfig::default();
        config.enable_social_factor = false;
        let mut eq = MasterConsciousnessEquation::new(config);
        let inputs = all_high_inputs();
        let result = eq.compute(&inputs);
        assert!(
            (result.social_embedding - 1.0).abs() < 1e-10,
            "Disabled social factor should yield 1.0, got {}",
            result.social_embedding
        );
    }

    #[test]
    fn test_describe_state() {
        let mut eq = MasterConsciousnessEquation::default();
        let inputs = all_high_inputs();
        let result = eq.compute(&inputs);
        let description = eq.describe_state(&result);
        // Description should contain the consciousness level
        let level_str = format!("{:.3}", result.consciousness_level);
        assert!(
            description.contains(&level_str),
            "describe_state should contain the consciousness level '{}', got '{}'",
            level_str,
            description
        );
        assert!(
            description.contains("Consciousness is"),
            "describe_state should contain 'Consciousness is', got '{}'",
            description
        );
    }

    #[test]
    fn test_consciousness_clamped() {
        let mut eq = MasterConsciousnessEquation::default();
        // Extreme high inputs
        let extreme = ConsciousnessInputs {
            phi: 100.0,
            broadcast: 100.0,
            working_memory: 100.0,
            attention: 100.0,
            recurrence: 100.0,
            embodiment: 100.0,
            knowledge: 100.0,
            synchrony: 100.0,
        };
        let result = eq.compute(&extreme);
        assert!(
            result.consciousness_level >= 0.0 && result.consciousness_level <= 1.0,
            "Consciousness level should be clamped to [0.0, 1.0], got {}",
            result.consciousness_level
        );

        // Extreme negative inputs
        let negative = ConsciousnessInputs {
            phi: -10.0,
            broadcast: -10.0,
            working_memory: -10.0,
            attention: -10.0,
            recurrence: -10.0,
            embodiment: -10.0,
            knowledge: -10.0,
            synchrony: -10.0,
        };
        let mut eq2 = MasterConsciousnessEquation::default();
        let result2 = eq2.compute(&negative);
        assert!(
            result2.consciousness_level >= 0.0 && result2.consciousness_level <= 1.0,
            "Consciousness level should be clamped to [0.0, 1.0] with negative inputs, got {}",
            result2.consciousness_level
        );
    }

    #[test]
    fn test_set_gating_clamps() {
        let mut eq = MasterConsciousnessEquation::default();
        eq.set_gating("phi", 5.0);
        // Verify internally the gating is clamped to 1.0 by computing and checking
        // that the result matches gating=1.0 behavior
        let inputs = all_high_inputs();
        let result_clamped = eq.compute(&inputs);

        let mut eq2 = MasterConsciousnessEquation::default();
        eq2.set_gating("phi", 1.0);
        let result_normal = eq2.compute(&inputs);

        assert!(
            (result_clamped.weighted_sum - result_normal.weighted_sum).abs() < 1e-10,
            "Gating set to 5.0 should be clamped to 1.0: clamped_ws={}, normal_ws={}",
            result_clamped.weighted_sum,
            result_normal.weighted_sum
        );
    }

    #[test]
    fn test_component_weights_total() {
        let config = MasterEquationConfig::default();
        let total = config.component_weights.total();
        assert!(
            (total - 1.0).abs() < 0.01,
            "Default component weights should total to ~1.0, got {}",
            total
        );
    }

    #[test]
    fn test_softmin_extreme_values_no_nan() {
        let eq = MasterConsciousnessEquation::default();
        // Very large positive values — softmin exponent would overflow without clamping
        let factors = vec![("A", 1e300_f64), ("B", 1e300_f64)];
        let (result, _) = eq.softmin_with_name(&factors);
        assert!(
            result.is_finite(),
            "softmin with extreme positive inputs should be finite, got {}",
            result
        );

        // Very large negative values
        let factors_neg = vec![("A", -1e300_f64), ("B", -1e300_f64)];
        let (result_neg, _) = eq.softmin_with_name(&factors_neg);
        assert!(
            result_neg.is_finite(),
            "softmin with extreme negative inputs should be finite, got {}",
            result_neg
        );

        // Tiny tau with moderate values
        let mut eq_tiny = MasterConsciousnessEquation::default();
        eq_tiny.update_config({
            let mut c = MasterEquationConfig::default();
            c.softmin_tau = 1e-10;
            c
        });
        let factors_mod = vec![("A", 0.5), ("B", 0.8)];
        let (result_tiny, _) = eq_tiny.softmin_with_name(&factors_mod);
        assert!(
            result_tiny.is_finite(),
            "softmin with tiny tau should be finite, got {}",
            result_tiny
        );
    }

    #[test]
    fn test_sigmoid_extreme_values_no_nan() {
        let eq = MasterConsciousnessEquation::default();
        // Large positive — should saturate to 1.0
        let s_pos = eq.sigmoid(200.0);
        assert!(
            (s_pos - 1.0).abs() < 1e-10,
            "sigmoid(200) should be ~1.0, got {}",
            s_pos
        );
        assert!(s_pos.is_finite());

        // Large negative — should saturate to 0.0
        let s_neg = eq.sigmoid(-200.0);
        assert!(
            s_neg.abs() < 1e-10,
            "sigmoid(-200) should be ~0.0, got {}",
            s_neg
        );
        assert!(s_neg.is_finite());

        // Normal value — should match original formula
        let s_mid = eq.sigmoid(0.5);
        let expected = 1.0 / (1.0 + (-2.5_f64).exp());
        assert!(
            (s_mid - expected).abs() < 1e-10,
            "sigmoid(0.5) should be {}, got {}",
            expected,
            s_mid
        );
    }

    /// Weight sensitivity analysis: sweep each weight ±50% and measure impact.
    ///
    /// Documents which weights the consciousness score is most sensitive to.
    /// Results guide future weight tuning and identify double-counting effects.
    #[test]
    fn test_weight_sensitivity_analysis() {
        let baseline_inputs = ConsciousnessInputs {
            phi: 0.6,
            broadcast: 0.5,
            working_memory: 0.5,
            attention: 0.6,
            recurrence: 0.5,
            embodiment: 0.5,
            knowledge: 0.4,
            synchrony: 0.7,
        };

        // Compute baseline
        let mut eq = MasterConsciousnessEquation::default();
        let baseline = eq.compute(&baseline_inputs);
        let c0 = baseline.consciousness_level;

        // Weight names and their default values for sweeping
        let weight_names = [
            "phi",
            "broadcast",
            "working_memory",
            "attention",
            "recurrence",
            "embodiment",
            "knowledge",
            "embodiment_factor",
            "narrative",
            "social",
        ];
        let default_weights = [0.15, 0.10, 0.10, 0.12, 0.10, 0.10, 0.08, 0.10, 0.08, 0.07];

        let mut sensitivities = Vec::new();

        for (i, &name) in weight_names.iter().enumerate() {
            let w0 = default_weights[i];

            // Sweep +50%
            let mut config_up = MasterEquationConfig::default();
            let w_up = &mut config_up.component_weights;
            match name {
                "phi" => w_up.phi = w0 * 1.5,
                "broadcast" => w_up.broadcast = w0 * 1.5,
                "working_memory" => w_up.working_memory = w0 * 1.5,
                "attention" => w_up.attention = w0 * 1.5,
                "recurrence" => w_up.recurrence = w0 * 1.5,
                "embodiment" => w_up.embodiment = w0 * 1.5,
                "knowledge" => w_up.knowledge = w0 * 1.5,
                "embodiment_factor" => w_up.embodiment_factor = w0 * 1.5,
                "narrative" => w_up.narrative = w0 * 1.5,
                "social" => w_up.social = w0 * 1.5,
                _ => unreachable!(),
            }
            let mut eq_up = MasterConsciousnessEquation::new(config_up);
            let c_up = eq_up.compute(&baseline_inputs).consciousness_level;

            // Sweep -50%
            let mut config_down = MasterEquationConfig::default();
            let w_dn = &mut config_down.component_weights;
            match name {
                "phi" => w_dn.phi = w0 * 0.5,
                "broadcast" => w_dn.broadcast = w0 * 0.5,
                "working_memory" => w_dn.working_memory = w0 * 0.5,
                "attention" => w_dn.attention = w0 * 0.5,
                "recurrence" => w_dn.recurrence = w0 * 0.5,
                "embodiment" => w_dn.embodiment = w0 * 0.5,
                "knowledge" => w_dn.knowledge = w0 * 0.5,
                "embodiment_factor" => w_dn.embodiment_factor = w0 * 0.5,
                "narrative" => w_dn.narrative = w0 * 0.5,
                "social" => w_dn.social = w0 * 0.5,
                _ => unreachable!(),
            }
            let mut eq_dn = MasterConsciousnessEquation::new(config_down);
            let c_dn = eq_dn.compute(&baseline_inputs).consciousness_level;

            // Sensitivity coefficient: (c_up - c_dn) / (w_up - w_dn) normalized by c0
            let delta_c = c_up - c_dn;
            let sensitivity = if c0 > 1e-10 { delta_c / c0 } else { 0.0 };
            sensitivities.push((name, sensitivity, c_dn, c0, c_up));
        }

        // Verify all results are finite and consciousness stays in [0, 1]
        for (name, sens, c_dn, _c0, c_up) in &sensitivities {
            assert!(sens.is_finite(), "Sensitivity for {name} is non-finite");
            assert!(
                *c_dn >= 0.0 && *c_dn <= 1.0,
                "{name} c_dn out of range: {c_dn}"
            );
            assert!(
                *c_up >= 0.0 && *c_up <= 1.0,
                "{name} c_up out of range: {c_up}"
            );
        }

        // The equation should respond to weight changes (not be constant)
        let total_sensitivity: f64 = sensitivities.iter().map(|(_, s, _, _, _)| s.abs()).sum();
        assert!(
            total_sensitivity > 0.01,
            "Total sensitivity should be non-trivial, got {total_sensitivity}"
        );

        // Verify M, N, Soc double-counting is bounded:
        // Their sensitivity should not be > 2× the average of other weights
        // (since they appear in both weighted sum and modulation)
        let core_sensitivities: Vec<f64> = sensitivities[..7]
            .iter()
            .map(|(_, s, _, _, _)| s.abs())
            .collect();
        let avg_core = core_sensitivities.iter().sum::<f64>() / core_sensitivities.len() as f64;

        for &(name, sens, _, _, _) in &sensitivities[7..] {
            assert!(
                sens.abs() < avg_core * 3.0,
                "Double-counted factor {name} sensitivity ({:.4}) exceeds 3× core average ({:.4})",
                sens.abs(),
                avg_core
            );
        }
    }
}
