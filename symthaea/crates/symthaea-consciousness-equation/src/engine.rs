use std::collections::VecDeque;
use std::time::Instant;

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
        let weighted_sum = self.compute_weighted_sum(inputs, m, n, soc);

        // Step 4: Compute temporal stability ρ(t)
        let temporal_stability = self.compute_temporal_stability();

        // Step 5: Final consciousness level with new factors
        // C(t) = σ(softmin) × weighted_sum × S × ρ(t) × M' × N' × Soc'
        //
        // M, N, Soc are already included in the weighted_sum via their component weights.
        // As raw multiplicatives they double-count AND cause catastrophic attenuation when
        // a factor is low (e.g., Soc=0.35 in non-social context → 65% permanent haircut).
        // Convert to soft modulations: map [0,1] → [0.5, 1.0] so low values attenuate
        // gently rather than crushing consciousness.
        // Science: Modular consciousness theories (Baars 2005) — subsystem deficits
        // reduce but don't eliminate consciousness.
        let m_mod = 0.5 + 0.5 * m;
        let n_mod = 0.5 + 0.5 * n;
        let soc_mod = 0.5 + 0.5 * soc;
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
    /// This is a smooth minimum that identifies the bottleneck
    fn softmin_with_name(&self, factors: &[(&str, f64)]) -> (f64, String) {
        let tau = self.config.softmin_tau;
        let epsilon = self.config.epsilon;

        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        let mut min_val = f64::MAX;
        let mut min_name = "Unknown".to_string();

        for (name, val) in factors {
            let weight = (-val / tau).exp();
            weighted_sum += val * weight;
            weight_sum += weight;

            if *val < min_val {
                min_val = *val;
                min_name = name.to_string();
            }
        }

        let softmin = if weight_sum > epsilon {
            weighted_sum / weight_sum
        } else {
            min_val
        };

        (softmin, min_name)
    }

    /// Sigmoid function: σ(x) = 1 / (1 + exp(-x))
    fn sigmoid(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x * 5.0).exp()) // Scaled for [0,1] inputs
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
}
