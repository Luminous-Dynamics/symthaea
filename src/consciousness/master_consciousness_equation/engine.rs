use std::collections::VecDeque;
use std::time::Instant;

use super::config::MasterEquationConfig;
use super::embodiment::EmbodimentFactor;
use super::narrative::NarrativeCoherence;
use super::social::SocialEmbedding;
use super::types::{ConsciousnessInputs, ConsciousnessResult};

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
        // C(t) = σ(softmin) × weighted_sum × S × ρ(t) × M × N × Soc
        let consciousness_level =
            sigmoid_bottleneck * weighted_sum * inputs.synchrony * temporal_stability * m * n * soc;

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
