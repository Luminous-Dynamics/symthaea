//! Multi-scale prediction, primitive state building, and consolidation.

use crate::consciousness::primitive_consciousness::{
    ActivationReason, ActivePrimitive, PrimitiveConsciousnessState,
};
use anyhow::Result;
use ndarray::Array1;

use super::CognitiveLoopService;

impl CognitiveLoopService {
    /// Get multi-scale prediction by averaging predictions at different time horizons
    ///
    /// This uses CfC's O(1) predict_forward to instantly query multiple future times,
    /// forcing the network to learn temporal "rules" rather than just noise patterns.
    pub(super) fn get_multi_scale_prediction(&mut self, input: &Array1<f32>) -> Vec<f32> {
        let horizons = &self.config.cfc_config.prediction_horizons;

        if horizons.is_empty() {
            // Fallback: single-step prediction
            return self
                .temporal_network
                .predict_forward(input, self.config.cfc_config.delta_t)
                .map(|arr| arr.to_vec())
                .unwrap_or_else(|_| vec![0.0; self.config.cfc_config.input_dim]);
        }

        // Collect predictions at multiple time horizons
        let mut predictions: Vec<Array1<f32>> = Vec::with_capacity(horizons.len());

        for &horizon in horizons {
            if let Ok(pred) = self.temporal_network.predict_forward(input, horizon) {
                predictions.push(pred);
            }
        }

        if predictions.is_empty() {
            return vec![0.0; self.config.cfc_config.input_dim];
        }

        // Average the multi-scale predictions
        // This forces temporal consistency across different timescales
        // Safe division: use max(1) to prevent division by zero
        let n = predictions.len().max(1) as f32;
        let dim = predictions[0].len();
        let mut result = vec![0.0f32; dim];

        for pred in &predictions {
            for (i, val) in pred.iter().enumerate() {
                if i < dim {
                    result[i] += val / n;
                }
            }
        }

        result
    }

    /// Compute coherence between multi-horizon predictions (0.0 = divergent, 1.0 = identical).
    ///
    /// Science: Bar (2009) — predictions at multiple time horizons should cohere when the
    /// model has captured genuine temporal structure (vs noise).
    /// Computes average pairwise cosine similarity across prediction horizons.
    pub(super) fn compute_prediction_coherence(&mut self, input: &Array1<f32>) -> f32 {
        let horizons = &self.config.cfc_config.prediction_horizons;
        if horizons.len() < 2 {
            return 1.0; // single horizon is trivially coherent
        }

        let mut predictions: Vec<Vec<f32>> = Vec::with_capacity(horizons.len());
        for &horizon in horizons {
            if let Ok(pred) = self.temporal_network.predict_forward(input, horizon) {
                predictions.push(pred.to_vec());
            }
        }

        if predictions.len() < 2 {
            return 1.0;
        }

        // Average pairwise cosine similarity
        let mut total_sim = 0.0f32;
        let mut pairs = 0u32;
        for i in 0..predictions.len() {
            for j in (i + 1)..predictions.len() {
                let mut dot = 0.0f32;
                let mut norm_a = 0.0f32;
                let mut norm_b = 0.0f32;
                for (a, b) in predictions[i].iter().zip(predictions[j].iter()) {
                    dot += a * b;
                    norm_a += a * a;
                    norm_b += b * b;
                }
                let denom = (norm_a.sqrt() * norm_b.sqrt()).max(1e-10);
                total_sim += (dot / denom).clamp(0.0, 1.0);
                pairs += 1;
            }
        }
        if pairs == 0 {
            1.0
        } else {
            total_sim / pairs as f32
        }
    }

    /// Build a lightweight [`PrimitiveConsciousnessState`] from detected primitive names.
    ///
    /// Maps primitive names to their most likely tier using keyword heuristics,
    /// then constructs `ActivePrimitive` entries with activation = 1.0 for each.
    pub(super) fn build_primitive_state(
        detected: &[String],
        phi: f64,
        timestamp: f64,
    ) -> PrimitiveConsciousnessState {
        use symthaea_core::hdc::BinaryHV;

        let mut state = PrimitiveConsciousnessState::new(timestamp);
        state.phi = phi;

        for name in detected {
            let tier = Self::classify_primitive_tier(name);
            let primitive = symthaea_core::hdc::primitive_system::Primitive {
                name: name.clone(),
                tier,
                domain: "detected".into(),
                encoding: BinaryHV::zero(),
                definition: String::new(),
                is_base: true,
                derivation: None,
            };
            let active = ActivePrimitive {
                primitive,
                activation: 1.0,
                activation_reason: ActivationReason::BottomUp {
                    input_similarity: 1.0,
                },
                duration: 1,
            };
            state.active_by_tier.entry(tier).or_default().push(active);
        }

        state
    }

    /// Classify a detected primitive name into its most likely tier.
    pub(super) fn classify_primitive_tier(
        name: &str,
    ) -> symthaea_core::hdc::primitive_system::PrimitiveTier {
        use symthaea_core::hdc::primitive_system::PrimitiveTier;

        let lower = name.to_lowercase();
        match lower.as_str() {
            "identity" | "bind" | "unbind" | "permute" | "bundle" | "protect" => PrimitiveTier::NSM,
            "addition" | "multiplication" | "implication" | "greater_than" | "less_than"
            | "equals" | "negation" => PrimitiveTier::Mathematical,
            "cause" | "effect" | "action" | "force" | "energy" => PrimitiveTier::Physical,
            "distance" | "angle" | "rotation" | "translation" | "manifold" => {
                PrimitiveTier::Geometric
            }
            "cooperate" | "compete" | "negotiate" | "trust" | "reciprocity" => {
                PrimitiveTier::Strategic
            }
            "reflect" | "metacognition" | "introspect" | "awareness" => {
                PrimitiveTier::MetaCognitive
            }
            "before" | "after" | "during" | "meets" | "overlaps" | "starts" | "finishes" => {
                PrimitiveTier::Temporal
            }
            "sequence" | "parallel" | "conditional" | "compose" | "recurse" => {
                PrimitiveTier::Compositional
            }
            _ => {
                // Fallback: try prefix matching
                if lower.starts_with("meta") {
                    PrimitiveTier::MetaCognitive
                } else if lower.starts_with("time") || lower.starts_with("temporal") {
                    PrimitiveTier::Temporal
                } else {
                    PrimitiveTier::NSM
                } // Default to NSM for unknown primitives
            }
        }
    }

    /// Run a background consolidation cycle
    ///
    /// This replays important experiences to strengthen learning using CfC.
    pub fn consolidate(&mut self) -> Result<f32> {
        if self.buffer.len() < 10 {
            return Ok(0.0);
        }

        self.is_consolidating = true;

        // Sort by importance and replay top experiences
        let mut experiences: Vec<_> = self.buffer.iter().collect();
        experiences.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut total_loss = 0.0;
        let replay_count = experiences.len().min(10);
        let delta_t = self.config.cfc_config.delta_t;
        let lr = self.config.cfc_config.learning_rate;

        for exp in experiences.iter().take(replay_count) {
            if let Some(ref next_state) = exp.next_state {
                // Reset CfC state for clean replay by injecting zeros
                let zeros = Array1::from_vec(vec![0.0f32; self.config.cfc_config.input_dim]);
                let _ = self.temporal_network.inject(&zeros);

                // Train using CfC's analytical gradient
                let prev_array = Array1::from_vec(exp.state.clone());
                let target_array = Array1::from_vec(next_state.clone());
                if let Ok(loss) =
                    self.temporal_network
                        .train_step(&prev_array, &target_array, delta_t, lr)
                {
                    total_loss += loss;
                }
            }
        }

        self.is_consolidating = false;

        Ok(total_loss / replay_count as f32)
    }
}
