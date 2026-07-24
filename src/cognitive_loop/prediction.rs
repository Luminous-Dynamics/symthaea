// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-scale prediction, primitive state building, and consolidation.

use crate::consciousness::primitive_consciousness::{
    ActivationReason, ActivePrimitive, PrimitiveConsciousnessState,
};
use anyhow::Result;
use ndarray::Array1;

use super::CognitiveLoopService;

impl CognitiveLoopService {
    /// Get multi-scale prediction by averaging predictions at different time horizons.
    ///
    /// This uses CfC's O(1) predict_forward to instantly query multiple future times,
    /// forcing the network to learn temporal "rules" rather than just noise patterns.
    ///
    /// Returns `(averaged_prediction, raw_predictions)`. The raw predictions are retained
    /// so `compute_prediction_coherence_from_cache` can reuse them without re-calling
    /// predict_forward (saving ~300µs every 11 cycles).
    pub(super) fn get_multi_scale_prediction(
        &mut self,
        input: &Array1<f32>,
    ) -> (Vec<f32>, Vec<Array1<f32>>) {
        // Adaptive prediction horizons: high PE → contract toward immediate (can't
        // predict far when surprised); low PE → expand (plan further ahead).
        // Friston (2010): precision-weighting of temporal predictions — uncertain
        // states warrant shorter planning horizons.
        let pe_horizon_scale = {
            let pe = self.stats.avg_prediction_error.clamp(0.0, 1.0);
            if pe > 0.3 {
                // High PE: contract horizons to 60-100% of base
                1.0 - (pe - 0.3) * 0.6 // at PE=1.0 → 0.58
            } else if pe < 0.05 {
                // Very low PE: expand horizons up to 130%
                1.0 + (0.05 - pe) * 6.0 // at PE=0 → 1.30
            } else {
                1.0
            }
        };

        // Consciousness-gated horizon expansion: higher consciousness permits
        // longer planning horizons (virtuous cycle: C↑ → horizon↑ → prediction↑ → K↑ → C↑).
        // Science: Baars (2002) — conscious access enables extended temporal integration.
        // Scale: C=0 → 0.9 (mild contraction), C=0.7 → 1.07, C=1.0 → 1.15 (15% expansion).
        let consciousness_horizon_mod =
            0.9 + 0.25 * self.carryover.history.consciousness_level as f32;

        // Scale prediction horizons by substrate tau_factor, PE-adaptive factor,
        // and consciousness modulation. Clamp to prevent extreme horizons.
        use crate::cognitive_loop::thresholds::{
            PREDICTION_HORIZON_MAX_SCALE, PREDICTION_HORIZON_MIN_SCALE,
        };
        let combined_scale =
            (self.substrate_manager.tau_factor * pe_horizon_scale * consciousness_horizon_mod)
                .clamp(PREDICTION_HORIZON_MIN_SCALE, PREDICTION_HORIZON_MAX_SCALE);
        let effective_horizons: Vec<f32> = self
            .config
            .cfc_config
            .prediction_horizons
            .iter()
            .map(|h| h * combined_scale)
            .collect();
        let horizons = &effective_horizons;

        if horizons.is_empty() {
            // Fallback: single-step prediction
            let pred = self
                .temporal_network
                .predict_forward(input, self.config.cfc_config.delta_t)
                .map(|arr| arr.to_vec())
                .unwrap_or_else(|e| {
                    tracing::warn!(err = %e, cycle = self.stats.total_cycles,
                        "predict_forward failed (no horizons) — substituting zero prediction, PE will saturate");
                    vec![0.0; self.config.cfc_config.input_dim]
                });
            return (pred, Vec::new());
        }

        // Predictions must not perturb live temporal state.
        //
        // HdcLtc (the default backend): predict_forward is PURE — it snapshots
        // and restores internal neuron state itself, so no bookkeeping here.
        // Classic CfC: predict_forward advances state, so save/restore via
        // read_state/inject (a true restore on that backend).
        //
        // HISTORY: this dance used to run unconditionally — but on HdcLtc,
        // read_state captures only the small projected output and inject RESETS
        // the network, so the "restore" wiped the temporal state every cycle
        // (the liquid brain was memoryless). See docs/PHI_SIGNAL_TRACE_2026-07-15.md.
        let saved_state = if self.temporal_network.prediction_is_pure() {
            None
        } else {
            self.temporal_network.read_state().ok()
        };

        // Collect predictions at multiple time horizons
        let mut predictions: Vec<Array1<f32>> = Vec::with_capacity(horizons.len());

        for &horizon in horizons {
            // Restore state before each prediction so all horizons predict
            // from the same starting point (not cascading advances).
            if let Some(ref state) = saved_state {
                let _ = self.temporal_network.inject(state);
            }
            if let Ok(pred) = self.temporal_network.predict_forward(input, horizon) {
                predictions.push(pred);
            }
        }

        // Restore original state — predictions should be read-only w.r.t. network state
        if let Some(state) = saved_state {
            let _ = self.temporal_network.inject(&state);
        }

        if predictions.is_empty() {
            // Every horizon's predict_forward failed. A zero vector here makes the
            // encoder's PE saturate at the 1.0 sentinel — say so instead of hiding it.
            tracing::warn!(
                cycle = self.stats.total_cycles,
                horizons = horizons.len(),
                "all predict_forward horizons failed — substituting zero prediction, PE will saturate"
            );
            return (vec![0.0; self.config.cfc_config.input_dim], Vec::new());
        }

        // Average the multi-scale predictions with causal-informed per-dimension weighting.
        // Dimensions with known causal edges are more predictable at longer horizons,
        // so they get more weight from distant predictions (Granger 1969, Pearl 2009).
        let n = predictions.len().max(1);
        let dim = predictions[0].len();
        let mut result = vec![0.0f32; dim];

        // Build per-dimension causal weight: dimensions involved in causal edges
        // get up-weighted for longer-horizon predictions (bias toward later horizons).
        // Only re-query the causal graph every 41 cycles (co-prime with world model update)
        // to avoid redundant graph copies. Between updates, use uniform weighting.
        let causal_dims: Option<Vec<bool>> = if self.stats.total_cycles % 41 == 0 {
            if let Some(ref enhancer) = self.memory.causal_enhancer {
                let graph = enhancer.current_graph();
                let mut has_edge = vec![false; dim];
                for e in &graph.edges {
                    if e.from < dim {
                        has_edge[e.from] = true;
                    }
                    if e.to < dim {
                        has_edge[e.to] = true;
                    }
                }
                // Only allocate if there are actual causal edges
                if has_edge.iter().any(|&b| b) {
                    Some(has_edge)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let inv_n = 1.0 / n as f32;
        if let Some(ref causal) = causal_dims {
            // Causal-weighted path (only on 41st cycle with actual edges)
            for (pred_idx, pred) in predictions.iter().enumerate() {
                let horizon_frac = (pred_idx + 1) as f32 / n as f32;
                for (i, val) in pred.iter().enumerate() {
                    if i < dim {
                        let w = if causal[i] {
                            (0.5 + horizon_frac) * inv_n
                        } else {
                            inv_n
                        };
                        result[i] += val * w;
                    }
                }
            }
        } else {
            // Uniform-weight fast path (no causal edges, 40/41 cycles)
            for pred in predictions.iter() {
                for (i, val) in pred.iter().enumerate() {
                    if i < dim {
                        result[i] += val * inv_n;
                    }
                }
            }
        }

        (result, predictions)
    }

    /// Compute coherence from cached predictions (avoids re-calling predict_forward).
    ///
    /// Takes the raw predictions already computed by `get_multi_scale_prediction`.
    /// Computes average pairwise cosine similarity across prediction horizons.
    pub(super) fn compute_prediction_coherence_from_cache(predictions: &[Array1<f32>]) -> f32 {
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
                let sim = dot / denom;
                total_sim += if sim.is_finite() {
                    sim.clamp(0.0, 1.0)
                } else {
                    0.0
                };
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

        // Sort by importance × epistemic value and replay top experiences.
        // Active inference: prioritize experiences where the model was most
        // uncertain (high PE = high epistemic value), weighted by importance.
        // Dyna-Q (Sutton 1991): model-based replay preferentially samples
        // transitions with high expected learning gain.
        let mut experiences: Vec<_> = self.buffer.iter().collect();
        experiences.sort_by(|a, b| {
            let a_score = a.importance * (1.0 + a.error); // high PE → more informative
            let b_score = b.importance * (1.0 + b.error);
            b_score
                .partial_cmp(&a_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut total_loss = 0.0;
        let replay_count = experiences.len().min(10);
        let delta_t = self.config.cfc_config.delta_t;
        let lr = self.config.cfc_config.learning_rate;

        // Replay deliberately resets state per experience for clean training —
        // but the LIVE cognitive state must survive consolidation. Snapshot it
        // now and restore after (previously it was silently destroyed every
        // consolidation; see docs/PHI_SIGNAL_TRACE_2026-07-15.md).
        let live_state = self.temporal_network.save_evolution_state();

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

        if let Some(ref backup) = live_state {
            self.temporal_network.restore_evolution_state(backup);
        }

        self.is_consolidating = false;

        Ok(total_loss / replay_count as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use symthaea_core::hdc::primitive_system::PrimitiveTier;

    // ---- compute_prediction_coherence_from_cache ----

    #[test]
    fn test_coherence_empty_predictions() {
        let predictions: &[Array1<f32>] = &[];
        let coherence = CognitiveLoopService::compute_prediction_coherence_from_cache(predictions);
        assert!(
            (coherence - 1.0).abs() < f32::EPSILON,
            "empty predictions should return 1.0, got {coherence}"
        );
    }

    #[test]
    fn test_coherence_single_prediction() {
        let predictions = [Array1::from_vec(vec![1.0, 2.0, 3.0])];
        let coherence = CognitiveLoopService::compute_prediction_coherence_from_cache(&predictions);
        assert!(
            (coherence - 1.0).abs() < f32::EPSILON,
            "single prediction should return 1.0, got {coherence}"
        );
    }

    #[test]
    fn test_coherence_identical_predictions() {
        let v = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let predictions = [v.clone(), v];
        let coherence = CognitiveLoopService::compute_prediction_coherence_from_cache(&predictions);
        assert!(
            (coherence - 1.0).abs() < 1e-5,
            "identical predictions should yield coherence ~1.0, got {coherence}"
        );
    }

    #[test]
    fn test_coherence_orthogonal_predictions() {
        // [1, 0, 0] and [0, 1, 0] are orthogonal => cosine = 0
        let a = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let b = Array1::from_vec(vec![0.0, 1.0, 0.0]);
        let predictions = [a, b];
        let coherence = CognitiveLoopService::compute_prediction_coherence_from_cache(&predictions);
        assert!(
            coherence < 0.01,
            "orthogonal predictions should yield coherence ~0.0, got {coherence}"
        );
    }

    #[test]
    fn test_coherence_opposite_predictions() {
        // [1, 2, 3] and [-1, -2, -3] => cosine = -1, clamped to 0
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![-1.0, -2.0, -3.0]);
        let predictions = [a, b];
        let coherence = CognitiveLoopService::compute_prediction_coherence_from_cache(&predictions);
        assert!(
            coherence.abs() < 1e-5,
            "opposite predictions should yield coherence ~0.0 (clamped), got {coherence}"
        );
    }

    // ---- classify_primitive_tier ----

    #[test]
    fn test_classify_nsm_primitives() {
        for name in &["identity", "bind", "unbind"] {
            let tier = CognitiveLoopService::classify_primitive_tier(name);
            assert_eq!(tier, PrimitiveTier::NSM, "{name} should be NSM");
        }
    }

    #[test]
    fn test_classify_mathematical_primitives() {
        for name in &["addition", "multiplication"] {
            let tier = CognitiveLoopService::classify_primitive_tier(name);
            assert_eq!(
                tier,
                PrimitiveTier::Mathematical,
                "{name} should be Mathematical"
            );
        }
    }

    #[test]
    fn test_classify_physical_primitives() {
        for name in &["cause", "effect"] {
            let tier = CognitiveLoopService::classify_primitive_tier(name);
            assert_eq!(tier, PrimitiveTier::Physical, "{name} should be Physical");
        }
    }

    #[test]
    fn test_classify_temporal_primitives() {
        for name in &["before", "after", "during"] {
            let tier = CognitiveLoopService::classify_primitive_tier(name);
            assert_eq!(tier, PrimitiveTier::Temporal, "{name} should be Temporal");
        }
    }

    #[test]
    fn test_classify_unknown_defaults_to_nsm() {
        let tier = CognitiveLoopService::classify_primitive_tier("foobar");
        assert_eq!(tier, PrimitiveTier::NSM, "unknown should default to NSM");
    }

    #[test]
    fn test_classify_prefix_metacognitive() {
        let tier = CognitiveLoopService::classify_primitive_tier("metacognition_xyz");
        assert_eq!(
            tier,
            PrimitiveTier::MetaCognitive,
            "meta-prefixed should be MetaCognitive"
        );
    }

    // ---- build_primitive_state ----

    #[test]
    fn test_build_primitive_state_empty() {
        let state = CognitiveLoopService::build_primitive_state(&[], 0.5, 100.0);
        assert!(
            state.active_by_tier.is_empty(),
            "empty detected list should produce no active primitives"
        );
        assert!((state.phi - 0.5).abs() < f64::EPSILON, "phi should be set");
        assert!(
            (state.timestamp - 100.0).abs() < f64::EPSILON,
            "timestamp should be set"
        );
    }

    #[test]
    fn test_build_primitive_state_populated() {
        let detected = vec!["cause".to_string(), "bind".to_string()];
        let state = CognitiveLoopService::build_primitive_state(&detected, 0.8, 42.0);

        // Should have entries in Physical (cause) and NSM (bind)
        assert!(
            state.active_by_tier.contains_key(&PrimitiveTier::Physical),
            "should contain Physical tier for 'cause'"
        );
        assert!(
            state.active_by_tier.contains_key(&PrimitiveTier::NSM),
            "should contain NSM tier for 'bind'"
        );

        let physical = &state.active_by_tier[&PrimitiveTier::Physical];
        assert_eq!(physical.len(), 1, "one Physical primitive expected");
        assert_eq!(physical[0].primitive.name, "cause");
        assert!((physical[0].activation - 1.0).abs() < f64::EPSILON);

        let nsm = &state.active_by_tier[&PrimitiveTier::NSM];
        assert_eq!(nsm.len(), 1, "one NSM primitive expected");
        assert_eq!(nsm[0].primitive.name, "bind");

        assert!((state.phi - 0.8).abs() < f64::EPSILON);
        assert!((state.timestamp - 42.0).abs() < f64::EPSILON);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Epistemic vs aleatoric uncertainty decomposition (Item #7)
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_coherence_high_implies_low_epistemic() {
        // Identical predictions → high coherence → low epistemic uncertainty
        let v = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let predictions = [v.clone(), v.clone(), v];
        let coherence = CognitiveLoopService::compute_prediction_coherence_from_cache(&predictions);
        let epistemic = 1.0 - coherence;
        assert!(
            epistemic < 0.01,
            "identical predictions → epistemic ≈ 0, got {epistemic}"
        );
    }

    #[test]
    fn test_coherence_low_implies_high_epistemic() {
        // Orthogonal predictions → low coherence → high epistemic uncertainty
        let a = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let b = Array1::from_vec(vec![0.0, 1.0, 0.0]);
        let c = Array1::from_vec(vec![0.0, 0.0, 1.0]);
        let predictions = [a, b, c];
        let coherence = CognitiveLoopService::compute_prediction_coherence_from_cache(&predictions);
        let epistemic = 1.0 - coherence;
        assert!(
            epistemic > 0.9,
            "orthogonal predictions → epistemic ≈ 1.0, got {epistemic}"
        );
    }

    #[test]
    fn test_aleatoric_from_per_dimension_variance() {
        // Two predictions with high per-dimension variance
        let a = Array1::from_vec(vec![0.0, 1.0, 0.0]);
        let b = Array1::from_vec(vec![0.0, -1.0, 0.0]);
        let predictions = [a, b];
        let dim = 3;
        let n = predictions.len() as f32;
        let mut mean_var = 0.0f32;
        for d in 0..dim {
            let mean: f32 = predictions.iter().map(|p| p[d]).sum::<f32>() / n;
            let var: f32 = predictions
                .iter()
                .map(|p| (p[d] - mean).powi(2))
                .sum::<f32>()
                / n;
            mean_var += var;
        }
        let aleatoric = (mean_var / dim as f32).sqrt();
        assert!(
            aleatoric > 0.3,
            "high per-dim variance → aleatoric > 0.3, got {aleatoric}"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Consolidation replay priority (Item #7)
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_consolidation_sort_prefers_high_error() {
        // Simulate the sort logic: importance × (1 + error)
        struct Exp {
            importance: f32,
            error: f32,
        }
        let mut exps = vec![
            Exp {
                importance: 0.8,
                error: 0.1,
            }, // score = 0.88
            Exp {
                importance: 0.5,
                error: 0.9,
            }, // score = 0.95 (high error wins)
            Exp {
                importance: 0.9,
                error: 0.0,
            }, // score = 0.90
        ];
        exps.sort_by(|a, b| {
            let a_score = a.importance * (1.0 + a.error);
            let b_score = b.importance * (1.0 + b.error);
            b_score
                .partial_cmp(&a_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        // First should be the high-error experience (0.95 > 0.90 > 0.88)
        assert!(
            (exps[0].error - 0.9).abs() < 0.01,
            "highest-error experience should be first, got error={}",
            exps[0].error
        );
        assert!(
            (exps[1].importance - 0.9).abs() < 0.01,
            "second should be high-importance/low-error"
        );
    }

    #[test]
    fn test_consolidation_sort_equal_importance_high_error_wins() {
        struct Exp {
            importance: f32,
            error: f32,
        }
        let mut exps = vec![
            Exp {
                importance: 1.0,
                error: 0.1,
            }, // score = 1.1
            Exp {
                importance: 1.0,
                error: 0.8,
            }, // score = 1.8
        ];
        exps.sort_by(|a, b| {
            let a_score = a.importance * (1.0 + a.error);
            let b_score = b.importance * (1.0 + b.error);
            b_score
                .partial_cmp(&a_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        assert!(
            exps[0].error > exps[1].error,
            "with equal importance, high-error should be first"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Causal prediction weight normalization (Item #4)
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_causal_weights_sum_correctly_non_causal() {
        // For non-causal dims, weight per prediction = 1/n → total = 1.0
        let n = 4usize;
        let total: f32 = (0..n).map(|_| 1.0 / n as f32).sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "non-causal weights should sum to 1.0: {total}"
        );
    }

    #[test]
    fn test_causal_weights_sum_correctly_causal() {
        // For causal dims, weight = (0.5 + horizon_frac) / n where horizon_frac = (idx+1)/n
        // Sum should be: Σ (0.5 + (i+1)/n) / n for i=0..n-1
        //   = (0.5*n + Σ(i+1)/n) / n = (0.5*n + (n+1)/2) / n = 0.5 + (n+1)/(2n)
        // For n=4: 0.5 + 5/8 = 1.125 — NOT 1.0. This is intentional:
        // causal dims get ~12.5% more total weight (emphasis on long horizons).
        let n = 4usize;
        let total: f32 = (0..n)
            .map(|i| {
                let frac = (i + 1) as f32 / n as f32;
                (0.5 + frac) / n as f32
            })
            .sum();
        // Verify the bias is bounded and positive (causal dims get MORE weight, not less)
        assert!(total > 1.0, "causal weights should exceed 1.0: {total}");
        assert!(
            total < 1.5,
            "causal weight excess should be bounded: {total}"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Prediction horizon bounds (Item #8)
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_horizon_scale_clamped_at_extremes() {
        use crate::cognitive_loop::thresholds::{
            PREDICTION_HORIZON_MAX_SCALE, PREDICTION_HORIZON_MIN_SCALE,
        };

        // Very high PE → contract horizons, but not below floor
        let pe = 1.0f32;
        let pe_scale = 1.0 - (pe - 0.3) * 0.6; // = 0.58
        let tau = 0.3f32; // very slow substrate
        let combined =
            (tau * pe_scale).clamp(PREDICTION_HORIZON_MIN_SCALE, PREDICTION_HORIZON_MAX_SCALE);
        assert!(
            combined >= PREDICTION_HORIZON_MIN_SCALE,
            "combined scale should not go below floor: {combined}"
        );

        // Very low PE + fast substrate → expand, but not above ceiling
        let _pe = 0.0f32;
        let pe_scale = 1.0 + 0.05 * 6.0; // = 1.30
        let tau = 3.0f32; // extremely fast substrate
        let combined =
            (tau * pe_scale).clamp(PREDICTION_HORIZON_MIN_SCALE, PREDICTION_HORIZON_MAX_SCALE);
        assert!(
            combined <= PREDICTION_HORIZON_MAX_SCALE,
            "combined scale should not exceed ceiling: {combined}"
        );
    }
}
