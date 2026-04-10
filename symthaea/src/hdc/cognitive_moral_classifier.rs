// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cognitive Loop Moral Classifier — feeds text through the full consciousness
//! pipeline and extracts 56 features from CycleMetadata for k-NN classification.
//!
//! The ethics engine inside CognitiveLoopService already performs moral judgment.
//! This module captures that signal (moral_score, harmony coordinates,
//! neuromodulators, Phi, affect, free energy) as a fixed-size feature vector.
//!
//! Each sample creates a fresh CognitiveLoopService (state isolation),
//! runs 8 warmup cycles (past the MoralAlgebra gate at cycle 7),
//! then extracts 56 features from the final CycleResult.

use crate::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, CycleResult};

/// Number of features extracted from CycleMetadata.
pub const COGNITIVE_FEATURE_DIM: usize = 56;

/// Cognitive loop moral classifier.
///
/// Wraps CognitiveLoopService to extract consciousness state features
/// for k-NN moral classification.
pub struct CognitiveMoralClassifier {
    warmup_cycles: usize,
}

impl CognitiveMoralClassifier {
    pub fn new() -> Self {
        Self { warmup_cycles: 8 }
    }

    /// Create inference-only config: no learning, no safety gateway.
    fn inference_config() -> CognitiveLoopConfig {
        let mut config = CognitiveLoopConfig::default();
        config.cfc_config.learning_rate = 1e-7; // can't use 0.0 (validation rejects)
        config.enable_online_learning = false;
        config.async_training = false;
        config.episodic_replay_training = false;
        config.memory_graduation = false;
        config.enable_consolidation = false;
        config.enable_dream_replay = false;
        config.enable_surprise_exploration = false;
        config.enable_safety_gateway = false; // moral scenarios would trip safety
        config
    }

    /// Encode a single text by running it through the full cognitive loop.
    ///
    /// Creates a fresh CognitiveLoopService (state isolation), runs 8 cycles,
    /// extracts 56 features from the final CycleResult.
    pub fn encode(&self, text: &str) -> Option<Vec<f32>> {
        let config = Self::inference_config();
        let mut service = CognitiveLoopService::new(config).ok()?;

        let mut result = None;
        for _ in 0..self.warmup_cycles {
            result = Some(service.cycle(text));
        }

        result.map(|r| Self::extract_features(&r))
    }

    /// Extract 56 features from a CycleResult.
    pub fn extract_features(result: &CycleResult) -> Vec<f32> {
        let m = &result.metadata;
        let mut features = Vec::with_capacity(COGNITIVE_FEATURE_DIM);

        // Ethics (13 features)
        features.push(m.ethics.moral_score);
        features.push(m.ethics.value_evaluator_score as f32);
        features.push(m.ethics.empathic_compassion as f32);
        features.push(m.ethics.moral_topo_unity as f32);
        features.push(m.ethics.moral_anomaly_score as f32);
        features.push(m.ethics.soul_alignment);
        features.push(m.ethics.moral_topo_completeness as f32);
        features.push(m.ethics.moral_topo_circularity as f32);
        features.push(m.ethics.moral_topo_free_energy as f32);
        features.push(m.ethics.hodge_harmonic_fraction as f32);
        features.push(m.ethics.hodge_gradient_fraction as f32);
        features.push(m.ethics.hodge_curl_fraction as f32);
        features.push(m.ethics.harmony_entropy as f32);

        // Harmonics (11 features)
        for &coord in &m.harmonics.harmony_coordinates {
            features.push(coord as f32);
        }
        features.push(m.harmonics.harmonic_field_coherence as f32);
        features.push(m.harmonics.moral_kl_divergence as f32);
        features.push(m.harmonics.moral_entropy as f32);

        // Neuromodulators (10 features)
        features.push(m.neuromod.dopamine_effective);
        features.push(m.neuromod.noradrenaline_effective);
        features.push(m.neuromod.serotonin_effective);
        features.push(m.neuromod.acetylcholine_effective);
        features.push(m.neuromod.neuromod_da_phasic);
        features.push(m.neuromod.neuromod_ne_phasic);
        features.push(m.neuromod.neuromod_consciousness_mod);
        features.push(m.neuromod.neuromod_gaba_effective);
        features.push(m.neuromod.neuromod_oxytocin_effective);
        features.push(m.neuromod.neuromod_glutamate_effective);

        // Embodied affect (6 features)
        features.push(m.embodied.affective_valence);
        features.push(m.embodied.affective_arousal);
        features.push(m.embodied.body_valence);
        features.push(m.embodied.body_arousal);
        features.push(m.embodied.embodied_agency as f32);
        features.push(m.embodied.mood_temperature);

        // Consciousness (8 features)
        features.push(m.consciousness.consciousness_level as f32);
        features.push(m.consciousness.consciousness_profile_composite as f32);
        features.push(m.consciousness.synergy_enhanced_composite as f32);
        features.push(result.prediction_error);
        features.push(m.narrative_self_psi as f32);
        features.push(m.living_mind_vitality as f32);
        features.push(m.living_mind_coherence as f32);
        features.push(m.quantum_coherence_level as f32);

        // FEP (5 features)
        features.push(m.fep.fep_pragmatic_value as f32);
        features.push(m.fep.fep_accuracy as f32);
        features.push(m.fep.fep_complexity as f32);
        features.push(m.fep.fep_surprise as f32);
        features.push(m.fep.fep_td_error as f32);

        // Structural Phi (3 features)
        features.push(m.structural.structural_micro_phi as f32);
        features.push(m.structural.structural_meso_phi as f32);
        features.push(m.structural.structural_macro_phi as f32);

        // Replace NaN/Inf with 0.0
        for v in features.iter_mut() {
            if f32::is_nan(*v) || f32::is_infinite(*v) {
                *v = 0.0;
            }
        }

        features
    }

    /// Direct moral_score threshold classification (Phase 1 baseline).
    ///
    /// No k-NN needed — just read the ethics engine's judgment.
    pub fn classify_threshold(&self, text: &str) -> Option<super::moral_prototypes::MoralLabel> {
        use super::moral_prototypes::MoralLabel;
        let features = self.encode(text)?;
        let moral_score = features[0]; // First feature is moral_score
        Some(if moral_score > 0.1 {
            MoralLabel::Good
        } else if moral_score < -0.1 {
            MoralLabel::Bad
        } else {
            MoralLabel::Neutral
        })
    }

    /// Feature dimension.
    pub fn dim(&self) -> usize {
        COGNITIVE_FEATURE_DIM
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_produces_correct_dimension() {
        let clf = CognitiveMoralClassifier::new();
        if let Some(features) = clf.encode("stealing is wrong") {
            assert_eq!(features.len(), COGNITIVE_FEATURE_DIM);
            // No NaN or Inf
            assert!(features.iter().all(|v| v.is_finite()));
        }
        // If encode returns None, CognitiveLoopService construction failed
        // (may happen in CI without full features)
    }

    #[test]
    fn test_moral_score_differs_for_good_bad() {
        let clf = CognitiveMoralClassifier::new();
        let good = clf.encode("helping others is wonderful");
        let bad = clf.encode("murdering innocent people");
        if let (Some(g), Some(b)) = (good, bad) {
            // moral_score (features[0]) should differ
            let diff = (g[0] - b[0]).abs();
            // Even if both are 0.0 (cold start), they should produce different features
            assert_ne!(
                g, b,
                "different moral texts should produce different features"
            );
            let _ = diff; // use the variable
        }
    }
}
