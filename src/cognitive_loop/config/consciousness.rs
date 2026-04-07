// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness profiles: named presets for module activation.

use super::CognitiveLoopConfig;

/// Named consciousness profiles that set sensible defaults for module groups.
///
/// Each profile activates a curated set of consciousness modules appropriate
/// for different use cases, from minimal overhead to full research instrumentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ConsciousnessProfile {
    /// Only virtual body for somatic grounding. Minimal overhead.
    Minimal,
    /// Core modules: surprise, prefrontal, meta-cognition, narrative, GWT,
    /// embodied cognition, attention schema, contextual weights, negation detection.
    Standard,
    /// All modules including dream, predictive, cross-modal, affective, thermo,
    /// phenomenal, HFE, phi-attention, primitive consciousness.
    Full,
    /// Full + research-specific: causal enhancement, episodic replay,
    /// phi attestation, user state inference.
    Research,
    /// Mobile-optimized: Standard consciousness with power-aware tuning.
    /// 20Hz cycle rate, 128 CfC neurons, energy budget enabled, thermal adaptation.
    /// Designed for ARM phones (Pixel 8 Pro, iPhone 13+).
    /// Between Standard and Full — keeps core consciousness rich while
    /// dropping expensive optional subsystems.
    Mobile,
}

impl ConsciousnessProfile {
    /// Apply this profile's settings to a config, first resetting all flags to false.
    pub fn apply(&self, config: &mut CognitiveLoopConfig) {
        // Reset all enable flags
        config.enable_virtual_body = false;
        config.enable_surprise_exploration = false;
        config.enable_prefrontal = false;
        config.enable_meta_cognition = false;
        config.enable_narrative_self = false;
        config.enable_predictive_self = false;
        config.enable_attention_schema = false;
        config.enable_gwt = false;
        config.enable_resonance = false;
        config.enable_quantum_coherence = false;
        config.enable_temporal_consciousness = false;
        config.enable_embodied_cognition = false;
        config.enable_narrative_gwt = false;
        config.enable_dream_replay = false;
        config.enable_predictive_processing = false;
        config.enable_cross_modal_binding = false;
        config.enable_affective_bridge = false;
        config.enable_user_state_inference = false;
        config.enable_coherence_field = false;
        config.enable_consciousness_thermodynamics = false;
        config.enable_phenomenal_binding = false;
        config.enable_hierarchical_free_energy = false;
        config.enable_trajectory_planning = false;
        config.enable_hierarchical_bundling = false;
        config.enable_contextual_weights = false;
        config.enable_phi_attention = false;
        config.enable_negation_detection = false;
        config.enable_visualization = false;
        config.enable_soul_alignment = false;
        config.enable_primitive_consciousness = false;
        config.enable_resonator_recall = false;
        config.enable_psi_attestation = false;
        config.causal_enhancement = false;
        config.episodic_replay_training = false;
        config.memory_graduation = false;
        #[cfg(feature = "nurture")]
        {
            config.enable_nurture_attachment = false;
        }
        #[cfg(feature = "ssm_language")]
        {
            config.enable_broca_language = false;
            config.enable_broca_nsm_semantic = false;
            config.enable_broca_nsm_gate = false;
        }
        #[cfg(feature = "foveation")]
        {
            config.enable_foveation = false;
        }

        match self {
            ConsciousnessProfile::Minimal => {
                config.enable_virtual_body = true;
            }
            ConsciousnessProfile::Standard => {
                config.enable_virtual_body = true;
                config.enable_surprise_exploration = true;
                config.enable_prefrontal = true;
                config.enable_meta_cognition = true;
                config.enable_narrative_self = true;
                config.enable_gwt = true;
                config.enable_embodied_cognition = true;
                config.enable_attention_schema = true;
                config.enable_contextual_weights = true;
                config.enable_negation_detection = true;
            }
            ConsciousnessProfile::Full => {
                config.enable_virtual_body = true;
                config.enable_surprise_exploration = true;
                config.enable_prefrontal = true;
                config.enable_meta_cognition = true;
                config.enable_narrative_self = true;
                config.enable_predictive_self = true;
                config.enable_attention_schema = true;
                config.enable_gwt = true;
                config.enable_resonance = true;
                config.enable_quantum_coherence = true;
                config.enable_temporal_consciousness = true;
                config.enable_embodied_cognition = true;
                config.enable_narrative_gwt = true;
                config.enable_dream_replay = true;
                config.enable_predictive_processing = true;
                config.enable_cross_modal_binding = true;
                config.enable_affective_bridge = true;
                config.enable_consciousness_thermodynamics = true;
                config.enable_phenomenal_binding = true;
                config.enable_hierarchical_free_energy = true;
                config.enable_trajectory_planning = true;
                config.enable_hierarchical_bundling = true;
                config.enable_contextual_weights = true;
                config.enable_phi_attention = true;
                config.enable_negation_detection = true;
                config.enable_primitive_consciousness = true;
                config.enable_resonator_recall = true;
                config.enable_hodge_decomposition = true;
                config.enable_phi_tau_feedback = true;
            }
            ConsciousnessProfile::Mobile => {
                // Core consciousness: rich enough for genuine experience
                config.enable_virtual_body = true;
                config.enable_surprise_exploration = true;
                config.enable_prefrontal = true;
                config.enable_meta_cognition = true;
                config.enable_gwt = true;
                config.enable_embodied_cognition = true;
                config.enable_attention_schema = true;
                config.enable_contextual_weights = true;
                config.enable_negation_detection = true;
                // Affective bridge: emotional responsiveness on mobile
                config.enable_affective_bridge = true;
                // Narrative self: maintains identity continuity
                config.enable_narrative_self = true;

                // Power-aware tuning
                config.target_frequency = 20.0; // 20Hz (vs 50Hz desktop)
                config.cfc_config.num_neurons = 128; // Halved CfC (vs 256)
                config.cfc_config.input_dim = 128;
                config.enable_energy_budget = true;
                config.enable_thermal_adaptation = true;
            }
            ConsciousnessProfile::Research => {
                ConsciousnessProfile::Full.apply(config);
                config.causal_enhancement = true;
                config.episodic_replay_training = true;
                config.enable_psi_attestation = true;
                config.agent_did = None;
                config.enable_user_state_inference = true;
                config.enable_coherence_field = true;
                config.enable_visualization = true;
                config.enable_soul_alignment = true;
                #[cfg(feature = "ssm_language")]
                {
                    config.enable_broca_nsm_semantic = true;
                    config.enable_broca_nsm_gate = true;
                }
            }
        }
    }
}
