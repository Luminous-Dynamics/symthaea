// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Language & Communication Manager — groups voice coherence, Broca language,
//! and user state inference into a single CLS field.

/// Consolidated manager for language and communication subsystems.
///
/// Replaces 4 top-level CLS fields:
/// - `voice_coherence` — CfC coherence + voice feedback + temporal signatures
/// - `broca_manager` — SSM language generation (feature-gated)
/// - `broca_lite` — lightweight always-on language fallback
/// - `last_broca_text` — most recent Broca output
/// - `user_state` — user state inference for adaptive responses
pub struct LanguageAndCommunicationManager {
    /// Voice-coherence bridge: CfC coherence + voice feedback + temporal signatures.
    pub voice_coherence: super::voice_coherence_bridge::VoiceCoherenceBridge,

    /// Broca SSM language center: consciousness-gated thought-to-text.
    #[cfg(feature = "ssm_language")]
    pub broca_manager: Option<super::broca_bridge::BrocaManager>,

    /// BrocaLite fallback: lightweight always-on language generation (512 tokens, 1024D).
    /// Used when `ssm_language` is not enabled or full Broca doesn't fire.
    pub broca_lite: super::broca_lite::BrocaLiteManager,

    /// Most recent Broca-generated text, drained into `CycleResult.language_output`.
    /// Always present (unconditional) — populated by full Broca or BrocaLite fallback.
    pub last_broca_text: Option<String>,

    /// Source tier that produced `last_broca_text`: "broca_lite", "broca", or "llm".
    pub last_language_source: Option<String>,

    /// User state inference for adaptive response generation.
    pub user_state: Option<crate::user_state_inference::UserStateInference>,
    /// Code channels injected by CodingAgent for Broca's CodeGate.
    /// `[syntax_complexity, type_confidence, algorithm_pattern, error_likelihood]`
    pub broca_code_channels: Option<[f32; 4]>,
}

impl std::fmt::Debug for LanguageAndCommunicationManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LanguageAndCommunicationManager")
            .field("user_state", &self.user_state.is_some())
            .finish_non_exhaustive()
    }
}

impl LanguageAndCommunicationManager {
    /// Reset all language/communication state for a new episode.
    pub fn reset(&mut self) {
        self.voice_coherence.reset();
        if let Some(ref mut usi) = self.user_state {
            usi.reset();
        }
        // Clear pending text (Broca manager preserves learned weights).
        self.last_broca_text = None;
        self.last_language_source = None;
    }
}
