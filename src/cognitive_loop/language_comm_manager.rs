//! Language & Communication Manager — groups voice coherence, Broca language,
//! and user state inference into a single CLS field.

/// Consolidated manager for language and communication subsystems.
///
/// Replaces 4 top-level CLS fields:
/// - `voice_coherence` — CfC coherence + voice feedback + temporal signatures
/// - `broca_manager` — SSM language generation (feature-gated)
/// - `last_broca_text` — most recent Broca output
/// - `user_state` — user state inference for adaptive responses
#[derive(Debug)]
pub struct LanguageAndCommunicationManager {
    /// Voice-coherence bridge: CfC coherence + voice feedback + temporal signatures.
    pub voice_coherence: super::voice_coherence_bridge::VoiceCoherenceBridge,

    /// Broca SSM language center: consciousness-gated thought-to-text.
    #[cfg(feature = "ssm_language")]
    pub broca_manager: Option<super::broca_bridge::BrocaManager>,

    /// Most recent Broca-generated text, drained into `CycleResult.language_output`.
    #[cfg(feature = "ssm_language")]
    pub last_broca_text: Option<String>,

    /// User state inference for adaptive response generation.
    pub user_state: Option<crate::user_state_inference::UserStateInference>,
}

impl LanguageAndCommunicationManager {
    /// Reset all language/communication state for a new episode.
    pub fn reset(&mut self) {
        self.voice_coherence.reset();
        if let Some(ref mut usi) = self.user_state {
            usi.reset();
        }
        #[cfg(feature = "ssm_language")]
        {
            // Broca manager preserves learned weights; just clear pending text.
            self.last_broca_text = None;
        }
    }
}
