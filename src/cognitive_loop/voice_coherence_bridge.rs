//! Voice-coherence bridge: groups CfC coherence, voice feedback, and temporal signature encoding.

use crate::dynamics::cfc_coherence::CfCCoherenceBridge;
use crate::dynamics::temporal_signatures::TemporalSignatureEncoder;
use crate::voice::voice_feedback::VoiceFeedbackBridge;

/// Groups the three voice-coherence subsystems that are always co-initialized and co-reset.
pub(crate) struct VoiceCoherenceBridge {
    pub coherence: CfCCoherenceBridge,
    pub voice_feedback: VoiceFeedbackBridge,
    pub temporal_encoder: TemporalSignatureEncoder,
}

impl VoiceCoherenceBridge {
    pub fn new(coherence_lr: f32) -> Self {
        use crate::dynamics::cfc_coherence::CoherenceConfig;
        use crate::dynamics::temporal_signatures::SignatureConfig;
        Self {
            coherence: CfCCoherenceBridge::new(CoherenceConfig {
                base_learning_rate: coherence_lr,
                ..CoherenceConfig::default()
            }),
            voice_feedback: VoiceFeedbackBridge::default(),
            temporal_encoder: TemporalSignatureEncoder::new(SignatureConfig::default()),
        }
    }

    pub fn reset(&mut self) {
        self.coherence.reset();
        self.voice_feedback.reset();
        self.temporal_encoder.reset();
    }
}
