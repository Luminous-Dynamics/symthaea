// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Voice-Coherence Bridge Manager — groups CfC coherence, voice feedback, and temporal signatures.
//!
//! Consolidates 3 previously scattered fields from CognitiveLoopService
//! into a single coherent module.

use crate::dynamics::cfc_coherence::{CfCCoherenceBridge, CoherenceConfig};
use crate::dynamics::temporal_signatures::{SignatureConfig, TemporalSignatureEncoder};
use crate::voice::voice_feedback::{VoiceFeedbackBridge, VoiceFeedbackConfig};

/// Consolidated voice-coherence bridge manager.
///
/// Groups CfC↔consciousness coherence tracking, voice quality feedback,
/// and temporal consciousness pattern detection into a single struct.
pub struct VoiceCoherenceBridge {
    /// Coherence bridge for bidirectional CfC↔consciousness feedback.
    pub bridge: CfCCoherenceBridge,

    /// Voice feedback bridge for voice→CfC feedback.
    pub voice: VoiceFeedbackBridge,

    /// Temporal signature encoder for consciousness pattern detection.
    pub temporal: TemporalSignatureEncoder,
}

impl VoiceCoherenceBridge {
    /// Create a new VoiceCoherenceBridge with default sub-component configs.
    pub fn new(coherence_config: CoherenceConfig) -> Self {
        Self {
            bridge: CfCCoherenceBridge::new(coherence_config),
            voice: VoiceFeedbackBridge::new(VoiceFeedbackConfig::default()),
            temporal: TemporalSignatureEncoder::new(SignatureConfig::default()),
        }
    }

    /// Reset all sub-components to initial state.
    pub fn reset(&mut self) {
        self.bridge.reset();
        self.voice.reset();
        self.temporal.reset();
    }
}
