//! Voice Interface Module - Natural Voice Conversation with LTC Pacing
//!
//! This module provides speech-to-text (STT) and text-to-speech (TTS) capabilities
//! integrated with Symthaea's LTC temporal dynamics for natural conversation flow.
//!
//! ## Components
//!
//! - **VoiceInput**: Speech recognition using whisper-rs (whisper.cpp bindings)
//! - **VoiceOutput**: Speech synthesis using Kokoro TTS (ONNX runtime)
//! - **VoiceConversation**: Full voice loop with LTC-aware pacing
//!
//! ## Features
//!
//! Enable voice features in Cargo.toml:
//! ```toml
//! symthaea = { features = ["voice"] }      # Full voice (STT + TTS)
//! symthaea = { features = ["voice-stt"] }  # STT only (whisper)
//! symthaea = { features = ["voice-tts"] }  # TTS only (Kokoro)
//! ```
//!
//! ## LTC Integration
//!
//! The voice interface uses LTC temporal dynamics to modulate:
//! - **Speech rate**: High flow state = slightly faster, confident delivery
//! - **Pause duration**: Rising Φ = shorter pauses, falling = longer pauses
//! - **Pitch variation**: Based on emotional valence (future)
//!
//! ## Example
//!
//! ```rust,ignore
//! use symthaea::voice::{VoiceConversation, VoiceConfig};
//!
//! let config = VoiceConfig::default();
//! let mut voice = VoiceConversation::new(config)?;
//! voice.run()?;  // Start voice conversation loop
//! ```

// Feature-gated submodules
#[cfg(feature = "voice-stt")]
pub mod input;

#[cfg(feature = "voice-tts")]
pub mod output;

pub mod conversation;
pub mod models;

// Re-exports
#[cfg(feature = "voice-stt")]
pub use input::{VoiceInput, VoiceInputConfig, TranscriptionResult};

#[cfg(feature = "voice-tts")]
pub use output::{VoiceOutput, VoiceOutputConfig, SynthesisResult};

pub use conversation::{VoiceConversation, VoiceConfig, VoiceEvent};
pub use models::{ModelManager, ModelPaths, WhisperModel, KokoroModel};

/// Voice interface error types
#[derive(Debug, thiserror::Error)]
pub enum VoiceError {
    #[error("Audio device error: {0}")]
    AudioDevice(String),

    #[error("Model loading error: {0}")]
    ModelLoad(String),

    #[error("Transcription error: {0}")]
    Transcription(String),

    #[error("Synthesis error: {0}")]
    Synthesis(String),

    #[error("Model download error: {0}")]
    Download(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Feature not enabled: {0}")]
    FeatureNotEnabled(String),
}

pub type VoiceResult<T> = Result<T, VoiceError>;

/// LTC-aware pacing parameters for voice output
#[derive(Debug, Clone)]
pub struct LTCPacing {
    /// Speech rate multiplier (0.8 - 1.2)
    pub speech_rate: f32,
    /// Pause duration in milliseconds
    pub pause_ms: u32,
    /// Whether we're in peak flow (for special phrases)
    pub peak_flow: bool,
}

impl LTCPacing {
    /// Calculate pacing from LTC snapshot
    pub fn from_ltc(flow_state: f32, phi_trend: f32) -> Self {
        // Flow state affects speech rate
        let speech_rate = if flow_state > 0.7 {
            1.1  // Peak flow = slightly faster, confident
        } else if flow_state > 0.4 {
            1.0  // Normal flow = natural pace
        } else {
            0.9  // Warming up = slower, thoughtful
        };

        // Φ trend affects pauses
        let pause_ms = if phi_trend > 0.02 {
            150  // Rising Φ = shorter pauses, engaged
        } else if phi_trend > 0.0 {
            250  // Stable = normal pauses
        } else if phi_trend > -0.02 {
            350  // Slightly falling = longer pauses
        } else {
            500  // Falling Φ = reflective pauses
        };

        Self {
            speech_rate,
            pause_ms,
            peak_flow: flow_state > 0.7,
        }
    }
}

impl Default for LTCPacing {
    fn default() -> Self {
        Self {
            speech_rate: 1.0,
            pause_ms: 250,
            peak_flow: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ltc_pacing_high_flow() {
        let pacing = LTCPacing::from_ltc(0.8, 0.05);
        assert!(pacing.speech_rate > 1.0);
        assert!(pacing.pause_ms < 200);
        assert!(pacing.peak_flow);
    }

    #[test]
    fn test_ltc_pacing_low_flow() {
        let pacing = LTCPacing::from_ltc(0.2, -0.05);
        assert!(pacing.speech_rate < 1.0);
        assert!(pacing.pause_ms > 400);
        assert!(!pacing.peak_flow);
    }

    #[test]
    fn test_ltc_pacing_normal() {
        let pacing = LTCPacing::from_ltc(0.5, 0.01);
        assert_eq!(pacing.speech_rate, 1.0);
        assert!(pacing.pause_ms >= 200 && pacing.pause_ms <= 300);
        assert!(!pacing.peak_flow);
    }
}
