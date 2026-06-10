// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Voice Module: Consciousness-Driven Speech Synthesis
//!
//! This module provides voice synthesis capabilities with LTC-driven prosody.
//! The key innovation is that speech rhythm emerges from the LTC network's
//! temporal dynamics, creating natural, consciousness-aware pacing.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │                    VOICE SYNTHESIS PIPELINE                       │
//! │                                                                   │
//! │   Text Input                                                      │
//! │       │                                                           │
//! │       ▼                                                           │
//! │   ┌──────────────┐                                                │
//! │   │ LTC Network  │ → Temporal dynamics for pacing                 │
//! │   └──────┬───────┘                                                │
//! │          │                                                        │
//! │          ▼                                                        │
//! │   ┌──────────────┐                                                │
//! │   │  LTCPacing   │ → Consciousness-aware rhythm                   │
//! │   └──────┬───────┘                                                │
//! │          │                                                        │
//! │          ▼                                                        │
//! │   ┌──────────────┐                                                │
//! │   │ TTS Engine   │ → Kokoro or simulated synthesis                │
//! │   │ (Optional)   │                                                │
//! │   └──────┬───────┘                                                │
//! │          │                                                        │
//! │          ▼                                                        │
//! │      Audio Output                                                 │
//! └──────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Features
//!
//! - **LTC Pacing**: Speech rhythm emerges from LTC temporal dynamics
//! - **Emotional Prosody**: Consciousness state influences tone
//! - **Kokoro TTS**: High-quality neural synthesis (when feature enabled)

use anyhow::Result;
use serde::{Deserialize, Serialize};

// TTS infrastructure modules
pub mod articulatory_synthesizer;
pub mod formant_targets;
pub mod phoneme_hdc;
pub mod vocoder;

// Cognitive loop integration
pub mod cognitive_bridge;
pub mod voice_feedback;

// Voice pipeline orchestrator: bridges STT, cognitive loop, Broca, and TTS
pub mod audio_bridge;
pub mod orchestrator;
pub mod synthesis_bridge;

// LTC-driven articulatory synthesis (Phase 17)
pub mod vocal_tract_controller;
pub mod vocal_tract_encoder;
pub mod vocal_tract_fep;
pub mod vocal_tract_metrics;

// Rap/rhythmic synthesis modules
pub mod beat_sync;
pub mod rap;
pub mod rhyme_hdc;

// Real-time audio output (cpal + ringbuf)
#[cfg(feature = "live-voice")]
pub mod audio_out;
#[cfg(feature = "live-voice")]
pub use audio_out::AudioOutput;

// Real-time streaming voice (G2P → synthesis → speaker)
#[cfg(feature = "live-voice")]
pub mod live_voice;
#[cfg(feature = "live-voice")]
pub use live_voice::LiveVoice;

// REPL voice output (consciousness-modulated speech for interactive use)
pub mod repl_voice;

// BigVGAN neural vocoder (ONNX-based waveform generation)
#[cfg(feature = "neural-vocoder")]
pub mod neural_vocoder;

// Kokoro TTS engine (ONNX-based neural TTS)
pub mod g2p;
pub mod kokoro_engine;

// Re-export formant types
pub use articulatory_synthesizer::{
    ArticulatoryConfig, ArticulatorySynthesizer, FormantFrame, TimedPhoneme,
};
pub use formant_targets::{FormantDatabase, FormantTarget};
pub use phoneme_hdc::{AcousticParams, Manner, PhonemeHdcCodec, PhonemeSpec, PitchContour, Place};
pub use vocoder::{
    FormantVocoder, VocoderConfig, VoiceQuality, cognitive_state_to_voice_quality,
    cognitive_state_to_voice_quality_extended,
};

// Re-export cognitive bridge types
pub use cognitive_bridge::{
    CognitivePacing, CognitiveVoiceBridge, SemanticCategory, SemanticProsody,
    VoiceConsciousnessSignals,
};

// Re-export voice feedback types
pub use voice_feedback::{
    VoiceFeedbackBridge, VoiceFeedbackConfig, VoiceOutputMetrics, VoiceQualitySummary,
};

// Re-export voice pipeline orchestrator types
pub use audio_bridge::{aggregate_audio_hvs, binary_to_continuous};
pub use orchestrator::VoiceOrchestrator;
#[cfg(feature = "ssm_language")]
pub use synthesis_bridge::cycle_result_to_thought_channels;

// Re-export vocal tract types (Phase 17)
pub use vocal_tract_controller::{
    ProsodyCorrection, ProsodyHead, SpeakerProfile, TrainingHyperparams, VocalTractConfig,
    VocalTractController,
};
pub use vocal_tract_controller::{
    refine_controller_ls, train_controller_cv_vc_transitions, train_controller_on_phoneme_db,
    train_controller_transitions,
};
pub use vocal_tract_encoder::{VocalTractHdcEncoder, VoiceCognitiveState};
#[cfg(feature = "vocal-tract")]
pub use vocal_tract_fep::{
    Intonation, ProsodyContext, StreamingVocalTract, VocalTractPipeline, predict_duration,
};
pub use vocal_tract_fep::{
    VocalAction, VocalTractFepAgent, VocalTractFepResult, VocalTractObservation,
};
#[cfg(feature = "vocal-tract")]
pub use vocal_tract_metrics::VocalTractMetrics;

// Re-export rap synthesis types
pub use beat_sync::{BeatPosition, BeatSync, FlowPattern, SwingConfig, SyllableTiming};
pub use rap::{
    FlowStyle, LyricLine, PhonemeDict, RapConfig, RapSynthesizer, SimplePhonemeDict, Verse,
};
pub use rhyme_hdc::{RhymeEncoder, RhymeScheme, RhymeScore, RhymeType};

// Re-export REPL voice types
pub use repl_voice::{
    AnalyzedWord, PhraseBoundary, ReplVoiceConfig, ReplVoiceOutput, SimpleG2P, TextAnalyzer,
};

// Re-export Kokoro TTS types
pub use g2p::G2PConverter;
pub use kokoro_engine::{KokoroConfig, KokoroEngine, save_wav};

/// LTC-driven speech pacing parameters
///
/// These parameters emerge from the LTC network's temporal dynamics
/// and control the rhythm and timing of synthesized speech.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LTCPacing {
    /// Speaking rate relative to base (1.0 = normal)
    pub rate: f32,

    /// Pause duration after phrases (seconds)
    pub phrase_pause: f32,

    /// Pause duration after sentences (seconds)
    pub sentence_pause: f32,

    /// Emphasis factor for stressed syllables
    pub emphasis: f32,

    /// Breath pause probability
    pub breath_probability: f32,

    /// Current emotional valence (-1.0 to 1.0)
    pub emotional_valence: f32,

    /// Current arousal level (0.0 to 1.0)
    pub arousal: f32,

    /// Time constant (tau) from LTC - slower = more deliberate speech
    pub tau: f32,
}

impl Default for LTCPacing {
    fn default() -> Self {
        Self {
            rate: 1.0,
            phrase_pause: 0.3,
            sentence_pause: 0.6,
            emphasis: 1.0,
            breath_probability: 0.1,
            emotional_valence: 0.0,
            arousal: 0.5,
            tau: 1.0,
        }
    }
}

impl LTCPacing {
    /// Create pacing for calm, deliberate speech
    pub fn calm() -> Self {
        Self {
            rate: 0.85,
            phrase_pause: 0.4,
            sentence_pause: 0.8,
            emphasis: 0.8,
            breath_probability: 0.15,
            emotional_valence: 0.2,
            arousal: 0.3,
            tau: 2.0,
        }
    }

    /// Create pacing for excited, energetic speech
    pub fn excited() -> Self {
        Self {
            rate: 1.2,
            phrase_pause: 0.2,
            sentence_pause: 0.4,
            emphasis: 1.3,
            breath_probability: 0.05,
            emotional_valence: 0.8,
            arousal: 0.9,
            tau: 0.5,
        }
    }

    /// Create pacing for focused, technical explanation
    pub fn focused() -> Self {
        Self {
            rate: 0.95,
            phrase_pause: 0.35,
            sentence_pause: 0.7,
            emphasis: 1.1,
            breath_probability: 0.12,
            emotional_valence: 0.1,
            arousal: 0.6,
            tau: 1.2,
        }
    }

    /// Create pacing from LTC state
    pub fn from_ltc_state(hidden: &[f32], tau: f32) -> Self {
        // Extract features from LTC hidden state
        let arousal = if !hidden.is_empty() {
            hidden.iter().map(|x| x.abs()).sum::<f32>() / hidden.len() as f32
        } else {
            0.5
        };

        let valence = if hidden.len() > 1 {
            // Use asymmetry of hidden state as valence proxy
            let pos_sum: f32 = hidden.iter().filter(|&&x| x > 0.0).sum();
            let neg_sum: f32 = hidden.iter().filter(|&&x| x < 0.0).map(|x| x.abs()).sum();
            let total = pos_sum + neg_sum;
            if total > 0.0 {
                (pos_sum - neg_sum) / total
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Map tau to speech rate (inverse relationship)
        let rate = 0.7 + 0.6 / (1.0 + tau * 0.5);

        // Map arousal to emphasis
        let emphasis = 0.8 + arousal * 0.4;

        Self {
            rate,
            phrase_pause: 0.2 + tau * 0.1,
            sentence_pause: 0.4 + tau * 0.2,
            emphasis,
            breath_probability: 0.05 + (1.0 - arousal) * 0.1,
            emotional_valence: valence,
            arousal,
            tau,
        }
    }

    /// Interpolate between two pacing states
    pub fn lerp(&self, other: &LTCPacing, t: f32) -> LTCPacing {
        let t = t.clamp(0.0, 1.0);
        LTCPacing {
            rate: self.rate + (other.rate - self.rate) * t,
            phrase_pause: self.phrase_pause + (other.phrase_pause - self.phrase_pause) * t,
            sentence_pause: self.sentence_pause + (other.sentence_pause - self.sentence_pause) * t,
            emphasis: self.emphasis + (other.emphasis - self.emphasis) * t,
            breath_probability: self.breath_probability
                + (other.breath_probability - self.breath_probability) * t,
            emotional_valence: self.emotional_valence
                + (other.emotional_valence - self.emotional_valence) * t,
            arousal: self.arousal + (other.arousal - self.arousal) * t,
            tau: self.tau + (other.tau - self.tau) * t,
        }
    }

    /// Apply adaptive behavior multipliers from consciousness state
    ///
    /// This modulates pacing based on the cognitive loop's adaptive behavior,
    /// creating consciousness-driven speech rhythm.
    pub fn apply_adaptive_behavior(
        &self,
        speech_rate_multiplier: f32,
        pause_multiplier: f32,
        attention_sensitivity: f32,
    ) -> LTCPacing {
        LTCPacing {
            // Rate scales with speech multiplier
            rate: (self.rate * speech_rate_multiplier).clamp(0.5, 2.0),
            // Pauses scale with pause multiplier
            phrase_pause: (self.phrase_pause * pause_multiplier).clamp(0.1, 2.0),
            sentence_pause: (self.sentence_pause * pause_multiplier).clamp(0.2, 3.0),
            // Emphasis modulated by attention sensitivity
            emphasis: (self.emphasis * attention_sensitivity).clamp(0.5, 2.0),
            // Breath probability increases with longer pauses
            breath_probability: (self.breath_probability * pause_multiplier).clamp(0.0, 0.5),
            // Keep emotional state unchanged
            emotional_valence: self.emotional_valence,
            arousal: self.arousal,
            tau: self.tau,
        }
    }
}

/// Configuration for voice output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceOutputConfig {
    /// Sample rate for audio output
    pub sample_rate: u32,

    /// Whether to enable TTS (requires `voice-tts` feature)
    pub enable_tts: bool,

    /// Path to TTS model (Kokoro ONNX)
    pub model_path: Option<String>,

    /// Default voice ID (0-based index)
    pub voice_id: usize,

    /// Volume level (0.0 to 1.0)
    pub volume: f32,

    /// Whether to stream audio output
    pub stream_output: bool,

    /// Buffer size for audio streaming
    pub buffer_size: usize,
}

impl Default for VoiceOutputConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24000,
            enable_tts: false,
            model_path: Some("models/kokoro-v1.0.onnx".to_string()),
            voice_id: 0,
            volume: 0.8,
            stream_output: false,
            buffer_size: 4096,
        }
    }
}

/// Voice output system
///
/// Provides text-to-speech synthesis with consciousness-driven prosody.
/// Uses Kokoro TTS when the `voice-tts` feature is enabled, otherwise
/// provides a mock implementation for testing.
pub struct VoiceOutput {
    /// Configuration
    config: VoiceOutputConfig,

    /// Current pacing state
    current_pacing: LTCPacing,

    /// Whether TTS is initialized
    initialized: bool,

    /// Kokoro TTS engine (when available)
    kokoro: Option<kokoro_engine::KokoroEngine>,

    /// Statistics
    stats: VoiceStats,
}

/// Voice synthesis statistics
#[derive(Debug, Clone, Default)]
pub struct VoiceStats {
    /// Total utterances synthesized
    pub total_utterances: u64,

    /// Total characters synthesized
    pub total_chars: u64,

    /// Average synthesis time (ms)
    pub avg_synthesis_time_ms: f32,

    /// Total audio seconds generated
    pub total_audio_seconds: f32,
}

impl VoiceOutput {
    /// Create a new voice output system
    pub fn new(config: VoiceOutputConfig) -> Self {
        Self {
            config,
            current_pacing: LTCPacing::default(),
            initialized: false,
            kokoro: None,
            stats: VoiceStats::default(),
        }
    }

    /// Initialize the TTS engine.
    ///
    /// Attempts to load the Kokoro ONNX model when the `voice-tts` feature is enabled.
    /// Falls back to simulated TTS (sine wave) if the model cannot be loaded.
    pub fn initialize(&mut self) -> Result<()> {
        // Attempt to load Kokoro TTS
        let kokoro_config = kokoro_engine::KokoroConfig {
            sample_rate: self.config.sample_rate,
            ..kokoro_engine::KokoroConfig::default()
        };
        self.kokoro = kokoro_engine::KokoroEngine::load(kokoro_config);
        if self.kokoro.is_some() {
            tracing::info!("VoiceOutput initialized with Kokoro TTS engine");
        } else {
            tracing::info!("VoiceOutput initialized with simulated TTS (Kokoro unavailable)");
        }
        self.initialized = true;
        Ok(())
    }

    /// Check if TTS is initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Update pacing from LTC state
    pub fn update_pacing(&mut self, hidden: &[f32], tau: f32) {
        self.current_pacing = LTCPacing::from_ltc_state(hidden, tau);
    }

    /// Set pacing directly
    pub fn set_pacing(&mut self, pacing: LTCPacing) {
        self.current_pacing = pacing;
    }

    /// Get current pacing
    pub fn pacing(&self) -> &LTCPacing {
        &self.current_pacing
    }

    /// Synthesize speech from text
    ///
    /// Returns audio samples at the configured sample rate.
    pub fn synthesize(&mut self, text: &str) -> Result<Vec<f32>> {
        self.synthesize_with_pacing(text, &self.current_pacing.clone())
    }

    /// Synthesize speech with specific pacing
    pub fn synthesize_with_pacing(&mut self, text: &str, pacing: &LTCPacing) -> Result<Vec<f32>> {
        let start = std::time::Instant::now();

        let samples = if self.config.enable_tts && self.initialized {
            // Try Kokoro TTS first, fall back to simulated
            if let Some(ref mut kokoro) = self.kokoro {
                if let Some(mut audio) = kokoro.synthesize(text, Some(self.config.voice_id)) {
                    // Apply pacing by resampling: rate > 1.0 = faster = fewer samples
                    if (pacing.rate - 1.0).abs() > 0.05 {
                        audio = resample_audio(&audio, pacing.rate);
                    }
                    // Apply volume
                    for s in &mut audio {
                        *s *= self.config.volume * pacing.emphasis * 0.5;
                    }
                    audio
                } else {
                    self.simulate_tts(text, pacing)
                }
            } else {
                self.simulate_tts(text, pacing)
            }
        } else {
            self.simulate_tts(text, pacing)
        };

        let elapsed = start.elapsed().as_secs_f32() * 1000.0;
        let audio_seconds = samples.len() as f32 / self.config.sample_rate as f32;

        // Update stats
        self.stats.total_utterances += 1;
        self.stats.total_chars += text.len() as u64;
        self.stats.total_audio_seconds += audio_seconds;
        let n = self.stats.total_utterances as f32;
        self.stats.avg_synthesis_time_ms =
            (self.stats.avg_synthesis_time_ms * (n - 1.0) + elapsed) / n;

        Ok(samples)
    }

    /// Simulate TTS for testing
    fn simulate_tts(&self, text: &str, pacing: &LTCPacing) -> Vec<f32> {
        let sample_rate = self.config.sample_rate as f32;

        // Estimate duration: ~0.1 seconds per character at base rate
        let base_duration = text.len() as f32 * 0.08 / pacing.rate;

        // Add pauses
        let pause_count =
            text.matches('.').count() + text.matches('?').count() + text.matches('!').count();
        let phrase_count = text.matches(',').count();

        let total_pause =
            pause_count as f32 * pacing.sentence_pause + phrase_count as f32 * pacing.phrase_pause;

        let total_duration = base_duration + total_pause;
        let num_samples = (total_duration * sample_rate) as usize;

        // Generate simple sine wave as placeholder audio
        let mut samples = Vec::with_capacity(num_samples);
        let base_freq = 150.0 + pacing.emotional_valence * 50.0; // Higher valence = higher pitch

        for i in 0..num_samples {
            let t = i as f32 / sample_rate;

            // Modulate frequency based on position (simple prosody)
            let phase = i as f32 / num_samples as f32;
            let freq_mod = 1.0 + 0.1 * (phase * std::f32::consts::TAU).sin();

            let sample = (t * base_freq * freq_mod * 2.0 * std::f32::consts::PI).sin();

            // Apply envelope
            let envelope = if phase < 0.1 {
                phase / 0.1 // Attack
            } else if phase > 0.9 {
                (1.0 - phase) / 0.1 // Release
            } else {
                1.0 // Sustain
            };

            samples.push(sample * envelope * self.config.volume * pacing.emphasis * 0.5);
        }

        samples
    }

    /// Get statistics
    pub fn stats(&self) -> &VoiceStats {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &VoiceOutputConfig {
        &self.config
    }

    /// Synthesize with adaptive behavior applied
    ///
    /// Applies consciousness-driven modulations to the pacing before synthesis.
    /// This creates speech that naturally reflects the system's cognitive state.
    pub fn synthesize_with_adaptive_behavior(
        &mut self,
        text: &str,
        speech_rate_multiplier: f32,
        pause_multiplier: f32,
        attention_sensitivity: f32,
    ) -> Result<Vec<f32>> {
        let adaptive_pacing = self.current_pacing.apply_adaptive_behavior(
            speech_rate_multiplier,
            pause_multiplier,
            attention_sensitivity,
        );
        self.synthesize_with_pacing(text, &adaptive_pacing)
    }

    /// Update pacing from LTC state with adaptive behavior
    pub fn update_pacing_adaptive(
        &mut self,
        hidden: &[f32],
        tau: f32,
        speech_rate_multiplier: f32,
        pause_multiplier: f32,
        attention_sensitivity: f32,
    ) {
        let base_pacing = LTCPacing::from_ltc_state(hidden, tau);
        self.current_pacing = base_pacing.apply_adaptive_behavior(
            speech_rate_multiplier,
            pause_multiplier,
            attention_sensitivity,
        );
    }
}

impl Default for VoiceOutput {
    fn default() -> Self {
        Self::new(VoiceOutputConfig::default())
    }
}

/// Simple linear resampling to adjust speech rate.
/// rate > 1.0 = faster (fewer output samples), rate < 1.0 = slower (more samples).
fn resample_audio(samples: &[f32], rate: f32) -> Vec<f32> {
    if samples.is_empty() || rate <= 0.0 {
        return samples.to_vec();
    }
    let raw_len = samples.len() as f32 / rate;
    let out_len = if raw_len.is_finite() {
        raw_len.clamp(0.0, samples.len() as f32 * 4.0) as usize
    } else {
        samples.len() // NaN/Inf rate → pass through unchanged
    };
    let mut output = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f32 * rate;
        let idx = src_pos as usize;
        let frac = src_pos - idx as f32;
        if idx + 1 < samples.len() {
            output.push(samples[idx] * (1.0 - frac) + samples[idx + 1] * frac);
        } else if idx < samples.len() {
            output.push(samples[idx]);
        }
    }
    output
}

/// Speech recognizer configuration (for future STT support)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechRecognizerConfig {
    /// Sample rate for input audio
    pub sample_rate: u32,

    /// Language code (e.g., "en-US")
    pub language: String,

    /// Whether to enable continuous recognition
    pub continuous: bool,

    /// Model path (for Whisper ONNX)
    pub model_path: Option<String>,
}

impl Default for SpeechRecognizerConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            language: "en-US".to_string(),
            continuous: false,
            model_path: Some("models/whisper-tiny.onnx".to_string()),
        }
    }
}

/// Phoneme representation for advanced TTS
#[derive(Debug, Clone)]
pub struct Phoneme {
    /// IPA symbol
    pub symbol: String,
    /// Duration in milliseconds
    pub duration_ms: u32,
    /// Stress level (0-2)
    pub stress: u8,
    /// Pitch modifier
    pub pitch_mod: f32,
}

/// Word with phoneme breakdown
#[derive(Debug, Clone)]
pub struct PhonemizedWord {
    /// Original text
    pub text: String,
    /// Phoneme sequence
    pub phonemes: Vec<Phoneme>,
    /// Part of speech tag
    pub pos_tag: Option<String>,
}

/// Configuration for a voice conversation session.
///
/// Used by the service binary to configure the voice interface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceConfig {
    /// Voice ID (index into available voice models)
    pub voice_id: u8,
    /// Whether to enable LTC-driven pacing
    pub ltc_pacing: bool,
    /// Sample rate for audio
    pub sample_rate: u32,
}

impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            voice_id: 0,
            ltc_pacing: true,
            sample_rate: 24000,
        }
    }
}

/// Voice conversation session combining STT (listen) and TTS (speak).
///
/// Wraps the voice output pipeline and provides a simple listen/speak API
/// for the service binary's conversational voice turns.
pub struct VoiceConversation {
    /// Voice output for TTS
    voice_output: VoiceOutput,
    /// Configuration
    _config: VoiceConfig,
}

impl VoiceConversation {
    /// Create a new voice conversation from config.
    pub fn new(config: VoiceConfig) -> Result<Self> {
        let voice_config = VoiceOutputConfig {
            sample_rate: config.sample_rate,
            voice_id: config.voice_id as usize,
            ..VoiceOutputConfig::default()
        };
        let mut voice_output = VoiceOutput::new(voice_config);
        voice_output.initialize()?;
        Ok(Self {
            voice_output,
            _config: config,
        })
    }

    /// Listen for user speech input (STT).
    ///
    /// Currently returns a placeholder — real STT integration requires
    /// the `stt` feature and an audio input device.
    pub fn listen(&mut self) -> Result<String> {
        // STT is not yet wired; return empty to indicate no speech captured.
        Ok(String::new())
    }

    /// Speak text aloud via TTS.
    pub fn speak(&mut self, text: &str) -> Result<()> {
        let _samples = self.voice_output.synthesize(text)?;
        // In a real implementation, samples would be played via audio output.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pacing_default() {
        let pacing = LTCPacing::default();
        assert_eq!(pacing.rate, 1.0);
        assert!(pacing.tau > 0.0);
    }

    #[test]
    fn test_pacing_from_ltc() {
        let hidden = vec![0.5f32; 64];
        let tau = 1.5;
        let pacing = LTCPacing::from_ltc_state(&hidden, tau);

        assert!(pacing.rate > 0.0);
        assert!(pacing.arousal > 0.0);
    }

    #[test]
    fn test_pacing_lerp() {
        let calm = LTCPacing::calm();
        let excited = LTCPacing::excited();

        let mid = calm.lerp(&excited, 0.5);
        assert!(mid.rate > calm.rate && mid.rate < excited.rate);
    }

    #[test]
    fn test_voice_output() {
        let mut voice = VoiceOutput::default();
        let samples = voice.synthesize("Hello, world!").unwrap();

        assert!(!samples.is_empty());
        assert!(voice.stats.total_utterances == 1);
    }

    #[test]
    fn test_voice_with_pacing() {
        let mut voice = VoiceOutput::default();
        let pacing = LTCPacing::calm();

        let samples = voice
            .synthesize_with_pacing("This is a calm message.", &pacing)
            .unwrap();
        assert!(!samples.is_empty());
    }

    #[test]
    fn test_adaptive_behavior_pacing() {
        let base = LTCPacing::default();

        // Slow and contemplative: lower rate, longer pauses
        let contemplative = base.apply_adaptive_behavior(0.8, 1.5, 0.7);
        assert!(contemplative.rate < base.rate);
        assert!(contemplative.phrase_pause > base.phrase_pause);

        // Excited: faster rate, shorter pauses
        let excited = base.apply_adaptive_behavior(1.2, 0.7, 1.3);
        assert!(excited.rate > base.rate);
        assert!(excited.phrase_pause < base.phrase_pause);
    }

    #[test]
    fn test_voice_with_adaptive_behavior() {
        let mut voice = VoiceOutput::default();

        // Synthesize with excited adaptive behavior
        let samples_excited = voice
            .synthesize_with_adaptive_behavior(
                "Hello world.",
                1.2, // faster speech
                0.7, // shorter pauses
                1.3, // higher attention
            )
            .unwrap();

        // Synthesize same text with contemplative behavior
        let samples_calm = voice
            .synthesize_with_adaptive_behavior(
                "Hello world.",
                0.8, // slower speech
                1.5, // longer pauses
                0.7, // lower attention
            )
            .unwrap();

        // Calm version should be longer due to slower rate and longer pauses
        assert!(samples_calm.len() > samples_excited.len());
    }

    #[test]
    fn test_update_pacing_adaptive() {
        let mut voice = VoiceOutput::default();
        let hidden = vec![0.5f32; 64];

        // Update with default multipliers
        voice.update_pacing_adaptive(&hidden, 1.0, 1.0, 1.0, 1.0);
        let base_rate = voice.pacing().rate;

        // Update with faster multiplier
        voice.update_pacing_adaptive(&hidden, 1.0, 1.3, 1.0, 1.0);
        assert!(voice.pacing().rate > base_rate);
    }
}
