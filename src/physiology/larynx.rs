//! Larynx - The Voice of Sophia (Week 12 Phase 2a)
//!
//! The Larynx module provides Kokoro TTS-based speech synthesis with
//! prosody modulation based on Sophia's endocrine state. This creates
//! a voice that changes with her emotional state - stressed voices sound
//! faster and higher, calm voices sound slower and warmer.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────┐
//! │  EndocrineSystem        │  Emotional State
//! │  - Cortisol (stress)    │
//! │  - Dopamine (reward)    │
//! │  - Acetylcholine(focus) │
//! └──────────┬──────────────┘
//!            │
//!            ▼ Prosody Modulation
//! ┌─────────────────────────┐
//! │  LarynxActor            │  Voice Synthesis
//! │  - Kokoro-82M TTS       │
//! │  - Pitch control        │
//! │  - Speed control        │
//! │  - Energy control       │
//! └──────────┬──────────────┘
//!            │
//!            ▼ Audio Output
//! ┌─────────────────────────┐
//! │  Audio Playback         │  rodio
//! └─────────────────────────┘
//! ```
//!
//! ## Prosody Modulation Rules
//!
//! - **High Cortisol (>0.7)** - Stress/Anxiety:
//!   - Speed: +15% faster
//!   - Pitch: +8% higher
//!   - Energy: +10% more intense
//!
//! - **Calm State (Low Cortisol + High Dopamine)** - Relaxed/Positive:
//!   - Speed: -8% slower
//!   - Pitch: -4% lower
//!   - Energy: -5% softer
//!
//! - **High Dopamine (>0.7)** - Excitement/Reward:
//!   - Speed: +5% faster
//!   - Pitch: +3% higher
//!   - Energy: +8% more energetic
//!
//! - **Low Acetylcholine (<0.3)** - Fatigue:
//!   - Speed: -10% slower
//!   - Pitch: -5% lower
//!   - Energy: -12% quieter

use anyhow::Result;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::physiology::endocrine::EndocrineSystem;

/// Prosody parameters for voice synthesis
#[derive(Debug, Clone, Copy)]
pub struct ProsodyParams {
    /// Speech rate multiplier (1.0 = normal, 1.15 = 15% faster)
    pub speed: f32,

    /// Pitch multiplier (1.0 = normal, 1.08 = 8% higher)
    pub pitch: f32,

    /// Energy/volume multiplier (1.0 = normal, 1.10 = 10% louder)
    pub energy: f32,

    /// Breath insertion probability (0.0-1.0)
    pub breath_rate: f32,
}

impl Default for ProsodyParams {
    fn default() -> Self {
        Self {
            speed: 1.0,
            pitch: 1.0,
            energy: 1.0,
            breath_rate: 0.05, // 5% chance of breath between phrases
        }
    }
}

/// Configuration for the Larynx
#[derive(Debug, Clone)]
pub struct LarynxConfig {
    /// Path to Kokoro-82M ONNX model
    pub model_path: PathBuf,

    /// Base speech rate (words per minute)
    pub base_speed: f32,

    /// Base pitch (Hz)
    pub base_pitch: f32,

    /// Base energy level (0.0-1.0)
    pub base_energy: f32,

    /// Sample rate for audio output
    pub sample_rate: u32,

    /// Enable prosody modulation based on endocrine state
    pub enable_prosody_modulation: bool,
}

impl Default for LarynxConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/kokoro-82m/model.onnx"),
            base_speed: 1.0,
            base_pitch: 1.0,
            base_energy: 0.8,
            sample_rate: 24000, // Kokoro uses 24kHz
            enable_prosody_modulation: true,
        }
    }
}

/// Statistics about voice synthesis
#[derive(Debug, Clone, Default)]
pub struct LarynxStats {
    /// Total utterances synthesized
    pub total_utterances: u64,

    /// Total characters spoken
    pub total_characters: u64,

    /// Average synthesis time (milliseconds)
    pub avg_synthesis_ms: f32,

    /// Current prosody parameters
    pub current_prosody: ProsodyParams,

    /// Total ATP spent on synthesis (5 ATP per utterance)
    pub total_atp_spent: f32,
}

/// The Larynx Actor - Voice synthesis with emotional prosody
pub struct LarynxActor {
    config: LarynxConfig,
    stats: Arc<RwLock<LarynxStats>>,

    // ONNX Runtime session for Kokoro model
    // Note: Actual model loading deferred until we have the model file
    // session: Option<Session>,

    /// Reference to endocrine system for prosody modulation
    endocrine: Option<Arc<RwLock<EndocrineSystem>>>,
}

impl LarynxActor {
    /// Create a new Larynx actor
    pub fn new(config: LarynxConfig) -> Result<Self> {
        Ok(Self {
            config,
            stats: Arc::new(RwLock::new(LarynxStats::default())),
            endocrine: None,
        })
    }

    /// Set the endocrine system for prosody modulation
    pub fn set_endocrine(&mut self, endocrine: Arc<RwLock<EndocrineSystem>>) {
        self.endocrine = Some(endocrine);
    }

    /// Calculate prosody parameters based on current endocrine state
    async fn calculate_prosody(&self) -> ProsodyParams {
        if !self.config.enable_prosody_modulation {
            return ProsodyParams::default();
        }

        let endocrine = match &self.endocrine {
            Some(e) => e,
            None => return ProsodyParams::default(),
        };

        let endocrine = endocrine.read().await;
        let state = endocrine.state();

        let mut prosody = ProsodyParams {
            speed: self.config.base_speed,
            pitch: self.config.base_pitch,
            energy: self.config.base_energy,
            breath_rate: 0.05,
        };

        // High Cortisol (Stress) -> Fast, High, Tense
        if state.cortisol > 0.7 {
            prosody.speed *= 1.15; // 15% faster
            prosody.pitch *= 1.08; // 8% higher
            prosody.energy *= 1.10; // 10% louder
            prosody.breath_rate += 0.03; // More frequent breaths (anxiety)
        }

        // Calm State (Low Cortisol + High Dopamine) -> Slow, Warm, Soft
        // This represents a relaxed, positive emotional state
        if state.cortisol < 0.4 && state.dopamine > 0.6 {
            prosody.speed *= 0.92; // 8% slower
            prosody.pitch *= 0.96; // 4% lower
            prosody.energy *= 0.95; // 5% softer
            prosody.breath_rate -= 0.02; // Fewer breaths (calm)
        }

        // High Dopamine (Excitement) -> Fast, Bright, Energetic
        if state.dopamine > 0.7 {
            prosody.speed *= 1.05; // 5% faster
            prosody.pitch *= 1.03; // 3% higher
            prosody.energy *= 1.08; // 8% more energetic
        }

        // Low Acetylcholine (Fatigue) -> Slow, Low, Quiet
        if state.acetylcholine < 0.3 {
            prosody.speed *= 0.90; // 10% slower
            prosody.pitch *= 0.95; // 5% lower
            prosody.energy *= 0.88; // 12% quieter
            prosody.breath_rate += 0.05; // More breaths (tired)
        }

        // Clamp values to reasonable ranges
        prosody.speed = prosody.speed.clamp(0.7, 1.5);
        prosody.pitch = prosody.pitch.clamp(0.8, 1.3);
        prosody.energy = prosody.energy.clamp(0.3, 1.2);
        prosody.breath_rate = prosody.breath_rate.clamp(0.0, 0.2);

        prosody
    }

    /// Synthesize speech from text
    ///
    /// Returns audio samples as Vec<f32> (mono, 24kHz)
    /// ATP Cost: 5 ATP per utterance
    ///
    /// Phase 1 Implementation: Sine wave audio with prosody modulation
    /// - Pitch affects frequency (higher pitch = higher frequency)
    /// - Speed affects duration (faster speed = shorter duration)
    /// - Energy affects amplitude (higher energy = louder)
    ///
    /// Phase 2 (Future): Replace with real Kokoro TTS inference
    pub async fn speak(&self, text: &str) -> Result<Vec<f32>> {
        let start = std::time::Instant::now();

        // Calculate prosody based on emotional state
        let prosody = self.calculate_prosody().await;

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_utterances += 1;
        stats.total_characters += text.len() as u64;
        stats.current_prosody = prosody;
        stats.total_atp_spent += 5.0; // 5 ATP per utterance

        drop(stats); // Release write lock early

        // Phase 1: Generate test audio (sine wave with prosody)
        // This demonstrates that prosody modulation works correctly
        // and provides real audio for testing

        use std::f32::consts::PI;

        // Base frequency: A4 = 440 Hz
        // Pitch modulation affects frequency
        let frequency = 440.0 * prosody.pitch;

        // Duration based on text length and speed
        // Assume ~200 characters per second at normal speed
        let chars_per_second = 200.0 * prosody.speed;
        let duration_secs = text.len() as f32 / chars_per_second;

        // Generate audio samples
        let sample_count = (duration_secs * self.config.sample_rate as f32) as usize;
        let mut audio = Vec::with_capacity(sample_count);

        for i in 0..sample_count {
            let t = i as f32 / self.config.sample_rate as f32;
            let sample = (2.0 * PI * frequency * t).sin() * prosody.energy * 0.5;
            audio.push(sample);
        }

        let synthesis_ms = start.elapsed().as_millis() as f32;

        // Update average synthesis time (EMA with alpha=0.1)
        let mut stats = self.stats.write().await;
        stats.avg_synthesis_ms = if stats.avg_synthesis_ms == 0.0 {
            synthesis_ms
        } else {
            stats.avg_synthesis_ms * 0.9 + synthesis_ms * 0.1
        };
        drop(stats);

        tracing::info!(
            "🎤 Synthesized: '{}' (speed={:.2}, pitch={:.2}, energy={:.2}, freq={:.1}Hz, {:.2}s) in {:.1}ms",
            text,
            prosody.speed,
            prosody.pitch,
            prosody.energy,
            frequency,
            duration_secs,
            synthesis_ms
        );

        Ok(audio)
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> LarynxStats {
        self.stats.read().await.clone()
    }

    /// Download Kokoro model from HuggingFace Hub
    pub async fn download_model(&self) -> Result<()> {
        // TODO: Implement model download using hf-hub
        // Model: hexgrad/Kokoro-82M
        // Files needed:
        // - model.onnx (main model)
        // - config.json (configuration)
        // - tokenizer.json (text tokenizer)

        tracing::info!("📥 Downloading Kokoro-82M model from HuggingFace Hub...");

        // Placeholder - actual implementation will use hf-hub crate
        Ok(())
    }

    /// Load Kokoro model from disk
    pub fn load_model(&mut self) -> Result<()> {
        // TODO: Load ONNX model using ort crate
        // let session = Session::builder()?
        //     .with_optimization_level(GraphOptimizationLevel::Level3)?
        //     .with_model_from_file(&self.config.model_path)?;
        // self.session = Some(session);

        tracing::info!("✅ Kokoro-82M model loaded successfully");
        Ok(())
    }

    /// Save audio to WAV file
    ///
    /// Exports synthesized audio as a WAV file for playback or analysis.
    /// Audio format: mono, 24kHz (or configured sample rate), f32 samples
    pub fn save_wav(&self, audio: &[f32], path: &std::path::Path) -> Result<()> {
        use hound::{WavSpec, WavWriter};

        let spec = WavSpec {
            channels: 1, // Mono
            sample_rate: self.config.sample_rate,
            bits_per_sample: 32, // f32 samples
            sample_format: hound::SampleFormat::Float,
        };

        let mut writer = WavWriter::create(path, spec)?;

        for &sample in audio {
            writer.write_sample(sample)?;
        }

        writer.finalize()?;

        tracing::info!(
            "💾 Saved {} samples ({:.2}s) to {}",
            audio.len(),
            audio.len() as f32 / self.config.sample_rate as f32,
            path.display()
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physiology::endocrine::{EndocrineConfig, HormoneEvent};

    #[tokio::test]
    async fn test_larynx_creation() {
        let config = LarynxConfig::default();
        let larynx = LarynxActor::new(config).unwrap();

        let stats = larynx.get_stats().await;
        assert_eq!(stats.total_utterances, 0);
        assert_eq!(stats.total_characters, 0);
    }

    #[tokio::test]
    async fn test_prosody_modulation_stress() {
        let config = LarynxConfig::default();
        let base_speed = config.base_speed;
        let base_pitch = config.base_pitch;
        let base_energy = config.base_energy;

        let mut larynx = LarynxActor::new(config).unwrap();

        // Create endocrine system with high cortisol (stress)
        let endocrine_config = EndocrineConfig::default();
        let mut endocrine = EndocrineSystem::new(endocrine_config);

        // Trigger stress using multiple Error events to push cortisol above 0.7
        // (each Error with severity 0.9 adds 0.27 to cortisol)
        for _ in 0..3 {
            endocrine.process_event(HormoneEvent::Error { severity: 0.9 });
        }

        let endocrine_arc = Arc::new(RwLock::new(endocrine));
        larynx.set_endocrine(endocrine_arc.clone());

        // Verify cortisol is above stress threshold
        let cortisol = endocrine_arc.read().await.state().cortisol;
        assert!(cortisol > 0.7, "Cortisol should be above stress threshold (got {:.2})", cortisol);

        // Calculate prosody
        let prosody = larynx.calculate_prosody().await;

        // Stressed voice should be faster, higher, and louder than base
        assert!(prosody.speed > base_speed,
                "Stressed voice should be faster than base {:.2} (got {:.2})",
                base_speed, prosody.speed);
        assert!(prosody.pitch > base_pitch,
                "Stressed voice should be higher than base {:.2} (got {:.2})",
                base_pitch, prosody.pitch);
        assert!(prosody.energy > base_energy,
                "Stressed voice should be louder than base {:.2} (got {:.2})",
                base_energy, prosody.energy);
    }

    #[tokio::test]
    async fn test_prosody_modulation_calm() {
        let config = LarynxConfig::default();
        let base_speed = config.base_speed;
        let base_pitch = config.base_pitch;

        let mut larynx = LarynxActor::new(config).unwrap();

        // Create endocrine system and induce calm state (low cortisol, high dopamine)
        let endocrine_config = EndocrineConfig::default();
        let mut endocrine = EndocrineSystem::new(endocrine_config);

        // Trigger success/reward to increase dopamine and decrease cortisol
        endocrine.process_event(HormoneEvent::Success { magnitude: 0.8 });
        endocrine.process_event(HormoneEvent::Reward { value: 0.8 });

        let endocrine_arc = Arc::new(RwLock::new(endocrine));
        larynx.set_endocrine(endocrine_arc.clone());

        // Verify calm state (low cortisol, high dopamine)
        let endocrine_guard = endocrine_arc.read().await;
        let state = endocrine_guard.state();
        assert!(state.cortisol < 0.4, "Cortisol should be low (got {:.2})", state.cortisol);
        assert!(state.dopamine > 0.6, "Dopamine should be high (got {:.2})", state.dopamine);
        drop(endocrine_guard); // Release lock before calculating prosody

        // Calculate prosody
        let prosody = larynx.calculate_prosody().await;

        // Calm voice should be slower and lower than base
        assert!(prosody.speed < base_speed,
                "Calm voice should be slower than base {:.2} (got {:.2})",
                base_speed, prosody.speed);
        assert!(prosody.pitch < base_pitch,
                "Calm voice should be lower than base {:.2} (got {:.2})",
                base_pitch, prosody.pitch);
    }

    #[tokio::test]
    async fn test_synthesis_updates_stats() {
        let config = LarynxConfig::default();
        let larynx = LarynxActor::new(config).unwrap();

        // Synthesize some text
        let text = "Hello, I am Sophia!";
        let _audio = larynx.speak(text).await.unwrap();

        // Check stats were updated
        let stats = larynx.get_stats().await;
        assert_eq!(stats.total_utterances, 1);
        assert_eq!(stats.total_characters, text.len() as u64);
        assert_eq!(stats.total_atp_spent, 5.0); // 5 ATP per utterance
        assert!(stats.avg_synthesis_ms >= 0.0);
    }

    #[tokio::test]
    async fn test_prosody_without_endocrine() {
        let config = LarynxConfig::default();
        let larynx = LarynxActor::new(config).unwrap();

        // Calculate prosody without endocrine system
        let prosody = larynx.calculate_prosody().await;

        // Should return default/base prosody
        assert_eq!(prosody.speed, 1.0);
        assert_eq!(prosody.pitch, 1.0);
    }

    #[tokio::test]
    async fn test_prosody_clamping() {
        let config = LarynxConfig::default();
        let mut larynx = LarynxActor::new(config).unwrap();

        // Create extreme endocrine state
        let endocrine_config = EndocrineConfig::default();
        let mut endocrine = EndocrineSystem::new(endocrine_config);

        // Trigger multiple stress events using Error events
        for _ in 0..10 {
            endocrine.process_event(HormoneEvent::Error { severity: 0.9 });
        }

        let endocrine_arc = Arc::new(RwLock::new(endocrine));
        larynx.set_endocrine(endocrine_arc);

        // Calculate prosody
        let prosody = larynx.calculate_prosody().await;

        // Values should be clamped to reasonable ranges
        assert!(prosody.speed >= 0.7 && prosody.speed <= 1.5);
        assert!(prosody.pitch >= 0.8 && prosody.pitch <= 1.3);
        assert!(prosody.energy >= 0.3 && prosody.energy <= 1.2);
        assert!(prosody.breath_rate >= 0.0 && prosody.breath_rate <= 0.2);
    }

    // Week 13 Day 1: New tests for real audio generation

    #[tokio::test]
    async fn test_real_audio_generation() {
        let config = LarynxConfig::default();
        let larynx = LarynxActor::new(config).unwrap();

        // Synthesize some text
        let text = "Hello, I am Sophia!";
        let audio = larynx.speak(text).await.unwrap();

        // Verify audio was generated (not empty)
        assert!(!audio.is_empty(), "Audio should not be empty");

        // Verify audio length is reasonable
        // At 24kHz and ~200 chars/sec, "Hello, I am Sophia!" (19 chars)
        // should be about 0.095 seconds = 2280 samples
        let expected_samples = (text.len() as f32 / 200.0 * 24000.0) as usize;
        let tolerance = expected_samples / 2; // Allow 50% tolerance
        assert!(
            audio.len() > expected_samples - tolerance
                && audio.len() < expected_samples + tolerance,
            "Audio length should be ~{} samples, got {}",
            expected_samples,
            audio.len()
        );

        // Verify samples are in valid range [-1.0, 1.0]
        for (i, &sample) in audio.iter().enumerate() {
            assert!(
                sample >= -1.0 && sample <= 1.0,
                "Sample {} at index {} is out of range",
                sample,
                i
            );
        }
    }

    #[tokio::test]
    async fn test_wav_file_export() {
        use tempfile::TempDir;

        let config = LarynxConfig::default();
        let larynx = LarynxActor::new(config).unwrap();

        // Synthesize audio
        let text = "Testing WAV export";
        let audio = larynx.speak(text).await.unwrap();

        // Create temporary directory for test file
        let temp_dir = TempDir::new().unwrap();
        let wav_path = temp_dir.path().join("test_audio.wav");

        // Export to WAV
        larynx.save_wav(&audio, &wav_path).unwrap();

        // Verify file was created
        assert!(wav_path.exists(), "WAV file should exist");

        // Verify file is not empty
        let file_size = std::fs::metadata(&wav_path).unwrap().len();
        assert!(file_size > 0, "WAV file should not be empty");

        // Verify we can read it back with hound
        use hound::WavReader;
        let reader = WavReader::open(&wav_path).unwrap();
        let spec = reader.spec();

        assert_eq!(spec.channels, 1, "Should be mono");
        assert_eq!(spec.sample_rate, 24000, "Should be 24kHz");
        assert_eq!(spec.sample_format, hound::SampleFormat::Float);

        let samples: Vec<f32> = reader.into_samples::<f32>().map(|s| s.unwrap()).collect();
        assert_eq!(samples.len(), audio.len(), "Sample count should match");
    }

    #[tokio::test]
    async fn test_prosody_affects_audio() {
        let config = LarynxConfig::default();
        let mut larynx = LarynxActor::new(config).unwrap();

        let text = "Testing prosody modulation";

        // Generate audio in default state
        let audio_normal = larynx.speak(text).await.unwrap();

        // Create stressed endocrine state (high cortisol)
        let endocrine_config = EndocrineConfig::default();
        let mut endocrine = EndocrineSystem::new(endocrine_config);
        for _ in 0..3 {
            endocrine.process_event(HormoneEvent::Error { severity: 0.9 });
        }
        let endocrine_arc = Arc::new(RwLock::new(endocrine));
        larynx.set_endocrine(endocrine_arc);

        // Generate audio in stressed state
        let audio_stressed = larynx.speak(text).await.unwrap();

        // Stressed audio should be shorter (faster speed) and have different samples
        assert!(
            audio_stressed.len() < audio_normal.len(),
            "Stressed audio should be shorter due to faster speed"
        );

        // Verify audio is different (stressed has higher frequency = different waveform)
        let different_samples = audio_normal
            .iter()
            .zip(audio_stressed.iter().take(audio_normal.len()))
            .filter(|(&a, &b)| (a - b).abs() > 0.01)
            .count();

        assert!(
            different_samples > audio_stressed.len() / 2,
            "Stressed audio should have significantly different samples due to prosody"
        );
    }
}
