//! Voice Output Module - Text-to-Speech using Kokoro TTS
//!
//! Provides natural speech synthesis using Kokoro-82M via ONNX runtime.
//! Supports LTC-aware speech pacing for natural conversation flow.

use std::path::Path;
use std::sync::Arc;

use super::{VoiceError, VoiceResult, LTCPacing};
use super::models::{ModelManager, KokoroModel};

/// Configuration for voice output
#[derive(Debug, Clone)]
pub struct VoiceOutputConfig {
    /// Kokoro model to use
    pub model: KokoroModel,
    /// Output sample rate (default: 24000)
    pub sample_rate: u32,
    /// Base speech rate multiplier
    pub base_rate: f32,
    /// Enable LTC-aware pacing
    pub ltc_pacing: bool,
    /// Voice ID (0-9 for different voices)
    pub voice_id: u8,
}

impl Default for VoiceOutputConfig {
    fn default() -> Self {
        Self {
            model: KokoroModel::V019,
            sample_rate: 24000,
            base_rate: 1.0,
            ltc_pacing: true,
            voice_id: 0,
        }
    }
}

/// Result of speech synthesis
#[derive(Debug, Clone)]
pub struct SynthesisResult {
    /// Audio samples (f32 in [-1, 1])
    pub samples: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
    /// Duration in milliseconds
    pub duration_ms: u64,
}

impl SynthesisResult {
    /// Save to WAV file
    #[cfg(feature = "voice-tts")]
    pub fn save_wav(&self, path: &Path) -> VoiceResult<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        let mut writer = hound::WavWriter::create(path, spec)
            .map_err(|e| VoiceError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        for sample in &self.samples {
            writer.write_sample(*sample)
                .map_err(|e| VoiceError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
        }

        writer.finalize()
            .map_err(|e| VoiceError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

        Ok(())
    }
}

/// Voice output handler for speech synthesis
pub struct VoiceOutput {
    config: VoiceOutputConfig,
    #[cfg(feature = "voice-tts")]
    session: ort::Session,
}

impl VoiceOutput {
    /// Create a new voice output handler
    #[cfg(feature = "voice-tts")]
    pub fn new(config: VoiceOutputConfig) -> VoiceResult<Self> {
        let manager = ModelManager::new()?;
        let model_path = manager.kokoro_path(config.model)?;
        Self::from_model_path(&model_path, config)
    }

    /// Create from a specific model path
    #[cfg(feature = "voice-tts")]
    pub fn from_model_path(path: &Path, config: VoiceOutputConfig) -> VoiceResult<Self> {
        use ort::GraphOptimizationLevel;

        let session = ort::Session::builder()
            .map_err(|e| VoiceError::ModelLoad(format!("ONNX builder error: {}", e)))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| VoiceError::ModelLoad(format!("Optimization error: {}", e)))?
            .with_intra_threads(4)
            .map_err(|e| VoiceError::ModelLoad(format!("Thread config error: {}", e)))?
            .commit_from_file(path)
            .map_err(|e| VoiceError::ModelLoad(format!("Failed to load model: {}", e)))?;

        Ok(Self { config, session })
    }

    /// Synthesize speech from text
    #[cfg(feature = "voice-tts")]
    pub fn synthesize(&self, text: &str) -> VoiceResult<SynthesisResult> {
        self.synthesize_with_pacing(text, LTCPacing::default())
    }

    /// Synthesize speech with LTC-aware pacing
    #[cfg(feature = "voice-tts")]
    pub fn synthesize_with_pacing(&self, text: &str, pacing: LTCPacing) -> VoiceResult<SynthesisResult> {
        use ort::inputs;
        use ndarray::Array1;

        // Prepare text tokens (simplified - Kokoro uses phoneme input)
        let tokens = self.text_to_tokens(text);
        let token_len = tokens.len();

        // Create input tensors
        let tokens_array = Array1::from_vec(tokens);
        let voice_id = Array1::from_vec(vec![self.config.voice_id as i64]);

        // Calculate effective speech rate
        let rate = self.config.base_rate * pacing.speech_rate;
        let speed = Array1::from_vec(vec![rate]);

        // Run inference
        let outputs = self.session.run(inputs![
            "tokens" => tokens_array.view(),
            "voice" => voice_id.view(),
            "speed" => speed.view(),
        ].map_err(|e| VoiceError::Synthesis(format!("Input error: {}", e)))?)
            .map_err(|e| VoiceError::Synthesis(format!("Inference error: {}", e)))?;

        // Extract audio
        let audio_tensor = outputs.get("audio")
            .ok_or_else(|| VoiceError::Synthesis("No audio output".into()))?;

        let audio_array = audio_tensor
            .try_extract_tensor::<f32>()
            .map_err(|e| VoiceError::Synthesis(format!("Extract error: {}", e)))?;

        let samples: Vec<f32> = audio_array.view().iter().copied().collect();
        let duration_ms = (samples.len() as f64 / self.config.sample_rate as f64 * 1000.0) as u64;

        Ok(SynthesisResult {
            samples,
            sample_rate: self.config.sample_rate,
            duration_ms,
        })
    }

    /// Convert text to token IDs (simplified phoneme mapping)
    #[cfg(feature = "voice-tts")]
    fn text_to_tokens(&self, text: &str) -> Vec<i64> {
        // Kokoro uses a specific vocabulary - this is a simplified version
        // In production, you'd use the actual tokenizer from the model
        let mut tokens = vec![0i64];  // Start token

        for ch in text.chars() {
            let token = match ch.to_ascii_lowercase() {
                'a'..='z' => (ch as i64 - 'a' as i64) + 1,
                ' ' => 27,
                '.' => 28,
                ',' => 29,
                '?' => 30,
                '!' => 31,
                '\'' => 32,
                _ => 27,  // Unknown -> space
            };
            tokens.push(token);
        }

        tokens.push(0);  // End token
        tokens
    }

    /// Play audio through default output device
    #[cfg(all(feature = "voice-tts", feature = "rodio"))]
    pub fn play(&self, result: &SynthesisResult) -> VoiceResult<()> {
        use rodio::{OutputStream, Sink, Source};
        use std::time::Duration;

        let (_stream, stream_handle) = OutputStream::try_default()
            .map_err(|e| VoiceError::AudioDevice(format!("Output device error: {}", e)))?;

        let sink = Sink::try_new(&stream_handle)
            .map_err(|e| VoiceError::AudioDevice(format!("Sink error: {}", e)))?;

        // Create audio source from samples
        let source = SamplesSource {
            samples: result.samples.clone(),
            sample_rate: result.sample_rate,
            position: 0,
        };

        sink.append(source);
        sink.sleep_until_end();

        Ok(())
    }

    /// Play audio with LTC-aware pauses
    #[cfg(all(feature = "voice-tts", feature = "rodio"))]
    pub fn play_with_pacing(&self, result: &SynthesisResult, pacing: LTCPacing) -> VoiceResult<()> {
        // Play the audio
        self.play(result)?;

        // Add post-speech pause based on LTC state
        std::thread::sleep(std::time::Duration::from_millis(pacing.pause_ms as u64));

        Ok(())
    }
}

/// Audio source adapter for rodio
#[cfg(all(feature = "voice-tts", feature = "rodio"))]
struct SamplesSource {
    samples: Vec<f32>,
    sample_rate: u32,
    position: usize,
}

#[cfg(all(feature = "voice-tts", feature = "rodio"))]
impl Iterator for SamplesSource {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.samples.len() {
            let sample = self.samples[self.position];
            self.position += 1;
            Some(sample)
        } else {
            None
        }
    }
}

#[cfg(all(feature = "voice-tts", feature = "rodio"))]
impl rodio::Source for SamplesSource {
    fn current_frame_len(&self) -> Option<usize> {
        Some(self.samples.len() - self.position)
    }

    fn channels(&self) -> u16 {
        1
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn total_duration(&self) -> Option<std::time::Duration> {
        let secs = self.samples.len() as f64 / self.sample_rate as f64;
        Some(std::time::Duration::from_secs_f64(secs))
    }
}

/// Split text into sentences for natural pacing
pub fn split_sentences(text: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    let mut start = 0;

    for (i, c) in text.char_indices() {
        if c == '.' || c == '!' || c == '?' {
            let end = i + c.len_utf8();
            let sentence = text[start..end].trim();
            if !sentence.is_empty() {
                sentences.push(sentence);
            }
            start = end;
        }
    }

    // Handle remaining text
    let remaining = text[start..].trim();
    if !remaining.is_empty() {
        sentences.push(remaining);
    }

    sentences
}

/// Add natural pauses to text for better TTS
pub fn add_pauses(text: &str) -> String {
    text.replace(", ", ", ... ")
        .replace("; ", "; ... ")
        .replace(": ", ": ... ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_output_config_default() {
        let config = VoiceOutputConfig::default();
        assert_eq!(config.sample_rate, 24000);
        assert_eq!(config.base_rate, 1.0);
        assert!(config.ltc_pacing);
    }

    #[test]
    fn test_synthesis_result() {
        let result = SynthesisResult {
            samples: vec![0.1, 0.2, 0.3],
            sample_rate: 24000,
            duration_ms: 125,
        };
        assert_eq!(result.samples.len(), 3);
    }

    #[test]
    fn test_split_sentences() {
        let text = "Hello! How are you? I am fine.";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "Hello!");
        assert_eq!(sentences[1], "How are you?");
        assert_eq!(sentences[2], "I am fine.");
    }

    #[test]
    fn test_split_sentences_no_punctuation() {
        let text = "Hello world";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 1);
    }

    #[test]
    fn test_add_pauses() {
        let text = "First, second; third: fourth.";
        let with_pauses = add_pauses(text);
        assert!(with_pauses.contains(", ..."));
        assert!(with_pauses.contains("; ..."));
        assert!(with_pauses.contains(": ..."));
    }

    #[test]
    fn test_ltc_pacing_affects_rate() {
        let fast_pacing = LTCPacing::from_ltc(0.8, 0.05);
        let slow_pacing = LTCPacing::from_ltc(0.2, -0.05);

        assert!(fast_pacing.speech_rate > slow_pacing.speech_rate);
        assert!(fast_pacing.pause_ms < slow_pacing.pause_ms);
    }
}
