//! Piper TTS Output - Fast native Rust text-to-speech
//!
//! Uses piper-rs crate which provides:
//! - Built-in espeak-rs for phonemization (correct for Piper models)
//! - VITS-based synthesis (fast, good quality)
//! - Many pre-trained voices from HuggingFace
//!
//! # Performance
//! - Typically faster than real-time on modern hardware
//! - No separate tokenizer needed (espeak-rs integrated)

use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PiperError {
    #[error("Piper model not found: {0}")]
    ModelNotFound(String),
    #[error("Synthesis failed: {0}")]
    SynthesisFailed(String),
    #[error("Audio output error: {0}")]
    AudioError(String),
    #[error("Feature not enabled: compile with --features voice-tts-piper")]
    FeatureNotEnabled,
}

/// Piper-based TTS output (recommended over Kokoro for Rust)
#[cfg(feature = "voice-tts-piper")]
pub struct PiperVoiceOutput {
    model: Box<dyn piper_rs::PiperModel + Send + Sync>,
    sample_rate: u32,
}

#[cfg(feature = "voice-tts-piper")]
impl PiperVoiceOutput {
    /// Create a new Piper voice output
    ///
    /// # Arguments
    /// * `config_path` - Path to the voice model config (.onnx.json file)
    /// * `speaker_id` - Optional speaker ID for multi-speaker models
    pub fn new(config_path: impl AsRef<Path>, speaker_id: Option<i64>) -> Result<Self, PiperError> {
        let config_path = config_path.as_ref();

        if !config_path.exists() {
            return Err(PiperError::ModelNotFound(config_path.display().to_string()));
        }

        // Load the model using piper-rs API
        let model = piper_rs::from_config_path(config_path)
            .map_err(|e| PiperError::ModelNotFound(format!("{:?}", e)))?;

        // Set speaker if specified
        if let Some(id) = speaker_id {
            model.set_speaker(id);
        }

        // Get sample rate from model
        let audio_info = model.audio_output_info();
        let sample_rate = audio_info.sample_rate as u32;

        Ok(Self {
            model,
            sample_rate,
        })
    }

    /// Synthesize text to audio samples
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>, PiperError> {
        // Get phonemes first
        let phonemes = self.model.phonemize_text(text, None)
            .map_err(|e| PiperError::SynthesisFailed(format!("{:?}", e)))?;

        // Synthesize each sentence
        let mut all_samples = Vec::new();
        for phoneme in phonemes {
            let audio = self.model.speak_one_sentence(&phoneme, None)
                .map_err(|e| PiperError::SynthesisFailed(format!("{:?}", e)))?;

            // Get samples and convert to f32
            let samples = audio.into_samples();
            all_samples.extend(samples.iter().map(|&s| s as f32 / 32768.0));
        }

        Ok(all_samples)
    }

    /// Synthesize text and save to WAV file
    #[cfg(feature = "hound")]
    pub fn synthesize_to_file(&self, text: &str, output_path: impl AsRef<Path>) -> Result<(), PiperError> {
        let samples = self.synthesize(text)?;

        // Write WAV file
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = hound::WavWriter::create(output_path.as_ref(), spec)
            .map_err(|e| PiperError::AudioError(e.to_string()))?;

        for sample in &samples {
            let sample_i16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
            writer.write_sample(sample_i16)
                .map_err(|e| PiperError::AudioError(e.to_string()))?;
        }

        writer.finalize()
            .map_err(|e| PiperError::AudioError(e.to_string()))?;

        Ok(())
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Play audio through default output device
    #[cfg(feature = "rodio")]
    pub fn speak(&self, text: &str) -> Result<(), PiperError> {
        use rodio::{OutputStream, Sink, buffer::SamplesBuffer};

        let samples = self.synthesize(text)?;

        let (_stream, stream_handle) = OutputStream::try_default()
            .map_err(|e| PiperError::AudioError(e.to_string()))?;

        let sink = Sink::try_new(&stream_handle)
            .map_err(|e| PiperError::AudioError(e.to_string()))?;

        let source = SamplesBuffer::new(1, self.sample_rate, samples);
        sink.append(source);
        sink.sleep_until_end();

        Ok(())
    }
}

/// Stub for when feature is not enabled
#[cfg(not(feature = "voice-tts-piper"))]
pub struct PiperVoiceOutput;

#[cfg(not(feature = "voice-tts-piper"))]
impl PiperVoiceOutput {
    pub fn new(_config_path: impl AsRef<Path>, _speaker_id: Option<i64>) -> Result<Self, PiperError> {
        Err(PiperError::FeatureNotEnabled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "voice-tts-piper")]
    fn test_piper_synthesis() {
        let config_path = "models/piper/en_US-amy-medium.onnx.json";
        if !Path::new(config_path).exists() {
            println!("Piper model not found, skipping test");
            return;
        }

        let output = PiperVoiceOutput::new(config_path, None).unwrap();
        let samples = output.synthesize("Hello world").unwrap();

        assert!(!samples.is_empty());
        println!("Generated {} samples at {} Hz", samples.len(), output.sample_rate());
    }
}
