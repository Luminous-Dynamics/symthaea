// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Kokoro TTS Engine
//!
//! Neural text-to-speech using the Kokoro ONNX model.
//! Downloads the model from HuggingFace Hub on first use.
//! Gracefully degrades to None when the model is unavailable.

use anyhow::Result;
use tracing::warn;

#[cfg(feature = "voice-tts")]
#[allow(unused_imports)]
use tracing::info;

use super::g2p::G2PConverter;

/// Kokoro TTS engine configuration.
#[derive(Debug, Clone)]
pub struct KokoroConfig {
    /// HuggingFace model repository ID.
    pub repo_id: String,
    /// ONNX model filename within the repo.
    pub model_filename: String,
    /// Voice pack filename (voice embeddings).
    pub voices_filename: String,
    /// Sample rate of the output audio (Kokoro uses 24kHz).
    pub sample_rate: u32,
    /// Default voice index.
    pub default_voice: usize,
}

impl Default for KokoroConfig {
    fn default() -> Self {
        Self {
            repo_id: "hexgrad/Kokoro-82M-v1.0-ONNX".to_string(),
            model_filename: "kokoro-v1.0.onnx".to_string(),
            voices_filename: "voices-v1.0.bin".to_string(),
            sample_rate: 24000,
            default_voice: 0,
        }
    }
}

/// Kokoro TTS Engine wrapping an ONNX Runtime session.
pub struct KokoroEngine {
    #[cfg(feature = "voice-tts")]
    session: ort::session::Session,
    #[cfg(feature = "voice-tts")]
    voices: Vec<Vec<f32>>,
    config: KokoroConfig,
    g2p: G2PConverter,
}

impl KokoroEngine {
    /// Attempt to load the Kokoro TTS model.
    ///
    /// Downloads from HuggingFace Hub if not cached locally.
    /// Returns `None` if model loading fails (missing model, ONNX runtime issues, etc).
    #[cfg(feature = "voice-tts")]
    pub fn load(config: KokoroConfig) -> Option<Self> {
        info!("Loading Kokoro TTS model from {}...", config.repo_id);

        // Try to download model from HuggingFace Hub
        let api = match hf_hub::api::sync::Api::new() {
            Ok(api) => api,
            Err(e) => {
                warn!(
                    "Failed to initialize HuggingFace Hub API: {}. TTS unavailable.",
                    e
                );
                return None;
            }
        };

        let repo = api.model(config.repo_id.clone());

        // Download ONNX model
        let model_path = match repo.get(&config.model_filename) {
            Ok(path) => path,
            Err(e) => {
                warn!(
                    "Failed to download Kokoro model '{}': {}. TTS unavailable.",
                    config.model_filename, e
                );
                return None;
            }
        };

        // Create ONNX Runtime session (ort 2.0 API)
        let session = match ort::session::Session::builder()
            .and_then(|builder| builder.commit_from_file(&model_path))
        {
            Ok(session) => session,
            Err(e) => {
                warn!("Failed to create ONNX session: {}. TTS unavailable.", e);
                return None;
            }
        };

        // Load voice embeddings
        let voices = match repo.get(&config.voices_filename) {
            Ok(voices_path) => {
                match std::fs::read(&voices_path) {
                    Ok(data) => parse_voice_pack(&data, config.sample_rate),
                    Err(e) => {
                        warn!("Failed to read voices file: {}. Using empty voice pack.", e);
                        vec![vec![0.0f32; 256]] // Fallback: single zero voice
                    }
                }
            }
            Err(e) => {
                warn!("Failed to download voices: {}. Using default voice.", e);
                vec![vec![0.0f32; 256]]
            }
        };

        info!("Kokoro TTS loaded: {} voices available", voices.len());
        Some(Self {
            session,
            voices,
            config,
            g2p: G2PConverter::new(),
        })
    }

    /// Stub load when voice-tts feature is not enabled.
    #[cfg(not(feature = "voice-tts"))]
    pub fn load(_config: KokoroConfig) -> Option<Self> {
        warn!("Kokoro TTS requires the 'voice-tts' feature. TTS unavailable.");
        None
    }

    /// Synthesize speech from text.
    ///
    /// Returns audio samples at 24kHz, or `None` if synthesis fails.
    #[cfg(feature = "voice-tts")]
    pub fn synthesize(&mut self, text: &str, voice_id: Option<usize>) -> Option<Vec<f32>> {
        let phoneme_ids = self.g2p.text_to_phonemes(text);
        if phoneme_ids.is_empty() {
            return None;
        }

        let voice_idx = voice_id.unwrap_or(self.config.default_voice);
        let voice_embed = self.voices.get(voice_idx).or_else(|| self.voices.first())?;

        // Prepare input tensors for ort 2.0 API (Tensor::from_array)
        let seq_len = phoneme_ids.len();
        let input_ids: Vec<i64> = phoneme_ids.iter().map(|&id| id as i64).collect();
        let style: Vec<f32> = voice_embed.clone();

        let input_ids_tensor =
            ort::value::Tensor::from_array((vec![1i64, seq_len as i64], input_ids)).ok()?;
        let style_tensor =
            ort::value::Tensor::from_array((vec![1i64, voice_embed.len() as i64], style)).ok()?;
        let speed_tensor = ort::value::Tensor::from_array((vec![1i64], vec![1.0f32])).ok()?;

        let outputs = match self.session.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "style" => style_tensor,
            "speed" => speed_tensor,
        ]) {
            Ok(outputs) => outputs,
            Err(e) => {
                warn!("Kokoro inference failed: {}", e);
                return None;
            }
        };

        // Extract audio output
        if let Some(output) = outputs.get("audio") {
            if let Ok((_shape, data)) = output.try_extract_tensor::<f32>() {
                if !data.is_empty() {
                    return Some(data.to_vec());
                }
            }
        }

        // Try alternative output name (first output)
        if let Ok((_shape, data)) = outputs[0].try_extract_tensor::<f32>() {
            if !data.is_empty() {
                return Some(data.to_vec());
            }
        }

        warn!("Kokoro produced no audio output");
        None
    }

    /// Stub synthesize when voice-tts feature is not enabled.
    #[cfg(not(feature = "voice-tts"))]
    pub fn synthesize(&mut self, _text: &str, _voice_id: Option<usize>) -> Option<Vec<f32>> {
        None
    }

    /// Get the sample rate of the engine output.
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// Get the number of available voices.
    #[cfg(feature = "voice-tts")]
    pub fn num_voices(&self) -> usize {
        self.voices.len()
    }

    /// Get the number of available voices (stub).
    #[cfg(not(feature = "voice-tts"))]
    pub fn num_voices(&self) -> usize {
        0
    }

    /// Get the G2P converter.
    pub fn g2p(&self) -> &G2PConverter {
        &self.g2p
    }
}

/// Parse a voice pack binary file into voice embeddings.
///
/// Format: sequence of 256-dimensional f32 vectors.
#[cfg(feature = "voice-tts")]
fn parse_voice_pack(data: &[u8], _sample_rate: u32) -> Vec<Vec<f32>> {
    let embed_dim = 256;
    let bytes_per_embed = embed_dim * 4; // f32 = 4 bytes
    let num_voices = data.len() / bytes_per_embed;

    let mut voices = Vec::with_capacity(num_voices);
    for i in 0..num_voices {
        let start = i * bytes_per_embed;
        let end = start + bytes_per_embed;
        if end > data.len() {
            break;
        }
        let voice: Vec<f32> = data[start..end]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        voices.push(voice);
    }

    if voices.is_empty() {
        vec![vec![0.0f32; embed_dim]]
    } else {
        voices
    }
}

/// Save audio samples to a WAV file.
#[cfg(feature = "voice-tts")]
pub fn save_wav(samples: &[f32], sample_rate: u32, path: &str) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &sample in samples {
        let amplitude = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer.write_sample(amplitude)?;
    }
    writer.finalize()?;
    info!(
        "Saved WAV to {}: {} samples at {}Hz",
        path,
        samples.len(),
        sample_rate
    );
    Ok(())
}

/// Save audio samples to a WAV file (stub).
#[cfg(not(feature = "voice-tts"))]
pub fn save_wav(_samples: &[f32], _sample_rate: u32, _path: &str) -> Result<()> {
    anyhow::bail!("WAV saving requires the 'voice-tts' feature")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kokoro_config_default() {
        let config = KokoroConfig::default();
        assert_eq!(config.sample_rate, 24000);
        assert!(config.repo_id.contains("Kokoro"));
    }

    #[test]
    fn test_g2p_integration() {
        let g2p = G2PConverter::new();
        let phonemes = g2p.text_to_phonemes("hello world");
        assert!(!phonemes.is_empty());
    }

    #[cfg(feature = "voice-tts")]
    #[test]
    fn test_parse_voice_pack_empty() {
        let voices = parse_voice_pack(&[], 24000);
        assert_eq!(voices.len(), 1); // Fallback voice
        assert_eq!(voices[0].len(), 256);
    }

    #[cfg(feature = "voice-tts")]
    #[test]
    fn test_parse_voice_pack_one_voice() {
        // Create a fake voice pack: 256 f32 values
        let mut data = Vec::new();
        for i in 0..256 {
            data.extend_from_slice(&(i as f32 / 256.0).to_le_bytes());
        }
        let voices = parse_voice_pack(&data, 24000);
        assert_eq!(voices.len(), 1);
        assert_eq!(voices[0].len(), 256);
    }
}
