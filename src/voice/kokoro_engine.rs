// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Kokoro TTS Engine — **VERIFIED LIVE 2026-07-16** (voice plan LF4)
//!
//! Neural text-to-speech using the Kokoro ONNX model.
//! Downloads the model from HuggingFace Hub on first use.
//! Gracefully degrades to None when the model is unavailable.
//!
//! ## Live verification (first successful run in this codebase)
//!
//! `examples/test_tts_pipeline` on 2026-07-16: model loaded from
//! onnx-community (fp32 ONNX), 510 style rows parsed, synthesized
//! "Hello, this is a test." → 2.05s audio → **Whisper large-v3 transcribed
//! it verbatim (0% WER)**. The three historic correctness gaps (wrong 45-ID
//! G2P vocab; voice pack misread as flat rows instead of per-voice
//! `[510, 256]` style tables; missing 0-padding) are fixed and confirmed.
//!
//! Runtime notes: needs `ORT_DYLIB_PATH` pointing at libonnxruntime.so
//! (the symthaea flake devShell exports it, but `nix develop -c <bin>` does
//! NOT run the shellHook -- set it manually) and espeak-ng for G2P.
//! CPU-only real-time factor measured 2026-07-18 (`examples/kokoro_realtime_factor.rs`,
//! engine loaded once, 5 sentences of varying length): RTF ~1.42 overall,
//! worsening on longer utterances (RTF 1.81 on the longest test sentence) --
//! slower than real time, fine for the distillation-teacher role (generating
//! reference audio to train the vocal-tract controller against, scored by
//! round-trip WER) but not yet comfortably interactive on CPU alone.
//! `KokoroConfig::use_gpu` (+ the `voice-tts-gpu` feature + a CUDA-enabled
//! `ORT_DYLIB_PATH`) requests the CUDA execution provider, with automatic
//! fallback to CPU if registration fails. **GPU verified live 2026-07-18**
//! (RTX 2070): RTF ~0.22, a ~6x speedup over CPU, comfortably faster than
//! real time. Getting there required an ort 2.0-rc.10 workaround -- see the
//! comment above the `ort::init().commit()` call in `load()` -- without it
//! CUDA EP registration silently fails and falls back to CPU (ort logs the
//! failure but still returns `Ok`, so it isn't visible without a tracing
//! subscriber). The consciousness coupling (`speed`, `voice_blend`) was
//! ported from the retired symthaea-voice crate.

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
    /// Per-voice style-table file (`voices/<name>.bin`): 510 rows × 256 f32,
    /// row selected by input token count (Kokoro v1.0 semantics).
    pub voices_filename: String,
    /// Sample rate of the output audio (Kokoro uses 24kHz).
    pub sample_rate: u32,
    /// Default voice index (legacy; with per-voice style files the style row
    /// is selected by token count, not by this index).
    pub default_voice: usize,
    /// Request CUDA execution provider acceleration (needs the
    /// `voice-tts-gpu` feature AND `ORT_DYLIB_PATH` pointing at a
    /// CUDA-enabled onnxruntime build). Measured 2026-07-18 on an RTX 2070:
    /// RTF ~0.22 (a ~6x speedup over CPU's ~1.42), see
    /// `examples/kokoro_realtime_factor.rs`. Falls back to CPU (with a
    /// warning) when the feature is off, when this is false, or when CUDA
    /// EP registration itself fails.
    pub use_gpu: bool,
}

impl Default for KokoroConfig {
    fn default() -> Self {
        Self {
            // LF4 fix (2026-07-15): the old default pointed at
            // hexgrad/Kokoro-82M-v1.0-ONNX with a single voices-v1.0.bin —
            // that repo now 401s and the flat parse misread the style tables.
            // onnx-community mirrors the same model with per-voice files.
            repo_id: "onnx-community/Kokoro-82M-v1.0-ONNX".to_string(),
            model_filename: "onnx/model.onnx".to_string(),
            voices_filename: "voices/af_heart.bin".to_string(),
            sample_rate: 24000,
            default_voice: 0,
            use_gpu: false,
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
    /// Liquid Kokoro: consciousness-driven speech rate (set before each
    /// synthesize call). Confused/low Ψ ≈ 0.75x, confident/high Ψ ≈ 1.1x.
    /// Ported from the retired symthaea-voice crate (2026-07-15).
    pub speed: Option<f32>,
    /// Liquid Kokoro: consciousness-driven voice blend. `(valence, arousal)`
    /// interpolates 30% of a secondary voice embedding into the base voice.
    pub voice_blend: Option<(f32, f32)>,
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

        // ort 2.0-rc.10 quirk (found 2026-07-18 wiring GPU support): if the
        // process-global `Environment` hasn't been explicitly committed
        // before the CUDA execution provider is registered, ort's provider
        // bridge hits "Attempt to use DefaultLogger but none has been
        // registered" and CUDA EP registration silently fails (falls back
        // to CPU with only an ort-internal WARN -- `with_execution_providers`
        // itself still returns `Ok`, so this can't be caught downstream).
        // Committing the environment first avoids the ordering bug.
        #[cfg(feature = "voice-tts-gpu")]
        if config.use_gpu {
            if let Err(e) = ort::init().commit() {
                warn!("Failed to pre-commit ort environment for GPU: {}", e);
            }
        }

        // Create ONNX Runtime session (ort 2.0 API)
        let builder = match ort::session::Session::builder() {
            Ok(b) => b,
            Err(e) => {
                warn!("Failed to create session builder: {}. TTS unavailable.", e);
                return None;
            }
        };

        #[cfg(feature = "voice-tts-gpu")]
        let builder = if config.use_gpu {
            info!("Attempting GPU (CUDA) acceleration for Kokoro...");
            match builder.with_execution_providers([
                ort::execution_providers::CUDAExecutionProvider::default().build(),
            ]) {
                Ok(b) => b,
                Err(e) => {
                    warn!("CUDA EP registration failed ({}), falling back to CPU", e);
                    match ort::session::Session::builder() {
                        Ok(b) => b,
                        Err(e2) => {
                            warn!("Fallback session builder failed: {}. TTS unavailable.", e2);
                            return None;
                        }
                    }
                }
            }
        } else {
            builder
        };

        let session = match builder.commit_from_file(&model_path) {
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

        info!(
            "Kokoro TTS loaded: {} style rows in {}",
            voices.len(),
            config.voices_filename
        );
        Some(Self {
            session,
            voices,
            config,
            g2p: G2PConverter::new(),
            speed: None,
            voice_blend: None,
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
    ///
    /// LF4 fix (2026-07-15): tokens now come from the REAL Kokoro-82M
    /// vocabulary (`g2p::text_to_kokoro_tokens`, no approximate fallback),
    /// the sequence is padded with the model's expected leading/trailing 0,
    /// and the style row is selected by token count from the per-voice
    /// `[510, 256]` style table — the three correctness gaps that made the
    /// old path produce garble by construction.
    #[cfg(feature = "voice-tts")]
    pub fn synthesize(&mut self, text: &str, _voice_id: Option<usize>) -> Option<Vec<f32>> {
        let raw_ids = self.g2p.text_to_kokoro_tokens(text)?;
        if raw_ids.is_empty() {
            return None;
        }

        // Model expects 0-padding at start and end.
        let mut phoneme_ids = Vec::with_capacity(raw_ids.len() + 2);
        phoneme_ids.push(0u32);
        phoneme_ids.extend(&raw_ids);
        phoneme_ids.push(0u32);

        // Style row = token count (Kokoro v1.0 per-voice table semantics).
        let style_row = phoneme_ids.len().min(self.voices.len().saturating_sub(1));
        let voice_embed = self
            .voices
            .get(style_row)
            .or_else(|| self.voices.last())
            .or_else(|| self.voices.first())?;

        // Liquid Kokoro: blend voice embeddings from consciousness state.
        // High valence = warmer voice (higher index), high arousal = brighter.
        // 70% base + 30% consciousness-selected secondary embedding.
        // Only meaningful for genuine multi-voice packs; with a per-voice
        // [510, 256] style table the "other rows" are other token counts,
        // so blending would inject a wrong-length style — skip it there.
        let looks_like_style_table = self.voices.len() >= 100;
        let style: Vec<f32> = if let (Some((valence, arousal)), false) =
            (self.voice_blend, looks_like_style_table)
        {
            let blend_idx = ((0.5 + valence * 0.3 + arousal * 0.2) * self.voices.len() as f32)
                .clamp(0.0, self.voices.len() as f32 - 1.0) as usize;
            let blend_voice = self.voices.get(blend_idx).unwrap_or(voice_embed);
            let t = 0.3;
            voice_embed
                .iter()
                .zip(blend_voice.iter())
                .map(|(&a, &b)| a * (1.0 - t) + b * t)
                .collect()
        } else {
            voice_embed.clone()
        };

        // Prepare input tensors for ort 2.0 API (Tensor::from_array)
        let seq_len = phoneme_ids.len();
        let input_ids: Vec<i64> = phoneme_ids.iter().map(|&id| id as i64).collect();

        let input_ids_tensor =
            ort::value::Tensor::from_array((vec![1i64, seq_len as i64], input_ids)).ok()?;
        let style_tensor =
            ort::value::Tensor::from_array((vec![1i64, voice_embed.len() as i64], style)).ok()?;
        // Liquid Kokoro: consciousness drives speech rate (1.0 when unset).
        let speed = self.speed.unwrap_or(1.0);
        let speed_tensor = ort::value::Tensor::from_array((vec![1i64], vec![speed])).ok()?;

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

/// Parse a per-voice style table into style rows.
///
/// Kokoro v1.0 per-voice files (`voices/<name>.bin`) are `[510, 256]` f32
/// tables — one 256-dim style row per input token count. This reads them as
/// consecutive 256-f32 rows; `synthesize` selects the row matching the padded
/// token count. (The old doc claimed each row was a separate "voice" — that
/// misreading of the layout was LF4 gap #2.)
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
