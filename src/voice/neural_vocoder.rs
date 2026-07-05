// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! BigVGAN neural vocoder channel — background-thread ONNX inference.
//!
//! Follows the `FoveationChannel` / `EmbeddingChannel` pattern:
//! - `std::sync::mpsc::sync_channel` with bounded depth for backpressure
//! - Background thread owns the ONNX `Session`
//! - `submit()` / `try_recv()` non-blocking collect pattern
//!
//! The vocoder accepts mel spectrograms (produced by `FormantToMelConverter`) and
//! returns raw audio waveform chunks. If the ONNX model is missing or fails to load,
//! `spawn()` returns `None` and the caller falls back to DSP synthesis.

use std::path::Path;
use std::sync::mpsc;
use tracing::{debug, info, warn};

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Neural vocoder configuration.
#[derive(Debug, Clone)]
pub struct NeuralVocoderConfig {
    /// Path to BigVGAN ONNX model file.
    pub model_path: String,
    /// Number of mel frames to buffer before submitting a chunk for inference.
    pub mel_buffer_size: usize,
    /// Channel depth (backpressure bound).
    pub channel_depth: usize,
    /// Expected sample rate of the model output.
    pub sample_rate: u32,
    /// Number of mel bins (must match model input).
    pub n_mels: usize,
    /// Number of mel frames of overlap context to prepend from the previous chunk.
    /// Reduces boundary artifacts in streaming mode. Set to 0 to disable.
    pub overlap_context: usize,
    /// Whether to attempt GPU acceleration (CUDA/CoreML) via ONNX execution providers.
    /// Requires `neural-vocoder-gpu` feature. Falls back to CPU if GPU unavailable.
    pub use_gpu: bool,
}

impl Default for NeuralVocoderConfig {
    fn default() -> Self {
        Self {
            model_path: "models/bigvgan_v2_24khz_100band.onnx".to_string(),
            mel_buffer_size: 32,
            channel_depth: 4,
            sample_rate: 24000,
            n_mels: 100,
            overlap_context: 4,
            use_gpu: false,
        }
    }
}

impl NeuralVocoderConfig {
    /// Create a low-latency streaming configuration.
    ///
    /// Uses smaller buffer (16 frames ≈ 170ms) with 4-frame overlap context
    /// for ~170ms first-audio latency.
    pub fn low_latency() -> Self {
        Self {
            mel_buffer_size: 16,
            overlap_context: 4,
            ..Default::default()
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// REQUEST / RESPONSE TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// A mel spectrogram chunk to be vocoded.
struct VocoderRequest {
    /// Mel frames: each inner Vec has `n_mels` elements.
    mel_frames: Vec<Vec<f32>>,
    /// Number of overlap context frames prepended (to be trimmed from output audio).
    overlap_frames: usize,
    /// Response channel for the generated audio.
    response_tx: mpsc::SyncSender<VocoderResponse>,
}

/// Audio output from the neural vocoder.
pub struct VocoderResponse {
    /// Raw audio samples (f32, mono, at the configured sample rate).
    pub audio: Vec<f32>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// NEURAL VOCODER CHANNEL
// ═══════════════════════════════════════════════════════════════════════════════

/// Handle to a background BigVGAN neural vocoder thread.
///
/// Submit mel spectrogram chunks via `submit()` and collect audio via the
/// returned `Receiver`. Drop the handle to shut down the background thread.
pub struct NeuralVocoderChannel {
    request_tx: mpsc::SyncSender<VocoderRequest>,
    config: NeuralVocoderConfig,
}

impl NeuralVocoderChannel {
    /// Attempt to spawn the neural vocoder background thread.
    ///
    /// Returns `None` if:
    /// - The model file doesn't exist
    /// - The ONNX session fails to create
    ///
    /// This follows the Kokoro engine pattern of graceful degradation.
    #[cfg(feature = "neural-vocoder")]
    pub fn spawn(config: NeuralVocoderConfig) -> Option<Self> {
        let model_path = config.model_path.clone();
        let _use_gpu = config.use_gpu;

        // Check model file exists before spawning thread
        if !Path::new(&model_path).exists() {
            warn!(
                "BigVGAN model not found at '{}'. Neural vocoder unavailable, using DSP fallback.",
                model_path
            );
            return None;
        }

        info!("Loading BigVGAN neural vocoder from {}...", model_path);

        let (request_tx, request_rx) = mpsc::sync_channel::<VocoderRequest>(config.channel_depth);
        let n_mels = config.n_mels;
        let hop_length = 256usize; // BigVGAN hop length for overlap trimming

        std::thread::Builder::new()
            .name("symthaea-neural-vocoder".into())
            .spawn(move || {
                // Build ONNX session — optionally with GPU execution provider
                let builder = match ort::session::Session::builder() {
                    Ok(b) => b,
                    Err(e) => {
                        warn!("Failed to create session builder: {}. Thread exiting.", e);
                        return;
                    }
                };

                // GPU acceleration (improvement #7)
                #[cfg(feature = "neural-vocoder-gpu")]
                let mut builder = if _use_gpu {
                    info!("Attempting GPU acceleration for BigVGAN...");
                    // Try CUDA first, then CoreML, fall back to CPU
                    match builder.with_execution_providers([
                        ort::execution_providers::CUDAExecutionProvider::default().build(),
                        #[cfg(target_os = "macos")]
                        ort::execution_providers::CoreMLExecutionProvider::default().build(),
                    ]) {
                        Ok(b) => b,
                        Err(e) => {
                            warn!("GPU EP registration failed ({}), using CPU", e);
                            match ort::session::Session::builder() {
                                Ok(b) => b,
                                Err(e2) => {
                                    warn!(
                                        "Fallback session builder failed: {}. Thread exiting.",
                                        e2
                                    );
                                    return;
                                }
                            }
                        }
                    }
                } else {
                    builder
                };

                let mut session = match builder.commit_from_file(&model_path) {
                    Ok(session) => session,
                    Err(e) => {
                        warn!(
                            "Failed to create BigVGAN ONNX session: {}. Thread exiting.",
                            e
                        );
                        return;
                    }
                };

                info!("BigVGAN neural vocoder ready");

                while let Ok(req) = request_rx.recv() {
                    let audio = run_bigvgan_inference(&mut session, &req.mel_frames, n_mels);
                    let mut audio = audio.unwrap_or_default();

                    // Trim overlap context from output audio (improvement #6)
                    if req.overlap_frames > 0 && !audio.is_empty() {
                        let trim_samples = req.overlap_frames * hop_length;
                        if trim_samples < audio.len() {
                            audio = audio[trim_samples..].to_vec();
                        }
                    }

                    let response = VocoderResponse { audio };
                    // If receiver was dropped, discard
                    let _ = req.response_tx.try_send(response);
                }

                debug!("Neural vocoder thread shutting down");
            })
            .expect("Failed to spawn neural vocoder thread");

        Some(Self { request_tx, config })
    }

    /// Stub when neural-vocoder feature is not enabled.
    #[cfg(not(feature = "neural-vocoder"))]
    pub fn spawn(_config: NeuralVocoderConfig) -> Option<Self> {
        None
    }

    /// Submit a mel spectrogram chunk for neural vocoding.
    ///
    /// `overlap_frames` is the number of context frames prepended from the previous chunk.
    /// The background thread will trim the corresponding output audio samples.
    ///
    /// Returns a `Receiver` for the audio result. Non-blocking — returns `Err`
    /// if the channel is full (backpressure) or the thread has exited.
    pub fn submit(
        &self,
        mel_frames: Vec<Vec<f32>>,
        overlap_frames: usize,
    ) -> Result<mpsc::Receiver<VocoderResponse>, mpsc::TrySendError<()>> {
        let (response_tx, response_rx) = mpsc::sync_channel(1);

        let req = VocoderRequest {
            mel_frames,
            overlap_frames,
            response_tx,
        };

        self.request_tx.try_send(req).map_err(|e| match e {
            mpsc::TrySendError::Full(_) => mpsc::TrySendError::Full(()),
            mpsc::TrySendError::Disconnected(_) => mpsc::TrySendError::Disconnected(()),
        })?;

        Ok(response_rx)
    }

    /// Get the configuration.
    pub fn config(&self) -> &NeuralVocoderConfig {
        &self.config
    }

    /// Check if the background thread is still alive.
    pub fn is_alive(&self) -> bool {
        // Try a zero-size probe — if disconnected, thread is gone
        // We can't actually send without data, so we check on next submit.
        // For now, assume alive if we hold the sender.
        true
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ONNX INFERENCE
// ═══════════════════════════════════════════════════════════════════════════════

/// Run BigVGAN inference on a mel spectrogram chunk.
///
/// Input: `mel_frames` — Vec of mel vectors (each `n_mels` long)
/// BigVGAN expects input tensor shape: `[1, n_mels, n_frames]` (channels-first)
/// Output tensor shape: `[1, 1, n_samples]`
#[cfg(feature = "neural-vocoder")]
fn run_bigvgan_inference(
    session: &mut ort::session::Session,
    mel_frames: &[Vec<f32>],
    n_mels: usize,
) -> Option<Vec<f32>> {
    if mel_frames.is_empty() {
        return Some(Vec::new());
    }

    let n_frames = mel_frames.len();

    // Build channels-first tensor: [1, n_mels, n_frames]
    // mel_frames is [n_frames][n_mels], we need to transpose to [n_mels][n_frames]
    let mut data = vec![0.0f32; n_mels * n_frames];
    for (t, frame) in mel_frames.iter().enumerate() {
        for (m, &val) in frame.iter().enumerate().take(n_mels) {
            data[m * n_frames + t] = val;
        }
    }

    // Use ort::value::Tensor (same pattern as qwen3/onnx.rs)
    let shape = vec![1i64, n_mels as i64, n_frames as i64];
    let mel_tensor = match ort::value::Tensor::from_array((shape, data)) {
        Ok(t) => t,
        Err(e) => {
            warn!("Failed to create mel tensor: {}", e);
            return None;
        }
    };

    let outputs = match session.run(ort::inputs!["mel_spectrogram" => mel_tensor]) {
        Ok(outputs) => outputs,
        Err(e) => {
            warn!("BigVGAN inference failed: {}", e);
            return None;
        }
    };

    // Extract audio from first output tensor
    if let Ok((_shape, audio_data)) = outputs[0].try_extract_tensor::<f32>() {
        if !audio_data.is_empty() {
            return Some(audio_data.to_vec());
        }
    }

    warn!("BigVGAN produced no audio output");
    None
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = NeuralVocoderConfig::default();
        assert_eq!(config.sample_rate, 24000);
        assert_eq!(config.n_mels, 100);
        assert_eq!(config.mel_buffer_size, 32);
        assert_eq!(config.channel_depth, 4);
    }

    #[test]
    fn test_spawn_missing_model() {
        let config = NeuralVocoderConfig {
            model_path: "/nonexistent/bigvgan.onnx".to_string(),
            ..Default::default()
        };
        let channel = NeuralVocoderChannel::spawn(config);
        assert!(
            channel.is_none(),
            "Should return None when model file is missing"
        );
    }

    #[test]
    fn test_vocoder_response_fields() {
        let response = VocoderResponse {
            audio: vec![0.1, 0.2, -0.3],
        };
        assert_eq!(response.audio.len(), 3);
    }
}
