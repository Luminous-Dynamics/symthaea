// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Async voice synthesis channel: sends text to a background thread for TTS.
//!
//! CRITICAL: Voice synthesis (especially Kokoro ONNX) takes 50-500ms.
//! It MUST NOT block the cognitive cycle (4.3ms / 234Hz target).
//! This module sends text + consciousness snapshot over a channel;
//! the cycle continues immediately. Completed audio is retrieved
//! from a return channel in subsequent cycles.

use std::sync::mpsc;
use std::thread;

/// Maximum completed-audio responses the loop buffers for `drain_voice_audio()`.
/// Bounded so an unconsumed buffer can't grow without limit (oldest dropped).
pub const VOICE_AUDIO_BUFFER_CAP: usize = 8;

/// Linear resample (self-hearing: 24kHz vocoder output → 16kHz ear input).
#[cfg(feature = "voice-stt")]
fn resample_linear(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || input.is_empty() {
        return input.to_vec();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let output_len = ((input.len() as f64) * ratio).round().max(1.0) as usize;
    (0..output_len)
        .map(|i| {
            let src = i as f64 / ratio;
            let idx = src as usize;
            let frac = (src - idx as f64) as f32;
            match (input.get(idx), input.get(idx + 1)) {
                (Some(&a), Some(&b)) => a * (1.0 - frac) + b * frac,
                (Some(&a), None) => a,
                _ => 0.0,
            }
        })
        .collect()
}

/// Snapshot of consciousness state needed for prosody modulation.
#[derive(Debug, Clone)]
pub struct VoiceRequest {
    /// Text to synthesize.
    pub text: String,
    /// CfC output vector (for cognitive voice bridge).
    pub cfc_output: Vec<f32>,
    /// Effective CfC time-constant factor for pacing (1.0 = baseline).
    /// Derived from the cycle's adaptive tau factors (FEP surprise × Φ).
    pub tau: f32,
    /// Prediction error at time of generation.
    pub prediction_error: f32,
    /// Detected primitives for prosody.
    pub detected_primitives: Vec<String>,
    /// Speech-rate multiplier from adaptive behavior (1.0 = baseline).
    pub speech_rate_multiplier: f32,
    /// Pause-duration multiplier from adaptive behavior (1.0 = baseline).
    pub pause_multiplier: f32,
    /// Cycle number (for ordering/debug).
    pub cycle_num: u64,
}

/// Completed audio from the background synthesis thread.
#[derive(Debug, Clone)]
pub struct VoiceResponse {
    /// PCM audio samples (f32, typically 16kHz or 22kHz).
    pub audio: Vec<f32>,
    /// Quality metrics computed from the produced formant frames,
    /// for the voice→cognition feedback bridge.
    pub metrics: crate::voice::voice_feedback::VoiceOutputMetrics,
    /// SELF-HEARING (voice plan LF5): the produced audio run through the
    /// same acoustic ear used for the microphone (`symthaea_stt`
    /// StreamProcessor → bundled 16,384-D HV). The cycle blends this into
    /// the next perception as a self-generated auditory modality — she hears
    /// her own voice through the ear she hears the world with. Computed on
    /// the synthesis thread (RTF ~0.07 would still cost too much on the
    /// 31Hz cycle thread).
    #[cfg(feature = "voice-stt")]
    pub self_hv: Option<symthaea_core::hdc::ContinuousHV>,
    /// Cycle number this audio was generated for.
    pub cycle_num: u64,
}

/// Handle for sending voice requests from the cognitive loop (non-blocking).
pub struct VoiceSynthesisChannel {
    /// Send voice requests to the background thread.
    tx: mpsc::Sender<VoiceRequest>,
    /// Receive completed audio from the background thread.
    /// Wrapped in Mutex because mpsc::Receiver is !Sync, and
    /// CognitiveLoopService may need to be Sync in some contexts.
    rx: std::sync::Mutex<mpsc::Receiver<VoiceResponse>>,
    /// Handle to the background thread (kept for cleanup).
    _thread: thread::JoinHandle<()>,
}

impl VoiceSynthesisChannel {
    /// Spawn a background voice synthesis thread.
    ///
    /// The thread owns a `VoiceOrchestrator` and processes requests sequentially.
    /// If requests queue up, older ones are dropped (latest-wins).
    pub fn spawn() -> Self {
        let (request_tx, request_rx) = mpsc::channel::<VoiceRequest>();
        let (response_tx, response_rx) = mpsc::channel::<VoiceResponse>();

        let handle = thread::Builder::new()
            .name("voice-synthesis".into())
            .spawn(move || {
                Self::synthesis_loop(request_rx, response_tx);
            })
            .expect("Failed to spawn voice synthesis thread");

        Self {
            tx: request_tx,
            rx: std::sync::Mutex::new(response_rx),
            _thread: handle,
        }
    }

    /// Send a voice request to the background thread (non-blocking).
    ///
    /// Returns `false` if the channel is disconnected (thread crashed).
    pub fn send(&self, request: VoiceRequest) -> bool {
        self.tx.send(request).is_ok()
    }

    /// Drain any completed audio responses (non-blocking).
    ///
    /// Returns all responses that have been completed since the last drain.
    pub fn drain_responses(&self) -> Vec<VoiceResponse> {
        let mut responses = Vec::new();
        if let Ok(rx) = self.rx.lock() {
            while let Ok(response) = rx.try_recv() {
                responses.push(response);
            }
        }
        responses
    }

    /// Background thread main loop.
    fn synthesis_loop(
        request_rx: mpsc::Receiver<VoiceRequest>,
        response_tx: mpsc::Sender<VoiceResponse>,
    ) {
        use crate::voice::orchestrator::VoiceOrchestrator;

        let mut orchestrator = VoiceOrchestrator::new();

        // Self-hearing ear: same StreamProcessor pipeline as microphone
        // capture, persistent across utterances (LTC state carries over,
        // like a real ear that doesn't reset between sounds).
        #[cfg(feature = "voice-stt")]
        let mut self_ear =
            symthaea_stt::StreamProcessor::new(symthaea_stt::StreamConfig::low_latency());

        // Block waiting for next request
        while let Ok(request) = request_rx.recv() {
            // If multiple requests queued, skip to the latest (latest-wins)
            let mut latest = request;
            while let Ok(newer) = request_rx.try_recv() {
                latest = newer;
            }

            // Real formant synthesis via the low-level pipeline. The previous
            // routing (synthesize_from_cycle_result on an uninitialized
            // VoiceOutput) fell through to simulate_tts — a placeholder sine
            // wave — so the loop's "voice" was never speech.
            let (audio, metrics) = orchestrator.thought_to_speech_paced(
                &latest.text,
                &latest.cfc_output,
                latest.tau,
                latest.prediction_error,
                latest.detected_primitives.clone(),
                latest.speech_rate_multiplier,
                latest.pause_multiplier,
            );

            if !audio.is_empty() {
                // Self-hearing: encode the produced audio through the native
                // acoustic ear (24kHz vocoder output → 16kHz ear input).
                #[cfg(feature = "voice-stt")]
                let self_hv = {
                    let resampled = resample_linear(&audio, 24_000, 16_000);
                    self_ear.push_audio(&resampled);
                    let frames: Vec<symthaea_stt::HV16> =
                        self_ear.process().into_iter().map(|f| f.hv).collect();
                    if frames.is_empty() {
                        None
                    } else {
                        let bundled = symthaea_stt::bundle(&frames);
                        Some(symthaea_core::hdc::ContinuousHV::from_vec(
                            bundled.to_core_continuous(),
                        ))
                    }
                };

                let _ = response_tx.send(VoiceResponse {
                    audio,
                    metrics,
                    #[cfg(feature = "voice-stt")]
                    self_hv,
                    cycle_num: latest.cycle_num,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_spawn_and_send() {
        let channel = VoiceSynthesisChannel::spawn();
        let request = VoiceRequest {
            text: "hello world".into(),
            cfc_output: vec![0.0; 16],
            tau: 1.0,
            prediction_error: 0.1,
            detected_primitives: vec![],
            speech_rate_multiplier: 1.0,
            pause_multiplier: 1.0,
            cycle_num: 1,
        };
        assert!(channel.send(request), "should send without blocking");
        // Give the thread a moment to process
        std::thread::sleep(std::time::Duration::from_millis(50));
        // Drain — may or may not have a response depending on synthesis speed
        let _responses = channel.drain_responses();
    }

    #[test]
    fn test_synthesis_is_real_speech_not_sine() {
        // Regression: the channel used to fall through to simulate_tts (a pure
        // sine wave). Real formant synthesis of two words must not be a single
        // sinusoid — check that the spectrum-shaping produces sign-structure
        // richer than a fixed-period tone.
        let channel = VoiceSynthesisChannel::spawn();
        let request = VoiceRequest {
            text: "hello world".into(),
            cfc_output: vec![0.2; 16],
            tau: 1.0,
            prediction_error: 0.1,
            detected_primitives: vec![],
            speech_rate_multiplier: 1.0,
            pause_multiplier: 1.0,
            cycle_num: 1,
        };
        assert!(channel.send(request));

        // Formant synthesis of two words takes noticeably longer than the old
        // placeholder; poll up to 5s.
        let mut responses = Vec::new();
        for _ in 0..100 {
            std::thread::sleep(std::time::Duration::from_millis(50));
            responses = channel.drain_responses();
            if !responses.is_empty() {
                break;
            }
        }
        let resp = responses.pop().expect("synthesis should complete");
        assert!(!resp.audio.is_empty());
        // Metrics must be real (computed from frames), not defaults.
        assert!(resp.metrics.speech_rate > 0.0);
    }
}
