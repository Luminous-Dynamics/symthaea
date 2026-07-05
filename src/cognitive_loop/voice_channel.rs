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

/// Snapshot of consciousness state needed for prosody modulation.
#[derive(Debug, Clone)]
pub struct VoiceRequest {
    /// Text to synthesize.
    pub text: String,
    /// CfC output vector (for cognitive voice bridge).
    pub cfc_output: Vec<f32>,
    /// Prediction error at time of generation.
    pub prediction_error: f32,
    /// Detected primitives for prosody.
    pub detected_primitives: Vec<String>,
    /// Cycle number (for ordering/debug).
    pub cycle_num: u64,
}

/// Completed audio from the background synthesis thread.
#[derive(Debug, Clone)]
pub struct VoiceResponse {
    /// PCM audio samples (f32, typically 16kHz or 22kHz).
    pub audio: Vec<f32>,
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

        // Block waiting for next request
        while let Ok(request) = request_rx.recv() {
            // If multiple requests queued, skip to the latest (latest-wins)
            let mut latest = request;
            while let Ok(newer) = request_rx.try_recv() {
                latest = newer;
            }

            // Build a minimal CycleResult for the orchestrator
            let cycle_result = crate::cognitive_loop::types::CycleResult {
                output: latest.cfc_output.clone(),
                prediction_error: latest.prediction_error,
                peak_attention: 0.0,
                detected_primitives: latest.detected_primitives.clone(),
                learning_occurred: false,
                training_loss: None,
                cycle_time_us: 0,
                metadata: Default::default(),
                thought_vector: Vec::new(),
                wisdom_hv: symthaea_core::hdc::BinaryHV::zero(),
                language_output: None,
                language_source: None,
                #[cfg(feature = "canvas")]
                canvas_svg: None,
                #[cfg(feature = "identity")]
                signed_output: None,
                #[cfg(feature = "identity")]
                assurance_level: Default::default(),
                #[cfg(feature = "vision-manifold")]
                mental_movie: None,
            };

            let audio = orchestrator.synthesize_from_cycle_result(&latest.text, &cycle_result);

            if !audio.is_empty() {
                let _ = response_tx.send(VoiceResponse {
                    audio,
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
            prediction_error: 0.1,
            detected_primitives: vec![],
            cycle_num: 1,
        };
        assert!(channel.send(request), "should send without blocking");
        // Give the thread a moment to process
        std::thread::sleep(std::time::Duration::from_millis(50));
        // Drain — may or may not have a response depending on synthesis speed
        let _responses = channel.drain_responses();
    }
}
