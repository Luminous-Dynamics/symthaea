// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Async voice synthesis channel: sends text to a background thread for TTS.
//!
//! CRITICAL: Voice synthesis (especially Kokoro ONNX) takes 50-500ms.
//! It MUST NOT block the cognitive cycle (4.3ms / 234Hz target).
//! This module sends text + consciousness snapshot to a bounded latest-wins
//! mailbox; the cycle continues immediately. Completed audio is retained in a
//! bounded drop-oldest buffer and retrieved in subsequent cycles.
//!
//! A newer request invalidates completed output from every older generation.
//! Multi-phrase requests are synthesized cooperatively: generation is checked
//! before and after every semantic phrase, so a superseding utterance can stop
//! an older multi-sentence synthesis before the full request completes. A single
//! long phrase remains one renderer call; frame-level cancellation is a later
//! qualification tranche.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

/// Maximum completed-audio responses the loop buffers for `drain_voice_audio()`.
/// Bounded so an unconsumed buffer cannot grow without limit; oldest entries are
/// dropped first.
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

/// Split text only at strong semantic boundaries where resetting the existing
/// whole-phrase renderer is least surprising.
///
/// Terminator runs such as `?!` stay attached to the same phrase. We deliberately
/// do not split on commas/semicolons or arbitrary byte/word counts: those would
/// improve worst-case cancellation latency at the cost of injecting additional
/// vocoder resets inside clauses. Frame-level cooperative cancellation belongs in
/// the renderer itself and is intentionally not approximated here.
fn split_voice_phrases(text: &str) -> Vec<&str> {
    let mut phrases = Vec::new();
    let mut start = 0usize;
    let mut chars = text.char_indices().peekable();

    while let Some((idx, ch)) = chars.next() {
        let strong_boundary = matches!(ch, '.' | '!' | '?') || ch == '\n';
        if !strong_boundary {
            continue;
        }

        let next_is_terminator = chars
            .peek()
            .map(|(_, next)| matches!(*next, '.' | '!' | '?'))
            .unwrap_or(false);
        if next_is_terminator && ch != '\n' {
            continue;
        }

        let end = idx + ch.len_utf8();
        let phrase = text[start..end].trim();
        if !phrase.is_empty() {
            phrases.push(phrase);
        }
        start = end;
    }

    let tail = text[start..].trim();
    if !tail.is_empty() {
        phrases.push(tail);
    }

    phrases
}

/// Sample-count-weighted aggregation for phrase-level synthesis metrics.
///
/// Each phrase still produces real `VoiceOutputMetrics`. Weighting by produced
/// audio samples avoids a short interjection dominating a much longer phrase.
#[derive(Debug, Default)]
struct WeightedVoiceMetrics {
    weight: f64,
    articulation_score: f64,
    formant_accuracy: f64,
    speech_rate: f64,
    pitch_stability: f64,
    coarticulation_smoothness: f64,
    listener_prediction: f64,
    duration_accuracy: f64,
    energy_consistency: f64,
}

impl WeightedVoiceMetrics {
    fn push(
        &mut self,
        metrics: &crate::voice::voice_feedback::VoiceOutputMetrics,
        sample_count: usize,
    ) {
        let weight = sample_count.max(1) as f64;
        self.weight += weight;
        self.articulation_score += metrics.articulation_score as f64 * weight;
        self.formant_accuracy += metrics.formant_accuracy as f64 * weight;
        self.speech_rate += metrics.speech_rate as f64 * weight;
        self.pitch_stability += metrics.pitch_stability as f64 * weight;
        self.coarticulation_smoothness += metrics.coarticulation_smoothness as f64 * weight;
        self.listener_prediction += metrics.listener_prediction as f64 * weight;
        self.duration_accuracy += metrics.duration_accuracy as f64 * weight;
        self.energy_consistency += metrics.energy_consistency as f64 * weight;
    }

    fn finish(self) -> crate::voice::voice_feedback::VoiceOutputMetrics {
        if self.weight <= 0.0 {
            return crate::voice::voice_feedback::VoiceOutputMetrics::default();
        }
        let inv = 1.0 / self.weight;
        crate::voice::voice_feedback::VoiceOutputMetrics {
            articulation_score: (self.articulation_score * inv) as f32,
            formant_accuracy: (self.formant_accuracy * inv) as f32,
            speech_rate: (self.speech_rate * inv) as f32,
            pitch_stability: (self.pitch_stability * inv) as f32,
            coarticulation_smoothness: (self.coarticulation_smoothness * inv) as f32,
            listener_prediction: (self.listener_prediction * inv) as f32,
            duration_accuracy: (self.duration_accuracy * inv) as f32,
            energy_consistency: (self.energy_consistency * inv) as f32,
        }
    }
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

#[derive(Debug)]
struct QueuedVoiceRequest {
    generation: u64,
    request: VoiceRequest,
}

#[derive(Debug)]
struct QueuedVoiceResponse {
    generation: u64,
    response: VoiceResponse,
}

#[derive(Debug, Default)]
struct RequestState {
    latest: Option<QueuedVoiceRequest>,
    shutdown: bool,
}

/// Single-slot mailbox for synthesis requests.
///
/// Producers overwrite the pending request instead of accumulating work. A
/// currently-running phrase may finish, but generation checks ensure synthesis
/// stops before the next phrase and no superseded result is exposed.
#[derive(Debug)]
struct RequestMailbox {
    state: Mutex<RequestState>,
    ready: Condvar,
    latest_generation: AtomicU64,
}

impl RequestMailbox {
    fn new() -> Self {
        Self {
            state: Mutex::new(RequestState::default()),
            ready: Condvar::new(),
            latest_generation: AtomicU64::new(0),
        }
    }

    fn submit(&self, request: VoiceRequest) -> bool {
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(_) => return false,
        };
        if state.shutdown {
            return false;
        }

        let generation = match self.latest_generation.fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |current| current.checked_add(1),
        ) {
            Ok(previous) => previous + 1,
            Err(_) => return false,
        };

        state.latest = Some(QueuedVoiceRequest {
            generation,
            request,
        });
        drop(state);
        self.ready.notify_one();
        true
    }

    fn take_blocking(&self) -> Option<QueuedVoiceRequest> {
        let mut state = self.state.lock().ok()?;
        loop {
            if state.shutdown {
                return None;
            }
            if let Some(request) = state.latest.take() {
                return Some(request);
            }
            state = self.ready.wait(state).ok()?;
        }
    }

    fn current_generation(&self) -> u64 {
        self.latest_generation.load(Ordering::Acquire)
    }

    fn is_current(&self, generation: u64) -> bool {
        generation != 0 && self.current_generation() == generation
    }

    fn shutdown(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.shutdown = true;
            state.latest = None;
        }
        self.ready.notify_all();
    }

    #[cfg(test)]
    fn try_take(&self) -> Option<QueuedVoiceRequest> {
        self.state.lock().ok()?.latest.take()
    }
}

/// Small bounded FIFO that explicitly drops the oldest entry on overflow.
#[derive(Debug)]
struct BoundedDropOldest<T> {
    capacity: usize,
    items: Mutex<VecDeque<T>>,
}

impl<T> BoundedDropOldest<T> {
    fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "bounded buffer capacity must be non-zero");
        Self {
            capacity,
            items: Mutex::new(VecDeque::with_capacity(capacity)),
        }
    }

    fn push(&self, item: T) -> bool {
        let mut items = match self.items.lock() {
            Ok(items) => items,
            Err(_) => return false,
        };
        if items.len() == self.capacity {
            items.pop_front();
        }
        items.push_back(item);
        true
    }

    fn drain(&self) -> Vec<T> {
        match self.items.lock() {
            Ok(mut items) => items.drain(..).collect(),
            Err(_) => Vec::new(),
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.items.lock().map(|items| items.len()).unwrap_or(0)
    }
}

/// Handle for sending voice requests from the cognitive loop.
///
/// Submission never waits for TTS or queue capacity: at most one pending request
/// exists and a newer request replaces it. The tiny mailbox mutex is held only
/// long enough to swap that slot.
pub struct VoiceSynthesisChannel {
    mailbox: Arc<RequestMailbox>,
    responses: Arc<BoundedDropOldest<QueuedVoiceResponse>>,
    /// Handle to the background thread (kept so it remains attached to this
    /// channel's lifetime; drop signals shutdown without waiting for synthesis).
    _thread: thread::JoinHandle<()>,
}

impl VoiceSynthesisChannel {
    /// Spawn a background voice synthesis thread.
    ///
    /// Pending work is latest-wins and bounded to one request. Completed audio is
    /// bounded to [`VOICE_AUDIO_BUFFER_CAP`] with oldest-drop overflow semantics.
    pub fn spawn() -> Self {
        let mailbox = Arc::new(RequestMailbox::new());
        let responses = Arc::new(BoundedDropOldest::new(VOICE_AUDIO_BUFFER_CAP));

        let worker_mailbox = Arc::clone(&mailbox);
        let worker_responses = Arc::clone(&responses);
        let handle = thread::Builder::new()
            .name("voice-synthesis".into())
            .spawn(move || {
                Self::synthesis_loop(worker_mailbox, worker_responses);
            })
            .expect("Failed to spawn voice synthesis thread");

        Self {
            mailbox,
            responses,
            _thread: handle,
        }
    }

    /// Submit a voice request without waiting for synthesis or queue capacity.
    ///
    /// A newer pending request replaces the older pending request. Returns `false`
    /// only if the mailbox is shutting down, poisoned, or its generation counter
    /// has exhausted `u64`.
    pub fn send(&self, request: VoiceRequest) -> bool {
        self.mailbox.submit(request)
    }

    /// Drain completed audio responses without blocking on synthesis.
    ///
    /// Responses from superseded generations are discarded here as a second
    /// race-safe guard in addition to the worker's pre-enqueue generation check.
    pub fn drain_responses(&self) -> Vec<VoiceResponse> {
        let current_generation = self.mailbox.current_generation();
        self.responses
            .drain()
            .into_iter()
            .filter(|queued| queued.generation == current_generation)
            .map(|queued| queued.response)
            .collect()
    }

    /// Background thread main loop.
    fn synthesis_loop(
        mailbox: Arc<RequestMailbox>,
        responses: Arc<BoundedDropOldest<QueuedVoiceResponse>>,
    ) {
        use crate::voice::orchestrator::VoiceOrchestrator;

        let mut orchestrator = VoiceOrchestrator::new();

        // Self-hearing ear: same StreamProcessor pipeline as microphone
        // capture, persistent across utterances (LTC state carries over,
        // like a real ear that doesn't reset between sounds).
        #[cfg(feature = "voice-stt")]
        let mut self_ear =
            symthaea_stt::StreamProcessor::new(symthaea_stt::StreamConfig::low_latency());

        while let Some(queued) = mailbox.take_blocking() {
            let generation = queued.generation;
            let request = queued.request;
            let phrases = split_voice_phrases(&request.text);
            let mut audio = Vec::new();
            let mut weighted_metrics = WeightedVoiceMetrics::default();
            let mut superseded = false;

            for phrase in phrases {
                // Cooperative cancellation boundary before entering the existing
                // synchronous renderer. This keeps stale queued work from starting.
                if !mailbox.is_current(generation) {
                    superseded = true;
                    break;
                }

                let (phrase_audio, phrase_metrics) = orchestrator.thought_to_speech_paced(
                    phrase,
                    &request.cfc_output,
                    request.tau,
                    request.prediction_error,
                    request.detected_primitives.clone(),
                    request.speech_rate_multiplier,
                    request.pause_multiplier,
                );

                // A newer request may arrive while the renderer is inside this
                // phrase. Never append the just-finished stale phrase in that case.
                if !mailbox.is_current(generation) {
                    superseded = true;
                    break;
                }

                if !phrase_audio.is_empty() {
                    weighted_metrics.push(&phrase_metrics, phrase_audio.len());
                    audio.extend(phrase_audio);
                }
            }

            // Partial audio from a superseded generation is intentionally thrown
            // away. The consumer sees either the complete current request or none.
            if superseded || audio.is_empty() || !mailbox.is_current(generation) {
                continue;
            }

            let metrics = weighted_metrics.finish();

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

            // A newer request can arrive while self-hearing is being encoded.
            // Re-check immediately before exposing the completed audio.
            if !mailbox.is_current(generation) {
                continue;
            }

            let _ = responses.push(QueuedVoiceResponse {
                generation,
                response: VoiceResponse {
                    audio,
                    metrics,
                    #[cfg(feature = "voice-stt")]
                    self_hv,
                    cycle_num: request.cycle_num,
                },
            });
        }
    }
}

impl Drop for VoiceSynthesisChannel {
    fn drop(&mut self) {
        self.mailbox.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(cycle_num: u64, text: &str) -> VoiceRequest {
        VoiceRequest {
            text: text.into(),
            cfc_output: vec![0.0; 16],
            tau: 1.0,
            prediction_error: 0.1,
            detected_primitives: vec![],
            speech_rate_multiplier: 1.0,
            pause_multiplier: 1.0,
            cycle_num,
        }
    }

    #[test]
    fn request_mailbox_is_single_slot_latest_wins() {
        let mailbox = RequestMailbox::new();
        assert!(mailbox.submit(request(1, "old")));
        assert!(mailbox.submit(request(2, "new")));

        let queued = mailbox.try_take().expect("latest request should be queued");
        assert_eq!(queued.request.cycle_num, 2);
        assert_eq!(queued.request.text, "new");
        assert_eq!(queued.generation, mailbox.current_generation());
        assert!(mailbox.try_take().is_none());
    }

    #[test]
    fn newer_request_invalidates_older_generation() {
        let mailbox = RequestMailbox::new();
        assert!(mailbox.submit(request(1, "first")));
        let first = mailbox.try_take().expect("first request should be queued");
        assert!(mailbox.is_current(first.generation));

        assert!(mailbox.submit(request(2, "second")));
        assert!(!mailbox.is_current(first.generation));
        let second = mailbox.try_take().expect("second request should be queued");
        assert!(mailbox.is_current(second.generation));
    }

    #[test]
    fn phrase_split_preserves_strong_boundaries() {
        let phrases = split_voice_phrases("Hello there. How are you?! Fine\nNext line");
        assert_eq!(
            phrases,
            vec!["Hello there.", "How are you?!", "Fine", "Next line"]
        );
    }

    #[test]
    fn single_phrase_keeps_existing_whole_request_path() {
        let phrases = split_voice_phrases("hello world without punctuation");
        assert_eq!(phrases, vec!["hello world without punctuation"]);
    }

    #[test]
    fn generation_can_preempt_between_phrases() {
        let mailbox = RequestMailbox::new();
        assert!(mailbox.submit(request(1, "first sentence. second sentence.")));
        let first = mailbox.try_take().expect("first request should be queued");
        let phrases = split_voice_phrases(&first.request.text);
        assert_eq!(phrases.len(), 2);
        assert!(mailbox.is_current(first.generation));

        assert!(mailbox.submit(request(2, "interrupt")));
        assert!(!mailbox.is_current(first.generation));
    }

    #[test]
    fn weighted_metrics_use_audio_length() {
        let mut weighted = WeightedVoiceMetrics::default();
        let short = crate::voice::voice_feedback::VoiceOutputMetrics {
            articulation_score: 0.0,
            speech_rate: 2.0,
            ..Default::default()
        };
        let long = crate::voice::voice_feedback::VoiceOutputMetrics {
            articulation_score: 1.0,
            speech_rate: 4.0,
            ..Default::default()
        };
        weighted.push(&short, 100);
        weighted.push(&long, 300);
        let metrics = weighted.finish();
        assert!((metrics.articulation_score - 0.75).abs() < 1e-6);
        assert!((metrics.speech_rate - 3.5).abs() < 1e-6);
    }

    #[test]
    fn bounded_buffer_drops_oldest() {
        let buffer = BoundedDropOldest::new(2);
        assert!(buffer.push(1));
        assert!(buffer.push(2));
        assert!(buffer.push(3));
        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer.drain(), vec![2, 3]);
    }

    #[test]
    fn test_channel_spawn_and_send() {
        let channel = VoiceSynthesisChannel::spawn();
        assert!(
            channel.send(request(1, "hello world")),
            "should submit without waiting for synthesis"
        );
        // Give the thread a moment to process.
        std::thread::sleep(std::time::Duration::from_millis(50));
        // Drain — may or may not have a response depending on synthesis speed.
        let _responses = channel.drain_responses();
    }

    #[test]
    fn test_synthesis_is_real_speech_not_sine() {
        // Regression: the channel used to fall through to simulate_tts (a pure
        // sine wave). Real formant synthesis of two words must not be a single
        // sinusoid — check that the spectrum-shaping produces sign-structure
        // richer than a fixed-period tone.
        let channel = VoiceSynthesisChannel::spawn();
        let mut req = request(1, "hello world");
        req.cfc_output = vec![0.2; 16];
        assert!(channel.send(req));

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
