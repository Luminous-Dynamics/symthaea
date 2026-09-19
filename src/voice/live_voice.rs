// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Real-time streaming voice: text → phonemes → synthesis → speaker output.
//!
//! Combines [`SimpleG2P`] + [`StreamingVocalTract`] + [`AudioOutput`] into a single
//! `speak()` call that streams audio incrementally (first audio within ~25ms).
//!
//! # Modes
//!
//! - **`speak()`** — synchronous, blocks caller until utterance is buffered
//! - **`speak_async()`** — spawns a background thread, returns a [`SpeakHandle`]
//! - **`speak_to_file()`** — writes WAV to disk (no audio device needed)
//!
//! # Prosody
//!
//! The cognitive state can be updated mid-utterance via the shared
//! [`Arc<parking_lot::Mutex<VoiceCognitiveState>>`] returned by [`LiveVoice::cognitive_state_handle()`].
//! Changes take effect on the next motor frame (~5ms latency).
//!
//! Feature-gated under `live-voice`.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use symthaea_core::genesis::GenesisSeed;

use super::audio_out::{AudioFlushHandle, AudioOutput, AudioProducerHandle};
use super::formant_targets::FormantDatabase;
use super::repl_voice::SimpleG2P;
use super::vocal_tract_controller::train_controller_on_phoneme_db;
use super::vocal_tract_encoder::VoiceCognitiveState;
use super::vocal_tract_fep::StreamingVocalTract;

const FRAME_RATE: u32 = 200;
const DT: f32 = 1.0 / FRAME_RATE as f32;
const BASE_PHONEME_DURATION: f32 = 0.06;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveVoiceSpeakOutcome {
    Completed,
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct LiveVoiceStopHandle {
    speaking: Arc<AtomicBool>,
    flush: AudioFlushHandle,
}

impl LiveVoiceStopHandle {
    fn new(speaking: Arc<AtomicBool>, flush: AudioFlushHandle) -> Self {
        Self { speaking, flush }
    }

    pub fn stop(&self) {
        self.speaking.store(false, Ordering::SeqCst);
        self.flush.request_flush();
    }

    pub fn is_speaking(&self) -> bool {
        self.speaking.load(Ordering::SeqCst)
    }
}

pub struct SpeakHandle {
    thread: Option<std::thread::JoinHandle<Result<()>>>,
    speaking: Arc<AtomicBool>,
    flush: AudioFlushHandle,
}

impl SpeakHandle {
    pub fn stop(&self) {
        self.speaking.store(false, Ordering::SeqCst);
        self.flush.request_flush();
    }

    pub fn is_speaking(&self) -> bool {
        self.speaking.load(Ordering::SeqCst)
    }

    pub fn join(mut self) -> Result<()> {
        if let Some(handle) = self.thread.take() {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("speak thread panicked"))?
        } else {
            Ok(())
        }
    }
}

pub struct LiveVoice {
    streaming: StreamingVocalTract,
    audio: AudioOutput,
    g2p: SimpleG2P,
    formant_db: FormantDatabase,
    cognitive_state: Arc<parking_lot::Mutex<VoiceCognitiveState>>,
    speaking: Arc<AtomicBool>,
    genesis: GenesisSeed,
}

impl LiveVoice {
    pub fn new(genesis: &GenesisSeed) -> Result<Self> {
        let audio = AudioOutput::new()?;
        let sample_rate = audio.sample_rate();
        let mut streaming = StreamingVocalTract::new(genesis, sample_rate, FRAME_RATE);
        let db = FormantDatabase::new();
        train_controller_on_phoneme_db(&mut streaming.pipeline.controller, genesis, &db, 30);

        Ok(Self {
            streaming,
            audio,
            g2p: SimpleG2P::new(),
            formant_db: db,
            cognitive_state: Arc::new(parking_lot::Mutex::new(VoiceCognitiveState::default())),
            speaking: Arc::new(AtomicBool::new(false)),
            genesis: genesis.clone(),
        })
    }

    pub fn new_headless(genesis: &GenesisSeed) -> Self {
        Self::new_headless_with_rate(genesis, 24000)
    }

    pub fn new_headless_with_rate(genesis: &GenesisSeed, sample_rate: u32) -> Self {
        let mut streaming = StreamingVocalTract::new(genesis, sample_rate, FRAME_RATE);
        let db = FormantDatabase::new();
        train_controller_on_phoneme_db(&mut streaming.pipeline.controller, genesis, &db, 30);

        Self {
            streaming,
            audio: AudioOutput::new_dummy(sample_rate),
            g2p: SimpleG2P::new(),
            formant_db: db,
            cognitive_state: Arc::new(parking_lot::Mutex::new(VoiceCognitiveState::default())),
            speaking: Arc::new(AtomicBool::new(false)),
            genesis: genesis.clone(),
        }
    }

    pub fn speak(&mut self, text: &str) -> Result<()> {
        let _ = self.speak_cancellable(text, || false)?;
        Ok(())
    }

    pub fn speak_cancellable<F>(
        &mut self,
        text: &str,
        cancelled: F,
    ) -> Result<LiveVoiceSpeakOutcome>
    where
        F: Fn() -> bool,
    {
        if cancelled() {
            self.speaking.store(false, Ordering::SeqCst);
            self.audio.request_flush();
            return Ok(LiveVoiceSpeakOutcome::Cancelled);
        }

        self.speaking.store(true, Ordering::SeqCst);
        if cancelled() {
            self.speaking.store(false, Ordering::SeqCst);
            self.audio.request_flush();
            return Ok(LiveVoiceSpeakOutcome::Cancelled);
        }

        let phonemes = self.g2p.text_to_phonemes(text, BASE_PHONEME_DURATION);
        let prosody = self.analyze_prosody(text);
        let mut was_cancelled = false;

        'phonemes: for timed in &phonemes {
            if cancelled() || !self.speaking.load(Ordering::SeqCst) {
                was_cancelled = true;
                break;
            }

            let n_frames = ((timed.duration / DT) as usize).max(1);
            let phoneme_str = if timed.phoneme == "SIL" {
                None
            } else {
                Some(timed.phoneme.as_str())
            };

            let mut state = self.cognitive_state.lock().clone();
            self.apply_prosody(&mut state, &prosody);

            for _ in 0..n_frames {
                if cancelled() || !self.speaking.load(Ordering::SeqCst) {
                    was_cancelled = true;
                    break 'phonemes;
                }

                let chunk = self.streaming.tick(&state, None, DT, phoneme_str);
                if !self.push_with_backpressure_cancellable(&chunk) {
                    was_cancelled = true;
                    break 'phonemes;
                }
            }
        }

        if cancelled() || !self.speaking.load(Ordering::SeqCst) {
            was_cancelled = true;
        }
        self.speaking.store(false, Ordering::SeqCst);

        if was_cancelled {
            self.audio.request_flush();
            Ok(LiveVoiceSpeakOutcome::Cancelled)
        } else {
            Ok(LiveVoiceSpeakOutcome::Completed)
        }
    }

    fn analyze_prosody(&self, _text: &str) -> ProsodyAnalysis {
        ProsodyAnalysis {
            pitch_range: 1.0,
            speaking_rate: 1.0,
            emphasis: Vec::new(),
        }
    }

    fn apply_prosody(&mut self, state: &mut VoiceCognitiveState, prosody: &ProsodyAnalysis) {
        state.emotional_arousal = prosody.pitch_range.clamp(0.0, 1.0);
        self.modulate_tau(1.0 / prosody.speaking_rate);
    }

    /// Background playback using a reusable producer-side capability.
    ///
    /// Unlike the historical implementation, this does not take producer ownership
    /// out of `AudioOutput`; later async utterances can obtain another handle to the
    /// same ring after the previous thread completes.
    pub fn speak_async(&mut self, text: &str) -> SpeakHandle {
        self.speaking.store(true, Ordering::SeqCst);

        let phonemes = self.g2p.text_to_phonemes(text, BASE_PHONEME_DURATION);
        let speaking = Arc::clone(&self.speaking);
        let cog_state = Arc::clone(&self.cognitive_state);

        let mut all_samples = Vec::new();
        for timed in &phonemes {
            if !speaking.load(Ordering::SeqCst) {
                break;
            }

            let n_frames = ((timed.duration / DT) as usize).max(1);
            let phoneme_str = if timed.phoneme == "SIL" {
                None
            } else {
                Some(timed.phoneme.as_str())
            };

            for _ in 0..n_frames {
                if !speaking.load(Ordering::SeqCst) {
                    break;
                }

                let state = cog_state.lock().clone();
                let chunk = self.streaming.tick(&state, None, DT, phoneme_str);
                all_samples.extend_from_slice(&chunk);
            }
        }

        let speaking_bg = Arc::clone(&self.speaking);
        let flush = self.audio.flush_handle();
        let flush_bg = flush.clone();
        let audio = self.audio.producer_handle();

        let thread = std::thread::Builder::new()
            .name("live-voice-push".into())
            .spawn(move || {
                let mut offset = 0;
                while offset < all_samples.len() {
                    if !speaking_bg.load(Ordering::SeqCst) {
                        break;
                    }
                    if let Some(ref producer) = audio {
                        let written = push_samples_to_producer_while_speaking(
                            producer,
                            &all_samples[offset..],
                            &speaking_bg,
                        );
                        offset += written;
                        if offset < all_samples.len() && speaking_bg.load(Ordering::SeqCst) {
                            std::thread::sleep(std::time::Duration::from_millis(1));
                        }
                    } else {
                        break;
                    }
                }

                let interrupted = offset < all_samples.len();
                speaking_bg.store(false, Ordering::SeqCst);
                if interrupted {
                    flush_bg.request_flush();
                }
                Ok(())
            })
            .expect("failed to spawn speak thread");

        SpeakHandle {
            thread: Some(thread),
            speaking: Arc::clone(&self.speaking),
            flush,
        }
    }

    pub fn speak_to_file(&mut self, text: &str, path: &Path) -> Result<usize> {
        let phonemes = self.g2p.text_to_phonemes(text, BASE_PHONEM_DURATION);
        let mut all_samples = Vec::new();

        for timed in &phonemes {
            let n_frames = ((timed.duration / DT) as usize).max(1);
            let phoneme_str = if timed.phoneme == "SIL" {
                None
            } else {
                Some(timed.phoneme.as_str())
            };

            let state = self.cognitive_state.lock().clone();
            for _ in 0..n_frames {
                let chunk = self.streaming.tick(&state, None, DT, phoneme_str);
                all_samples.extend_from_slice(&chunk);
            }
        }

        let sample_rate = self.streaming.vocoder.sample_rate();
        write_wav(path, &all_samples, sample_rate)?;
        Ok(all_samples.len())
    }

    fn push_with_backpressure_cancellable(&mut self, samples: &[f32]) -> bool {
        let mut offset = 0;
        while offset < samples.len() {
            if !self.speaking.load(Ordering::SeqCst) {
                return false;
            }
            let written = self.audio.push_samples(&samples[offset..]);
            offset += written;
            if offset < samples.len() {
                if !self.speaking.load(Ordering::SeqCst) {
                    return false;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        true
    }

    pub fn stop(&self) {
        self.stop_handle().stop();
    }

    pub fn is_speaking(&self) -> bool {
        self.speaking.load(Ordering::SeqCst)
    }

    pub fn stop_handle(&self) -> LiveVoiceStopHandle {
        LiveVoiceStopHandle::new(Arc::clone(&self.speaking), self.audio.flush_handle())
    }

    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.speaking)
    }

    pub fn cognitive_state_handle(&self) -> Arc<parking_lot::Mutex<VoiceCognitiveState>> {
        Arc::clone(&self.cognitive_state)
    }

    pub fn set_cognitive_state(&self, state: VoiceCognitiveState) {
        *self.cognitive_state.lock() = state;
    }

    pub fn modulate_tau(&mut self, factor: f32) {
        self.streaming.pipeline.controller.modulate_tau(factor);
    }

    pub fn train(&mut self, epochs: usize) {
        train_controller_on_phoneme_db(
            &mut self.streaming.pipeline.controller,
            &self.genesis,
            &self.formant_db,
            epochs,
        );
    }

    pub fn sample_rate(&self) -> u32 {
        self.audio.sample_rate()
    }

    pub fn reset(&mut self) {
        self.streaming.reset();
    }
}

fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        let amplitude = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer.write_sample(amplitude)?;
    }
    writer.finalize()?;
    Ok(())
}

fn push_samples_to_producer_while_speaking(
    producer: &AudioProducerHandle,
    samples: &[f32],
    speaking: &AtomicBool,
) -> usize {
    producer.push_samples_while(samples, || speaking.load(Ordering::SeqCst))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phoneme_sequence_generation() {
        let g2p = SimpleG2P::new();
        let phonemes = g2p.text_to_phonemes("hello world", BASE_PHONEME_DURATION);
        assert!(!phonemes.is_empty());
        let non_silence: Vec<_> = phonemes.iter().filter(|p| p.phoneme != "SIL").collect();
        assert!(non_silence.len() >= 4);
    }

    #[test]
    fn test_stop_flag_works() {
        let flag = Arc::new(AtomicBool::new(true));
        assert!(flag.load(Ordering::SeqCst));
        flag.store(false, Ordering::SeqCst);
        assert!(!flag.load(Ordering::SeqCst));
    }

    #[test]
    fn typed_stop_handle_stops_and_requests_flush() {
        let flag = Arc::new(AtomicBool::new(true));
        let audio = AudioOutput::new_dummy(24000);
        let flush = audio.flush_handle();
        let first = LiveVoiceStopHandle::new(Arc::clone(&flag), flush.clone());
        let second = first.clone();

        assert!(first.is_speaking());
        assert!(!flush.is_pending());
        second.stop();
        assert!(!first.is_speaking());
        assert!(flush.is_pending());
    }

    #[test]
    fn cancellable_speak_honors_pre_start_cancellation_and_requests_flush() {
        let genesis = GenesisSeed::from_phrase("test-pre-cancel");
        let mut voice = LiveVoice::new_headless(&genesis);
        let flush = voice.audio.flush_handle();
        let outcome = voice
            .speak_cancellable("this must never begin", || true)
            .unwrap();
        assert_eq!(outcome, LiveVoiceSpeakOutcome::Cancelled);
        assert!(!voice.is_speaking());
        assert!(flush.is_pending());
    }

    #[test]
    fn reusable_producer_helper_obeys_stop_signal() {
        let rb = ringbuf::HeapRb::<f32>::new(8);
        let (producer, mut consumer) = ringbuf::traits::Split::split(rb);
        let producer = AudioProducerHandle::new(producer);
        let speaking = AtomicBool::new(false);
        let written = push_samples_to_producer_while_speaking(
            &producer,
            &[0.1, 0.2, 0.3],
            &speaking,
        );
        assert_eq!(written, 0);
        use ringbuf::traits::Consumer;
        assert!(consumer.try_pop().is_none());
    }

    #[test]
    fn test_speak_to_file_headless() {
        let genesis = GenesisSeed::from_phrase("test-headless");
        let mut voice = LiveVoice::new_headless(&genesis);
        let dir = tempfile::tempdir().expect("tempdir");
        let wav_path = dir.path().join("test.wav");
        let n_samples = voice
            .speak_to_file("hello", &wav_path)
            .expect("speak_to_file should succeed");
        assert!(n_samples > 0);
        assert!(wav_path.exists());
        let reader = hound::WavReader::open(&wav_path).expect("Should read WAV");
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.spec().sample_rate, 24000);
        assert!(reader.len() > 0);
    }

    #[test]
    fn test_cognitive_state_handle() {
        let state = Arc::new(parking_lot::Mutex::new(VoiceCognitiveState::default()));
        let handle = Arc::clone(&state);
        {
            let mut s = handle.lock();
            s.emotional_arousal = 0.9;
        }
        let current = state.lock().clone();
        assert!((current.emotional_arousal - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_speak_to_file_cognitive_modulation() {
        let genesis = GenesisSeed::from_phrase("test-prosody");
        let mut voice = LiveVoice::new_headless(&genesis);
        let dir = tempfile::tempdir().expect("tempdir");

        voice.set_cognitive_state(VoiceCognitiveState {
            emotional_arousal: 0.1,
            ..Default::default()
        });
        let calm_path = dir.path().join("calm.wav");
        let calm_n = voice.speak_to_file("hello", &calm_path).unwrap();

        voice.reset();
        voice.set_cognitive_state(VoiceCognitiveState {
            emotional_arousal: 0.9,
            emotional_valence: 0.8,
            consciousness_level: 0.9,
            ..Default::default()
        });
        let excited_path = dir.path().join("excited.wav");
        let excited_n = voice.speak_to_file("hello", &excited_path).unwrap();

        assert!(calm_n > 0);
        assert!(excited_n > 0);

        let calm_reader = hound::WavReader::open(&calm_path).unwrap();
        let excited_reader = hound::WavReader::open(&excited_path).unwrap();
        let calm_samples: Vec<f32> = calm_reader
            .into_samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32767.0)
            .collect();
        let excited_samples: Vec<f32> = excited_reader
            .into_samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32767.0)
            .collect();
        assert!(rms(&calm_samples) > 1e-6);
        assert!(rms(&excited_samples) > 1e-6);
    }

    #[test]
    #[ignore]
    fn test_live_voice_speak_produces_audio() {
        let genesis = GenesisSeed::from_phrase("test-live-voice");
        let mut voice = LiveVoice::new(&genesis).expect("Should create LiveVoice");
        assert!(voice.sample_rate() > 0);
        voice.speak("hello").expect("Should speak without error");
        assert!(!voice.is_speaking());
    }

    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }
}
