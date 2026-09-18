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
//! - **`speak()`** — synchronous synthesis + playback buffering
//! - **`speak_async()`** — legacy API: synthesis still occurs on the caller, then
//!   ring-buffer pushing runs on a background thread and returns a [`SpeakHandle`]
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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use symthaea_core::genesis::GenesisSeed;

use super::audio_out::AudioOutput;
use super::formant_targets::FormantDatabase;
use super::repl_voice::SimpleG2P;
use super::vocal_tract_controller::train_controller_on_phoneme_db;
use super::vocal_tract_encoder::VoiceCognitiveState;
use super::vocal_tract_fep::StreamingVocalTract;

/// Motor frame rate (Hz). Each frame produces `sample_rate / FRAME_RATE` audio samples.
const FRAME_RATE: u32 = 200;

/// Motor frame timestep (seconds).
const DT: f32 = 1.0 / FRAME_RATE as f32;

/// Base phoneme duration (seconds) for G2P timing.
const BASE_PHONEME_DURATION: f32 = 0.06;

/// Maximum producer-side batch attempted while holding playback-generation control.
///
/// Bounding the batch keeps a newer generation from waiting behind a large ring
/// fill while still ensuring that generation check + producer write are ordered
/// against generation replacement.
const PLAYBACK_PUSH_BATCH_SAMPLES: usize = 512;

/// Handle to a background playback push started by `speak_async()`.
///
/// Dropping the handle does NOT stop playback — call [`SpeakHandle::stop()`] explicitly,
/// or use [`SpeakHandle::join()`] to wait for producer completion.
pub struct SpeakHandle {
    thread: Option<std::thread::JoinHandle<Result<()>>>,
    speaking: Arc<AtomicBool>,
    /// Generation represented by this handle.
    generation: u64,
    /// Current generation allowed to write/stop playback.
    current_playback: Arc<AtomicU64>,
    /// Serializes generation transitions with producer-side write batches.
    playback_control: Arc<Mutex<()>>,
    /// Shared lock-free signal consumed by the audio callback.
    flush_requested: Arc<AtomicBool>,
}

impl SpeakHandle {
    /// Stop this utterance only if it is still the current playback generation.
    ///
    /// A stale handle is intentionally a no-op: it cannot stop or flush audio that
    /// belongs to a newer utterance. Generation transition and stop are serialized
    /// with producer-side batches; the device callback remains lock-free.
    pub fn stop(&self) {
        let _guard = self
            .playback_control
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.current_playback.load(Ordering::Acquire) != self.generation {
            return;
        }

        self.current_playback.store(0, Ordering::Release);
        self.speaking.store(false, Ordering::SeqCst);
        self.flush_requested.store(true, Ordering::Release);
    }

    /// Whether this handle still owns the actively-pushing generation.
    pub fn is_speaking(&self) -> bool {
        self.current_playback.load(Ordering::Acquire) == self.generation
            && self.speaking.load(Ordering::SeqCst)
    }

    /// Block until this generation's background producer thread exits.
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

/// Real-time streaming voice: text → phonemes → synthesis → speaker output.
///
/// # Example
///
/// ```no_run
/// # use symthaea_core::genesis::GenesisSeed;
/// # use symthaea::voice::live_voice::LiveVoice;
/// let genesis = GenesisSeed::from_phrase("my-voice");
/// let mut voice = LiveVoice::new(&genesis).unwrap();
/// voice.speak("hello world").unwrap();
/// ```
pub struct LiveVoice {
    streaming: StreamingVocalTract,
    audio: AudioOutput,
    g2p: SimpleG2P,
    formant_db: FormantDatabase,
    /// Shared cognitive state — can be updated from another thread mid-utterance.
    cognitive_state: Arc<parking_lot::Mutex<VoiceCognitiveState>>,
    /// Legacy/global stop signal. Generation identity is authoritative for async
    /// ownership; this flag remains for compatibility with `stop_flag()`.
    speaking: Arc<AtomicBool>,
    /// Monotonic allocator for playback generations. Zero is reserved for none.
    playback_counter: AtomicU64,
    /// Current playback generation, or zero when explicitly stopped/uninitialized.
    current_playback: Arc<AtomicU64>,
    /// Non-real-time control serialization. The CPAL callback never takes this lock.
    playback_control: Arc<Mutex<()>>,
    genesis: GenesisSeed,
}

impl LiveVoice {
    /// Create all components, train the controller, and open audio output.
    ///
    /// Training runs 30 epochs on the full FormantDatabase (~43 phonemes).
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
            playback_counter: AtomicU64::new(0),
            current_playback: Arc::new(AtomicU64::new(0)),
            playback_control: Arc::new(Mutex::new(())),
            genesis: genesis.clone(),
        })
    }

    /// Create a LiveVoice without an audio device (for `speak_to_file()` only).
    ///
    /// Uses a default sample rate of 24000 Hz.
    pub fn new_headless(genesis: &GenesisSeed) -> Self {
        Self::new_headless_with_rate(genesis, 24000)
    }

    /// Create a headless LiveVoice with a specific sample rate.
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
            playback_counter: AtomicU64::new(0),
            current_playback: Arc::new(AtomicU64::new(0)),
            playback_control: Arc::new(Mutex::new(())),
            genesis: genesis.clone(),
        }
    }

    /// Allocate and install a new playback generation.
    ///
    /// If the previous generation is still actively pushing, its queued playback
    /// is invalidated before this transition completes. If the previous producer
    /// already finished, its queued tail is allowed to drain naturally so normal
    /// sequential speech does not truncate itself.
    fn begin_playback_generation(&self) -> u64 {
        let _guard = self
            .playback_control
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let previous_was_pushing = self.current_playback.load(Ordering::Acquire) != 0
            && self.speaking.load(Ordering::SeqCst);

        let previous = self
            .playback_counter
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(1)
            })
            .expect("playback generation counter exhausted");
        let generation = previous + 1;

        self.current_playback.store(generation, Ordering::Release);
        self.speaking.store(true, Ordering::SeqCst);
        if previous_was_pushing {
            self.audio.request_flush();
        }
        generation
    }

    fn generation_is_active(&self, generation: u64) -> bool {
        generation != 0
            && self.current_playback.load(Ordering::Acquire) == generation
            && self.speaking.load(Ordering::SeqCst)
    }

    /// Mark producer work complete if this generation still owns playback.
    ///
    /// We retain `current_playback = generation` after producer completion so its
    /// handle may still flush samples already queued in the device ring. A future
    /// generation replaces this token atomically under the same control lock.
    fn finish_playback_generation(&self, generation: u64) {
        let _guard = self
            .playback_control
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.current_playback.load(Ordering::Acquire) == generation {
            self.speaking.store(false, Ordering::SeqCst);
        }
    }

    /// Speak text in real time with enhanced prosody control.
    pub fn speak(&mut self, text: &str) -> Result<()> {
        let generation = self.begin_playback_generation();

        let phonemes = self.g2p.text_to_phonemes(text, BASE_PHONEME_DURATION);
        let prosody = self.analyze_prosody(text);

        'phonemes: for timed in &phonemes {
            if !self.generation_is_active(generation) {
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
                if !self.generation_is_active(generation) {
                    break 'phonemes;
                }

                let chunk = self.streaming.tick(&state, None, DT, phoneme_str);
                if !self.push_with_backpressure(&chunk, generation) {
                    break 'phonemes;
                }
            }
        }

        self.finish_playback_generation(generation);
        Ok(())
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

    /// Pre-render text, then push the rendered audio on a background thread.
    ///
    /// This legacy method is asynchronous only for playback pushing: G2P and vocal
    /// tract synthesis still run on the caller before the returned [`SpeakHandle`]
    /// exists. The cognitive-loop voice worker uses a separate persistent worker.
    ///
    /// Playback generations make overlapping calls latest-wins on the producer
    /// side: starting a newer call invalidates the older writer and flushes its
    /// queued samples if that older generation is still actively pushing. A stale
    /// `SpeakHandle` cannot stop or flush the newer generation.
    pub fn speak_async(&mut self, text: &str) -> SpeakHandle {
        let generation = self.begin_playback_generation();

        let phonemes = self.g2p.text_to_phonemes(text, BASE_PHONEME_DURATION);
        let cog_state = Arc::clone(&self.cognitive_state);

        let mut all_samples = Vec::new();
        'phonemes: for timed in &phonemes {
            if !self.generation_is_active(generation) {
                break;
            }

            let n_frames = ((timed.duration / DT) as usize).max(1);
            let phoneme_str = if timed.phoneme == "SIL" {
                None
            } else {
                Some(timed.phoneme.as_str())
            };

            for _ in 0..n_frames {
                if !self.generation_is_active(generation) {
                    break 'phonemes;
                }

                let state = cog_state.lock().clone();
                let chunk = self.streaming.tick(&state, None, DT, phoneme_str);
                all_samples.extend_from_slice(&chunk);
            }
        }

        let speaking_bg = Arc::clone(&self.speaking);
        let current_playback = Arc::clone(&self.current_playback);
        let playback_control = Arc::clone(&self.playback_control);
        let flush_requested = self.audio.flush_handle();
        let flush_bg = Arc::clone(&flush_requested);
        let audio = self.audio.producer_handle();

        let thread = std::thread::Builder::new()
            .name("live-voice-push".into())
            .spawn(move || {
                let mut offset = 0usize;

                while offset < all_samples.len() {
                    let mut stop_loop = false;
                    let written = {
                        let _guard = playback_control
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());

                        if current_playback.load(Ordering::Acquire) != generation {
                            // Superseded: the newer generation owns flushing and
                            // playback now. A stale writer must not flush it.
                            stop_loop = true;
                            0
                        } else if !speaking_bg.load(Ordering::SeqCst) {
                            // Current generation was stopped through the legacy
                            // global stop flag. Invalidate and flush while control
                            // is still serialized against a newer generation.
                            current_playback.store(0, Ordering::Release);
                            flush_bg.store(true, Ordering::Release);
                            stop_loop = true;
                            0
                        } else if !audio.is_attached() {
                            speaking_bg.store(false, Ordering::SeqCst);
                            stop_loop = true;
                            0
                        } else {
                            let end = (offset + PLAYBACK_PUSH_BATCH_SAMPLES)
                                .min(all_samples.len());
                            audio.push_samples(&all_samples[offset..end])
                        }
                    };

                    if stop_loop {
                        break;
                    }

                    offset += written;
                    if offset < all_samples.len() {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                }

                // Completion may race with a newer generation. Only the current
                // generation may mutate the shared speaking flag.
                let _guard = playback_control
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if current_playback.load(Ordering::Acquire) == generation {
                    speaking_bg.store(false, Ordering::SeqCst);
                }
                Ok(())
            })
            .expect("failed to spawn speak thread");

        SpeakHandle {
            thread: Some(thread),
            speaking: Arc::clone(&self.speaking),
            generation,
            current_playback: Arc::clone(&self.current_playback),
            playback_control: Arc::clone(&self.playback_control),
            flush_requested,
        }
    }

    /// Synthesize text to a WAV file. No audio device needed.
    ///
    /// Uses the same G2P → frame-by-frame synthesis pipeline as `speak()`,
    /// but collects all samples and writes them to disk via `hound`.
    pub fn speak_to_file(&mut self, text: &str, path: &Path) -> Result<usize> {
        let phonemes = self.g2p.text_to_phonemes(text, BASE_PHONEME_DURATION);
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

    /// Push samples to the ring buffer while preserving playback-generation order.
    ///
    /// Returns false if this generation was stopped or superseded.
    fn push_with_backpressure(&mut self, samples: &[f32], generation: u64) -> bool {
        let mut offset = 0usize;
        while offset < samples.len() {
            let written = {
                let _guard = self
                    .playback_control
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());

                if self.current_playback.load(Ordering::Acquire) != generation {
                    return false;
                }
                if !self.speaking.load(Ordering::SeqCst) {
                    self.current_playback.store(0, Ordering::Release);
                    self.audio.request_flush();
                    return false;
                }

                let end = (offset + PLAYBACK_PUSH_BATCH_SAMPLES).min(samples.len());
                self.audio.push_samples(&samples[offset..end])
            };

            offset += written;
            if offset < samples.len() {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        true
    }

    /// Stop whichever playback generation is current and invalidate queued audio.
    pub fn stop(&self) {
        let _guard = self
            .playback_control
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.current_playback.store(0, Ordering::Release);
        self.speaking.store(false, Ordering::SeqCst);
        self.audio.request_flush();
    }

    /// Whether the current generation is still synthesizing/pushing.
    pub fn is_speaking(&self) -> bool {
        self.current_playback.load(Ordering::Acquire) != 0
            && self.speaking.load(Ordering::SeqCst)
    }

    /// Get the legacy global stop flag for cross-thread interruption.
    ///
    /// This flag is not generation-scoped: setting it false requests that whichever
    /// generation is current stop. Prefer [`Self::stop`] or [`SpeakHandle::stop`]
    /// when generation-safe ownership matters.
    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.speaking)
    }

    /// Clone the playback flush flag for integration with a higher-level turn or
    /// utterance cancellation token.
    pub fn playback_flush_handle(&self) -> Arc<AtomicBool> {
        self.audio.flush_handle()
    }

    /// Get a handle to the shared cognitive state for real-time prosody modulation.
    pub fn cognitive_state_handle(&self) -> Arc<parking_lot::Mutex<VoiceCognitiveState>> {
        Arc::clone(&self.cognitive_state)
    }

    /// Set the cognitive state (convenience wrapper — locks internally).
    pub fn set_cognitive_state(&self, state: VoiceCognitiveState) {
        *self.cognitive_state.lock() = state;
    }

    /// Modulate the LTC controller's time constant for speech rate control.
    pub fn modulate_tau(&mut self, factor: f32) {
        self.streaming.pipeline.controller.modulate_tau(factor);
    }

    /// Run additional training epochs on the formant database.
    pub fn train(&mut self, epochs: usize) {
        train_controller_on_phoneme_db(
            &mut self.streaming.pipeline.controller,
            &self.genesis,
            &self.formant_db,
            epochs,
        );
    }

    /// Audio sample rate from the output device (or headless default).
    pub fn sample_rate(&self) -> u32 {
        self.audio.sample_rate()
    }

    /// Reset the vocal tract pipeline state.
    pub fn reset(&mut self) {
        self.streaming.reset();
    }
}

/// Write 16-bit PCM mono WAV via hound.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_handle(
        generation: u64,
        current_generation: u64,
        speaking_value: bool,
    ) -> (SpeakHandle, Arc<AtomicBool>, Arc<AtomicU64>, Arc<AtomicBool>) {
        let speaking = Arc::new(AtomicBool::new(speaking_value));
        let current_playback = Arc::new(AtomicU64::new(current_generation));
        let flush_requested = Arc::new(AtomicBool::new(false));
        let handle = SpeakHandle {
            thread: None,
            speaking: Arc::clone(&speaking),
            generation,
            current_playback: Arc::clone(&current_playback),
            playback_control: Arc::new(Mutex::new(())),
            flush_requested: Arc::clone(&flush_requested),
        };
        (handle, speaking, current_playback, flush_requested)
    }

    #[test]
    fn test_phoneme_sequence_generation() {
        let g2p = SimpleG2P::new();
        let phonemes = g2p.text_to_phonemes("hello world", BASE_PHONEME_DURATION);
        assert!(
            !phonemes.is_empty(),
            "Should produce phonemes for 'hello world'"
        );

        let non_silence: Vec<_> = phonemes.iter().filter(|p| p.phoneme != "SIL").collect();
        assert!(
            non_silence.len() >= 4,
            "Should have at least 4 non-silence phonemes, got {}",
            non_silence.len()
        );
    }

    #[test]
    fn current_handle_stop_invalidates_and_flushes() {
        let (handle, speaking, current, flush) = test_handle(7, 7, true);
        handle.stop();
        assert_eq!(current.load(Ordering::Acquire), 0);
        assert!(!speaking.load(Ordering::SeqCst));
        assert!(flush.load(Ordering::Acquire));
    }

    #[test]
    fn stale_handle_cannot_stop_or_flush_newer_generation() {
        let (handle, speaking, current, flush) = test_handle(7, 8, true);
        handle.stop();
        assert_eq!(current.load(Ordering::Acquire), 8);
        assert!(speaking.load(Ordering::SeqCst));
        assert!(!flush.load(Ordering::Acquire));
        assert!(!handle.is_speaking());
    }

    #[test]
    fn handle_is_speaking_is_generation_scoped() {
        let (current_handle, _, _, _) = test_handle(4, 4, true);
        assert!(current_handle.is_speaking());

        let (stale_handle, _, _, _) = test_handle(3, 4, true);
        assert!(!stale_handle.is_speaking());
    }

    #[test]
    fn beginning_new_generation_supersedes_active_generation() {
        let genesis = GenesisSeed::from_phrase("test-generation");
        let voice = LiveVoice::new_headless(&genesis);
        let first = voice.begin_playback_generation();
        let second = voice.begin_playback_generation();
        assert!(second > first);
        assert_eq!(voice.current_playback.load(Ordering::Acquire), second);
        assert!(voice.generation_is_active(second));
        assert!(!voice.generation_is_active(first));
    }

    #[test]
    fn live_voice_stop_invalidates_current_generation_and_flushes() {
        let genesis = GenesisSeed::from_phrase("test-stop-flush");
        let voice = LiveVoice::new_headless(&genesis);
        let flush = voice.playback_flush_handle();
        let generation = voice.begin_playback_generation();
        assert!(voice.generation_is_active(generation));

        voice.stop();
        assert_eq!(voice.current_playback.load(Ordering::Acquire), 0);
        assert!(!voice.is_speaking());
        assert!(flush.load(Ordering::Acquire));
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

        assert!(n_samples > 0, "Should produce audio samples");
        assert!(wav_path.exists(), "WAV file should be created");

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

        let calm_rms = rms(&calm_samples);
        let excited_rms = rms(&excited_samples);

        assert!(
            calm_rms > 1e-6,
            "Calm audio should have content: rms={calm_rms}"
        );
        assert!(
            excited_rms > 1e-6,
            "Excited audio should have content: rms={excited_rms}"
        );
    }

    #[test]
    #[ignore] // Requires audio device
    fn sequential_speak_async_reuses_live_producer() {
        let genesis = GenesisSeed::from_phrase("test-live-voice-async-repeat");
        let mut voice = LiveVoice::new(&genesis).expect("Should create LiveVoice");

        let first = voice.speak_async("hello");
        first.join().expect("first async push should complete");

        let second = voice.speak_async("again");
        second.join().expect("second async push should reuse producer");
    }

    #[test]
    #[ignore] // Requires audio device
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

#[derive(Debug, Clone)]
struct ProsodyAnalysis {
    pitch_range: f32,
    speaking_rate: f32,
    #[allow(dead_code)]
    emphasis: Vec<String>,
}
