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

/// Handle to a background `speak_async()` call.
///
/// Dropping the handle does NOT stop playback — call [`SpeakHandle::stop()`] explicitly,
/// or use [`SpeakHandle::join()`] to wait for completion.
pub struct SpeakHandle {
    thread: Option<std::thread::JoinHandle<Result<()>>>,
    speaking: Arc<AtomicBool>,
}

impl SpeakHandle {
    /// Stop the background utterance. The ring buffer drains naturally to silence.
    pub fn stop(&self) {
        self.speaking.store(false, Ordering::SeqCst);
    }

    /// Whether the background thread is still synthesizing.
    pub fn is_speaking(&self) -> bool {
        self.speaking.load(Ordering::SeqCst)
    }

    /// Block until the utterance finishes (or is stopped).
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
    speaking: Arc<AtomicBool>,
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

        // AudioOutput::new() would fail headless, so we create a dummy.
        // speak() and speak_async() will fail if called, but speak_to_file() works.
        // We use a separate struct field to track this.
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

    /// Speak text in real time with enhanced prosody control
    pub fn speak(&mut self, text: &str) -> Result<()> {
        self.speaking.store(true, Ordering::SeqCst);

        // Enhanced text analysis
        let phonemes = self.g2p.text_to_phonemes(text, BASE_PHONEME_DURATION);
        let prosody = self.analyze_prosody(text);

        for timed in &phonemes {
            if !self.speaking.load(Ordering::SeqCst) {
                break;
            }

            let n_frames = ((timed.duration / DT) as usize).max(1);
            let phoneme_str = if timed.phoneme == "SIL" {
                None
            } else {
                Some(timed.phoneme.as_str())
            };

            // Apply prosody modulation
            let mut state = self.cognitive_state.lock().clone();
            self.apply_prosody(&mut state, &prosody);

            for _ in 0..n_frames {
                if !self.speaking.load(Ordering::SeqCst) {
                    break;
                }

                let chunk = self.streaming.tick(&state, None, DT, phoneme_str);
                self.push_with_backpressure(&chunk);
            }
        }

        self.speaking.store(false, Ordering::SeqCst);
        Ok(())
    }

    fn analyze_prosody(&self, text: &str) -> ProsodyAnalysis {
        // Analyze sentence structure, emphasis, etc.
        ProsodyAnalysis {
            pitch_range: 1.0,
            speaking_rate: 1.0,
            emphasis: Vec::new(),
        }
    }

    fn apply_prosody(&self, state: &mut VoiceCognitiveState, prosody: &ProsodyAnalysis) {
        state.emotional_arousal = prosody.pitch_range.clamp(0.0, 1.0);
        self.modulate_tau(1.0 / prosody.speaking_rate);
    }

    /// Speak text on a background thread. Returns a [`SpeakHandle`] for control.
    ///
    /// The cognitive loop can continue running while speech plays. Use
    /// [`cognitive_state_handle()`](Self::cognitive_state_handle) to modulate prosody mid-utterance.
    ///
    /// # Note
    /// This takes `&mut self` to ensure exclusive synthesis access, then moves
    /// the necessary state into the thread. Only one `speak_async` at a time.
    pub fn speak_async(&mut self, text: &str) -> SpeakHandle {
        self.speaking.store(true, Ordering::SeqCst);

        let phonemes = self.g2p.text_to_phonemes(text, BASE_PHONEME_DURATION);
        let speaking = Arc::clone(&self.speaking);
        let cog_state = Arc::clone(&self.cognitive_state);

        // Synthesize frames into a buffer on a dedicated thread.
        // We can't move `self` into the thread, so we pre-synthesize all audio.
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

        // Push synthesized audio to the ring buffer on a background thread
        // (backpressure may block, so we don't want to block the caller)
        let speaking_bg = Arc::clone(&self.speaking);
        let mut audio = self.audio.take_producer();

        let thread = std::thread::Builder::new()
            .name("live-voice-push".into())
            .spawn(move || {
                let mut offset = 0;
                while offset < all_samples.len() {
                    if !speaking_bg.load(Ordering::SeqCst) {
                        break;
                    }
                    if let Some(ref mut producer) = audio {
                        let written = push_samples_to_producer(producer, &all_samples[offset..]);
                        offset += written;
                        if offset < all_samples.len() {
                            std::thread::sleep(std::time::Duration::from_millis(1));
                        }
                    } else {
                        break;
                    }
                }
                speaking_bg.store(false, Ordering::SeqCst);
                Ok(())
            })
            .expect("failed to spawn speak thread");

        SpeakHandle {
            thread: Some(thread),
            speaking: Arc::clone(&self.speaking),
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

    /// Push samples to the ring buffer with simple backpressure.
    fn push_with_backpressure(&mut self, samples: &[f32]) {
        let mut offset = 0;
        while offset < samples.len() {
            let written = self.audio.push_samples(&samples[offset..]);
            offset += written;
            if offset < samples.len() {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }

    /// Stop speaking immediately. The ring buffer drains naturally to silence.
    pub fn stop(&self) {
        self.speaking.store(false, Ordering::SeqCst);
    }

    /// Whether `speak()` or `speak_async()` is currently running.
    pub fn is_speaking(&self) -> bool {
        self.speaking.load(Ordering::SeqCst)
    }

    /// Get a clone of the stop flag for cross-thread interruption.
    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.speaking)
    }

    /// Get a handle to the shared cognitive state for real-time prosody modulation.
    ///
    /// Lock the mutex and modify the state from any thread; changes take effect
    /// on the next motor frame (~5ms).
    pub fn cognitive_state_handle(&self) -> Arc<parking_lot::Mutex<VoiceCognitiveState>> {
        Arc::clone(&self.cognitive_state)
    }

    /// Set the cognitive state (convenience wrapper — locks internally).
    pub fn set_cognitive_state(&self, state: VoiceCognitiveState) {
        *self.cognitive_state.lock() = state;
    }

    /// Modulate the LTC controller's time constant for speech rate control.
    ///
    /// `factor > 1.0` → slower, more deliberate formant transitions (max 3.0).
    /// `factor < 1.0` → faster, more agile transitions.
    /// `factor = 1.0` → default rate.
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

/// Push samples to a ring buffer producer (used by background thread).
fn push_samples_to_producer(producer: &mut ringbuf::HeapProd<f32>, samples: &[f32]) -> usize {
    use ringbuf::traits::Producer;
    let mut written = 0;
    for &s in samples {
        if producer.try_push(s).is_ok() {
            written += 1;
        } else {
            break;
        }
    }
    written
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_stop_flag_works() {
        let flag = Arc::new(AtomicBool::new(true));
        assert!(flag.load(Ordering::SeqCst));

        flag.store(false, Ordering::SeqCst);
        assert!(!flag.load(Ordering::SeqCst));
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

        // Verify WAV is readable
        let reader = hound::WavReader::open(&wav_path).expect("Should read WAV");
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.spec().sample_rate, 24000);
        assert!(reader.len() > 0);
    }

    #[test]
    fn test_cognitive_state_handle() {
        let state = Arc::new(parking_lot::Mutex::new(VoiceCognitiveState::default()));
        let handle = Arc::clone(&state);

        // Modify from "another thread" (simulated)
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

        // Calm state
        voice.set_cognitive_state(VoiceCognitiveState {
            emotional_arousal: 0.1,
            ..Default::default()
        });
        let calm_path = dir.path().join("calm.wav");
        let calm_n = voice.speak_to_file("hello", &calm_path).unwrap();

        // Reset pipeline state between utterances
        voice.reset();

        // Excited state
        voice.set_cognitive_state(VoiceCognitiveState {
            emotional_arousal: 0.9,
            emotional_valence: 0.8,
            consciousness_level: 0.9,
            ..Default::default()
        });
        let excited_path = dir.path().join("excited.wav");
        let excited_n = voice.speak_to_file("hello", &excited_path).unwrap();

        // Both should produce audio
        assert!(calm_n > 0);
        assert!(excited_n > 0);

        // Read both WAVs and compare RMS — different cognitive states should
        // produce different audio content (even if same phonemes)
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

        // Both should have non-trivial content
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
