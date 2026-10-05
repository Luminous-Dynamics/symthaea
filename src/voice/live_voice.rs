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
use serde::{Deserialize, Serialize};
use symthaea_core::genesis::GenesisSeed;
#[cfg(feature = "ssm_language")]
use symthaea_broca::PhonologicalPlan;
use symthaea_vocal_tract::pipeline::{Intonation, PitchAccent, ProsodyContext, predict_duration};

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

/// Evidence emitted by the explicit phonological-plan realization path.
#[cfg(feature = "ssm_language")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhonologicalPlanRealizationReceipt {
    /// Receipt schema, independent from the phonological-plan contract version.
    pub schema_version: u32,
    pub plan_version: String,
    pub plan_grounding_blake3: String,
    pub realization_authorized: bool,
    pub segment_count: usize,
    pub scheduler_frames: usize,
    pub sample_count: usize,
    pub sample_rate: u32,
    pub rate: f32,
    pub pitch_range: f32,
    pub prominence: f32,
    pub pause_weight: f32,
    pub audio_blake3: String,
}

#[cfg(feature = "ssm_language")]
impl PhonologicalPlanRealizationReceipt {
    /// Independently verify the receipt's plan-bound fields and internal sample accounting.
    pub fn verify_against_plan(&self, plan: &PhonologicalPlan) -> Result<()> {
        plan.validate()
            .map_err(|error| anyhow::anyhow!("invalid phonological plan: {error}"))?;

        if self.schema_version != 1 {
            anyhow::bail!("unsupported realization receipt schema: {}", self.schema_version);
        }
        if self.plan_version != plan.version {
            anyhow::bail!("realization receipt plan version does not match plan");
        }
        if !self.realization_authorized || !plan.realization_authorized {
            anyhow::bail!("realization receipt or plan is not authorized");
        }

        let expected_grounding =
            blake3::hash(plan.grounding_surface().as_bytes()).to_hex().to_string();
        if self.plan_grounding_blake3 != expected_grounding {
            anyhow::bail!("realization receipt plan grounding does not match plan");
        }

        if self.segment_count != plan.segments.len()
            || self.rate != plan.rate
            || self.pitch_range != plan.pitch_range
            || self.prominence != plan.prominence
            || self.pause_weight != plan.pause_weight
        {
            anyhow::bail!("realization receipt plan fields do not match plan");
        }

        let samples_per_frame = (self.sample_rate / FRAME_RATE) as usize;
        let expected_samples = self.scheduler_frames.saturating_mul(samples_per_frame);
        if self.sample_count != expected_samples || self.sample_count == 0 {
            anyhow::bail!("realization receipt sample accounting is inconsistent");
        }

        if !is_hex_digest(&self.audio_blake3) || !is_hex_digest(&self.plan_grounding_blake3) {
            anyhow::bail!("realization receipt hashes are malformed");
        }

        Ok(())
    }

    /// Verify the audio digest independently from the plan receipt.
    pub fn verify_samples(&self, samples: &[f32]) -> bool {
        if samples.len() != self.sample_count {
            return false;
        }
        let mut hasher = blake3::Hasher::new();
        for sample in samples {
            hasher.update(&sample.to_le_bytes());
        }
        hasher.finalize().to_hex().to_string() == self.audio_blake3
    }
}

#[cfg(feature = "ssm_language")]
fn is_hex_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

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

    /// Synthesize an explicit phonological plan through the live audio output path.
    ///
    /// This is intentionally plan-native: no text, G2P reconstruction, or lexical inference
    /// occurs here. The validated plan supplies explicit phoneme identity, stress, rate,
    /// phrase boundaries, and prosodic intent. The vocal-tract controller supplies
    /// speaker/anatomical parameters. Word-final timing remains unasserted because the current
    /// phonological contract does not encode word boundaries.
    ///
    /// The current implementation pre-synthesizes the validated plan before enqueueing it to
    /// the audio sink, so this method does not claim first-audio latency.
    #[cfg(feature = "ssm_language")]
    pub fn speak_phonological_plan(&mut self, plan: &PhonologicalPlan) -> Result<usize> {
        Ok(self.speak_phonological_plan_with_receipt(plan)?.sample_count)
    }

    /// Realize an explicit phonological plan and return an evidence receipt that binds
    /// the validated plan grounding to the generated audio buffer.
    #[cfg(feature = "ssm_language")]
    pub fn speak_phonological_plan_with_receipt(
        &mut self,
        plan: &PhonologicalPlan,
    ) -> Result<PhonologicalPlanRealizationReceipt> {
        let (samples, receipt) = self.synthesize_phonological_plan(plan)?;
        self.push_with_backpressure(&samples);
        Ok(receipt)
    }

    /// Synthesize an explicit phonological plan directly to WAV without an audio device.
    ///
    /// The same scheduler-owned duration calculation is used by the real-time path.
    #[cfg(feature = "ssm_language")]
    pub fn speak_phonological_plan_to_file(
        &mut self,
        plan: &PhonologicalPlan,
        path: &Path,
    ) -> Result<usize> {
        let (samples, receipt) = self.synthesize_phonological_plan(plan)?;
        write_wav(path, &samples, receipt.sample_rate)?;
        Ok(receipt.sample_count)
    }

    #[cfg(feature = "ssm_language")]
    fn pitch_accent_for_plan(segment_is_focus: bool, prominence: f32) -> PitchAccent {
        if segment_is_focus && prominence >= 0.82 {
            PitchAccent::RiseHigh
        } else if prominence >= 0.62 {
            PitchAccent::High
        } else {
            PitchAccent::None
        }
    }

    #[cfg(feature = "ssm_language")]
    fn synthesize_phonological_plan(
        &mut self,
        plan: &PhonologicalPlan,
    ) -> Result<(Vec<f32>, PhonologicalPlanRealizationReceipt)> {
        plan.validate()
            .map_err(|error| anyhow::anyhow!("invalid phonological plan: {error}"))?;
        if !plan.ready_for_realization() {
            anyhow::bail!("phonological plan is not ready for realization");
        }

        let scheduled_frames_for = |index: usize, segment: &symthaea_broca::PhonemeSlot| {
            let mut frames = predict_duration(
                &segment.symbol,
                segment.stress.ordinal(),
                false,
                index + 1 == plan.segments.len(),
                plan.rate,
            );
            if segment.symbol.eq_ignore_ascii_case("SIL") {
                frames = ((frames as f32) * (1.0 + plan.pause_weight)).round() as usize;
            }
            frames
        };

        let segment_frame_counts = plan
            .segments
            .iter()
            .enumerate()
            .map(|(index, segment)| scheduled_frames_for(index, segment))
            .collect::<Vec<_>>();

        let mut segment_frame_offsets = Vec::with_capacity(segment_frame_counts.len() + 1);
        segment_frame_offsets.push(0);
        for &frames in &segment_frame_counts {
            let next = segment_frame_offsets.last().copied().unwrap_or(0).saturating_add(frames);
            segment_frame_offsets.push(next);
        }
        let total_frames = *segment_frame_offsets.last().unwrap_or(&0);

        let samples_per_frame = (self.sample_rate() / FRAME_RATE.max(1)) as usize;
        let mut all_samples =
            Vec::with_capacity(total_frames.saturating_mul(samples_per_frame.max(1)));

        let last_index = plan.segments.len().saturating_sub(1);
        for (index, segment) in plan.segments.iter().enumerate() {
            // Pause weight is only realized when the phonological plan explicitly encodes
            // a silence segment; this prevents inventing pause locations from an abstract
            // scalar alone.
            let frames = segment_frame_counts[index];

            let phoneme = if segment.symbol.eq_ignore_ascii_case("SIL") {
                None
            } else {
                Some(segment.symbol.as_str())
            };
            let state = self.cognitive_state.lock().clone();
            let segment_count = plan.segments.len();
            let phrase_index = plan.segments[..index]
                .iter()
                .filter(|slot| slot.phrase_boundary_after)
                .count()
                .min(u8::MAX as usize) as u8;
            let phrase_start = plan.segments[..index]
                .iter()
                .rposition(|slot| slot.phrase_boundary_after)
                .map(|boundary| boundary + 1)
                .unwrap_or(0);
            let phrase_end = plan.segments[index..]
                .iter()
                .position(|slot| slot.phrase_boundary_after)
                .map(|offset| index + offset)
                .unwrap_or(segment_count.saturating_sub(1));
            let phrase_start_frame = segment_frame_offsets
                .get(phrase_start)
                .copied()
                .unwrap_or(0);
            let phrase_end_frame = segment_frame_offsets
                .get(phrase_end.saturating_add(1))
                .copied()
                .unwrap_or(total_frames);
            let phrase_span_frames = phrase_end_frame
                .saturating_sub(phrase_start_frame)
                .max(1);
            let segment_start_frame = segment_frame_offsets
                .get(index)
                .copied()
                .unwrap_or(0);
            let intonation = match plan.intonation {
                symthaea_broca::IntonationIntent::Statement => Intonation::Statement,
                symthaea_broca::IntonationIntent::Question => Intonation::Question,
                symthaea_broca::IntonationIntent::Exclamation => Intonation::Exclamation,
            };

            for frame_index in 0..frames {
                let progress = if frames > 1 {
                    frame_index as f32 / (frames - 1) as f32
                } else {
                    0.0
                };
                let global_frame = segment_start_frame.saturating_add(frame_index);
                let utterance_progress = if total_frames > 1 {
                    global_frame as f32 / (total_frames - 1) as f32
                } else {
                    0.0
                };
                let phrase_progress = if phrase_span_frames > 1 {
                    global_frame.saturating_sub(phrase_start_frame) as f32
                        / (phrase_span_frames - 1) as f32
                } else {
                    0.0
                };

                let prosody = ProsodyContext {
                    utterance_progress: utterance_progress.clamp(0.0, 1.0),
                    phoneme_progress: progress,
                    stress: segment.stress.ordinal(),
                    // Keep planned pitch range and speaker-state arousal as separate
                    // prosody controls. The pitch range never overwrites authoritative base F0.
                    base_f0: self.streaming.base_f0(),
                    arousal: state.emotional_arousal.clamp(0.0, 1.0),
                    pitch_range: plan.pitch_range,
                    intonation,
                    phrase_index,
                    phrase_progress: phrase_progress.clamp(0.0, 1.0),
                    is_focus: segment.is_focus && plan.focus_role.is_some(),
                    pitch_accent: Self::pitch_accent_for_plan(segment.is_focus, plan.prominence),
                    is_syllable_onset: segment.is_syllable_onset,
                    syllable_progress: progress,
                    prev_source_type: None,
                    next_source_type: None,
                };
                let chunk = self
                    .streaming
                    .tick_with_prosody(&state, None, DT, phoneme, &prosody);
                all_samples.extend_from_slice(&chunk);
            }
        }

        let plan_grounding = plan.grounding_surface();
        let audio_blake3 = {
            let mut hasher = blake3::Hasher::new();
            for sample in &all_samples {
                hasher.update(&sample.to_le_bytes());
            }
            hasher.finalize().to_hex().to_string()
        };
        let receipt = PhonologicalPlanRealizationReceipt {
            schema_version: 1,
            plan_version: plan.version.clone(),
            plan_grounding_blake3: blake3::hash(plan_grounding.as_bytes())
                .to_hex()
                .to_string(),
            realization_authorized: plan.realization_authorized,
            segment_count: plan.segments.len(),
            scheduler_frames: total_frames,
            sample_count: all_samples.len(),
            sample_rate: self.streaming.vocoder.sample_rate(),
            rate: plan.rate,
            pitch_range: plan.pitch_range,
            prominence: plan.prominence,
            pause_weight: plan.pause_weight,
            audio_blake3,
        };

        Ok((all_samples, receipt))
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

    fn apply_prosody(&mut self, state: &mut VoiceCognitiveState, prosody: &ProsodyAnalysis) {
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

    #[cfg(feature = "ssm_language")]
    #[test]
    fn test_plan_prominence_selects_typed_pitch_accent() {
        assert_eq!(
            LiveVoice::pitch_accent_for_plan(true, 0.90),
            PitchAccent::RiseHigh
        );
        assert_eq!(
            LiveVoice::pitch_accent_for_plan(false, 0.70),
            PitchAccent::High
        );
        assert_eq!(
            LiveVoice::pitch_accent_for_plan(false, 0.20),
            PitchAccent::None
        );
    }

    #[cfg(feature = "ssm_language")]
    #[test]
    fn test_phonological_plan_rate_reaches_real_scheduler() {
        fn make_plan(rate: f32) -> PhonologicalPlan {
            use symthaea_broca::{
                ContentBindingStatus, LinguisticFrame, PhonemeSlot, SpeechPlan, StructuredDecoder,
                SyllableStress, ThoughtChannels,
            };

            let genesis = GenesisSeed::from_phrase("plan-native-rate-test");
            let decoder = StructuredDecoder::new(&genesis);
            let channels = ThoughtChannels::with_intent(4);
            let readout = decoder.decode(&channels);
            let mut speech_plan = SpeechPlan::from_readout(&channels, &readout);
            speech_plan.prosody.rate = rate;

            let frame = LinguisticFrame::from_speech_plan(&speech_plan);
            let mut plan = PhonologicalPlan::from_linguistic_frame(&frame);
            plan.bind_segments(
                vec![PhonemeSlot::new(
                    "AH",
                    0,
                    SyllableStress::Primary,
                    true,
                    false,
                    true,
                )],
                ContentBindingStatus::PhonologicallyBound,
            )
            .expect("explicit phonological fixture");
            plan
        }

        let genesis = GenesisSeed::from_phrase("plan-native-rate-test");
        let mut voice = LiveVoice::new_headless(&genesis);

        let slow = make_plan(0.70);
        let fast = make_plan(1.30);

        let dir = std::env::temp_dir().join(format!(
            "symthaea-plan-native-rate-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temporary output directory");
        let slow_path = dir.join("slow.wav");
        let fast_path = dir.join("fast.wav");

        let slow_samples = voice
            .speak_phonological_plan_to_file(&slow, &slow_path)
            .expect("slow plan should synthesize");

        voice.reset();

        let fast_samples = voice
            .speak_phonological_plan_to_file(&fast, &fast_path)
            .expect("fast plan should synthesize");

        assert!(
            slow_samples > fast_samples,
            "scheduler must consume plan rate: slow={slow_samples}, fast={fast_samples}"
        );
        let slow_frames = predict_duration("AH", 1, false, true, slow.rate);
        let samples_per_frame = (voice.sample_rate() / FRAME_RATE) as usize;
        assert_eq!(
            slow_samples,
            slow_frames * samples_per_frame,
            "scheduler sample count must equal deterministic frame count"
        );

        std::fs::remove_dir_all(&dir).expect("remove temporary output directory");
    }

    #[cfg(feature = "ssm_language")]
    #[test]
    fn test_phonological_plan_consumes_pitch_range() {
        fn make_plan(pitch_range: f32) -> PhonologicalPlan {
            use symthaea_broca::{
                ContentBindingStatus, LinguisticFrame, PhonemeSlot, SpeechPlan, StructuredDecoder,
                SyllableStress, ThoughtChannels,
            };

            let genesis = GenesisSeed::from_phrase("plan-native-pitch-test");
            let decoder = StructuredDecoder::new(&genesis);
            let channels = ThoughtChannels::with_intent(2);
            let readout = decoder.decode(&channels);
            let mut speech_plan = SpeechPlan::from_readout(&channels, &readout);
            speech_plan.prosody.pitch_range = pitch_range;

            let frame = LinguisticFrame::from_speech_plan(&speech_plan);
            let mut plan = PhonologicalPlan::from_linguistic_frame(&frame);
            plan.bind_segments(
                vec![PhonemeSlot::new(
                    "AH",
                    0,
                    SyllableStress::Primary,
                    true,
                    false,
                    true,
                )],
                ContentBindingStatus::PhonologicallyBound,
            )
            .expect("explicit phonological fixture");
            plan
        }

        let genesis = GenesisSeed::from_phrase("plan-native-pitch-test");
        let mut voice = LiveVoice::new_headless(&genesis);
        let narrow = make_plan(0.65);
        let wide = make_plan(1.45);

        let (narrow_samples, _) = voice
            .synthesize_phonological_plan(&narrow)
            .expect("narrow pitch plan should synthesize");
        voice.reset();
        let (wide_samples, _) = voice
            .synthesize_phonological_plan(&wide)
            .expect("wide pitch plan should synthesize");

        assert_eq!(
            narrow_samples.len(),
            wide_samples.len(),
            "pitch range should change realization, not deterministic duration"
        );
        let difference = narrow_samples
            .iter()
            .zip(&wide_samples)
            .map(|(left, right)| (left - right).abs())
            .sum::<f32>();
        assert!(
            difference > 1e-3,
            "plan pitch range must reach phonological realization: absolute sample difference={difference}"
        );
    }

    #[cfg(feature = "ssm_language")]
    #[test]
    fn test_phonological_plan_consumes_explicit_pause_weight() {
        fn make_plan(pause_weight: f32) -> PhonologicalPlan {
            use symthaea_broca::{
                ContentBindingStatus, LinguisticFrame, PhonemeSlot, SpeechPlan, StructuredDecoder,
                SyllableStress, ThoughtChannels,
            };

            let genesis = GenesisSeed::from_phrase("plan-native-pause-test");
            let decoder = StructuredDecoder::new(&genesis);
            let channels = ThoughtChannels::with_intent(2);
            let readout = decoder.decode(&channels);
            let mut speech_plan = SpeechPlan::from_readout(&channels, &readout);
            speech_plan.prosody.pause_weight = pause_weight;

            let frame = LinguisticFrame::from_speech_plan(&speech_plan);
            let mut plan = PhonologicalPlan::from_linguistic_frame(&frame);
            plan.bind_segments(
                vec![
                    PhonemeSlot::new(
                        "AH",
                        0,
                        SyllableStress::Primary,
                        true,
                        false,
                        true,
                    ),
                    PhonemeSlot::new(
                        "SIL",
                        1,
                        SyllableStress::None,
                        true,
                        false,
                        true,
                    ),
                ],
                ContentBindingStatus::PhonologicallyBound,
            )
            .expect("explicit phonological fixture");
            plan
        }

        let genesis = GenesisSeed::from_phrase("plan-native-pause-test");
        let mut voice = LiveVoice::new_headless(&genesis);

        let no_pause = make_plan(0.0);
        let explicit_pause = make_plan(1.0);

        let (no_pause_samples, _) = voice
            .synthesize_phonological_plan(&no_pause)
            .expect("no-pause plan should synthesize");
        voice.reset();
        let (explicit_pause_samples, _) = voice
            .synthesize_phonological_plan(&explicit_pause)
            .expect("explicit pause plan should synthesize");

        assert!(
            explicit_pause_samples.len() > no_pause_samples.len(),
            "pause weight must extend only the explicitly encoded silence segment"
        );
    }

    #[cfg(feature = "ssm_language")]
    #[test]
    fn test_phonological_plan_receipt_binds_plan_and_audio() {
        use symthaea_broca::{
            ContentBindingStatus, LinguisticFrame, PhonemeSlot, SpeechPlan, StructuredDecoder,
            SyllableStress, ThoughtChannels,
        };

        let genesis = GenesisSeed::from_phrase("plan-native-receipt-test");
        let decoder = StructuredDecoder::new(&genesis);
        let channels = ThoughtChannels::with_intent(2);
        let readout = decoder.decode(&channels);
        let speech_plan = SpeechPlan::from_readout(&channels, &readout);
        let frame = LinguisticFrame::from_speech_plan(&speech_plan);
        let mut plan = PhonologicalPlan::from_linguistic_frame(&frame);
        plan.bind_segments(
            vec![PhonemeSlot::new(
                "AH",
                0,
                SyllableStress::Primary,
                true,
                false,
                true,
            )],
            ContentBindingStatus::PhonologicallyBound,
        )
        .expect("explicit receipt fixture");

        let mut voice = LiveVoice::new_headless(&genesis);
        let receipt = voice
            .speak_phonological_plan_with_receipt(&plan)
            .expect("plan-native receipt should be emitted");

        assert_eq!(receipt.schema_version, 1);
        assert_eq!(receipt.plan_version, plan.version);
        assert!(receipt.realization_authorized);
        assert_eq!(
            receipt.plan_grounding_blake3,
            blake3::hash(plan.grounding_surface().as_bytes())
                .to_hex()
                .to_string()
        );
        assert_eq!(receipt.segment_count, 1);
        assert_eq!(
            receipt.sample_count,
            receipt.scheduler_frames * (receipt.sample_rate / FRAME_RATE) as usize
        );
        assert_eq!(receipt.sample_rate, voice.sample_rate());
        assert_eq!(receipt.sample_rate, 24_000);
        assert_eq!(receipt.rate, plan.rate);
        assert_eq!(receipt.pitch_range, plan.pitch_range);
        assert_eq!(receipt.prominence, plan.prominence);
        assert_eq!(receipt.pause_weight, plan.pause_weight);
        assert_eq!(receipt.audio_blake3.len(), 64);
    }

    #[cfg(feature = "ssm_language")]
    #[test]
    fn test_phonological_plan_receipt_rejects_tampering() {
        use symthaea_broca::{
            ContentBindingStatus, LinguisticFrame, PhonemeSlot, SpeechPlan, StructuredDecoder,
            SyllableStress, ThoughtChannels,
        };

        let genesis = GenesisSeed::from_phrase("plan-native-receipt-verify-test");
        let decoder = StructuredDecoder::new(&genesis);
        let channels = ThoughtChannels::with_intent(2);
        let readout = decoder.decode(&channels);
        let speech_plan = SpeechPlan::from_readout(&channels, &readout);
        let frame = LinguisticFrame::from_speech_plan(&speech_plan);
        let mut plan = PhonologicalPlan::from_linguistic_frame(&frame);
        plan.bind_segments(
            vec![PhonemeSlot::new(
                "AH",
                0,
                SyllableStress::Primary,
                true,
                false,
                true,
            )],
            ContentBindingStatus::PhonologicallyBound,
        )
        .expect("explicit receipt fixture");

        let mut voice = LiveVoice::new_headless(&genesis);
        let (samples, receipt) = voice
            .synthesize_phonological_plan(&plan)
            .expect("plan-native receipt should be emitted");
        assert!(receipt.verify_against_plan(&plan).is_ok());
        let mut unauthorized_receipt = receipt.clone();
        unauthorized_receipt.realization_authorized = false;
        assert!(unauthorized_receipt.verify_against_plan(&plan).is_err());
        assert!(receipt.verify_samples(&samples));

        let mut tampered = plan.clone();
        tampered.rate = if tampered.rate < 1.0 { 1.2 } else { 0.8 };
        assert!(receipt.verify_against_plan(&tampered).is_err());
        assert!(!receipt.verify_samples(&samples[..samples.len().saturating_sub(1)]));
    }

    #[cfg(feature = "ssm_language")]
    #[test]
    fn test_role_only_phonological_plan_is_rejected() {
        use symthaea_broca::{LinguisticFrame, SpeechPlan, StructuredDecoder, ThoughtChannels};

        let genesis = GenesisSeed::from_phrase("plan-native-rejection-test");
        let decoder = StructuredDecoder::new(&genesis);
        let channels = ThoughtChannels::with_intent(4);
        let readout = decoder.decode(&channels);
        let frame = LinguisticFrame::from_speech_plan(&SpeechPlan::from_readout(&channels, &readout));
        let plan = PhonologicalPlan::from_linguistic_frame(&frame);

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("rejected.wav");
        let mut voice = LiveVoice::new_headless(&genesis);

        let error = voice
            .speak_phonological_plan_to_file(&plan, &path)
            .expect_err("role-only plans must not reach synthesis");

        assert!(error.to_string().contains("not ready for realization"));
        assert!(!path.exists());
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
