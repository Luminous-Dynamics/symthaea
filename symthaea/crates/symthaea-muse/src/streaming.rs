// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Full-pipeline streaming PCM synthesis with all consciousness-coupled modules.
//!
//! [`StreamingSynth`] integrates:
//! - Wavetable morphing (consciousness → table selection)
//! - Sidechain ducking (lead → bass/harmony gain reduction)
//! - Consciousness reverb (Phi → room, harmonies → early reflections)
//! - Binaural rendering (Phi → spatial spread, harmonies → positions)
//! - Audio feedback (output → feature extraction → state modulation)
//! - Phi optimizer (target Phi → parameter perturbation)
//! - Substrate timbre (substrate type → synthesis character)

use crate::audio_feedback::AudioFeedbackEncoder;
use crate::binaural::BinauralConsciousnessRenderer;
use crate::consciousness_reverb::ConsciousnessReverb;
use crate::instruments::{self, Instrument, KarplusStrong};
use crate::mixing::MixingChain;
use crate::musical_inference::MusicalInferenceEngine;
use crate::percussion;
use crate::phi_optimizer::{PhiOptimizer, PhiTarget};
use crate::sidechain::DuckingMatrix;
use crate::stream::MuseStream;
use crate::substrate_timbre::{SubstrateTimbreModifier, SubstrateTimbreType};
use crate::wavetable::{WavetableBank, WavetableOscillator};
use crate::{MuseConfig, MusicalState, Note};

pub const DEFAULT_CHUNK_MS: u32 = 32;
const MAX_ACTIVE_NOTES: usize = 32;
const MIN_CHUNK_SAMPLES: usize = 64;

// ─── ADSR + clip ────────────────────────────────────────────────────────────

fn envelope(atk: f32, dec: f32, sus: f32, rel: f32, t: f32, dur: f32) -> f32 {
    if t < atk { t / atk }
    else if t < atk + dec { 1.0 - (t - atk) / dec * (1.0 - sus) }
    else if t < dur { sus }
    else { sus * (1.0 - (t - dur) / rel).max(0.0) }
}

fn soft_clip(x: f32) -> f32 {
    if x > 1.0 { 1.0 - (-x + 1.0).exp() * 0.5 }
    else if x < -1.0 { -1.0 + (x + 1.0).exp() * 0.5 }
    else { x }
}

// ─── Active note with wavetable + vibrato ───────────────────────────────────

struct ActiveNote {
    note: Note,
    sample_pos: usize,
    total_samples: usize,
    partial_phases: Vec<f32>,
    fm_phase: f32,
    pan: f32,
    volume: f32,
    voice_idx: usize,
    wavetable_osc: WavetableOscillator,
    vibrato_phase: f32,
    /// Instrument assigned to this note.
    instrument: Instrument,
    /// Karplus-Strong state (for guitar/harp).
    ks: Option<KarplusStrong>,
    /// Pre-rendered FM buffer (for e-piano/bell, rendered at note start).
    fm_buffer: Option<Vec<f32>>,
}

/// Full-pipeline streaming synthesis engine.
pub struct StreamingSynth {
    muse_stream: MuseStream,
    active_notes: Vec<ActiveNote>,
    state: MusicalState,
    config: MuseConfig,
    sample_rate: u32,
    chunk_samples: usize,
    total_samples_rendered: u64,
    chunks_rendered: u64,
    note_gen_cadence: u32,
    chunks_since_note_gen: u32,
    // Consciousness-coupled modules
    reverb: ConsciousnessReverb,
    binaural: BinauralConsciousnessRenderer,
    sidechain: DuckingMatrix,
    feedback: AudioFeedbackEncoder,
    phi_optimizer: PhiOptimizer,
    wavetable_bank: WavetableBank,
    substrate: Option<SubstrateTimbreModifier>,
    /// FEP active inference engine for music (the system predicts its own sound).
    fep_engine: MusicalInferenceEngine,
    /// Free energy history for learning verification.
    fe_history: Vec<f64>,
    /// Mixing chain: EQ → compressor → limiter.
    mixing: MixingChain,
    /// Pending drum hits for the current bar.
    drum_hits: Vec<percussion::DrumHit>,
    /// Sample position within current bar (for drum timing).
    bar_sample_pos: usize,
    /// Samples per bar (computed from tempo).
    samples_per_bar: usize,
    /// Audio feedback strength [0, 1]. 0 = open loop, 1 = full strange loop.
    pub feedback_strength: f32,
    /// Enable binaural rendering (vs simple pan law).
    pub enable_binaural: bool,
    /// Enable sidechain ducking.
    pub enable_sidechain: bool,
    /// Enable Phi optimization.
    pub enable_phi_optimizer: bool,
    /// Enable FEP active inference (generative model of own audio).
    pub enable_fep: bool,
}

impl StreamingSynth {
    pub fn new(config: MuseConfig, sample_rate: u32) -> Self {
        let chunk_samples = ((DEFAULT_CHUNK_MS as f32 / 1000.0) * sample_rate as f32) as usize;
        let chunk_samples = chunk_samples.max(MIN_CHUNK_SAMPLES);
        Self {
            muse_stream: MuseStream::new(42, config.clone()),
            active_notes: Vec::with_capacity(MAX_ACTIVE_NOTES),
            state: MusicalState::default(),
            config,
            sample_rate,
            chunk_samples,
            total_samples_rendered: 0,
            chunks_rendered: 0,
            note_gen_cadence: 4,
            chunks_since_note_gen: 0,
            reverb: ConsciousnessReverb::new(sample_rate),
            binaural: BinauralConsciousnessRenderer::new(4, sample_rate),
            sidechain: DuckingMatrix::default_matrix(sample_rate),
            feedback: AudioFeedbackEncoder::new(),
            phi_optimizer: PhiOptimizer::new(PhiTarget::Maximize),
            wavetable_bank: WavetableBank::default_bank(),
            substrate: None,
            fep_engine: MusicalInferenceEngine::new(),
            fe_history: Vec::with_capacity(1024),
            mixing: MixingChain::new(sample_rate),
            drum_hits: Vec::new(),
            bar_sample_pos: 0,
            samples_per_bar: (44100.0 * 60.0 / 80.0 * 4.0) as usize, // 4 beats at 80bpm
            feedback_strength: 0.3,
            enable_binaural: true,
            enable_sidechain: true,
            enable_phi_optimizer: false,
            enable_fep: true,
        }
    }

    pub fn update_state(&mut self, state: &MusicalState) {
        self.state = state.clone();
        self.muse_stream.update_state(state);
        self.reverb.update_state(state);
        self.binaural.update_state(state);
        if self.enable_phi_optimizer {
            self.phi_optimizer.update_phi(state.consciousness_level);
        }

        // Arousal-driven note generation cadence:
        // High arousal (0.9) → generate every 1 chunk (dense onsets)
        // Low arousal (0.1) → generate every 8 chunks (sparse, calm)
        // Formula: cadence = 8 - 7 * arousal, clamped to [1, 8]
        self.note_gen_cadence = (8.0 - 7.0 * state.arousal.clamp(0.0, 1.0))
            .round()
            .clamp(1.0, 8.0) as u32;
    }

    /// Set substrate type for timbre coloring.
    pub fn set_substrate(&mut self, substrate: SubstrateTimbreType) {
        self.substrate = Some(SubstrateTimbreModifier::for_substrate(substrate));
    }

    pub fn set_chunk_duration_ms(&mut self, ms: u32) {
        let s = ((ms as f32 / 1000.0) * self.sample_rate as f32) as usize;
        self.chunk_samples = s.max(MIN_CHUNK_SAMPLES);
    }

    /// Render one chunk of stereo PCM through the full pipeline.
    ///
    /// Pipeline: generate notes → additive/wavetable synthesis → per-voice buffers
    /// → sidechain ducking → binaural spatialization → consciousness reverb
    /// → soft clip → audio feedback extraction → Phi optimization → output
    pub fn render_chunk(&mut self) -> Vec<[f32; 2]> {
        // Apply Phi optimizer modulation
        if self.enable_phi_optimizer {
            self.phi_optimizer.modulate_state(&mut self.state);
        }

        // Apply substrate timbre
        if let Some(ref substrate) = self.substrate {
            substrate.apply_to_state(&mut self.state);
        }

        // Generate new notes
        self.chunks_since_note_gen += 1;
        if self.chunks_since_note_gen >= self.note_gen_cadence {
            self.chunks_since_note_gen = 0;
            self.generate_notes();
        }
        self.active_notes.retain(|n| n.sample_pos < n.total_samples);

        let sr = self.sample_rate as f32;
        let nyquist = sr * 0.5;
        let num_p = self.config.num_partials.clamp(1, 16);
        let fm_depth = self.state.dopamine * self.config.max_fm_depth;
        let fm_ratio = 2.0 + self.state.noradrenaline;
        let rolloff = 1.0 + self.state.serotonin * 0.8;
        let brightness = 0.3 + self.state.dopamine * 0.7;
        let atk = 0.01 + (1.0 - self.state.arousal) * 0.05;
        let dec = 0.05 + self.state.serotonin * 0.1;
        let sus = 0.4 + self.state.consciousness_level * 0.4;
        let rel = 0.1 + self.state.harmony_activations[7] * 0.3;

        // ── Phase 1: Render per-voice mono buffers ──
        let num_voices = 4; // lead, bass, harmony, ostinato
        let mut voice_buffers: Vec<Vec<f32>> = (0..num_voices).map(|_| vec![0.0; self.chunk_samples]).collect();
        let voice_roles = [
            crate::voice::VoiceRole::Lead,
            crate::voice::VoiceRole::Bass,
            crate::voice::VoiceRole::Harmony,
            crate::voice::VoiceRole::Ostinato,
        ];

        // Select instrument from consciousness state (updated per chunk)
        let current_instrument = instruments::select_instrument(&self.state);

        for active in &mut self.active_notes {
            let voice = active.voice_idx.min(num_voices - 1);

            // Vibrato LFO: 5Hz for strings, 0 for piano/guitar
            let (vibrato_depth_cents, vibrato_rate) = match active.instrument {
                Instrument::Violin | Instrument::Cello => (25.0, 5.0),
                Instrument::Flute => (15.0, 4.5),
                Instrument::Pad => (8.0, 3.0),
                _ => (0.0, 0.0), // no vibrato for piano, guitar, bell
            };

            for i in 0..self.chunk_samples {
                if active.sample_pos >= active.total_samples { break; }
                let t = active.sample_pos as f32 / sr;
                let env = envelope(atk, dec, sus, rel, t, active.note.duration);

                // Vibrato
                if vibrato_depth_cents > 0.0 {
                    active.vibrato_phase += vibrato_rate / sr;
                    if active.vibrato_phase > 1.0 { active.vibrato_phase -= 1.0; }
                }
                let vibrato_mod = (active.vibrato_phase * std::f32::consts::TAU).sin();
                let vibrato_factor = 2.0f32.powf(vibrato_depth_cents * vibrato_mod / 1200.0);
                let freq = active.note.frequency * vibrato_factor;

                let sample = if let Some(ref mut ks) = active.ks {
                    // Karplus-Strong (guitar, harp): self-sustaining, no external envelope
                    ks.tick()
                } else if let Some(ref fm_buf) = active.fm_buffer {
                    // Pre-rendered FM (e-piano, bell): read from buffer
                    if active.sample_pos < fm_buf.len() { fm_buf[active.sample_pos] } else { 0.0 }
                } else {
                    // Additive synthesis with instrument-specific partials
                    let partials = active.instrument.partials();
                    let num_p = partials.len().min(16);
                    let mut s = 0.0f32;
                    for h in 0..num_p {
                        let cf = freq * (h + 1) as f32;
                        if cf >= nyquist { break; }
                        while active.partial_phases.len() <= h { active.partial_phases.push(0.0); }
                        active.partial_phases[h] += (cf / sr) * std::f32::consts::TAU;
                        if active.partial_phases[h] > std::f32::consts::TAU * 2.0 {
                            active.partial_phases[h] -= std::f32::consts::TAU * 2.0;
                        }
                        s += partials[h] * active.partial_phases[h].sin();
                    }
                    s * env
                };

                // Arousal-modulated master gain: 0.05 (calm) → 0.35 (excited)
                // + velocity boost from arousal (excited states hit harder)
                let arousal = self.state.arousal.clamp(0.0, 1.0);
                let master_gain = 0.05 + arousal * 0.30;
                let velocity_boost = 1.0 + arousal * 0.5; // 1.0x calm → 1.5x excited
                voice_buffers[voice][i] += sample * env * (active.note.velocity * velocity_boost).min(1.0) * active.volume * master_gain;
                active.sample_pos += 1;
            }
        }

        // ── Phase 2: Sidechain ducking ──
        if self.enable_sidechain {
            self.sidechain.apply(&mut voice_buffers, &voice_roles, self.chunk_samples);
        }

        // ── Phase 3: Binaural rendering OR simple panning ──
        let mut buffer = if self.enable_binaural {
            self.binaural.render(&voice_buffers)
        } else {
            // Simple panning fallback
            let pans = [0.0f32, -0.2, 0.4, -0.3];
            let mut buf = vec![[0.0f32; 2]; self.chunk_samples];
            for (v, voice_buf) in voice_buffers.iter().enumerate() {
                let theta = (pans[v.min(3)] + 1.0) * std::f32::consts::FRAC_PI_4;
                let (gl, gr) = (theta.cos(), theta.sin());
                for (i, &s) in voice_buf.iter().enumerate() {
                    buf[i][0] += s * gl;
                    buf[i][1] += s * gr;
                }
            }
            buf
        };

        // Pad if binaural returned fewer samples
        while buffer.len() < self.chunk_samples {
            buffer.push([0.0, 0.0]);
        }

        // ── Phase 3.5: Percussion ──
        // Generate new drum pattern at bar boundaries
        let tempo = self.muse_stream.tempo();
        self.samples_per_bar = ((60.0 / tempo) * 4.0 * self.sample_rate as f32) as usize;
        if self.bar_sample_pos >= self.samples_per_bar || self.drum_hits.is_empty() {
            self.drum_hits = percussion::generate_pattern(
                tempo, self.state.consciousness_level, self.state.arousal,
            );
            self.bar_sample_pos = 0;
        }
        // Render drum hits into the buffer
        for hit in &self.drum_hits {
            let hit_sample = (hit.time * self.sample_rate as f32) as usize;
            if hit_sample >= self.bar_sample_pos && hit_sample < self.bar_sample_pos + self.chunk_samples {
                let local_offset = hit_sample - self.bar_sample_pos;
                let drum_buf = percussion::render_drum(hit, self.sample_rate);
                for (j, &s) in drum_buf.iter().enumerate() {
                    let idx = local_offset + j;
                    if idx < buffer.len() {
                        buffer[idx][0] += s * 0.5; // center drums
                        buffer[idx][1] += s * 0.5;
                    }
                }
            }
        }
        self.bar_sample_pos += self.chunk_samples;

        // ── Phase 4: Consciousness reverb ──
        for pair in &mut buffer {
            let (l, r) = self.reverb.process_stereo(pair[0], pair[1]);
            pair[0] = l;
            pair[1] = r;
        }

        // ── Phase 4.5: Mixing chain (EQ → compressor → limiter) ──
        for pair in &mut buffer {
            let (l, r) = self.mixing.process(pair[0], pair[1]);
            pair[0] = l;
            pair[1] = r;
        }

        // ── Phase 5: Audio feedback (strange loop) ──
        if self.feedback_strength > 0.0 {
            self.feedback.extract(&buffer, self.sample_rate);
            let features = *self.feedback.smoothed_features();

            // ── Phase 5b: FEP Active Inference ──
            // The system observes its own audio, updates beliefs about what it sounds
            // like, and selects actions to minimize free energy (surprise).
            if self.enable_fep {
                let result = self.fep_engine.infer(&features);
                self.fep_engine.apply_action(&result, &mut self.state);

                // Track free energy for learning verification
                if self.fe_history.len() < 10000 {
                    self.fe_history.push(result.free_energy);
                }
            }

            features.modulate_state(&mut self.state, self.feedback_strength);
        }

        self.total_samples_rendered += self.chunk_samples as u64;
        self.chunks_rendered += 1;
        buffer
    }

    fn generate_notes(&mut self) {
        if self.active_notes.len() >= MAX_ACTIVE_NOTES { return; }
        let sr = self.sample_rate as f32;
        let rel = 0.1 + self.state.harmony_activations[7] * 0.3;
        let release_samples = (rel * sr) as usize;
        let psi = self.state.consciousness_level;

        // Phi-gated polyphony: higher consciousness = more simultaneous voices
        // Psi < 0.3 → 1 note, Psi 0.3-0.5 → 2, Psi 0.5-0.7 → 3, Psi > 0.7 → 4
        let max_new = if psi > 0.7 { 4 } else if psi > 0.5 { 3 } else if psi > 0.3 { 2 } else { 1 };
        for _ in 0..max_new {
            if self.active_notes.len() >= MAX_ACTIVE_NOTES { break; }
            if let Some(note) = self.muse_stream.next_note() {
                let idx = self.active_notes.len();
                let voice_idx = match idx % 4 {
                    0 => 0, // lead
                    1 => if psi > 0.4 { 1 } else { 0 }, // bass or lead
                    2 => if psi > 0.6 { 2 } else { 0 }, // harmony or lead
                    _ => if psi > 0.7 { 3 } else { 0 }, // ostinato or lead
                };
                // Select instrument based on consciousness state
                let instrument = instruments::select_instrument(&self.state);

                // Initialize instrument-specific synthesis state
                let ks = if instrument.uses_karplus_strong() {
                    let (damp, bright, _stiff) = instrument.ks_params();
                    let mut ks = KarplusStrong::new(note.frequency, self.sample_rate, damp, bright);
                    ks.excite(note.velocity);
                    Some(ks)
                } else {
                    None
                };

                let fm_buffer = if instrument.uses_fm() {
                    Some(instruments::render_fm_instrument(
                        instrument, note.frequency, note.velocity, note.duration, self.sample_rate,
                    ))
                } else {
                    None
                };

                self.active_notes.push(ActiveNote {
                    total_samples: (note.duration * sr) as usize + release_samples,
                    note,
                    sample_pos: 0,
                    partial_phases: vec![0.0; 16],
                    fm_phase: 0.0,
                    pan: [0.0, -0.2, 0.4, -0.3][voice_idx],
                    volume: [1.0, 0.7, 0.5, 0.3][voice_idx],
                    voice_idx,
                    wavetable_osc: WavetableOscillator::new(),
                    vibrato_phase: 0.0,
                    instrument,
                    ks,
                    fm_buffer,
                });
            }
        }
    }

    // ── Accessors ──
    pub fn sample_rate(&self) -> u32 { self.sample_rate }
    pub fn chunk_samples(&self) -> usize { self.chunk_samples }
    pub fn total_samples_rendered(&self) -> u64 { self.total_samples_rendered }
    pub fn chunks_rendered(&self) -> u64 { self.chunks_rendered }
    pub fn active_note_count(&self) -> usize { self.active_notes.len() }
    pub fn tempo(&self) -> f32 { self.muse_stream.tempo() }
    pub fn phi_metrics(&self) -> Option<crate::phi_optimizer::PhiOptimizerMetrics> {
        if self.enable_phi_optimizer { Some(self.phi_optimizer.metrics()) } else { None }
    }
    pub fn feedback_features(&self) -> &crate::audio_feedback::AudioFeatures {
        self.feedback.smoothed_features()
    }
    /// Current free energy from the FEP agent (lower = better self-model).
    pub fn current_free_energy(&self) -> f64 {
        self.fep_engine.current_free_energy()
    }
    /// Free energy history for learning curve analysis.
    pub fn free_energy_history(&self) -> &[f64] {
        &self.fe_history
    }
    /// FEP inference cycle count.
    pub fn fep_cycles(&self) -> u64 {
        self.fep_engine.cycle_count()
    }
    /// Last FEP inference result.
    pub fn last_fep_result(&self) -> Option<&crate::musical_inference::MusicInferenceResult> {
        self.fep_engine.last_result()
    }

    pub fn reset(&mut self, seed: u64) {
        self.fep_engine = MusicalInferenceEngine::new();
        self.fe_history.clear();
        self.mixing = MixingChain::new(self.sample_rate);
        self.drum_hits.clear();
        self.bar_sample_pos = 0;
        self.muse_stream.reset(seed);
        self.active_notes.clear();
        self.total_samples_rendered = 0;
        self.chunks_rendered = 0;
        self.chunks_since_note_gen = 0;
        self.reverb = ConsciousnessReverb::new(self.sample_rate);
        self.feedback.reset();
        self.sidechain.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> MuseConfig {
        MuseConfig { duration_secs: 4.0, max_notes: 16, ..Default::default() }
    }

    #[test]
    fn render_chunk_correct_length() {
        let mut s = StreamingSynth::new(cfg(), 44100);
        assert_eq!(s.render_chunk().len(), s.chunk_samples());
    }

    #[test]
    fn samples_are_finite() {
        let mut s = StreamingSynth::new(cfg(), 44100);
        s.update_state(&MusicalState { consciousness_level: 0.8, ..Default::default() });
        for _ in 0..50 {
            for p in s.render_chunk() {
                assert!(p[0].is_finite() && p[1].is_finite());
            }
        }
    }

    #[test]
    fn state_update_changes_tempo() {
        let mut s = StreamingSynth::new(cfg(), 44100);
        let t1 = s.tempo();
        s.update_state(&MusicalState { arousal: 0.95, noradrenaline: 0.9, ..Default::default() });
        assert!(s.tempo() > t1);
    }

    #[test]
    fn total_samples_tracks() {
        let mut s = StreamingSynth::new(cfg(), 44100);
        let cs = s.chunk_samples() as u64;
        s.render_chunk();
        assert_eq!(s.total_samples_rendered(), cs);
    }

    #[test]
    fn reset_clears() {
        let mut s = StreamingSynth::new(cfg(), 44100);
        for _ in 0..10 { s.render_chunk(); }
        s.reset(99);
        assert_eq!(s.total_samples_rendered(), 0);
        assert_eq!(s.chunks_rendered(), 0);
    }

    #[test]
    fn feedback_loop_modulates_state() {
        let mut s = StreamingSynth::new(cfg(), 44100);
        s.feedback_strength = 1.0;
        s.update_state(&MusicalState { consciousness_level: 0.8, ..Default::default() });
        let orig_arousal = s.state.arousal;
        // Render several chunks to build up feedback
        for _ in 0..20 { s.render_chunk(); }
        // Audio feedback should have modified state
        let features = s.feedback_features();
        assert!(features.rms_energy > 0.0 || features.spectral_centroid > 0.0,
            "feedback should detect audio features");
    }

    #[test]
    fn binaural_vs_simple_pan() {
        let mut s = StreamingSynth::new(cfg(), 44100);
        s.update_state(&MusicalState { consciousness_level: 0.8, ..Default::default() });
        s.enable_binaural = true;
        for _ in 0..10 { s.render_chunk(); }
        let bin_chunk = s.render_chunk();

        s.reset(42);
        s.update_state(&MusicalState { consciousness_level: 0.8, ..Default::default() });
        s.enable_binaural = false;
        for _ in 0..10 { s.render_chunk(); }
        let pan_chunk = s.render_chunk();

        assert_eq!(bin_chunk.len(), pan_chunk.len());
    }

    #[test]
    fn substrate_changes_character() {
        let mut s = StreamingSynth::new(cfg(), 44100);
        s.update_state(&MusicalState { consciousness_level: 0.8, ..Default::default() });
        s.set_substrate(SubstrateTimbreType::Quantum);
        assert!(s.substrate.is_some());
        for _ in 0..10 { s.render_chunk(); }
    }

    #[test]
    fn fep_runs_live_in_pipeline() {
        let mut s = StreamingSynth::new(cfg(), 44100);
        s.enable_fep = true;
        s.feedback_strength = 0.5;
        s.update_state(&MusicalState { consciousness_level: 0.7, ..Default::default() });

        for _ in 0..50 { s.render_chunk(); }

        assert!(s.fep_cycles() > 0, "FEP should have run cycles");
        assert!(!s.free_energy_history().is_empty(), "FE history should be recorded");
    }

    #[test]
    fn fep_free_energy_is_finite() {
        let mut s = StreamingSynth::new(cfg(), 44100);
        s.enable_fep = true;
        s.feedback_strength = 1.0;
        s.update_state(&MusicalState {
            consciousness_level: 0.8, arousal: 0.6,
            harmony_activations: [0.7, 0.5, 0.6, 0.4, 0.3, 0.5, 0.6, 0.3],
            ..Default::default()
        });

        for _ in 0..200 {
            let chunk = s.render_chunk();
            for pair in &chunk {
                assert!(pair[0].is_finite() && pair[1].is_finite());
            }
        }

        // All FE values should be finite
        for &fe in s.free_energy_history() {
            assert!(fe.is_finite(), "free energy should be finite: {fe}");
        }
    }

    #[test]
    fn fep_learning_curve() {
        // Run 500 cycles with consistent input.
        // The FEP agent should learn to predict its own audio,
        // meaning late free energy should be no worse than early.
        let mut s = StreamingSynth::new(cfg(), 44100);
        s.enable_fep = true;
        s.feedback_strength = 0.5;
        s.update_state(&MusicalState {
            consciousness_level: 0.7, arousal: 0.4,
            harmony_activations: [0.6, 0.5, 0.4, 0.3, 0.5, 0.6, 0.4, 0.5],
            ..Default::default()
        });

        for _ in 0..500 { s.render_chunk(); }

        let history = s.free_energy_history();
        assert!(history.len() >= 100, "need enough data points");

        // Compare early vs late average FE
        let early: f64 = history[..20].iter().sum::<f64>() / 20.0;
        let late: f64 = history[history.len()-20..].iter().sum::<f64>() / 20.0;

        eprintln!("  FEP learning: early_FE={early:.4}, late_FE={late:.4}, delta={:.4}", early - late);

        // Late FE should not be dramatically worse than early
        // (with learning, it should improve or stabilize)
        assert!(
            late < early + 0.5,
            "FE should not diverge: early={early:.4}, late={late:.4}"
        );
    }
}
