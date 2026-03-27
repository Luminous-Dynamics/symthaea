// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Streaming PCM synthesis with persistent DSP state.
//!
//! [`StreamingSynth`] wraps [`MuseStream`](crate::stream::MuseStream) with
//! persistent oscillator phases and a simple feedback-delay reverb to produce
//! gapless stereo PCM chunks for ring-buffer audio output.

use crate::stream::MuseStream;
use crate::{MuseConfig, MusicalState, Note};

/// Default chunk duration in milliseconds (~32ms for 31Hz cognitive loop).
pub const DEFAULT_CHUNK_MS: u32 = 32;
const MAX_ACTIVE_NOTES: usize = 32;
const MIN_CHUNK_SAMPLES: usize = 64;

// ─── Simple stereo delay reverb ──────────────────────────────────────────────

struct DelayReverb {
    buf_l: Vec<f32>,
    buf_r: Vec<f32>,
    write_pos: usize,
    delay_samples: usize,
    feedback: f32,
    wet: f32,
}

impl DelayReverb {
    fn new(sample_rate: u32, delay_ms: f32, feedback: f32, wet: f32) -> Self {
        let delay_samples = ((delay_ms / 1000.0) * sample_rate as f32) as usize;
        let cap = delay_samples.max(1);
        Self {
            buf_l: vec![0.0; cap],
            buf_r: vec![0.0; cap],
            write_pos: 0,
            delay_samples: cap,
            feedback: feedback.clamp(0.0, 0.95),
            wet: wet.clamp(0.0, 1.0),
        }
    }

    fn process(&mut self, in_l: f32, in_r: f32) -> (f32, f32) {
        let read_pos = self.write_pos;
        let delayed_l = self.buf_l[read_pos];
        let delayed_r = self.buf_r[read_pos];
        self.buf_l[self.write_pos] = in_l + delayed_l * self.feedback;
        self.buf_r[self.write_pos] = in_r + delayed_r * self.feedback;
        self.write_pos = (self.write_pos + 1) % self.delay_samples;
        let dry = 1.0 - self.wet;
        (in_l * dry + delayed_l * self.wet, in_r * dry + delayed_r * self.wet)
    }
}

// ─── ADSR ────────────────────────────────────────────────────────────────────

fn envelope(attack: f32, decay: f32, sustain: f32, release: f32, t: f32, note_dur: f32) -> f32 {
    if t < attack {
        t / attack
    } else if t < attack + decay {
        1.0 - (t - attack) / decay * (1.0 - sustain)
    } else if t < note_dur {
        sustain
    } else {
        let r = (t - note_dur) / release;
        sustain * (1.0 - r).max(0.0)
    }
}

fn soft_clip(x: f32) -> f32 {
    if x > 1.0 { 1.0 - (-x + 1.0).exp() * 0.5 }
    else if x < -1.0 { -1.0 + (x + 1.0).exp() * 0.5 }
    else { x }
}

// ─── StreamingSynth ──────────────────────────────────────────────────────────

struct ActiveNote {
    note: Note,
    sample_pos: usize,
    total_samples: usize,
    partial_phases: Vec<f32>,
    fm_phase: f32,
    pan: f32,
    volume: f32,
}

/// Streaming PCM synthesis engine with persistent DSP state.
pub struct StreamingSynth {
    muse_stream: MuseStream,
    reverb: DelayReverb,
    active_notes: Vec<ActiveNote>,
    state: MusicalState,
    config: MuseConfig,
    sample_rate: u32,
    chunk_samples: usize,
    total_samples_rendered: u64,
    chunks_rendered: u64,
    note_gen_cadence: u32,
    chunks_since_note_gen: u32,
}

impl StreamingSynth {
    pub fn new(config: MuseConfig, sample_rate: u32) -> Self {
        let chunk_samples = ((DEFAULT_CHUNK_MS as f32 / 1000.0) * sample_rate as f32) as usize;
        let chunk_samples = chunk_samples.max(MIN_CHUNK_SAMPLES);
        let reverb = DelayReverb::new(sample_rate, 120.0, 0.3, 0.15);
        Self {
            muse_stream: MuseStream::new(42, config.clone()),
            reverb,
            active_notes: Vec::with_capacity(MAX_ACTIVE_NOTES),
            state: MusicalState::default(),
            config,
            sample_rate,
            chunk_samples,
            total_samples_rendered: 0,
            chunks_rendered: 0,
            note_gen_cadence: 4,
            chunks_since_note_gen: 0,
        }
    }

    pub fn update_state(&mut self, state: &MusicalState) {
        self.state = state.clone();
        self.muse_stream.update_state(state);
        self.reverb.wet = 0.1 + state.consciousness_level * 0.3;
    }

    pub fn set_chunk_duration_ms(&mut self, ms: u32) {
        let s = ((ms as f32 / 1000.0) * self.sample_rate as f32) as usize;
        self.chunk_samples = s.max(MIN_CHUNK_SAMPLES);
    }

    pub fn render_chunk(&mut self) -> Vec<[f32; 2]> {
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

        // Timbre: harmonic rolloff modulated by serotonin
        let rolloff = 1.0 + self.state.serotonin * 0.8;
        let brightness = 0.3 + self.state.dopamine * 0.7;

        // ADSR from state
        let atk = 0.01 + (1.0 - self.state.arousal) * 0.05;
        let dec = 0.05 + self.state.serotonin * 0.1;
        let sus = 0.4 + self.state.consciousness_level * 0.4;
        let rel = 0.1 + self.state.harmony_activations[7] * 0.3;

        let mut buffer = vec![[0.0f32; 2]; self.chunk_samples];

        for active in &mut self.active_notes {
            let theta = (active.pan + 1.0) * std::f32::consts::FRAC_PI_4;
            let (gain_l, gain_r) = (theta.cos(), theta.sin());

            for i in 0..self.chunk_samples {
                if active.sample_pos >= active.total_samples {
                    break;
                }
                let t = active.sample_pos as f32 / sr;
                let env = envelope(atk, dec, sus, rel, t, active.note.duration);

                // FM
                active.fm_phase += (active.note.frequency * fm_ratio / sr) * std::f32::consts::TAU;
                if active.fm_phase > std::f32::consts::TAU {
                    active.fm_phase -= std::f32::consts::TAU;
                }
                let fm_off = fm_depth * active.fm_phase.sin();

                // Additive synthesis
                let mut sample = 0.0f32;
                for h in 0..num_p {
                    let cf = active.note.frequency * (h + 1) as f32;
                    if cf >= nyquist {
                        break;
                    }
                    while active.partial_phases.len() <= h {
                        active.partial_phases.push(0.0);
                    }
                    active.partial_phases[h] += (cf / sr) * std::f32::consts::TAU;
                    if active.partial_phases[h] > std::f32::consts::TAU * 2.0 {
                        active.partial_phases[h] -= std::f32::consts::TAU * 2.0;
                    }
                    let amp = if h == 0 {
                        1.0
                    } else {
                        brightness / ((h + 1) as f32).powf(rolloff)
                    };
                    sample += amp * (active.partial_phases[h] + fm_off).sin();
                }

                let out = sample * env * active.note.velocity * active.volume * 0.15;
                buffer[i][0] += out * gain_l;
                buffer[i][1] += out * gain_r;
                active.sample_pos += 1;
            }
        }

        // Reverb + soft clip
        for pair in &mut buffer {
            let (l, r) = self.reverb.process(pair[0], pair[1]);
            pair[0] = soft_clip(l);
            pair[1] = soft_clip(r);
        }

        self.total_samples_rendered += self.chunk_samples as u64;
        self.chunks_rendered += 1;
        buffer
    }

    fn generate_notes(&mut self) {
        if self.active_notes.len() >= MAX_ACTIVE_NOTES {
            return;
        }
        let sr = self.sample_rate as f32;
        let rel = 0.1 + self.state.harmony_activations[7] * 0.3;
        let release_samples = (rel * sr) as usize;
        let psi = self.state.consciousness_level;

        for _ in 0..4 {
            if self.active_notes.len() >= MAX_ACTIVE_NOTES {
                break;
            }
            if let Some(note) = self.muse_stream.next_note() {
                let idx = self.active_notes.len();
                let pan = match idx % 3 {
                    0 => 0.0,
                    1 => if psi > 0.4 { -0.2 } else { 0.0 },
                    _ => if psi > 0.6 { 0.4 } else { 0.0 },
                };
                self.active_notes.push(ActiveNote {
                    total_samples: (note.duration * sr) as usize + release_samples,
                    note,
                    sample_pos: 0,
                    partial_phases: vec![0.0; self.config.num_partials.clamp(1, 16)],
                    fm_phase: 0.0,
                    pan,
                    volume: if idx == 0 { 1.0 } else { 0.6 },
                });
            }
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    pub fn chunk_samples(&self) -> usize {
        self.chunk_samples
    }
    pub fn total_samples_rendered(&self) -> u64 {
        self.total_samples_rendered
    }
    pub fn chunks_rendered(&self) -> u64 {
        self.chunks_rendered
    }
    pub fn active_note_count(&self) -> usize {
        self.active_notes.len()
    }
    pub fn tempo(&self) -> f32 {
        self.muse_stream.tempo()
    }
    pub fn reset(&mut self, seed: u64) {
        self.muse_stream.reset(seed);
        self.active_notes.clear();
        self.total_samples_rendered = 0;
        self.chunks_rendered = 0;
        self.chunks_since_note_gen = 0;
        self.reverb = DelayReverb::new(self.sample_rate, 120.0, 0.3, 0.15);
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
}
