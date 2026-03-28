// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hi-Fi hybrid additive + FM synthesis with Freeverb and stereo rendering.
//!
//! Features: configurable partials (1-16), Freeverb (8 comb + 4 allpass),
//! sub-bass, detuned unison, filtered noise, per-voice stereo panning.

use crate::voice::Arrangement;
use crate::{AudioData, MuseConfig, MusicalState, Note, OutputFormat};

pub(crate) struct Adsr {
    pub(crate) attack: f32,
    pub(crate) decay: f32,
    pub(crate) sustain: f32,
    pub(crate) release: f32,
}

// ─── Freeverb ───────────────────────────────────────────────────────────────
const COMB_DELAYS: [usize; 8] = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617];
const ALLPASS_DELAYS: [usize; 4] = [556, 441, 341, 225];
const STEREO_SPREAD: usize = 23;

pub(crate) struct Freeverb {
    comb_l: Vec<CombFilter>,
    comb_r: Vec<CombFilter>,
    allpass_l: Vec<AllpassFilter>,
    allpass_r: Vec<AllpassFilter>,
    pub(crate) wet: f32,
}
struct CombFilter {
    buffer: Vec<f32>,
    index: usize,
    feedback: f32,
    damp1: f32,
    damp2: f32,
    filterstore: f32,
}
struct AllpassFilter {
    buffer: Vec<f32>,
    index: usize,
}

impl Freeverb {
    pub(crate) fn new(sample_rate: u32, room_size: f32, damping: f32, wet: f32) -> Self {
        let scale = sample_rate as f32 / 44100.0;
        let fb = 0.28 + room_size * 0.7;
        let mc = |d: usize, o: usize| CombFilter {
            buffer: vec![0.0; ((d + o) as f32 * scale) as usize + 1],
            index: 0,
            feedback: fb,
            damp1: damping,
            damp2: 1.0 - damping,
            filterstore: 0.0,
        };
        let ma = |d: usize, o: usize| AllpassFilter {
            buffer: vec![0.0; ((d + o) as f32 * scale) as usize + 1],
            index: 0,
        };
        Self {
            comb_l: COMB_DELAYS.iter().map(|&d| mc(d, 0)).collect(),
            comb_r: COMB_DELAYS.iter().map(|&d| mc(d, STEREO_SPREAD)).collect(),
            allpass_l: ALLPASS_DELAYS.iter().map(|&d| ma(d, 0)).collect(),
            allpass_r: ALLPASS_DELAYS
                .iter()
                .map(|&d| ma(d, STEREO_SPREAD))
                .collect(),
            wet,
        }
    }
    pub(crate) fn process_stereo(&mut self, il: f32, ir: f32) -> (f32, f32) {
        let (mut ol, mut or) = (0.0f32, 0.0f32);
        for c in &mut self.comb_l {
            ol += cp(c, il);
        }
        for c in &mut self.comb_r {
            or += cp(c, ir);
        }
        for a in &mut self.allpass_l {
            ol = ap(a, ol);
        }
        for a in &mut self.allpass_r {
            or = ap(a, or);
        }
        let d = 1.0 - self.wet;
        (il * d + ol * self.wet, ir * d + or * self.wet)
    }
}
fn cp(c: &mut CombFilter, input: f32) -> f32 {
    let o = c.buffer[c.index];
    c.filterstore = o * c.damp2 + c.filterstore * c.damp1;
    c.buffer[c.index] = input + c.filterstore * c.feedback;
    c.index = (c.index + 1) % c.buffer.len();
    o
}
fn ap(a: &mut AllpassFilter, input: f32) -> f32 {
    let b = a.buffer[a.index];
    let o = b - input;
    a.buffer[a.index] = input + b * 0.5;
    a.index = (a.index + 1) % a.buffer.len();
    o
}

// ─── Public API ─────────────────────────────────────────────────────────────
pub fn render_arrangement(
    arrangement: &Arrangement,
    sample_rate: u32,
    total_samples: usize,
    state: &MusicalState,
    config: &MuseConfig,
) -> AudioData {
    let sr = sample_rate as f32;
    let partials = compute_timbre(state, config.num_partials.clamp(1, 16));
    let adsr = compute_adsr(state);
    let fm_depth = state.dopamine * config.max_fm_depth;
    let fm_ratio = 2.0 + state.noradrenaline;
    let chord_intervals = compute_chord_intervals(state);
    let mut bl = vec![0.0f32; total_samples];
    let mut br = vec![0.0f32; total_samples];

    for voice in &arrangement.voices {
        let theta = (voice.pan + 1.0) * std::f32::consts::FRAC_PI_4;
        let (gl, gr) = (theta.cos(), theta.sin());
        for note in &voice.notes {
            for &ir in &chord_intervals {
                let freq = note.frequency * ir;
                let cv = if (ir - 1.0).abs() < 0.01 { 1.0 } else { 0.35 };
                render_tone(
                    &mut bl,
                    &mut br,
                    sr,
                    note,
                    freq,
                    cv * voice.volume,
                    gl,
                    gr,
                    &partials,
                    &adsr,
                    fm_depth,
                    fm_ratio,
                );
                if config.enable_sub_bass && (ir - 1.0).abs() < 0.01 {
                    let sf = freq * 0.5;
                    if sf >= 20.0 {
                        render_sub(
                            &mut bl,
                            &mut br,
                            sr,
                            note,
                            sf,
                            (1.0 - state.serotonin) * 0.4 * voice.volume,
                            gl,
                            gr,
                            &adsr,
                        );
                    }
                }
                if config.unison_detune > 0.0 && (ir - 1.0).abs() < 0.01 {
                    let d = config.unison_detune;
                    for &m in &[1.0 + d, 1.0 - d] {
                        render_tone(
                            &mut bl,
                            &mut br,
                            sr,
                            note,
                            freq * m,
                            cv * voice.volume * 0.3,
                            gl,
                            gr,
                            &partials,
                            &adsr,
                            fm_depth,
                            fm_ratio,
                        );
                    }
                }
            }
        }
    }
    if config.noise_mix > 0.0 {
        render_noise(
            &mut bl,
            &mut br,
            sr,
            total_samples,
            config.noise_mix,
            state.noradrenaline,
        );
    }
    let rw = 0.1 + state.consciousness_level * 0.3;
    let mut rv = Freeverb::new(
        sample_rate,
        config.reverb.room_size,
        config.reverb.damping,
        rw,
    );
    for i in 0..total_samples {
        let (l, r) = rv.process_stereo(bl[i], br[i]);
        bl[i] = soft_clip(l);
        br[i] = soft_clip(r);
    }
    if config.reverb.width < 1.0 {
        let w = config.reverb.width;
        for i in 0..total_samples {
            let (m, s) = ((bl[i] + br[i]) * 0.5, (bl[i] - br[i]) * 0.5);
            bl[i] = m + s * w;
            br[i] = m - s * w;
        }
    }
    match config.output_format {
        OutputFormat::StereoF32 => {
            AudioData::StereoF32(bl.iter().zip(br.iter()).map(|(&l, &r)| [l, r]).collect())
        }
        OutputFormat::MonoF32 => AudioData::F32(
            bl.iter()
                .zip(br.iter())
                .map(|(&l, &r)| (l + r) * 0.5)
                .collect(),
        ),
        OutputFormat::Mono16 => AudioData::I16(
            bl.iter()
                .zip(br.iter())
                .map(|(&l, &r)| (soft_clip((l + r) * 0.5) * i16::MAX as f32) as i16)
                .collect(),
        ),
    }
}

pub fn render_notes(
    notes: &[Note],
    sample_rate: u32,
    total_samples: usize,
    state: &MusicalState,
) -> AudioData {
    let config = MuseConfig {
        sample_rate,
        output_format: OutputFormat::Mono16,
        ..Default::default()
    };
    let arr = crate::voice::arrange(notes, state);
    render_arrangement(&arr, sample_rate, total_samples, state, &config)
}

// ─── Core ───────────────────────────────────────────────────────────────────
fn render_tone(
    bl: &mut [f32],
    br: &mut [f32],
    sr: f32,
    note: &Note,
    freq: f32,
    vol: f32,
    gl: f32,
    gr: f32,
    partials: &[f32],
    adsr: &Adsr,
    fm_depth: f32,
    fm_ratio: f32,
) {
    let start = (note.start_time * sr) as usize;
    let dur = (note.duration * sr) as usize;
    let rel = (adsr.release * sr) as usize;
    let mf = freq * fm_ratio;
    let ny = sr * 0.5;
    for i in 0..dur + rel {
        let si = start + i;
        if si >= bl.len() {
            break;
        }
        let t = i as f32 / sr;
        let env = envelope(adsr, t, note.duration);
        let fm = fm_depth * (std::f32::consts::TAU * mf * t).sin();
        let mut s = 0.0f32;
        for (h, &a) in partials.iter().enumerate() {
            let cf = freq * (h + 1) as f32;
            if cf >= ny {
                break;
            }
            s += a * (std::f32::consts::TAU * cf * t + fm).sin();
        }
        let o = s * env * note.velocity * vol * 0.15;
        bl[si] += o * gl;
        br[si] += o * gr;
    }
}

fn render_sub(
    bl: &mut [f32],
    br: &mut [f32],
    sr: f32,
    note: &Note,
    freq: f32,
    vol: f32,
    gl: f32,
    gr: f32,
    adsr: &Adsr,
) {
    let start = (note.start_time * sr) as usize;
    let dur = (note.duration * sr) as usize;
    let rel = (adsr.release * sr) as usize;
    for i in 0..dur + rel {
        let si = start + i;
        if si >= bl.len() {
            break;
        }
        let t = i as f32 / sr;
        let o = (std::f32::consts::TAU * freq * t).sin()
            * envelope(adsr, t, note.duration)
            * note.velocity
            * vol
            * 0.15;
        bl[si] += o * gl;
        br[si] += o * gr;
    }
}

fn render_noise(bl: &mut [f32], br: &mut [f32], sr: f32, n: usize, mix: f32, ne: f32) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let cut = 500.0 + ne * 3000.0;
    let a = (1.0 / sr) / (1.0 / (std::f32::consts::TAU * cut) + 1.0 / sr);
    let (mut pl, mut pr) = (0.0f32, 0.0f32);
    for i in 0..n {
        let (wl, wr) = (rng.gen::<f32>() * 2.0 - 1.0, rng.gen::<f32>() * 2.0 - 1.0);
        pl += a * (wl - pl);
        pr += a * (wr - pr);
        bl[i] += pl * mix;
        br[i] += pr * mix;
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────
pub(crate) fn compute_chord_intervals(state: &MusicalState) -> Vec<f32> {
    let mut v = vec![1.0];
    if state.harmony_activations[1] > 0.5 {
        v.push(1.25);
    }
    if state.harmony_activations[2] > 0.4 {
        v.push(1.5);
    }
    if state.harmony_activations[3] > 0.6 {
        v.push(1.8);
    }
    if state.harmony_activations[4] > 0.5 {
        v.push(1.333);
    }
    v.truncate(3);
    v
}

pub fn compute_timbre(state: &MusicalState, n: usize) -> Vec<f32> {
    let n = n.clamp(1, 16);
    let re = 1.0 + state.serotonin * 0.8;
    let br = 0.3 + state.dopamine * 0.7;
    (0..n)
        .map(|i| {
            if i == 0 {
                1.0
            } else {
                let h = (i + 1) as f32;
                br / h.powf(re) + state.noradrenaline * 0.05 / h
            }
        })
        .collect()
}

pub(crate) fn compute_adsr(state: &MusicalState) -> Adsr {
    Adsr {
        attack: 0.01 + (1.0 - state.arousal) * 0.05,
        decay: 0.05 + state.serotonin * 0.1,
        sustain: 0.4 + state.consciousness_level * 0.4,
        release: 0.1 + state.harmony_activations[7] * 0.3,
    }
}

pub(crate) fn envelope(adsr: &Adsr, t: f32, dur: f32) -> f32 {
    if t < adsr.attack {
        t / adsr.attack
    } else if t < adsr.attack + adsr.decay {
        1.0 - (t - adsr.attack) / adsr.decay * (1.0 - adsr.sustain)
    } else if t < dur {
        adsr.sustain
    } else {
        adsr.sustain * (1.0 - (t - dur) / adsr.release).max(0.0)
    }
}

pub(crate) fn soft_clip(x: f32) -> f32 {
    if x > 1.0 {
        1.0 - (-x + 1.0).exp() * 0.5
    } else if x < -1.0 {
        -1.0 + (x + 1.0).exp() * 0.5
    } else {
        x
    }
}
