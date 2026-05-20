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
    /// Update reverb parameters WITHOUT destroying the tail.
    pub(crate) fn set_params(&mut self, room_size: f32, damping: f32, wet: f32) {
        let fb = 0.28 + room_size.clamp(0.0, 1.0) * 0.7;
        let d1 = damping.clamp(0.0, 1.0);
        let d2 = 1.0 - d1;
        for c in self.comb_l.iter_mut().chain(self.comb_r.iter_mut()) {
            c.feedback = fb;
            c.damp1 = d1;
            c.damp2 = d2;
        }
        self.wet = wet.clamp(0.0, 1.0);
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
    // At high arousal, use FEWER partials to reduce intermodulation distortion
    // when many voices play simultaneously. User feedback: high-arousal + chord
    // accompaniment produced static (spectral flatness > 0.5). Fewer partials
    // → less harmonic density → less beating/noise.
    let partial_cap = if state.arousal > 0.6 {
        4 // minimal partials for clean high-energy sound
    } else if state.arousal > 0.4 {
        6
    } else {
        config.num_partials.clamp(1, 16)
    };
    let partials = compute_timbre(state, partial_cap);
    let adsr = compute_adsr(state);
    // FM synthesis: emotion-gated.
    // Tense: moderate FM (the original B-grade sound used moderate FM, not extreme).
    // Joyful: light FM for shimmer. Others: none.
    let fm_depth = if state.valence < -0.2 && state.arousal > 0.5 {
        // Tense: moderate FM — enough for edge without being metallic noise
        0.4 * config.max_fm_depth
    } else if state.valence > 0.2 && state.arousal > 0.5 {
        // Joyful: light FM shimmer
        0.1 * config.max_fm_depth
    } else {
        0.0
    };
    let fm_ratio = 2.0; // fixed ratio for all (original B-grade tense used 2.0)
    // Filter envelope parameters: emotion-dependent spectral movement
    let (filter_open, filter_close) = if state.valence > 0.2 && state.arousal > 0.5 {
        (12.0, 8.0) // Joyful: stay very bright
    } else if state.valence > 0.2 {
        (4.0, 2.0) // Contemplative: dark
    } else if state.arousal > 0.5 {
        (8.0, 5.0) // Tense: bright throughout (aggressive character)
    } else {
        (3.0, 1.5) // Sorrowful: always dark
    };
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
                    filter_open,
                    filter_close,
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
                            filter_open,
                            filter_close,
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

    // ── Loudness normalization + peak limiter ────────────────────────
    // Target: RMS ≈ -18 dB (0.126), ceiling at -1 dB (0.891).
    // This ensures consistent loudness across emotional states and
    // prevents clipping regardless of harmonic density.
    {
        const TARGET_RMS: f32 = 0.126; // -18 dB
        const CEILING: f32 = 0.891; // -1 dB

        // Measure current RMS (stereo)
        let rms = {
            let sum: f32 = bl.iter().zip(br.iter()).map(|(&l, &r)| l * l + r * r).sum();
            (sum / (2.0 * total_samples as f32)).sqrt()
        };

        // Apply gain to reach target RMS (only if signal is audible)
        if rms > 0.0001 {
            let gain = (TARGET_RMS / rms).min(4.0); // cap boost at +12 dB
            for i in 0..total_samples {
                bl[i] *= gain;
                br[i] *= gain;
            }
        }

        // Peak limiter: soft-knee at ceiling
        for i in 0..total_samples {
            bl[i] = peak_limit(bl[i], CEILING);
            br[i] = peak_limit(br[i], CEILING);
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
    filter_open: f32,
    filter_close: f32,
) {
    let start = (note.start_time * sr) as usize;
    let dur = (note.duration * sr) as usize;
    let rel = (adsr.release * sr) as usize;
    let mf = freq * fm_ratio;
    let ny = sr * 0.5;

    // Filter envelope: cutoff sweeps from filter_open to filter_close over the note.
    // Values are passed as parameters (computed per-emotion in render_arrangement).

    for i in 0..dur + rel {
        let si = start + i;
        if si >= bl.len() {
            break;
        }
        let t = i as f32 / sr;
        let env = envelope(adsr, t, note.duration);

        // Filter cutoff follows envelope: bright at attack, darker at sustain
        let filter_ratio = filter_close + (filter_open - filter_close) * env;
        let cutoff = freq * filter_ratio;

        let fm = fm_depth * (std::f32::consts::TAU * mf * t).sin();
        let mut s = 0.0f32;
        for (h, &a) in partials.iter().enumerate() {
            let cf = freq * (h + 1) as f32;
            if cf >= ny {
                break;
            }
            // Apply spectral rolloff above cutoff (6dB/oct lowpass approximation)
            let filter_atten = if cf > cutoff {
                (cutoff / cf).min(1.0)
            } else {
                1.0
            };
            // PER-PARTIAL ENVELOPE: upper partials decay faster than fundamentals.
            // Real instruments (piano, strings, wind) have this physics: higher
            // harmonics dissipate energy faster due to string stiffness, air
            // absorption, body filtering. The fundamental rings; harmonics fade.
            //
            // Decay rate scales with partial index: partial N decays ~N× faster
            // than fundamental. At t=1s, partial 8 is at 0.37× its initial
            // amplitude while fundamental is still near 1.0.
            let partial_decay = (-t * 0.8 * (h as f32).sqrt()).exp();
            s += a * filter_atten * partial_decay * (std::f32::consts::TAU * cf * t + fm).sin();
        }
        let o = s * env * note.velocity * vol * 0.2;
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
        let (wl, wr) = (
            rng.r#gen::<f32>() * 2.0 - 1.0,
            rng.r#gen::<f32>() * 2.0 - 1.0,
        );
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

/// Compute harmonic partial amplitudes based on emotional state.
///
/// Maps (valence, arousal) to recognizable timbres:
/// - High valence + high arousal → bright saw-like pad (rich harmonics, 1/h rolloff)
/// - High valence + low arousal → soft bell/pad (odd harmonics, fast rolloff)
/// - Low valence + high arousal → harsh lead (all harmonics + FM emphasis)
/// - Low valence + low arousal → warm filtered tone (few harmonics, 1/h² rolloff)
///
/// These map to timbres CLAP recognizes as "music" rather than "electronic noise."
pub fn compute_timbre(state: &MusicalState, n: usize) -> Vec<f32> {
    let n = n.clamp(1, 16);
    let v = state.valence;
    let a = state.arousal;

    (0..n)
        .map(|i| {
            if i == 0 {
                1.0 // fundamental always present
            } else {
                let h = (i + 1) as f32;
                let harmonic_idx = i + 1;

                if v > 0.2 && a > 0.5 {
                    // JOYFUL: Very bright, pluck-like spectrum (close to square wave).
                    // Strong odd harmonics for "plucky" character that CLAP recognizes
                    // as upbeat electronic. Think: marimba, kalimba, bright arp.
                    let odd_boost = if harmonic_idx % 2 == 1 { 1.0 } else { 0.4 };
                    odd_boost * 0.8 / h.powf(0.7) // slow rolloff = very bright
                } else if v > 0.2 && a <= 0.5 {
                    // CONTEMPLATIVE: Pure, minimal harmonics. Almost sine-like.
                    // CLAP "ambient meditation" expects simple, clean tones.
                    // Fundamental dominates; slight 5th (3rd harmonic) for warmth.
                    if harmonic_idx == 3 {
                        0.3
                    }
                    // perfect 5th overtone
                    else if harmonic_idx == 5 {
                        0.1
                    }
                    // subtle octave+3rd
                    else {
                        0.05 / h.powf(2.5)
                    } // everything else nearly silent
                } else if v < -0.2 && a > 0.5 {
                    // TENSE: Saw-like spectrum — dense harmonics, moderate rolloff.
                    // CLAP "dark dramatic electronic" matches standard sawtooth synth.
                    0.7 / h // classic sawtooth: 1/h amplitude ratio
                } else {
                    // SORROWFUL: Warm sine-dominant, barely any overtones.
                    // CLAP "sad melancholic" expects almost pure tone, dark, intimate.
                    0.3 / h.powf(2.8) // very fast rolloff = nearly sine wave
                }
            }
        })
        .collect()
}

/// Compute ADSR envelope based on emotional state.
///
/// Emotion-appropriate envelopes make the difference between "beep" and "music":
/// - Pads need slow attack (0.05-0.3s), long sustain, long release
/// - Plucks need fast attack, short sustain
/// - Leads need fast attack but long release
pub(crate) fn compute_adsr(state: &MusicalState) -> Adsr {
    let v = state.valence;
    let a = state.arousal;

    if v > 0.2 && a > 0.5 {
        // JOYFUL: Pluck/arpeggio character — very fast attack, short decay.
        // CLAP expects bright transients for "upbeat electronic music."
        // Think: arpeggiator, plucked synth, marimba-like.
        Adsr {
            attack: 0.002, // near-instant (pluck)
            decay: 0.15 + (1.0 - a) * 0.1,
            sustain: 0.25, // low sustain = notes have clear end
            release: 0.08, // short release = rhythmic clarity
        }
    } else if v > 0.2 && a <= 0.5 {
        // CONTEMPLATIVE: Drone/pad — extremely slow, barely any transient.
        // CLAP expects "ambient meditation" = evolving texture, no rhythm.
        Adsr {
            attack: 0.3 + (1.0 - a) * 0.5, // 0.3-0.8s (glacial)
            decay: 0.2,
            sustain: 0.85, // high sustain = continuous drone
            release: 1.0 + state.harmony_activations[7] * 1.0, // 1-2s tail
        }
    } else if v < -0.2 && a > 0.5 {
        // TENSE: Aggressive stab — fast attack, moderate sustain, punchy.
        // CLAP expects "dark dramatic" = hard transients, bass-heavy.
        Adsr {
            attack: 0.001, // instant
            decay: 0.03,
            sustain: 0.6 + state.noradrenaline * 0.3,
            release: 0.1,
        }
    } else {
        // SORROWFUL: Slow swell — very long attack, endless release.
        // CLAP expects "sad melancholic" = barely audible onsets, long decay.
        Adsr {
            attack: 0.4 + (1.0 - a) * 0.4, // 0.4-0.8s
            decay: 0.3,
            sustain: 0.5,
            release: 1.5 + state.harmony_activations[7] * 1.0, // 1.5-2.5s
        }
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

/// Peak limiter with soft knee at the ceiling.
/// Below 0.8×ceiling: pass through. Above: smooth compression to ceiling.
#[inline]
fn peak_limit(x: f32, ceiling: f32) -> f32 {
    let knee = ceiling * 0.8;
    let ax = x.abs();
    if ax <= knee {
        x
    } else {
        // Smooth compression: maps [knee, ∞) → [knee, ceiling)
        let over = ax - knee;
        let range = ceiling - knee;
        let compressed = knee + range * (1.0 - (-over / range).exp());
        compressed.copysign(x)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voice::{Arrangement, Voice, VoiceRole};
    use crate::{MuseConfig, MusicalState, Note, OutputFormat};

    fn one_note_arrangement(freq: f32) -> Arrangement {
        let note = Note {
            frequency: freq,
            start_time: 0.0,
            duration: 0.5,
            velocity: 0.8,
        };
        Arrangement {
            voices: vec![Voice {
                role: VoiceRole::Lead,
                notes: vec![note],
                pitch_range: (130.0, 1000.0),
                volume: 1.0,
                pan: 0.0,
            }],
        }
    }

    // ─── soft_clip ────────────────────────────────────────────────────────────

    #[test]
    fn soft_clip_identity_in_range() {
        for x in [-0.9f32, -0.5, 0.0, 0.5, 0.9] {
            assert!(
                (soft_clip(x) - x).abs() < 1e-6,
                "soft_clip({x}) should be identity"
            );
        }
    }

    #[test]
    fn soft_clip_bounded() {
        for x in [-10.0f32, -2.0, 2.0, 10.0] {
            let y = soft_clip(x);
            assert!(y > -1.5 && y < 1.5, "soft_clip({x}) = {y} exceeds bounds");
        }
    }

    #[test]
    fn soft_clip_monotone_in_linear_region() {
        // soft_clip is linear (identity) in [-1, 1] — verify monotone there
        let mut prev = soft_clip(-1.0);
        for i in 1..=20 {
            let x = -1.0 + i as f32 * 0.1;
            let y = soft_clip(x);
            assert!(
                y >= prev - 1e-6,
                "soft_clip not monotone at {x}: {y} < {prev}"
            );
            prev = y;
        }
    }

    // ─── ADSR envelope ────────────────────────────────────────────────────────

    #[test]
    fn adsr_from_default_state() {
        let state = MusicalState::default();
        let adsr = compute_adsr(&state);
        assert!(adsr.attack > 0.0);
        assert!(adsr.sustain > 0.0 && adsr.sustain <= 1.0);
        assert!(adsr.release > 0.0);
    }

    #[test]
    fn envelope_attack_phase() {
        let state = MusicalState::default();
        let adsr = compute_adsr(&state);
        let y = envelope(&adsr, adsr.attack * 0.5, 1.0);
        assert!(
            y > 0.0 && y < 1.0,
            "mid-attack should be between 0 and 1: {y}"
        );
    }

    #[test]
    fn envelope_sustain_plateau() {
        let state = MusicalState::default();
        let adsr = compute_adsr(&state);
        let t_sustain = adsr.attack + adsr.decay + 0.01;
        let y = envelope(&adsr, t_sustain, t_sustain + 0.1);
        assert!(
            (y - adsr.sustain).abs() < 0.05,
            "sustain phase should be near sustain level {}: got {y}",
            adsr.sustain
        );
    }

    #[test]
    fn high_arousal_faster_attack() {
        let calm = MusicalState {
            arousal: 0.1,
            ..MusicalState::default()
        };
        let excited = MusicalState {
            arousal: 0.9,
            ..MusicalState::default()
        };
        let adsr_calm = compute_adsr(&calm);
        let adsr_excited = compute_adsr(&excited);
        assert!(
            adsr_excited.attack < adsr_calm.attack,
            "high arousal should have faster attack: {} < {}",
            adsr_excited.attack,
            adsr_calm.attack
        );
    }

    // ─── render_arrangement ───────────────────────────────────────────────────

    #[test]
    fn render_produces_samples() {
        let arr = one_note_arrangement(440.0);
        let config = MuseConfig {
            duration_secs: 1.0,
            ..Default::default()
        };
        let state = MusicalState::default();
        let audio = render_arrangement(&arr, 44100, 44100, &state, &config);
        assert!(!audio.is_empty());
    }

    #[test]
    fn render_stereo_format() {
        let arr = one_note_arrangement(261.63);
        let config = MuseConfig {
            duration_secs: 0.5,
            output_format: OutputFormat::StereoF32,
            ..Default::default()
        };
        let state = MusicalState::default();
        let audio = render_arrangement(&arr, 44100, 22050, &state, &config);
        assert!(
            matches!(audio, AudioData::StereoF32(_)),
            "expected StereoF32"
        );
    }

    #[test]
    fn render_mono16_format() {
        let arr = one_note_arrangement(330.0);
        let config = MuseConfig {
            duration_secs: 0.5,
            output_format: OutputFormat::Mono16,
            ..Default::default()
        };
        let state = MusicalState::default();
        let audio = render_arrangement(&arr, 44100, 22050, &state, &config);
        assert!(matches!(audio, AudioData::I16(_)), "expected Mono16 (I16)");
    }

    #[test]
    fn render_no_nan_inf() {
        let arr = one_note_arrangement(440.0);
        let config = MuseConfig {
            duration_secs: 1.0,
            output_format: OutputFormat::StereoF32,
            num_partials: 8,
            max_fm_depth: 3.0,
            ..Default::default()
        };
        let state = MusicalState {
            harmony_activations: [0.8; 8],
            consciousness_level: 0.9,
            ..MusicalState::default()
        };
        let audio = render_arrangement(&arr, 44100, 44100, &state, &config);
        if let AudioData::StereoF32(samples) = audio {
            for (i, s) in samples.iter().enumerate() {
                assert!(s[0].is_finite(), "left NaN/Inf at sample {i}: {}", s[0]);
                assert!(s[1].is_finite(), "right NaN/Inf at sample {i}: {}", s[1]);
            }
        }
    }

    #[test]
    fn render_horror_config_no_panic() {
        let arr = one_note_arrangement(110.0);
        let config = MuseConfig {
            duration_secs: 0.5,
            ..MuseConfig::horror()
        };
        let state = MusicalState::default();
        let audio = render_arrangement(&arr, 44100, 22050, &state, &config);
        assert!(!audio.is_empty());
    }

    #[test]
    fn render_fm_depth_affects_output() {
        let arr = one_note_arrangement(440.0);
        let state = MusicalState::default();
        let samples = 22050;

        let clean = MuseConfig {
            duration_secs: 0.5,
            output_format: OutputFormat::MonoF32,
            max_fm_depth: 0.0,
            ..Default::default()
        };
        let fm = MuseConfig {
            duration_secs: 0.5,
            output_format: OutputFormat::MonoF32,
            max_fm_depth: 5.0,
            ..Default::default()
        };

        let clean_audio = render_arrangement(&arr, 44100, samples, &state, &clean);
        let fm_audio = render_arrangement(&arr, 44100, samples, &state, &fm);

        // FM-modulated output should differ from clean
        if let (AudioData::F32(c), AudioData::F32(f)) = (clean_audio, fm_audio) {
            let diff: f32 = c.iter().zip(f.iter()).map(|(a, b)| (a - b).abs()).sum();
            assert!(
                diff > 0.1,
                "FM depth should change the output (diff = {diff})"
            );
        }
    }

    #[test]
    fn render_partials_affect_timbre() {
        let arr = one_note_arrangement(220.0);
        let state = MusicalState::default();
        let n = 22050;

        let thin = MuseConfig {
            duration_secs: 0.5,
            output_format: OutputFormat::MonoF32,
            num_partials: 1,
            ..Default::default()
        };
        let rich = MuseConfig {
            duration_secs: 0.5,
            output_format: OutputFormat::MonoF32,
            num_partials: 16,
            ..Default::default()
        };

        let thin_audio = render_arrangement(&arr, 44100, n, &state, &thin);
        let rich_audio = render_arrangement(&arr, 44100, n, &state, &rich);

        if let (AudioData::F32(t), AudioData::F32(r)) = (thin_audio, rich_audio) {
            // Rich should have higher RMS (more partials = more energy)
            let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
            assert!(
                rms(&r) > rms(&t),
                "rich ({} partials) should have higher RMS than thin (1 partial)",
                rich.num_partials
            );
        }
    }
}
