// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hi-Fi hybrid additive + FM synthesis with Freeverb and stereo rendering.
//!
//! Features: configurable partials (1-16), Freeverb (8 comb + 4 allpass),
//! sub-bass, detuned unison, filtered noise, per-voice stereo panning.

use crate::instruments::Instrument;
use crate::voice::{Arrangement, VoiceRole};
use crate::{AudioData, MuseConfig, MusicalState, Note, OutputFormat};

#[derive(Clone, Copy)]
pub(crate) struct Adsr {
    pub(crate) attack: f32,
    pub(crate) decay: f32,
    pub(crate) sustain: f32,
    pub(crate) release: f32,
}

impl From<(f32, f32, f32, f32)> for Adsr {
    fn from((attack, decay, sustain, release): (f32, f32, f32, f32)) -> Self {
        Adsr {
            attack,
            decay,
            sustain,
            release,
        }
    }
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
    /// Current wet gain, smoothed per sample toward `target_wet`. `set_params`
    /// must never step wet, feedback, or damping directly: wet multiplies the
    /// whole tail, and a feedback step (e.g. consciousness 0.1→0.9 retunes
    /// fb 0.41→0.85) writes a ~2× jump into the recirculating comb buffers —
    /// both are audible zipper clicks. All three glide here per sample.
    pub(crate) wet: f32,
    target_wet: f32,
    feedback: f32,
    target_feedback: f32,
    damping: f32,
    target_damping: f32,
}
struct CombFilter {
    buffer: Vec<f32>,
    index: usize,
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
            target_wet: wet,
            feedback: fb,
            target_feedback: fb,
            damping,
            target_damping: damping,
        }
    }
    /// Update reverb parameters WITHOUT destroying the tail. Targets only —
    /// the live values glide per sample in `process_stereo` (see field doc).
    pub(crate) fn set_params(&mut self, room_size: f32, damping: f32, wet: f32) {
        self.target_feedback = 0.28 + room_size.clamp(0.0, 1.0) * 0.7;
        self.target_damping = damping.clamp(0.0, 1.0);
        self.target_wet = wet.clamp(0.0, 1.0);
    }
    pub(crate) fn process_stereo(&mut self, il: f32, ir: f32) -> (f32, f32) {
        // Glide all parameters toward their targets (~23ms time constant at
        // 44.1kHz) — see the field doc: stepping any of them clicks.
        self.wet += (self.target_wet - self.wet) * 0.001;
        self.feedback += (self.target_feedback - self.feedback) * 0.001;
        self.damping += (self.target_damping - self.damping) * 0.001;
        let (fb, d1) = (self.feedback, self.damping);
        let d2 = 1.0 - d1;
        let (mut ol, mut or) = (0.0f32, 0.0f32);
        for c in &mut self.comb_l {
            ol += cp(c, il, fb, d1, d2);
        }
        for c in &mut self.comb_r {
            or += cp(c, ir, fb, d1, d2);
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
/// Flush subnormal floats to zero.
///
/// Every feedback/recursive filter in the crate must pass its state through
/// this: as a reverb tail or filter state decays toward silence it enters the
/// subnormal range, where x86 float ops run 10-100× slower — a classic
/// real-time audio CPU spike. Threshold 1e-20 is ~120 dB below the smallest
/// audible sample, far outside anything musically meaningful.
#[inline(always)]
pub(crate) fn flush_denormal(x: f32) -> f32 {
    if x.abs() < 1e-20 { 0.0 } else { x }
}

fn cp(c: &mut CombFilter, input: f32, feedback: f32, damp1: f32, damp2: f32) -> f32 {
    let o = c.buffer[c.index];
    c.filterstore = flush_denormal(o * damp2 + c.filterstore * damp1);
    c.buffer[c.index] = flush_denormal(input + c.filterstore * feedback);
    c.index = (c.index + 1) % c.buffer.len();
    o
}
fn ap(a: &mut AllpassFilter, input: f32) -> f32 {
    let b = a.buffer[a.index];
    let o = b - input;
    a.buffer[a.index] = flush_denormal(input + b * 0.5);
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

        // A voice with a real `instrument` set gets that instrument's OWN
        // acoustic partials/envelope (or its physical model — Karplus-Strong
        // / FM — entirely bypassing the additive path below) instead of the
        // single mood-derived timbre every voice used to share. This is the
        // fix for "every voice sounds like the same chiptune patch in a
        // different register": melody/harmony/bass can now genuinely be a
        // violin, a piano, and a cello. See `voice::Voice::instrument`.
        let additive_instrument = voice
            .instrument
            .filter(|i| !i.uses_karplus_strong() && !i.uses_fm());
        let local_adsr: Adsr = additive_instrument
            .map(|i| Adsr::from(i.default_adsr()))
            .unwrap_or(adsr);

        for (ni, note) in voice.notes.iter().enumerate() {
            // Legato: when the previous note in this voice ends where this
            // one begins (within 35ms) AND the motion is stepwise, the
            // recorded ATTACK of the incoming sample is the enemy — a fresh
            // bow/tongue transient inside a slur is the "MIDI preview"
            // sound. The gate is deliberately narrow (first version slurred
            // EVERYTHING adjacent, which turned the walking bass into
            // attack-less mush — "the strings are not played properly"):
            // - stepwise only (≤3.5 semitones): a leap takes a fresh bow;
            // - never the bass voice: a walking bass re-articulates
            //   every note, that IS the walk;
            // - only bowed/blown instruments: plucked/struck timbres are
            //   their attack.
            #[cfg(not(target_arch = "wasm32"))]
            let legato_from_prev = ni > 0
                && voice.role != VoiceRole::Bass
                && voice
                    .instrument
                    .map(instrument_benefits_from_legato)
                    .unwrap_or(false)
                && {
                    let prev = &voice.notes[ni - 1];
                    let gap = note.start_time - (prev.start_time + prev.duration);
                    let semis = (12.0 * (note.frequency / prev.frequency).log2()).abs();
                    (-0.035..=0.035).contains(&gap) && semis <= 3.5
                };
            // Long sustained bowed/blown notes get a gentle messa di voce
            // (swell toward ~40%, relax after) — a flat-gain 3-second note
            // is a note PARKED, not played. The held-arrival cadence tones
            // and the held climax are exactly these notes.
            #[cfg(not(target_arch = "wasm32"))]
            let sustained_shape = note.duration >= 1.2
                && voice
                    .instrument
                    .map(instrument_benefits_from_legato)
                    .unwrap_or(false);
            if let Some(instrument) = voice.instrument {
                // Sampled instruments first: when the VCSL library is active
                // (SYMTHAEA_VCSL_DIR or vcsl::init) and has a bank for this
                // instrument, the note plays from a REAL recording. Inactive
                // or unmapped → the synthesis paths below, unchanged.
                #[cfg(not(target_arch = "wasm32"))]
                if let Some(lib) = crate::vcsl::library() {
                    // Articulation-aware: notes short enough to be PLAYED
                    // short use the real staccato/spiccato bank when the
                    // library ships one — a truncated sustain never sounds
                    // like an actual short bow. Falls through to the
                    // sustain bank, then to synthesis.
                    let mut done = false;
                    // 130ms, not 250: the MAESTRO articulation model can
                    // shorten ordinary detached eighths below 250ms at
                    // brisk tempi, and merely-detached notes must not flip
                    // to spiccato -- that articulation is for notes PLAYED
                    // short (true staccato, grace notes). Defensive guard:
                    // at slow tempi nothing lands in the 130-250ms band.
                    if note.duration < 0.13
                        && let Some(stac) = lib.staccato_bank(instrument)
                    {
                        // A staccato note IS its attack — never legato.
                        done = render_vcsl_note(
                            &mut bl,
                            &mut br,
                            sr,
                            note,
                            voice.volume,
                            gl,
                            gr,
                            stac,
                            false,
                            false,
                        );
                    }
                    if !done && let Some(bank) = lib.bank(instrument) {
                        done = render_vcsl_note(
                            &mut bl,
                            &mut br,
                            sr,
                            note,
                            voice.volume,
                            gl,
                            gr,
                            bank,
                            legato_from_prev,
                            sustained_shape,
                        );
                    }
                    if done {
                        continue;
                    }
                }
                if instrument.uses_karplus_strong() {
                    render_karplus_note(
                        &mut bl,
                        &mut br,
                        sr,
                        note,
                        voice.volume,
                        gl,
                        gr,
                        instrument,
                    );
                    continue;
                }
                if instrument.uses_fm() {
                    render_fm_note(
                        &mut bl,
                        &mut br,
                        sample_rate,
                        note,
                        voice.volume,
                        gl,
                        gr,
                        instrument,
                    );
                    continue;
                }
            }
            // Per-NOTE timbre: velocity shapes the spectrum (ff is brighter,
            // not just louder), piano gets stretched partials, sustained
            // instruments keep their brightness, and the instrument's attack
            // transient (bow bite / chiff / hammer) speaks at the onset.
            // Voices with no instrument keep the legacy mood-derived timbre.
            let timbre = match additive_instrument {
                Some(i) => NoteTimbre::for_instrument(i, note),
                None => NoteTimbre::legacy(&partials),
            };
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
                    &timbre,
                    &local_adsr,
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
                            &local_adsr,
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
                            &timbre,
                            &local_adsr,
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
    let rw = config.reverb.wet_floor + state.consciousness_level * 0.3;
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

/// Deterministic per-note variation seed: two notes at different times (or
/// different pitches/lengths) get different seeds, while re-rendering the
/// same piece reproduces identical audio. This is the round-robin identity —
/// before it existed, repeated notes rendered bit-identical audio (additive
/// partials all phase-locked at 0; every Karplus-Strong pluck used one
/// hardcoded RNG seed), which reads as "machine gun" to the ear.
pub(crate) fn note_seed(note: &Note) -> u32 {
    note.start_time.to_bits()
        ^ note.frequency.to_bits().rotate_left(13)
        ^ note.duration.to_bits().rotate_left(27)
}

/// Per-note timbre for the additive path. Before this existed, a voice had
/// ONE static spectrum for the whole piece: velocity was a pure amplitude
/// multiplier ("organ mode" — a ff note was just a louder pp note), piano
/// partials were perfectly harmonic, sustained bowed notes dulled over time
/// as if they'd been struck, and no instrument had an onset transient.
pub(crate) struct NoteTimbre {
    /// Velocity-shaped partial amplitudes (see
    /// [`Instrument::partials_at_velocity`]).
    partials: Vec<f32>,
    /// Stiff-string inharmonicity coefficient B (0 = perfectly harmonic).
    inharmonicity: f32,
    /// Per-partial decay speed: ~0.8 for struck/plucked (ringing notes dull),
    /// much lower for continuously-energized instruments (bowed/blown notes
    /// keep their brightness while played).
    decay_rate: f32,
    /// Attack transient noise (bow bite / breath chiff / hammer click):
    /// amplitude and duration in seconds. 0.0 disables.
    attack_amp: f32,
    attack_dur: f32,
    /// Vibrato `(rate_hz, depth_ratio, onset_delay_s)` — see
    /// [`Instrument::vibrato`]. `None` = stationary pitch (the pre-vibrato
    /// behavior, and the honest setting for organ/struck/plucked timbres).
    vibrato: Option<(f32, f32, f32)>,
}

impl NoteTimbre {
    /// Timbre for one note of a known instrument.
    fn for_instrument(instrument: Instrument, note: &Note) -> Self {
        let (attack_amp, attack_dur) = instrument.attack_noise();
        NoteTimbre {
            partials: instrument.partials_at_velocity(note.velocity).to_vec(),
            inharmonicity: instrument.inharmonicity(note.frequency),
            decay_rate: if instrument.sustains() { 0.12 } else { 0.8 },
            attack_amp,
            attack_dur,
            vibrato: instrument.vibrato(),
        }
    }

    /// Legacy mood-derived timbre (voices with no instrument assigned):
    /// preserves the pre-existing behavior exactly — harmonic partials,
    /// struck-style decay, no transient.
    fn legacy(partials: &[f32]) -> Self {
        NoteTimbre {
            partials: partials.to_vec(),
            inharmonicity: 0.0,
            decay_rate: 0.8,
            attack_amp: 0.0,
            attack_dur: 0.0,
            vibrato: None,
        }
    }
}

/// Per-partial initial phases for one note, derived from its seed. Real
/// instruments never start every harmonic at phase 0 twice in a row; the
/// phase set changes the attack's crest pattern, so repeated notes stop
/// being sample-identical while each render stays deterministic.
fn partial_phases(seed: u32) -> [f32; 16] {
    let mut phases = [0.0f32; 16];
    for (h, p) in phases.iter_mut().enumerate() {
        let x = seed
            .wrapping_mul(2654435761)
            .wrapping_add((h as u32).wrapping_mul(0x9E37_79B9));
        *p = (x >> 8) as f32 / 16777216.0 * std::f32::consts::TAU;
    }
    phases
}

fn render_tone(
    bl: &mut [f32],
    br: &mut [f32],
    sr: f32,
    note: &Note,
    freq: f32,
    vol: f32,
    gl: f32,
    gr: f32,
    timbre: &NoteTimbre,
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
    // Round-robin: this note's own partial start-phases (see partial_phases).
    // Also fold the actual rendered freq in, so chord-interval copies of the
    // same note don't share a phase set.
    let seed = note_seed(note) ^ freq.to_bits();
    let phases = partial_phases(seed);

    // Partial frequencies, hoisted out of the sample loop: stiff-string
    // inharmonicity stretches partial h to f·(h+1)·√(1+B(h+1)²) — for piano
    // this is the Conklin-measured B, for everything else B=0 gives the
    // plain harmonic series.
    let b = timbre.inharmonicity;
    let mut cfs = [0.0f32; 16];
    for (h, cf) in cfs.iter_mut().enumerate().take(timbre.partials.len()) {
        let n = (h + 1) as f32;
        let stretch = if b > 0.0 {
            (1.0 + b * n * n).sqrt()
        } else {
            1.0
        };
        *cf = freq * n * stretch;
    }

    // Attack transient noise state (bow bite / breath chiff / hammer click).
    // Runs OUTSIDE the amplitude envelope: the whole point of a transient is
    // that it speaks before the tone swells.
    let mut noise_rng = seed ^ 0x5EED_A77A;
    let attack_samples = (timbre.attack_dur * sr) as usize;

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
        // Vibrato: true frequency modulation via the phase integral —
        // f(t) = cf·(1 + d·sin(2πf_v t)) integrates to a phase term
        // −(cf·d/f_v)·cos(2πf_v t). The depth ramps in over 0.3s after the
        // instrument's onset delay (players don't vibrate the attack), and
        // the (cf-proportional) term is applied per partial so the whole
        // harmonic stack breathes coherently instead of shimmering apart.
        let vib_factor = match timbre.vibrato {
            Some((rate, depth, delay)) => {
                let ramp = ((t - delay) / 0.3).clamp(0.0, 1.0);
                if ramp > 0.0 {
                    -(depth * ramp / rate) * (std::f32::consts::TAU * rate * t).cos()
                } else {
                    0.0
                }
            }
            None => 0.0,
        };
        let mut s = 0.0f32;
        for (h, &a) in timbre.partials.iter().enumerate() {
            let cf = cfs[h];
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
            // Rate comes from the timbre: struck/plucked notes dull as they
            // ring; continuously-energized (bowed/blown) notes keep their
            // brightness while played.
            let partial_decay = (-t * timbre.decay_rate * (h as f32).sqrt()).exp();
            let phase = phases[h.min(phases.len() - 1)];
            s += a
                * filter_atten
                * partial_decay
                * (std::f32::consts::TAU * cf * t + cf * vib_factor + fm + phase).sin();
        }
        let mut o = s * env * note.velocity * vol * 0.2;
        if i < attack_samples {
            noise_rng = noise_rng.wrapping_mul(1103515245).wrapping_add(12345);
            let n = (noise_rng >> 8) as f32 / 8388608.0 - 1.0; // [-1, 1)
            let ramp = 1.0 - i as f32 / attack_samples as f32;
            o += timbre.attack_amp * n * ramp * note.velocity * vol * 0.2;
        }
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

/// Render one note through Karplus-Strong plucked-string synthesis (see
/// [`crate::instruments::KarplusStrong`]) instead of the additive engine.
/// The string is excited once and left to ring past the note's nominal
/// duration — a real pluck decays on its own, it doesn't have an ADSR
/// release phase bolted onto a sustained tone.
// Matches the existing arg-count shape of render_tone/render_sub in this
// same file (buffer pair + sample rate + note + gain/pan + synthesis
// params) -- consistent with the established local convention rather than
// introducing a bespoke params struct just for this one function.
#[allow(clippy::too_many_arguments)]
fn render_karplus_note(
    bl: &mut [f32],
    br: &mut [f32],
    sr: f32,
    note: &Note,
    vol: f32,
    gl: f32,
    gr: f32,
    instrument: Instrument,
) {
    let (damping, brightness, _stiffness) = instrument.ks_params();
    // Velocity → brightness: a hard pluck excites more high-frequency string
    // modes than a gentle one. The loop filter `b·x[n] + (1−b)·x[n+1]` damps
    // highs by |2b−1| per pass — DULLEST at b = 0.5, brighter on EITHER side
    // (a naive `b × velocity_factor` scaling can therefore darken a hard
    // pluck; a test caught exactly that). So soft notes pull b toward 0.5
    // (more averaging) and full velocity restores the instrument's own value,
    // keeping v=1.0 renders identical to the instrument's designed character.
    let brightness = 0.5 + (brightness - 0.5) * (0.4 + 0.6 * note.velocity.clamp(0.0, 1.0));
    let mut ks =
        crate::instruments::KarplusStrong::new(note.frequency, sr as u32, damping, brightness);
    // Round-robin: a per-note excitation seed, so repeated notes and chord
    // strums stop sounding like the same pluck sample retriggered.
    ks.excite_seeded(note.velocity, note_seed(note));
    let start = (note.start_time * sr) as usize;
    // Let the string ring out well past its nominal duration -- a pluck's
    // decay IS its release, there's no separate ADSR release to add.
    let ring = (note.duration * sr) as usize + (0.8 * sr) as usize;
    for i in 0..ring {
        let si = start + i;
        if si >= bl.len() {
            break;
        }
        let s = ks.tick() * vol * 0.6;
        bl[si] += s * gl;
        br[si] += s * gr;
    }
}

/// 4-point Catmull-Rom (Hermite) interpolation — the standard sampler
/// interpolator. Linear interpolation leaves high-frequency imaging
/// artifacts that read as HARSHNESS, worst on upward repitches; cubic
/// reduces them by orders of magnitude for a few extra multiplies.
/// Edge points clamp to the boundary sample.
#[cfg(not(target_arch = "wasm32"))]
fn hermite_interpolate(frames: &[f32], idx: usize, frac: f32) -> f32 {
    let xm1 = frames[idx.saturating_sub(1)];
    let x0 = frames[idx];
    let x1 = frames[(idx + 1).min(frames.len() - 1)];
    let x2 = frames[(idx + 2).min(frames.len() - 1)];
    let c = (x1 - xm1) * 0.5;
    let v = x0 - x1;
    let w = c + v;
    let a = w + v + (x2 - x0) * 0.5;
    let b = w + a;
    ((a * frac - b) * frac + c) * frac + x0
}

/// The bowed/blown sustain instruments whose recorded ATTACK transient
/// should be skipped on legato continuations. Plucked/struck timbres
/// (harp, piano, guitar, mallets) are their attack — never skip those.
#[cfg(not(target_arch = "wasm32"))]
fn instrument_benefits_from_legato(instrument: Instrument) -> bool {
    matches!(
        instrument,
        Instrument::Violin
            | Instrument::Cello
            | Instrument::Flute
            | Instrument::Clarinet
            | Instrument::Trumpet
    )
}

/// Render one note from a VCSL recorded sample (see [`crate::vcsl`]):
/// nearest-pitch sample repitched by linear-interpolation resampling, dynamic
/// layer chosen by note velocity, round-robin by the note's seed. Returns
/// `false` when the bank has nothing within a musical fifth of the target —
/// the caller then falls through to synthesis. The recording carries its own
/// envelope; we only add a 3 ms declick fade-in and a release fade after the
/// written duration.
///
/// `legato_from_prev`: this note continues a slur — start playback past
/// the worst of the recording's attack transient (~40ms in, bounded by a
/// quarter of the sample) with a 15ms fade-in. The previous note's 250ms
/// release tail is still sounding underneath, so the two recordings
/// crossfade into one gesture instead of re-attacking — the fix for the
/// per-note "MIDI preview" sound on slurred lines. The CALLER gates this
/// musically (stepwise motion only, never the bass, bowed/blown only).
///
/// `sustained_shape`: a long held note gets a gentle messa di voce swell
/// instead of parking at constant gain for seconds.
#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)] // matches render_tone/render_karplus_note convention
fn render_vcsl_note(
    bl: &mut [f32],
    br: &mut [f32],
    sr: f32,
    note: &Note,
    vol: f32,
    gl: f32,
    gr: f32,
    bank: &crate::vcsl::InstrumentBank,
    legato_from_prev: bool,
    sustained_shape: bool,
) -> bool {
    let target_midi = 69.0 + 12.0 * (note.frequency / 440.0).log2();
    // Strict window first; then relaxed (a far-shifted recording beats a
    // mid-line timbre switch to synthesis — see pick_with_window).
    let picked = bank
        .pick(target_midi, note.velocity, note_seed(note))
        .or_else(|| bank.pick_with_window(target_midi, note.velocity, note_seed(note), 12.0, 24.0));
    let Some(sample) = picked else {
        return false;
    };
    let ratio =
        2f32.powf((target_midi - sample.midi as f32) / 12.0) * (sample.sample_rate as f32 / sr);
    if !ratio.is_finite() || ratio <= 0.0 {
        return false;
    }
    let start = (note.start_time * sr) as usize;
    let release = 0.25f32;
    let out_len = ((note.duration + release) * sr) as usize;
    let fade_in = if legato_from_prev {
        ((0.015 * sr) as usize).max(1)
    } else {
        ((0.003 * sr) as usize).max(1)
    };
    let fade_start = (note.duration * sr) as usize;
    // The sample was RECORDED at its dynamic layer (a soft layer already
    // sounds soft), so velocity only trims level mildly on top.
    let gain = 0.5 * (0.6 + 0.4 * note.velocity) * vol;
    // 40ms: enough to soften the bow/tongue transient inside a slur while
    // keeping some articulation — the first cut (80ms) erased the note
    // starts entirely and strings read as "not played properly".
    let sample_secs = sample.frames.len() as f32 / sample.sample_rate as f32;
    let attack_skip_secs = if legato_from_prev {
        0.04f32.min(sample_secs * 0.25)
    } else {
        0.0
    };
    let mut src_pos = attack_skip_secs * sample.sample_rate as f32;
    // How many output samples this recording can actually supply at
    // `ratio`'s playback speed before running out -- a note whose
    // duration+release outlasts the recording used to hard-stop here with
    // no fade at all (`idx + 1 >= sample.frames.len() { break; }` below),
    // an audible dropout on exactly the long/held notes (arrival cadences,
    // climaxes) that matter most. When exhaustion would land before the
    // note's own natural end, ramp the envelope to zero over the last
    // `FADE_TAIL_SAMPLES` instead of cutting off mid-sustain.
    const FADE_TAIL_SECS: f32 = 0.01;
    let i_exhaust = if ratio > 0.0 {
        ((sample.frames.len() as f32 - 1.0 - src_pos).max(0.0) / ratio).floor() as usize
    } else {
        usize::MAX
    };
    let fade_tail = ((FADE_TAIL_SECS * sr) as usize).max(1);
    let fade_out_start = i_exhaust.saturating_sub(fade_tail);
    for i in 0..out_len {
        let si = start + i;
        if si >= bl.len() {
            break;
        }
        let idx = src_pos as usize;
        if idx + 1 >= sample.frames.len() {
            break;
        }
        let frac = src_pos - idx as f32;
        let s = hermite_interpolate(&sample.frames, idx, frac);
        let mut env = 1.0f32;
        if i < fade_in {
            env = i as f32 / fade_in as f32;
        }
        if i >= fade_start && out_len > fade_start {
            env *= 1.0 - (i - fade_start) as f32 / (out_len - fade_start) as f32;
        }
        if i_exhaust < out_len && i >= fade_out_start {
            env *= (1.0 - (i - fade_out_start) as f32 / fade_tail as f32).clamp(0.0, 1.0);
        }
        if sustained_shape && fade_start > 0 {
            // Messa di voce: 0.85 at the bow start, swelling to 1.12 at
            // 40% of the written duration, relaxing to 0.85 by the end.
            // Piecewise-linear, so it adds no discontinuities of its own.
            let pos = (i as f32 / fade_start as f32).min(1.0);
            let arch = if pos < 0.4 {
                0.85 + 0.27 * (pos / 0.4)
            } else {
                1.12 - 0.27 * ((pos - 0.4) / 0.6)
            };
            env *= arch;
        }
        let o = s * env * gain;
        bl[si] += o * gl;
        br[si] += o * gr;
        src_pos += ratio;
    }
    true
}

/// Render one note through FM synthesis (see
/// [`crate::instruments::render_fm_instrument`]) instead of the additive
/// engine — used for the DX7-style/percussive-FM instruments
/// ([`Instrument::uses_fm`]).
#[allow(clippy::too_many_arguments)]
fn render_fm_note(
    bl: &mut [f32],
    br: &mut [f32],
    sample_rate: u32,
    note: &Note,
    vol: f32,
    gl: f32,
    gr: f32,
    instrument: Instrument,
) {
    let buf = crate::instruments::render_fm_instrument(
        instrument,
        note.frequency,
        note.velocity * vol,
        note.duration,
        sample_rate,
    );
    let start = (note.start_time * sample_rate as f32) as usize;
    for (i, &s) in buf.iter().enumerate() {
        let si = start + i;
        if si >= bl.len() {
            break;
        }
        let o = s * 0.6;
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

/// Soft clipper: identity below |x| = 0.8, smooth compression above, output
/// bounded to (-1, 1). Delegates to `peak_limit` so the transfer curve is
/// continuous and monotone. (A previous piecewise version jumped from 1.0
/// down to 0.5 at |x| = 1.0, wavefolding any over-full-scale sample.)
pub(crate) fn soft_clip(x: f32) -> f32 {
    peak_limit(x, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voice::{Arrangement, Voice, VoiceRole};
    use crate::{MuseConfig, MusicalState, Note, OutputFormat};

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn legato_continuation_skips_the_recorded_attack() {
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../data/samples/vcsl");
        let Some(lib) = crate::vcsl::VcslLibrary::load(&root) else {
            eprintln!("VCSL not on disk — skipping");
            return;
        };
        let Some(bank) = lib.bank(crate::instruments::Instrument::Violin) else {
            eprintln!("violin bank absent — skipping");
            return;
        };
        let sr = 44100.0f32;
        let note = Note {
            frequency: 440.0,
            start_time: 0.0,
            duration: 0.6,
            velocity: 0.7,
        };
        let render = |legato: bool| {
            let n = (sr * 1.0) as usize;
            let (mut bl, mut br) = (vec![0.0f32; n], vec![0.0f32; n]);
            assert!(render_vcsl_note(
                &mut bl, &mut br, sr, &note, 1.0, 0.7, 0.7, bank, legato, false
            ));
            bl
        };
        let plain = render(false);
        let legato = render(true);
        // The first 60ms must genuinely differ — the legato render starts
        // playback past the recorded attack transient.
        let w = (0.06 * sr) as usize;
        let diff: f32 = plain[..w]
            .iter()
            .zip(&legato[..w])
            .map(|(a, b)| (a - b).abs())
            .sum();
        let energy: f32 = plain[..w].iter().map(|s| s.abs()).sum();
        assert!(
            diff > energy * 0.2,
            "legato start must differ from the attack (diff {diff}, energy {energy})"
        );
        // And it is still real audio, not silence.
        assert!(legato.iter().any(|s| s.abs() > 1e-4));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn sustained_shape_swells_toward_the_middle_of_a_long_note() {
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../data/samples/vcsl");
        let Some(lib) = crate::vcsl::VcslLibrary::load(&root) else {
            eprintln!("VCSL not on disk — skipping");
            return;
        };
        let Some(bank) = lib.bank(crate::instruments::Instrument::Violin) else {
            eprintln!("violin bank absent — skipping");
            return;
        };
        let sr = 44100.0f32;
        let note = Note {
            frequency: 440.0,
            start_time: 0.0,
            duration: 2.0,
            velocity: 0.7,
        };
        let render = |shape: bool| {
            let n = (sr * 2.5) as usize;
            let (mut bl, mut br) = (vec![0.0f32; n], vec![0.0f32; n]);
            assert!(render_vcsl_note(
                &mut bl, &mut br, sr, &note, 1.0, 0.7, 0.7, bank, false, shape
            ));
            bl
        };
        let flat = render(false);
        let shaped = render(true);
        // Same sample, same pick — the only difference is the arch. The
        // shaped/flat gain ratio must be higher at 40% of the note than
        // near its start (0.85 → 1.12 swell).
        let rms = |b: &[f32], at: f32| {
            let c = (at * 2.0 * sr) as usize;
            let w = (0.1 * sr) as usize;
            (b[c..c + w].iter().map(|s| s * s).sum::<f32>() / w as f32).sqrt()
        };
        let ratio_start = rms(&shaped, 0.05) / rms(&flat, 0.05).max(1e-9);
        let ratio_peak = rms(&shaped, 0.40) / rms(&flat, 0.40).max(1e-9);
        assert!(
            ratio_peak > ratio_start + 0.1,
            "arch must swell: start ratio {ratio_start:.3}, peak ratio {ratio_peak:.3}"
        );
    }

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
                instrument: None,
            }],
        }
    }

    // ─── round-robin variation ──────────────────────────────────────────────

    #[test]
    fn repeated_identical_notes_render_differently() {
        // Two notes identical in every field except start_time must NOT
        // produce bit-identical waveforms — that phase-locked repetition is
        // the "machine gun" artifact. Render each note alone into its own
        // buffer and compare the two note windows sample-for-sample.
        let sr = 44100.0f32;
        let n = 4410; // 100ms window
        let make = |start: f32| Note {
            frequency: 440.0,
            start_time: start,
            duration: 0.5,
            velocity: 0.8,
        };
        let render = |note: &Note| -> Vec<f32> {
            let total = ((note.start_time + 1.5) * sr) as usize;
            let (mut bl, mut br) = (vec![0.0f32; total], vec![0.0f32; total]);
            let timbre = NoteTimbre::legacy(&[1.0f32, 0.5, 0.33, 0.25]);
            let adsr = Adsr {
                attack: 0.01,
                decay: 0.1,
                sustain: 0.7,
                release: 0.2,
            };
            render_tone(
                &mut bl,
                &mut br,
                sr,
                note,
                note.frequency,
                1.0,
                0.7,
                0.7,
                &timbre,
                &adsr,
                0.0,
                2.0,
                8.0,
                4.0,
            );
            let s = (note.start_time * sr) as usize;
            bl[s..s + n].to_vec()
        };
        let a = render(&make(0.0));
        let b = render(&make(1.0));
        assert_eq!(a.len(), b.len());
        assert!(
            a.iter().zip(&b).any(|(x, y)| x != y),
            "notes at different times must not be bit-identical"
        );
        // Same note re-rendered must reproduce exactly (determinism).
        let a2 = render(&make(0.0));
        assert_eq!(a, a2, "same note must render identically across runs");
        // And the variation must be phase-level, not loudness-level: the two
        // windows carry comparable energy.
        let e = |v: &[f32]| v.iter().map(|x| (x * x) as f64).sum::<f64>();
        let (ea, eb) = (e(&a), e(&b));
        assert!(
            (ea / eb).max(eb / ea) < 1.5,
            "round-robin must not change loudness: {ea:.4} vs {eb:.4}"
        );
    }

    // ─── per-note timbre (velocity→spectrum, transients, inharmonicity) ─────

    fn tone_with_timbre(timbre: &NoteTimbre, secs: f32) -> Vec<f32> {
        let sr = 44100.0f32;
        let note = Note {
            frequency: 220.0,
            start_time: 0.0,
            duration: secs,
            velocity: 0.8,
        };
        let total = ((secs + 0.5) * sr) as usize;
        let (mut bl, mut br) = (vec![0.0f32; total], vec![0.0f32; total]);
        let adsr = Adsr {
            attack: 0.01,
            decay: 0.05,
            sustain: 0.8,
            release: 0.2,
        };
        render_tone(
            &mut bl, &mut br, sr, &note, 220.0, 1.0, 0.7, 0.7, timbre, &adsr, 0.0, 2.0, 12.0, 8.0,
        );
        bl
    }

    #[test]
    fn attack_transient_speaks_only_at_the_onset() {
        let base = NoteTimbre::legacy(&[1.0f32, 0.5, 0.33]);
        let with_chiff = NoteTimbre {
            attack_amp: 0.12,
            attack_dur: 0.04,
            ..NoteTimbre::legacy(&[1.0f32, 0.5, 0.33])
        };
        let a = tone_with_timbre(&base, 1.0);
        let b = tone_with_timbre(&with_chiff, 1.0);
        let window = (0.04f32 * 44100.0) as usize;
        assert!(
            a[..window].iter().zip(&b[..window]).any(|(x, y)| x != y),
            "transient must alter the onset"
        );
        assert_eq!(
            &a[window..],
            &b[window..],
            "transient must be silent after its duration"
        );
    }

    #[test]
    fn sustained_instruments_keep_brightness_while_struck_ones_dull() {
        // Same partials, same envelope — only the per-partial decay differs.
        // Late in a long note, the sustained timbre must retain more energy
        // (its upper partials are still being energized by the bow/breath).
        let struck = NoteTimbre {
            decay_rate: 0.8,
            ..NoteTimbre::legacy(&[1.0f32, 0.8, 0.6, 0.5, 0.4])
        };
        let sustained = NoteTimbre {
            decay_rate: 0.12,
            ..NoteTimbre::legacy(&[1.0f32, 0.8, 0.6, 0.5, 0.4])
        };
        let a = tone_with_timbre(&struck, 2.0);
        let b = tone_with_timbre(&sustained, 2.0);
        let late = (1.5f32 * 44100.0) as usize..(2.0f32 * 44100.0) as usize;
        let e = |v: &[f32]| v.iter().map(|x| (x * x) as f64).sum::<f64>();
        assert!(
            e(&b[late.clone()]) > e(&a[late]) * 1.2,
            "sustained timbre must keep noticeably more late-note energy"
        );
    }

    #[test]
    fn piano_inharmonicity_changes_the_waveform() {
        let harmonic = NoteTimbre::legacy(&[1.0f32, 0.8, 0.6, 0.5]);
        let stretched = NoteTimbre {
            inharmonicity: 0.0004,
            ..NoteTimbre::legacy(&[1.0f32, 0.8, 0.6, 0.5])
        };
        let a = tone_with_timbre(&harmonic, 0.5);
        let b = tone_with_timbre(&stretched, 0.5);
        assert!(
            a.iter().zip(&b).any(|(x, y)| x != y),
            "stretched partials must render differently"
        );
    }

    #[test]
    fn ks_velocity_brightens_the_pluck() {
        // A hard pluck must carry more high-frequency content than a soft
        // one — measured here as zero crossings over the first 200ms
        // (loudness alone cannot change a zero-crossing count).
        let sr = 44100.0f32;
        let render = |velocity: f32| -> Vec<f32> {
            let note = Note {
                frequency: 220.0,
                start_time: 0.0,
                duration: 0.5,
                velocity,
            };
            let total = (1.5 * sr) as usize;
            let (mut bl, mut br) = (vec![0.0f32; total], vec![0.0f32; total]);
            render_karplus_note(
                &mut bl,
                &mut br,
                sr,
                &note,
                1.0,
                0.7,
                0.7,
                Instrument::AcousticGuitar,
            );
            bl
        };
        let zc = |v: &[f32]| {
            v.windows(2)
                .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
                .count()
        };
        let window = (0.2 * sr) as usize;
        let soft = render(0.2);
        let hard = render(1.0);
        assert!(
            zc(&hard[..window]) > zc(&soft[..window]),
            "hard pluck must be brighter: soft={} hard={}",
            zc(&soft[..window]),
            zc(&hard[..window])
        );
    }

    #[test]
    fn vibrato_spreads_energy_off_the_carrier() {
        // Goertzel |X| at the exact carrier frequency: a vibrato-modulated
        // tone moves energy into sidebands, so the bin at cf must lose
        // energy vs the identical stationary tone. (An amplitude change
        // could not fake this — both renders share every other parameter.)
        let goertzel = |v: &[f32], freq: f32, sr: f32| -> f32 {
            let w = std::f32::consts::TAU * freq / sr;
            let coeff = 2.0 * w.cos();
            let (mut s1, mut s2) = (0.0f32, 0.0f32);
            for &x in v {
                let s0 = x + coeff * s1 - s2;
                s2 = s1;
                s1 = s0;
            }
            (s1 * s1 + s2 * s2 - coeff * s1 * s2).sqrt()
        };
        // Frequency matters: the FM index is β = cf·depth/rate, and the
        // carrier retains J₀(β) of its amplitude. At 220 Hz with real violin
        // vibrato (d=0.007, 5.5 Hz), β≈0.28 → J₀≈0.98 — a correct, subtle
        // vibrato that a coarse threshold can't see (the first version of
        // this test failed exactly there). At 880 Hz (violin E-string
        // register), β≈1.12 → J₀≈0.66 — a robustly detectable 34% drop.
        let sr = 44100.0f32;
        let render = |vibrato: Option<(f32, f32, f32)>| -> Vec<f32> {
            let note = Note {
                frequency: 880.0,
                start_time: 0.0,
                duration: 2.0,
                velocity: 0.8,
            };
            let timbre = NoteTimbre {
                decay_rate: 0.12,
                vibrato,
                ..NoteTimbre::legacy(&[1.0f32])
            };
            let total = (2.5 * sr) as usize;
            let (mut bl, mut br) = (vec![0.0f32; total], vec![0.0f32; total]);
            let adsr = Adsr {
                attack: 0.01,
                decay: 0.05,
                sustain: 0.8,
                release: 0.2,
            };
            render_tone(
                &mut bl, &mut br, sr, &note, 880.0, 1.0, 0.7, 0.7, &timbre, &adsr, 0.0, 2.0, 12.0,
                8.0,
            );
            bl
        };
        let stationary = render(None);
        let vibed = render(Some((5.5, 0.007, 0.2)));
        // Late window: vibrato fully developed, envelope in flat sustain.
        let (a, b) = ((1.0f32 * sr) as usize, (1.9f32 * sr) as usize);
        let on_carrier_stationary = goertzel(&stationary[a..b], 880.0, sr);
        let on_carrier_vibed = goertzel(&vibed[a..b], 880.0, sr);
        assert!(
            on_carrier_vibed < on_carrier_stationary * 0.8,
            "vibrato must move energy off the carrier: {on_carrier_vibed} vs {on_carrier_stationary}"
        );
    }

    #[test]
    fn hermite_beats_linear_on_a_known_sine() {
        // Resample a 1kHz sine by a fractional ratio; compare against the
        // TRUE analytic values. Hermite must be at least 5x more accurate
        // than linear — the interpolation error is exactly the "harsh"
        // imaging noise the sampler used to add.
        let sr = 44100.0f32;
        let src: Vec<f32> = (0..2048)
            .map(|i| (std::f32::consts::TAU * 1000.0 * i as f32 / sr).sin())
            .collect();
        let ratio = 1.29739f32; // awkward fraction, worst case for interp
        let (mut err_h, mut err_l, mut n) = (0.0f64, 0.0f64, 0);
        let mut pos = 4.0f32;
        while (pos as usize) + 3 < src.len() {
            let idx = pos as usize;
            let frac = pos - idx as f32;
            let truth = (std::f32::consts::TAU * 1000.0 * pos / sr).sin();
            let lin = src[idx] * (1.0 - frac) + src[idx + 1] * frac;
            let her = hermite_interpolate(&src, idx, frac);
            err_l += (lin - truth).abs() as f64;
            err_h += (her - truth).abs() as f64;
            n += 1;
            pos += ratio;
        }
        let (mae_h, mae_l) = (err_h / n as f64, err_l / n as f64);
        assert!(
            mae_h * 5.0 < mae_l,
            "hermite MAE {mae_h} must be well under linear MAE {mae_l}"
        );
    }

    // ─── VCSL sampled rendering ─────────────────────────────────────────────

    /// Uses a LOCAL library instance on purpose: initializing the
    /// process-global `vcsl::library()` from a test would silently switch
    /// every other render test in this process onto samples.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn vcsl_sample_renders_real_audio_when_library_present() {
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../data/samples/vcsl");
        let Some(lib) = crate::vcsl::VcslLibrary::load(&root) else {
            eprintln!("VCSL not on disk — skipping");
            return;
        };
        let Some(bank) = lib.bank(Instrument::Harp) else {
            panic!("harp bank must index when the library is present");
        };
        let sr = 44100.0f32;
        let note = Note {
            frequency: 261.63, // C4
            start_time: 0.0,
            duration: 1.0,
            velocity: 0.8,
        };
        let n = (2.0 * sr) as usize;
        let (mut bl, mut br) = (vec![0.0f32; n], vec![0.0f32; n]);
        assert!(
            render_vcsl_note(
                &mut bl, &mut br, sr, &note, 1.0, 0.7, 0.7, bank, false, false
            ),
            "a concert-harp C4 must be renderable from samples"
        );
        let energy: f32 = bl.iter().map(|x| x * x).sum();
        assert!(energy > 0.0, "sampled note must produce audio");
        assert!(bl.iter().chain(br.iter()).all(|x| x.is_finite()));
        // Declick: the very first sample must be (near) zero, not a step.
        assert!(bl[0].abs() < 1e-3);
    }

    // ─── soft_clip ────────────────────────────────────────────────────────────

    #[test]
    fn soft_clip_identity_in_linear_region() {
        // Identity below the 0.8 knee
        for x in [-0.79f32, -0.5, 0.0, 0.5, 0.79] {
            assert!(
                (soft_clip(x) - x).abs() < 1e-6,
                "soft_clip({x}) should be identity"
            );
        }
    }

    #[test]
    fn soft_clip_bounded_at_full_scale() {
        // The compression curve asymptotes to the ceiling; at f32 precision
        // the exp() underflows for huge inputs and the output reaches exactly
        // ±1.0 — never beyond.
        for x in [-1000.0f32, -10.0, -2.0, -1.01, 1.01, 2.0, 10.0, 1000.0] {
            let y = soft_clip(x);
            assert!(
                y.abs() <= 1.0,
                "soft_clip({x}) = {y} must never exceed ±1.0"
            );
        }
    }

    #[test]
    fn soft_clip_monotone_and_continuous() {
        // The whole curve must be monotone non-decreasing and continuous —
        // the old implementation jumped 1.0 → 0.5 just above x = 1.0
        // (wavefolding). Sweep across the knee and the old discontinuity.
        let mut prev = soft_clip(-3.0);
        let mut x = -3.0f32;
        while x <= 3.0 {
            let y = soft_clip(x);
            assert!(
                y >= prev - 1e-6,
                "soft_clip not monotone at {x}: {y} < {prev}"
            );
            assert!(
                (y - prev).abs() < 0.05,
                "soft_clip discontinuity near {x}: jumped {prev} -> {y}"
            );
            prev = y;
            x += 0.01;
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
        // FM is emotion-gated (render_arrangement:140): it only engages for
        // tense (valence < -0.2, arousal > 0.5) or joyful states. The default
        // state gates FM to zero, which would make both renders identical.
        let state = MusicalState {
            valence: -0.5,
            arousal: 0.7,
            ..Default::default()
        };
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
        // Timbre is emotion-gated (compute_timbre): the default state falls in
        // the "sorrowful" branch, deliberately near-sine (0.3/h^2.8 rolloff),
        // where num_partials is inaudible — and arousal > 0.4 caps partials
        // regardless of config. Use the contemplative branch (valence > 0.2,
        // arousal ≤ 0.4), which carries real 3rd/5th-harmonic content and
        // honors config.num_partials.
        let state = MusicalState {
            valence: 0.5,
            arousal: 0.3,
            ..Default::default()
        };
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
            // Loudness normalization equalizes RMS across timbres, so compare
            // spectral content instead: normalized first-difference energy is a
            // high-frequency proxy — 16 partials must carry more HF energy
            // than a single 220 Hz sine.
            let hf_ratio = |v: &[f32]| {
                let total: f32 = v.iter().map(|x| x * x).sum();
                let diff: f32 = v.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
                diff / total.max(1e-12)
            };
            let (thin_hf, rich_hf) = (hf_ratio(&t), hf_ratio(&r));
            // Expected ≈1.5× from the contemplative 3rd/5th harmonics after
            // the per-emotion filter envelope; 1.15 leaves headroom for the
            // shared chord/reverb layers diluting the contrast.
            assert!(
                rich_hf > thin_hf * 1.15,
                "rich ({} partials, hf={rich_hf}) should carry more high-frequency \
                 energy than thin (1 partial, hf={thin_hf})",
                rich.num_partials
            );
        } else {
            panic!("expected MonoF32 output");
        }
    }

    #[test]
    fn render_output_stays_within_full_scale() {
        // Guards the soft_clip wavefolding regression: drive a hot, dense
        // state (sub-bass + unison + noise + big reverb) and assert no sample
        // ever leaves ±1.0 in the final mix.
        let mut arr = one_note_arrangement(220.0);
        for freq in [277.18, 329.63, 440.0, 554.37] {
            arr.voices[0].notes.push(Note {
                frequency: freq,
                start_time: 0.0,
                duration: 0.5,
                velocity: 1.0,
            });
        }
        let state = MusicalState {
            valence: -0.8,
            arousal: 1.0,
            noradrenaline: 1.0,
            consciousness_level: 1.0,
            ..Default::default()
        };
        let config = MuseConfig {
            duration_secs: 0.5,
            output_format: OutputFormat::StereoF32,
            ..MuseConfig::horror()
        };
        let audio = render_arrangement(&arr, 44100, 22050, &state, &config);
        if let AudioData::StereoF32(frames) = audio {
            for (i, [l, r]) in frames.iter().enumerate() {
                assert!(
                    l.abs() <= 1.0 && r.abs() <= 1.0,
                    "sample {i} out of full scale: L={l} R={r}"
                );
                assert!(l.is_finite() && r.is_finite(), "sample {i} not finite");
            }
        } else {
            panic!("expected StereoF32 output");
        }
    }
}
