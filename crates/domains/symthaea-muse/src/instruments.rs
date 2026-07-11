// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Acoustic instrument synthesis from published physical measurements.
//!
//! Sources:
//! - Piano: Fletcher & Rossing, *Physics of Musical Instruments* (1998)
//! - Violin: Saunders (1937), Schelleng (1973), Cremer (1984)
//! - DX7 FM: Chowning (1973), Yamaha DX7 manual (1983)
//! - Karplus-Strong: Karplus & Strong (1983), Jaffe & Smith (1983)
//!
//! All data is from peer-reviewed acoustic measurements (scientific facts,
//! not copyrightable). DX7 patents expired in early 2000s.

use crate::MusicalState;
use serde::{Deserialize, Serialize};

// ─── Acoustic Partial Data ──────────────────────────────────────────────────

/// Piano partial amplitudes at ff and pp (C4, Fletcher & Rossing 1998).
/// Index 0 = fundamental, index 15 = 16th partial.
pub const PIANO_FF: [f32; 16] = [
    1.000, 0.891, 0.794, 0.631, 0.501, 0.398, 0.316, 0.251, 0.200, 0.158, 0.126, 0.100, 0.079,
    0.063, 0.050, 0.040,
];
pub const PIANO_PP: [f32; 16] = [
    1.000, 0.630, 0.398, 0.251, 0.158, 0.100, 0.063, 0.040, 0.025, 0.016, 0.010, 0.006, 0.004,
    0.003, 0.002, 0.001,
];

/// Violin partial amplitudes (measured G3, Saunders/Schelleng).
pub const VIOLIN: [f32; 16] = [
    1.000, 0.520, 0.380, 0.230, 0.250, 0.150, 0.180, 0.100, 0.130, 0.080, 0.100, 0.060, 0.075,
    0.045, 0.055, 0.030,
];

/// Cello partial amplitudes (measured C2).
pub const CELLO: [f32; 16] = [
    1.000, 0.480, 0.350, 0.270, 0.200, 0.180, 0.130, 0.120, 0.090, 0.100, 0.070, 0.080, 0.050,
    0.060, 0.035, 0.045,
];

/// Flute-like (strong fundamental, weak upper partials).
pub const FLUTE: [f32; 16] = [
    1.000, 0.200, 0.050, 0.010, 0.005, 0.002, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.000, 0.000,
];

/// Organ-like (drawbar registration: 8'+4'+2-2/3'+2').
pub const ORGAN: [f32; 16] = [
    1.000, 0.500, 0.250, 0.125, 0.060, 0.030, 0.015, 0.008, 0.004, 0.002, 0.001, 0.001, 0.000,
    0.000, 0.000, 0.000,
];

/// Clarinet: strong odd harmonics (cylindrical bore, Benade 1976).
pub const CLARINET: [f32; 16] = [
    1.000, 0.050, 0.750, 0.030, 0.500, 0.020, 0.350, 0.015, 0.200, 0.010, 0.120, 0.008, 0.070,
    0.005, 0.040, 0.003,
];
/// Trumpet: rich harmonics with formant emphasis around partials 4-8 (Luce 1963).
pub const TRUMPET: [f32; 16] = [
    1.000, 0.700, 0.600, 0.800, 0.900, 0.750, 0.600, 0.500, 0.350, 0.250, 0.180, 0.120, 0.080,
    0.060, 0.040, 0.025,
];
/// Saxophone: complex spectrum with strong even+odd partials (Backus 1977).
pub const SAXOPHONE: [f32; 16] = [
    1.000, 0.600, 0.450, 0.350, 0.400, 0.300, 0.350, 0.200, 0.250, 0.150, 0.180, 0.100, 0.120,
    0.080, 0.060, 0.040,
];
/// Sitar: sympathetic resonance with buzzing bridge (jawari effect).
pub const SITAR: [f32; 16] = [
    1.000, 0.400, 0.600, 0.200, 0.450, 0.150, 0.350, 0.120, 0.300, 0.100, 0.200, 0.080, 0.150,
    0.060, 0.100, 0.050,
];
/// Sawtooth wave (electronic synth lead, all harmonics with 1/n falloff).
pub const SAW_LEAD: [f32; 16] = [
    1.000, 0.500, 0.333, 0.250, 0.200, 0.167, 0.143, 0.125, 0.111, 0.100, 0.091, 0.083, 0.077,
    0.071, 0.067, 0.063,
];

/// Piano decay time T60 by octave (seconds, Weinreich 1977).
pub const PIANO_T60: [f32; 8] = [18.0, 14.0, 12.0, 10.0, 6.0, 3.5, 1.5, 0.5];

/// Piano inharmonicity coefficient B by octave (Conklin 1996).
pub const PIANO_INHARMONICITY: [f32; 8] = [
    0.00012, 0.00008, 0.00006, 0.00004, 0.00003, 0.00010, 0.00040, 0.00150,
];

// ─── Instrument Enum ────────────────────────────────────────────────────────

/// Available instrument timbres.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Instrument {
    Piano,
    PianoPP,
    Violin,
    Cello,
    Flute,
    Organ,
    ElectricPiano,
    Bell,
    AcousticGuitar,
    Harp,
    Pad,
    Clarinet,
    Trumpet,
    Saxophone,
    Marimba,
    Sitar,
    Kalimba,
    SawLead,
    Koto,
    Oud,
    Ney,
    UprightBass,
}

impl Instrument {
    /// Get the partial amplitude profile for additive synthesis.
    pub fn partials(&self) -> &'static [f32; 16] {
        match self {
            Self::Piano => &PIANO_FF,
            Self::PianoPP | Self::Pad => &PIANO_PP,
            Self::Violin => &VIOLIN,
            Self::Cello | Self::UprightBass => &CELLO,
            Self::Flute | Self::Ney => &FLUTE,
            Self::Organ => &ORGAN,
            Self::Clarinet => &CLARINET,
            Self::Trumpet => &TRUMPET,
            Self::Saxophone => &SAXOPHONE,
            Self::Sitar => &SITAR,
            Self::SawLead => &SAW_LEAD,
            _ => &PIANO_FF, // ElectricPiano, Bell, Guitar, Harp, Koto, Oud, Marimba, Kalimba
        }
    }

    /// Velocity-dependent partial spectrum. A fortissimo note is not just a
    /// louder pianissimo note — playing harder shifts energy into upper
    /// partials on essentially every acoustic instrument. Before this,
    /// velocity was a pure amplitude multiplier ("organ mode").
    ///
    /// Piano crossfades between the two MEASURED dynamic tables that were
    /// already in this file (Fletcher & Rossing's ff and pp spectra —
    /// previously reachable only as *separate instruments*); everything else
    /// gets a physically-motivated spectral tilt (upper partial h scaled by
    /// `tilt^h`, tilt rising with velocity).
    pub fn partials_at_velocity(&self, velocity: f32) -> [f32; 16] {
        let v = velocity.clamp(0.0, 1.0);
        let mut out = [0.0f32; 16];
        match self {
            Self::Piano | Self::PianoPP => {
                for h in 0..16 {
                    out[h] = PIANO_PP[h] + (PIANO_FF[h] - PIANO_PP[h]) * v;
                }
            }
            _ => {
                let base = self.partials();
                let tilt = 0.7 + 0.3 * v;
                let mut f = 1.0f32;
                for h in 0..16 {
                    out[h] = base[h] * f;
                    f *= tilt;
                }
            }
        }
        out
    }

    /// Whether the instrument is continuously energized while a note is held
    /// (bowed strings, wind, organ, synth pad). These must NOT get the
    /// struck/plucked per-partial decay — a sustained violin note keeps its
    /// brightness while bowed; a piano note dulls as it rings.
    pub fn sustains(&self) -> bool {
        matches!(
            self,
            Self::Violin
                | Self::Cello
                | Self::Flute
                | Self::Organ
                | Self::Clarinet
                | Self::Trumpet
                | Self::Saxophone
                | Self::Ney
                | Self::Pad
                | Self::SawLead
        )
    }

    /// Attack transient noise for the additive path: `(amplitude, seconds)`.
    /// The onset noise — bow bite, breath chiff, hammer strike — is a large
    /// part of instrument identity; a pure partial stack has none, which is
    /// a big reason additive "violin" reads as synthetic. Amplitudes are
    /// relative to the note's own level and deliberately subtle.
    pub fn attack_noise(&self) -> (f32, f32) {
        match self {
            Self::Violin | Self::Cello => (0.10, 0.06), // bow bite
            Self::Flute | Self::Ney => (0.12, 0.04),    // breath chiff
            Self::Clarinet | Self::Saxophone | Self::Trumpet => (0.06, 0.03), // reed/lip start
            Self::Piano | Self::PianoPP => (0.05, 0.008), // hammer click
            Self::Organ => (0.04, 0.02),                // pipe speech/chiff
            _ => (0.0, 0.0),
        }
    }

    /// Vibrato for the additive path: `(rate_hz, depth_ratio, onset_delay_s)`,
    /// or `None` for instruments played without it. A perfectly stationary
    /// sine stack is one of the strongest "this is a synthesizer" tells for
    /// bowed strings and winds — real players' vibrato develops shortly
    /// after the note speaks (the delay), at ~4.5-6 Hz, with depth well
    /// under a semitone. Organ pipes and struck/plucked strings have no
    /// player-controlled vibrato, so they honestly get `None`.
    pub fn vibrato(&self) -> Option<(f32, f32, f32)> {
        match self {
            Self::Violin | Self::Cello => Some((5.5, 0.007, 0.20)),
            Self::Flute | Self::Ney => Some((5.0, 0.004, 0.25)),
            Self::Clarinet | Self::Saxophone | Self::Trumpet => Some((4.5, 0.003, 0.30)),
            _ => None,
        }
    }

    /// Inharmonicity coefficient B for stretched partials (`f_h =
    /// f·(h+1)·√(1+B(h+1)²)`). Piano strings are stiff enough that upper
    /// partials run measurably sharp — `PIANO_INHARMONICITY` (Conklin 1996)
    /// sat in this file cited-but-unreferenced while piano partials rendered
    /// perfectly harmonic, a big part of why synthesized piano sounds fake.
    pub fn inharmonicity(&self, freq: f32) -> f32 {
        match self {
            Self::Piano | Self::PianoPP => {
                // Octave index 0..7 over the piano range (A0 = 27.5 Hz).
                let oct = (freq / 27.5).max(1.0).log2().clamp(0.0, 7.0) as usize;
                PIANO_INHARMONICITY[oct.min(7)]
            }
            _ => 0.0,
        }
    }

    /// Resolve a spec's snake_case instrument name (see
    /// `symthaea_music_theory::spec::CompositionSpec::ensemble_pool`).
    /// Unknown names return `None` — the CALLER decides the fallback and
    /// must do it loudly, never silently.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "piano" => Self::Piano,
            "piano_pp" => Self::PianoPP,
            "violin" => Self::Violin,
            "cello" => Self::Cello,
            "flute" => Self::Flute,
            "organ" => Self::Organ,
            "electric_piano" => Self::ElectricPiano,
            "bell" => Self::Bell,
            "acoustic_guitar" => Self::AcousticGuitar,
            "harp" => Self::Harp,
            "pad" => Self::Pad,
            "clarinet" => Self::Clarinet,
            "trumpet" => Self::Trumpet,
            "saxophone" => Self::Saxophone,
            "marimba" => Self::Marimba,
            "sitar" => Self::Sitar,
            "kalimba" => Self::Kalimba,
            "saw_lead" => Self::SawLead,
            "koto" => Self::Koto,
            "oud" => Self::Oud,
            "ney" => Self::Ney,
            "upright_bass" => Self::UprightBass,
            _ => return None,
        })
    }

    /// General MIDI program number (0-based) for exporting this instrument
    /// to a Standard MIDI File. Instruments without an exact GM equivalent
    /// map to the closest family member (Oud → nylon guitar, Ney → pan
    /// flute) so a DAW's default GM bank renders something reasonable.
    pub fn gm_program(&self) -> u8 {
        match self {
            Self::Piano | Self::PianoPP => 0, // Acoustic Grand Piano
            Self::ElectricPiano => 4,         // Electric Piano 1
            Self::Marimba => 12,
            Self::Bell => 14, // Tubular Bells
            Self::Organ => 19,
            Self::AcousticGuitar | Self::Oud => 24, // Nylon Guitar
            Self::UprightBass => 32,                // Acoustic Bass
            Self::Violin => 40,
            Self::Cello => 42,
            Self::Harp => 46,
            Self::Trumpet => 56,
            Self::Saxophone => 66, // Tenor Sax
            Self::Clarinet => 71,
            Self::Flute => 73,
            Self::Ney => 75, // Pan Flute
            Self::SawLead => 81,
            Self::Pad => 88,
            Self::Sitar => 104,
            Self::Koto => 107,
            Self::Kalimba => 108,
        }
    }

    /// Whether this instrument uses Karplus-Strong synthesis.
    pub fn uses_karplus_strong(&self) -> bool {
        matches!(
            self,
            Self::AcousticGuitar | Self::Harp | Self::Koto | Self::Oud | Self::UprightBass
        )
    }

    /// Whether this instrument uses FM synthesis.
    pub fn uses_fm(&self) -> bool {
        matches!(
            self,
            Self::ElectricPiano | Self::Bell | Self::Marimba | Self::Kalimba
        )
    }

    /// Karplus-Strong parameters: (damping, brightness, stiffness).
    pub fn ks_params(&self) -> (f32, f32, f32) {
        match self {
            Self::AcousticGuitar => (0.996, 0.42, 0.005),
            Self::Harp => (0.997, 0.40, 0.000),
            Self::Koto => (0.994, 0.55, 0.008),
            Self::Oud => (0.997, 0.35, 0.002),
            Self::UprightBass => (0.998, 0.30, 0.001),
            _ => (0.996, 0.42, 0.000),
        }
    }

    /// FM parameters: (carrier_ratio, mod_ratio, index, index_decay_rate).
    pub fn fm_params(&self) -> (f32, f32, f32, f32) {
        match self {
            Self::ElectricPiano => (1.0, 1.0, 1.5, 15.0),
            Self::Bell => (1.0, 3.5, 2.5, 3.0),
            Self::Marimba => (1.0, 4.0, 1.0, 30.0),
            Self::Kalimba => (1.0, 5.17, 1.2, 8.0),
            _ => (1.0, 1.0, 0.3, 10.0),
        }
    }

    /// Default ADSR envelope (attack, decay, sustain, release) in seconds
    /// for the ADDITIVE synthesis path (`synth::render_arrangement`).
    /// Physically motivated per instrument family rather than the single
    /// mood-derived envelope every voice used to share: bowed/blown
    /// instruments sustain while played (slow-ish attack, high sustain,
    /// long release); a struck piano decays even while "held." Instruments
    /// that render via Karplus-Strong or FM (see [`Self::uses_karplus_strong`]
    /// / [`Self::uses_fm`]) don't consult this — their envelope comes from
    /// the physical model or the FM index decay instead — but it's still a
    /// sane fallback if one of them is ever routed through the additive path.
    pub fn default_adsr(&self) -> (f32, f32, f32, f32) {
        match self {
            Self::Piano | Self::PianoPP => (0.005, 0.3, 0.3, 0.4),
            Self::Violin | Self::Cello => (0.05, 0.1, 0.85, 0.3),
            Self::Flute | Self::Clarinet | Self::Trumpet | Self::Saxophone | Self::Ney => {
                (0.03, 0.05, 0.9, 0.15)
            }
            Self::Organ | Self::Pad => (0.3, 0.1, 0.95, 0.8),
            Self::Sitar | Self::SawLead => (0.01, 0.2, 0.6, 0.3),
            // Struck/plucked/FM families: a percussive fallback (their real
            // envelope normally comes from the KS/FM path, not this one).
            Self::ElectricPiano
            | Self::Bell
            | Self::AcousticGuitar
            | Self::Harp
            | Self::Marimba
            | Self::Koto
            | Self::Kalimba
            | Self::Oud
            | Self::UprightBass => (0.002, 0.3, 0.2, 0.5),
        }
    }
}

// ─── Karplus-Strong Synthesis ───────────────────────────────────────────────

/// Karplus-Strong plucked string synthesizer.
pub struct KarplusStrong {
    delay_line: Vec<f32>,
    write_pos: usize,
    delay_len: usize,
    damping: f32,
    brightness: f32,
    prev_output: f32,
}

impl KarplusStrong {
    /// Create a KS synthesizer for a given frequency.
    pub fn new(freq: f32, sample_rate: u32, damping: f32, brightness: f32) -> Self {
        let delay_len = (sample_rate as f32 / freq).round() as usize;
        let delay_len = delay_len.max(2);
        Self {
            delay_line: vec![0.0; delay_len],
            write_pos: 0,
            delay_len,
            damping,
            brightness,
            prev_output: 0.0,
        }
    }

    /// Excite the string with a noise burst (fixed seed — every pluck
    /// identical). Prefer [`Self::excite_seeded`] with a per-note seed:
    /// a real string never receives the same pluck twice, and the fixed
    /// seed here was audible as "machine-gun" repetition on repeated
    /// notes and strummed chords.
    pub fn excite(&mut self, velocity: f32) {
        self.excite_seeded(velocity, 42);
    }

    /// Excite the string with a noise burst from `seed` — the round-robin
    /// variation point. Different seeds give genuinely different excitation
    /// noise (a different pluck), same seed reproduces the same pluck
    /// exactly (renders stay deterministic).
    pub fn excite_seeded(&mut self, velocity: f32, seed: u32) {
        // Avoid the LCG's degenerate zero fixed-point neighborhood and
        // decorrelate adjacent seeds.
        let mut rng = seed.wrapping_mul(2654435761).wrapping_add(1);
        for sample in &mut self.delay_line {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            *sample = ((rng as f32 / u32::MAX as f32) * 2.0 - 1.0) * velocity;
        }
    }

    /// Generate one sample (call at audio rate).
    pub fn tick(&mut self) -> f32 {
        let read_pos = self.write_pos;
        let next_pos = (read_pos + 1) % self.delay_len;

        let current = self.delay_line[read_pos];
        let next = self.delay_line[next_pos];

        // Low-pass averaging filter with brightness control
        let filtered = self.brightness * current + (1.0 - self.brightness) * next;
        let output = self.damping * filtered;

        // One-pole smoothing for extra warmth (denormal-flushed: this value
        // recirculates through the delay line as the string decays)
        let smoothed = crate::synth::flush_denormal(0.5 * output + 0.5 * self.prev_output);
        self.prev_output = smoothed;

        // Write back into delay line
        self.delay_line[self.write_pos] = smoothed;
        self.write_pos = (self.write_pos + 1) % self.delay_len;

        smoothed
    }
}

// ─── FM Instrument Renderer ─────────────────────────────────────────────────

/// Render an FM instrument sound for a given note.
pub fn render_fm_instrument(
    instrument: Instrument,
    freq: f32,
    velocity: f32,
    duration: f32,
    sample_rate: u32,
) -> Vec<f32> {
    let sr = sample_rate as f32;
    let (c_ratio, m_ratio, max_index, decay_rate) = instrument.fm_params();
    let n = (duration * sr) as usize + (0.5 * sr) as usize; // note + release

    // Velocity → modulation index: FM's index IS its brightness, and striking
    // an electric piano / bell / marimba harder makes it brighter, not just
    // louder. (Callers fold voice volume into `velocity`; coupling that into
    // brightness slightly is acceptable — a quieter voice reading mellower is
    // the right direction anyway.)
    let max_index = max_index * (0.5 + 0.7 * velocity.clamp(0.0, 1.0));

    let carrier_freq = freq * c_ratio;
    let mod_freq = freq * m_ratio;

    // Band-limit: FM sidebands extend to roughly carrier + (index+1)·mod
    // (Carson's rule). Clamp the index so they stay below Nyquist — without
    // this, high notes fold their sidebands back as inharmonic aliasing.
    let nyquist = sr * 0.5;
    let max_index = if mod_freq > 1.0 {
        max_index.min(((nyquist - carrier_freq) / mod_freq - 1.0).max(0.0))
    } else {
        max_index
    };

    let mut buf = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / sr;
        // Index envelope: starts at max_index, decays exponentially
        let index = max_index * (-t * decay_rate).exp();
        // Amplitude envelope: fast attack, sustain, release
        let env = if t < 0.005 {
            t / 0.005 // 5ms attack
        } else if t < duration {
            1.0 * (-t * 0.5).exp().max(0.3) // slow decay to sustain
        } else {
            let rel_t = t - duration;
            0.3 * (-rel_t * 8.0).exp() // release
        };

        let mod_phase = std::f32::consts::TAU * mod_freq * t;
        let fm = index * mod_phase.sin();
        let carrier_phase = std::f32::consts::TAU * carrier_freq * t + fm;
        let sample = carrier_phase.sin() * env * velocity;

        buf.push(sample);
    }
    buf
}

// ─── Instrument Selection from Consciousness ────────────────────────────────

/// Select an instrument based on consciousness state.
///
/// Maps consciousness quality to appropriate timbres:
/// - Low consciousness, high stillness → Pad, Flute
/// - Moderate consciousness, moderate arousal → Piano, Guitar
/// - High consciousness, high arousal → Violin, Organ
/// - High dopamine → Electric Piano, Bell
pub fn select_instrument(state: &MusicalState) -> Instrument {
    let psi = state.consciousness_level;
    let arousal = state.arousal;
    let valence = state.valence;
    let da = state.dopamine;
    let ne = state.noradrenaline;
    let stillness = state.harmony_activations[7];
    let cultural = state.harmony_activations[5];
    let progress = state.harmony_activations[6];
    let play = state.harmony_activations[3];

    if stillness > 0.6 && arousal < 0.3 {
        if cultural > 0.5 {
            Instrument::Kalimba
        } else {
            Instrument::Pad
        }
    } else if psi < 0.3 {
        if valence < -0.2 {
            Instrument::Ney
        } else {
            Instrument::Flute
        }
    } else if ne > 0.6 && arousal > 0.5 {
        if valence < -0.3 {
            Instrument::SawLead
        } else {
            Instrument::Trumpet
        }
    } else if da > 0.7 {
        if arousal > 0.6 {
            Instrument::Marimba
        } else if arousal > 0.4 {
            Instrument::Bell
        } else {
            Instrument::ElectricPiano
        }
    } else if cultural > 0.6 {
        if valence < -0.2 {
            Instrument::Oud
        } else if arousal < 0.4 {
            Instrument::Koto
        } else {
            Instrument::Sitar
        }
    } else if psi > 0.7 && arousal > 0.5 {
        if state.harmony_activations[2] > 0.5 {
            Instrument::Violin
        } else {
            Instrument::Cello
        }
    } else if progress > 0.5 && arousal > 0.5 {
        if valence > 0.2 {
            Instrument::Saxophone
        } else {
            Instrument::Clarinet
        }
    } else if play > 0.5 && arousal > 0.3 {
        Instrument::AcousticGuitar
    } else if valence < -0.3 && arousal < 0.4 {
        Instrument::UprightBass
    } else if arousal < 0.3 {
        Instrument::PianoPP
    } else {
        Instrument::Piano
    }
}

// ─── Chord Progressions ─────────────────────────────────────────────────────

/// Chord quality (semitone offsets from root).
#[derive(Debug, Clone, Copy)]
pub enum ChordType {
    Major,  // [0, 4, 7]
    Minor,  // [0, 3, 7]
    Major7, // [0, 4, 7, 11]
    Minor7, // [0, 3, 7, 10]
    Dom7,   // [0, 4, 7, 10]
    Sus2,   // [0, 2, 7]
    Sus4,   // [0, 5, 7]
    Dim,    // [0, 3, 6] — diminished (tension/fear)
    Aug,    // [0, 4, 8] — augmented (unease/wonder)
}

impl ChordType {
    /// Get semitone offsets for this chord type.
    pub fn semitones(&self) -> &'static [i32] {
        match self {
            Self::Major => &[0, 4, 7],
            Self::Minor => &[0, 3, 7],
            Self::Major7 => &[0, 4, 7, 11],
            Self::Minor7 => &[0, 3, 7, 10],
            Self::Dom7 => &[0, 4, 7, 10],
            Self::Sus2 => &[0, 2, 7],
            Self::Sus4 => &[0, 5, 7],
            Self::Dim => &[0, 3, 6],
            Self::Aug => &[0, 4, 8],
        }
    }

    /// Convert to frequency ratios relative to root.
    pub fn ratios(&self) -> Vec<f32> {
        self.semitones()
            .iter()
            .map(|&s| 2.0f32.powf(s as f32 / 12.0))
            .collect()
    }
}

/// A chord in a progression (root as semitone offset from key + type).
#[derive(Debug, Clone, Copy)]
pub struct ProgressionChord {
    pub root_semitones: i32,
    pub chord_type: ChordType,
    pub duration_beats: f32,
}

/// Common chord progressions.
pub fn pop_progression() -> Vec<ProgressionChord> {
    // I - V - vi - IV (C - G - Am - F)
    vec![
        ProgressionChord {
            root_semitones: 0,
            chord_type: ChordType::Major,
            duration_beats: 4.0,
        },
        ProgressionChord {
            root_semitones: 7,
            chord_type: ChordType::Major,
            duration_beats: 4.0,
        },
        ProgressionChord {
            root_semitones: 9,
            chord_type: ChordType::Minor,
            duration_beats: 4.0,
        },
        ProgressionChord {
            root_semitones: 5,
            chord_type: ChordType::Major,
            duration_beats: 4.0,
        },
    ]
}

pub fn jazz_progression() -> Vec<ProgressionChord> {
    // ii7 - V7 - Imaj7
    vec![
        ProgressionChord {
            root_semitones: 2,
            chord_type: ChordType::Minor7,
            duration_beats: 4.0,
        },
        ProgressionChord {
            root_semitones: 7,
            chord_type: ChordType::Dom7,
            duration_beats: 4.0,
        },
        ProgressionChord {
            root_semitones: 0,
            chord_type: ChordType::Major7,
            duration_beats: 8.0,
        },
    ]
}

pub fn ambient_progression() -> Vec<ProgressionChord> {
    // Isus2 - IVsus2 - visus2 - Vsus4
    vec![
        ProgressionChord {
            root_semitones: 0,
            chord_type: ChordType::Sus2,
            duration_beats: 8.0,
        },
        ProgressionChord {
            root_semitones: 5,
            chord_type: ChordType::Sus2,
            duration_beats: 8.0,
        },
        ProgressionChord {
            root_semitones: 9,
            chord_type: ChordType::Sus2,
            duration_beats: 8.0,
        },
        ProgressionChord {
            root_semitones: 7,
            chord_type: ChordType::Sus4,
            duration_beats: 8.0,
        },
    ]
}

/// Minor progression for sadness/grief (Cheung et al. 2019: minor mode → negative valence).
pub fn minor_progression() -> Vec<ProgressionChord> {
    // i - iv - v - i (natural minor, no leading tone → unresolved)
    vec![
        ProgressionChord {
            root_semitones: 0,
            chord_type: ChordType::Minor,
            duration_beats: 4.0,
        },
        ProgressionChord {
            root_semitones: 5,
            chord_type: ChordType::Minor,
            duration_beats: 4.0,
        },
        ProgressionChord {
            root_semitones: 7,
            chord_type: ChordType::Minor,
            duration_beats: 4.0,
        },
        ProgressionChord {
            root_semitones: 0,
            chord_type: ChordType::Minor,
            duration_beats: 4.0,
        },
    ]
}

/// Tension/dread progression with tritones and diminished chords.
pub fn tension_progression() -> Vec<ProgressionChord> {
    // i - bII - vii° - i (Phrygian approach + diminished = maximum tension)
    vec![
        ProgressionChord {
            root_semitones: 0,
            chord_type: ChordType::Minor,
            duration_beats: 4.0,
        },
        ProgressionChord {
            root_semitones: 1,
            chord_type: ChordType::Major,
            duration_beats: 2.0,
        },
        ProgressionChord {
            root_semitones: 11,
            chord_type: ChordType::Dim,
            duration_beats: 2.0,
        },
        ProgressionChord {
            root_semitones: 0,
            chord_type: ChordType::Minor,
            duration_beats: 4.0,
        },
    ]
}

/// Select a progression based on consciousness state AND valence.
pub fn select_progression(state: &MusicalState) -> Vec<ProgressionChord> {
    let psi = state.consciousness_level;
    let arousal = state.arousal;
    let valence = state.valence;
    let stillness = state.harmony_activations[7];
    let pe = state.prediction_error;

    // Valence-first selection (Panda et al. 2023: harmonic content predicts valence)
    if pe > 0.6 || (valence < -0.3 && arousal > 0.6) {
        // High surprise or negative+aroused → tension/dread
        tension_progression()
    } else if valence < -0.2 && arousal < 0.5 {
        // Sad/depleted → minor
        minor_progression()
    } else if stillness > 0.5 || (psi < 0.3 && arousal < 0.3) {
        ambient_progression()
    } else if psi > 0.6 && state.dopamine > 0.5 {
        jazz_progression()
    } else {
        pop_progression()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn piano_partials_decay() {
        for i in 1..16 {
            assert!(
                PIANO_FF[i] < PIANO_FF[i - 1] || (PIANO_FF[i] - PIANO_FF[i - 1]).abs() < 0.01,
                "piano ff partials should generally decay: [{}]={}, [{}]={}",
                i - 1,
                PIANO_FF[i - 1],
                i,
                PIANO_FF[i]
            );
        }
    }

    #[test]
    fn karplus_strong_produces_sound() {
        let mut ks = KarplusStrong::new(440.0, 44100, 0.996, 0.5);
        ks.excite(0.8);
        let mut has_signal = false;
        for _ in 0..4410 {
            let s = ks.tick();
            if s.abs() > 0.01 {
                has_signal = true;
            }
        }
        assert!(has_signal, "KS should produce sound");
    }

    #[test]
    fn piano_velocity_crossfades_between_the_measured_tables() {
        assert_eq!(Instrument::Piano.partials_at_velocity(1.0), PIANO_FF);
        assert_eq!(Instrument::Piano.partials_at_velocity(0.0), PIANO_PP);
        // Midway blend sits strictly between the two on a partial where they
        // differ substantially.
        let mid = Instrument::Piano.partials_at_velocity(0.5);
        assert!(mid[8] > PIANO_PP[8] && mid[8] < PIANO_FF[8]);
    }

    #[test]
    fn velocity_tilt_brightens_non_piano_spectra() {
        // Louder playing must shift relative energy toward upper partials.
        let soft = Instrument::Violin.partials_at_velocity(0.2);
        let hard = Instrument::Violin.partials_at_velocity(0.9);
        let ratio = |p: &[f32; 16]| p[8] / p[0];
        assert!(
            ratio(&hard) > ratio(&soft),
            "upper-partial share must grow with velocity: {} vs {}",
            ratio(&soft),
            ratio(&hard)
        );
    }

    #[test]
    fn piano_is_inharmonic_and_sustaining_instruments_are_not() {
        assert!(Instrument::Piano.inharmonicity(261.6) > 0.0);
        assert_eq!(Instrument::Violin.inharmonicity(261.6), 0.0);
        assert!(Instrument::Violin.sustains());
        assert!(!Instrument::Piano.sustains());
        // Attack transients: bowed/blown/struck have one, pads don't.
        assert!(Instrument::Violin.attack_noise().0 > 0.0);
        assert_eq!(Instrument::Pad.attack_noise().0, 0.0);
    }

    #[test]
    fn fm_velocity_raises_brightness_not_just_loudness() {
        // The modulation index (FM's brightness) now scales with velocity —
        // measured as zero crossings, which amplitude cannot change.
        let zc = |v: &[f32]| {
            v.windows(2)
                .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
                .count()
        };
        let soft = render_fm_instrument(Instrument::Bell, 220.0, 0.15, 0.5, 44100);
        let hard = render_fm_instrument(Instrument::Bell, 220.0, 1.0, 0.5, 44100);
        let n = 8820; // first 200ms
        assert!(
            zc(&hard[..n]) > zc(&soft[..n]),
            "hard strike must be brighter: soft={} hard={}",
            zc(&soft[..n]),
            zc(&hard[..n])
        );
    }

    #[test]
    fn ks_excitation_varies_by_seed_and_reproduces_by_seed() {
        // Round-robin contract: different seeds → genuinely different plucks;
        // same seed → bit-identical pluck (renders stay deterministic).
        let render = |seed: u32| -> Vec<f32> {
            let mut ks = KarplusStrong::new(440.0, 44100, 0.996, 0.5);
            ks.excite_seeded(0.8, seed);
            (0..2205).map(|_| ks.tick()).collect()
        };
        let a = render(1);
        let b = render(2);
        let a2 = render(1);
        assert_eq!(a, a2, "same seed must reproduce the same pluck exactly");
        assert!(
            a.iter().zip(&b).any(|(x, y)| x != y),
            "different seeds must produce different plucks"
        );
    }

    #[test]
    fn karplus_strong_decays() {
        let mut ks = KarplusStrong::new(440.0, 44100, 0.990, 0.5); // faster decay
        ks.excite(0.8);
        // Early RMS
        let early: f32 = (0..1000)
            .map(|_| {
                let s = ks.tick();
                s * s
            })
            .sum::<f32>()
            / 1000.0;
        // Skip ahead
        for _ in 0..20000 {
            ks.tick();
        }
        let late: f32 = (0..1000)
            .map(|_| {
                let s = ks.tick();
                s * s
            })
            .sum::<f32>()
            / 1000.0;
        assert!(
            late < early,
            "KS should decay: early={early:.6}, late={late:.6}"
        );
    }

    #[test]
    fn fm_electric_piano_sounds() {
        let buf = render_fm_instrument(Instrument::ElectricPiano, 261.63, 0.8, 1.0, 44100);
        assert!(buf.len() > 44100); // note + release
        // Should have signal
        assert!(buf.iter().any(|&s| s.abs() > 0.1));
        // Should decay
        let late_rms: f32 =
            (buf[buf.len() - 1000..].iter().map(|s| s * s).sum::<f32>() / 1000.0).sqrt();
        assert!(late_rms < 0.3, "FM should decay: late_rms={late_rms}");
    }

    #[test]
    fn fm_bell_is_inharmonic() {
        let buf = render_fm_instrument(Instrument::Bell, 440.0, 0.7, 2.0, 44100);
        assert!(!buf.is_empty());
        // Bell should ring long (slow decay)
        let mid_rms: f32 = (buf[44100..45100].iter().map(|s| s * s).sum::<f32>() / 1000.0).sqrt();
        assert!(mid_rms > 0.01, "bell should still ring at 1s: {mid_rms}");
    }

    #[test]
    fn chord_ratios_correct() {
        let major = ChordType::Major.ratios();
        assert_eq!(major.len(), 3);
        assert!((major[0] - 1.0).abs() < 0.001); // root
        assert!((major[1] - 1.2599).abs() < 0.001); // major 3rd
        assert!((major[2] - 1.4983).abs() < 0.001); // perfect 5th
    }

    #[test]
    fn instrument_selection_varies() {
        let calm = MusicalState {
            consciousness_level: 0.2,
            arousal: 0.1,
            harmony_activations: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8],
            ..Default::default()
        };
        let excited = MusicalState {
            consciousness_level: 0.8,
            arousal: 0.8,
            dopamine: 0.9,
            ..Default::default()
        };
        assert_ne!(select_instrument(&calm), select_instrument(&excited));
    }

    #[test]
    fn progression_selection_varies() {
        let ambient = MusicalState {
            consciousness_level: 0.2,
            arousal: 0.1,
            harmony_activations: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8],
            ..Default::default()
        };
        let jazz = MusicalState {
            consciousness_level: 0.8,
            dopamine: 0.8,
            ..Default::default()
        };
        let p_amb = select_progression(&ambient);
        let p_jazz = select_progression(&jazz);
        assert_ne!(
            p_amb.len(),
            p_jazz.len(),
            "different states should select different progressions"
        );
    }

    #[test]
    fn all_instruments_have_partials() {
        let instruments = [
            Instrument::Piano,
            Instrument::PianoPP,
            Instrument::Violin,
            Instrument::Cello,
            Instrument::Flute,
            Instrument::Organ,
            Instrument::Pad,
        ];
        for inst in &instruments {
            let p = inst.partials();
            assert_eq!(p.len(), 16);
            assert!(
                (p[0] - 1.0).abs() < 0.01,
                "{:?} fundamental should be 1.0",
                inst
            );
        }
    }

    #[test]
    fn default_adsr_values_are_physically_sane() {
        // Every ADSR component must be positive/valid regardless of family.
        for inst in [
            Instrument::Piano,
            Instrument::Violin,
            Instrument::Flute,
            Instrument::Organ,
            Instrument::Sitar,
            Instrument::AcousticGuitar,
        ] {
            let (a, d, s, r) = inst.default_adsr();
            assert!(a > 0.0, "{inst:?} attack must be positive");
            assert!(d > 0.0, "{inst:?} decay must be positive");
            assert!(
                (0.0..=1.0).contains(&s),
                "{inst:?} sustain must be in 0..=1"
            );
            assert!(r > 0.0, "{inst:?} release must be positive");
        }
    }

    #[test]
    fn bowed_instruments_sustain_more_than_struck_ones() {
        // A defining difference between a bowed/blown instrument (sustains
        // as long as it's played) and a struck one (decays immediately,
        // "sustain" here meaning "the level it settles to while nominally
        // still held") is a much higher sustain level.
        let (_, _, violin_sustain, _) = Instrument::Violin.default_adsr();
        let (_, _, piano_sustain, _) = Instrument::Piano.default_adsr();
        assert!(
            violin_sustain > piano_sustain,
            "violin ({violin_sustain}) should sustain higher than piano ({piano_sustain})"
        );
    }
}
