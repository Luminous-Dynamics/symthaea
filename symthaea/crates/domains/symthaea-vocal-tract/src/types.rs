// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core vocal tract types: FormantFrame, FormantTarget.
//!
//! Pure data types with no external dependencies beyond serde.

use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════════
// SOURCE TYPE — manner of articulation for vocoder excitation selection
// ═══════════════════════════════════════════════════════════════════════════════

/// Manner of articulation for vocoder source selection.
///
/// The vocoder uses different excitation signals depending on the phoneme's
/// manner of articulation:
/// - **Vowel/Liquid**: Glottal pulse + aspiration + noise mix
/// - **Stop**: Silence (closure) + noise burst
/// - **Fricative**: Shaped noise + optional voicing bar
/// - **Nasal**: Voiced source + nasal anti-formant
/// - **Affricate**: Closure + burst + frication
/// - **Silent**: Zero output
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SourceType {
    /// Vowels and vowel-like sounds (default excitation)
    #[default]
    Vowel,
    /// Plosive stops (P, T, K, B, D, G) — closure + burst
    Stop,
    /// Fricatives (S, SH, F, TH, ...) — shaped noise
    Fricative,
    /// Nasals (M, N, NG) — voiced + anti-formant
    Nasal,
    /// Affricates (CH, JH) — stop + fricative
    Affricate,
    /// Liquids and glides (L, R, W, Y) — vowel-like
    Liquid,
    /// Silence
    Silent,
}

// ═══════════════════════════════════════════════════════════════════════════════
// FORMANT FRAME
// ═══════════════════════════════════════════════════════════════════════════════

/// Single frame of formant parameters for vocoder
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormantFrame {
    /// First formant frequency (Hz)
    pub f1: f32,
    /// Second formant frequency (Hz)
    pub f2: f32,
    /// Third formant frequency (Hz)
    pub f3: f32,
    /// Bandwidth of F1 (Hz)
    pub b1: f32,
    /// Bandwidth of F2 (Hz)
    pub b2: f32,
    /// Bandwidth of F3 (Hz)
    pub b3: f32,
    /// Fundamental frequency / pitch (Hz)
    pub f0: f32,
    /// Energy/amplitude (0.0 to 1.0)
    pub energy: f32,
    /// Voicing probability (0.0 = unvoiced, 1.0 = fully voiced)
    pub voicing: f32,
    /// Timestamp in seconds
    pub time: f32,
    /// Source excitation type (manner of articulation for vocoder).
    pub source_type: SourceType,
    /// Nasal anti-formant (zero) frequency (Hz). 0.0 = no nasal zero.
    pub nasal_zero_freq: f32,
    /// Nasal anti-formant bandwidth (Hz). 0.0 = default 200 Hz.
    pub nasal_zero_bw: f32,
}

impl FormantFrame {
    /// Create a silent frame
    pub fn silent(time: f32) -> Self {
        Self {
            f1: 0.0,
            f2: 0.0,
            f3: 0.0,
            b1: 60.0,
            b2: 90.0,
            b3: 150.0,
            f0: 0.0,
            energy: 0.0,
            voicing: 0.0,
            time,
            source_type: SourceType::Silent,
            nasal_zero_freq: 0.0,
            nasal_zero_bw: 0.0,
        }
    }

    /// Create from formant target with pitch and energy
    pub fn from_target(target: &FormantTarget, f0: f32, energy: f32, time: f32) -> Self {
        Self {
            f1: target.f1,
            f2: target.f2,
            f3: target.f3,
            b1: target.b1,
            b2: target.b2,
            b3: target.b3,
            f0,
            energy,
            voicing: if target.is_voiced { 1.0 } else { 0.0 },
            time,
            source_type: target.manner,
            nasal_zero_freq: target.nasal_zero_freq,
            nasal_zero_bw: target.nasal_zero_bw,
        }
    }

    /// Linear interpolation between frames
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            f1: self.f1 + (other.f1 - self.f1) * t,
            f2: self.f2 + (other.f2 - self.f2) * t,
            f3: self.f3 + (other.f3 - self.f3) * t,
            b1: self.b1 + (other.b1 - self.b1) * t,
            b2: self.b2 + (other.b2 - self.b2) * t,
            b3: self.b3 + (other.b3 - self.b3) * t,
            f0: self.f0 + (other.f0 - self.f0) * t,
            energy: self.energy + (other.energy - self.energy) * t,
            voicing: self.voicing + (other.voicing - self.voicing) * t,
            time: self.time + (other.time - self.time) * t,
            source_type: if t < 0.5 {
                self.source_type
            } else {
                other.source_type
            },
            nasal_zero_freq: self.nasal_zero_freq
                + (other.nasal_zero_freq - self.nasal_zero_freq) * t,
            nasal_zero_bw: self.nasal_zero_bw + (other.nasal_zero_bw - self.nasal_zero_bw) * t,
        }
    }
}

impl Default for FormantFrame {
    fn default() -> Self {
        Self {
            f1: 0.0,
            f2: 0.0,
            f3: 0.0,
            b1: 60.0,
            b2: 90.0,
            b3: 150.0,
            f0: 0.0,
            energy: 0.0,
            voicing: 0.0,
            time: 0.0,
            source_type: SourceType::Silent,
            nasal_zero_freq: 0.0,
            nasal_zero_bw: 0.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FORMANT TARGET
// ═══════════════════════════════════════════════════════════════════════════════

/// Formant target for a single phoneme
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FormantTarget {
    /// First formant frequency (Hz) - tongue height
    pub f1: f32,
    /// Second formant frequency (Hz) - tongue frontness
    pub f2: f32,
    /// Third formant frequency (Hz) - lip rounding
    pub f3: f32,
    /// Bandwidth of F1 (Hz)
    pub b1: f32,
    /// Bandwidth of F2 (Hz)
    pub b2: f32,
    /// Bandwidth of F3 (Hz)
    pub b3: f32,
    /// Typical duration (ms)
    pub duration_ms: f32,
    /// Whether this is a vowel
    pub is_vowel: bool,
    /// Whether this is voiced
    pub is_voiced: bool,
    /// Manner of articulation (determines vocoder source type).
    pub manner: SourceType,
    /// Nasal anti-formant (zero) frequency (Hz). 0.0 = no nasal zero.
    pub nasal_zero_freq: f32,
    /// Nasal anti-formant bandwidth (Hz). 0.0 = default 200 Hz.
    pub nasal_zero_bw: f32,
    /// Diphthong offset F1 (Hz). 0.0 = not a diphthong.
    pub f1_offset: f32,
    /// Diphthong offset F2 (Hz). 0.0 = not a diphthong.
    pub f2_offset: f32,
    /// Diphthong offset F3 (Hz). 0.0 = not a diphthong.
    pub f3_offset: f32,
}

impl FormantTarget {
    /// Create a vowel formant target
    pub const fn vowel(f1: f32, f2: f32, f3: f32, duration_ms: f32) -> Self {
        Self {
            f1,
            f2,
            f3,
            b1: 60.0,
            b2: 90.0,
            b3: 150.0,
            duration_ms,
            is_vowel: true,
            is_voiced: true,
            manner: SourceType::Vowel,
            nasal_zero_freq: 0.0,
            nasal_zero_bw: 0.0,
            f1_offset: 0.0,
            f2_offset: 0.0,
            f3_offset: 0.0,
        }
    }

    /// Create a voiced consonant formant target
    pub const fn voiced_consonant(f1: f32, f2: f32, f3: f32, duration_ms: f32) -> Self {
        Self {
            f1,
            f2,
            f3,
            b1: 80.0,
            b2: 120.0,
            b3: 200.0,
            duration_ms,
            is_vowel: false,
            is_voiced: true,
            manner: SourceType::Liquid,
            nasal_zero_freq: 0.0,
            nasal_zero_bw: 0.0,
            f1_offset: 0.0,
            f2_offset: 0.0,
            f3_offset: 0.0,
        }
    }

    /// Create an unvoiced consonant formant target
    pub const fn unvoiced_consonant(f1: f32, f2: f32, f3: f32, duration_ms: f32) -> Self {
        Self {
            f1,
            f2,
            f3,
            b1: 100.0,
            b2: 150.0,
            b3: 250.0,
            duration_ms,
            is_vowel: false,
            is_voiced: false,
            manner: SourceType::Fricative,
            nasal_zero_freq: 0.0,
            nasal_zero_bw: 0.0,
            f1_offset: 0.0,
            f2_offset: 0.0,
            f3_offset: 0.0,
        }
    }

    /// Override the manner of articulation (builder pattern).
    pub const fn with_manner(mut self, manner: SourceType) -> Self {
        self.manner = manner;
        self
    }

    /// Override formant bandwidths (builder pattern).
    ///
    /// Based on Klatt (1980) and Hawks & Miller (1995):
    /// high/close vowels have narrower bandwidths, open vowels wider.
    pub const fn with_bandwidths(mut self, b1: f32, b2: f32, b3: f32) -> Self {
        self.b1 = b1;
        self.b2 = b2;
        self.b3 = b3;
        self
    }

    /// Set nasal anti-formant (zero) frequency and bandwidth (builder pattern).
    ///
    /// For nasal consonants, the oral cavity produces a zero (anti-resonance)
    /// whose frequency depends on the place of articulation:
    /// - M (bilabial): ~750 Hz
    /// - N (alveolar): ~1450 Hz
    /// - NG (velar): ~3000 Hz
    pub const fn with_nasal_zero(mut self, freq: f32, bw: f32) -> Self {
        self.nasal_zero_freq = freq;
        self.nasal_zero_bw = bw;
        self
    }

    /// Create a diphthong target with onset and offset formant values.
    /// The offset frequencies represent the end-point of the diphthong glide.
    /// `f1_offset`, `f2_offset`, `f3_offset` are the endpoint formant frequencies.
    pub const fn with_diphthong_offset(
        mut self,
        f1_offset: f32,
        f2_offset: f32,
        f3_offset: f32,
    ) -> Self {
        self.f1_offset = f1_offset;
        self.f2_offset = f2_offset;
        self.f3_offset = f3_offset;
        self
    }

    /// Whether this target represents a diphthong (has offset formants).
    pub fn is_diphthong(&self) -> bool {
        self.f1_offset > 0.0
    }

    /// Interpolate between two formant targets
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            f1: self.f1 + (other.f1 - self.f1) * t,
            f2: self.f2 + (other.f2 - self.f2) * t,
            f3: self.f3 + (other.f3 - self.f3) * t,
            b1: self.b1 + (other.b1 - self.b1) * t,
            b2: self.b2 + (other.b2 - self.b2) * t,
            b3: self.b3 + (other.b3 - self.b3) * t,
            duration_ms: self.duration_ms + (other.duration_ms - self.duration_ms) * t,
            is_vowel: if t < 0.5 {
                self.is_vowel
            } else {
                other.is_vowel
            },
            is_voiced: if t < 0.5 {
                self.is_voiced
            } else {
                other.is_voiced
            },
            manner: if t < 0.5 { self.manner } else { other.manner },
            nasal_zero_freq: self.nasal_zero_freq
                + (other.nasal_zero_freq - self.nasal_zero_freq) * t,
            nasal_zero_bw: self.nasal_zero_bw + (other.nasal_zero_bw - self.nasal_zero_bw) * t,
            f1_offset: self.f1_offset + (other.f1_offset - self.f1_offset) * t,
            f2_offset: self.f2_offset + (other.f2_offset - self.f2_offset) * t,
            f3_offset: self.f3_offset + (other.f3_offset - self.f3_offset) * t,
        }
    }

    /// Apply speaking rate modifier
    pub fn with_rate(&self, rate: f32) -> Self {
        let mut result = *self;
        result.duration_ms /= rate.max(0.1);
        result
    }

    /// Apply pitch modifier (shifts formants slightly)
    pub fn with_pitch_shift(&self, semitones: f32) -> Self {
        let ratio = 2.0_f32.powf(semitones / 12.0);
        Self {
            f1: self.f1 * ratio.powf(0.3),
            f2: self.f2 * ratio.powf(0.2),
            f3: self.f3 * ratio.powf(0.1),
            ..*self
        }
    }
}

impl Default for FormantTarget {
    fn default() -> Self {
        // Schwa (neutral vowel)
        Self::vowel(500.0, 1500.0, 2500.0, 80.0)
    }
}
