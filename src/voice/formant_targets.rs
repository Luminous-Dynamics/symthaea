//! # Formant Target Database
//!
//! Acoustic phonetics data for articulatory TTS synthesis.
//!
//! This module provides formant frequency targets (F1, F2, F3) for
//! English phonemes based on acoustic phonetics research. These targets
//! are used by the CfC-based synthesizer to generate smooth formant
//! trajectories during speech synthesis.
//!
//! ## References
//!
//! - Peterson & Barney (1952): Classic vowel formant data
//! - Hillenbrand et al. (1995): Updated measurements
//! - Stevens (1998): Acoustic Phonetics
//!
//! ## Formant Overview
//!
//! - **F1**: Related to tongue height (low vowels have high F1)
//! - **F2**: Related to tongue frontness/backness
//! - **F3**: Related to lip rounding and pharyngeal constriction

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════════
// FORMANT TARGET STRUCTURE
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
}

impl FormantTarget {
    /// Create a vowel formant target
    pub const fn vowel(f1: f32, f2: f32, f3: f32, duration_ms: f32) -> Self {
        Self {
            f1,
            f2,
            f3,
            b1: 60.0,   // Typical bandwidth for F1
            b2: 90.0,   // Typical bandwidth for F2
            b3: 150.0,  // Typical bandwidth for F3
            duration_ms,
            is_vowel: true,
            is_voiced: true,
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
        }
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
            is_vowel: if t < 0.5 { self.is_vowel } else { other.is_vowel },
            is_voiced: if t < 0.5 { self.is_voiced } else { other.is_voiced },
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
            f1: self.f1 * ratio.powf(0.3),  // F1 shifts less than pitch
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

// ═══════════════════════════════════════════════════════════════════════════════
// ARPABET FORMANT DATABASE
// ═══════════════════════════════════════════════════════════════════════════════

/// ARPABET phoneme formant targets
///
/// Based on Peterson & Barney (1952) and Hillenbrand et al. (1995)
/// Values are for adult male; female/child speakers scale by ~15%/30%
pub fn get_formant_database() -> HashMap<String, FormantTarget> {
    let mut db = HashMap::new();

    // ═══════════════════════════════════════════════════════════════════════════
    // VOWELS (Monophthongs)
    // ═══════════════════════════════════════════════════════════════════════════

    // Front vowels
    db.insert("IY".into(), FormantTarget::vowel(270.0, 2290.0, 3010.0, 100.0));  // "beat"
    db.insert("IH".into(), FormantTarget::vowel(390.0, 1990.0, 2550.0, 80.0));   // "bit"
    db.insert("EY".into(), FormantTarget::vowel(476.0, 2089.0, 2691.0, 120.0));  // "bait" (diphthong start)
    db.insert("EH".into(), FormantTarget::vowel(530.0, 1840.0, 2480.0, 80.0));   // "bet"
    db.insert("AE".into(), FormantTarget::vowel(660.0, 1720.0, 2410.0, 100.0));  // "bat"

    // Central vowels
    db.insert("AH".into(), FormantTarget::vowel(520.0, 1190.0, 2390.0, 80.0));   // "but" (stressed)
    db.insert("AX".into(), FormantTarget::vowel(500.0, 1500.0, 2500.0, 60.0));   // schwa (unstressed)
    db.insert("ER".into(), FormantTarget::vowel(490.0, 1350.0, 1690.0, 100.0));  // "bird" (r-colored)
    db.insert("AXR".into(), FormantTarget::vowel(500.0, 1400.0, 1750.0, 80.0));  // unstressed r-colored

    // Back vowels
    db.insert("AA".into(), FormantTarget::vowel(730.0, 1090.0, 2440.0, 100.0));  // "bot"
    db.insert("AO".into(), FormantTarget::vowel(570.0, 840.0, 2410.0, 100.0));   // "bought"
    db.insert("OW".into(), FormantTarget::vowel(497.0, 910.0, 2459.0, 120.0));   // "boat" (diphthong)
    db.insert("UH".into(), FormantTarget::vowel(440.0, 1020.0, 2240.0, 80.0));   // "book"
    db.insert("UW".into(), FormantTarget::vowel(300.0, 870.0, 2240.0, 100.0));   // "boot"

    // Diphthongs
    db.insert("AY".into(), FormantTarget::vowel(710.0, 1100.0, 2540.0, 150.0));  // "bite" (start)
    db.insert("AW".into(), FormantTarget::vowel(698.0, 1109.0, 2459.0, 150.0));  // "bout" (start)
    db.insert("OY".into(), FormantTarget::vowel(570.0, 840.0, 2410.0, 150.0));   // "boy" (start)

    // ═══════════════════════════════════════════════════════════════════════════
    // CONSONANTS - Stops
    // ═══════════════════════════════════════════════════════════════════════════

    // Bilabial stops
    db.insert("P".into(), FormantTarget::unvoiced_consonant(200.0, 1000.0, 2200.0, 60.0));
    db.insert("B".into(), FormantTarget::voiced_consonant(200.0, 1000.0, 2200.0, 60.0));

    // Alveolar stops
    db.insert("T".into(), FormantTarget::unvoiced_consonant(400.0, 1800.0, 2600.0, 50.0));
    db.insert("D".into(), FormantTarget::voiced_consonant(400.0, 1800.0, 2600.0, 50.0));

    // Velar stops
    db.insert("K".into(), FormantTarget::unvoiced_consonant(350.0, 1500.0, 2500.0, 70.0));
    db.insert("G".into(), FormantTarget::voiced_consonant(350.0, 1500.0, 2500.0, 70.0));

    // Glottal stop
    db.insert("Q".into(), FormantTarget::unvoiced_consonant(300.0, 1500.0, 2500.0, 30.0));

    // ═══════════════════════════════════════════════════════════════════════════
    // CONSONANTS - Fricatives
    // ═══════════════════════════════════════════════════════════════════════════

    // Labiodental fricatives
    db.insert("F".into(), FormantTarget::unvoiced_consonant(300.0, 1300.0, 2400.0, 80.0));
    db.insert("V".into(), FormantTarget::voiced_consonant(300.0, 1300.0, 2400.0, 80.0));

    // Dental fricatives
    db.insert("TH".into(), FormantTarget::unvoiced_consonant(300.0, 1500.0, 2500.0, 80.0));  // "think"
    db.insert("DH".into(), FormantTarget::voiced_consonant(300.0, 1500.0, 2500.0, 60.0));    // "this"

    // Alveolar fricatives
    db.insert("S".into(), FormantTarget::unvoiced_consonant(320.0, 1700.0, 2600.0, 100.0));
    db.insert("Z".into(), FormantTarget::voiced_consonant(320.0, 1700.0, 2600.0, 100.0));

    // Postalveolar fricatives
    db.insert("SH".into(), FormantTarget::unvoiced_consonant(300.0, 1900.0, 2700.0, 100.0)); // "ship"
    db.insert("ZH".into(), FormantTarget::voiced_consonant(300.0, 1900.0, 2700.0, 80.0));    // "measure"

    // Glottal fricative
    db.insert("HH".into(), FormantTarget::unvoiced_consonant(500.0, 1500.0, 2500.0, 60.0));

    // ═══════════════════════════════════════════════════════════════════════════
    // CONSONANTS - Affricates
    // ═══════════════════════════════════════════════════════════════════════════

    db.insert("CH".into(), FormantTarget::unvoiced_consonant(300.0, 1800.0, 2800.0, 100.0)); // "church"
    db.insert("JH".into(), FormantTarget::voiced_consonant(300.0, 1800.0, 2800.0, 100.0));   // "judge"

    // ═══════════════════════════════════════════════════════════════════════════
    // CONSONANTS - Nasals
    // ═══════════════════════════════════════════════════════════════════════════

    db.insert("M".into(), FormantTarget::voiced_consonant(280.0, 1000.0, 2200.0, 70.0));
    db.insert("N".into(), FormantTarget::voiced_consonant(280.0, 1500.0, 2500.0, 70.0));
    db.insert("NG".into(), FormantTarget::voiced_consonant(280.0, 1900.0, 2600.0, 70.0));

    // ═══════════════════════════════════════════════════════════════════════════
    // CONSONANTS - Liquids
    // ═══════════════════════════════════════════════════════════════════════════

    db.insert("L".into(), FormantTarget::voiced_consonant(350.0, 1050.0, 2750.0, 70.0));
    db.insert("R".into(), FormantTarget::voiced_consonant(420.0, 1300.0, 1660.0, 70.0));  // Low F3 is characteristic

    // ═══════════════════════════════════════════════════════════════════════════
    // CONSONANTS - Semivowels/Glides
    // ═══════════════════════════════════════════════════════════════════════════

    db.insert("W".into(), FormantTarget::voiced_consonant(300.0, 750.0, 2200.0, 60.0));
    db.insert("Y".into(), FormantTarget::voiced_consonant(280.0, 2200.0, 2960.0, 60.0));

    // ═══════════════════════════════════════════════════════════════════════════
    // SPECIAL SYMBOLS
    // ═══════════════════════════════════════════════════════════════════════════

    // Silence
    db.insert("SIL".into(), FormantTarget {
        f1: 0.0, f2: 0.0, f3: 0.0,
        b1: 0.0, b2: 0.0, b3: 0.0,
        duration_ms: 100.0,
        is_vowel: false,
        is_voiced: false,
    });

    // Short pause
    db.insert("SP".into(), FormantTarget {
        f1: 0.0, f2: 0.0, f3: 0.0,
        b1: 0.0, b2: 0.0, b3: 0.0,
        duration_ms: 50.0,
        is_vowel: false,
        is_voiced: false,
    });

    db
}

// ═══════════════════════════════════════════════════════════════════════════════
// FORMANT DATABASE WRAPPER
// ═══════════════════════════════════════════════════════════════════════════════

/// Formant database with lookup and interpolation
#[derive(Debug, Clone)]
pub struct FormantDatabase {
    targets: HashMap<String, FormantTarget>,
}

impl FormantDatabase {
    /// Create a new database with default ARPABET targets
    pub fn new() -> Self {
        Self {
            targets: get_formant_database(),
        }
    }

    /// Lookup formant target for a phoneme
    pub fn lookup(&self, phoneme: &str) -> Option<&FormantTarget> {
        self.targets.get(phoneme)
    }

    /// Lookup with fallback to schwa
    pub fn lookup_or_default(&self, phoneme: &str) -> FormantTarget {
        self.targets.get(phoneme).copied().unwrap_or_default()
    }

    /// Get transition formants between two phonemes
    ///
    /// Returns interpolated formants for coarticulation
    pub fn get_transition(&self, from: &str, to: &str, position: f32) -> FormantTarget {
        let from_target = self.lookup_or_default(from);
        let to_target = self.lookup_or_default(to);
        from_target.lerp(&to_target, position)
    }

    /// Add or update a phoneme target
    pub fn set(&mut self, phoneme: impl Into<String>, target: FormantTarget) {
        self.targets.insert(phoneme.into(), target);
    }

    /// Scale all formants for a different speaker type
    ///
    /// - Female: scale ~1.15
    /// - Child: scale ~1.30
    pub fn scale_for_speaker(&self, scale: f32) -> Self {
        let mut scaled = HashMap::new();
        for (phoneme, target) in &self.targets {
            scaled.insert(phoneme.clone(), FormantTarget {
                f1: target.f1 * scale,
                f2: target.f2 * scale,
                f3: target.f3 * scale,
                b1: target.b1 * scale,
                b2: target.b2 * scale,
                b3: target.b3 * scale,
                ..*target
            });
        }
        Self { targets: scaled }
    }

    /// Get all vowel phonemes
    pub fn vowels(&self) -> Vec<&str> {
        self.targets
            .iter()
            .filter(|(_, t)| t.is_vowel)
            .map(|(p, _)| p.as_str())
            .collect()
    }

    /// Get all consonant phonemes
    pub fn consonants(&self) -> Vec<&str> {
        self.targets
            .iter()
            .filter(|(_, t)| !t.is_vowel)
            .map(|(p, _)| p.as_str())
            .collect()
    }
}

impl Default for FormantDatabase {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_formant_database_creation() {
        let db = FormantDatabase::new();

        // Check we have vowels
        let vowels = db.vowels();
        assert!(!vowels.is_empty());
        assert!(vowels.contains(&"IY"));
        assert!(vowels.contains(&"AA"));

        // Check we have consonants
        let consonants = db.consonants();
        assert!(!consonants.is_empty());
        assert!(consonants.contains(&"S"));
        assert!(consonants.contains(&"M"));
    }

    #[test]
    fn test_formant_lookup() {
        let db = FormantDatabase::new();

        let iy = db.lookup("IY").expect("IY should exist");
        assert!(iy.f1 < 400.0);  // High vowel = low F1
        assert!(iy.f2 > 2000.0); // Front vowel = high F2
        assert!(iy.is_vowel);
        assert!(iy.is_voiced);

        let s = db.lookup("S").expect("S should exist");
        assert!(!s.is_vowel);
        assert!(!s.is_voiced);
    }

    #[test]
    fn test_formant_interpolation() {
        let db = FormantDatabase::new();

        let transition = db.get_transition("IY", "AA", 0.5);

        let iy = db.lookup("IY").unwrap();
        let aa = db.lookup("AA").unwrap();

        // F1 should be between IY and AA
        assert!(transition.f1 > iy.f1);
        assert!(transition.f1 < aa.f1);
    }

    #[test]
    fn test_speaker_scaling() {
        let db = FormantDatabase::new();
        let female_db = db.scale_for_speaker(1.15);

        let male_iy = db.lookup("IY").unwrap();
        let female_iy = female_db.lookup("IY").unwrap();

        assert!((female_iy.f1 - male_iy.f1 * 1.15).abs() < 1.0);
    }

    #[test]
    fn test_rate_modifier() {
        let target = FormantTarget::vowel(500.0, 1500.0, 2500.0, 100.0);
        let fast = target.with_rate(2.0);

        assert!((fast.duration_ms - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_pitch_shift() {
        let target = FormantTarget::vowel(500.0, 1500.0, 2500.0, 100.0);
        let higher = target.with_pitch_shift(12.0);  // One octave up

        // Formants should shift up, but less than pitch
        assert!(higher.f1 > target.f1);
        assert!(higher.f1 < target.f1 * 2.0);  // Less than octave
    }
}
