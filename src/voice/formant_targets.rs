// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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

// ═══════════════════════════════════════════════════════════════════════════════
// FORMANT TARGET — canonical definition in symthaea-vocal-tract sub-crate
// ═══════════════════════════════════════════════════════════════════════════════

pub use symthaea_vocal_tract::types::{FormantTarget, SourceType};

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

    // Front vowels — bandwidths from Klatt (1980) / Hawks & Miller (1995)
    db.insert(
        "IY".into(),
        FormantTarget::vowel(270.0, 2290.0, 3010.0, 100.0).with_bandwidths(40.0, 70.0, 130.0),
    ); // "beat" — high tongue, tight coupling
    db.insert(
        "IH".into(),
        FormantTarget::vowel(390.0, 1990.0, 2550.0, 80.0).with_bandwidths(50.0, 80.0, 140.0),
    ); // "bit"
    db.insert(
        "EY".into(),
        FormantTarget::vowel(530.0, 1840.0, 2480.0, 130.0)
            .with_bandwidths(50.0, 80.0, 140.0)
            .with_diphthong_offset(270.0, 2290.0, 3010.0),
    ); // "day" — onset ~EH, offset ~IY
    db.insert(
        "EH".into(),
        FormantTarget::vowel(530.0, 1840.0, 2480.0, 80.0),
    ); // "bet" — mid (default 60/90/150)
    db.insert(
        "AE".into(),
        FormantTarget::vowel(660.0, 1720.0, 2410.0, 100.0).with_bandwidths(80.0, 100.0, 160.0),
    ); // "bat" — lower tongue, wider

    // Central vowels
    db.insert(
        "AH".into(),
        FormantTarget::vowel(520.0, 1190.0, 2390.0, 80.0).with_bandwidths(70.0, 90.0, 150.0),
    ); // "but" (stressed)
    db.insert(
        "AX".into(),
        FormantTarget::vowel(500.0, 1500.0, 2500.0, 60.0),
    ); // schwa (unchanged)
    db.insert(
        "ER".into(),
        FormantTarget::vowel(490.0, 1350.0, 1690.0, 100.0).with_bandwidths(70.0, 100.0, 200.0),
    ); // "bird" — R-coloring widens B3
    db.insert(
        "AXR".into(),
        FormantTarget::vowel(500.0, 1400.0, 1750.0, 80.0).with_bandwidths(70.0, 100.0, 200.0),
    ); // unstressed r-colored

    // Back vowels
    db.insert(
        "AA".into(),
        FormantTarget::vowel(730.0, 1090.0, 2440.0, 100.0).with_bandwidths(100.0, 110.0, 170.0),
    ); // "bot" — open, loose coupling
    db.insert(
        "AO".into(),
        FormantTarget::vowel(570.0, 840.0, 2410.0, 100.0).with_bandwidths(90.0, 100.0, 160.0),
    ); // "bought"
    db.insert(
        "OW".into(),
        FormantTarget::vowel(570.0, 840.0, 2410.0, 130.0)
            .with_bandwidths(80.0, 90.0, 150.0)
            .with_diphthong_offset(300.0, 870.0, 2240.0),
    ); // "go" — onset ~AO, offset ~UW
    db.insert(
        "UH".into(),
        FormantTarget::vowel(440.0, 1020.0, 2240.0, 80.0).with_bandwidths(60.0, 80.0, 140.0),
    ); // "book" — tight rounding
    db.insert(
        "UW".into(),
        FormantTarget::vowel(300.0, 870.0, 2240.0, 100.0).with_bandwidths(45.0, 75.0, 135.0),
    ); // "boot" — very tight rounding

    // Diphthongs
    db.insert(
        "AY".into(),
        FormantTarget::vowel(730.0, 1090.0, 2440.0, 150.0)
            .with_bandwidths(80.0, 90.0, 150.0)
            .with_diphthong_offset(270.0, 2290.0, 3010.0),
    ); // "my" — onset ~AA, offset ~IY
    db.insert(
        "AW".into(),
        FormantTarget::vowel(730.0, 1090.0, 2440.0, 150.0)
            .with_bandwidths(80.0, 90.0, 150.0)
            .with_diphthong_offset(300.0, 870.0, 2240.0),
    ); // "how" — onset ~AA, offset ~UW
    db.insert(
        "OY".into(),
        FormantTarget::vowel(570.0, 840.0, 2410.0, 150.0)
            .with_bandwidths(80.0, 90.0, 150.0)
            .with_diphthong_offset(270.0, 2290.0, 3010.0),
    ); // "boy" — onset ~AO, offset ~IY

    // ═══════════════════════════════════════════════════════════════════════════
    // CONSONANTS - Stops
    // ═══════════════════════════════════════════════════════════════════════════

    // Bilabial stops
    db.insert(
        "P".into(),
        FormantTarget::unvoiced_consonant(200.0, 1000.0, 2200.0, 60.0)
            .with_manner(SourceType::Stop),
    );
    db.insert(
        "B".into(),
        FormantTarget::voiced_consonant(200.0, 1000.0, 2200.0, 60.0).with_manner(SourceType::Stop),
    );

    // Alveolar stops
    db.insert(
        "T".into(),
        FormantTarget::unvoiced_consonant(400.0, 1800.0, 2600.0, 50.0)
            .with_manner(SourceType::Stop),
    );
    db.insert(
        "D".into(),
        FormantTarget::voiced_consonant(400.0, 1800.0, 2600.0, 50.0).with_manner(SourceType::Stop),
    );

    // Velar stops
    db.insert(
        "K".into(),
        FormantTarget::unvoiced_consonant(350.0, 1500.0, 2500.0, 70.0)
            .with_manner(SourceType::Stop),
    );
    db.insert(
        "G".into(),
        FormantTarget::voiced_consonant(350.0, 1500.0, 2500.0, 70.0).with_manner(SourceType::Stop),
    );

    // Glottal stop
    db.insert(
        "Q".into(),
        FormantTarget::unvoiced_consonant(300.0, 1500.0, 2500.0, 30.0)
            .with_manner(SourceType::Stop),
    );

    // ═══════════════════════════════════════════════════════════════════════════
    // CONSONANTS - Fricatives
    // ═══════════════════════════════════════════════════════════════════════════

    // Labiodental fricatives
    db.insert(
        "F".into(),
        FormantTarget::unvoiced_consonant(300.0, 1300.0, 2400.0, 80.0),
    );
    db.insert(
        "V".into(),
        FormantTarget::voiced_consonant(300.0, 1300.0, 2400.0, 80.0),
    );

    // Dental fricatives
    db.insert(
        "TH".into(),
        FormantTarget::unvoiced_consonant(300.0, 1500.0, 2500.0, 80.0),
    ); // "think"
    db.insert(
        "DH".into(),
        FormantTarget::voiced_consonant(300.0, 1500.0, 2500.0, 60.0),
    ); // "this"

    // Alveolar fricatives
    db.insert(
        "S".into(),
        FormantTarget::unvoiced_consonant(320.0, 1700.0, 2600.0, 100.0),
    );
    db.insert(
        "Z".into(),
        FormantTarget::voiced_consonant(320.0, 1700.0, 2600.0, 100.0),
    );

    // Postalveolar fricatives
    db.insert(
        "SH".into(),
        FormantTarget::unvoiced_consonant(300.0, 1900.0, 2700.0, 100.0),
    ); // "ship"
    db.insert(
        "ZH".into(),
        FormantTarget::voiced_consonant(300.0, 1900.0, 2700.0, 80.0),
    ); // "measure"

    // Glottal fricative
    db.insert(
        "HH".into(),
        FormantTarget::unvoiced_consonant(500.0, 1500.0, 2500.0, 60.0),
    );

    // ═══════════════════════════════════════════════════════════════════════════
    // CONSONANTS - Affricates
    // ═══════════════════════════════════════════════════════════════════════════

    db.insert(
        "CH".into(),
        FormantTarget::unvoiced_consonant(300.0, 1800.0, 2800.0, 100.0)
            .with_manner(SourceType::Affricate),
    ); // "church"
    db.insert(
        "JH".into(),
        FormantTarget::voiced_consonant(300.0, 1800.0, 2800.0, 100.0)
            .with_manner(SourceType::Affricate),
    ); // "judge"

    // ═══════════════════════════════════════════════════════════════════════════
    // CONSONANTS - Nasals
    // ═══════════════════════════════════════════════════════════════════════════

    db.insert(
        "M".into(),
        FormantTarget::voiced_consonant(280.0, 1000.0, 2200.0, 70.0)
            .with_manner(SourceType::Nasal)
            .with_nasal_zero(750.0, 200.0),
    ); // Bilabial — zero from oral cavity
    db.insert(
        "N".into(),
        FormantTarget::voiced_consonant(280.0, 1500.0, 2500.0, 70.0)
            .with_manner(SourceType::Nasal)
            .with_nasal_zero(1450.0, 250.0),
    ); // Alveolar — zero near F2
    db.insert(
        "NG".into(),
        FormantTarget::voiced_consonant(280.0, 1900.0, 2600.0, 70.0)
            .with_manner(SourceType::Nasal)
            .with_nasal_zero(3000.0, 300.0),
    ); // Velar — zero in F3 region

    // ═══════════════════════════════════════════════════════════════════════════
    // CONSONANTS - Liquids
    // ═══════════════════════════════════════════════════════════════════════════

    db.insert(
        "L".into(),
        FormantTarget::voiced_consonant(350.0, 1050.0, 2750.0, 70.0),
    );
    db.insert(
        "R".into(),
        FormantTarget::voiced_consonant(420.0, 1300.0, 1660.0, 70.0),
    ); // Low F3 is characteristic

    // ═══════════════════════════════════════════════════════════════════════════
    // CONSONANTS - Semivowels/Glides
    // ═══════════════════════════════════════════════════════════════════════════

    db.insert(
        "W".into(),
        FormantTarget::voiced_consonant(300.0, 750.0, 2200.0, 60.0),
    );
    db.insert(
        "Y".into(),
        FormantTarget::voiced_consonant(280.0, 2200.0, 2960.0, 60.0),
    );

    // ═══════════════════════════════════════════════════════════════════════════
    // SPECIAL SYMBOLS
    // ═══════════════════════════════════════════════════════════════════════════

    // Silence
    db.insert(
        "SIL".into(),
        FormantTarget {
            f1: 0.0,
            f2: 0.0,
            f3: 0.0,
            b1: 0.0,
            b2: 0.0,
            b3: 0.0,
            duration_ms: 100.0,
            is_vowel: false,
            is_voiced: false,
            manner: SourceType::Silent,
            nasal_zero_freq: 0.0,
            nasal_zero_bw: 0.0,
            f1_offset: 0.0,
            f2_offset: 0.0,
            f3_offset: 0.0,
        },
    );

    // Short pause
    db.insert(
        "SP".into(),
        FormantTarget {
            f1: 0.0,
            f2: 0.0,
            f3: 0.0,
            b1: 0.0,
            b2: 0.0,
            b3: 0.0,
            duration_ms: 50.0,
            is_vowel: false,
            is_voiced: false,
            manner: SourceType::Silent,
            nasal_zero_freq: 0.0,
            nasal_zero_bw: 0.0,
            f1_offset: 0.0,
            f2_offset: 0.0,
            f3_offset: 0.0,
        },
    );

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

    /// Get all phoneme names in the database.
    pub fn all_phonemes(&self) -> Vec<String> {
        let mut names: Vec<String> = self.targets.keys().cloned().collect();
        names.sort();
        names
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
            scaled.insert(
                phoneme.clone(),
                FormantTarget {
                    f1: target.f1 * scale,
                    f2: target.f2 * scale,
                    f3: target.f3 * scale,
                    b1: target.b1 * scale,
                    b2: target.b2 * scale,
                    b3: target.b3 * scale,
                    nasal_zero_freq: target.nasal_zero_freq * scale,
                    f1_offset: target.f1_offset * scale,
                    f2_offset: target.f2_offset * scale,
                    f3_offset: target.f3_offset * scale,
                    manner: target.manner,
                    ..*target
                },
            );
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
        assert!(iy.f1 < 400.0); // High vowel = low F1
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
        let higher = target.with_pitch_shift(12.0); // One octave up

        // Formants should shift up, but less than pitch
        assert!(higher.f1 > target.f1);
        assert!(higher.f1 < target.f1 * 2.0); // Less than octave
    }

    // ═══════════════════════════════════════════════════════════════════
    // Item 4: Bandwidth training tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_bandwidth_database_variety() {
        let db = FormantDatabase::new();
        // Not all vowels should have B1=60 (default)
        let iy = db.lookup("IY").unwrap();
        let aa = db.lookup("AA").unwrap();
        assert_ne!(
            iy.b1 as u32, aa.b1 as u32,
            "IY and AA should have different B1: IY={}, AA={}",
            iy.b1, aa.b1
        );
    }

    #[test]
    fn test_bandwidth_iy_narrow() {
        let db = FormantDatabase::new();
        let iy = db.lookup("IY").unwrap();
        assert!(
            iy.b1 < 50.0,
            "IY B1 should be narrow (high tongue): B1={}",
            iy.b1
        );
    }

    #[test]
    fn test_bandwidth_aa_wide() {
        let db = FormantDatabase::new();
        let aa = db.lookup("AA").unwrap();
        assert!(
            aa.b1 > 90.0,
            "AA B1 should be wide (open vowel): B1={}",
            aa.b1
        );
    }

    #[test]
    fn test_backward_compat_vowel_constructor() {
        // FormantTarget::vowel() still produces default bandwidths
        let target = FormantTarget::vowel(500.0, 1500.0, 2500.0, 100.0);
        assert!(
            (target.b1 - 60.0).abs() < 0.01,
            "vowel() default B1 should be 60: {}",
            target.b1
        );
        assert!(
            (target.b2 - 90.0).abs() < 0.01,
            "vowel() default B2 should be 90: {}",
            target.b2
        );
        assert!(
            (target.b3 - 150.0).abs() < 0.01,
            "vowel() default B3 should be 150: {}",
            target.b3
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    // Item 5: Nasal pole-zero tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_nasal_m_vs_n_differ() {
        let db = FormantDatabase::new();
        let m = db.lookup("M").unwrap();
        let n = db.lookup("N").unwrap();
        assert!(
            (m.nasal_zero_freq - n.nasal_zero_freq).abs() > 100.0,
            "M and N should have different zero freqs: M={}, N={}",
            m.nasal_zero_freq,
            n.nasal_zero_freq
        );
    }

    #[test]
    fn test_nasal_backward_compat() {
        // Non-nasal phonemes should have nasal_zero_freq=0.0
        let db = FormantDatabase::new();
        let iy = db.lookup("IY").unwrap();
        assert!(
            iy.nasal_zero_freq == 0.0,
            "Non-nasal phoneme should have zero nasal freq: {}",
            iy.nasal_zero_freq
        );
    }

    #[test]
    fn test_nasal_zero_freq_from_target() {
        use symthaea_vocal_tract::types::FormantFrame;
        let db = FormantDatabase::new();
        let m = db.lookup("M").unwrap();
        let frame = FormantFrame::from_target(m, 120.0, 0.7, 0.0);
        assert!(
            frame.nasal_zero_freq > 0.0,
            "FormantFrame should carry nasal zero freq from target: {}",
            frame.nasal_zero_freq
        );
        assert!(
            (frame.nasal_zero_freq - 750.0).abs() < 1.0,
            "M nasal zero should be ~750 Hz: {}",
            frame.nasal_zero_freq
        );
    }

    #[test]
    fn test_nasal_interpolation() {
        use symthaea_vocal_tract::types::FormantFrame;
        let db = FormantDatabase::new();
        let m = db.lookup("M").unwrap();
        let iy = db.lookup("IY").unwrap();

        let frame_m = FormantFrame::from_target(m, 120.0, 0.7, 0.0);
        let frame_iy = FormantFrame::from_target(iy, 120.0, 0.7, 0.1);

        let mid = frame_m.lerp(&frame_iy, 0.5);
        // Should interpolate: M has 750 Hz, IY has 0 Hz → mid ~375
        assert!(
            mid.nasal_zero_freq > 0.0 && mid.nasal_zero_freq < frame_m.nasal_zero_freq,
            "Interpolated nasal zero should be between M and IY: {}",
            mid.nasal_zero_freq
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    // Diphthong trajectory tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_diphthong_ay_has_offset() {
        let db = FormantDatabase::new();
        let ay = db.lookup("AY").unwrap();
        assert!(ay.is_diphthong(), "AY should be a diphthong");
        assert!(ay.f1_offset > 0.0, "AY should have offset F1");
        // AY onset is near AA, offset is near IY
        assert!(ay.f1 > 600.0, "AY onset F1 should be high (AA-like)");
        assert!(ay.f1_offset < 350.0, "AY offset F1 should be low (IY-like)");
    }

    #[test]
    fn test_diphthong_offset_f2_movement() {
        let db = FormantDatabase::new();
        let ay = db.lookup("AY").unwrap();
        // AY has large F2 movement (low->high)
        let f2_delta = (ay.f2_offset - ay.f2).abs();
        assert!(
            f2_delta > 500.0,
            "AY should have >500Hz F2 movement, got {f2_delta}"
        );
    }

    #[test]
    fn test_monophthong_not_diphthong() {
        let db = FormantDatabase::new();
        let iy = db.lookup("IY").unwrap();
        assert!(!iy.is_diphthong(), "IY should not be a diphthong");
        assert_eq!(iy.f1_offset, 0.0);
    }

    #[test]
    fn test_all_diphthongs_have_offsets() {
        let db = FormantDatabase::new();
        for ph in &["AY", "AW", "OY", "EY", "OW"] {
            let target = db.lookup(ph).expect(ph);
            assert!(target.is_diphthong(), "{ph} should be a diphthong");
        }
    }
}
