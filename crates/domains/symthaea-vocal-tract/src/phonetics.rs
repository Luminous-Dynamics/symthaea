// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Phonetics utilities and ARPAbet articulatory classifications.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhonemeClass {
    Vowel,
    Stop,
    Fricative,
    Nasal,
    Liquid,
    Glide,
    Affricate,
    Silence,
}

#[derive(Debug, Clone, Copy)]
pub struct ArticulationMetadata {
    pub class: PhonemeClass,
    pub f1: f32,
    pub f2: f32,
    pub f3: f32,
    pub voiced: bool,
}

pub fn canonical_arpabet_symbol(ph: &str) -> Option<&'static str> {
    let clean = ph
        .trim_end_matches(|c: char| c.is_ascii_digit())
        .to_uppercase();
    match clean.as_str() {
        "AA" | "AE" | "AH" | "AO" | "AW" | "AY" | "EH" | "ER" | "EY" | "IH" | "IY" | "OW"
        | "OY" | "UH" | "UW" => Some("VOWEL"),
        "P" | "B" | "T" | "D" | "K" | "G" => Some("STOP"),
        "F" | "V" | "TH" | "DH" | "S" | "Z" | "SH" | "ZH" | "HH" => Some("FRICATIVE"),
        "M" | "N" | "NG" => Some("NASAL"),
        "L" | "R" => Some("LIQUID"),
        "W" | "Y" => Some("GLIDE"),
        "CH" | "JH" => Some("AFFRICATE"),
        "SIL" | "SP" | "" => Some("SILENCE"),
        _ => None,
    }
}

pub fn arpabet_articulation(ph: &str) -> ArticulationMetadata {
    let clean = ph
        .trim_end_matches(|c: char| c.is_ascii_digit())
        .to_uppercase();
    match clean.as_str() {
        "AA" => ArticulationMetadata {
            class: PhonemeClass::Vowel,
            f1: 730.0,
            f2: 1090.0,
            f3: 2440.0,
            voiced: true,
        },
        "AE" => ArticulationMetadata {
            class: PhonemeClass::Vowel,
            f1: 660.0,
            f2: 1720.0,
            f3: 2410.0,
            voiced: true,
        },
        "AH" => ArticulationMetadata {
            class: PhonemeClass::Vowel,
            f1: 520.0,
            f2: 1190.0,
            f3: 2390.0,
            voiced: true,
        },
        "AO" => ArticulationMetadata {
            class: PhonemeClass::Vowel,
            f1: 570.0,
            f2: 840.0,
            f3: 2410.0,
            voiced: true,
        },
        "EH" => ArticulationMetadata {
            class: PhonemeClass::Vowel,
            f1: 530.0,
            f2: 1840.0,
            f3: 2480.0,
            voiced: true,
        },
        "IH" => ArticulationMetadata {
            class: PhonemeClass::Vowel,
            f1: 390.0,
            f2: 1990.0,
            f3: 2550.0,
            voiced: true,
        },
        "IY" => ArticulationMetadata {
            class: PhonemeClass::Vowel,
            f1: 270.0,
            f2: 2290.0,
            f3: 3010.0,
            voiced: true,
        },
        "UH" => ArticulationMetadata {
            class: PhonemeClass::Vowel,
            f1: 440.0,
            f2: 1020.0,
            f3: 2240.0,
            voiced: true,
        },
        "UW" => ArticulationMetadata {
            class: PhonemeClass::Vowel,
            f1: 300.0,
            f2: 870.0,
            f3: 2240.0,
            voiced: true,
        },
        // ER is a monophthong (r-colored vowel) with a real published steady-state
        // formant target: Peterson & Barney (1952), "bird", F1=490/F2=1350/F3=1690 Hz
        // -- same source as the AA/AE/AH/AO/EH/IH/IY/UH/UW entries above.
        "ER" => ArticulationMetadata {
            class: PhonemeClass::Vowel,
            f1: 490.0,
            f2: 1350.0,
            f3: 1690.0,
            voiced: true,
        },
        // AW/AY/EY/OW/OY are genuine diphthongs -- they glide between two target
        // qualities over their duration, which this crate's single static (F1,F2,F3)
        // triple per phoneme cannot represent. Approximated here by each diphthong's
        // ONSET vowel quality (the dominant perceptual target in most reduced/short
        // realizations), reusing this table's own existing monophthong entries rather
        // than inventing new numbers. Cross-checked against real diphthong onset
        // measurements where available: Holbrook & Fairbanks (1962) measured OY's
        // onset at F1~550/F2~800 Hz, matching this crate's own AO entry (570/840)
        // closely -- supporting onset-vowel reuse as a reasonable simplification, not
        // just a guess. A future improvement would model the full onset->offset glide;
        // out of scope for this fix (this table has no time-varying representation).
        "AY" => ArticulationMetadata {
            class: PhonemeClass::Vowel,
            f1: 730.0,
            f2: 1090.0,
            f3: 2440.0,
            voiced: true,
        }, // onset ~ AA ("father")
        "AW" => ArticulationMetadata {
            class: PhonemeClass::Vowel,
            f1: 520.0,
            f2: 1190.0,
            f3: 2390.0,
            voiced: true,
        }, // onset ~ AH ("but")
        "EY" => ArticulationMetadata {
            class: PhonemeClass::Vowel,
            f1: 530.0,
            f2: 1840.0,
            f3: 2480.0,
            voiced: true,
        }, // onset ~ EH ("bet")
        "OW" | "OY" => ArticulationMetadata {
            class: PhonemeClass::Vowel,
            f1: 570.0,
            f2: 840.0,
            f3: 2410.0,
            voiced: true,
        }, // onset ~ AO ("bought") -- matches Holbrook & Fairbanks' OY onset measurement
        "P" | "B" | "T" | "D" | "K" | "G" => ArticulationMetadata {
            class: PhonemeClass::Stop,
            f1: 300.0,
            f2: 1500.0,
            f3: 2500.0,
            voiced: matches!(clean.as_str(), "B" | "D" | "G"),
        },
        "F" | "V" | "TH" | "DH" | "S" | "Z" | "SH" | "ZH" | "HH" => ArticulationMetadata {
            class: PhonemeClass::Fricative,
            f1: 400.0,
            f2: 1800.0,
            f3: 2600.0,
            voiced: matches!(clean.as_str(), "V" | "DH" | "Z" | "ZH"),
        },
        "M" | "N" | "NG" => ArticulationMetadata {
            class: PhonemeClass::Nasal,
            f1: 280.0,
            f2: 1100.0,
            f3: 2200.0,
            voiced: true,
        },
        "L" | "R" => ArticulationMetadata {
            class: PhonemeClass::Liquid,
            f1: 400.0,
            f2: 1200.0,
            f3: 2300.0,
            voiced: true,
        },
        "W" | "Y" => ArticulationMetadata {
            class: PhonemeClass::Glide,
            f1: 300.0,
            f2: 1600.0,
            f3: 2500.0,
            voiced: true,
        },
        "CH" | "JH" => ArticulationMetadata {
            class: PhonemeClass::Affricate,
            f1: 350.0,
            f2: 1700.0,
            f3: 2600.0,
            voiced: clean == "JH",
        },
        _ => ArticulationMetadata {
            class: PhonemeClass::Silence,
            f1: 0.0,
            f2: 0.0,
            f3: 0.0,
            voiced: false,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CANONICAL_NONSILENCE_SYMBOLS: &[&str] = &[
        "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW",
        "P", "B", "T", "D", "K", "G", "F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH", "M", "N",
        "NG", "L", "R", "W", "Y", "CH", "JH",
    ];

    /// Regression for the ARPAbet routing bug found in the 2026-07-29 verification
    /// ledger: `canonical_arpabet_symbol` accepted AW/AY/ER/EY/OW/OY as valid vowels
    /// while `arpabet_articulation` silently classified all six as `Silence` (no
    /// match arm existed), producing zero-formant, unvoiced articulation metadata
    /// for phonemes the crate itself considers valid speech sounds.
    #[test]
    fn every_canonical_symbol_produces_nonsilence_articulation() {
        for &sym in CANONICAL_NONSILENCE_SYMBOLS {
            assert!(
                canonical_arpabet_symbol(sym).is_some(),
                "{sym} should be a recognized canonical symbol"
            );
            let meta = arpabet_articulation(sym);
            assert_ne!(
                meta.class,
                PhonemeClass::Silence,
                "{sym} is a canonical non-silence phoneme but routed to Silence"
            );
            assert!(
                meta.f1 > 0.0 && meta.f2 > 0.0 && meta.f3 > 0.0,
                "{sym} produced zero formants: {meta:?}"
            );
        }
    }

    /// Stressed forms (e.g. "AY1") must normalize identically to their base symbol,
    /// not silently fall through the same way an unrecognized symbol would.
    #[test]
    fn stressed_forms_normalize_to_the_same_nonsilence_articulation() {
        for base in ["AA", "AY", "ER", "EY", "OW", "OY", "AW", "IY"] {
            let unstressed = arpabet_articulation(base);
            for stress in ["0", "1", "2"] {
                let stressed = arpabet_articulation(&format!("{base}{stress}"));
                assert_eq!(
                    stressed.class, unstressed.class,
                    "{base}{stress} should classify the same as {base}"
                );
                assert!(
                    (stressed.f1 - unstressed.f1).abs() < f32::EPSILON,
                    "{base}{stress} should have identical F1 to {base}"
                );
            }
        }
    }

    #[test]
    fn unknown_symbol_still_routes_to_silence() {
        // Confirms the wildcard fallback is reached only for genuinely unrecognized
        // input, not for canonical symbols that happen to lack a match arm.
        assert_eq!(canonical_arpabet_symbol("XYZ"), None);
        assert_eq!(arpabet_articulation("XYZ").class, PhonemeClass::Silence);
    }
}
