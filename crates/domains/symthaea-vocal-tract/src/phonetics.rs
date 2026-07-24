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
