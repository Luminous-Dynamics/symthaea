// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lightweight speech synthesis: text → IPA phonemes → formant frames → PCM.
//!
//! Ported from the retired `symthaea-voice` crate (2026-07-15, voice plan
//! P0.6): that crate was an orphaned April-2026 fork of the root crate's
//! `src/voice/` with exactly one consumer (`symthaea-muse`'s voice/singing
//! bridges) and a broken embedded CMUdict (a 37-byte gitignored placeholder —
//! the `**/data/**` trap). The pieces muse actually uses live here now, in
//! the crate both sides already depend on; the CMUdict branch was dropped
//! (it was empty at runtime, so behavior is identical), and Kokoro support
//! stays in the root crate's `voice::kokoro_engine`.
//!
//! This is deliberately the *small*, legacy synthesis path (3 resonators, one
//! file). It is not the Series 23 physical articulatory renderer. The root
//! crate's `voice::vocoder::FormantVocoder` (LF glottal model, 5 formants,
//! jitter/shimmer) is the fuller acoustic voice path.

/// Consciousness state that shapes voice prosody.
#[derive(Debug, Clone, Default)]
pub struct VoiceProsody {
    pub arousal: f32,
    pub valence: f32,
    pub consciousness: f32,
    pub serotonin: f32,
}

/// Explicit synthesis backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeechSynthesisBackend {
    /// Lightweight rule-based IPA/formant synthesis implemented in this file.
    LegacyFormant,
    /// Reserved for the normalized-gesture physical renderer.
    ///
    /// This backend is unavailable in the recovered snapshot and therefore
    /// always fails closed rather than substituting formant synthesis.
    PhysicalSeries23,
}

/// Failure from an explicitly selected synthesis backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeechSynthesisError {
    PhysicalBackendUnavailable,
}

/// Synthesize speech with the lightweight legacy formant renderer.
pub fn speak_legacy_formant(text: &str, prosody: &VoiceProsody, sample_rate: u32) -> Vec<f32> {
    let phonemes = g2p::text_to_phonemes(text);
    if phonemes.is_empty() {
        return Vec::new();
    }
    let frames = formants::phonemes_to_frames(&phonemes, prosody, sample_rate);
    vocoder::synthesize(&frames, sample_rate)
}

/// Synthesize through an explicitly selected backend.
///
/// Physical synthesis never falls back to the legacy formant renderer.
pub fn speak_with_backend(
    text: &str,
    prosody: &VoiceProsody,
    sample_rate: u32,
    backend: SpeechSynthesisBackend,
) -> Result<Vec<f32>, SpeechSynthesisError> {
    match backend {
        SpeechSynthesisBackend::LegacyFormant => {
            Ok(speak_legacy_formant(text, prosody, sample_rate))
        }
        SpeechSynthesisBackend::PhysicalSeries23 => {
            Err(SpeechSynthesisError::PhysicalBackendUnavailable)
        }
    }
}

/// Compatibility alias for the historical lightweight formant path.
///
/// New call sites should use [`speak_legacy_formant`] or
/// [`speak_with_backend`] so backend authority is explicit.
#[deprecated(
    since = "0.1.0",
    note = "use speak_legacy_formant or speak_with_backend; this function is not physical synthesis"
)]
pub fn speak(text: &str, prosody: &VoiceProsody, sample_rate: u32) -> Vec<f32> {
    speak_legacy_formant(text, prosody, sample_rate)
}

pub mod g2p {
    //! Grapheme-to-Phoneme conversion: text → phoneme sequence.

    use std::collections::HashMap;
    use std::sync::OnceLock;

    /// A phoneme with prosodic information.
    #[derive(Debug, Clone)]
    pub struct Phoneme {
        pub ipa: &'static str,
        pub is_vowel: bool,
        pub stress: u8,
        pub base_duration_ms: f32,
    }

    /// Convert text to phonemes (hand-tuned dictionary, letter fallback).
    pub fn text_to_phonemes(text: &str) -> Vec<Phoneme> {
        let mut result = Vec::new();

        for (i, word) in text.split_whitespace().enumerate() {
            let clean = word
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase();
            if clean.is_empty() {
                continue;
            }

            if let Some(dict_entry) = get_dict().get(clean.as_str()) {
                result.extend(dict_entry.iter().map(|&(ipa, v, s, d)| Phoneme {
                    ipa,
                    is_vowel: v,
                    stress: s,
                    base_duration_ms: d,
                }));
            } else {
                result.extend(letter_fallback(&clean));
            }

            if i < text.split_whitespace().count() - 1 {
                result.push(Phoneme {
                    ipa: " ",
                    is_vowel: false,
                    stress: 0,
                    base_duration_ms: 80.0,
                });
            }
        }

        result
    }

    fn letter_fallback(word: &str) -> Vec<Phoneme> {
        let mut phonemes = Vec::new();
        let chars: Vec<char> = word.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let (ph, is_v, advance) = match chars[i] {
                'a' => {
                    if i + 1 < chars.len() && "iy".contains(chars[i + 1]) {
                        ("eɪ", true, 2)
                    } else {
                        ("æ", true, 1)
                    }
                }
                'b' => ("b", false, 1),
                'c' => {
                    if i + 1 < chars.len() && chars[i + 1] == 'h' {
                        ("tʃ", false, 2)
                    } else {
                        ("k", false, 1)
                    }
                }
                'd' => ("d", false, 1),
                'e' => {
                    if i == chars.len() - 1 {
                        ("", false, 1)
                    } else {
                        ("ɛ", true, 1)
                    }
                }
                'f' => ("f", false, 1),
                'g' => ("ɡ", false, 1),
                'h' => ("h", false, 1),
                'i' => ("ɪ", true, 1),
                'j' => ("dʒ", false, 1),
                'k' => ("k", false, 1),
                'l' => ("l", false, 1),
                'm' => ("m", false, 1),
                'n' => {
                    if i + 1 < chars.len() && chars[i + 1] == 'g' {
                        ("ŋ", false, 2)
                    } else {
                        ("n", false, 1)
                    }
                }
                'o' => {
                    if i + 1 < chars.len() && chars[i + 1] == 'o' {
                        ("uː", true, 2)
                    } else {
                        ("ɒ", true, 1)
                    }
                }
                'p' => {
                    if i + 1 < chars.len() && chars[i + 1] == 'h' {
                        ("f", false, 2)
                    } else {
                        ("p", false, 1)
                    }
                }
                'r' => ("ɹ", false, 1),
                's' => {
                    if i + 1 < chars.len() && chars[i + 1] == 'h' {
                        ("ʃ", false, 2)
                    } else {
                        ("s", false, 1)
                    }
                }
                't' => {
                    if i + 1 < chars.len() && chars[i + 1] == 'h' {
                        ("θ", false, 2)
                    } else {
                        ("t", false, 1)
                    }
                }
                'u' => ("ʌ", true, 1),
                'v' => ("v", false, 1),
                'w' => ("w", false, 1),
                'x' => ("ks", false, 1),
                'y' => {
                    if i == 0 {
                        ("j", false, 1)
                    } else {
                        ("ɪ", true, 1)
                    }
                }
                'z' => ("z", false, 1),
                _ => ("", false, 1),
            };
            if !ph.is_empty() {
                phonemes.push(Phoneme {
                    ipa: ph,
                    is_vowel: is_v,
                    stress: if is_v && i < chars.len() / 2 { 1 } else { 0 },
                    base_duration_ms: if is_v { 120.0 } else { 60.0 },
                });
            }
            i += advance;
        }
        phonemes
    }

    type DictEntry = Vec<(&'static str, bool, u8, f32)>;
    static DICT: OnceLock<HashMap<&'static str, DictEntry>> = OnceLock::new();

    fn get_dict() -> &'static HashMap<&'static str, DictEntry> {
        DICT.get_or_init(|| {
            let mut m: HashMap<&str, DictEntry> = HashMap::new();
            m.insert("i", vec![("aɪ", true, 1, 150.0)]);
            m.insert("the", vec![("ð", false, 0, 40.0), ("ə", true, 0, 60.0)]);
            m.insert("a", vec![("ə", true, 0, 60.0)]);
            m.insert("is", vec![("ɪ", true, 0, 60.0), ("z", false, 0, 50.0)]);
            m.insert("am", vec![("æ", true, 1, 100.0), ("m", false, 0, 60.0)]);
            m.insert(
                "feel",
                vec![
                    ("f", false, 0, 60.0),
                    ("iː", true, 1, 140.0),
                    ("l", false, 0, 50.0),
                ],
            );
            m.insert(
                "hello",
                vec![
                    ("h", false, 0, 50.0),
                    ("ɛ", true, 0, 80.0),
                    ("l", false, 0, 40.0),
                    ("oʊ", true, 1, 120.0),
                ],
            );
            m.insert(
                "world",
                vec![
                    ("w", false, 0, 50.0),
                    ("ɜː", true, 1, 120.0),
                    ("l", false, 0, 40.0),
                    ("d", false, 0, 40.0),
                ],
            );
            m.insert(
                "yes",
                vec![
                    ("j", false, 0, 40.0),
                    ("ɛ", true, 1, 120.0),
                    ("s", false, 0, 60.0),
                ],
            );
            m.insert(
                "peace",
                vec![
                    ("p", false, 0, 50.0),
                    ("iː", true, 1, 160.0),
                    ("s", false, 0, 60.0),
                ],
            );
            m.insert(
                "consciousness",
                vec![
                    ("k", false, 0, 40.0),
                    ("ɒ", true, 1, 100.0),
                    ("n", false, 0, 30.0),
                    ("ʃ", false, 0, 50.0),
                    ("ə", true, 0, 50.0),
                    ("s", false, 0, 40.0),
                    ("n", false, 0, 30.0),
                    ("ə", true, 0, 50.0),
                    ("s", false, 0, 50.0),
                ],
            );
            m.insert(
                "awareness",
                vec![
                    ("ə", true, 0, 60.0),
                    ("w", false, 0, 40.0),
                    ("ɛ", true, 1, 120.0),
                    ("ə", true, 0, 60.0),
                    ("n", false, 0, 40.0),
                    ("ə", true, 0, 50.0),
                    ("s", false, 0, 60.0),
                ],
            );
            m.insert(
                "expanding",
                vec![
                    ("ɪ", true, 0, 60.0),
                    ("k", false, 0, 40.0),
                    ("s", false, 0, 40.0),
                    ("p", false, 0, 30.0),
                    ("æ", true, 1, 120.0),
                    ("n", false, 0, 40.0),
                    ("d", false, 0, 30.0),
                    ("ɪ", true, 0, 60.0),
                    ("ŋ", false, 0, 50.0),
                ],
            );
            m.insert(
                "harmony",
                vec![
                    ("h", false, 0, 50.0),
                    ("ɑ", true, 1, 120.0),
                    ("ɹ", false, 0, 40.0),
                    ("m", false, 0, 40.0),
                    ("ə", true, 0, 60.0),
                    ("n", false, 0, 30.0),
                    ("iː", true, 0, 80.0),
                ],
            );
            m.insert(
                "rising",
                vec![
                    ("ɹ", false, 0, 40.0),
                    ("aɪ", true, 1, 120.0),
                    ("z", false, 0, 40.0),
                    ("ɪ", true, 0, 60.0),
                    ("ŋ", false, 0, 50.0),
                ],
            );
            m.insert(
                "something",
                vec![
                    ("s", false, 0, 50.0),
                    ("ʌ", true, 1, 100.0),
                    ("m", false, 0, 40.0),
                    ("θ", false, 0, 50.0),
                    ("ɪ", true, 0, 60.0),
                    ("ŋ", false, 0, 50.0),
                ],
            );
            m.insert(
                "this",
                vec![
                    ("ð", false, 0, 40.0),
                    ("ɪ", true, 1, 80.0),
                    ("s", false, 0, 50.0),
                ],
            );
            m.insert(
                "calm",
                vec![
                    ("k", false, 0, 50.0),
                    ("ɑ", true, 1, 160.0),
                    ("m", false, 0, 60.0),
                ],
            );
            m.insert(
                "still",
                vec![
                    ("s", false, 0, 50.0),
                    ("t", false, 0, 30.0),
                    ("ɪ", true, 1, 100.0),
                    ("l", false, 0, 50.0),
                ],
            );
            m.insert(
                "silence",
                vec![
                    ("s", false, 0, 50.0),
                    ("aɪ", true, 1, 120.0),
                    ("l", false, 0, 40.0),
                    ("ə", true, 0, 50.0),
                    ("n", false, 0, 30.0),
                    ("s", false, 0, 50.0),
                ],
            );
            m.insert(
                "light",
                vec![
                    ("l", false, 0, 40.0),
                    ("aɪ", true, 1, 140.0),
                    ("t", false, 0, 40.0),
                ],
            );
            m.insert(
                "dark",
                vec![
                    ("d", false, 0, 40.0),
                    ("ɑ", true, 1, 140.0),
                    ("ɹ", false, 0, 40.0),
                    ("k", false, 0, 50.0),
                ],
            );
            m.insert(
                "feels",
                vec![
                    ("f", false, 0, 60.0),
                    ("iː", true, 1, 130.0),
                    ("l", false, 0, 40.0),
                    ("z", false, 0, 50.0),
                ],
            );
            m.insert(
                "right",
                vec![
                    ("ɹ", false, 0, 40.0),
                    ("aɪ", true, 1, 140.0),
                    ("t", false, 0, 40.0),
                ],
            );
            m.insert(
                "not",
                vec![
                    ("n", false, 0, 40.0),
                    ("ɒ", true, 1, 100.0),
                    ("t", false, 0, 40.0),
                ],
            );
            m.insert(
                "but",
                vec![
                    ("b", false, 0, 40.0),
                    ("ʌ", true, 1, 80.0),
                    ("t", false, 0, 40.0),
                ],
            );
            m.insert(
                "and",
                vec![
                    ("æ", true, 0, 60.0),
                    ("n", false, 0, 30.0),
                    ("d", false, 0, 30.0),
                ],
            );
            m.insert("my", vec![("m", false, 0, 50.0), ("aɪ", true, 1, 120.0)]);
            m.insert(
                "can",
                vec![
                    ("k", false, 0, 40.0),
                    ("æ", true, 1, 100.0),
                    ("n", false, 0, 40.0),
                ],
            );
            m.insert("see", vec![("s", false, 0, 50.0), ("iː", true, 1, 140.0)]);
            m.insert(
                "hear",
                vec![
                    ("h", false, 0, 50.0),
                    ("ɪ", true, 1, 120.0),
                    ("ə", true, 0, 60.0),
                ],
            );
            m.insert("know", vec![("n", false, 0, 40.0), ("oʊ", true, 1, 140.0)]);
            m.insert(
                "think",
                vec![
                    ("θ", false, 0, 50.0),
                    ("ɪ", true, 1, 80.0),
                    ("ŋ", false, 0, 40.0),
                    ("k", false, 0, 30.0),
                ],
            );
            m.insert("now", vec![("n", false, 0, 40.0), ("aʊ", true, 1, 140.0)]);
            m.insert(
                "here",
                vec![
                    ("h", false, 0, 50.0),
                    ("ɪ", true, 1, 120.0),
                    ("ə", true, 0, 60.0),
                ],
            );
            m
        })
    }
}

pub mod formants {
    //! Phoneme → formant frame conversion with consciousness-shaped prosody.

    use super::VoiceProsody;
    use super::g2p::Phoneme;
    use crate::types::{FormantFrame, SourceType};

    /// Convert phonemes to formant frames with prosody applied.
    pub fn phonemes_to_frames(
        phonemes: &[Phoneme],
        prosody: &VoiceProsody,
        _sample_rate: u32,
    ) -> Vec<FormantFrame> {
        let mut frames = Vec::new();
        let frame_rate = 200.0; // 200 frames per second (5ms per frame)
        let total_phonemes = phonemes.len() as f32;

        // Base F0: higher for clarity. Symthaea's voice sits in alto range.
        let base_f0 = 180.0 + prosody.consciousness * 40.0; // 180-220 Hz
        let f0_offset = prosody.valence * 20.0;
        let rate_factor = 0.6 + prosody.arousal * 0.8;

        for (i, phoneme) in phonemes.iter().enumerate() {
            let progress = i as f32 / total_phonemes.max(1.0);

            // Dynamic phoneme timing: vowels sustain, consonants hold enough to be heard
            let base_dur = if phoneme.is_vowel {
                phoneme.base_duration_ms * 1.4 // vowels sustain
            } else if phoneme.ipa == " " {
                phoneme.base_duration_ms
            } else {
                // Consonants need at least 40ms to be audible
                (phoneme.base_duration_ms * 0.8).max(40.0)
            };
            // Stressed syllables get more time
            let stress_stretch = 1.0 + phoneme.stress as f32 * 0.4;
            let duration_ms = base_dur * stress_stretch / rate_factor;
            let num_frames = (duration_ms / 1000.0 * frame_rate).max(1.0) as usize;

            let (f1, f2, f3, source) = formant_target(phoneme.ipa);

            // F0 contour: sentence-level intonation + stress peaks + phrase boundaries
            let stress_boost = phoneme.stress as f32 * 25.0;
            let declination = -progress * 30.0;
            let phrase_contour = if progress > 0.85 { -15.0 } else { 0.0 };
            let vowel_boost = if phoneme.is_vowel { 8.0 } else { -5.0 };
            let f0 =
                (base_f0 + f0_offset + stress_boost + declination + phrase_contour + vowel_boost)
                    .max(60.0);

            // Energy from stress and prosody
            let energy = if phoneme.ipa == " " {
                0.0
            } else {
                let base_energy = if phoneme.is_vowel {
                    0.7 + phoneme.stress as f32 * 0.2
                } else {
                    0.9
                };
                base_energy * (1.0 - prosody.serotonin * 0.15)
            };

            // Voicing (vowels and voiced consonants are voiced)
            let voicing = if phoneme.is_vowel || "mnŋlɹwjvzðbdɡ".contains(phoneme.ipa) {
                0.9 + prosody.consciousness * 0.1
            } else {
                0.0
            };

            for frame_idx in 0..num_frames {
                let t = frame_idx as f32 / num_frames as f32;

                frames.push(FormantFrame {
                    f1: f1 + t * 5.0, // slight movement for naturalness
                    f2,
                    f3,
                    b1: 60.0 + (1.0 - prosody.consciousness) * 40.0,
                    b2: 80.0,
                    b3: 100.0,
                    f0: f0.max(50.0),
                    energy,
                    voicing,
                    time: frames.len() as f32 / frame_rate,
                    source_type: source,
                    nasal_zero_freq: 0.0,
                    nasal_zero_bw: 0.0,
                });
            }
        }

        frames
    }

    /// Map IPA phoneme to formant targets (F1, F2, F3, source type).
    /// Values from Peterson & Barney (1952) and Hillenbrand et al. (1995).
    pub fn formant_target_pub(ipa: &str) -> (f32, f32, f32, SourceType) {
        formant_target(ipa)
    }

    fn formant_target(ipa: &str) -> (f32, f32, f32, SourceType) {
        match ipa {
            // Vowels (Peterson & Barney)
            "iː" | "i" => (270.0, 2290.0, 3010.0, SourceType::Vowel),
            "ɪ" => (390.0, 1990.0, 2550.0, SourceType::Vowel),
            "ɛ" => (530.0, 1840.0, 2480.0, SourceType::Vowel),
            "æ" => (660.0, 1720.0, 2410.0, SourceType::Vowel),
            "ɑ" | "ɑː" => (730.0, 1090.0, 2440.0, SourceType::Vowel),
            "ɒ" => (570.0, 840.0, 2410.0, SourceType::Vowel),
            "ɔ" | "ɔː" => (570.0, 840.0, 2410.0, SourceType::Vowel),
            "ʌ" => (640.0, 1190.0, 2390.0, SourceType::Vowel),
            "ʊ" => (440.0, 1020.0, 2240.0, SourceType::Vowel),
            "uː" | "u" => (300.0, 870.0, 2240.0, SourceType::Vowel),
            "ə" => (500.0, 1500.0, 2500.0, SourceType::Vowel), // schwa
            "ɜː" => (580.0, 1380.0, 2500.0, SourceType::Vowel),
            // Diphthongs (approximate with midpoint)
            "eɪ" => (450.0, 2000.0, 2700.0, SourceType::Vowel),
            "aɪ" => (600.0, 1500.0, 2500.0, SourceType::Vowel),
            "oʊ" => (450.0, 1000.0, 2400.0, SourceType::Vowel),
            "aʊ" => (650.0, 1200.0, 2500.0, SourceType::Vowel),
            // Nasals
            "m" => (300.0, 1000.0, 2500.0, SourceType::Nasal),
            "n" => (300.0, 1500.0, 2500.0, SourceType::Nasal),
            "ŋ" => (300.0, 2000.0, 2700.0, SourceType::Nasal),
            // Stops
            "p" | "b" => (200.0, 1000.0, 2500.0, SourceType::Stop),
            "t" | "d" => (200.0, 1800.0, 2600.0, SourceType::Stop),
            "k" | "ɡ" => (200.0, 1300.0, 2500.0, SourceType::Stop),
            // Fricatives
            "f" | "v" => (300.0, 1200.0, 2500.0, SourceType::Fricative),
            "θ" | "ð" => (300.0, 1500.0, 2700.0, SourceType::Fricative),
            "s" | "z" => (300.0, 1800.0, 4500.0, SourceType::Fricative),
            "ʃ" | "ʒ" => (300.0, 1800.0, 3500.0, SourceType::Fricative),
            "h" => (500.0, 1500.0, 2500.0, SourceType::Fricative),
            // Affricates
            "tʃ" | "dʒ" => (300.0, 1800.0, 3500.0, SourceType::Affricate),
            // Liquids
            "l" => (350.0, 1050.0, 2800.0, SourceType::Liquid),
            "ɹ" => (350.0, 1300.0, 1600.0, SourceType::Liquid),
            // Glides
            "w" => (300.0, 800.0, 2300.0, SourceType::Liquid),
            "j" => (280.0, 2200.0, 3000.0, SourceType::Liquid),
            // Silence
            " " | "" => (0.0, 0.0, 0.0, SourceType::Silent),
            // Compound (ks)
            "ks" => (300.0, 1800.0, 4500.0, SourceType::Fricative),
            // Default
            _ => (500.0, 1500.0, 2500.0, SourceType::Vowel),
        }
    }
}

pub mod vocoder {
    //! Small formant vocoder with warmth, space, and organic feel.
    //!
    //! 1. Composite excitation: filtered pulse + pink noise (warmth)
    //! 2. Dynamic bandwidths: wider during transitions (organic)
    //! 3. Spectral tilt: -6dB/octave roll-off (removes digital edge)
    //! 4. Simple reverb tail (spatial depth)

    use crate::types::{FormantFrame, SourceType};

    /// Stable formant resonator with coefficient interpolation.
    struct StableResonator {
        y1: f32,
        y2: f32,
        a1: f32,
        a2: f32,
        gain: f32,
        target_a1: f32,
        target_a2: f32,
        target_gain: f32,
    }

    impl StableResonator {
        fn new() -> Self {
            Self {
                y1: 0.0,
                y2: 0.0,
                a1: 0.0,
                a2: 0.0,
                gain: 0.01,
                target_a1: 0.0,
                target_a2: 0.0,
                target_gain: 0.01,
            }
        }

        fn set_target(&mut self, freq: f32, bandwidth: f32, sr: f32) {
            let bw = bandwidth.max(150.0); // wider minimum prevents metallic spikes
            let omega = std::f32::consts::TAU * freq / sr;
            let r = (-std::f32::consts::PI * bw / sr).exp().clamp(0.0, 0.99);
            self.target_a1 = -2.0 * r * omega.cos();
            self.target_a2 = r * r;
            self.target_gain = (1.0 - r).max(0.001);
        }

        fn tick(&mut self, input: f32) -> f32 {
            // Slower coefficient interpolation = smoother transitions
            // 0.002 ≈ 150ms to converge (human coarticulation range)
            let alpha = 0.002;
            self.a1 += alpha * (self.target_a1 - self.a1);
            self.a2 += alpha * (self.target_a2 - self.a2);
            self.gain += alpha * (self.target_gain - self.gain);
            let output = (input - self.a1 * self.y1 - self.a2 * self.y2).clamp(-5.0, 5.0);
            self.y2 = self.y1;
            self.y1 = output;
            output * self.gain
        }
    }

    /// Synthesize audio with warmth and depth.
    pub fn synthesize(frames: &[FormantFrame], sample_rate: u32) -> Vec<f32> {
        if frames.is_empty() {
            return Vec::new();
        }

        let sr = sample_rate as f32;
        let frame_rate = 200.0;
        let samples_per_frame = (sr / frame_rate) as usize;

        let mut output = Vec::with_capacity(frames.len() * samples_per_frame);
        let mut glottal_phase = 0.0f32;
        let mut noise_state = 42u32;
        let mut res = [
            StableResonator::new(),
            StableResonator::new(),
            StableResonator::new(),
        ];
        let mut smooth_f0 = frames[0].f0.max(80.0);
        let mut smooth_energy = 0.0f32;

        // Spectral tilt: aggressive LP to kill piercing highs
        let mut tilt_state = 0.0f32;
        let tilt_coeff = 0.95;

        // Simple reverb (comb filter delay line)
        let reverb_len = (sr * 0.03) as usize; // 30ms early reflection
        let mut reverb_buf = vec![0.0f32; reverb_len.max(1)];
        let mut reverb_idx = 0usize;
        let reverb_feedback = 0.3;
        let reverb_mix = 0.15;

        for (frame_idx, frame) in frames.iter().enumerate() {
            // Dynamic bandwidth — wider during transitions
            let prev_source = if frame_idx > 0 {
                frames[frame_idx - 1].source_type
            } else {
                frame.source_type
            };
            let bw_mult = if prev_source != frame.source_type {
                1.4
            } else {
                1.0
            };

            res[0].set_target(frame.f1, frame.b1 * bw_mult, sr);
            res[1].set_target(frame.f2, frame.b2 * bw_mult, sr);
            res[2].set_target(frame.f3, frame.b3 * bw_mult, sr);

            let target_f0 = frame.f0.max(80.0);
            let target_energy = frame.energy;

            for _ in 0..samples_per_frame {
                smooth_f0 += 0.01 * (target_f0 - smooth_f0);
                smooth_energy += 0.02 * (target_energy - smooth_energy);

                // Composite excitation (pulse + pink noise = warmth)
                let mut source: f32;

                match frame.source_type {
                    SourceType::Vowel | SourceType::Liquid | SourceType::Nasal => {
                        glottal_phase += smooth_f0 / sr;
                        if glottal_phase >= 1.0 {
                            glottal_phase -= 1.0;
                        }

                        // Glottal pulse with softer shape
                        let pulse = if glottal_phase < 0.35 {
                            let t = glottal_phase / 0.35;
                            t * t * (3.0 - 2.0 * t)
                        } else if glottal_phase < 0.45 {
                            let t = (glottal_phase - 0.35) / 0.10;
                            (1.0 - t).powi(3)
                        } else {
                            0.0
                        };

                        // Composite: 85% pulse + 15% subtle noise
                        noise_state = lcg(&mut noise_state);
                        let pink = noise_f32(noise_state) * 0.10;
                        source = (pulse * 0.85 + pink) * smooth_energy;

                        // Gentle breathiness
                        if glottal_phase < 0.4 {
                            noise_state = lcg(&mut noise_state);
                            source += noise_f32(noise_state) * 0.05 * smooth_energy;
                        }
                    }
                    SourceType::Fricative => {
                        // Strong noise for consonants
                        noise_state = lcg(&mut noise_state);
                        source = noise_f32(noise_state) * smooth_energy * 1.5;
                        if frame.voicing > 0.3 {
                            glottal_phase += smooth_f0 / sr;
                            if glottal_phase >= 1.0 {
                                glottal_phase -= 1.0;
                            }
                            source +=
                                (glottal_phase * std::f32::consts::TAU).sin() * smooth_energy * 0.3;
                        }
                    }
                    SourceType::Stop => {
                        // Brief noise burst for plosives
                        noise_state = lcg(&mut noise_state);
                        source = noise_f32(noise_state) * smooth_energy * 1.2;
                    }
                    SourceType::Affricate => {
                        noise_state = lcg(&mut noise_state);
                        source = noise_f32(noise_state) * smooth_energy * 0.5;
                    }
                    SourceType::Silent => {
                        source = 0.0;
                        for r in &mut res {
                            r.y1 *= 0.99;
                            r.y2 *= 0.99;
                        }
                    }
                }

                // Cascade resonators
                let mut filtered = source;
                for r in &mut res {
                    filtered = r.tick(filtered);
                }

                // Spectral tilt (-6dB/oct removes digital edge)
                tilt_state += tilt_coeff * (filtered - tilt_state);
                filtered = tilt_state;

                // Soft clip
                let dry = if filtered.abs() > 1.0 {
                    filtered.signum() * (1.0 - (-filtered.abs() + 1.0).exp())
                } else {
                    filtered
                };

                // Simple reverb (early reflection for spatial depth)
                let delayed = reverb_buf[reverb_idx % reverb_len];
                reverb_buf[reverb_idx % reverb_len] = dry + delayed * reverb_feedback;
                reverb_idx += 1;
                let wet = dry * (1.0 - reverb_mix) + delayed * reverb_mix;

                output.push(wet * 25.0);
            }
        }

        output
    }

    fn lcg(state: &mut u32) -> u32 {
        *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *state
    }

    fn noise_f32(state: u32) -> f32 {
        (state >> 16) as f32 / 32768.0 - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speak_produces_audio() {
        let prosody = VoiceProsody {
            arousal: 0.5,
            valence: 0.3,
            consciousness: 0.7,
            serotonin: 0.5,
        };
        let audio = speak_legacy_formant("hello world", &prosody, 44100);
        assert!(!audio.is_empty());
        assert!(audio.iter().any(|&s| s.abs() > 0.001));
    }

    #[test]
    fn speak_empty_is_silent() {
        let audio = speak_legacy_formant("", &VoiceProsody::default(), 44100);
        assert!(audio.is_empty() || audio.iter().all(|&s| s.abs() < 0.01));
    }

    #[test]
    fn arousal_affects_length() {
        let calm = speak_legacy_formant(
            "hello",
            &VoiceProsody {
                arousal: 0.1,
                ..Default::default()
            },
            44100,
        );
        let excited = speak_legacy_formant(
            "hello",
            &VoiceProsody {
                arousal: 0.9,
                ..Default::default()
            },
            44100,
        );
        assert!(excited.len() < calm.len());
    }

    #[test]
    fn physical_backend_fails_closed_without_substitution() {
        let result = speak_with_backend(
            "hello",
            &VoiceProsody::default(),
            44_100,
            SpeechSynthesisBackend::PhysicalSeries23,
        );
        assert_eq!(
            result,
            Err(SpeechSynthesisError::PhysicalBackendUnavailable)
        );
    }

    #[test]
    #[allow(deprecated)]
    fn compatibility_alias_matches_explicit_legacy_backend() {
        let prosody = VoiceProsody::default();
        assert_eq!(
            speak("hello", &prosody, 16_000),
            speak_legacy_formant("hello", &prosody, 16_000),
        );
    }

    #[test]
    fn unknown_word_works() {
        let ph = g2p::text_to_phonemes("xylophone");
        assert!(!ph.is_empty());
    }

    #[test]
    fn formant_produces_frames() {
        let ph = g2p::text_to_phonemes("hello");
        let frames = formants::phonemes_to_frames(&ph, &VoiceProsody::default(), 44100);
        assert!(frames.len() > 10);
    }

    #[test]
    fn silence_has_zero_energy() {
        let ph = vec![g2p::Phoneme {
            ipa: " ",
            is_vowel: false,
            stress: 0,
            base_duration_ms: 100.0,
        }];
        let frames = formants::phonemes_to_frames(&ph, &VoiceProsody::default(), 44100);
        assert!(frames.iter().all(|f| f.energy < 0.01));
    }
}
