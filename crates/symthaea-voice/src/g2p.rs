// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
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

/// Convert text to phonemes.
pub fn text_to_phonemes(text: &str) -> Vec<Phoneme> {
    let mut result = Vec::new();

    for (i, word) in text.split_whitespace().enumerate() {
        let clean = word
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase();
        if clean.is_empty() {
            continue;
        }

        // Priority: 1) CMUdict (134K words), 2) hand-tuned dict, 3) letter fallback
        if let Some(cmu_phonemes) = crate::cmudict::lookup(&clean) {
            result.extend(cmu_phonemes.iter().cloned());
        } else if let Some(dict_entry) = get_dict().get(clean.as_str()) {
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

static DICT: OnceLock<HashMap<&'static str, Vec<(&'static str, bool, u8, f32)>>> = OnceLock::new();

fn get_dict() -> &'static HashMap<&'static str, Vec<(&'static str, bool, u8, f32)>> {
    DICT.get_or_init(|| {
        let mut m: HashMap<&str, Vec<(&str, bool, u8, f32)>> = HashMap::new();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_has_phonemes() {
        let ph = text_to_phonemes("hello world");
        assert!(!ph.is_empty());
        assert!(ph.iter().any(|p| p.is_vowel));
    }

    #[test]
    fn unknown_word_works() {
        let ph = text_to_phonemes("xylophone");
        assert!(!ph.is_empty());
    }

    #[test]
    fn consciousness_in_dict() {
        let ph = text_to_phonemes("consciousness");
        assert!(ph.len() >= 7);
    }
}
