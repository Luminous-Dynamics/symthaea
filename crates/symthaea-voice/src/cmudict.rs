// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! CMU Pronouncing Dictionary: 134K English words → ARPAbet phonemes.
//!
//! Loaded at first use from embedded data. Maps ARPAbet to our IPA phonemes.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::g2p::Phoneme;

static CMUDICT: OnceLock<HashMap<String, Vec<Phoneme>>> = OnceLock::new();

/// Get the CMUdict. Loads and parses on first call.
pub fn get_cmudict() -> &'static HashMap<String, Vec<Phoneme>> {
    CMUDICT.get_or_init(|| {
        let raw = include_str!("../data/cmudict.txt");
        parse_cmudict(raw)
    })
}

/// Look up a word in CMUdict.
pub fn lookup(word: &str) -> Option<&'static Vec<Phoneme>> {
    get_cmudict().get(&word.to_uppercase())
}

fn parse_cmudict(raw: &str) -> HashMap<String, Vec<Phoneme>> {
    let mut dict = HashMap::new();

    for line in raw.lines() {
        if line.starts_with(";;;") || line.is_empty() {
            continue;
        }

        // Format: WORD  PH1 PH2 PH3...  (double space separator)
        let parts: Vec<&str> = line.splitn(2, "  ").collect();
        if parts.len() < 2 {
            continue;
        }

        let word = parts[0].trim();
        // Skip pronunciation variants like WORD(2)
        let word = if word.contains('(') {
            word.split('(').next().unwrap_or(word)
        } else {
            word
        };

        let phonemes_str = parts[1].trim();
        let phonemes: Vec<Phoneme> = phonemes_str
            .split_whitespace()
            .filter_map(|arp| arpabet_to_phoneme(arp))
            .collect();

        if !phonemes.is_empty() {
            dict.entry(word.to_string()).or_insert(phonemes);
        }
    }

    dict
}

/// Convert ARPAbet symbol to our Phoneme type.
fn arpabet_to_phoneme(arp: &str) -> Option<Phoneme> {
    // ARPAbet has stress markers: AA0, AA1, AA2
    let (base, stress) = if arp.len() > 2 && arp.as_bytes()[arp.len() - 1].is_ascii_digit() {
        (
            &arp[..arp.len() - 1],
            (arp.as_bytes()[arp.len() - 1] - b'0'),
        )
    } else {
        (arp, 0u8)
    };

    let (ipa, is_vowel, dur) = match base {
        // Vowels
        "AA" => ("ɑ", true, 120.0),
        "AE" => ("æ", true, 120.0),
        "AH" => ("ʌ", true, 100.0),
        "AO" => ("ɔ", true, 120.0),
        "AW" => ("aʊ", true, 140.0),
        "AY" => ("aɪ", true, 140.0),
        "EH" => ("ɛ", true, 100.0),
        "ER" => ("ɜː", true, 120.0),
        "EY" => ("eɪ", true, 130.0),
        "IH" => ("ɪ", true, 80.0),
        "IY" => ("iː", true, 120.0),
        "OW" => ("oʊ", true, 130.0),
        "OY" => ("ɔɪ", true, 140.0),
        "UH" => ("ʊ", true, 80.0),
        "UW" => ("uː", true, 120.0),
        // Consonants
        "B" => ("b", false, 40.0),
        "CH" => ("tʃ", false, 60.0),
        "D" => ("d", false, 40.0),
        "DH" => ("ð", false, 40.0),
        "F" => ("f", false, 60.0),
        "G" => ("ɡ", false, 40.0),
        "HH" => ("h", false, 50.0),
        "JH" => ("dʒ", false, 60.0),
        "K" => ("k", false, 50.0),
        "L" => ("l", false, 50.0),
        "M" => ("m", false, 60.0),
        "N" => ("n", false, 40.0),
        "NG" => ("ŋ", false, 50.0),
        "P" => ("p", false, 50.0),
        "R" => ("ɹ", false, 40.0),
        "S" => ("s", false, 60.0),
        "SH" => ("ʃ", false, 60.0),
        "T" => ("t", false, 40.0),
        "TH" => ("θ", false, 50.0),
        "V" => ("v", false, 50.0),
        "W" => ("w", false, 40.0),
        "Y" => ("j", false, 40.0),
        "Z" => ("z", false, 50.0),
        "ZH" => ("ʒ", false, 60.0),
        _ => return None,
    };

    Some(Phoneme {
        ipa,
        is_vowel,
        stress,
        base_duration_ms: dur,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cmudict_loads() {
        let dict = get_cmudict();
        assert!(
            dict.len() > 100000,
            "should have 100K+ entries: {}",
            dict.len()
        );
    }

    #[test]
    fn hello_in_dict() {
        let ph = lookup("hello");
        assert!(ph.is_some(), "HELLO should be in CMUdict");
        let ph = ph.unwrap();
        assert!(ph.len() >= 4, "hello should have 4+ phonemes: {}", ph.len());
    }

    #[test]
    fn consciousness_in_dict() {
        let ph = lookup("consciousness");
        assert!(ph.is_some(), "CONSCIOUSNESS should be in CMUdict");
    }

    #[test]
    fn arpabet_conversion() {
        let ph = arpabet_to_phoneme("AA1").unwrap();
        assert_eq!(ph.ipa, "ɑ");
        assert_eq!(ph.stress, 1);
        assert!(ph.is_vowel);
    }
}
