// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Isolated pronunciation-frontend probe.
//!
//! This is diagnostic only: Piper Plus output is a candidate hypothesis, not
//! ground truth. Compare it with the Muse frontend and human-reviewed
//! references before considering any production integration.

use piper_plus_g2p::Phonemizer;
use piper_plus_g2p::english::EnglishPhonemizer;

const CASES: &[(&str, &str)] = &[
    ("baseline", "hello world"),
    ("clusters", "quick queen strength sixth"),
    ("irregular_ough", "though through rough tough cough bough"),
    ("heteronym_read", "I read the book yesterday and I will read it tomorrow"),
    ("heteronym_lead", "the lead singer will lead the band"),
    ("silent_letters", "knight debt island subtle"),
    ("contractions", "don't we'll they're I'd"),
    ("multisyllabic", "colonel anemone phenomenon"),
    ("punctuation", "\"Hello, world!\""),
    ("unknown_candidate", "xqzpl flurble"),
];

fn main() {
    let phonemizer = EnglishPhonemizer::new_bundled()
        .expect("bundled Piper Plus English dictionary must initialize");

    println!("case_id\ttext\tpiper_ipa_tokens\tmuse_ipa_tokens\tprosody_debug");
    for (case_id, text) in CASES {
        let muse_syllable_count = symthaea_muse::singing_bridge::syllable_count(text);
        let melody: Vec<symthaea_muse::Note> = (0..muse_syllable_count)
            .map(|index| symthaea_muse::Note {
                frequency: 220.0,
                start_time: index as f32 * 0.4,
                duration: 0.4,
                velocity: 0.7,
            })
            .collect();
        let muse_tokens: Vec<&str> = symthaea_muse::singing_bridge::sing(text, &melody)
            .into_iter()
            .flat_map(|syllable| syllable.phonemes.into_iter().map(|phoneme| phoneme.ipa))
            .collect();

        match phonemizer.phonemize_with_prosody(text) {
            Ok((tokens, prosody)) => {
                println!("{case_id}\t{text}\t{tokens:?}\t{muse_tokens:?}\t{prosody:?}");
            }
            Err(error) => {
                println!("{case_id}\t{text}\tERROR: {error}\t{muse_tokens:?}\t-");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_phonemizer_initializes_and_produces_tokens() {
        let phonemizer = EnglishPhonemizer::new_bundled();
        let (tokens, _prosody) = phonemizer
            .phonemize_with_prosody("hello world")
            .expect("known words should produce an output");
        assert!(!tokens.is_empty(), "known words must not return no phones");
        assert!(
            tokens.iter().all(|token| !token.trim().is_empty()),
            "phoneme token list must not contain empty entries: {tokens:?}"
        );
    }

    #[test]
    fn current_muse_g2p_baseline_is_observable_on_same_fixture() {
        let count = symthaea_muse::singing_bridge::syllable_count("hello world");
        assert!(count > 0, "the current Muse G2P baseline should emit syllables");
        let notes: Vec<symthaea_muse::Note> = (0..count)
            .map(|index| symthaea_muse::Note {
                frequency: 220.0,
                start_time: index as f32 * 0.4,
                duration: 0.4,
                velocity: 0.7,
            })
            .collect();
        let output = symthaea_muse::singing_bridge::sing("hello world", &notes);
        assert!(!output.is_empty(), "Muse baseline output should be inspectable");
    }

    #[test]
    fn diagnostic_cases_are_independent_and_keep_case_labels() {
        assert!(CASES.iter().any(|(id, _)| *id == "heteronym_read"));
        assert!(CASES.iter().any(|(id, _)| *id == "unknown_candidate"));
    }
}
