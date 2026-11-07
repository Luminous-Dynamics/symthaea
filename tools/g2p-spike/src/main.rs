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
    let phonemizer = EnglishPhonemizer::new()
        .expect("Piper Plus English phonemizer must initialize");

    println!("case_id\ttext\tipa_tokens_debug\tprosody_debug");
    for (case_id, text) in CASES {
        match phonemizer.phonemize_with_prosody(text) {
            Ok((tokens, prosody)) => {
                println!("{case_id}\t{text}\t{tokens:?}\t{prosody:?}");
            }
            Err(error) => {
                println!("{case_id}\t{text}\tERROR\t{error}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_phonemizer_initializes_and_produces_tokens() {
        let phonemizer = EnglishPhonemizer::new().expect("English phonemizer");
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
    fn diagnostic_cases_are_independent_and_keep_case_labels() {
        assert!(CASES.iter().any(|(id, _)| *id == "heteronym_read"));
        assert!(CASES.iter().any(|(id, _)| *id == "unknown_candidate"));
    }
}
