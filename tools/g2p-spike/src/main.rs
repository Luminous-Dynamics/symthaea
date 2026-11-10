// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Isolated pronunciation-frontend probe.
//!
//! This is diagnostic only: Piper Plus output is a candidate hypothesis, not
//! ground truth. Compare it with the Muse frontend and human-reviewed
//! references before considering any production integration.

use piper_plus_g2p::Phonemizer;
use piper_plus_g2p::english::EnglishPhonemizer;

mod normalize;
use normalize::normalize_piper_english;

const CASES: &[(&str, &str)] = &[
    ("baseline", "hello world"),
    ("clusters", "quick queen strength sixth"),
    ("irregular_ough", "though through rough tough cough bough"),
    ("heteronym_read", "I read the book yesterday and I will read it tomorrow"),
    ("heteronym_lead", "the lead pipe is heavy and the guide will lead the band"),
    ("silent_letters", "knight debt island subtle"),
    ("contractions", "don't we'll they're I'd"),
    ("multisyllabic", "colonel anemone phenomenon"),
    ("punctuation", "\"Hello, world!\""),
    ("unknown_candidate", "xqzpl flurble"),
    ("ugt_abbreviations", "ppl r here imo"),
    ("ugt_initialisms", "brb idk tbh"),
    ("ugt_noncanonical", "u r gonna be ok"),
    ("ugt_elongation", "soooo this is reallly good"),
    ("ugt_slang", "tho cuz wanna hafta"),
    ("ugt_digit_homophone", "gr8 2 c u"),
    ("ugt_a1_vowel_drop", "ppl pls"),
    ("ugt_a2_other_drop", "bc k"),
    ("ugt_a3_phrase_shortening", "smh imo"),
    ("ugt_b2_iteration", "waitwait"),
    ("ugt_c1_eye_direct", "luv"),
    ("ugt_c2_regular", "droppin"),
    ("ugt_c4_pronounceable", "rofl"),
];

fn main() {
    let phonemizer = EnglishPhonemizer::new_bundled()
        .expect("bundled Piper Plus English dictionary must initialize");

    println!("case_id\ttext\tpiper_ipa_tokens\tnormalized_word_hypotheses_json\tmuse_word_ipa_tokens\tprosody_debug");
    for (case_id, text) in CASES {
        let words: Vec<&str> = text.split_whitespace().collect();
        let muse_word_syllable_counts: Vec<usize> = words
            .iter()
            .map(|word| symthaea_muse::singing_bridge::syllable_count(word))
            .collect();
        let muse_syllable_count: usize = muse_word_syllable_counts.iter().sum();
        let melody: Vec<symthaea_muse::Note> = (0..muse_syllable_count)
            .map(|index| symthaea_muse::Note {
                frequency: 220.0,
                start_time: index as f32 * 0.4,
                duration: 0.4,
                velocity: 0.7,
            })
            .collect();
        let muse_syllables = symthaea_muse::singing_bridge::sing(text, &melody);
        let mut cursor = 0usize;
        let muse_word_tokens: Vec<Vec<&str>> = muse_word_syllable_counts
            .iter()
            .map(|count| {
                let word_phones = muse_syllables[cursor..cursor + count]
                    .iter()
                    .flat_map(|syllable| syllable.phonemes.iter().map(|phoneme| phoneme.ipa))
                    .collect();
                cursor += count;
                word_phones
            })
            .collect();

        match phonemizer.phonemize_with_prosody(text) {
            Ok((tokens, prosody)) => {
                match normalize_piper_english(&tokens, &prosody) {
                    Ok(hypothesis) => {
                        let normalized_json =
                            serde_json::to_string(&hypothesis).expect("hypothesis serialization");
                        println!(
                            "{case_id}\t{text}\t{tokens:?}\t{normalized_json}\t{muse_word_tokens:?}\t{prosody:?}"
                        );
                    }
                    Err(error) => {
                        println!(
                            "{case_id}\t{text}\t{tokens:?}\tNORMALIZATION_ERROR: {error}\t{muse_word_tokens:?}\t{prosody:?}"
                        );
                    }
                }
            }
            Err(error) => {
                println!("{case_id}\t{text}\tERROR: {error}\t-\t{muse_word_tokens:?}\t-");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_phonemizer_initializes_and_produces_tokens() {
        let phonemizer = EnglishPhonemizer::new_bundled()
            .expect("bundled English phonemizer must initialize");
        let (tokens, prosody) = phonemizer
            .phonemize_with_prosody("hello world")
            .expect("known words should produce an output");
        assert!(!tokens.is_empty(), "known words must not return no phones");
        assert!(
            tokens.iter().all(|token| !token.is_empty()),
            "phoneme token stream must not contain empty entries: {tokens:?}"
        );
        assert!(
            tokens
                .iter()
                .any(|token| !token.trim().is_empty() && !matches!(token.as_str(), "ˈ" | "ˌ" | "ː")),
            "known words must contain at least one IPA phone: {tokens:?}"
        );
        assert!(
            tokens.iter().any(|token| token == " "),
            "word boundaries must remain explicit in the IPA stream: {tokens:?}"
        );
        assert_eq!(
            tokens.len(),
            prosody.len(),
            "prosody entries must remain aligned one-to-one with IPA character tokens"
        );
        let normalized = normalize_piper_english(&tokens, &prosody)
            .expect("raw Piper output must satisfy the loss-preserving adapter");
        assert_eq!(normalized.words.len(), 2);
        assert_eq!(normalized.words[0].resolution_status, "needs_review");
        assert_eq!(normalized.words[0].syllable_boundaries, None);
        assert_eq!(normalized.words[0].oov_status, "not_assessed");
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
