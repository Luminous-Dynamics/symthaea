// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared deterministic text encoding for therapeutic memory.
//!
//! Whole-string hashing makes every wording change look unrelated. This module
//! instead bundles deterministic word hypervectors so texts that share content
//! words retain measurable representational overlap. It is deliberately a
//! lexical encoder, not a claim of full semantic understanding.

use symthaea_core::hdc::BinaryHV;

/// Low-information words omitted from therapeutic-memory encodings.
///
/// Negation words are intentionally retained: "safe" and "not safe" must not
/// collapse to identical representations.
const THERAPEUTIC_STOPWORDS: &[&str] = &[
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being", "i", "me", "my",
    "we", "our", "you", "your", "he", "she", "it", "its", "they", "them", "their", "this", "that",
    "these", "those", "in", "on", "at", "to", "for", "of", "with", "by", "from", "as", "and", "or",
    "but", "so", "if", "then", "than", "when", "while", "do", "did", "does", "have", "has", "had",
    "will", "would", "could", "should", "can", "may", "might", "shall", "must", "just", "very",
    "really", "also", "too", "even", "still", "already", "about", "into", "over", "after",
    "before", "between", "through", "up", "down", "out", "off", "all", "each", "every", "both",
    "here", "there", "where", "how", "what", "which", "who", "whom", "some", "any", "only", "own",
    "same", "much", "many", "more", "most", "other", "such",
];

/// Encode therapeutic text as a deterministic bag of content-word vectors.
///
/// Returns `None` when no lexical content remains after normalization. Callers
/// may preserve that absence or use [`encode_or_fallback`] when a vector is
/// structurally required.
pub(crate) fn encode_therapeutic_text(text: &str) -> Option<BinaryHV> {
    encode_bag_of_words(text, "therapeutic", THERAPEUTIC_STOPWORDS)
}

/// Encode text using deterministic content-word hypervectors in a namespace.
pub(crate) fn encode_bag_of_words(
    text: &str,
    namespace: &str,
    stopwords: &[&str],
) -> Option<BinaryHV> {
    let tokens: Vec<String> = text
        .split(|ch: char| !ch.is_alphanumeric() && ch != '\'')
        .map(|token| token.trim_matches('\'').to_lowercase())
        .filter(|token| token.len() > 1)
        .filter(|token| !stopwords.contains(&token.as_str()))
        .collect();

    if tokens.is_empty() {
        return None;
    }

    let vectors: Vec<BinaryHV> = tokens
        .iter()
        .map(|token| deterministic_vector(namespace, "word", token))
        .collect();

    Some(BinaryHV::bundle(&vectors))
}

/// Encode text compositionally, falling back to a deterministic whole-text
/// vector only when normalization leaves no content words.
pub(crate) fn encode_or_fallback(text: &str, namespace: &str, stopwords: &[&str]) -> BinaryHV {
    encode_bag_of_words(text, namespace, stopwords).unwrap_or_else(|| {
        let normalized = text.trim().to_lowercase();
        deterministic_vector(namespace, "fallback", &normalized)
    })
}

fn deterministic_vector(namespace: &str, kind: &str, value: &str) -> BinaryHV {
    let hash = blake3::hash(format!("{namespace}_{kind}:{value}").as_bytes());
    let seed = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
    BinaryHV::random(seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_is_case_and_punctuation_stable() {
        let a = encode_therapeutic_text("Fear of abandonment!").unwrap();
        let b = encode_therapeutic_text("fear of abandonment").unwrap();
        assert_eq!(a.similarity(&b), 1.0);
    }

    #[test]
    fn shared_content_is_more_similar_than_unrelated_content() {
        let source = encode_therapeutic_text("intense anger toward authority").unwrap();
        let related = encode_therapeutic_text("anger around an authority figure").unwrap();
        let unrelated = encode_therapeutic_text("peaceful walk beside the ocean").unwrap();

        assert!(
            source.similarity(&related) > source.similarity(&unrelated),
            "shared content words should create greater overlap"
        );
    }

    #[test]
    fn negation_is_not_discarded() {
        let positive = encode_therapeutic_text("I feel safe").unwrap();
        let negated = encode_therapeutic_text("I do not feel safe").unwrap();
        assert!(positive.similarity(&negated) < 1.0);
    }

    #[test]
    fn empty_text_has_no_encoding() {
        assert!(encode_therapeutic_text("").is_none());
        assert!(encode_therapeutic_text("the and of").is_none());
    }
}
