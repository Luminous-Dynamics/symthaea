//! Grapheme-to-Phoneme (G2P) Conversion
//!
//! Basic English G2P using a lookup table approach.
//! Maps text to phoneme IDs for Kokoro TTS model input.
//! Can be upgraded to a full G2P model later.

use std::collections::HashMap;

/// Phoneme ID mapping for Kokoro TTS.
///
/// Kokoro uses a fixed phoneme vocabulary. This struct maps
/// English text to sequences of phoneme IDs.
pub struct G2PConverter {
    /// Word → phoneme ID sequence lookup table
    word_map: HashMap<String, Vec<u32>>,
    /// Character → phoneme ID fallback
    char_map: HashMap<char, u32>,
    /// Silence token ID
    pub silence_id: u32,
    /// Padding token ID
    pub pad_id: u32,
}

impl G2PConverter {
    /// Create a new G2P converter with a built-in vocabulary.
    pub fn new() -> Self {
        let mut word_map = HashMap::new();
        let mut char_map = HashMap::new();

        // Common English words → phoneme ID sequences
        // These IDs correspond to Kokoro's phoneme vocabulary
        // In production, load from a pronunciation dictionary (CMUDict)
        let common_words: &[(&str, &[u32])] = &[
            ("hello", &[16, 7, 21, 24, 28]),
            ("world", &[34, 9, 21, 5]),
            ("the", &[31, 6]),
            ("a", &[6]),
            ("an", &[8, 23]),
            ("is", &[17, 36]),
            ("and", &[8, 23, 5]),
            ("to", &[33, 35]),
            ("of", &[10, 33]),
            ("in", &[17, 23]),
            ("for", &[13, 11, 27]),
            ("on", &[10, 23]),
            ("it", &[17, 33]),
            ("that", &[31, 8, 33]),
            ("this", &[31, 17, 29]),
            ("with", &[34, 17, 31]),
            ("not", &[23, 10, 33]),
            ("you", &[18, 35]),
            ("we", &[34, 14]),
            ("consciousness", &[20, 10, 23, 30, 6, 29, 23, 6, 29]),
            ("symthaea", &[29, 17, 22, 31, 14, 6]),
            ("intelligence", &[17, 23, 33, 7, 21, 17, 5, 6, 23, 29]),
            ("neural", &[23, 35, 27, 6, 21]),
            ("system", &[29, 17, 29, 33, 6, 22]),
        ];

        for (word, phonemes) in common_words {
            word_map.insert(word.to_string(), phonemes.to_vec());
        }

        // Character-level fallback mapping (rough approximation)
        let char_phonemes: &[(char, u32)] = &[
            ('a', 8), ('b', 4), ('c', 20), ('d', 5), ('e', 7),
            ('f', 13), ('g', 15), ('h', 16), ('i', 17), ('j', 5),
            ('k', 20), ('l', 21), ('m', 22), ('n', 23), ('o', 10),
            ('p', 25), ('q', 20), ('r', 27), ('s', 29), ('t', 33),
            ('u', 35), ('v', 33), ('w', 34), ('x', 20), ('y', 18),
            ('z', 36), (' ', 0),
        ];

        for (ch, id) in char_phonemes {
            char_map.insert(*ch, *id);
        }

        Self {
            word_map,
            char_map,
            silence_id: 0,
            pad_id: 0,
        }
    }

    /// Convert text to a sequence of phoneme IDs.
    pub fn text_to_phonemes(&self, text: &str) -> Vec<u32> {
        let mut phoneme_ids = Vec::new();
        let lower = text.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();

        for (i, word) in words.iter().enumerate() {
            // Strip punctuation for lookup
            let clean: String = word.chars().filter(|c| c.is_alphanumeric()).collect();

            if let Some(phonemes) = self.word_map.get(&clean) {
                phoneme_ids.extend_from_slice(phonemes);
            } else {
                // Fall back to character-by-character mapping
                for ch in clean.chars() {
                    if let Some(&id) = self.char_map.get(&ch) {
                        phoneme_ids.push(id);
                    }
                }
            }

            // Add silence between words
            if i < words.len() - 1 {
                phoneme_ids.push(self.silence_id);
            }
        }

        // Add sentence-end silence
        phoneme_ids.push(self.silence_id);
        phoneme_ids
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        40 // Kokoro uses ~40 phoneme tokens
    }

    /// Get the number of words in the lookup table.
    pub fn dictionary_size(&self) -> usize {
        self.word_map.len()
    }
}

impl Default for G2PConverter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_known_word() {
        let g2p = G2PConverter::new();
        let phonemes = g2p.text_to_phonemes("hello");
        assert!(!phonemes.is_empty());
        // Should use dictionary lookup
        assert_eq!(phonemes[..phonemes.len()-1], [16, 7, 21, 24, 28]);
    }

    #[test]
    fn test_unknown_word_fallback() {
        let g2p = G2PConverter::new();
        let phonemes = g2p.text_to_phonemes("xyz");
        assert!(!phonemes.is_empty());
    }

    #[test]
    fn test_multi_word() {
        let g2p = G2PConverter::new();
        let phonemes = g2p.text_to_phonemes("hello world");
        // Should have silence between words
        assert!(phonemes.contains(&g2p.silence_id));
    }

    #[test]
    fn test_empty_input() {
        let g2p = G2PConverter::new();
        let phonemes = g2p.text_to_phonemes("");
        // Just the trailing silence
        assert_eq!(phonemes, vec![0]);
    }

    #[test]
    fn test_dictionary_size() {
        let g2p = G2PConverter::new();
        assert!(g2p.dictionary_size() > 10);
    }
}
