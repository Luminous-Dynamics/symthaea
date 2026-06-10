// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Lexicon and text-to-phoneme conversion
//!
//! This module provides:
//! - CMU Pronouncing Dictionary loader
//! - Text to phoneme sequence conversion
//! - ARPAbet to IPA conversion

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// ARPAbet to IPA conversion table
pub fn arpabet_to_ipa(arpabet: &str) -> Option<&'static str> {
    // Strip stress markers
    let clean = arpabet.trim_end_matches(|c: char| c.is_ascii_digit());

    Some(match clean {
        // Vowels
        "AA" => "ɑ",
        "AE" => "æ",
        "AH" => "ʌ",
        "AO" => "ɔ",
        "AW" => "aʊ",
        "AX" => "ə",
        "AXR" => "ɚ",
        "AY" => "aɪ",
        "EH" => "ɛ",
        "ER" => "ɝ",
        "EY" => "e",
        "IH" => "ɪ",
        "IX" => "ɨ",
        "IY" => "i",
        "OW" => "o",
        "OY" => "ɔɪ",
        "UH" => "ʊ",
        "UW" => "u",
        "UX" => "ʉ",
        // Consonants
        "B" => "b",
        "CH" => "tʃ",
        "D" => "d",
        "DH" => "ð",
        "DX" => "ɾ",
        "EL" => "l̩",
        "EM" => "m̩",
        "EN" => "n̩",
        "F" => "f",
        "G" => "ɡ",
        "HH" => "h",
        "H" => "h",
        "JH" => "dʒ",
        "K" => "k",
        "L" => "l",
        "M" => "m",
        "N" => "n",
        "NG" => "ŋ",
        "NX" => "ɾ̃",
        "P" => "p",
        "Q" => "ʔ",
        "R" => "ɹ",
        "S" => "s",
        "SH" => "ʃ",
        "T" => "t",
        "TH" => "θ",
        "V" => "v",
        "W" => "w",
        "WH" => "ʍ",
        "Y" => "j",
        "Z" => "z",
        "ZH" => "ʒ",
        // Special
        "SIL" => "sil",
        "SPN" => "spn",
        "NSN" => "nsn",
        _ => return None,
    })
}

/// Pronunciation entry with optional variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PronunciationEntry {
    /// Word (lowercase)
    pub word: String,
    /// Primary pronunciation (ARPAbet)
    pub primary: Vec<String>,
    /// Alternative pronunciations
    pub alternates: Vec<Vec<String>>,
}

/// CMU Pronouncing Dictionary
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CmuDictionary {
    entries: HashMap<String, PronunciationEntry>,
}

impl CmuDictionary {
    /// Create an empty dictionary
    pub fn new() -> Self {
        Self::default()
    }

    /// Load from CMU dict file
    ///
    /// Format: WORD  P H O N E M E S (double space) or WORD P H O N E M E S (single space)
    /// Alternates: WORD(1)  P H O N E M E S
    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut dict = Self::new();

        for line in reader.lines() {
            let line = line?;

            // Skip comments and empty lines
            if line.starts_with(";;;") || line.starts_with('#') || line.is_empty() {
                continue;
            }

            // Parse: WORD  PHONEME PHONEME ... (try double space first, then single space)
            let (word_part, phonemes): (&str, Vec<String>) = if line.contains("  ") {
                // Double-space format (traditional CMU dict)
                let parts: Vec<&str> = line.splitn(2, "  ").collect();
                if parts.len() != 2 {
                    continue;
                }
                (
                    parts[0],
                    parts[1].split_whitespace().map(|s| s.to_string()).collect(),
                )
            } else {
                // Single-space format (simplified CMU dict)
                let mut parts = line.split_whitespace();
                let word = match parts.next() {
                    Some(w) => w,
                    None => continue,
                };
                let phones: Vec<String> = parts.map(|s| s.to_string()).collect();
                if phones.is_empty() {
                    continue;
                }
                (word, phones)
            };

            // Check for alternate pronunciation marker: WORD(N)
            if let Some(paren_idx) = word_part.find('(') {
                let word = word_part[..paren_idx].to_lowercase();
                if let Some(entry) = dict.entries.get_mut(&word) {
                    entry.alternates.push(phonemes);
                }
            } else {
                let word = word_part.to_lowercase();
                dict.entries.insert(
                    word.clone(),
                    PronunciationEntry {
                        word,
                        primary: phonemes,
                        alternates: Vec::new(),
                    },
                );
            }
        }

        Ok(dict)
    }

    /// Look up a word
    pub fn get(&self, word: &str) -> Option<&PronunciationEntry> {
        self.entries.get(&word.to_lowercase())
    }

    /// Get primary pronunciation as IPA
    pub fn get_ipa(&self, word: &str) -> Option<Vec<String>> {
        self.get(word).map(|entry| {
            entry
                .primary
                .iter()
                .filter_map(|p| arpabet_to_ipa(p).map(|s| s.to_string()))
                .collect()
        })
    }

    /// Number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Add a word manually
    pub fn add(&mut self, word: &str, phonemes: Vec<String>) {
        let word_lower = word.to_lowercase();
        self.entries.insert(
            word_lower.clone(),
            PronunciationEntry {
                word: word_lower,
                primary: phonemes,
                alternates: Vec::new(),
            },
        );
    }

    /// Get all words
    pub fn words(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(|s| s.as_str())
    }
}

/// Convert text to phoneme sequences
#[derive(Debug)]
pub struct TextToPhonemes {
    dictionary: CmuDictionary,
    /// Add silence tokens between words
    insert_silence: bool,
}

impl TextToPhonemes {
    /// Create with a dictionary
    pub fn new(dictionary: CmuDictionary) -> Self {
        Self {
            dictionary,
            insert_silence: true,
        }
    }

    /// Create without dictionary (for testing)
    pub fn empty() -> Self {
        Self {
            dictionary: CmuDictionary::new(),
            insert_silence: true,
        }
    }

    /// Set whether to insert silence tokens
    pub fn with_silence(mut self, insert: bool) -> Self {
        self.insert_silence = insert;
        self
    }

    /// Convert text to ARPAbet phoneme sequence
    ///
    /// Unknown words are returned as-is in angle brackets: <UNKNOWN>
    pub fn convert(&self, text: &str) -> Vec<String> {
        let mut result = Vec::new();

        // Normalize and tokenize
        let words = self.tokenize(text);

        for (i, word) in words.iter().enumerate() {
            if let Some(entry) = self.dictionary.get(word) {
                result.extend(entry.primary.clone());
            } else {
                // Unknown word - try letter-by-letter
                let spelled = self.spell_out(word);
                if spelled.is_empty() {
                    result.push(format!("<{}>", word.to_uppercase()));
                } else {
                    result.extend(spelled);
                }
            }

            // Add inter-word silence
            if self.insert_silence && i < words.len() - 1 {
                result.push("SIL".to_string());
            }
        }

        result
    }

    /// Convert to IPA
    pub fn convert_ipa(&self, text: &str) -> Vec<String> {
        self.convert(text)
            .iter()
            .filter_map(|p| {
                if p.starts_with('<') {
                    Some(p.clone())
                } else if p == "SIL" {
                    Some("sil".to_string())
                } else {
                    arpabet_to_ipa(p).map(|s| s.to_string())
                }
            })
            .collect()
    }

    /// Simple tokenization
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphabetic() && c != '\'')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
    }

    /// Spell out unknown word letter by letter
    fn spell_out(&self, word: &str) -> Vec<String> {
        let mut result = Vec::new();

        for c in word.chars() {
            if let Some(entry) = self.dictionary.get(&c.to_string()) {
                result.extend(entry.primary.clone());
            }
        }

        result
    }

    /// Get dictionary reference
    pub fn dictionary(&self) -> &CmuDictionary {
        &self.dictionary
    }
}

/// Pronunciation trie for efficient prefix lookup
#[derive(Debug, Clone, Default)]
pub struct PronunciationTrie {
    nodes: Vec<TrieNode>,
}

#[derive(Debug, Clone, Default)]
struct TrieNode {
    children: HashMap<char, usize>,
    pronunciation: Option<Vec<String>>,
}

impl PronunciationTrie {
    /// Create a new trie
    pub fn new() -> Self {
        Self {
            nodes: vec![TrieNode::default()],
        }
    }

    /// Build from dictionary
    pub fn from_dictionary(dict: &CmuDictionary) -> Self {
        let mut trie = Self::new();
        for (word, entry) in &dict.entries {
            trie.insert(word, entry.primary.clone());
        }
        trie
    }

    /// Insert a word
    pub fn insert(&mut self, word: &str, pronunciation: Vec<String>) {
        let mut node_idx = 0;

        for c in word.chars() {
            if !self.nodes[node_idx].children.contains_key(&c) {
                let new_idx = self.nodes.len();
                self.nodes.push(TrieNode::default());
                self.nodes[node_idx].children.insert(c, new_idx);
            }
            node_idx = self.nodes[node_idx].children[&c];
        }

        self.nodes[node_idx].pronunciation = Some(pronunciation);
    }

    /// Look up a word
    pub fn get(&self, word: &str) -> Option<&Vec<String>> {
        let mut node_idx = 0;

        for c in word.chars() {
            match self.nodes[node_idx].children.get(&c) {
                Some(&next) => node_idx = next,
                None => return None,
            }
        }

        self.nodes[node_idx].pronunciation.as_ref()
    }

    /// Find longest matching prefix
    pub fn longest_prefix(&self, text: &str) -> Option<(usize, &Vec<String>)> {
        let mut node_idx = 0;
        let mut best_match: Option<(usize, &Vec<String>)> = None;

        for (i, c) in text.chars().enumerate() {
            match self.nodes[node_idx].children.get(&c) {
                Some(&next) => {
                    node_idx = next;
                    if let Some(ref pron) = self.nodes[node_idx].pronunciation {
                        best_match = Some((i + 1, pron));
                    }
                }
                None => break,
            }
        }

        best_match
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arpabet_to_ipa() {
        assert_eq!(arpabet_to_ipa("P"), Some("p"));
        assert_eq!(arpabet_to_ipa("IY1"), Some("i"));
        assert_eq!(arpabet_to_ipa("TH"), Some("θ"));
        assert_eq!(arpabet_to_ipa("INVALID"), None);
    }

    #[test]
    fn test_dictionary_add_get() {
        let mut dict = CmuDictionary::new();
        dict.add(
            "hello",
            vec!["HH".into(), "AH0".into(), "L".into(), "OW1".into()],
        );

        let entry = dict.get("hello").unwrap();
        assert_eq!(entry.primary, vec!["HH", "AH0", "L", "OW1"]);

        // Case insensitive
        assert!(dict.get("HELLO").is_some());
    }

    #[test]
    fn test_text_to_phonemes() {
        let mut dict = CmuDictionary::new();
        dict.add("the", vec!["DH".into(), "AH0".into()]);
        dict.add("cat", vec!["K".into(), "AE1".into(), "T".into()]);

        let t2p = TextToPhonemes::new(dict).with_silence(false);
        let phonemes = t2p.convert("The cat");

        assert_eq!(phonemes, vec!["DH", "AH0", "K", "AE1", "T"]);
    }

    #[test]
    fn test_trie_lookup() {
        let mut trie = PronunciationTrie::new();
        trie.insert(
            "hello",
            vec!["HH".into(), "EH".into(), "L".into(), "OW".into()],
        );
        trie.insert(
            "help",
            vec!["HH".into(), "EH".into(), "L".into(), "P".into()],
        );

        assert!(trie.get("hello").is_some());
        assert!(trie.get("help").is_some());
        assert!(trie.get("hell").is_none()); // Not a complete word
    }

    #[test]
    fn test_silence_insertion() {
        let mut dict = CmuDictionary::new();
        dict.add("a", vec!["AH0".into()]);
        dict.add("b", vec!["B".into(), "IY1".into()]);

        let t2p = TextToPhonemes::new(dict);
        let phonemes = t2p.convert("a b");

        assert!(phonemes.contains(&"SIL".to_string()));
    }
}
