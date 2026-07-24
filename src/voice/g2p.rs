// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Grapheme-to-Phoneme (G2P) Conversion
//!
//! Converts text to phoneme IDs for Kokoro TTS model input.
//! Uses espeak-ng via espeak-rs when available for accurate IPA phonemization,
//! then converts IPA to Misaki phoneme format expected by Kokoro.
//!
//! Falls back to a basic lookup table when espeak-ng is not available.

use std::collections::HashMap;

/// Misaki phoneme vocabulary for Kokoro TTS.
/// Maps IPA symbols to Kokoro phoneme IDs (0-44).
/// Based on: <https://github.com/hexgrad/misaki/blob/main/EN_PHONES.md>
pub struct MisakiVocab {
    ipa_to_id: HashMap<&'static str, u32>,
}

impl MisakiVocab {
    /// Create the Misaki phoneme vocabulary.
    pub fn new() -> Self {
        let mut ipa_to_id = HashMap::new();

        // American English phonemes (45 total)
        // Vowels - uppercase = long/diphthong, lowercase = short
        ipa_to_id.insert("A", 0); // FACE diphthong
        ipa_to_id.insert("I", 1); // PRICE diphthong
        ipa_to_id.insert("W", 2); // MOUTH diphthong
        ipa_to_id.insert("Y", 3); // CHOICE diphthong
        ipa_to_id.insert("O", 4); // GOAT diphthong

        // Consonants
        ipa_to_id.insert("b", 5);
        ipa_to_id.insert("d", 6);
        ipa_to_id.insert("f", 7);
        ipa_to_id.insert("h", 8);
        ipa_to_id.insert("j", 10); // y as in yes
        ipa_to_id.insert("k", 11);
        ipa_to_id.insert("l", 12);
        ipa_to_id.insert("m", 13);
        ipa_to_id.insert("n", 14);
        ipa_to_id.insert("p", 15);

        // IPA vowels
        ipa_to_id.insert("ɑ", 16); // PALM/LOT (American merged)
        ipa_to_id.insert("ɔ", 17); // THOUGHT
        ipa_to_id.insert("ə", 18); // schwa (COMMA)
        ipa_to_id.insert("ɛ", 19); // DRESS
        ipa_to_id.insert("ɜ", 20); // NURSE
        ipa_to_id.insert("ɪ", 21); // KIT
        ipa_to_id.insert("ʊ", 22); // FOOT
        ipa_to_id.insert("ʌ", 23); // STRUT
        ipa_to_id.insert("i", 24); // FLEECE
        ipa_to_id.insert("u", 25); // GOOSE
        ipa_to_id.insert("ɹ", 26); // r (American)
        ipa_to_id.insert("ɾ", 27); // flap t/d
        ipa_to_id.insert("ᵻ", 28); // unstressed KIT/schwa
        ipa_to_id.insert("z", 29);
        ipa_to_id.insert("v", 30);
        ipa_to_id.insert("w", 31);

        // Fricatives and affricates
        ipa_to_id.insert("ʃ", 32); // SH
        ipa_to_id.insert("ʒ", 33); // ZH (measure)
        ipa_to_id.insert("ʤ", 34); // J (judge)
        ipa_to_id.insert("dʒ", 34); // alternate J
        ipa_to_id.insert("ʧ", 35); // CH (church)
        ipa_to_id.insert("tʃ", 35); // alternate CH

        // Prosody
        ipa_to_id.insert("ˈ", 36); // primary stress
        ipa_to_id.insert("ˌ", 37); // secondary stress

        // More consonants
        ipa_to_id.insert("θ", 38); // TH (think)
        ipa_to_id.insert("ð", 39); // TH (this)
        ipa_to_id.insert("ŋ", 40); // NG (sing)
        ipa_to_id.insert("ɡ", 41); // g (IPA symbol)
        ipa_to_id.insert("g", 41); // g (ASCII fallback)
        ipa_to_id.insert("ᵊ", 42); // optional schwa
        ipa_to_id.insert("æ", 43); // TRAP

        // Common IPA variants and alternates
        ipa_to_id.insert("s", 29); // map s to z slot (they share)
        ipa_to_id.insert("t", 6); // t maps to d slot (unvoiced pair)
        ipa_to_id.insert("r", 26); // r ASCII to ɹ
        ipa_to_id.insert("ʔ", 18); // glottal stop → schwa (reduced)
        ipa_to_id.insert("ː", 18); // length marker → ignored (schwa)
        ipa_to_id.insert("ɐ", 23); // near-open central → STRUT
        ipa_to_id.insert("e", 19); // close-mid front → DRESS
        ipa_to_id.insert("o", 17); // close-mid back → THOUGHT
        ipa_to_id.insert("a", 16); // open front → PALM

        Self { ipa_to_id }
    }

    /// Convert an IPA string to Misaki phoneme IDs.
    pub fn ipa_to_phoneme_ids(&self, ipa: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let chars: Vec<char> = ipa.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            // Try two-character sequences first (affricates, etc.)
            if i + 1 < chars.len() {
                let two_char: String = chars[i..=i + 1].iter().collect();
                if let Some(&id) = self.ipa_to_id.get(two_char.as_str()) {
                    ids.push(id);
                    i += 2;
                    continue;
                }
            }

            // Single character
            let one_char = chars[i].to_string();
            if let Some(&id) = self.ipa_to_id.get(one_char.as_str()) {
                ids.push(id);
            }
            // Skip unknown characters (spaces, punctuation, etc.)

            i += 1;
        }

        ids
    }
}

impl Default for MisakiVocab {
    fn default() -> Self {
        Self::new()
    }
}

/// The REAL Kokoro-82M token vocabulary (115 tokens, IDs 0-177, sparse),
/// generated verbatim from `onnx-community/Kokoro-82M-v1.0-ONNX/tokenizer.json`
/// (fetched 2026-07-15, voice plan LF4). Every token is a single char.
///
/// This is deliberately separate from [`MisakiVocab`]: that 45-ID space is
/// also consumed by the orchestrator's phoneme→formant fallback table, so it
/// must not change; the Kokoro engine needs *these* IDs to produce anything
/// but garbled audio.
pub fn kokoro_token_id(c: char) -> Option<u32> {
    match c {
        '$' => Some(0),
        ';' => Some(1),
        ':' => Some(2),
        ',' => Some(3),
        '.' => Some(4),
        '!' => Some(5),
        '?' => Some(6),
        '—' => Some(9),
        '…' => Some(10),
        '"' => Some(11),
        '(' => Some(12),
        ')' => Some(13),
        '“' => Some(14),
        '”' => Some(15),
        ' ' => Some(16),
        '\u{303}' => Some(17),
        'ʣ' => Some(18),
        'ʥ' => Some(19),
        'ʦ' => Some(20),
        'ʨ' => Some(21),
        'ᵝ' => Some(22),
        'ꭧ' => Some(23),
        'A' => Some(24),
        'I' => Some(25),
        'O' => Some(31),
        'Q' => Some(33),
        'S' => Some(35),
        'T' => Some(36),
        'W' => Some(39),
        'Y' => Some(41),
        'ᵊ' => Some(42),
        'a' => Some(43),
        'b' => Some(44),
        'c' => Some(45),
        'd' => Some(46),
        'e' => Some(47),
        'f' => Some(48),
        'h' => Some(50),
        'i' => Some(51),
        'j' => Some(52),
        'k' => Some(53),
        'l' => Some(54),
        'm' => Some(55),
        'n' => Some(56),
        'o' => Some(57),
        'p' => Some(58),
        'q' => Some(59),
        'r' => Some(60),
        's' => Some(61),
        't' => Some(62),
        'u' => Some(63),
        'v' => Some(64),
        'w' => Some(65),
        'x' => Some(66),
        'y' => Some(67),
        'z' => Some(68),
        'ɑ' => Some(69),
        'ɐ' => Some(70),
        'ɒ' => Some(71),
        'æ' => Some(72),
        'β' => Some(75),
        'ɔ' => Some(76),
        'ɕ' => Some(77),
        'ç' => Some(78),
        'ɖ' => Some(80),
        'ð' => Some(81),
        'ʤ' => Some(82),
        'ə' => Some(83),
        'ɚ' => Some(85),
        'ɛ' => Some(86),
        'ɜ' => Some(87),
        'ɟ' => Some(90),
        'ɡ' => Some(92),
        'ɥ' => Some(99),
        'ɨ' => Some(101),
        'ɪ' => Some(102),
        'ʝ' => Some(103),
        'ɯ' => Some(110),
        'ɰ' => Some(111),
        'ŋ' => Some(112),
        'ɳ' => Some(113),
        'ɲ' => Some(114),
        'ɴ' => Some(115),
        'ø' => Some(116),
        'ɸ' => Some(118),
        'θ' => Some(119),
        'œ' => Some(120),
        'ɹ' => Some(123),
        'ɾ' => Some(125),
        'ɻ' => Some(126),
        'ʁ' => Some(128),
        'ɽ' => Some(129),
        'ʂ' => Some(130),
        'ʃ' => Some(131),
        'ʈ' => Some(132),
        'ʧ' => Some(133),
        'ʊ' => Some(135),
        'ʋ' => Some(136),
        'ʌ' => Some(138),
        'ɣ' => Some(139),
        'ɤ' => Some(140),
        'χ' => Some(142),
        'ʎ' => Some(143),
        'ʒ' => Some(147),
        'ʔ' => Some(148),
        'ˈ' => Some(156),
        'ˌ' => Some(157),
        'ː' => Some(158),
        'ʰ' => Some(162),
        'ʲ' => Some(164),
        '↓' => Some(169),
        '→' => Some(171),
        '↗' => Some(172),
        '↘' => Some(173),
        'ᵻ' => Some(177),
        _ => None,
    }
}

/// G2P converter using espeak-ng when available, with fallback to lookup table.
pub struct G2PConverter {
    /// Misaki vocabulary for IPA → phoneme ID conversion
    _vocab: MisakiVocab,
    /// Word → phoneme ID sequence lookup table (fallback)
    word_map: HashMap<String, Vec<u32>>,
    /// Character → phoneme ID fallback
    char_map: HashMap<char, u32>,
    /// Whether espeak-ng is available
    #[cfg(feature = "voice-tts")]
    espeak_available: bool,
    /// Silence token ID
    pub silence_id: u32,
    /// Padding token ID
    pub pad_id: u32,
}

impl G2PConverter {
    /// Create a new G2P converter.
    /// Attempts to initialize espeak-ng if the voice-tts feature is enabled.
    pub fn new() -> Self {
        let vocab = MisakiVocab::new();

        #[cfg(feature = "voice-tts")]
        let espeak_available = Self::init_espeak();

        let mut word_map = HashMap::new();
        let mut char_map = HashMap::new();

        // Common English words → phoneme ID sequences (fallback)
        // These use Misaki phoneme IDs
        let common_words: &[(&str, &[u32])] = &[
            ("hello", &[8, 19, 12, 4]),  // hɛloʊ → h ɛ l O
            ("world", &[31, 20, 12, 6]), // wɜld → w ɜ l d
            ("the", &[39, 18]),          // ðə → ð ə
            ("a", &[18]),                // ə
            ("an", &[18, 14]),           // ən → ə n
            ("is", &[21, 29]),           // ɪz → ɪ z
            ("and", &[43, 14, 6]),       // ænd → æ n d
            ("to", &[6, 25]),            // tu → t u
            ("of", &[23, 30]),           // ʌv → ʌ v
            ("in", &[21, 14]),           // ɪn → ɪ n
            ("for", &[7, 17, 26]),       // fɔr → f ɔ r
            ("on", &[16, 14]),           // ɑn → ɑ n
            ("it", &[21, 6]),            // ɪt → ɪ t
            ("that", &[39, 43, 6]),      // ðæt → ð æ t
            ("this", &[39, 21, 29]),     // ðɪs → ð ɪ s
            ("with", &[31, 21, 38]),     // wɪθ → w ɪ θ
            ("not", &[14, 16, 6]),       // nɑt → n ɑ t
            ("you", &[10, 25]),          // ju → j u
            ("we", &[31, 24]),           // wi → w i
            ("consciousness", &[11, 16, 14, 30, 18, 29, 14, 18, 29]), // kɑnʃəsnəs
            ("symthaea", &[29, 21, 13, 38, 24, 18]), // sɪmθiə
            ("intelligence", &[21, 14, 6, 19, 12, 21, 6, 18, 14, 29]), // ɪntɛlɪdʒəns
            ("neural", &[14, 22, 26, 18, 12]), // nʊrəl
            ("system", &[29, 21, 29, 6, 18, 13]), // sɪstəm
        ];

        for (word, phonemes) in common_words {
            word_map.insert(word.to_string(), phonemes.to_vec());
        }

        // Character-level fallback mapping (rough approximation)
        let char_phonemes: &[(char, u32)] = &[
            ('a', 43),
            ('b', 5),
            ('c', 11),
            ('d', 6),
            ('e', 19),
            ('f', 7),
            ('g', 41),
            ('h', 8),
            ('i', 21),
            ('j', 34),
            ('k', 11),
            ('l', 12),
            ('m', 13),
            ('n', 14),
            ('o', 16),
            ('p', 15),
            ('q', 11),
            ('r', 26),
            ('s', 29),
            ('t', 6),
            ('u', 23),
            ('v', 30),
            ('w', 31),
            ('x', 11),
            ('y', 10),
            ('z', 29),
            (' ', 0),
        ];

        for (ch, id) in char_phonemes {
            char_map.insert(*ch, *id);
        }

        Self {
            _vocab: vocab,
            word_map,
            char_map,
            #[cfg(feature = "voice-tts")]
            espeak_available,
            silence_id: 0,
            pad_id: 0,
        }
    }

    /// Initialize espeak-ng.
    #[cfg(feature = "voice-tts")]
    fn init_espeak() -> bool {
        // espeak-rs initializes on first use
        // We'll check availability when we first try to use it
        true
    }

    /// Convert text to phoneme IDs using espeak-ng.
    #[cfg(feature = "voice-tts")]
    fn text_to_phonemes_espeak(&self, text: &str) -> Option<Vec<u32>> {
        use espeak_rs::text_to_phonemes;

        match text_to_phonemes(text, "en-us", None, false, false) {
            Ok(phoneme_strings) => {
                let mut ids = Vec::new();
                for phoneme_str in phoneme_strings {
                    ids.extend(self._vocab.ipa_to_phoneme_ids(&phoneme_str));
                    ids.push(self.silence_id); // silence between sentences
                }
                Some(ids)
            }
            Err(_) => None,
        }
    }

    /// Convert text to REAL Kokoro-82M token IDs (espeak IPA → per-char
    /// [`kokoro_token_id`] lookup). Returns `None` when espeak is unavailable
    /// or produced nothing — there is deliberately no fallback here, because
    /// feeding approximate IDs to the neural model is how the old 45-ID
    /// mapping produced garble. Unknown IPA chars are skipped.
    #[cfg(feature = "voice-tts")]
    pub fn text_to_kokoro_tokens(&self, text: &str) -> Option<Vec<u32>> {
        use espeak_rs::text_to_phonemes;

        if !self.espeak_available {
            return None;
        }
        let phoneme_strings = text_to_phonemes(text, "en-us", None, false, false).ok()?;
        let mut ids = Vec::new();
        for (i, sentence) in phoneme_strings.iter().enumerate() {
            if i > 0 {
                // Sentence boundary → period token.
                ids.push(4);
            }
            for c in sentence.trim().chars() {
                // espeak emits newlines between clauses; normalize to space.
                let c = if c == '\n' { ' ' } else { c };
                if let Some(id) = kokoro_token_id(c) {
                    ids.push(id);
                }
            }
        }
        if ids.is_empty() { None } else { Some(ids) }
    }

    /// Convert text to a sequence of phoneme IDs.
    pub fn text_to_phonemes(&self, text: &str) -> Vec<u32> {
        // Try espeak-ng first if available
        #[cfg(feature = "voice-tts")]
        if self.espeak_available {
            if let Some(ids) = self.text_to_phonemes_espeak(text) {
                if !ids.is_empty() {
                    return ids;
                }
            }
        }

        // Fallback to lookup table
        self.text_to_phonemes_fallback(text)
    }

    /// Fallback phoneme conversion using lookup table.
    fn text_to_phonemes_fallback(&self, text: &str) -> Vec<u32> {
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
        45 // Misaki uses 45 phoneme tokens for American English
    }

    /// Get the number of words in the lookup table.
    pub fn dictionary_size(&self) -> usize {
        self.word_map.len()
    }

    /// Check if espeak-ng is available.
    pub fn has_espeak(&self) -> bool {
        #[cfg(feature = "voice-tts")]
        {
            self.espeak_available
        }
        #[cfg(not(feature = "voice-tts"))]
        {
            false
        }
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
    fn test_misaki_vocab() {
        let vocab = MisakiVocab::new();

        // Test basic IPA conversion
        let ids = vocab.ipa_to_phoneme_ids("hɛˈloʊ");
        assert!(!ids.is_empty());
        assert!(ids.contains(&8)); // h
        assert!(ids.contains(&19)); // ɛ
    }

    #[test]
    fn test_known_word() {
        let g2p = G2PConverter::new();
        let phonemes = g2p.text_to_phonemes("hello");
        assert!(!phonemes.is_empty());
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

    #[test]
    fn test_vocab_size() {
        let g2p = G2PConverter::new();
        assert_eq!(g2p.vocab_size(), 45);
    }

    #[test]
    fn test_kokoro_vocab_spot_checks() {
        // Verbatim values from onnx-community/Kokoro-82M-v1.0-ONNX
        // tokenizer.json (2026-07-15).
        assert_eq!(kokoro_token_id('$'), Some(0));
        assert_eq!(kokoro_token_id(' '), Some(16));
        assert_eq!(kokoro_token_id('a'), Some(43));
        assert_eq!(kokoro_token_id('z'), Some(68));
        assert_eq!(kokoro_token_id('ə'), Some(83));
        assert_eq!(kokoro_token_id('ʃ'), Some(131));
        assert_eq!(kokoro_token_id('ˈ'), Some(156));
        assert_eq!(kokoro_token_id('ᵻ'), Some(177));
        assert_eq!(kokoro_token_id('~'), None);
    }

    #[cfg(feature = "voice-tts")]
    #[test]
    fn test_kokoro_tokens_via_espeak() {
        let g2p = G2PConverter::new();
        if let Some(ids) = g2p.text_to_kokoro_tokens("hello world") {
            assert!(ids.len() >= 6, "expected several tokens: {ids:?}");
            // All IDs must be within the model's 0-177 sparse space.
            assert!(ids.iter().all(|&id| id <= 177));
            // Word boundary space token should be present.
            assert!(ids.contains(&16));
        }
        // None (espeak missing at runtime) is acceptable — no fallback by design.
    }
}
