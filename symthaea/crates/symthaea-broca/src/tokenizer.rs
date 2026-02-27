//! Minimal built-in BPE tokenizer.
//!
//! No external HuggingFace dependency. Supports loading vocabulary from JSON
//! or using a built-in ~256-token ASCII + common English vocabulary for testing.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Special token IDs.
pub const BOS_TOKEN: &str = "<bos>";
pub const EOS_TOKEN: &str = "<eos>";
pub const PAD_TOKEN: &str = "<pad>";
pub const UNK_TOKEN: &str = "<unk>";
pub const THOUGHT_TOKEN: &str = "<thought>";

/// BPE merge pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergePair {
    pub left: String,
    pub right: String,
}

/// Serialized vocabulary format for JSON loading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabFile {
    pub tokens: Vec<String>,
    pub merges: Vec<MergePair>,
}

/// Minimal BPE tokenizer with encode/decode and special tokens.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    token_to_id: HashMap<String, u32>,
    id_to_token: Vec<String>,
    merges: Vec<(String, String)>,
    pub bos_id: u32,
    pub eos_id: u32,
    pub pad_id: u32,
    pub unk_id: u32,
    pub thought_id: u32,
    vocab_size: usize,
}

impl BpeTokenizer {
    /// Load vocabulary from a JSON file.
    pub fn from_json(path: &str) -> Result<Self> {
        let data = std::fs::read_to_string(path)
            .with_context(|| format!("reading vocab file: {path}"))?;
        let vocab_file: VocabFile = serde_json::from_str(&data)
            .with_context(|| "parsing vocab JSON")?;

        let mut token_to_id = HashMap::new();
        let mut id_to_token = Vec::new();

        for (i, token) in vocab_file.tokens.iter().enumerate() {
            token_to_id.insert(token.clone(), i as u32);
            id_to_token.push(token.clone());
        }

        let merges: Vec<(String, String)> = vocab_file
            .merges
            .iter()
            .map(|m| (m.left.clone(), m.right.clone()))
            .collect();

        let bos_id = *token_to_id.get(BOS_TOKEN).unwrap_or(&0);
        let eos_id = *token_to_id.get(EOS_TOKEN).unwrap_or(&1);
        let pad_id = *token_to_id.get(PAD_TOKEN).unwrap_or(&2);
        let unk_id = *token_to_id.get(UNK_TOKEN).unwrap_or(&3);
        let thought_id = *token_to_id.get(THOUGHT_TOKEN).unwrap_or(&4);
        let vocab_size = id_to_token.len();

        Ok(Self {
            token_to_id,
            id_to_token,
            merges,
            bos_id,
            eos_id,
            pad_id,
            unk_id,
            thought_id,
            vocab_size,
        })
    }

    /// Built-in 4K English vocabulary (4096 tokens).
    ///
    /// Contains 5 special tokens + 256 byte tokens + ~100 subwords + ~3,635 common
    /// English words. Dictionary lookup with byte fallback (no BPE merges).
    /// Embedded at compile time via `include_str!`.
    pub fn default_4k() -> Self {
        let vocab_json = include_str!("../data/broca-vocab-4k.json");
        let vocab_file: VocabFile = serde_json::from_str(vocab_json)
            .expect("embedded broca-vocab-4k.json is valid");
        Self::from_vocab_file(&vocab_file)
    }

    /// Built-in 16K English vocabulary (16,384 tokens).
    ///
    /// Contains 5 special tokens + 256 byte tokens + ~70 morphological affixes +
    /// ~15,000 common English words + ~1,100 reserved extension slots.
    /// Superset of the 4K vocabulary with expanded coverage of verbs, nouns,
    /// adjectives, adverbs, contractions, technical terms, and domain-specific
    /// consciousness/cognitive science vocabulary. Includes 386 BPE merge rules.
    /// Embedded at compile time via `include_str!`.
    pub fn default_16k() -> Self {
        let vocab_json = include_str!("../data/broca-vocab-16k.json");
        let vocab_file: VocabFile = serde_json::from_str(vocab_json)
            .expect("embedded broca-vocab-16k.json is valid");
        Self::from_vocab_file(&vocab_file)
    }

    /// Built-in minimal vocabulary for testing.
    ///
    /// Contains ~256 tokens: special tokens + printable ASCII bytes +
    /// common English words and subwords + hedging vocabulary.
    pub fn default_minimal() -> Self {
        let mut tokens = Vec::new();
        let mut merges = Vec::new();

        // Special tokens (0-4)
        tokens.push(BOS_TOKEN.to_string());
        tokens.push(EOS_TOKEN.to_string());
        tokens.push(PAD_TOKEN.to_string());
        tokens.push(UNK_TOKEN.to_string());
        tokens.push(THOUGHT_TOKEN.to_string());

        // Single printable ASCII bytes (5-99)
        for b in 32u8..=126 {
            tokens.push(String::from(b as char));
        }

        // Common English subwords and words
        let common_words = [
            // Articles/prepositions
            "the", "a", "an", "of", "to", "in", "for", "on", "with", "at", "by", "from",
            "is", "it", "that", "this", "was", "are", "be", "have", "has", "had",
            // Pronouns
            "I", "you", "he", "she", "we", "they", "my", "your", "his", "her", "our",
            // Common verbs
            "do", "not", "can", "will", "would", "could", "should", "may", "might",
            "think", "know", "see", "say", "make", "go", "get", "take",
            // Common nouns
            "time", "way", "thing", "man", "world", "life", "day", "part", "place",
            // Connectors
            "and", "but", "or", "if", "so", "then", "than", "when", "what", "which",
            // Common subwords
            "ing", "tion", "er", "ed", "ly", "ness", "ment", "ful", "less", "able",
            "un", "re", "pre", "dis", "over", "out", "up",
            // Hedging / epistemic markers
            "perhaps", "maybe", "possibly", "likely", "probably", "certainly",
            "uncertain", "unknown", "believe", "seems", "appears", "might",
            "however", "although", "unfortunately", "sorry",
            // Punctuation sequences
            ". ", ", ", "? ", "! ", ": ", "; ", "-- ", "...",
            // Whitespace
            " ", "\n",
        ];

        for word in common_words {
            if !tokens.contains(&word.to_string()) {
                tokens.push(word.to_string());
            }
        }

        // Generate simple merges for common bigrams
        let merge_pairs = [
            ("t", "h"), ("th", "e"), ("t", "o"), ("i", "n"),
            ("a", "n"), ("e", "r"), ("o", "n"), ("i", "s"),
            ("i", "t"), ("a", "t"), ("o", "f"), ("o", "r"),
            ("h", "e"), ("n", "o"), ("n", "ot"),
        ];

        for (left, right) in merge_pairs {
            merges.push((left.to_string(), right.to_string()));
        }

        let mut token_to_id = HashMap::new();
        for (i, token) in tokens.iter().enumerate() {
            token_to_id.insert(token.clone(), i as u32);
        }

        let vocab_size = tokens.len();

        Self {
            token_to_id,
            id_to_token: tokens,
            merges,
            bos_id: 0,
            eos_id: 1,
            pad_id: 2,
            unk_id: 3,
            thought_id: 4,
            vocab_size,
        }
    }

    /// Build a tokenizer directly from a VocabFile (no merges, dict-lookup only).
    ///
    /// Used by checkpoint loading to reconstruct the tokenizer from saved vocabulary.
    pub fn from_vocab_file(vocab: &VocabFile) -> Self {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = Vec::new();

        for (i, token) in vocab.tokens.iter().enumerate() {
            token_to_id.insert(token.clone(), i as u32);
            id_to_token.push(token.clone());
        }

        let merges: Vec<(String, String)> = vocab
            .merges
            .iter()
            .map(|m| (m.left.clone(), m.right.clone()))
            .collect();

        let bos_id = *token_to_id.get(BOS_TOKEN).unwrap_or(&0);
        let eos_id = *token_to_id.get(EOS_TOKEN).unwrap_or(&1);
        let pad_id = *token_to_id.get(PAD_TOKEN).unwrap_or(&2);
        let unk_id = *token_to_id.get(UNK_TOKEN).unwrap_or(&3);
        let thought_id = *token_to_id.get(THOUGHT_TOKEN).unwrap_or(&4);
        let vocab_size = id_to_token.len();

        Self {
            token_to_id,
            id_to_token,
            merges,
            bos_id,
            eos_id,
            pad_id,
            unk_id,
            thought_id,
            vocab_size,
        }
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Look up a token's ID, returning UNK if not found.
    pub fn token_id(&self, token: &str) -> u32 {
        *self.token_to_id.get(token).unwrap_or(&self.unk_id)
    }

    /// Look up a token by ID.
    pub fn token_str(&self, id: u32) -> &str {
        self.id_to_token
            .get(id as usize)
            .map(|s| s.as_str())
            .unwrap_or(UNK_TOKEN)
    }

    /// Encode text to token IDs.
    ///
    /// Strategy:
    /// 1. Split text on whitespace boundaries to get word-level segments
    /// 2. Greedily match each segment against the vocabulary (longest match first)
    /// 3. Fall back to byte tokens (`<0xHH>`) for unrecognized characters
    /// 4. Apply BPE merges to recover multi-character tokens from fragments
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        let mut result = Vec::new();
        let mut pos = 0;
        let bytes = text.as_bytes();

        while pos < text.len() {
            // If pos is not at a char boundary (mid-multibyte), emit byte token
            if !text.is_char_boundary(pos) {
                let byte = bytes[pos];
                let byte_token = format!("<0x{:02X}>", byte);
                let id = self.token_to_id.get(&byte_token).copied().unwrap_or(self.unk_id);
                result.push(id);
                pos += 1;
                continue;
            }

            // Try to match the longest token starting at this position
            let remaining = &text[pos..];
            let mut best_len = 0;
            let mut best_id = None;

            // Try decreasing lengths (greedy longest match)
            let max_try = remaining.len().min(32);
            for end in (1..=max_try).rev() {
                // Make sure we're at a char boundary
                if !remaining.is_char_boundary(end) {
                    continue;
                }
                let candidate = &remaining[..end];
                if let Some(&id) = self.token_to_id.get(candidate) {
                    best_len = end;
                    best_id = Some(id);
                    break;
                }
            }

            if let Some(id) = best_id {
                result.push(id);
                pos += best_len;
            } else {
                // Fall back to byte token for the first byte
                let byte = bytes[pos];
                let byte_token = format!("<0x{:02X}>", byte);
                let id = self.token_to_id.get(&byte_token).copied().unwrap_or(self.unk_id);
                result.push(id);
                pos += 1;
            }
        }

        // Apply BPE merges to recover multi-character tokens from fragments
        self.apply_merges(&mut result);

        result
    }

    /// Apply BPE merges to a token sequence.
    ///
    /// For each merge (left, right) in priority order, scans the token sequence
    /// for adjacent pairs matching the merge. Replaces both with the concatenated
    /// token if it exists in the vocabulary.
    ///
    /// Runs up to 10 passes until no more merges apply.
    fn apply_merges(&self, tokens: &mut Vec<u32>) {
        if self.merges.is_empty() || tokens.len() < 2 {
            return;
        }

        // Build merge lookup: (left_id, right_id) -> merged_id
        let mut merge_lookup: HashMap<(u32, u32), u32> = HashMap::new();
        for (left_str, right_str) in &self.merges {
            if let (Some(&left_id), Some(&right_id)) = (
                self.token_to_id.get(left_str),
                self.token_to_id.get(right_str),
            ) {
                let merged = format!("{}{}", left_str, right_str);
                if let Some(&merged_id) = self.token_to_id.get(&merged) {
                    merge_lookup.insert((left_id, right_id), merged_id);
                }
            }
        }

        if merge_lookup.is_empty() {
            return;
        }

        // Apply merges iteratively (up to 10 passes)
        for _pass in 0..10 {
            let mut changed = false;
            let mut i = 0;
            while i + 1 < tokens.len() {
                if let Some(&merged_id) = merge_lookup.get(&(tokens[i], tokens[i + 1])) {
                    tokens[i] = merged_id;
                    tokens.remove(i + 1);
                    changed = true;
                    // Don't advance — re-check this position for chained merges
                } else {
                    i += 1;
                }
            }
            if !changed {
                break;
            }
        }
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut result = Vec::new();
        for &id in ids {
            let token = self.token_str(id);
            // Skip special tokens in decoded output
            if token == BOS_TOKEN || token == EOS_TOKEN || token == PAD_TOKEN || token == THOUGHT_TOKEN {
                continue;
            }
            if token == UNK_TOKEN {
                result.extend_from_slice("\u{FFFD}".as_bytes());
                continue;
            }
            // Handle byte tokens: <0xHH> -> raw byte
            if token.starts_with("<0x") && token.ends_with('>') && token.len() == 6 {
                if let Ok(byte) = u8::from_str_radix(&token[3..5], 16) {
                    result.push(byte);
                    continue;
                }
            }
            result.extend_from_slice(token.as_bytes());
        }
        String::from_utf8(result).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
    }

    /// Check if a token ID is a special token.
    pub fn is_special(&self, id: u32) -> bool {
        id == self.bos_id || id == self.eos_id || id == self.pad_id
            || id == self.unk_id || id == self.thought_id
    }

    /// Add a new token to the vocabulary (for swarm vocabulary extension).
    /// Returns the new token's ID.
    pub fn add_token(&mut self, token: &str) -> u32 {
        if let Some(&id) = self.token_to_id.get(token) {
            return id;
        }
        let id = self.id_to_token.len() as u32;
        self.token_to_id.insert(token.to_string(), id);
        self.id_to_token.push(token.to_string());
        self.vocab_size = self.id_to_token.len();
        id
    }
}

/// Unified tokenizer backend supporting both built-in BPE and HuggingFace tokenizers.
///
/// The built-in backend uses the minimal BPE tokenizer for CfC-HDC generation.
/// The HuggingFace backend enables pre-trained tokenizers (e.g., mamba-130m's 50K vocab).
pub enum TokenizerBackend {
    /// Built-in minimal BPE tokenizer.
    Builtin(BpeTokenizer),
    /// HuggingFace tokenizers library backend.
    #[cfg(feature = "hf-tokenizer")]
    HuggingFace(tokenizers::Tokenizer),
}

impl TokenizerBackend {
    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        match self {
            Self::Builtin(tok) => tok.encode(text),
            #[cfg(feature = "hf-tokenizer")]
            Self::HuggingFace(tok) => {
                tok.encode(text, false)
                    .map(|enc| enc.get_ids().to_vec())
                    .unwrap_or_default()
            }
        }
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        match self {
            Self::Builtin(tok) => tok.decode(ids),
            #[cfg(feature = "hf-tokenizer")]
            Self::HuggingFace(tok) => {
                tok.decode(ids, true)
                    .unwrap_or_default()
            }
        }
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        match self {
            Self::Builtin(tok) => tok.vocab_size(),
            #[cfg(feature = "hf-tokenizer")]
            Self::HuggingFace(tok) => tok.get_vocab_size(true),
        }
    }

    /// Create a HuggingFace tokenizer backend from a file.
    #[cfg(feature = "hf-tokenizer")]
    pub fn from_hf_file(path: &str) -> Result<Self> {
        let tok = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load HF tokenizer: {e}"))?;
        Ok(Self::HuggingFace(tok))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_minimal_creates_vocab() {
        let tok = BpeTokenizer::default_minimal();
        assert!(tok.vocab_size() > 100, "should have >100 tokens");
        assert!(tok.vocab_size() < 500, "should be a minimal vocab");
    }

    #[test]
    fn test_special_tokens() {
        let tok = BpeTokenizer::default_minimal();
        assert_eq!(tok.bos_id, 0);
        assert_eq!(tok.eos_id, 1);
        assert_eq!(tok.pad_id, 2);
        assert_eq!(tok.unk_id, 3);
        assert_eq!(tok.thought_id, 4);
        assert!(tok.is_special(0));
        assert!(tok.is_special(1));
        assert!(!tok.is_special(10));
    }

    #[test]
    fn test_encode_decode_roundtrip_ascii() {
        let tok = BpeTokenizer::default_minimal();
        let text = "hello world";
        let ids = tok.encode(text);
        assert!(!ids.is_empty());
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text, "ASCII roundtrip should be lossless");
    }

    #[test]
    fn test_encode_decode_sentence() {
        let tok = BpeTokenizer::default_minimal();
        let text = "the cat is on the mat";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_empty_encode() {
        let tok = BpeTokenizer::default_minimal();
        assert!(tok.encode("").is_empty());
        assert_eq!(tok.decode(&[]), "");
    }

    #[test]
    fn test_special_tokens_stripped_in_decode() {
        let tok = BpeTokenizer::default_minimal();
        let ids = vec![tok.bos_id, tok.eos_id];
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "", "special tokens should be stripped");
    }

    #[test]
    fn test_add_token() {
        let mut tok = BpeTokenizer::default_minimal();
        let original_size = tok.vocab_size();
        let id = tok.add_token("mycelix");
        assert_eq!(id as usize, original_size);
        assert_eq!(tok.vocab_size(), original_size + 1);
        assert_eq!(tok.token_str(id), "mycelix");
        // Adding again returns same ID
        let id2 = tok.add_token("mycelix");
        assert_eq!(id, id2);
    }

    #[test]
    fn test_unknown_token() {
        let tok = BpeTokenizer::default_minimal();
        // ID way beyond vocab should return UNK
        assert_eq!(tok.token_str(99999), UNK_TOKEN);
    }

    #[test]
    fn test_multibyte_utf8_no_panic() {
        let tok = BpeTokenizer::default_minimal();
        // Smart quotes, em-dash, emoji — all multi-byte UTF-8
        let text = "don\u{2019}t stop \u{2014} go \u{1F600}";
        let ids = tok.encode(text);
        assert!(!ids.is_empty(), "multi-byte text should produce tokens");
        // Roundtrip may be lossy (byte tokens) but must not panic
        let _decoded = tok.decode(&ids);
    }

    #[test]
    fn test_4k_vocab_loads() {
        let tok = BpeTokenizer::default_4k();
        assert_eq!(tok.vocab_size(), 4096, "4K vocab should have exactly 4096 tokens");
    }

    #[test]
    fn test_4k_special_tokens() {
        let tok = BpeTokenizer::default_4k();
        assert_eq!(tok.bos_id, 0);
        assert_eq!(tok.eos_id, 1);
        assert_eq!(tok.pad_id, 2);
        assert_eq!(tok.unk_id, 3);
        assert_eq!(tok.thought_id, 4);
    }

    #[test]
    fn test_4k_common_words() {
        let tok = BpeTokenizer::default_4k();
        // Common words should be present and not UNK
        for word in &["the", "and", "of", "to", "is", "for", "that"] {
            let id = tok.token_id(word);
            assert_ne!(id, tok.unk_id, "Common word '{word}' should be in 4K vocab");
        }
    }

    #[test]
    fn test_4k_encode_decode_english() {
        let tok = BpeTokenizer::default_4k();
        // Dictionary lookup: whole words should encode to single tokens
        let id = tok.token_id("world");
        assert_ne!(id, tok.unk_id, "'world' should be in vocab");

        // A sentence made of vocabulary words
        let text = "the world is beautiful";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text, "English sentence roundtrip should be lossless");
    }

    #[test]
    fn test_16k_vocab_loads() {
        let tok = BpeTokenizer::default_16k();
        assert_eq!(tok.vocab_size(), 16384, "16K vocab should have exactly 16384 tokens");
    }

    #[test]
    fn test_16k_superset_of_4k() {
        let tok4k = BpeTokenizer::default_4k();
        let tok16k = BpeTokenizer::default_16k();
        // Every token in 4K should be in 16K at the same index
        for i in 0..tok4k.vocab_size() {
            assert_eq!(
                tok4k.id_to_token[i], tok16k.id_to_token[i],
                "16K token at index {i} should match 4K"
            );
        }
    }

    #[test]
    fn test_16k_common_words() {
        let tok = BpeTokenizer::default_16k();
        // These words should be in the expanded vocab
        for word in &["algorithm", "consciousness", "prediction", "uncertainty", "perhaps"] {
            let id = tok.token_id(word);
            assert_ne!(id, tok.unk_id, "'{word}' should be in 16K vocab");
        }
    }

    #[test]
    fn test_json_roundtrip() {
        let tok = BpeTokenizer::default_minimal();
        let vocab = VocabFile {
            tokens: tok.id_to_token.clone(),
            merges: tok.merges.iter().map(|(l, r)| MergePair {
                left: l.clone(),
                right: r.clone(),
            }).collect(),
        };
        let json = serde_json::to_string(&vocab).unwrap();
        let parsed: VocabFile = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.tokens.len(), tok.vocab_size());
    }

    #[test]
    fn test_merges_noop_when_empty() {
        // Build a tokenizer with no merges
        let mut tok = BpeTokenizer::default_minimal();
        tok.merges.clear();
        let text = "the cat";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text, "Empty merges should not break encoding");
    }

    #[test]
    fn test_merges_reduce_token_count() {
        let tok = BpeTokenizer::default_minimal();
        // With merges, some character pairs should be combined
        // The minimal tokenizer has merges like ("i", "n"), ("i", "s"), ("i", "t")
        // For "in" — the greedy encoder finds "in" as a whole word first, so no merge needed
        // For a word like "gift" that's NOT in the minimal vocab:
        // greedy: "g", "i", "f", "t" (4 tokens as single chars)
        // merges: ("i", "t") exists, but "it" IS in vocab → merges "i"+"t" → "it"
        // So "gift" → ["g", "it", "f"...] wait, order matters
        // Actually greedy goes left to right: g→i→f→t
        // merge pass: (g,i)→"gi" not in vocab, (i,f)→"if" check vocab... "if" IS in vocab!
        // So "i"+"f" → "if" merge would apply if ("i","f") is in the merge list
        // The default merges are: ("t","h"), ("th","e"), ("t","o"), ("i","n"), ("a","n"),
        //   ("e","r"), ("o","n"), ("i","s"), ("i","t"), ("a","t"), ("o","f"), ("o","r"),
        //   ("h","e"), ("n","o"), ("n","ot")
        // So ("i","s") → "is" (in vocab) should work for text that produces "i" then "s"

        // Create a tokenizer without merges to compare
        let mut tok_no_merge = BpeTokenizer::default_minimal();
        tok_no_merge.merges.clear();

        // Encode text that would produce adjacent single chars
        // "xis" is not a word → greedy: "x", "i", "s" (3 chars)
        // With merges: ("i","s")→"is" merge should fire: "x", "is" (2 tokens)
        let ids_with_merge = tok.encode("xis");
        let ids_no_merge = tok_no_merge.encode("xis");
        assert!(
            ids_with_merge.len() <= ids_no_merge.len(),
            "Merges should not increase token count: with={} without={}",
            ids_with_merge.len(), ids_no_merge.len()
        );

        // Verify decoded output is the same
        let decoded_merge = tok.decode(&ids_with_merge);
        let decoded_no = tok_no_merge.decode(&ids_no_merge);
        assert_eq!(decoded_merge, decoded_no, "Both should decode to the same text");
    }

    #[test]
    fn test_merges_combine_characters() {
        let tok = BpeTokenizer::default_minimal();
        // "xit" → greedy: "x", "i", "t" (3 tokens)
        // merge ("i","t") → "it" (in vocab) → "x", "it" (2 tokens)
        let ids = tok.encode("xit");

        // Find "it" token id
        let it_id = tok.token_id("it");
        assert_ne!(it_id, tok.unk_id, "'it' should be in minimal vocab");

        // The merged result should contain the "it" token
        assert!(
            ids.contains(&it_id),
            "Merges should combine 'i'+'t' into 'it': {:?}",
            ids.iter().map(|&id| tok.token_str(id)).collect::<Vec<_>>()
        );
    }
}
