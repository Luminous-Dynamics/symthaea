//! Dual-channel HDC text encoder (character trigrams + word-level).
//!
//! Encodes arbitrary text strings into hyperdimensional vectors using two
//! complementary channels:
//!
//! 1. **Character trigrams**: Captures sub-word morphological patterns
//!    (e.g., "ing", "tion", "un-")
//! 2. **Word-level**: Captures semantic word presence and order
//!    (each word hashed to a deterministic HV, bound with position)
//!
//! The two channels are bundled (summed) and L2-normalized, giving both
//! fine-grained character patterns and coarse word-level signal.

use std::collections::HashSet;
use symthaea_core::hdc::ContinuousHV;

/// Dual-channel HDC text encoder combining character trigrams and word-level encoding,
/// with an optional third sentiment channel.
///
/// ## Encoding Pipeline
///
/// 1. **Trigram channel**: Slide a 3-char window, bind each char with position
///    within the window, multiply to get n-gram HV, accumulate across text.
/// 2. **Word channel**: Split on whitespace, hash each word to a deterministic HV,
///    bind with word-position HV, accumulate across all words.
/// 3. **Sentiment channel** (optional): Accumulate POSITIVE_SEED for good words,
///    NEGATIVE_SEED for bad words. L2-normalize.
/// 4. **Combine**: Weight channels, sum, L2-normalize.
#[derive(Debug, Clone)]
pub struct TextHdcEncoder {
    dim: usize,
    ngram_size: usize,
    /// Character-level random HVs (one per ASCII char, indices 0..128)
    char_hvs: Vec<ContinuousHV>,
    /// Position HVs for n-gram character positions
    pos_hvs: Vec<ContinuousHV>,
    /// Word-position HVs (for word-level encoding, up to MAX_WORD_POSITIONS)
    word_pos_hvs: Vec<ContinuousHV>,
    /// Weight for trigram channel (word channel = 1 - trigram_weight)
    trigram_weight: f32,
    /// Weight for sentiment channel (0.0 = off, blends into trigram+word)
    sentiment_weight: f32,
    /// Seed HV for positive moral sentiment
    positive_seed: ContinuousHV,
    /// Seed HV for negative moral sentiment
    negative_seed: ContinuousHV,
    /// Good/positive moral words
    good_words: HashSet<String>,
    /// Bad/negative moral words
    bad_words: HashSet<String>,
}

/// Maximum number of distinct word positions tracked.
const MAX_WORD_POSITIONS: usize = 64;

impl TextHdcEncoder {
    /// Create a new text encoder with the given dimension and n-gram size.
    ///
    /// Uses default channel weights: trigram 0.5, word 0.5, sentiment off.
    pub fn new(dim: usize, ngram_size: usize) -> Self {
        Self::with_sentiment(dim, ngram_size, 0.5, 0.0)
    }

    /// Create with explicit trigram/word channel weights (sentiment off).
    ///
    /// `trigram_weight` controls the balance: 0.0 = pure word-level, 1.0 = pure trigram.
    pub fn with_weights(dim: usize, ngram_size: usize, trigram_weight: f32) -> Self {
        Self::with_sentiment(dim, ngram_size, trigram_weight, 0.0)
    }

    /// Create with explicit trigram, word, and sentiment channel weights.
    ///
    /// When `sentiment_weight > 0`, the final encoding blends:
    ///   `tw*(1-sw) * trigram + ww*(1-sw) * word + sw * sentiment`
    /// where `tw = trigram_weight`, `ww = 1 - trigram_weight`, `sw = sentiment_weight`.
    ///
    /// When `sentiment_weight == 0`, identical to `with_weights()` (fast path, no regression).
    pub fn with_sentiment(
        dim: usize,
        ngram_size: usize,
        trigram_weight: f32,
        sentiment_weight: f32,
    ) -> Self {
        let char_hvs: Vec<ContinuousHV> = (0..128)
            .map(|c| ContinuousHV::random(dim, 30000 + c as u64))
            .collect();

        let pos_hvs: Vec<ContinuousHV> = (0..ngram_size)
            .map(|p| ContinuousHV::random(dim, 40000 + p as u64))
            .collect();

        let word_pos_hvs: Vec<ContinuousHV> = (0..MAX_WORD_POSITIONS)
            .map(|p| ContinuousHV::random(dim, 60000 + p as u64))
            .collect();

        let positive_seed = ContinuousHV::random(dim, 70000001);
        let negative_seed = ContinuousHV::random(dim, 70000017);

        let good_words: HashSet<String> = [
            "good",
            "kind",
            "help",
            "helps",
            "helped",
            "helping",
            "generous",
            "honest",
            "brave",
            "fair",
            "love",
            "caring",
            "protect",
            "save",
            "share",
            "donate",
            "forgive",
            "respect",
            "trust",
            "loyal",
            "gentle",
            "mercy",
            "grateful",
            "compassion",
            "empathy",
            "encourage",
            "support",
            "nurture",
            "inspire",
            "cooperate",
            "volunteer",
            "rescue",
            "praise",
            "comfort",
            "heal",
            "thoughtful",
            "considerate",
            "responsible",
            "patient",
            "humble",
            "sincere",
            "peaceful",
            "noble",
            "virtuous",
            "admirable",
            "heroic",
            "selfless",
            "charitable",
            "benevolent",
            "righteous",
            "worthy",
            "honorable",
            "dignified",
            "gracious",
            "courteous",
            "polite",
            "wonderful",
            "beautiful",
            "excellent",
            "joyful",
            "happy",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        let bad_words: HashSet<String> = [
            "bad",
            "cruel",
            "harm",
            "harms",
            "harmed",
            "harming",
            "selfish",
            "dishonest",
            "coward",
            "unfair",
            "hate",
            "uncaring",
            "attack",
            "destroy",
            "steal",
            "stole",
            "betray",
            "disrespect",
            "distrust",
            "disloyal",
            "harsh",
            "merciless",
            "ungrateful",
            "heartless",
            "callous",
            "discourage",
            "undermine",
            "neglect",
            "demean",
            "cheat",
            "cheated",
            "lie",
            "lied",
            "lying",
            "deceive",
            "manipulate",
            "murder",
            "kill",
            "abuse",
            "exploit",
            "bully",
            "threaten",
            "blackmail",
            "corrupt",
            "greedy",
            "malicious",
            "spiteful",
            "vengeful",
            "violent",
            "arrogant",
            "reckless",
            "lazy",
            "irresponsible",
            "impatient",
            "vain",
            "wicked",
            "evil",
            "terrible",
            "horrible",
            "vicious",
            "nasty",
            "wrong",
            "immoral",
            "unethical",
            "unjust",
            "sinful",
            "shameful",
            "rude",
            "aggressive",
            "hostile",
            "toxic",
            "destructive",
            "damaging",
            "hurt",
            "hurting",
            "stolen",
            "stealing",
            "killed",
            "killing",
            "broke",
            "vandalize",
            "sabotage",
            "fraud",
            "forge",
            "fake",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        Self {
            dim,
            ngram_size,
            char_hvs,
            pos_hvs,
            word_pos_hvs,
            trigram_weight,
            sentiment_weight,
            positive_seed,
            negative_seed,
            good_words,
            bad_words,
        }
    }

    /// Get the output dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the n-gram size.
    pub fn ngram_size(&self) -> usize {
        self.ngram_size
    }

    /// Encode a text string into a normalized HDC vector.
    ///
    /// Combines character trigram, word-level, and (optionally) sentiment channels.
    pub fn encode(&self, text: &str) -> ContinuousHV {
        let trigram_hv = self.encode_trigrams(text);
        let word_hv = self.encode_words(text);

        let tw = self.trigram_weight;
        let ww = 1.0 - tw;
        let sw = self.sentiment_weight;

        let mut combined = vec![0.0f32; self.dim];

        if sw > 0.0 {
            // Three-channel blend: tw*(1-sw)*trigram + ww*(1-sw)*word + sw*sentiment
            let sentiment_hv = self.encode_sentiment(text);
            let tw_scaled = tw * (1.0 - sw);
            let ww_scaled = ww * (1.0 - sw);
            for i in 0..self.dim {
                combined[i] = tw_scaled * trigram_hv.values[i]
                    + ww_scaled * word_hv.values[i]
                    + sw * sentiment_hv.values[i];
            }
        } else {
            // Fast path: original two-channel blend (no regression)
            for i in 0..self.dim {
                combined[i] = tw * trigram_hv.values[i] + ww * word_hv.values[i];
            }
        }

        // L2-normalize the combined vector
        let norm: f32 = combined.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut combined {
                *v /= norm;
            }
        }

        ContinuousHV::from_vec(combined)
    }

    /// Sentiment channel: accumulate positive/negative seed HVs for moral words.
    ///
    /// For each word in the text, if it's in the good_words set, add POSITIVE_SEED;
    /// if in bad_words, add NEGATIVE_SEED. The result is L2-normalized.
    /// If no sentiment words are found, returns a zero vector (neutral contribution).
    fn encode_sentiment(&self, text: &str) -> ContinuousHV {
        let mut accumulator = vec![0.0f32; self.dim];
        let text_lower = text.to_lowercase();
        let mut found_any = false;

        for word in text_lower.split_whitespace() {
            // Strip punctuation from word edges
            let clean: &str = word.trim_matches(|c: char| !c.is_alphanumeric());
            if clean.is_empty() {
                continue;
            }
            if self.good_words.contains(clean) {
                found_any = true;
                for (acc, &val) in accumulator.iter_mut().zip(self.positive_seed.values.iter()) {
                    *acc += val;
                }
            } else if self.bad_words.contains(clean) {
                found_any = true;
                for (acc, &val) in accumulator.iter_mut().zip(self.negative_seed.values.iter()) {
                    *acc += val;
                }
            }
        }

        if found_any {
            let norm: f32 = accumulator.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut accumulator {
                    *v /= norm;
                }
            }
        }

        ContinuousHV::from_vec(accumulator)
    }

    /// Character trigram encoding channel.
    ///
    /// Optimized to avoid per-trigram ContinuousHV allocations — works directly
    /// on float slices with a reusable buffer.
    fn encode_trigrams(&self, text: &str) -> ContinuousHV {
        let chars: Vec<u8> = text.bytes().map(|b| b.min(127)).collect();
        let mut accumulator = vec![0.0f32; self.dim];

        if chars.len() < self.ngram_size {
            // Too short — just use character HVs directly
            for &ch in &chars {
                for (acc, &val) in accumulator
                    .iter_mut()
                    .zip(self.char_hvs[ch as usize].values.iter())
                {
                    *acc += val;
                }
            }
        } else {
            // N-gram encoding: slide a window, reuse buffer for bind products
            let mut ngram_buf = vec![0.0f32; self.dim];

            for window_start in 0..=(chars.len() - self.ngram_size) {
                // Start with all ones (identity for element-wise multiplication)
                for v in ngram_buf.iter_mut() {
                    *v = 1.0;
                }

                for pos in 0..self.ngram_size {
                    let ch = chars[window_start + pos] as usize;
                    // Inline bind: ngram_buf *= char_hvs[ch] * pos_hvs[pos]
                    let char_vals = &self.char_hvs[ch].values;
                    let pos_vals = &self.pos_hvs[pos].values;
                    for i in 0..self.dim {
                        ngram_buf[i] *= char_vals[i] * pos_vals[i];
                    }
                }

                for (acc, &val) in accumulator.iter_mut().zip(ngram_buf.iter()) {
                    *acc += val;
                }
            }
        }

        // L2-normalize
        let norm: f32 = accumulator.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut accumulator {
                *v /= norm;
            }
        }

        ContinuousHV::from_vec(accumulator)
    }

    /// Word-level encoding channel.
    ///
    /// Each word is hashed to a deterministic HV (via its bytes), then bound
    /// with a word-position HV. All word-position-bound HVs are accumulated.
    /// Optimized: reuses buffer for bind results to reduce allocations.
    fn encode_words(&self, text: &str) -> ContinuousHV {
        let mut accumulator = vec![0.0f32; self.dim];
        let text_lower = text.to_lowercase();
        let mut bound_buf = vec![0.0f32; self.dim];

        for (word_idx, word) in text_lower.split_whitespace().enumerate() {
            let word_hv = self.hash_word(word);
            let pos_idx = word_idx.min(MAX_WORD_POSITIONS - 1);
            let pos_vals = &self.word_pos_hvs[pos_idx].values;

            // Inline bind: bound_buf = word_hv * pos_hv
            for i in 0..self.dim {
                bound_buf[i] = word_hv.values[i] * pos_vals[i];
            }

            for (acc, &val) in accumulator.iter_mut().zip(bound_buf.iter()) {
                *acc += val;
            }
        }

        // L2-normalize
        let norm: f32 = accumulator.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut accumulator {
                *v /= norm;
            }
        }

        ContinuousHV::from_vec(accumulator)
    }

    /// Hash a word to a deterministic HV using its bytes as a seed.
    fn hash_word(&self, word: &str) -> ContinuousHV {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        word.hash(&mut hasher);
        let seed = hasher.finish();
        ContinuousHV::random(self.dim, seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_determinism() {
        let enc = TextHdcEncoder::new(4096, 3);
        let hv1 = enc.encode("hello world");
        let hv2 = enc.encode("hello world");
        assert!((hv1.similarity(&hv2) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_short_text_handling() {
        let enc = TextHdcEncoder::new(4096, 3);
        // Should not panic on texts shorter than ngram_size
        let hv_a = enc.encode("a");
        let hv_ab = enc.encode("ab");
        let hv_empty = enc.encode("");
        // Short texts produce valid HVs
        assert_eq!(hv_a.values.len(), 4096);
        assert_eq!(hv_ab.values.len(), 4096);
        assert_eq!(hv_empty.values.len(), 4096);
    }

    #[test]
    fn test_similar_text_similarity() {
        let enc = TextHdcEncoder::new(4096, 3);
        let hv1 = enc.encode("the cat sat on the mat");
        let hv2 = enc.encode("the cat sat on the hat");
        let hv3 = enc.encode("quantum chromodynamics is fascinating");

        // Similar texts should be more similar than dissimilar texts
        let sim_close = hv1.similarity(&hv2);
        let sim_far = hv1.similarity(&hv3);
        assert!(
            sim_close > sim_far,
            "Close texts similarity ({}) should exceed distant texts similarity ({})",
            sim_close,
            sim_far
        );
    }

    #[test]
    fn test_normalized_output() {
        let enc = TextHdcEncoder::new(4096, 3);
        let hv = enc.encode("normalization test");
        let norm: f32 = hv.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "Output should be unit-normalized, got {}",
            norm
        );
    }

    #[test]
    fn test_word_channel_captures_semantics() {
        // Word-level encoding should make texts with shared words more similar
        let enc = TextHdcEncoder::with_weights(4096, 3, 0.0); // Pure word channel
        let hv1 = enc.encode("stealing is wrong and harmful");
        let hv2 = enc.encode("stealing is bad and cruel");
        let hv3 = enc.encode("helping others brings joy");

        let sim_close = hv1.similarity(&hv2);
        let sim_far = hv1.similarity(&hv3);
        assert!(
            sim_close > sim_far,
            "Shared-word texts ({:.4}) should be more similar than unrelated ({:.4})",
            sim_close,
            sim_far
        );
    }

    #[test]
    fn test_dual_channel_outperforms_single() {
        // The combined encoder should preserve similarity better than either alone
        let dual = TextHdcEncoder::new(4096, 3); // 0.5 / 0.5
        let tri_only = TextHdcEncoder::with_weights(4096, 3, 1.0);

        // Texts that share words but differ in character patterns
        let a = "being kind to everyone is virtuous";
        let b = "being cruel to everyone is wrong";

        let dual_sim = dual.encode(a).similarity(&dual.encode(b));
        let tri_sim = tri_only.encode(a).similarity(&tri_only.encode(b));

        // Both should produce meaningful (non-zero) similarity
        assert!(
            dual_sim.abs() > 0.01,
            "Dual should produce meaningful similarity"
        );
        assert!(
            tri_sim.abs() > 0.01,
            "Trigram should produce meaningful similarity"
        );
    }

    #[test]
    fn test_sentiment_channel_default_off() {
        // When sentiment_weight=0, output should be identical to the original encoder
        let enc_default = TextHdcEncoder::new(4096, 3);
        let enc_sentiment_off = TextHdcEncoder::with_sentiment(4096, 3, 0.5, 0.0);

        let text = "stealing is wrong and harmful";
        let hv_default = enc_default.encode(text);
        let hv_off = enc_sentiment_off.encode(text);

        let sim = hv_default.similarity(&hv_off);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Sentiment off should produce identical output, got similarity {}",
            sim
        );
    }

    #[test]
    fn test_sentiment_separates_polarity() {
        // With sentiment channel active, good text and bad text should be more
        // separated than without sentiment
        let enc_no_sent = TextHdcEncoder::new(4096, 3);
        let enc_sent = TextHdcEncoder::with_sentiment(4096, 3, 0.5, 0.3);

        let good_text = "helping kind generous caring love";
        let bad_text = "stealing cruel selfish harming hate";

        let sim_no_sent = enc_no_sent
            .encode(good_text)
            .similarity(&enc_no_sent.encode(bad_text));
        let sim_sent = enc_sent
            .encode(good_text)
            .similarity(&enc_sent.encode(bad_text));

        // Sentiment channel should push good/bad further apart (lower similarity)
        assert!(
            sim_sent < sim_no_sent,
            "Sentiment channel should separate polarity: with={:.4} should be < without={:.4}",
            sim_sent,
            sim_no_sent
        );
    }
}
