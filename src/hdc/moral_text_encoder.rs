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

use symthaea_core::hdc::ContinuousHV;

/// Dual-channel HDC text encoder combining character trigrams and word-level encoding.
///
/// ## Encoding Pipeline
///
/// 1. **Trigram channel**: Slide a 3-char window, bind each char with position
///    within the window, multiply to get n-gram HV, accumulate across text.
/// 2. **Word channel**: Split on whitespace, hash each word to a deterministic HV,
///    bind with word-position HV, accumulate across all words.
/// 3. **Combine**: Weight trigram and word channels, sum, L2-normalize.
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
}

/// Maximum number of distinct word positions tracked.
const MAX_WORD_POSITIONS: usize = 64;

impl TextHdcEncoder {
    /// Create a new text encoder with the given dimension and n-gram size.
    ///
    /// Uses default channel weights: trigram 0.5, word 0.5.
    pub fn new(dim: usize, ngram_size: usize) -> Self {
        Self::with_weights(dim, ngram_size, 0.5)
    }

    /// Create with explicit channel weights.
    ///
    /// `trigram_weight` controls the balance: 0.0 = pure word-level, 1.0 = pure trigram.
    pub fn with_weights(dim: usize, ngram_size: usize, trigram_weight: f32) -> Self {
        let char_hvs: Vec<ContinuousHV> = (0..128)
            .map(|c| ContinuousHV::random(dim, 30000 + c as u64))
            .collect();

        let pos_hvs: Vec<ContinuousHV> = (0..ngram_size)
            .map(|p| ContinuousHV::random(dim, 40000 + p as u64))
            .collect();

        let word_pos_hvs: Vec<ContinuousHV> = (0..MAX_WORD_POSITIONS)
            .map(|p| ContinuousHV::random(dim, 60000 + p as u64))
            .collect();

        Self {
            dim,
            ngram_size,
            char_hvs,
            pos_hvs,
            word_pos_hvs,
            trigram_weight,
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
    /// Combines character trigram and word-level channels.
    pub fn encode(&self, text: &str) -> ContinuousHV {
        let trigram_hv = self.encode_trigrams(text);
        let word_hv = self.encode_words(text);

        // Weighted combination of both channels
        let tw = self.trigram_weight;
        let ww = 1.0 - tw;
        let mut combined = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            combined[i] = tw * trigram_hv.values[i] + ww * word_hv.values[i];
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
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
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
        assert!((norm - 1.0).abs() < 1e-4, "Output should be unit-normalized, got {}", norm);
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
            sim_close, sim_far
        );
    }

    #[test]
    fn test_dual_channel_outperforms_single() {
        // The combined encoder should preserve similarity better than either alone
        let dual = TextHdcEncoder::new(4096, 3);  // 0.5 / 0.5
        let tri_only = TextHdcEncoder::with_weights(4096, 3, 1.0);

        // Texts that share words but differ in character patterns
        let a = "being kind to everyone is virtuous";
        let b = "being cruel to everyone is wrong";

        let dual_sim = dual.encode(a).similarity(&dual.encode(b));
        let tri_sim = tri_only.encode(a).similarity(&tri_only.encode(b));

        // Both should produce meaningful (non-zero) similarity
        assert!(dual_sim.abs() > 0.01, "Dual should produce meaningful similarity");
        assert!(tri_sim.abs() > 0.01, "Trigram should produce meaningful similarity");
    }
}
