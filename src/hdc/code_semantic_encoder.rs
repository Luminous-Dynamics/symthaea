use symthaea_core::hdc::ContinuousHV;

/// Encodes code-related text into HDC vectors with synonym awareness.
pub struct CodeSemanticEncoder {
    dim: usize,
}

impl CodeSemanticEncoder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Encode a text string into a continuous HD vector using character-trigram hashing.
    pub fn encode_text(&self, text: &str) -> ContinuousHV {
        let mut v = vec![0.0f32; self.dim];
        let bytes = text.as_bytes();
        for window in bytes.windows(3) {
            let hash = (window[0] as usize)
                .wrapping_mul(31)
                .wrapping_add(window[1] as usize)
                .wrapping_mul(31)
                .wrapping_add(window[2] as usize);
            let idx = hash % self.dim;
            let sign = if (hash / self.dim) % 2 == 0 { 1.0f32 } else { -1.0f32 };
            v[idx] += sign;
        }
        // Normalize
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        for val in v.iter_mut() {
            *val /= norm;
        }
        ContinuousHV::from_vec(v)
    }
}
