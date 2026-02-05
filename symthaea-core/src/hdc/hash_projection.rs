//! Deterministic hash-based projection to bit-packed hypervectors (HV16)
//!
//! This follows the v1.2 design: any byte sequence is projected via BLAKE3
//! into a stable 2048-bit vector. Same input → same vector across runs and
//! machines, making HDC behavior reproducible.

use blake3::Hasher;

use super::binary_hv::HV16;

/// Content-addressed specification for deterministic encoding.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct ContentSpec {
    /// Semantic content (e.g., "firefox", "install").
    pub content: String,
    /// Optional namespace for disambiguation (e.g., domain, modality).
    pub namespace: Option<String>,
    /// Version for schema evolution; change to avoid collisions when formats shift.
    pub version: u8,
}

impl ContentSpec {
    /// Deterministic 32-byte hash of the specification.
    pub fn content_hash(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(self.content.as_bytes());
        if let Some(ns) = &self.namespace {
            hasher.update(b"::");
            hasher.update(ns.as_bytes());
        }
        hasher.update(&[self.version]);
        *hasher.finalize().as_bytes()
    }
}

/// Project any byte sequence to a bit-packed HV16 using BLAKE3 XOF.
pub fn project_to_hv(bytes: &[u8]) -> HV16 {
    let seed = blake3::hash(bytes);
    expand_hash_to_hv(seed.as_bytes())
}

/// Expand a 32-byte hash to a 2048-bit hypervector.
pub fn expand_hash_to_hv(seed: &[u8; 32]) -> HV16 {
    let mut result = [0u8; HV16::BYTES];

    // Use extendable output to fill the full 256 bytes.
    let mut hasher = Hasher::new();
    hasher.update(seed);
    let mut output = hasher.finalize_xof();
    output.fill(&mut result);

    HV16(result)
}

/// Deterministically encode text into a sequence of HV16 vectors (one per token).
///
/// Splits on ASCII whitespace; each token is hashed independently so identical
/// tokens across runs map to the same hypervector without storing state.
pub fn encode_text_to_hv16s(text: &str) -> Vec<HV16> {
    text.split_whitespace()
        .map(|tok| project_to_hv(tok.as_bytes()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_projection() {
        let data = b"install firefox";
        let hv1 = project_to_hv(data);
        let hv2 = project_to_hv(data);
        assert_eq!(hv1, hv2, "Projection must be deterministic");
    }

    #[test]
    fn different_inputs_diverge() {
        let hv1 = project_to_hv(b"install firefox");
        let hv2 = project_to_hv(b"install chromium");
        // Random-like vectors should land near 0.5 similarity.
        let sim = hv1.similarity(&hv2);
        assert!(sim > 0.45 && sim < 0.55, "Different inputs should be near-orthogonal (got {sim})");
    }

    #[test]
    fn content_spec_hash_is_stable() {
        let spec = ContentSpec {
            content: "firefox".into(),
            namespace: Some("pkg".into()),
            version: 1,
        };
        let hash1 = spec.content_hash();
        let hash2 = spec.content_hash();
        assert_eq!(hash1, hash2, "ContentSpec hash must be stable");

        let hv1 = expand_hash_to_hv(&hash1);
        let hv2 = expand_hash_to_hv(&hash2);
        assert_eq!(hv1, hv2, "Hash expansion must be deterministic");
    }

    #[test]
    fn encode_text_is_deterministic_per_token() {
        let hv1 = encode_text_to_hv16s("install ripgrep");
        let hv2 = encode_text_to_hv16s("install ripgrep");
        assert_eq!(hv1, hv2, "Token encoding must be deterministic");
        assert_eq!(hv1.len(), 2);

        let hv3 = encode_text_to_hv16s("install ripgrep --profile dev");
        assert!(hv3.len() >= 2);
        // First two tokens should match regardless of trailing flags.
        assert_eq!(hv1[0], hv3[0]);
        assert_eq!(hv1[1], hv3[1]);
    }
}
