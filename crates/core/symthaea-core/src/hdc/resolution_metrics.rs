//! Measurement primitives for adaptive-resolution projection experiments.
//!
//! These metrics intentionally operate on the packed research type. They are
//! not claims about production HDC semantics; they quantify how much a
//! projection changes observable binary geometry.

use super::adaptive_resolution::PackedBinaryHv;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimilarityMetrics {
    pub source_similarity: f64,
    pub projected_similarity: f64,
    pub absolute_error: f64,
}

fn popcount_xor(a: &[u8], b: &[u8]) -> u64 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones() as u64).sum()
}

/// Binary Hamming similarity: 1.0 identical, 0.0 maximally different.
pub fn hamming_similarity(a: &PackedBinaryHv, b: &PackedBinaryHv) -> Option<f64> {
    if a.resolution() != b.resolution() {
        return None;
    }
    let bits = a.resolution().bits() as u64;
    Some(1.0 - popcount_xor(a.as_bytes(), b.as_bytes()) as f64 / bits as f64)
}

/// Compare the similarity relationship of a pair before and after projection.
pub fn pair_similarity_metrics(
    source_a: &PackedBinaryHv,
    source_b: &PackedBinaryHv,
    projected_a: &PackedBinaryHv,
    projected_b: &PackedBinaryHv,
) -> Option<SimilarityMetrics> {
    let source_similarity = hamming_similarity(source_a, source_b)?;
    let projected_similarity = hamming_similarity(projected_a, projected_b)?;
    Some(SimilarityMetrics {
        source_similarity,
        projected_similarity,
        absolute_error: (source_similarity - projected_similarity).abs(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::adaptive_resolution::HdcResolution;

    fn hv(r: HdcResolution, fill: u8) -> PackedBinaryHv {
        PackedBinaryHv::from_bytes(r, vec![fill; r.bytes()]).unwrap()
    }

    #[test]
    fn identical_vectors_have_unit_similarity() {
        let a = hv(HdcResolution::D4K, 0xaa);
        assert_eq!(hamming_similarity(&a, &a), Some(1.0));
    }

    #[test]
    fn_opposites_have_zero_similarity() {
        let a = hv(HdcResolution::D4K, 0x00);
        let b = hv(HdcResolution::D4K, 0xff);
        assert_eq!(hamming_similarity(&a, &b), Some(0.0));
    }

    #[test]
    fn mismatched_resolution_is_not_silently_comparable() {
        let a = hv(HdcResolution::D4K, 0xaa);
        let b = hv(HdcResolution::D8K, 0xaa);
        assert_eq!(hamming_similarity(&a, &b), None);
    }

    #[test]
    fn pair_error_is_explicit() {
        let a = hv(HdcResolution::D4K, 0x00);
        let b = hv(HdcResolution::D4K, 0xff);
        let pa = hv(HdcResolution::D2K, 0x00);
        let pb = hv(HdcResolution::D2K, 0x00);
        let m = pair_similarity_metrics(&a, &b, &pa, &pb).unwrap();
        assert_eq!(m.source_similarity, 0.0);
        assert_eq!(m.projected_similarity, 1.0);
        assert_eq!(m.absolute_error, 1.0);
    }
}
