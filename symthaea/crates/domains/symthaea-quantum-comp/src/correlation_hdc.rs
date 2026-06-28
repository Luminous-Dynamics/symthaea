//! Correlation-style binding sketches.
//!
//! This module deliberately does not represent physical entanglement. It models
//! a classical parity/correlation sketch that can later be exported into tiny
//! circuit experiments or compared against quantum backend encodings.

use crate::classical_hdc::BinaryHypervector;
use crate::errors::Result;

/// A compact pairwise-correlation binding sketch.
///
/// For binary HDC, the stored parity vector is equivalent to XOR binding. The
/// value of this wrapper is conceptual and experimental: it makes the correlation
/// assumption explicit so future circuit export, backend telemetry, and topology
/// probes have a stable type to target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorrelationBindingSketch {
    dimension: usize,
    parity: BinaryHypervector,
}

impl CorrelationBindingSketch {
    /// Creates a sketch from an item and key by recording their pairwise parity.
    pub fn bind(item: &BinaryHypervector, key: &BinaryHypervector) -> Result<Self> {
        let parity = item.bind_xor(key)?;
        Ok(Self {
            dimension: item.dimension(),
            parity,
        })
    }

    /// Creates a sketch directly from a parity hypervector.
    pub fn from_parity(parity: BinaryHypervector) -> Self {
        Self {
            dimension: parity.dimension(),
            parity,
        }
    }

    /// Returns the sketch dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the stored parity hypervector.
    pub fn parity(&self) -> &BinaryHypervector {
        &self.parity
    }

    /// Recovers the item side when the key side is supplied.
    pub fn recover_item(&self, key: &BinaryHypervector) -> Result<BinaryHypervector> {
        self.parity.unbind_xor(key)
    }

    /// Recovers the key side when the item side is supplied.
    pub fn recover_key(&self, item: &BinaryHypervector) -> Result<BinaryHypervector> {
        self.parity.unbind_xor(item)
    }

    /// Similarity between the stored parity sketch and another parity-like vector.
    pub fn parity_similarity(&self, other: &BinaryHypervector) -> Result<f32> {
        self.parity.similarity(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correlation_sketch_recovers_item() {
        let item = BinaryHypervector::random(512, 10).unwrap();
        let key = BinaryHypervector::random(512, 11).unwrap();
        let sketch = CorrelationBindingSketch::bind(&item, &key).unwrap();
        let recovered = sketch.recover_item(&key).unwrap();
        assert_eq!(item, recovered);
    }
}
