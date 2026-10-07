//! Deterministic, seed-addressable hypervector materialization research primitive.
//!
//! This module deliberately does not replace the production fixed-width
//! BinaryHV. It provides a reproducible way to study resident-vs-regenerated
//! representations across the 1K..64K research ladder.
//!
//! Generator identity is part of representation identity: changing the
//! algorithm/version, seed, or dimension changes the materialization subject.

use blake3::Hasher;

/// Versioned identity of the deterministic binary materializer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BinaryMaterializerSpec {
    /// Generator schema/version.
    pub version: u8,
    /// Public deterministic seed.
    pub seed: u64,
    /// Number of output bits. Must be a multiple of 8.
    pub bits: usize,
}

impl BinaryMaterializerSpec {
    pub const VERSION: u8 = 1;

    pub const fn new(seed: u64, bits: usize) -> Self {
        Self { version: Self::VERSION, seed, bits }
    }

    pub const fn bytes(self) -> usize {
        self.bits / 8
    }

    pub const fn is_byte_aligned(self) -> bool {
        self.bits != 0 && self.bits % 8 == 0
    }
}

/// Materialize a deterministic binary HV from an explicit specification.
///
/// The byte stream is generated with BLAKE3 XOF over the canonical
/// little-endian encoding of (version, seed, bits). Including the dimension
/// and generator version prevents accidental cross-resolution aliasing.
///
/// This is a research primitive, not a cryptographic key generator.
pub fn materialize_binary(spec: BinaryMaterializerSpec) -> Option<Vec<u8>> {
    if !spec.is_byte_aligned() {
        return None;
    }

    let mut hasher = Hasher::new();
    hasher.update(&[spec.version]);
    hasher.update(&spec.seed.to_le_bytes());
    hasher.update(&(spec.bits as u64).to_le_bytes());

    let mut bytes = vec![0u8; spec.bytes()];
    hasher.finalize_xof().fill(&mut bytes);
    Some(bytes)
}

/// Compact resident-state estimate for a regenerable binary HV.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaterializationCost {
    pub specification_bytes: usize,
    pub materialized_bytes: usize,
}

/// Return the representation-size comparison without materializing the HV.
pub const fn cost(spec: BinaryMaterializerSpec) -> Option<MaterializationCost> {
    if !spec.is_byte_aligned() {
        return None;
    }

    // version + seed + bits. This is the canonical serialized scalar payload;
    // allocator/header overhead is intentionally excluded.
    Some(MaterializationCost {
        specification_bytes: 1 + 8 + 8,
        materialized_bytes: spec.bytes(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_spec_is_bit_exact() {
        let spec = BinaryMaterializerSpec::new(42, 16_384);
        assert_eq!(materialize_binary(spec), materialize_binary(spec));
    }

    #[test]
    fn dimension_is_part_of_identity() {
        let a = BinaryMaterializerSpec::new(42, 8_192);
        let b = BinaryMaterializerSpec::new(42, 16_384);
        let a_bytes = materialize_binary(a).unwrap();
        let b_bytes = materialize_binary(b).unwrap();
        assert_ne!(a_bytes, b_bytes[..a.bytes()]);
    }

    #[test]
    fn ladder_sizes_are_exact() {
        let expected = [
            (1_024, 128),
            (2_048, 256),
            (4_096, 512),
            (8_192, 1_024),
            (16_384, 2_048),
            (32_768, 4_096),
            (65_536, 8_192),
        ];

        for (bits, bytes) in expected {
            let spec = BinaryMaterializerSpec::new(7, bits);
            assert_eq!(spec.bytes(), bytes);
            assert_eq!(materialize_binary(spec).unwrap().len(), bytes);
        }
    }

    #[test]
    fn invalid_alignment_fails_closed() {
        assert!(materialize_binary(BinaryMaterializerSpec::new(7, 1_025)).is_none());
        assert!(cost(BinaryMaterializerSpec::new(7, 0)).is_none());
    }

    #[test]
    fn compact_spec_is_smaller_than_16k_resident_vector() {
        let spec = BinaryMaterializerSpec::new(7, 16_384);
        let measured = cost(spec).unwrap();
        assert!(measured.specification_bytes < measured.materialized_bytes);
        assert_eq!(measured.specification_bytes, 17);
    }

    #[test]
    fn version_changes_materialization_identity() {
        let a = BinaryMaterializerSpec::new(7, 16_384);
        let b = BinaryMaterializerSpec { version: 2, ..a };
        assert_ne!(materialize_binary(a), materialize_binary(b));
    }
}
