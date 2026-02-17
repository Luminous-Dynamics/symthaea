//! Optional Symthaea-backed HyperFeel integration.
//!
//! When the `symthaea-hv` feature is enabled, this module provides
//! conversion helpers between the local `Hypervector` type and
//! Symthaea's `BinaryHV` / `HDC_DIMENSION`.

#![cfg(feature = "symthaea-hv")]

use symthaea::core::{BinaryHV, HDC_DIMENSION};

use crate::hypervector::Hypervector;

/// Dimension alias for Symthaea-backed hypervectors.
pub const SYMTHAEA_DIMENSION: usize = HDC_DIMENSION;

/// Number of bytes for a Symthaea `BinaryHV`.
pub const SYMTHAEA_BYTES: usize = HDC_DIMENSION / 8;

/// Convert a local `Hypervector` (i8 values) into a Symthaea `BinaryHV`.
///
/// Convention:
/// - Values > 0 map to bit 1 (+1 in bipolar)
/// - Values <= 0 map to bit 0 (-1 in bipolar)
pub fn to_symthaea_binary(hv: &Hypervector) -> BinaryHV {
    let mut bytes = vec![0u8; SYMTHAEA_BYTES];
    let dim = hv.data.len().min(HDC_DIMENSION);

    for i in 0..dim {
        if hv.data[i] > 0 {
            let byte = i / 8;
            let bit = i % 8;
            bytes[byte] |= 1 << bit;
        }
    }

    BinaryHV::from_bytes(bytes)
}

/// Convert a Symthaea `BinaryHV` into a local `Hypervector`.
///
/// Convention:
/// - +1 in bipolar maps to 1_i8
/// - -1 in bipolar maps to -1_i8
pub fn from_symthaea_binary(binary: &BinaryHV) -> Hypervector {
    let mut data = vec![0i8; Hypervector::DIMENSION];
    let dim = Hypervector::DIMENSION.min(binary.dim());

    for i in 0..dim {
        let v = binary.get_bipolar(i);
        data[i] = if v >= 0 { 1 } else { -1 };
    }

    Hypervector { data }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_sign_preservation_for_prefix() {
        // Create a random Symthaea BinaryHV
        let original = BinaryHV::random(42);

        // Convert to local Hypervector and back
        let hv = from_symthaea_binary(&original);
        let back = to_symthaea_binary(&hv);

        // Check the first 256 bits for sign consistency
        let check_bits = 256.min(original.dim());
        for i in 0..check_bits {
            let a = original.get_bipolar(i);
            let b = back.get_bipolar(i);
            assert_eq!(a.signum(), b.signum(), "sign mismatch at bit {}", i);
        }
    }
}
