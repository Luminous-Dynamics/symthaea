// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Ported from fl-aggregator/src/proofs/field.rs
//! Field element conversions for zkSTARK proofs.
//!
//! Utilities for converting between Rust types and Winterfell's f128 field
//! elements used in STARK proofs.

use crate::error::{ProofError, ProofResult};
use winterfell::math::{fields::f128::BaseElement, FieldElement, StarkField};

/// Scale factor for converting f32 to field elements.
/// Preserves 3 decimal places of precision.
pub const F32_SCALE: u64 = 1000;

/// Maximum value representable without overflow.
pub const MAX_SCALED_F32: u64 = u64::MAX / F32_SCALE;

/// Convert f32 to a scaled BaseElement.
///
/// Scales by `F32_SCALE` (1000) to preserve decimal precision.
/// Negative values are not supported (field elements are unsigned).
pub fn f32_to_field(value: f32) -> ProofResult<BaseElement> {
    if value < 0.0 {
        return Err(ProofError::FieldOverflow(format!(
            "Negative values not supported: {}",
            value
        )));
    }

    let scaled = (value * F32_SCALE as f32) as u64;
    if scaled > MAX_SCALED_F32 {
        return Err(ProofError::FieldOverflow(format!(
            "Value {} exceeds maximum representable value",
            value
        )));
    }

    Ok(BaseElement::from(scaled))
}

/// Convert a scaled BaseElement back to f32.
pub fn field_to_f32(element: BaseElement) -> f32 {
    let value: u128 = element.as_int();
    (value as f64 / F32_SCALE as f64) as f32
}

/// Convert u64 to BaseElement.
pub fn u64_to_field(value: u64) -> BaseElement {
    BaseElement::from(value)
}

/// Convert u32 to BaseElement.
pub fn u32_to_field(value: u32) -> BaseElement {
    BaseElement::from(value as u64)
}

/// Convert bool to BaseElement (0 or 1).
pub fn bool_to_field(value: bool) -> BaseElement {
    if value {
        BaseElement::ONE
    } else {
        BaseElement::ZERO
    }
}

/// Convert BaseElement to u64 (may truncate for large field values).
pub fn field_to_u64(element: BaseElement) -> u64 {
    element.as_int() as u64
}

/// Convert a 256-bit hash (32 bytes) to 4 BaseElements.
///
/// Each BaseElement holds 64 bits of the hash.
pub fn hash_to_field_elements(hash: &[u8; 32]) -> [BaseElement; 4] {
    let mut elements = [BaseElement::ZERO; 4];

    for (i, chunk) in hash.chunks(8).enumerate() {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(chunk);
        elements[i] = BaseElement::from(u64::from_le_bytes(bytes));
    }

    elements
}

/// Convert 4 BaseElements back to a 256-bit hash.
pub fn field_elements_to_hash(elements: &[BaseElement; 4]) -> [u8; 32] {
    let mut hash = [0u8; 32];

    for (i, element) in elements.iter().enumerate() {
        let value: u128 = element.as_int();
        let bytes = (value as u64).to_le_bytes();
        hash[i * 8..(i + 1) * 8].copy_from_slice(&bytes);
    }

    hash
}

/// Trait for types that can be converted to field elements.
pub trait ToFieldElements {
    fn to_field_elements(&self) -> ProofResult<Vec<BaseElement>>;
}

impl ToFieldElements for f32 {
    fn to_field_elements(&self) -> ProofResult<Vec<BaseElement>> {
        Ok(vec![f32_to_field(*self)?])
    }
}

impl ToFieldElements for u64 {
    fn to_field_elements(&self) -> ProofResult<Vec<BaseElement>> {
        Ok(vec![u64_to_field(*self)])
    }
}

impl ToFieldElements for u32 {
    fn to_field_elements(&self) -> ProofResult<Vec<BaseElement>> {
        Ok(vec![u32_to_field(*self)])
    }
}

impl ToFieldElements for bool {
    fn to_field_elements(&self) -> ProofResult<Vec<BaseElement>> {
        Ok(vec![bool_to_field(*self)])
    }
}

impl ToFieldElements for [u8; 32] {
    fn to_field_elements(&self) -> ProofResult<Vec<BaseElement>> {
        Ok(hash_to_field_elements(self).to_vec())
    }
}

/// Decompose a value into bits (least significant first).
pub fn decompose_to_bits(value: u64, num_bits: usize) -> Vec<BaseElement> {
    let mut bits = Vec::with_capacity(num_bits);
    let mut v = value;

    for _ in 0..num_bits {
        bits.push(bool_to_field(v & 1 == 1));
        v >>= 1;
    }

    bits
}

/// Recompose bits back to a value.
pub fn recompose_from_bits(bits: &[BaseElement]) -> BaseElement {
    let mut result = BaseElement::ZERO;
    let mut power = BaseElement::ONE;
    let two = BaseElement::from(2u64);

    for bit in bits {
        result += *bit * power;
        power *= two;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_conversion() {
        let original = 0.75f32;
        let field = f32_to_field(original).unwrap();
        let recovered = field_to_f32(field);
        assert!((original - recovered).abs() < 0.001);
    }

    #[test]
    fn test_f32_negative_fails() {
        assert!(f32_to_field(-1.0).is_err());
    }

    #[test]
    fn test_u64_conversion() {
        let original = 42u64;
        let field = u64_to_field(original);
        let recovered = field_to_u64(field);
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_bool_conversion() {
        assert_eq!(bool_to_field(true), BaseElement::ONE);
        assert_eq!(bool_to_field(false), BaseElement::ZERO);
    }

    #[test]
    fn test_hash_roundtrip() {
        let hash = [42u8; 32];
        let elements = hash_to_field_elements(&hash);
        let recovered = field_elements_to_hash(&elements);
        assert_eq!(hash, recovered);
    }

    #[test]
    fn test_bit_decomposition() {
        let value = 13u64; // 1101 in binary
        let bits = decompose_to_bits(value, 4);

        assert_eq!(bits[0], BaseElement::ONE);  // LSB = 1
        assert_eq!(bits[1], BaseElement::ZERO); // 0
        assert_eq!(bits[2], BaseElement::ONE);  // 1
        assert_eq!(bits[3], BaseElement::ONE);  // MSB = 1

        let recomposed = recompose_from_bits(&bits);
        assert_eq!(field_to_u64(recomposed), value);
    }

    #[test]
    fn test_bit_decomposition_roundtrip() {
        for value in [0, 1, 127, 255, 65535, u64::MAX >> 1] {
            let bits = decompose_to_bits(value, 64);
            let recomposed = recompose_from_bits(&bits);
            assert_eq!(field_to_u64(recomposed), value);
        }
    }

    #[test]
    fn test_to_field_elements_trait() {
        assert!(0.5f32.to_field_elements().is_ok());
        assert!(42u64.to_field_elements().is_ok());
        assert!(10u32.to_field_elements().is_ok());
        assert!(true.to_field_elements().is_ok());
        assert!([0u8; 32].to_field_elements().is_ok());
    }
}
