// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! The binary field GF(2⁸) as used by AES and Reed-Solomon codes.
//!
//! Elements are bytes (polynomials of degree < 8 over GF(2)); addition is XOR
//! and multiplication is carry-less multiplication reduced modulo the AES
//! irreducible polynomial `x⁸ + x⁴ + x³ + x + 1` (0x11B).

/// Addition in GF(2⁸) — the same as subtraction — is XOR.
pub fn add(a: u8, b: u8) -> u8 {
    a ^ b
}

/// Multiplication in GF(2⁸) modulo the AES polynomial (Russian-peasant with
/// reduction by 0x1B).
pub fn mul(mut a: u8, mut b: u8) -> u8 {
    let mut product: u8 = 0;
    for _ in 0..8 {
        if b & 1 != 0 {
            product ^= a;
        }
        let high_bit = a & 0x80;
        a <<= 1; // u8 shift drops the overflow bit
        if high_bit != 0 {
            a ^= 0x1B; // reduce modulo x⁸ + x⁴ + x³ + x + 1
        }
        b >>= 1;
    }
    product
}

/// `aᵉ` by repeated squaring.
pub fn pow(a: u8, mut e: u32) -> u8 {
    let mut result: u8 = 1;
    let mut base = a;
    while e > 0 {
        if e & 1 == 1 {
            result = mul(result, base);
        }
        base = mul(base, base);
        e >>= 1;
    }
    result
}

/// The multiplicative inverse in GF(2⁸): `a^254` (since `a^255 = 1` for every
/// nonzero `a`). `None` for `0`.
pub fn inverse(a: u8) -> Option<u8> {
    if a == 0 {
        return None;
    }
    Some(pow(a, 254))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fips197_multiplication_example() {
        // The canonical FIPS-197 example: 0x57 · 0x83 = 0xC1.
        assert_eq!(mul(0x57, 0x83), 0xC1);
    }

    #[test]
    fn addition_is_xor_and_self_inverse() {
        assert_eq!(add(0x57, 0x83), 0x57 ^ 0x83);
        assert_eq!(add(0xAB, 0xAB), 0); // a + a = 0
    }

    #[test]
    fn one_is_identity_and_inverses_round_trip() {
        assert_eq!(mul(0x53, 1), 0x53);
        for a in 1u8..=255 {
            let inv = inverse(a).unwrap();
            assert_eq!(mul(a, inv), 1, "a={a:#x}");
        }
        assert_eq!(inverse(0), None);
    }

    #[test]
    fn multiplication_is_commutative() {
        for &(a, b) in &[(0x57u8, 0x83u8), (0x02, 0xF6), (0x11, 0x77)] {
            assert_eq!(mul(a, b), mul(b, a));
        }
    }
}
