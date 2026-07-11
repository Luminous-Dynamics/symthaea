// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Reed-Solomon codes over **GF(2⁸)** (the AES field, via
//! `symthaea-finite-field`) — systematic encoding and syndrome-based error
//! detection. This is the bridge from finite-field algebra to real-world codes
//! (QR codes, CDs, deep-space telemetry).
//!
//! Polynomials are most-significant-coefficient first. The generator has roots
//! `α⁰ … α^{nsym−1}` with `α = 0x02` (a primitive element of GF(2⁸)), so a valid
//! codeword vanishes at those points — the basis of detection.

use symthaea_finite_field::binary::{add, mul, pow};

/// Polynomial product over GF(2⁸).
fn poly_mul(a: &[u8], b: &[u8]) -> Vec<u8> {
    let mut r = vec![0u8; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            r[i + j] = add(r[i + j], mul(ai, bj));
        }
    }
    r
}

/// Evaluate a polynomial at `x` (Horner's method) over GF(2⁸).
fn poly_eval(poly: &[u8], x: u8) -> u8 {
    poly.iter().fold(0u8, |y, &c| add(mul(y, x), c))
}

/// The generator polynomial `g(x) = ∏_{i=0}^{nsym−1} (x − αⁱ)`.
pub fn generator_poly(nsym: usize) -> Vec<u8> {
    let mut g = vec![1u8];
    for i in 0..nsym {
        g = poly_mul(&g, &[1, pow(2, i as u32)]);
    }
    g
}

/// Systematically encode `msg` with `nsym` parity symbols: returns
/// `msg ++ parity`, a codeword divisible by the generator.
pub fn encode(msg: &[u8], nsym: usize) -> Vec<u8> {
    let gen_poly = generator_poly(nsym);
    let mut buf = msg.to_vec();
    buf.extend(std::iter::repeat_n(0u8, nsym));
    // Synthetic division of buf by the (monic) generator; the tail is the remainder.
    for i in 0..msg.len() {
        let coef = buf[i];
        if coef != 0 {
            for j in 1..gen_poly.len() {
                buf[i + j] = add(buf[i + j], mul(gen_poly[j], coef));
            }
        }
    }
    let mut out = msg.to_vec();
    out.extend_from_slice(&buf[msg.len()..]);
    out
}

/// The `nsym` syndromes `C(αⁱ)`. All zero iff the word is a valid codeword.
pub fn syndromes(codeword: &[u8], nsym: usize) -> Vec<u8> {
    (0..nsym)
        .map(|i| poly_eval(codeword, pow(2, i as u32)))
        .collect()
}

/// Whether `codeword` is a valid Reed-Solomon codeword (no detected errors).
pub fn is_valid(codeword: &[u8], nsym: usize) -> bool {
    syndromes(codeword, nsym).iter().all(|&s| s == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoded_word_is_valid() {
        let msg = [0x12, 0x34, 0x56];
        let nsym = 4;
        let cw = encode(&msg, nsym);
        assert_eq!(cw.len(), msg.len() + nsym);
        assert_eq!(&cw[..3], &msg); // systematic: message preserved
        assert!(is_valid(&cw, nsym)); // all syndromes zero
    }

    #[test]
    fn detects_corruption() {
        let msg = [0xAA, 0xBB, 0xCC, 0xDD];
        let nsym = 4;
        let mut cw = encode(&msg, nsym);
        assert!(is_valid(&cw, nsym));
        // A single flipped symbol is detected.
        cw[2] ^= 0x01;
        assert!(!is_valid(&cw, nsym));
    }

    #[test]
    fn generator_has_correct_degree() {
        assert_eq!(generator_poly(4).len(), 5); // degree nsym
        assert_eq!(generator_poly(1), vec![1, 1]); // x - α⁰ = x + 1
    }
}
