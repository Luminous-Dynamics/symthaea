// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hamming distance/weight and the single-error-correcting **Hamming(7,4)**
//! code.

/// The Hamming weight (number of 1-bits) of a byte slice.
pub fn weight(a: &[u8]) -> u32 {
    a.iter().map(|b| b.count_ones()).sum()
}

/// The Hamming distance (number of differing bits) between equal-length byte
/// slices. `None` on a length mismatch.
pub fn distance(a: &[u8], b: &[u8]) -> Option<u32> {
    if a.len() != b.len() {
        return None;
    }
    Some(a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum())
}

/// Encode 4 data bits (each 0 or 1) into a 7-bit Hamming codeword. Layout:
/// positions `[p1, p2, d1, p3, d2, d3, d4]`.
pub fn hamming74_encode(data: [u8; 4]) -> [u8; 7] {
    let [d1, d2, d3, d4] = data.map(|b| b & 1);
    let p1 = d1 ^ d2 ^ d4;
    let p2 = d1 ^ d3 ^ d4;
    let p3 = d2 ^ d3 ^ d4;
    [p1, p2, d1, p3, d2, d3, d4]
}

/// Decode a 7-bit Hamming codeword, correcting any single-bit error, and return
/// the 4 data bits.
pub fn hamming74_decode(codeword: [u8; 7]) -> [u8; 4] {
    let c = codeword.map(|b| b & 1);
    // Syndrome: each parity check covers the positions whose index has that bit.
    let s1 = c[0] ^ c[2] ^ c[4] ^ c[6]; // positions 1,3,5,7
    let s2 = c[1] ^ c[2] ^ c[5] ^ c[6]; // positions 2,3,6,7
    let s3 = c[3] ^ c[4] ^ c[5] ^ c[6]; // positions 4,5,6,7
    let syndrome = s1 + (s2 << 1) + (s3 << 2); // 1-based error position, 0 = none
    let mut c = c;
    if syndrome != 0 {
        let idx = (syndrome - 1) as usize;
        c[idx] ^= 1;
    }
    [c[2], c[4], c[5], c[6]] // d1, d2, d3, d4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_and_distance() {
        assert_eq!(weight(&[0b1011]), 3);
        assert_eq!(distance(&[0b1010], &[0b0011]), Some(2));
        assert_eq!(distance(&[0], &[0, 0]), None);
    }

    #[test]
    fn hamming74_corrects_every_single_error() {
        // For every 4-bit message and every single-bit flip, decoding recovers
        // the original data — the defining property of a 1-error-correcting code.
        for m in 0u8..16 {
            let data = [m & 1, (m >> 1) & 1, (m >> 2) & 1, (m >> 3) & 1];
            let cw = hamming74_encode(data);
            assert_eq!(hamming74_decode(cw), data); // no error
            for flip in 0..7 {
                let mut corrupted = cw;
                corrupted[flip] ^= 1;
                assert_eq!(hamming74_decode(corrupted), data, "m={m}, flip={flip}");
            }
        }
    }
}
