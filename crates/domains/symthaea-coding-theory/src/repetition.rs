// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! The repetition code — the simplest error-correcting code: send each bit `n`
//! times and decode by majority vote (corrects up to `⌊(n−1)/2⌋` errors per
//! bit).

/// Repeat each input bit `n` times.
pub fn encode(bits: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(bits.len() * n);
    for &b in bits {
        for _ in 0..n {
            out.push(b & 1);
        }
    }
    out
}

/// Majority-decode `n`-repeated bits back to the original bits. Trailing partial
/// groups are ignored.
pub fn decode(received: &[u8], n: usize) -> Vec<u8> {
    if n == 0 {
        return Vec::new();
    }
    received
        .chunks_exact(n)
        .map(|group| {
            let ones: usize = group.iter().map(|&b| (b & 1) as usize).sum();
            u8::from(ones * 2 > n)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_then_decode_is_identity() {
        let data = [1, 0, 1, 1, 0];
        assert_eq!(decode(&encode(&data, 3), 3), data);
    }

    #[test]
    fn corrects_one_error_per_group() {
        let data = [1, 0, 1];
        let mut rx = encode(&data, 3); // [1,1,1, 0,0,0, 1,1,1]
        rx[0] = 0; // one error in group 0
        rx[4] = 1; // one error in group 1
        rx[8] = 0; // one error in group 2
        assert_eq!(decode(&rx, 3), data); // majority still correct
    }
}
