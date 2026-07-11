// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Channel capacity.

use crate::entropy::binary_entropy;

/// The capacity of a binary symmetric channel with crossover probability `p`:
/// `C = 1 − H(p)` bits per use. Maximised (1 bit) at `p = 0` or `1`; zero at
/// `p = 1/2`.
pub fn bsc_capacity(p: f64) -> f64 {
    1.0 - binary_entropy(p)
}

/// The capacity of a binary erasure channel with erasure probability `e`:
/// `C = 1 − e`.
pub fn bec_capacity(e: f64) -> f64 {
    1.0 - e
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bsc_landmarks() {
        assert!((bsc_capacity(0.0) - 1.0).abs() < 1e-12); // noiseless
        assert!(bsc_capacity(0.5).abs() < 1e-12); // useless
        assert!((bsc_capacity(1.0) - 1.0).abs() < 1e-12); // deterministic flip
        // A slightly noisy channel has capacity strictly between 0 and 1.
        let c = bsc_capacity(0.1);
        assert!(c > 0.0 && c < 1.0);
    }

    #[test]
    fn bec_capacity_is_one_minus_erasure() {
        assert!((bec_capacity(0.0) - 1.0).abs() < 1e-12);
        assert!((bec_capacity(0.25) - 0.75).abs() < 1e-12);
    }
}
