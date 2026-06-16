// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Stochastic Noise — Controlled perturbation of hypervectors.
//!
//! Implements stochastic resonance for program space exploration:
//! adding the RIGHT amount of noise to a program's HDC encoding makes
//! the execution oracle MORE discriminative, not less.
//!
//! ## Science
//!
//! Symthaea's stochastic resonance paper validates: ∂Φ/∂noise = +0.341.
//! Noise INCREASES consciousness (integrated information) in HDC systems.
//! Applied to code: noise-perturbed ProgramNode encodings explore a broader
//! region of program space, and the oracle discriminates better at σ ≈ 0.05-0.15.
//!
//! ## Protocol
//!
//! Stochastic bit-flipping with probability σ ∈ [0, 0.5]:
//! - σ = 0: no perturbation (identity)
//! - σ = 0.05-0.15: resonance sweet spot (validated)
//! - σ = 0.5: completely random (destroys signal)

use symthaea_core::hdc::binary_hv::BinaryHV;

/// Perturb a BinaryHV by flipping bits with probability σ.
///
/// Uses a deterministic xorshift64 PRNG seeded from `seed` for reproducibility.
/// Each bit is flipped independently with probability σ.
///
/// # Arguments
/// - `hv`: the hypervector to perturb
/// - `sigma`: flip probability per bit, in [0.0, 0.5]
/// - `seed`: deterministic RNG seed
///
/// # Returns
/// A new BinaryHV with ~σ×16384 bits flipped.
pub fn perturb(hv: &BinaryHV, sigma: f32, seed: u64) -> BinaryHV {
    let sigma = sigma.clamp(0.0, 0.5);
    if sigma < 1e-6 {
        return *hv; // No perturbation
    }

    let mut result = *hv;
    let mut rng_state = seed.wrapping_add(0xDEAD_CAFE_0000_0001);

    // Threshold for flipping: map sigma to a u64 threshold
    // If xorshift output < threshold, flip the bit
    let threshold = (sigma as f64 * u32::MAX as f64) as u32;

    for byte_idx in 0..BinaryHV::BYTES {
        let mut byte = result.0[byte_idx];
        for bit in 0..8u8 {
            rng_state = xorshift64(rng_state);
            if (rng_state as u32) < threshold {
                byte ^= 1 << bit; // Flip this bit
            }
        }
        result.0[byte_idx] = byte;
    }

    result
}

/// Perturb and measure the actual flip rate (for calibration).
///
/// Returns (perturbed_hv, actual_flip_rate).
pub fn perturb_measured(hv: &BinaryHV, sigma: f32, seed: u64) -> (BinaryHV, f32) {
    let result = perturb(hv, sigma, seed);
    let flipped = hamming_distance(hv, &result);
    let rate = flipped as f32 / BinaryHV::DIM as f32;
    (result, rate)
}

/// Adaptive perturbation: adjust sigma based on improvement.
///
/// If the last perturbation improved the score, increase sigma (explore more).
/// If it degraded, decrease sigma (exploit current region).
/// Clamps to [sigma_min, sigma_max].
pub fn adapt_sigma(
    current_sigma: f32,
    improved: bool,
    sigma_min: f32,
    sigma_max: f32,
    adapt_rate: f32,
) -> f32 {
    let new_sigma = if improved {
        current_sigma * (1.0 + adapt_rate) // Explore more
    } else {
        current_sigma * (1.0 - adapt_rate) // Exploit more
    };
    new_sigma.clamp(sigma_min, sigma_max)
}

/// Count bits that differ between two BinaryHVs (Hamming distance).
fn hamming_distance(a: &BinaryHV, b: &BinaryHV) -> usize {
    let mut count = 0usize;
    for i in 0..BinaryHV::BYTES {
        count += (a.0[i] ^ b.0[i]).count_ones() as usize;
    }
    count
}

/// xorshift64 PRNG — fast, deterministic, good distribution.
#[inline(always)]
fn xorshift64(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perturb_zero_sigma() {
        let hv = BinaryHV::random(42);
        let perturbed = perturb(&hv, 0.0, 123);
        assert_eq!(hv.similarity(&perturbed), 1.0, "σ=0 should not change HV");
    }

    #[test]
    fn test_perturb_half_sigma() {
        let hv = BinaryHV::random(42);
        let perturbed = perturb(&hv, 0.5, 123);
        let sim = hv.similarity(&perturbed);
        // At σ=0.5, ~50% bits flip → similarity ≈ 0.5 (random)
        assert!(
            (sim - 0.5).abs() < 0.05,
            "σ=0.5 should produce ~0.5 similarity, got {sim}"
        );
    }

    #[test]
    fn test_perturb_resonance_range() {
        let hv = BinaryHV::random(42);
        // Resonance sweet spot: σ ≈ 0.1
        let perturbed = perturb(&hv, 0.1, 123);
        let sim = hv.similarity(&perturbed);
        // Should be high but not 1.0 (~90% bits preserved → sim ≈ 0.8)
        assert!(sim > 0.7, "σ=0.1 should preserve most structure, got {sim}");
        assert!(sim < 0.95, "σ=0.1 should have some perturbation, got {sim}");
    }

    #[test]
    fn test_perturb_deterministic() {
        let hv = BinaryHV::random(42);
        let p1 = perturb(&hv, 0.1, 999);
        let p2 = perturb(&hv, 0.1, 999);
        assert_eq!(
            p1.similarity(&p2),
            1.0,
            "same seed should produce identical perturbation"
        );
    }

    #[test]
    fn test_perturb_different_seeds() {
        let hv = BinaryHV::random(42);
        let p1 = perturb(&hv, 0.1, 111);
        let p2 = perturb(&hv, 0.1, 222);
        let sim = p1.similarity(&p2);
        // Different seeds at same sigma should produce different perturbations
        // but both still similar to original
        assert!(sim < 0.99, "different seeds should differ, got {sim}");
    }

    #[test]
    fn test_perturb_measured_rate() {
        let hv = BinaryHV::random(42);
        let (_, rate) = perturb_measured(&hv, 0.1, 123);
        // Actual flip rate should be close to sigma
        assert!(
            (rate - 0.1).abs() < 0.02,
            "measured flip rate {rate} should be ≈ 0.1"
        );
    }

    #[test]
    fn test_adapt_sigma_improves() {
        let sigma = adapt_sigma(0.1, true, 0.01, 0.3, 0.1);
        assert!(sigma > 0.1, "improvement should increase sigma");
        assert!(sigma < 0.3, "should stay below max");
    }

    #[test]
    fn test_adapt_sigma_degrades() {
        let sigma = adapt_sigma(0.1, false, 0.01, 0.3, 0.1);
        assert!(sigma < 0.1, "degradation should decrease sigma");
        assert!(sigma > 0.01, "should stay above min");
    }

    #[test]
    fn test_monotonic_sigma_effect() {
        let hv = BinaryHV::random(42);
        let sim_low = hv.similarity(&perturb(&hv, 0.05, 100));
        let sim_mid = hv.similarity(&perturb(&hv, 0.15, 100));
        let sim_high = hv.similarity(&perturb(&hv, 0.4, 100));
        // Higher sigma → lower similarity (more perturbation)
        assert!(
            sim_low > sim_mid && sim_mid > sim_high,
            "similarity should decrease with sigma: {sim_low} > {sim_mid} > {sim_high}"
        );
    }
}
