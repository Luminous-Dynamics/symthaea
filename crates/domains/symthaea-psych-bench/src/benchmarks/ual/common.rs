// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared low-level helpers for the UAL probe packet: the same xorshift PRNG
//! and softmax-choice pattern used verbatim in `neuromod::reward_learning`
//! and `motor::srtt` (kept in one place here, rather than duplicated per
//! file a third time, since all three UAL probes need it), plus a
//! near-chance-similarity `ContinuousHV` generator used by P2/P4a to rule out
//! representational-overlap confounds at stimulus-generation time.

use symthaea_core::hdc::ContinuousHV;

/// Same xorshift step as `reward_learning.rs`/`srtt.rs`.
pub fn next_seed(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Same softmax-choice pattern as `reward_learning.rs`.
pub fn softmax_choice(values: &[f64], temperature: f64, rng: &mut u64) -> usize {
    let max_v = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = values
        .iter()
        .map(|v| ((v - max_v) / temperature).exp())
        .collect();
    let sum: f64 = exps.iter().sum();
    let probs: Vec<f64> = exps.iter().map(|e| e / sum).collect();

    let r = (next_seed(rng) % 10000) as f64 / 10000.0;
    let mut cum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r < cum {
            return i;
        }
    }
    probs.len() - 1
}

/// Generate a `ContinuousHV` whose similarity to every hypervector in
/// `existing` falls within `|similarity| < threshold` — a preregistered
/// near-chance band. Re-draws (bumping the seed) on failure, up to
/// `max_attempts`, so representational overlap cannot itself explain later
/// transfer in P2/P4a (design doc's "representational similarity" alternative
/// explanation). At these HDC dimensions two independently random vectors
/// are near-orthogonal essentially always, so rejection should be rare in
/// practice — this is a safety net, not the primary mechanism.
pub fn generate_near_chance_hv(
    dim: usize,
    base_seed: u64,
    existing: &[&ContinuousHV],
    threshold: f32,
    max_attempts: u32,
) -> ContinuousHV {
    let mut seed = base_seed;
    for _ in 0..max_attempts.max(1) {
        let candidate = ContinuousHV::random(dim, seed);
        let ok = existing
            .iter()
            .all(|hv| candidate.similarity(hv).abs() < threshold);
        if ok {
            return candidate;
        }
        seed = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    }
    // Fall through: accept the last candidate rather than loop forever.
    // At real HDC dimensions (>=128) this branch should never be exercised;
    // if it is, a test asserting near-chance similarity will fail loudly
    // rather than silently passing on a biased draw.
    ContinuousHV::random(dim, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn near_chance_generator_produces_low_similarity_pairs() {
        let dim = 512;
        let a = ContinuousHV::random(dim, 1);
        let b = generate_near_chance_hv(dim, 2, &[&a], 0.1, 50);
        assert!(
            a.similarity(&b).abs() < 0.1,
            "similarity should be near-chance: {}",
            a.similarity(&b)
        );
    }

    #[test]
    fn softmax_choice_is_deterministic_given_seed() {
        let mut rng_a = 123u64;
        let mut rng_b = 123u64;
        let a = softmax_choice(&[0.5, 0.5], 0.3, &mut rng_a);
        let b = softmax_choice(&[0.5, 0.5], 0.3, &mut rng_b);
        assert_eq!(a, b);
    }
}
