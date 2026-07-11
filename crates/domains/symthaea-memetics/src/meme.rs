// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! The meme: a transmissible unit of idea, carried as an HDC hypervector.
//!
//! A [`Meme`] is deliberately *just* an idea plus lineage — no cognition, no
//! network. Its `payload` is a `BinaryHV` so it rides the exact same 16,384-D
//! medium the resonance graph and immune threat-memory already speak (plan
//! constraint: HDC is the medium; no parallel embedding space).

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::binary_hv::BinaryHV;

/// A transmissible unit of idea.
///
/// Replication is lossy: [`Meme::transmit`] copies the payload with a bounded
/// bit-flip mutation, so descendants drift. [`Meme::fidelity`] measures how
/// faithfully one meme copied another (Hamming similarity of payloads).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Meme {
    /// Unique id within a lineage/run.
    pub id: u64,
    /// The idea itself, as a 16,384-D hypervector.
    pub payload: BinaryHV,
    /// Intrinsic replication advantage in `[0, 1]` — how readily this idea is
    /// adopted *independent of* how well it resonates with a given mind (its
    /// "stickiness": simplicity, emotional charge, memorability). Scales the
    /// per-contact adoption probability in `propagation`.
    pub fitness: f32,
    /// Generation number; the seed meme is generation 0, each transmission +1.
    pub generation: u32,
    /// The meme this one was copied from, if any.
    pub parent: Option<u64>,
}

impl Meme {
    /// A fresh seed meme (generation 0, no parent).
    ///
    /// `fitness` is clamped to `[0, 1]`.
    pub fn seed(id: u64, payload: BinaryHV, fitness: f32) -> Self {
        Self {
            id,
            payload,
            fitness: fitness.clamp(0.0, 1.0),
            generation: 0,
            parent: None,
        }
    }

    /// Transmit (replicate) this meme, mutating the payload.
    ///
    /// `mutation ∈ [0, 1]` is the mutation strength — the inverse of transmission
    /// fidelity. `0.0` is a perfect copy; higher values drift the idea. The child
    /// gets a new `id`, `generation + 1`, and `parent = self.id`. Fitness is
    /// inherited unchanged (selection acts elsewhere).
    ///
    /// Note the underlying `BinaryHV::add_noise` gates each candidate flip on an
    /// internal ~50% mask, so it flips **≈ mutation/2** of the bits: fidelity to
    /// the parent ≈ `1 − mutation/2`. Thus `mutation = 1.0` yields ~0.5 fidelity
    /// (unrelated) as the floor — a mutated idea drifts to *unrelated*, never to
    /// *anti-correlated*, which is the sensible semantics for an idea.
    pub fn transmit(&self, child_id: u64, mutation: f32, seed: u64) -> Meme {
        Meme {
            id: child_id,
            payload: self.payload.add_noise(mutation.clamp(0.0, 1.0), seed),
            fitness: self.fitness,
            generation: self.generation + 1,
            parent: Some(self.id),
        }
    }

    /// Transmission fidelity to another meme: Hamming similarity of payloads in
    /// `[0, 1]` (1.0 = identical idea, ~0.5 = unrelated). Typically called on a
    /// (parent, child) pair to measure how faithfully the idea copied.
    pub fn fidelity(&self, other: &Meme) -> f32 {
        self.payload.similarity(&other.payload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_transmission_is_full_fidelity() {
        let seed = Meme::seed(0, BinaryHV::random(1), 0.5);
        let child = seed.transmit(1, 0.0, 123);
        assert_eq!(child.generation, 1);
        assert_eq!(child.parent, Some(0));
        assert!(
            (seed.fidelity(&child) - 1.0).abs() < 1e-6,
            "mutation=0 must copy exactly, fidelity={}",
            seed.fidelity(&child)
        );
    }

    #[test]
    fn mutation_lowers_fidelity_monotonically() {
        let seed = Meme::seed(0, BinaryHV::random(7), 0.5);
        let low = seed.fidelity(&seed.transmit(1, 0.05, 9));
        let mid = seed.fidelity(&seed.transmit(2, 0.20, 9));
        let high = seed.fidelity(&seed.transmit(3, 0.50, 9));
        // More mutation ⇒ less fidelity (same seed ⇒ nested flip sets ⇒ strictly
        // monotone). `add_noise` flips ≈ p/2 of bits, so fidelity ≈ 1 − p/2:
        // p=0.5 ⇒ ~0.75. The floor is ~0.5 (unrelated) at p=1.0, not below.
        assert!(low > mid && mid > high, "{low} > {mid} > {high}");
        assert!(
            (high - 0.75).abs() < 0.05,
            "p=0.5 ⇒ ~0.75 fidelity, got {high}"
        );
    }

    #[test]
    fn fitness_is_clamped_and_inherited() {
        let seed = Meme::seed(0, BinaryHV::random(1), 5.0);
        assert_eq!(seed.fitness, 1.0);
        assert_eq!(seed.transmit(1, 0.1, 1).fitness, 1.0);
    }
}
