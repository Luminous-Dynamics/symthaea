// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Deterministic Genesis Seed
//!
//! Provides cryptographically deterministic initialization for HDC vectors
//! using SHAKE-256 (Keccak sponge). Same phrase + domain label produces
//! identical vectors on any machine, forever.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea_core::genesis::{GenesisSeed, GenesisCovenant};
//!
//! let genesis = GenesisSeed::from_phrase("We hold these truths...");
//! let hv = genesis.hv("concept:justice", 16384);
//!
//! // Hierarchical sub-universe
//! let covenant = GenesisCovenant::new(&genesis, "mycelix");
//! let sub_hv = covenant.seed().hv("neuron_0::gate_weight", 16384);
//! ```

use crate::hdc::unified_hv::ContinuousHV;
use rand::RngCore;
use sha3::{
    Shake256,
    digest::{ExtendableOutput, Update, XofReader},
};

/// The root singularity: a seed phrase that deterministically generates
/// all downstream vectors via domain-separated SHAKE-256 streams.
#[derive(Debug, Clone)]
pub struct GenesisSeed {
    seed_phrase: String,
    timeline_id: String,
}

impl GenesisSeed {
    /// Create a genesis seed from a constitution phrase.
    ///
    /// The `timeline_id` is derived as the hex-encoded first 16 bytes of
    /// SHAKE-256(phrase).
    pub fn from_phrase(phrase: &str) -> Self {
        let mut hasher = Shake256::default();
        hasher.update(phrase.as_bytes());
        let mut reader = hasher.finalize_xof();
        let mut id_bytes = [0u8; 16];
        reader.read(&mut id_bytes);
        let timeline_id = hex_encode(&id_bytes);
        Self {
            seed_phrase: phrase.to_string(),
            timeline_id,
        }
    }

    /// Create a domain-separated SHAKE-256 RNG stream.
    ///
    /// Absorbs `phrase || "::" || label`, then squeezes infinite bytes.
    pub fn domain(&self, label: &str) -> ShakeRng {
        let mut hasher = Shake256::default();
        hasher.update(self.seed_phrase.as_bytes());
        hasher.update(b"::");
        hasher.update(label.as_bytes());
        ShakeRng {
            reader: hasher.finalize_xof(),
        }
    }

    /// Squeeze a full `ContinuousHV` from domain `label`.
    ///
    /// Each f32 component is produced by reading 4 bytes from the SHAKE-256
    /// stream and mapping uniformly to \[-1, 1\].
    pub fn hv(&self, label: &str, dim: usize) -> ContinuousHV {
        let mut rng = self.domain(label);
        let mut values = Vec::with_capacity(dim);
        for _ in 0..dim {
            let mut buf = [0u8; 4];
            rng.fill_bytes(&mut buf);
            let u = u32::from_le_bytes(buf);
            let normalized = (u as f32 / u32::MAX as f32) * 2.0 - 1.0;
            values.push(normalized);
        }
        ContinuousHV::from_vec(values)
    }

    /// Create a versioned, unambiguous SHAKE-256 stream for a domain label.
    ///
    /// Unlike the legacy domain format, V2 length-prefixes the seed phrase and
    /// label. This prevents delimiter-boundary aliases such as
    /// ("alpha::beta", "gamma") and ("alpha", "beta::gamma") from producing
    /// the same stream. The versioned namespace is intentionally separate from
    /// V1: this method must not change historical model initialization.
    pub fn domain_v2(&self, label: &str) -> ShakeRng {
        let mut hasher = Shake256::default();
        hasher.update(b"LuminousDynamics::GenesisSeed::domain::v2\\0");
        hasher.update(&(self.seed_phrase.len() as u64).to_le_bytes());
        hasher.update(self.seed_phrase.as_bytes());
        hasher.update(&(label.len() as u64).to_le_bytes());
        hasher.update(label.as_bytes());
        ShakeRng {
            reader: hasher.finalize_xof(),
        }
    }

    /// Squeeze a deterministic V2 ContinuousHV using unambiguous domain input.
    ///
    /// Each coordinate uses the high 24 bits of one little-endian SHAKE word,
    /// mapped onto a representable, near-uniform grid in [-1, 1). This avoids
    /// the precision collapse in V1's u32-to-f32 scaling path. The vector's
    /// prefix is invariant to the requested dimension, as in V1.
    ///
    /// V2 is opt-in. It is not intended to reproduce any V1 vector.
    pub fn hv_v2(&self, label: &str, dim: usize) -> ContinuousHV {
        let mut rng = self.domain_v2(label);
        let mut values = Vec::with_capacity(dim);
        for _ in 0..dim {
            let mut buf = [0u8; 4];
            rng.fill_bytes(&mut buf);
            let mantissa = u32::from_le_bytes(buf) >> 8;
            let normalized = (mantissa as f32) * (1.0 / 8_388_608.0) - 1.0;
            values.push(normalized);
        }
        ContinuousHV::from_vec(values)
    }

    /// The timeline identifier (hex string derived from the phrase).
    pub fn timeline_id(&self) -> &str {
        &self.timeline_id
    }

    /// The original seed phrase.
    pub fn phrase(&self) -> &str {
        &self.seed_phrase
    }
}

/// A SHAKE-256 extendable-output stream that implements `rand::RngCore`.
pub struct ShakeRng {
    reader: <Shake256 as ExtendableOutput>::Reader,
}

impl std::fmt::Debug for ShakeRng {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShakeRng").finish_non_exhaustive()
    }
}

impl RngCore for ShakeRng {
    fn next_u32(&mut self) -> u32 {
        let mut buf = [0u8; 4];
        self.reader.read(&mut buf);
        u32::from_le_bytes(buf)
    }

    fn next_u64(&mut self) -> u64 {
        let mut buf = [0u8; 8];
        self.reader.read(&mut buf);
        u64::from_le_bytes(buf)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.reader.read(dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.reader.read(dest);
        Ok(())
    }
}

/// Hierarchical sub-universe derived from a parent `GenesisSeed`.
///
/// The covenant's internal seed phrase is `"{parent_phrase}::covenant::{name}"`,
/// producing a completely separate but deterministic namespace.
#[derive(Debug, Clone)]
pub struct GenesisCovenant {
    inner: GenesisSeed,
}

impl GenesisCovenant {
    /// Derive a child universe from a parent seed.
    pub fn new(parent: &GenesisSeed, name: &str) -> Self {
        let derived_phrase = format!("{}::covenant::{}", parent.phrase(), name);
        Self {
            inner: GenesisSeed::from_phrase(&derived_phrase),
        }
    }

    /// Access the derived genesis seed.
    pub fn seed(&self) -> &GenesisSeed {
        &self.inner
    }
}

/// Minimal hex encoder (avoids pulling in the `hex` crate).
fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reproducibility() {
        let a = GenesisSeed::from_phrase("test phrase").hv("a", 1024);
        let b = GenesisSeed::from_phrase("test phrase").hv("a", 1024);
        assert_eq!(
            a.values, b.values,
            "Same phrase+label must produce identical HVs"
        );
    }

    #[test]
    fn test_domain_separation() {
        let genesis = GenesisSeed::from_phrase("test phrase");
        let a = genesis.hv("a", 1024);
        let b = genesis.hv("b", 1024);
        let sim = a.similarity(&b);
        assert!(
            sim.abs() < 0.05,
            "Different domains should be near-orthogonal, got similarity {}",
            sim
        );
    }

    #[test]
    fn v2_domain_encoding_prevents_legacy_delimiter_aliases() {
        let left = GenesisSeed::from_phrase("alpha::beta");
        let right = GenesisSeed::from_phrase("alpha");

        // Document the compatibility hazard in the historical V1 encoding.
        assert_eq!(
            left.hv("gamma", 128).values,
            right.hv("beta::gamma", 128).values,
            "V1 output must remain unchanged, including its legacy delimiter alias"
        );

        assert_ne!(
            left.hv_v2("gamma", 128).values,
            right.hv_v2("beta::gamma", 128).values,
            "V2 length-prefixing must distinguish the seed/label tuples"
        );
        assert_ne!(
            left.domain_v2("gamma").next_u64(),
            left.domain_v2("delta").next_u64(),
            "different V2 labels should derive independent streams"
        );
    }

    #[test]
    fn v2_hypervector_is_repeatable_and_dimension_prefix_stable() {
        let seed = GenesisSeed::from_phrase("v2 reproducibility fixture");
        let small = seed.hv_v2("broca::token_emb::7", 128);
        let large = seed.hv_v2("broca::token_emb::7", 512);

        assert_eq!(small.values, seed.hv_v2("broca::token_emb::7", 128).values);
        assert_eq!(
            small.values,
            large.values[..128],
            "asking for more dimensions must preserve the existing vector prefix"
        );
        assert!(small.values.iter().all(|value| value.is_finite()));
        assert_ne!(
            small.values,
            seed.hv("broca::token_emb::7", 128).values,
            "V2 is a distinct, explicitly versioned initializer"
        );
    }

    #[test]
    fn v2_hypervector_distribution_is_centered_and_scale_bounded() {
        let seed = GenesisSeed::from_phrase("v2 distribution fixture");
        let vector = seed.hv_v2("distribution::probe", 16_384);
        let values = &vector.values;
        assert!(values.iter().all(|value| (-1.0..1.0).contains(value)));

        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let rms = (values.iter().map(|value| value * value).sum::<f32>()
            / values.len() as f32)
            .sqrt();
        assert!(mean.abs() < 0.02, "expected near-zero mean, got {mean}");
        assert!(
            (0.56..0.60).contains(&rms),
            "uniform [-1, 1) coordinates should have RMS near sqrt(1/3), got {rms}"
        );

        let other = seed.hv_v2("distribution::other", 16_384);
        let similarity = vector.similarity(&other);
        assert!(
            similarity.abs() < 0.05,
            "independent V2 domains should be near-orthogonal, got {similarity}"
        );
    }

    #[test]
    fn test_shake_rng_core() {
        let genesis = GenesisSeed::from_phrase("rng test");

        // Get u64 via next_u64
        let mut rng1 = genesis.domain("x");
        let val = rng1.next_u64();

        // Get same u64 via fill_bytes
        let mut rng2 = genesis.domain("x");
        let mut buf = [0u8; 8];
        rng2.fill_bytes(&mut buf);
        let val2 = u64::from_le_bytes(buf);

        assert_eq!(val, val2, "next_u64 and fill_bytes must agree");
    }

    #[test]
    fn test_covenant_derivation() {
        let parent = GenesisSeed::from_phrase("parent");
        let covenant = GenesisCovenant::new(&parent, "child");

        let parent_hv = parent.hv("x", 1024);
        let child_hv = covenant.seed().hv("x", 1024);

        // Must be deterministic
        let child_hv2 = GenesisCovenant::new(&GenesisSeed::from_phrase("parent"), "child")
            .seed()
            .hv("x", 1024);
        assert_eq!(
            child_hv.values, child_hv2.values,
            "Covenant must be deterministic"
        );

        // Must differ from parent
        let sim = parent_hv.similarity(&child_hv);
        assert!(
            sim.abs() < 0.05,
            "Covenant HV should differ from parent, got similarity {}",
            sim
        );
    }

    #[test]
    fn test_timeline_id_deterministic() {
        let a = GenesisSeed::from_phrase("hello");
        let b = GenesisSeed::from_phrase("hello");
        assert_eq!(a.timeline_id(), b.timeline_id());

        let c = GenesisSeed::from_phrase("world");
        assert_ne!(a.timeline_id(), c.timeline_id());
    }
}
