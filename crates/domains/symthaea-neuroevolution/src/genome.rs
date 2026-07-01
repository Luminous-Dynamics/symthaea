// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Neural genome encoding: BinaryHV (2KB) → CfC neuron configuration.
//!
//! Bit-range loci decode to `UnifiedConfig` + `UnifiedNetworkConfig` fields.
//! Pattern inspired by `hdc_genetics.rs` locus encoding.

use serde::{Deserialize, Serialize};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::hdc_ltc_unified::{UnifiedActivation, UnifiedConfig, UnifiedNetworkConfig};

// ═══════════════════════════════════════════════════════════════════════════════
// LOCUS DEFINITIONS (bit ranges within the 16,384-bit BinaryHV)
// ═══════════════════════════════════════════════════════════════════════════════

const TAU_BASE_START: usize = 0;
const TAU_BASE_BITS: usize = 16;

const BACKBONE_TAU_START: usize = 16;
const BACKBONE_TAU_BITS: usize = 16;

const ACTIVATION_START: usize = 32;
const ACTIVATION_BITS: usize = 3;

const LEARNING_RATE_START: usize = 35;
const LEARNING_RATE_BITS: usize = 16;

const MOMENTUM_START: usize = 51;
const MOMENTUM_BITS: usize = 16;

const WEIGHT_DECAY_START: usize = 67;
const WEIGHT_DECAY_BITS: usize = 16;

const GATING_STEEPNESS_START: usize = 83;
const GATING_STEEPNESS_BITS: usize = 16;

const FOURIER_COUNT_START: usize = 99;
const FOURIER_COUNT_BITS: usize = 4;

const FOURIER_FREQS_START: usize = 103;
const FOURIER_FREQ_BITS: usize = 16;
const MAX_FOURIER_FREQS: usize = 8;

const LAYER_COUNT_START: usize = 231;
const LAYER_COUNT_BITS: usize = 3;

const LAYER_SIZES_START: usize = 234;
const LAYER_SIZE_BITS: usize = 16;
const MAX_LAYERS: usize = 5;

const SKIP_CONNECTIONS_BIT: usize = 314;
const LAYER_BINDING_BIT: usize = 315;

const WEIGHT_SEED_START: usize = 316;
const WEIGHT_SEED_BITS: usize = 64;

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// A CfC neural genome encoded as a 16,384-bit BinaryHV (2KB).
///
/// Bit-range loci decode to `UnifiedConfig` + `UnifiedNetworkConfig` fields.
/// Mutation is bit-flipping via `BinaryHV::add_noise`, crossover is uniform byte selection.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NeuralGenome {
    pub hv: BinaryHV,
}

/// Decoded phenotype from a `NeuralGenome`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralPhenotype {
    pub neuron_config: UnifiedConfig,
    pub network_config: UnifiedNetworkConfig,
    pub weight_seed: u64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// BIT EXTRACTION HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

/// Extract `num_bits` starting at `start_bit` from a BinaryHV, returned as u64.
pub(crate) fn extract_bits(hv: &BinaryHV, start_bit: usize, num_bits: usize) -> u64 {
    let mut value: u64 = 0;
    for i in 0..num_bits {
        let bit_index = start_bit + i;
        let byte_index = bit_index / 8;
        let bit_offset = bit_index % 8;
        if hv.0[byte_index] & (1 << bit_offset) != 0 {
            value |= 1 << i;
        }
    }
    value
}

/// Set `num_bits` starting at `start_bit` in a BinaryHV from a u64 value.
pub(crate) fn set_bits(hv: &mut BinaryHV, start_bit: usize, num_bits: usize, value: u64) {
    for i in 0..num_bits {
        let bit_index = start_bit + i;
        let byte_index = bit_index / 8;
        let bit_offset = bit_index % 8;
        if value & (1 << i) != 0 {
            hv.0[byte_index] |= 1 << bit_offset;
        } else {
            hv.0[byte_index] &= !(1 << bit_offset);
        }
    }
}

/// Map a u64 (from `num_bits` bits) to a f32 in [lo, hi] linearly.
pub(crate) fn bits_to_linear(raw: u64, num_bits: usize, lo: f32, hi: f32) -> f32 {
    let max_val = ((1u64 << num_bits) - 1) as f32;
    if max_val == 0.0 {
        return lo;
    }
    lo + (raw as f32 / max_val) * (hi - lo)
}

/// Map a u64 (from `num_bits` bits) to a f32 in [lo, hi] logarithmically.
fn bits_to_log(raw: u64, num_bits: usize, lo: f32, hi: f32) -> f32 {
    let t = bits_to_linear(raw, num_bits, 0.0, 1.0);
    let log_lo = lo.ln();
    let log_hi = hi.ln();
    (log_lo + t * (log_hi - log_lo)).exp()
}

/// Inverse of bits_to_linear: f32 in [lo, hi] → u64 for `num_bits`.
pub(crate) fn linear_to_bits(val: f32, num_bits: usize, lo: f32, hi: f32) -> u64 {
    let max_val = ((1u64 << num_bits) - 1) as f32;
    let t = ((val - lo) / (hi - lo)).clamp(0.0, 1.0);
    (t * max_val).round() as u64
}

/// Inverse of bits_to_log: f32 in [lo, hi] → u64 for `num_bits`.
fn log_to_bits(val: f32, num_bits: usize, lo: f32, hi: f32) -> u64 {
    let log_lo = lo.ln();
    let log_hi = hi.ln();
    let t = ((val.ln() - log_lo) / (log_hi - log_lo)).clamp(0.0, 1.0);
    let max_val = ((1u64 << num_bits) - 1) as f32;
    (t * max_val).round() as u64
}

// ═══════════════════════════════════════════════════════════════════════════════
// IMPL
// ═══════════════════════════════════════════════════════════════════════════════

impl NeuralGenome {
    /// Create a random genome from a deterministic seed.
    pub fn random(seed: u64) -> Self {
        Self {
            hv: BinaryHV::random(seed),
        }
    }

    /// Create a genome from a genesis seed with a domain label.
    pub fn from_genesis(genesis: &GenesisSeed, label: &str) -> Self {
        let mut rng = genesis.domain(&format!("{label}::neural_genome"));
        let mut bytes = [0u8; 2048];
        use rand::RngCore;
        rng.fill_bytes(&mut bytes);
        Self {
            hv: BinaryHV(bytes),
        }
    }

    /// Decode the genome into a phenotype (CfC configuration).
    pub fn decode(&self) -> NeuralPhenotype {
        let tau_base = bits_to_log(
            extract_bits(&self.hv, TAU_BASE_START, TAU_BASE_BITS),
            TAU_BASE_BITS,
            0.01,
            1.0,
        );

        let backbone_tau = bits_to_linear(
            extract_bits(&self.hv, BACKBONE_TAU_START, BACKBONE_TAU_BITS),
            BACKBONE_TAU_BITS,
            0.1,
            2.0,
        );

        let activation_raw = extract_bits(&self.hv, ACTIVATION_START, ACTIVATION_BITS) as usize;
        let activation = match activation_raw % 5 {
            0 => UnifiedActivation::Tanh,
            1 => UnifiedActivation::Sigmoid,
            2 => UnifiedActivation::SiLU,
            3 => UnifiedActivation::Identity,
            _ => UnifiedActivation::BoundedTanh { scale: 1.5 },
        };

        let learning_rate = bits_to_log(
            extract_bits(&self.hv, LEARNING_RATE_START, LEARNING_RATE_BITS),
            LEARNING_RATE_BITS,
            0.0001,
            0.1,
        );

        let momentum = bits_to_linear(
            extract_bits(&self.hv, MOMENTUM_START, MOMENTUM_BITS),
            MOMENTUM_BITS,
            0.8,
            0.999,
        );

        let weight_decay = bits_to_linear(
            extract_bits(&self.hv, WEIGHT_DECAY_START, WEIGHT_DECAY_BITS),
            WEIGHT_DECAY_BITS,
            0.0,
            0.01,
        );

        let gating_steepness = bits_to_linear(
            extract_bits(&self.hv, GATING_STEEPNESS_START, GATING_STEEPNESS_BITS),
            GATING_STEEPNESS_BITS,
            0.1,
            5.0,
        );

        let fourier_count =
            extract_bits(&self.hv, FOURIER_COUNT_START, FOURIER_COUNT_BITS) as usize;
        let fourier_count = fourier_count.min(MAX_FOURIER_FREQS);

        let mut fourier_frequencies = Vec::with_capacity(fourier_count);
        for i in 0..fourier_count {
            let freq = bits_to_linear(
                extract_bits(
                    &self.hv,
                    FOURIER_FREQS_START + i * FOURIER_FREQ_BITS,
                    FOURIER_FREQ_BITS,
                ),
                FOURIER_FREQ_BITS,
                0.1,
                100.0,
            );
            fourier_frequencies.push(freq);
        }

        let layer_count_raw = extract_bits(&self.hv, LAYER_COUNT_START, LAYER_COUNT_BITS) as usize;
        let layer_count = (layer_count_raw % MAX_LAYERS) + 1; // 1..=5

        let mut layer_sizes = Vec::with_capacity(layer_count);
        for i in 0..layer_count {
            let size_raw = extract_bits(
                &self.hv,
                LAYER_SIZES_START + i * LAYER_SIZE_BITS,
                LAYER_SIZE_BITS,
            ) as usize;
            let size = (size_raw % 32) + 1; // 1..=32
            layer_sizes.push(size);
        }

        let skip_connections = extract_bits(&self.hv, SKIP_CONNECTIONS_BIT, 1) != 0;
        let use_layer_binding = extract_bits(&self.hv, LAYER_BINDING_BIT, 1) != 0;

        let weight_seed = extract_bits(&self.hv, WEIGHT_SEED_START, WEIGHT_SEED_BITS);

        let neuron_config = UnifiedConfig {
            tau_base,
            backbone_tau,
            dimension: 16_384,
            activation,
            learning_rate,
            momentum,
            weight_decay,
            gating_steepness,
            interp_bias: 0.0,
            fourier_frequencies,
            fourier_amplitude: 0.1,
        };

        let network_config = UnifiedNetworkConfig {
            layer_sizes,
            neuron_config: neuron_config.clone(),
            use_layer_binding,
            skip_connections,
        };

        NeuralPhenotype {
            neuron_config,
            network_config,
            weight_seed,
        }
    }

    /// Encode a phenotype back into a genome.
    pub fn encode(phenotype: &NeuralPhenotype) -> Self {
        let mut hv = BinaryHV::zero();
        let nc = &phenotype.neuron_config;
        let netc = &phenotype.network_config;

        set_bits(
            &mut hv,
            TAU_BASE_START,
            TAU_BASE_BITS,
            log_to_bits(nc.tau_base.clamp(0.01, 1.0), TAU_BASE_BITS, 0.01, 1.0),
        );

        set_bits(
            &mut hv,
            BACKBONE_TAU_START,
            BACKBONE_TAU_BITS,
            linear_to_bits(nc.backbone_tau.clamp(0.1, 2.0), BACKBONE_TAU_BITS, 0.1, 2.0),
        );

        let activation_idx = match nc.activation {
            UnifiedActivation::Tanh => 0u64,
            UnifiedActivation::Sigmoid => 1,
            UnifiedActivation::SiLU => 2,
            UnifiedActivation::Identity => 3,
            UnifiedActivation::BoundedTanh { .. } => 4,
        };
        set_bits(&mut hv, ACTIVATION_START, ACTIVATION_BITS, activation_idx);

        set_bits(
            &mut hv,
            LEARNING_RATE_START,
            LEARNING_RATE_BITS,
            log_to_bits(
                nc.learning_rate.clamp(0.0001, 0.1),
                LEARNING_RATE_BITS,
                0.0001,
                0.1,
            ),
        );

        set_bits(
            &mut hv,
            MOMENTUM_START,
            MOMENTUM_BITS,
            linear_to_bits(nc.momentum.clamp(0.8, 0.999), MOMENTUM_BITS, 0.8, 0.999),
        );

        set_bits(
            &mut hv,
            WEIGHT_DECAY_START,
            WEIGHT_DECAY_BITS,
            linear_to_bits(
                nc.weight_decay.clamp(0.0, 0.01),
                WEIGHT_DECAY_BITS,
                0.0,
                0.01,
            ),
        );

        set_bits(
            &mut hv,
            GATING_STEEPNESS_START,
            GATING_STEEPNESS_BITS,
            linear_to_bits(
                nc.gating_steepness.clamp(0.1, 5.0),
                GATING_STEEPNESS_BITS,
                0.1,
                5.0,
            ),
        );

        let freq_count = nc.fourier_frequencies.len().min(MAX_FOURIER_FREQS);
        set_bits(
            &mut hv,
            FOURIER_COUNT_START,
            FOURIER_COUNT_BITS,
            freq_count as u64,
        );
        for (i, &freq) in nc
            .fourier_frequencies
            .iter()
            .take(MAX_FOURIER_FREQS)
            .enumerate()
        {
            set_bits(
                &mut hv,
                FOURIER_FREQS_START + i * FOURIER_FREQ_BITS,
                FOURIER_FREQ_BITS,
                linear_to_bits(freq.clamp(0.1, 100.0), FOURIER_FREQ_BITS, 0.1, 100.0),
            );
        }

        let layer_count = netc.layer_sizes.len().clamp(1, MAX_LAYERS);
        set_bits(
            &mut hv,
            LAYER_COUNT_START,
            LAYER_COUNT_BITS,
            (layer_count - 1) as u64,
        );
        for (i, &size) in netc.layer_sizes.iter().take(MAX_LAYERS).enumerate() {
            let clamped = size.clamp(1, 32) - 1;
            set_bits(
                &mut hv,
                LAYER_SIZES_START + i * LAYER_SIZE_BITS,
                LAYER_SIZE_BITS,
                clamped as u64,
            );
        }

        set_bits(
            &mut hv,
            SKIP_CONNECTIONS_BIT,
            1,
            u64::from(netc.skip_connections),
        );
        set_bits(
            &mut hv,
            LAYER_BINDING_BIT,
            1,
            u64::from(netc.use_layer_binding),
        );

        set_bits(
            &mut hv,
            WEIGHT_SEED_START,
            WEIGHT_SEED_BITS,
            phenotype.weight_seed,
        );

        Self { hv }
    }

    /// Mutate the genome by flipping bits at the given rate.
    pub fn mutate(&self, rate: f32, seed: u64) -> Self {
        Self {
            hv: self.hv.add_noise(rate, seed),
        }
    }

    /// Uniform byte crossover with another genome (avoids BinaryHV::bundle tie-breaking).
    pub fn crossover(&self, other: &Self, seed: u64) -> Self {
        let mut rng = ShakeRngAdapter::new(seed);
        let mut child_bytes = [0u8; 2048];
        for i in 0..2048 {
            if rng.next_bit() {
                child_bytes[i] = self.hv.0[i];
            } else {
                child_bytes[i] = other.hv.0[i];
            }
        }
        Self {
            hv: BinaryHV(child_bytes),
        }
    }

    /// Hamming distance (fraction of differing bits, 0.0 = identical, 1.0 = maximally different).
    pub fn distance(&self, other: &Self) -> f32 {
        self.hv.hamming_distance(&other.hv) as f32 / BinaryHV::DIM as f32
    }

    pub fn decode_thresholds(&self) -> crate::threshold_genome::ThresholdPhenotype {
        crate::threshold_genome::decode_thresholds(&self.hv)
    }

    pub fn encode_thresholds(&mut self, pheno: &crate::threshold_genome::ThresholdPhenotype) {
        crate::threshold_genome::encode_thresholds(&mut self.hv, pheno);
    }

    /// Biased mutation for Lamarckian evolution.
    /// `targets`: slice of (start_bit, num_bits, confidence) tuples identifying
    /// threshold loci to mutate with boosted rate (3× base_rate × confidence).
    pub fn mutate_biased(
        &self,
        base_rate: f32,
        seed: u64,
        targets: &[(usize, usize, f32)],
    ) -> Self {
        let mut child = self.hv.add_noise(base_rate, seed);
        let boost_rate = (base_rate * 3.0).min(0.5);
        for &(start, bits, confidence) in targets {
            let mut rng_state = seed ^ (start as u64 * 0x517CC1B727220A95);
            for bit_idx in start..start + bits {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                let r = (rng_state >> 40) as f32 / (1u64 << 24) as f32;
                if r < boost_rate * confidence {
                    let byte_idx = bit_idx / 8;
                    let bit_offset = bit_idx % 8;
                    child.0[byte_idx] ^= 1 << bit_offset;
                }
            }
        }
        Self { hv: child }
    }
}

/// Simple deterministic bit generator using BLAKE3.
struct ShakeRngAdapter {
    bytes: Vec<u8>,
    index: usize,
    bit_offset: usize,
}

impl ShakeRngAdapter {
    fn new(seed: u64) -> Self {
        // Generate enough random bytes for 2048 decisions
        let mut bytes = vec![0u8; 256];
        // Generate 256 bytes via BLAKE3 chaining
        for chunk in 0..8 {
            let h = blake3::Hasher::new()
                .update(&seed.to_le_bytes())
                .update(&(chunk as u64).to_le_bytes())
                .finalize();
            bytes[chunk * 32..(chunk + 1) * 32].copy_from_slice(h.as_bytes());
        }
        Self {
            bytes,
            index: 0,
            bit_offset: 0,
        }
    }

    fn next_bit(&mut self) -> bool {
        if self.index >= self.bytes.len() {
            // Wrap around (deterministic)
            self.index = 0;
        }
        let bit = (self.bytes[self.index] >> self.bit_offset) & 1 != 0;
        self.bit_offset += 1;
        if self.bit_offset >= 8 {
            self.bit_offset = 0;
            self.index += 1;
        }
        bit
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_genome_decodes_valid() {
        let genome = NeuralGenome::random(42);
        let phenotype = genome.decode();
        assert!(phenotype.neuron_config.tau_base >= 0.01);
        assert!(phenotype.neuron_config.tau_base <= 1.0);
        assert!(phenotype.neuron_config.backbone_tau >= 0.1);
        assert!(phenotype.neuron_config.backbone_tau <= 2.0);
        assert!(phenotype.neuron_config.learning_rate >= 0.0001);
        assert!(phenotype.neuron_config.learning_rate <= 0.1);
        assert!(phenotype.neuron_config.momentum >= 0.8);
        assert!(phenotype.neuron_config.momentum <= 0.999);
        assert!(phenotype.neuron_config.weight_decay >= 0.0);
        assert!(phenotype.neuron_config.weight_decay <= 0.01);
        assert!(phenotype.neuron_config.gating_steepness >= 0.1);
        assert!(phenotype.neuron_config.gating_steepness <= 5.0);
        assert!(!phenotype.network_config.layer_sizes.is_empty());
        assert!(phenotype.network_config.layer_sizes.len() <= 5);
        for &size in &phenotype.network_config.layer_sizes {
            assert!(size >= 1 && size <= 32);
        }
    }

    #[test]
    fn test_encode_decode_roundtrip_tau_base() {
        let genome = NeuralGenome::random(123);
        let phenotype = genome.decode();
        let re_encoded = NeuralGenome::encode(&phenotype);
        let re_decoded = re_encoded.decode();
        // Log-scale round-trip: allow 1% tolerance
        let ratio = re_decoded.neuron_config.tau_base / phenotype.neuron_config.tau_base;
        assert!(
            (ratio - 1.0).abs() < 0.02,
            "tau_base roundtrip: {} vs {}",
            phenotype.neuron_config.tau_base,
            re_decoded.neuron_config.tau_base
        );
    }

    #[test]
    fn test_encode_decode_roundtrip_linear_fields() {
        let genome = NeuralGenome::random(456);
        let phenotype = genome.decode();
        let re_decoded = NeuralGenome::encode(&phenotype).decode();
        let tolerance = 0.001;
        assert!(
            (re_decoded.neuron_config.backbone_tau - phenotype.neuron_config.backbone_tau).abs()
                < tolerance
        );
        assert!(
            (re_decoded.neuron_config.momentum - phenotype.neuron_config.momentum).abs()
                < tolerance
        );
    }

    #[test]
    fn test_encode_decode_roundtrip_activation() {
        for seed in 0..20 {
            let genome = NeuralGenome::random(seed * 1000);
            let phenotype = genome.decode();
            let re_decoded = NeuralGenome::encode(&phenotype).decode();
            assert_eq!(
                std::mem::discriminant(&phenotype.neuron_config.activation),
                std::mem::discriminant(&re_decoded.neuron_config.activation)
            );
        }
    }

    #[test]
    fn test_encode_decode_roundtrip_layers() {
        let genome = NeuralGenome::random(789);
        let phenotype = genome.decode();
        let re_decoded = NeuralGenome::encode(&phenotype).decode();
        assert_eq!(
            phenotype.network_config.layer_sizes.len(),
            re_decoded.network_config.layer_sizes.len()
        );
        assert_eq!(
            phenotype.network_config.skip_connections,
            re_decoded.network_config.skip_connections
        );
        assert_eq!(
            phenotype.network_config.use_layer_binding,
            re_decoded.network_config.use_layer_binding
        );
    }

    #[test]
    fn test_encode_decode_roundtrip_weight_seed() {
        let genome = NeuralGenome::random(101);
        let phenotype = genome.decode();
        let re_decoded = NeuralGenome::encode(&phenotype).decode();
        assert_eq!(phenotype.weight_seed, re_decoded.weight_seed);
    }

    #[test]
    fn test_mutation_preserves_validity() {
        let genome = NeuralGenome::random(42);
        for rate_x100 in [1, 2, 5, 10, 25, 50] {
            let rate = rate_x100 as f32 / 100.0;
            let mutated = genome.mutate(rate, 999 + rate_x100);
            let phenotype = mutated.decode();
            assert!(phenotype.neuron_config.tau_base >= 0.01);
            assert!(phenotype.neuron_config.tau_base <= 1.0);
            assert!(!phenotype.network_config.layer_sizes.is_empty());
        }
    }

    #[test]
    fn test_crossover_produces_intermediate() {
        let parent_a = NeuralGenome::random(100);
        let parent_b = NeuralGenome::random(200);
        let child = parent_a.crossover(&parent_b, 300);

        let dist_a = child.distance(&parent_a);
        let dist_b = child.distance(&parent_b);
        let parent_dist = parent_a.distance(&parent_b);

        // Child should be closer to both parents than parents are to each other
        // (on average — with uniform crossover, ~50% bits from each)
        assert!(
            dist_a < parent_dist || dist_b < parent_dist,
            "Child should be intermediate: dist_a={dist_a}, dist_b={dist_b}, parent_dist={parent_dist}"
        );
    }

    #[test]
    fn test_distance_symmetry() {
        let a = NeuralGenome::random(1);
        let b = NeuralGenome::random(2);
        assert!((a.distance(&b) - b.distance(&a)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_distance_identity() {
        let a = NeuralGenome::random(42);
        assert!(a.distance(&a) < f32::EPSILON);
    }

    #[test]
    fn test_distance_range() {
        let a = NeuralGenome::random(1);
        let b = NeuralGenome::random(2);
        let d = a.distance(&b);
        assert!(d >= 0.0 && d <= 1.0);
    }

    #[test]
    fn test_from_genesis_deterministic() {
        let genesis = GenesisSeed::from_phrase("test-neuroevo");
        let g1 = NeuralGenome::from_genesis(&genesis, "org_0");
        let g2 = NeuralGenome::from_genesis(&genesis, "org_0");
        assert_eq!(g1, g2);

        let g3 = NeuralGenome::from_genesis(&genesis, "org_1");
        assert_ne!(g1, g3);
    }
}
