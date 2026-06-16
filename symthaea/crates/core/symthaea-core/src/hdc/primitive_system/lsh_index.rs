// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use super::Primitive;
use crate::hdc::binary_hv::BinaryHV;
use std::collections::HashMap;

/// Locality Sensitive Hashing index for fast approximate similarity search.
///
/// For binary hypervectors like BinaryHV, we use a simple bit-sampling LSH scheme:
/// - Create `num_bands` hash tables
/// - Each table samples `bits_per_band` random bit positions
/// - Vectors with the same sampled bits hash to the same bucket
/// - Query returns all primitives that share a bucket with the query in any table
///
/// This provides O(1) expected time for candidate retrieval instead of O(n) linear scan.
/// The tradeoff is recall vs. speed: more bands = better recall, more memory.
#[derive(Debug, Clone)]
pub struct LshIndex {
    /// Hash tables: band_idx -> bucket_key -> primitive_names
    tables: Vec<HashMap<u64, Vec<String>>>,
    /// Bit indices sampled for each band
    bit_indices: Vec<Vec<usize>>,
    /// Number of bits per band
    bits_per_band: usize,
}

impl LshIndex {
    /// Build an LSH index from a collection of primitives.
    ///
    /// # Parameters
    /// - `primitives`: Map of primitive name to Primitive
    /// - `num_bands`: Number of hash tables (8-16 is typical)
    /// - `bits_per_band`: Bits per hash (32-128 typical for 16K vectors)
    pub fn build(
        primitives: &HashMap<String, Primitive>,
        num_bands: usize,
        bits_per_band: usize,
    ) -> Self {
        // Generate deterministic random bit indices for each band
        let total_bits = 16384; // BinaryHV dimension
        let mut bit_indices = Vec::with_capacity(num_bands);

        for band in 0..num_bands {
            let mut indices = Vec::with_capacity(bits_per_band);
            // Use deterministic seed based on band number
            let mut seed = 0x5f3759dfu64.wrapping_mul(band as u64 + 1);
            for _ in 0..bits_per_band {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let idx = (seed >> 32) as usize % total_bits;
                indices.push(idx);
            }
            bit_indices.push(indices);
        }

        // Build hash tables
        let mut tables: Vec<HashMap<u64, Vec<String>>> = vec![HashMap::new(); num_bands];

        for (name, prim) in primitives {
            for (band_idx, indices) in bit_indices.iter().enumerate() {
                let hash = Self::compute_hash(&prim.encoding, indices);
                tables[band_idx].entry(hash).or_default().push(name.clone());
            }
        }

        Self {
            tables,
            bit_indices,
            bits_per_band,
        }
    }

    /// Compute hash for a vector using the given bit indices.
    fn compute_hash(hv: &BinaryHV, indices: &[usize]) -> u64 {
        let mut hash: u64 = 0;
        for (i, &bit_idx) in indices.iter().enumerate() {
            let byte_idx = bit_idx / 8;
            let bit_in_byte = bit_idx % 8;
            let bit = (hv.0[byte_idx] >> bit_in_byte) & 1;
            if bit == 1 && i < 64 {
                hash |= 1u64 << (i % 64);
            }
        }
        hash
    }

    /// Query the index for candidate primitives similar to the given encoding.
    ///
    /// Returns a set of primitive names that share at least one bucket with the query.
    /// These are candidates for full similarity comparison.
    pub fn query_candidates(&self, encoding: &BinaryHV) -> Vec<String> {
        let mut candidates = std::collections::HashSet::new();

        for (band_idx, indices) in self.bit_indices.iter().enumerate() {
            let hash = Self::compute_hash(encoding, indices);
            if let Some(bucket) = self.tables[band_idx].get(&hash) {
                for name in bucket {
                    candidates.insert(name.clone());
                }
            }
        }

        candidates.into_iter().collect()
    }

    /// Get statistics about the index.
    pub fn stats(&self) -> LshStats {
        let total_entries: usize = self
            .tables
            .iter()
            .map(|t| t.values().map(|v| v.len()).sum::<usize>())
            .sum();
        let total_buckets: usize = self.tables.iter().map(|t| t.len()).sum();
        let avg_bucket_size = if total_buckets > 0 {
            total_entries as f32 / total_buckets as f32
        } else {
            0.0
        };

        LshStats {
            num_bands: self.tables.len(),
            bits_per_band: self.bits_per_band,
            total_buckets,
            total_entries,
            avg_bucket_size,
        }
    }
}

/// Statistics about an LSH index
#[derive(Debug, Clone)]
pub struct LshStats {
    pub num_bands: usize,
    pub bits_per_band: usize,
    pub total_buckets: usize,
    pub total_entries: usize,
    pub avg_bucket_size: f32,
}
