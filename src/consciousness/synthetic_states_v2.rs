//! Synthetic Consciousness States - CORRECTED VERSION
//!
//! **Critical Fix (Dec 26, 2025)**: Redesigned with DIRECT similarity encoding
//!
//! ## The Problem (Version 1)
//!
//! Original generators used bundling of many patterns to represent graph connectivity.
//! This caused "bundle dilution": bundle([A,B,C,D,E]) has only 0.2 similarity to A.
//! Result: More bundling → LOWER similarity → INVERTED Φ correlation (r = -0.803)!
//!
//! ## The Solution (Version 2)
//!
//! Use **shared pattern ratios** to control similarity directly:
//! - High integration: bundle([shared1, shared2, shared3, unique]) = 3/4 shared → HIGH similarity
//! - Low integration: bundle([shared1, unique1, unique2, unique3]) = 1/4 shared → LOW similarity
//!
//! ## Theory
//!
//! For k components bundled, where m are shared patterns:
//! - Expected similarity ≈ m/k (proportion of shared patterns)
//! - High integration needs m/k ≈ 0.75 (mostly shared)
//! - Low integration needs m/k ≈ 0.25 (mostly unique)
//!
//! This creates the CORRECT relationship:
//! - More integration → More shared patterns → Higher similarity → Higher Φ ✅

use crate::hdc::binary_hv::HV16;

/// Generator for synthetic consciousness states with CORRECT similarity encoding
pub struct SyntheticStateGenerator {
    seed: u64,
    num_components: usize,
}

impl SyntheticStateGenerator {
    pub fn new(num_components: usize, seed: u64) -> Self {
        Self {
            seed,
            num_components,
        }
    }

    /// Generate next random seed using simple LCG
    fn next_seed(&mut self) -> u64 {
        // Linear Congruential Generator
        self.seed = self.seed.wrapping_mul(1664525).wrapping_add(1013904223);
        self.seed
    }

    /// DEEP ANESTHESIA (Φ: 0.00-0.05): Pure Random Independence
    ///
    /// No shared patterns at all - completely independent components
    /// Expected similarity: ~0.5 (HDV random baseline)
    pub fn generate_random_state(&mut self) -> Vec<HV16> {
        (0..self.num_components)
            .map(|_| HV16::random(self.next_seed()))
            .collect()
    }

    /// LIGHT ANESTHESIA (Φ: 0.05-0.15): Isolated Pairs
    ///
    /// Components grouped in pairs, each pair shares 1 pattern (50% shared within pair)
    /// Cross-pair: no shared patterns (only random overlap ~0.5)
    /// Expected within-pair similarity: ~0.5, cross-pair: ~0.5 → minimal Φ
    pub fn generate_fragmented_state(&mut self) -> Vec<HV16> {
        let mut components = Vec::with_capacity(self.num_components);

        for i in 0..self.num_components {
            if i % 2 == 0 {
                // First of pair - create pair pattern
                let pair_shared = HV16::random(self.next_seed());
                let unique = HV16::random(self.next_seed());
                components.push(HV16::bundle(&[pair_shared, unique]));
            } else {
                // Second of pair - share pattern with previous
                let pair_shared = if i > 0 {
                    // Extract shared pattern from previous (approximation: use XOR property)
                    components[i - 1].clone()
                } else {
                    HV16::random(self.next_seed())
                };
                let unique = HV16::random(self.next_seed());
                components.push(HV16::bundle(&[pair_shared, unique]));
            }
        }

        components
    }

    /// DEEP SLEEP (Φ: 0.15-0.25): Small Disconnected Clusters
    ///
    /// Clusters of 3-4 components, each cluster shares 1 pattern (33% shared)
    /// Expected within-cluster similarity: ~0.33, cross-cluster: ~0.5
    pub fn generate_isolated_state(&mut self) -> Vec<HV16> {
        let mut components = Vec::with_capacity(self.num_components);
        let cluster_size = 3;

        let mut i = 0;
        while i < self.num_components {
            let cluster_shared = HV16::random(self.next_seed());

            for _ in 0..cluster_size.min(self.num_components - i) {
                let unique1 = HV16::random(self.next_seed());
                let unique2 = HV16::random(self.next_seed());
                // 1 shared + 2 unique = 33% shared
                components.push(HV16::bundle(&[cluster_shared.clone(), unique1, unique2]));
                i += 1;
            }
        }

        components
    }

    /// LIGHT SLEEP (Φ: 0.25-0.35): Modular with Bridges
    ///
    /// 3 modules, each module shares 1 pattern (40% shared within module)
    /// Bridge components share patterns from 2 modules
    /// Expected within-module similarity: ~0.40, cross-module (via bridges): ~0.25
    pub fn generate_low_integration(&mut self) -> Vec<HV16> {
        let module_a = HV16::random(self.next_seed());
        let module_b = HV16::random(self.next_seed());
        let module_c = HV16::random(self.next_seed());
        let mut components = Vec::with_capacity(self.num_components);

        let module_size = self.num_components / 3;

        for i in 0..self.num_components {
            let unique1 = HV16::random(self.next_seed());
            let unique2 = HV16::random(self.next_seed());

            if i < module_size {
                // Module A: 2 shared (module_a + global) + 2 unique = 50% shared
                components.push(HV16::bundle(&[module_a.clone(), unique1, unique2]));
            } else if i < module_size * 2 {
                // Module B
                components.push(HV16::bundle(&[module_b.clone(), unique1, unique2]));
            } else if i == module_size * 2 {
                // Bridge component (shares A and B)
                components.push(HV16::bundle(&[module_a.clone(), module_b.clone(), unique1]));
            } else {
                // Module C
                components.push(HV16::bundle(&[module_c.clone(), unique1, unique2]));
            }
        }

        components
    }

    /// DROWSY (Φ: 0.35-0.45): Ring Topology
    ///
    /// All components share 1 global ring pattern (50% shared)
    /// Expected similarity: ~0.50 (all pairs share ring pattern)
    pub fn generate_moderate_low_integration(&mut self) -> Vec<HV16> {
        let ring_shared = HV16::random(self.next_seed());

        (0..self.num_components)
            .map(|_| {
                let unique = HV16::random(self.next_seed());
                // 1 shared + 1 unique = 50% shared
                HV16::bundle(&[ring_shared.clone(), unique])
            })
            .collect()
    }

    /// RESTING AWAKE (Φ: 0.45-0.55): Small-World Network
    ///
    /// All components share 2 global patterns (67% shared)
    /// Expected similarity: ~0.67
    pub fn generate_moderate_integration(&mut self) -> Vec<HV16> {
        let global1 = HV16::random(self.next_seed());
        let global2 = HV16::random(self.next_seed());

        (0..self.num_components)
            .map(|_| {
                let unique = HV16::random(self.next_seed());
                // 2 shared + 1 unique = 67% shared
                HV16::bundle(&[global1.clone(), global2.clone(), unique])
            })
            .collect()
    }

    /// AWAKE (Φ: 0.55-0.65): Dense Network
    ///
    /// All components share 3 global patterns (75% shared)
    /// Expected similarity: ~0.75
    pub fn generate_moderate_high_integration(&mut self) -> Vec<HV16> {
        let global1 = HV16::random(self.next_seed());
        let global2 = HV16::random(self.next_seed());
        let global3 = HV16::random(self.next_seed());

        (0..self.num_components)
            .map(|_| {
                let unique = HV16::random(self.next_seed());
                // 3 shared + 1 unique = 75% shared
                HV16::bundle(&[global1.clone(), global2.clone(), global3.clone(), unique])
            })
            .collect()
    }

    /// ALERT FOCUSED (Φ: 0.65-0.85): Star Topology (Maximum Integration)
    ///
    /// All components share 4 global patterns (80% shared)
    /// Expected similarity: ~0.80
    pub fn generate_high_integration(&mut self) -> Vec<HV16> {
        let global1 = HV16::random(self.next_seed());
        let global2 = HV16::random(self.next_seed());
        let global3 = HV16::random(self.next_seed());
        let global4 = HV16::random(self.next_seed());

        (0..self.num_components)
            .map(|_| {
                let unique = HV16::random(self.next_seed());
                // 4 shared + 1 unique = 80% shared
                HV16::bundle(&[
                    global1.clone(),
                    global2.clone(),
                    global3.clone(),
                    global4.clone(),
                    unique,
                ])
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that generators produce expected similarity patterns
    #[test]
    fn test_similarity_gradation() {
        let mut gen = SyntheticStateGenerator::new(10, 42);

        // Generate all states
        let random = gen.generate_random_state();
        gen.seed = 43;
        let fragmented = gen.generate_fragmented_state();
        gen.seed = 44;
        let isolated = gen.generate_isolated_state();
        gen.seed = 45;
        let low = gen.generate_low_integration();
        gen.seed = 46;
        let mod_low = gen.generate_moderate_low_integration();
        gen.seed = 47;
        let moderate = gen.generate_moderate_integration();
        gen.seed = 48;
        let mod_high = gen.generate_moderate_high_integration();
        gen.seed = 49;
        let high = gen.generate_high_integration();

        // Measure average pairwise similarities
        let sim_random = measure_avg_similarity(&random);
        let sim_fragmented = measure_avg_similarity(&fragmented);
        let sim_isolated = measure_avg_similarity(&isolated);
        let sim_low = measure_avg_similarity(&low);
        let sim_mod_low = measure_avg_similarity(&mod_low);
        let sim_moderate = measure_avg_similarity(&moderate);
        let sim_mod_high = measure_avg_similarity(&mod_high);
        let sim_high = measure_avg_similarity(&high);

        println!("Similarity progression:");
        println!("  Random:       {:.4}", sim_random);
        println!("  Fragmented:   {:.4}", sim_fragmented);
        println!("  Isolated:     {:.4}", sim_isolated);
        println!("  Low:          {:.4}", sim_low);
        println!("  Moderate-Low: {:.4}", sim_mod_low);
        println!("  Moderate:     {:.4}", sim_moderate);
        println!("  Moderate-High:{:.4}", sim_mod_high);
        println!("  High:         {:.4}", sim_high);

        // Verify monotonic increase (allowing small variance)
        assert!(sim_high > sim_moderate, "High should have higher similarity than moderate");
        assert!(sim_moderate > sim_mod_low, "Moderate should have higher similarity than mod-low");
        assert!(sim_mod_low > sim_low, "Mod-low should have higher similarity than low");
    }

    fn measure_avg_similarity(components: &[HV16]) -> f64 {
        let n = components.len();
        if n < 2 {
            return 0.0;
        }

        let mut total = 0.0;
        let mut count = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                total += components[i].similarity(&components[j]) as f64;
                count += 1;
            }
        }

        total / count as f64
    }
}
