// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hierarchical Bundling Protocol for BinaryHV.
//!
//! Implements per-region accumulation with role-based binding for scalable
//! aggregation across cortical regions. Each region's bundle is bound with
//! a unique role HV, enabling unbinding (recovery) via XOR self-inverse.
//!
//! ## Design
//!
//! - Each `CorticalRegion` gets a deterministic role HV (from seed)
//! - Vectors are accumulated per-region, then majority-vote bundled
//! - Role binding: `role_bound = bundle(region) ⊕ role_hv`
//! - Aggregate: bundle all role-bound vectors
//! - Unbind: `recovered = aggregate ⊕ role_hv` (XOR is self-inverse)
//!
//! ## SNR Considerations
//!
//! For D=16,384 BinaryHV:
//! - Bundling N vectors: SNR ∝ √(D/N)
//! - Reliable for N ≤ ~200 per level
//! - Warning emitted for N > 200

use std::collections::HashMap;

use super::binary_hv::BinaryHV;
use super::substrate_independence::CorticalRegion;

/// Maximum recommended vectors per bundle level before SNR degrades.
pub const MAX_RELIABLE_BUNDLE: usize = 200;

/// Role HVs for each cortical region, generated deterministically from a seed.
#[derive(Debug, Clone)]
pub struct RegionRoleHVs {
    roles: HashMap<CorticalRegion, BinaryHV>,
}

impl RegionRoleHVs {
    /// Create role HVs from a seed. Each region gets a unique deterministic HV.
    pub fn from_seed(seed: u64) -> Self {
        let mut roles = HashMap::new();
        for (i, region) in CorticalRegion::ALL.iter().enumerate() {
            // Each region gets a unique role HV derived from seed + region index
            let role_seed = seed.wrapping_add(i as u64).wrapping_mul(0x9E3779B97F4A7C15);
            roles.insert(*region, BinaryHV::random(role_seed));
        }
        Self { roles }
    }

    /// Get the role HV for a specific region.
    pub fn get(&self, region: &CorticalRegion) -> Option<&BinaryHV> {
        self.roles.get(region)
    }
}

/// A bundled result for a single cortical region.
#[derive(Debug, Clone)]
pub struct RegionBundle {
    /// Which cortical region this bundle represents.
    pub region: CorticalRegion,
    /// The majority-vote bundle of all contributors.
    pub local_bundle: BinaryHV,
    /// Number of vectors that contributed to this bundle.
    pub contributor_count: usize,
    /// The bundle bound with the region's role HV.
    pub role_bound: BinaryHV,
}

/// Hierarchical bundler that accumulates vectors per-region and produces
/// structured aggregates.
#[derive(Debug)]
pub struct HierarchicalBundler {
    /// Role HVs for binding.
    role_hvs: RegionRoleHVs,
    /// Per-region accumulators.
    accumulators: HashMap<CorticalRegion, Vec<BinaryHV>>,
    /// Maximum vectors per region before warning.
    max_per_region: usize,
}

impl HierarchicalBundler {
    /// Create a new bundler with the given seed for role HV generation.
    pub fn new(seed: u64) -> Self {
        Self {
            role_hvs: RegionRoleHVs::from_seed(seed),
            accumulators: HashMap::new(),
            max_per_region: MAX_RELIABLE_BUNDLE,
        }
    }

    /// Create a new bundler with a custom max-per-region limit.
    pub fn with_max_per_region(seed: u64, max: usize) -> Self {
        Self {
            role_hvs: RegionRoleHVs::from_seed(seed),
            accumulators: HashMap::new(),
            max_per_region: max,
        }
    }

    /// Add a vector to a region's accumulator.
    pub fn add(&mut self, region: CorticalRegion, hv: BinaryHV) {
        let acc = self.accumulators.entry(region).or_default();
        if acc.len() >= self.max_per_region {
            eprintln!(
                "[HierarchicalBundler] Warning: region {:?} accumulator ({}) exceeds max ({}) — SNR may degrade",
                region,
                acc.len(),
                self.max_per_region,
            );
        }
        acc.push(hv);
    }

    /// Bundle a single region's accumulated vectors.
    ///
    /// Returns `None` if the region has no accumulated vectors.
    pub fn bundle_region(&self, region: &CorticalRegion) -> Option<RegionBundle> {
        let acc = self.accumulators.get(region)?;
        if acc.is_empty() {
            return None;
        }

        let local_bundle = BinaryHV::bundle(acc);
        let role_hv = self.role_hvs.get(region)?;
        let role_bound = local_bundle.bind(role_hv);

        Some(RegionBundle {
            region: *region,
            local_bundle,
            contributor_count: acc.len(),
            role_bound,
        })
    }

    /// Bundle all regions and aggregate into a single HV.
    ///
    /// The aggregate preserves per-region information via role binding:
    /// each region's bundle is XOR-bound with its unique role HV before
    /// the final majority-vote aggregation.
    pub fn aggregate(&self) -> Option<BinaryHV> {
        let bundles: Vec<RegionBundle> = CorticalRegion::ALL
            .iter()
            .filter_map(|r| self.bundle_region(r))
            .collect();

        if bundles.is_empty() {
            return None;
        }

        let role_bound_refs: Vec<BinaryHV> = bundles.iter().map(|b| b.role_bound).collect();
        Some(BinaryHV::bundle(&role_bound_refs))
    }

    /// Unbind a specific region from an aggregate HV.
    ///
    /// Because XOR is self-inverse, `aggregate ⊕ role_hv` recovers
    /// an approximation of the original region bundle.
    pub fn unbind_region(&self, aggregate: &BinaryHV, region: &CorticalRegion) -> Option<BinaryHV> {
        let role_hv = self.role_hvs.get(region)?;
        Some(aggregate.bind(role_hv)) // XOR is self-inverse
    }

    /// Get all region bundles.
    pub fn all_bundles(&self) -> Vec<RegionBundle> {
        CorticalRegion::ALL
            .iter()
            .filter_map(|r| self.bundle_region(r))
            .collect()
    }

    /// Clear all accumulators.
    pub fn clear(&mut self) {
        self.accumulators.clear();
    }

    /// Number of regions with accumulated vectors.
    pub fn active_region_count(&self) -> usize {
        self.accumulators.values().filter(|v| !v.is_empty()).count()
    }

    /// Total vectors across all regions.
    pub fn total_vectors(&self) -> usize {
        self.accumulators.values().map(|v| v.len()).sum()
    }

    /// Estimate SNR for a given bundle size at D=16,384.
    pub fn estimate_snr(bundle_size: usize) -> f64 {
        if bundle_size == 0 {
            return f64::INFINITY;
        }
        (16_384.0_f64 / bundle_size as f64).sqrt()
    }
}

/// Multi-level hierarchy for progressive bundling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HierarchyLevel {
    /// Individual node level.
    Node,
    /// Local neighborhood (10-50 nodes).
    Neighborhood,
    /// Ward-level aggregation (50-200 nodes).
    Ward,
    /// Guild-level aggregation (200+ nodes).
    Guild,
}

/// Progressive bundler that supports multi-level hierarchical aggregation.
#[derive(Debug)]
pub struct ProgressiveBundler {
    /// Current hierarchy level.
    pub level: HierarchyLevel,
    /// Inner bundler for this level.
    bundler: HierarchicalBundler,
    /// Role HVs for hierarchy levels.
    hierarchy_roles: HashMap<HierarchyLevel, BinaryHV>,
    /// Bundles received from child levels.
    child_bundles: Vec<BinaryHV>,
}

impl ProgressiveBundler {
    /// Create a new progressive bundler at the given level.
    pub fn new(level: HierarchyLevel, seed: u64) -> Self {
        let mut hierarchy_roles = HashMap::new();
        hierarchy_roles.insert(
            HierarchyLevel::Node,
            BinaryHV::random(seed.wrapping_add(1000)),
        );
        hierarchy_roles.insert(
            HierarchyLevel::Neighborhood,
            BinaryHV::random(seed.wrapping_add(2000)),
        );
        hierarchy_roles.insert(
            HierarchyLevel::Ward,
            BinaryHV::random(seed.wrapping_add(3000)),
        );
        hierarchy_roles.insert(
            HierarchyLevel::Guild,
            BinaryHV::random(seed.wrapping_add(4000)),
        );

        Self {
            level,
            bundler: HierarchicalBundler::new(seed),
            hierarchy_roles,
            child_bundles: Vec::new(),
        }
    }

    /// Add a vector from a child level.
    pub fn add_child_bundle(&mut self, bundle: BinaryHV) {
        self.child_bundles.push(bundle);
    }

    /// Add a vector to a specific region.
    pub fn add(&mut self, region: CorticalRegion, hv: BinaryHV) {
        self.bundler.add(region, hv);
    }

    /// Produce the aggregate for this level, including child bundles.
    pub fn aggregate(&self) -> Option<BinaryHV> {
        let mut all_vectors = Vec::new();

        // Include region-aggregated vectors
        if let Some(region_agg) = self.bundler.aggregate() {
            all_vectors.push(region_agg);
        }

        // Include child bundles
        all_vectors.extend_from_slice(&self.child_bundles);

        if all_vectors.is_empty() {
            return None;
        }

        let level_bundle = BinaryHV::bundle(&all_vectors);

        // Bind with hierarchy level role
        if let Some(role) = self.hierarchy_roles.get(&self.level) {
            Some(level_bundle.bind(role))
        } else {
            Some(level_bundle)
        }
    }

    /// Clear all state.
    pub fn clear(&mut self) {
        self.bundler.clear();
        self.child_bundles.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_role_hvs_deterministic() {
        let roles1 = RegionRoleHVs::from_seed(42);
        let roles2 = RegionRoleHVs::from_seed(42);

        for region in &CorticalRegion::ALL {
            let r1 = roles1.get(region).unwrap();
            let r2 = roles2.get(region).unwrap();
            assert_eq!(r1.similarity(r2), 1.0);
        }
    }

    #[test]
    fn region_role_hvs_unique() {
        let roles = RegionRoleHVs::from_seed(42);
        let regions: Vec<&CorticalRegion> = CorticalRegion::ALL.iter().collect();

        for i in 0..regions.len() {
            for j in (i + 1)..regions.len() {
                let r1 = roles.get(regions[i]).unwrap();
                let r2 = roles.get(regions[j]).unwrap();
                // Random HVs should have ~0.5 similarity
                let sim = r1.similarity(r2);
                assert!(
                    (sim - 0.5).abs() < 0.1,
                    "Role HVs should be quasi-orthogonal, got sim={sim}"
                );
            }
        }
    }

    #[test]
    fn bundle_single_region() {
        let mut bundler = HierarchicalBundler::new(42);

        for i in 0..5 {
            bundler.add(CorticalRegion::Prefrontal, BinaryHV::random(i));
        }

        let bundle = bundler.bundle_region(&CorticalRegion::Prefrontal);
        assert!(bundle.is_some());
        let bundle = bundle.unwrap();
        assert_eq!(bundle.contributor_count, 5);
        assert_eq!(bundle.region, CorticalRegion::Prefrontal);
    }

    #[test]
    fn unbind_recovers_region() {
        let mut bundler = HierarchicalBundler::new(42);

        // Add vectors to two regions
        let prefrontal_hv = BinaryHV::random(100);
        bundler.add(CorticalRegion::Prefrontal, prefrontal_hv);

        let motor_hv = BinaryHV::random(200);
        bundler.add(CorticalRegion::Motor, motor_hv);

        // Get region bundles for comparison
        let prefrontal_bundle = bundler.bundle_region(&CorticalRegion::Prefrontal).unwrap();

        // Aggregate
        let aggregate = bundler.aggregate().unwrap();

        // Unbind prefrontal from aggregate
        let recovered = bundler
            .unbind_region(&aggregate, &CorticalRegion::Prefrontal)
            .unwrap();

        // The recovered vector should be similar to the original prefrontal bundle
        // (not identical due to interference from other regions in the aggregate)
        let similarity = recovered.similarity(&prefrontal_bundle.local_bundle);
        assert!(
            similarity > 0.6,
            "Unbinding should recover the region bundle (got sim={similarity})"
        );
    }

    #[test]
    fn aggregate_preserves_multiple_regions() {
        let mut bundler = HierarchicalBundler::new(42);

        // Add vectors to 4 regions
        for i in 0..3 {
            bundler.add(CorticalRegion::Prefrontal, BinaryHV::random(i * 100));
            bundler.add(CorticalRegion::Motor, BinaryHV::random(i * 100 + 1));
            bundler.add(CorticalRegion::Sensory, BinaryHV::random(i * 100 + 2));
            bundler.add(CorticalRegion::Memory, BinaryHV::random(i * 100 + 3));
        }

        let aggregate = bundler.aggregate().unwrap();

        // Should be able to unbind each region
        for region in &[
            CorticalRegion::Prefrontal,
            CorticalRegion::Motor,
            CorticalRegion::Sensory,
            CorticalRegion::Memory,
        ] {
            let recovered = bundler.unbind_region(&aggregate, region).unwrap();
            let original = bundler.bundle_region(region).unwrap();
            let sim = recovered.similarity(&original.local_bundle);
            assert!(
                sim > 0.55,
                "Region {:?} should be recoverable (got sim={sim})",
                region
            );
        }
    }

    #[test]
    fn snr_estimate() {
        assert!(HierarchicalBundler::estimate_snr(1) > 100.0);
        assert!(HierarchicalBundler::estimate_snr(100) > 10.0);
        assert!(HierarchicalBundler::estimate_snr(200) > 8.0);
        assert!(HierarchicalBundler::estimate_snr(0).is_infinite());
    }

    #[test]
    fn bundler_stats() {
        let mut bundler = HierarchicalBundler::new(42);
        assert_eq!(bundler.active_region_count(), 0);
        assert_eq!(bundler.total_vectors(), 0);

        bundler.add(CorticalRegion::Prefrontal, BinaryHV::random(1));
        bundler.add(CorticalRegion::Motor, BinaryHV::random(2));
        bundler.add(CorticalRegion::Prefrontal, BinaryHV::random(3));

        assert_eq!(bundler.active_region_count(), 2);
        assert_eq!(bundler.total_vectors(), 3);

        bundler.clear();
        assert_eq!(bundler.active_region_count(), 0);
    }

    #[test]
    fn progressive_bundler_two_levels() {
        // Node level
        let mut node = ProgressiveBundler::new(HierarchyLevel::Node, 42);
        node.add(CorticalRegion::Prefrontal, BinaryHV::random(1));
        node.add(CorticalRegion::Motor, BinaryHV::random(2));
        let node_agg = node.aggregate().unwrap();

        // Neighborhood level receives node bundles
        let mut neighborhood = ProgressiveBundler::new(HierarchyLevel::Neighborhood, 43);
        neighborhood.add_child_bundle(node_agg);
        neighborhood.add(CorticalRegion::Sensory, BinaryHV::random(3));

        let neighborhood_agg = neighborhood.aggregate();
        assert!(neighborhood_agg.is_some());
    }

    #[test]
    fn empty_bundler_returns_none() {
        let bundler = HierarchicalBundler::new(42);
        assert!(bundler.aggregate().is_none());
        assert!(bundler.bundle_region(&CorticalRegion::Prefrontal).is_none());
    }
}
