// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Glasser Atlas Parcellation — mapping 360 cortical parcels to 12 CorticalRegions.
//!
//! Maps the Human Connectome Project Multi-Modal Parcellation (MMP/Glasser atlas,
//! 180 parcels × 2 hemispheres = 360 total) onto Symthaea's 12 `CorticalRegion`
//! variants. This enables comparison of Symthaea's simulated activations against
//! voxel-level fMRI predictions from brain encoding models like TRIBE v2.
//!
//! The mapping is based on neuroanatomical consensus:
//! - Glasser et al. (2016). *Nature*, 536, 171-178.
//! - Parcels are identified by 1-based indices (1-180 left hemisphere,
//!   181-360 right hemisphere, mirrored mapping).
//!
//! # Usage
//!
//! ```rust
//! use symthaea_core::hdc::glasser_parcellation::GlasserMapping;
//! use symthaea_core::hdc::substrate_independence::CorticalRegion;
//!
//! let mapping = GlasserMapping::default();
//! let region = mapping.parcel_to_region(1); // V1 → Visual
//! assert_eq!(region, Some(CorticalRegion::Visual));
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::substrate_independence::CorticalRegion;

/// Glasser atlas parcel index (1-360).
pub type ParcelIndex = u16;

/// Mapping between 360 Glasser parcels and 12 CorticalRegion variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlasserMapping {
    /// Parcel index → CorticalRegion.
    forward: HashMap<ParcelIndex, CorticalRegion>,
    /// CorticalRegion → list of parcel indices.
    inverse: HashMap<CorticalRegion, Vec<ParcelIndex>>,
}

impl GlasserMapping {
    /// Look up which CorticalRegion a parcel belongs to.
    pub fn parcel_to_region(&self, parcel: ParcelIndex) -> Option<CorticalRegion> {
        self.forward.get(&parcel).copied()
    }

    /// Get all parcels assigned to a region.
    pub fn region_parcels(&self, region: CorticalRegion) -> &[ParcelIndex] {
        self.inverse
            .get(&region)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Total number of mapped parcels.
    pub fn num_parcels(&self) -> usize {
        self.forward.len()
    }

    /// Aggregate voxel-level data (360 values, one per parcel) into 12 region means.
    ///
    /// `voxel_data` must have at least 360 elements (indexed 0-based, mapped as
    /// parcel index - 1). Unmapped parcels are ignored.
    pub fn aggregate_to_regions(&self, voxel_data: &[f32]) -> HashMap<CorticalRegion, f32> {
        let mut sums: HashMap<CorticalRegion, (f64, usize)> = HashMap::new();

        for (&parcel, &region) in &self.forward {
            let idx = (parcel as usize).saturating_sub(1);
            if idx < voxel_data.len() {
                let entry = sums.entry(region).or_insert((0.0, 0));
                entry.0 += voxel_data[idx] as f64;
                entry.1 += 1;
            }
        }

        sums.into_iter()
            .map(|(region, (sum, count))| {
                let mean = if count > 0 { sum / count as f64 } else { 0.0 };
                (region, mean as f32)
            })
            .collect()
    }

    /// Build the default mapping based on neuroanatomical consensus.
    ///
    /// Parcel indices follow the Glasser et al. (2016) numbering:
    /// - 1-180: left hemisphere
    /// - 181-360: right hemisphere (mirror of left)
    fn build_default() -> Self {
        let mut forward = HashMap::new();
        let mut inverse: HashMap<CorticalRegion, Vec<ParcelIndex>> = HashMap::new();

        // Helper: assign parcel range to region (left hemisphere).
        // Right hemisphere mirror is parcel + 180.
        let mut assign = |parcels: &[u16], region: CorticalRegion| {
            for &p in parcels {
                // Left hemisphere
                forward.insert(p, region);
                inverse.entry(region).or_default().push(p);
                // Right hemisphere mirror
                let rh = p + 180;
                forward.insert(rh, region);
                inverse.entry(region).or_default().push(rh);
            }
        };

        // ================================================================
        // Visual cortex (V1-V4, MT+, V3A, V3B, V6, V6A, V7, V8, FFC, PIT, VVC, VMV)
        // Glasser parcels: 1-6 (V1, MST, V6, V2, V3, V4),
        //   13-14 (V8, 4), mapped below
        // ================================================================
        assign(
            &[
                1, 2, 3, 4, 5, 6, 13, 14, 15, 18, 19, 153, 154, 155, 156, 160, 163,
            ],
            CorticalRegion::Visual,
        );

        // ================================================================
        // Auditory cortex (A1, belt, parabelt, RI, STGa)
        // ================================================================
        assign(
            &[24, 104, 105, 124, 125, 126, 173, 174],
            CorticalRegion::Auditory,
        );

        // ================================================================
        // Language areas (Broca: 44, 45; Wernicke: posterior STS, STSdp, STSvp, TPOJ1)
        // ================================================================
        assign(
            &[75, 76, 77, 128, 129, 130, 131, 132, 133, 134, 175, 176],
            CorticalRegion::Language,
        );

        // ================================================================
        // Motor cortex (M1/area 4, premotor/area 6, SMA, FEF)
        // ================================================================
        assign(
            &[8, 55, 56, 57, 58, 59, 60, 61, 62, 78, 79],
            CorticalRegion::Motor,
        );

        // ================================================================
        // Somatosensory cortex (S1: areas 3a, 3b, 1, 2; S2, OP1-4)
        // ================================================================
        assign(
            &[9, 51, 52, 53, 54, 106, 107, 108, 109],
            CorticalRegion::Sensory,
        );

        // ================================================================
        // Prefrontal cortex (dlPFC, vmPFC, OFC, areas 9, 10, 46, 47, 8)
        // ================================================================
        assign(
            &[
                68, 69, 70, 71, 72, 73, 74, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
            ],
            CorticalRegion::Prefrontal,
        );

        // ================================================================
        // Memory (hippocampus, parahippocampal, entorhinal, perirhinal, TF, TH)
        // Note: hippocampus is subcortical; mapped here via adjacent cortical parcels
        // ================================================================
        assign(
            &[20, 21, 22, 23, 135, 136, 137, 164, 165],
            CorticalRegion::Memory,
        );

        // ================================================================
        // Emotional (amygdala-adjacent, insula anterior, ACC areas 24, 25, 32, OFC)
        // ================================================================
        assign(
            &[91, 92, 93, 94, 95, 96, 97, 98, 110, 111, 112],
            CorticalRegion::Emotional,
        );

        // ================================================================
        // Social (TPJ, posterior STS, mPFC areas, pSTS)
        // ================================================================
        assign(&[139, 140, 141, 142, 143, 144, 145], CorticalRegion::Social);

        // ================================================================
        // Executive (dlPFC/ACC overlap, anterior insula, areas 8C, i6-8, s6-8)
        // Some parcels shared with Prefrontal — executive is the "active control" subset
        // ================================================================
        assign(
            &[63, 64, 65, 66, 67, 99, 100, 101],
            CorticalRegion::Executive,
        );

        // ================================================================
        // Creative (default mode network: precuneus, mPFC, angular gyrus, LTC)
        // ================================================================
        assign(
            &[26, 27, 28, 29, 30, 31, 146, 147, 148, 149, 150],
            CorticalRegion::Creative,
        );

        // ================================================================
        // Integration (thalamus-adjacent, claustrum-adjacent, posterior parietal,
        //              precuneus, retrosplenial, POS)
        // ================================================================
        assign(
            &[
                7, 10, 11, 12, 16, 17, 25, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                46, 47, 48, 49, 50, 102, 103, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
                123, 127, 138, 151, 152, 157, 158, 159, 161, 162, 166, 167, 168, 169, 170, 171,
                172, 177, 178, 179, 180,
            ],
            CorticalRegion::Integration,
        );

        Self { forward, inverse }
    }
}

impl Default for GlasserMapping {
    fn default() -> Self {
        Self::build_default()
    }
}

// ============================================================================
// Desikan-Killiany Atlas (68 parcels → 12 CorticalRegions)
// ============================================================================

/// Desikan-Killiany atlas mapping (34 parcels × 2 hemispheres = 68 total).
///
/// A coarser parcellation widely used in FreeSurfer-based analysis.
/// Reference: Desikan et al. (2006). *NeuroImage*, 31(3), 968-980.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesikanKillianyMapping {
    forward: HashMap<ParcelIndex, CorticalRegion>,
    inverse: HashMap<CorticalRegion, Vec<ParcelIndex>>,
}

impl DesikanKillianyMapping {
    pub fn parcel_to_region(&self, parcel: ParcelIndex) -> Option<CorticalRegion> {
        self.forward.get(&parcel).copied()
    }

    pub fn region_parcels(&self, region: CorticalRegion) -> &[ParcelIndex] {
        self.inverse
            .get(&region)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn num_parcels(&self) -> usize {
        self.forward.len()
    }

    pub fn aggregate_to_regions(&self, voxel_data: &[f32]) -> HashMap<CorticalRegion, f32> {
        let mut sums: HashMap<CorticalRegion, (f64, usize)> = HashMap::new();
        for (&parcel, &region) in &self.forward {
            let idx = (parcel as usize).saturating_sub(1);
            if idx < voxel_data.len() {
                let entry = sums.entry(region).or_insert((0.0, 0));
                entry.0 += voxel_data[idx] as f64;
                entry.1 += 1;
            }
        }
        sums.into_iter()
            .map(|(region, (sum, count))| {
                (
                    region,
                    if count > 0 {
                        (sum / count as f64) as f32
                    } else {
                        0.0
                    },
                )
            })
            .collect()
    }

    fn build_default() -> Self {
        let mut forward = HashMap::new();
        let mut inverse: HashMap<CorticalRegion, Vec<ParcelIndex>> = HashMap::new();

        let mut assign = |parcels: &[u16], region: CorticalRegion| {
            for &p in parcels {
                forward.insert(p, region);
                inverse.entry(region).or_default().push(p);
                let rh = p + 34;
                forward.insert(rh, region);
                inverse.entry(region).or_default().push(rh);
            }
        };

        // DK parcels (left hemisphere 1-34, right 35-68)
        // Mapping based on Desikan et al. (2006) anatomical labels
        assign(&[5, 11, 21], CorticalRegion::Visual); // cuneus, lingual, pericalcarine
        assign(&[30, 31], CorticalRegion::Auditory); // superiortemporal, transversetemporal
        assign(&[18, 19, 20], CorticalRegion::Language); // parsorbitalis, parstriangularis, parsopercularis
        assign(&[22, 23], CorticalRegion::Motor); // precentral, paracentral
        assign(&[24, 29], CorticalRegion::Sensory); // postcentral, supramarginal
        assign(&[27, 28, 3, 14, 25], CorticalRegion::Prefrontal); // superiorfrontal, frontalpole, caudalmiddlefrontal, medialorbitofrontal, rostralmiddlefrontal
        assign(&[6, 7, 17], CorticalRegion::Memory); // entorhinal, fusiform, parahippocampal
        assign(&[1, 10, 26], CorticalRegion::Emotional); // caudalanteriorcingulate, isthmuscingulate, rostralanteriorcingulate
        assign(&[2, 16], CorticalRegion::Social); // bankssts (STS), lateralorbitofrontal
        assign(&[8, 12, 15], CorticalRegion::Executive); // inferiorparietal, lateraloccipital, middletemporal
        assign(&[9, 13, 34], CorticalRegion::Creative); // inferiortemporal, lingual(overlap), insula
        assign(&[4, 32, 33], CorticalRegion::Integration); // corpuscallosum, precuneus, posteriorcingulate

        Self { forward, inverse }
    }
}

impl Default for DesikanKillianyMapping {
    fn default() -> Self {
        Self::build_default()
    }
}

// ============================================================================
// Schaefer Atlas (100 parcels → 12 CorticalRegions)
// ============================================================================

/// Schaefer 100-parcel atlas mapping (50 per hemisphere).
///
/// Functionally defined parcels based on resting-state fMRI, grouped into
/// 7 Yeo networks. Finer than DK, coarser than Glasser.
/// Reference: Schaefer et al. (2018). *Cerebral Cortex*, 28(9), 3095-3114.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchaeferMapping {
    forward: HashMap<ParcelIndex, CorticalRegion>,
    inverse: HashMap<CorticalRegion, Vec<ParcelIndex>>,
}

impl SchaeferMapping {
    pub fn parcel_to_region(&self, parcel: ParcelIndex) -> Option<CorticalRegion> {
        self.forward.get(&parcel).copied()
    }

    pub fn region_parcels(&self, region: CorticalRegion) -> &[ParcelIndex] {
        self.inverse
            .get(&region)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn num_parcels(&self) -> usize {
        self.forward.len()
    }

    pub fn aggregate_to_regions(&self, voxel_data: &[f32]) -> HashMap<CorticalRegion, f32> {
        let mut sums: HashMap<CorticalRegion, (f64, usize)> = HashMap::new();
        for (&parcel, &region) in &self.forward {
            let idx = (parcel as usize).saturating_sub(1);
            if idx < voxel_data.len() {
                let entry = sums.entry(region).or_insert((0.0, 0));
                entry.0 += voxel_data[idx] as f64;
                entry.1 += 1;
            }
        }
        sums.into_iter()
            .map(|(region, (sum, count))| {
                (
                    region,
                    if count > 0 {
                        (sum / count as f64) as f32
                    } else {
                        0.0
                    },
                )
            })
            .collect()
    }

    fn build_default() -> Self {
        let mut forward = HashMap::new();
        let mut inverse: HashMap<CorticalRegion, Vec<ParcelIndex>> = HashMap::new();

        let mut assign = |parcels: &[u16], region: CorticalRegion| {
            for &p in parcels {
                forward.insert(p, region);
                inverse.entry(region).or_default().push(p);
                let rh = p + 50;
                forward.insert(rh, region);
                inverse.entry(region).or_default().push(rh);
            }
        };

        // Schaefer 100 parcels mapped via Yeo 7 networks → CorticalRegion
        // Parcels 1-50 left hemisphere, 51-100 right hemisphere
        assign(&[1, 2, 3, 4, 5, 6, 7], CorticalRegion::Visual); // Visual network
        assign(&[8, 9, 10, 11], CorticalRegion::Sensory); // Somatomotor A
        assign(&[12, 13, 14], CorticalRegion::Motor); // Somatomotor B
        assign(&[15, 16, 17, 18, 19], CorticalRegion::Integration); // Dorsal Attention
        assign(&[20, 21, 22, 23, 24], CorticalRegion::Executive); // Ventral Attention / Salience
        assign(&[25, 26, 27, 28, 29, 30, 31], CorticalRegion::Prefrontal); // Frontoparietal / Control
        assign(&[32, 33, 34, 35, 36, 37, 38], CorticalRegion::Creative); // Default Mode A (medial)
        assign(&[39, 40, 41, 42], CorticalRegion::Social); // Default Mode B (lateral temporal)
        assign(&[43, 44], CorticalRegion::Memory); // Limbic / temporal pole
        assign(&[45, 46], CorticalRegion::Emotional); // Limbic / OFC
        assign(&[47, 48], CorticalRegion::Language); // Language (perisylvian)
        assign(&[49, 50], CorticalRegion::Auditory); // Auditory (STG)

        Self { forward, inverse }
    }
}

impl Default for SchaeferMapping {
    fn default() -> Self {
        Self::build_default()
    }
}

/// Atlas type selector for multi-parcellation comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AtlasType {
    /// Glasser MMP: 360 parcels (180 per hemisphere). Gold standard.
    Glasser360,
    /// Desikan-Killiany: 68 parcels (34 per hemisphere). FreeSurfer default.
    DesikanKilliany68,
    /// Schaefer: 100 parcels (50 per hemisphere). Functional networks.
    Schaefer100,
}

impl AtlasType {
    /// All available atlas types.
    pub const ALL: [AtlasType; 3] = [
        AtlasType::Glasser360,
        AtlasType::DesikanKilliany68,
        AtlasType::Schaefer100,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            AtlasType::Glasser360 => "Glasser-360",
            AtlasType::DesikanKilliany68 => "Desikan-Killiany-68",
            AtlasType::Schaefer100 => "Schaefer-100",
        }
    }

    pub fn num_parcels(&self) -> usize {
        match self {
            AtlasType::Glasser360 => 360,
            AtlasType::DesikanKilliany68 => 68,
            AtlasType::Schaefer100 => 100,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_360_parcels_mapped() {
        let mapping = GlasserMapping::default();
        // Every parcel 1-360 should be assigned to some region.
        let mut unmapped = Vec::new();
        for p in 1..=360 {
            if mapping.parcel_to_region(p).is_none() {
                unmapped.push(p);
            }
        }
        assert!(
            unmapped.is_empty(),
            "Unmapped parcels: {:?} ({} total)",
            &unmapped[..unmapped.len().min(20)],
            unmapped.len()
        );
    }

    #[test]
    fn test_all_regions_have_parcels() {
        let mapping = GlasserMapping::default();
        for region in CorticalRegion::ALL {
            let parcels = mapping.region_parcels(region);
            assert!(
                !parcels.is_empty(),
                "Region {:?} has no parcels assigned",
                region
            );
        }
    }

    #[test]
    fn test_hemisphere_symmetry() {
        let mapping = GlasserMapping::default();
        // Each left-hemisphere parcel (1-180) should have a right-hemisphere
        // mirror (181-360) in the same region.
        for p in 1..=180 {
            let left = mapping.parcel_to_region(p);
            let right = mapping.parcel_to_region(p + 180);
            assert_eq!(
                left,
                right,
                "Parcel {p} (left={left:?}) and {} (right={right:?}) differ",
                p + 180
            );
        }
    }

    #[test]
    fn test_v1_is_visual() {
        let mapping = GlasserMapping::default();
        assert_eq!(mapping.parcel_to_region(1), Some(CorticalRegion::Visual));
    }

    #[test]
    fn test_aggregate_uniform() {
        let mapping = GlasserMapping::default();
        // All voxels = 1.0 → all region means = 1.0
        let data = vec![1.0f32; 360];
        let agg = mapping.aggregate_to_regions(&data);
        for region in CorticalRegion::ALL {
            let v = agg.get(&region).copied().unwrap_or(0.0);
            assert!(
                (v - 1.0).abs() < 1e-4,
                "Region {:?} should be 1.0, got {}",
                region,
                v
            );
        }
    }

    #[test]
    fn test_aggregate_zeros() {
        let mapping = GlasserMapping::default();
        let data = vec![0.0f32; 360];
        let agg = mapping.aggregate_to_regions(&data);
        for region in CorticalRegion::ALL {
            let v = agg.get(&region).copied().unwrap_or(0.0);
            assert!(v.abs() < 1e-6, "Region {:?} should be 0, got {}", region, v);
        }
    }

    #[test]
    fn test_no_parcel_double_mapped() {
        let mapping = GlasserMapping::default();
        // Verify forward map has exactly one region per parcel.
        // (HashMap guarantees uniqueness of keys, but let's verify count.)
        assert_eq!(mapping.num_parcels(), 360);
    }

    #[test]
    fn test_inverse_consistency() {
        let mapping = GlasserMapping::default();
        // Every parcel in inverse lists should point back correctly.
        for region in CorticalRegion::ALL {
            for &p in mapping.region_parcels(region) {
                assert_eq!(
                    mapping.parcel_to_region(p),
                    Some(region),
                    "Parcel {p} in {:?} inverse list but forward maps to {:?}",
                    region,
                    mapping.parcel_to_region(p)
                );
            }
        }
    }

    // ── Desikan-Killiany tests ──

    #[test]
    fn test_dk_all_68_parcels_mapped() {
        let mapping = DesikanKillianyMapping::default();
        assert_eq!(mapping.num_parcels(), 68);
        for p in 1..=68 {
            assert!(
                mapping.parcel_to_region(p).is_some(),
                "DK parcel {p} unmapped"
            );
        }
    }

    #[test]
    fn test_dk_all_regions_covered() {
        let mapping = DesikanKillianyMapping::default();
        for region in CorticalRegion::ALL {
            assert!(
                !mapping.region_parcels(region).is_empty(),
                "DK region {:?} has no parcels",
                region
            );
        }
    }

    #[test]
    fn test_dk_hemisphere_symmetry() {
        let mapping = DesikanKillianyMapping::default();
        for p in 1..=34 {
            let left = mapping.parcel_to_region(p);
            let right = mapping.parcel_to_region(p + 34);
            assert_eq!(left, right, "DK parcel {p} left/right mismatch");
        }
    }

    // ── Schaefer tests ──

    #[test]
    fn test_schaefer_all_100_parcels_mapped() {
        let mapping = SchaeferMapping::default();
        assert_eq!(mapping.num_parcels(), 100);
        for p in 1..=100 {
            assert!(
                mapping.parcel_to_region(p).is_some(),
                "Schaefer parcel {p} unmapped"
            );
        }
    }

    #[test]
    fn test_schaefer_all_regions_covered() {
        let mapping = SchaeferMapping::default();
        for region in CorticalRegion::ALL {
            assert!(
                !mapping.region_parcels(region).is_empty(),
                "Schaefer region {:?} has no parcels",
                region
            );
        }
    }

    #[test]
    fn test_schaefer_hemisphere_symmetry() {
        let mapping = SchaeferMapping::default();
        for p in 1..=50 {
            let left = mapping.parcel_to_region(p);
            let right = mapping.parcel_to_region(p + 50);
            assert_eq!(left, right, "Schaefer parcel {p} left/right mismatch");
        }
    }

    // ── AtlasType tests ──

    #[test]
    fn test_atlas_type_all() {
        assert_eq!(AtlasType::ALL.len(), 3);
        assert_eq!(AtlasType::Glasser360.num_parcels(), 360);
        assert_eq!(AtlasType::DesikanKilliany68.num_parcels(), 68);
        assert_eq!(AtlasType::Schaefer100.num_parcels(), 100);
    }
}
