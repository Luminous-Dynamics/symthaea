// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Six biosphere coherence dimensions, computed from paleontological proxy data.
//!
//! Each dimension is normalized to [0, 1] relative to Phanerozoic maxima.
//! Network integration and energy throughput use metabolic scaling laws
//! (Kleiber's Law, West-Brown-Enquist) rather than pure guesswork.
//!
//! CITATION: Kleiber (1932), West, Brown & Enquist (1997), Bar-On et al. (2018).

use serde::{Deserialize, Serialize};

use super::deep_time_data::DeepTimeData;
use super::mass_extinctions::{extinction_multiplier, MassExtinctionEvent};
use super::temporal_bins::{MaAge, TemporalBin, TimeResolution};
use super::uncertainty;

/// Raw (unnormalized) biosphere dimension values at a single temporal bin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiosphereDimensionsRaw {
    /// Genus-level richness (taphonomically corrected).
    pub biodiversity: f64,
    /// Maximum organism complexity grade [1-10].
    /// 1=prokaryote, 2=eukaryote, 3=multicellular, 4=bilateral,
    /// 5=vertebrate, 6=tetrapod, 7=amniote, 8=mammal, 9=primate, 10=H. sapiens.
    pub complexity_grade: f64,
    /// Background extinction rate (genera per Ma).
    /// Low rate = stable, resilient ecosystem. High rate = stressed.
    /// CITATION: Alroy (2008), Bambach (2006).
    pub background_extinction_rate: f64,
    /// Trophic network connectivity estimate [0, 1].
    /// Derived from Kleiber's Law + O2 levels + guild count.
    pub network_integration: f64,
    /// Energy throughput: metabolic rate × biomass proxy (W).
    /// Computed from atmospheric O2 × estimated total biomass.
    pub energy_throughput: f64,
    /// Information capacity: genome complexity × species richness.
    pub information_capacity: f64,
}

/// Normalized [0, 1] dimension values with uncertainty bounds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiosphereDimensionsNorm {
    pub values: [f64; 6],
    /// 95% CI half-widths for each dimension.
    pub ci_half_widths: [f64; 6],
}

pub const DIMENSION_NAMES: [&str; 6] = [
    "Biodiversity",
    "Complexity",
    "Resilience",
    "Network Integration",
    "Energy Throughput",
    "Information Capacity",
];

/// Observed maxima for normalization (calibrated to present-day or Phanerozoic peak).
pub struct BiosphereMaxima {
    /// Maximum taphonomically-corrected genus count (~4200 at Holocene).
    pub max_biodiversity: f64,
    /// Maximum complexity grade (10.0 for H. sapiens).
    pub max_complexity: f64,
    /// Maximum background extinction rate (per Ma) — used to invert resilience.
    /// End-Permian peak: ~0.35/Ma (Bambach 2006).
    pub max_extinction_rate: f64,
    /// Maximum network integration (1.0 by definition).
    pub max_network_integration: f64,
    /// Maximum energy throughput (present-day biosphere ~130 TW).
    pub max_energy_throughput: f64,
    /// Maximum information capacity.
    pub max_information_capacity: f64,
}

impl Default for BiosphereMaxima {
    fn default() -> Self {
        Self {
            max_biodiversity: 4500.0, // Holocene corrected diversity
            max_complexity: 10.0,
            max_extinction_rate: 0.35, // End-Permian peak (Bambach 2006)
            max_network_integration: 1.0,
            max_energy_throughput: 130.0,  // TW, Bar-On 2018 estimate
            max_information_capacity: 1.0, // Normalized internally
        }
    }
}

/// Look up organism complexity grade at a given age.
/// Based on first appearance datums of major clades.
///
/// CITATION: Knoll & Bambach (2000), Erwin et al. (2011).
fn complexity_grade_at(age_ma: MaAge) -> f64 {
    if age_ma > 3500.0 {
        return 1.0;
    } // Prokaryotes only
    if age_ma > 2100.0 {
        return 1.5;
    } // Cyanobacteria, complex prokaryotes
    if age_ma > 1800.0 {
        return 2.0;
    } // First eukaryotes (Grypania)
    if age_ma > 800.0 {
        return 2.5;
    } // Complex eukaryotes
    if age_ma > 635.0 {
        return 3.0;
    } // First multicellular (Ediacaran)
    if age_ma > 541.0 {
        return 3.5;
    } // Ediacaran biota (frond, disc)
    if age_ma > 520.0 {
        return 4.0;
    } // Bilateral (Cambrian Explosion)
    if age_ma > 480.0 {
        return 5.0;
    } // First vertebrates (Arandaspis)
    if age_ma > 375.0 {
        return 6.0;
    } // First tetrapods (Tiktaalik)
    if age_ma > 312.0 {
        return 7.0;
    } // First amniotes
    if age_ma > 160.0 {
        return 7.5;
    } // Early mammals (Juramaia)
    if age_ma > 66.0 {
        return 8.0;
    } // Diversified mammals
    if age_ma > 7.0 {
        return 9.0;
    } // Great apes, hominids
    10.0 // Homo sapiens
}

/// Compute network integration using Kleiber's Law and O2 levels.
///
/// The maximum trophic chain length is bounded by available metabolic energy,
/// which scales with body mass^(3/4) (Kleiber 1932) and is limited by
/// atmospheric O2 (aerobic metabolism requires >~2% O2 for multicellular life).
///
/// Network density = (O2_fraction / 0.21) × complexity_factor × diversity_factor
///
/// CITATION: West, Brown & Enquist (1997) "A general model for the origin of
///           allometric scaling laws in biology."
fn network_integration_at(
    _age_ma: MaAge,
    o2_fraction: f64,
    complexity: f64,
    diversity_norm: f64,
) -> f64 {
    // O2 factor: aerobic trophic webs require O2 > 0.02
    let o2_factor = if o2_fraction < 0.02 {
        o2_fraction / 0.02 * 0.1 // Anaerobic: minimal trophic complexity
    } else {
        (o2_fraction / 0.21).min(1.0) // Scale to modern levels
    };

    // Complexity factor: more complex organisms = more trophic levels
    let complexity_factor = (complexity / 10.0).powf(0.75); // Kleiber exponent

    // Diversity contributes to web connectivity
    let diversity_factor = diversity_norm.sqrt();

    (o2_factor * complexity_factor * diversity_factor).clamp(0.0, 1.0)
}

/// Estimate biosphere energy throughput (TW) from O₂, standing biomass, and land area.
///
/// Total biosphere metabolic power ≈ GPP ≈ f(O₂, biomass, land_area)
/// Modern GPP ≈ 130 TW (Bar-On 2018).
///
/// **Critical fix**: Energy throughput must scale with **standing biomass**
/// (proxied by PBDB genus diversity), not just O₂. If 96% of genera die
/// at the End-Permian, the standing biomass crashes proportionally, and
/// energy throughput must reflect that — even if O₂ stays at 16%.
///
/// CITATION: Bar-On, Phillips & Milo (2018) "The biomass distribution on Earth."
fn energy_throughput_at(
    age_ma: MaAge,
    o2_fraction: f64,
    complexity: f64,
    standing_biomass_fraction: f64,
) -> f64 {
    // O₂ sets the metabolic ceiling (photosynthetic capacity)
    let photosynthetic_ceiling = o2_fraction / 0.21;

    // Standing biomass determines how much of that ceiling is actually used.
    // This is the critical factor: a mass extinction crashes biomass even if
    // O₂ stays high, so energy throughput must drop with the die-off.
    let biomass_factor = standing_biomass_fraction;

    // Land colonization boost (post-Silurian, ~420 Ma)
    let land_factor = if age_ma < 420.0 {
        1.0 + (420.0 - age_ma) / 420.0 * 0.5
    } else {
        0.5 // Marine only
    };

    // Complexity-driven metabolic efficiency (higher-order organisms metabolize faster)
    let efficiency = 1.0 + (complexity - 1.0) * 0.03;

    (photosynthetic_ceiling * biomass_factor * land_factor * efficiency * 130.0).max(0.001)
}

/// Compute raw dimensions for a temporal bin.
pub fn compute_raw_dimensions(
    bin: &TemporalBin,
    data: &DeepTimeData,
    _extinctions: &[MassExtinctionEvent],
) -> BiosphereDimensionsRaw {
    let age = bin.midpoint_ma;

    // 1. Biodiversity (taphonomically corrected)
    let biodiversity = data.corrected_diversity(age);

    // 2. Complexity grade
    let complexity_grade = complexity_grade_at(age);

    // 3. Background extinction rate — measures ecosystem stability
    // Low rate = resilient, stable biosphere. High rate = stressed, unstable.
    let background_extinction_rate = data.interpolate_extinction_rate(age);

    // 4. Network integration (Kleiber + O2)
    let o2 = data.interpolate_o2(age);
    let max_bio = BiosphereMaxima::default().max_biodiversity;
    let diversity_norm = (biodiversity / max_bio).clamp(0.0, 1.0);
    let network_integration = network_integration_at(age, o2, complexity_grade, diversity_norm);

    // 5. Energy throughput — scaled by standing biomass AND εp fractionation
    // εp (δ¹³C_carb - δ¹³C_org) is a direct proxy for photosynthetic efficiency.
    // Higher εp → more efficient carbon fixation → higher actual productivity.
    // This replaces raw δ¹³C excursions with a thermodynamically grounded proxy.
    // CITATION: Hayes et al. (1999), Freeman & Hayes (1992).
    let epsilon_p = data.interpolate_epsilon_p(age);
    let ep_factor = (epsilon_p / 27.0).clamp(0.5, 1.5); // Normalize to modern εp ≈ 27‰
    let energy_throughput =
        energy_throughput_at(age, o2, complexity_grade, diversity_norm) * ep_factor;

    // 6. Information capacity — weighted by functional group occupancy
    // Raw genus count suffers from functional redundancy: 1000 trilobite species
    // performing the same ecological role ≠ high information capacity.
    // Functional occupancy (Bambach ecospace) measures unique modes of life,
    // not just taxonomic names.
    // CITATION: Bambach et al. (2007), Bush & Bambach (2011).
    let functional_occupancy = data.interpolate_functional_occupancy(age);
    let info_raw = (complexity_grade / 10.0).powi(2) * functional_occupancy;
    let information_capacity = info_raw;

    BiosphereDimensionsRaw {
        biodiversity,
        complexity_grade,
        background_extinction_rate,
        network_integration,
        energy_throughput,
        information_capacity,
    }
}

impl BiosphereDimensionsRaw {
    /// Normalize against observed maxima to [0, 1].
    pub fn normalize(
        &self,
        maxima: &BiosphereMaxima,
        resolution: TimeResolution,
        age_ma: MaAge,
        d13c_excursion: Option<f64>,
    ) -> BiosphereDimensionsNorm {
        let values = [
            // Biodiversity
            (self.biodiversity / maxima.max_biodiversity).clamp(0.0, 1.0),
            // Complexity
            (self.complexity_grade / maxima.max_complexity).clamp(0.0, 1.0),
            // Resilience (inverted: low extinction rate = high resilience)
            // 0.35/Ma (End-Permian) → 0.0, 0.0/Ma (perfectly stable) → 1.0
            (1.0 - (self.background_extinction_rate / maxima.max_extinction_rate).min(1.0))
                .clamp(0.0, 1.0),
            // Network integration (already [0, 1])
            self.network_integration,
            // Energy throughput
            (self.energy_throughput / maxima.max_energy_throughput).clamp(0.0, 1.0),
            // Information capacity (already [0, 1] range)
            self.information_capacity.clamp(0.0, 1.0),
        ];

        let ci_half_widths = [
            uncertainty::ci_half_width(resolution, 0, age_ma, d13c_excursion),
            uncertainty::ci_half_width(resolution, 1, age_ma, d13c_excursion),
            uncertainty::ci_half_width(resolution, 2, age_ma, d13c_excursion),
            uncertainty::ci_half_width(resolution, 3, age_ma, d13c_excursion),
            uncertainty::ci_half_width(resolution, 4, age_ma, d13c_excursion),
            uncertainty::ci_half_width(resolution, 5, age_ma, d13c_excursion),
        ];

        BiosphereDimensionsNorm {
            values,
            ci_half_widths,
        }
    }
}

/// Compute B(t) as the geometric mean of 6 normalized dimensions.
/// Mirrors K(t)'s geometric mean aggregation.
/// Geometric mean enforces non-substitutability: if any dimension → 0, B(t) → 0.
pub fn compute_bt(dims: &BiosphereDimensionsNorm) -> f64 {
    let epsilon = 1e-10;
    let log_sum: f64 = dims.values.iter().map(|v| (v + epsilon).ln()).sum();
    (log_sum / 6.0).exp()
}

/// Compute B(t) confidence interval bounds.
pub fn compute_bt_ci(dims: &BiosphereDimensionsNorm) -> (f64, f64) {
    let bt = compute_bt(dims);

    // Propagate uncertainty through geometric mean via log-space
    // CI of geometric mean ≈ exp(mean_log ± sqrt(sum(ci_log²) / n²))
    let epsilon = 1e-10;
    let ci_log_sq_sum: f64 = dims
        .values
        .iter()
        .zip(dims.ci_half_widths.iter())
        .map(|(v, ci)| {
            let v_safe = v + epsilon;
            let upper = v_safe + ci;
            let log_range = (upper / v_safe).ln();
            log_range * log_range
        })
        .sum();

    let ci_log = (ci_log_sq_sum / 36.0).sqrt(); // /n² where n=6
    let ci_lower = (bt.ln() - ci_log).exp().max(0.0);
    let ci_upper = (bt.ln() + ci_log).exp().min(1.0);

    (ci_lower, ci_upper)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::biosphere::temporal_bins::build_deep_time_bins;

    #[test]
    fn complexity_monotonically_increases() {
        // Ages from oldest (3800) to youngest (0) — complexity should increase
        let ages: Vec<f64> = (0..=38).map(|i| i as f64 * 100.0).rev().collect();
        let grades: Vec<f64> = ages.iter().map(|&a| complexity_grade_at(a)).collect();
        for pair in grades.windows(2) {
            assert!(
                pair[0] <= pair[1],
                "Complexity should not decrease toward present: grade at {} Ma ({}) > grade at {} Ma ({})",
                ages[grades.iter().position(|g| (*g - pair[0]).abs() < 0.01).unwrap()],
                pair[0],
                ages[grades.iter().position(|g| (*g - pair[1]).abs() < 0.01).unwrap_or(0)],
                pair[1],
            );
        }
    }

    #[test]
    fn normalized_values_in_unit_range() {
        let data = DeepTimeData::load();
        let extinctions = super::super::mass_extinctions::canonical_mass_extinctions();
        let maxima = BiosphereMaxima::default();

        for bin in build_deep_time_bins() {
            let raw = compute_raw_dimensions(&bin, &data, &extinctions);
            let norm = raw.normalize(&maxima, bin.resolution, bin.midpoint_ma, None);
            for (i, v) in norm.values.iter().enumerate() {
                assert!(
                    *v >= 0.0 && *v <= 1.0,
                    "Dimension {} out of range at {} Ma: {}",
                    DIMENSION_NAMES[i],
                    bin.midpoint_ma,
                    v
                );
            }
        }
    }

    #[test]
    fn geometric_mean_less_than_arithmetic() {
        let dims = BiosphereDimensionsNorm {
            values: [0.5, 0.8, 0.3, 0.6, 0.7, 0.4],
            ci_half_widths: [0.1; 6],
        };
        let geo = compute_bt(&dims);
        let arith: f64 = dims.values.iter().sum::<f64>() / 6.0;
        assert!(
            geo <= arith + 1e-10,
            "Geometric mean ({}) must be <= arithmetic mean ({})",
            geo,
            arith
        );
    }

    #[test]
    fn bt_ci_contains_central_value() {
        let dims = BiosphereDimensionsNorm {
            values: [0.5, 0.8, 0.3, 0.6, 0.7, 0.4],
            ci_half_widths: [0.1; 6],
        };
        let bt = compute_bt(&dims);
        let (lo, hi) = compute_bt_ci(&dims);
        assert!(lo <= bt, "CI lower ({}) should be <= B(t) ({})", lo, bt);
        assert!(hi >= bt, "CI upper ({}) should be >= B(t) ({})", hi, bt);
    }

    #[test]
    fn end_permian_bt_is_low() {
        let data = DeepTimeData::load();
        let extinctions = super::super::mass_extinctions::canonical_mass_extinctions();
        let maxima = BiosphereMaxima::default();

        // Find the Lopingian bin (includes 252 Ma End-Permian)
        let bins = build_deep_time_bins();
        let permian_bin = bins
            .iter()
            .find(|b| b.midpoint_ma > 250.0 && b.midpoint_ma < 260.0)
            .unwrap();

        let raw = compute_raw_dimensions(permian_bin, &data, &extinctions);
        let norm = raw.normalize(
            &maxima,
            permian_bin.resolution,
            permian_bin.midpoint_ma,
            None,
        );
        let bt = compute_bt(&norm);
        // Apply extinction multiplier at the nadir (252 Ma)
        let ext_mult = extinction_multiplier(252.0, &extinctions);
        let adjusted = bt * ext_mult;
        // End-Permian should show clear depression relative to pre-Permian
        let pre_permian_bt = {
            let pre_bin = bins
                .iter()
                .find(|b| b.midpoint_ma > 265.0 && b.midpoint_ma < 280.0)
                .unwrap();
            let pre_raw = compute_raw_dimensions(pre_bin, &data, &extinctions);
            let pre_norm =
                pre_raw.normalize(&maxima, pre_bin.resolution, pre_bin.midpoint_ma, None);
            compute_bt(&pre_norm) * extinction_multiplier(pre_bin.midpoint_ma, &extinctions)
        };
        assert!(
            adjusted < pre_permian_bt,
            "End-Permian B(t) ({}) should be less than pre-Permian ({})",
            adjusted,
            pre_permian_bt
        );
    }
}
