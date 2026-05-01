// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Resolution-dependent uncertainty modeling with geochemical bounding.
//!
//! Rather than uniform confidence intervals, we use:
//! 1. **Resolution-dependent base widths** — Precambrian is inherently more uncertain
//! 2. **Dimension-dependent scaling** — biodiversity is better constrained than
//!    network integration
//! 3. **Geochemical HMM bounding** (δ¹³C proxy) — isotopic shifts bound total
//!    biomass even when species identity is unknown, tightening Precambrian CIs
//!
//! CITATION: Kump & Arthur (1999), Hayes et al. (1999) — δ¹³C as biomass proxy.
//! The fractionation between organic carbon (δ¹³C_org ≈ -25‰) and inorganic
//! carbon (δ¹³C_carb ≈ 0‰) is driven by photosynthetic biomass. Large excursions
//! in δ¹³C_carb indicate massive changes in total biosphere productivity.

use super::deep_time_data::TaphonomicRegime;
use super::temporal_bins::TimeResolution;

/// Dimension indices matching BiosphereDimensionsNorm.values[].
pub const DIM_BIODIVERSITY: usize = 0;
pub const DIM_COMPLEXITY: usize = 1;
pub const DIM_RESILIENCE: usize = 2;
pub const DIM_NETWORK: usize = 3;
pub const DIM_ENERGY: usize = 4;
pub const DIM_INFORMATION: usize = 5;

/// Base CI half-width for each (resolution, dimension) pair.
///
/// These are epistemic uncertainty estimates, NOT statistical bootstrap CIs.
/// They honestly reflect how much we know about each dimension at each timescale.
pub fn base_ci_half_width(resolution: TimeResolution, dim_idx: usize) -> f64 {
    let resolution_factor = match resolution {
        TimeResolution::Precambrian => 1.0,
        TimeResolution::PhanerozoicStage => 0.4,
        TimeResolution::Holocene => 0.15,
    };

    // Dimension-specific base uncertainty (at Precambrian resolution).
    // Better-constrained dimensions get lower base values.
    let dim_base = match dim_idx {
        DIM_BIODIVERSITY => 0.25, // Fossil record exists, preservation bias known
        DIM_COMPLEXITY => 0.15,   // First appearances well-dated
        DIM_RESILIENCE => 0.20,   // Recovery times reasonably constrained
        DIM_NETWORK => 0.35,      // Theoretical — trophic webs rarely preserved
        DIM_ENERGY => 0.20,       // O2 proxy is decent via GEOCARB
        DIM_INFORMATION => 0.30,  // Genome reconstruction is speculative
        _ => 0.30,
    };

    dim_base * resolution_factor
}

/// Apply geochemical bounding to tighten CIs using δ¹³C data.
///
/// When large δ¹³C excursions are observed, they bound total biomass changes:
/// - A +5‰ positive excursion means massive organic carbon burial → high productivity
/// - A -5‰ negative excursion means organic carbon release → biomass crash
///
/// This constrains the BIODIVERSITY and ENERGY dimensions even when no fossils exist.
///
/// The correction factor reduces the CI half-width when strong isotopic signal is present.
/// `d13c_excursion`: magnitude of δ¹³C departure from baseline (‰, absolute).
pub fn geochemical_ci_correction(d13c_excursion_abs: f64, dim_idx: usize) -> f64 {
    // Only biodiversity and energy throughput benefit from δ¹³C bounding
    let sensitivity = match dim_idx {
        DIM_BIODIVERSITY => 0.5,
        DIM_ENERGY => 0.6,
        _ => 0.0, // Other dimensions don't benefit from this proxy
    };

    if sensitivity == 0.0 {
        return 1.0; // No correction
    }

    // Strong isotopic signal (>3‰ excursion) tightens CI by up to 40%
    // Weak signal (<1‰) provides negligible constraint
    let signal_strength = (d13c_excursion_abs / 5.0).min(1.0);
    let max_tightening = 0.4 * sensitivity;

    1.0 - max_tightening * signal_strength
}

/// Compute full CI half-width for a dimension at a given time point.
///
/// Combines:
/// 1. Base resolution-dependent width
/// 2. Taphonomic penalty (for biodiversity dimension)
/// 3. Geochemical bounding correction (from δ¹³C)
pub fn ci_half_width(
    resolution: TimeResolution,
    dim_idx: usize,
    age_ma: f64,
    d13c_excursion_abs: Option<f64>,
) -> f64 {
    let mut width = base_ci_half_width(resolution, dim_idx);

    // Taphonomic penalty for biodiversity dimension
    if dim_idx == DIM_BIODIVERSITY {
        width += TaphonomicRegime::for_age(age_ma).ci_penalty();
    }

    // Geochemical bounding (tightens CI when strong isotopic signal available)
    if let Some(excursion) = d13c_excursion_abs {
        width *= geochemical_ci_correction(excursion, dim_idx);
    }

    width.clamp(0.01, 0.50)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precambrian_wider_than_phanerozoic() {
        for dim in 0..6 {
            let pre = base_ci_half_width(TimeResolution::Precambrian, dim);
            let pha = base_ci_half_width(TimeResolution::PhanerozoicStage, dim);
            let hol = base_ci_half_width(TimeResolution::Holocene, dim);
            assert!(
                pre > pha,
                "Precambrian should be wider than Phanerozoic for dim {}",
                dim
            );
            assert!(
                pha > hol,
                "Phanerozoic should be wider than Holocene for dim {}",
                dim
            );
        }
    }

    #[test]
    fn geochemical_bounding_tightens_ci() {
        let no_signal = geochemical_ci_correction(0.0, DIM_BIODIVERSITY);
        let strong_signal = geochemical_ci_correction(5.0, DIM_BIODIVERSITY);
        assert!(
            (no_signal - 1.0).abs() < 0.01,
            "No signal should not tighten"
        );
        assert!(
            strong_signal < 0.85,
            "Strong signal should tighten by >15%, got {}",
            strong_signal
        );
    }

    #[test]
    fn non_bio_dimensions_unaffected_by_d13c() {
        let correction = geochemical_ci_correction(5.0, DIM_COMPLEXITY);
        assert!(
            (correction - 1.0).abs() < 0.01,
            "Complexity should be unaffected by δ¹³C"
        );
    }

    #[test]
    fn full_ci_reasonable_range() {
        let precambrian = ci_half_width(TimeResolution::Precambrian, DIM_NETWORK, 2000.0, None);
        assert!(precambrian > 0.2, "Precambrian network CI should be wide");
        assert!(precambrian < 0.5, "CI should not exceed 0.5");

        let modern = ci_half_width(TimeResolution::Holocene, DIM_BIODIVERSITY, 0.005, None);
        assert!(modern < 0.10, "Holocene biodiversity CI should be narrow");
    }
}
