// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cumulative-emissions warming via TCRE and carbon-budget accounting.
//!
//! The Transient Climate Response to cumulative Emissions (TCRE) is the
//! near-linear relationship between cumulative CO₂ emissions and global warming
//! (Matthews et al. 2009; IPCC AR6). This module turns emitted carbon into
//! warming and computes the remaining budget for a temperature target.

/// TCRE central estimate: warming per gigatonne of *carbon* emitted
/// (≈ 1.65 °C per 1000 GtC, IPCC AR6 range 1.0–2.3).
pub const TCRE_C_PER_GTC: f64 = 1.65e-3;

/// Mass ratio CO₂ : C (44.009 / 12.011 ≈ 3.664) — divide GtCO₂ by this to get GtC.
pub const CO2_TO_C_MASS_RATIO: f64 = 44.009 / 12.011;

/// Warming (°C) from cumulative carbon emissions (GtC).
pub fn warming_from_cumulative_carbon(gt_carbon: f64) -> f64 {
    TCRE_C_PER_GTC * gt_carbon
}

/// Warming (°C) from cumulative CO₂ emissions (GtCO₂).
pub fn warming_from_cumulative_co2(gt_co2: f64) -> f64 {
    warming_from_cumulative_carbon(gt_co2 / CO2_TO_C_MASS_RATIO)
}

/// Remaining carbon budget (GtC) to stay under `target_warming` given
/// `warming_so_far`. Clamped to zero once the target is exceeded.
pub fn remaining_carbon_budget(target_warming: f64, warming_so_far: f64) -> f64 {
    let headroom = target_warming - warming_so_far;
    if headroom <= 0.0 {
        0.0
    } else {
        headroom / TCRE_C_PER_GTC
    }
}

/// Remaining CO₂ budget (GtCO₂) to stay under `target_warming`.
pub fn remaining_co2_budget(target_warming: f64, warming_so_far: f64) -> f64 {
    remaining_carbon_budget(target_warming, warming_so_far) * CO2_TO_C_MASS_RATIO
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thousand_gtc_gives_canonical_warming() {
        // 1000 GtC ⇒ ≈ 1.65 °C.
        assert!((warming_from_cumulative_carbon(1000.0) - 1.65).abs() < 1e-9);
    }

    #[test]
    fn co2_and_carbon_pathways_are_consistent() {
        // 1000 GtC ≈ 3664 GtCO₂ ⇒ same warming.
        let via_c = warming_from_cumulative_carbon(1000.0);
        let via_co2 = warming_from_cumulative_co2(1000.0 * CO2_TO_C_MASS_RATIO);
        assert!((via_c - via_co2).abs() < 1e-9);
    }

    #[test]
    fn warming_is_linear_in_emissions() {
        let a = warming_from_cumulative_carbon(500.0);
        let b = warming_from_cumulative_carbon(1000.0);
        assert!((b / a - 2.0).abs() < 1e-9);
    }

    #[test]
    fn remaining_budget_for_1p5_from_1p1() {
        // Headroom 0.4 °C / 1.65e-3 ≈ 242 GtC.
        let budget = remaining_carbon_budget(1.5, 1.1);
        assert!((budget - 242.4).abs() < 1.0, "budget={budget}");
        // Same headroom in CO₂ terms ≈ 888 GtCO₂.
        let co2 = remaining_co2_budget(1.5, 1.1);
        assert!((co2 - 888.0).abs() < 5.0, "co2={co2}");
    }

    #[test]
    fn exceeded_target_gives_zero_budget() {
        assert_eq!(remaining_carbon_budget(1.5, 1.6), 0.0);
        assert_eq!(remaining_co2_budget(1.5, 1.6), 0.0);
    }
}
