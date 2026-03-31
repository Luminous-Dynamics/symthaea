// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Transition timeline — fossil deposits fade, renewables grow.
//!
//! Year 0 = present day. Year 300 = deep future.
//! Epochs match the Leptos version's timeline slider.

use crate::types::FossilDeposit;

// ─── Epochs ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Epoch {
    Foundation,  // 0-25: building the base
    Growth,      // 26-50: rapid expansion
    Expansion,   // 51-100: global reach
    Maturation,  // 101-150: systems mature
    Hardening,   // 151-300: resilient civilization
    DeepTime,    // 301+: long-term stewardship
}

impl Epoch {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Foundation => "Foundation",
            Self::Growth => "Growth",
            Self::Expansion => "Expansion",
            Self::Maturation => "Maturation",
            Self::Hardening => "Hardening",
            Self::DeepTime => "Deep Time",
        }
    }
}

pub fn epoch_for_year(year: u32) -> Epoch {
    match year {
        0..=25 => Epoch::Foundation,
        26..=50 => Epoch::Growth,
        51..=100 => Epoch::Expansion,
        101..=150 => Epoch::Maturation,
        151..=300 => Epoch::Hardening,
        _ => Epoch::DeepTime,
    }
}

// ─── Visibility functions ───────────────────────────────────────

/// Fossil deposit opacity (0.0-1.0) based on timeline year and deposit status.
/// Producing deposits fade gradually; depleted ones fade faster.
pub fn fossil_opacity(deposit: &FossilDeposit, year: u32) -> f32 {
    let base = match deposit.status.as_str() {
        "producing" => 1.0 - (year as f32 / 300.0).min(1.0),
        "declining" => 1.0 - (year as f32 / 200.0).min(1.0),
        "depleted" => (1.0 - (year as f32 / 50.0).min(1.0)).max(0.0),
        "undeveloped" => 1.0 - (year as f32 / 150.0).min(1.0),
        _ => 1.0 - (year as f32 / 300.0).min(1.0),
    };
    base.clamp(0.0, 1.0)
}

/// Renewable site opacity — grows from dim to full over the first 100 years.
pub fn renewable_opacity(year: u32) -> f32 {
    let t = (year as f32 / 100.0).min(1.0);
    0.3 + 0.7 * t
}

/// Nuclear site opacity — bridge fuel, peaks mid-timeline then fades slightly.
pub fn nuclear_opacity(year: u32) -> f32 {
    if year <= 100 {
        0.5 + 0.5 * (year as f32 / 100.0)
    } else {
        1.0 - 0.3 * ((year as f32 - 100.0) / 200.0).min(1.0)
    }
}

/// Resontia vault visibility by index — more vaults appear over time.
/// Matches the Leptos implementation tiers.
pub fn vault_visible(index: usize, year: u32) -> bool {
    let max_visible = match year {
        0..=49 => 0,
        50..=74 => 2,
        75..=99 => 4,
        100..=124 => 7,
        125..=149 => 10,
        _ => usize::MAX,
    };
    index < max_visible
}

/// Maglev corridor visibility by index — corridors built over time.
pub fn corridor_visible(index: usize, year: u32) -> bool {
    let max_visible = match year {
        0..=24 => 0,
        25..=49 => 3,
        50..=99 => 6,
        100..=149 => 10,
        150..=199 => 13,
        _ => usize::MAX,
    };
    index < max_visible
}

/// EROI decline over time as easy reserves deplete.
/// Oil declines ~2%/yr, gas ~1%/yr, coal ~0.5%/yr. Floors at 1.0 (thermodynamic minimum).
pub fn fossil_eroi_at_year(base_eroi: f64, fuel_type: &crate::types::FuelType, year: u32) -> f64 {
    let annual_decline = match fuel_type {
        crate::types::FuelType::Oil => 0.02,
        crate::types::FuelType::Gas => 0.01,
        crate::types::FuelType::Coal => 0.005,
        crate::types::FuelType::TarSands => 0.03, // fastest decline — diminishing returns
    };
    let eroi = base_eroi * (1.0_f64 - annual_decline).powf(year as f64);
    eroi.max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FuelType;

    fn deposit(status: &str) -> FossilDeposit {
        FossilDeposit {
            name: "Test".into(),
            lat: 0.0, lon: 0.0,
            fuel_type: FuelType::Oil,
            proven_reserves_mboe: 1000.0,
            annual_production_mboe: 100.0,
            status: status.into(),
            country: "Test".into(),
            discovery_year: 1950,
            extraction_cost_per_boe: None,
            decommission_cost_m: None,
            eroi: None,
        }
    }

    #[test]
    fn test_epoch_boundaries() {
        assert_eq!(epoch_for_year(0), Epoch::Foundation);
        assert_eq!(epoch_for_year(25), Epoch::Foundation);
        assert_eq!(epoch_for_year(26), Epoch::Growth);
        assert_eq!(epoch_for_year(150), Epoch::Maturation);
        assert_eq!(epoch_for_year(301), Epoch::DeepTime);
    }

    #[test]
    fn test_fossil_opacity_year_zero() {
        assert!((fossil_opacity(&deposit("producing"), 0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_fossil_opacity_depleted_fades_fast() {
        let depleted = fossil_opacity(&deposit("depleted"), 50);
        let producing = fossil_opacity(&deposit("producing"), 50);
        assert!(depleted < producing, "depleted should fade faster");
    }

    #[test]
    fn test_fossil_opacity_year_300() {
        assert!((fossil_opacity(&deposit("producing"), 300)).abs() < 0.01);
    }

    #[test]
    fn test_renewable_opacity_grows() {
        let early = renewable_opacity(0);
        let late = renewable_opacity(100);
        assert!(early < late);
        assert!((early - 0.3).abs() < 0.01);
        assert!((late - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_vault_visibility_tiers() {
        assert!(!vault_visible(0, 0));
        assert!(vault_visible(0, 50));
        assert!(!vault_visible(2, 50));
        assert!(vault_visible(11, 150));
    }

    #[test]
    fn test_eroi_decline_over_time() {
        let base = 25.0;
        let at_50 = fossil_eroi_at_year(base, &FuelType::Oil, 50);
        let at_100 = fossil_eroi_at_year(base, &FuelType::Oil, 100);
        assert!(at_50 < base, "EROI should decline");
        assert!(at_100 < at_50, "EROI should continue declining");
        assert!(at_100 >= 1.0, "EROI floors at 1.0");
    }

    #[test]
    fn test_eroi_year_zero_unchanged() {
        assert!((fossil_eroi_at_year(25.0, &FuelType::Oil, 0) - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_eroi_tar_sands_declines_fastest() {
        let oil_100 = fossil_eroi_at_year(10.0, &FuelType::Oil, 100);
        let tar_100 = fossil_eroi_at_year(10.0, &FuelType::TarSands, 100);
        assert!(tar_100 < oil_100, "tar sands should decline faster than oil");
    }

    #[test]
    fn test_corridor_visibility_tiers() {
        assert!(!corridor_visible(0, 0));
        assert!(corridor_visible(0, 25));
        assert!(!corridor_visible(3, 25));
        assert!(corridor_visible(12, 150));
    }
}
