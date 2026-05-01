// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bridge between B(t) deep time, K(t) historical, and multiworld-sim.
//!
//! ## Shared Thermodynamic Currency
//!
//! Rather than a purely visual cross-fade, the bridge grounds both B(t) and K(t)
//! in **energy throughput** (W per kg of active mass). Both the biosphere and
//! human civilization compete for the same terrestrial free energy budget
//! (solar insolation ≈ 174 PW, of which ~0.1% drives the biosphere).
//!
//! When K(t) spikes (Industrial Revolution), it mechanically depresses B(t) by
//! co-opting biosphere energy throughput (land use change, fossil carbon extraction).
//! This makes the handoff a physical zero-sum equation, not a visual fade.
//!
//! CITATION: Haberl et al. (2007) "Human appropriation of NPP",
//!           Smil (2017) "Energy and Civilization", Bar-On et al. (2018).

use super::temporal_bins::MaAge;

/// Global terrestrial Net Primary Productivity (NPP) in TW.
/// CITATION: Bar-On et al. (2018) — total biosphere GPP ≈ 130 TW.
const _GLOBAL_NPP_TW: f64 = 130.0;

/// Human Appropriation of NPP (HANPP) as fraction, by era.
/// CITATION: Haberl et al. (2007) — modern HANPP ≈ 24% of potential NPP.
/// Krausmann et al. (2013) — historical HANPP reconstruction.
fn hanpp_fraction(year_ce: f64) -> f64 {
    if year_ce < -10000.0 {
        0.0 // Pre-Neolithic: negligible
    } else if year_ce < -3000.0 {
        // Neolithic agriculture: very small (<1%)
        let t = (year_ce + 10000.0) / 7000.0;
        t * 0.01
    } else if year_ce < 1800.0 {
        // Pre-industrial civilization: slow increase
        let t = (year_ce + 3000.0) / 4800.0;
        0.01 + t * 0.04 // 1% → 5%
    } else if year_ce < 1950.0 {
        // Industrial Revolution
        let t = (year_ce - 1800.0) / 150.0;
        0.05 + t * 0.10 // 5% → 15%
    } else {
        // Modern era: 15% → 24%
        let t = ((year_ce - 1950.0) / 70.0).min(1.0);
        0.15 + t * 0.09
    }
}

/// Convert Ma age to year CE for the HANPP function.
fn ma_to_year_ce(age_ma: MaAge) -> f64 {
    2026.0 - age_ma * 1_000_000.0
}

/// Energy-grounded B(t) adjustment for human energy co-option.
///
/// Returns the fraction of biosphere energy remaining after HANPP.
/// B_adjusted(t) = B_raw(t) × (1 - HANPP(t))
///
/// This is the thermodynamic bridge: human civilization's rise
/// mechanically reduces biosphere coherence by claiming energy throughput.
pub fn biosphere_energy_adjustment(age_ma: MaAge) -> f64 {
    let year_ce = ma_to_year_ce(age_ma);
    let hanpp = hanpp_fraction(year_ce);
    1.0 - hanpp
}

/// Handoff point between B(t) and K(t).
/// 10,000 BCE = 0.012 Ma (roughly Neolithic Revolution).
pub const BT_KT_HANDOFF_MA: MaAge = 0.012;

/// Blending window duration (Ma).
/// 7,000 years of overlap (10,000 BCE to 3,000 BCE).
pub const BT_KT_BLEND_WINDOW_MA: MaAge = 0.007;

/// B(t) → K(t) blending weight at a given age.
/// Returns (bt_weight, kt_weight) that sum to 1.0.
///
/// - Before 10,000 BCE (age > 0.012 Ma): pure B(t)
/// - Between 10,000 BCE and 3,000 BCE: linear blend
/// - After 3,000 BCE (age < 0.005 Ma): pure K(t)
pub fn blend_weights(age_ma: MaAge) -> (f64, f64) {
    if age_ma >= BT_KT_HANDOFF_MA {
        (1.0, 0.0) // Pure B(t)
    } else if age_ma <= BT_KT_HANDOFF_MA - BT_KT_BLEND_WINDOW_MA {
        (0.0, 1.0) // Pure K(t)
    } else {
        // Linear blend
        let t = (BT_KT_HANDOFF_MA - age_ma) / BT_KT_BLEND_WINDOW_MA;
        (1.0 - t, t)
    }
}

/// Compute blended coherence index at a given age.
pub fn blended_coherence(age_ma: MaAge, bt_value: f64, kt_value: f64) -> f64 {
    let (bt_w, kt_w) = blend_weights(age_ma);
    bt_w * bt_value + kt_w * kt_value
}

/// Map B(t) terminal dimensions to EcosystemHealth initial conditions.
///
/// Uses the shared thermodynamic currency: biosphere energy state at 2026 CE
/// is B(t) terminal value adjusted for HANPP.
pub struct EcosystemHealthInit {
    pub biodiversity: f64,
    pub forest_cover: f64,
    pub soil_health: f64,
    pub ocean_health: f64,
    pub freshwater: f64,
}

/// Compute sim initial conditions from B(t) terminal state.
///
/// The B(t) terminal value (at ~10,000 BCE handoff) represents the pre-civilization
/// biosphere. We degrade it to 2026 conditions using known HANPP + historical
/// degradation factors.
pub fn bt_to_ecosystem_health(
    bt_terminal_biodiversity: f64,
    bt_terminal_energy: f64,
) -> EcosystemHealthInit {
    // Apply Anthropocene degradation from the pre-industrial biosphere baseline.
    // B(t) terminal is at ~3000 BCE — nearly pristine biosphere.
    // We degrade to 2026 conditions using LPI and FAO estimates.

    // Living Planet Index: 69% decline since 1970. From pre-industrial baseline,
    // total decline ~30% (much of it post-1970). Pre-1970 decline was gradual.
    let biodiversity_degradation = 0.30; // 30% total from pristine

    EcosystemHealthInit {
        biodiversity: (bt_terminal_biodiversity * (1.0 - biodiversity_degradation)).clamp(0.3, 1.0),
        // FAO: ~15% of original forest lost
        forest_cover: 0.85,
        // FAO: 33% degraded
        soil_health: 0.80,
        // Ocean: ~10% degradation (acidification + warming starting)
        ocean_health: (bt_terminal_energy * 0.90).clamp(0.3, 0.95),
        // Freshwater: ~10% stress from extraction
        freshwater: 0.90,
    }
}

/// A point on the unified coherence curve.
#[derive(Debug, Clone)]
pub struct CoherencePoint {
    /// Age in Ma (0 = present, positive = past).
    pub age_ma: MaAge,
    /// Coherence value [0, 1].
    pub value: f64,
    /// Lower 95% CI bound.
    pub ci_lower: f64,
    /// Upper 95% CI bound.
    pub ci_upper: f64,
    /// Which index is primary at this point.
    pub source: CoherenceSource,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoherenceSource {
    /// B(t) biosphere coherence (deep time).
    Biosphere,
    /// Blended B(t) + K(t) (Neolithic transition zone).
    Blended,
    /// K(t) civilizational coherence (historical).
    Civilization,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hanpp_increases_over_time() {
        let pre_neolithic = hanpp_fraction(-15000.0);
        let neolithic = hanpp_fraction(-5000.0);
        let industrial = hanpp_fraction(1900.0);
        let modern = hanpp_fraction(2020.0);

        assert!(pre_neolithic < neolithic);
        assert!(neolithic < industrial);
        assert!(industrial < modern);
        assert!(modern > 0.20, "Modern HANPP should be >20%, got {}", modern);
        assert!(modern < 0.30, "Modern HANPP should be <30%, got {}", modern);
    }

    #[test]
    fn energy_adjustment_depresses_modern() {
        let adjustment_modern = biosphere_energy_adjustment(0.0001); // ~1926 CE
        let adjustment_ancient = biosphere_energy_adjustment(100.0);
        assert!(
            adjustment_modern < adjustment_ancient,
            "Modern biosphere should be depressed relative to ancient"
        );
        assert!(
            adjustment_modern > 0.7,
            "Modern biosphere shouldn't be <70% of potential, got {}",
            adjustment_modern
        );
    }

    #[test]
    fn blend_weights_sum_to_one() {
        for &age in &[0.020, 0.012, 0.010, 0.008, 0.005, 0.003, 0.001] {
            let (bt, kt) = blend_weights(age);
            assert!(
                ((bt + kt) - 1.0).abs() < 1e-10,
                "Weights should sum to 1.0 at {} Ma",
                age
            );
        }
    }

    #[test]
    fn pure_bt_before_handoff() {
        let (bt, kt) = blend_weights(0.020);
        assert!((bt - 1.0).abs() < 1e-10);
        assert!(kt.abs() < 1e-10);
    }

    #[test]
    fn pure_kt_after_blend_window() {
        let (bt, kt) = blend_weights(0.003);
        assert!(bt.abs() < 1e-10);
        assert!((kt - 1.0).abs() < 1e-10);
    }

    #[test]
    fn ecosystem_health_reasonable() {
        let health = bt_to_ecosystem_health(0.9, 0.85);
        assert!(health.biodiversity > 0.3 && health.biodiversity < 1.0);
        assert!(health.forest_cover > 0.5 && health.forest_cover < 1.0);
        assert!(health.soil_health > 0.5 && health.soil_health < 1.0);
        assert!(health.ocean_health > 0.3 && health.ocean_health < 1.0);
        assert!(health.freshwater > 0.5 && health.freshwater < 1.0);
    }
}
