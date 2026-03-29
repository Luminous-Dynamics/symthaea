// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Neutron Shielding
//!
//! Simplified shielding calculations for D-D neutrons (2.45 MeV).

use serde::{Deserialize, Serialize};

/// Shielding calculation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShieldingResult {
    /// Material name
    pub material: String,
    /// Required thickness for target attenuation (cm)
    pub thickness_cm: f64,
    /// Mass per unit area (g/cm²)
    pub areal_density: f64,
    /// Final dose rate (mSv/hr)
    pub dose_rate_msv_hr: f64,
    /// Is public-safe (< 1 mSv/year at 2000 hr exposure)?
    pub public_safe: bool,
    /// Total mass for spherical shell (kg)
    pub total_mass_kg: f64,
}

/// Shielding material properties.
#[derive(Debug, Clone)]
pub struct ShieldingMaterial {
    /// Material name
    pub name: &'static str,
    /// Density (g/cm³)
    pub density: f64,
    /// Tenth-value layer for 2.45 MeV neutrons (cm)
    pub tvl_cm: f64,
    /// Half-value layer (cm)
    pub hvl_cm: f64,
}

impl ShieldingMaterial {
    /// Borated polyethylene (5% B)
    pub const BORATED_PE: ShieldingMaterial = ShieldingMaterial {
        name: "Borated Polyethylene (5% B)",
        density: 1.0,
        tvl_cm: 15.0,
        hvl_cm: 4.5,
    };

    /// Water
    pub const WATER: ShieldingMaterial = ShieldingMaterial {
        name: "Water",
        density: 1.0,
        tvl_cm: 20.0,
        hvl_cm: 6.0,
    };

    /// Concrete
    pub const CONCRETE: ShieldingMaterial = ShieldingMaterial {
        name: "Concrete",
        density: 2.3,
        tvl_cm: 25.0,
        hvl_cm: 7.5,
    };

    /// Polyethylene (no boron)
    pub const POLYETHYLENE: ShieldingMaterial = ShieldingMaterial {
        name: "Polyethylene",
        density: 0.94,
        tvl_cm: 18.0,
        hvl_cm: 5.4,
    };
}

/// Neutron shielding calculator.
pub struct NeutronShielding;

impl NeutronShielding {
    /// Calculate unshielded dose rate.
    ///
    /// # Arguments
    /// * `neutron_rate` - Neutron production rate (n/s)
    /// * `distance_m` - Distance from source (m)
    pub fn unshielded_dose(neutron_rate: f64, distance_m: f64) -> f64 {
        // Flux at distance (1/r² for point source)
        let flux = neutron_rate / (4.0 * std::f64::consts::PI * (distance_m * 100.0).powi(2));
        let flux_cm2 = flux; // Already in n/cm²/s

        // Dose conversion factor for 2.45 MeV neutrons
        // ICRP-116: ~10 pSv·cm²/n for fast neutrons
        let dcf = 10e-12; // Sv·cm²/n

        let dose_sv_s = flux_cm2 * dcf;
        dose_sv_s * 3600.0 * 1000.0 // Convert to mSv/hr
    }

    /// Calculate required shielding.
    ///
    /// # Arguments
    /// * `neutron_rate` - Neutron production rate (n/s)
    /// * `distance_m` - Distance from source (m)
    /// * `material` - Shielding material
    /// * `target_dose_msv_hr` - Target dose rate (mSv/hr)
    pub fn required_shielding(
        neutron_rate: f64,
        distance_m: f64,
        material: &ShieldingMaterial,
        target_dose_msv_hr: f64,
    ) -> ShieldingResult {
        let unshielded = Self::unshielded_dose(neutron_rate, distance_m);

        // Required attenuation
        let attenuation_needed = target_dose_msv_hr / unshielded;

        // Number of TVLs needed
        let tvls = if attenuation_needed < 1.0 {
            -attenuation_needed.log10()
        } else {
            0.0
        };

        let thickness_cm = tvls * material.tvl_cm;
        let areal_density = thickness_cm * material.density;

        // Final dose with this shielding
        let dose_rate = unshielded * 10.0_f64.powf(-tvls);

        // Public safety: < 1 mSv/year at 2000 hr/year exposure
        let annual_dose = dose_rate * 2000.0;
        let public_safe = annual_dose < 1.0;

        // Total mass for spherical shell at distance
        // Shell volume = 4π/3 × ((r + t)³ - r³)
        let r_inner_cm = distance_m * 100.0;
        let r_outer_cm = r_inner_cm + thickness_cm;
        let shell_volume_cm3 = (4.0 / 3.0) * std::f64::consts::PI *
            (r_outer_cm.powi(3) - r_inner_cm.powi(3));
        let total_mass_kg = shell_volume_cm3 * material.density / 1000.0;

        ShieldingResult {
            material: material.name.to_string(),
            thickness_cm,
            areal_density,
            dose_rate_msv_hr: dose_rate,
            public_safe,
            total_mass_kg,
        }
    }

    /// Calculate shielded dose rate.
    pub fn shielded_dose(
        neutron_rate: f64,
        distance_m: f64,
        material: &ShieldingMaterial,
        thickness_cm: f64,
    ) -> f64 {
        let unshielded = Self::unshielded_dose(neutron_rate, distance_m);
        let tvls = thickness_cm / material.tvl_cm;
        unshielded * 10.0_f64.powf(-tvls)
    }

    /// Recommend optimal shielding for portable application.
    pub fn recommend_portable(neutron_rate: f64) -> ShieldingResult {
        // Target: public-safe at 1 m distance
        Self::required_shielding(
            neutron_rate,
            1.0,
            &ShieldingMaterial::BORATED_PE,
            0.0005, // 0.5 μSv/hr (very conservative)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unshielded_dose() {
        let dose = NeutronShielding::unshielded_dose(1e6, 1.0);
        assert!(dose > 0.0);
    }

    #[test]
    fn test_shielding_calculation() {
        // Use high neutron rate to ensure shielding is needed
        let result = NeutronShielding::required_shielding(
            1e10,  // 10 billion n/s - definitely needs shielding
            1.0,
            &ShieldingMaterial::BORATED_PE,
            0.0001, // Very low target dose
        );

        assert!(result.thickness_cm > 0.0, "Should need shielding for high flux");
        assert!(result.dose_rate_msv_hr <= 0.0002, "Should achieve target dose");
    }

    #[test]
    fn test_shielded_dose() {
        let unshielded = NeutronShielding::unshielded_dose(1e6, 1.0);
        let shielded = NeutronShielding::shielded_dose(
            1e6,
            1.0,
            &ShieldingMaterial::BORATED_PE,
            30.0, // 2 TVL
        );

        assert!(shielded < unshielded);
        // 2 TVL should give ~100× reduction
        assert!(shielded < unshielded / 50.0);
    }
}
