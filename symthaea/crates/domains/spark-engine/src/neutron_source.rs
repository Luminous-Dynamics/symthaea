// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Compact Neutron Source Design (Path D)
//!
//! Even with Q < 1, LCF may be viable as a compact neutron generator for:
//! - Neutron activation analysis (NAA)
//! - Boron neutron capture therapy (BNCT)
//! - Material testing
//! - Security screening
//! - Isotope production

use crate::constants::*;
use serde::{Deserialize, Serialize};

/// Neutron source specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceSpec {
    /// Target neutron output (n/s)
    pub target_rate_n_s: f64,
    /// Neutron energy (MeV)
    pub neutron_energy_mev: f64,
    /// Application category
    pub application: NeutronApplication,
    /// Required beam purity (n/γ ratio)
    pub purity_ratio: f64,
    /// Portability requirement
    pub portability: Portability,
    /// Maximum acceptable input power (W)
    pub max_input_power_w: f64,
}

/// Neutron source applications.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeutronApplication {
    /// Neutron activation analysis
    NAA,
    /// Boron neutron capture therapy
    BNCT,
    /// Material testing / radiography
    MaterialTesting,
    /// Security screening (cargo, etc.)
    SecurityScreening,
    /// Isotope production
    IsotopeProduction,
    /// Research / general
    Research,
}

impl NeutronApplication {
    /// Typical neutron rate requirement (n/s).
    pub fn typical_rate(&self) -> f64 {
        match self {
            Self::NAA => 1e6,                // 10⁶ n/s
            Self::BNCT => 1e9,               // 10⁹ n/s (epithermal)
            Self::MaterialTesting => 1e7,    // 10⁷ n/s
            Self::SecurityScreening => 1e8,  // 10⁸ n/s
            Self::IsotopeProduction => 1e12, // 10¹² n/s
            Self::Research => 1e4,           // 10⁴ n/s (flexible)
        }
    }

    /// Description of the application.
    pub fn description(&self) -> &'static str {
        match self {
            Self::NAA => "Elemental analysis via neutron activation and gamma spectroscopy",
            Self::BNCT => "Cancer treatment using boron-10 neutron capture",
            Self::MaterialTesting => "Non-destructive testing and radiography",
            Self::SecurityScreening => "Detection of explosives, drugs, nuclear materials",
            Self::IsotopeProduction => "Production of medical and industrial radioisotopes",
            Self::Research => "General neutron science experiments",
        }
    }
}

/// Portability classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Portability {
    /// Fixed installation
    Fixed,
    /// Transportable (truck-mounted)
    Transportable,
    /// Portable (hand-carried)
    Portable,
    /// Handheld device
    Handheld,
}

impl Portability {
    /// Maximum mass for this portability class (kg).
    pub fn max_mass_kg(&self) -> f64 {
        match self {
            Self::Fixed => f64::INFINITY,
            Self::Transportable => 1000.0,
            Self::Portable => 25.0,
            Self::Handheld => 5.0,
        }
    }

    /// Maximum volume for this portability class (L).
    pub fn max_volume_l(&self) -> f64 {
        match self {
            Self::Fixed => f64::INFINITY,
            Self::Transportable => 1000.0,
            Self::Portable => 20.0,
            Self::Handheld => 2.0,
        }
    }
}

/// Neutron source design result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeutronSourceDesign {
    /// Specification this design addresses
    pub spec: SourceSpec,
    /// Active PdD volume (cm³)
    pub active_volume_cm3: f64,
    /// Pd mass (g)
    pub pd_mass_g: f64,
    /// Required trigger power (W)
    pub trigger_power_w: f64,
    /// Estimated neutron output (n/s)
    pub estimated_rate_n_s: f64,
    /// Q factor (will be << 1)
    pub q_factor: f64,
    /// Shielding mass (kg)
    pub shielding_mass_kg: f64,
    /// Total system mass (kg)
    pub total_mass_kg: f64,
    /// Total system volume (L)
    pub total_volume_l: f64,
    /// Estimated cost ($)
    pub cost_usd: f64,
    /// Feasibility assessment
    pub feasibility: SourceFeasibility,
    /// Design notes
    pub notes: Vec<String>,
}

/// Feasibility assessment for a neutron source design.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceFeasibility {
    /// Overall feasibility (0-1)
    pub score: f64,
    /// Physics feasibility (can we get the neutrons?)
    pub physics_ok: bool,
    /// Engineering feasibility (can we build it?)
    pub engineering_ok: bool,
    /// Meets portability requirement
    pub portability_ok: bool,
    /// Cost acceptable
    pub cost_ok: bool,
    /// Regulatory path exists
    pub regulatory_ok: bool,
    /// Primary limiting factor
    pub limiting_factor: String,
}

/// Neutron source designer.
pub struct NeutronSourceDesigner;

impl NeutronSourceDesigner {
    /// Design a neutron source for the given specification.
    ///
    /// Note: This assumes the anomalous enhancement observed by NASA is real.
    /// If using standard Gamow physics (Q << 1), the design may not be viable.
    pub fn design(spec: &SourceSpec, assume_anomaly: bool) -> NeutronSourceDesign {
        // Enhancement factor: 1 for standard physics, ~10^50 if anomaly is real
        let _enhancement = if assume_anomaly {
            // NASA observed 10³ n/s from ~0.01 cm³ volume
            // This implies an effective enhancement of ~10^50 over standard Gamow
            1e50
        } else {
            1.0
        };

        // Required active volume based on target rate
        // Using NASA scaling: ~10³ n/s per 0.01 cm³ with anomaly
        let base_rate_per_cm3 = if assume_anomaly {
            1e5 // 10⁵ n/s/cm³ based on NASA extrapolation
        } else {
            // Standard physics: essentially zero
            1e-45
        };

        let required_volume_cm3 = spec.target_rate_n_s / base_rate_per_cm3;

        // Pd mass (density 12.02 g/cm³)
        let pd_mass_g = required_volume_cm3 * PD_DENSITY;

        // Trigger power: estimate ~1 W/cm³ for X-ray trigger
        let trigger_power_w = required_volume_cm3 * 1.0;

        // Q factor
        let fusion_power_w = spec.target_rate_n_s * DD_Q_NEUTRON * 1.602e-13;
        let q_factor = fusion_power_w / trigger_power_w.max(1.0);

        // Shielding for 2.45 MeV D-D neutrons
        // ~15 cm of polyethylene for 99% attenuation
        let shielding_thickness_cm = 15.0;
        let core_radius_cm =
            (required_volume_cm3 * 3.0 / (4.0 * std::f64::consts::PI)).powf(1.0 / 3.0);
        let shielding_radius_cm = core_radius_cm + shielding_thickness_cm;
        let shielding_volume_cm3 = (4.0 / 3.0)
            * std::f64::consts::PI
            * (shielding_radius_cm.powi(3) - core_radius_cm.powi(3));
        let shielding_mass_kg = shielding_volume_cm3 * 0.94 / 1000.0; // Polyethylene density

        // Total system
        let electronics_mass_kg = 5.0; // Trigger electronics
        let structure_mass_kg = 2.0;
        let total_mass_kg =
            pd_mass_g / 1000.0 + shielding_mass_kg + electronics_mass_kg + structure_mass_kg;
        let total_volume_l =
            (4.0 / 3.0) * std::f64::consts::PI * shielding_radius_cm.powi(3) / 1000.0 + 5.0; // Electronics volume

        // Cost
        let pd_cost = pd_mass_g * 70.0; // $70/g for Pd
        let shielding_cost = shielding_mass_kg * 10.0; // $10/kg for polyethylene
        let electronics_cost = 10000.0; // X-ray trigger system
        let assembly_cost = 5000.0;
        let cost_usd = pd_cost + shielding_cost + electronics_cost + assembly_cost;

        // Feasibility
        let physics_ok = assume_anomaly || spec.target_rate_n_s < 1.0;
        let engineering_ok = required_volume_cm3 < 1e6;
        let portability_ok = total_mass_kg <= spec.portability.max_mass_kg()
            && total_volume_l <= spec.portability.max_volume_l();
        let cost_ok = cost_usd < 1e6;
        let regulatory_ok = true; // D-D neutrons, no tritium fuel

        let score = [
            if physics_ok { 0.3 } else { 0.0 },
            if engineering_ok { 0.2 } else { 0.0 },
            if portability_ok { 0.2 } else { 0.0 },
            if cost_ok { 0.15 } else { 0.0 },
            if regulatory_ok { 0.15 } else { 0.0 },
        ]
        .iter()
        .sum();

        let limiting_factor = if !physics_ok {
            "Physics: standard Gamow gives Q << 1".to_string()
        } else if !portability_ok {
            format!(
                "Size/mass exceeds {} limit",
                match spec.portability {
                    Portability::Handheld => "handheld",
                    Portability::Portable => "portable",
                    Portability::Transportable => "transportable",
                    Portability::Fixed => "fixed",
                }
            )
        } else if !cost_ok {
            "Cost exceeds $1M".to_string()
        } else {
            "None - design is feasible".to_string()
        };

        let feasibility = SourceFeasibility {
            score,
            physics_ok,
            engineering_ok,
            portability_ok,
            cost_ok,
            regulatory_ok,
            limiting_factor,
        };

        let mut notes = vec![];
        if assume_anomaly {
            notes.push("Design assumes NASA anomalous enhancement is real".to_string());
            notes.push("If standard physics, multiply Pd mass by ~10^50".to_string());
        } else {
            notes.push("Using standard Gamow physics - design is NOT viable".to_string());
        }
        if q_factor < 1.0 {
            notes.push(format!(
                "Q = {:.2e} << 1: net energy consumer, not generator",
                q_factor
            ));
        }

        NeutronSourceDesign {
            spec: spec.clone(),
            active_volume_cm3: required_volume_cm3,
            pd_mass_g,
            trigger_power_w,
            estimated_rate_n_s: spec.target_rate_n_s,
            q_factor,
            shielding_mass_kg,
            total_mass_kg,
            total_volume_l,
            cost_usd,
            feasibility,
            notes,
        }
    }

    /// Compare LCF source to competing technologies.
    pub fn compare_to_competitors(_spec: &SourceSpec) -> Vec<CompetitorComparison> {
        vec![
            CompetitorComparison {
                name: "Californium-252".to_string(),
                neutron_rate: 2.3e6, // per μg
                cost_usd: 50000.0,   // per μg
                portability: Portability::Portable,
                pros: vec!["Very compact".to_string(), "Passive operation".to_string()],
                cons: vec![
                    "Decays (2.6 year half-life)".to_string(),
                    "Expensive".to_string(),
                ],
            },
            CompetitorComparison {
                name: "D-D Generator (accelerator)".to_string(),
                neutron_rate: 1e8,
                cost_usd: 200000.0,
                portability: Portability::Transportable,
                pros: vec![
                    "No radioactive source".to_string(),
                    "Controllable".to_string(),
                ],
                cons: vec![
                    "High voltage (>100 kV)".to_string(),
                    "Tritium buildup".to_string(),
                ],
            },
            CompetitorComparison {
                name: "D-T Generator (accelerator)".to_string(),
                neutron_rate: 1e10,
                cost_usd: 300000.0,
                portability: Portability::Transportable,
                pros: vec!["High output".to_string(), "Well-established".to_string()],
                cons: vec![
                    "Tritium source required".to_string(),
                    "14.1 MeV neutrons".to_string(),
                ],
            },
            CompetitorComparison {
                name: "Reactor (research)".to_string(),
                neutron_rate: 1e14,
                cost_usd: 1e8,
                portability: Portability::Fixed,
                pros: vec!["Very high flux".to_string(), "Thermal neutrons".to_string()],
                cons: vec![
                    "Fixed installation".to_string(),
                    "Heavy regulation".to_string(),
                ],
            },
        ]
    }
}

/// Comparison to a competing neutron source technology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompetitorComparison {
    /// Technology name
    pub name: String,
    /// Typical neutron rate (n/s)
    pub neutron_rate: f64,
    /// Typical cost ($)
    pub cost_usd: f64,
    /// Portability class
    pub portability: Portability,
    /// Advantages
    pub pros: Vec<String>,
    /// Disadvantages
    pub cons: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_application_rates() {
        assert!(NeutronApplication::NAA.typical_rate() > 0.0);
        assert!(NeutronApplication::BNCT.typical_rate() > NeutronApplication::NAA.typical_rate());
    }

    #[test]
    fn test_design_with_anomaly() {
        let spec = SourceSpec {
            target_rate_n_s: 1e6,
            neutron_energy_mev: DD_NEUTRON_ENERGY,
            application: NeutronApplication::NAA,
            purity_ratio: 100.0,
            portability: Portability::Transportable,
            max_input_power_w: 10000.0,
        };

        let design = NeutronSourceDesigner::design(&spec, true);

        assert!(design.pd_mass_g > 0.0);
        assert!(design.q_factor < 1.0);
        assert!(design.feasibility.physics_ok);
    }

    #[test]
    fn test_design_standard_physics() {
        let spec = SourceSpec {
            target_rate_n_s: 1e6,
            neutron_energy_mev: DD_NEUTRON_ENERGY,
            application: NeutronApplication::NAA,
            purity_ratio: 100.0,
            portability: Portability::Transportable,
            max_input_power_w: 10000.0,
        };

        let design = NeutronSourceDesigner::design(&spec, false);

        // With standard physics, should NOT be feasible
        assert!(!design.feasibility.physics_ok);
    }

    #[test]
    fn test_competitor_comparison() {
        let spec = SourceSpec {
            target_rate_n_s: 1e6,
            neutron_energy_mev: DD_NEUTRON_ENERGY,
            application: NeutronApplication::NAA,
            purity_ratio: 100.0,
            portability: Portability::Transportable,
            max_input_power_w: 10000.0,
        };

        let competitors = NeutronSourceDesigner::compare_to_competitors(&spec);
        assert!(!competitors.is_empty());
    }
}
