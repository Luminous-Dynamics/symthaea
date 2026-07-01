// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Neutron Shielding: Radiation Containment for Spark Engine
//!
//! Models neutron transport, attenuation, and shielding requirements
//! for safe operation of the LCF power system.
//!
//! ## Neutron Physics
//!
//! - **D-D reaction**: 2.45 MeV neutrons
//! - **D-T reaction**: 14.1 MeV neutrons
//!
//! ## Attenuation Mechanisms
//!
//! 1. **Elastic scattering**: Neutron bounces off nucleus, loses energy
//! 2. **Inelastic scattering**: Neutron excites nucleus, loses more energy
//! 3. **Absorption**: Neutron captured by nucleus
//!
//! ## Shielding Strategy
//!
//! ```text
//! Fusion Zone → Moderator (slow down) → Absorber (capture) → Environment
//!               H-rich material         B, Gd, Cd, Li          Safe dose
//! ```
//!
//! ## Key Metrics
//!
//! - **Mean free path (λ)**: Average distance between interactions
//! - **Macroscopic cross-section (Σ)**: 1/λ = N * σ
//! - **Tenth-value layer (TVL)**: Thickness to reduce flux by 10x

use super::radiation_damage::FusionReaction;
use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;

/// Neutron interaction cross-sections for a material
#[derive(Debug, Clone)]
pub struct NeutronCrossSection {
    /// Material name
    pub name: String,
    /// Atomic number
    pub z: u8,
    /// Atomic mass (u)
    pub a: f64,
    /// Density (kg/m³)
    pub density: f64,
    /// Elastic scattering cross-section (barn) at 14.1 MeV
    pub sigma_elastic_14mev: f64,
    /// Elastic scattering cross-section (barn) at 2.45 MeV
    pub sigma_elastic_2mev: f64,
    /// Absorption cross-section (barn) at thermal energies
    pub sigma_abs_thermal: f64,
    /// Inelastic threshold (MeV)
    pub inelastic_threshold: f64,
}

impl NeutronCrossSection {
    /// Number density (atoms/m³)
    pub fn number_density(&self) -> f64 {
        let n_a = 6.022_140_76e23; // Avogadro's number (2019 SI exact)
        self.density * n_a / (self.a * 1e-3)
    }

    /// Macroscopic cross-section (1/m) at given energy
    pub fn macro_cross_section(&self, energy_mev: f64) -> f64 {
        let sigma = if energy_mev > 10.0 {
            self.sigma_elastic_14mev
        } else if energy_mev > 1.0 {
            self.sigma_elastic_2mev
        } else {
            // Thermal range - absorption dominates
            self.sigma_abs_thermal
        };

        // Convert barn to m² (1 barn = 1e-28 m²)
        self.number_density() * sigma * 1e-28
    }

    /// Mean free path (m)
    pub fn mean_free_path(&self, energy_mev: f64) -> f64 {
        1.0 / self.macro_cross_section(energy_mev)
    }

    /// Half-value layer (m)
    pub fn hvl(&self, energy_mev: f64) -> f64 {
        2.0_f64.ln() / self.macro_cross_section(energy_mev)
    }

    /// Tenth-value layer (m)
    pub fn tvl(&self, energy_mev: f64) -> f64 {
        10.0_f64.ln() / self.macro_cross_section(energy_mev)
    }

    /// Average energy loss per collision (fraction)
    pub fn avg_energy_loss(&self) -> f64 {
        // For elastic scattering: ξ = 1 - (A-1)²/(2A) * ln((A+1)/(A-1))
        // Simplified: ξ ≈ 2/(A + 2/3) for A > 10
        if self.a > 10.0 {
            2.0 / (self.a + 0.667)
        } else {
            // Hydrogen is special: ξ ≈ 1 (can lose all energy in one collision)
            1.0 - ((self.a - 1.0) / (self.a + 1.0)).powi(2)
        }
    }

    /// Collisions to thermalize from given energy
    pub fn collisions_to_thermal(&self, start_energy_mev: f64) -> f64 {
        let thermal_energy = 0.025e-6; // 25 meV in MeV
        let energy_ratio = start_energy_mev / thermal_energy;
        energy_ratio.ln() / self.avg_energy_loss()
    }
}

/// Shielding material with neutron properties
#[derive(Debug, Clone)]
pub struct ShieldingMaterial {
    /// Cross-section data
    pub cross_section: NeutronCrossSection,
    /// Primary function
    pub function: ShieldFunction,
    /// Gamma production (secondary radiation)
    pub gamma_production: f64, // Relative scale 0-1
    /// Activation potential (long-lived isotopes)
    pub activation: f64, // Relative scale 0-1
}

/// Shielding function
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShieldFunction {
    /// Slows neutrons via elastic scattering (H-rich)
    Moderator,
    /// Captures thermal neutrons (B, Gd, Cd)
    Absorber,
    /// Reflects neutrons back (Be, graphite)
    Reflector,
    /// Combined moderation and absorption
    Composite,
}

/// Shielding analysis result
#[derive(Debug, Clone)]
pub struct ShieldingAnalysis {
    /// Material name
    pub material: String,
    /// Thickness required for 99% attenuation (m)
    pub thickness_99: f64,
    /// Thickness required for 99.9% attenuation (m)
    pub thickness_999: f64,
    /// Mass per unit area (kg/m²)
    pub areal_density: f64,
    /// Secondary gamma production
    pub gamma_dose_factor: f64,
    /// Long-term activation
    pub activation_factor: f64,
    /// Overall shielding quality (0-1)
    pub quality_score: f32,
}

/// Neutron dose result
#[derive(Debug, Clone)]
pub struct NeutronDose {
    /// Neutron flux at surface (n/cm²/s)
    pub flux: f64,
    /// Dose equivalent rate (mSv/hr)
    pub dose_rate_msv_hr: f64,
    /// Is below occupational limit (20 mSv/year)?
    pub occupational_safe: bool,
    /// Is below public limit (1 mSv/year)?
    pub public_safe: bool,
    /// Required additional shielding (m) for public safety
    pub additional_shielding_m: f64,
}

/// Neutron shielding system
#[derive(Debug, Clone)]
pub struct NeutronShielding {
    /// Concept vectors
    pub moderation: ContinuousHV,
    pub absorption: ContinuousHV,
    pub reflection: ContinuousHV,
    pub activation: ContinuousHV,

    /// Material database
    pub materials: Vec<ShieldingMaterial>,
}

impl NeutronShielding {
    /// Create from genesis
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        let moderation = genesis.hv("shielding::moderation", PHYSICS_DIM);
        let absorption = genesis.hv("shielding::absorption", PHYSICS_DIM);
        let reflection = genesis.hv("shielding::reflection", PHYSICS_DIM);
        let activation = genesis.hv("shielding::activation", PHYSICS_DIM);

        let mut system = Self {
            moderation,
            absorption,
            reflection,
            activation,
            materials: Vec::new(),
        };

        system.init_materials();
        system
    }

    /// Initialize shielding materials database
    fn init_materials(&mut self) {
        // Moderators (H-rich)
        self.materials.push(ShieldingMaterial {
            cross_section: NeutronCrossSection {
                name: "Water (H2O)".to_string(),
                z: 1,
                a: 18.0,
                density: 1000.0,
                sigma_elastic_14mev: 0.7,
                sigma_elastic_2mev: 2.5,
                sigma_abs_thermal: 0.33,
                inelastic_threshold: 0.0,
            },
            function: ShieldFunction::Moderator,
            gamma_production: 0.3,
            activation: 0.1,
        });

        self.materials.push(ShieldingMaterial {
            cross_section: NeutronCrossSection {
                name: "Polyethylene".to_string(),
                z: 1,
                a: 14.0,
                density: 940.0,
                sigma_elastic_14mev: 0.8,
                sigma_elastic_2mev: 3.0,
                sigma_abs_thermal: 0.00,
                inelastic_threshold: 0.0,
            },
            function: ShieldFunction::Moderator,
            gamma_production: 0.1,
            activation: 0.05,
        });

        // Absorbers
        self.materials.push(ShieldingMaterial {
            cross_section: NeutronCrossSection {
                name: "Borated Polyethylene".to_string(),
                z: 5,
                a: 14.5,
                density: 1000.0,
                sigma_elastic_14mev: 0.8,
                sigma_elastic_2mev: 3.0,
                sigma_abs_thermal: 767.0, // B-10 has huge absorption
                inelastic_threshold: 0.0,
            },
            function: ShieldFunction::Composite,
            gamma_production: 0.5, // (n,α) produces gammas
            activation: 0.1,
        });

        self.materials.push(ShieldingMaterial {
            cross_section: NeutronCrossSection {
                name: "Boron Carbide (B4C)".to_string(),
                z: 5,
                a: 55.0,
                density: 2520.0,
                sigma_elastic_14mev: 1.0,
                sigma_elastic_2mev: 2.0,
                sigma_abs_thermal: 600.0,
                inelastic_threshold: 0.5,
            },
            function: ShieldFunction::Absorber,
            gamma_production: 0.4,
            activation: 0.15,
        });

        self.materials.push(ShieldingMaterial {
            cross_section: NeutronCrossSection {
                name: "Gadolinium".to_string(),
                z: 64,
                a: 157.0,
                density: 7900.0,
                sigma_elastic_14mev: 5.0,
                sigma_elastic_2mev: 6.0,
                sigma_abs_thermal: 49000.0, // Highest known
                inelastic_threshold: 0.1,
            },
            function: ShieldFunction::Absorber,
            gamma_production: 0.8, // High gamma production
            activation: 0.5,
        });

        // Structural/shielding materials
        self.materials.push(ShieldingMaterial {
            cross_section: NeutronCrossSection {
                name: "Concrete".to_string(),
                z: 14,
                a: 23.0,
                density: 2300.0,
                sigma_elastic_14mev: 2.5,
                sigma_elastic_2mev: 4.0,
                sigma_abs_thermal: 0.003,
                inelastic_threshold: 1.0,
            },
            function: ShieldFunction::Composite,
            gamma_production: 0.4,
            activation: 0.3,
        });

        self.materials.push(ShieldingMaterial {
            cross_section: NeutronCrossSection {
                name: "Steel".to_string(),
                z: 26,
                a: 56.0,
                density: 7800.0,
                sigma_elastic_14mev: 2.0,
                sigma_elastic_2mev: 3.0,
                sigma_abs_thermal: 2.6,
                inelastic_threshold: 0.85,
            },
            function: ShieldFunction::Composite,
            gamma_production: 0.6,
            activation: 0.4,
        });

        self.materials.push(ShieldingMaterial {
            cross_section: NeutronCrossSection {
                name: "Lead".to_string(),
                z: 82,
                a: 207.0,
                density: 11340.0,
                sigma_elastic_14mev: 5.0,
                sigma_elastic_2mev: 7.0,
                sigma_abs_thermal: 0.17,
                inelastic_threshold: 0.8,
            },
            function: ShieldFunction::Reflector,
            gamma_production: 0.2, // Good gamma shield too
            activation: 0.2,
        });

        // Specialty materials
        self.materials.push(ShieldingMaterial {
            cross_section: NeutronCrossSection {
                name: "Lithium Hydride (LiH)".to_string(),
                z: 3,
                a: 8.0,
                density: 780.0,
                sigma_elastic_14mev: 1.0,
                sigma_elastic_2mev: 4.0,
                sigma_abs_thermal: 71.0, // Li-6 absorption
                inelastic_threshold: 0.0,
            },
            function: ShieldFunction::Composite,
            gamma_production: 0.1, // Clean absorption (tritium product)
            activation: 0.05,
        });
    }

    /// Analyze shielding effectiveness
    pub fn analyze(
        &self,
        material: &ShieldingMaterial,
        reaction: FusionReaction,
    ) -> ShieldingAnalysis {
        let energy = reaction.neutron_energy_mev().unwrap_or(2.45);

        let tvl = material.cross_section.tvl(energy);

        // 99% = 2 TVL, 99.9% = 3 TVL
        let thickness_99 = 2.0 * tvl;
        let thickness_999 = 3.0 * tvl;

        let areal_density = thickness_99 * material.cross_section.density;

        // Quality score considers multiple factors
        let thickness_factor = (0.5 / thickness_99).min(1.0);
        let gamma_factor = 1.0 - material.gamma_production;
        let activation_factor = 1.0 - material.activation;

        let quality_score =
            (thickness_factor * 0.4 + gamma_factor * 0.3 + activation_factor * 0.3) as f32;

        ShieldingAnalysis {
            material: material.cross_section.name.clone(),
            thickness_99,
            thickness_999,
            areal_density,
            gamma_dose_factor: material.gamma_production,
            activation_factor: material.activation,
            quality_score,
        }
    }

    /// Calculate dose at given distance from unshielded source
    ///
    /// This calculation includes:
    /// - Neutron branching ratio (D-D is only 50% neutron-producing)
    /// - Self-shielding within the core (moderation and absorption)
    /// - Geometric escape fraction
    pub fn unshielded_dose(
        &self,
        reaction: FusionReaction,
        power_kw: f64,
        distance_m: f64,
    ) -> NeutronDose {
        let energy = reaction.neutron_energy_mev().unwrap_or(2.45);
        let total_energy = reaction.total_energy_mev();

        // Total fusion reactions per second
        // Power (W) = reactions/s * energy/reaction
        let reactions_per_s = power_kw * 1000.0 / (total_energy * super::constants::MEV_TO_J);

        // Apply neutron yield fraction (branching ratio)
        // D-D: only ~50% of reactions produce neutrons (He-3+n branch)
        // D-T: 100% neutron yield
        let neutron_yield = reaction.neutron_yield_fraction();
        let neutrons_per_s = reactions_per_s * neutron_yield;

        // Apply self-shielding factor
        // Accounts for neutrons absorbed/moderated within the core before escaping
        // Estimate core radius from power (higher power = larger core)
        let core_radius_m = (power_kw / 50.0).powf(1.0 / 3.0) * 0.1; // ~10cm for 50 kW
        let self_shielding = reaction.self_shielding_factor(core_radius_m);
        let escaping_neutrons = neutrons_per_s * self_shielding;

        // Flux at distance (1/r² falloff for point source)
        let flux = escaping_neutrons / (4.0 * std::f64::consts::PI * distance_m.powi(2));
        let flux_cm2 = flux * 1e-4; // Convert from n/m²/s to n/cm²/s

        // Dose conversion factor (ICRP-116 values)
        // Fast neutrons (>1 MeV): ~10 pSv·cm²/n
        // Thermal neutrons: ~2 pSv·cm²/n
        let dcf = if energy > 1.0 { 10e-12 } else { 2e-12 }; // Sv·cm²/n

        let dose_rate_sv_s = flux_cm2 * dcf;
        let dose_rate_msv_hr = dose_rate_sv_s * 3600.0 * 1000.0;

        // Annual dose at 2000 hours exposure
        let annual_dose = dose_rate_msv_hr * 2000.0;

        let occupational_safe = annual_dose < 20.0; // 20 mSv/year
        let public_safe = annual_dose < 1.0; // 1 mSv/year

        // Additional shielding needed
        let additional = if public_safe {
            0.0
        } else {
            // Need to reduce by factor of (annual_dose / 1.0)
            let reduction_factor = annual_dose / 1.0;
            let tvls_needed = reduction_factor.log10();
            // Assume borated polyethylene
            let tvl = 0.15; // ~15 cm for fast neutrons
            tvls_needed * tvl
        };

        NeutronDose {
            flux: flux_cm2,
            dose_rate_msv_hr,
            occupational_safe,
            public_safe,
            additional_shielding_m: additional,
        }
    }

    /// Calculate shielded dose
    pub fn shielded_dose(
        &self,
        reaction: FusionReaction,
        power_kw: f64,
        distance_m: f64,
        shield_material: &ShieldingMaterial,
        shield_thickness_m: f64,
    ) -> NeutronDose {
        let unshielded = self.unshielded_dose(reaction, power_kw, distance_m);

        let energy = reaction.neutron_energy_mev().unwrap_or(2.45);
        let macro_xs = shield_material.cross_section.macro_cross_section(energy);

        // Exponential attenuation: I = I0 * exp(-Σ * x)
        let attenuation = (-macro_xs * shield_thickness_m).exp();

        let flux_shielded = unshielded.flux * attenuation;
        let dose_rate_shielded = unshielded.dose_rate_msv_hr * attenuation;

        let annual_dose = dose_rate_shielded * 2000.0;

        NeutronDose {
            flux: flux_shielded,
            dose_rate_msv_hr: dose_rate_shielded,
            occupational_safe: annual_dose < 20.0,
            public_safe: annual_dose < 1.0,
            additional_shielding_m: 0.0, // Already shielded
        }
    }

    /// Find optimal shielding combination
    ///
    /// Uses target dose rate of 0.001 mSv/hr (public safety limit)
    /// and calculates the minimum thickness required to achieve it.
    pub fn optimize_shielding(
        &self,
        reaction: FusionReaction,
        power_kw: f64,
        distance_m: f64,
        max_thickness_m: f64,
        max_mass_kg_m2: f64,
    ) -> OptimalShielding {
        // Target dose rate for public safety
        let target_dose_msv_hr = 0.001;

        // Get unshielded dose first
        let unshielded = self.unshielded_dose(reaction, power_kw, distance_m);

        // Required attenuation factor
        let required_attenuation = target_dose_msv_hr / unshielded.dose_rate_msv_hr;

        let energy = reaction.neutron_energy_mev().unwrap_or(2.45);

        let mut best: Option<(String, f64, f64, f64, f32)> = None; // (name, thickness, mass, dose, quality)

        for material in &self.materials {
            let analysis = self.analyze(material, reaction);
            let macro_xs = material.cross_section.macro_cross_section(energy);

            // Calculate required thickness: attenuation = exp(-Σ*x)
            // x = -ln(attenuation) / Σ
            let required_thickness = if required_attenuation >= 1.0 {
                0.0 // No shielding needed
            } else if required_attenuation <= 0.0 || macro_xs <= 0.0 {
                f64::INFINITY // Impossible to achieve
            } else {
                -required_attenuation.ln() / macro_xs
            };

            // Use the calculated thickness, capped at max
            let actual_thickness = required_thickness.min(max_thickness_m);

            // Check mass constraint
            let mass_per_area = actual_thickness * material.cross_section.density;
            if mass_per_area > max_mass_kg_m2 {
                continue;
            }

            // Calculate actual dose with this thickness
            let shielded =
                self.shielded_dose(reaction, power_kw, distance_m, material, actual_thickness);

            // Score: prioritize meeting target, then minimize thickness
            let dose_factor = if shielded.dose_rate_msv_hr <= target_dose_msv_hr {
                1.0 // Meets target
            } else {
                target_dose_msv_hr / shielded.dose_rate_msv_hr // Partial credit
            };

            let thickness_factor = 1.0 - (actual_thickness / max_thickness_m).min(1.0);
            let quality = analysis.quality_score as f64;

            let score = dose_factor * 0.6 + thickness_factor * 0.2 + quality * 0.2;

            if best.is_none() || score > best.as_ref().unwrap().4 as f64 {
                best = Some((
                    material.cross_section.name.clone(),
                    actual_thickness,
                    mass_per_area,
                    shielded.dose_rate_msv_hr,
                    score as f32,
                ));
            }
        }

        let (name, thickness, mass, dose, quality) =
            best.expect("Should find at least one material");

        OptimalShielding {
            primary_material: name,
            thickness_m: thickness,
            mass_per_area: mass,
            final_dose: dose,
            public_safe: dose <= target_dose_msv_hr,
            quality_score: quality,
        }
    }
}

/// Optimal shielding result
#[derive(Debug, Clone)]
pub struct OptimalShielding {
    /// Primary shielding material
    pub primary_material: String,
    /// Required thickness (m)
    pub thickness_m: f64,
    /// Mass per unit area (kg/m²)
    pub mass_per_area: f64,
    /// Final dose rate (mSv/hr)
    pub final_dose: f64,
    /// Is public-safe?
    pub public_safe: bool,
    /// Quality score
    pub quality_score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> NeutronShielding {
        let genesis = GenesisSeed::from_phrase("shielding test");
        NeutronShielding::from_genesis(&genesis)
    }

    #[test]
    fn test_cross_section_calculations() {
        let water = NeutronCrossSection {
            name: "Water".to_string(),
            z: 1,
            a: 18.0,
            density: 1000.0,
            sigma_elastic_14mev: 0.7,
            sigma_elastic_2mev: 2.5,
            sigma_abs_thermal: 0.33,
            inelastic_threshold: 0.0,
        };

        assert!(water.number_density() > 0.0);
        assert!(water.mean_free_path(14.1) > 0.0);
        assert!(water.hvl(14.1) > 0.0);
        assert!(water.tvl(14.1) > water.hvl(14.1));
    }

    #[test]
    fn test_materials_loaded() {
        let shielding = setup();
        assert!(shielding.materials.len() >= 5);
    }

    #[test]
    fn test_shielding_analysis() {
        let shielding = setup();

        let borated_pe = shielding
            .materials
            .iter()
            .find(|m| m.cross_section.name.contains("Borated"))
            .expect("Should have borated polyethylene");

        let analysis = shielding.analyze(borated_pe, FusionReaction::DT);

        assert!(analysis.thickness_99 > 0.0);
        assert!(analysis.thickness_999 > analysis.thickness_99);
        assert!(analysis.quality_score > 0.0);
    }

    #[test]
    fn test_unshielded_dose() {
        let shielding = setup();

        let dose = shielding.unshielded_dose(FusionReaction::DT, 100.0, 1.0);

        assert!(dose.flux > 0.0);
        assert!(dose.dose_rate_msv_hr > 0.0);
        // 100 kW at 1m should definitely be unsafe
        assert!(!dose.public_safe);
    }

    #[test]
    fn test_shielded_dose() {
        let shielding = setup();

        let borated_pe = shielding
            .materials
            .iter()
            .find(|m| m.cross_section.name.contains("Borated"))
            .unwrap();

        let unshielded = shielding.unshielded_dose(FusionReaction::DD, 5.0, 2.0);
        let shielded = shielding.shielded_dose(
            FusionReaction::DD,
            5.0,
            2.0,
            borated_pe,
            0.3, // 30 cm
        );

        assert!(shielded.dose_rate_msv_hr < unshielded.dose_rate_msv_hr);
    }

    #[test]
    fn test_optimize_shielding() {
        let shielding = setup();

        let optimal = shielding.optimize_shielding(
            FusionReaction::DD,
            5.0,   // 5 kW
            2.0,   // 2 m distance
            0.5,   // Max 50 cm thickness
            500.0, // Max 500 kg/m²
        );

        assert!(optimal.thickness_m > 0.0);
        assert!(optimal.thickness_m <= 0.5);
    }
}
