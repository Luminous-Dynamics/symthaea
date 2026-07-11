// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Spark Engine: Unified LCF Power System Specification
//!
//! This module integrates all components of the Lattice Confinement Fusion
//! power system into a single, coherent specification.
//!
//! ## Physics honesty — read before citing this header
//!
//! The v1.0 diagram below is a **hypothetical design target, not a validated
//! device**. Under standard Gamow physics, D-D fusion at 300 K gives
//! `<σv>` ≈ 5×10⁻⁹⁰ cm³/s and Q ≈ 10⁻⁵⁷ — no net energy gain is possible
//! (Q > 1 requires T > 10⁷ K). The design is contingent on the unexplained
//! ~40-order-of-magnitude rate anomaly reported by NASA Glenn (Steinetz et
//! al. 2020, Phys. Rev. C) being real, controllable, AND vastly scalable —
//! none of which is established. See `crates/domains/spark-engine/docs/
//! TECHNICAL_REPORT.md` for the honest assessment; the credible near-term
//! application is a compact neutron source, not a power system. The Q-factor
//! enforcement in this physics stack flags Q < 1 explicitly — keep it that way.
//!
//! ## Architecture (hypothetical target)
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    SPARK ENGINE v1.0                        │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                             │
//! │  ┌─────────────────────────────────────────────────────┐    │
//! │  │  OUTER SHELL: HEA (TiVZrNbPd)                       │    │
//! │  │  • D₂ storage + LCF trigger zone                    │    │
//! │  │  • Self-healing via sluggish diffusion              │    │
//! │  ├─────────────────────────────────────────────────────┤    │
//! │  │  INTERFACE: Nano-laminate (Pd/Nb)                   │    │
//! │  │  • Defect sink layer                                │    │
//! │  │  • H-permeable barrier                              │    │
//! │  ├─────────────────────────────────────────────────────┤    │
//! │  │  INNER CORE: Liquid Metal (Galinstan)               │    │
//! │  │  • Infinite self-healing                            │    │
//! │  │  • Heat transfer medium                             │    │
//! │  └─────────────────────────────────────────────────────┘    │
//! │                                                             │
//! │  TRIGGER: X-ray @ 0.1nm, femtosecond pulse, 300K            │
//! │  FUEL: D₂ from seawater ($0.01/g)                           │
//! │  OUTPUT: ~80,000 GJ/g (fusion scale)                        │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! let spec = SparkEngineSpec::design(&genesis, SparkTarget::default());
//! println!("{}", spec.summary());
//! ```

use super::{
    AdvancedMaterials, FusionReaction, HEADesigner, HEATarget, Hadrons, HighEntropyAlloy,
    PeriodicTable, PhononDynamics, RadiationDamageSystem, StandardModel, TriggerMethod,
    TriggerProtocol,
};
use crate::genesis::GenesisSeed;

/// Target specification for Spark Engine design
#[derive(Debug, Clone)]
pub struct SparkTarget {
    /// Target power output in kW
    pub power_kw: f64,
    /// Target lifetime in years
    pub lifetime_years: f64,
    /// Cost constraint (0 = budget, 1 = unlimited)
    pub cost_tolerance: f32,
    /// Size constraint (0 = compact, 1 = industrial)
    pub size_class: f32,
    /// Safety requirement (0 = laboratory, 1 = consumer)
    pub safety_level: f32,
}

impl Default for SparkTarget {
    fn default() -> Self {
        Self {
            power_kw: 10.0,       // 10 kW residential
            lifetime_years: 20.0, // 20 year lifespan
            cost_tolerance: 0.5,  // Mid-range budget
            size_class: 0.3,      // Compact
            safety_level: 0.9,    // Near consumer-grade
        }
    }
}

impl SparkTarget {
    /// Industrial-scale power plant
    pub fn industrial() -> Self {
        Self {
            power_kw: 100_000.0, // 100 MW
            lifetime_years: 40.0,
            cost_tolerance: 0.8,
            size_class: 1.0,
            safety_level: 0.7,
        }
    }

    /// Compact consumer device
    pub fn consumer() -> Self {
        Self {
            power_kw: 5.0, // 5 kW home
            lifetime_years: 25.0,
            cost_tolerance: 0.3,
            size_class: 0.1,
            safety_level: 1.0,
        }
    }

    /// Research prototype
    pub fn prototype() -> Self {
        Self {
            power_kw: 1.0,
            lifetime_years: 1.0,
            cost_tolerance: 1.0,
            size_class: 0.5,
            safety_level: 0.5,
        }
    }
}

/// Shell material specification
#[derive(Debug, Clone)]
pub struct ShellSpec {
    /// Material name
    pub name: String,
    /// Material type
    pub material_type: ShellMaterial,
    /// Composition (for HEA)
    pub composition: String,
    /// H-solubility (0-1)
    pub h_solubility: f32,
    /// Self-healing score (0-1)
    pub healing_score: f32,
    /// Lifetime extension factor
    pub lifetime_factor: f64,
    /// Recommended thickness (mm)
    pub thickness_mm: f64,
}

/// Shell material types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellMaterial {
    HighEntropyAlloy,
    MAXPhase,
    NanoLaminate,
    Hybrid,
}

/// Interface layer specification
#[derive(Debug, Clone)]
pub struct InterfaceSpec {
    /// Material name
    pub name: String,
    /// Layer type
    pub layer_type: InterfaceType,
    /// Thickness (nm)
    pub thickness_nm: f64,
    /// Defect sink efficiency
    pub sink_efficiency: f32,
    /// H-permeability
    pub h_permeable: bool,
}

/// Interface types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterfaceType {
    NanoLaminate,
    MAXPhase,
    OxideBarrier,
    None,
}

/// Core specification
#[derive(Debug, Clone)]
pub struct CoreSpec {
    /// Material name
    pub name: String,
    /// Core type
    pub core_type: CoreType,
    /// Operating temperature (K)
    pub operating_temp_k: f64,
    /// Heat capacity (J/kg·K)
    pub heat_capacity: f64,
    /// Self-healing (always 1.0 for liquid)
    pub healing_score: f32,
}

/// Core types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreType {
    LiquidMetal,
    MoltenSalt,
    HeatPipe,
    Solid,
}

/// Complete Spark Engine specification
#[derive(Debug, Clone)]
pub struct SparkEngineSpec {
    /// Engine name/version
    pub name: String,
    /// Fusion reaction type
    pub reaction: FusionReaction,
    /// Shell specification
    pub shell: ShellSpec,
    /// Interface specification
    pub interface: InterfaceSpec,
    /// Core specification
    pub core: CoreSpec,
    /// Trigger protocol
    pub trigger: TriggerProtocol,
    /// Target power output (kW)
    pub power_kw: f64,
    /// Fuel consumption rate (g/year)
    pub fuel_rate_g_year: f64,
    /// Estimated lifetime (years)
    pub lifetime_years: f64,
    /// Energy density (GJ/g fuel)
    pub energy_density_gj_g: f64,
    /// Overall confidence score (0-1)
    pub confidence: f32,
    /// Design notes
    pub notes: Vec<String>,
}

impl SparkEngineSpec {
    /// Design optimal Spark Engine for given target
    pub fn design(genesis: &GenesisSeed, target: SparkTarget) -> Self {
        // Initialize all physics layers
        let model = StandardModel::from_genesis(genesis);
        let hadrons = Hadrons::from_model(&model, genesis);
        let table = PeriodicTable::from_model(&model, &hadrons, genesis);
        let _phonons = PhononDynamics::from_genesis(genesis, &table);
        let radiation = RadiationDamageSystem::from_genesis(genesis);
        let hea_designer = HEADesigner::from_genesis(genesis);
        let advanced = AdvancedMaterials::from_genesis(genesis);

        // Determine fusion reaction
        let reaction = if target.safety_level > 0.8 {
            FusionReaction::DD // Lower neutron energy, safer
        } else {
            FusionReaction::DT // Higher energy density
        };

        let neutron_energy = reaction.neutron_energy_mev().unwrap_or(2.45);
        let damage = radiation.calculate_damage(neutron_energy, 1e14, 22);

        // Select shell material
        let hea_target = HEATarget {
            h_solubility: 0.9,
            neutron_tolerance: 0.9,
            self_healing: 0.95,
            phonon_coupling: 0.8,
            cost_tolerance: target.cost_tolerance,
        };

        let hea_results = hea_designer.search(&hea_target, &radiation, reaction);
        let best_hea = hea_results.first();

        let shell = if let Some(hea) = best_hea {
            if hea.alloy.name.contains("Galinstan") {
                // Skip liquid for shell, use second best
                let solid_hea = hea_results
                    .iter()
                    .find(|r| !r.alloy.name.contains("Galinstan"))
                    .unwrap_or(hea);
                Self::make_shell_spec(&solid_hea.alloy, &solid_hea.healing, target.lifetime_years)
            } else {
                Self::make_shell_spec(&hea.alloy, &hea.healing, target.lifetime_years)
            }
        } else {
            // Fallback to MAX phase
            let (best_max, _) = advanced
                .best_max_phase(&damage)
                .expect("spark_engine: no MAX phase material in advanced_materials catalog");
            ShellSpec {
                name: best_max.name.clone(),
                material_type: ShellMaterial::MAXPhase,
                composition: format!(
                    "M{}A{}X{}",
                    best_max.m_element, best_max.a_element, best_max.x_element
                ),
                h_solubility: 0.3, // MAX phases have limited H storage
                healing_score: best_max.healing_score,
                lifetime_factor: 5.0,
                thickness_mm: 10.0,
            }
        };

        // Select interface layer
        let (best_laminate, lam_analysis) = advanced
            .best_nano_laminate(&damage)
            .expect("spark_engine: no nano-laminate material in advanced_materials catalog");

        let interface = if lam_analysis.h_compatible {
            InterfaceSpec {
                name: best_laminate.name.clone(),
                layer_type: InterfaceType::NanoLaminate,
                thickness_nm: best_laminate.spacing_nm * 100.0, // ~100 layers
                sink_efficiency: best_laminate.sink_efficiency as f32,
                h_permeable: true,
            }
        } else {
            InterfaceSpec {
                name: "None".to_string(),
                layer_type: InterfaceType::None,
                thickness_nm: 0.0,
                sink_efficiency: 0.0,
                h_permeable: false,
            }
        };

        // Core is always liquid metal for self-healing
        let core = CoreSpec {
            name: "Galinstan (GaInSn)".to_string(),
            core_type: CoreType::LiquidMetal,
            operating_temp_k: 350.0, // Slightly above room temp
            heat_capacity: 296.0,    // J/kg·K for Galinstan
            healing_score: 1.0,
        };

        // Trigger protocol
        let trigger = TriggerProtocol {
            method: TriggerMethod::XRay,
            wavelength_nm: Some(0.1),
            pulse_duration: Some("femtosecond".to_string()),
            temperature_k: Some(300.0),
            notes: "Phonon-assisted bremsstrahlung trigger".to_string(),
        };

        // Calculate performance metrics
        // Energy density: E_mev × J/MeV × N_A/mol / (fuel_mass_g_per_mol × 1e9 J/GJ)
        let fuel_mass_g_mol = match reaction {
            FusionReaction::DD => 4.028,   // 2 deuterium atoms
            FusionReaction::DT => 5.030,   // deuterium + tritium
            FusionReaction::DHe3 => 5.008, // deuterium + He-3
            _ => 5.0,                      // conservative default
        };
        let energy_density =
            reaction.total_energy_mev() * super::constants::MEV_TO_J * super::constants::N_AVOGADRO
                / (fuel_mass_g_mol * 1e9); // GJ/g
        let fuel_rate = target.power_kw * 3.15e7 / (energy_density * 1e9); // g/year

        let lifetime = target.lifetime_years * shell.lifetime_factor.min(100.0) / 10.0;

        // Confidence based on component scores
        let confidence = (shell.healing_score * 0.4
            + interface.sink_efficiency * 0.2
            + core.healing_score * 0.3
            + 0.1) // Base confidence
            .min(0.95);

        // Notes
        let mut notes = vec![
            format!(
                "Reaction: {:?} ({:.1} MeV neutrons)",
                reaction, neutron_energy
            ),
            format!("Shell healing: {:.0}%", shell.healing_score * 100.0),
        ];

        if interface.layer_type != InterfaceType::None {
            notes.push(format!(
                "Interface: {} (sink eff: {:.0}%)",
                interface.name,
                interface.sink_efficiency * 100.0
            ));
        }

        if target.safety_level > 0.8 {
            notes.push("Safety: Consumer-grade (D-D reaction, lower neutron flux)".to_string());
        }

        Self {
            name: format!("SPARK v1.0 ({:.0}kW)", target.power_kw),
            reaction,
            shell,
            interface,
            core,
            trigger,
            power_kw: target.power_kw,
            fuel_rate_g_year: fuel_rate,
            lifetime_years: lifetime.min(target.lifetime_years),
            energy_density_gj_g: energy_density,
            confidence,
            notes,
        }
    }

    fn make_shell_spec(
        alloy: &HighEntropyAlloy,
        healing: &super::HealingAnalysis,
        target_lifetime: f64,
    ) -> ShellSpec {
        let composition = if alloy.elements.is_empty() {
            alloy.name.clone()
        } else {
            alloy
                .elements
                .iter()
                .map(|e| format!("{}{:.0}", e.symbol, e.fraction * 100.0))
                .collect::<Vec<_>>()
                .join("-")
        };

        // Thickness scales with target lifetime
        let thickness = 5.0 + target_lifetime / 5.0;

        ShellSpec {
            name: alloy.name.clone(),
            material_type: ShellMaterial::HighEntropyAlloy,
            composition,
            h_solubility: alloy.h_solubility as f32,
            healing_score: healing.healing_score,
            lifetime_factor: healing.lifetime_factor,
            thickness_mm: thickness,
        }
    }

    /// Generate human-readable summary
    pub fn summary(&self) -> String {
        let mut s = String::new();

        s.push_str(&format!("\n{}\n", "═".repeat(60)));
        s.push_str(&format!("  {}\n", self.name));
        s.push_str(&format!("{}\n\n", "═".repeat(60)));

        s.push_str("  ARCHITECTURE:\n\n");
        s.push_str(&format!(
            "    Shell:     {} ({:?})\n",
            self.shell.name, self.shell.material_type
        ));
        s.push_str(&format!(
            "               Composition: {}\n",
            self.shell.composition
        ));
        s.push_str(&format!(
            "               H-Solubility: {:.0}%\n",
            self.shell.h_solubility * 100.0
        ));
        s.push_str(&format!(
            "               Thickness: {:.1} mm\n\n",
            self.shell.thickness_mm
        ));

        if self.interface.layer_type != InterfaceType::None {
            s.push_str(&format!(
                "    Interface: {} ({:?})\n",
                self.interface.name, self.interface.layer_type
            ));
            s.push_str(&format!(
                "               Thickness: {:.0} nm\n",
                self.interface.thickness_nm
            ));
            s.push_str(&format!(
                "               Sink Eff: {:.0}%\n\n",
                self.interface.sink_efficiency * 100.0
            ));
        }

        s.push_str(&format!(
            "    Core:      {} ({:?})\n",
            self.core.name, self.core.core_type
        ));
        s.push_str(&format!(
            "               Temp: {:.0} K\n\n",
            self.core.operating_temp_k
        ));

        s.push_str("  TRIGGER:\n\n");
        s.push_str(&format!("    Method:    {:?}\n", self.trigger.method));
        if let Some(wl) = self.trigger.wavelength_nm {
            s.push_str(&format!("    Wavelength: {wl:.2} nm\n"));
        }
        if let Some(ref pulse) = self.trigger.pulse_duration {
            s.push_str(&format!("    Pulse:     {pulse}\n"));
        }
        s.push('\n');

        s.push_str("  PERFORMANCE:\n\n");
        s.push_str(&format!("    Power:     {:.1} kW\n", self.power_kw));
        s.push_str(&format!(
            "    Lifetime:  {:.1} years\n",
            self.lifetime_years
        ));
        s.push_str(&format!(
            "    Fuel Rate: {:.4} g/year\n",
            self.fuel_rate_g_year
        ));
        s.push_str(&format!(
            "    Energy:    {:.0} GJ/g\n\n",
            self.energy_density_gj_g
        ));

        s.push_str(&format!(
            "  CONFIDENCE: {:.0}%\n\n",
            self.confidence * 100.0
        ));

        s.push_str("  NOTES:\n");
        for note in &self.notes {
            s.push_str(&format!("    • {note}\n"));
        }

        s.push_str(&format!("\n{}\n", "═".repeat(60)));

        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spark_design() {
        let genesis = GenesisSeed::from_phrase("spark engine test");
        let target = SparkTarget::default();

        let spec = SparkEngineSpec::design(&genesis, target);

        assert!(spec.power_kw > 0.0);
        assert!(spec.lifetime_years > 0.0);
        assert!(spec.confidence > 0.0);
    }

    #[test]
    fn test_industrial_design() {
        let genesis = GenesisSeed::from_phrase("industrial spark");
        let target = SparkTarget::industrial();

        let spec = SparkEngineSpec::design(&genesis, target);

        assert!(spec.power_kw >= 100_000.0);
        assert_eq!(spec.reaction, FusionReaction::DT); // Industrial uses D-T
    }

    #[test]
    fn test_consumer_design() {
        let genesis = GenesisSeed::from_phrase("consumer spark");
        let target = SparkTarget::consumer();

        let spec = SparkEngineSpec::design(&genesis, target);

        assert!(spec.power_kw <= 10.0);
        assert_eq!(spec.reaction, FusionReaction::DD); // Consumer uses safer D-D
    }

    #[test]
    fn test_summary_generation() {
        let genesis = GenesisSeed::from_phrase("summary test");
        let target = SparkTarget::default();

        let spec = SparkEngineSpec::design(&genesis, target);
        let summary = spec.summary();

        assert!(summary.contains("SPARK"));
        assert!(summary.contains("Shell"));
        assert!(summary.contains("Core"));
        assert!(summary.contains("TRIGGER"));
    }
}
