// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Alternative Trigger Systems for Lattice Confinement Fusion
//!
//! The original Spark Engine design calls for 0.1nm femtosecond X-rays,
//! which requires synchrotron/XFEL-class equipment. This module explores
//! practical alternatives that could enable desktop-scale LCF reactors.
//!
//! ## The Trigger Problem
//!
//! LCF requires:
//! 1. Loading deuterium into a metal lattice (Pd, Ti, Ni, etc.)
//! 2. Creating transient high-density D-D proximity
//! 3. Triggering fusion via lattice phonons + external excitation
//!
//! The challenge: How to provide enough localized energy to overcome
//! the Coulomb barrier without massive equipment?
//!
//! ## Alternative Approaches
//!
//! ```text
//! Approach              Equipment Cost    TRL    Trigger Rate    Notes
//! ─────────────────────────────────────────────────────────────────────
//! XFEL X-ray            $1B+              9      Hz-kHz          Reference
//! Electron Beam         $10K-100K         7      MHz             Compact
//! Laser Bremsstrahlung  $50K-500K         5      kHz             Tunable
//! NEEC Cascade          $100K-1M          3      Hz-kHz          Novel
//! Piezo Phonon          $1K-10K           6      MHz             Simplest
//! Pulsed Electrolysis   $1K-50K           7      Hz              Proven
//! Muon Catalysis        $10M+             4      Low             Exotic
//! ```

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Extended trigger method taxonomy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExtendedTriggerMethod {
    // === Electromagnetic ===
    /// Synchrotron/XFEL hard X-ray (reference, impractical)
    XfelXRay,
    /// Compact X-ray tube with Bremsstrahlung spectrum
    CompactXRay,
    /// Laser-driven inverse Compton X-ray source
    LaserCompton,
    /// Laser-driven Bremsstrahlung (target conversion)
    LaserBremsstrahlung,
    /// Direct UV/VUV laser (for NEEC-mediated)
    DirectLaser,

    // === Particle Beam ===
    /// Electron beam direct irradiation
    ElectronBeam,
    /// Proton beam (for recoil-induced fusion)
    ProtonBeam,
    /// Deuteron beam (beam-target fusion hybrid)
    DeuteronBeam,

    // === Nuclear/Exotic ===
    /// NEEC cascade through Th-229m or similar isomer
    NeecCascade,
    /// Muon-catalyzed fusion (μCF)
    MuonCatalysis,
    /// Pyroelectric crystal neutron source
    PyroelectricNeutron,

    // === Mechanical/Acoustic ===
    /// Piezoelectric phonon injection
    PiezoPhonon,
    /// Ultrasonic cavitation (sonofusion-inspired)
    Ultrasonic,
    /// Shock compression (dynamic loading)
    ShockCompression,

    // === Electrochemical ===
    /// Pulsed electrolysis (Fleischmann-Pons style, refined)
    PulsedElectrolysis,
    /// Glow discharge plasma electrolysis
    GlowDischarge,
    /// Supercritical water electrolysis
    SupercriticalElectrolysis,

    // === Thermal ===
    /// Rapid thermal cycling
    ThermalCycling,
    /// Laser ablation heating
    LaserAblation,
}

/// Technology Readiness Level (NASA scale 1-9)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[allow(clippy::upper_case_acronyms)]
pub struct TRL(pub u8);

impl TRL {
    pub fn basic_principles() -> Self {
        TRL(1)
    }
    pub fn technology_concept() -> Self {
        TRL(2)
    }
    pub fn proof_of_concept() -> Self {
        TRL(3)
    }
    pub fn lab_validation() -> Self {
        TRL(4)
    }
    pub fn lab_demonstration() -> Self {
        TRL(5)
    }
    pub fn prototype_demo() -> Self {
        TRL(6)
    }
    pub fn system_demo() -> Self {
        TRL(7)
    }
    pub fn qualified() -> Self {
        TRL(8)
    }
    pub fn proven() -> Self {
        TRL(9)
    }

    pub fn description(&self) -> &'static str {
        match self.0 {
            1 => "Basic principles observed",
            2 => "Technology concept formulated",
            3 => "Proof of concept",
            4 => "Lab validation",
            5 => "Lab-scale demonstration",
            6 => "Prototype demonstration",
            7 => "System demonstration",
            8 => "System qualified",
            9 => "Flight/operation proven",
            _ => "Unknown",
        }
    }
}

/// Equipment cost category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CostCategory {
    /// < $10K - Hobbyist/maker accessible
    Maker,
    /// $10K - $100K - University lab scale
    LabScale,
    /// $100K - $1M - Research facility
    Research,
    /// $1M - $100M - Major facility
    MajorFacility,
    /// > $100M - National/international facility
    NationalFacility,
}

impl CostCategory {
    pub fn range_usd(&self) -> (f64, f64) {
        match self {
            CostCategory::Maker => (1_000.0, 10_000.0),
            CostCategory::LabScale => (10_000.0, 100_000.0),
            CostCategory::Research => (100_000.0, 1_000_000.0),
            CostCategory::MajorFacility => (1_000_000.0, 100_000_000.0),
            CostCategory::NationalFacility => (100_000_000.0, 10_000_000_000.0),
        }
    }

    pub fn median_usd(&self) -> f64 {
        let (low, high) = self.range_usd();
        (low * high).sqrt() // Geometric mean
    }
}

/// Complete trigger system specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerSystemSpec {
    /// Trigger method
    pub method: ExtendedTriggerMethod,
    /// Human-readable name
    pub name: String,
    /// Detailed description
    pub description: String,
    /// Technology readiness level
    pub trl: TRL,
    /// Equipment cost category
    pub cost: CostCategory,
    /// Estimated cost in USD
    pub cost_estimate_usd: f64,
    /// Maximum trigger repetition rate (Hz)
    pub max_rep_rate_hz: f64,
    /// Energy delivered per trigger (J)
    pub energy_per_trigger_j: f64,
    /// Electrical-to-trigger efficiency
    pub wall_plug_efficiency: f64,
    /// Physical footprint (m³)
    pub footprint_m3: f64,
    /// Key advantages
    pub advantages: Vec<String>,
    /// Key challenges/limitations
    pub challenges: Vec<String>,
    /// Physics model for trigger effectiveness
    pub physics: TriggerPhysics,
}

/// Physics model for trigger effectiveness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerPhysics {
    /// Primary energy deposition mechanism
    pub mechanism: EnergyDepositionMechanism,
    /// Characteristic penetration depth in Pd (μm)
    pub penetration_depth_um: f64,
    /// Energy deposition profile (exponential decay constant)
    pub deposition_profile_um: f64,
    /// Phonon generation efficiency (0-1)
    pub phonon_efficiency: f64,
    /// Estimated screening enhancement factor
    pub screening_factor: f64,
    /// Theoretical trigger-to-fusion conversion efficiency
    pub conversion_efficiency: f64,
}

/// How energy is deposited into the lattice
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnergyDepositionMechanism {
    /// Photoelectric absorption (X-ray)
    Photoelectric,
    /// Compton scattering (hard X-ray/gamma)
    Compton,
    /// Electronic stopping (charged particles)
    ElectronicStopping,
    /// Nuclear stopping (heavy ions)
    NuclearStopping,
    /// Direct phonon injection (acoustic)
    PhononInjection,
    /// Electrochemical (ion transport)
    Electrochemical,
    /// Thermal conduction
    Thermal,
    /// Nuclear recoil (neutrons, muons)
    NuclearRecoil,
}

/// Trigger system library with all alternatives
pub struct TriggerSystemLibrary {
    pub systems: Vec<TriggerSystemSpec>,
}

impl TriggerSystemLibrary {
    /// Initialize library with all known trigger systems
    #[allow(clippy::vec_init_then_push)]
    pub fn new() -> Self {
        let mut systems = Vec::new();

        // === Reference: XFEL X-ray ===
        systems.push(TriggerSystemSpec {
            method: ExtendedTriggerMethod::XfelXRay,
            name: "XFEL Hard X-ray".to_string(),
            description: "Free-electron laser producing coherent, femtosecond X-ray pulses. \
                         Current reference design but requires billion-dollar facilities."
                .to_string(),
            trl: TRL::proven(),
            cost: CostCategory::NationalFacility,
            cost_estimate_usd: 1_000_000_000.0,
            max_rep_rate_hz: 27_000.0,   // European XFEL
            energy_per_trigger_j: 0.001, // ~1 mJ per pulse
            wall_plug_efficiency: 1e-6,  // Extremely inefficient
            footprint_m3: 1_000_000.0,   // 3.4 km tunnel
            advantages: vec![
                "Highest peak brilliance".to_string(),
                "Tunable wavelength".to_string(),
                "Proven technology".to_string(),
                "Femtosecond time resolution".to_string(),
            ],
            challenges: vec![
                "Cost: >$1B".to_string(),
                "Size: kilometers".to_string(),
                "Not portable".to_string(),
                "Limited access".to_string(),
            ],
            physics: TriggerPhysics {
                mechanism: EnergyDepositionMechanism::Photoelectric,
                penetration_depth_um: 10.0, // ~10 keV X-rays in Pd
                deposition_profile_um: 5.0,
                phonon_efficiency: 0.3,      // Good coupling to lattice
                screening_factor: 50.0,      // High due to coherent excitation
                conversion_efficiency: 1e-6, // Theoretical
            },
        });

        // === Electron Beam ===
        systems.push(TriggerSystemSpec {
            method: ExtendedTriggerMethod::ElectronBeam,
            name: "Compact Electron Beam".to_string(),
            description: "Desktop-scale electron gun (10-100 keV) directly irradiating \
                         deuterated target. Electrons deposit energy via electronic stopping, \
                         generating phonons and local heating."
                .to_string(),
            trl: TRL::system_demo(),
            cost: CostCategory::LabScale,
            cost_estimate_usd: 50_000.0,
            max_rep_rate_hz: 1_000_000.0, // MHz pulsing possible
            energy_per_trigger_j: 1e-6,   // μJ per pulse
            wall_plug_efficiency: 0.3,    // Reasonable efficiency
            footprint_m3: 0.1,            // Desktop
            advantages: vec![
                "Compact and affordable".to_string(),
                "High rep rate (MHz)".to_string(),
                "Direct energy deposition".to_string(),
                "Well-understood technology".to_string(),
                "Tunable energy (1-100 keV)".to_string(),
            ],
            challenges: vec![
                "Limited penetration depth".to_string(),
                "Vacuum required".to_string(),
                "Surface damage over time".to_string(),
                "Bremsstrahlung X-ray generation".to_string(),
            ],
            physics: TriggerPhysics {
                mechanism: EnergyDepositionMechanism::ElectronicStopping,
                penetration_depth_um: 5.0, // 50 keV electrons in Pd
                deposition_profile_um: 2.0,
                phonon_efficiency: 0.5, // Good phonon generation
                screening_factor: 20.0, // Moderate screening enhancement
                conversion_efficiency: 1e-8,
            },
        });

        // === Laser Bremsstrahlung ===
        systems.push(TriggerSystemSpec {
            method: ExtendedTriggerMethod::LaserBremsstrahlung,
            name: "Laser-Driven Bremsstrahlung X-ray".to_string(),
            description: "High-intensity laser (>10^18 W/cm²) focused on high-Z target \
                         generates hot electrons that produce Bremsstrahlung X-rays. \
                         Compact alternative to synchrotrons."
                .to_string(),
            trl: TRL::lab_demonstration(),
            cost: CostCategory::Research,
            cost_estimate_usd: 300_000.0,
            max_rep_rate_hz: 1000.0,    // kHz lasers available
            energy_per_trigger_j: 0.01, // 10 mJ X-ray yield
            wall_plug_efficiency: 1e-4, // Low conversion
            footprint_m3: 5.0,          // Optical table + laser
            advantages: vec![
                "Desktop X-ray source".to_string(),
                "Tunable spectrum".to_string(),
                "Ultra-short pulses possible".to_string(),
                "No accelerator needed".to_string(),
            ],
            challenges: vec![
                "Low conversion efficiency".to_string(),
                "Target debris".to_string(),
                "Laser maintenance".to_string(),
                "Broad spectrum (not monochromatic)".to_string(),
            ],
            physics: TriggerPhysics {
                mechanism: EnergyDepositionMechanism::Photoelectric,
                penetration_depth_um: 20.0, // Broad spectrum
                deposition_profile_um: 10.0,
                phonon_efficiency: 0.2,
                screening_factor: 30.0,
                conversion_efficiency: 1e-7,
            },
        });

        // === NEEC Cascade ===
        systems.push(TriggerSystemSpec {
            method: ExtendedTriggerMethod::NeecCascade,
            name: "NEEC Cascade via Th-229m".to_string(),
            description: "Uses Thorium-229m's uniquely low nuclear transition (8.28 eV, VUV) \
                         as an intermediary. VUV laser excites Th-229m, which then transfers \
                         energy to nearby D nuclei through lattice phonons. Requires Th-doped \
                         deuteride target."
                .to_string(),
            trl: TRL::proof_of_concept(),
            cost: CostCategory::Research,
            cost_estimate_usd: 500_000.0,
            max_rep_rate_hz: 10_000.0,  // Limited by Th-229m lifetime
            energy_per_trigger_j: 1e-9, // nJ photon, amplified by nuclear
            wall_plug_efficiency: 0.01, // Potentially high
            footprint_m3: 1.0,          // VUV laser system
            advantages: vec![
                "Nuclear-scale energy from optical photon".to_string(),
                "Highly selective excitation".to_string(),
                "Potential for very high efficiency".to_string(),
                "Novel physics pathway".to_string(),
            ],
            challenges: vec![
                "Th-229 availability/cost".to_string(),
                "VUV laser technology immature".to_string(),
                "Nuclear-lattice coupling unproven".to_string(),
                "Regulatory hurdles (thorium)".to_string(),
            ],
            physics: TriggerPhysics {
                mechanism: EnergyDepositionMechanism::NuclearRecoil,
                penetration_depth_um: 1000.0, // VUV absorbed at surface, energy propagates
                deposition_profile_um: 100.0,
                phonon_efficiency: 0.8,      // High if coupling works
                screening_factor: 100.0,     // Potentially very high
                conversion_efficiency: 1e-5, // Speculative but promising
            },
        });

        // === Piezoelectric Phonon Injection ===
        systems.push(TriggerSystemSpec {
            method: ExtendedTriggerMethod::PiezoPhonon,
            name: "Piezoelectric Phonon Injection".to_string(),
            description: "High-frequency piezoelectric transducers (MHz-GHz) mechanically \
                         coupled to deuteride lattice. Generates coherent phonon waves that \
                         create transient D-D proximity enhancement."
                .to_string(),
            trl: TRL::prototype_demo(),
            cost: CostCategory::Maker,
            cost_estimate_usd: 5_000.0,
            max_rep_rate_hz: 1_000_000_000.0, // GHz
            energy_per_trigger_j: 1e-9,       // nJ acoustic energy
            wall_plug_efficiency: 0.5,        // Piezo efficiency
            footprint_m3: 0.001,              // Tiny
            advantages: vec![
                "Extremely simple and cheap".to_string(),
                "No vacuum required".to_string(),
                "GHz repetition rates".to_string(),
                "Direct phonon generation".to_string(),
                "Scalable array designs".to_string(),
            ],
            challenges: vec![
                "Low energy per pulse".to_string(),
                "Acoustic impedance matching".to_string(),
                "Heating at high power".to_string(),
                "Unknown fusion coupling".to_string(),
            ],
            physics: TriggerPhysics {
                mechanism: EnergyDepositionMechanism::PhononInjection,
                penetration_depth_um: 1000.0, // Acoustic waves propagate
                deposition_profile_um: 500.0,
                phonon_efficiency: 1.0,       // Direct phonon generation
                screening_factor: 5.0,        // Lower than ionizing radiation
                conversion_efficiency: 1e-10, // Low but compensated by rate
            },
        });

        // === Pulsed Electrolysis ===
        systems.push(TriggerSystemSpec {
            method: ExtendedTriggerMethod::PulsedElectrolysis,
            name: "Pulsed Electrolysis Loading".to_string(),
            description: "Refined Fleischmann-Pons approach: Pulsed electrolysis in D2O \
                         creates dynamic deuterium loading/unloading cycles in Pd cathode. \
                         Pulsing creates transient high-loading states with enhanced screening."
                .to_string(),
            trl: TRL::system_demo(),
            cost: CostCategory::Maker,
            cost_estimate_usd: 2_000.0,
            max_rep_rate_hz: 100.0,     // Limited by electrochemistry
            energy_per_trigger_j: 0.1,  // 100 mJ electrical
            wall_plug_efficiency: 0.01, // Most goes to electrolysis
            footprint_m3: 0.01,         // Benchtop
            advantages: vec![
                "Simplest possible setup".to_string(),
                "Self-loading of deuterium".to_string(),
                "Proven to produce excess heat (disputed)".to_string(),
                "No exotic materials".to_string(),
            ],
            challenges: vec![
                "Controversial physics".to_string(),
                "Poor reproducibility historically".to_string(),
                "Slow trigger rate".to_string(),
                "Material degradation".to_string(),
            ],
            physics: TriggerPhysics {
                mechanism: EnergyDepositionMechanism::Electrochemical,
                penetration_depth_um: 100.0, // D loading depth
                deposition_profile_um: 50.0,
                phonon_efficiency: 0.1, // Indirect
                screening_factor: 10.0, // Loading-dependent
                conversion_efficiency: 1e-9,
            },
        });

        // === Ultrasonic Cavitation ===
        systems.push(TriggerSystemSpec {
            method: ExtendedTriggerMethod::Ultrasonic,
            name: "Ultrasonic Cavitation".to_string(),
            description: "High-power ultrasound in deuterated liquid (D2O or organic) creates \
                         collapsing cavitation bubbles with extreme transient temperatures and \
                         pressures. Sonofusion-inspired but with solid target immersed."
                .to_string(),
            trl: TRL::lab_validation(),
            cost: CostCategory::LabScale,
            cost_estimate_usd: 20_000.0,
            max_rep_rate_hz: 100_000.0, // Acoustic frequency
            energy_per_trigger_j: 1e-6, // Per bubble collapse
            wall_plug_efficiency: 0.1,
            footprint_m3: 0.1,
            advantages: vec![
                "Creates extreme local conditions".to_string(),
                "Self-focusing energy".to_string(),
                "Continuous operation possible".to_string(),
                "No vacuum needed".to_string(),
            ],
            challenges: vec![
                "Sonofusion claims disputed".to_string(),
                "Difficult to control/measure".to_string(),
                "Erosion of target".to_string(),
                "Complex fluid dynamics".to_string(),
            ],
            physics: TriggerPhysics {
                mechanism: EnergyDepositionMechanism::PhononInjection,
                penetration_depth_um: 10.0, // Bubble collapse is localized
                deposition_profile_um: 5.0,
                phonon_efficiency: 0.3,
                screening_factor: 15.0, // High-pressure screening
                conversion_efficiency: 1e-9,
            },
        });

        // === Glow Discharge ===
        systems.push(TriggerSystemSpec {
            method: ExtendedTriggerMethod::GlowDischarge,
            name: "Glow Discharge Plasma".to_string(),
            description: "Low-pressure D2 gas plasma with Pd cathode. Energetic D+ ions \
                         (100 eV - 1 keV) implant into cathode while cathode heating drives \
                         loading. Combines ion implantation with thermal effects."
                .to_string(),
            trl: TRL::prototype_demo(),
            cost: CostCategory::LabScale,
            cost_estimate_usd: 15_000.0,
            max_rep_rate_hz: 1000.0, // Pulsed discharge
            energy_per_trigger_j: 0.01,
            wall_plug_efficiency: 0.05,
            footprint_m3: 0.1,
            advantages: vec![
                "Ion implantation + loading".to_string(),
                "Creates defects (D trapping sites)".to_string(),
                "Continuous or pulsed operation".to_string(),
                "Observable excess heat reports".to_string(),
            ],
            challenges: vec![
                "Cathode sputtering".to_string(),
                "Plasma instabilities".to_string(),
                "Contamination issues".to_string(),
                "Limited ion energy".to_string(),
            ],
            physics: TriggerPhysics {
                mechanism: EnergyDepositionMechanism::NuclearStopping,
                penetration_depth_um: 0.1, // Shallow implantation
                deposition_profile_um: 0.05,
                phonon_efficiency: 0.4,
                screening_factor: 25.0, // Ion channeling effects
                conversion_efficiency: 1e-8,
            },
        });

        // === Muon Catalysis (exotic) ===
        systems.push(TriggerSystemSpec {
            method: ExtendedTriggerMethod::MuonCatalysis,
            name: "Muon-Catalyzed Fusion Hybrid".to_string(),
            description: "Muons replace electrons in D-D molecules, reducing internuclear \
                         distance by factor of 207. Enables fusion at room temperature. \
                         Limited by muon sticking and production cost."
                .to_string(),
            trl: TRL::lab_validation(),
            cost: CostCategory::MajorFacility,
            cost_estimate_usd: 10_000_000.0,
            max_rep_rate_hz: 1.0,                   // Muon lifetime limited
            energy_per_trigger_j: 17.6e6 * 1.6e-19, // D-T fusion energy
            wall_plug_efficiency: 1e-6,             // Muon production expensive
            footprint_m3: 100.0,                    // Accelerator + target
            advantages: vec![
                "Proven fusion mechanism".to_string(),
                "Room temperature operation".to_string(),
                "~150 fusions per muon".to_string(),
                "Well-understood physics".to_string(),
            ],
            challenges: vec![
                "Muon production requires accelerator".to_string(),
                "Alpha sticking limits yield".to_string(),
                "Energy breakeven difficult".to_string(),
                "Short muon lifetime (2.2 μs)".to_string(),
            ],
            physics: TriggerPhysics {
                mechanism: EnergyDepositionMechanism::NuclearRecoil,
                penetration_depth_um: 1000.0, // Muons penetrate deeply
                deposition_profile_um: 500.0,
                phonon_efficiency: 0.1,     // Not phonon-mediated
                screening_factor: 207.0,    // Mass ratio screening
                conversion_efficiency: 1.0, // Near-certain fusion per capture
            },
        });

        // === Compact X-ray Tube ===
        systems.push(TriggerSystemSpec {
            method: ExtendedTriggerMethod::CompactXRay,
            name: "Compact X-ray Tube".to_string(),
            description: "Standard medical/industrial X-ray tube (30-150 kVp) producing \
                         Bremsstrahlung + characteristic X-rays. Not coherent or ultrafast \
                         but readily available and well-understood."
                .to_string(),
            trl: TRL::proven(),
            cost: CostCategory::LabScale,
            cost_estimate_usd: 30_000.0,
            max_rep_rate_hz: 1000.0,    // Pulsed tubes
            energy_per_trigger_j: 0.1,  // Integrated pulse energy
            wall_plug_efficiency: 0.01, // X-ray generation efficiency
            footprint_m3: 0.1,
            advantages: vec![
                "Proven, available technology".to_string(),
                "No R&D needed".to_string(),
                "Regulatory framework exists".to_string(),
                "Serviceable parts".to_string(),
            ],
            challenges: vec![
                "Not coherent".to_string(),
                "Broad spectrum".to_string(),
                "Limited peak intensity".to_string(),
                "Shielding required".to_string(),
            ],
            physics: TriggerPhysics {
                mechanism: EnergyDepositionMechanism::Photoelectric,
                penetration_depth_um: 100.0, // Depending on spectrum
                deposition_profile_um: 50.0,
                phonon_efficiency: 0.2,
                screening_factor: 15.0,
                conversion_efficiency: 1e-9,
            },
        });

        // === Thermal Cycling ===
        systems.push(TriggerSystemSpec {
            method: ExtendedTriggerMethod::ThermalCycling,
            name: "Rapid Thermal Cycling".to_string(),
            description: "Cyclic heating/cooling of deuterated metal creates expansion/ \
                         contraction that modulates D-D spacing and generates thermal phonons. \
                         Simplest possible trigger but low energy density."
                .to_string(),
            trl: TRL::system_demo(),
            cost: CostCategory::Maker,
            cost_estimate_usd: 1_000.0,
            max_rep_rate_hz: 10.0,     // Thermal mass limited
            energy_per_trigger_j: 1.0, // Heating energy
            wall_plug_efficiency: 0.9, // Resistive heating efficient
            footprint_m3: 0.01,
            advantages: vec![
                "Absolute simplest trigger".to_string(),
                "No special equipment".to_string(),
                "Easy to control".to_string(),
                "Inherent temperature monitoring".to_string(),
            ],
            challenges: vec![
                "Slow cycle time".to_string(),
                "Thermal stress on materials".to_string(),
                "Low peak excitation".to_string(),
                "Deuterium desorption at high T".to_string(),
            ],
            physics: TriggerPhysics {
                mechanism: EnergyDepositionMechanism::Thermal,
                penetration_depth_um: 10000.0, // Bulk heating
                deposition_profile_um: 5000.0,
                phonon_efficiency: 0.9, // Thermal = phonons
                screening_factor: 2.0,  // Minimal enhancement
                conversion_efficiency: 1e-12,
            },
        });

        Self { systems }
    }

    /// Find systems meeting specific criteria
    pub fn filter(&self, criteria: &TriggerCriteria) -> Vec<&TriggerSystemSpec> {
        self.systems
            .iter()
            .filter(|sys| {
                // TRL filter
                if let Some(min_trl) = criteria.min_trl
                    && sys.trl.0 < min_trl
                {
                    return false;
                }

                // Cost filter
                if let Some(max_cost) = criteria.max_cost_usd
                    && sys.cost_estimate_usd > max_cost
                {
                    return false;
                }

                // Rep rate filter
                if let Some(min_rate) = criteria.min_rep_rate_hz
                    && sys.max_rep_rate_hz < min_rate
                {
                    return false;
                }

                // Footprint filter
                if let Some(max_footprint) = criteria.max_footprint_m3
                    && sys.footprint_m3 > max_footprint
                {
                    return false;
                }

                true
            })
            .collect()
    }

    /// Rank systems by a weighted score
    pub fn rank(&self, weights: &TriggerWeights) -> Vec<(&TriggerSystemSpec, f64)> {
        let mut scored: Vec<_> = self
            .systems
            .iter()
            .map(|sys| {
                let score = self.compute_score(sys, weights);
                (sys, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    /// Compute weighted score for a system
    fn compute_score(&self, sys: &TriggerSystemSpec, weights: &TriggerWeights) -> f64 {
        // Normalize each factor to 0-1 scale

        // TRL: higher is better
        let trl_score = sys.trl.0 as f64 / 9.0;

        // Cost: lower is better (log scale)
        let cost_score = 1.0 - (sys.cost_estimate_usd.ln() / 25.0).clamp(0.0, 1.0);

        // Rep rate: higher is better (log scale)
        let rate_score = (sys.max_rep_rate_hz.ln() / 25.0).clamp(0.0, 1.0);

        // Efficiency: higher is better
        let efficiency_score = (sys.wall_plug_efficiency.ln() + 10.0) / 10.0;

        // Footprint: smaller is better (log scale)
        let footprint_score = 1.0 - (sys.footprint_m3.ln() + 5.0) / 20.0;

        // Conversion efficiency: higher is better (log scale)
        let conversion_score = (sys.physics.conversion_efficiency.log10() + 12.0) / 12.0;

        // Phonon efficiency: higher is better for LCF
        let phonon_score = sys.physics.phonon_efficiency;

        // Weighted sum
        weights.trl * trl_score
            + weights.cost * cost_score
            + weights.rep_rate * rate_score
            + weights.efficiency * efficiency_score
            + weights.footprint * footprint_score
            + weights.conversion * conversion_score
            + weights.phonon * phonon_score
    }

    /// Get optimal trigger for specific power scale
    pub fn optimal_for_power(&self, target_power_w: f64) -> &TriggerSystemSpec {
        // Different triggers optimal at different scales
        if target_power_w < 100.0 {
            // Sub-100W: piezo or thermal cycling
            self.systems
                .iter()
                .find(|s| s.method == ExtendedTriggerMethod::PiezoPhonon)
                .unwrap_or(&self.systems[0])
        } else if target_power_w < 10_000.0 {
            // 100W - 10kW: electron beam or glow discharge
            self.systems
                .iter()
                .find(|s| s.method == ExtendedTriggerMethod::ElectronBeam)
                .unwrap_or(&self.systems[0])
        } else if target_power_w < 1_000_000.0 {
            // 10kW - 1MW: laser bremsstrahlung or compact X-ray
            self.systems
                .iter()
                .find(|s| s.method == ExtendedTriggerMethod::LaserBremsstrahlung)
                .unwrap_or(&self.systems[0])
        } else {
            // >1MW: NEEC or muon catalysis for theoretical high gain
            self.systems
                .iter()
                .find(|s| s.method == ExtendedTriggerMethod::NeecCascade)
                .unwrap_or(&self.systems[0])
        }
    }
}

impl Default for TriggerSystemLibrary {
    fn default() -> Self {
        Self::new()
    }
}

/// Criteria for filtering trigger systems
#[derive(Debug, Clone, Default)]
pub struct TriggerCriteria {
    pub min_trl: Option<u8>,
    pub max_cost_usd: Option<f64>,
    pub min_rep_rate_hz: Option<f64>,
    pub max_footprint_m3: Option<f64>,
}

/// Weights for ranking trigger systems
#[derive(Debug, Clone)]
pub struct TriggerWeights {
    pub trl: f64,
    pub cost: f64,
    pub rep_rate: f64,
    pub efficiency: f64,
    pub footprint: f64,
    pub conversion: f64,
    pub phonon: f64,
}

impl Default for TriggerWeights {
    fn default() -> Self {
        Self {
            trl: 1.0,
            cost: 1.5, // Cost is important
            rep_rate: 0.8,
            efficiency: 1.0,
            footprint: 1.2,  // Compactness important
            conversion: 1.5, // Conversion critical
            phonon: 1.0,
        }
    }
}

impl TriggerWeights {
    /// Weights optimized for desktop/maker scale
    pub fn desktop() -> Self {
        Self {
            trl: 1.5,  // Want proven tech
            cost: 2.0, // Cost critical
            rep_rate: 0.5,
            efficiency: 0.8,
            footprint: 2.0, // Must be small
            conversion: 1.0,
            phonon: 1.2,
        }
    }

    /// Weights for research/prototype
    pub fn research() -> Self {
        Self {
            trl: 0.5, // OK with less proven
            cost: 0.8,
            rep_rate: 1.0,
            efficiency: 1.0,
            footprint: 0.5,
            conversion: 2.0, // Want best performance
            phonon: 1.5,
        }
    }

    /// Weights for industrial/power generation
    pub fn industrial() -> Self {
        Self {
            trl: 2.0, // Must be reliable
            cost: 1.0,
            rep_rate: 1.5,   // Throughput matters
            efficiency: 2.0, // Efficiency critical at scale
            footprint: 0.3,
            conversion: 1.5,
            phonon: 0.8,
        }
    }
}

/// Estimate fusion yield for a trigger system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionYieldEstimate {
    /// Trigger system used
    pub trigger: ExtendedTriggerMethod,
    /// Reactions per second estimate
    pub reactions_per_second: f64,
    /// Thermal power output (W)
    pub thermal_power_w: f64,
    /// Q factor (fusion power / input power)
    pub q_factor: f64,
    /// Confidence level (0-1)
    pub confidence: f64,
    /// Notes on calculation
    pub notes: String,
}

/// Estimate fusion yield from a trigger system
pub fn estimate_fusion_yield(
    sys: &TriggerSystemSpec,
    target_mass_g: f64,
    d_loading_ratio: f64, // D/Pd ratio
) -> FusionYieldEstimate {
    // Number of D atoms in target
    // Assume Pd: 106.4 g/mol, so moles = mass / 106.4
    let pd_moles = target_mass_g / 106.4;
    let d_atoms = pd_moles * d_loading_ratio * super::constants::N_AVOGADRO;

    // D-D pairs that could fuse (rough estimate)
    let d_pairs = d_atoms * d_loading_ratio / 2.0;

    // Trigger volume fraction
    let trigger_volume = PI
        * (sys.physics.penetration_depth_um * 1e-4).powi(2)
        * sys.physics.deposition_profile_um
        * 1e-4; // cm³
    let target_volume = target_mass_g / 12.0; // Pd density ~12 g/cm³
    let volume_fraction = (trigger_volume / target_volume).min(1.0);

    // Affected D pairs per trigger
    let affected_pairs = d_pairs * volume_fraction;

    // Fusion probability per affected pair per trigger
    // Base rate from conversion efficiency, enhanced by screening
    let base_prob = sys.physics.conversion_efficiency;
    let screening_enhancement = sys.physics.screening_factor;
    let phonon_factor = 1.0 + sys.physics.phonon_efficiency * 10.0;

    let fusion_prob = base_prob * screening_enhancement * phonon_factor;

    // Fusions per trigger
    let fusions_per_trigger = affected_pairs * fusion_prob;

    // Fusions per second (at max rep rate)
    let fusions_per_second = fusions_per_trigger * sys.max_rep_rate_hz;

    // Energy per D-D fusion: ~3.65 MeV average (D-D has two branches)
    let mev_per_fusion = 3.65;
    let j_per_fusion = mev_per_fusion * super::constants::MEV_TO_J;

    let thermal_power_w = fusions_per_second * j_per_fusion;

    // Input power estimate
    let input_power_w = sys.energy_per_trigger_j * sys.max_rep_rate_hz / sys.wall_plug_efficiency;

    let q_factor = if input_power_w > 0.0 {
        thermal_power_w / input_power_w
    } else {
        0.0
    };

    // Confidence based on TRL and mechanism understanding
    let confidence = match sys.method {
        ExtendedTriggerMethod::MuonCatalysis => 0.9, // Well-understood
        ExtendedTriggerMethod::XfelXRay => 0.7,
        ExtendedTriggerMethod::ElectronBeam => 0.5,
        ExtendedTriggerMethod::PulsedElectrolysis => 0.3,
        ExtendedTriggerMethod::NeecCascade => 0.2,
        _ => 0.4,
    };

    FusionYieldEstimate {
        trigger: sys.method,
        reactions_per_second: fusions_per_second,
        thermal_power_w,
        q_factor,
        confidence,
        notes: format!(
            "Estimated for {target_mass_g:.1}g Pd at D/Pd={d_loading_ratio:.2}. \
             Affected volume: {trigger_volume:.2e} cm³, pairs: {affected_pairs:.2e}, prob/pair: {fusion_prob:.2e}"
        ),
    }
}

/// Compare all trigger systems for a given configuration
pub fn compare_triggers(target_mass_g: f64, d_loading: f64) -> Vec<FusionYieldEstimate> {
    let library = TriggerSystemLibrary::new();
    let mut estimates: Vec<_> = library
        .systems
        .iter()
        .map(|sys| estimate_fusion_yield(sys, target_mass_g, d_loading))
        .collect();

    estimates.sort_by(|a, b| {
        b.q_factor
            .partial_cmp(&a.q_factor)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    estimates
}

// ============================================================================
// LITERATURE-CALIBRATED PHYSICS CONSTANTS
// ============================================================================

/// Result of full Gamow peak thermal integration for D-D reaction rate.
#[derive(Debug, Clone)]
pub struct GamowIntegrationResult {
    /// Thermally averaged reaction rate <σv> (cm³/s)
    pub sigma_v_cm3_s: f64,
    /// Gamow peak energy (keV)
    pub gamow_peak_kev: f64,
    /// Gamow peak width (keV)
    pub gamow_width_kev: f64,
    /// Screening enhancement factor (ratio of screened to bare <σv>)
    pub screening_enhancement: f64,
    /// Phonon enhancement factor (ratio with/without phonon modes)
    pub phonon_enhancement: f64,
}

/// Detail for a single D-D reaction channel.
#[derive(Debug, Clone)]
pub struct DDChannelDetail {
    /// Channel name (e.g., "D+D → He3 + n")
    pub name: String,
    /// Branching ratio (fraction of total reactions)
    pub branching_ratio: f64,
    /// Reaction rate for this channel (reactions/s)
    pub reaction_rate_s: f64,
    /// Q-value for this channel (MeV)
    pub q_value_mev: f64,
    /// Thermal power from this channel (W)
    pub thermal_power_w: f64,
    /// Whether this channel produces neutrons
    pub produces_neutrons: bool,
    /// Whether this channel produces tritium
    pub produces_tritium: bool,
}

/// Multi-channel D-D branching yield result.
#[derive(Debug, Clone)]
pub struct DDChannelResult {
    /// Total reaction rate (reactions/s)
    pub total_reaction_rate_s: f64,
    /// Total thermal power from all channels (W)
    pub total_thermal_power_w: f64,
    /// Neutron production rate (neutrons/s) — for shielding calculations
    pub neutron_production_rate_s: f64,
    /// Tritium accumulation rate (g/s) — for safety tracking
    pub tritium_production_rate_g_s: f64,
    /// Per-channel details
    pub channels: Vec<DDChannelDetail>,
}

/// LCF-specific physics constants calibrated against published literature.
///
/// ## Key References
///
/// - **Screening Enhancement**: Raiola et al., Eur. Phys. J. A 19 (2004) 283-287
///   "Enhanced electron screening in d(d,p)t for deuterated metals"
///   Measured Ue = 309 ± 12 eV for Pd, 320 ± 15 eV for Pt
///
/// - **D-D Cross-Sections**: ENDF/B-VIII.0, Brown et al., NDS 148 (2018) 1-142
///   S-factor: S(E) ≈ 55 keV·b at low energies
///
/// - **Electron Stopping**: NIST ESTAR database (Berger et al.)
///   CSDA ranges and stopping powers for electrons in Pd
///
/// - **NASA LCF Results**: Steinetz et al., Phys. Rev. C 101 (2020) 044610
///   "Energetic particle and gamma-ray emission from D-loaded Pd/Au"
///   Observed ~10³ neutrons/s from 1 cm² target under X-ray irradiation
///
/// - **Muon Catalysis**: Jones et al., Phys. Rev. Lett. 56 (1986) 588
///   ~150 fusions per muon, α-sticking ~0.5%
///
/// - **NEEC Theory**: Pálffy et al., Phys. Rev. C 73 (2006) 014602
///   NEEC cross-sections typically 10⁻²⁴ - 10⁻²⁰ cm² depending on resonance
pub struct LcfPhysicsConstants;

impl LcfPhysicsConstants {
    // === D-D Fusion Constants ===

    /// D-D astrophysical S-factor at zero energy (keV·barn)
    /// ENDF/B-VIII.0, confirmed by Krauss et al. (1987)
    pub const DD_S_FACTOR_KEV_BARN: f64 = 55.0;

    /// D-D Gamow peak energy at 300K (eV)
    /// E_G = (π α Z₁ Z₂)² μ c² / 2 ≈ 5.5 keV for D-D
    pub const DD_GAMOW_PEAK_EV: f64 = 5500.0;

    /// D-D bare Coulomb barrier energy (keV)
    pub const DD_COULOMB_BARRIER_KEV: f64 = 430.0;

    /// D-T Q-value (MeV) - NRL Plasma Formulary
    pub const DT_Q_VALUE_MEV: f64 = 17.589;

    /// D-D Q-value, neutron branch (MeV) - He3 + n
    pub const DD_Q_NEUTRON_MEV: f64 = 3.269;

    /// D-D Q-value, proton branch (MeV) - T + p
    pub const DD_Q_PROTON_MEV: f64 = 4.033;

    /// D-D average Q-value (50/50 branching, MeV)
    pub const DD_Q_AVERAGE_MEV: f64 = 3.651;

    /// D-He3 Q-value (MeV) - aneutronic
    pub const DHE3_Q_VALUE_MEV: f64 = 18.353;

    // === Screening Enhancement Factors ===
    // Measured by Raiola et al. (2004), Czerski et al. (2001)

    /// Electron screening energy in Pd (eV)
    /// Raiola et al. EPJ A 19 (2004): 309 ± 12 eV
    pub const SCREENING_UE_PD_EV: f64 = 309.0;

    /// Electron screening energy in Pt (eV)
    /// Raiola et al. EPJ A 19 (2004): 320 ± 15 eV
    pub const SCREENING_UE_PT_EV: f64 = 320.0;

    /// Electron screening energy in Ti (eV)
    /// Czerski et al. EPL 68 (2004): 363 ± 20 eV
    pub const SCREENING_UE_TI_EV: f64 = 363.0;

    /// Electron screening energy in Ta (eV)
    /// Czerski et al. EPL 68 (2004): 322 ± 15 eV
    pub const SCREENING_UE_TA_EV: f64 = 322.0;

    /// Theoretical Debye screening energy for metals (eV)
    /// Ichimaru (1993) - much lower than measured values
    pub const SCREENING_UE_DEBYE_EV: f64 = 25.0;

    /// Screening enhancement factor using the Assenbaum formula.
    ///
    /// σ_screened(E) = S(E)/(E+Ue) * exp(-B_G/√(E+Ue))
    /// σ_bare(E) = S(E)/E * exp(-B_G/√E)
    ///
    /// Enhancement f = σ_screened/σ_bare
    ///   = [E/(E+Ue)] * exp(B_G * [1/√E - 1/√(E+Ue)])
    ///
    /// where B_G = π α Z₁Z₂ √(2μc²) ≈ 31.38 keV^(1/2) for D-D
    ///
    /// At 1 keV with Pd (Ue=309 eV): ~100× enhancement
    /// At 10 keV with Pd: ~1.2× enhancement
    ///
    /// Reference: Assenbaum et al., Z. Phys. A 327 (1987) 461-468
    pub fn screening_enhancement_at_energy(ue_ev: f64, e_cm_ev: f64) -> f64 {
        let e_kev = e_cm_ev / 1000.0;
        let ue_kev = ue_ev / 1000.0;

        if e_kev <= 0.01 {
            return 1.0;
        } // Below validity range

        // Gamow constant for D-D: B_G = π α √(2 μ c²) ≈ 31.38 keV^(1/2)
        let b_gamow = 31.38_f64;

        // Assenbaum formula
        let e_screened = e_kev + ue_kev;
        let exponent = b_gamow * (1.0 / e_kev.sqrt() - 1.0 / e_screened.sqrt());
        let prefactor = e_kev / e_screened;

        let enhancement = prefactor * exponent.exp();

        // Cap at physically reasonable values
        enhancement.clamp(1.0, 1e6)
    }

    // === Electron Stopping Powers (NIST ESTAR) ===

    /// Electron CSDA range in Pd at 50 keV (μm)
    /// NIST ESTAR: 8.7 μm (ρ = 12.02 g/cm³)
    pub const ELECTRON_RANGE_50KEV_PD_UM: f64 = 8.7;

    /// Electron CSDA range in Pd at 100 keV (μm)
    /// NIST ESTAR: 27.3 μm
    pub const ELECTRON_RANGE_100KEV_PD_UM: f64 = 27.3;

    /// Electron total stopping power in Pd at 50 keV (MeV·cm²/g)
    /// NIST ESTAR: ~3.5 MeV·cm²/g
    pub const ELECTRON_STOPPING_50KEV_PD: f64 = 3.5;

    /// Electron total stopping power in Pd at 100 keV (MeV·cm²/g)
    /// NIST ESTAR: ~2.1 MeV·cm²/g
    pub const ELECTRON_STOPPING_100KEV_PD: f64 = 2.1;

    // === X-ray Absorption ===

    /// Pd mass attenuation coefficient at 10 keV (cm²/g)
    /// NIST XCOM: ~220 cm²/g (above K-edge at 24.35 keV)
    pub const PD_MASS_ATTEN_10KEV: f64 = 220.0;

    /// Pd mass attenuation coefficient at 30 keV (cm²/g)
    /// NIST XCOM: ~35 cm²/g (above K-edge)
    pub const PD_MASS_ATTEN_30KEV: f64 = 35.0;

    /// Pd K-edge energy (keV)
    pub const PD_K_EDGE_KEV: f64 = 24.35;

    // === NASA LCF Experimental Results ===

    /// Observed neutron rate from Steinetz et al. (2020)
    /// ~10³ n/s from 1 cm² PdAu deuteride under 12 keV X-rays
    pub const NASA_NEUTRON_RATE_PER_CM2: f64 = 1e3;

    /// NASA LCF X-ray energy used (keV)
    pub const NASA_XRAY_ENERGY_KEV: f64 = 12.0;

    /// NASA LCF target thickness (μm)
    pub const NASA_TARGET_THICKNESS_UM: f64 = 25.0;

    // === Muon Catalysis Constants ===

    /// Fusions per muon (Jones et al. 1986)
    pub const MUON_FUSIONS_PER_MUON: f64 = 150.0;

    /// Muon lifetime (μs)
    pub const MUON_LIFETIME_US: f64 = 2.196;

    /// Alpha sticking probability (fraction)
    /// Limits the number of fusions per muon
    pub const MUON_ALPHA_STICKING: f64 = 0.005;

    /// Muon cycling rate (μs⁻¹) in D-T (Jones et al. 1986)
    /// λ_c ≈ 1.2 × 10^8 s⁻¹ = 120 per μs at liquid density
    pub const MUON_CYCLING_RATE: f64 = 120.0; // per μs

    // === Phonon Properties ===

    /// Pd Debye temperature (K) - CRC Handbook
    pub const PD_DEBYE_TEMP_K: f64 = 274.0;

    /// Pd maximum phonon energy (meV) ≈ kB * θD
    pub const PD_MAX_PHONON_MEV: f64 = 23.6;

    /// Ti Debye temperature (K) - CRC Handbook
    pub const TI_DEBYE_TEMP_K: f64 = 420.0;

    /// PdD₀.₇ optical phonon energy (meV)
    /// From inelastic neutron scattering: Ross et al. (1998)
    pub const PDD_OPTICAL_PHONON_MEV: f64 = 56.0;

    /// PdD₀.₇ optical phonon energy in keV (for Gamow integration)
    pub const PDD_OPTICAL_PHONON_KEV: f64 = 0.056;

    /// D-D tunneling probability at energy E in Pd with screening
    /// Estimated from WKB approximation with Ue = 309 eV
    pub fn dd_tunneling_probability(e_cm_ev: f64) -> f64 {
        let e_kev = e_cm_ev / 1000.0;
        if e_kev <= 0.0 {
            return 0.0;
        }

        // Gamow factor: exp(-B_G / √E) where B_G ≈ 31.38 for D-D
        let b_gamow = 31.38_f64;
        let bare_prob = (-b_gamow / e_kev.sqrt()).exp();

        // Apply screening enhancement
        let screened_prob =
            bare_prob * Self::screening_enhancement_at_energy(Self::SCREENING_UE_PD_EV, e_cm_ev);

        screened_prob.min(1.0)
    }

    /// D-D fusion cross-section at center-of-mass energy E (barn)
    ///
    /// Uses astrophysical S-factor parameterization:
    ///   σ(E) = S(E) / E * exp(-B_G / √E)
    ///
    /// where B_G = π α Z₁Z₂ √(2μc²) ≈ 31.38 keV^(1/2) for D-D
    ///
    /// Reference values (ENDF/B-VIII.0):
    ///   σ(10 keV) ≈ 2.8 × 10⁻⁴ barn
    ///   σ(100 keV) ≈ 2.0 × 10⁻² barn
    pub fn dd_cross_section_barn(e_cm_kev: f64) -> f64 {
        if e_cm_kev <= 0.0 {
            return 0.0;
        }

        // Gamow constant for D-D
        let b_gamow = 31.38_f64;

        // Gamow factor
        let gamow = (-b_gamow / e_cm_kev.sqrt()).exp();

        // S-factor (roughly constant at low energies)
        let s_factor = Self::DD_S_FACTOR_KEV_BARN; // keV·barn

        // Cross-section
        s_factor / e_cm_kev * gamow
    }

    /// Screened D-D fusion cross-section (barn)
    pub fn dd_cross_section_screened_barn(e_cm_kev: f64, host_ue_ev: f64) -> f64 {
        let bare = Self::dd_cross_section_barn(e_cm_kev);
        let enhancement = Self::screening_enhancement_at_energy(host_ue_ev, e_cm_kev * 1000.0);
        bare * enhancement
    }

    /// Estimate LCF fusion rate based on NASA results and scaling
    ///
    /// NASA observed ~10³ n/s from 25 μm PdAu at 12 keV X-ray.
    /// Scale to different conditions using:
    /// - Target thickness (linear if < penetration depth)
    /// - X-ray flux (linear)
    /// - Screening enhancement (exponential with host material)
    pub fn estimate_lcf_rate(
        target_area_cm2: f64,
        target_thickness_um: f64,
        trigger_energy_kev: f64,
        trigger_flux_per_cm2_s: f64,
        host_ue_ev: f64,
    ) -> f64 {
        // Scale from NASA baseline
        let area_factor = target_area_cm2 / 1.0; // NASA used 1 cm²
        let thickness_factor = (target_thickness_um / Self::NASA_TARGET_THICKNESS_UM).min(5.0);

        // X-ray absorption efficiency (higher energy = less absorption)
        let atten_factor = if trigger_energy_kev < Self::PD_K_EDGE_KEV {
            (trigger_energy_kev / Self::NASA_XRAY_ENERGY_KEV).powf(0.5)
        } else {
            0.5 // Less efficient above K-edge
        };

        // Screening enhancement relative to Pd baseline
        let screening_ratio = host_ue_ev / Self::SCREENING_UE_PD_EV;

        // Flux scaling (NASA X-ray flux estimated at ~10¹² photons/cm²/s)
        let flux_factor = trigger_flux_per_cm2_s / 1e12;

        Self::NASA_NEUTRON_RATE_PER_CM2
            * area_factor
            * thickness_factor
            * atten_factor
            * screening_ratio
            * flux_factor
    }

    // === Direction A: Physics Fidelity ===

    /// Temperature-dependent screening energy.
    ///
    /// At room temperature (~300K), returns the measured screening energy unchanged.
    /// Above ~800K, transitions to Debye model where Ue ∝ √(T/300).
    /// Uses Gaussian crossover to smoothly blend the two regimes.
    ///
    /// Reference: Czerski et al., J. Phys. G 35 (2008) 014012
    pub fn screening_energy_at_temperature(ue_measured_ev: f64, t_lattice_k: f64) -> f64 {
        // Measured value valid near 300K
        let ue_room = ue_measured_ev;
        // Debye model scaling: Ue ∝ √(T/300)
        let ue_debye = ue_measured_ev * (t_lattice_k / 300.0).sqrt();
        // Gaussian crossover centered at 300K with width 1000K
        let f = (-(t_lattice_k - 300.0).powi(2) / (2.0 * 1000.0_f64.powi(2))).exp();
        // f ≈ 1 near 300K (use measured), f → 0 far from 300K (use Debye)
        f * ue_room + (1.0 - f) * ue_debye
    }

    /// Full Gamow peak thermal integration for D-D reaction rate <σv>.
    ///
    /// Replaces single-energy cross-section evaluation with:
    /// ```text
    /// <σv> = √(8/πμ) · (kT)^(-3/2) · ∫ σ(E) · E · exp(-E/kT) dE
    /// ```
    ///
    /// Uses 200-point trapezoidal integration over ±2.5 Gamow widths,
    /// with log-space arithmetic to avoid over/underflow at room temperature
    /// where <σv> ~ 10⁻⁵⁰ cm³/s.
    ///
    /// Parameters:
    /// - `t_lattice_k`: Lattice temperature in Kelvin
    /// - `host_ue_ev`: Measured screening energy of host material (eV)
    /// - `n_phonon_modes`: Number of coherent phonon modes (0-5)
    ///
    /// Reference: Clayton, "Principles of Stellar Evolution and Nucleosynthesis" (1983)
    pub fn dd_reaction_rate_integrated(
        t_lattice_k: f64,
        host_ue_ev: f64,
        n_phonon_modes: u32,
    ) -> GamowIntegrationResult {
        // Physical constants
        let kb_kev = 8.617e-5 * 1e-3; // Boltzmann constant in keV/K
        let kt_kev = kb_kev * t_lattice_k;

        if kt_kev <= 0.0 {
            return GamowIntegrationResult {
                sigma_v_cm3_s: 0.0,
                gamow_peak_kev: 0.0,
                gamow_width_kev: 0.0,
                screening_enhancement: 1.0,
                phonon_enhancement: 1.0,
            };
        }

        // Gamow constant for D-D: B_G ≈ 31.38 keV^(1/2)
        let b_g = 31.38_f64;

        // Gamow peak: E_0 = (B_G² · kT² / 4)^(1/3)
        let e0_kev = (b_g * b_g * kt_kev * kt_kev / 4.0).powf(1.0 / 3.0);

        // Gamow width: Δ = 4√(E_0·kT/3)
        let delta_kev = 4.0 * (e0_kev * kt_kev / 3.0).sqrt();

        // Phonon enhancement: each coherent mode adds ℏω = 0.056 keV to effective CM energy
        let phonon_boost_kev = (n_phonon_modes as f64) * Self::PDD_OPTICAL_PHONON_KEV;

        // Temperature-dependent screening
        let ue_ev = Self::screening_energy_at_temperature(host_ue_ev, t_lattice_k);
        let ue_kev = ue_ev / 1000.0;

        // Integration range: ±2.5 Gamow widths around peak
        let e_min = (e0_kev - 2.5 * delta_kev).max(0.001);
        let e_max = e0_kev + 2.5 * delta_kev;
        let n_points: usize = 200;
        let de = (e_max - e_min) / (n_points as f64);

        // Reduced mass of D-D system in keV/c²
        // μ = m_D/2 = 2.014 amu / 2 = 1.007 amu
        // 1 amu = 931494 keV/c²
        let _mu_kev = 1.007 * 931494.0;

        // Prefactor: √(8/πμ) · (kT)^(-3/2) in appropriate units
        // We work in log space to avoid numerical issues

        // 200-point trapezoidal integration in log space
        // Integrand: σ(E) · E · exp(-E/kT)
        // σ(E) = S/(E+Ue) · exp(-B_G/√(E+Ue))
        // With phonon boost: E_eff = E + phonon_boost
        let mut log_integrand_values: Vec<f64> = Vec::with_capacity(n_points + 1);
        let mut energies: Vec<f64> = Vec::with_capacity(n_points + 1);

        for i in 0..=n_points {
            let e_kev = e_min + (i as f64) * de;
            let e_eff = e_kev + phonon_boost_kev;

            // Log of cross-section: log(S/(E+Ue)) - B_G/√(E+Ue)
            let e_screened = e_eff + ue_kev;
            let log_sigma =
                (Self::DD_S_FACTOR_KEV_BARN / e_screened).ln() - b_g / e_screened.sqrt();

            // Log of integrand: log(σ) + log(E) - E/kT
            let log_val = log_sigma + e_kev.ln() - e_kev / kt_kev;

            energies.push(e_kev);
            log_integrand_values.push(log_val);
        }

        // Find maximum log value for numerical stability
        let log_max = log_integrand_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // Trapezoidal integration in shifted space
        let mut integral = 0.0;
        for i in 0..n_points {
            let f0 = (log_integrand_values[i] - log_max).exp();
            let f1 = (log_integrand_values[i + 1] - log_max).exp();
            integral += 0.5 * (f0 + f1) * de;
        }

        // log of integral = log_max + log(integral)
        let log_integral = log_max + integral.ln();

        // Compute <σv> in cm³/s
        // <σv> = √(8/(π·μ)) · (kT)^(-3/2) · integral
        // σ in barn = 10⁻²⁴ cm², E in keV
        // Need unit conversions:
        //   √(8/(π·μ_keV)) with velocities: v = √(2E/μ) where μ in keV/c²
        //   Factor = √(8/π) · (1/μ)^(1/2) · (kT)^(-3/2) · c
        //   In CGS: factor × 3e10 cm/s × (keV conversion)
        // Standard formula: <σv> = (8/π·μ)^(1/2) · (kT)^(-3/2) · integral × unit_factors
        // Using the NRL Plasma Formulary result:
        //   <σv> [cm³/s] = 3.727e-12 / (μ_amu^(1/2) · T_keV^(3/2)) · integral [keV·barn·keV]
        // integral has units of [barn·keV²] (σ·E·dE with Boltzmann absorbed)
        let mu_amu: f64 = 1.007;
        let prefactor: f64 = 3.727e-12 / (mu_amu.sqrt() * kt_kev.powf(1.5));

        // Convert barn·keV² to barn·keV (already keV integral)
        // integral is in barn · keV (σ in barn, E·dE in keV²... but σ = S/E·exp, so integral ~ barn·keV)
        let log_sigma_v: f64 = prefactor.ln() + log_integral;
        let sigma_v = log_sigma_v.exp();

        // Compute screening enhancement relative to bare
        let bare_result = {
            let mut log_bare_vals: Vec<f64> = Vec::with_capacity(n_points + 1);
            for i in 0..=n_points {
                let e_kev = energies[i];
                let log_sigma = (Self::DD_S_FACTOR_KEV_BARN / e_kev).ln() - b_g / e_kev.sqrt();
                let log_val = log_sigma + e_kev.ln() - e_kev / kt_kev;
                log_bare_vals.push(log_val);
            }
            let log_max_b = log_bare_vals
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let mut int_b = 0.0;
            for i in 0..n_points {
                let f0 = (log_bare_vals[i] - log_max_b).exp();
                let f1 = (log_bare_vals[i + 1] - log_max_b).exp();
                int_b += 0.5 * (f0 + f1) * de;
            }
            log_max_b + int_b.ln()
        };

        let screening_enhancement = (log_integral - bare_result).exp();

        // Phonon enhancement: ratio with phonon modes to without
        let phonon_enhancement = if n_phonon_modes > 0 {
            let no_phonon = Self::dd_reaction_rate_integrated(t_lattice_k, host_ue_ev, 0);
            if no_phonon.sigma_v_cm3_s > 0.0 {
                sigma_v / no_phonon.sigma_v_cm3_s
            } else {
                1.0
            }
        } else {
            1.0
        };

        GamowIntegrationResult {
            sigma_v_cm3_s: sigma_v,
            gamow_peak_kev: e0_kev,
            gamow_width_kev: delta_kev,
            screening_enhancement,
            phonon_enhancement,
        }
    }

    /// Multi-channel D-D branching yield calculation.
    ///
    /// Tracks both D-D reaction channels:
    /// - Neutron channel: D+D → He3 + n (3.27 MeV), ~50% branching
    /// - Proton channel: D+D → T + p (4.03 MeV), ~50% branching
    ///
    /// Energy-dependent branching: R_n(E) = 0.5·(1 - 0.003·E_keV) for E < 100 keV
    ///
    /// Reference: Bosch & Hale, Nucl. Fusion 32 (1992) 611-631
    pub fn dd_branched_yield(total_reaction_rate_s: f64, e_cm_kev: f64) -> DDChannelResult {
        // Energy-dependent branching ratio for neutron channel
        let e_clamped = e_cm_kev.clamp(0.0, 100.0);
        let r_neutron = 0.5 * (1.0 - 0.003 * e_clamped);
        let r_proton = 1.0 - r_neutron;

        let neutron_rate = total_reaction_rate_s * r_neutron;
        let proton_rate = total_reaction_rate_s * r_proton;

        // Energy per reaction (MeV → J)
        let mev_to_j = super::constants::MEV_TO_J;

        let neutron_channel = DDChannelDetail {
            name: "D+D → He3 + n".to_string(),
            branching_ratio: r_neutron,
            reaction_rate_s: neutron_rate,
            q_value_mev: Self::DD_Q_NEUTRON_MEV,
            thermal_power_w: neutron_rate * Self::DD_Q_NEUTRON_MEV * mev_to_j,
            produces_neutrons: true,
            produces_tritium: false,
        };

        let proton_channel = DDChannelDetail {
            name: "D+D → T + p".to_string(),
            branching_ratio: r_proton,
            reaction_rate_s: proton_rate,
            q_value_mev: Self::DD_Q_PROTON_MEV,
            thermal_power_w: proton_rate * Self::DD_Q_PROTON_MEV * mev_to_j,
            produces_neutrons: false,
            produces_tritium: true,
        };

        let total_power = neutron_channel.thermal_power_w + proton_channel.thermal_power_w;

        // Tritium accumulation: each proton-channel reaction produces one T atom
        // T mass rate: rate × 3.016 amu / N_A
        let amu_to_g = 1.6605e-24;
        let tritium_production_g_s = proton_rate * 3.016 * amu_to_g;

        DDChannelResult {
            total_reaction_rate_s,
            total_thermal_power_w: total_power,
            neutron_production_rate_s: neutron_rate,
            tritium_production_rate_g_s: tritium_production_g_s,
            channels: vec![neutron_channel, proton_channel],
        }
    }
}

/// Literature-validated trigger physics for a specific trigger type
pub struct CalibratedTriggerPhysics;

impl CalibratedTriggerPhysics {
    /// Calibrate electron beam physics against NIST ESTAR
    pub fn electron_beam_penetration_um(energy_kev: f64) -> f64 {
        // Kanaya-Okayama model for electron range in Pd (Z=46, A=106.42, ρ=12.02)
        // R = (0.0276 * A / (Z^0.89 * ρ)) * E^1.67  (μm, keV)
        let a = 106.42;
        let z = 46.0_f64;
        let rho = 12.02;
        0.0276 * a / (z.powf(0.89) * rho) * energy_kev.powf(1.67)
    }

    /// Calibrate X-ray penetration depth in Pd (μm)
    pub fn xray_penetration_um(energy_kev: f64) -> f64 {
        // 1/e depth = 1/(μ * ρ) where μ is mass attenuation coefficient
        let mu = if energy_kev < LcfPhysicsConstants::PD_K_EDGE_KEV {
            // Below K-edge: μ ∝ E^-3 (photoelectric)
            LcfPhysicsConstants::PD_MASS_ATTEN_10KEV * (10.0 / energy_kev).powi(3)
        } else {
            // Above K-edge
            LcfPhysicsConstants::PD_MASS_ATTEN_30KEV * (30.0 / energy_kev).powi(3)
        };

        // Convert to μm: depth = 1/(μ * ρ) in cm, then * 10⁴ for μm
        let depth_cm = 1.0 / (mu * 12.02);
        depth_cm * 1e4
    }

    /// Calculate phonon generation efficiency from electron beam
    /// Based on electron-phonon coupling constant for Pd
    pub fn ebeam_phonon_efficiency(energy_kev: f64) -> f64 {
        // At low energies, most energy goes to electronic excitations
        // which eventually couple to phonons via electron-phonon interaction
        // Pd electron-phonon coupling: λ ≈ 0.39 (Allen & Dynes 1975)
        let lambda_ep = 0.39;

        // Fraction that goes to phonons depends on energy
        // Low energy (< 1 keV): mostly electronic stopping → phonon via e-p coupling
        // High energy (> 100 keV): some nuclear stopping → direct phonon
        let nuclear_fraction = if energy_kev < 1.0 {
            0.01 // Almost all electronic
        } else if energy_kev < 10.0 {
            0.05
        } else {
            0.1 // Some nuclear stopping
        };

        // Total phonon generation: nuclear stopping + electronic → phonon
        nuclear_fraction + (1.0 - nuclear_fraction) * lambda_ep * 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_creation() {
        let lib = TriggerSystemLibrary::new();
        assert!(
            lib.systems.len() >= 10,
            "Should have multiple trigger systems"
        );
    }

    #[test]
    fn test_filter_by_cost() {
        let lib = TriggerSystemLibrary::new();
        let cheap = lib.filter(&TriggerCriteria {
            max_cost_usd: Some(50_000.0),
            ..Default::default()
        });

        assert!(!cheap.is_empty(), "Should find cheap triggers");
        for sys in cheap {
            assert!(sys.cost_estimate_usd <= 50_000.0);
        }
    }

    #[test]
    fn test_filter_by_trl() {
        let lib = TriggerSystemLibrary::new();
        let proven = lib.filter(&TriggerCriteria {
            min_trl: Some(7),
            ..Default::default()
        });

        assert!(!proven.is_empty(), "Should find proven triggers");
        for sys in proven {
            assert!(sys.trl.0 >= 7);
        }
    }

    #[test]
    fn test_ranking() {
        let lib = TriggerSystemLibrary::new();
        let ranked = lib.rank(&TriggerWeights::desktop());

        // Desktop ranking should favor cheap, compact options
        let top = &ranked[0].0;
        assert!(
            top.cost_estimate_usd < 100_000.0,
            "Desktop top pick should be affordable"
        );
    }

    #[test]
    fn test_fusion_yield_estimate() {
        let lib = TriggerSystemLibrary::new();
        let ebeam = lib
            .systems
            .iter()
            .find(|s| s.method == ExtendedTriggerMethod::ElectronBeam)
            .unwrap();

        let estimate = estimate_fusion_yield(ebeam, 10.0, 0.7);

        assert!(estimate.reactions_per_second >= 0.0);
        assert!(estimate.thermal_power_w >= 0.0);
        assert!(estimate.confidence > 0.0 && estimate.confidence <= 1.0);
    }

    #[test]
    fn test_optimal_for_power_scale() {
        let lib = TriggerSystemLibrary::new();

        let low_power = lib.optimal_for_power(10.0);
        let mid_power = lib.optimal_for_power(5_000.0);
        let _high_power = lib.optimal_for_power(100_000.0);

        // Should return different systems at different scales
        assert_eq!(low_power.method, ExtendedTriggerMethod::PiezoPhonon);
        assert_eq!(mid_power.method, ExtendedTriggerMethod::ElectronBeam);
    }

    #[test]
    fn test_compare_all_triggers() {
        let estimates = compare_triggers(10.0, 0.7);
        assert!(!estimates.is_empty());

        // Should be sorted by Q factor
        for i in 1..estimates.len() {
            assert!(estimates[i - 1].q_factor >= estimates[i].q_factor);
        }
    }

    // === Literature-calibrated physics tests ===

    #[test]
    fn test_dd_cross_section_at_10kev() {
        // At 10 keV CM, D-D cross-section should be ~2.8e-4 barn (ENDF/B-VIII.0)
        let sigma = LcfPhysicsConstants::dd_cross_section_barn(10.0);
        assert!(
            sigma > 1e-5 && sigma < 1e-2,
            "D-D σ at 10 keV should be ~3e-4 barn, got {:.2e}",
            sigma
        );
    }

    #[test]
    fn test_dd_cross_section_at_100kev() {
        // At 100 keV CM, D-D cross-section ~2e-2 barn
        let sigma = LcfPhysicsConstants::dd_cross_section_barn(100.0);
        assert!(
            sigma > 1e-3 && sigma < 1.0,
            "D-D σ at 100 keV should be ~2e-2 barn, got {:.2e}",
            sigma
        );
    }

    #[test]
    fn test_dd_cross_section_monotonic() {
        // Cross-section should increase with energy in this range
        let sigma_10 = LcfPhysicsConstants::dd_cross_section_barn(10.0);
        let sigma_50 = LcfPhysicsConstants::dd_cross_section_barn(50.0);
        let sigma_100 = LcfPhysicsConstants::dd_cross_section_barn(100.0);

        assert!(sigma_50 > sigma_10, "σ(50) > σ(10)");
        assert!(sigma_100 > sigma_50, "σ(100) > σ(50)");
    }

    #[test]
    fn test_screening_enhancement_at_1kev() {
        // At 1 keV CM, Pd screening (Ue=309 eV) gives ~100× enhancement
        // Assenbaum formula: f = [E/(E+Ue)] * exp(B_G * [1/√E - 1/√(E+Ue)])
        let enhancement = LcfPhysicsConstants::screening_enhancement_at_energy(
            LcfPhysicsConstants::SCREENING_UE_PD_EV,
            1000.0, // 1 keV = 1000 eV
        );
        assert!(
            enhancement > 10.0 && enhancement < 1000.0,
            "Screening enhancement at 1 keV should be ~100×, got {:.1}×",
            enhancement
        );
    }

    #[test]
    fn test_screening_small_at_high_energy() {
        // At 100 keV, screening should be negligible (~1.01×)
        let enhancement = LcfPhysicsConstants::screening_enhancement_at_energy(
            LcfPhysicsConstants::SCREENING_UE_PD_EV,
            100_000.0, // 100 keV
        );
        assert!(
            enhancement > 0.9 && enhancement < 1.1,
            "Screening at 100 keV should be negligible, got {:.4}×",
            enhancement
        );
    }

    #[test]
    fn test_screening_increases_with_ue() {
        // At 1 keV, higher Ue should give higher enhancement
        let enh_pd = LcfPhysicsConstants::screening_enhancement_at_energy(
            LcfPhysicsConstants::SCREENING_UE_PD_EV,
            1000.0,
        );
        let enh_ti = LcfPhysicsConstants::screening_enhancement_at_energy(
            LcfPhysicsConstants::SCREENING_UE_TI_EV,
            1000.0,
        );

        assert!(
            enh_ti > enh_pd,
            "Ti (Ue={} eV, enh={:.1}×) should screen more than Pd (Ue={} eV, enh={:.1}×)",
            LcfPhysicsConstants::SCREENING_UE_TI_EV,
            enh_ti,
            LcfPhysicsConstants::SCREENING_UE_PD_EV,
            enh_pd
        );
    }

    #[test]
    fn test_electron_beam_range_vs_nist() {
        // Compare Kanaya-Okayama model against NIST ESTAR values
        let range_50 = CalibratedTriggerPhysics::electron_beam_penetration_um(50.0);
        let nist_50 = LcfPhysicsConstants::ELECTRON_RANGE_50KEV_PD_UM;

        // Should be within 50% (K-O is approximate)
        let error = (range_50 - nist_50).abs() / nist_50;
        assert!(
            error < 0.5,
            "K-O range at 50 keV ({:.1} μm) vs NIST ({:.1} μm): {:.0}% error",
            range_50,
            nist_50,
            error * 100.0
        );
    }

    #[test]
    fn test_xray_depth_increases_with_energy() {
        let depth_5 = CalibratedTriggerPhysics::xray_penetration_um(5.0);
        let depth_10 = CalibratedTriggerPhysics::xray_penetration_um(10.0);
        let depth_50 = CalibratedTriggerPhysics::xray_penetration_um(50.0);

        assert!(depth_10 > depth_5, "Higher energy X-rays penetrate deeper");
        assert!(depth_50 > depth_10, "Higher energy X-rays penetrate deeper");
    }

    #[test]
    fn test_lcf_rate_scales_with_area() {
        let rate_1cm2 = LcfPhysicsConstants::estimate_lcf_rate(
            1.0,
            25.0,
            12.0,
            1e12,
            LcfPhysicsConstants::SCREENING_UE_PD_EV,
        );
        let rate_10cm2 = LcfPhysicsConstants::estimate_lcf_rate(
            10.0,
            25.0,
            12.0,
            1e12,
            LcfPhysicsConstants::SCREENING_UE_PD_EV,
        );

        assert!(
            (rate_10cm2 / rate_1cm2 - 10.0).abs() < 0.1,
            "Rate should scale linearly with area"
        );
    }

    #[test]
    fn test_lcf_rate_matches_nasa_baseline() {
        // With NASA parameters, should reproduce ~10³ n/s
        let rate = LcfPhysicsConstants::estimate_lcf_rate(
            1.0, // 1 cm²
            LcfPhysicsConstants::NASA_TARGET_THICKNESS_UM,
            LcfPhysicsConstants::NASA_XRAY_ENERGY_KEV,
            1e12, // estimated flux
            LcfPhysicsConstants::SCREENING_UE_PD_EV,
        );

        assert!(
            (rate - LcfPhysicsConstants::NASA_NEUTRON_RATE_PER_CM2).abs()
                / LcfPhysicsConstants::NASA_NEUTRON_RATE_PER_CM2
                < 0.01,
            "Should reproduce NASA baseline: got {:.0} vs expected {:.0}",
            rate,
            LcfPhysicsConstants::NASA_NEUTRON_RATE_PER_CM2
        );
    }

    #[test]
    fn test_phonon_efficiency_physical_range() {
        // Phonon generation should be between 0 and 1
        for energy in [1.0, 10.0, 50.0, 100.0] {
            let eff = CalibratedTriggerPhysics::ebeam_phonon_efficiency(energy);
            assert!(
                eff > 0.0 && eff < 1.0,
                "Phonon efficiency at {} keV = {}, should be in (0,1)",
                energy,
                eff
            );
        }
    }

    #[test]
    fn test_muon_catalysis_constants() {
        // Verify muon constants are internally consistent
        // The alpha sticking probability limits the cycle count:
        // N_fusions ≈ 1 / ω_s where ω_s is the sticking probability
        // This gives an upper bound of 1/0.005 = 200 fusions
        let upper_bound = 1.0 / LcfPhysicsConstants::MUON_ALPHA_STICKING;

        // The observed ~150 fusions is below this upper bound
        // (limited also by muon lifetime and cycling rate)
        assert!(
            LcfPhysicsConstants::MUON_FUSIONS_PER_MUON < upper_bound,
            "Observed fusions ({:.0}) should be below sticking limit ({:.0})",
            LcfPhysicsConstants::MUON_FUSIONS_PER_MUON,
            upper_bound
        );

        // Cycling rate × lifetime gives max possible cycles
        let max_cycles =
            LcfPhysicsConstants::MUON_CYCLING_RATE * LcfPhysicsConstants::MUON_LIFETIME_US;
        assert!(
            max_cycles > LcfPhysicsConstants::MUON_FUSIONS_PER_MUON,
            "Max cycles ({:.0}) should exceed observed fusions ({:.0})",
            max_cycles,
            LcfPhysicsConstants::MUON_FUSIONS_PER_MUON
        );
    }

    // === Direction A: Physics Fidelity Tests ===

    #[test]
    fn test_gamow_integration_at_300k() {
        // At 300K, D-D reaction rate should be negligibly small (~10⁻⁵⁰ cm³/s or less)
        let result = LcfPhysicsConstants::dd_reaction_rate_integrated(300.0, 0.0, 0);
        // The rate is astronomically small but should be finite and positive
        assert!(result.sigma_v_cm3_s >= 0.0);
        assert!(result.gamow_peak_kev > 0.0);
        assert!(result.gamow_width_kev > 0.0);
    }

    #[test]
    fn test_gamow_integration_monotonic_with_temperature() {
        // Reaction rate should increase monotonically with temperature
        let temps = [1000.0, 5000.0, 10000.0, 50000.0];
        let mut prev_rate = 0.0_f64;
        for &t in &temps {
            let result = LcfPhysicsConstants::dd_reaction_rate_integrated(
                t,
                LcfPhysicsConstants::SCREENING_UE_PD_EV,
                0,
            );
            assert!(
                result.sigma_v_cm3_s >= prev_rate,
                "<σv> at {}K ({:.2e}) should be >= at lower T ({:.2e})",
                t,
                result.sigma_v_cm3_s,
                prev_rate
            );
            prev_rate = result.sigma_v_cm3_s;
        }
    }

    #[test]
    fn test_gamow_integration_screening_enhances() {
        // Screened rate should exceed bare rate
        let bare = LcfPhysicsConstants::dd_reaction_rate_integrated(5000.0, 0.0, 0);
        let screened = LcfPhysicsConstants::dd_reaction_rate_integrated(
            5000.0,
            LcfPhysicsConstants::SCREENING_UE_PD_EV,
            0,
        );
        assert!(
            screened.sigma_v_cm3_s >= bare.sigma_v_cm3_s,
            "Screened rate ({:.2e}) should >= bare ({:.2e})",
            screened.sigma_v_cm3_s,
            bare.sigma_v_cm3_s
        );
        assert!(screened.screening_enhancement >= 1.0);
    }

    #[test]
    fn test_gamow_integration_phonon_enhancement() {
        // Adding phonon modes should increase reaction rate
        let no_phonon = LcfPhysicsConstants::dd_reaction_rate_integrated(
            5000.0,
            LcfPhysicsConstants::SCREENING_UE_PD_EV,
            0,
        );
        let with_phonon = LcfPhysicsConstants::dd_reaction_rate_integrated(
            5000.0,
            LcfPhysicsConstants::SCREENING_UE_PD_EV,
            3,
        );
        assert!(
            with_phonon.sigma_v_cm3_s >= no_phonon.sigma_v_cm3_s,
            "Phonon-enhanced rate ({:.2e}) should >= bare ({:.2e})",
            with_phonon.sigma_v_cm3_s,
            no_phonon.sigma_v_cm3_s
        );
        assert!(with_phonon.phonon_enhancement >= 1.0);
    }

    #[test]
    fn test_screening_at_room_temperature() {
        // At 300K, screening_energy_at_temperature should return ~measured value
        let ue = LcfPhysicsConstants::screening_energy_at_temperature(
            LcfPhysicsConstants::SCREENING_UE_PD_EV,
            300.0,
        );
        // Should be very close to measured (Gaussian f ≈ 1 at 300K)
        assert!(
            (ue - LcfPhysicsConstants::SCREENING_UE_PD_EV).abs() < 1.0,
            "Ue at 300K ({:.1} eV) should match measured ({:.1} eV)",
            ue,
            LcfPhysicsConstants::SCREENING_UE_PD_EV
        );
    }

    #[test]
    fn test_screening_debye_at_high_temperature() {
        // At very high T, should transition toward Debye scaling
        let ue_1500k = LcfPhysicsConstants::screening_energy_at_temperature(
            LcfPhysicsConstants::SCREENING_UE_PD_EV,
            1500.0,
        );
        // At 1500K, Debye model gives Ue × √(1500/300) ≈ Ue × 2.24
        // The Gaussian crossover means we're mostly in Debye regime
        assert!(
            ue_1500k > LcfPhysicsConstants::SCREENING_UE_PD_EV,
            "Ue at 1500K ({:.1} eV) should exceed room-temperature value ({:.1} eV)",
            ue_1500k,
            LcfPhysicsConstants::SCREENING_UE_PD_EV
        );
    }

    #[test]
    fn test_dd_branching_sums_to_one() {
        let result = LcfPhysicsConstants::dd_branched_yield(1e6, 10.0);
        let sum: f64 = result.channels.iter().map(|c| c.branching_ratio).sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Branching ratios should sum to 1.0, got {:.10}",
            sum
        );
    }

    #[test]
    fn test_dd_branching_near_50_50() {
        // At low energy, branching should be ~50/50
        let result = LcfPhysicsConstants::dd_branched_yield(1e6, 1.0);
        for channel in &result.channels {
            assert!(
                (channel.branching_ratio - 0.5).abs() < 0.01,
                "Channel '{}' branching {:.4} should be near 0.5 at low energy",
                channel.name,
                channel.branching_ratio
            );
        }
    }

    #[test]
    fn test_dd_branching_energy_conservation() {
        let rate = 1e10;
        let result = LcfPhysicsConstants::dd_branched_yield(rate, 5.0);
        // Total power should equal sum of channel powers
        let sum_power: f64 = result.channels.iter().map(|c| c.thermal_power_w).sum();
        assert!(
            (sum_power - result.total_thermal_power_w).abs() / result.total_thermal_power_w < 1e-10,
            "Channel powers should sum to total"
        );
    }

    #[test]
    fn test_dd_branching_neutron_production() {
        let result = LcfPhysicsConstants::dd_branched_yield(1e6, 5.0);
        // Neutron rate should be ~50% of total
        assert!(
            result.neutron_production_rate_s > 0.0,
            "Should produce neutrons"
        );
        assert!(
            result.neutron_production_rate_s < result.total_reaction_rate_s,
            "Neutron rate should be less than total rate"
        );
    }

    #[test]
    fn test_dd_branching_tritium_accumulation() {
        let result = LcfPhysicsConstants::dd_branched_yield(1e6, 5.0);
        assert!(
            result.tritium_production_rate_g_s > 0.0,
            "Should accumulate tritium from p+T channel"
        );
    }

    #[test]
    fn test_gamow_peak_increases_with_temperature() {
        let low_t = LcfPhysicsConstants::dd_reaction_rate_integrated(1000.0, 0.0, 0);
        let high_t = LcfPhysicsConstants::dd_reaction_rate_integrated(10000.0, 0.0, 0);
        assert!(
            high_t.gamow_peak_kev > low_t.gamow_peak_kev,
            "Gamow peak should shift to higher energy at higher T"
        );
    }
}
