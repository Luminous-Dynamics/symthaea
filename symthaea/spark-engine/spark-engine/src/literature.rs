// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Literature Mining (Option B)
//!
//! Systematic database of LCF/LENR observations to find patterns.
//! Credibility-weighted analysis to separate signal from noise.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A published observation of anomalous nuclear effects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// Unique identifier
    pub id: String,
    /// First author
    pub author: String,
    /// Publication year
    pub year: u16,
    /// Journal/venue
    pub venue: String,
    /// DOI if available
    pub doi: Option<String>,
    /// Host material (Pd, Ti, Ni, etc.)
    pub host_material: HostMaterial,
    /// Fuel (D, H, or mixture)
    pub fuel: Fuel,
    /// Trigger method
    pub trigger: TriggerMethod,
    /// Observed effects
    pub effects: Vec<ObservedEffect>,
    /// Claimed signal magnitude
    pub signal: SignalClaim,
    /// Control experiments performed
    pub controls: Vec<ControlExperiment>,
    /// Credibility assessment
    pub credibility: Credibility,
    /// Replication status
    pub replications: ReplicationStatus,
    /// Key conditions
    pub conditions: ExperimentalConditions,
    /// Notes
    pub notes: String,
}

/// Host material for the lattice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HostMaterial {
    Palladium,
    Titanium,
    Nickel,
    Platinum,
    Tungsten,
    Iron,
    PdAu,     // Alloy
    PdAg,     // Alloy
    NiCu,     // Alloy
    Other,
}

impl HostMaterial {
    /// Known screening energy (eV) if measured.
    pub fn screening_ev(&self) -> Option<f64> {
        match self {
            Self::Palladium => Some(309.0),  // Raiola 2004
            Self::Platinum => Some(320.0),   // Raiola 2004
            Self::Titanium => Some(363.0),   // Czerski 2004
            Self::Nickel => Some(250.0),     // Estimated
            _ => None,
        }
    }

    /// Crystal structure.
    pub fn structure(&self) -> &'static str {
        match self {
            Self::Palladium | Self::Platinum | Self::Nickel | Self::PdAu | Self::PdAg | Self::NiCu => "FCC",
            Self::Titanium => "HCP",
            Self::Tungsten | Self::Iron => "BCC",
            Self::Other => "Unknown",
        }
    }
}

/// Fuel type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Fuel {
    Deuterium,
    Hydrogen,
    DeuteriumHydrogen, // Mixture
    Tritium,           // Rare
}

/// Trigger method used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TriggerMethod {
    Electrolysis,
    GasLoading,
    XRay,
    ElectronBeam,
    LaserStimulation,
    Ultrasonic,
    ThermalCycling,
    IonImplantation,
    GlowDischarge,
    None, // Passive observation
}

/// Type of observed effect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ObservedEffect {
    ExcessHeat { power_w: f64, duration_s: f64 },
    Neutrons { rate_per_s: f64, energy_mev: Option<f64> },
    ChargedParticles { type_: String, rate_per_s: f64 },
    Gamma { energy_kev: f64, rate_per_s: f64 },
    Transmutation { from: String, to: String, amount: String },
    Tritium { amount_bq: f64 },
    Helium4 { amount_atoms: f64 },
}

/// Signal claim with uncertainty.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalClaim {
    /// Primary metric (e.g., "neutron rate", "excess power")
    pub metric: String,
    /// Claimed value
    pub value: f64,
    /// Unit
    pub unit: String,
    /// Uncertainty if reported
    pub uncertainty: Option<f64>,
    /// Signal-to-background ratio
    pub signal_to_background: Option<f64>,
}

/// Control experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlExperiment {
    /// Control type
    pub control_type: ControlType,
    /// Result
    pub result: ControlResult,
    /// Notes
    pub notes: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlType {
    HydrogenControl,    // H instead of D
    NoTrigger,          // Trigger off
    EmptyCell,          // No fuel
    BackgroundRun,      // Extended background measurement
    DifferentMaterial,  // Different host
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlResult {
    NullAsExpected,     // Control showed no signal (good)
    SignalPersisted,    // Signal present in control (bad)
    Ambiguous,          // Unclear
    NotReported,        // Control not described
}

/// Credibility assessment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Credibility {
    /// Overall score (0-10)
    pub score: f64,
    /// Peer-reviewed?
    pub peer_reviewed: bool,
    /// Multiple detection methods?
    pub multi_method: bool,
    /// Proper controls?
    pub proper_controls: bool,
    /// Statistical rigor?
    pub statistical_rigor: bool,
    /// Author track record
    pub author_reputation: Reputation,
    /// Conflicts of interest
    pub conflicts: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Reputation {
    Established,    // Known expert in field
    Academic,       // University researcher
    National,       // National lab
    Industrial,     // Industry researcher
    Independent,    // Independent researcher
    Unknown,
}

/// Replication status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationStatus {
    /// Number of successful replications
    pub successes: u32,
    /// Number of failed replications
    pub failures: u32,
    /// Number of partial replications
    pub partial: u32,
    /// Notable replication attempts
    pub notes: Vec<String>,
}

/// Experimental conditions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentalConditions {
    /// Temperature (K)
    pub temperature_k: Option<f64>,
    /// Pressure (atm)
    pub pressure_atm: Option<f64>,
    /// Loading ratio (D/M or H/M)
    pub loading_ratio: Option<f64>,
    /// Current density for electrolysis (mA/cm²)
    pub current_density: Option<f64>,
    /// Trigger power (W)
    pub trigger_power_w: Option<f64>,
    /// Sample mass (g)
    pub sample_mass_g: Option<f64>,
    /// Run duration (hours)
    pub duration_hours: Option<f64>,
}

/// Literature database.
pub struct LiteratureDatabase {
    observations: Vec<Observation>,
}

impl LiteratureDatabase {
    /// Create database with canonical observations.
    pub fn canonical() -> Self {
        let mut db = Self { observations: Vec::new() };
        db.load_canonical();
        db
    }

    /// Load canonical observations from literature.
    fn load_canonical(&mut self) {
        // NASA Steinetz 2020 - Key modern result
        self.observations.push(Observation {
            id: "steinetz2020".to_string(),
            author: "Steinetz".to_string(),
            year: 2020,
            venue: "Physical Review C".to_string(),
            doi: Some("10.1103/PhysRevC.101.044610".to_string()),
            host_material: HostMaterial::PdAu,
            fuel: Fuel::Deuterium,
            trigger: TriggerMethod::XRay,
            effects: vec![
                ObservedEffect::Neutrons { rate_per_s: 1e3, energy_mev: Some(2.45) },
            ],
            signal: SignalClaim {
                metric: "neutron rate".to_string(),
                value: 1e3,
                unit: "n/s".to_string(),
                uncertainty: Some(100.0),
                signal_to_background: Some(10.0),
            },
            controls: vec![
                ControlExperiment {
                    control_type: ControlType::HydrogenControl,
                    result: ControlResult::NullAsExpected,
                    notes: "No signal with H".to_string(),
                },
                ControlExperiment {
                    control_type: ControlType::NoTrigger,
                    result: ControlResult::NullAsExpected,
                    notes: "No signal without X-ray".to_string(),
                },
            ],
            credibility: Credibility {
                score: 8.5,
                peer_reviewed: true,
                multi_method: true,
                proper_controls: true,
                statistical_rigor: true,
                author_reputation: Reputation::National,
                conflicts: None,
            },
            replications: ReplicationStatus {
                successes: 0,
                failures: 0,
                partial: 0,
                notes: vec!["No independent replication yet".to_string()],
            },
            conditions: ExperimentalConditions {
                temperature_k: Some(300.0),
                pressure_atm: None,
                loading_ratio: Some(0.7),
                current_density: None,
                trigger_power_w: Some(100.0),
                sample_mass_g: Some(0.5),
                duration_hours: Some(10.0),
            },
            notes: "Key modern result with proper controls. Uses 12 keV X-rays.".to_string(),
        });

        // Fleischmann-Pons 1989 - Original claim
        self.observations.push(Observation {
            id: "fleischmann1989".to_string(),
            author: "Fleischmann".to_string(),
            year: 1989,
            venue: "Journal of Electroanalytical Chemistry".to_string(),
            doi: Some("10.1016/0022-0728(89)80006-3".to_string()),
            host_material: HostMaterial::Palladium,
            fuel: Fuel::Deuterium,
            trigger: TriggerMethod::Electrolysis,
            effects: vec![
                ObservedEffect::ExcessHeat { power_w: 4.0, duration_s: 3600.0 },
            ],
            signal: SignalClaim {
                metric: "excess power".to_string(),
                value: 4.0,
                unit: "W".to_string(),
                uncertainty: Some(1.0),
                signal_to_background: Some(2.0),
            },
            controls: vec![
                ControlExperiment {
                    control_type: ControlType::HydrogenControl,
                    result: ControlResult::NullAsExpected,
                    notes: "Less heat with H2O".to_string(),
                },
            ],
            credibility: Credibility {
                score: 5.0,
                peer_reviewed: true,
                multi_method: false,
                proper_controls: false,
                statistical_rigor: false,
                author_reputation: Reputation::Established,
                conflicts: None,
            },
            replications: ReplicationStatus {
                successes: 50,
                failures: 200,
                partial: 100,
                notes: vec![
                    "Many attempts, inconsistent results".to_string(),
                    "SRI partial success with high loading".to_string(),
                ],
            },
            conditions: ExperimentalConditions {
                temperature_k: Some(300.0),
                pressure_atm: Some(1.0),
                loading_ratio: Some(0.7),
                current_density: Some(500.0),
                trigger_power_w: None,
                sample_mass_g: Some(10.0),
                duration_hours: Some(100.0),
            },
            notes: "Original 'cold fusion' claim. Controversial. Calorimetry questioned.".to_string(),
        });

        // Raiola 2004 - Screening measurement
        self.observations.push(Observation {
            id: "raiola2004".to_string(),
            author: "Raiola".to_string(),
            year: 2004,
            venue: "European Physical Journal A".to_string(),
            doi: Some("10.1140/epja/i2003-10206-6".to_string()),
            host_material: HostMaterial::Palladium,
            fuel: Fuel::Deuterium,
            trigger: TriggerMethod::IonImplantation,
            effects: vec![
                ObservedEffect::ChargedParticles {
                    type_: "protons".to_string(),
                    rate_per_s: 1e4
                },
            ],
            signal: SignalClaim {
                metric: "screening energy".to_string(),
                value: 309.0,
                unit: "eV".to_string(),
                uncertainty: Some(12.0),
                signal_to_background: Some(100.0),
            },
            controls: vec![
                ControlExperiment {
                    control_type: ControlType::DifferentMaterial,
                    result: ControlResult::NullAsExpected,
                    notes: "Different Ue for different metals".to_string(),
                },
            ],
            credibility: Credibility {
                score: 9.0,
                peer_reviewed: true,
                multi_method: false,
                proper_controls: true,
                statistical_rigor: true,
                author_reputation: Reputation::Academic,
                conflicts: None,
            },
            replications: ReplicationStatus {
                successes: 5,
                failures: 0,
                partial: 2,
                notes: vec![
                    "Czerski et al. confirmed".to_string(),
                    "Multiple labs agree on Ue values".to_string(),
                ],
            },
            conditions: ExperimentalConditions {
                temperature_k: Some(300.0),
                pressure_atm: None,
                loading_ratio: Some(0.7),
                current_density: None,
                trigger_power_w: None,
                sample_mass_g: Some(0.1),
                duration_hours: Some(1.0),
            },
            notes: "Clean accelerator measurement of screening. Highly credible.".to_string(),
        });

        // Jones 1989 - Muon catalysis in metal
        self.observations.push(Observation {
            id: "jones1989".to_string(),
            author: "Jones".to_string(),
            year: 1989,
            venue: "Nature".to_string(),
            doi: Some("10.1038/338737a0".to_string()),
            host_material: HostMaterial::Titanium,
            fuel: Fuel::Deuterium,
            trigger: TriggerMethod::Electrolysis,
            effects: vec![
                ObservedEffect::Neutrons { rate_per_s: 1.0, energy_mev: Some(2.45) },
            ],
            signal: SignalClaim {
                metric: "neutron rate".to_string(),
                value: 1.0,
                unit: "n/s".to_string(),
                uncertainty: Some(0.5),
                signal_to_background: Some(2.0),
            },
            controls: vec![
                ControlExperiment {
                    control_type: ControlType::BackgroundRun,
                    result: ControlResult::NullAsExpected,
                    notes: "Background measured".to_string(),
                },
            ],
            credibility: Credibility {
                score: 6.0,
                peer_reviewed: true,
                multi_method: false,
                proper_controls: true,
                statistical_rigor: true,
                author_reputation: Reputation::Academic,
                conflicts: None,
            },
            replications: ReplicationStatus {
                successes: 2,
                failures: 10,
                partial: 5,
                notes: vec!["Mixed results".to_string()],
            },
            conditions: ExperimentalConditions {
                temperature_k: Some(300.0),
                pressure_atm: Some(1.0),
                loading_ratio: Some(1.5),
                current_density: Some(100.0),
                trigger_power_w: None,
                sample_mass_g: Some(5.0),
                duration_hours: Some(50.0),
            },
            notes: "Low-level neutron claim. Ti host instead of Pd.".to_string(),
        });

        // Mizuno 1996 - Japanese excess heat
        self.observations.push(Observation {
            id: "mizuno1996".to_string(),
            author: "Mizuno".to_string(),
            year: 1996,
            venue: "Fusion Technology".to_string(),
            doi: None,
            host_material: HostMaterial::Palladium,
            fuel: Fuel::Deuterium,
            trigger: TriggerMethod::Electrolysis,
            effects: vec![
                ObservedEffect::ExcessHeat { power_w: 50.0, duration_s: 36000.0 },
                ObservedEffect::Transmutation {
                    from: "Cs".to_string(),
                    to: "Pr".to_string(),
                    amount: "trace".to_string()
                },
            ],
            signal: SignalClaim {
                metric: "excess power".to_string(),
                value: 50.0,
                unit: "W".to_string(),
                uncertainty: Some(10.0),
                signal_to_background: Some(5.0),
            },
            controls: vec![],
            credibility: Credibility {
                score: 4.0,
                peer_reviewed: true,
                multi_method: true,
                proper_controls: false,
                statistical_rigor: false,
                author_reputation: Reputation::Academic,
                conflicts: None,
            },
            replications: ReplicationStatus {
                successes: 1,
                failures: 5,
                partial: 3,
                notes: vec!["Some Japanese replications".to_string()],
            },
            conditions: ExperimentalConditions {
                temperature_k: Some(350.0),
                pressure_atm: Some(1.0),
                loading_ratio: Some(0.9),
                current_density: Some(300.0),
                trigger_power_w: None,
                sample_mass_g: Some(20.0),
                duration_hours: Some(100.0),
            },
            notes: "High excess heat claim. Transmutation claims controversial.".to_string(),
        });

        // Arata 2008 - Gas loading
        self.observations.push(Observation {
            id: "arata2008".to_string(),
            author: "Arata".to_string(),
            year: 2008,
            venue: "Journal of High Temperature Society Japan".to_string(),
            doi: None,
            host_material: HostMaterial::Palladium,
            fuel: Fuel::Deuterium,
            trigger: TriggerMethod::GasLoading,
            effects: vec![
                ObservedEffect::ExcessHeat { power_w: 10.0, duration_s: 86400.0 },
                ObservedEffect::Helium4 { amount_atoms: 1e15 },
            ],
            signal: SignalClaim {
                metric: "excess energy".to_string(),
                value: 200.0,
                unit: "kJ".to_string(),
                uncertainty: Some(50.0),
                signal_to_background: Some(3.0),
            },
            controls: vec![
                ControlExperiment {
                    control_type: ControlType::HydrogenControl,
                    result: ControlResult::NullAsExpected,
                    notes: "No He-4 with H".to_string(),
                },
            ],
            credibility: Credibility {
                score: 5.5,
                peer_reviewed: true,
                multi_method: true,
                proper_controls: true,
                statistical_rigor: false,
                author_reputation: Reputation::Established,
                conflicts: None,
            },
            replications: ReplicationStatus {
                successes: 2,
                failures: 3,
                partial: 2,
                notes: vec!["Some replications in Japan".to_string()],
            },
            conditions: ExperimentalConditions {
                temperature_k: Some(400.0),
                pressure_atm: Some(50.0),
                loading_ratio: Some(0.8),
                current_density: None,
                trigger_power_w: None,
                sample_mass_g: Some(5.0),
                duration_hours: Some(24.0),
            },
            notes: "Nano-powder Pd. Gas loading method.".to_string(),
        });

        // Czerski 2004 - Screening in metals
        self.observations.push(Observation {
            id: "czerski2004".to_string(),
            author: "Czerski".to_string(),
            year: 2004,
            venue: "Europhysics Letters".to_string(),
            doi: Some("10.1209/epl/i2004-10095-7".to_string()),
            host_material: HostMaterial::Titanium,
            fuel: Fuel::Deuterium,
            trigger: TriggerMethod::IonImplantation,
            effects: vec![
                ObservedEffect::ChargedParticles {
                    type_: "protons".to_string(),
                    rate_per_s: 1e4,
                },
            ],
            signal: SignalClaim {
                metric: "screening energy".to_string(),
                value: 363.0,
                unit: "eV".to_string(),
                uncertainty: Some(25.0),
                signal_to_background: Some(50.0),
            },
            controls: vec![
                ControlExperiment {
                    control_type: ControlType::DifferentMaterial,
                    result: ControlResult::NullAsExpected,
                    notes: "Different Ue for Ta, Al".to_string(),
                },
            ],
            credibility: Credibility {
                score: 8.5,
                peer_reviewed: true,
                multi_method: false,
                proper_controls: true,
                statistical_rigor: true,
                author_reputation: Reputation::Academic,
                conflicts: None,
            },
            replications: ReplicationStatus {
                successes: 3,
                failures: 1,
                partial: 1,
                notes: vec!["Confirmed by multiple accelerator labs".to_string()],
            },
            conditions: ExperimentalConditions {
                temperature_k: Some(300.0),
                pressure_atm: None,
                loading_ratio: Some(1.5), // TiD2
                current_density: None,
                trigger_power_w: None,
                sample_mass_g: Some(0.2),
                duration_hours: Some(2.0),
            },
            notes: "Confirms anomalously large screening in metals. Ti shows Ue = 363 eV.".to_string(),
        });

        // McKubre 1998 - SRI excess heat
        self.observations.push(Observation {
            id: "mckubre1998".to_string(),
            author: "McKubre".to_string(),
            year: 1998,
            venue: "ICCF-7 Proceedings".to_string(),
            doi: None,
            host_material: HostMaterial::Palladium,
            fuel: Fuel::Deuterium,
            trigger: TriggerMethod::Electrolysis,
            effects: vec![
                ObservedEffect::ExcessHeat { power_w: 20.0, duration_s: 86400.0 },
            ],
            signal: SignalClaim {
                metric: "excess power".to_string(),
                value: 20.0,
                unit: "W".to_string(),
                uncertainty: Some(5.0),
                signal_to_background: Some(4.0),
            },
            controls: vec![
                ControlExperiment {
                    control_type: ControlType::HydrogenControl,
                    result: ControlResult::NullAsExpected,
                    notes: "H2O showed no excess".to_string(),
                },
            ],
            credibility: Credibility {
                score: 6.5,
                peer_reviewed: false, // Conference proceedings
                multi_method: true,
                proper_controls: true,
                statistical_rigor: true,
                author_reputation: Reputation::National, // SRI
                conflicts: Some("EPRI funding".to_string()),
            },
            replications: ReplicationStatus {
                successes: 10,
                failures: 20,
                partial: 15,
                notes: vec![
                    "Required D/Pd > 0.9".to_string(),
                    "Highly variable results".to_string(),
                ],
            },
            conditions: ExperimentalConditions {
                temperature_k: Some(320.0),
                pressure_atm: Some(1.0),
                loading_ratio: Some(0.95),
                current_density: Some(200.0),
                trigger_power_w: None,
                sample_mass_g: Some(15.0),
                duration_hours: Some(200.0),
            },
            notes: "SRI systematic studies. Shows loading threshold D/Pd > 0.9.".to_string(),
        });

        // Storms 2007 - Review/meta-analysis
        self.observations.push(Observation {
            id: "storms2007".to_string(),
            author: "Storms".to_string(),
            year: 2007,
            venue: "The Science of Low Energy Nuclear Reaction".to_string(),
            doi: None,
            host_material: HostMaterial::Palladium,
            fuel: Fuel::Deuterium,
            trigger: TriggerMethod::Electrolysis,
            effects: vec![
                ObservedEffect::ExcessHeat { power_w: 10.0, duration_s: 1e6 },
                ObservedEffect::Helium4 { amount_atoms: 1e14 },
            ],
            signal: SignalClaim {
                metric: "He-4/heat correlation".to_string(),
                value: 24.0,
                unit: "MeV/He".to_string(),
                uncertainty: Some(5.0),
                signal_to_background: None,
            },
            controls: vec![],
            credibility: Credibility {
                score: 5.0,
                peer_reviewed: false, // Book
                multi_method: true,
                proper_controls: false,
                statistical_rigor: false,
                author_reputation: Reputation::Independent,
                conflicts: Some("LENR advocate".to_string()),
            },
            replications: ReplicationStatus {
                successes: 0,
                failures: 0,
                partial: 0,
                notes: vec!["Review paper, not original experiment".to_string()],
            },
            conditions: ExperimentalConditions {
                temperature_k: None,
                pressure_atm: None,
                loading_ratio: None,
                current_density: None,
                trigger_power_w: None,
                sample_mass_g: None,
                duration_hours: None,
            },
            notes: "Meta-analysis claiming He-4/heat correlation = 24 MeV (D+D→He4 Q-value).".to_string(),
        });

        // Iwamura 2002 - Transmutation claims
        self.observations.push(Observation {
            id: "iwamura2002".to_string(),
            author: "Iwamura".to_string(),
            year: 2002,
            venue: "Japanese Journal of Applied Physics".to_string(),
            doi: Some("10.1143/JJAP.41.4642".to_string()),
            host_material: HostMaterial::Palladium,
            fuel: Fuel::Deuterium,
            trigger: TriggerMethod::GasLoading,
            effects: vec![
                ObservedEffect::Transmutation {
                    from: "Cs".to_string(),
                    to: "Pr".to_string(),
                    amount: "10^12 atoms".to_string(),
                },
                ObservedEffect::Transmutation {
                    from: "Sr".to_string(),
                    to: "Mo".to_string(),
                    amount: "10^11 atoms".to_string(),
                },
            ],
            signal: SignalClaim {
                metric: "transmutation rate".to_string(),
                value: 1e12,
                unit: "atoms".to_string(),
                uncertainty: None,
                signal_to_background: Some(100.0),
            },
            controls: vec![
                ControlExperiment {
                    control_type: ControlType::HydrogenControl,
                    result: ControlResult::NullAsExpected,
                    notes: "No transmutation with H".to_string(),
                },
            ],
            credibility: Credibility {
                score: 4.5,
                peer_reviewed: true,
                multi_method: true, // XPS, SIMS
                proper_controls: true,
                statistical_rigor: false,
                author_reputation: Reputation::Industrial, // Mitsubishi Heavy Industries
                conflicts: Some("MHI funding".to_string()),
            },
            replications: ReplicationStatus {
                successes: 1,
                failures: 5,
                partial: 2,
                notes: vec![
                    "Toyota partially replicated".to_string(),
                    "Most attempts failed".to_string(),
                ],
            },
            conditions: ExperimentalConditions {
                temperature_k: Some(343.0),
                pressure_atm: Some(1.0),
                loading_ratio: Some(0.7),
                current_density: None,
                trigger_power_w: None,
                sample_mass_g: Some(1.0),
                duration_hours: Some(100.0),
            },
            notes: "Controversial transmutation claims. Requires Cs or Sr surface layer.".to_string(),
        });

        // Huizenga 1993 - Negative review
        self.observations.push(Observation {
            id: "huizenga1993".to_string(),
            author: "Huizenga".to_string(),
            year: 1993,
            venue: "Cold Fusion: The Scientific Fiasco of the Century".to_string(),
            doi: None,
            host_material: HostMaterial::Palladium,
            fuel: Fuel::Deuterium,
            trigger: TriggerMethod::Electrolysis,
            effects: vec![],
            signal: SignalClaim {
                metric: "no effect".to_string(),
                value: 0.0,
                unit: "N/A".to_string(),
                uncertainty: None,
                signal_to_background: None,
            },
            controls: vec![],
            credibility: Credibility {
                score: 7.0,
                peer_reviewed: false,
                multi_method: false,
                proper_controls: false,
                statistical_rigor: false,
                author_reputation: Reputation::Established, // DOE panel chair
                conflicts: Some("DOE panel chair - skeptical position".to_string()),
            },
            replications: ReplicationStatus {
                successes: 0,
                failures: 0,
                partial: 0,
                notes: vec!["Review/critique, not experiment".to_string()],
            },
            conditions: ExperimentalConditions {
                temperature_k: None,
                pressure_atm: None,
                loading_ratio: None,
                current_density: None,
                trigger_power_w: None,
                sample_mass_g: None,
                duration_hours: None,
            },
            notes: "Skeptical review. Documents failed replications at major labs.".to_string(),
        });
    }

    /// Get all observations.
    pub fn all(&self) -> &[Observation] {
        &self.observations
    }

    /// Filter by host material.
    pub fn by_material(&self, material: HostMaterial) -> Vec<&Observation> {
        self.observations.iter()
            .filter(|o| o.host_material == material)
            .collect()
    }

    /// Filter by trigger method.
    pub fn by_trigger(&self, trigger: TriggerMethod) -> Vec<&Observation> {
        self.observations.iter()
            .filter(|o| o.trigger == trigger)
            .collect()
    }

    /// Filter by minimum credibility score.
    pub fn by_credibility(&self, min_score: f64) -> Vec<&Observation> {
        self.observations.iter()
            .filter(|o| o.credibility.score >= min_score)
            .collect()
    }

    /// Find observations with neutron detection.
    pub fn with_neutrons(&self) -> Vec<&Observation> {
        self.observations.iter()
            .filter(|o| o.effects.iter().any(|e| matches!(e, ObservedEffect::Neutrons { .. })))
            .collect()
    }

    /// Find observations with excess heat.
    pub fn with_excess_heat(&self) -> Vec<&Observation> {
        self.observations.iter()
            .filter(|o| o.effects.iter().any(|e| matches!(e, ObservedEffect::ExcessHeat { .. })))
            .collect()
    }

    /// Compute pattern statistics.
    pub fn compute_patterns(&self) -> PatternAnalysis {
        let total = self.observations.len();

        let mut material_counts: HashMap<HostMaterial, usize> = HashMap::new();
        let mut trigger_counts: HashMap<TriggerMethod, usize> = HashMap::new();
        let mut effect_counts: HashMap<String, usize> = HashMap::new();

        let mut total_credibility = 0.0;
        let mut neutron_rates: Vec<f64> = Vec::new();
        let mut heat_powers: Vec<f64> = Vec::new();

        for obs in &self.observations {
            *material_counts.entry(obs.host_material).or_insert(0) += 1;
            *trigger_counts.entry(obs.trigger).or_insert(0) += 1;
            total_credibility += obs.credibility.score;

            for effect in &obs.effects {
                let name = match effect {
                    ObservedEffect::Neutrons { rate_per_s, .. } => {
                        neutron_rates.push(*rate_per_s);
                        "Neutrons"
                    }
                    ObservedEffect::ExcessHeat { power_w, .. } => {
                        heat_powers.push(*power_w);
                        "ExcessHeat"
                    }
                    ObservedEffect::ChargedParticles { .. } => "ChargedParticles",
                    ObservedEffect::Gamma { .. } => "Gamma",
                    ObservedEffect::Transmutation { .. } => "Transmutation",
                    ObservedEffect::Tritium { .. } => "Tritium",
                    ObservedEffect::Helium4 { .. } => "Helium4",
                };
                *effect_counts.entry(name.to_string()).or_insert(0) += 1;
            }
        }

        PatternAnalysis {
            total_observations: total,
            material_distribution: material_counts,
            trigger_distribution: trigger_counts,
            effect_distribution: effect_counts,
            average_credibility: total_credibility / total as f64,
            neutron_rate_range: if neutron_rates.is_empty() {
                None
            } else {
                Some((
                    neutron_rates.iter().cloned().fold(f64::INFINITY, f64::min),
                    neutron_rates.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                ))
            },
            heat_power_range: if heat_powers.is_empty() {
                None
            } else {
                Some((
                    heat_powers.iter().cloned().fold(f64::INFINITY, f64::min),
                    heat_powers.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                ))
            },
        }
    }
}

/// Pattern analysis results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternAnalysis {
    pub total_observations: usize,
    pub material_distribution: HashMap<HostMaterial, usize>,
    pub trigger_distribution: HashMap<TriggerMethod, usize>,
    pub effect_distribution: HashMap<String, usize>,
    pub average_credibility: f64,
    pub neutron_rate_range: Option<(f64, f64)>,
    pub heat_power_range: Option<(f64, f64)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_database_creation() {
        let db = LiteratureDatabase::canonical();
        assert!(!db.all().is_empty());
    }

    #[test]
    fn test_filter_by_material() {
        let db = LiteratureDatabase::canonical();
        let pd_obs = db.by_material(HostMaterial::Palladium);
        assert!(!pd_obs.is_empty());
    }

    #[test]
    fn test_filter_by_credibility() {
        let db = LiteratureDatabase::canonical();
        let high_cred = db.by_credibility(7.0);
        for obs in high_cred {
            assert!(obs.credibility.score >= 7.0);
        }
    }

    #[test]
    fn test_pattern_analysis() {
        let db = LiteratureDatabase::canonical();
        let patterns = db.compute_patterns();
        assert!(patterns.total_observations > 0);
        assert!(patterns.average_credibility > 0.0);
    }

    #[test]
    fn test_neutron_observations() {
        let db = LiteratureDatabase::canonical();
        let neutron_obs = db.with_neutrons();
        assert!(!neutron_obs.is_empty());
        // NASA should be in there
        assert!(neutron_obs.iter().any(|o| o.id == "steinetz2020"));
    }
}
