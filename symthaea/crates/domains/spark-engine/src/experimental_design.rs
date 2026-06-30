// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Experimental Design Tool (Option C)
//!
//! Tools for designing discriminating experiments to test LCF hypotheses.
//! The goal is to design experiments that can distinguish between:
//! 1. Real anomalous fusion enhancement
//! 2. Systematic measurement errors
//! 3. Different proposed mechanisms

use crate::hypothesis_models::HypothesisType;
use serde::{Deserialize, Serialize};

/// A designed experiment to test LCF hypotheses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentDesign {
    /// Experiment name
    pub name: String,
    /// What question this experiment answers
    pub research_question: String,
    /// Hypotheses this discriminates between
    pub hypotheses_tested: Vec<HypothesisType>,
    /// Experimental setup
    pub setup: ExperimentalSetup,
    /// Expected outcomes for each hypothesis
    pub expected_outcomes: Vec<ExpectedOutcome>,
    /// Required instrumentation
    pub instrumentation: Vec<Instrument>,
    /// Estimated cost ($)
    pub estimated_cost_usd: f64,
    /// Estimated duration (months)
    pub duration_months: f64,
    /// Priority score (0-1)
    pub priority: f64,
    /// Critical success criteria
    pub success_criteria: Vec<String>,
}

/// Experimental setup specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentalSetup {
    /// Host material
    pub host_material: String,
    /// Target D/M loading ratio
    pub loading_ratio: f64,
    /// Sample geometry
    pub sample_geometry: SampleGeometry,
    /// Trigger mechanism
    pub trigger: TriggerSpec,
    /// Temperature range (K)
    pub temperature_range: (f64, f64),
    /// Control experiments
    pub controls: Vec<ControlExperiment>,
}

/// Sample geometry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleGeometry {
    /// Shape
    pub shape: String,
    /// Dimensions (cm)
    pub dimensions_cm: Vec<f64>,
    /// Active volume (cm³)
    pub active_volume_cm3: f64,
    /// Surface area (cm²)
    pub surface_area_cm2: f64,
}

/// Trigger specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerSpec {
    /// Trigger type
    pub trigger_type: String,
    /// Energy/power
    pub intensity: f64,
    /// Units for intensity
    pub intensity_units: String,
    /// Pulse duration (if applicable)
    pub pulse_duration_s: Option<f64>,
    /// Repetition rate (Hz, if applicable)
    pub repetition_rate_hz: Option<f64>,
}

/// Control experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlExperiment {
    /// Control type
    pub control_type: ControlType,
    /// Description
    pub description: String,
    /// What this rules out
    pub rules_out: String,
}

/// Types of control experiments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlType {
    /// H instead of D (rules out nuclear fusion)
    HydrogenControl,
    /// Unloaded host (rules out host material effects)
    UnloadedControl,
    /// No trigger (rules out trigger-independent effects)
    NoTriggerControl,
    /// Different geometry (rules out geometry-specific effects)
    GeometryControl,
    /// Background measurement
    BackgroundControl,
}

/// Expected outcome for a hypothesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedOutcome {
    /// Hypothesis this applies to
    pub hypothesis: HypothesisType,
    /// Predicted neutron rate (n/s)
    pub predicted_rate_range: (f64, f64),
    /// Predicted neutron energy (MeV)
    pub predicted_energy_mev: f64,
    /// Predicted temperature dependence
    pub temperature_dependence: String,
    /// Predicted trigger dependence
    pub trigger_dependence: String,
    /// Distinguishing signatures
    pub signatures: Vec<String>,
}

/// Instrumentation requirement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instrument {
    /// Instrument type
    pub instrument_type: InstrumentType,
    /// Required specifications
    pub specifications: String,
    /// Estimated cost ($)
    pub cost_usd: f64,
    /// Availability
    pub availability: Availability,
}

/// Instrument types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InstrumentType {
    /// Neutron detector
    NeutronDetector,
    /// Gamma spectrometer
    GammaSpectrometer,
    /// Charged particle detector
    ChargedParticleDetector,
    /// X-ray source
    XRaySource,
    /// Temperature sensor
    TemperatureSensor,
    /// Calorimeter
    Calorimeter,
    /// Mass spectrometer (for He detection)
    MassSpectrometer,
    /// Loading monitor
    LoadingMonitor,
}

/// Availability status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Availability {
    /// Commercially available
    Commercial,
    /// Available at national labs
    NationalLab,
    /// Custom fabrication required
    Custom,
    /// Research prototype only
    Prototype,
}

/// Experiment designer.
pub struct ExperimentDesigner;

impl ExperimentDesigner {
    /// Design a complete experimental program.
    pub fn design_program() -> ExperimentalProgram {
        ExperimentalProgram {
            name: "LCF Anomaly Investigation Program".to_string(),
            phases: vec![
                Self::design_phase_1_validation(),
                Self::design_phase_2_mechanism(),
                Self::design_phase_3_optimization(),
            ],
            total_cost_usd: 700_000.0,
            total_duration_months: 30.0,
        }
    }

    /// Phase 1: Validate the anomaly exists.
    fn design_phase_1_validation() -> ExperimentPhase {
        ExperimentPhase {
            name: "Validation".to_string(),
            goal: "Confirm anomalous neutron production is real and from D-D fusion".to_string(),
            experiments: vec![
                Self::nasa_replication_experiment(),
                Self::hydrogen_control_experiment(),
                Self::neutron_spectroscopy_experiment(),
            ],
            duration_months: 6.0,
            cost_usd: 200_000.0,
            success_criteria: vec![
                "Observe >100 n/s above background with D loading".to_string(),
                "H control shows <1 n/s (background level)".to_string(),
                "Neutron energy consistent with 2.45 MeV D-D".to_string(),
            ],
            go_nogo: "All three criteria must be met to proceed to Phase 2".to_string(),
        }
    }

    /// Phase 2: Identify the mechanism.
    fn design_phase_2_mechanism() -> ExperimentPhase {
        ExperimentPhase {
            name: "Mechanism Identification".to_string(),
            goal: "Discriminate between hot spot, phonon cascade, and screening hypotheses"
                .to_string(),
            experiments: vec![
                Self::temperature_mapping_experiment(),
                Self::acoustic_drive_experiment(),
                Self::host_material_comparison(),
            ],
            duration_months: 12.0,
            cost_usd: 300_000.0,
            success_criteria: vec![
                "Identify which hypothesis best matches observed dependencies".to_string(),
                "Quantify enhancement factor to within 2 orders of magnitude".to_string(),
            ],
            go_nogo: "Must identify a plausible mechanism to proceed".to_string(),
        }
    }

    /// Phase 3: Optimize for applications.
    fn design_phase_3_optimization() -> ExperimentPhase {
        ExperimentPhase {
            name: "Optimization".to_string(),
            goal: "Maximize neutron output for practical applications".to_string(),
            experiments: vec![
                Self::scaling_experiment(),
                Self::lifetime_experiment(),
                Self::efficiency_experiment(),
            ],
            duration_months: 12.0,
            cost_usd: 200_000.0,
            success_criteria: vec![
                "Achieve >10⁶ n/s sustained output".to_string(),
                "Demonstrate >1000 hour operational lifetime".to_string(),
                "Document reproducible fabrication procedure".to_string(),
            ],
            go_nogo: "Proceed to commercialization if goals met".to_string(),
        }
    }

    /// NASA replication experiment.
    pub fn nasa_replication_experiment() -> ExperimentDesign {
        ExperimentDesign {
            name: "NASA LCF Replication".to_string(),
            research_question: "Can the NASA Steinetz 2020 results be replicated?".to_string(),
            hypotheses_tested: vec![
                HypothesisType::HotSpots,
                HypothesisType::PhononCascade,
                HypothesisType::SuperScreening,
            ],
            setup: ExperimentalSetup {
                host_material: "ErD2 on TiD2".to_string(),
                loading_ratio: 0.7,
                sample_geometry: SampleGeometry {
                    shape: "Thin film stack".to_string(),
                    dimensions_cm: vec![1.0, 1.0, 0.0025], // 25 μm thick
                    active_volume_cm3: 0.0025,
                    surface_area_cm2: 2.0,
                },
                trigger: TriggerSpec {
                    trigger_type: "Bremsstrahlung X-ray".to_string(),
                    intensity: 100.0,
                    intensity_units: "kV".to_string(),
                    pulse_duration_s: None,
                    repetition_rate_hz: None,
                },
                temperature_range: (250.0, 350.0),
                controls: vec![
                    ControlExperiment {
                        control_type: ControlType::HydrogenControl,
                        description: "Replace D with H in identical setup".to_string(),
                        rules_out: "Non-nuclear neutron sources".to_string(),
                    },
                    ControlExperiment {
                        control_type: ControlType::NoTriggerControl,
                        description: "Loaded sample without X-ray trigger".to_string(),
                        rules_out: "Spontaneous fusion".to_string(),
                    },
                ],
            },
            expected_outcomes: vec![
                ExpectedOutcome {
                    hypothesis: HypothesisType::HotSpots,
                    predicted_rate_range: (100.0, 10000.0),
                    predicted_energy_mev: 2.45,
                    temperature_dependence: "Weak - hot spots dominate".to_string(),
                    trigger_dependence: "Strong - proportional to X-ray power".to_string(),
                    signatures: vec![
                        "Rate scales with X-ray intensity".to_string(),
                        "Burst structure following trigger".to_string(),
                    ],
                },
                ExpectedOutcome {
                    hypothesis: HypothesisType::PhononCascade,
                    predicted_rate_range: (10.0, 1000.0),
                    predicted_energy_mev: 2.45,
                    temperature_dependence: "Moderate - phonon population matters".to_string(),
                    trigger_dependence: "Moderate - phonon excitation".to_string(),
                    signatures: vec![
                        "Resonances at phonon frequencies".to_string(),
                        "Enhancement with acoustic drive".to_string(),
                    ],
                },
            ],
            instrumentation: vec![
                Instrument {
                    instrument_type: InstrumentType::NeutronDetector,
                    specifications: "³He proportional counter, >10% efficiency at 2.45 MeV"
                        .to_string(),
                    cost_usd: 15000.0,
                    availability: Availability::Commercial,
                },
                Instrument {
                    instrument_type: InstrumentType::XRaySource,
                    specifications: "100 kV, 10 mA tungsten target".to_string(),
                    cost_usd: 25000.0,
                    availability: Availability::Commercial,
                },
                Instrument {
                    instrument_type: InstrumentType::LoadingMonitor,
                    specifications: "4-probe resistance for D/M ratio".to_string(),
                    cost_usd: 5000.0,
                    availability: Availability::Commercial,
                },
            ],
            estimated_cost_usd: 75000.0,
            duration_months: 3.0,
            priority: 1.0,
            success_criteria: vec![
                "Observe neutrons >10× background with D".to_string(),
                "H control consistent with background".to_string(),
                "Rate reproducible across 3+ runs".to_string(),
            ],
        }
    }

    /// Hydrogen control experiment.
    pub fn hydrogen_control_experiment() -> ExperimentDesign {
        ExperimentDesign {
            name: "Hydrogen Control".to_string(),
            research_question: "Is the neutron signal from D-D fusion or background?".to_string(),
            hypotheses_tested: vec![],
            setup: ExperimentalSetup {
                host_material: "ErH2 on TiH2".to_string(),
                loading_ratio: 0.7,
                sample_geometry: SampleGeometry {
                    shape: "Thin film stack".to_string(),
                    dimensions_cm: vec![1.0, 1.0, 0.0025],
                    active_volume_cm3: 0.0025,
                    surface_area_cm2: 2.0,
                },
                trigger: TriggerSpec {
                    trigger_type: "Bremsstrahlung X-ray".to_string(),
                    intensity: 100.0,
                    intensity_units: "kV".to_string(),
                    pulse_duration_s: None,
                    repetition_rate_hz: None,
                },
                temperature_range: (250.0, 350.0),
                controls: vec![],
            },
            expected_outcomes: vec![ExpectedOutcome {
                hypothesis: HypothesisType::HotSpots, // Placeholder
                predicted_rate_range: (0.0, 1.0),     // Background only
                predicted_energy_mev: 0.0,            // No fusion
                temperature_dependence: "None".to_string(),
                trigger_dependence: "None - only background".to_string(),
                signatures: vec!["No neutrons above background".to_string()],
            }],
            instrumentation: vec![Instrument {
                instrument_type: InstrumentType::NeutronDetector,
                specifications: "Same as D experiment for direct comparison".to_string(),
                cost_usd: 0.0, // Shared with D experiment
                availability: Availability::Commercial,
            }],
            estimated_cost_usd: 10000.0, // Just sample fabrication
            duration_months: 1.0,
            priority: 1.0,
            success_criteria: vec![
                "Neutron rate consistent with background (<1 n/s)".to_string(),
                "No correlation with X-ray trigger".to_string(),
            ],
        }
    }

    /// Neutron spectroscopy experiment.
    pub fn neutron_spectroscopy_experiment() -> ExperimentDesign {
        ExperimentDesign {
            name: "Neutron Energy Spectroscopy".to_string(),
            research_question: "Is the neutron energy 2.45 MeV (D-D fusion)?".to_string(),
            hypotheses_tested: vec![],
            setup: ExperimentalSetup {
                host_material: "ErD2 on TiD2".to_string(),
                loading_ratio: 0.7,
                sample_geometry: SampleGeometry {
                    shape: "Thin film stack".to_string(),
                    dimensions_cm: vec![1.0, 1.0, 0.0025],
                    active_volume_cm3: 0.0025,
                    surface_area_cm2: 2.0,
                },
                trigger: TriggerSpec {
                    trigger_type: "Bremsstrahlung X-ray".to_string(),
                    intensity: 100.0,
                    intensity_units: "kV".to_string(),
                    pulse_duration_s: None,
                    repetition_rate_hz: None,
                },
                temperature_range: (300.0, 300.0),
                controls: vec![],
            },
            expected_outcomes: vec![ExpectedOutcome {
                hypothesis: HypothesisType::HotSpots,
                predicted_rate_range: (100.0, 10000.0),
                predicted_energy_mev: 2.45,
                temperature_dependence: "N/A".to_string(),
                trigger_dependence: "N/A".to_string(),
                signatures: vec![
                    "Sharp peak at 2.45 MeV".to_string(),
                    "No 14.1 MeV peak (no D-T)".to_string(),
                ],
            }],
            instrumentation: vec![Instrument {
                instrument_type: InstrumentType::NeutronDetector,
                specifications: "Time-of-flight spectrometer, ΔE/E < 10%".to_string(),
                cost_usd: 50000.0,
                availability: Availability::NationalLab,
            }],
            estimated_cost_usd: 50000.0,
            duration_months: 2.0,
            priority: 0.95,
            success_criteria: vec![
                "Energy peak at 2.45 ± 0.3 MeV".to_string(),
                "FWHM consistent with thermal broadening".to_string(),
            ],
        }
    }

    /// Temperature mapping experiment.
    pub fn temperature_mapping_experiment() -> ExperimentDesign {
        ExperimentDesign {
            name: "Temperature Dependence Mapping".to_string(),
            research_question: "How does neutron rate depend on lattice temperature?".to_string(),
            hypotheses_tested: vec![
                HypothesisType::HotSpots,
                HypothesisType::PhononCascade,
                HypothesisType::SuperScreening,
            ],
            setup: ExperimentalSetup {
                host_material: "PdD".to_string(),
                loading_ratio: 0.7,
                sample_geometry: SampleGeometry {
                    shape: "Disk".to_string(),
                    dimensions_cm: vec![1.0, 0.1], // 1 cm dia, 1 mm thick
                    active_volume_cm3: 0.08,
                    surface_area_cm2: 3.5,
                },
                trigger: TriggerSpec {
                    trigger_type: "Bremsstrahlung X-ray".to_string(),
                    intensity: 100.0,
                    intensity_units: "kV".to_string(),
                    pulse_duration_s: None,
                    repetition_rate_hz: None,
                },
                temperature_range: (77.0, 600.0), // LN2 to D desorption
                controls: vec![ControlExperiment {
                    control_type: ControlType::HydrogenControl,
                    description: "PdH at same temperatures".to_string(),
                    rules_out: "Non-nuclear temperature effects".to_string(),
                }],
            },
            expected_outcomes: vec![
                ExpectedOutcome {
                    hypothesis: HypothesisType::HotSpots,
                    predicted_rate_range: (10.0, 10000.0),
                    predicted_energy_mev: 2.45,
                    temperature_dependence: "Weak - hot spots override bulk T".to_string(),
                    trigger_dependence: "Strong".to_string(),
                    signatures: vec![
                        "Rate nearly constant with bulk T".to_string(),
                        "Slight decrease at high T (less contrast)".to_string(),
                    ],
                },
                ExpectedOutcome {
                    hypothesis: HypothesisType::PhononCascade,
                    predicted_rate_range: (1.0, 1000.0),
                    predicted_energy_mev: 2.45,
                    temperature_dependence: "Strong - phonon population scales with T".to_string(),
                    trigger_dependence: "Moderate".to_string(),
                    signatures: vec![
                        "Rate increases with T (more phonons)".to_string(),
                        "Possible plateau above Debye T".to_string(),
                    ],
                },
                ExpectedOutcome {
                    hypothesis: HypothesisType::SuperScreening,
                    predicted_rate_range: (1e-10, 1.0),
                    predicted_energy_mev: 2.45,
                    temperature_dependence: "Gamow scaling: steep increase with T".to_string(),
                    trigger_dependence: "Weak".to_string(),
                    signatures: vec![
                        "Arrhenius-like T dependence".to_string(),
                        "Negligible rate at 77K".to_string(),
                    ],
                },
            ],
            instrumentation: vec![
                Instrument {
                    instrument_type: InstrumentType::NeutronDetector,
                    specifications: "³He detector with thermal shielding".to_string(),
                    cost_usd: 20000.0,
                    availability: Availability::Commercial,
                },
                Instrument {
                    instrument_type: InstrumentType::TemperatureSensor,
                    specifications: "Pt RTD, 77-600K, ±0.1K".to_string(),
                    cost_usd: 500.0,
                    availability: Availability::Commercial,
                },
                Instrument {
                    instrument_type: InstrumentType::Calorimeter,
                    specifications: "Cryostat with heater, LN2 to 600K".to_string(),
                    cost_usd: 30000.0,
                    availability: Availability::Commercial,
                },
            ],
            estimated_cost_usd: 80000.0,
            duration_months: 4.0,
            priority: 0.9,
            success_criteria: vec![
                "Map rate vs T at >10 temperatures".to_string(),
                "Determine functional form (constant, linear, Arrhenius)".to_string(),
                "Uncertainty <50% at each point".to_string(),
            ],
        }
    }

    /// Acoustic drive experiment (tests phonon hypothesis).
    pub fn acoustic_drive_experiment() -> ExperimentDesign {
        ExperimentDesign {
            name: "Acoustic/Phonon Drive".to_string(),
            research_question: "Does acoustic excitation enhance fusion rate?".to_string(),
            hypotheses_tested: vec![HypothesisType::PhononCascade],
            setup: ExperimentalSetup {
                host_material: "PdD".to_string(),
                loading_ratio: 0.7,
                sample_geometry: SampleGeometry {
                    shape: "Disk on piezo".to_string(),
                    dimensions_cm: vec![1.0, 0.05],
                    active_volume_cm3: 0.04,
                    surface_area_cm2: 2.0,
                },
                trigger: TriggerSpec {
                    trigger_type: "Ultrasonic piezo".to_string(),
                    intensity: 10.0,
                    intensity_units: "W".to_string(),
                    pulse_duration_s: None,
                    repetition_rate_hz: Some(1e6), // 1 MHz
                },
                temperature_range: (300.0, 300.0),
                controls: vec![ControlExperiment {
                    control_type: ControlType::NoTriggerControl,
                    description: "Same sample without acoustic drive".to_string(),
                    rules_out: "Baseline fusion rate".to_string(),
                }],
            },
            expected_outcomes: vec![ExpectedOutcome {
                hypothesis: HypothesisType::PhononCascade,
                predicted_rate_range: (1.0, 10000.0),
                predicted_energy_mev: 2.45,
                temperature_dependence: "N/A - fixed T".to_string(),
                trigger_dependence: "Strong - rate scales with acoustic power".to_string(),
                signatures: vec![
                    "Enhancement at phonon resonances".to_string(),
                    "Rate proportional to acoustic power".to_string(),
                    "Frequency-dependent response".to_string(),
                ],
            }],
            instrumentation: vec![Instrument {
                instrument_type: InstrumentType::NeutronDetector,
                specifications: "Fast response for time correlation".to_string(),
                cost_usd: 20000.0,
                availability: Availability::Commercial,
            }],
            estimated_cost_usd: 40000.0,
            duration_months: 3.0,
            priority: 0.8,
            success_criteria: vec![
                "Measure rate vs acoustic frequency".to_string(),
                "Measure rate vs acoustic power".to_string(),
                "Identify any resonance structure".to_string(),
            ],
        }
    }

    /// Host material comparison.
    pub fn host_material_comparison() -> ExperimentDesign {
        ExperimentDesign {
            name: "Host Material Survey".to_string(),
            research_question: "Which host material gives highest fusion rate?".to_string(),
            hypotheses_tested: vec![HypothesisType::SuperScreening, HypothesisType::HotSpots],
            setup: ExperimentalSetup {
                host_material: "Pd, Ti, Ni, Er (compare)".to_string(),
                loading_ratio: 0.7,
                sample_geometry: SampleGeometry {
                    shape: "Thin film".to_string(),
                    dimensions_cm: vec![1.0, 1.0, 0.0025],
                    active_volume_cm3: 0.0025,
                    surface_area_cm2: 2.0,
                },
                trigger: TriggerSpec {
                    trigger_type: "Bremsstrahlung X-ray".to_string(),
                    intensity: 100.0,
                    intensity_units: "kV".to_string(),
                    pulse_duration_s: None,
                    repetition_rate_hz: None,
                },
                temperature_range: (300.0, 300.0),
                controls: vec![],
            },
            expected_outcomes: vec![
                ExpectedOutcome {
                    hypothesis: HypothesisType::SuperScreening,
                    predicted_rate_range: (0.0, 10000.0),
                    predicted_energy_mev: 2.45,
                    temperature_dependence: "N/A".to_string(),
                    trigger_dependence: "N/A".to_string(),
                    signatures: vec![
                        "Rate correlates with measured screening (Er > Pd > Ti > Ni)".to_string(),
                    ],
                },
                ExpectedOutcome {
                    hypothesis: HypothesisType::HotSpots,
                    predicted_rate_range: (0.0, 10000.0),
                    predicted_energy_mev: 2.45,
                    temperature_dependence: "N/A".to_string(),
                    trigger_dependence: "N/A".to_string(),
                    signatures: vec![
                        "Rate correlates with X-ray absorption".to_string(),
                        "High-Z materials (Er) show more enhancement".to_string(),
                    ],
                },
            ],
            instrumentation: vec![Instrument {
                instrument_type: InstrumentType::NeutronDetector,
                specifications: "Same detector for all materials".to_string(),
                cost_usd: 15000.0,
                availability: Availability::Commercial,
            }],
            estimated_cost_usd: 60000.0,
            duration_months: 4.0,
            priority: 0.85,
            success_criteria: vec![
                "Rank materials by fusion rate".to_string(),
                "Correlate with screening energy".to_string(),
                "Correlate with X-ray absorption".to_string(),
            ],
        }
    }

    /// Scaling experiment.
    pub fn scaling_experiment() -> ExperimentDesign {
        ExperimentDesign {
            name: "Scale-Up Study".to_string(),
            research_question: "How does neutron rate scale with active volume?".to_string(),
            hypotheses_tested: vec![],
            setup: ExperimentalSetup {
                host_material: "Best performer from survey".to_string(),
                loading_ratio: 0.7,
                sample_geometry: SampleGeometry {
                    shape: "Variable (1-100 cm³)".to_string(),
                    dimensions_cm: vec![],
                    active_volume_cm3: 50.0, // Average
                    surface_area_cm2: 100.0,
                },
                trigger: TriggerSpec {
                    trigger_type: "Optimized from Phase 2".to_string(),
                    intensity: 0.0,
                    intensity_units: "TBD".to_string(),
                    pulse_duration_s: None,
                    repetition_rate_hz: None,
                },
                temperature_range: (300.0, 300.0),
                controls: vec![],
            },
            expected_outcomes: vec![],
            instrumentation: vec![Instrument {
                instrument_type: InstrumentType::NeutronDetector,
                specifications: "Array for higher rates".to_string(),
                cost_usd: 30000.0,
                availability: Availability::Commercial,
            }],
            estimated_cost_usd: 50000.0,
            duration_months: 4.0,
            priority: 0.7,
            success_criteria: vec![
                "Determine scaling law (linear, sublinear, superlinear)".to_string(),
                "Identify any saturation effects".to_string(),
            ],
        }
    }

    /// Lifetime experiment.
    pub fn lifetime_experiment() -> ExperimentDesign {
        ExperimentDesign {
            name: "Operational Lifetime".to_string(),
            research_question: "How long can sustained neutron production continue?".to_string(),
            hypotheses_tested: vec![],
            setup: ExperimentalSetup {
                host_material: "Best performer".to_string(),
                loading_ratio: 0.7,
                sample_geometry: SampleGeometry {
                    shape: "Standard from Phase 2".to_string(),
                    dimensions_cm: vec![],
                    active_volume_cm3: 1.0,
                    surface_area_cm2: 10.0,
                },
                trigger: TriggerSpec {
                    trigger_type: "Continuous operation".to_string(),
                    intensity: 0.0,
                    intensity_units: "TBD".to_string(),
                    pulse_duration_s: None,
                    repetition_rate_hz: None,
                },
                temperature_range: (300.0, 300.0),
                controls: vec![],
            },
            expected_outcomes: vec![],
            instrumentation: vec![
                Instrument {
                    instrument_type: InstrumentType::NeutronDetector,
                    specifications: "Long-term monitoring".to_string(),
                    cost_usd: 15000.0,
                    availability: Availability::Commercial,
                },
                Instrument {
                    instrument_type: InstrumentType::LoadingMonitor,
                    specifications: "Track D/M ratio over time".to_string(),
                    cost_usd: 5000.0,
                    availability: Availability::Commercial,
                },
            ],
            estimated_cost_usd: 30000.0,
            duration_months: 6.0,
            priority: 0.65,
            success_criteria: vec![
                "Run continuously for >100 hours".to_string(),
                "Track rate decay vs time".to_string(),
                "Correlate with D depletion/damage".to_string(),
            ],
        }
    }

    /// Efficiency experiment.
    pub fn efficiency_experiment() -> ExperimentDesign {
        ExperimentDesign {
            name: "Energy Efficiency".to_string(),
            research_question: "What is the neutron output per input energy?".to_string(),
            hypotheses_tested: vec![],
            setup: ExperimentalSetup {
                host_material: "Best performer".to_string(),
                loading_ratio: 0.7,
                sample_geometry: SampleGeometry {
                    shape: "Standard".to_string(),
                    dimensions_cm: vec![],
                    active_volume_cm3: 1.0,
                    surface_area_cm2: 10.0,
                },
                trigger: TriggerSpec {
                    trigger_type: "Variable power".to_string(),
                    intensity: 0.0,
                    intensity_units: "W".to_string(),
                    pulse_duration_s: None,
                    repetition_rate_hz: None,
                },
                temperature_range: (300.0, 300.0),
                controls: vec![],
            },
            expected_outcomes: vec![],
            instrumentation: vec![
                Instrument {
                    instrument_type: InstrumentType::NeutronDetector,
                    specifications: "Calibrated for absolute rate".to_string(),
                    cost_usd: 20000.0,
                    availability: Availability::Commercial,
                },
                Instrument {
                    instrument_type: InstrumentType::Calorimeter,
                    specifications: "Input power measurement".to_string(),
                    cost_usd: 10000.0,
                    availability: Availability::Commercial,
                },
            ],
            estimated_cost_usd: 40000.0,
            duration_months: 3.0,
            priority: 0.6,
            success_criteria: vec![
                "Measure n/s per Watt input".to_string(),
                "Compare to conventional sources".to_string(),
                "Identify optimization path".to_string(),
            ],
        }
    }

    /// Prioritize experiments based on information value.
    pub fn prioritize_experiments(experiments: &[ExperimentDesign]) -> Vec<(usize, f64)> {
        let mut ranked: Vec<(usize, f64)> = experiments
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let mut score = e.priority * 100.0;

                // Bonus for discriminating between hypotheses
                score += (e.hypotheses_tested.len() as f64) * 10.0;

                // Penalty for high cost
                score -= (e.estimated_cost_usd / 100000.0).min(30.0);

                // Penalty for long duration
                score -= e.duration_months;

                (i, score)
            })
            .collect();

        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }
}

/// A phase of the experimental program.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentPhase {
    /// Phase name
    pub name: String,
    /// Phase goal
    pub goal: String,
    /// Experiments in this phase
    pub experiments: Vec<ExperimentDesign>,
    /// Duration (months)
    pub duration_months: f64,
    /// Cost ($)
    pub cost_usd: f64,
    /// Success criteria
    pub success_criteria: Vec<String>,
    /// Go/no-go decision criteria
    pub go_nogo: String,
}

/// Complete experimental program.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentalProgram {
    /// Program name
    pub name: String,
    /// Phases
    pub phases: Vec<ExperimentPhase>,
    /// Total cost ($)
    pub total_cost_usd: f64,
    /// Total duration (months)
    pub total_duration_months: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nasa_replication_design() {
        let exp = ExperimentDesigner::nasa_replication_experiment();

        assert!(!exp.name.is_empty());
        assert!(!exp.hypotheses_tested.is_empty());
        assert!(!exp.instrumentation.is_empty());
        assert!(exp.estimated_cost_usd > 0.0);
    }

    #[test]
    fn test_experimental_program() {
        let program = ExperimentDesigner::design_program();

        assert_eq!(program.phases.len(), 3);
        assert!(program.total_cost_usd > 0.0);
        assert!(program.total_duration_months > 0.0);
    }

    #[test]
    fn test_experiment_prioritization() {
        let experiments = vec![
            ExperimentDesigner::nasa_replication_experiment(),
            ExperimentDesigner::hydrogen_control_experiment(),
            ExperimentDesigner::temperature_mapping_experiment(),
        ];

        let ranked = ExperimentDesigner::prioritize_experiments(&experiments);

        assert_eq!(ranked.len(), experiments.len());
        // NASA replication should be highest priority
        assert_eq!(ranked[0].0, 0);
    }

    #[test]
    fn test_control_experiments() {
        let exp = ExperimentDesigner::nasa_replication_experiment();

        // Should have H control
        assert!(
            exp.setup
                .controls
                .iter()
                .any(|c| c.control_type == ControlType::HydrogenControl)
        );
    }

    #[test]
    fn test_phase_structure() {
        let program = ExperimentDesigner::design_program();

        // Phase 1 should be validation
        assert_eq!(program.phases[0].name, "Validation");

        // Each phase should have experiments
        for phase in &program.phases {
            assert!(!phase.experiments.is_empty());
            assert!(!phase.success_criteria.is_empty());
        }
    }
}
