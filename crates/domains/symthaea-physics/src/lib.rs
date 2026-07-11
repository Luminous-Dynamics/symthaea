// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Symthaea Physics
//!
//! Physics simulation and sensor encoding for the Spark Engine.
//!
//! ## Modules
//!
//! - [`plasma_hdc_encoder`]: HDC encoding for tokamak/plasma fusion sensor data
//! - [`plasma_control`]: Phi-based plasma control feedback system
//! - [`cmod_adapter`]: C-Mod dataset adapter for plasma disruption prediction
//!
//! ## C-Mod Dataset Adapter
//!
//! The C-Mod adapter provides tools for loading and processing plasma disruption
//! data from the Alcator C-Mod tokamak, following the Multi-Machine Disruption
//! Prediction Challenge format. It includes:
//!
//! - CSV/HDF5 data loading
//! - Streaming iterators for real-time playback
//! - Disruption labeling (Normal/Warning/Critical)
//! - Integration with PlasmaHdcEncoder for Phi monitoring
//! - Synthetic data generation for testing

#![deny(unsafe_code)]

pub mod accelerator;
pub mod antimatter;
pub mod cmod_adapter;
#[allow(
    clippy::useless_format,
    clippy::single_char_add_str,
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop
)]
pub mod cmod_evaluation;
#[allow(
    clippy::useless_format,
    clippy::single_char_add_str,
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop
)]
pub mod cross_machine_evaluation;
pub mod datacenter;
pub mod fission;
pub mod fission_bench;
pub mod fission_dispatch;
pub mod fusion_twin;
pub mod grid;
pub mod multiscale;
pub mod nppad_validation;
pub mod plasma_control;
pub mod plasma_hdc_encoder;
pub mod threat;

pub use plasma_hdc_encoder::{
    // Phi integration
    DisruptionAlert,
    PlasmaEncoderConfig,
    // Encoder
    PlasmaHdcEncoder,
    PlasmaPhiMonitor,
    PlasmaPhiThresholds,
    PlasmaReading,
    // Sensor types
    PlasmaSensorType,
    PlasmaState,
    // Streaming
    PlasmaStateBuffer,
    StabilityAssessment,
};

pub use plasma_control::{
    DisruptionScenario,
    GasSpecies,
    // Actions and control
    PlasmaControlAction,
    // Configuration
    PlasmaControlConfig,
    PlasmaControlLoop,
    // State types
    PlasmaControlState,
    PlasmaControlStats,
    // Simulation
    PlasmaSimulator,
    PlasmaStabilityAssessment,
    StabilityRegime,
    TrendDirection,
};

pub use multiscale::{
    CrossScaleCoherence, MultiScaleUnifier, PhysicalScale, ScaleEncoder, ScalePredictor,
};

pub use fusion_twin::{
    DisruptionRisk, FusionDigitalTwin, FusionTwinOutput, PLASMA_HORIZON_LABELS, PLASMA_HORIZONS,
    PlasmaFepAction, PlasmaFepAgent, PlasmaMultiScalePredictor,
};

pub use grid::{
    GRID_HORIZON_LABELS, GRID_HORIZONS, GridFepAction, GridFepAgent, GridHdcEncoder, GridOutput,
    GridPredictor, GridReading, GridSafetyLevel, GridTwin,
};

pub use fission::{
    FISSION_HORIZON_LABELS, FISSION_HORIZONS, FissionFepAction, FissionFepAgent, FissionHdcEncoder,
    FissionOutput, FissionPredictor, FissionReading, FissionSafetyLevel, FissionTwin,
};

pub use fission_bench::{
    BenchmarkReport, FaultKind, FaultScenario, FreeEnergyDetector, PLANT_CHANNEL_LABELS,
    PLANT_CHANNELS, PlantHdcEncoder, PlantReading, PlantSimulator, ScenarioResult, run_benchmark,
    run_healthy_false_alarm_check, run_scenario, standard_fault_suite,
};

pub use fission_dispatch::{
    FissionTelemetryPayload, SensorNodeRegistration, SimulatedRadZoneInspectionOrder,
    simulated_dispatch_for_degradation,
};

pub use accelerator::{
    ACCELERATOR_HORIZON_LABELS, ACCELERATOR_HORIZONS, AcceleratorFepAction, AcceleratorFepAgent,
    AcceleratorHdcEncoder, AcceleratorOutput, AcceleratorPredictor, AcceleratorReading,
    AcceleratorSafetyLevel, AcceleratorTwin,
};

pub use threat::{
    THREAT_HORIZON_LABELS, THREAT_HORIZONS, ThreatFepAction, ThreatFepAgent, ThreatHdcEncoder,
    ThreatLevel, ThreatOutput, ThreatPredictor, ThreatReading, ThreatTwin,
};

pub use datacenter::{
    DATACENTER_HORIZON_LABELS, DATACENTER_HORIZONS, DatacenterFepAction, DatacenterFepAgent,
    DatacenterHdcEncoder, DatacenterOutput, DatacenterPredictor, DatacenterReading,
    DatacenterSafetyLevel, DatacenterTwin,
};

pub use cmod_evaluation::{
    ConfusionMatrix, DensityLimitEncoder, DensityLimitEncoderV3, DensityLimitSample,
    DensityLimitShot, EvaluationConfig, EvaluationConfigV2, EvaluationConfigV3, EvaluationReport,
    LeadTimeStats, NormalizationRanges, RocPoint, compute_auc, compute_roc_curve,
    evaluate_density_limit, evaluate_density_limit_v2, evaluate_density_limit_v3,
    greenwald_fraction, load_density_limit_csv,
};

pub use cross_machine_evaluation::{
    CrossMachineResult, GenericEvalConfig, GenericHdcEncoder, GenericNormRanges, GenericSample,
    GenericShot, MachineDatasetConfig, MachineEvaluationResult, evaluate_cross_machine,
    evaluate_machine, generate_cross_machine_report, generate_synthetic_machine, load_generic_csv,
};

pub use cmod_adapter::{
    // Benchmarking
    BenchmarkResult,
    CModHdcEncoder,
    CModPhiMonitor,
    // HDC integration
    CModPlasmaSample,
    // Core data structures
    CModSample,
    CModShot,
    // Streaming
    CModStream,
    DatasetStats,
    // Labels
    DisruptionLabel,
    LabelConfig,
    // Missing value handling
    MissingValueStrategy,
    // Pipeline
    PipelineResults,
    SensorNormalizer,
    // Statistics
    SensorStats,
    // Synthetic data
    SyntheticConfig,
    benchmark_encoding,
    compute_statistics,
    example_pipeline,
    fill_missing_values,
    generate_synthetic_data,
    label_samples,
    // File loading
    load_csv,
    load_hdf5,
    to_cmod_plasma_sample,
};
