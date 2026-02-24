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

pub mod cmod_adapter;
pub mod plasma_control;
pub mod plasma_hdc_encoder;

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

pub use cmod_adapter::{
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
};
