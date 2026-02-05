//! # Physics Module
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

pub mod plasma_hdc_encoder;
pub mod plasma_control;
pub mod cmod_adapter;

pub use plasma_hdc_encoder::{
    // Sensor types
    PlasmaSensorType,
    PlasmaReading,
    PlasmaState,
    // Encoder
    PlasmaHdcEncoder,
    PlasmaEncoderConfig,
    // Streaming
    PlasmaStateBuffer,
    // Phi integration
    DisruptionAlert,
    PlasmaPhiThresholds,
    PlasmaPhiMonitor,
    StabilityAssessment,
};

pub use plasma_control::{
    // Configuration
    PlasmaControlConfig,
    // Actions and control
    PlasmaControlAction,
    PlasmaControlLoop,
    PlasmaStabilityAssessment,
    // State types
    PlasmaControlState,
    GasSpecies,
    TrendDirection,
    StabilityRegime,
    // Simulation
    PlasmaSimulator,
    DisruptionScenario,
    PlasmaControlStats,
};

pub use cmod_adapter::{
    // Core data structures
    CModSample,
    CModShot,
    // Labels
    DisruptionLabel,
    LabelConfig,
    label_samples,
    // Statistics
    SensorStats,
    SensorNormalizer,
    DatasetStats,
    compute_statistics,
    // File loading
    load_csv,
    load_hdf5,
    // Missing value handling
    MissingValueStrategy,
    fill_missing_values,
    // Streaming
    CModStream,
    // HDC integration
    CModPlasmaSample,
    CModHdcEncoder,
    CModPhiMonitor,
    to_cmod_plasma_sample,
    // Benchmarking
    BenchmarkResult,
    benchmark_encoding,
    // Synthetic data
    SyntheticConfig,
    generate_synthetic_data,
    // Pipeline
    PipelineResults,
    example_pipeline,
};
