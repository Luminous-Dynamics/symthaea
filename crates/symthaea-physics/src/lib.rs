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
pub mod cmod_adapter;
#[allow(
    clippy::useless_format,
    clippy::single_char_add_str,
    clippy::manual_is_multiple_of,
    clippy::needless_range_loop,
    clippy::manual_saturating_arithmetic
)]
pub mod cmod_evaluation;
pub mod datacenter;
pub mod fission;
pub mod fusion_twin;
pub mod grid;
pub mod multiscale;
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
    DisruptionRisk, FusionDigitalTwin, FusionTwinOutput, PlasmaFepAction, PlasmaFepAgent,
    PlasmaMultiScalePredictor, PLASMA_HORIZONS, PLASMA_HORIZON_LABELS,
};

pub use grid::{
    GridFepAction, GridFepAgent, GridHdcEncoder, GridOutput, GridPredictor, GridReading,
    GridSafetyLevel, GridTwin, GRID_HORIZONS, GRID_HORIZON_LABELS,
};

pub use fission::{
    FissionFepAction, FissionFepAgent, FissionHdcEncoder, FissionOutput, FissionPredictor,
    FissionReading, FissionSafetyLevel, FissionTwin, FISSION_HORIZONS, FISSION_HORIZON_LABELS,
};

pub use accelerator::{
    AcceleratorFepAction, AcceleratorFepAgent, AcceleratorHdcEncoder, AcceleratorOutput,
    AcceleratorPredictor, AcceleratorReading, AcceleratorSafetyLevel, AcceleratorTwin,
    ACCELERATOR_HORIZONS, ACCELERATOR_HORIZON_LABELS,
};

pub use threat::{
    ThreatFepAction, ThreatFepAgent, ThreatHdcEncoder, ThreatLevel, ThreatOutput, ThreatPredictor,
    ThreatReading, ThreatTwin, THREAT_HORIZONS, THREAT_HORIZON_LABELS,
};

pub use datacenter::{
    DatacenterFepAction, DatacenterFepAgent, DatacenterHdcEncoder, DatacenterOutput,
    DatacenterPredictor, DatacenterReading, DatacenterSafetyLevel, DatacenterTwin,
    DATACENTER_HORIZONS, DATACENTER_HORIZON_LABELS,
};

pub use cmod_evaluation::{
    compute_auc, compute_roc_curve, evaluate_density_limit, evaluate_density_limit_v2,
    evaluate_density_limit_v3, greenwald_fraction, load_density_limit_csv, ConfusionMatrix,
    DensityLimitEncoder, DensityLimitEncoderV3, DensityLimitSample, DensityLimitShot,
    EvaluationConfig, EvaluationConfigV2, EvaluationConfigV3, EvaluationReport, LeadTimeStats,
    NormalizationRanges, RocPoint,
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
