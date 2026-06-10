// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # C-Mod Dataset Adapter for Plasma Disruption Prediction
//!
//! This module provides a complete adapter for loading and processing plasma disruption
//! data from the Alcator C-Mod tokamak, following the Multi-Machine Disruption Prediction
//! Challenge data format.
//!
//! ## Overview
//!
//! The C-Mod (Compact Modular) tokamak at MIT produced extensive diagnostic data for
//! plasma disruption research. This adapter handles:
//!
//! - **Data Loading**: CSV and optional HDF5 file parsing
//! - **Streaming**: Real-time playback of shot sequences
//! - **Statistics**: Dataset-wide analysis and sensor statistics
//! - **Labeling**: Configurable disruption warning labels
//! - **HDC Integration**: Conversion to plasma samples for Phi monitoring
//!
//! ## Data Format
//!
//! C-Mod disruption data typically contains:
//! - Shot ID: Unique plasma discharge identifier
//! - Time: Milliseconds from shot start
//! - Ip: Plasma current (MA)
//! - ne: Electron density (10^20 m^-3)
//! - Te: Electron temperature (keV)
//! - Prad: Radiated power (MW)
//! - Vloop: Loop voltage (V)
//! - q95: Edge safety factor (dimensionless)
//! - Wmhd: Stored energy (MJ)
//! - Beta: Normalized beta (dimensionless)
//!
//! ## Integration with PlasmaHdcEncoder
//!
//! This adapter provides conversion functions to integrate with the existing
//! `PlasmaHdcEncoder` and `PlasmaState` types in `plasma_hdc_encoder.rs`.
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use symthaea::physics::cmod_adapter::*;
//! use symthaea::physics::plasma_hdc_encoder::PlasmaHdcEncoder;
//!
//! // Load CSV data
//! let shots = load_csv(Path::new("cmod_disruptions.csv"))?;
//!
//! // Create streaming iterator and encoder
//! let mut stream = CModStream::new(shots, 1.0);
//! let encoder = PlasmaHdcEncoder::new(42);
//!
//! // Process samples with HDC encoding
//! for sample in stream {
//!     let plasma_state = sample.to_plasma_state();
//!     let encoded = encoder.encode_state(&plasma_state);
//!     // Monitor Phi for disruption precursors
//! }
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

// Re-export core types for HDC integration
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

// Import existing plasma types for integration
use crate::plasma_hdc_encoder::PlasmaState;

// =============================================================================
// CORE DATA STRUCTURES
// =============================================================================

/// A single time-slice sample from a C-Mod plasma shot
///
/// Contains all diagnostic measurements at a specific time point.
/// NaN values indicate missing or invalid sensor readings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CModSample {
    /// Unique shot identifier (plasma discharge number)
    pub shot_id: u32,

    /// Time from shot start in milliseconds
    pub time_ms: f64,

    /// Plasma current in Mega-Amperes (MA)
    /// Typical range: 0.2 - 1.2 MA
    pub ip: f32,

    /// Line-averaged electron density (10^20 m^-3)
    /// Typical range: 0.5 - 4.0
    pub ne: f32,

    /// Electron temperature in keV
    /// Typical range: 0.5 - 5.0 keV
    pub te: f32,

    /// Total radiated power in Megawatts (MW)
    /// Typical range: 0.1 - 10.0 MW
    pub prad: f32,

    /// Loop voltage in Volts
    /// Typical range: 0.5 - 3.0 V
    pub vloop: f32,

    /// Edge safety factor q95 (dimensionless)
    /// Typical range: 2.0 - 8.0
    pub q95: f32,

    /// Stored energy in Mega-Joules (MJ)
    /// Typical range: 0.01 - 0.3 MJ
    pub wmhd: f32,

    /// Normalized beta (dimensionless)
    /// Typical range: 0.1 - 2.5
    pub beta: f32,

    /// Whether this shot eventually disrupts
    pub is_disruption: bool,

    /// Time remaining until disruption (None if no disruption)
    pub time_to_disruption_ms: Option<f64>,
}

impl CModSample {
    /// Create a new sample with default (NaN) sensor values
    pub fn new(shot_id: u32, time_ms: f64) -> Self {
        Self {
            shot_id,
            time_ms,
            ip: f32::NAN,
            ne: f32::NAN,
            te: f32::NAN,
            prad: f32::NAN,
            vloop: f32::NAN,
            q95: f32::NAN,
            wmhd: f32::NAN,
            beta: f32::NAN,
            is_disruption: false,
            time_to_disruption_ms: None,
        }
    }

    /// Check if any sensor value is NaN (missing data)
    pub fn has_missing_data(&self) -> bool {
        self.ip.is_nan()
            || self.ne.is_nan()
            || self.te.is_nan()
            || self.prad.is_nan()
            || self.vloop.is_nan()
            || self.q95.is_nan()
            || self.wmhd.is_nan()
            || self.beta.is_nan()
    }

    /// Get all sensor values as a vector (for encoding)
    pub fn to_feature_vector(&self) -> Vec<f32> {
        vec![
            self.ip, self.ne, self.te, self.prad, self.vloop, self.q95, self.wmhd, self.beta,
        ]
    }

    /// Get sensor values normalized to [0, 1] using provided ranges
    pub fn to_normalized_vector(&self, normalizer: &SensorNormalizer) -> Vec<f32> {
        vec![
            normalizer.normalize_ip(self.ip),
            normalizer.normalize_ne(self.ne),
            normalizer.normalize_te(self.te),
            normalizer.normalize_prad(self.prad),
            normalizer.normalize_vloop(self.vloop),
            normalizer.normalize_q95(self.q95),
            normalizer.normalize_wmhd(self.wmhd),
            normalizer.normalize_beta(self.beta),
        ]
    }

    /// Convert to PlasmaState for integration with PlasmaHdcEncoder
    ///
    /// Uses default values for Bt (toroidal field) and Bp (poloidal field)
    /// as these are not typically in C-Mod disruption datasets.
    pub fn to_plasma_state(&self) -> PlasmaState {
        // Convert time from ms to seconds
        let timestamp_s = self.time_ms / 1000.0;

        // Handle NaN values by replacing with typical defaults
        let safe_value =
            |v: f32, default: f64| -> f64 { if v.is_nan() { default } else { v as f64 } };

        PlasmaState::new(
            timestamp_s,
            safe_value(self.ip, 1.0),    // MA
            safe_value(self.ne, 1.5),    // 10^20 m^-3
            safe_value(self.te, 3.0),    // keV
            safe_value(self.prad, 1.0),  // MW
            safe_value(self.vloop, 1.0), // V
            5.0,                         // Bt (T) - default for C-Mod
            0.5,                         // Bp (T) - default for C-Mod
            safe_value(self.q95, 3.5),   // q95
            safe_value(self.wmhd, 0.1),  // MJ
            safe_value(self.beta, 1.0),  // %
        )
    }
}

/// A complete plasma shot containing all time samples
///
/// Represents one complete plasma discharge from initiation to termination
/// (either controlled shutdown or disruption).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CModShot {
    /// Unique shot identifier
    pub shot_id: u32,

    /// Time-ordered samples for this shot
    pub samples: Vec<CModSample>,

    /// Whether this shot experienced a disruption
    pub disrupted: bool,

    /// Time of disruption (None if no disruption)
    pub disruption_time_ms: Option<f64>,
}

impl CModShot {
    /// Create a new empty shot
    pub fn new(shot_id: u32) -> Self {
        Self {
            shot_id,
            samples: Vec::new(),
            disrupted: false,
            disruption_time_ms: None,
        }
    }

    /// Get shot duration in milliseconds
    pub fn duration_ms(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let first = self
            .samples
            .first()
            .expect("samples checked non-empty above")
            .time_ms;
        let last = self
            .samples
            .last()
            .expect("samples checked non-empty above")
            .time_ms;
        last - first
    }

    /// Add a sample to this shot
    pub fn add_sample(&mut self, mut sample: CModSample) {
        sample.is_disruption = self.disrupted;
        if let Some(disruption_time) = self.disruption_time_ms {
            sample.time_to_disruption_ms = Some(disruption_time - sample.time_ms);
        }
        self.samples.push(sample);
    }

    /// Sort samples by time
    pub fn sort_by_time(&mut self) {
        self.samples.sort_by(|a, b| a.time_ms.total_cmp(&b.time_ms));
    }

    /// Get sample count
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if shot is empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

// =============================================================================
// DISRUPTION LABELS
// =============================================================================

/// Disruption proximity labels for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisruptionLabel {
    /// Normal operation, no disruption expected
    Normal,
    /// Warning: Approaching disruption (configurable window)
    Warning,
    /// Critical: Disruption imminent (configurable threshold)
    Critical,
    /// Post-disruption state (if tracking continues)
    PostDisruption,
}

impl DisruptionLabel {
    /// Convert to numeric value for machine learning
    pub fn to_numeric(&self) -> f32 {
        match self {
            Self::Normal => 0.0,
            Self::Warning => 0.5,
            Self::Critical => 1.0,
            Self::PostDisruption => -1.0,
        }
    }

    /// Get label color for visualization
    pub fn color(&self) -> &'static str {
        match self {
            Self::Normal => "green",
            Self::Warning => "yellow",
            Self::Critical => "red",
            Self::PostDisruption => "gray",
        }
    }
}

/// Label all samples in a shot based on time-to-disruption
///
/// # Arguments
/// * `shot` - The shot to label
/// * `warning_window_ms` - Time before disruption to start warning (e.g., 100.0 ms)
/// * `critical_window_ms` - Time before disruption for critical state (e.g., 20.0 ms)
///
/// # Returns
/// Vector of labels corresponding to each sample
pub fn label_samples(
    shot: &CModShot,
    warning_window_ms: f64,
    critical_window_ms: f64,
) -> Vec<DisruptionLabel> {
    shot.samples
        .iter()
        .map(|sample| {
            if !shot.disrupted {
                return DisruptionLabel::Normal;
            }

            match sample.time_to_disruption_ms {
                Some(ttd) if ttd < 0.0 => DisruptionLabel::PostDisruption,
                Some(ttd) if ttd <= critical_window_ms => DisruptionLabel::Critical,
                Some(ttd) if ttd <= warning_window_ms => DisruptionLabel::Warning,
                _ => DisruptionLabel::Normal,
            }
        })
        .collect()
}

/// Label configuration for customizing disruption windows
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelConfig {
    /// Warning window in ms (default: 100.0)
    pub warning_window_ms: f64,
    /// Critical window in ms (default: 20.0)
    pub critical_window_ms: f64,
}

impl Default for LabelConfig {
    fn default() -> Self {
        Self {
            warning_window_ms: 100.0,
            critical_window_ms: 20.0,
        }
    }
}

// =============================================================================
// SENSOR NORMALIZATION
// =============================================================================

/// Statistics for a single sensor across the dataset
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SensorStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std_dev: f32,
    pub count: usize,
    pub nan_count: usize,
}

impl SensorStats {
    /// Compute statistics from a slice of values
    pub fn from_values(values: &[f32]) -> Self {
        let valid: Vec<f32> = values.iter().filter(|v| !v.is_nan()).copied().collect();
        let nan_count = values.len() - valid.len();

        if valid.is_empty() {
            return Self {
                min: f32::NAN,
                max: f32::NAN,
                mean: f32::NAN,
                std_dev: f32::NAN,
                count: 0,
                nan_count,
            };
        }

        let min = valid.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = valid.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let sum: f32 = valid.iter().sum();
        let mean = sum / valid.len() as f32;

        let variance: f32 =
            valid.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / valid.len() as f32;
        let std_dev = variance.sqrt();

        Self {
            min,
            max,
            mean,
            std_dev,
            count: valid.len(),
            nan_count,
        }
    }
}

/// Normalizer for converting raw sensor values to [0, 1] range
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorNormalizer {
    /// Ip range (MA)
    pub ip_range: (f32, f32),
    /// ne range (10^20 m^-3)
    pub ne_range: (f32, f32),
    /// Te range (keV)
    pub te_range: (f32, f32),
    /// Prad range (MW)
    pub prad_range: (f32, f32),
    /// Vloop range (V)
    pub vloop_range: (f32, f32),
    /// q95 range
    pub q95_range: (f32, f32),
    /// Wmhd range (MJ)
    pub wmhd_range: (f32, f32),
    /// Beta range
    pub beta_range: (f32, f32),
}

impl Default for SensorNormalizer {
    fn default() -> Self {
        // Default ranges based on typical C-Mod operating parameters
        Self {
            ip_range: (0.0, 1.5),     // MA
            ne_range: (0.0, 5.0),     // 10^20 m^-3
            te_range: (0.0, 6.0),     // keV
            prad_range: (0.0, 15.0),  // MW
            vloop_range: (-1.0, 5.0), // V
            q95_range: (1.5, 10.0),   // dimensionless
            wmhd_range: (0.0, 0.5),   // MJ
            beta_range: (0.0, 3.0),   // dimensionless
        }
    }
}

impl SensorNormalizer {
    /// Create normalizer from dataset statistics
    pub fn from_stats(stats: &DatasetStats) -> Self {
        Self {
            ip_range: (stats.ip_stats.min, stats.ip_stats.max),
            ne_range: (stats.ne_stats.min, stats.ne_stats.max),
            te_range: (stats.te_stats.min, stats.te_stats.max),
            prad_range: (stats.prad_stats.min, stats.prad_stats.max),
            vloop_range: (stats.vloop_stats.min, stats.vloop_stats.max),
            q95_range: (stats.q95_stats.min, stats.q95_stats.max),
            wmhd_range: (stats.wmhd_stats.min, stats.wmhd_stats.max),
            beta_range: (stats.beta_stats.min, stats.beta_stats.max),
        }
    }

    fn normalize(value: f32, range: (f32, f32)) -> f32 {
        if value.is_nan() || range.0 >= range.1 {
            return 0.5; // Default to middle for invalid values
        }
        ((value - range.0) / (range.1 - range.0)).clamp(0.0, 1.0)
    }

    pub fn normalize_ip(&self, value: f32) -> f32 {
        Self::normalize(value, self.ip_range)
    }
    pub fn normalize_ne(&self, value: f32) -> f32 {
        Self::normalize(value, self.ne_range)
    }
    pub fn normalize_te(&self, value: f32) -> f32 {
        Self::normalize(value, self.te_range)
    }
    pub fn normalize_prad(&self, value: f32) -> f32 {
        Self::normalize(value, self.prad_range)
    }
    pub fn normalize_vloop(&self, value: f32) -> f32 {
        Self::normalize(value, self.vloop_range)
    }
    pub fn normalize_q95(&self, value: f32) -> f32 {
        Self::normalize(value, self.q95_range)
    }
    pub fn normalize_wmhd(&self, value: f32) -> f32 {
        Self::normalize(value, self.wmhd_range)
    }
    pub fn normalize_beta(&self, value: f32) -> f32 {
        Self::normalize(value, self.beta_range)
    }
}

// =============================================================================
// DATASET STATISTICS
// =============================================================================

/// Comprehensive statistics for the entire dataset
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DatasetStats {
    /// Total number of shots
    pub total_shots: usize,
    /// Number of disrupted shots
    pub disrupted_shots: usize,
    /// Number of non-disrupted shots
    pub non_disrupted_shots: usize,
    /// Total number of samples
    pub total_samples: usize,

    /// Per-sensor statistics
    pub ip_stats: SensorStats,
    pub ne_stats: SensorStats,
    pub te_stats: SensorStats,
    pub prad_stats: SensorStats,
    pub vloop_stats: SensorStats,
    pub q95_stats: SensorStats,
    pub wmhd_stats: SensorStats,
    pub beta_stats: SensorStats,

    /// Time-to-disruption distribution for disrupted shots
    pub ttd_stats: SensorStats,
}

/// Compute comprehensive statistics for a dataset
pub fn compute_statistics(shots: &[CModShot]) -> DatasetStats {
    let total_shots = shots.len();
    let disrupted_shots = shots.iter().filter(|s| s.disrupted).count();
    let non_disrupted_shots = total_shots - disrupted_shots;

    // Collect all sensor values
    let mut ip_vals: Vec<f32> = Vec::new();
    let mut ne_vals: Vec<f32> = Vec::new();
    let mut te_vals: Vec<f32> = Vec::new();
    let mut prad_vals: Vec<f32> = Vec::new();
    let mut vloop_vals: Vec<f32> = Vec::new();
    let mut q95_vals: Vec<f32> = Vec::new();
    let mut wmhd_vals: Vec<f32> = Vec::new();
    let mut beta_vals: Vec<f32> = Vec::new();
    let mut ttd_vals: Vec<f32> = Vec::new();

    for shot in shots {
        for sample in &shot.samples {
            ip_vals.push(sample.ip);
            ne_vals.push(sample.ne);
            te_vals.push(sample.te);
            prad_vals.push(sample.prad);
            vloop_vals.push(sample.vloop);
            q95_vals.push(sample.q95);
            wmhd_vals.push(sample.wmhd);
            beta_vals.push(sample.beta);

            if let Some(ttd) = sample.time_to_disruption_ms {
                if ttd > 0.0 {
                    ttd_vals.push(ttd as f32);
                }
            }
        }
    }

    DatasetStats {
        total_shots,
        disrupted_shots,
        non_disrupted_shots,
        total_samples: ip_vals.len(),
        ip_stats: SensorStats::from_values(&ip_vals),
        ne_stats: SensorStats::from_values(&ne_vals),
        te_stats: SensorStats::from_values(&te_vals),
        prad_stats: SensorStats::from_values(&prad_vals),
        vloop_stats: SensorStats::from_values(&vloop_vals),
        q95_stats: SensorStats::from_values(&q95_vals),
        wmhd_stats: SensorStats::from_values(&wmhd_vals),
        beta_stats: SensorStats::from_values(&beta_vals),
        ttd_stats: SensorStats::from_values(&ttd_vals),
    }
}

// =============================================================================
// FILE PARSING
// =============================================================================

/// Load C-Mod data from CSV file
///
/// Expected CSV format:
/// ```csv
/// shot_id,time_ms,ip,ne,te,prad,vloop,q95,wmhd,beta,disrupted,disruption_time_ms
/// 1150101001,0.0,0.5,1.2,2.5,1.0,1.5,3.5,0.1,0.8,1,150.0
/// ```
///
/// # Arguments
/// * `path` - Path to the CSV file
///
/// # Returns
/// Vector of shots parsed from the file
pub fn load_csv(path: &Path) -> Result<Vec<CModShot>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path)
        .map_err(|e| anyhow!("Failed to open CSV file: {e}"))?;

    // Group samples by shot_id
    let mut shots_map: HashMap<u32, CModShot> = HashMap::new();

    for result in reader.records() {
        let record = result.map_err(|e| anyhow!("CSV parse error: {e}"))?;

        // Parse required fields
        let shot_id: u32 = record
            .get(0)
            .ok_or_else(|| anyhow!("Missing shot_id"))?
            .parse()
            .map_err(|e| anyhow!("Invalid shot_id: {e}"))?;

        let time_ms: f64 = record
            .get(1)
            .ok_or_else(|| anyhow!("Missing time_ms"))?
            .parse()
            .map_err(|e| anyhow!("Invalid time_ms: {e}"))?;

        // Parse sensor values (allow NaN for missing)
        let parse_f32 = |idx: usize| -> f32 {
            record
                .get(idx)
                .and_then(|s: &str| s.parse::<f32>().ok())
                .unwrap_or(f32::NAN)
        };

        let sample = CModSample {
            shot_id,
            time_ms,
            ip: parse_f32(2),
            ne: parse_f32(3),
            te: parse_f32(4),
            prad: parse_f32(5),
            vloop: parse_f32(6),
            q95: parse_f32(7),
            wmhd: parse_f32(8),
            beta: parse_f32(9),
            is_disruption: false,        // Will be set by shot
            time_to_disruption_ms: None, // Will be computed
        };

        // Parse disruption info
        let disrupted: bool = record
            .get(10)
            .map(|s: &str| s == "1" || s.to_lowercase() == "true")
            .unwrap_or(false);

        let disruption_time_ms: Option<f64> =
            record.get(11).and_then(|s: &str| s.parse::<f64>().ok());

        // Add to shot
        let shot = shots_map.entry(shot_id).or_insert_with(|| {
            let mut s = CModShot::new(shot_id);
            s.disrupted = disrupted;
            s.disruption_time_ms = disruption_time_ms;
            s
        });

        shot.add_sample(sample);
    }

    // Convert to vector and sort
    let mut shots: Vec<CModShot> = shots_map.into_values().collect();
    for shot in &mut shots {
        shot.sort_by_time();
    }
    shots.sort_by_key(|s| s.shot_id);

    Ok(shots)
}

/// Load C-Mod data from HDF5 file.
///
/// HDF5 support was removed (no system HDF5 library available).
/// Use `load_csv` or `load_synthetic` instead.
pub fn load_hdf5(_path: &Path) -> Result<Vec<CModShot>> {
    Err(anyhow!(
        "HDF5 support removed. Use load_csv or load_synthetic instead."
    ))
}

// =============================================================================
// MISSING VALUE HANDLING
// =============================================================================

/// Strategy for handling missing (NaN) values
#[derive(Debug, Clone, Copy)]
pub enum MissingValueStrategy {
    /// Replace with zero
    Zero,
    /// Replace with sensor mean
    Mean,
    /// Linear interpolation from surrounding values
    Interpolate,
    /// Forward-fill (use last known value)
    ForwardFill,
    /// Drop samples with any missing values
    Drop,
}

/// Fill missing values in a shot using the specified strategy
pub fn fill_missing_values(
    shot: &mut CModShot,
    strategy: MissingValueStrategy,
    stats: &DatasetStats,
) {
    match strategy {
        MissingValueStrategy::Zero => fill_with_value(shot, 0.0),
        MissingValueStrategy::Mean => fill_with_mean(shot, stats),
        MissingValueStrategy::Interpolate => interpolate_missing(shot),
        MissingValueStrategy::ForwardFill => forward_fill(shot),
        MissingValueStrategy::Drop => drop_missing(shot),
    }
}

fn fill_with_value(shot: &mut CModShot, value: f32) {
    for sample in &mut shot.samples {
        if sample.ip.is_nan() {
            sample.ip = value;
        }
        if sample.ne.is_nan() {
            sample.ne = value;
        }
        if sample.te.is_nan() {
            sample.te = value;
        }
        if sample.prad.is_nan() {
            sample.prad = value;
        }
        if sample.vloop.is_nan() {
            sample.vloop = value;
        }
        if sample.q95.is_nan() {
            sample.q95 = value;
        }
        if sample.wmhd.is_nan() {
            sample.wmhd = value;
        }
        if sample.beta.is_nan() {
            sample.beta = value;
        }
    }
}

fn fill_with_mean(shot: &mut CModShot, stats: &DatasetStats) {
    for sample in &mut shot.samples {
        if sample.ip.is_nan() {
            sample.ip = stats.ip_stats.mean;
        }
        if sample.ne.is_nan() {
            sample.ne = stats.ne_stats.mean;
        }
        if sample.te.is_nan() {
            sample.te = stats.te_stats.mean;
        }
        if sample.prad.is_nan() {
            sample.prad = stats.prad_stats.mean;
        }
        if sample.vloop.is_nan() {
            sample.vloop = stats.vloop_stats.mean;
        }
        if sample.q95.is_nan() {
            sample.q95 = stats.q95_stats.mean;
        }
        if sample.wmhd.is_nan() {
            sample.wmhd = stats.wmhd_stats.mean;
        }
        if sample.beta.is_nan() {
            sample.beta = stats.beta_stats.mean;
        }
    }
}

fn interpolate_missing(shot: &mut CModShot) {
    if shot.samples.len() < 2 {
        return;
    }

    // Interpolate each sensor independently
    interpolate_sensor(shot, |s| s.ip, |s, v| s.ip = v);
    interpolate_sensor(shot, |s| s.ne, |s, v| s.ne = v);
    interpolate_sensor(shot, |s| s.te, |s, v| s.te = v);
    interpolate_sensor(shot, |s| s.prad, |s, v| s.prad = v);
    interpolate_sensor(shot, |s| s.vloop, |s, v| s.vloop = v);
    interpolate_sensor(shot, |s| s.q95, |s, v| s.q95 = v);
    interpolate_sensor(shot, |s| s.wmhd, |s, v| s.wmhd = v);
    interpolate_sensor(shot, |s| s.beta, |s, v| s.beta = v);
}

fn interpolate_sensor<F, G>(shot: &mut CModShot, getter: F, setter: G)
where
    F: Fn(&CModSample) -> f32,
    G: Fn(&mut CModSample, f32),
{
    let n = shot.samples.len();
    let mut i = 0;

    while i < n {
        let value = getter(&shot.samples[i]);
        if value.is_nan() {
            // Find previous valid value
            let prev_idx = (0..i).rev().find(|&j| !getter(&shot.samples[j]).is_nan());
            // Find next valid value
            let next_idx = ((i + 1)..n).find(|&j| !getter(&shot.samples[j]).is_nan());

            let interpolated = match (prev_idx, next_idx) {
                (Some(p), Some(n)) => {
                    // Linear interpolation
                    let prev_val = getter(&shot.samples[p]);
                    let next_val = getter(&shot.samples[n]);
                    let t = (i - p) as f32 / (n - p) as f32;
                    prev_val + t * (next_val - prev_val)
                }
                (Some(p), None) => getter(&shot.samples[p]),
                (None, Some(n)) => getter(&shot.samples[n]),
                (None, None) => 0.0,
            };

            setter(&mut shot.samples[i], interpolated);
        }
        i += 1;
    }
}

fn forward_fill(shot: &mut CModShot) {
    let mut last_ip = 0.0f32;
    let mut last_ne = 0.0f32;
    let mut last_te = 0.0f32;
    let mut last_prad = 0.0f32;
    let mut last_vloop = 0.0f32;
    let mut last_q95 = 0.0f32;
    let mut last_wmhd = 0.0f32;
    let mut last_beta = 0.0f32;

    for sample in &mut shot.samples {
        if sample.ip.is_nan() {
            sample.ip = last_ip;
        } else {
            last_ip = sample.ip;
        }
        if sample.ne.is_nan() {
            sample.ne = last_ne;
        } else {
            last_ne = sample.ne;
        }
        if sample.te.is_nan() {
            sample.te = last_te;
        } else {
            last_te = sample.te;
        }
        if sample.prad.is_nan() {
            sample.prad = last_prad;
        } else {
            last_prad = sample.prad;
        }
        if sample.vloop.is_nan() {
            sample.vloop = last_vloop;
        } else {
            last_vloop = sample.vloop;
        }
        if sample.q95.is_nan() {
            sample.q95 = last_q95;
        } else {
            last_q95 = sample.q95;
        }
        if sample.wmhd.is_nan() {
            sample.wmhd = last_wmhd;
        } else {
            last_wmhd = sample.wmhd;
        }
        if sample.beta.is_nan() {
            sample.beta = last_beta;
        } else {
            last_beta = sample.beta;
        }
    }
}

fn drop_missing(shot: &mut CModShot) {
    shot.samples.retain(|s| !s.has_missing_data());
}

// =============================================================================
// STREAMING ITERATOR
// =============================================================================

/// Streaming iterator for real-time playback of C-Mod shot data
///
/// Supports configurable playback speed and seamless shot transitions.
#[derive(Debug)]
pub struct CModStream {
    /// All shots in the dataset
    shots: Vec<CModShot>,
    /// Current shot index
    current_shot: usize,
    /// Current sample index within shot
    current_sample: usize,
    /// Playback speed multiplier (1.0 = real-time) - reserved for rate limiting
    _playback_speed: f32,
    /// Whether to loop when reaching end
    loop_playback: bool,
    /// Last emit time for rate limiting
    last_emit_time: Option<Instant>,
    /// Last sample time for delta calculation
    last_sample_time_ms: Option<f64>,
}

impl CModStream {
    /// Create a new stream from shots
    pub fn new(shots: Vec<CModShot>, _playback_speed: f32) -> Self {
        Self {
            shots,
            current_shot: 0,
            current_sample: 0,
            _playback_speed,
            loop_playback: false,
            last_emit_time: None,
            last_sample_time_ms: None,
        }
    }

    /// Enable looping playback
    pub fn with_loop(mut self, loop_playback: bool) -> Self {
        self.loop_playback = loop_playback;
        self
    }

    /// Reset to beginning
    pub fn reset(&mut self) {
        self.current_shot = 0;
        self.current_sample = 0;
        self.last_emit_time = None;
        self.last_sample_time_ms = None;
    }

    /// Get current shot
    pub fn current_shot(&self) -> Option<&CModShot> {
        self.shots.get(self.current_shot)
    }

    /// Get progress (0.0 to 1.0)
    pub fn progress(&self) -> f32 {
        if self.shots.is_empty() {
            return 0.0;
        }

        let total_samples: usize = self.shots.iter().map(|s| s.samples.len()).sum();
        let completed: usize = self.shots[..self.current_shot]
            .iter()
            .map(|s| s.samples.len())
            .sum::<usize>()
            + self.current_sample;

        completed as f32 / total_samples as f32
    }

    /// Get next sample without timing constraints (instant)
    pub fn next_instant(&mut self) -> Option<CModSample> {
        if self.shots.is_empty() {
            return None;
        }

        // Check if we've exhausted current shot
        while self.current_shot < self.shots.len() {
            let shot = &self.shots[self.current_shot];
            if self.current_sample < shot.samples.len() {
                let sample = shot.samples[self.current_sample].clone();
                self.current_sample += 1;
                return Some(sample);
            }
            // Move to next shot
            self.current_shot += 1;
            self.current_sample = 0;
        }

        // Handle loop
        if self.loop_playback {
            self.reset();
            return self.next_instant();
        }

        None
    }
}

impl Iterator for CModStream {
    type Item = CModSample;

    fn next(&mut self) -> Option<Self::Item> {
        // For now, just use instant mode
        // Real-time mode would require async or blocking with timing
        self.next_instant()
    }
}

// =============================================================================
// HDC INTEGRATION
// =============================================================================

/// Normalized plasma sample structure for C-Mod HDC encoding
///
/// Simplified representation suitable for hyperdimensional encoding
/// and Phi (integrated information) monitoring.
#[derive(Debug, Clone)]
pub struct CModPlasmaSample {
    /// Encoded sensor vector (normalized to [0, 1])
    pub sensors: Vec<f32>,
    /// Time in milliseconds
    pub time_ms: f64,
    /// Disruption label
    pub label: DisruptionLabel,
    /// Original shot ID for tracing
    pub shot_id: u32,
}

/// Convert C-Mod sample to CModPlasmaSample for HDC encoding
pub fn to_cmod_plasma_sample(
    cmod: &CModSample,
    normalizer: &SensorNormalizer,
    label: DisruptionLabel,
) -> CModPlasmaSample {
    CModPlasmaSample {
        sensors: cmod.to_normalized_vector(normalizer),
        time_ms: cmod.time_ms,
        label,
        shot_id: cmod.shot_id,
    }
}

/// C-Mod specific HDC Encoder for converting normalized plasma samples
///
/// Uses amplitude encoding with learned basis vectors for each sensor channel.
/// This is a lightweight encoder specifically for the 8-channel C-Mod format.
/// For full plasma state encoding, use the main `PlasmaHdcEncoder` from
/// `plasma_hdc_encoder.rs`.
#[derive(Debug)]
pub struct CModHdcEncoder {
    /// HDC dimension (default: HDC_DIMENSION = 16,384)
    dimension: usize,
    /// Basis vectors for each sensor channel
    sensor_bases: Vec<ContinuousHV>,
    /// Level vectors for amplitude encoding
    level_vectors: Vec<ContinuousHV>,
    /// Number of quantization levels
    num_levels: usize,
    /// Sensor names for debugging - reserved for diagnostics
    _sensor_names: Vec<&'static str>,
}

impl CModHdcEncoder {
    /// Number of sensor channels in C-Mod data
    pub const NUM_SENSORS: usize = 8;
    /// Sensor names
    pub const SENSOR_NAMES: [&'static str; 8] =
        ["ip", "ne", "te", "prad", "vloop", "q95", "wmhd", "beta"];

    /// Create a new C-Mod encoder
    pub fn new(dimension: usize, num_levels: usize) -> Self {
        // Generate deterministic basis vectors for each sensor
        let sensor_bases: Vec<ContinuousHV> = (0..Self::NUM_SENSORS)
            .map(|i| ContinuousHV::random(dimension, 100_000 + i as u64))
            .collect();

        // Generate level vectors for amplitude quantization
        let level_vectors: Vec<ContinuousHV> = (0..num_levels)
            .map(|i| ContinuousHV::random(dimension, 200_000 + i as u64))
            .collect();

        Self {
            dimension,
            sensor_bases,
            level_vectors,
            num_levels,
            _sensor_names: Self::SENSOR_NAMES.to_vec(),
        }
    }

    /// Create with default settings (HDC_DIMENSION, 32 levels)
    pub fn default_encoder() -> Self {
        Self::new(HDC_DIMENSION, 32)
    }

    /// Encode a plasma sample to a hyperdimensional vector
    ///
    /// # Algorithm
    /// 1. For each sensor, quantize the normalized value to a level
    /// 2. Bind the sensor basis with the level vector
    /// 3. Bundle all bound vectors to create the final encoding
    pub fn encode(&self, sample: &CModPlasmaSample) -> ContinuousHV {
        let mut bound_vectors: Vec<ContinuousHV> = Vec::with_capacity(Self::NUM_SENSORS);

        for (i, &value) in sample.sensors.iter().enumerate() {
            // Quantize value to level index
            let level_idx = (value * (self.num_levels - 1) as f32)
                .round()
                .clamp(0.0, (self.num_levels - 1) as f32) as usize;

            // Bind sensor basis with level vector
            let bound = self.sensor_bases[i].bind(&self.level_vectors[level_idx]);
            bound_vectors.push(bound);
        }

        // Bundle all bound vectors
        ContinuousHV::bundle_owned(&bound_vectors)
    }

    /// Encode a batch of samples
    pub fn encode_batch(&self, samples: &[CModPlasmaSample]) -> Vec<ContinuousHV> {
        samples.iter().map(|s| self.encode(s)).collect()
    }

    /// Get encoding dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Phi monitor for tracking integrated information over time
///
/// Monitors the evolution of Phi (consciousness metric) as plasma
/// state changes, useful for detecting critical state transitions.
///
/// Uses a simplified correlation-based approximation of integration
/// that can be computed efficiently from ContinuousHV encodings.
#[derive(Debug)]
pub struct CModPhiMonitor {
    /// History of Phi values
    history: Vec<(f64, f64)>, // (time_ms, phi)
    /// Rolling window size
    window_size: usize,
    /// Recent encodings for correlation analysis
    recent_encodings: Vec<ContinuousHV>,
}

impl CModPhiMonitor {
    /// Create a new Phi monitor
    pub fn new(window_size: usize) -> Self {
        Self {
            history: Vec::new(),
            window_size,
            recent_encodings: Vec::new(),
        }
    }

    /// Create with default settings
    pub fn default_monitor() -> Self {
        Self::new(100)
    }

    /// Update with a new encoded sample
    pub fn update(&mut self, time_ms: f64, encoding: &ContinuousHV) {
        // Add to recent encodings
        self.recent_encodings.push(encoding.clone());
        if self.recent_encodings.len() > self.window_size {
            self.recent_encodings.remove(0);
        }

        // Compute Phi approximation if we have enough samples
        if self.recent_encodings.len() >= 4 {
            let phi = self.compute_integration_approximation();
            self.history.push((time_ms, phi));
        }
    }

    /// Compute a simplified integration measure based on encoding similarity
    ///
    /// This measures how integrated/coherent the recent plasma states are.
    /// High integration (high Phi) indicates stable plasma; decreasing
    /// integration may precede disruptions.
    fn compute_integration_approximation(&self) -> f64 {
        if self.recent_encodings.len() < 2 {
            return 0.5;
        }

        // Compute average pairwise similarity of recent encodings
        let n = self.recent_encodings.len().min(10); // Use last 10 samples
        let recent = &self.recent_encodings[self.recent_encodings.len() - n..];

        let mut sum_similarity = 0.0;
        let mut count = 0;

        for i in 0..recent.len() {
            for j in (i + 1)..recent.len() {
                sum_similarity += recent[i].similarity(&recent[j]) as f64;
                count += 1;
            }
        }

        if count == 0 {
            return 0.5;
        }

        // Average similarity in [0, 1] range
        // High similarity = integrated states = high "Phi"
        sum_similarity / count as f64
    }

    /// Get current Phi value
    pub fn current_phi(&self) -> Option<f64> {
        self.history.last().map(|(_, phi)| *phi)
    }

    /// Get Phi history
    pub fn history(&self) -> &[(f64, f64)] {
        &self.history
    }

    /// Detect if Phi is dropping (potential precursor to disruption)
    pub fn is_declining(&self) -> bool {
        if self.history.len() < 10 {
            return false;
        }

        let recent: Vec<f64> = self
            .history
            .iter()
            .rev()
            .take(10)
            .map(|(_, p)| *p)
            .collect();
        let mean_recent = recent.iter().sum::<f64>() / recent.len() as f64;
        let mean_older: f64 = self
            .history
            .iter()
            .rev()
            .skip(10)
            .take(10)
            .map(|(_, p)| *p)
            .sum::<f64>()
            / 10.0;

        mean_recent < mean_older * 0.9 // 10% decline
    }
}

// =============================================================================
// BENCHMARKING
// =============================================================================

/// Benchmark result for encoding throughput
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Total samples processed
    pub samples_processed: usize,
    /// Total time in milliseconds
    pub total_time_ms: f64,
    /// Samples per second
    pub throughput: f64,
    /// Average time per sample in microseconds
    pub avg_time_per_sample_us: f64,
}

/// Benchmark encoding throughput
pub fn benchmark_encoding(
    encoder: &CModHdcEncoder,
    samples: &[CModPlasmaSample],
    iterations: usize,
) -> BenchmarkResult {
    let start = Instant::now();

    for _ in 0..iterations {
        for sample in samples {
            let _ = encoder.encode(sample);
        }
    }

    let elapsed = start.elapsed();
    let total_samples = samples.len() * iterations;
    let total_time_ms = elapsed.as_secs_f64() * 1000.0;

    BenchmarkResult {
        samples_processed: total_samples,
        total_time_ms,
        throughput: total_samples as f64 / elapsed.as_secs_f64(),
        avg_time_per_sample_us: elapsed.as_micros() as f64 / total_samples as f64,
    }
}

// =============================================================================
// SYNTHETIC DATA GENERATION
// =============================================================================

/// Configuration for synthetic data generation
#[derive(Debug, Clone)]
pub struct SyntheticConfig {
    /// Number of shots to generate
    pub num_shots: usize,
    /// Probability of a shot being a disruption (0.0 to 1.0)
    pub disruption_probability: f32,
    /// Average number of samples per shot
    pub samples_per_shot: usize,
    /// Sample interval in milliseconds
    pub sample_interval_ms: f64,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for SyntheticConfig {
    fn default() -> Self {
        Self {
            num_shots: 100,
            disruption_probability: 0.3,
            samples_per_shot: 200,
            sample_interval_ms: 1.0,
            seed: 42,
        }
    }
}

/// Generate synthetic C-Mod data for testing
///
/// Creates realistic-looking plasma discharge data without requiring
/// access to real experimental data.
pub fn generate_synthetic_data(config: &SyntheticConfig) -> Vec<CModShot> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut shots = Vec::with_capacity(config.num_shots);

    for shot_idx in 0..config.num_shots {
        let shot_id = 1_000_000 + shot_idx as u32;
        let mut shot = CModShot::new(shot_id);

        // Determine if this shot disrupts
        shot.disrupted = rng.r#gen::<f32>() < config.disruption_probability;

        // Set disruption time if applicable (towards end of shot)
        let num_samples = config.samples_per_shot + rng.gen_range(0..50);
        if shot.disrupted {
            // Disruption happens somewhere in the last 10-30% of the shot
            let disruption_offset =
                (num_samples as f64 * (0.7 + rng.r#gen::<f64>() * 0.2)) as usize;
            let disruption_sample = disruption_offset.max(1);
            shot.disruption_time_ms = Some(disruption_sample as f64 * config.sample_interval_ms);
        }

        // Generate samples
        // Base values with some shot-to-shot variation
        let base_ip = 0.6 + rng.r#gen::<f32>() * 0.4;
        let base_ne = 1.5 + rng.r#gen::<f32>() * 1.5;
        let base_te = 2.0 + rng.r#gen::<f32>() * 2.0;
        let base_prad = 1.0 + rng.r#gen::<f32>() * 2.0;
        let base_vloop = 1.0 + rng.r#gen::<f32>() * 0.5;
        let base_q95 = 3.0 + rng.r#gen::<f32>() * 2.0;
        let base_wmhd = 0.1 + rng.r#gen::<f32>() * 0.1;
        let base_beta = 0.8 + rng.r#gen::<f32>() * 0.6;

        for sample_idx in 0..num_samples {
            let time_ms = sample_idx as f64 * config.sample_interval_ms;

            // Calculate time-to-disruption if applicable
            let ttd = if shot.disrupted {
                shot.disruption_time_ms.map(|dt| dt - time_ms)
            } else {
                None
            };

            // Add temporal evolution and noise
            let t_norm = sample_idx as f32 / num_samples as f32;
            let mut noise = || (rng.r#gen::<f32>() - 0.5) * 0.1;

            // Disruption precursor effects
            let disruption_factor = if let Some(ttd_val) = ttd {
                if ttd_val < 0.0 {
                    0.1 // Post-disruption collapse
                } else if ttd_val < 20.0 {
                    0.3 + 0.7 * (ttd_val as f32 / 20.0) // Critical phase
                } else if ttd_val < 100.0 {
                    0.7 + 0.3 * (ttd_val as f32 / 100.0) // Warning phase
                } else {
                    1.0
                }
            } else {
                1.0
            };

            let sample = CModSample {
                shot_id,
                time_ms,
                ip: (base_ip * (1.0 + 0.1 * t_norm) * disruption_factor + noise()).max(0.0),
                ne: (base_ne * (1.0 - 0.05 * t_norm) * disruption_factor + noise()).max(0.0),
                te: (base_te * (1.0 - 0.1 * t_norm) * disruption_factor + noise()).max(0.0),
                prad: (base_prad * (1.0 + 0.3 * t_norm / disruption_factor) + noise()).max(0.0),
                vloop: (base_vloop * (1.0 + 0.2 * t_norm / disruption_factor) + noise()).max(0.0),
                q95: (base_q95 * (1.0 - 0.05 * t_norm) + noise()).max(1.5),
                wmhd: (base_wmhd * (1.0 - 0.1 * t_norm) * disruption_factor + noise()).max(0.0),
                beta: (base_beta * (1.0 - 0.05 * t_norm) * disruption_factor + noise()).max(0.0),
                is_disruption: shot.disrupted,
                time_to_disruption_ms: ttd,
            };

            shot.samples.push(sample);
        }

        shots.push(shot);
    }

    shots
}

// =============================================================================
// EXAMPLE PIPELINE
// =============================================================================

/// Example: Complete pipeline from data loading to Phi monitoring
///
/// Demonstrates the full workflow:
/// 1. Load or generate data
/// 2. Compute statistics and create normalizer
/// 3. Stream samples through HDC encoder
/// 4. Monitor Phi values for disruption precursors
///
/// # Example
/// ```rust,ignore
/// use symthaea::physics::cmod_adapter::*;
///
/// // Generate synthetic data for testing
/// let shots = generate_synthetic_data(&SyntheticConfig::default());
///
/// // Run the example pipeline
/// let results = example_pipeline(&shots)?;
/// println!("Processed {} samples, detected {} anomalies",
///     results.samples_processed, results.anomalies_detected);
/// ```
#[derive(Debug)]
pub struct PipelineResults {
    /// Total samples processed
    pub samples_processed: usize,
    /// Anomalies detected (Phi drops)
    pub anomalies_detected: usize,
    /// Final Phi value
    pub final_phi: Option<f64>,
    /// Encoding throughput (samples/sec)
    pub throughput: f64,
}

/// Run example pipeline on shot data
pub fn example_pipeline(shots: &[CModShot]) -> Result<PipelineResults> {
    // Compute statistics
    let stats = compute_statistics(shots);
    println!(
        "Dataset: {} shots ({} disrupted), {} total samples",
        stats.total_shots, stats.disrupted_shots, stats.total_samples
    );

    // Create normalizer and encoder
    let normalizer = SensorNormalizer::from_stats(&stats);
    let encoder = CModHdcEncoder::default_encoder();
    let mut phi_monitor = CModPhiMonitor::default_monitor();

    // Process samples
    let mut stream = CModStream::new(shots.to_vec(), 1.0);
    let label_config = LabelConfig::default();

    let start = Instant::now();
    let mut samples_processed = 0;
    let mut anomalies_detected = 0;

    for sample in stream.by_ref() {
        // Get label for this sample
        let label = if sample.is_disruption {
            match sample.time_to_disruption_ms {
                Some(ttd) if ttd <= label_config.critical_window_ms => DisruptionLabel::Critical,
                Some(ttd) if ttd <= label_config.warning_window_ms => DisruptionLabel::Warning,
                Some(ttd) if ttd < 0.0 => DisruptionLabel::PostDisruption,
                _ => DisruptionLabel::Normal,
            }
        } else {
            DisruptionLabel::Normal
        };

        // Convert to plasma sample and encode
        let plasma_sample = to_cmod_plasma_sample(&sample, &normalizer, label);
        let encoding = encoder.encode(&plasma_sample);

        // Update Phi monitor
        phi_monitor.update(sample.time_ms, &encoding);

        // Check for anomalies
        if phi_monitor.is_declining() {
            anomalies_detected += 1;
        }

        samples_processed += 1;
    }

    let elapsed = start.elapsed();
    let throughput = samples_processed as f64 / elapsed.as_secs_f64();

    Ok(PipelineResults {
        samples_processed,
        anomalies_detected,
        final_phi: phi_monitor.current_phi(),
        throughput,
    })
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cmod_sample_creation() {
        let sample = CModSample::new(123, 100.0);
        assert_eq!(sample.shot_id, 123);
        assert_eq!(sample.time_ms, 100.0);
        assert!(sample.has_missing_data());
    }

    #[test]
    fn test_cmod_shot_creation() {
        let mut shot = CModShot::new(12345);
        assert_eq!(shot.shot_id, 12345);
        assert!(shot.is_empty());

        shot.disrupted = true;
        shot.disruption_time_ms = Some(150.0);

        let sample = CModSample::new(12345, 100.0);
        shot.add_sample(sample);

        assert_eq!(shot.len(), 1);
        assert!(shot.samples[0].is_disruption);
        assert_eq!(shot.samples[0].time_to_disruption_ms, Some(50.0));
    }

    #[test]
    fn test_labeling() {
        let mut shot = CModShot::new(1);
        shot.disrupted = true;
        shot.disruption_time_ms = Some(200.0);

        // Add samples at various times
        // warning_window = 100ms, critical_window = 20ms
        // disruption at t=200ms
        // Logic: ttd <= critical (20) -> Critical, ttd <= warning (100) -> Warning, else Normal
        for t in [
            0.0, 50.0, 100.0, 150.0, 175.0, 180.0, 190.0, 195.0, 200.0, 210.0,
        ] {
            let mut sample = CModSample::new(1, t);
            sample.time_to_disruption_ms = Some(200.0 - t);
            shot.samples.push(sample);
        }

        let labels = label_samples(&shot, 100.0, 20.0);

        // ttd values: 200, 150, 100, 50, 25, 20, 10, 5, 0, -10
        assert_eq!(labels[0], DisruptionLabel::Normal); // t=0, ttd=200 (> 100 = Normal)
        assert_eq!(labels[1], DisruptionLabel::Normal); // t=50, ttd=150 (> 100 = Normal)
        assert_eq!(labels[2], DisruptionLabel::Warning); // t=100, ttd=100 (<= 100 warning)
        assert_eq!(labels[3], DisruptionLabel::Warning); // t=150, ttd=50 (<= 100 warning)
        assert_eq!(labels[4], DisruptionLabel::Warning); // t=175, ttd=25 (25 > 20, so Warning)
        assert_eq!(labels[5], DisruptionLabel::Critical); // t=180, ttd=20 (<= 20 critical)
        assert_eq!(labels[6], DisruptionLabel::Critical); // t=190, ttd=10 (<= 20 critical)
        assert_eq!(labels[7], DisruptionLabel::Critical); // t=195, ttd=5 (<= 20 critical)
        assert_eq!(labels[8], DisruptionLabel::Critical); // t=200, ttd=0 (<= 20 critical)
        assert_eq!(labels[9], DisruptionLabel::PostDisruption); // t=210, ttd=-10 (< 0 = post)
    }

    #[test]
    fn test_sensor_stats() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, f32::NAN];
        let stats = SensorStats::from_values(&values);

        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert!((stats.mean - 3.0).abs() < 0.001);
        assert_eq!(stats.count, 5);
        assert_eq!(stats.nan_count, 1);
    }

    #[test]
    fn test_normalizer() {
        let normalizer = SensorNormalizer::default();

        // Test middle of range
        let norm_ip = normalizer.normalize_ip(0.75);
        assert!(norm_ip > 0.4 && norm_ip < 0.6);

        // Test boundary values
        assert_eq!(normalizer.normalize_ip(0.0), 0.0);
        assert_eq!(normalizer.normalize_ip(1.5), 1.0);

        // Test clipping
        assert_eq!(normalizer.normalize_ip(10.0), 1.0);
    }

    #[test]
    fn test_synthetic_data_generation() {
        let config = SyntheticConfig {
            num_shots: 10,
            disruption_probability: 0.5,
            samples_per_shot: 50,
            sample_interval_ms: 1.0,
            seed: 42,
        };

        let shots = generate_synthetic_data(&config);

        assert_eq!(shots.len(), 10);

        // Check that we have some disrupted and some non-disrupted shots
        let disrupted_count = shots.iter().filter(|s| s.disrupted).count();
        assert!(disrupted_count > 0);
        assert!(disrupted_count < shots.len());

        // Check sample values are reasonable
        for shot in &shots {
            assert!(!shot.is_empty());
            for sample in &shot.samples {
                assert!(sample.ip >= 0.0);
                assert!(sample.ne >= 0.0);
                assert!(sample.te >= 0.0);
                assert!(sample.q95 >= 1.5);
            }
        }
    }

    #[test]
    fn test_cmod_encoder() {
        let encoder = CModHdcEncoder::default_encoder();
        let sample = CModPlasmaSample {
            sensors: vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            time_ms: 100.0,
            label: DisruptionLabel::Normal,
            shot_id: 1,
        };

        let encoded = encoder.encode(&sample);
        assert_eq!(encoded.values.len(), HDC_DIMENSION);

        // Encoded vector should not be all zeros
        let norm: f32 = encoded.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm > 0.0);
    }

    #[test]
    fn test_stream_iterator() {
        let config = SyntheticConfig {
            num_shots: 3,
            samples_per_shot: 10,
            ..Default::default()
        };
        let shots = generate_synthetic_data(&config);

        let stream = CModStream::new(shots.clone(), 1.0);
        let samples: Vec<_> = stream.collect();

        // Should have samples from all shots
        let total_expected: usize = shots.iter().map(|s| s.len()).sum();
        assert_eq!(samples.len(), total_expected);
    }

    #[test]
    fn test_missing_value_handling() {
        let config = SyntheticConfig {
            num_shots: 1,
            samples_per_shot: 10,
            ..Default::default()
        };
        let mut shots = generate_synthetic_data(&config);

        // Introduce some NaN values
        shots[0].samples[3].ip = f32::NAN;
        shots[0].samples[5].ne = f32::NAN;

        let stats = compute_statistics(&shots);

        // Test forward fill
        let mut shot_ff = shots[0].clone();
        fill_missing_values(&mut shot_ff, MissingValueStrategy::ForwardFill, &stats);
        assert!(!shot_ff.samples[3].ip.is_nan());

        // Test interpolation
        let mut shot_interp = shots[0].clone();
        fill_missing_values(&mut shot_interp, MissingValueStrategy::Interpolate, &stats);
        assert!(!shot_interp.samples[3].ip.is_nan());
    }

    #[test]
    fn test_encoding_determinism() {
        let encoder = CModHdcEncoder::default_encoder();
        let sample = CModPlasmaSample {
            sensors: vec![0.3, 0.6, 0.4, 0.7, 0.5, 0.8, 0.2, 0.9],
            time_ms: 50.0,
            label: DisruptionLabel::Warning,
            shot_id: 42,
        };

        let enc1 = encoder.encode(&sample);
        let enc2 = encoder.encode(&sample);

        // Same input should produce same output
        assert_eq!(enc1.values, enc2.values);
    }

    #[test]
    fn test_encoding_similarity() {
        let encoder = CModHdcEncoder::default_encoder();

        // Identical inputs should produce identical encodings
        let sample1 = CModPlasmaSample {
            sensors: vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            time_ms: 100.0,
            label: DisruptionLabel::Normal,
            shot_id: 1,
        };

        let sample1_copy = CModPlasmaSample {
            sensors: vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            time_ms: 100.0,
            label: DisruptionLabel::Normal,
            shot_id: 1,
        };

        // Different inputs should produce different encodings
        let sample2 = CModPlasmaSample {
            sensors: vec![0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4],
            time_ms: 100.0,
            label: DisruptionLabel::Normal,
            shot_id: 1,
        };

        let enc1 = encoder.encode(&sample1);
        let enc1_copy = encoder.encode(&sample1_copy);
        let enc2 = encoder.encode(&sample2);

        // Identical inputs produce identical encodings
        let sim_identical = enc1.similarity(&enc1_copy);
        assert!(
            (sim_identical - 1.0).abs() < 0.001,
            "Identical inputs should have ~1.0 similarity, got {}",
            sim_identical
        );

        // Different inputs should have lower similarity
        let sim_different = enc1.similarity(&enc2);
        assert!(
            sim_different < 0.9,
            "Different inputs should have <0.9 similarity, got {}",
            sim_different
        );
    }

    #[test]
    fn test_dataset_statistics() {
        let config = SyntheticConfig {
            num_shots: 20,
            disruption_probability: 0.4,
            samples_per_shot: 100,
            ..Default::default()
        };
        let shots = generate_synthetic_data(&config);

        let stats = compute_statistics(&shots);

        assert_eq!(stats.total_shots, 20);
        assert!(stats.disrupted_shots > 0);
        assert!(stats.non_disrupted_shots > 0);
        assert_eq!(
            stats.total_samples,
            shots.iter().map(|s| s.len()).sum::<usize>()
        );

        // Sensor stats should be valid
        assert!(stats.ip_stats.min < stats.ip_stats.max);
        assert!(stats.ip_stats.mean > 0.0);
    }

    #[test]
    #[ignore = "performance test - run with cargo test --release --ignored"]
    fn test_benchmark() {
        let encoder = CModHdcEncoder::default_encoder();
        let normalizer = SensorNormalizer::default();

        let config = SyntheticConfig {
            num_shots: 1,
            samples_per_shot: 100,
            ..Default::default()
        };
        let shots = generate_synthetic_data(&config);

        let samples: Vec<CModPlasmaSample> = shots[0]
            .samples
            .iter()
            .map(|s| to_cmod_plasma_sample(s, &normalizer, DisruptionLabel::Normal))
            .collect();

        let iterations = 10;
        let result = benchmark_encoding(&encoder, &samples, iterations);

        // Due to random sample count variation (100 + 0-49), just verify it's in range
        let expected_min = 100 * iterations;
        let expected_max = 150 * iterations;
        assert!(result.samples_processed >= expected_min);
        assert!(result.samples_processed <= expected_max);
        assert!(result.throughput > 0.0);
        assert!(result.avg_time_per_sample_us < 10_000.0); // Should be reasonably fast
    }

    #[test]
    fn test_plasma_state_conversion() {
        let mut sample = CModSample::new(12345, 500.0);
        sample.ip = 1.2;
        sample.ne = 2.0;
        sample.te = 4.0;
        sample.prad = 1.5;
        sample.vloop = 1.2;
        sample.q95 = 3.5;
        sample.wmhd = 0.15;
        sample.beta = 1.2;

        let plasma_state = sample.to_plasma_state();

        // Check conversion (time_ms -> seconds)
        assert!((plasma_state.timestamp - 0.5).abs() < 0.001);
        assert!((plasma_state.ip - 1.2).abs() < 0.001);
        assert!((plasma_state.ne - 2.0).abs() < 0.001);
        assert!((plasma_state.te - 4.0).abs() < 0.001);
        assert!((plasma_state.q95 - 3.5).abs() < 0.001);
    }
}
