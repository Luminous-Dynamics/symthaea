// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Cross-Machine Disruption Prediction Evaluation
//!
//! Generic evaluator that works with ANY tokamak dataset from the ITPA Global
//! Disruption Database (or any other source). The key insight: HDC encodes
//! normalized sensor values into 16,384D space and measures cosine distance
//! from a healthy reference — this is substrate-independent across machines.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use symthaea_physics::cross_machine_evaluation::*;
//! let config = MachineDatasetConfig::cmod();
//! let eval = GenericEvalConfig::default();
//! let result = evaluate_machine(&config, &eval).unwrap();
//! println!("AUC V1: {:.4}", result.auc_v1);
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

use crate::cmod_evaluation::{
    LeadTimeStats, RocPoint, classify_at_threshold, compute_auc, compute_roc_curve,
};

// ── Configuration ────────────────────────────────────────────────────────────

/// Configuration for a generic tokamak dataset.
///
/// Describes the CSV layout so the evaluator can work with any machine's data
/// without hard-coded column names.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MachineDatasetConfig {
    /// Human-readable machine name (e.g., "Alcator C-Mod", "JET", "DIII-D").
    pub machine_name: String,
    /// Path to the CSV file.
    pub csv_path: PathBuf,
    /// Column name for the shot/discharge identifier.
    pub shot_id_column: String,
    /// Column name for the time axis (seconds).
    pub time_column: String,
    /// Column name for the disruption label (binary).
    pub label_column: String,
    /// Ordered list of numeric sensor columns to encode.
    pub sensor_columns: Vec<String>,
    /// The string value in label_column that means "disruption". Default: "1".
    pub label_positive_value: String,
}

impl MachineDatasetConfig {
    /// Pre-built config for the Alcator C-Mod DL_DataFrame.csv.
    ///
    /// Matches the existing `cmod_evaluation` module's expected format so we
    /// can verify the generic evaluator reproduces the same results.
    pub fn cmod() -> Self {
        Self {
            machine_name: "Alcator C-Mod".to_string(),
            csv_path: PathBuf::from("data/cmod/DL_DataFrame.csv"),
            shot_id_column: "discharge_ID".to_string(),
            time_column: "time".to_string(),
            label_column: "density_limit_phase".to_string(),
            sensor_columns: vec![
                "density".to_string(),
                "elongation".to_string(),
                "minor_radius".to_string(),
                "plasma_current".to_string(),
                "toroidal_B_field".to_string(),
                "triangularity".to_string(),
            ],
            label_positive_value: "1".to_string(),
        }
    }

    /// Template config for ITPA Global Disruption Database files.
    ///
    /// Column names are placeholders — update once the actual ITPA schema is
    /// confirmed from the downloaded data.
    pub fn itpa_template() -> Self {
        Self {
            machine_name: "ITPA (template)".to_string(),
            csv_path: PathBuf::from("data/itpa/scalar_parameters.csv"),
            shot_id_column: "shot".to_string(),
            time_column: "time".to_string(),
            label_column: "disrupted".to_string(),
            sensor_columns: vec![
                "ip".to_string(),          // plasma current
                "bt".to_string(),          // toroidal field
                "ne".to_string(),          // electron density
                "q95".to_string(),         // safety factor
                "li".to_string(),          // internal inductance
                "betan".to_string(),       // normalized beta
                "wmhd".to_string(),        // stored energy
                "kappa".to_string(),       // elongation
                "delta_upper".to_string(), // upper triangularity
            ],
            label_positive_value: "1".to_string(),
        }
    }
}

/// Configuration controlling the evaluation harness behaviour.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericEvalConfig {
    /// Fraction of shots used for training (rest for test). Default: 0.8.
    pub train_fraction: f64,
    /// Temporal window size for V2 evaluation. Default: 5.
    pub window_size: usize,
    /// Fraction of each shot's early samples used as per-shot reference. Default: 0.2.
    pub reference_fraction: f64,
    /// Maximum threshold for fine sweep. Default: 0.30.
    pub fine_threshold_max: f64,
    /// Step size for fine threshold sweep. Default: 0.005.
    pub fine_threshold_step: f64,
    /// Minimum samples needed for per-shot reference (fallback to global). Default: 10.
    pub min_reference_samples: usize,
    /// Deterministic seed base for basis vector generation. Default: 0xCA00_0001.
    pub seed_base: u64,
}

impl Default for GenericEvalConfig {
    fn default() -> Self {
        Self {
            train_fraction: 0.8,
            window_size: 5,
            reference_fraction: 0.2,
            fine_threshold_max: 0.30,
            fine_threshold_step: 0.005,
            min_reference_samples: 10,
            seed_base: 0xCA00_0001,
        }
    }
}

// ── Generic Data Structures ──────────────────────────────────────────────────

/// A single time-step sample with arbitrary sensor columns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericSample {
    /// Shot/discharge identifier (parsed as string to handle any format).
    pub shot_id: String,
    /// Time value (seconds from shot start).
    pub time_s: f64,
    /// Sensor values in the same order as `MachineDatasetConfig::sensor_columns`.
    pub sensor_values: Vec<f32>,
    /// Whether this sample is labeled as a disruption.
    pub is_disruption: bool,
}

/// A complete shot grouped by shot identifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericShot {
    /// Shot identifier.
    pub shot_id: String,
    /// Time-ordered samples.
    pub samples: Vec<GenericSample>,
    /// Whether any sample in this shot is labeled as disruption.
    pub has_disruption: bool,
}

impl GenericShot {
    /// Time of the first disruption label in this shot (if any).
    pub fn disruption_onset_time(&self) -> Option<f64> {
        self.samples
            .iter()
            .find(|s| s.is_disruption)
            .map(|s| s.time_s)
    }
}

// ── Per-sensor Normalization ─────────────────────────────────────────────────

/// Per-sensor normalization ranges for mapping to [0, 1].
///
/// Unlike the fixed-size `NormalizationRanges` in `cmod_evaluation`, this
/// supports an arbitrary number of sensors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericNormRanges {
    pub mins: Vec<f32>,
    pub maxs: Vec<f32>,
}

impl GenericNormRanges {
    /// Compute normalization ranges from a set of shots.
    pub fn from_shots(shots: &[GenericShot], n_sensors: usize) -> Self {
        let mut mins = vec![f32::INFINITY; n_sensors];
        let mut maxs = vec![f32::NEG_INFINITY; n_sensors];

        for shot in shots {
            for sample in &shot.samples {
                for (i, &v) in sample.sensor_values.iter().enumerate() {
                    if i < n_sensors && v.is_finite() {
                        mins[i] = mins[i].min(v);
                        maxs[i] = maxs[i].max(v);
                    }
                }
            }
        }

        // Guard against degenerate ranges
        for i in 0..n_sensors {
            if !mins[i].is_finite() {
                mins[i] = 0.0;
            }
            if !maxs[i].is_finite() {
                maxs[i] = 1.0;
            }
            if (maxs[i] - mins[i]).abs() < 1e-10 {
                maxs[i] = mins[i] + 1.0;
            }
        }

        Self { mins, maxs }
    }

    /// Normalize a single sensor value to [0, 1].
    pub fn normalize(&self, sensor_idx: usize, value: f32) -> f32 {
        if sensor_idx >= self.mins.len() || !value.is_finite() {
            return 0.0;
        }
        let range = self.maxs[sensor_idx] - self.mins[sensor_idx];
        if range.abs() < 1e-10 {
            return 0.0;
        }
        ((value - self.mins[sensor_idx]) / range).clamp(0.0, 1.0)
    }

    /// Normalize all sensor values for a sample.
    pub fn normalize_sample(&self, sample: &GenericSample) -> Vec<f32> {
        sample
            .sensor_values
            .iter()
            .enumerate()
            .map(|(i, &v)| self.normalize(i, v))
            .collect()
    }
}

// ── Generic HDC Encoder ──────────────────────────────────────────────────────

/// HDC encoder for an arbitrary number of sensor columns.
///
/// Generates one deterministic basis vector per sensor. Encoding is a weighted
/// superposition: `encode_weighted(bases, normalized_values)`.
#[derive(Clone)]
pub struct GenericHdcEncoder {
    bases: Vec<ContinuousHV>,
    ranges: GenericNormRanges,
}

impl GenericHdcEncoder {
    /// Create encoder with pre-computed normalization ranges.
    ///
    /// `seed_base` is offset by the sensor index to produce a unique but
    /// deterministic basis vector for each sensor column.
    pub fn new(ranges: GenericNormRanges, seed_base: u64) -> Self {
        let n = ranges.mins.len();
        let bases: Vec<ContinuousHV> = (0..n)
            .map(|i| ContinuousHV::random(HDC_DIMENSION, seed_base + i as u64))
            .collect();
        Self { bases, ranges }
    }

    /// Create encoder from training shots, computing normalization ranges.
    pub fn from_shots(shots: &[GenericShot], n_sensors: usize, seed_base: u64) -> Self {
        let ranges = GenericNormRanges::from_shots(shots, n_sensors);
        Self::new(ranges, seed_base)
    }

    /// Encode a single sample into a 16,384D ContinuousHV.
    pub fn encode(&self, sample: &GenericSample) -> ContinuousHV {
        let normed = self.ranges.normalize_sample(sample);
        ContinuousHV::encode_weighted(&self.bases, &normed)
    }

    /// Encode a temporal window of consecutive samples into a single HV.
    pub fn encode_window(&self, samples: &[GenericSample]) -> ContinuousHV {
        if samples.is_empty() {
            return ContinuousHV::zero(HDC_DIMENSION);
        }
        if samples.len() == 1 {
            return self.encode(&samples[0]);
        }
        let encoded: Vec<ContinuousHV> = samples.iter().map(|s| self.encode(s)).collect();
        ContinuousHV::bundle_owned(&encoded)
    }

    /// Build a reference HV from healthy (non-disruption) samples.
    pub fn build_reference(&self, shots: &[GenericShot]) -> ContinuousHV {
        let healthy_encodings: Vec<ContinuousHV> = shots
            .iter()
            .flat_map(|shot| shot.samples.iter())
            .filter(|s| !s.is_disruption)
            .map(|s| self.encode(s))
            .collect();

        if healthy_encodings.is_empty() {
            return ContinuousHV::zero(HDC_DIMENSION);
        }

        ContinuousHV::bundle_owned(&healthy_encodings)
    }

    /// Access the normalization ranges.
    pub fn ranges(&self) -> &GenericNormRanges {
        &self.ranges
    }
}

// ── CSV Loading ──────────────────────────────────────────────────────────────

/// Load a generic CSV and group by shot ID.
///
/// Works with any CSV that has the columns described in `MachineDatasetConfig`.
/// Rows with NaN/Inf in any sensor column are silently skipped.
pub fn load_generic_csv(config: &MachineDatasetConfig) -> Result<Vec<GenericShot>> {
    load_generic_csv_from_path(&config.csv_path, config)
}

/// Load from an explicit path (useful for testing with temp files).
fn load_generic_csv_from_path(
    path: &Path,
    config: &MachineDatasetConfig,
) -> Result<Vec<GenericShot>> {
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("Failed to open CSV: {}", path.display()))?;

    // Resolve column indices from headers
    let headers = reader.headers()?.clone();
    let shot_idx = find_column_index(&headers, &config.shot_id_column)?;
    let time_idx = find_column_index(&headers, &config.time_column)?;
    let label_idx = find_column_index(&headers, &config.label_column)?;
    let sensor_indices: Vec<usize> = config
        .sensor_columns
        .iter()
        .map(|name| find_column_index(&headers, name))
        .collect::<Result<Vec<_>>>()?;

    let mut shots_map: HashMap<String, Vec<GenericSample>> = HashMap::new();

    for result in reader.records() {
        let record = result.context("Failed to parse CSV row")?;

        let shot_id = record.get(shot_idx).unwrap_or("").trim().to_string();

        let time_s: f64 = match record.get(time_idx).unwrap_or("").trim().parse() {
            Ok(t) if f64::is_finite(t) => t,
            _ => continue,
        };

        let label_str = record.get(label_idx).unwrap_or("").trim().to_string();
        let is_disruption = label_str == config.label_positive_value;

        // Parse sensor values, skip row if any is non-finite
        let mut sensor_values = Vec::with_capacity(sensor_indices.len());
        let mut skip = false;
        for &idx in &sensor_indices {
            match record.get(idx).unwrap_or("").trim().parse::<f32>() {
                Ok(v) if v.is_finite() => sensor_values.push(v),
                _ => {
                    skip = true;
                    break;
                }
            }
        }
        if skip {
            continue;
        }

        let sample = GenericSample {
            shot_id: shot_id.clone(),
            time_s,
            sensor_values,
            is_disruption,
        };

        shots_map.entry(shot_id).or_default().push(sample);
    }

    // Sort samples within each shot by time, collect into sorted shots
    let mut shots: Vec<GenericShot> = shots_map
        .into_iter()
        .map(|(shot_id, mut samples)| {
            samples.sort_by(|a, b| {
                a.time_s
                    .partial_cmp(&b.time_s)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let has_disruption = samples.iter().any(|s| s.is_disruption);
            GenericShot {
                shot_id,
                samples,
                has_disruption,
            }
        })
        .collect();

    shots.sort_by(|a, b| a.shot_id.cmp(&b.shot_id));
    Ok(shots)
}

/// Resolve a column name to its 0-based index in the CSV header row.
fn find_column_index(headers: &csv::StringRecord, name: &str) -> Result<usize> {
    headers
        .iter()
        .position(|h| h.trim() == name)
        .with_context(|| format!("Column '{}' not found in CSV headers", name))
}

// ── Result Types ─────────────────────────────────────────────────────────────

/// Results for a single machine evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MachineEvaluationResult {
    /// Machine name from the config.
    pub machine_name: String,
    /// Total number of shots loaded.
    pub total_shots: usize,
    /// Number of disrupted shots.
    pub disrupted_shots: usize,
    /// Total number of time-step samples in the test set.
    pub total_samples: usize,
    /// Number of sensor columns used.
    pub n_sensors: usize,
    /// AUC for V1 (single-sample, global reference).
    pub auc_v1: f64,
    /// AUC for V2 (temporal window + per-shot reference).
    pub auc_v2: f64,
    /// Best F1 score for V1.
    pub best_f1_v1: f64,
    /// Best F1 score for V2.
    pub best_f1_v2: f64,
    /// Mean detection lead time (ms) at best V2 threshold.
    pub lead_time_mean_ms: f64,
    /// Median detection lead time (ms) at best V2 threshold.
    pub lead_time_median_ms: f64,
}

/// Cross-machine transfer evaluation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossMachineResult {
    /// Per-machine independent evaluation results.
    pub per_machine: Vec<MachineEvaluationResult>,
    /// Mean V1 AUC across all machines.
    pub mean_auc_v1: f64,
    /// Mean V2 AUC across all machines.
    pub mean_auc_v2: f64,
    /// Cross-machine transfer AUC (train on machine A, test on machine B).
    /// Average over all (A, B) pairs where A != B.
    pub cross_machine_transfer_auc: f64,
}

// ── Scoring Functions ────────────────────────────────────────────────────────

/// Score each test sample: compute free energy relative to a reference HV.
fn score_samples_generic(
    encoder: &GenericHdcEncoder,
    reference: &ContinuousHV,
    samples: &[GenericSample],
) -> Vec<(f64, bool)> {
    samples
        .iter()
        .map(|sample| {
            let hv = encoder.encode(sample);
            let sim = hv.similarity(reference) as f64;
            let fe = if sim.is_finite() {
                (1.0 - sim).max(0.0)
            } else {
                1.0
            };
            (fe, sample.is_disruption)
        })
        .collect()
}

/// Score test shots using V2 (temporal windowing + per-shot early-phase reference).
fn score_shots_v2_generic(
    encoder: &GenericHdcEncoder,
    global_reference: &ContinuousHV,
    test_shots: &[GenericShot],
    config: &GenericEvalConfig,
) -> Vec<(f64, bool)> {
    let mut scored = Vec::new();

    for shot in test_shots {
        let n_samples = shot.samples.len();
        let n_ref = ((n_samples as f64) * config.reference_fraction).ceil() as usize;
        let n_ref = n_ref.min(n_samples);

        // Build per-shot reference from early samples
        let reference = if n_ref >= config.min_reference_samples {
            let ref_encodings: Vec<ContinuousHV> = shot.samples[..n_ref]
                .iter()
                .map(|s| encoder.encode(s))
                .collect();
            ContinuousHV::bundle_owned(&ref_encodings)
        } else {
            global_reference.clone()
        };

        // Score using temporal windows
        let w = config.window_size;
        for i in 0..n_samples {
            let start = (i + 1).saturating_sub(w);
            let window = &shot.samples[start..=i];
            let hv = encoder.encode_window(window);
            let sim = hv.similarity(&reference) as f64;
            let fe = if sim.is_finite() {
                (1.0 - sim).max(0.0)
            } else {
                1.0
            };
            scored.push((fe, shot.samples[i].is_disruption));
        }
    }

    scored
}

/// Compute lead times for disrupted test shots using V2 windowed scoring.
fn compute_lead_times_generic(
    encoder: &GenericHdcEncoder,
    global_reference: &ContinuousHV,
    test_shots: &[GenericShot],
    threshold: f64,
    config: &GenericEvalConfig,
) -> Vec<f64> {
    let mut lead_times = Vec::new();

    for shot in test_shots {
        if !shot.has_disruption {
            continue;
        }
        let onset_time = match shot.disruption_onset_time() {
            Some(t) => t,
            None => continue,
        };

        let n_samples = shot.samples.len();
        let n_ref = ((n_samples as f64) * config.reference_fraction).ceil() as usize;
        let n_ref = n_ref.min(n_samples);

        let reference = if n_ref >= config.min_reference_samples {
            let ref_encodings: Vec<ContinuousHV> = shot.samples[..n_ref]
                .iter()
                .map(|s| encoder.encode(s))
                .collect();
            ContinuousHV::bundle_owned(&ref_encodings)
        } else {
            global_reference.clone()
        };

        let w = config.window_size;
        for i in 0..n_samples {
            let start = (i + 1).saturating_sub(w);
            let window = &shot.samples[start..=i];
            let hv = encoder.encode_window(window);
            let sim = hv.similarity(&reference) as f64;
            let fe = if sim.is_finite() {
                (1.0 - sim).max(0.0)
            } else {
                1.0
            };

            if fe >= threshold {
                let lead = onset_time - shot.samples[i].time_s;
                if lead.is_finite() {
                    lead_times.push(lead);
                }
                break;
            }
        }
    }

    lead_times
}

// ── Train/Test Split ─────────────────────────────────────────────────────────

/// Stratified train/test split ensuring disrupted shots appear in both sets.
fn stratified_split(
    shots: Vec<GenericShot>,
    train_fraction: f64,
) -> (Vec<GenericShot>, Vec<GenericShot>) {
    let mut disrupted: Vec<GenericShot> = Vec::new();
    let mut normal: Vec<GenericShot> = Vec::new();

    for shot in shots {
        if shot.has_disruption {
            disrupted.push(shot);
        } else {
            normal.push(shot);
        }
    }

    let d_split = ((disrupted.len() as f64) * train_fraction).ceil() as usize;
    let d_split = d_split
        .min(disrupted.len().saturating_sub(1))
        .max(if disrupted.len() > 1 { 1 } else { 0 });
    let n_split = ((normal.len() as f64) * train_fraction).ceil() as usize;
    let n_split = n_split
        .min(normal.len().saturating_sub(1))
        .max(if normal.is_empty() { 0 } else { 1 });

    let mut train = Vec::new();
    let mut test = Vec::new();

    train.extend(disrupted.drain(..d_split));
    test.extend(disrupted);
    train.extend(normal.drain(..n_split));
    test.extend(normal);

    (train, test)
}

// ── Threshold Sweep ──────────────────────────────────────────────────────────

/// Sweep thresholds and return (best_threshold, best_f1, roc, auc).
fn sweep_thresholds(
    scored: &[(f64, bool)],
    max_thresh: f64,
    step: f64,
) -> (f64, f64, Vec<RocPoint>, f64) {
    let mut best_threshold = step;
    let mut best_f1 = 0.0;

    let mut thresh = step;
    while thresh <= max_thresh + 1e-9 {
        let cm = classify_at_threshold(scored, thresh);
        let f1 = cm.f1();
        if f1 > best_f1 {
            best_f1 = f1;
            best_threshold = thresh;
        }
        thresh += step;
    }

    let roc = compute_roc_curve(scored);
    let auc = compute_auc(&roc);

    (best_threshold, best_f1, roc, auc)
}

// ── Single Machine Evaluation ────────────────────────────────────────────────

/// Evaluate a single machine's dataset using both V1 and V2 methods.
///
/// V1: single-sample encoding, global healthy reference, coarse threshold sweep.
/// V2: temporal windowing, per-shot early-phase reference, fine threshold sweep.
pub fn evaluate_machine(
    config: &MachineDatasetConfig,
    eval_config: &GenericEvalConfig,
) -> Result<MachineEvaluationResult> {
    let shots = load_generic_csv(config)?;
    evaluate_machine_from_shots(config, eval_config, shots)
}

/// Inner evaluation on already-loaded shots (used by cross-machine transfer).
fn evaluate_machine_from_shots(
    config: &MachineDatasetConfig,
    eval_config: &GenericEvalConfig,
    shots: Vec<GenericShot>,
) -> Result<MachineEvaluationResult> {
    anyhow::ensure!(
        !shots.is_empty(),
        "No shots loaded for {}",
        config.machine_name
    );

    let total_shots = shots.len();
    let disrupted_shots = shots.iter().filter(|s| s.has_disruption).count();
    let n_sensors = config.sensor_columns.len();

    let (train_shots, test_shots) = stratified_split(shots, eval_config.train_fraction);

    // Build encoder from training normalization ranges
    let encoder = GenericHdcEncoder::from_shots(&train_shots, n_sensors, eval_config.seed_base);
    let global_reference = encoder.build_reference(&train_shots);

    let total_samples: usize = test_shots.iter().map(|s| s.samples.len()).sum();

    // ── V1: single-sample, global reference ──
    let test_samples_flat: Vec<GenericSample> = test_shots
        .iter()
        .flat_map(|shot| shot.samples.iter().cloned())
        .collect();

    let scored_v1 = score_samples_generic(&encoder, &global_reference, &test_samples_flat);
    let (_best_thresh_v1, best_f1_v1, _roc_v1, auc_v1) = sweep_thresholds(&scored_v1, 0.95, 0.05);

    // ── V2: temporal window + per-shot reference ──
    let scored_v2 = score_shots_v2_generic(&encoder, &global_reference, &test_shots, eval_config);
    let (best_thresh_v2, best_f1_v2, _roc_v2, auc_v2) = sweep_thresholds(
        &scored_v2,
        eval_config.fine_threshold_max,
        eval_config.fine_threshold_step,
    );

    // ── Lead time analysis at best V2 threshold ──
    let lead_times = compute_lead_times_generic(
        &encoder,
        &global_reference,
        &test_shots,
        best_thresh_v2,
        eval_config,
    );
    let total_disrupted = test_shots.iter().filter(|s| s.has_disruption).count();
    let lt_stats = LeadTimeStats::from_lead_times(&lead_times, total_disrupted);

    Ok(MachineEvaluationResult {
        machine_name: config.machine_name.clone(),
        total_shots,
        disrupted_shots,
        total_samples,
        n_sensors,
        auc_v1,
        auc_v2,
        best_f1_v1,
        best_f1_v2,
        lead_time_mean_ms: lt_stats.mean_s * 1000.0,
        lead_time_median_ms: lt_stats.median_s * 1000.0,
    })
}

// ── Cross-Machine Transfer ───────────────────────────────────────────────────

/// Evaluate each machine independently AND measure cross-machine transfer.
///
/// Cross-machine transfer: for each ordered pair (A, B) where A != B, train
/// the encoder on machine A's healthy data and test on machine B's test set.
/// The reported `cross_machine_transfer_auc` is the average over all such pairs.
pub fn evaluate_cross_machine(
    configs: &[MachineDatasetConfig],
    eval_config: &GenericEvalConfig,
) -> Result<CrossMachineResult> {
    anyhow::ensure!(
        !configs.is_empty(),
        "At least one machine config is required"
    );

    // Load all datasets upfront
    let mut all_data: Vec<(MachineDatasetConfig, Vec<GenericShot>)> = Vec::new();
    for cfg in configs {
        let shots = load_generic_csv(cfg)
            .with_context(|| format!("Failed to load data for {}", cfg.machine_name))?;
        all_data.push((cfg.clone(), shots));
    }

    // Independent per-machine evaluation
    let mut per_machine = Vec::new();
    for (cfg, shots) in &all_data {
        let result = evaluate_machine_from_shots(cfg, eval_config, shots.clone())?;
        per_machine.push(result);
    }

    let mean_auc_v1 = if per_machine.is_empty() {
        0.0
    } else {
        per_machine.iter().map(|r| r.auc_v1).sum::<f64>() / per_machine.len() as f64
    };
    let mean_auc_v2 = if per_machine.is_empty() {
        0.0
    } else {
        per_machine.iter().map(|r| r.auc_v2).sum::<f64>() / per_machine.len() as f64
    };

    // Cross-machine transfer: train on A, test on B
    let mut transfer_aucs = Vec::new();

    if all_data.len() >= 2 {
        for (i, (cfg_a, shots_a)) in all_data.iter().enumerate() {
            let n_sensors_a = cfg_a.sensor_columns.len();

            // Use all of A's healthy data as the reference
            let encoder_a =
                GenericHdcEncoder::from_shots(shots_a, n_sensors_a, eval_config.seed_base);
            let reference_a = encoder_a.build_reference(shots_a);

            for (j, (_cfg_b, shots_b)) in all_data.iter().enumerate() {
                if i == j {
                    continue;
                }

                // Only attempt transfer if sensor counts match
                let n_sensors_b = _cfg_b.sensor_columns.len();
                if n_sensors_a != n_sensors_b {
                    continue;
                }

                // Score B's data against A's reference + A's encoder (normalization)
                let samples_b: Vec<GenericSample> = shots_b
                    .iter()
                    .flat_map(|s| s.samples.iter().cloned())
                    .collect();

                let scored = score_samples_generic(&encoder_a, &reference_a, &samples_b);
                let roc = compute_roc_curve(&scored);
                let auc = compute_auc(&roc);
                transfer_aucs.push(auc);
            }
        }
    }

    let cross_machine_transfer_auc = if transfer_aucs.is_empty() {
        0.0
    } else {
        transfer_aucs.iter().sum::<f64>() / transfer_aucs.len() as f64
    };

    Ok(CrossMachineResult {
        per_machine,
        mean_auc_v1,
        mean_auc_v2,
        cross_machine_transfer_auc,
    })
}

// ── Report Generation ────────────────────────────────────────────────────────

/// Generate a Markdown comparison table from cross-machine results.
pub fn generate_cross_machine_report(result: &CrossMachineResult) -> String {
    let mut md = String::new();

    md.push_str("# Cross-Machine Disruption Prediction Report\n\n");

    md.push_str("## Per-Machine Results\n\n");
    md.push_str("| Machine | Shots | Disrupted | Sensors | AUC V1 | AUC V2 | F1 V1 | F1 V2 | Lead Time (ms) |\n");
    md.push_str("|---------|-------|-----------|---------|--------|--------|-------|-------|----------------|\n");

    for r in &result.per_machine {
        md.push_str(&format!(
            "| {} | {} | {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.1} |\n",
            r.machine_name,
            r.total_shots,
            r.disrupted_shots,
            r.n_sensors,
            r.auc_v1,
            r.auc_v2,
            r.best_f1_v1,
            r.best_f1_v2,
            r.lead_time_median_ms,
        ));
    }

    md.push_str("\n## Aggregate Metrics\n\n");
    md.push_str("| Metric | Value |\n");
    md.push_str("|--------|-------|\n");
    md.push_str(&format!("| Mean AUC V1 | {:.4} |\n", result.mean_auc_v1));
    md.push_str(&format!("| Mean AUC V2 | {:.4} |\n", result.mean_auc_v2));
    md.push_str(&format!(
        "| Cross-Machine Transfer AUC | {:.4} |\n",
        result.cross_machine_transfer_auc
    ));

    md.push_str("\n## Methodology\n\n");
    md.push_str("- **V1**: Single-sample HDC encoding, global healthy reference, coarse threshold sweep (0.05-0.95)\n");
    md.push_str("- **V2**: Temporal window (5 samples) + per-shot early-phase reference, fine threshold sweep (0.005-0.30)\n");
    md.push_str("- **Cross-Machine Transfer**: Train encoder/reference on machine A's healthy data, test on machine B\n");
    md.push_str("- **HDC Dimension**: 16,384\n");
    md.push_str("- **Normalization**: Per-machine min-max to [0, 1]\n");

    md
}

// ── Synthetic Data Generation (for testing) ──────────────────────────────────

/// Generate synthetic shots for a hypothetical machine.
///
/// Creates `n_shots` with `samples_per_shot` time steps. Disrupted shots have
/// their sensor values diverge from the healthy reference in the second half.
pub fn generate_synthetic_machine(
    machine_name: &str,
    n_sensors: usize,
    n_shots: usize,
    samples_per_shot: usize,
    disruption_fraction: f64,
    seed: u64,
) -> Vec<GenericShot> {
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n_disrupted = ((n_shots as f64) * disruption_fraction).ceil() as usize;

    let mut shots = Vec::with_capacity(n_shots);

    for shot_idx in 0..n_shots {
        let is_disrupted = shot_idx < n_disrupted;
        let shot_id = format!("{}-{:04}", machine_name, shot_idx);

        let mut samples = Vec::with_capacity(samples_per_shot);

        // Base sensor values for this shot (healthy regime)
        let base_values: Vec<f32> = (0..n_sensors).map(|_| rng.gen_range(0.3..0.7)).collect();

        for t_idx in 0..samples_per_shot {
            let time_s = t_idx as f64 * 0.001; // 1 ms steps
            let fraction = t_idx as f64 / samples_per_shot as f64;

            let mut sensor_values = Vec::with_capacity(n_sensors);
            let mut is_disruption = false;

            for (s, &base) in base_values.iter().enumerate() {
                let noise: f32 = rng.gen_range(-0.05..0.05);
                let value = if is_disrupted && fraction > 0.5 {
                    is_disruption = true;
                    // Diverge: sensor values shift significantly
                    let drift = (fraction - 0.5) * 2.0; // 0..1 over second half
                    let direction = if s % 2 == 0 { 1.0 } else { -1.0 };
                    base + direction * drift as f32 * 0.5 + noise
                } else {
                    base + noise
                };
                sensor_values.push(value);
            }

            samples.push(GenericSample {
                shot_id: shot_id.clone(),
                time_s,
                sensor_values,
                is_disruption,
            });
        }

        let has_disruption = samples.iter().any(|s| s.is_disruption);
        shots.push(GenericShot {
            shot_id,
            samples,
            has_disruption,
        });
    }

    shots
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Helper: write a temp CSV with custom column names and return the path.
    fn write_temp_csv(
        dir: &tempfile::TempDir,
        filename: &str,
        headers: &[&str],
        rows: &[Vec<&str>],
    ) -> PathBuf {
        let path = dir.path().join(filename);
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "{}", headers.join(",")).unwrap();
        for row in rows {
            writeln!(f, "{}", row.join(",")).unwrap();
        }
        path
    }

    #[test]
    fn test_generic_csv_loading() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_temp_csv(
            &dir,
            "test_machine.csv",
            &["pulse", "t", "flag", "sensor_a", "sensor_b"],
            &[
                vec!["100", "0.0", "0", "1.5", "2.3"],
                vec!["100", "0.001", "0", "1.6", "2.2"],
                vec!["100", "0.002", "1", "3.0", "0.5"],
                vec!["200", "0.0", "0", "1.4", "2.4"],
                vec!["200", "0.001", "0", "1.5", "2.3"],
                // Row with NaN should be skipped
                vec!["200", "0.002", "0", "NaN", "2.1"],
            ],
        );

        let config = MachineDatasetConfig {
            machine_name: "Test Machine".to_string(),
            csv_path: path,
            shot_id_column: "pulse".to_string(),
            time_column: "t".to_string(),
            label_column: "flag".to_string(),
            sensor_columns: vec!["sensor_a".to_string(), "sensor_b".to_string()],
            label_positive_value: "1".to_string(),
        };

        let shots = load_generic_csv(&config).unwrap();
        assert_eq!(shots.len(), 2, "should have 2 shots");

        // Shot 100: 3 samples, has disruption
        let shot100 = shots.iter().find(|s| s.shot_id == "100").unwrap();
        assert_eq!(shot100.samples.len(), 3);
        assert!(shot100.has_disruption);
        assert!(!shot100.samples[0].is_disruption);
        assert!(shot100.samples[2].is_disruption);

        // Shot 200: 2 samples (NaN row skipped), no disruption
        let shot200 = shots.iter().find(|s| s.shot_id == "200").unwrap();
        assert_eq!(shot200.samples.len(), 2);
        assert!(!shot200.has_disruption);
    }

    #[test]
    fn test_generic_encoder_round_trip() {
        // Generate synthetic data and verify encoding produces finite non-zero HVs
        let shots = generate_synthetic_machine("enc-test", 4, 5, 20, 0.4, 42);
        let encoder = GenericHdcEncoder::from_shots(&shots, 4, 0xBEEF);
        let reference = encoder.build_reference(&shots);

        // Reference should be non-zero
        let ref_norm: f32 = reference
            .as_slice()
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!(ref_norm > 0.0, "reference HV should be non-zero");

        // Encode a healthy sample and a disrupted sample
        let healthy = shots.iter().find(|s| !s.has_disruption).unwrap();
        let disrupted = shots.iter().find(|s| s.has_disruption).unwrap();

        let h_hv = encoder.encode(&healthy.samples[0]);
        let d_hv = encoder.encode(disrupted.samples.last().unwrap());

        let h_sim = h_hv.similarity(&reference);
        let d_sim = d_hv.similarity(&reference);

        assert!(h_sim.is_finite());
        assert!(d_sim.is_finite());
    }

    #[test]
    fn test_normalization_ranges_generic() {
        let shots = generate_synthetic_machine("norm-test", 3, 4, 10, 0.5, 99);
        let ranges = GenericNormRanges::from_shots(&shots, 3);

        assert_eq!(ranges.mins.len(), 3);
        assert_eq!(ranges.maxs.len(), 3);

        for i in 0..3 {
            assert!(ranges.mins[i].is_finite());
            assert!(ranges.maxs[i].is_finite());
            assert!(ranges.maxs[i] >= ranges.mins[i]);
        }

        // Normalized values should be in [0, 1]
        for shot in &shots {
            for sample in &shot.samples {
                let normed = ranges.normalize_sample(sample);
                for v in &normed {
                    assert!(
                        *v >= 0.0 && *v <= 1.0,
                        "normalized value {} out of range",
                        v
                    );
                }
            }
        }
    }

    #[test]
    fn test_cross_machine_on_synthetic() {
        // Create two synthetic "machines" with different sensor distributions
        let shots_a = generate_synthetic_machine("Alpha", 4, 20, 50, 0.3, 1234);
        let shots_b = generate_synthetic_machine("Beta", 4, 20, 50, 0.3, 5678);

        // Write them as CSVs so we can use the full pipeline
        let dir = tempfile::tempdir().unwrap();

        let path_a = write_synthetic_csv(&dir, "alpha.csv", &shots_a, 4);
        let path_b = write_synthetic_csv(&dir, "beta.csv", &shots_b, 4);

        let config_a = MachineDatasetConfig {
            machine_name: "Alpha Tokamak".to_string(),
            csv_path: path_a,
            shot_id_column: "shot_id".to_string(),
            time_column: "time".to_string(),
            label_column: "disrupted".to_string(),
            sensor_columns: (0..4).map(|i| format!("sensor_{}", i)).collect(),
            label_positive_value: "1".to_string(),
        };

        let config_b = MachineDatasetConfig {
            machine_name: "Beta Tokamak".to_string(),
            csv_path: path_b,
            shot_id_column: "shot_id".to_string(),
            time_column: "time".to_string(),
            label_column: "disrupted".to_string(),
            sensor_columns: (0..4).map(|i| format!("sensor_{}", i)).collect(),
            label_positive_value: "1".to_string(),
        };

        let eval_config = GenericEvalConfig::default();
        let result = evaluate_cross_machine(&[config_a, config_b], &eval_config).unwrap();

        // Basic structural assertions
        assert_eq!(result.per_machine.len(), 2);
        assert_eq!(result.per_machine[0].machine_name, "Alpha Tokamak");
        assert_eq!(result.per_machine[1].machine_name, "Beta Tokamak");

        // AUCs should be finite and in [0, 1]
        for r in &result.per_machine {
            assert!(
                r.auc_v1 >= 0.0 && r.auc_v1 <= 1.0,
                "AUC V1 out of range: {}",
                r.auc_v1
            );
            assert!(
                r.auc_v2 >= 0.0 && r.auc_v2 <= 1.0,
                "AUC V2 out of range: {}",
                r.auc_v2
            );
            assert!(r.best_f1_v1 >= 0.0 && r.best_f1_v1 <= 1.0);
            assert!(r.best_f1_v2 >= 0.0 && r.best_f1_v2 <= 1.0);
        }

        assert!(result.mean_auc_v1 >= 0.0 && result.mean_auc_v1 <= 1.0);
        assert!(result.mean_auc_v2 >= 0.0 && result.mean_auc_v2 <= 1.0);
        assert!(
            result.cross_machine_transfer_auc >= 0.0 && result.cross_machine_transfer_auc <= 1.0
        );

        // Report generation should not panic
        let report = generate_cross_machine_report(&result);
        assert!(report.contains("Alpha Tokamak"));
        assert!(report.contains("Beta Tokamak"));
        assert!(report.contains("Cross-Machine Transfer AUC"));
    }

    #[test]
    fn test_single_machine_synthetic() {
        let dir = tempfile::tempdir().unwrap();
        let shots = generate_synthetic_machine("Gamma", 3, 15, 40, 0.4, 7777);
        let path = write_synthetic_csv(&dir, "gamma.csv", &shots, 3);

        let config = MachineDatasetConfig {
            machine_name: "Gamma Tokamak".to_string(),
            csv_path: path,
            shot_id_column: "shot_id".to_string(),
            time_column: "time".to_string(),
            label_column: "disrupted".to_string(),
            sensor_columns: (0..3).map(|i| format!("sensor_{}", i)).collect(),
            label_positive_value: "1".to_string(),
        };

        let eval_config = GenericEvalConfig::default();
        let result = evaluate_machine(&config, &eval_config).unwrap();

        assert_eq!(result.machine_name, "Gamma Tokamak");
        assert_eq!(result.total_shots, 15);
        assert_eq!(result.n_sensors, 3);
        assert!(result.auc_v1 >= 0.0 && result.auc_v1 <= 1.0);
        assert!(result.auc_v2 >= 0.0 && result.auc_v2 <= 1.0);
    }

    #[test]
    fn test_cmod_config_structure() {
        // Verify the C-Mod config has the right shape (cannot run without real data)
        let config = MachineDatasetConfig::cmod();
        assert_eq!(config.machine_name, "Alcator C-Mod");
        assert_eq!(config.shot_id_column, "discharge_ID");
        assert_eq!(config.time_column, "time");
        assert_eq!(config.label_column, "density_limit_phase");
        assert_eq!(config.sensor_columns.len(), 6);
        assert_eq!(config.label_positive_value, "1");
    }

    #[test]
    fn test_itpa_template_config() {
        let config = MachineDatasetConfig::itpa_template();
        assert_eq!(config.machine_name, "ITPA (template)");
        assert_eq!(config.sensor_columns.len(), 9);
    }

    #[test]
    fn test_report_generation() {
        let result = CrossMachineResult {
            per_machine: vec![
                MachineEvaluationResult {
                    machine_name: "Machine A".to_string(),
                    total_shots: 100,
                    disrupted_shots: 30,
                    total_samples: 5000,
                    n_sensors: 6,
                    auc_v1: 0.85,
                    auc_v2: 0.91,
                    best_f1_v1: 0.78,
                    best_f1_v2: 0.84,
                    lead_time_mean_ms: 50.0,
                    lead_time_median_ms: 45.0,
                },
                MachineEvaluationResult {
                    machine_name: "Machine B".to_string(),
                    total_shots: 80,
                    disrupted_shots: 25,
                    total_samples: 4000,
                    n_sensors: 6,
                    auc_v1: 0.80,
                    auc_v2: 0.87,
                    best_f1_v1: 0.72,
                    best_f1_v2: 0.80,
                    lead_time_mean_ms: 60.0,
                    lead_time_median_ms: 55.0,
                },
            ],
            mean_auc_v1: 0.825,
            mean_auc_v2: 0.89,
            cross_machine_transfer_auc: 0.75,
        };

        let report = generate_cross_machine_report(&result);
        assert!(report.contains("Machine A"));
        assert!(report.contains("Machine B"));
        assert!(report.contains("0.825"));
        assert!(report.contains("0.7500"));
        assert!(report.contains("HDC Dimension"));
    }

    /// Helper: write synthetic shots to CSV for the full pipeline test.
    fn write_synthetic_csv(
        dir: &tempfile::TempDir,
        filename: &str,
        shots: &[GenericShot],
        n_sensors: usize,
    ) -> PathBuf {
        let path = dir.path().join(filename);
        let mut f = std::fs::File::create(&path).unwrap();

        // Header
        let mut header = "shot_id,time,disrupted".to_string();
        for i in 0..n_sensors {
            header.push_str(&format!(",sensor_{}", i));
        }
        writeln!(f, "{}", header).unwrap();

        // Rows
        for shot in shots {
            for sample in &shot.samples {
                let label = if sample.is_disruption { "1" } else { "0" };
                let mut row = format!("{},{},{}", sample.shot_id, sample.time_s, label);
                for v in &sample.sensor_values {
                    row.push_str(&format!(",{}", v));
                }
                writeln!(f, "{}", row).unwrap();
            }
        }

        path
    }
}
