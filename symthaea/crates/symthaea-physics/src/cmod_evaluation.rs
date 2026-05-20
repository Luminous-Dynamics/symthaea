// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # C-Mod Density Limit Evaluation Module
//!
//! Evaluates the Fusion Digital Twin (Genesis Challenge 3) against real Alcator C-Mod
//! tokamak data from the MIT PSFC Open Density Limit Database (`DL_DataFrame.csv`).
//!
//! ## Data Format
//!
//! The DL_DataFrame has 6 plasma sensors per time step:
//! - **density**: Line-averaged electron density (10^20 m^-3)
//! - **elongation**: Plasma elongation (dimensionless)
//! - **minor_radius**: Plasma minor radius (m)
//! - **plasma_current**: Plasma current (MA)
//! - **toroidal_B_field**: Toroidal magnetic field (T)
//! - **triangularity**: Plasma triangularity (dimensionless)
//!
//! Label: `density_limit_phase` (0 = normal, 1 = density limit precursor/disruption)
//!
//! ## Architecture
//!
//! 1. **DensityLimitEncoder**: 6-basis-vector HDC encoder mapping sensor values to 16,384D ContinuousHV
//! 2. **Free Energy Classification**: cosine distance from a healthy-plasma reference HV
//! 3. **Phi-Decline Detection**: sliding-window integration decline as secondary signal
//! 4. **Threshold Sweep**: finds optimal free-energy threshold via F1 maximization
//! 5. **ROC/AUC**: full receiver operating characteristic with trapezoidal AUC

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

use crate::cmod_adapter::CModPhiMonitor;

// ── Constants ────────────────────────────────────────────────────────────────

/// Number of sensors in the DL_DataFrame.
const NUM_SENSORS: usize = 6;

/// Deterministic seeds for basis vector generation (one per sensor).
const SENSOR_SEEDS: [u64; NUM_SENSORS] = [
    0xD10_0001, // density
    0xD10_0002, // elongation
    0xD10_0003, // minor_radius
    0xD10_0004, // plasma_current
    0xD10_0005, // toroidal_B_field
    0xD10_0006, // triangularity
];

/// Sensor names matching CSV column order.
const SENSOR_NAMES: [&str; NUM_SENSORS] = [
    "density",
    "elongation",
    "minor_radius",
    "plasma_current",
    "toroidal_B_field",
    "triangularity",
];

// ── Data Structures ──────────────────────────────────────────────────────────

/// A single time-step sample from the DL_DataFrame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DensityLimitSample {
    /// Discharge (shot) identifier.
    pub shot_id: u64,
    /// Time in seconds from shot start.
    pub time_s: f64,
    /// Line-averaged electron density (10^20 m^-3).
    pub density: f32,
    /// Plasma elongation (dimensionless).
    pub elongation: f32,
    /// Plasma minor radius (m).
    pub minor_radius: f32,
    /// Plasma current (MA).
    pub plasma_current: f32,
    /// Toroidal magnetic field (T).
    pub toroidal_b_field: f32,
    /// Plasma triangularity (dimensionless).
    pub triangularity: f32,
    /// Whether this sample is in a density-limit phase (disruption precursor).
    pub is_disruption: bool,
}

impl DensityLimitSample {
    /// Return sensor values as a 6-element array in canonical order.
    pub fn sensor_values(&self) -> [f32; NUM_SENSORS] {
        [
            self.density,
            self.elongation,
            self.minor_radius,
            self.plasma_current,
            self.toroidal_b_field,
            self.triangularity,
        ]
    }
}

/// A complete shot (discharge) grouped by shot_id.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DensityLimitShot {
    /// Discharge identifier.
    pub shot_id: u64,
    /// Time-ordered samples for this shot.
    pub samples: Vec<DensityLimitSample>,
    /// Whether any sample in this shot is labeled as disruption.
    pub has_disruption: bool,
}

impl DensityLimitShot {
    /// Time of the first disruption label in this shot (if any).
    pub fn disruption_onset_time(&self) -> Option<f64> {
        self.samples
            .iter()
            .find(|s| s.is_disruption)
            .map(|s| s.time_s)
    }
}

// ── CSV Loading ──────────────────────────────────────────────────────────────

/// Raw CSV row matching the DL_DataFrame columns.
#[derive(Debug, Deserialize)]
#[allow(non_snake_case)]
struct CsvRow {
    discharge_ID: u64,
    time: f64,
    density_limit_phase: u8,
    density: f32,
    elongation: f32,
    minor_radius: f32,
    plasma_current: f32,
    toroidal_B_field: f32,
    triangularity: f32,
}

/// Load the DL_DataFrame CSV and group by discharge_ID into shots.
///
/// Shots are sorted by shot_id; samples within each shot are sorted by time.
/// Rows with NaN in any sensor column are silently skipped.
pub fn load_density_limit_csv(path: &Path) -> Result<Vec<DensityLimitShot>> {
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("Failed to open CSV: {}", path.display()))?;

    let mut shots_map: HashMap<u64, Vec<DensityLimitSample>> = HashMap::new();

    for result in reader.deserialize() {
        let row: CsvRow = result.context("Failed to parse CSV row")?;

        // Skip rows with non-finite sensor values
        if !row.density.is_finite()
            || !row.elongation.is_finite()
            || !row.minor_radius.is_finite()
            || !row.plasma_current.is_finite()
            || !row.toroidal_B_field.is_finite()
            || !row.triangularity.is_finite()
            || !row.time.is_finite()
        {
            continue;
        }

        let sample = DensityLimitSample {
            shot_id: row.discharge_ID,
            time_s: row.time,
            density: row.density,
            elongation: row.elongation,
            minor_radius: row.minor_radius,
            plasma_current: row.plasma_current,
            toroidal_b_field: row.toroidal_B_field,
            triangularity: row.triangularity,
            is_disruption: row.density_limit_phase != 0,
        };

        shots_map.entry(row.discharge_ID).or_default().push(sample);
    }

    // Sort samples within each shot by time, then collect into sorted shots
    let mut shots: Vec<DensityLimitShot> = shots_map
        .into_iter()
        .map(|(shot_id, mut samples)| {
            samples.sort_by(|a, b| {
                a.time_s
                    .partial_cmp(&b.time_s)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let has_disruption = samples.iter().any(|s| s.is_disruption);
            DensityLimitShot {
                shot_id,
                samples,
                has_disruption,
            }
        })
        .collect();

    shots.sort_by_key(|s| s.shot_id);
    Ok(shots)
}

// ── Normalization Ranges ─────────────────────────────────────────────────────

/// Per-sensor normalization range [min, max] for mapping to [0, 1].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationRanges {
    pub mins: [f32; NUM_SENSORS],
    pub maxs: [f32; NUM_SENSORS],
}

impl NormalizationRanges {
    /// Compute normalization ranges from a set of shots.
    pub fn from_shots(shots: &[DensityLimitShot]) -> Self {
        let mut mins = [f32::INFINITY; NUM_SENSORS];
        let mut maxs = [f32::NEG_INFINITY; NUM_SENSORS];

        for shot in shots {
            for sample in &shot.samples {
                let vals = sample.sensor_values();
                for (i, &v) in vals.iter().enumerate() {
                    if v.is_finite() {
                        mins[i] = mins[i].min(v);
                        maxs[i] = maxs[i].max(v);
                    }
                }
            }
        }

        // Guard against degenerate ranges
        for i in 0..NUM_SENSORS {
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

    /// Normalize a sensor value to [0, 1].
    pub fn normalize(&self, sensor_idx: usize, value: f32) -> f32 {
        if sensor_idx >= NUM_SENSORS || !value.is_finite() {
            return 0.0;
        }
        let range = self.maxs[sensor_idx] - self.mins[sensor_idx];
        if range.abs() < 1e-10 {
            return 0.0;
        }
        ((value - self.mins[sensor_idx]) / range).clamp(0.0, 1.0)
    }

    /// Normalize all 6 sensors for a sample.
    pub fn normalize_sample(&self, sample: &DensityLimitSample) -> [f32; NUM_SENSORS] {
        let vals = sample.sensor_values();
        let mut normed = [0.0f32; NUM_SENSORS];
        for i in 0..NUM_SENSORS {
            normed[i] = self.normalize(i, vals[i]);
        }
        normed
    }
}

// ── HDC Encoder ──────────────────────────────────────────────────────────────

/// HDC encoder for the 6 DL_DataFrame sensors.
///
/// Each sensor gets a deterministic random basis vector. Encoding works by
/// weighted superposition: `encode_weighted(bases, normalized_values)`.
#[derive(Clone)]
pub struct DensityLimitEncoder {
    bases: Vec<ContinuousHV>,
    ranges: NormalizationRanges,
}

impl DensityLimitEncoder {
    /// Create encoder with pre-computed normalization ranges.
    pub fn new(ranges: NormalizationRanges) -> Self {
        let bases: Vec<ContinuousHV> = SENSOR_SEEDS
            .iter()
            .map(|&seed| ContinuousHV::random(HDC_DIMENSION, seed))
            .collect();
        Self { bases, ranges }
    }

    /// Create encoder and compute normalization ranges from training shots.
    pub fn from_shots(shots: &[DensityLimitShot]) -> Self {
        let ranges = NormalizationRanges::from_shots(shots);
        Self::new(ranges)
    }

    /// Encode a single sample into a 16,384D ContinuousHV.
    pub fn encode(&self, sample: &DensityLimitSample) -> ContinuousHV {
        let normed = self.ranges.normalize_sample(sample);
        ContinuousHV::encode_weighted(&self.bases, &normed)
    }

    /// Build a reference HV from a set of healthy (non-disruption) samples.
    ///
    /// Returns the element-wise average (bundle) of all healthy encodings.
    /// If no healthy samples are found, returns a zero HV.
    pub fn build_reference(&self, shots: &[DensityLimitShot]) -> ContinuousHV {
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

    /// Encode a temporal window of consecutive samples into a single HV.
    ///
    /// Takes `N` consecutive samples, encodes each individually, then bundles
    /// them into a single ContinuousHV that captures temporal context.
    /// A disrupting plasma evolves differently over time than a stable one —
    /// windowed encoding captures this trajectory information.
    pub fn encode_window(&self, samples: &[DensityLimitSample]) -> ContinuousHV {
        if samples.is_empty() {
            return ContinuousHV::zero(HDC_DIMENSION);
        }
        if samples.len() == 1 {
            return self.encode(&samples[0]);
        }
        let encoded: Vec<ContinuousHV> = samples.iter().map(|s| self.encode(s)).collect();
        ContinuousHV::bundle_owned(&encoded)
    }

    /// Access the normalization ranges.
    pub fn ranges(&self) -> &NormalizationRanges {
        &self.ranges
    }
}

// ── Confusion Matrix & Metrics ───────────────────────────────────────────────

/// Standard binary classification confusion matrix.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ConfusionMatrix {
    /// True positives (predicted disruption, was disruption).
    pub tp: u64,
    /// False positives (predicted disruption, was normal).
    pub fp: u64,
    /// True negatives (predicted normal, was normal).
    pub tn: u64,
    /// False negatives (predicted normal, was disruption).
    pub fn_: u64,
}

impl ConfusionMatrix {
    /// Create from counts.
    pub fn new(tp: u64, fp: u64, tn: u64, fn_: u64) -> Self {
        Self { tp, fp, tn, fn_ }
    }

    /// Total number of samples.
    pub fn total(&self) -> u64 {
        self.tp + self.fp + self.tn + self.fn_
    }

    /// Accuracy: (TP + TN) / total.
    pub fn accuracy(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        (self.tp + self.tn) as f64 / total as f64
    }

    /// Precision: TP / (TP + FP).
    pub fn precision(&self) -> f64 {
        let denom = self.tp + self.fp;
        if denom == 0 {
            return 0.0;
        }
        self.tp as f64 / denom as f64
    }

    /// Recall (sensitivity, true positive rate): TP / (TP + FN).
    pub fn recall(&self) -> f64 {
        let denom = self.tp + self.fn_;
        if denom == 0 {
            return 0.0;
        }
        self.tp as f64 / denom as f64
    }

    /// F1 score: harmonic mean of precision and recall.
    pub fn f1(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if (p + r).abs() < 1e-12 {
            return 0.0;
        }
        2.0 * p * r / (p + r)
    }

    /// Specificity (true negative rate): TN / (TN + FP).
    pub fn specificity(&self) -> f64 {
        let denom = self.tn + self.fp;
        if denom == 0 {
            return 0.0;
        }
        self.tn as f64 / denom as f64
    }

    /// False positive rate: FP / (FP + TN).
    pub fn fpr(&self) -> f64 {
        1.0 - self.specificity()
    }

    /// True positive rate (same as recall).
    pub fn tpr(&self) -> f64 {
        self.recall()
    }
}

// ── ROC Curve ────────────────────────────────────────────────────────────────

/// A single point on the ROC curve.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RocPoint {
    /// False positive rate.
    pub fpr: f64,
    /// True positive rate.
    pub tpr: f64,
    /// Threshold that produced this point.
    pub threshold: f64,
}

/// Compute ROC curve from scored predictions.
///
/// `scored`: Vec of (free_energy, true_label_is_disruption).
/// Returns ROC points sorted by ascending FPR.
pub fn compute_roc_curve(scored: &[(f64, bool)]) -> Vec<RocPoint> {
    if scored.is_empty() {
        return vec![];
    }

    let total_pos = scored.iter().filter(|(_, lab)| *lab).count() as f64;
    let total_neg = scored.iter().filter(|(_, lab)| !*lab).count() as f64;

    if total_pos < 1.0 || total_neg < 1.0 {
        // Degenerate: only one class present
        return vec![
            RocPoint {
                fpr: 0.0,
                tpr: 0.0,
                threshold: 1.0,
            },
            RocPoint {
                fpr: 1.0,
                tpr: 1.0,
                threshold: 0.0,
            },
        ];
    }

    // Collect unique thresholds from scores
    let mut thresholds: Vec<f64> = scored.iter().map(|(s, _)| *s).collect();
    thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    thresholds.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

    // Add boundary thresholds
    let mut all_thresholds = vec![thresholds.last().copied().unwrap_or(1.0) + 0.01];
    all_thresholds.extend(thresholds.iter().rev());

    let mut points = Vec::with_capacity(all_thresholds.len());

    for &thresh in &all_thresholds {
        let mut tp = 0u64;
        let mut fp = 0u64;

        for &(score, is_pos) in scored {
            if score >= thresh {
                if is_pos {
                    tp += 1;
                } else {
                    fp += 1;
                }
            }
        }

        let tpr = tp as f64 / total_pos;
        let fpr = fp as f64 / total_neg;
        points.push(RocPoint {
            fpr,
            tpr,
            threshold: thresh,
        });
    }

    // Sort by FPR ascending, then TPR ascending for consistent curve
    points.sort_by(|a, b| {
        a.fpr
            .partial_cmp(&b.fpr)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                a.tpr
                    .partial_cmp(&b.tpr)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });

    points
}

/// Compute AUC via trapezoidal integration of ROC points.
///
/// Points must be sorted by ascending FPR.
pub fn compute_auc(roc: &[RocPoint]) -> f64 {
    if roc.len() < 2 {
        return 0.0;
    }

    let mut auc = 0.0;
    for i in 1..roc.len() {
        let dx = roc[i].fpr - roc[i - 1].fpr;
        let avg_tpr = (roc[i].tpr + roc[i - 1].tpr) / 2.0;
        auc += dx * avg_tpr;
    }

    auc.clamp(0.0, 1.0)
}

// ── Lead Time Analysis ───────────────────────────────────────────────────────

/// Statistics on detection lead time before disruption onset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeadTimeStats {
    /// Mean lead time (seconds before disruption onset).
    pub mean_s: f64,
    /// Median lead time (seconds).
    pub median_s: f64,
    /// Minimum lead time (seconds).
    pub min_s: f64,
    /// Maximum lead time (seconds).
    pub max_s: f64,
    /// Number of disrupted shots with a valid detection.
    pub detected_count: usize,
    /// Total number of disrupted shots.
    pub total_disrupted: usize,
}

impl LeadTimeStats {
    /// Compute from a list of lead times in seconds.
    pub fn from_lead_times(lead_times: &[f64], total_disrupted: usize) -> Self {
        if lead_times.is_empty() {
            return Self {
                mean_s: 0.0,
                median_s: 0.0,
                min_s: 0.0,
                max_s: 0.0,
                detected_count: 0,
                total_disrupted,
            };
        }

        let mut sorted = lead_times.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let sum: f64 = sorted.iter().sum();
        let mean = sum / sorted.len() as f64;
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        Self {
            mean_s: mean,
            median_s: median,
            min_s: sorted[0],
            max_s: sorted[sorted.len() - 1],
            detected_count: sorted.len(),
            total_disrupted,
        }
    }
}

// ── Evaluation Config ────────────────────────────────────────────────────────

/// Configuration for the density-limit evaluation harness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationConfig {
    /// Fraction of shots used for training (rest for test). Default: 0.8.
    pub train_fraction: f64,
    /// Sliding window size for Phi-decline monitor. Default: 10.
    pub window_size: usize,
    /// Phi decline ratio threshold for `is_declining()`. Default: 0.9.
    pub phi_decline_threshold: f64,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            train_fraction: 0.8,
            window_size: 10,
            phi_decline_threshold: 0.9,
        }
    }
}

// ── Evaluation Report ────────────────────────────────────────────────────────

/// Complete evaluation report for the density-limit prediction task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationReport {
    /// Best free-energy threshold (maximizes F1).
    pub best_threshold: f64,
    /// Best F1 score achieved.
    pub best_f1: f64,
    /// Confusion matrix at the best threshold.
    pub confusion_matrix: ConfusionMatrix,
    /// ROC curve points.
    pub roc_curve: Vec<RocPoint>,
    /// Area under the ROC curve.
    pub auc: f64,
    /// Lead time statistics for disrupted shots.
    pub lead_time_stats: LeadTimeStats,
    /// Number of training shots.
    pub total_shots_train: usize,
    /// Number of test shots.
    pub total_shots_test: usize,
    /// Samples per class in the test set: (normal, disruption).
    pub samples_per_class: (usize, usize),
    /// Phi-decline method confusion matrix (for comparison).
    pub phi_decline_cm: ConfusionMatrix,
    /// Normalization ranges used.
    pub normalization_ranges: NormalizationRanges,
    /// All threshold sweep results: (threshold, f1, precision, recall).
    pub threshold_sweep: Vec<(f64, f64, f64, f64)>,
}

impl EvaluationReport {
    /// Render the report as a human-readable Markdown string.
    pub fn to_markdown(&self) -> String {
        let cm = &self.confusion_matrix;
        let phi_cm = &self.phi_decline_cm;
        let lt = &self.lead_time_stats;

        let mut md = String::new();
        md.push_str("# C-Mod Density Limit Evaluation Report\n\n");
        md.push_str("## Dataset Split\n\n");
        md.push_str(&format!("| Metric | Value |\n|--------|-------|\n"));
        md.push_str(&format!(
            "| Training shots | {} |\n",
            self.total_shots_train
        ));
        md.push_str(&format!("| Test shots | {} |\n", self.total_shots_test));
        md.push_str(&format!(
            "| Test normal samples | {} |\n",
            self.samples_per_class.0
        ));
        md.push_str(&format!(
            "| Test disruption samples | {} |\n",
            self.samples_per_class.1
        ));
        md.push_str("\n");

        md.push_str("## Free Energy Classification (Best Threshold)\n\n");
        md.push_str(&format!("| Metric | Value |\n|--------|-------|\n"));
        md.push_str(&format!(
            "| Best threshold | {:.4} |\n",
            self.best_threshold
        ));
        md.push_str(&format!("| F1 | {:.4} |\n", self.best_f1));
        md.push_str(&format!("| Accuracy | {:.4} |\n", cm.accuracy()));
        md.push_str(&format!("| Precision | {:.4} |\n", cm.precision()));
        md.push_str(&format!("| Recall | {:.4} |\n", cm.recall()));
        md.push_str(&format!("| Specificity | {:.4} |\n", cm.specificity()));
        md.push_str(&format!("| AUC | {:.4} |\n", self.auc));
        md.push_str("\n");

        md.push_str("### Confusion Matrix\n\n");
        md.push_str("| | Predicted Normal | Predicted Disruption |\n");
        md.push_str("|---|---|---|\n");
        md.push_str(&format!(
            "| Actual Normal | TN={} | FP={} |\n",
            cm.tn, cm.fp
        ));
        md.push_str(&format!(
            "| Actual Disruption | FN={} | TP={} |\n",
            cm.fn_, cm.tp
        ));
        md.push_str("\n");

        md.push_str("## Phi-Decline Method (Comparison)\n\n");
        md.push_str(&format!("| Metric | Value |\n|--------|-------|\n"));
        md.push_str(&format!("| Accuracy | {:.4} |\n", phi_cm.accuracy()));
        md.push_str(&format!("| Precision | {:.4} |\n", phi_cm.precision()));
        md.push_str(&format!("| Recall | {:.4} |\n", phi_cm.recall()));
        md.push_str(&format!("| F1 | {:.4} |\n", phi_cm.f1()));
        md.push_str("\n");

        md.push_str("## Lead Time Analysis\n\n");
        md.push_str(&format!("| Metric | Value |\n|--------|-------|\n"));
        md.push_str(&format!(
            "| Detected shots | {} / {} |\n",
            lt.detected_count, lt.total_disrupted
        ));
        md.push_str(&format!("| Mean lead time | {:.4} s |\n", lt.mean_s));
        md.push_str(&format!("| Median lead time | {:.4} s |\n", lt.median_s));
        md.push_str(&format!("| Min lead time | {:.4} s |\n", lt.min_s));
        md.push_str(&format!("| Max lead time | {:.4} s |\n", lt.max_s));
        md.push_str("\n");

        md.push_str("## Normalization Ranges\n\n");
        md.push_str("| Sensor | Min | Max |\n|--------|-----|-----|\n");
        for i in 0..NUM_SENSORS {
            md.push_str(&format!(
                "| {} | {:.6} | {:.6} |\n",
                SENSOR_NAMES[i],
                self.normalization_ranges.mins[i],
                self.normalization_ranges.maxs[i]
            ));
        }
        md.push_str("\n");

        md.push_str("## Threshold Sweep\n\n");
        md.push_str(
            "| Threshold | F1 | Precision | Recall |\n|-----------|-----|-----------|--------|\n",
        );
        for &(thresh, f1, prec, rec) in &self.threshold_sweep {
            md.push_str(&format!(
                "| {:.2} | {:.4} | {:.4} | {:.4} |\n",
                thresh, f1, prec, rec
            ));
        }

        md
    }

    /// Serialize the report to JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).context("Failed to serialize report to JSON")
    }
}

// ── Evaluation Harness ───────────────────────────────────────────────────────

/// Score each test sample: compute free energy relative to reference HV.
fn score_samples(
    encoder: &DensityLimitEncoder,
    reference: &ContinuousHV,
    samples: &[DensityLimitSample],
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

/// Classify predictions at a given threshold and return a ConfusionMatrix.
pub fn classify_at_threshold(scored: &[(f64, bool)], threshold: f64) -> ConfusionMatrix {
    let mut tp = 0u64;
    let mut fp = 0u64;
    let mut tn = 0u64;
    let mut fn_ = 0u64;

    for &(score, is_disruption) in scored {
        let predicted = score >= threshold;
        match (predicted, is_disruption) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, false) => tn += 1,
            (false, true) => fn_ += 1,
        }
    }

    ConfusionMatrix::new(tp, fp, tn, fn_)
}

/// Compute lead times for disrupted test shots at a given threshold.
///
/// For each disrupted shot, finds the first sample where free_energy >= threshold
/// and computes how far before the disruption onset time that detection occurred.
fn compute_lead_times(
    encoder: &DensityLimitEncoder,
    reference: &ContinuousHV,
    test_shots: &[DensityLimitShot],
    threshold: f64,
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

        // Find first detection time
        for sample in &shot.samples {
            let hv = encoder.encode(sample);
            let sim = hv.similarity(reference) as f64;
            let fe = if sim.is_finite() {
                (1.0 - sim).max(0.0)
            } else {
                1.0
            };

            if fe >= threshold {
                let lead = onset_time - sample.time_s;
                if lead.is_finite() {
                    lead_times.push(lead);
                }
                break;
            }
        }
    }

    lead_times
}

/// Evaluate Phi-decline method on the test set.
fn evaluate_phi_decline(
    encoder: &DensityLimitEncoder,
    test_shots: &[DensityLimitShot],
    window_size: usize,
) -> ConfusionMatrix {
    let mut tp = 0u64;
    let mut fp = 0u64;
    let mut tn = 0u64;
    let mut fn_ = 0u64;

    for shot in test_shots {
        let mut monitor = CModPhiMonitor::new(window_size);

        for sample in &shot.samples {
            let encoding = encoder.encode(sample);
            monitor.update(sample.time_s * 1000.0, &encoding);

            let predicted_disruption = monitor.is_declining();
            match (predicted_disruption, sample.is_disruption) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, false) => tn += 1,
                (false, true) => fn_ += 1,
            }
        }
    }

    ConfusionMatrix::new(tp, fp, tn, fn_)
}

/// Run the full evaluation pipeline on DL_DataFrame data.
///
/// 1. Loads CSV and groups by shot
/// 2. Splits into train/test by sorted shot_id (deterministic)
/// 3. Builds encoder and reference HV from healthy training samples
/// 4. Sweeps free-energy thresholds to find optimal F1
/// 5. Computes ROC curve, AUC, and lead-time statistics
/// 6. Evaluates Phi-decline method for comparison
pub fn evaluate_density_limit(path: &Path, config: &EvaluationConfig) -> Result<EvaluationReport> {
    // 1. Load data
    let shots = load_density_limit_csv(path)?;
    anyhow::ensure!(
        !shots.is_empty(),
        "No shots found in CSV: {}",
        path.display()
    );

    // 2. Stratified train/test split: ensure disrupted shots appear in both sets
    let mut disrupted_shots: Vec<DensityLimitShot> = Vec::new();
    let mut normal_shots: Vec<DensityLimitShot> = Vec::new();
    for shot in shots {
        if shot.has_disruption {
            disrupted_shots.push(shot);
        } else {
            normal_shots.push(shot);
        }
    }

    let d_split = ((disrupted_shots.len() as f64) * config.train_fraction).ceil() as usize;
    let d_split = d_split
        .min(disrupted_shots.len().saturating_sub(1))
        .max(if disrupted_shots.len() > 1 { 1 } else { 0 });
    let n_split = ((normal_shots.len() as f64) * config.train_fraction).ceil() as usize;
    let n_split = n_split.min(normal_shots.len().saturating_sub(1)).max(1);

    let mut train_shots_owned: Vec<DensityLimitShot> = Vec::new();
    let mut test_shots_owned: Vec<DensityLimitShot> = Vec::new();

    train_shots_owned.extend(disrupted_shots.drain(..d_split));
    test_shots_owned.extend(disrupted_shots); // remaining disrupted → test
    train_shots_owned.extend(normal_shots.drain(..n_split));
    test_shots_owned.extend(normal_shots); // remaining normal → test

    let train_shots = &train_shots_owned[..];
    let test_shots = &test_shots_owned[..];

    // 3. Build encoder from training data normalization ranges
    let encoder = DensityLimitEncoder::from_shots(train_shots);
    let reference = encoder.build_reference(train_shots);

    // 4. Score all test samples
    let test_samples: Vec<DensityLimitSample> = test_shots
        .iter()
        .flat_map(|shot| shot.samples.iter().cloned())
        .collect();

    let normal_count = test_samples.iter().filter(|s| !s.is_disruption).count();
    let disruption_count = test_samples.iter().filter(|s| s.is_disruption).count();

    let scored = score_samples(&encoder, &reference, &test_samples);

    // 5. Threshold sweep: 0.05 to 0.95 in steps of 0.05
    let mut threshold_sweep = Vec::new();
    let mut best_threshold = 0.5;
    let mut best_f1 = 0.0;

    for step in 1..=19 {
        let threshold = step as f64 * 0.05;
        let cm = classify_at_threshold(&scored, threshold);
        let f1 = cm.f1();
        threshold_sweep.push((threshold, f1, cm.precision(), cm.recall()));
        if f1 > best_f1 {
            best_f1 = f1;
            best_threshold = threshold;
        }
    }

    // 6. Final classification at best threshold
    let confusion_matrix = classify_at_threshold(&scored, best_threshold);

    // 7. ROC curve and AUC
    let roc_curve = compute_roc_curve(&scored);
    let auc = compute_auc(&roc_curve);

    // 8. Lead time analysis
    let lead_times = compute_lead_times(&encoder, &reference, test_shots, best_threshold);
    let total_disrupted = test_shots.iter().filter(|s| s.has_disruption).count();
    let lead_time_stats = LeadTimeStats::from_lead_times(&lead_times, total_disrupted);

    // 9. Phi-decline comparison
    let phi_decline_cm = evaluate_phi_decline(&encoder, test_shots, config.window_size);

    Ok(EvaluationReport {
        best_threshold,
        best_f1,
        confusion_matrix,
        roc_curve,
        auc,
        lead_time_stats,
        total_shots_train: train_shots.len(),
        total_shots_test: test_shots.len(),
        samples_per_class: (normal_count, disruption_count),
        phi_decline_cm,
        normalization_ranges: encoder.ranges().clone(),
        threshold_sweep,
    })
}

// ── V2 Evaluation: Temporal Windows + Per-Shot Reference ─────────────────

/// Configuration for the improved V2 evaluation harness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationConfigV2 {
    /// Fraction of shots used for training (rest for test). Default: 0.8.
    pub train_fraction: f64,
    /// Temporal window size (consecutive samples bundled into one HV). Default: 5.
    pub window_size: usize,
    /// Fraction of each shot's early samples used as per-shot reference. Default: 0.2.
    pub reference_fraction: f64,
    /// Maximum threshold for fine sweep. Default: 0.30.
    pub fine_threshold_max: f64,
    /// Step size for fine threshold sweep. Default: 0.005.
    pub fine_threshold_step: f64,
    /// Minimum samples needed for per-shot reference (fallback to global). Default: 10.
    pub min_reference_samples: usize,
    /// Sliding window size for Phi-decline monitor. Default: 10.
    pub phi_window_size: usize,
}

impl Default for EvaluationConfigV2 {
    fn default() -> Self {
        Self {
            train_fraction: 0.8,
            window_size: 5,
            reference_fraction: 0.2,
            fine_threshold_max: 0.30,
            fine_threshold_step: 0.005,
            min_reference_samples: 10,
            phi_window_size: 10,
        }
    }
}

/// Score test shots using temporal windowing and per-shot early-phase reference.
///
/// For each shot:
/// 1. Take the first `reference_fraction` of samples as the "healthy baseline"
/// 2. Build a per-shot reference HV from those early samples
/// 3. Encode remaining samples using sliding temporal windows
/// 4. Compute free energy of each window against the per-shot reference
/// 5. Fall back to global reference if per-shot reference has too few samples
fn score_shots_v2(
    encoder: &DensityLimitEncoder,
    global_reference: &ContinuousHV,
    test_shots: &[DensityLimitShot],
    config: &EvaluationConfigV2,
) -> Vec<(f64, bool)> {
    let mut scored = Vec::new();

    for shot in test_shots {
        let n_samples = shot.samples.len();
        let n_ref = ((n_samples as f64) * config.reference_fraction).ceil() as usize;
        let n_ref = n_ref.min(n_samples);

        // Build per-shot reference from early samples (should be healthy baseline)
        let reference = if n_ref >= config.min_reference_samples {
            let ref_samples = &shot.samples[..n_ref];
            let ref_encodings: Vec<ContinuousHV> =
                ref_samples.iter().map(|s| encoder.encode(s)).collect();
            ContinuousHV::bundle_owned(&ref_encodings)
        } else {
            global_reference.clone()
        };

        // Score all samples using temporal windows
        // For each position i, the window is samples[max(0, i+1-W)..=i]
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

/// Run the improved V2 evaluation pipeline with temporal windowing and per-shot reference.
///
/// Improvements over V1:
/// 1. **Temporal windowing**: encodes N consecutive samples as a bundle (captures trajectory)
/// 2. **Per-shot reference**: each shot is compared against its own early-phase baseline
/// 3. **Finer threshold sweep**: 0.005 to 0.30 in 0.005 steps (60 thresholds)
pub fn evaluate_density_limit_v2(
    path: &Path,
    config: &EvaluationConfigV2,
) -> Result<EvaluationReport> {
    // 1. Load data
    let shots = load_density_limit_csv(path)?;
    anyhow::ensure!(
        !shots.is_empty(),
        "No shots found in CSV: {}",
        path.display()
    );

    // 2. Stratified train/test split (same logic as V1)
    let mut disrupted_shots: Vec<DensityLimitShot> = Vec::new();
    let mut normal_shots: Vec<DensityLimitShot> = Vec::new();
    for shot in shots {
        if shot.has_disruption {
            disrupted_shots.push(shot);
        } else {
            normal_shots.push(shot);
        }
    }

    let d_split = ((disrupted_shots.len() as f64) * config.train_fraction).ceil() as usize;
    let d_split = d_split
        .min(disrupted_shots.len().saturating_sub(1))
        .max(if disrupted_shots.len() > 1 { 1 } else { 0 });
    let n_split = ((normal_shots.len() as f64) * config.train_fraction).ceil() as usize;
    let n_split = n_split.min(normal_shots.len().saturating_sub(1)).max(1);

    let mut train_shots_owned: Vec<DensityLimitShot> = Vec::new();
    let mut test_shots_owned: Vec<DensityLimitShot> = Vec::new();

    train_shots_owned.extend(disrupted_shots.drain(..d_split));
    test_shots_owned.extend(disrupted_shots);
    train_shots_owned.extend(normal_shots.drain(..n_split));
    test_shots_owned.extend(normal_shots);

    let train_shots = &train_shots_owned[..];
    let test_shots = &test_shots_owned[..];

    // 3. Build encoder and global reference from training data
    let encoder = DensityLimitEncoder::from_shots(train_shots);
    let global_reference = encoder.build_reference(train_shots);

    // 4. Count test samples
    let normal_count = test_shots
        .iter()
        .flat_map(|s| &s.samples)
        .filter(|s| !s.is_disruption)
        .count();
    let disruption_count = test_shots
        .iter()
        .flat_map(|s| &s.samples)
        .filter(|s| s.is_disruption)
        .count();

    // 5. Score with V2 method (temporal windows + per-shot reference)
    let scored = score_shots_v2(&encoder, &global_reference, test_shots, config);

    // 6. Fine threshold sweep
    let mut threshold_sweep = Vec::new();
    let mut best_threshold = 0.05;
    let mut best_f1 = 0.0;

    let mut thresh = config.fine_threshold_step;
    while thresh <= config.fine_threshold_max + 1e-9 {
        let cm = classify_at_threshold(&scored, thresh);
        let f1 = cm.f1();
        threshold_sweep.push((thresh, f1, cm.precision(), cm.recall()));
        if f1 > best_f1 {
            best_f1 = f1;
            best_threshold = thresh;
        }
        thresh += config.fine_threshold_step;
    }

    // 7. Final classification at best threshold
    let confusion_matrix = classify_at_threshold(&scored, best_threshold);

    // 8. ROC curve and AUC
    let roc_curve = compute_roc_curve(&scored);
    let auc = compute_auc(&roc_curve);

    // 9. Lead time analysis (using windowed scoring per shot)
    let total_disrupted = test_shots.iter().filter(|s| s.has_disruption).count();
    let lead_times = compute_lead_times_v2(
        &encoder,
        &global_reference,
        test_shots,
        best_threshold,
        config,
    );
    let lead_time_stats = LeadTimeStats::from_lead_times(&lead_times, total_disrupted);

    // 10. Phi-decline comparison (unchanged from V1)
    let phi_decline_cm = evaluate_phi_decline(&encoder, test_shots, config.phi_window_size);

    Ok(EvaluationReport {
        best_threshold,
        best_f1,
        confusion_matrix,
        roc_curve,
        auc,
        lead_time_stats,
        total_shots_train: train_shots.len(),
        total_shots_test: test_shots.len(),
        samples_per_class: (normal_count, disruption_count),
        phi_decline_cm,
        normalization_ranges: encoder.ranges().clone(),
        threshold_sweep,
    })
}

/// Compute lead times for disrupted test shots using V2 windowed scoring.
fn compute_lead_times_v2(
    encoder: &DensityLimitEncoder,
    global_reference: &ContinuousHV,
    test_shots: &[DensityLimitShot],
    threshold: f64,
    config: &EvaluationConfigV2,
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

// ── V3 Evaluation: Physics-Informed Features (Greenwald + Rates) ─────────

/// Number of features in the V3 encoder:
/// 6 raw sensors + 1 Greenwald fraction + 6 rate-of-change + 1 Greenwald rate = 14
const NUM_FEATURES_V3: usize = 14;

/// Deterministic seeds for V3 basis vectors.
/// First 6 reuse SENSOR_SEEDS, next 8 are new for derived features.
const V3_SEEDS: [u64; NUM_FEATURES_V3] = [
    0xD10_0001, // density
    0xD10_0002, // elongation
    0xD10_0003, // minor_radius
    0xD10_0004, // plasma_current
    0xD10_0005, // toroidal_B_field
    0xD10_0006, // triangularity
    0xD10_0007, // Greenwald fraction
    0xD10_0008, // delta density
    0xD10_0009, // delta elongation
    0xD10_000A, // delta minor_radius
    0xD10_000B, // delta plasma_current
    0xD10_000C, // delta toroidal_B_field
    0xD10_000D, // delta triangularity
    0xD10_000E, // delta Greenwald fraction
];

/// Compute the Greenwald density fraction.
///
/// f_G = density * pi * minor_radius^2 / plasma_current
///
/// The Greenwald limit is n_G = I_p / (pi * a^2). When density > n_G, the
/// plasma exceeds the density limit. The fraction f_G = density / n_G
/// = density * pi * a^2 / I_p. Values > 1.0 indicate disruption regime.
///
/// DL_DataFrame density is in 10^20 m^-3 units, plasma_current in MA,
/// minor_radius in m, so no unit conversion is needed because the Greenwald
/// formula in standard SI uses n_G[10^20 m^-3] = I_p[MA] / (pi * a[m]^2).
pub fn greenwald_fraction(density: f32, minor_radius: f32, plasma_current: f32) -> f32 {
    if plasma_current.abs() < 0.01 {
        return 0.0;
    }
    let pi = std::f32::consts::PI;
    density * pi * minor_radius * minor_radius / plasma_current
}

/// Configuration for the V3 physics-informed evaluation harness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationConfigV3 {
    /// Fraction of shots used for training (rest for test). Default: 0.8.
    pub train_fraction: f64,
    /// Temporal window size (consecutive samples bundled into one HV). Default: 5.
    pub window_size: usize,
    /// Fraction of each shot's early samples used as per-shot reference. Default: 0.2.
    pub reference_fraction: f64,
    /// Maximum threshold for fine sweep. Default: 0.50.
    pub fine_threshold_max: f64,
    /// Step size for fine threshold sweep. Default: 0.002.
    pub fine_threshold_step: f64,
    /// Minimum samples needed for per-shot reference (fallback to global). Default: 10.
    pub min_reference_samples: usize,
    /// Sliding window size for Phi-decline monitor. Default: 10.
    pub phi_window_size: usize,
    /// Whether to include the Greenwald fraction feature. Default: true.
    pub use_greenwald: bool,
    /// Whether to include rate-of-change features. Default: true.
    pub use_rates: bool,
}

impl Default for EvaluationConfigV3 {
    fn default() -> Self {
        Self {
            train_fraction: 0.8,
            window_size: 5,
            reference_fraction: 0.2,
            fine_threshold_max: 0.50,
            fine_threshold_step: 0.002,
            min_reference_samples: 10,
            phi_window_size: 10,
            use_greenwald: true,
            use_rates: true,
        }
    }
}

/// HDC encoder with physics-informed derived features.
///
/// Encodes up to 14 features per sample:
/// - 6 raw sensors (normalized)
/// - 1 Greenwald density fraction (f_G, normalized to [0,1] via clamp(f_G/2))
/// - 6 rate-of-change features (delta of each sensor, mapped from [-1,1] to [0,1])
/// - 1 Greenwald fraction rate (delta f_G, mapped from [-1,1] to [0,1])
#[derive(Clone)]
pub struct DensityLimitEncoderV3 {
    bases: Vec<ContinuousHV>,
    ranges: NormalizationRanges,
    use_greenwald: bool,
    use_rates: bool,
    /// Number of active features (depends on config flags).
    num_active: usize,
}

impl DensityLimitEncoderV3 {
    /// Create a V3 encoder with the given normalization ranges and feature flags.
    pub fn new(ranges: NormalizationRanges, use_greenwald: bool, use_rates: bool) -> Self {
        // Determine which features are active:
        // Always: 6 raw sensors
        // Optional: 1 Greenwald fraction, 6 rate deltas, 1 Greenwald rate
        let mut active_indices: Vec<usize> = (0..NUM_SENSORS).collect();
        if use_greenwald {
            active_indices.push(6); // Greenwald fraction
        }
        if use_rates {
            active_indices.extend(7..13); // 6 sensor deltas
            if use_greenwald {
                active_indices.push(13); // Greenwald rate
            }
        }
        let num_active = active_indices.len();

        let bases: Vec<ContinuousHV> = active_indices
            .iter()
            .map(|&idx| ContinuousHV::random(HDC_DIMENSION, V3_SEEDS[idx]))
            .collect();

        Self {
            bases,
            ranges,
            use_greenwald,
            use_rates,
            num_active,
        }
    }

    /// Create encoder and compute normalization ranges from training shots.
    pub fn from_shots(shots: &[DensityLimitShot], use_greenwald: bool, use_rates: bool) -> Self {
        let ranges = NormalizationRanges::from_shots(shots);
        Self::new(ranges, use_greenwald, use_rates)
    }

    /// Number of basis vectors (active features).
    pub fn num_features(&self) -> usize {
        self.num_active
    }

    /// Compute the feature vector for a sample, given an optional previous sample
    /// for rate-of-change computation.
    ///
    /// Returns a Vec<f32> of length `num_active` with all values in [0, 1].
    pub fn compute_features(
        &self,
        sample: &DensityLimitSample,
        prev: Option<&DensityLimitSample>,
    ) -> Vec<f32> {
        let mut features = Vec::with_capacity(self.num_active);

        // 1. Raw 6 normalized sensors
        let normed = self.ranges.normalize_sample(sample);
        features.extend_from_slice(&normed);

        // 2. Greenwald fraction (if enabled)
        let f_g = greenwald_fraction(sample.density, sample.minor_radius, sample.plasma_current);
        if self.use_greenwald {
            // Normalize to [0, 1] by dividing by 2 and clamping
            features.push((f_g / 2.0).clamp(0.0, 1.0));
        }

        // 3. Rate-of-change features (if enabled)
        if self.use_rates {
            let prev_vals = prev.map(|p| p.sensor_values());
            let curr_vals = sample.sensor_values();

            for i in 0..NUM_SENSORS {
                let delta = if let Some(ref pv) = prev_vals {
                    let raw_delta = curr_vals[i] - pv[i];
                    // Normalize delta using the sensor range
                    let range = self.ranges.maxs[i] - self.ranges.mins[i];
                    if range.abs() > 1e-10 {
                        (raw_delta / range).clamp(-1.0, 1.0)
                    } else {
                        0.0
                    }
                } else {
                    0.0 // First sample in window: delta = 0
                };
                // Map [-1, 1] → [0, 1]
                features.push((delta + 1.0) / 2.0);
            }

            // Greenwald rate (if Greenwald is also enabled)
            if self.use_greenwald {
                let prev_f_g = prev
                    .map(|p| greenwald_fraction(p.density, p.minor_radius, p.plasma_current))
                    .unwrap_or(f_g);
                let delta_fg = (f_g - prev_f_g).clamp(-1.0, 1.0);
                features.push((delta_fg + 1.0) / 2.0);
            }
        }

        features
    }

    /// Encode a single sample with physics-informed features.
    pub fn encode(
        &self,
        sample: &DensityLimitSample,
        prev: Option<&DensityLimitSample>,
    ) -> ContinuousHV {
        let features = self.compute_features(sample, prev);
        debug_assert_eq!(features.len(), self.num_active);
        ContinuousHV::encode_weighted(&self.bases, &features)
    }

    /// Encode a temporal window of consecutive samples into a single HV.
    ///
    /// Each sample is encoded with its predecessor for rate-of-change features,
    /// then all encodings are bundled.
    pub fn encode_window(&self, samples: &[DensityLimitSample]) -> ContinuousHV {
        if samples.is_empty() {
            return ContinuousHV::zero(HDC_DIMENSION);
        }
        if samples.len() == 1 {
            return self.encode(&samples[0], None);
        }
        let encoded: Vec<ContinuousHV> = samples
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let prev = if i > 0 { Some(&samples[i - 1]) } else { None };
                self.encode(s, prev)
            })
            .collect();
        ContinuousHV::bundle_owned(&encoded)
    }

    /// Build a reference HV from a set of healthy (non-disruption) samples within shots.
    ///
    /// Iterates sequentially through each shot to correctly compute rate features.
    pub fn build_reference(&self, shots: &[DensityLimitShot]) -> ContinuousHV {
        let mut healthy_encodings: Vec<ContinuousHV> = Vec::new();

        for shot in shots {
            let mut prev: Option<&DensityLimitSample> = None;
            for sample in &shot.samples {
                if !sample.is_disruption {
                    healthy_encodings.push(self.encode(sample, prev));
                }
                prev = Some(sample);
            }
        }

        if healthy_encodings.is_empty() {
            return ContinuousHV::zero(HDC_DIMENSION);
        }

        ContinuousHV::bundle_owned(&healthy_encodings)
    }

    /// Access the normalization ranges.
    pub fn ranges(&self) -> &NormalizationRanges {
        &self.ranges
    }
}

/// Score test shots using V3 physics-informed encoding with temporal windowing
/// and per-shot early-phase reference.
fn score_shots_v3(
    encoder: &DensityLimitEncoderV3,
    global_reference: &ContinuousHV,
    test_shots: &[DensityLimitShot],
    config: &EvaluationConfigV3,
) -> Vec<(f64, bool)> {
    let mut scored = Vec::new();

    for shot in test_shots {
        let n_samples = shot.samples.len();
        let n_ref = ((n_samples as f64) * config.reference_fraction).ceil() as usize;
        let n_ref = n_ref.min(n_samples);

        // Build per-shot reference from early samples (sequential for rate features)
        let reference = if n_ref >= config.min_reference_samples {
            let ref_samples = &shot.samples[..n_ref];
            let ref_encodings: Vec<ContinuousHV> = ref_samples
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let prev = if i > 0 {
                        Some(&ref_samples[i - 1])
                    } else {
                        None
                    };
                    encoder.encode(s, prev)
                })
                .collect();
            ContinuousHV::bundle_owned(&ref_encodings)
        } else {
            global_reference.clone()
        };

        // Score all samples using temporal windows
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

/// Compute lead times for disrupted test shots using V3 windowed scoring.
fn compute_lead_times_v3(
    encoder: &DensityLimitEncoderV3,
    global_reference: &ContinuousHV,
    test_shots: &[DensityLimitShot],
    threshold: f64,
    config: &EvaluationConfigV3,
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
            let ref_samples = &shot.samples[..n_ref];
            let ref_encodings: Vec<ContinuousHV> = ref_samples
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let prev = if i > 0 {
                        Some(&ref_samples[i - 1])
                    } else {
                        None
                    };
                    encoder.encode(s, prev)
                })
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

/// Run the V3 physics-informed evaluation pipeline.
///
/// Improvements over V2:
/// 1. **Greenwald fraction**: f_G = density * pi * a^2 / I_p (the key disruption physics)
/// 2. **Rate-of-change features**: temporal derivatives of all 6 sensors + Greenwald fraction
/// 3. **14-feature encoder**: captures both state and trajectory in physics-meaningful space
/// 4. **Finer threshold sweep**: 0.002 to 0.50 in 0.002 steps (250 thresholds)
pub fn evaluate_density_limit_v3(
    path: &Path,
    config: &EvaluationConfigV3,
) -> Result<EvaluationReport> {
    // 1. Load data
    let shots = load_density_limit_csv(path)?;
    anyhow::ensure!(
        !shots.is_empty(),
        "No shots found in CSV: {}",
        path.display()
    );

    // 2. Stratified train/test split (same logic as V1/V2)
    let mut disrupted_shots: Vec<DensityLimitShot> = Vec::new();
    let mut normal_shots: Vec<DensityLimitShot> = Vec::new();
    for shot in shots {
        if shot.has_disruption {
            disrupted_shots.push(shot);
        } else {
            normal_shots.push(shot);
        }
    }

    let d_split = ((disrupted_shots.len() as f64) * config.train_fraction).ceil() as usize;
    let d_split = d_split
        .min(disrupted_shots.len().saturating_sub(1))
        .max(if disrupted_shots.len() > 1 { 1 } else { 0 });
    let n_split = ((normal_shots.len() as f64) * config.train_fraction).ceil() as usize;
    let n_split = n_split.min(normal_shots.len().saturating_sub(1)).max(1);

    let mut train_shots_owned: Vec<DensityLimitShot> = Vec::new();
    let mut test_shots_owned: Vec<DensityLimitShot> = Vec::new();

    train_shots_owned.extend(disrupted_shots.drain(..d_split));
    test_shots_owned.extend(disrupted_shots);
    train_shots_owned.extend(normal_shots.drain(..n_split));
    test_shots_owned.extend(normal_shots);

    let train_shots = &train_shots_owned[..];
    let test_shots = &test_shots_owned[..];

    // 3. Build V3 encoder and global reference from training data
    let encoder =
        DensityLimitEncoderV3::from_shots(train_shots, config.use_greenwald, config.use_rates);
    let global_reference = encoder.build_reference(train_shots);

    eprintln!(
        "V3 encoder: {} active features (greenwald={}, rates={})",
        encoder.num_features(),
        config.use_greenwald,
        config.use_rates
    );

    // 4. Count test samples
    let normal_count = test_shots
        .iter()
        .flat_map(|s| &s.samples)
        .filter(|s| !s.is_disruption)
        .count();
    let disruption_count = test_shots
        .iter()
        .flat_map(|s| &s.samples)
        .filter(|s| s.is_disruption)
        .count();

    // 5. Score with V3 method (physics-informed + temporal windows + per-shot reference)
    let scored = score_shots_v3(&encoder, &global_reference, test_shots, config);

    // 6. Fine threshold sweep: 0.002 to 0.50 in 0.002 steps
    let mut threshold_sweep = Vec::new();
    let mut best_threshold = 0.05;
    let mut best_f1 = 0.0;

    let mut thresh = config.fine_threshold_step;
    while thresh <= config.fine_threshold_max + 1e-9 {
        let cm = classify_at_threshold(&scored, thresh);
        let f1 = cm.f1();
        threshold_sweep.push((thresh, f1, cm.precision(), cm.recall()));
        if f1 > best_f1 {
            best_f1 = f1;
            best_threshold = thresh;
        }
        thresh += config.fine_threshold_step;
    }

    // 7. Final classification at best threshold
    let confusion_matrix = classify_at_threshold(&scored, best_threshold);

    // 8. ROC curve and AUC
    let roc_curve = compute_roc_curve(&scored);
    let auc = compute_auc(&roc_curve);

    // 9. Lead time analysis
    let total_disrupted = test_shots.iter().filter(|s| s.has_disruption).count();
    let lead_times = compute_lead_times_v3(
        &encoder,
        &global_reference,
        test_shots,
        best_threshold,
        config,
    );
    let lead_time_stats = LeadTimeStats::from_lead_times(&lead_times, total_disrupted);

    // 10. Phi-decline comparison (uses V1 encoder for apples-to-apples)
    let v1_encoder = DensityLimitEncoder::from_shots(train_shots);
    let phi_decline_cm = evaluate_phi_decline(&v1_encoder, test_shots, config.phi_window_size);

    Ok(EvaluationReport {
        best_threshold,
        best_f1,
        confusion_matrix,
        roc_curve,
        auc,
        lead_time_stats,
        total_shots_train: train_shots.len(),
        total_shots_test: test_shots.len(),
        samples_per_class: (normal_count, disruption_count),
        phi_decline_cm,
        normalization_ranges: encoder.ranges().clone(),
        threshold_sweep,
    })
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Helper: generate synthetic CSV content.
    fn synthetic_csv(num_shots: usize, samples_per_shot: usize) -> String {
        let mut csv = String::from(
            "discharge_ID,time,density_limit_phase,density,elongation,minor_radius,plasma_current,toroidal_B_field,triangularity\n",
        );

        for shot in 0..num_shots {
            let shot_id = 1000000 + shot as u64;
            let is_disruptive = shot % 3 == 0; // every 3rd shot disrupts

            for t in 0..samples_per_shot {
                let time = t as f64 * 0.01;
                let frac = t as f32 / samples_per_shot as f32;

                // Normal values with slight variation per shot
                let density = 1.0 + (shot as f32 * 0.01);
                let elongation = 1.6 + (frac * 0.1);
                let minor_radius = 0.21;
                let plasma_current = 0.75 + (frac * 0.05);
                let b_field = 5.4;
                let triangularity = 0.37;

                // Label: disruption in second half of disruptive shots
                let phase = if is_disruptive && frac > 0.5 { 1 } else { 0 };

                csv.push_str(&format!(
                    "{},{},{},{},{},{},{},{},{}\n",
                    shot_id,
                    time,
                    phase,
                    density,
                    elongation,
                    minor_radius,
                    plasma_current,
                    b_field,
                    triangularity
                ));
            }
        }
        csv
    }

    /// Write synthetic CSV to a temp file and return the path.
    fn write_temp_csv(content: &str) -> (tempfile::NamedTempFile, std::path::PathBuf) {
        let mut file = tempfile::NamedTempFile::new().expect("create temp file");
        file.write_all(content.as_bytes()).expect("write csv");
        file.flush().expect("flush");
        let path = file.path().to_path_buf();
        (file, path)
    }

    #[test]
    fn test_load_density_limit_synthetic() {
        let csv = synthetic_csv(4, 5); // 4 shots, 5 samples each = 20 rows
        let (_file, path) = write_temp_csv(&csv);

        let shots = load_density_limit_csv(&path).expect("load CSV");
        assert_eq!(shots.len(), 4, "should have 4 shots");

        let total_samples: usize = shots.iter().map(|s| s.samples.len()).sum();
        assert_eq!(total_samples, 20, "should have 20 total samples");

        // Shots are sorted by ID
        for i in 1..shots.len() {
            assert!(shots[i].shot_id > shots[i - 1].shot_id);
        }

        // Check disruption labels: shots 0, 3 (every 3rd) should have disruptions
        assert!(shots[0].has_disruption, "shot 0 should have disruption");
        assert!(
            !shots[1].has_disruption,
            "shot 1 should not have disruption"
        );
        assert!(
            !shots[2].has_disruption,
            "shot 2 should not have disruption"
        );
        assert!(shots[3].has_disruption, "shot 3 should have disruption");

        // Check sample fields are populated
        let first = &shots[0].samples[0];
        assert!(first.density > 0.0);
        assert!(first.elongation > 0.0);
        assert!(first.minor_radius > 0.0);
    }

    #[test]
    fn test_encoder_determinism() {
        let csv = synthetic_csv(2, 5);
        let (_file, path) = write_temp_csv(&csv);
        let shots = load_density_limit_csv(&path).unwrap();

        let encoder = DensityLimitEncoder::from_shots(&shots);
        let sample = &shots[0].samples[0];

        let hv1 = encoder.encode(sample);
        let hv2 = encoder.encode(sample);

        assert_eq!(hv1.dim(), HDC_DIMENSION);
        assert_eq!(hv2.dim(), HDC_DIMENSION);

        // Same input must produce bit-identical output (deterministic encoding)
        for (i, (a, b)) in hv1.values.iter().zip(hv2.values.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "Dimension {} differs: {} vs {}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn test_confusion_matrix_metrics() {
        // Perfect confusion matrix: 50 TP, 10 FP, 80 TN, 5 FN
        let cm = ConfusionMatrix::new(50, 10, 80, 5);

        assert_eq!(cm.total(), 145);

        let acc = cm.accuracy();
        assert!((acc - (130.0 / 145.0)).abs() < 1e-10, "accuracy={}", acc);

        let prec = cm.precision();
        assert!((prec - (50.0 / 60.0)).abs() < 1e-10, "precision={}", prec);

        let rec = cm.recall();
        assert!((rec - (50.0 / 55.0)).abs() < 1e-10, "recall={}", rec);

        let spec = cm.specificity();
        assert!((spec - (80.0 / 90.0)).abs() < 1e-10, "specificity={}", spec);

        let f1 = cm.f1();
        let expected_f1 = 2.0 * prec * rec / (prec + rec);
        assert!((f1 - expected_f1).abs() < 1e-10, "f1={}", f1);

        // Edge case: empty matrix
        let empty = ConfusionMatrix::new(0, 0, 0, 0);
        assert_eq!(empty.accuracy(), 0.0);
        assert_eq!(empty.precision(), 0.0);
        assert_eq!(empty.recall(), 0.0);
        assert_eq!(empty.f1(), 0.0);
    }

    #[test]
    fn test_roc_auc_perfect_classifier() {
        // Perfect classifier: all positives get score 1.0, all negatives get score 0.0
        let mut scored = Vec::new();
        for _ in 0..50 {
            scored.push((1.0, true));
        }
        for _ in 0..50 {
            scored.push((0.0, false));
        }

        let roc = compute_roc_curve(&scored);
        let auc = compute_auc(&roc);

        assert!(
            auc > 0.99,
            "Perfect classifier should have AUC ~1.0, got {}",
            auc
        );
    }

    #[test]
    fn test_roc_auc_random_classifier() {
        // Random classifier: scores are independent of labels
        use rand::Rng;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut scored = Vec::new();
        for _ in 0..500 {
            let score: f64 = rng.r#gen();
            let label = rng.gen_bool(0.5);
            scored.push((score, label));
        }

        let roc = compute_roc_curve(&scored);
        let auc = compute_auc(&roc);

        assert!(
            (auc - 0.5).abs() < 0.15,
            "Random classifier should have AUC ~0.5, got {}",
            auc
        );
    }

    #[test]
    fn test_evaluation_on_synthetic() {
        // Generate enough data for a meaningful evaluation
        let csv = synthetic_csv(20, 20); // 20 shots, 20 samples each = 400 rows
        let (_file, path) = write_temp_csv(&csv);

        let config = EvaluationConfig {
            train_fraction: 0.8,
            window_size: 5,
            phi_decline_threshold: 0.9,
        };

        let report = evaluate_density_limit(&path, &config).expect("evaluation should succeed");

        // Basic sanity checks on the report
        assert_eq!(report.total_shots_train + report.total_shots_test, 20);
        assert!(
            report.total_shots_train >= 15,
            "should have ~16 training shots"
        );
        assert!(report.total_shots_test >= 3, "should have ~4 test shots");
        assert!(report.best_threshold >= 0.05 && report.best_threshold <= 0.95);
        assert!(report.best_f1 >= 0.0 && report.best_f1 <= 1.0);
        assert!(report.auc >= 0.0 && report.auc <= 1.0);
        assert!(!report.roc_curve.is_empty(), "ROC curve should have points");
        assert_eq!(report.threshold_sweep.len(), 19, "19 threshold steps");
        assert_eq!(
            report.samples_per_class.0 + report.samples_per_class.1,
            report.total_shots_test * 20,
            "sample counts should match"
        );

        // Confusion matrix should account for all test samples
        let cm = &report.confusion_matrix;
        assert_eq!(
            cm.total() as usize,
            report.samples_per_class.0 + report.samples_per_class.1,
            "CM total should match test samples"
        );

        // Markdown report should be non-empty
        let md = report.to_markdown();
        assert!(md.contains("C-Mod Density Limit Evaluation Report"));
        assert!(md.contains("Confusion Matrix"));
        assert!(md.contains("Lead Time Analysis"));

        // JSON should be valid
        let json = report.to_json().expect("JSON serialization");
        assert!(json.contains("best_threshold"));
    }

    #[test]
    fn test_lead_time_stats() {
        let lead_times = vec![0.1, 0.2, 0.3, 0.5, 1.0];
        let stats = LeadTimeStats::from_lead_times(&lead_times, 7);

        assert_eq!(stats.detected_count, 5);
        assert_eq!(stats.total_disrupted, 7);
        assert!((stats.min_s - 0.1).abs() < 1e-10);
        assert!((stats.max_s - 1.0).abs() < 1e-10);
        assert!((stats.median_s - 0.3).abs() < 1e-10);
        let expected_mean = (0.1 + 0.2 + 0.3 + 0.5 + 1.0) / 5.0;
        assert!((stats.mean_s - expected_mean).abs() < 1e-10);

        // Empty case
        let empty = LeadTimeStats::from_lead_times(&[], 3);
        assert_eq!(empty.detected_count, 0);
        assert_eq!(empty.total_disrupted, 3);
    }

    #[test]
    fn test_normalization_ranges() {
        let csv = synthetic_csv(3, 5);
        let (_file, path) = write_temp_csv(&csv);
        let shots = load_density_limit_csv(&path).unwrap();

        let ranges = NormalizationRanges::from_shots(&shots);

        // All mins should be <= maxs
        for i in 0..NUM_SENSORS {
            assert!(
                ranges.mins[i] <= ranges.maxs[i],
                "Sensor {} min ({}) > max ({})",
                SENSOR_NAMES[i],
                ranges.mins[i],
                ranges.maxs[i]
            );
        }

        // Normalization should produce values in [0, 1]
        for shot in &shots {
            for sample in &shot.samples {
                let normed = ranges.normalize_sample(sample);
                for (i, &v) in normed.iter().enumerate() {
                    assert!(
                        v >= 0.0 && v <= 1.0,
                        "Sensor {} normalized to {} (out of [0,1])",
                        SENSOR_NAMES[i],
                        v
                    );
                }
            }
        }
    }

    #[test]
    fn test_density_limit_shot_onset_time() {
        let shot = DensityLimitShot {
            shot_id: 1,
            has_disruption: true,
            samples: vec![
                DensityLimitSample {
                    shot_id: 1,
                    time_s: 0.1,
                    density: 1.0,
                    elongation: 1.6,
                    minor_radius: 0.21,
                    plasma_current: 0.75,
                    toroidal_b_field: 5.4,
                    triangularity: 0.37,
                    is_disruption: false,
                },
                DensityLimitSample {
                    shot_id: 1,
                    time_s: 0.2,
                    density: 1.2,
                    elongation: 1.65,
                    minor_radius: 0.21,
                    plasma_current: 0.78,
                    toroidal_b_field: 5.4,
                    triangularity: 0.38,
                    is_disruption: true,
                },
            ],
        };

        assert_eq!(shot.disruption_onset_time(), Some(0.2));
    }

    #[test]
    fn test_encode_window_single_sample() {
        let csv = synthetic_csv(2, 5);
        let (_file, path) = write_temp_csv(&csv);
        let shots = load_density_limit_csv(&path).unwrap();
        let encoder = DensityLimitEncoder::from_shots(&shots);

        let sample = &shots[0].samples[0];
        let single_hv = encoder.encode(sample);
        let window_hv = encoder.encode_window(&[sample.clone()]);

        // Single-sample window should be identical to single encode
        assert_eq!(single_hv.dim(), window_hv.dim());
        for (a, b) in single_hv.values.iter().zip(window_hv.values.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn test_encode_window_temporal_context() {
        // Use samples from different shots to guarantee different sensor values
        let csv = synthetic_csv(10, 20);
        let (_file, path) = write_temp_csv(&csv);
        let shots = load_density_limit_csv(&path).unwrap();
        let encoder = DensityLimitEncoder::from_shots(&shots);

        // Take samples from different shots (different density baselines)
        let mut diverse_samples = vec![
            shots[0].samples[0].clone(),
            shots[1].samples[5].clone(),
            shots[2].samples[10].clone(),
            shots[3].samples[15].clone(),
            shots[4].samples[19].clone(),
        ];
        // Ensure they have different values
        diverse_samples[0].density = 0.5;
        diverse_samples[1].density = 1.5;
        diverse_samples[2].density = 2.5;
        diverse_samples[3].density = 3.5;
        diverse_samples[4].density = 4.5;

        let window_hv = encoder.encode_window(&diverse_samples);
        let single_hv = encoder.encode(&diverse_samples[2]); // middle sample

        // Window bundles multiple different encodings, so it should not be identical to one
        let sim = window_hv.similarity(&single_hv);
        assert!(
            sim > 0.0,
            "window vs single similarity should be positive: {}",
            sim
        );
        assert_eq!(window_hv.dim(), HDC_DIMENSION);

        // Different windows should produce different HVs
        let window2_hv = encoder.encode_window(&diverse_samples[0..3]);
        let sim2 = window_hv.similarity(&window2_hv);
        assert!(sim2 > 0.0, "two windows should have positive similarity");
        assert_eq!(window2_hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_encode_window_empty() {
        let csv = synthetic_csv(2, 5);
        let (_file, path) = write_temp_csv(&csv);
        let shots = load_density_limit_csv(&path).unwrap();
        let encoder = DensityLimitEncoder::from_shots(&shots);

        let empty_hv = encoder.encode_window(&[]);
        assert_eq!(empty_hv.dim(), HDC_DIMENSION);
        // All zeros
        assert!(empty_hv.values.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_evaluation_v2_on_synthetic() {
        // Generate enough data for a meaningful V2 evaluation
        let csv = synthetic_csv(20, 40); // 20 shots, 40 samples each
        let (_file, path) = write_temp_csv(&csv);

        let config = EvaluationConfigV2 {
            train_fraction: 0.8,
            window_size: 5,
            reference_fraction: 0.2,
            fine_threshold_max: 0.30,
            fine_threshold_step: 0.005,
            min_reference_samples: 5, // lower for synthetic data
            phi_window_size: 10,
        };

        let report =
            evaluate_density_limit_v2(&path, &config).expect("V2 evaluation should succeed");

        // Basic sanity checks
        assert_eq!(report.total_shots_train + report.total_shots_test, 20);
        assert!(report.best_threshold >= 0.005 && report.best_threshold <= 0.30);
        assert!(report.best_f1 >= 0.0 && report.best_f1 <= 1.0);
        assert!(report.auc >= 0.0 && report.auc <= 1.0);
        assert!(!report.roc_curve.is_empty(), "ROC curve should have points");

        // Fine sweep should have ~60 thresholds
        assert!(
            report.threshold_sweep.len() >= 50,
            "Fine sweep should have ~60 thresholds, got {}",
            report.threshold_sweep.len()
        );

        // Confusion matrix should account for all test samples
        let cm = &report.confusion_matrix;
        assert_eq!(
            cm.total() as usize,
            report.samples_per_class.0 + report.samples_per_class.1,
            "CM total should match test samples"
        );
    }

    #[test]
    fn test_per_shot_reference_fallback() {
        // With very few samples per shot, should fall back to global reference
        let csv = synthetic_csv(10, 5); // only 5 samples per shot
        let (_file, path) = write_temp_csv(&csv);

        let config = EvaluationConfigV2 {
            min_reference_samples: 10, // 20% of 5 = 1, which is < 10 → fallback
            ..EvaluationConfigV2::default()
        };

        // Should not panic — falls back to global reference
        let report = evaluate_density_limit_v2(&path, &config).expect("fallback should work");
        assert!(report.auc >= 0.0);
    }

    // ── V3 Tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_greenwald_fraction_computation() {
        // Typical C-Mod: I_p = 0.8 MA, a = 0.22 m
        // n_G = I_p / (pi * a^2) = 0.8 / (pi * 0.0484) = 0.8 / 0.15205 ≈ 5.26
        // If density = 5.26, f_G should be ≈ 1.0
        let pi = std::f32::consts::PI;
        let ip = 0.8f32;
        let a = 0.22f32;
        let n_g = ip / (pi * a * a); // ≈ 5.26

        let f_g = greenwald_fraction(n_g, a, ip);
        assert!(
            (f_g - 1.0).abs() < 0.01,
            "At Greenwald limit, f_G should be ~1.0, got {}",
            f_g
        );

        // Double the density → f_G ≈ 2.0
        let f_g2 = greenwald_fraction(n_g * 2.0, a, ip);
        assert!(
            (f_g2 - 2.0).abs() < 0.01,
            "At 2x Greenwald limit, f_G should be ~2.0, got {}",
            f_g2
        );

        // Half the density → f_G ≈ 0.5
        let f_g_half = greenwald_fraction(n_g * 0.5, a, ip);
        assert!(
            (f_g_half - 0.5).abs() < 0.01,
            "At 0.5x Greenwald limit, f_G should be ~0.5, got {}",
            f_g_half
        );

        // Division by zero guard
        let f_g_zero = greenwald_fraction(5.0, 0.22, 0.001);
        assert_eq!(f_g_zero, 0.0, "Near-zero current should return 0.0");
    }

    #[test]
    fn test_v3_encoder_dimension() {
        let csv = synthetic_csv(4, 10);
        let (_file, path) = write_temp_csv(&csv);
        let shots = load_density_limit_csv(&path).unwrap();

        // Full V3: 6 raw + 1 Greenwald + 6 deltas + 1 Greenwald delta = 14
        let encoder_full = DensityLimitEncoderV3::from_shots(&shots, true, true);
        assert_eq!(
            encoder_full.num_features(),
            14,
            "full V3 should have 14 features"
        );

        // No Greenwald, no rates: just 6 raw sensors
        let encoder_raw = DensityLimitEncoderV3::from_shots(&shots, false, false);
        assert_eq!(
            encoder_raw.num_features(),
            6,
            "raw-only should have 6 features"
        );

        // Greenwald only (no rates): 6 raw + 1 Greenwald = 7
        let encoder_gw = DensityLimitEncoderV3::from_shots(&shots, true, false);
        assert_eq!(
            encoder_gw.num_features(),
            7,
            "greenwald-only should have 7 features"
        );

        // Rates only (no Greenwald): 6 raw + 6 deltas = 12
        let encoder_rates = DensityLimitEncoderV3::from_shots(&shots, false, true);
        assert_eq!(
            encoder_rates.num_features(),
            12,
            "rates-only should have 12 features"
        );
    }

    #[test]
    fn test_v3_evaluation_on_synthetic() {
        // Generate enough data for a meaningful V3 evaluation
        let csv = synthetic_csv(20, 40); // 20 shots, 40 samples each
        let (_file, path) = write_temp_csv(&csv);

        let config = EvaluationConfigV3 {
            train_fraction: 0.8,
            window_size: 5,
            reference_fraction: 0.2,
            fine_threshold_max: 0.50,
            fine_threshold_step: 0.002,
            min_reference_samples: 5,
            phi_window_size: 10,
            use_greenwald: true,
            use_rates: true,
        };

        let report =
            evaluate_density_limit_v3(&path, &config).expect("V3 evaluation should succeed");

        // Basic sanity checks
        assert_eq!(report.total_shots_train + report.total_shots_test, 20);
        assert!(report.best_threshold >= 0.002 && report.best_threshold <= 0.50);
        assert!(report.best_f1 >= 0.0 && report.best_f1 <= 1.0);
        assert!(report.auc >= 0.0 && report.auc <= 1.0);
        assert!(!report.roc_curve.is_empty(), "ROC curve should have points");

        // Fine sweep should have ~250 thresholds
        assert!(
            report.threshold_sweep.len() >= 200,
            "Fine sweep should have ~250 thresholds, got {}",
            report.threshold_sweep.len()
        );

        // Confusion matrix should account for all test samples
        let cm = &report.confusion_matrix;
        assert_eq!(
            cm.total() as usize,
            report.samples_per_class.0 + report.samples_per_class.1,
            "CM total should match test samples"
        );
    }
}
